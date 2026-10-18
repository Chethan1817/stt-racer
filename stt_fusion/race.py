"""Run several STT engines on the same audio at the same time."""

from __future__ import annotations

import asyncio
import difflib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from .types import Candidate

# STT.recognize() retries on its own. One retry rides out a flaky request without
# letting a dead provider hold up the race.
ENGINE_CONN_OPTIONS = APIConnectOptions(max_retry=1, retry_interval=0.5, timeout=10.0)

_WORD = re.compile(r"[\w']+")


@dataclass(frozen=True)
class Engine:
    """An STT instance and the label used for it in logs and in the fusion prompt."""

    name: str
    stt: stt.STT
    warm_up: bool = False
    """Send the engine a tiny silent clip when the user starts speaking. For providers
    whose first request after a pause is slow (Deepgram: ~1.4 s cold vs ~0.5 s warm),
    this has the connection and their servers ready when the real segment arrives."""


async def race(
    engines: Sequence[Engine],
    buffer: utils.AudioBuffer,
    *,
    timeout_s: float,
    grace_s: float | None = None,
    quorum: int | None = None,
    language: NotGivenOr[str] = NOT_GIVEN,
) -> list[Candidate]:
    """Start every engine on `buffer` at the same moment and collect the results.

    Waits until every engine has answered or `timeout_s` has passed. With `grace_s`,
    once `quorum` engines (default: a majority) have returned text, the rest get at
    most `grace_s` more, which bounds the slowest engine's tail. Engines still running
    then are cancelled and reported as "timeout". Nothing is raised: a failing engine
    comes back as a Candidate with status "error". Results keep engine order.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + timeout_s
    needed = quorum if quorum is not None else len(engines) // 2 + 1
    tasks = [
        asyncio.create_task(_transcribe(engine, buffer, language), name=f"race:{engine.name}")
        for engine in engines
    ]
    pending = set(tasks)
    with_text = 0
    try:
        while pending:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            done, pending = await asyncio.wait(
                pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED
            )
            if grace_s is not None and with_text < needed:
                with_text += sum(task.result().usable for task in done)
                if with_text >= needed:
                    deadline = min(deadline, loop.time() + grace_s)
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending)

    waited = loop.time() - started
    results: list[Candidate] = []
    for engine, task in zip(engines, tasks, strict=True):
        if task.done() and not task.cancelled():
            results.append(task.result())
        else:
            results.append(
                Candidate(
                    engine=engine.name,
                    text="",
                    status="timeout",
                    latency_s=waited,
                    error=f"no answer within {waited:.1f}s",
                )
            )
    return results


def consensus_pick(candidates: Sequence[Candidate]) -> str:
    """Pick the transcript that agrees most with the other engines.

    Returns "" when most of the engines that answered heard nothing. That marks the
    segment as noise, so a lone hallucination (Whisper's "Thank you." on a cough)
    never starts a turn. Ties go to the engine listed first.
    """
    answered = [c for c in candidates if c.status in ("ok", "empty")]
    heard = [c for c in answered if c.usable]
    if not heard or (len(answered) - len(heard)) * 2 > len(answered):
        return ""
    words = [_words(c.text) for c in heard]

    def agreement(i: int) -> float:
        return sum(
            difflib.SequenceMatcher(None, words[i], words[j]).ratio()
            for j in range(len(heard))
            if j != i
        )

    best = max(range(len(heard)), key=lambda i: (agreement(i), -i))
    return heard[best].text


async def _transcribe(
    engine: Engine, buffer: utils.AudioBuffer, language: NotGivenOr[str]
) -> Candidate:
    started = time.perf_counter()
    confidence: float | None = None
    try:
        if engine.stt.capabilities.offline_recognize:
            event = await engine.stt.recognize(
                buffer, language=language, conn_options=ENGINE_CONN_OPTIONS
            )
            alt = event.alternatives[0] if event.alternatives else None
            text = alt.text.strip() if alt else ""
            if alt and alt.confidence:
                confidence = alt.confidence
        else:
            text = await _transcribe_via_stream(engine.stt, buffer, language)
    except Exception as e:  # one engine failing must never end the race
        return Candidate(
            engine=engine.name,
            text="",
            status="error",
            latency_s=time.perf_counter() - started,
            error=f"{type(e).__name__}: {e}"[:300],
        )
    return Candidate(
        engine=engine.name,
        text=text,
        status="ok" if text else "empty",
        latency_s=time.perf_counter() - started,
        confidence=confidence,
    )


async def _transcribe_via_stream(
    engine_stt: stt.STT, buffer: utils.AudioBuffer, language: NotGivenOr[str]
) -> str:
    """Replay a finished clip through a streaming-only STT and join its final transcripts."""
    frames = buffer if isinstance(buffer, list) else [buffer]
    texts: list[str] = []
    async with engine_stt.stream(language=language, conn_options=ENGINE_CONN_OPTIONS) as stream:
        for frame in frames:
            stream.push_frame(frame)
        stream.end_input()
        async for event in stream:
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                texts.append(event.alternatives[0].text.strip())
    return " ".join(t for t in texts if t)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())
