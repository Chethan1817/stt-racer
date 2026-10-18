"""An STT that races several engines on each VAD speech segment."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterable, Sequence
from typing import Any

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils.audio import calculate_audio_duration
from livekit.agents.utils.misc import is_given

from .race import Engine, consensus_pick, race
from .types import SegmentResult

logger = logging.getLogger("stt-fusion")

SEGMENT_METADATA_KEY = "stt_fusion.segment"

# The race retries per engine; the wrapper stream itself must never retry.
_STREAM_CONN_OPTIONS = APIConnectOptions(max_retry=0, timeout=DEFAULT_API_CONNECT_OPTIONS.timeout)
_WARM_UP_CONN_OPTIONS = APIConnectOptions(max_retry=0, timeout=5.0)
_SAMPLE_RATE = 16000
# How long an engine stays warm after a request. Deepgram cools down within a few
# seconds, so during long utterances the clip is re-sent at this interval.
_WARM_FOR_S = 2.5


class MultiSTT(stt.STT):
    """Races several STT engines on identical audio.

    Speech is cut into segments with a VAD, the same approach as `stt.StreamAdapter`.
    Each segment goes to every engine at once. The FINAL transcript's text is the
    consensus pick, and every engine's result rides along in `SpeechData.metadata`
    under `SEGMENT_METADATA_KEY` (read it back with `segment_from_event`).
    """

    def __init__(
        self,
        *,
        engines: Sequence[Engine],
        vad: vad.VAD,
        timeout_s: float = 10.0,
        grace_s: float | None = None,
        min_segment_s: float = 0.2,
        language: str = "en",
    ) -> None:
        if not engines:
            raise ValueError("MultiSTT needs at least one engine")
        names = [e.name for e in engines]
        if len(set(names)) != len(names):
            raise ValueError(f"engine names must be unique, got {names}")
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self._engines = tuple(engines)
        self._vad = vad
        self._timeout_s = timeout_s
        self._grace_s = grace_s
        self._min_segment_s = min_segment_s
        self._language = language
        self._last_used: dict[str, float] = {}
        self._warm_up_tasks: set[asyncio.Task[None]] = set()
        # Usage metrics are forwarded so session.usage covers every engine. Error events
        # are deliberately not forwarded: the session treats an unrecoverable STT error
        # as fatal, and one failing engine must not end the call.
        for engine in self._engines:
            engine.stt.on("metrics_collected", self._on_engine_metrics)

    @property
    def engines(self) -> tuple[Engine, ...]:
        return self._engines

    @property
    def model(self) -> str:
        return "+".join(e.name for e in self._engines)

    @property
    def provider(self) -> str:
        return "stt-fusion"

    async def transcribe_segment(
        self, buffer: utils.AudioBuffer, *, index: int, language: NotGivenOr[str] = NOT_GIVEN
    ) -> SegmentResult:
        candidates = await race(
            self._engines,
            buffer,
            timeout_s=self._timeout_s,
            grace_s=self._grace_s,
            language=language,
        )
        now = time.monotonic()
        self._last_used.update((engine.name, now) for engine in self._engines)
        return SegmentResult(
            index=index,
            audio_s=round(calculate_audio_duration(buffer), 3),
            candidates=tuple(candidates),
            provisional=consensus_pick(candidates),
        )

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        segment = await self.transcribe_segment(buffer, index=0, language=language)
        return _final_event(segment, language if is_given(language) else self._language)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> MultiSTTStream:
        return MultiSTTStream(self, language=language)

    def prewarm(self) -> None:
        for engine in self._engines:
            engine.stt.prewarm()

    def warm_up(self) -> None:
        """Called while the user is speaking: wake engines that opted in (Engine.warm_up)."""
        now = time.monotonic()
        for engine in self._engines:
            if (
                not engine.warm_up
                or now - self._last_used.get(engine.name, -_WARM_FOR_S) < _WARM_FOR_S
            ):
                continue
            self._last_used[engine.name] = now
            task = asyncio.create_task(self._warm_up(engine), name=f"warm_up:{engine.name}")
            self._warm_up_tasks.add(task)
            task.add_done_callback(self._warm_up_tasks.discard)

    async def _warm_up(self, engine: Engine) -> None:
        silence = rtc.AudioFrame(
            data=b"\0\0" * (_SAMPLE_RATE // 4),
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=_SAMPLE_RATE // 4,
        )
        try:
            await engine.stt.recognize(silence, conn_options=_WARM_UP_CONN_OPTIONS)
        except Exception as e:  # only an optimisation; the real request will report errors
            logger.debug("warm-up of %s failed: %s", engine.name, e)

    async def aclose(self) -> None:
        for task in self._warm_up_tasks:
            task.cancel()
        for engine in self._engines:
            engine.stt.off("metrics_collected", self._on_engine_metrics)

    def _on_engine_metrics(self, *args: Any, **kwargs: Any) -> None:
        self.emit("metrics_collected", *args, **kwargs)


class MultiSTTStream(stt.RecognizeStream):
    def __init__(self, multi: MultiSTT, *, language: NotGivenOr[str]) -> None:
        super().__init__(stt=multi, conn_options=_STREAM_CONN_OPTIONS, sample_rate=_SAMPLE_RATE)
        self._multi = multi
        self._language = language
        self._segment_index = 0

    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[stt.SpeechEvent]) -> None:
        # Usage comes from each engine's own recognize() call, forwarded by MultiSTT.
        async for _ in event_aiter:
            pass

    async def _run(self) -> None:
        vad_stream = self._multi._vad.stream()

        async def _forward_input() -> None:
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
            vad_stream.end_input()

        async def _recognize() -> None:
            async for event in vad_stream:
                if event.type == vad.VADEventType.INFERENCE_DONE:
                    if event.speaking:
                        self._multi.warm_up()  # throttled to once per _WARM_FOR_S
                elif event.type == vad.VADEventType.START_OF_SPEECH:
                    self._multi.warm_up()
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                    )
                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    speech_end_time = (
                        time.time() - event.silence_duration - event.inference_duration
                    )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.END_OF_SPEECH,
                            speech_end_time=speech_end_time,
                        )
                    )
                    await self._transcribe(event.frames, speech_end_time)

        tasks = [
            asyncio.create_task(_forward_input(), name="multi_stt.forward_input"),
            asyncio.create_task(_recognize(), name="multi_stt.recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await vad_stream.aclose()

    async def _transcribe(self, frames: list[Any], speech_end_time: float) -> None:
        if not frames:
            return
        buffer = utils.merge_frames(frames)
        duration = calculate_audio_duration(buffer)
        if duration < self._multi._min_segment_s:
            logger.debug("skipping %.2fs speech segment (too short)", duration)
            return

        segment = await self._multi.transcribe_segment(
            buffer, index=self._segment_index, language=self._language
        )
        self._segment_index += 1
        if not segment.provisional:
            if not any(c.status in ("ok", "empty") for c in segment.candidates):
                errors = "; ".join(f"{c.engine}: {c.error}" for c in segment.candidates)
                logger.error("every STT engine failed on a %.1fs segment: %s", duration, errors)
            else:
                heard = {c.engine: c.text for c in segment.candidates if c.usable}
                logger.info("dropped %.1fs segment as noise; text from: %s", duration, heard)
            return

        language = self._language if is_given(self._language) else self._multi._language
        final = _final_event(segment, language)
        final.speech_end_time = speech_end_time
        self._event_ch.send_nowait(final)


def segment_from_event(event: stt.SpeechEvent) -> SegmentResult | None:
    """Return the race result attached to a MultiSTT FINAL event, if there is one."""
    if event.type != stt.SpeechEventType.FINAL_TRANSCRIPT or not event.alternatives:
        return None
    data = (event.alternatives[0].metadata or {}).get(SEGMENT_METADATA_KEY)
    return SegmentResult.from_dict(data) if data else None


def _final_event(segment: SegmentResult, language: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[
            stt.SpeechData(
                language=language,
                text=segment.provisional,
                metadata={SEGMENT_METADATA_KEY: segment.to_dict()},
            )
        ],
    )
