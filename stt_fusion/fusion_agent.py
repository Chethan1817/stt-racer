"""The agent: collect each segment's race results, fuse them once the user's turn ends."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any

from livekit import rtc
from livekit.agents import Agent, ModelSettings, StopResponse, llm, stt

from .fusion import TranscriptFuser
from .multi_stt import segment_from_event
from .recorder import TurnRecorder
from .types import SegmentResult, merge_turn

logger = logging.getLogger("stt-fusion")

EventSink = Callable[[dict[str, Any]], Awaitable[None]]
"""Receives {"type": "segment", ...} as each race finishes and {"type": "turn", ...}
after each fusion (used to drive the web UI)."""


class FusionAgent(Agent):
    """Rewrites each user message with the fused transcript before the LLM node runs.

    The session must use `MultiSTT`. Its FINAL events pass through `stt_node`, where
    the race results are buffered; the session can only end the turn after it has
    seen those events, so the buffer is complete by `on_user_turn_completed`.
    """

    def __init__(
        self,
        *,
        instructions: str,
        fuser: TranscriptFuser,
        recorder: TurnRecorder,
        greeting: str | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        super().__init__(instructions=instructions)
        self._fuser = fuser
        self._recorder = recorder
        self._greeting = greeting
        self._on_event = on_event
        self._pending: list[SegmentResult] = []
        self._event_tasks: set[asyncio.Task[None]] = set()

    async def on_enter(self) -> None:
        if self._greeting:
            # A fixed line spoken by TTS: no LLM call before the user has said anything.
            self.session.say(self._greeting)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> AsyncIterable[stt.SpeechEvent]:
        async for event in Agent.default.stt_node(self, audio, model_settings):
            if (segment := segment_from_event(event)) is not None:
                self._pending.append(segment)
                self._emit({"type": "segment", "segment": segment.to_dict()})
            yield event

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        heard = new_message.text_content or ""
        segments = self.take_segments(heard)
        if not segments:
            return  # typed input (console text mode): nothing was raced

        candidates = merge_turn(segments)
        outcome = await self._fuser.fuse(candidates, fallback=heard, chat_ctx=turn_ctx)
        new_message.content = [outcome.text]
        new_message.extra["stt_fusion"] = {
            "heard": heard,
            "candidates": [c.to_dict() for c in candidates],
            "fusion": outcome.to_dict(),
        }
        entry = self._recorder.record(
            segments=segments, candidates=candidates, heard=heard, outcome=outcome
        )
        self._emit({"type": "turn", **entry})
        if not outcome.text.strip():
            raise StopResponse()  # the fuser judged the audio to be noise: don't reply

    def take_segments(self, heard: str) -> list[SegmentResult]:
        """Pop the buffered segments whose texts make up this turn's transcript.

        The session builds a turn's transcript by joining FINAL texts with spaces. A
        segment that finished after the session ended the turn belongs to the next
        turn, so only an exact run of segments matching `heard` is taken; anything
        before that run is stale. If nothing matches, everything buffered is used.
        """
        target = _normalize(heard)
        for start in range(len(self._pending)):
            joined = ""
            for end in range(start, len(self._pending)):
                joined = f"{joined} {_normalize(self._pending[end].provisional)}".strip()
                if joined == target:
                    taken = self._pending[start : end + 1]
                    if start:
                        logger.warning("discarding %d stale speech segment(s)", start)
                    self._pending = self._pending[end + 1 :]
                    return taken
                if not target.startswith(joined):
                    break
        if self._pending:
            logger.warning("turn transcript does not match buffered segments; using all of them")
        taken, self._pending = self._pending, []
        return taken

    def _emit(self, event: dict[str, Any]) -> None:
        """Hand an event to the sink in the background, so the pipeline never waits on it."""
        if self._on_event is None:
            return
        task = asyncio.create_task(self._send(event))
        self._event_tasks.add(task)
        task.add_done_callback(self._event_tasks.discard)

    async def _send(self, event: dict[str, Any]) -> None:
        assert self._on_event is not None
        try:
            await self._on_event(event)
        except Exception:
            logger.exception("failed to publish %s event", event.get("type"))


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
