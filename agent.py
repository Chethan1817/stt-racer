"""LiveKit voice agent: race three STT engines, fuse their transcripts with an LLM, reply.

uv run agent.py console   # talk to it in the terminal (local mic + speakers)
uv run agent.py dev       # serve rooms on a LiveKit server (e.g. `livekit-server --dev`)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
)
from livekit.plugins import deepgram, openai

from stt_fusion.config import Settings
from stt_fusion.fusion_agent import FusionAgent
from stt_fusion.multi_stt import MultiSTT
from stt_fusion.providers import build_engines, build_fuser
from stt_fusion.recorder import TurnRecorder

load_dotenv()
logger = logging.getLogger("stt-fusion")

INSTRUCTIONS = """\
You are a friendly, concise voice assistant. The user speaks into a microphone; their \
words are transcribed by three speech engines and reconciled by a separate step before \
they reach you, so the text is usually right. If a name, number or email still looks \
wrong or ambiguous, ask a short clarifying question instead of guessing. When the user \
gives important details (names, phone numbers, emails, dates, amounts), repeat them \
back. Keep replies to one to three short sentences. Your replies are spoken aloud, so \
never use markdown, lists, emojis or special characters."""

GREETING = "Hi! I'm listening through three speech engines at once. What can I do for you?"

UI_TOPIC = "stt-fusion"  # text-stream topic for segment, turn and latency events
UI_PIPELINE_ATTRIBUTE = "stt_fusion.pipeline"  # agent participant attribute

# LiveKit's per-turn timings (ChatMessage.metrics), measured from the end of the user's speech
# except where noted.
LATENCY_STAGES = {
    "transcription_delay": "transcript",  # segment closed + engine race
    "end_of_turn_delay": "end of turn",
    "on_user_turn_completed_delay": "fusion",  # duration of the hook itself
    "llm_node_ttft": "reply first token",  # from the start of the LLM call
    "tts_node_ttfb": "first audio",  # from the first text sent to TTS
    "e2e_latency": "total",
}

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=MultiSTT(
            engines=build_engines(settings),
            # Separate from the session VAD: longer pauses make fewer, fuller segments,
            # which batch STT models transcribe more accurately.
            vad=inference.VAD(min_silence_duration=settings.segment_min_silence_s),
            timeout_s=settings.race_timeout_s,
            grace_s=settings.race_grace_s,
            language=settings.stt_language,
        ),
        vad=inference.VAD(),
        llm=openai.LLM(model=settings.agent_llm_model),
        tts=deepgram.TTS(model=settings.tts_model),
        turn_handling=TurnHandlingOptions(
            # Pinned to the local model: by default dev/console mode would try LiveKit Cloud.
            turn_detection=(
                inference.TurnDetector(version="v1-mini")
                if settings.turn_detection == "eou"
                else "vad"
            ),
            # The default "adaptive" detector is also a LiveKit Cloud service.
            interruption={"mode": "vad"},
            # On by default; it would start the reply before the fused transcript exists.
            preemptive_generation={"enabled": False},
        ),
    )

    async def _log_usage() -> None:
        logger.info("session usage: %s", session.usage)

    ctx.add_shutdown_callback(_log_usage)

    recorder = TurnRecorder(settings.log_dir, session_name=ctx.room.name)
    if recorder.path:
        logger.info("writing turn log to %s", recorder.path)

    async def publish_to_room(event: dict[str, Any]) -> None:
        # The web UI listens on this topic. Console mode has no connected room.
        if ctx.room.isconnected():
            await ctx.room.local_participant.send_text(json.dumps(event), topic=UI_TOPIC)

    pending_publishes: set[asyncio.Task[None]] = set()

    @session.on("conversation_item_added")
    def _report_latency(ev: ConversationItemAddedEvent) -> None:
        reply = ev.item
        if reply.type != "message" or reply.role != "assistant":
            return
        if "e2e_latency" not in reply.metrics:
            return  # e.g. the greeting: no user turn to measure from
        users = [m for m in session.history.items if m.type == "message" and m.role == "user"]
        metrics = {**(users[-1].metrics if users else {}), **reply.metrics}
        latency = {
            key: round(value, 3)
            for key in LATENCY_STAGES
            if isinstance(value := metrics.get(key), (int, float))
        }
        logger.info(
            "turn latency: %s",
            ", ".join(f"{LATENCY_STAGES[key]} {value:.2f}s" for key, value in latency.items()),
        )
        task = asyncio.create_task(publish_to_room({"type": "latency", **latency}))
        pending_publishes.add(task)
        task.add_done_callback(pending_publishes.discard)

    await session.start(
        agent=FusionAgent(
            instructions=INSTRUCTIONS,
            fuser=build_fuser(settings),
            recorder=recorder,
            greeting=GREETING,
            on_event=publish_to_room,
        ),
        room=ctx.room,
    )

    if ctx.room.isconnected():
        # Shown in the UI header; attributes also reach participants who join later.
        pipeline = {
            "engines": [engine.name for engine in session.stt.engines],  # type: ignore[union-attr]
            "fusion_llm": settings.fusion_llm_model,
            "reply_llm": settings.agent_llm_model,
            "tts": settings.tts_model,
        }
        await ctx.room.local_participant.set_attributes(
            {UI_PIPELINE_ATTRIBUTE: json.dumps(pipeline)}
        )


if __name__ == "__main__":
    cli.run_app(server)
