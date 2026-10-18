"""A full AgentSession run with no network: WAV in, and the LLM node sees the fused text.

Real pieces: AgentSession, its turn handling, the local VAD, MultiSTT and FusionAgent.
Fakes: the three STT engines, the fusion LLM, and an echo LLM for the reply.
"""

import asyncio
import json
from pathlib import Path

from fakes import EchoLLM, FakeCompleter, FakeSTT, WavAudioInput
from livekit.agents import AgentSession, ConversationItemAddedEvent, TurnHandlingOptions, inference

from stt_fusion.fusion import TranscriptFuser
from stt_fusion.fusion_agent import FusionAgent
from stt_fusion.multi_stt import MultiSTT
from stt_fusion.race import Engine
from stt_fusion.recorder import TurnRecorder

# "Book a table for four at Toit tonight." (macOS `say`)
ONE_PHRASE = Path(__file__).parent / "data" / "one_phrase.wav"


async def test_llm_node_receives_the_fused_transcript(tmp_path):
    fused = "Book a table for four at Toit tonight."
    completer = FakeCompleter(json.dumps({"transcript": fused, "rationale": "toyt vs Toit"}))
    session = AgentSession(
        stt=MultiSTT(
            engines=[
                Engine("a", FakeSTT("book a table for four at toyt tonight")),
                Engine("b", FakeSTT("book a table for four at toit tonight")),
                Engine("c", FakeSTT("book a table for for at toit to night")),
            ],
            vad=inference.VAD(min_silence_duration=0.6),
            timeout_s=2,
        ),
        vad=inference.VAD(),
        llm=EchoLLM(),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "vad"},
            preemptive_generation={"enabled": False},
        ),
    )
    replies: asyncio.Queue[str] = asyncio.Queue()

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            replies.put_nowait(ev.item.text_content or "")

    ui_events: list[dict] = []

    async def on_event(event: dict) -> None:
        ui_events.append(event)

    agent = FusionAgent(
        instructions="test",
        fuser=TranscriptFuser(completer),
        recorder=TurnRecorder(tmp_path, session_name="e2e"),
        on_event=on_event,
    )
    session.input.audio = WavAudioInput(ONE_PHRASE)
    await session.start(agent)
    try:
        reply = await asyncio.wait_for(replies.get(), timeout=20)
    finally:
        await session.aclose()

    # The reply LLM ran after fusion and saw the corrected text, not the consensus pick.
    assert reply == f"You said: {fused}"
    users = [m for m in agent.chat_ctx.messages() if m.role == "user"]
    assert users[-1].text_content == fused
    assert users[-1].extra["stt_fusion"]["heard"] == "book a table for four at toit tonight"
    assert len(completer.requests) == 1  # exactly one fusion call for the turn

    # What the web UI receives: the race result as soon as it's in, then the fused turn.
    assert [e["type"] for e in ui_events] == ["segment", "turn"]
    assert [c["engine"] for c in ui_events[0]["segment"]["candidates"]] == ["a", "b", "c"]
    assert ui_events[1]["fusion"]["text"] == fused
    assert ui_events[1]["changed"] is True
