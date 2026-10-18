import json

import pytest
from fakes import FakeCompleter, empty, ok
from livekit.agents import StopResponse, llm

from stt_fusion.fusion import TranscriptFuser
from stt_fusion.fusion_agent import FusionAgent
from stt_fusion.recorder import TurnRecorder
from stt_fusion.types import Candidate, SegmentResult


def seg(index: int, provisional: str, *candidates: Candidate) -> SegmentResult:
    cands = candidates or (ok(provisional, "a"),)
    return SegmentResult(index=index, audio_s=1.0, candidates=cands, provisional=provisional)


def make_agent(completer: FakeCompleter, log_dir=None) -> FusionAgent:
    return FusionAgent(
        instructions="test",
        fuser=TranscriptFuser(completer),
        recorder=TurnRecorder(log_dir, session_name="test"),
    )


def user_message(text: str) -> llm.ChatMessage:
    return llm.ChatMessage(role="user", content=[text])


async def test_user_message_is_replaced_by_the_fused_transcript(tmp_path):
    completer = FakeCompleter(json.dumps({"transcript": "Book a table at Toit.", "rationale": "x"}))
    agent = make_agent(completer, log_dir=tmp_path)
    agent._pending = [
        seg(
            0,
            "book a table at toyt",
            ok("book a table at toyt", "a"),
            ok("book a table at toit", "b"),
        )
    ]
    message = user_message("book a table at toyt")

    await agent.on_user_turn_completed(llm.ChatContext.empty(), message)

    assert message.text_content == "Book a table at Toit."
    details = message.extra["stt_fusion"]
    assert details["heard"] == "book a table at toyt"
    assert details["fusion"]["source"] == "llm"
    [log_file] = tmp_path.iterdir()
    entry = json.loads(log_file.read_text())
    assert entry["changed"] is True and entry["turn"] == 1


async def test_noise_turn_gets_no_reply():
    completer = FakeCompleter(json.dumps({"transcript": "", "rationale": "artifact"}))
    agent = make_agent(completer)
    agent._pending = [seg(0, "Thank you.", empty("a"), ok("Thank you.", "b"))]

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(llm.ChatContext.empty(), user_message("Thank you."))


async def test_typed_input_is_left_alone():
    completer = FakeCompleter()
    agent = make_agent(completer)
    message = user_message("typed in the console")

    await agent.on_user_turn_completed(llm.ChatContext.empty(), message)

    assert message.text_content == "typed in the console"
    assert completer.requests == []


def test_later_segments_stay_for_the_next_turn():
    agent = make_agent(FakeCompleter())
    agent._pending = [seg(0, "book a table"), seg(1, "for four"), seg(2, "actually five")]

    taken = agent.take_segments("Book a table  for four")

    assert [s.index for s in taken] == [0, 1]
    assert [s.index for s in agent._pending] == [2]


def test_stale_segments_before_the_turn_are_discarded():
    agent = make_agent(FakeCompleter())
    agent._pending = [seg(0, "left over"), seg(1, "hello there")]

    assert [s.index for s in agent.take_segments("hello there")] == [1]
    assert agent._pending == []


def test_everything_is_used_when_nothing_matches():
    agent = make_agent(FakeCompleter())
    agent._pending = [seg(0, "one"), seg(1, "two")]

    assert [s.index for s in agent.take_segments("something else")] == [0, 1]
    assert agent._pending == []
