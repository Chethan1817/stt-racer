import json

from fakes import FakeCompleter, empty, failed, ok
from livekit.agents import llm

from stt_fusion.fusion import FUSION_INSTRUCTIONS, TranscriptFuser


def reply(transcript: str, rationale: str = "engines agreed") -> str:
    return json.dumps({"transcript": transcript, "rationale": rationale})


async def test_uses_the_llm_transcript():
    completer = FakeCompleter(reply("Book a table at Toit.", "toyt vs Toit"))
    outcome = await TranscriptFuser(completer).fuse(
        [ok("book a table at toyt", "a"), ok("Book a table at Toit.", "b")],
        fallback="book a table at toyt",
    )
    assert outcome.source == "llm"
    assert outcome.text == "Book a table at Toit."
    assert outcome.rationale == "toyt vs Toit"


async def test_tolerates_text_around_the_json():
    completer = FakeCompleter("```json\n" + reply("hello there") + "\n```")
    outcome = await TranscriptFuser(completer).fuse([ok("hello there")], fallback="x")
    assert outcome.source == "llm" and outcome.text == "hello there"


async def test_falls_back_on_invalid_json():
    completer = FakeCompleter('{"transcript": ')
    outcome = await TranscriptFuser(completer).fuse([ok("hello")], fallback="hello")
    assert outcome.source == "fallback" and outcome.text == "hello"


async def test_falls_back_when_the_llm_call_fails():
    completer = FakeCompleter(error=RuntimeError("model not found"))
    outcome = await TranscriptFuser(completer).fuse([ok("hello")], fallback="hello")
    assert outcome.source == "fallback"
    assert "model not found" in (outcome.error or "")


async def test_falls_back_on_timeout():
    completer = FakeCompleter(reply("hello"), delay=1)
    outcome = await TranscriptFuser(completer, timeout_s=0.05).fuse([ok("hello")], fallback="hi")
    assert outcome.source == "fallback" and outcome.text == "hi"
    assert "timed out" in (outcome.error or "")


async def test_rejects_an_answer_instead_of_a_transcript():
    answer = "Sure! The capital of France is Paris, a city famous for the Eiffel Tower and food."
    completer = FakeCompleter(reply(answer))
    outcome = await TranscriptFuser(completer).fuse(
        [ok("what is the capital of france", "a"), ok("what's the capital of france", "b")],
        fallback="what is the capital of france",
    )
    assert outcome.source == "fallback"
    assert outcome.text == "what is the capital of france"
    assert (outcome.error or "").startswith("rejected")


async def test_rejects_empty_output_when_engines_clearly_heard_speech():
    completer = FakeCompleter(reply(""))
    outcome = await TranscriptFuser(completer).fuse(
        [ok("call me at five", "a"), ok("call me at nine", "b")], fallback="call me at five"
    )
    assert outcome.source == "fallback" and outcome.text == "call me at five"


async def test_accepts_empty_output_for_noise():
    completer = FakeCompleter(reply("", "only one engine heard 'Thank you.'"))
    outcome = await TranscriptFuser(completer).fuse(
        [empty("a"), ok("Thank you.", "b")], fallback="Thank you."
    )
    assert outcome.source == "llm" and outcome.text == ""


async def test_skips_the_llm_when_no_engine_returned_text():
    completer = FakeCompleter(reply("never used"))
    outcome = await TranscriptFuser(completer).fuse([failed("a"), empty("b")], fallback="")
    assert outcome.source == "skipped"
    assert completer.requests == []


async def test_skip_if_unanimous_avoids_the_call_when_engines_agree():
    completer = FakeCompleter(reply("never used"))
    fuser = TranscriptFuser(completer, skip_if_unanimous=True)
    outcome = await fuser.fuse([ok("Hello there.", "a"), ok("hello there", "b")], fallback="hi")
    assert outcome.source == "skipped" and completer.requests == []


async def test_prompt_carries_candidates_context_and_key_terms():
    completer = FakeCompleter(reply("chethan.rao@gmail.com"))
    history = llm.ChatContext.empty()
    history.add_message(role="assistant", content="What's your email address?")
    fuser = TranscriptFuser(completer, keyterms=["Chethan"])

    await fuser.fuse(
        [
            ok("chetan dot rao at gmail dot com", "deepgram", confidence=0.91),
            empty("openai"),
            failed("groq"),
        ],
        fallback="chetan dot rao at gmail dot com",
        chat_ctx=history,
    )

    system, user = completer.requests[0].items
    assert system.text_content == FUSION_INSTRUCTIONS
    payload = json.loads(user.text_content)
    assert payload["recent_conversation"] == [
        {"role": "assistant", "text": "What's your email address?"}
    ]
    assert payload["key_terms"] == ["Chethan"]
    # the engine that heard nothing is evidence; the one that errored is not
    assert payload["candidates"] == [
        {"engine": "deepgram", "transcript": "chetan dot rao at gmail dot com", "confidence": 0.91},
        {"engine": "openai", "transcript": ""},
    ]


async def test_key_terms_keep_the_llm_in_the_loop_even_when_engines_agree():
    completer = FakeCompleter(reply("Chethan here"))
    fuser = TranscriptFuser(completer, skip_if_unanimous=True, keyterms=["Chethan"])
    outcome = await fuser.fuse([ok("Chetan here", "a"), ok("chetan here", "b")], fallback="x")
    assert outcome.source == "llm" and outcome.text == "Chethan here"
