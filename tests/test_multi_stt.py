"""MultiSTT end to end on a real clip: real local VAD, fake engines, no network."""

from pathlib import Path

import pytest
from fakes import FakeSTT, silence
from livekit.agents import inference, stt

from stt_fusion.multi_stt import SEGMENT_METADATA_KEY, MultiSTT, segment_from_event
from stt_fusion.race import Engine
from stt_fusion.wav import read_wav_frames

# "Book a table for four. <1.2 s pause> At seven thirty tonight." (macOS `say`)
TWO_PHRASES = Path(__file__).parent / "data" / "two_phrases.wav"


def make_multi(*engines: Engine) -> MultiSTT:
    return MultiSTT(engines=engines, vad=inference.VAD(min_silence_duration=0.6), timeout_s=2)


async def run_stream(multi: MultiSTT, path: Path) -> list[stt.SpeechEvent]:
    stream = multi.stream()
    for frame in read_wav_frames(path):
        stream.push_frame(frame)
    stream.end_input()
    events = [event async for event in stream]
    await stream.aclose()
    return events


async def test_one_final_per_speech_segment_with_every_engines_result():
    multi = make_multi(
        Engine("a", FakeSTT("same words")),
        Engine("b", FakeSTT("same words")),
        Engine("c", FakeSTT("")),
    )
    events = await run_stream(multi, TWO_PHRASES)

    types = [e.type for e in events]
    assert types.count(stt.SpeechEventType.START_OF_SPEECH) == 2
    finals = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert len(finals) == 2
    # each segment: START, END, then the FINAL once the race is over
    first_final = types.index(stt.SpeechEventType.FINAL_TRANSCRIPT)
    assert types[:first_final] == [
        stt.SpeechEventType.START_OF_SPEECH,
        stt.SpeechEventType.END_OF_SPEECH,
    ]

    segments = [segment_from_event(e) for e in finals]
    assert [s.index for s in segments if s] == [0, 1]
    first = segments[0]
    assert first is not None
    assert finals[0].alternatives[0].text == first.provisional == "same words"
    assert [c.engine for c in first.candidates] == ["a", "b", "c"]
    assert [c.status for c in first.candidates] == ["ok", "ok", "empty"]
    assert first.audio_s > 0.5


async def test_segment_most_engines_call_silence_is_dropped():
    multi = make_multi(
        Engine("a", FakeSTT("")),
        Engine("b", FakeSTT("")),
        Engine("whisper", FakeSTT("Thank you.")),
    )
    events = await run_stream(multi, TWO_PHRASES)
    assert not [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert [e.type for e in events].count(stt.SpeechEventType.END_OF_SPEECH) == 2


async def test_recognize_attaches_the_race_result():
    multi = make_multi(Engine("a", FakeSTT("hello")), Engine("b", FakeSTT(error=RuntimeError())))
    event = await multi.recognize(silence(1.0))
    assert event.alternatives[0].text == "hello"
    assert SEGMENT_METADATA_KEY in (event.alternatives[0].metadata or {})


def test_engine_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        make_multi(Engine("a", FakeSTT()), Engine("a", FakeSTT()))


async def test_opted_in_engines_get_a_warm_up_clip_when_speech_starts():
    warm, plain = FakeSTT("same words"), FakeSTT("same words")
    multi = make_multi(Engine("warm", warm, warm_up=True), Engine("plain", plain))
    await run_stream(multi, TWO_PHRASES)

    # two segments each; the warm engine also got one silent clip at the first start of
    # speech, but not at the second, because it had just been used
    assert plain.calls == 2
    assert warm.calls == 3
