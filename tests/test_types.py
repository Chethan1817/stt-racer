from fakes import empty, failed, ok

from stt_fusion.types import Candidate, SegmentResult, merge_turn


def segment(index: int, *candidates: Candidate) -> SegmentResult:
    return SegmentResult(index=index, audio_s=1.0, candidates=candidates, provisional="x")


def test_merge_joins_each_engines_text_across_segments():
    merged = merge_turn(
        [
            segment(0, ok("book a table", "a", 0.8), ok("book a cable", "b")),
            segment(1, ok("for four", "a", 0.6), ok("for for", "b")),
        ]
    )
    assert [(c.engine, c.text, c.status) for c in merged] == [
        ("a", "book a table for four", "ok"),
        ("b", "book a cable for for", "ok"),
    ]
    assert merged[0].confidence == 0.7
    assert merged[1].confidence is None


def test_engine_that_missed_a_segment_is_flagged():
    merged = merge_turn([segment(0, ok("hello", "a")), segment(1, failed("a", "timeout"))])
    assert merged[0].status == "ok"
    assert merged[0].error is not None and "missed 1 of 2" in merged[0].error


def test_engine_without_text_reports_its_worst_failure():
    merged = merge_turn([segment(0, empty("a")), segment(1, failed("a"))])
    assert merged[0].status == "error"
    merged = merge_turn([segment(0, empty("a")), segment(1, empty("a"))])
    assert merged[0].status == "empty" and merged[0].error is None


def test_segment_round_trips_through_a_dict():
    original = segment(3, ok("hi", "a", 0.5), failed("b"))
    assert SegmentResult.from_dict(original.to_dict()) == original
