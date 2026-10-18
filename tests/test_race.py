import time

from fakes import FakeStreamingSTT, FakeSTT, empty, failed, ok, silence

from stt_fusion.race import Engine, consensus_pick, race


async def test_results_keep_engine_order_and_failures_are_reported():
    engines = [
        Engine("slow", FakeSTT("hello world", delay=0.05, confidence=0.9)),
        Engine("broken", FakeSTT(error=RuntimeError("bad key"))),
        Engine("fast", FakeSTT("hello word")),
    ]
    results = await race(engines, silence(), timeout_s=2)

    assert [c.engine for c in results] == ["slow", "broken", "fast"]
    assert results[0].status == "ok" and results[0].text == "hello world"
    assert results[0].confidence == 0.9
    assert results[1].status == "error" and "bad key" in (results[1].error or "")
    assert results[2].status == "ok" and results[2].confidence is None


async def test_engines_run_concurrently():
    engines = [Engine(f"e{i}", FakeSTT("hi", delay=0.3)) for i in range(3)]
    started = time.perf_counter()
    await race(engines, silence(), timeout_s=2)
    assert time.perf_counter() - started < 0.6


async def test_slow_engine_times_out_without_holding_up_the_others():
    engines = [Engine("fast", FakeSTT("hi")), Engine("stuck", FakeSTT("late", delay=5))]
    started = time.perf_counter()
    results = await race(engines, silence(), timeout_s=0.2)

    assert time.perf_counter() - started < 1
    assert results[0].status == "ok"
    assert results[1].status == "timeout"


async def test_grace_window_starts_once_a_majority_has_answered():
    engines = [
        Engine("fast", FakeSTT("hi")),
        Engine("medium", FakeSTT("hi", delay=0.2)),
        Engine("slow", FakeSTT("hi", delay=2)),
    ]
    started = time.perf_counter()
    results = await race(engines, silence(), timeout_s=5, grace_s=0.1)

    assert time.perf_counter() - started < 1
    assert [c.status for c in results] == ["ok", "ok", "timeout"]


async def test_grace_window_still_waits_for_a_majority():
    # one fast answer is not a majority of three: the race waits for a second engine
    engines = [
        Engine("fast", FakeSTT("hi")),
        Engine("medium", FakeSTT("hi", delay=0.4)),
        Engine("slow", FakeSTT("hi", delay=2)),
    ]
    results = await race(engines, silence(), timeout_s=5, grace_s=0.1)
    assert [c.status for c in results] == ["ok", "ok", "timeout"]


async def test_empty_answer_is_not_an_error():
    results = await race([Engine("quiet", FakeSTT(""))], silence(), timeout_s=1)
    assert results[0].status == "empty"


async def test_streaming_only_engine_is_replayed_through_a_stream():
    results = await race([Engine("live", FakeStreamingSTT("streamed"))], silence(), timeout_s=2)
    assert results[0].status == "ok" and results[0].text == "streamed"


def test_consensus_picks_the_reading_closest_to_the_others():
    candidates = [
        ok("book a table at toyt", "a"),
        ok("book a table at toit", "b"),
        ok("book a table at toit tonight", "c"),
    ]
    assert consensus_pick(candidates) == "book a table at toit"


def test_consensus_ties_go_to_the_first_engine():
    assert consensus_pick([ok("yes", "a"), ok("yeah", "b")]) == "yes"


def test_consensus_drops_a_lone_hallucination():
    assert consensus_pick([empty("a"), empty("b"), ok("Thank you.", "c")]) == ""


def test_consensus_keeps_text_when_engines_split_evenly():
    assert consensus_pick([empty("a"), ok("yes", "b")]) == "yes"


def test_consensus_ignores_engines_that_failed():
    candidates = [failed("a"), failed("b", status="timeout"), ok("hello", "c")]
    assert consensus_pick(candidates) == "hello"
