"""Per-turn output: a readable block in the log and one JSON line per turn on disk."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .fusion import FusionOutcome
from .types import Candidate, SegmentResult

logger = logging.getLogger("stt-fusion")


class TurnRecorder:
    def __init__(self, log_dir: Path | None, *, session_name: str) -> None:
        self._turn = 0
        self._path: Path | None = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^\w.-]+", "_", session_name) or "session"
            self._path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{safe_name}.jsonl"

    @property
    def path(self) -> Path | None:
        return self._path

    def record(
        self,
        *,
        segments: Sequence[SegmentResult],
        candidates: Sequence[Candidate],
        heard: str,
        outcome: FusionOutcome,
    ) -> dict[str, Any]:
        """Log the turn, append it to the JSONL file, and return the logged entry."""
        self._turn += 1
        logger.info(
            "STT race + fusion\n%s", format_turn(self._turn, segments, candidates, heard, outcome)
        )
        entry = {
            "turn": self._turn,
            "time": time.time(),
            "segments": [s.to_dict() for s in segments],
            "candidates": [c.to_dict() for c in candidates],
            "heard": heard,
            "fusion": outcome.to_dict(),
            "changed": _changed(heard, outcome.text),
        }
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def format_turn(
    turn: int,
    segments: Sequence[SegmentResult],
    candidates: Sequence[Candidate],
    heard: str,
    outcome: FusionOutcome,
) -> str:
    audio_s = sum(s.audio_s for s in segments)
    labels = [c.engine for c in candidates] + ["consensus", "fused"]
    width = max(len(label) for label in labels)
    lines = [f"turn {turn}: {len(segments)} segment(s), {audio_s:.1f}s of speech"]
    for c in candidates:
        lines.append(f"  {c.engine:<{width}}  {c.latency_s * 1000:6.0f} ms  {_describe(c)}")
    lines.append(f"  {'consensus':<{width}}  {'':>9}  {heard!r}")
    tag = outcome.source + (", changed" if _changed(heard, outcome.text) else "")
    lines.append(
        f"  {'fused':<{width}}  {outcome.latency_s * 1000:6.0f} ms  {outcome.text!r}  [{tag}]"
    )
    if outcome.rationale:
        lines.append(f"  {'':<{width}}  why: {outcome.rationale}")
    if outcome.error:
        lines.append(f"  {'':<{width}}  fallback reason: {outcome.error}")
    return "\n".join(lines)


def _describe(c: Candidate) -> str:
    if c.status == "ok":
        return repr(c.text) + (f"  (note: {c.error})" if c.error else "")
    if c.status == "empty":
        return "(heard nothing)"
    return f"({c.status}: {c.error})"


def _changed(before: str, after: str) -> bool:
    return " ".join(before.split()) != " ".join(after.split())
