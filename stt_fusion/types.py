"""Plain data passed between the STT race, the agent and the fuser.

Everything converts to and from JSON-friendly dicts: segments travel inside
`SpeechData.metadata` and end up in `ChatMessage.extra` and the turn log.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

CandidateStatus = Literal["ok", "empty", "error", "timeout"]

# When an engine produced no text for a whole turn, report its most telling failure.
_FAILURE_RANK: dict[str, int] = {"error": 3, "timeout": 2, "empty": 1, "ok": 0}


@dataclass(frozen=True)
class Candidate:
    """One engine's transcript of one piece of audio."""

    engine: str
    text: str
    status: CandidateStatus
    latency_s: float
    confidence: float | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        return cls(**data)


@dataclass(frozen=True)
class SegmentResult:
    """The race result for one VAD speech segment."""

    index: int
    audio_s: float
    candidates: tuple[Candidate, ...]
    provisional: str
    """Consensus pick. The session receives this as the segment's FINAL transcript."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "audio_s": self.audio_s,
            "candidates": [c.to_dict() for c in self.candidates],
            "provisional": self.provisional,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentResult:
        return cls(
            index=data["index"],
            audio_s=data["audio_s"],
            candidates=tuple(Candidate.from_dict(c) for c in data["candidates"]),
            provisional=data["provisional"],
        )


def merge_turn(segments: Sequence[SegmentResult]) -> list[Candidate]:
    """Join each engine's text across the segments of one user turn.

    Engines keep the order in which they first appear. An engine is "ok" for the
    turn if it produced text for any segment; if it also failed on some segment,
    the failure is kept in `error` so the fuser knows that text may be incomplete.
    """
    per_engine: dict[str, list[Candidate]] = {}
    for segment in segments:
        for cand in segment.candidates:
            per_engine.setdefault(cand.engine, []).append(cand)

    merged: list[Candidate] = []
    for engine, cands in per_engine.items():
        texts = [c.text.strip() for c in cands if c.usable and c.text.strip()]
        confidences = [c.confidence for c in cands if c.usable and c.confidence is not None]
        failures = [c for c in cands if c.status in ("error", "timeout")]
        if texts:
            status: CandidateStatus = "ok"
        else:
            status = max((c.status for c in cands), key=lambda s: _FAILURE_RANK[s])
        error = None
        if failures:
            reasons = dict.fromkeys(c.error or c.status for c in failures)
            error = "; ".join(reasons)
            if texts:
                error = f"missed {len(failures)} of {len(cands)} segments ({error})"
        merged.append(
            Candidate(
                engine=engine,
                text=" ".join(texts),
                status=status,
                latency_s=max(c.latency_s for c in cands),
                confidence=sum(confidences) / len(confidences) if confidences else None,
                error=error,
            )
        )
    return merged
