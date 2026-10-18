"""Settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

KNOWN_ENGINES = ("deepgram", "openai", "groq")


@dataclass(frozen=True)
class Settings:
    stt_engines: tuple[str, ...] = KNOWN_ENGINES
    stt_language: str = "en"
    deepgram_stt_model: str = "nova-3"
    openai_stt_model: str = "gpt-4o-transcribe"
    openai_stt_prompt: str = (
        "A person speaking English, possibly with an Indian or other non-native accent. "
        "Transcribe in English."
    )
    groq_stt_model: str = "whisper-large-v3"

    race_timeout_s: float = 10.0
    race_grace_s: float | None = 2.0
    segment_min_silence_s: float = 0.4

    fusion_llm_model: str = "gpt-4.1-mini"
    fusion_temperature: float | None = 0.0  # unset it for reasoning models such as gpt-5.5
    fusion_timeout_s: float = 15.0
    fusion_context_turns: int = 6
    fusion_keyterms: tuple[str, ...] = ()
    fusion_skip_if_unanimous: bool = False  # the brief asks for the LLM pass on every turn

    agent_llm_model: str = "gpt-4.1-mini"
    tts_model: str = "aura-2-andromeda-en"
    turn_detection: Literal["eou", "vad"] = "eou"
    log_dir: Path | None = Path("logs")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        d = cls()

        def text(name: str, default: str) -> str:
            return env.get(name, "").strip() or default

        def number(name: str, default: float | None) -> float | None:
            raw = env.get(name)
            if raw is None:
                return default
            raw = raw.strip()
            return float(raw) if raw else None

        engines = tuple(
            e.strip().lower() for e in text("STT_ENGINES", ",".join(d.stt_engines)).split(",")
        )
        engines = tuple(e for e in engines if e)
        if unknown := sorted(set(engines) - set(KNOWN_ENGINES)):
            raise ValueError(f"STT_ENGINES has unknown engines {unknown}; use {KNOWN_ENGINES}")
        if not engines:
            raise ValueError("STT_ENGINES must list at least one engine")

        turn_detection = text("TURN_DETECTION", d.turn_detection).lower()
        if turn_detection not in ("eou", "vad"):
            raise ValueError("TURN_DETECTION must be 'eou' or 'vad'")

        log_dir = env.get("LOG_DIR")
        return cls(
            stt_engines=engines,
            stt_language=text("STT_LANGUAGE", d.stt_language),
            deepgram_stt_model=text("DEEPGRAM_STT_MODEL", d.deepgram_stt_model),
            openai_stt_model=text("OPENAI_STT_MODEL", d.openai_stt_model),
            openai_stt_prompt=text("OPENAI_STT_PROMPT", d.openai_stt_prompt),
            groq_stt_model=text("GROQ_STT_MODEL", d.groq_stt_model),
            race_timeout_s=number("RACE_TIMEOUT_S", d.race_timeout_s) or d.race_timeout_s,
            race_grace_s=number("RACE_GRACE_S", d.race_grace_s),
            segment_min_silence_s=number("SEGMENT_MIN_SILENCE_S", d.segment_min_silence_s)
            or d.segment_min_silence_s,
            fusion_llm_model=text("FUSION_LLM_MODEL", d.fusion_llm_model),
            fusion_temperature=number("FUSION_TEMPERATURE", d.fusion_temperature),
            fusion_timeout_s=number("FUSION_TIMEOUT_S", d.fusion_timeout_s) or d.fusion_timeout_s,
            fusion_context_turns=int(number("FUSION_CONTEXT_TURNS", d.fusion_context_turns) or 0),
            fusion_keyterms=tuple(
                t.strip() for t in env.get("FUSION_KEYTERMS", "").split(",") if t.strip()
            ),
            fusion_skip_if_unanimous=text("FUSION_SKIP_IF_UNANIMOUS", "false").lower()
            in ("1", "true", "yes"),
            agent_llm_model=text("AGENT_LLM_MODEL", d.agent_llm_model),
            tts_model=text("TTS_MODEL", d.tts_model),
            turn_detection=turn_detection,  # type: ignore[arg-type]
            log_dir=d.log_dir if log_dir is None else (Path(log_dir) if log_dir.strip() else None),
        )
