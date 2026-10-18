"""Build the STT engines and the fusion LLM from Settings."""

from __future__ import annotations

from typing import Any

import aiohttp
from livekit.plugins import deepgram, groq, openai

from .config import Settings
from .fusion import TranscriptFuser, llm_completer
from .race import Engine


def build_engines(
    settings: Settings, *, http_session: aiohttp.ClientSession | None = None
) -> list[Engine]:
    """Create the engines listed in `settings.stt_engines`, in that order.

    `http_session` is only needed outside a LiveKit job (scripts, tests): the Deepgram
    plugin otherwise borrows the job's shared aiohttp session.
    """
    builders = {
        "deepgram": lambda: Engine(
            f"deepgram/{settings.deepgram_stt_model}",
            deepgram.STT(
                model=settings.deepgram_stt_model,
                language=settings.stt_language,
                smart_format=True,
                http_session=http_session,
            ),
            # measured: ~1.4 s for the first request after a pause, ~0.5 s once warm
            warm_up=True,
        ),
        "openai": lambda: Engine(
            f"openai/{settings.openai_stt_model}",
            openai.STT(
                model=settings.openai_stt_model,
                language=settings.stt_language,
                # Without this the model sometimes answers an accented speaker in another
                # script (Telugu, Japanese) even with the language set.
                prompt=settings.openai_stt_prompt,
            ),
        ),
        "groq": lambda: Engine(
            f"groq/{settings.groq_stt_model}",
            groq.STT(model=settings.groq_stt_model, language=settings.stt_language),
        ),
    }
    return [builders[name]() for name in settings.stt_engines]


def build_fuser(settings: Settings) -> TranscriptFuser:
    llm_options: dict[str, Any] = {"model": settings.fusion_llm_model}
    if settings.fusion_temperature is not None:
        llm_options["temperature"] = settings.fusion_temperature
    fusion_llm = openai.LLM(**llm_options)
    # The session prewarms its own LLM; this one is ours, so open its connection now
    # rather than on the first turn.
    fusion_llm.prewarm()
    return TranscriptFuser(
        llm_completer(fusion_llm),
        timeout_s=settings.fusion_timeout_s,
        context_turns=settings.fusion_context_turns,
        keyterms=settings.fusion_keyterms,
        skip_if_unanimous=settings.fusion_skip_if_unanimous,
    )
