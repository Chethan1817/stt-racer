"""Replay WAV files through the STT race and the fusion step, with no LiveKit room or mic.

Each file is treated as one user turn and goes through the same MultiSTT stream
(VAD segmentation + race) and TranscriptFuser the agent uses.

    uv run scripts/replay_wav.py samples/*.wav
    uv run scripts/replay_wav.py --context "What's your email address?" samples/email.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aiohttp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from livekit.agents import inference, llm  # noqa: E402

from stt_fusion.config import Settings  # noqa: E402
from stt_fusion.multi_stt import MultiSTT, segment_from_event  # noqa: E402
from stt_fusion.providers import build_engines, build_fuser  # noqa: E402
from stt_fusion.recorder import format_turn  # noqa: E402
from stt_fusion.types import SegmentResult, merge_turn  # noqa: E402
from stt_fusion.wav import read_wav_frames  # noqa: E402


async def race_file(multi: MultiSTT, path: Path) -> list[SegmentResult]:
    stream = multi.stream()
    for frame in read_wav_frames(path):
        stream.push_frame(frame)
    stream.end_input()
    segments = [seg async for event in stream if (seg := segment_from_event(event))]
    await stream.aclose()
    return segments


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wavs", nargs="+", type=Path, help="16-bit PCM WAV files")
    parser.add_argument("--context", help="assistant line the user is answering (helps fusion)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show library logs")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR, format="%(levelname)s %(message)s"
    )
    logging.getLogger("stt-fusion").setLevel(logging.INFO)
    settings = Settings.from_env()
    fuser = build_fuser(settings)
    history = llm.ChatContext.empty()
    if args.context:
        history.add_message(role="assistant", content=args.context)

    async with aiohttp.ClientSession() as http:
        multi = MultiSTT(
            engines=build_engines(settings, http_session=http),
            vad=inference.VAD(min_silence_duration=settings.segment_min_silence_s),
            timeout_s=settings.race_timeout_s,
            grace_s=settings.race_grace_s,
            language=settings.stt_language,
        )
        for turn, path in enumerate(args.wavs, start=1):
            print(f"\n=== {path}")
            segments = await race_file(multi, path)
            if not segments:
                print("no transcript: no speech, judged as noise, or every engine failed (see log)")
                continue
            heard = " ".join(s.provisional for s in segments)
            candidates = merge_turn(segments)
            outcome = await fuser.fuse(candidates, fallback=heard, chat_ctx=history)
            print(format_turn(turn, segments, candidates, heard, outcome))


if __name__ == "__main__":
    asyncio.run(main())
