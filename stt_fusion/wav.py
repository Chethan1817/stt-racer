"""Read WAV files into LiveKit audio frames (for the replay script and tests)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from livekit import rtc


def read_wav_frames(
    path: str | Path, *, frame_ms: int = 20, trailing_silence_s: float = 1.0
) -> list[rtc.AudioFrame]:
    """Load a 16-bit PCM WAV as mono frames, plus trailing silence so a VAD closes the
    last speech segment."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM audio")
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    samples = np.concatenate([samples, np.zeros(int(rate * trailing_silence_s), dtype=np.int16)])

    per_frame = rate * frame_ms // 1000
    padding = (-len(samples)) % per_frame
    samples = np.concatenate([samples, np.zeros(padding, dtype=np.int16)])
    return [
        rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=rate,
            num_channels=1,
            samples_per_channel=per_frame,
        )
        for chunk in samples.reshape(-1, per_frame)
    ]
