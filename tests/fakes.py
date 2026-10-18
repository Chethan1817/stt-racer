"""Offline stand-ins for STT engines, LLMs and audio input."""

from __future__ import annotations

import asyncio
from pathlib import Path

from livekit import rtc
from livekit.agents import APIConnectOptions, llm, stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.voice import io

from stt_fusion.types import Candidate
from stt_fusion.wav import read_wav_frames


class FakeSTT(stt.STT):
    """A batch STT that answers with fixed text after a delay, or raises."""

    def __init__(
        self,
        text: str = "",
        *,
        delay: float = 0.0,
        error: Exception | None = None,
        confidence: float = 0.0,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.text = text
        self.delay = delay
        self.error = error
        self.confidence = confidence
        self.calls = 0

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language="en", text=self.text, confidence=self.confidence)
            ],
        )


class FakeStreamingSTT(stt.STT):
    """A streaming-only STT: emits one FINAL with fixed text once its input ends."""

    def __init__(self, text: str) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True, interim_results=False, offline_recognize=False
            )
        )
        self.text = text

    async def _recognize_impl(self, *args: object, **kwargs: object) -> stt.SpeechEvent:
        raise NotImplementedError

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return _FakeStream(stt=self, conn_options=conn_options, text=self.text)


class _FakeStream(stt.RecognizeStream):
    def __init__(self, *, stt: stt.STT, conn_options: APIConnectOptions, text: str) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._text = text

    async def _run(self) -> None:
        async for _ in self._input_ch:
            pass
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text=self._text)],
            )
        )


class FakeCompleter:
    """Stands in for the fusion LLM: records each request and returns a fixed reply."""

    def __init__(
        self, reply: str = "", *, delay: float = 0.0, error: Exception | None = None
    ) -> None:
        self.reply = reply
        self.delay = delay
        self.error = error
        self.requests: list[llm.ChatContext] = []

    async def __call__(self, chat_ctx: llm.ChatContext) -> str:
        self.requests.append(chat_ctx)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


def silence(seconds: float = 0.5, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = int(seconds * sample_rate)
    return rtc.AudioFrame(
        data=b"\0\0" * samples,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def ok(text: str, engine: str = "e", confidence: float | None = None) -> Candidate:
    return Candidate(engine=engine, text=text, status="ok", latency_s=0.1, confidence=confidence)


def empty(engine: str = "e") -> Candidate:
    return Candidate(engine=engine, text="", status="empty", latency_s=0.1)


def failed(engine: str = "e", status: str = "error") -> Candidate:
    return Candidate(engine=engine, text="", status=status, latency_s=0.1, error="boom")  # type: ignore[arg-type]


class EchoLLM(llm.LLM):
    """Replies "You said: <latest user message>", which shows exactly what the LLM node got."""

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: object,
    ) -> llm.LLMStream:
        return _EchoStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _EchoStream(llm.LLMStream):
    async def _run(self) -> None:
        users = [m for m in self._chat_ctx.messages() if m.role == "user"]
        text = users[-1].text_content if users else ""
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="echo", delta=llm.ChoiceDelta(role="assistant", content=f"You said: {text}")
            )
        )


class WavAudioInput(io.AudioInput):
    """Plays a WAV file into an AgentSession in real time, then silence forever."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(label="wav")
        self._frames = read_wav_frames(path, trailing_silence_s=0)
        self._silence = rtc.AudioFrame(
            data=b"\0\0" * self._frames[0].samples_per_channel,
            sample_rate=self._frames[0].sample_rate,
            num_channels=1,
            samples_per_channel=self._frames[0].samples_per_channel,
        )
        self._next = 0

    async def __anext__(self) -> rtc.AudioFrame:
        await asyncio.sleep(0.02)
        if self._next < len(self._frames):
            self._next += 1
            return self._frames[self._next - 1]
        return self._silence
