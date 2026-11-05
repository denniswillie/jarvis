"""OpenAI PCM transport for expert-authored speech.

This module does not own tasks, persist intents, select when to speak, or claim
that audio was heard. The attention coordinator must commit exact input and
attempt identity in authorize_submission, then recheck its source/claim under
the common phone writer as well as through is_current below.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import time

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError


SAMPLE_RATE = 24_000
SAMPLE_BYTES = 2
DEFAULT_INSTRUCTIONS = (
    "Speak only the supplied words clearly. Preserve spoken punctuation, units, "
    "and uncertainty. Do not add commentary."
)


class SpeechError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None):
        self.code = code
        self.status_code = status_code
        super().__init__(code)


def provider_error(error: APIError) -> SpeechError:
    if isinstance(error, APITimeoutError):
        return SpeechError("tts_timeout")
    if isinstance(error, APIConnectionError):
        return SpeechError("tts_connection_failed")
    status = error.status_code if isinstance(error, APIStatusError) else None
    code = {
        400: "tts_configuration_rejected",
        401: "tts_authentication_failed",
        403: "tts_access_denied",
        404: "tts_configuration_unavailable",
        429: "tts_rate_limited",
    }.get(status, "tts_provider_failed")
    return SpeechError(code, status_code=status)


@dataclass(frozen=True, slots=True)
class SpeechConfiguration:
    model: str = "gpt-4o-mini-tts-2025-12-15"
    voice: str = "cedar"
    timeout_seconds: float = 45.0
    # Deliberately conservative operational cap, not a claim that characters
    # equal tokens. The endpoint allows 4096 characters and the model 2000
    # input tokens. Bound the combined text/instructions by UTF-8 bytes until
    # a documented exact tokenizer contract is available.
    max_input_utf8_bytes: int = 1800
    max_audio_seconds: float = 240.0
    max_frame_bytes: int = 24_000

    def __post_init__(self):
        if not self.model.strip() or not self.voice.strip():
            raise ValueError("Speech model and voice must be configured")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Speech timeout must be finite and positive")
        if not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0:
            raise ValueError("Speech duration bound must be finite and positive")
        if not 1 <= self.max_input_utf8_bytes <= 2000:
            raise ValueError("Speech UTF-8 input bound must be between 1 and 2000")
        if self.max_frame_bytes < SAMPLE_BYTES or self.max_frame_bytes % SAMPLE_BYTES:
            raise ValueError("Speech frame bound must contain whole PCM samples")


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
    text: str
    model: str
    voice: str
    instructions: str = DEFAULT_INSTRUCTIONS

    @property
    def digest(self) -> str:
        content = json.dumps({
            "text": self.text, "model": self.model, "voice": self.voice,
            "instructions": self.instructions, "format": "pcm_s16le",
            "sample_rate": SAMPLE_RATE, "channels": 1,
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SpeechFrame:
    data: bytes
    start_sample: int

    @property
    def end_sample(self) -> int:
        return self.start_sample + len(self.data) // SAMPLE_BYTES


class SpeechStream:
    def __init__(self, response, configuration: SpeechConfiguration, is_current, started_at):
        self._response = response
        self._configuration = configuration
        self._is_current = is_current
        self._started_at = started_at
        self._iterated = False
        self.closed = False
        self.generation_complete = False
        self.provider_request_id = response.headers.get("x-request-id")
        self.first_bytes_ms: float | None = None
        self.received_bytes = 0
        self.samples_yielded = 0

    async def _chunks(self):
        try:
            async for data in self._response.iter_bytes():
                yield data
        except APIError as error:
            # Provider messages may contain input text. Do not expose them as
            # tool/public error strings or silently retry a partial utterance.
            raise provider_error(error) from None

    async def _check_current(self):
        if self.closed:
            raise SpeechError("tts_stream_closed")
        if not await self._is_current():
            raise SpeechError("tts_source_revoked")

    async def frames(self) -> AsyncIterator[SpeechFrame]:
        if self._iterated:
            raise SpeechError("tts_stream_already_consumed")
        self._iterated = True
        await self._check_current()
        pending = b""
        async for data in self._chunks():
            if not data:
                continue
            if self.first_bytes_ms is None:
                self.first_bytes_ms = (time.perf_counter() - self._started_at) * 1000
            self.received_bytes += len(data)
            if self.received_bytes > self._configuration.max_audio_seconds * SAMPLE_RATE * SAMPLE_BYTES:
                raise SpeechError("tts_audio_limit_exceeded")
            pending += data
            # HTTP packet boundaries are not playback frame boundaries. Tiny
            # packets would otherwise each incur the durable phone-writer
            # authorization, making delivery slower than the audio itself.
            # Buffer at most one frame between reads, then flush the EOF tail.
            usable = len(pending) // self._configuration.max_frame_bytes * self._configuration.max_frame_bytes
            for start in range(0, usable, self._configuration.max_frame_bytes):
                await self._check_current()
                piece = pending[start:min(start + self._configuration.max_frame_bytes, usable)]
                frame = SpeechFrame(piece, self.samples_yielded)
                self.samples_yielded = frame.end_sample
                yield frame
            pending = pending[usable:]
        usable = len(pending) - len(pending) % SAMPLE_BYTES
        if usable:
            await self._check_current()
            frame = SpeechFrame(pending[:usable], self.samples_yielded)
            self.samples_yielded = frame.end_sample
            yield frame
            pending = pending[usable:]
        if pending:
            raise SpeechError("tts_partial_pcm_sample")
        if not self.samples_yielded:
            raise SpeechError("tts_empty_audio")
        await self._check_current()
        self.generation_complete = True


class OpenAISpeechProvider:
    def __init__(self, client, configuration: SpeechConfiguration | None = None):
        self.configuration = configuration or SpeechConfiguration()
        # Delivery retries belong to the durable coordinator, never the SDK.
        self._client = client.with_options(max_retries=0, timeout=self.configuration.timeout_seconds)

    def prepare(self, text: str, *, instructions: str = DEFAULT_INSTRUCTIONS) -> PreparedSpeech:
        request = PreparedSpeech(text, self.configuration.model, self.configuration.voice, instructions)
        self._validate(request)
        return request

    def _validate(self, request: PreparedSpeech):
        if not request.model.strip() or not request.voice.strip() or not request.text.strip():
            raise SpeechError("tts_input_invalid")
        if len(request.text) > 4096 or len(request.instructions) > 4096:
            raise SpeechError("tts_input_limit_exceeded")
        size = len(request.text.encode("utf-8")) + len(request.instructions.encode("utf-8"))
        if size > self.configuration.max_input_utf8_bytes:
            raise SpeechError("tts_input_limit_exceeded")

    @asynccontextmanager
    async def open_stream(
        self,
        request: PreparedSpeech,
        *,
        authorize_submission: Callable[[PreparedSpeech], Awaitable[bool]],
        is_current: Callable[[], Awaitable[bool]],
    ):
        """Close HTTP on consumer exit/cancellation, including early breaks.

        Authorization must durably bind this exact request digest and source
        revision to a current delivery attempt before permitting network I/O.
        Yielding PCM is not a phone-write, playback, or hearing receipt.
        """
        self._validate(request)
        if not await authorize_submission(request):
            raise SpeechError("tts_submission_revoked")
        started = time.perf_counter()
        async with AsyncExitStack() as stack:
            try:
                response = await stack.enter_async_context(
                    self._client.audio.speech.with_streaming_response.create(
                        model=request.model, voice=request.voice, input=request.text,
                        instructions=request.instructions, response_format="pcm", stream_format="audio",
                    )
                )
            except APIError as error:
                raise provider_error(error) from None
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in {"audio/pcm", "application/octet-stream"}:
                raise SpeechError("tts_unexpected_format")
            stream = SpeechStream(response, self.configuration, is_current, started)
            try:
                yield stream
            finally:
                stream.closed = True
