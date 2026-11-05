from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

from openai import APIStatusError
import pytest

from app.attention.tts import (
    OpenAISpeechProvider, PreparedSpeech, SpeechConfiguration, SpeechError,
)


async def allow(*_):
    return True


class FakeResponse:
    def __init__(self, chunks, *, content_type="audio/pcm", read_error=None):
        self.headers = {"content-type": content_type, "x-request-id": "synthetic"}
        self.http_response = SimpleNamespace(is_closed=False)
        self.chunks = chunks
        self.read_error = read_error

    async def iter_bytes(self):
        for chunk in self.chunks:
            yield chunk
        if self.read_error:
            raise self.read_error


class FakeClient:
    def __init__(self, response, *, opening_error=None):
        self.response = response
        self.opening_error = opening_error
        self.calls = []
        self.options = None
        self.audio = SimpleNamespace(speech=SimpleNamespace(with_streaming_response=self))

    def with_options(self, **options):
        self.options = options
        return self

    @asynccontextmanager
    async def create(self, **parameters):
        self.calls.append(parameters)
        if self.opening_error:
            raise self.opening_error
        try:
            yield self.response
        finally:
            self.response.http_response.is_closed = True


def setup(chunks=(b"\x01\x02",), *, response=None, **configuration):
    raw = response or FakeResponse(chunks)
    client = FakeClient(raw)
    provider = OpenAISpeechProvider(client, SpeechConfiguration(**configuration))
    return provider, client, raw


@pytest.mark.asyncio
async def test_odd_network_boundaries_preserve_every_pcm_sample():
    provider, client, raw = setup((b"\x01", b"\x02\x03", b"\x04", b""))
    request = provider.prepare("Read these supplied words.")
    admitted = []

    async def authorize(value):
        admitted.append(value.digest)
        assert client.calls == []
        return True

    async with provider.open_stream(request, authorize_submission=authorize, is_current=allow) as stream:
        frames = [frame async for frame in stream.frames()]
        assert b"".join(f.data for f in frames) == b"\x01\x02\x03\x04"
        assert [f.start_sample for f in frames] == [0]
        assert stream.generation_complete
        assert stream.samples_yielded == 2
        assert stream.first_bytes_ms is not None
        assert stream.provider_request_id == "synthetic"
    assert admitted == [request.digest]
    assert stream.closed and raw.http_response.is_closed
    assert client.options == {"max_retries": 0, "timeout": 45.0}
    assert client.calls == [{
        "model": request.model, "voice": request.voice, "input": request.text,
        "instructions": request.instructions, "response_format": "pcm", "stream_format": "audio",
    }]


@pytest.mark.asyncio
async def test_large_network_chunk_is_bounded_and_fenced_per_frame():
    provider, _, _ = setup((bytes(range(10)),), max_frame_bytes=4)
    checks = 0

    async def current():
        nonlocal checks
        checks += 1
        return True

    async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=current) as stream:
        frames = [frame async for frame in stream.frames()]
    assert [len(f.data) for f in frames] == [4, 4, 2]
    assert [f.start_sample for f in frames] == [0, 2, 4]
    assert checks == 5  # initial, every frame, and normal generation completion


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks,code", [
    ((b"\x01",), "tts_partial_pcm_sample"),
    ((b"",), "tts_empty_audio"),
])
async def test_malformed_or_empty_pcm_does_not_complete(chunks, code):
    provider, client, raw = setup(chunks)
    with pytest.raises(SpeechError) as raised:
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
            [frame async for frame in stream.frames()]
    assert raised.value.code == code
    assert not stream.generation_complete
    assert raw.http_response.is_closed
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_denied_submission_has_no_provider_call():
    provider, client, _ = setup()

    async def deny(_):
        return False

    with pytest.raises(SpeechError, match="tts_submission_revoked"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=deny, is_current=allow):
            pytest.fail("denied stream opened")
    assert client.calls == []


@pytest.mark.asyncio
async def test_revocation_between_frames_stops_buffered_audio():
    provider, _, raw = setup((bytes(range(8)),), max_frame_bytes=4)
    current = True

    async def is_current():
        return current

    with pytest.raises(SpeechError, match="tts_source_revoked"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=is_current) as stream:
            frames = stream.frames()
            first = await anext(frames)
            assert first.data == bytes(range(4))
            current = False
            await anext(frames)
    assert stream.samples_yielded == 2
    assert not stream.generation_complete
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_revoked_source_never_yields_first_audio():
    provider, _, raw = setup()

    async def deny():
        return False

    with pytest.raises(SpeechError, match="tts_source_revoked"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=deny) as stream:
            await anext(stream.frames())
    assert stream.samples_yielded == 0
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_early_consumer_exit_closes_http_without_claiming_completion():
    provider, _, raw = setup((b"\x01\x02", b"\x03\x04"))
    async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
        iterator = stream.frames()
        await anext(iterator)
    assert raw.http_response.is_closed and stream.closed
    assert not stream.generation_complete
    with pytest.raises(SpeechError, match="tts_stream_closed"):
        await anext(iterator)


@pytest.mark.asyncio
async def test_consumer_error_is_not_misclassified_as_provider_error():
    provider, _, raw = setup()
    with pytest.raises(ValueError, match="consumer failure"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow):
            raise ValueError("consumer failure")
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_cancellation_while_reading_closes_http():
    entered = asyncio.Event()
    wait = asyncio.Event()

    class WaitingResponse(FakeResponse):
        async def iter_bytes(self):
            entered.set()
            await wait.wait()
            yield b"\x01\x02"

    provider, _, raw = setup(response=WaitingResponse(()))

    async def consume():
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
            [frame async for frame in stream.frames()]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert raw.http_response.is_closed


def status_error(status):
    response = SimpleNamespace(status_code=status, headers={}, request=SimpleNamespace())
    return APIStatusError("sensitive provider echo", response=response, body=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [
    (400, "tts_configuration_rejected"), (401, "tts_authentication_failed"),
    (403, "tts_access_denied"), (404, "tts_configuration_unavailable"),
    (429, "tts_rate_limited"), (500, "tts_provider_failed"),
])
async def test_provider_rejection_is_explicit_without_retry_or_text_leak(status, code):
    provider, client, _ = setup()
    client.opening_error = status_error(status)
    with pytest.raises(SpeechError) as raised:
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow):
            pytest.fail("provider rejection opened a stream")
    assert raised.value.code == code
    assert raised.value.status_code == status
    assert "sensitive" not in str(raised.value)
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_midstream_provider_failure_is_not_replayed():
    provider, client, raw = setup(response=FakeResponse((b"\x01\x02",), read_error=status_error(500)), max_frame_bytes=2)
    with pytest.raises(SpeechError, match="tts_provider_failed"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
            frames = stream.frames()
            await anext(frames)
            await anext(frames)
    assert len(client.calls) == 1
    assert stream.samples_yielded == 1 and not stream.generation_complete
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_wrong_content_type_is_rejected_and_closed():
    provider, _, raw = setup(response=FakeResponse((b"{}",), content_type="application/json"))
    with pytest.raises(SpeechError, match="tts_unexpected_format"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow):
            pytest.fail("JSON is not PCM")
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_response_audio_bound_closes_excessive_stream():
    provider, _, raw = setup((bytes(100),), max_audio_seconds=0.001)
    with pytest.raises(SpeechError, match="tts_audio_limit_exceeded"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
            [frame async for frame in stream.frames()]
    assert raw.http_response.is_closed
    assert stream.samples_yielded == 0


@pytest.mark.asyncio
async def test_stream_cannot_be_consumed_twice():
    provider, _, _ = setup()
    async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
        [frame async for frame in stream.frames()]
        with pytest.raises(SpeechError, match="tts_stream_already_consumed"):
            await anext(stream.frames())


@pytest.mark.parametrize("configuration", [
    {"model": ""}, {"voice": " "}, {"timeout_seconds": 0},
    {"timeout_seconds": float("inf")}, {"max_audio_seconds": -1},
    {"max_input_utf8_bytes": 2001}, {"max_frame_bytes": 3},
])
def test_invalid_configuration_is_rejected_on_explicit_construction(configuration):
    with pytest.raises(ValueError):
        SpeechConfiguration(**configuration)


def test_exact_text_and_configuration_digest_and_conservative_input_limit():
    provider, client, _ = setup()
    request = provider.prepare("Words")
    assert request.digest == replace(request).digest
    for changed in (replace(request, text="Other words"), replace(request, voice="other"),
                    replace(request, instructions="Other instructions"), replace(request, model="other")):
        assert changed.digest != request.digest
    with pytest.raises(SpeechError, match="tts_input_limit_exceeded"):
        provider.prepare("\u20ac" * 700)
    with pytest.raises(SpeechError, match="tts_input_limit_exceeded"):
        provider.prepare("a" * 4097)
    with pytest.raises(SpeechError, match="tts_input_invalid"):
        provider.prepare(" ")
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("packet_bytes", [1, 2, 3, 7, 24])
async def test_fragmented_packets_produce_bounded_frames_not_one_write_per_packet(packet_bytes):
    audio = bytes(range(22))
    chunks = tuple(audio[i:i + packet_bytes] for i in range(0, len(audio), packet_bytes))
    provider, _, _ = setup(chunks, max_frame_bytes=8)
    async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
        frames = [frame async for frame in stream.frames()]
    assert [len(frame.data) for frame in frames] == [8, 8, 6]
    assert [frame.start_sample for frame in frames] == [0, 4, 8]
    assert b"".join(frame.data for frame in frames) == audio
    assert stream.samples_yielded == 11 and stream.generation_complete


@pytest.mark.asyncio
async def test_buffered_eof_tail_is_not_forwarded_after_revocation():
    current = True

    class RevokingResponse(FakeResponse):
        async def iter_bytes(self):
            nonlocal current
            yield b"\x01\x02"
            current = False

    async def is_current():
        return current

    provider, _, raw = setup(response=RevokingResponse(()))
    with pytest.raises(SpeechError, match="tts_source_revoked"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=is_current) as stream:
            await anext(stream.frames())
    assert stream.samples_yielded == 0 and not stream.generation_complete
    assert raw.http_response.is_closed


@pytest.mark.asyncio
async def test_odd_eof_tail_preserves_whole_samples_but_never_completes():
    provider, _, raw = setup((bytes(range(11)),), max_frame_bytes=8)
    frames = []
    with pytest.raises(SpeechError, match="tts_partial_pcm_sample"):
        async with provider.open_stream(provider.prepare("Words"), authorize_submission=allow, is_current=allow) as stream:
            async for frame in stream.frames():
                frames.append(frame)
    assert b"".join(frame.data for frame in frames) == bytes(range(10))
    assert stream.samples_yielded == 5 and not stream.generation_complete
    assert raw.http_response.is_closed
