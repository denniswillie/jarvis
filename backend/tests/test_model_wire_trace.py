import json
import os
from unittest.mock import AsyncMock

import httpx
import pytest

from evals.responsive_voice.model_wire_trace import (
    TracedSocket, WireTrace, creation_session, readable, render, sanitized_payload,
)


@pytest.fixture
def trace(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    value = WireTrace(root)
    yield value
    value.close()


@pytest.mark.asyncio
async def test_exact_payload_and_order_are_recorded_without_changing_wire(trace):
    raw = '{"type":"response.create", "response":{"instructions":"first\\nsecond", "tools":[{"name":"cancel_background_task","parameters":{"properties":{"jarvis_request_source":{"enum":["retained","current"]}}}}]}}'
    socket = AsyncMock()
    socket.recv.return_value = '{"type":"response.created","response":{"id":"r1"}}'
    wrapped = TracedSocket(socket, trace, "connection1")
    await wrapped.send(raw)
    assert await wrapped.recv() == socket.recv.return_value
    socket.send.assert_awaited_once_with(raw)
    trace.close()
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [row["stage"] for row in rows] == [
        "capture_started", "send_attempt", "send_returned", "receive", "capture_closed"]
    assert rows[1]["raw"] == raw
    assert rows[2]["attempt_sequence"] == rows[1]["sequence"]
    assert trace.path.stat().st_mode & 0o077 == 0
    result = render(trace.root, trace.root / "timeline.md")
    text = (trace.root / "timeline.md").read_text()
    assert result["incomplete"] is False
    assert "L0001 | first" in text and "L0002 | second" in text
    assert "jarvis_request_source.enum[0]" in text
    assert "cancel_background_task" in text
    assert "Direct phone WebRTC" in text


@pytest.mark.asyncio
async def test_failed_send_is_not_reported_as_returned(trace):
    socket = AsyncMock()
    socket.send.side_effect = TimeoutError()
    with pytest.raises(TimeoutError):
        await TracedSocket(socket, trace, "c").send('{"type":"response.cancel"}')
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert rows[-1]["stage"] == "send_failed"
    assert not any(row["stage"] == "send_returned" for row in rows)


def test_credential_redaction_does_not_change_original_input(trace):
    original = {"session":{"client_secret":{"value":"secret-value"},"instructions":"all prompt lines"}}
    trace.record("receive", raw=json.dumps(original))
    assert original["session"]["client_secret"]["value"] == "secret-value"
    assert "secret-value" not in trace.path.read_text()
    assert "all prompt lines" in trace.path.read_text()
    value, paths = sanitized_payload(original)
    assert paths == ["session.client_secret"]
    assert value["session"]["instructions"] == "all prompt lines"


@pytest.mark.asyncio
async def test_http_creation_extracts_only_session_not_sdp_or_headers():
    request = httpx.Request("POST", "https://api.openai.com/v1/realtime/calls",
        headers={"Authorization":"Bearer secret"}, files={
            "sdp": (None, "private SDP"),
            "session": (None, '{"model":"gpt-realtime-2.1"}', "application/json")})
    assert await creation_session(request) == '{"model":"gpt-realtime-2.1"}'
    assert b"private SDP" in request.content


def test_trace_bound_marks_incomplete(tmp_path):
    root = tmp_path / "bounded"
    root.mkdir(mode=0o700)
    trace = WireTrace(root, limit=1024)
    trace.record("send_attempt", raw=json.dumps({"instructions":"x" * 2048}))
    trace.close()
    assert trace.incomplete == "capture_byte_limit"
    assert "capture_incomplete" in trace.path.read_text()


def test_insecure_directory_rejected(tmp_path):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        WireTrace(root)


def test_readable_keeps_empty_values_and_multiline_text():
    lines = readable({"prompt":"first\n\nlast", "tools":[], "options":{}, "enabled":False})
    assert "$.prompt:L0002 | " in lines
    assert "$.tools = []" in lines and "$.options = {}" in lines
    assert "$.enabled = false" in lines
