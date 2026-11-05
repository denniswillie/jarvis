"""Server-owned call setup and control; microphone/speaker media bypass Jarvis."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from urllib.parse import urlencode, urlsplit

import httpx
import websockets

from app.realtime.openai_client import OpenAIRealtimeConnection, safety_identifier_for_owner


LOGGER = logging.getLogger(__name__)
CALLS_URL = "https://api.openai.com/v1/realtime/calls"
MAX_SDP_BYTES = 96_000
MAX_ERROR_METADATA_BYTES = 4096
LOCAL_CLOSE_TIMEOUT_SECONDS = 1.0
HANGUP_TIMEOUT_SECONDS = 5.0
HANGUP_HTTP_TIMEOUT_SECONDS = 3.0
RESPONSE_CLOSE_TIMEOUT_SECONDS = 1.0
HTTP_CLOSE_TIMEOUT_SECONDS = 1.0
SAFE_TRANSPORT_EXCEPTIONS = frozenset({
    "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "ConnectError", "ReadError",
    "WriteError", "CloseError", "RemoteProtocolError", "LocalProtocolError", "ProxyError",
    "UnsupportedProtocol", "DecodingError", "TooManyRedirects", "TimeoutError", "CancelledError",
})
HANGUP_TRACE_PHASES = {
    "connection.connect_tcp": "connect_tcp", "connection.start_tls": "tls",
    **{protocol + "." + operation: phase for protocol in ("http11", "http2") for operation, phase in (
        ("send_request_headers", "request_headers"), ("send_request_body", "request_body"),
        ("receive_response_headers", "response_headers"), ("receive_response_body", "response_body"))},
}
# Provider-controlled fields are not arbitrary log text. Unknown values are
# redacted, not truncated (a truncated secret is still a secret).
SAFE_ERROR_CODES = frozenset({
    "invalid_request", "invalid_request_error", "invalid_value", "invalid_parameter",
    "missing_required_parameter", "unknown_parameter", "unsupported_parameter", "unsupported_value",
    "invalid_api_key", "invalid_model", "model_not_found", "invalid_voice", "invalid_sdp",
    "invalid_session", "invalid_session_config", "unauthorized", "permission_denied",
    "rate_limit_exceeded", "insufficient_quota", "server_error", "internal_error",
    "request_timeout", "session_expired", "call_not_found",
})
SAFE_ERROR_PARAMS = frozenset({
    "sdp", "session", "type", "model", "output_modalities", "output_modalities[0]",
    "audio", "audio.input", "audio.output", "audio.output.voice", "audio.output.format",
    "audio.input.format", "audio.input.transcription", "audio.input.transcription.model",
    "audio.input.turn_detection", "audio.input.turn_detection.type", "audio.input.turn_detection.eagerness",
    "audio.input.turn_detection.create_response", "audio.input.turn_detection.interrupt_response",
    "instructions", "tools", "tool_choice",
})
SAFE_ERROR_PARAMS = SAFE_ERROR_PARAMS | {"session." + item for item in SAFE_ERROR_PARAMS if item != "session"}


class WebRTCTransportError(RuntimeError):
    """Only sanitized metadata escapes; no underlying HTTP/SDP exception text."""

    def __init__(self, diagnostic):
        self.diagnostic = dict(diagnostic)
        super().__init__("OpenAI WebRTC setup failed at " + diagnostic["stage"] + ".")


def log_webrtc_setup_failure(logger, connection, error):
    """Used only by the session's setup-exception logger; no route semantics."""
    if not isinstance(connection, OpenAIWebRTCConnection):
        return False
    logger.error("Jarvis OpenAI realtime setup failed error_type=%s transport_diagnostic=%s",
                 type(error).__name__, json.dumps(connection.setup_failure_diagnostic(), sort_keys=True))
    return True


def validate_offer(payload):
    if not isinstance(payload, dict) or payload.get("type") != "audio.webrtc.offer":
        raise ValueError("A WebRTC offer must be the first control message.")
    sdp, mode = payload.get("sdp"), payload.get("capture_mode", "auto")
    if (not isinstance(sdp, str) or not sdp.startswith("v=0")
            or len(sdp.encode()) > MAX_SDP_BYTES or "m=audio " not in sdp
            or mode not in {"auto", "manual"}):
        raise ValueError("Invalid audio WebRTC offer or capture mode.")
    return sdp, mode


class OpenAIWebRTCConnection(OpenAIRealtimeConnection):
    transport = "webrtc"

    def __init__(self, *, settings, owner_id, offer, capture_mode="auto", http=None, native_turns=False):
        super().__init__(settings=settings, owner_id=owner_id)
        if type(native_turns) is not bool:
            raise ValueError("Native turn ownership must be explicitly selected.")
        self.native_turns = native_turns
        self.offer, self.capture_mode = offer, capture_mode
        self.answer = None
        self.call_id = None
        # Retain known identity independently of the active local handle. This
        # is in-memory evidence, not a durable recovery ledger.
        self.created_call_id = None
        self.creation_outcome = "not_requested"
        self.closure_outcome = "not_requested"
        self.creation_diagnostic = None
        self.closure_diagnostic = None
        self._setup_diagnostic = None
        self._creation_http_status = None
        self._creation_request_id = None
        self._creation_attempted = False
        self._close_task = None
        self._close_finished = False
        self._hangup_response = None
        self._hangup_started = None
        self._hangup_metadata = {}
        self._hangup_fields = dict(
            http_phase="not_started", http_phase_state="not_started", request_headers_sent=False,
            request_body_sent=False, response_headers_received=False, caller_cancelled=False,
            deadline_source=None, exception_type=None, response_body_outcome="not_read",
            response_close_outcome="not_requested", http_close_outcome="not_requested")
        self._http = http
        self._owns_http = http is None

    def _safe_metadata(self, value, *, choices=None, pattern=None):
        if value is None:
            return None
        key = self._settings.OPENAI_API_KEY
        if (not isinstance(value, str) or len(value) > 192
                or (key and key in value) or "sk-" in value.lower()
                or (choices is not None and value not in choices)
                or (pattern is not None and not re.fullmatch(pattern, value))):
            return "[redacted]"
        return value

    def _response_metadata(self, response=None, *, error=None):
        status, request_id = None, None
        if response is not None:
            candidate = getattr(response, "status_code", None)
            status = candidate if type(candidate) is int and 100 <= candidate <= 599 else None
            headers = getattr(response, "headers", {})
            request_id = self._safe_metadata(headers.get("x-request-id"),
                pattern=r"(?:req_[A-Za-z0-9_-]{1,128}|[a-fA-F0-9]{32}|[a-fA-F0-9-]{36})")
            if status is not None and status >= 400:
                try:
                    body = response.content if isinstance(response, httpx.Response) else response.body
                    if isinstance(body, (bytes, bytearray)) and len(body) <= MAX_ERROR_METADATA_BYTES:
                        payload = json.loads(body)
                        if isinstance(payload, dict):
                            error = payload.get("error")
                except (AttributeError, ValueError, TypeError, RecursionError, httpx.ResponseNotRead):
                    pass
        error = error if isinstance(error, dict) else {}
        return dict(http_status=status, provider_request_id=request_id,
                    provider_error_code=self._safe_metadata(error.get("code"), choices=SAFE_ERROR_CODES),
                    provider_error_param=self._safe_metadata(error.get("param"), choices=SAFE_ERROR_PARAMS))

    def _diagnostic(self, stage, reason, response=None, *, error=None):
        return dict(stage=stage, reason=reason, **self._response_metadata(response, error=error),
                    call_id=self._safe_metadata(self.created_call_id, pattern=r"rtc_[A-Za-z0-9_-]{1,180}"),
                    creation_http_status=self._creation_http_status, creation_request_id=self._creation_request_id,
                    creation_outcome=self.creation_outcome, closure_outcome=self.closure_outcome,
                    remote_closure_confirmed=False)

    def _record_setup_failure(self, stage, reason, response=None):
        self._setup_diagnostic = self._diagnostic(stage, reason, response)
        if stage in {"create_http", "validate_call_identity", "validate_sdp_answer"}:
            self.creation_diagnostic = dict(self._setup_diagnostic)
        return WebRTCTransportError(self._setup_diagnostic)

    def setup_failure_diagnostic(self):
        # A post-attach failure could be readiness, context preparation, or a
        # write. Do not infer which from exception text or claim readiness.
        result = dict(self._setup_diagnostic or self._diagnostic("post_attach_setup", "session_setup_failed"))
        result.update(creation_outcome=self.creation_outcome, closure_outcome=self.closure_outcome)
        return result

    def session_config(self, config):
        # Copy nested configuration: the relay path's configuration is unchanged.
        result = {**config}
        audio = {**result.get("audio", {})}
        incoming = {**audio.get("input", {})}
        outgoing = {**audio.get("output", {})}
        incoming.pop("format", None)
        outgoing.pop("format", None)
        incoming["turn_detection"] = ({
            "type": "semantic_vad", "eagerness": "auto",
            "create_response": self.native_turns, "interrupt_response": self.native_turns,
        } if self.capture_mode == "auto" else None)
        audio.update(input=incoming, output=outgoing)
        result["audio"] = audio
        return result

    async def connect(self):
        if self._creation_attempted or self._close_task is not None or self._close_finished:
            raise WebRTCTransportError(self._diagnostic("create_http", "connection_is_single_use"))
        headers = {"Authorization": "Bearer " + self._settings.OPENAI_API_KEY,
                   "OpenAI-Safety-Identifier": safety_identifier_for_owner(self._owner_id)}
        self._http = self._http or httpx.AsyncClient(timeout=20)
        config = self.session_config({
            "type": "realtime", "model": self._settings.OPENAI_REALTIME_MODEL,
            "output_modalities": ["audio"],
            "audio": {"output": {"voice": self._settings.OPENAI_REALTIME_VOICE}},
        })
        self._creation_attempted = True
        self.creation_outcome = "requested"
        self.creation_diagnostic = self._diagnostic("create_http", "request_started")
        try:
            response = await self._http.post(CALLS_URL, headers=headers, files={
                "sdp": (None, self.offer, "application/sdp"),
                "session": (None, json.dumps(config), "application/json"),
            })
        except asyncio.CancelledError:
            self.creation_outcome = "unknown"
            self._record_setup_failure("create_http", "request_cancelled_without_response")
            raise
        except Exception:
            self.creation_outcome = "unknown"
            raise self._record_setup_failure("create_http", "request_failed_without_response") from None
        metadata = self._response_metadata(response)
        self._creation_http_status = metadata["http_status"]
        self._creation_request_id = metadata["provider_request_id"]
        # Provider bodies can contain submitted SDP or credentials; never log them.
        if response.status_code != 201:
            # Rejection of this request is distinct from loss of its outcome.
            # Timeout/server/unexpected-success responses cannot prove absence
            # of a remotely created call. Never retry this connection object.
            self.creation_outcome = ("rejected" if 400 <= response.status_code < 500
                                     and response.status_code != 408 else "unknown")
            raise self._record_setup_failure("create_http", "unexpected_http_status", response) from None
        self.creation_outcome = "unknown"
        try:
            location = urlsplit(response.headers.get("location", ""))
            identity = location.path.removeprefix("/v1/realtime/calls/")
            valid = (location.hostname in {None, "api.openai.com"}
                     and location.path.startswith("/v1/realtime/calls/")
                     and re.fullmatch(r"rtc_[A-Za-z0-9_-]{1,180}", identity))
        except ValueError:
            valid = False
        if not valid:
            raise self._record_setup_failure("validate_call_identity", "call_identity_unavailable", response) from None
        self.call_id = identity
        self.created_call_id = identity
        self.creation_outcome = "created"
        self.creation_diagnostic = self._diagnostic("create_http", "created_with_identity", response)
        try:
            valid_answer = len(response.content) <= MAX_SDP_BYTES and response.text.startswith("v=0")
        except Exception:
            valid_answer = False
        if not valid_answer:
            raise self._record_setup_failure("validate_sdp_answer", "invalid_sdp_answer", response) from None
        self.answer = response.text
        uri = "wss://api.openai.com/v1/realtime?" + urlencode({"call_id": self.call_id})
        try:
            self._websocket = await websockets.connect(uri, additional_headers=headers)
        except asyncio.CancelledError:
            self._record_setup_failure("sideband_connect", "attach_cancelled")
            raise
        except Exception as error:
            raise self._record_setup_failure("sideband_connect", "attach_failed", getattr(error, "response", None)) from None

    async def receive_event(self):
        payload = await super().receive_event()
        if payload.get("type") == "error":
            # Retain only filtered metadata before the session converts this
            # event into an exception. Never retain/log its message or payload.
            self._setup_diagnostic = self._diagnostic("provider_control_error", "provider_error_event", error=payload.get("error"))
        return payload

    async def send_event(self, event, *, write_guard=None):
        async def prepared():
            import inspect
            payload = event() if callable(event) else event
            if inspect.isawaitable(payload):
                payload = await payload
            if payload is not None and payload.get("type") == "session.update":
                payload = {**payload, "session": self.session_config(payload["session"])}
            return payload
        return await super().send_event(prepared, write_guard=write_guard)

    async def close(self):
        if self._close_finished:
            return dict(self.closure_diagnostic)
        if self._close_task is None:
            # One task owns all teardown, including HTTP lifetime. Caller
            # cancellation must not cancel hangup and then destroy its client.
            # There is no detached retry: every caller joins this same task.
            self._close_task = asyncio.create_task(self._close_owned(), name="webrtc-owned-teardown")
        task = self._close_task
        cancelled = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                cancelled = error
                self._hangup_fields["caller_cancelled"] = True
                if self.closure_diagnostic is not None:
                    self.closure_diagnostic["caller_cancelled"] = True
                # Only an internal cancellation may cancel the owned task.
                # Repeated caller cancellation is remembered but cannot detach it.
                if task.cancelled():
                    break
            except Exception:
                break
        try:
            result = task.result()
        except BaseException:
            if cancelled is not None:
                raise cancelled from None
            raise
        if cancelled is not None:
            raise cancelled
        return result

    def _record_hangup(self, reason, *, error=None):
        if error is not None:
            name = type(error).__name__
            self._hangup_fields["exception_type"] = name if name in SAFE_TRANSPORT_EXCEPTIONS else "other"
        elapsed = (None if self._hangup_started is None else
                   round(max(0, asyncio.get_running_loop().time() - self._hangup_started) * 1000, 3))
        self.closure_diagnostic = {
            **self._diagnostic("hangup_http", reason), **self._hangup_metadata, **self._hangup_fields,
            "hangup_elapsed_ms": elapsed,
            "hangup_total_timeout_ms": HANGUP_TIMEOUT_SECONDS * 1000,
            "http_phase_timeout_ms": HANGUP_HTTP_TIMEOUT_SECONDS * 1000,
        }
        level = logging.INFO if self.closure_outcome in {"accepted", "requested", "not_requested"} else logging.WARNING
        LOGGER.log(level, "WebRTC remote cleanup diagnostic=%s", json.dumps(self.closure_diagnostic, sort_keys=True))

    async def _hangup_trace(self, name, _info):
        # HTTPcore trace info contains URLs, headers and network objects. Never
        # inspect or retain it. Only fixed operation names become diagnostics.
        if not isinstance(name, str) or len(name) > 96:
            return
        operation, _, state = name.rpartition(".")
        phase = HANGUP_TRACE_PHASES.get(operation)
        if phase is None or state not in {"started", "complete", "failed"}:
            return
        self._hangup_fields.update(http_phase=phase, http_phase_state=state)
        if state == "complete" and phase in {"request_headers", "request_body"}:
            self._hangup_fields[phase + "_sent"] = True

    async def _close_owned(self):
        try:
            try:
                async with asyncio.timeout(LOCAL_CLOSE_TIMEOUT_SECONDS):
                    await super().close()
            finally:
                self.call_id = None
                try:
                    async with asyncio.timeout(HANGUP_TIMEOUT_SECONDS):
                        await self._request_hangup()
                except TimeoutError as error:
                    self._hangup_fields["deadline_source"] = "hangup_total"
                    received = self._hangup_fields["response_headers_received"]
                    if not received:
                        self.closure_outcome = "unknown"
                    self._record_hangup("hangup_deadline_after_response" if received else
                                        "hangup_deadline_without_response", error=error)
        finally:
            try:
                if self._hangup_response is not None:
                    try:
                        async with asyncio.timeout(RESPONSE_CLOSE_TIMEOUT_SECONDS):
                            await self._hangup_response.aclose()
                    except BaseException as error:
                        self._hangup_fields["response_close_outcome"] = (
                            "deadline" if isinstance(error, TimeoutError) else
                            "cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
                        self._record_hangup(self.closure_diagnostic["reason"], error=error)
                        raise
                    else:
                        self._hangup_fields["response_close_outcome"] = "closed"
            finally:
                try:
                    if self._owns_http and self._http is not None:
                        try:
                            async with asyncio.timeout(HTTP_CLOSE_TIMEOUT_SECONDS):
                                await self._http.aclose()
                        except BaseException as error:
                            self._hangup_fields["http_close_outcome"] = (
                                "deadline" if isinstance(error, TimeoutError) else
                                "cancelled" if isinstance(error, asyncio.CancelledError) else "failed")
                            self._record_hangup(self.closure_diagnostic["reason"], error=error)
                            raise
                        else:
                            self._hangup_fields["http_close_outcome"] = "closed"
                    elif self._http is not None:
                        self._hangup_fields["http_close_outcome"] = "borrowed"
                finally:
                    self._http = None
                    self._hangup_response = None
                    self._close_finished = True
                    if self.closure_diagnostic is not None:
                        self._record_hangup(self.closure_diagnostic["reason"])
        return dict(self.closure_diagnostic)

    async def _read_hangup_error(self, response):
        self._hangup_fields.update(http_phase="response_body", http_phase_state="started")
        body = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=MAX_ERROR_METADATA_BYTES + 1):
            if len(body) + len(chunk) > MAX_ERROR_METADATA_BYTES:
                self._hangup_fields["response_body_outcome"] = "too_large"
                return
            body.extend(chunk)
        self._hangup_fields["http_phase_state"] = "complete"
        try:
            payload = json.loads(body)
        except (ValueError, TypeError, RecursionError):
            self._hangup_fields["response_body_outcome"] = "invalid_json"
            return
        metadata = self._response_metadata(error=payload.get("error") if isinstance(payload, dict) else None)
        for field in ("provider_error_code", "provider_error_param"):
            self._hangup_metadata[field] = metadata[field]
        self._hangup_fields["response_body_outcome"] = "read"

    async def _request_hangup(self):
        if self.created_call_id is None or self._http is None:
            self.closure_outcome = "unknown" if self.creation_outcome in {"requested", "unknown", "created"} else "not_requested"
            self._record_hangup("call_identity_or_transport_unavailable")
            return
        self._hangup_started = asyncio.get_running_loop().time()
        self._hangup_fields.update(http_phase="request_dispatch", http_phase_state="started")
        self.closure_outcome = "requested"
        self._record_hangup("request_started")
        try:
            request = self._http.build_request("POST", CALLS_URL + "/" + self.created_call_id + "/hangup",
                headers={"Authorization": "Bearer " + self._settings.OPENAI_API_KEY},
                timeout=httpx.Timeout(HANGUP_HTTP_TIMEOUT_SECONDS), extensions={"trace": self._hangup_trace})
            # Streaming here means headers are observable before body completion.
            # No redirects or retries, and only the owned task closes resources.
            response = await self._http.send(request, stream=True, follow_redirects=False)
            self._hangup_response = response
            self._hangup_metadata = self._response_metadata(response)
            self._hangup_fields.update(response_headers_received=True,
                                       http_phase="response_headers", http_phase_state="complete")
            if response.status_code == 200:
                self.closure_outcome, reason = "accepted", "teardown_accepted_not_confirmed_closed"
            elif 200 <= response.status_code < 300:
                self.closure_outcome, reason = "unknown", "undocumented_success_status"
            else:
                self.closure_outcome, reason = "failed", "hangup_http_failure"
            self._record_hangup(reason)
            if response.status_code >= 400:
                await self._read_hangup_error(response)
            else:
                # A 200 is accepted teardown, not confirmed closure. No success
                # body is required to preserve that already-observed status.
                self._hangup_fields["response_body_outcome"] = "not_needed"
            self._record_hangup(reason)
        except asyncio.CancelledError as error:
            received = self._hangup_fields["response_headers_received"]
            if not received:
                self.closure_outcome = "unknown"
            elif self._hangup_fields["response_body_outcome"] == "not_read":
                self._hangup_fields["response_body_outcome"] = "cancelled"
            self._hangup_fields["deadline_source"] = "cancelled"
            self._record_hangup("request_cancelled_after_response" if received else
                                "request_cancelled_without_response", error=error)
            raise
        except Exception as error:
            received = self._hangup_fields["response_headers_received"]
            if not received:
                self.closure_outcome = "unknown"
            elif self._hangup_fields["response_body_outcome"] == "not_read":
                self._hangup_fields["response_body_outcome"] = "failed"
            if isinstance(error, httpx.TimeoutException):
                self._hangup_fields["deadline_source"] = "http_phase"
            self._record_hangup("request_failed_after_response" if received else
                                "request_failed_without_response", error=error)
