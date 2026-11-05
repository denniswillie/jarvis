"""GPT Live client-delegation protocol, independent of media and task ownership.

This is not an activated VoiceRoute. The authenticated transport owns reading,
serialized guarded writes, and playback fences. No append acknowledgment here
is evidence that speech was generated, rendered, or heard. See the P3 handoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import json
import math
import time
from typing import Callable, Literal
from urllib.parse import quote
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


LIVE_SESSION_URL = "https://api.openai.com/v1/live/sessions"
TokenCounter = Callable[[str], int]


class LiveProtocolError(ValueError):
    """A bounded diagnostic code, never a provider message or supplied text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _text(value: object, *, maximum: int, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()) or "\x00" in value:
        raise LiveProtocolError("invalid_text")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise LiveProtocolError("invalid_text") from None
    if size > maximum:
        raise LiveProtocolError("text_limit_exceeded")
    return value


def _identifier(value: object) -> str:
    result = _text(value, maximum=256)
    if any(ord(char) < 32 for char in result):
        raise LiveProtocolError("invalid_identifier")
    return result


def _tokens(text: str, counter: TokenCounter, maximum: int) -> None:
    count = counter(text)
    if type(count) is not int or count < 0 or count > maximum:
        raise LiveProtocolError("token_budget_exceeded")


def _milliseconds(value: object) -> int:
    if type(value) is not int or value < 0:
        raise LiveProtocolError("invalid_timeline")
    return value


def _interval(raw: dict) -> tuple[int, int]:
    start = _milliseconds(raw.get("start_ms"))
    end = _milliseconds(raw.get("end_ms"))
    if end < start:
        raise LiveProtocolError("invalid_timeline")
    return start, end


@dataclass(frozen=True, slots=True)
class LiveHistoryMessage:
    role: Literal["developer", "user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class LiveSessionConfiguration:
    instructions: str
    model: str = "gpt-live-1"
    voice: str = "marin"
    time_zone: str = "Europe/Dublin"


def webrtc_session_request(
    config: LiveSessionConfiguration,
    offer_sdp: str,
    *,
    history: tuple[LiveHistoryMessage, ...] = (),
    count_tokens: TokenCounter,
    now: datetime | None = None,
) -> dict:
    """Call only with server-owned instructions and source-guarded history.

    The caller supplies a qualified token counter. Provider validation remains
    authoritative. No truncation, recording, tools, or managed Responses agent
    is enabled implicitly. SDP is media negotiation, not logged context.
    """
    _text(offer_sdp, maximum=65536)
    instructions = _text(config.instructions, maximum=262144, empty=True)
    # Clock metadata is allowed alongside static instructions. Generate it for
    # each new provider session, never from retained dialogue or model knowledge.
    instant = now if now is not None else datetime.now(timezone.utc)
    if not isinstance(instant, datetime) or instant.utcoffset() is None:
        raise LiveProtocolError("invalid_session_clock")
    zone_name = _identifier(config.time_zone)
    try:
        zone = ZoneInfo(zone_name)
    except (ZoneInfoNotFoundError, ValueError):
        raise LiveProtocolError("invalid_session_timezone") from None
    stamp = instant.astimezone(zone).isoformat(timespec="seconds")
    instructions += f"\n\nCurrent date/time at session start: {stamp} ({zone_name})."
    _tokens(instructions, count_tokens, 16384)
    _identifier(config.model)
    _identifier(config.voice)
    if len(history) > 128:
        raise LiveProtocolError("history_message_limit_exceeded")
    messages = []
    for message in history:
        if message.role not in {"developer", "user", "assistant"}:
            raise LiveProtocolError("invalid_history_role")
        text = _text(message.text, maximum=131072, empty=True)
        messages.append({"type": "message", "role": message.role, "content": [{
            "type": "output_text" if message.role == "assistant" else "input_text", "text": text,
        }]})
    _tokens("\n".join(message.text for message in history), count_tokens, 8192)
    return {
        "session": {
            "model": config.model,
            "instructions": instructions,
            "delegation": {"type": "client"},
            "audio": {"output": {"voice": config.voice}},
            "input": messages,
            "store": False,
        },
        "transport": {"type": "webrtc", "sdp": offer_sdp},
    }


@dataclass(frozen=True, slots=True)
class LiveSessionAnswer:
    session_id: str
    answer_sdp: str

    @property
    def sideband_url(self) -> str:
        return f"wss://api.openai.com/v1/live/sessions/{quote(self.session_id, safe='')}/attach"


def parse_webrtc_session_answer(raw: object) -> LiveSessionAnswer:
    if not isinstance(raw, dict):
        raise LiveProtocolError("invalid_session_answer")
    session, transport = raw.get("session"), raw.get("transport")
    if not isinstance(session, dict) or not isinstance(transport, dict) or transport.get("type") != "webrtc":
        raise LiveProtocolError("invalid_session_answer")
    return LiveSessionAnswer(_identifier(session.get("id")), _text(transport.get("sdp"), maximum=65536))


class LiveAppendKind(StrEnum):
    THINKING = "session.thinking.append"
    COMMENTARY = "session.commentary.append"
    INSTRUCTIONS = "session.instructions.append"


class LiveCommandStatus(StrEnum):
    PREPARED = "prepared"
    SUBMISSION_STARTED = "submission_started"
    WRITTEN = "written"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


_TERMINAL = {LiveCommandStatus.ACCEPTED, LiveCommandStatus.REJECTED, LiveCommandStatus.UNKNOWN}
_ACKS = {
    "session.thinking.appended": LiveAppendKind.THINKING.value,
    "session.commentary.appended": LiveAppendKind.COMMENTARY.value,
    "session.instructions.appended": LiveAppendKind.INSTRUCTIONS.value,
    "session.input_audio.muted": "session.input_audio.mute",
    "session.input_audio.unmuted": "session.input_audio.unmute",
}


@dataclass(frozen=True, slots=True)
class LiveCommandObservation:
    command_id: str
    kind: str
    status: LiveCommandStatus
    delegation_id: str | None
    deadline: float
    start_ms: int | None = None
    end_ms: int | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LiveTranscriptFragment:
    event_id: str
    speaker: Literal["user", "assistant"]
    delta: str
    start_ms: int
    end_ms: int
    # A fragment has no provider turn ID/final marker or audio output identity.


@dataclass(frozen=True, slots=True)
class LiveDelegationRequest:
    event_id: str
    delegation_id: str
    offset_ms: int
    # Intentionally no inferred utterance, tool arguments, or action permission.


@dataclass(frozen=True, slots=True)
class LiveSessionFinalization:
    session_id: str
    reason: str
    usage_seconds: float | None
    # Provider finalization does not establish a client playback fence.


@dataclass(frozen=True, slots=True)
class LiveProviderFailure:
    command_id: str | None
    code: str


class LiveControlProtocol:
    """Single-loop, nonblocking protocol bookkeeping for one authenticated route.

    Transport callbacks must carry the actual bound session ID. Caller policy
    must run canonical guards immediately around take_for_write and its write.
    Deadline expiry never triggers retry. Capacity/finalization failures require
    caller-owned output fencing and reconnection, not silent event eviction.
    """

    def __init__(
        self, *, route_id: UUID, route_generation: int, provider_session_id: str,
        count_tokens: TokenCounter, clock: Callable[[], float] = time.monotonic,
        max_commands: int = 2048, max_pending: int = 32, max_events: int = 32768,
    ):
        if not isinstance(route_id, UUID) or type(route_generation) is not int or route_generation < 1:
            raise LiveProtocolError("invalid_route_identity")
        if any(type(value) is not int or value < 1 for value in (max_commands, max_pending, max_events)):
            raise LiveProtocolError("invalid_capacity")
        if max_pending > max_commands or max_commands > 16384 or max_events > 131072:
            raise LiveProtocolError("invalid_capacity")
        self.route_id = route_id
        self.route_generation = route_generation
        self.provider_session_id = _identifier(provider_session_id)
        self._count_tokens, self._clock = count_tokens, clock
        self._max_commands, self._max_pending, self._max_events = max_commands, max_pending, max_events
        self._records: dict[str, LiveCommandObservation] = {}
        self._prepared: dict[str, dict] = {}
        self._seen: dict[str, str] = {}
        self._delegations: dict[str, int] = {}
        self._attached = False
        self._closing = False
        self._closed = False
        self._retired = False
        self._close_id: str | None = None

    @property
    def reconnect_required(self) -> bool:
        return self._retired

    @property
    def ready(self) -> bool:
        return self._attached and not (self._closing or self._closed or self._retired)

    def mark_attached(self, *, authorized_session_id: str) -> None:
        if authorized_session_id != self.provider_session_id or self._closing or self._closed or self._retired:
            raise LiveProtocolError("session_binding_mismatch")
        self._attached = True

    def observation(self, command_id: str) -> LiveCommandObservation:
        try:
            return self._records[command_id]
        except KeyError:
            raise LiveProtocolError("unknown_command") from None

    def _replace(self, original: LiveCommandObservation, status: LiveCommandStatus, *,
                 reason: str | None = None, start_ms: int | None = None, end_ms: int | None = None):
        value = LiveCommandObservation(original.command_id, original.kind, status, original.delegation_id,
                                       original.deadline, start_ms, end_ms, reason)
        self._records[original.command_id] = value
        return value

    def _prepare(self, kind: str, payload: dict, *, command_id: str | None, timeout: float):
        if not self.ready:
            raise LiveProtocolError("session_not_ready")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise LiveProtocolError("invalid_timeout")
        self.expire()
        identifier = _identifier(command_id) if command_id is not None else str(uuid4())
        if identifier in self._records:
            raise LiveProtocolError("command_id_reused")
        if len(self._records) >= self._max_commands:
            self.retire()
            raise LiveProtocolError("command_capacity_exceeded")
        if sum(item.status not in _TERMINAL for item in self._records.values()) >= self._max_pending:
            raise LiveProtocolError("pending_command_capacity_exceeded")
        record = LiveCommandObservation(identifier, kind, LiveCommandStatus.PREPARED,
                                        payload.get("delegation_id"), self._clock() + timeout)
        self._records[identifier] = record
        self._prepared[identifier] = {"type": kind, "event_id": identifier, **payload}
        return record

    def prepare_append(self, kind: LiveAppendKind, content: str, *, delegation_id: str | None = None,
                       command_id: str | None = None, timeout: float = 5.0) -> LiveCommandObservation:
        if not isinstance(kind, LiveAppendKind):
            raise LiveProtocolError("invalid_append_kind")
        _text(content, maximum=32768, empty=True)
        _tokens(content, self._count_tokens, 500)
        if delegation_id is not None and delegation_id not in self._delegations:
            raise LiveProtocolError("unknown_client_delegation")
        return self._prepare(kind.value, {"content": content, "delegation_id": delegation_id},
                             command_id=command_id, timeout=timeout)

    def prepare_input_mute(self, *, muted: bool, command_id: str | None = None, timeout: float = 5.0):
        if type(muted) is not bool:
            raise LiveProtocolError("invalid_mute_value")
        return self._prepare("session.input_audio.mute" if muted else "session.input_audio.unmute", {},
                             command_id=command_id, timeout=timeout)

    def prepare_close(self, *, command_id: str | None = None, timeout: float = 5.0):
        record = self._prepare("session.close", {}, command_id=command_id, timeout=timeout)
        self._close_id = record.command_id
        self._closing = True
        return record

    def take_for_write(self, command_id: str) -> dict:
        self.expire()
        record = self.observation(command_id)
        if self._retired or self._closed or (self._closing and command_id != self._close_id):
            raise LiveProtocolError("session_not_ready")
        if record.status != LiveCommandStatus.PREPARED:
            raise LiveProtocolError("command_not_writable")
        payload = self._prepared.pop(command_id)
        self._replace(record, LiveCommandStatus.SUBMISSION_STARTED)
        return payload

    def mark_written(self, command_id: str) -> LiveCommandObservation:
        self.expire()
        record = self.observation(command_id)
        if record.status == LiveCommandStatus.SUBMISSION_STARTED:
            return self._replace(record, LiveCommandStatus.WRITTEN)
        if record.status in _TERMINAL:
            return record
        raise LiveProtocolError("command_not_submitted")

    def expire(self) -> tuple[LiveCommandObservation, ...]:
        now = self._clock()
        expired = []
        for record in tuple(self._records.values()):
            if record.status not in _TERMINAL and now >= record.deadline:
                self._prepared.pop(record.command_id, None)
                expired.append(self._replace(record, LiveCommandStatus.UNKNOWN, reason="command_deadline_exceeded"))
        return tuple(expired)

    def retire(self) -> None:
        self._retired = True
        self._prepared.clear()
        for record in tuple(self._records.values()):
            if record.status not in _TERMINAL:
                self._replace(record, LiveCommandStatus.UNKNOWN, reason="route_retired")

    def transport_lost(self) -> None:
        self.retire()

    def _remember(self, event_id: str, raw: dict) -> bool:
        try:
            digest = sha256(json.dumps(raw, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        except (TypeError, ValueError):
            raise LiveProtocolError("invalid_provider_event") from None
        previous = self._seen.get(event_id)
        if previous is not None:
            if previous != digest:
                self.retire()
                raise LiveProtocolError("provider_event_id_reused")
            return False
        if len(self._seen) >= self._max_events:
            self.retire()
            raise LiveProtocolError("event_capacity_exceeded")
        self._seen[event_id] = digest
        return True

    def accept(self, raw: object, *, provider_session_id: str):
        """Called by the sole authenticated provider reader; never by phone JSON."""
        if provider_session_id != self.provider_session_id:
            raise LiveProtocolError("session_binding_mismatch")
        if self._retired or self._closed:
            return None
        if not self._attached or not isinstance(raw, dict):
            raise LiveProtocolError("session_not_ready")
        kind = raw.get("type")
        known = set(_ACKS) | {"session.input_transcript.delta", "session.output_transcript.delta",
                             "session.delegation.created", "session.closed", "error"}
        if not isinstance(kind, str) or kind not in known:
            return None
        event_id = _identifier(raw.get("event_id"))
        if not self._remember(event_id, raw):
            return None
        self.expire()
        if kind in _ACKS:
            command_id = _identifier(raw.get("client_event_id"))
            record = self._records.get(command_id)
            if record is None or record.status in _TERMINAL:
                return None
            if record.kind != _ACKS[kind] or record.status == LiveCommandStatus.PREPARED:
                raise LiveProtocolError("acknowledgment_mismatch")
            if "delegation_id" in raw and raw["delegation_id"] != record.delegation_id:
                raise LiveProtocolError("acknowledgment_mismatch")
            start, end = _interval(raw) if record.kind in set(LiveAppendKind) else (None, None)
            return self._replace(record, LiveCommandStatus.ACCEPTED, start_ms=start, end_ms=end)
        if kind.endswith("transcript.delta"):
            start, end = _interval(raw)
            return LiveTranscriptFragment(event_id, "user" if kind == "session.input_transcript.delta" else "assistant",
                                          _text(raw.get("delta"), maximum=65536, empty=True), start, end)
        if kind == "session.delegation.created":
            delegation = raw.get("delegation")
            if not isinstance(delegation, dict) or delegation.get("target") != "client" or delegation.get("type") != "delegation":
                self.retire()
                raise LiveProtocolError("unexpected_delegation_mode")
            identity, offset = _identifier(delegation.get("id")), _milliseconds(raw.get("offset_ms"))
            if identity in self._delegations:
                if self._delegations[identity] != offset:
                    self.retire()
                    raise LiveProtocolError("delegation_id_reused")
                return None
            self._delegations[identity] = offset
            return LiveDelegationRequest(event_id, identity, offset)
        if kind == "error":
            error = raw.get("error")
            if not isinstance(error, dict):
                raise LiveProtocolError("invalid_provider_error")
            command_id = error.get("client_event_id")
            record = self._records.get(command_id) if isinstance(command_id, str) else None
            code = error.get("code")
            if not isinstance(code, str) or not 1 <= len(code) <= 80 or any(c not in "abcdefghijklmnopqrstuvwxyz_0123456789" for c in code):
                code = "provider_error"
            if record is not None and record.status not in _TERMINAL:
                self._prepared.pop(record.command_id, None)
                self._replace(record, LiveCommandStatus.REJECTED, reason=code)
            return LiveProviderFailure(record.command_id if record else None, code)
        session = raw.get("session")
        if not isinstance(session, dict) or session.get("id") != self.provider_session_id:
            raise LiveProtocolError("session_binding_mismatch")
        usage = raw.get("usage")
        seconds = usage.get("seconds") if isinstance(usage, dict) else None
        if seconds is not None and (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
            raise LiveProtocolError("invalid_usage")
        reason = raw.get("reason")
        if reason not in {"close_requested", "expired", "content", "remote_hangup", "connection_lost"}:
            reason = "unknown"
        self._closed = True
        self._prepared.clear()
        for record in tuple(self._records.values()):
            if record.status not in _TERMINAL:
                accepted = record.command_id == self._close_id and record.status != LiveCommandStatus.PREPARED
                self._replace(record, LiveCommandStatus.ACCEPTED if accepted else LiveCommandStatus.UNKNOWN,
                              reason="session_finalized")
        return LiveSessionFinalization(self.provider_session_id, reason, seconds)
