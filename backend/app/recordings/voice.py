"""Lean voice commands over recording authority and bounded native control.

The domain owns admission/finalization; this session-local adapter owns only
device capability hints, command delivery deduplication and client reports.
No microphone, recording task engine, transcript gate or second voice source.
"""
import asyncio
from uuid import UUID

from app.assistant.tool_limits import ToolCallLimitError
from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileTransferPolicy, MP4_CAPTURE_POLICY
from app.knowledge.models import SourceError
from app.live_state.admission import CommandAdmission
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition


def _uuid(value):
    if not isinstance(value, str):
        raise ValueError("An exact recording identity is required")
    return UUID(value)


def _result(outcome, summary, *, effect=False, **fields):
    return {"status": "success", "outcome": outcome, "effect_committed": effect,
            "summary": summary, **fields}


class LeanRecordingDispatcher:
    def __init__(self, *, owner_id, session_id, admissions, policy,
                 provider_session_key, is_active, on_command, timeout=5.0):
        self.owner_id, self.session_id = owner_id, session_id
        self.admissions, self.policy = admissions, policy
        self.provider_session_key, self.is_active = provider_session_key, is_active
        self.on_command, self.timeout = on_command, timeout
        self.current_id = None
        self.capabilities = None
        self.generations, self.reports = {}, {}
        self.recording_options = {}
        self.notified, self.stop_requested = set(), set()
        self.command_lock = asyncio.Lock()
        self.report_changed = asyncio.Event()

    async def execute_confirmed(self, arguments, *, confirmation_timeout=3.0, **kwargs):
        """Use the existing phone report before confirming a capture command."""
        result = await self.execute(arguments, **kwargs)
        identity = result.get("recording_id")
        if not identity or arguments.get("action") not in {"start", "stop"} or result.get("saved"):
            return result
        action = arguments["action"]
        pending = result | {"summary": "The phone has not confirmed this recording command. Do not claim it started or stopped.",
                            "capture_confirmed": False}
        deadline = asyncio.get_running_loop().time() + confirmation_timeout
        while True:
            self.report_changed.clear()
            report = self.reports.get(UUID(identity))
            current = (self.is_active() and self.capabilities is not None and report is not None
                       and report["generation_id"] == str(self.capabilities["generation_id"]))
            state = report["state"] if current else None
            if state in {"failed", "stopped"} or (action == "start" and state == "started"):
                summaries = {
                    "started": "The phone confirms recording has started. Saving is not yet confirmed.",
                    "stopped": "The phone confirms recording has stopped. Saving is not yet confirmed.",
                    "failed": "The phone reports recording failed. It is not recording.",
                }
                return result | {"outcome": "capture_" + state,
                    "summary": summaries[state], "last_client_report": report,
                    "capture_confirmed": state == "started"}
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0 or not self.is_active() or not result.get("device_command_enqueued"):
                return pending
            try:
                await asyncio.wait_for(self.report_changed.wait(), remaining)
            except TimeoutError:
                return pending

    def definition(self):
        async def unbound(arguments):
            return _result("unavailable", "Recording requires the authenticated native voice dispatcher.")
        return ToolDefinition(name="live_recording", mutates_state=True, handler=unbound,
            description=("Start, stop or inspect a glasses recording only when requested. Ask whether to include "
                "Jarvis audio before starting if unclear. Admission is not capture; a phone report is not a save. "
                "Speech interruption does not stop recording. Status is on demand, not automatic polling."),
            input_schema={"type": "object", "additionalProperties": False, "required": ["action"],
                "properties": {"action": {"type": "string", "enum": ["start", "stop", "status"]},
                    "include_jarvis_audio": {"type": "boolean"},
                    "recording_id": {"type": "string", "format": "uuid"}}})

    def configure_device(self, event):
        try:
            keys = {"type", "version", "generation_id", "available", "include_jarvis_audio",
                    "max_bytes", "max_duration_ms"}
            if event.get("version") == 2:
                keys |= {"include_media_audio", "voiceover"}
                if event.get("include_media_audio") is not True or event.get("voiceover") is not True:
                    raise ValueError()
            if (set(event) != keys or type(event["version"]) is not int or event["version"] not in (1, 2)
                    or type(event["available"]) is not bool or type(event["include_jarvis_audio"]) is not bool
                    or type(event["max_bytes"]) is not int or not 1 <= event["max_bytes"] <= CAPTURE_MAX_BYTES
                    or type(event["max_duration_ms"]) is not int or not 1 <= event["max_duration_ms"] <= CAPTURE_MAX_DURATION_MS):
                raise ValueError()
            generation = _uuid(event["generation_id"])
        except (KeyError, TypeError, ValueError):
            self.capabilities = None
            return False
        self.capabilities = dict(event, generation_id=generation)
        return True

    def accept_report(self, event):
        try:
            if (set(event) != {"type", "version", "generation_id", "recording_id", "sequence", "state"}
                    or type(event["version"]) is not int or event["version"] != 1
                    or type(event["sequence"]) is not int or not 1 <= event["sequence"] <= 1_000_000_000
                    or event["state"] not in {"started", "stopped", "failed"}):
                return None
            identity, generation = _uuid(event["recording_id"]), _uuid(event["generation_id"])
            if (not self.is_active() or self.capabilities is None
                    or self.capabilities["generation_id"] != generation
                    or self.generations.get(identity) != generation):
                return None
            prior = self.reports.get(identity)
            if (prior is not None and (event["sequence"] <= prior["sequence"]
                    or (prior["state"] in {"stopped", "failed"} and event["state"] == "started"))):
                return None
            if event["state"] == "started" and identity in self.stop_requested:
                return None
        except (KeyError, TypeError, ValueError):
            return None
        report = {"recording_id": str(identity), "generation_id": str(generation),
                  "sequence": event["sequence"], "state": event["state"],
                  "basis": "client_report", "saved": False}
        self.reports[identity] = report
        self.report_changed.set()
        return report

    def _normalize(self, arguments):
        if not isinstance(arguments, dict):
            raise ValueError()
        action = arguments.get("action")
        if action == "start":
            if (set(arguments) - {"action", "include_jarvis_audio", "recording_mode"}
                    or type(arguments.get("include_jarvis_audio")) is not bool):
                raise ValueError("Clarify whether Jarvis audio should be included")
            mode = arguments.get("recording_mode", "ordinary")
            if mode not in ("ordinary", "voiceover") or (mode == "voiceover" and arguments["include_jarvis_audio"]):
                raise ValueError("Voiceover records only the microphone")
            normalized = {"include_jarvis_audio": arguments["include_jarvis_audio"]}
            if self.capabilities and self.capabilities["version"] == 2:
                normalized.update(recording_mode=mode, include_media_audio=mode == "ordinary")
            elif mode == "voiceover":
                raise ValueError("This phone has not offered voiceover recording")
            return action, normalized
        if action in {"stop", "status"} and set(arguments) <= {"action", "recording_id"}:
            identity = _uuid(arguments["recording_id"]) if "recording_id" in arguments else self.current_id
            return action, {"recording_id": str(identity)} if identity is not None else {}
        raise ValueError()

    async def _status(self, identity):
        async with self.admissions.sources.transaction(self.owner_id) as connection:
            await self.admissions.tasks.lock_owner(connection, self.owner_id)
            scope = await self.admissions.require_admitted_recording(
                connection, self.owner_id, self.session_id, identity)
            saved = await self.admissions.get_completion(connection, scope)
            allowed = False
            if saved is None:
                try:
                    await self.admissions.require_admitted_recording(
                        connection, self.owner_id, self.session_id, identity, for_capture=True)
                    allowed = True
                except SourceError as error:
                    if error.code != "recording_capture_closed":
                        raise
        result = _result("saved" if saved is not None else "admitted" if allowed else "capture_closed",
            "The recording has an exact saved receipt." if saved is not None else
            "This is recording authority status; it does not prove current capture or saved media.",
            effect=saved is not None, recording_id=str(identity), capture_allowed=allowed,
            include_jarvis_audio=scope.include_jarvis_audio)
        if saved is not None:
            result["saved"] = {key: str(getattr(saved, key))
                               for key in ("file_id", "source_id", "version_id", "receipt_event_id")}
        report = self.reports.get(identity)
        if report is not None and self.capabilities is not None and report["generation_id"] == str(self.capabilities["generation_id"]):
            result["last_client_report"] = report
        return result

    async def execute(self, arguments, *, provider_call_id, budget=None, admit_if=None):
        if current_tool_invocation().task is not None or not self.is_active():
            return _result("unavailable", "No foreground recording command was admitted.")
        try:
            action, normalized = self._normalize(arguments)
            if budget is not None:
                budget.record(tool_name="live_recording", arguments={"action": action, **normalized})
        except (TypeError, ValueError, ToolCallLimitError):
            return _result("rejected", "Use valid recording arguments; clarify whether to include Jarvis audio before starting.")
        if action != "start" and not normalized:
            return _result("unavailable", "No recording identity is known in this voice session; no action was taken.")
        command = None
        submitted = False
        if action == "stop":
            # A stop arriving while start admission waits prevents its later
            # native start hint. Ordinary speech interruption never sets this.
            self.stop_requested.add(UUID(normalized["recording_id"]))
        try:
            async with asyncio.timeout(self.timeout):
                if action == "status":
                    return await self._status(UUID(normalized["recording_id"]))
                command = CommandAdmission(owner_id=self.owner_id, session_id=self.session_id,
                    provider_session_key=self.provider_session_key(), provider_call_id=provider_call_id,
                    tool_name="live_recording", capability="recordings." + action,
                    normalized_arguments=normalized)
                async with self.command_lock:
                    if not self.is_active() or (admit_if is not None and not admit_if()):
                        return _result("rejected", "The interrupted recording command was not admitted.")
                    if action == "start":
                        device = self.capabilities
                        if (device is None or not device["available"]
                                or device["max_duration_ms"] < self.policy.max_duration_ms
                                or (normalized["include_jarvis_audio"] and not device["include_jarvis_audio"])):
                            return _result("unavailable", "The phone has not offered the requested recording capability; capture was not requested.")
                        if self.current_id is not None and self.current_id != command.operation_id:
                            previous = await self._status(self.current_id)
                            if previous["capture_allowed"]:
                                return dict(previous, summary="An existing recording admission is still open; no second capture was requested.")
                        if len(self.generations) >= 128 and command.operation_id not in self.generations:
                            return _result("unavailable", "This voice session has reached its recording-control capacity.")
                        selected = await self.admissions.selections.read(self.owner_id, self.session_id)
                        self.current_id = command.operation_id
                        self.generations[command.operation_id] = device["generation_id"]
                        if "recording_mode" in normalized:
                            self.recording_options[command.operation_id] = {
                                key: normalized[key] for key in ("recording_mode", "include_media_audio")}
                        effective = FileTransferPolicy(MP4_CAPTURE_POLICY, min(self.policy.max_bytes, device["max_bytes"]))
                        submitted = True
                        receipt = await self.admissions.admit(command, area_id=UUID(selected["area_id"]), policy=effective)
                    else:
                        submitted = True
                        receipt = await self.admissions.stop(command)
                    identity = receipt.scope.recording_id
                    enqueue = False
                    if (command.operation_id not in self.notified and self.is_active()
                            and identity in self.generations
                            and (action == "stop" or (receipt.capture_allowed and identity not in self.stop_requested))):
                        payload = {"version": 1, "action": action, "operation_id": str(command.operation_id),
                            "recording_id": str(identity), "session_id": str(self.session_id),
                            "generation_id": str(self.generations[identity]),
                            "include_jarvis_audio": receipt.scope.include_jarvis_audio,
                            "max_bytes": receipt.scope.max_bytes, "max_duration_ms": receipt.scope.max_duration_ms}
                        if identity in self.recording_options:
                            payload.update(version=2, **self.recording_options[identity])
                        try:
                            enqueue = self.on_command(payload) is True
                        except Exception:
                            enqueue = False
                        if enqueue:
                            self.notified.add(command.operation_id)
                    return _result(receipt.outcome, "Recording authority was updated. Capture and final saving are not confirmed by this receipt.",
                        effect=True, operation_id=str(command.operation_id), recording_id=str(identity),
                        capture_allowed=receipt.capture_allowed, device_command_enqueued=enqueue,
                        max_duration_ms=receipt.scope.max_duration_ms, max_bytes=receipt.scope.max_bytes)
        except asyncio.CancelledError:
            raise
        except Exception:
            return _result("outcome_unknown" if submitted else "unavailable",
                "The recording operation is unconfirmed; inspect its status instead of blindly repeating it.",
                effect=None if submitted else False,
                operation_id=str(command.operation_id) if command is not None else None,
                recording_id=str(self.current_id) if self.current_id is not None else None)
