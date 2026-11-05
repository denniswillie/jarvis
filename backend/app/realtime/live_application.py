"""Native application context/control for the existing Live audio session.

This binding does not interpret speech, create a worker or grant a task access.
The server command receiver owns interpretation admission and call identity.
"""
from contextvars import ContextVar

from app.recordings.voice import LeanRecordingDispatcher
from app.realtime.resource_context import LiveResourceContext


class LiveApplicationBridge:
    def __init__(self, *, context, services, is_current, is_ready, offer_event):
        self.context, self.is_current, self.is_ready = context, is_current, is_ready
        self.offer_event = offer_event
        self.resources = LiveResourceContext(context.jarvis_session_id, context.audio_session_id)
        self._backend_session = ContextVar("jarvis_live_backend_session", default=None)
        admissions = getattr(services, "recording_admission", None)
        policy = getattr(services, "recording_policy", None)
        self.recording = None
        if admissions is not None and policy is not None:
            self.recording = LeanRecordingDispatcher(owner_id=context.owner_id,
                session_id=context.jarvis_session_id, admissions=admissions, policy=policy,
                provider_session_key=self._backend_session.get, is_active=is_ready,
                on_command=lambda payload: self.offer_event({"type": "recording.command",
                    **payload, "audio_session_id": context.audio_session_id}))
        self.app_controls = getattr(services, "app_controls", None)
        if self.app_controls is not None:
            self.app_controls.bind(self)

    def session_created(self):
        event = {"type": "session.created", "session_id": str(self.context.jarvis_session_id),
                 "audio_session_id": self.context.audio_session_id, "audio_provider": "openai",
                 "voice_route": "live", "conversation_mode": "legacy"}
        if self.recording is not None:
            event["recording_protocol_version"] = 2
        return event

    def receive(self, event):
        if not self.is_current() or type(event) is not dict:
            return False
        kind = event.get("type")
        if kind == "workspace.resource_context":
            return self.resources.receive(event)
        if self.recording is not None:
            if kind == "recording.capabilities":
                return self.recording.configure_device(event)
            if kind == "recording.report":
                return self.recording.accept_report(event) is not None
        return False

    def snapshot(self):
        value = {"resource_context": self.resources.snapshot()}
        if self.recording is not None:
            identity = self.recording.current_id
            value["recording"] = {"recording_id": str(identity) if identity else None,
                "device_capabilities_received": self.recording.capabilities is not None,
                "last_client_report": dict(self.recording.reports[identity])
                    if identity in self.recording.reports else None,
                "meaning": "Native capability/report context only; consult the recording owner for saved state."}
        return value

    async def execute_recording(self, arguments, *, provider_call_id, backend_session_key, admit_if=None):
        """Called only by the authenticated server command receiver, not the phone.

        The existing dispatcher still rejects generic worker invocation scopes.
        This method does not clear/change that scope or infer a tool call ID.
        """
        if (self.recording is None or not self.is_ready()
                or type(backend_session_key) is not str or not backend_session_key.strip()
                or len(backend_session_key) > 256 or "\x00" in backend_session_key):
            return {"status": "success", "outcome": "unavailable", "effect_committed": False,
                    "summary": "Recording command transport is unavailable."}
        token = self._backend_session.set(backend_session_key)
        try:
            return await self.recording.execute_confirmed(arguments, provider_call_id=provider_call_id, admit_if=admit_if)
        finally:
            self._backend_session.reset(token)

    def close(self):
        if self.app_controls is not None:
            self.app_controls.unbind(self)
        self.resources.close()
        if self.recording is not None:
            self.recording.capabilities = None
            self.recording.report_changed.set()
        # No stop/cancel is inferred. The recording owner retains its receipt.
