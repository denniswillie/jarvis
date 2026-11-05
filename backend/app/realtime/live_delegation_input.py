"""Bounded original observations for a server-owned Live command bridge.

This packet is context, not CommandAdmission, a completed speech turn, or a
provider function call. The owning bridge interprets the original dialogue and
uses existing canonical command admission. No second transcript/task store.
"""
from dataclasses import dataclass
import json

from app.tasks.models import TaskError
from app.tasks.readonly import FRAGMENT_FORMAT
from app.voice_routes.contracts import RouteIdentity, VoiceContextUpdate
from app.voice_routes.live_adapter import LiveDelegationWindow


@dataclass(frozen=True)
class LiveDelegationInput:
    route: RouteIdentity
    audio_session_id: str
    window: LiveDelegationWindow
    context_json: str
    observation_payload_ids: tuple
    live_context: VoiceContextUpdate

    @classmethod
    def from_window(cls, route, audio_session_id, window, observations, live_context, *, application_context=None):
        if (not window.parts
                or window.received_part_count < len(window.parts)
                or len(observations) != len(window.parts) or live_context.route != route):
            raise TaskError("live_delegation_context_unavailable", "The received context window is unavailable.")
        dialogue = []
        for part, record in zip(window.parts, observations):
            body = record.content or {}
            fragment = part.fragment
            exact = {"format": FRAGMENT_FORMAT, "voice_route": "live",
                "route": route.model_dump(mode="json"), "audio_provider": "openai",
                "audio_session_id": audio_session_id, "caption_id": str(part.transcript_id),
                "caption_revision": part.revision, "provider_event_id": fragment.event_id,
                "speaker": fragment.speaker, "delta": fragment.delta,
                "start_ms": fragment.start_ms, "end_ms": fragment.end_ms,
                "speech_finality": "provisional", "request_boundary": "unavailable"}
            if (record.owner_id != route.scope.owner_id or record.session_id != route.scope.session_id
                    or record.kind != "observation" or record.trust != "derived"
                    or not record.is_current or record.availability != "available"
                    or any(body.get(key) != value for key, value in exact.items())):
                raise TaskError("live_delegation_source_mismatch", "An original observation is unavailable.")
            dialogue.append({"context_event_id": str(record.id), "context_revision": record.revision,
                "payload_id": str(record.payload_id), "content_digest": record.content_digest,
                "sequence": record.sequence, "origin_sequence": record.origin_sequence,
                "speaker": fragment.speaker, "delta": fragment.delta,
                "start_ms": fragment.start_ms, "end_ms": fragment.end_ms})
        if sum(len(part.fragment.delta.encode("utf-8")) for part in window.parts) > 32768:
            raise TaskError("live_delegation_context_budget", "The original context window exceeds its bound.")
        context = {"format": "jarvis.live_delegation_input.v1",
            "route": route.model_dump(mode="json"), "audio_session_id": audio_session_id,
            "delegation": {"event_id": window.request.event_id,
                "delegation_id": window.request.delegation_id, "offset_ms": window.request.offset_ms},
            "window": {"boundary": "received_before_delegation_event",
                "received_part_count": window.received_part_count,
                "included_part_count": len(window.parts),
                "omitted_prefix_parts": window.omitted_prefix_parts,
                "speech_finality": "unavailable", "timing": "approximate_not_a_turn_boundary",
                "originals": "canonical_context_events_in_this_owner_session"},
            "dialogue": dialogue,
            "live_state": {"text": live_context.text,
                "expires_at": live_context.expires_at.isoformat() if live_context.expires_at is not None else None,
                "source_payload_ids": [str(value) for value in
                    (live_context.guard_binding.payload_root_ids if live_context.guard_binding else ())]}}
        if application_context is not None:
            application_json = json.dumps(application_context, ensure_ascii=False, allow_nan=False)
            if len(application_json.encode("utf-8")) > 40000:
                raise TaskError("live_application_context_budget", "The current application context exceeds its bound.")
            context["application_context"] = json.loads(application_json)
        return cls(route, audio_session_id, window,
            json.dumps(context, ensure_ascii=False, separators=(",", ":")),
            tuple(record.payload_id for record in observations), live_context)

    def as_context(self):
        """A fresh data projection, with exact deltas and explicit omissions."""
        return json.loads(self.context_json)
