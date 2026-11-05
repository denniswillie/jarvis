"""Offline exact-source correlation of owner-generated voice evidence.

Callers supply canonical ContextRecords from a scoped, current export/snapshot,
not model text or adjacent messages. This reader does no I/O and grants no task
authority. A successful source join is still not native acceptance or hearing.
"""
from __future__ import annotations

from uuid import UUID

from app.tasks.context_events import ContextRecord
from app.tasks.models import digest


def resolve_response_input(*, output: ContextRecord, request: ContextRecord | None,
                           owner_id: str, session_id: UUID, audio_session_id: str,
                           response_id: str, item_id: str) -> dict:
    """Join one exact assistant binding to one actual canonical input revision.

    Absence/withdrawal is not reconstructed from cached payloads. Availability
    is evidence at the supplied snapshot, not a promise of future availability.
    """
    def result(status, reason, canonical_input=None):
        return {"status": status, "reason": reason, "canonical_input": canonical_input,
                "heard_state": "unknown", "full_chain_proven": False}

    namespace = "voice:" + digest({"provider": "openai", "audio_session_id": audio_session_id})
    content = output.content
    if (output.owner_id != owner_id or output.session_id != session_id
            or output.source_namespace != namespace
            or output.source_event_id != "output:" + digest({"identity": item_id})
            or output.kind != "assistant_output" or output.trust != "assistant_output"
            or not output.is_current or output.availability != "available"
            or output.finality == "unavailable" or not isinstance(content, dict)
            or content.get("audio_provider") != "openai"
            or content.get("audio_session_id") != audio_session_id
            or content.get("response_id") != response_id or content.get("item_id") != item_id):
        return result("unavailable", "output_scope_or_availability")
    binding = content.get("response_input_association")
    if not isinstance(binding, dict):
        return result("unresolved", "association_missing")
    expected = {"authority": "session_context_capture", "owner_id": owner_id,
                "session_id": str(session_id), "audio_provider": "openai",
                "audio_session_id": audio_session_id, "source_namespace": namespace,
                "response_id": response_id, "canonical_event_id": None,
                "canonical_revision": None, "heard_state": "unknown"}
    if (type(binding.get("schema_version")) is not int or binding["schema_version"] != 1
            or any(key not in binding or binding[key] != value for key, value in expected.items())
            or binding.get("full_chain_proven") is not False):
        return result("unavailable", "association_scope_or_contract")
    if binding.get("binding_status") != "source_bound":
        return result("unresolved", "association_not_bound")
    identity, generation = binding.get("input_id"), binding.get("turn_generation")
    if (not isinstance(identity, str) or not identity.strip() or "\x00" in identity
            or len(identity.encode("utf-8")) > 512
            or type(generation) is not int or not 0 <= generation < 2**63
            or binding.get("source_event_id") != "input:" + digest({"identity": identity})):
        return result("unavailable", "association_identity_invalid")
    if request is None:
        return result("unresolved", "canonical_input_missing")
    if (request.owner_id != owner_id or request.session_id != session_id
            or request.source_namespace != namespace
            or request.source_event_id != binding["source_event_id"]):
        return result("unavailable", "canonical_input_scope_mismatch")
    source = request.content
    if (not request.authorizes_request or request.kind != "user_transcript"
            or request.revision < 1 or not isinstance(source, dict)
            or source.get("input_id") != identity
            or source.get("audio_provider") != "openai"
            or source.get("audio_session_id") != audio_session_id
            or source.get("transcript_status") != "finalized"
            or not isinstance(source.get("text"), str) or not source["text"].strip()):
        return result("unavailable", "canonical_input_not_current_final_available")
    provider_item = source.get("provider_item_id")
    if (not isinstance(provider_item, str) or not provider_item.strip()
            or "\x00" in provider_item or len(provider_item.encode("utf-8")) > 512):
        provider_item = None
    return result("resolved", "exact_canonical_source", {
        "event_id": str(request.id), "payload_id": str(request.payload_id),
        "revision": request.revision, "source_namespace": request.source_namespace,
        "source_event_id": request.source_event_id, "provider_item_id": provider_item,
    })
