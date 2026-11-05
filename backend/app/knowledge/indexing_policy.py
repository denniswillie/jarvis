"""Semantic eligibility derived from canonical bindings, never vendor prose."""
from app.tasks.models import digest


TRANSPORT_RECEIPT_EXCLUDED = "semantic_index_excluded_transport_receipt"
_STATES = {"socket_written", "provider_acknowledged", "write_outcome_unknown",
           "not_written", "provider_rejected"}
_FIELDS = {"audio_provider", "audio_session_id", "images", "source_time_basis",
           "delivery_id", "delivery_id_basis", "delivery_state", "delivery_state_scope",
           "image_view", "current_view_verified", "provider_acknowledged",
           "model_understanding", "evidence_gap"}


def indexing_exclusion(row, content):
    """Recognize only the exact SessionContextCapture image receipt envelope.

    The row is the owner/area-checked SOURCE_VERSION_SQL projection, including
    the actual context-source binding. Receipt-looking user JSON is insufficient.
    Unknown/new envelopes remain eligible rather than silently dropping content.
    """
    if (row.get("context_binding_matches") is not True
            or row.get("context_event_id") is None
            or row.get("source_type") != "observation" or row.get("trust") != "sensor"
            or row.get("context_kind") != "observation" or row.get("context_trust") != "sensor"
            or not isinstance(content, dict) or set(content) != _FIELDS):
        return None
    provider, session = content["audio_provider"], content["audio_session_id"]
    state, identity = content["delivery_state"], content["delivery_id"]
    if (not isinstance(provider, str) or not provider or not isinstance(session, str) or not session
            or not isinstance(state, str) or state not in _STATES
            or not isinstance(identity, str) or not 1 <= len(identity) <= 512
            or content["delivery_state_scope"] != "immutable_receipt"
            or content["model_understanding"] != "unknown"
            or content["current_view_verified"] is not False
            or content["provider_acknowledged"] is not (state == "provider_acknowledged")
            or state == "provider_acknowledged" and provider != "openai"
            or content["delivery_id_basis"] != ("provider_item" if provider == "openai" else "application_image_write")):
        return None
    images = content["images"]
    if not isinstance(images, list) or len(images) > 1:
        return None
    if images:
        if (not isinstance(images[0], dict) or content["source_time_basis"] != "image_capture"
                or content["image_view"] != "captured_snapshot" or content["evidence_gap"] is not None):
            return None
    elif (content["source_time_basis"] != "application_ingress_not_physical_capture"
          or content["image_view"] != "unavailable" or content["evidence_gap"] != "image_reference_unavailable"):
        return None
    namespace = "voice:" + digest({"provider": provider, "audio_session_id": session})
    event = "image:" + digest({"identity": digest({"delivery_id": identity, "state": state})})
    if (row.get("context_source_namespace") != namespace or row.get("context_source_event_id") != event
            or row.get("source_namespace") != "context:" + digest({"namespace": namespace})
            or row.get("source_key") != event):
        return None
    return TRANSPORT_RECEIPT_EXCLUDED
