"""Frozen legacy population mapping. SQL identifiers are code-owned, never model input."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Population:
    table: str
    keys: tuple[str, ...]
    mode: str
    fields: tuple[str, ...]
    parent: str | None = None
    parent_key: str | None = None
    owned: bool = True
    optional: bool = False


POPULATIONS = (
    Population("jarvis_behavior_profiles", ("owner_id",), "retained_operational",
        ("owner_id",)),
    Population("jarvis_file_publications", ("id",), "retained_operational",
        ("id",), optional=True),
    Population("jarvis_files", ("id",), "external_only",
        ("id",)),
    Population("jarvis_messages", ("id",), "source",
        ("content", "created_at", "id", "metadata", "modality", "owner_id", "provider_item_id", "provider_response_id", "role", "session_id", "source_payload_id")),
    Population("jarvis_notes", ("id",), "source",
        ("body", "created_at", "id", "metadata", "owner_id", "session_id", "source", "title", "updated_at")),
    Population("jarvis_owner_working_memory", ("owner_id",), "retained_operational",
        ("owner_id",)),
    Population("jarvis_prospective_delivery_events", ("id",), "retained_operational",
        ("id",)),
    Population("jarvis_reminder_events", ("id",), "retained_operational",
        ("id",)),
    Population("jarvis_reminders", ("id",), "retained_operational",
        ("id",)),
    Population("jarvis_session_working_memory", ("owner_id", "session_id"), "retained_operational",
        ("owner_id", "session_id")),
    Population("jarvis_sessions", ("id",), "external_only",
        ("id",)),
    Population("jarvis_tool_executions", ("id",), "retained_operational",
        ("id",)),
    Population("jarvis_user_preferences", ("id",), "retained_operational",
        ("id",)),
    Population("jarvis_visual_focuses", ("id",), "retained_operational",
        ("id",)),
    Population("experience_event_embeddings", ("experience_event_id", "embedding_kind", "provider", "model"), "retained_rebuildable",
        ("experience_event_id", "embedding_kind", "provider", "model"), parent="experience_events", parent_key="experience_event_id", owned=False),
    Population("experience_event_evidence", ("id",), "attached",
        ("event_id", "evidence_ref_id", "evidence_type", "id", "ingested_at", "metadata", "occurred_at", "source_sequence"), parent="experience_events", parent_key="event_id", owned=False),
    Population("experience_event_keyframes", ("event_id", "visual_frame_id"), "attached",
        ("created_at", "event_id", "quality_score", "rank", "reason", "selected_by", "visual_frame_id"), parent="experience_events", parent_key="event_id", owned=False),
    Population("experience_event_retention_reasons", ("id",), "retained_operational",
        ("id",)),
    Population("experience_events", ("id",), "source",
        ("activity", "boundary_reason", "confidence", "consolidated_at", "consolidation_model", "consolidation_provider", "consolidation_ready_at", "consolidation_result", "created_at", "ended_at", "event_kind", "evidence_reduced_at", "id", "importance", "keyframes_finalized_at", "modalities", "owner_id", "promoted_episode_memory_id", "promotion_decided_at", "promotion_decision", "promotion_reason", "retention_expires_at", "scene", "semantic_reconciled_at", "session_id", "started_at", "state", "status", "summary", "title", "updated_at")),
    Population("google_oauth_tokens", ("owner_id",), "excluded_credentials",
        ("owner_id",)),
    Population("memory_entities", ("id",), "source",
        ("aliases", "canonical_name", "confidence", "created_at", "description", "entity_type", "external_id", "external_namespace", "id", "metadata", "normalized_name", "owner_id", "updated_at")),
    Population("memory_entity_embeddings", ("entity_id", "embedding_kind", "provider", "model"), "retained_rebuildable",
        ("entity_id", "embedding_kind", "provider", "model"), parent="memory_entities", parent_key="entity_id", owned=False),
    Population("memory_entity_mentions", ("id",), "attached",
        ("confidence", "created_at", "entity_id", "id", "memory_item_id", "owner_id", "role"), parent="memory_entities", parent_key="entity_id", owned=True),
    Population("memory_entity_relations", ("id",), "attached",
        ("confidence", "created_at", "id", "metadata", "object_entity_id", "object_value", "owner_id", "predicate", "status", "subject_entity_id", "superseded_by_relation_id", "supporting_memory_id", "updated_at", "valid_from", "valid_to"), parent="memory_entities", parent_key="subject_entity_id", owned=True),
    Population("memory_evidence", ("id",), "attached",
        ("confidence", "created_at", "evidence_ref_id", "evidence_time", "evidence_type", "excerpt", "id", "memory_item_id", "metadata", "owner_id"), parent="memory_items", parent_key="memory_item_id", owned=True),
    Population("memory_forgetting_tombstones", ("id",), "retained_operational",
        ("id", "source_memory_id", "overridden_at")),
    Population("memory_item_embeddings", ("memory_item_id", "embedding_kind", "provider", "model"), "retained_rebuildable",
        ("memory_item_id", "embedding_kind", "provider", "model"), parent="memory_items", parent_key="memory_item_id", owned=False),
    Population("memory_items", ("id",), "source",
        ("confidence", "content", "created_at", "id", "importance", "is_explicit", "memory_type", "metadata", "occurred_at", "owner_id", "session_id", "source", "source_experience_event_id", "status", "summary", "superseded_by_memory_id", "title", "updated_at", "valid_from", "valid_to")),
    Population("memory_jobs", ("id",), "retained_rebuildable",
        ("id",)),
    Population("memory_links", ("id",), "attached",
        ("confidence", "created_at", "id", "metadata", "owner_id", "relationship", "source_memory_id", "supporting_memory_id", "target_memory_id", "updated_at", "valid_from", "valid_to"), parent="memory_items", parent_key="source_memory_id", owned=True),
    Population("memory_profile_projections", ("owner_id",), "retained_rebuildable",
        ("owner_id",)),
    Population("memory_retrieval_runs", ("id",), "retained_rebuildable",
        ("id",)),
    Population("perception_observation_frames", ("observation_id", "visual_frame_id"), "attached",
        ("created_at", "observation_id", "ordinal", "role", "visual_frame_id"), parent="perception_observations", parent_key="observation_id", owned=False),
    Population("perception_observations", ("id",), "source",
        ("activity", "activity_confidence", "analyzer_contract_version", "analyzer_model", "analyzer_provider", "boundary_reason", "compacted_at", "confidence", "created_at", "ended_at", "entities", "id", "interrupt_candidate", "keyframe_candidates", "memory_tags", "owner_id", "primary_visual_frame_id", "scene", "scene_confidence", "session_id", "source", "started_at", "starts_new_event", "state_changes", "summary", "visible_text")),
    Population("visual_frame_retention_reasons", ("id",), "retained_operational",
        ("id",)),
    Population("visual_frames", ("id",), "external_only",
        ("id", "raw_image_status", "captured_at")),
    Population("visual_interrupts", ("id",), "retained_operational",
        ("id",)),
    Population("web_fetch_cache", ("id",), "external_only",
        ("id",)),
)

# Operational snapshots require account-specific authorization and are never an
# automatic vendor import. Presence without supported owner scope is reported.
POPULATIONS += (Population("calendar_event_snapshots", ("id",), "external_only", ("id",), optional=True),)
BY_TABLE = {item.table: item for item in POPULATIONS}
SOURCE_TABLES = frozenset(item.table for item in POPULATIONS if item.mode == "source")
EVIDENCE_TABLES = {
    "message": "jarvis_messages", "observation": "perception_observations",
    "experience_event": "experience_events", "memory": "memory_items",
    "frame": "visual_frames", "tool_execution": "jarvis_tool_executions",
    "file": "jarvis_files", "reminder": "jarvis_reminders", "calendar_event": "calendar_event_snapshots",
}
