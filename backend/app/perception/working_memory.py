from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from re import sub
from typing import Any, Mapping
from uuid import UUID, uuid5

from app.perception.models import VisualObservation


_ENTITY_ID_NAMESPACE = UUID("fc43bf80-2707-46c2-9ee8-ef9312f6d433")
_MIN_TRANSITION_CONFIDENCE = 0.6
_MAX_VISIBLE_TEXT_ITEMS = 20
_MAX_OWNER_LIST_ITEMS = 20
_OWNER_FIELDS = frozenset(
    {
        "active_goal_ids",
        "active_visual_focus",
        "pending_commitments",
        "carried_context",
    }
)


@dataclass(frozen=True)
class WorkingMemoryPolicy:
    ttl_seconds: int
    entity_ttl_seconds: int
    visible_text_ttl_seconds: int
    max_entities: int
    max_recent_changes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("ttl_seconds", self.ttl_seconds),
            ("entity_ttl_seconds", self.entity_ttl_seconds),
            ("visible_text_ttl_seconds", self.visible_text_ttl_seconds),
            ("max_entities", self.max_entities),
            ("max_recent_changes", self.max_recent_changes),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")


def reduce_session_working_memory(
    previous_state: Mapping[str, Any],
    observation: VisualObservation,
    *,
    policy: WorkingMemoryPolicy,
) -> dict[str, Any]:
    """Apply one ordered observation to bounded session working memory."""

    observed_at = observation.ended_at
    state = prune_session_working_memory(
        previous_state,
        now=observed_at,
        policy=policy,
    )
    observation_id = str(observation.id)

    state["scene"] = _reduce_temporal_label(
        state.get("scene"),
        label=observation.scene,
        confidence=(
            observation.scene_confidence
            if observation.scene_confidence is not None
            else observation.confidence
        ),
        observed_at=observed_at,
        source_observation_id=observation_id,
        first_seen_key="first_seen_at",
        last_seen_key="last_seen_at",
    )
    state["activity"] = _reduce_temporal_label(
        state.get("activity"),
        label=observation.activity,
        confidence=(
            observation.activity_confidence
            if observation.activity_confidence is not None
            else observation.confidence
        ),
        observed_at=observed_at,
        source_observation_id=observation_id,
        first_seen_key="started_at",
        last_seen_key="last_observed_at",
    )

    entities = _mapping_copy(state.get("entities"))
    local_entity_ids: dict[str, str] = {}
    for entity in observation.entities:
        attributes = dict(entity.attributes)
        existing_id = _non_blank_string(attributes.pop("entity_id", None))
        local_key = _non_blank_string(attributes.pop("local_key", None))
        entity_id = existing_id
        if entity_id is None and local_key is not None:
            entity_id = _session_entity_id(observation, local_key)
            local_entity_ids[local_key] = entity_id
        if entity_id is None:
            continue
        if existing_id is not None and entity_id not in entities:
            continue

        previous = _mapping_copy(entities.get(entity_id))
        entity_state = attributes.pop("state", None)
        last_visible_frame_id = attributes.pop("last_visible_frame_id", None)
        entities[entity_id] = {
            "entity_id": entity_id,
            "type": entity.category or previous.get("type") or "unknown",
            "name": entity.name,
            "state": entity_state,
            "attributes": {
                **_mapping_copy(previous.get("attributes")),
                **attributes,
            },
            "confidence": entity.confidence,
            "first_seen_at": previous.get("first_seen_at") or observed_at.isoformat(),
            "last_seen_at": observed_at.isoformat(),
            "last_visible_frame_id": last_visible_frame_id,
            "source_observation_id": observation_id,
        }

    state["entities"] = _bound_entities(entities, policy.max_entities)

    recent_changes = _list_of_mappings(state.get("recent_changes"))
    for change in observation.state_changes:
        subject_id = local_entity_ids.get(change.subject, change.subject)
        if subject_id not in state["entities"]:
            continue
        recent_changes.append(
            {
                "entity_id": subject_id,
                "attribute": change.attribute,
                "before": change.before,
                "after": change.after,
                "confidence": change.confidence,
                "observed_at": observed_at.isoformat(),
                "source_observation_id": observation_id,
            }
        )
    state["recent_changes"] = recent_changes[-policy.max_recent_changes :]
    state["visible_text"] = _reduce_visible_text(
        state.get("visible_text"),
        observation=observation,
        max_items=_MAX_VISIBLE_TEXT_ITEMS,
    )
    state.update(
        {
            "summary": observation.summary,
            "description": observation.summary,
            "confidence": observation.confidence,
            "location": _state_label(state.get("scene")),
            "current_activity": _state_label(state.get("activity")),
            "source_frame_ids": (
                [str(observation.primary_visual_frame_id)]
                if observation.primary_visual_frame_id is not None
                else []
            ),
            "last_observation_id": observation_id,
            "latest_experience_event_id": state.get("latest_experience_event_id"),
            "updated_at": observed_at.isoformat(),
        }
    )
    return state


def prune_session_working_memory(
    state: Mapping[str, Any],
    *,
    now: datetime,
    policy: WorkingMemoryPolicy,
) -> dict[str, Any]:
    """Return an expired-and-bounded copy without mutating persisted input."""

    result = deepcopy(dict(state))
    entity_cutoff = now.timestamp() - policy.entity_ttl_seconds
    entities = {
        entity_id: entity
        for entity_id, entity in _mapping_copy(result.get("entities")).items()
        if _timestamp(_mapping_copy(entity).get("last_seen_at")) >= entity_cutoff
    }
    result["entities"] = _bound_entities(entities, policy.max_entities)

    text_cutoff = now.timestamp() - policy.visible_text_ttl_seconds
    result["visible_text"] = [
        item
        for item in _list_of_mappings(result.get("visible_text"))
        if _timestamp(item.get("last_seen_at")) >= text_cutoff
    ][-_MAX_VISIBLE_TEXT_ITEMS:]

    evidence_cutoff = now.timestamp() - policy.ttl_seconds
    result["recent_changes"] = [
        item
        for item in _list_of_mappings(result.get("recent_changes"))
        if _timestamp(item.get("observed_at")) >= evidence_cutoff
    ][-policy.max_recent_changes:]
    result.setdefault("latest_experience_event_id", None)
    return result


def empty_owner_working_memory() -> dict[str, Any]:
    return {}


def normalize_owner_working_memory(
    state: Mapping[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Validate, expire, and copy the bounded owner working-memory fields."""

    # Companion state and plugin inventory share storage, not this contract.
    # Their owners validate them; this legacy projection must leave them alone.
    shared_fields = {"companion_live_state", "companion_installed_plugins"}
    state = {field: value for field, value in state.items() if field not in shared_fields}
    unexpected = set(state) - _OWNER_FIELDS
    if unexpected:
        raise ValueError(
            "unsupported owner working-memory fields: " + ", ".join(sorted(unexpected))
        )

    normalized: dict[str, Any] = {}
    for field, raw_entry in state.items():
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"owner working-memory field {field} must be an object")
        unexpected_entry_keys = set(raw_entry) - {"value", "expires_at", "provenance"}
        if unexpected_entry_keys:
            raise ValueError(
                f"unsupported keys for owner working-memory field {field}: "
                + ", ".join(sorted(unexpected_entry_keys))
            )
        if "value" not in raw_entry:
            raise ValueError(f"owner working-memory field {field} requires value")
        expires_at = _required_aware_datetime(raw_entry.get("expires_at"), field)
        provenance = raw_entry.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError(
                f"owner working-memory field {field} requires non-empty provenance"
            )
        if expires_at <= now:
            continue
        normalized[field] = {
            "value": _normalize_owner_value(field, raw_entry["value"]),
            "expires_at": expires_at.isoformat(),
            "provenance": deepcopy(dict(provenance)),
        }
    return normalized


def owner_working_memory_expires_at(state: Mapping[str, Any]) -> datetime | None:
    expiries = [
        _required_aware_datetime(_mapping_copy(entry).get("expires_at"), field)
        for field, entry in state.items()
    ]
    return max(expiries, default=None)


def _reduce_temporal_label(
    current: object,
    *,
    label: str | None,
    confidence: float,
    observed_at: datetime,
    source_observation_id: str,
    first_seen_key: str,
    last_seen_key: str,
) -> dict[str, Any] | None:
    previous = _mapping_copy(current)
    display_label = _non_blank_string(label)
    if display_label is None:
        return previous or None
    normalized = _normalize_label(display_label)
    previous_normalized = _non_blank_string(previous.get("normalized_label"))
    if previous and previous_normalized != normalized and confidence < _MIN_TRANSITION_CONFIDENCE:
        return previous
    first_seen_at = (
        previous.get(first_seen_key)
        if previous_normalized == normalized
        else observed_at.isoformat()
    )
    return {
        "label": display_label,
        "normalized_label": normalized,
        "confidence": confidence,
        first_seen_key: first_seen_at or observed_at.isoformat(),
        last_seen_key: observed_at.isoformat(),
        "source_observation_id": source_observation_id,
    }


def _reduce_visible_text(
    current: object,
    *,
    observation: VisualObservation,
    max_items: int,
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in _list_of_mappings(current):
        key = _visible_text_key(item.get("text"), item.get("context"))
        if key:
            by_key[key] = item
    observed_at = observation.ended_at.isoformat()
    for item in observation.visible_text:
        key = _visible_text_key(item.text, item.context)
        previous = by_key.get(key, {})
        by_key[key] = {
            "text": item.text,
            "context": item.context,
            "confidence": item.confidence,
            "first_seen_at": previous.get("first_seen_at") or observed_at,
            "last_seen_at": observed_at,
            "source_observation_id": str(observation.id),
            "source_frame_id": (
                str(observation.primary_visual_frame_id)
                if observation.primary_visual_frame_id is not None
                else None
            ),
        }
    return sorted(
        by_key.values(),
        key=lambda item: (str(item.get("last_seen_at", "")), str(item.get("text", ""))),
    )[-max_items:]


def _session_entity_id(observation: VisualObservation, local_key: str) -> str:
    value = uuid5(
        _ENTITY_ID_NAMESPACE,
        f"{observation.session_id}:{observation.id}:{local_key}",
    )
    return f"session_entity_{value.hex}"


def _bound_entities(entities: Mapping[str, Any], limit: int) -> dict[str, Any]:
    ordered = sorted(
        (
            (entity_id, _mapping_copy(entity))
            for entity_id, entity in entities.items()
            if isinstance(entity_id, str) and isinstance(entity, Mapping)
        ),
        key=lambda item: (str(item[1].get("last_seen_at", "")), item[0]),
        reverse=True,
    )[:limit]
    return dict(sorted(ordered, key=lambda item: item[0]))


def _mapping_copy(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _state_label(value: object) -> str:
    return _non_blank_string(_mapping_copy(value).get("label")) or ""


def _required_aware_datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"owner working-memory field {field} requires expires_at")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"owner working-memory field {field} has invalid expires_at"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"owner working-memory field {field} expires_at must include a timezone"
        )
    return parsed


def _normalize_owner_value(field: str, value: Any) -> Any:
    if field == "active_goal_ids":
        if not isinstance(value, list):
            raise ValueError("active_goal_ids value must be a list")
        normalized = [_non_blank_string(item) for item in value]
        if any(item is None for item in normalized):
            raise ValueError("active_goal_ids must contain non-blank strings")
        if len(normalized) > _MAX_OWNER_LIST_ITEMS:
            raise ValueError("active_goal_ids exceeds the item limit")
        return normalized
    if field in {"pending_commitments", "carried_context"}:
        if not isinstance(value, list):
            raise ValueError(f"{field} value must be a list")
        if len(value) > _MAX_OWNER_LIST_ITEMS:
            raise ValueError(f"{field} exceeds the item limit")
    return deepcopy(value)


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _non_blank_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_label(value: str) -> str:
    return sub(r"\s+", " ", value.strip()).casefold()


def _visible_text_key(text: object, context: object) -> str:
    normalized_text = _non_blank_string(text)
    if normalized_text is None:
        return ""
    normalized_context = _non_blank_string(context) or ""
    return f"{_normalize_label(normalized_text)}|{_normalize_label(normalized_context)}"


def _timestamp(value: object) -> float:
    if not isinstance(value, str):
        return float("-inf")
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return float("-inf")
