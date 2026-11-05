from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from app.memory.consolidation.contracts import (
        ConsolidationEvidenceInput,
        ConsolidationRequest,
    )


# Quote content, never JSON keys, transport identifiers, or retrieval labels.
_SOURCE_FIELDS = {
    "observation": (
        "summary", "scene", "activity", "entities", "visible_text",
        "state_changes", "keyframe_candidates", "confidence", "compacted",
    ),
    "message": ("content",),
    "tool_execution": ("arguments", "result", "error", "mutates_state"),
}
# Root transport fields are excluded by _SOURCE_FIELDS. Within actual content,
# status/model/role/provider may be business facts, not transport metadata.
_NON_CONTENT_FIELDS = frozenset({
    "id", "metadata", "storage_bucket", "storage_path", "content_type",
})


@dataclass(frozen=True, slots=True)
class CitationQuotation:
    source_path: str
    text: str


@dataclass(frozen=True, slots=True)
class CitationSource:
    source: ConsolidationEvidenceInput
    quotations: tuple[CitationQuotation, ...]

    def contains_quote(self, text: str) -> bool:
        # No normalization: case, punctuation, Unicode, and whitespace are evidence.
        return bool(text.strip()) and any(text in item.text for item in self.quotations)


CitationCatalog = dict[tuple[str, UUID], CitationSource]


def event_state_for_evidence(state: Mapping[str, Any]) -> dict[str, Any]:
    """Keep event observations without aggregate retrieval-only tags."""
    return {key: value for key, value in state.items() if key != "tags"}


def source_payload_for_context(source: ConsolidationEvidenceInput) -> dict[str, Any]:
    """Exclude generated observation tags even when a source is context-only."""
    payload = dict(source.payload)
    if source.evidence_type == "observation":
        payload.pop("memory_tags", None)
    # Do not recursively filter: user and tool content can legitimately contain tags.
    return payload


def is_citable_source(source: ConsolidationEvidenceInput) -> bool:
    return (
        source.citation_allowed
        and source.evidence_id is not None
        and source.evidence_type in _SOURCE_FIELDS
        # A provider image's user role is not a spoken or typed user assertion.
        and not (
            source.evidence_type == "message"
            and (
                source.payload.get("modality") == "visual"
                or not message_is_experience_evidence(source.payload)
            )
        )
    )


def message_is_experience_evidence(payload: Mapping[str, Any]) -> bool:
    """Honor transport provenance without inferring that forwarded audio was heard."""
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return True  # Legacy messages have no delivery diagnostics.
    eligible = metadata.get("experience_eligible")
    if eligible is not None and eligible is not True:
        return False
    completion = metadata.get("completion_state")
    if completion is not None and (
        not isinstance(completion, str) or completion not in {"completed", "finalized"}
    ):
        return False
    if payload.get("role") != "assistant":
        return True
    delivery = metadata.get("delivery_status")
    playback = metadata.get("playback_status")
    if delivery is not None and (
        not isinstance(delivery, str) or delivery != "forwarded"
    ):
        return False
    if playback is not None and (
        not isinstance(playback, str) or playback not in {"unconfirmed", "completed"}
    ):
        return False
    return True


def source_provenance(source: ConsolidationEvidenceInput) -> dict[str, Any]:
    """Expose only interpretation-relevant diagnostics, never arbitrary metadata."""
    if source.evidence_type != "message":
        return {}
    metadata = source.payload.get("metadata")
    provenance: dict[str, Any] = {"delivery_confirmation": "unknown"}
    if isinstance(metadata, Mapping):
        for key in (
            "experience_eligible", "completion_state", "delivery_status", "playback_status",
        ):
            value = metadata.get(key)
            if isinstance(value, (bool, str)):
                provenance[key] = value[:80] if isinstance(value, str) else value
    return provenance


def build_citation_catalog(request: ConsolidationRequest) -> CitationCatalog:
    """Build only from attached sources already scoped by the repositories."""
    catalog: CitationCatalog = {}
    for source in request.evidence:
        if not is_citable_source(source):
            continue
        key = (source.evidence_type, source.evidence_ref_id)
        if key in catalog:
            raise ValueError("duplicate consolidation evidence reference")
        quotations = tuple(
            CitationQuotation(source_path=path, text=value)
            for path, value in source_content_values(source)
            if isinstance(value, str) and value.strip()
        )
        catalog[key] = CitationSource(source=source, quotations=quotations)
    return catalog


def source_content_values(
    source: ConsolidationEvidenceInput,
) -> Iterator[tuple[str, Any]]:
    """Share content selection between literal quotes and interpretation context."""
    for field in _SOURCE_FIELDS.get(source.evidence_type, ()):
        if field in source.payload:
            yield from _content_values(source.payload[field], field)


def _content_values(value: Any, path: str) -> Iterator[tuple[str, Any]]:
    if value is None or isinstance(value, (str, bool, int, float)):
        yield path, value
    elif isinstance(value, Mapping):
        if not value:
            yield path, {}
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            if key in _NON_CONTENT_FIELDS or key.endswith(("_id", "_ids")):
                continue
            yield from _content_values(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if not value:
            yield path, []
        for index, child in enumerate(value):
            yield from _content_values(child, f"{path}[{index}]")
