"""Strict scene validation with independently rejectable derived candidates."""
from __future__ import annotations

from dataclasses import dataclass
import json

from pydantic import ValidationError

from app.live_state.perception import PerceptionFinding
from app.perception.contracts import (
    FastInterruptCandidate, FastKeyframeCandidate, FastPerceptionOutput,
    FastPerceptionOutputError, FastPerceptionRequest, FastProspectiveMatch,
    FastStateChange, _strip_json_fence, parse_fast_perception_output, validate_perception_finding,
)
from app.perception.references import PerceptionReferenceError, PerceptionReferences


@dataclass(frozen=True)
class RejectedCandidate:
    field: str
    index: int | None
    reason: str


@dataclass(frozen=True)
class ValidatedPerception:
    output: FastPerceptionOutput
    rejected: tuple[RejectedCandidate, ...]


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _invalid_constant(_):
    raise ValueError("non_json_number")


def _list_limit(field):
    return next(item.max_length for item in FastPerceptionOutput.model_fields[field].metadata
                if getattr(item, "max_length", None) is not None)


def parse_referenced_perception_output(
    raw: str, *, request: FastPerceptionRequest, references: PerceptionReferences,
) -> ValidatedPerception:
    references.require_scope(request)
    try:
        payload = json.loads(_strip_json_fence(raw), object_pairs_hook=_unique_object,
                             parse_constant=_invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError("not_object")
    except (ValueError, TypeError, AttributeError):
        raise FastPerceptionOutputError("Fast perception output failed contract validation: invalid_json") from None

    optional = {field: payload.pop(field, None if field == "interrupt_candidate" else []) for field in (
        "interrupt_candidate", "prospective_matches", "state_changes", "keyframe_candidates", "perception_findings",
    )}
    try:
        for entity in payload.get("entities", []) if isinstance(payload.get("entities", []), list) else []:
            if isinstance(entity, dict):
                if entity.get("existing_entity_id") is not None:
                    entity["existing_entity_id"] = references.entity(entity["existing_entity_id"])
                if "last_visible_frame_id" in entity:
                    entity["last_visible_frame_id"] = references.frame(entity["last_visible_frame_id"])
        for item in payload.get("visible_text", []) if isinstance(payload.get("visible_text", []), list) else []:
            if isinstance(item, dict) and "frame_id" in item:
                item["frame_id"] = references.frame(item["frame_id"])
        # Validate the core before salvaging anything. Unknown core fields,
        # entities and text outside this window invalidate the observation.
        core = FastPerceptionOutput.model_validate(payload)
    except PerceptionReferenceError:
        raise
    except ValidationError:
        raise FastPerceptionOutputError("Fast perception output failed contract validation: required_structure") from None

    rejected = []
    accepted = {}
    entity_ids = {value for entity in core.entities for value in (entity.existing_entity_id, entity.new_local_key) if value}
    specifications = {
        "prospective_matches": FastProspectiveMatch,
        "state_changes": FastStateChange,
        "keyframe_candidates": FastKeyframeCandidate,
        "perception_findings": PerceptionFinding,
    }
    for field, model in specifications.items():
        candidates = optional[field]
        accepted[field] = []
        if not isinstance(candidates, list) or len(candidates) > _list_limit(field):
            rejected.append(RejectedCandidate(field, None, "invalid_candidate_collection"))
            continue
        finding_indices = []
        for index, candidate in enumerate(candidates):
            try:
                if not isinstance(candidate, dict):
                    raise ValueError("invalid_candidate_shape")
                candidate = dict(candidate)
                if field == "prospective_matches":
                    if "subject_type" in candidate:
                        # Preserve explicit legacy identity checks. Never repair
                        # contradictory subject types or silently reassign work.
                        candidate["subject_id"] = references.subject(candidate["subject_type"], candidate.get("subject_id"))
                    else:
                        candidate["subject_type"], candidate["subject_id"] = references.subject_identity(candidate.get("subject_id"))
                    if isinstance(candidate.get("supporting_frame_ids"), list):
                        candidate["supporting_frame_ids"] = [references.frame(value) for value in candidate["supporting_frame_ids"]]
                elif field == "keyframe_candidates":
                    candidate["frame_id"] = references.frame(candidate.get("frame_id"))
                elif field == "perception_findings":
                    # Versions are trusted request data, never model-authored.
                    # Even an apparently correct supplied check_ref is rejected:
                    # the model contract carries only question_id/check_id.
                    if "check_ref" in candidate:
                        raise PerceptionReferenceError("model_authored_check_version")
                    question_id, check_ref = references.finding_identity(
                        candidate.get("question_id"), candidate.pop("check_id", None))
                    candidate["question_id"], candidate["check_ref"] = question_id, check_ref
                    if isinstance(candidate.get("evidence_frame_ids"), list):
                        candidate["evidence_frame_ids"] = [references.frame(value) for value in candidate["evidence_frame_ids"]]
                else:
                    subject = candidate.get("subject_local_key")
                    if isinstance(subject, str) and subject in references.entities:
                        subject = references.entity(subject)
                    if not isinstance(subject, str) or subject not in entity_ids:
                        raise PerceptionReferenceError("unknown_output_entity")
                    candidate["subject_local_key"] = subject
                parsed = model.model_validate(candidate)
                if field == "perception_findings":
                    validate_perception_finding(parsed, request=request)
                    finding_indices.append(index)
                accepted[field].append(parsed)
            except PerceptionReferenceError as error:
                rejected.append(RejectedCandidate(field, index, str(error)))
            except (ValidationError, ValueError, TypeError):
                rejected.append(RejectedCandidate(field, index, "invalid_candidate_shape"))

        if field == "perception_findings":
            counts = {}
            for finding in accepted[field]:
                counts[finding.question_id] = counts.get(finding.question_id, 0) + 1
            kept = []
            for index, finding in zip(finding_indices, accepted[field]):
                if counts[finding.question_id] > 1:
                    rejected.append(RejectedCandidate(field, index, "duplicate_question_finding"))
                else:
                    kept.append(finding)
            accepted[field] = kept

    accepted["interrupt_candidate"] = None
    if optional["interrupt_candidate"] is not None:
        try:
            interrupt = FastInterruptCandidate.model_validate(optional["interrupt_candidate"])
            if (request.active_focus or request.active_reminders) and interrupt.scope != "general":
                # An unassociated spoken finding must not leave a real search
                # active, allowing its later tracked match to announce again.
                # Do not infer an ID from prose or suppress unrelated alerts.
                rejected.append(RejectedCandidate("interrupt_candidate", None, "unattributed_interrupt_with_active_watch"))
            else:
                accepted["interrupt_candidate"] = interrupt
        except ValidationError:
            rejected.append(RejectedCandidate("interrupt_candidate", None, "invalid_candidate_shape"))

    # Reuse the existing final contract/reference checks and tag normalization.
    # Internal/storage values remain UUIDs and durable entity IDs, never handles.
    output = parse_fast_perception_output(core.model_copy(update=accepted).model_dump_json(), request=request)
    return ValidatedPerception(output, tuple(rejected))
