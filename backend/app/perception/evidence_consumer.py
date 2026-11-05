"""Opt-in assignment evidence ingress, not a second check/notification authority.

The pipeline supplies its ORIGINAL request and PerceptionEffectGuard. That guard
owns current episode/check/focus/cue applicability and borrows the canonical
transaction; this module must not reconstruct those decisions from model prose.
Accepted findings become removable, replay-safe canonical context observations.

Neither existing owner currently exposes a qualified visual predicate settlement
API. Accordingly, a recorded receipt is evidence ONLY: it does not modify a
check, clear a focus, complete a physical action, or create a notification. The
receipt's exact references can be passed to a future qualified owner adapter,
which must repeat applicability checks at its own effect boundary.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Literal
from uuid import UUID

from asyncpg import PostgresError

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.knowledge.observations import OBSERVATION_NAMESPACE, ObservationPublication
from app.live_state.canonical import context_dependency
from app.live_state.episode_context import validate_episode_dependencies
from app.live_state.perception_focus import (
    CanonicalFocusInputs, FOCUS_CHECK, FOCUS_SUBJECT, focus_dependencies,
)
from app.live_state.perception import (
    EvidenceReference, ObservationalQuestion, PerceptionBrief, PerceptionEvidence, PerceptionFinding,
    PerceptionFrameReference,
)
from app.perception.applicability import PerceptionEffectGuard, PerceptionEffectUnavailable
from app.perception.contracts import (
    FastPerceptionOutput, FastPerceptionOutputError, FastPerceptionRequest,
    bound_perception_questions, validate_perception_finding,
)
from app.perception.models import VisualObservation
from app.reminders.episode_checks import EpisodeCheckSnapshot
from app.tasks.context_events import ContextIngress, ContextRecord
from app.tasks.context_repository import RECORD_SQL, context_record
from app.tasks.models import TaskError, digest


EVIDENCE_NAMESPACE = "perception:assignment_evidence:v1"
EpisodeSnapshots = Callable[[object, PerceptionBrief], Awaitable[Sequence[EpisodeCheckSnapshot]]]


@dataclass(frozen=True)
class OwnerFindingEvidence:
    question: ObservationalQuestion
    finding: PerceptionFinding
    source_refs: tuple[EvidenceReference, ...]
    # Literal observations are not a qualified intended-predicate decision.
    resolution_status: Literal["validator_unavailable"] = "validator_unavailable"


@dataclass(frozen=True)
class EvidenceConsumption:
    status: Literal["disabled", "no_findings", "recorded", "not_applied", "unavailable"]
    reason: str | None = None
    evidence: PerceptionEvidence | None = None
    record: ContextRecord | None = None
    owner_findings: tuple[OwnerFindingEvidence, ...] = ()


def _reject(reason):
    raise SourceError("perception_evidence_" + reason, "Assignment evidence could not be applied.")


def canonical_episode_snapshots(canonical) -> EpisodeSnapshots:
    """Explicitly bind the SAME canonical binder used by the effect guard.

    This reads through the existing episode owner on the borrowed transaction.
    It does not inspect a bound guard method, create another owner, or replace
    the effect guard's cue/applicability checks. Application composition supplies
    this callback; absence is not permission to guess an episode start time.
    """
    async def snapshots(connection, brief):
        if brief.scope != canonical.scope:
            _reject("episode_scope_mismatch")
        if not connection.is_in_transaction():
            raise RuntimeError("Episode evidence requires the existing effect transaction")
        dependencies = brief.dependencies
        if (any(ref.source_namespace in (FOCUS_CHECK, FOCUS_SUBJECT) for ref in dependencies)
                or any(question.check_ref is not None and question.check_ref.source_namespace == FOCUS_CHECK
                       for question in brief.questions)):
            # The shared episode namespace is not blanket permission to drop
            # references. Recheck the complete focus assignment and removable
            # receipt through its owner on this SAME effect transaction first.
            await CanonicalFocusInputs(canonical).validate(connection, brief)
            owned = focus_dependencies(brief)
            if (len(owned) != 3 or len(set(owned)) != 3
                    or any(ref not in owned for ref in dependencies
                           if ref.source_namespace in (FOCUS_CHECK, FOCUS_SUBJECT))):
                _reject("focus_partition_invalid")
            dependencies = tuple(ref for ref in dependencies if ref not in owned)
        return await validate_episode_dependencies(canonical, connection, dependencies)
    return snapshots


class PerceptionEvidenceConsumer:
    """One bounded canonical write on the background analyzer path.

    ``camera_is_current(request, stream)`` validates the ORIGINAL captured
    connection identity through the session's real camera owner. It must become
    false on stop/replacement, even if visual_session_id is reused. New frames
    alone must not invalidate that connection. The older ``stream_is_current``
    fixture seam is used only without captured connection fields and without a
    camera callback; it is not a production substitute for socket ownership.
    Missing proof fails closed. Camera ownership and the existing effect guard
    are checked within the write transaction, with no provider/network work.

    Use the existing ObservationPublication instance. This object borrows it and
    owns no services, tasks, database schema, queue or independent lifecycle.
    """

    def __init__(self, publication: ObservationPublication, *, enabled=False,
                 stream_is_current: Callable[[str, UUID, UUID], bool] | None = None,
                 camera_is_current: Callable[[FastPerceptionRequest, UUID], bool] | None = None,
                 episode_snapshots: EpisodeSnapshots | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 timeout_seconds: float = 2.0):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Evidence timeout must be finite and positive")
        self.publication, self.enabled = publication, enabled
        self.stream_is_current, self.clock = stream_is_current, clock
        self.camera_is_current = camera_is_current
        self.episode_snapshots = episode_snapshots
        self.timeout_seconds = timeout_seconds

    def _current(self, request, effect_guard, stream):
        self.publication._authorize(request.owner_id)
        brief = request.perception_brief
        if not self.enabled or not effect_guard.is_live():
            raise PerceptionEffectUnavailable()
        if not brief.created_at <= self.clock() < brief.expires_at:
            _reject("assignment_expired")
        if request.camera_connection_id is not None and request.camera_visual_session_id != stream:
            _reject("camera_stream_mismatch")
        if self.camera_is_current is not None:
            current = self.camera_is_current(request, stream)
        elif request.camera_connection_id is not None:
            # A UUID-only predicate cannot establish the captured connection.
            current = False
        else:
            current = self.stream_is_current is not None and self.stream_is_current(
                request.owner_id, request.session_id, stream)
        if current is not True:
            _reject("camera_stream_unavailable")

    async def _validate_episode_frames(self, connection, request, findings, questions, frames):
        assigned = tuple(finding for finding in findings if finding.check_ref is not None
                         and finding.check_ref.source_namespace == "responsibility_check")
        if not assigned:
            return
        if self.episode_snapshots is None:
            _reject("episode_snapshot_unavailable")
        snapshots = await self.episode_snapshots(connection, request.perception_brief)
        if not isinstance(snapshots, (tuple, list)) or len(snapshots) > 16:
            _reject("episode_snapshot_invalid")
        by_check = {}
        for snapshot in snapshots:
            if not isinstance(snapshot, EpisodeCheckSnapshot) or snapshot.check.id in by_check:
                _reject("episode_snapshot_invalid")
            by_check[snapshot.check.id] = snapshot
        for finding in assigned:
            question = questions[finding.question_id]
            snapshot = by_check.get(finding.check_ref.source_id)
            if snapshot is None:
                _reject("episode_snapshot_unavailable")
            check = snapshot.check
            owner = check.canonical_owner_ref
            if (check.scope != request.perception_brief.scope or check.revision != finding.check_ref.revision
                    or finding.check_ref.generation is not None or check.continuity != "durable"
                    or check.lifecycle.value != "tracking" or owner is None
                    or owner.source_namespace != "responsibility" or owner.generation is not None
                    or question.episode_or_object_ref != check.episode_or_object_ref
                    or any(ref not in snapshot.dependencies or ref not in request.perception_brief.dependencies
                           for ref in (owner, finding.check_ref, check.episode_or_object_ref))):
                _reject("episode_snapshot_mismatch")
            # Every cited frame must belong to this episode. Checking only the
            # newest/primary frame would let an older supporting frame establish
            # a condition for a new episode. Equality at the original start is OK.
            if any(frames[identity]["captured_at"] < snapshot.payload.episode_started_at
                   for identity in finding.evidence_frame_ids):
                _reject("frame_predates_episode")

    async def consume(self, *, request: FastPerceptionRequest, output: FastPerceptionOutput,
                      observation: VisualObservation, analysis_input: ObservationInputSnapshot | None,
                      effect_guard: PerceptionEffectGuard | None) -> EvidenceConsumption:
        if not self.enabled:
            return EvidenceConsumption("disabled")
        if request.perception_brief is None or not output.perception_findings:
            return EvidenceConsumption("no_findings")
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._consume(request, output, observation, analysis_input, effect_guard)
        except (SourceError, TaskError) as error:
            return EvidenceConsumption("not_applied", error.code)
        except (FastPerceptionOutputError, ValueError, TypeError, KeyError):
            return EvidenceConsumption("not_applied", "perception_evidence_invalid")
        except (PostgresError, OSError, TimeoutError):
            # A missing commit receipt is not proof the write never happened.
            # Retrying the SAME observation identity is idempotent, not a new effect.
            return EvidenceConsumption("unavailable", "perception_evidence_write_unconfirmed")

    async def _consume(self, request, output, observation, analysis_input, effect_guard):
        brief = request.perception_brief
        if (not isinstance(effect_guard, PerceptionEffectGuard)
                or effect_guard.brief != brief or effect_guard.binding != request.perception_brief_binding
                or not isinstance(analysis_input, ObservationInputSnapshot)
                or (analysis_input.owner_id, analysis_input.session_id) != (request.owner_id, request.session_id)
                or request.perception_brief_input_digest != analysis_input.request_digest
                or digest(request_document(request)) != analysis_input.request_digest):
            _reject("capture_or_guard_mismatch")
        if (observation.owner_id, observation.session_id, observation.primary_visual_frame_id,
                observation.started_at, observation.ended_at) != (
                request.owner_id, request.session_id, request.primary_frame.frame_id,
                request.frames[0].captured_at, request.primary_frame.captured_at):
            _reject("observation_mismatch")
        if not request.requested_at <= observation.created_at <= self.clock():
            _reject("analysis_time_invalid")
        questions = bound_perception_questions(request)
        findings = tuple(output.perception_findings)
        for finding in findings:
            validate_perception_finding(finding, request=request, questions=questions)

        # This is the same canonical transaction, including owner cancellation,
        # snooze/replacement, original cue relevance and source-generation guards.
        # No comparison with a global Live State revision or newest whole brief.
        async with effect_guard() as connection:
            await self.publication.inputs.check(connection, analysis_input,
                owner_id=request.owner_id, session_id=request.session_id,
                frame_ids=tuple(frame.frame_id for frame in request.frames))
            row = await connection.fetchrow(RECORD_SQL + """
                join jarvis_observation_sources b on b.owner_id=e.owner_id and b.payload_id=e.payload_id
                where b.owner_id=$1 and b.session_id=$2 and b.observation_id=$3
                  and e.source_namespace=$4 and e.source_event_id=$5""",
                request.owner_id, request.session_id, observation.id, OBSERVATION_NAMESPACE, str(observation.id))
            if row is None:
                _reject("original_unavailable")
            original = context_record(row)
            content = original.content or {}
            stored = content.get("record", {})
            if (original.owner_id != request.owner_id or original.session_id != request.session_id
                    or original.kind != "observation" or original.trust != "derived"
                    or original.finality not in {"final", "revised"} or not original.is_current
                    or original.availability != "available" or original.source_time != observation.ended_at
                    or original.payload_id != effect_guard.observation_payload_id
                    or content.get("analysis_input_id") != str(analysis_input.payload_id)
                    or stored.get("id") != str(observation.id)
                    or stored.get("owner_id") != request.owner_id
                    or stored.get("session_id") != str(request.session_id)
                    or stored.get("primary_visual_frame_id") != str(request.primary_frame.frame_id)
                    or stored.get("summary") != output.summary
                    or datetime.fromisoformat(stored.get("created_at", "")) != observation.created_at
                    or datetime.fromisoformat(stored.get("started_at", "")) != observation.started_at
                    or datetime.fromisoformat(stored.get("ended_at", "")) != observation.ended_at
                    or tuple(item["frame_id"] for item in content.get("frame_links", ()))
                        != tuple(str(frame.frame_id) for frame in request.frames)):
                _reject("original_mismatch")
            frames = await connection.fetch("""select id,owner_id,session_id,visual_session_id,
                frame_sequence,captured_at,source,content_type from visual_frames
                where owner_id=$1 and session_id=$2 and id=any($3::uuid[]) for share""",
                request.owner_id, request.session_id, [frame.frame_id for frame in request.frames])
            by_id = {frame["id"]: frame for frame in frames}
            if len(frames) != len(request.frames) or set(by_id) != request.allowed_frame_ids:
                _reject("frame_window_mismatch")
            streams = {frame["visual_session_id"] for frame in frames}
            if len(streams) != 1 or not isinstance(next(iter(streams)), UUID):
                _reject("mixed_camera_streams")
            stream = next(iter(streams))
            for frame in request.frames:
                saved = by_id[frame.frame_id]
                if (saved["owner_id"], saved["session_id"], saved["frame_sequence"],
                        saved["captured_at"], saved["source"], saved["content_type"]) != (
                        request.owner_id, request.session_id, frame.frame_sequence,
                        frame.captured_at, frame.source, frame.content_type):
                    _reject("frame_metadata_changed")
            self._current(request, effect_guard, stream)
            await self._validate_episode_frames(connection, request, findings, questions, by_id)
            self._current(request, effect_guard, stream)
            source = context_dependency(original)
            evidence = PerceptionEvidence(brief=brief, window_id=request.window_id,
                camera_stream_epoch=stream, requested_at=request.requested_at,
                analyzed_at=observation.created_at, findings=findings,
                frames=tuple(PerceptionFrameReference(frame_id=frame.frame_id,
                    frame_sequence=frame.frame_sequence, captured_at=frame.captured_at, source=source)
                    for frame in request.frames))
            parents = tuple(sorted({original.payload_id, analysis_input.payload_id,
                                   *request.perception_brief_binding.payload_root_ids}))
            ingress = ContextIngress(session_id=request.session_id, source_namespace=EVIDENCE_NAMESPACE,
                source_event_id=str(observation.id), receipt_id=str(observation.id),
                kind="observation", trust="derived", source_time=observation.ended_at,
                expected_revision=0, source_payload_ids=parents, payload={
                    "format": "jarvis.perception_evidence.v1",
                    "observation_id": str(observation.id), "original_source": source.model_dump(mode="json"),
                    "analysis_input_id": str(analysis_input.payload_id),
                    "request_digest": analysis_input.request_digest,
                    **({"camera_connection_id": str(request.camera_connection_id)}
                       if request.camera_connection_id is not None else {}),
                    "evidence": evidence.model_dump(mode="json"),
                    # Keep exact structured gaps in the removable envelope, not
                    # invented summaries or falsely empty PerceptionEvidence gaps.
                    "input_context_gaps": list(request.input_context_gaps),
                    "owner_effect": "none", "resolution_status": "validator_unavailable",
                    "meaning": "Literal model findings under the original assignment; not proof of a physical action. "
                        "Recorded applicability is not continuing permission. Recheck owners and sources before effects.",
                })
            record = await self.publication.context.append_in_transaction(connection, request.owner_id, ingress)
            if (record.owner_id != request.owner_id or record.session_id != request.session_id
                    or record.source_namespace != ingress.source_namespace
                    or record.source_event_id != ingress.source_event_id or record.source_time != ingress.source_time
                    or record.kind != "observation" or record.trust != "derived" or record.finality != "final"
                    or not record.is_current or record.availability != "available" or record.content != ingress.payload):
                _reject("receipt_unavailable")
            # A replayed ingress receipt does not itself reacquire its originals.
            if not await connection.fetchval("select lock_jarvis_copy_sources($1,$2::uuid[])",
                    request.owner_id, [*parents, record.payload_id]):
                _reject("source_withdrawn")
            self._current(request, effect_guard, stream)
            recorded_source = context_dependency(record)
            frame_times = {frame.frame_id: frame.captured_at for frame in request.frames}
            owner_findings = tuple(OwnerFindingEvidence(question=questions[finding.question_id], finding=finding,
                source_refs=tuple(EvidenceReference(source=recorded_source, captured_at=frame_times[identity],
                    episode_or_object_ref=questions[finding.question_id].episode_or_object_ref,
                    frame_id=identity, visibility=finding.visibility) for identity in finding.evidence_frame_ids))
                for finding in findings)
        # A real context record AND successful transaction exit precede success.
        return EvidenceConsumption("recorded", evidence=evidence, record=record, owner_findings=owner_findings)
