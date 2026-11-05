"""Opt-in focus owner adapter after canonical typed-evidence recording.

Main supplies the existing focus service, canonical binder, effect guard and an
owner-qualified async predicate evaluator. The evaluator receives literal data;
it returns VisualPredicateDecision, not a notification or lifecycle command. A
missing evaluator fails closed. No analyzer, scheduler, task or provider is started.

Integration: extend the existing brief binder with focus_dependencies(), routing
that complete group to CanonicalFocusInputs.validate before responsibility episode
validation. Call consume after PerceptionEvidenceConsumer.consume, then retain the
ordinary process_matches call for legacy results. Both use the same prospective
claim/reservation and attention route. Never call this on the realtime voice path.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
import math
from typing import Callable
from uuid import UUID

from pydantic import Field, ValidationError

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import request_document
from app.live_state.canonical import context_dependency
from app.live_state.models import ContractModel
from app.live_state.perception import CheckResolution, EvidenceReference
from app.live_state.perception_focus import CanonicalFocusInputs, FOCUS_CHECK
from app.perception.applicability import PerceptionEffectGuard
from app.perception.contracts import FastProspectiveMatch
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE
from app.reminders.visual_resolution import VisualPredicateDecision
from app.tasks.context_events import ContextIngress
from app.tasks.models import digest


DECISION_NAMESPACE = "perception:focus_predicate:v1"


class FocusEvidencePolicy(ContractModel):
    minimum_confidence: float = Field(default=0.9, ge=0.8, le=1)
    max_evidence_age_seconds: float = Field(default=10, gt=0, le=30)


@dataclass(frozen=True)
class FocusEvidenceOutcome:
    status: str
    reason: str | None = None


def reject(reason):
    raise SourceError("focus_evidence_" + reason, "The focus evidence is not currently authorized.")


async def candidate(connection, *, canonical, request, evidence, question_id, policy, camera_is_current):
    """Repeated at the owner boundary; never use the newest brief as a substitute."""
    now = canonical.clock()
    brief = request.perception_brief
    if (evidence.status != "recorded" or evidence.record is None or evidence.evidence is None
            or brief is None or brief.scope != canonical.scope or evidence.evidence.brief != brief
            or (request.owner_id, request.session_id) != (canonical.scope.owner_id, canonical.scope.session_id)
            or request.window_id != evidence.evidence.window_id
            or request.requested_at != evidence.evidence.requested_at
            or request.camera_connection_id is None or camera_is_current is None
            or request.camera_visual_session_id != evidence.evidence.camera_stream_epoch
            or camera_is_current(request, evidence.evidence.camera_stream_epoch) is not True
            or not brief.created_at <= now < brief.expires_at
            or evidence.evidence.analyzed_at > now):
        reject("assignment_or_camera")
    await canonical._lock_roots(connection, (evidence.record.payload_id,))
    stored = await canonical.context._record(connection, request.owner_id, evidence.record.id)
    body = stored.content or {}
    if (stored != evidence.record or not stored.is_current or stored.availability != "available"
            or stored.owner_id != request.owner_id or stored.session_id != request.session_id
            or stored.kind != "observation" or stored.trust != "derived" or stored.finality != "final"
            or stored.source_namespace != EVIDENCE_NAMESPACE
            or body.get("format") != "jarvis.perception_evidence.v1"
            or body.get("evidence") != evidence.evidence.model_dump(mode="json")
            or body.get("request_digest") != digest(request_document(request))
            or body.get("camera_connection_id") != str(request.camera_connection_id)):
        reject("source_changed")
    selected = await CanonicalFocusInputs(canonical).validate(connection, brief)
    question = selected.question
    if question.id != question_id:
        reject("question_changed")
    finding = next((item for item in evidence.evidence.findings if item.question_id == question_id), None)
    if finding is None or finding.check_ref != question.check_ref:
        reject("finding_changed")
    frames = {item.frame_id: item for item in request.frames}
    saved = {item.frame_id: item for item in evidence.evidence.frames}
    if set(frames) != set(saved):
        reject("frames_changed")
    for identity, frame in frames.items():
        original = saved[identity]
        if (frame.captured_at, frame.frame_sequence) != (original.captured_at, original.frame_sequence):
            reject("frames_changed")
    for identity in finding.evidence_frame_ids:
        frame = frames.get(identity)
        if (frame is None or not selected.focus.created_at <= frame.captured_at <= request.requested_at <= now
                or (now - frame.captured_at).total_seconds() > policy.max_evidence_age_seconds):
            reject("evidence_age_or_episode")
    return selected, finding


@dataclass(frozen=True)
class FocusEvidenceDecision:
    canonical: object
    request: object
    evidence: object
    record: object
    policy: FocusEvidencePolicy
    camera_is_current: Callable
    question_id: UUID
    focus_id: UUID

    async def validate(self, connection, focus):
        selected, finding = await candidate(connection, canonical=self.canonical, request=self.request,
            evidence=self.evidence, question_id=self.question_id, policy=self.policy,
            camera_is_current=self.camera_is_current)
        if focus != selected.focus or focus.owner_id != self.request.owner_id or focus.id != self.focus_id:
            reject("owner_changed")
        if focus.mode == "find_once" and focus.check_resolution is not None:
            reject("already_resolved")
        await self.canonical._lock_roots(connection, (self.record.payload_id,))
        fresh = await self.canonical.context._record(connection, focus.owner_id, self.record.id)
        body = fresh.content or {}
        question = selected.question
        if (fresh != self.record or not fresh.is_current or fresh.availability != "available"
                or fresh.owner_id != focus.owner_id or fresh.session_id != self.request.session_id
                or fresh.source_namespace != DECISION_NAMESPACE or fresh.kind != "observation"
                or fresh.trust != "derived" or fresh.finality != "final"
                or fresh.source_time is None or not self.evidence.evidence.analyzed_at <= fresh.source_time <= self.canonical.clock()
                or body.get("format") != "jarvis.focus_predicate.v1"
                or body.get("policy") != self.policy.model_dump(mode="json")
                or body.get("question") != question.model_dump(mode="json")
                or body.get("finding") != finding.model_dump(mode="json")
                or body.get("evidence_source") != context_dependency(self.evidence.record).model_dump(mode="json")
                or body.get("request_digest") != digest(request_document(self.request))):
            reject("decision_changed")
        verdict = VisualPredicateDecision.model_validate(body.get("decision"))
        if (finding.visibility != "visible" or finding.confidence < self.policy.minimum_confidence
                or not verdict.authorizes(self.policy)
                or len(set(verdict.supporting_frame_ids)) != len(verdict.supporting_frame_ids)
                or not set(verdict.supporting_frame_ids) <= set(finding.evidence_frame_ids)):
            reject("predicate_not_established")
        frames = {frame.frame_id: frame for frame in self.request.frames}
        episode = question.episode_or_object_ref
        return CheckResolution(basis="visual_observation", resolved_at=self.canonical.clock(),
            applicable_episode_or_object_ref=episode,
            source_refs=tuple(EvidenceReference(source=context_dependency(self.evidence.record),
                captured_at=frames[identity].captured_at, frame_id=identity, visibility="visible",
                episode_or_object_ref=episode) for identity in verdict.supporting_frame_ids) + (
                EvidenceReference(source=context_dependency(fresh), captured_at=fresh.source_time,
                                  episode_or_object_ref=episode),))


class FocusEvidenceAdapter:
    """One caller-owned, bounded evaluation; source/owner locks never span analysis.

    ``evaluate(focus, question, finding, frames)`` must establish the described
    target/observable condition, not merely visibility, and preserve uncertainty.
    Runtime composition must borrow its existing evaluator/limiter, not add an
    unbounded provider queue. No evaluator means no evidence-based settlement.
    """

    def __init__(self, *, canonical, prospective, evaluate=None, policy=None,
                 enabled=False, timeout_seconds=10):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Invalid focus evaluation timeout")
        self.canonical, self.prospective, self.evaluate = canonical, prospective, evaluate
        self.policy = policy or FocusEvidencePolicy()
        self.enabled, self.timeout_seconds = enabled, timeout_seconds
        self._busy, self._closed = False, False

    def retire(self):
        # Synchronous applicability fence, including cancellation-resistant evaluators.
        self._closed = True

    async def consume(self, *, request, evidence, effect_guard, camera_is_current):
        if not self.enabled or self._closed:
            return FocusEvidenceOutcome("disabled")
        if self.evaluate is None:
            return FocusEvidenceOutcome("validator_unavailable")
        if self._busy:
            return FocusEvidenceOutcome("busy")
        self._busy = True
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._consume(request, evidence, effect_guard, camera_is_current)
        except asyncio.CancelledError:
            raise
        except SourceError as error:
            return FocusEvidenceOutcome("not_applied", error.code)
        except (ValueError, TypeError, KeyError, ValidationError):
            return FocusEvidenceOutcome("not_applied", "focus_evidence_invalid")
        except Exception:
            return FocusEvidenceOutcome("unavailable", "focus_evidence_unconfirmed")
        finally:
            self._busy = False

    async def _consume(self, request, evidence, guard, camera_is_current):
        if (not isinstance(guard, PerceptionEffectGuard) or guard.brief != request.perception_brief
                or guard.binding != request.perception_brief_binding):
            reject("guard_mismatch")
        if evidence.status != "recorded" or evidence.evidence is None:
            return FocusEvidenceOutcome("no_findings")
        findings = [item for item in evidence.evidence.findings
                    if item.check_ref is not None and item.check_ref.source_namespace == FOCUS_CHECK]
        if not findings:
            return FocusEvidenceOutcome("no_findings")
        if len(findings) != 1:
            reject("ambiguous")
        finding = findings[0]

        def current(request, stream):
            return not self._closed and camera_is_current is not None and camera_is_current(request, stream) is True

        async def inspect(connection):
            return await candidate(connection, canonical=self.canonical, request=request, evidence=evidence,
                question_id=finding.question_id, policy=self.policy, camera_is_current=current)

        # Lock all evidence roots before the focus owner, including on re-entry.
        @asynccontextmanager
        async def evidence_guard(brief, binding, *, effect_payload_ids=()):
            async with guard.canonical_guard(brief, binding,
                    effect_payload_ids=tuple(sorted({*effect_payload_ids, evidence.record.payload_id}))) as connection:
                yield connection

        captured_guard = replace(guard, canonical_guard=evidence_guard,
            is_live=lambda: guard.is_live() and not self._closed)
        async with captured_guard() as connection:
            selected, finding = await inspect(connection)
        if finding.visibility != "visible" or finding.confidence < self.policy.minimum_confidence:
            return FocusEvidenceOutcome("observed_without_resolution")
        frames = tuple(frame for frame in request.frames if frame.frame_id in finding.evidence_frame_ids)
        if sum(len(frame.image_bytes) for frame in frames) > 4_000_000:
            reject("image_budget")
        decision = VisualPredicateDecision.model_validate(
            await self.evaluate(selected.focus, selected.question, finding, frames))
        if (not decision.authorizes(self.policy)
                or len(set(decision.supporting_frame_ids)) != len(decision.supporting_frame_ids)
                or not set(decision.supporting_frame_ids) <= set(finding.evidence_frame_ids)):
            return FocusEvidenceOutcome("observed_without_resolution")
        # Re-enter the ORIGINAL assignment after analysis. Replacement, clear,
        # source withdrawal, expiry and camera retirement all fail before write.
        async with captured_guard() as connection:
            latest, latest_finding = await inspect(connection)
            if latest != selected or latest_finding != finding:
                reject("changed_during_analysis")
            body = {"format": "jarvis.focus_predicate.v1", "policy": self.policy.model_dump(mode="json"),
                "question": selected.question.model_dump(mode="json"), "finding": finding.model_dump(mode="json"),
                "decision": decision.model_dump(mode="json"),
                "evidence_source": context_dependency(evidence.record).model_dump(mode="json"),
                "request_digest": digest(request_document(request)),
                "meaning": "Qualified watch match only; no physical action or delivery is established."}
            receipt = await self.canonical.context.append_in_transaction(connection, request.owner_id, ContextIngress(
                session_id=request.session_id, source_namespace=DECISION_NAMESPACE,
                source_event_id=f"{selected.focus.id}/{selected.focus.revision}/{evidence.record.id}/{finding.question_id}",
                receipt_id="decision", kind="observation", trust="derived", source_time=self.canonical.clock(),
                expected_revision=0, payload=body,
                source_payload_ids=(evidence.record.payload_id, *request.perception_brief_binding.payload_root_ids)))
        qualified = FocusEvidenceDecision(self.canonical, request, evidence, receipt, self.policy, current,
                                         finding.question_id, selected.focus.id)

        @asynccontextmanager
        async def decision_guard(brief, binding, *, effect_payload_ids=()):
            async with evidence_guard(brief, binding,
                    effect_payload_ids=(*effect_payload_ids, receipt.payload_id)) as connection:
                yield connection

        final_guard = replace(captured_guard, canonical_guard=decision_guard)
        # Message reports observation, not user action or confirmed playback.
        message = ("I see a match for " + selected.focus.target + ": " + finding.literal_finding)[:500]
        match = FastProspectiveMatch(subject_type="visual_focus", subject_id=selected.focus.id,
            confidence=min(finding.confidence, decision.confidence),
            observed_condition=decision.observed_condition[:500],
            supporting_frame_ids=list(decision.supporting_frame_ids), suggested_message=message)
        await self.prospective.process_matches(request=request, matches=[match],
            source_payload_id=guard.observation_payload_id, effect_guard=final_guard, focus_decision=qualified)
        return FocusEvidenceOutcome("owner_processed")
