"""Pure, opt-in brief composition. No admission, lifecycle effects or I/O.

The caller supplies current scoped projections and a nonblocking, scope-bound
dependency predicate. Schema validity and absence from an invalidation list are
not authorization. Reuse the returned composition on the next call; losing that
state requires a new scope_epoch to avoid reusing brief identities/versions.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import json
from uuid import UUID, uuid5

from app.live_state.models import LiveSnapshot, SourceDependency
from app.live_state.perception import (
    CheckLifecycle,
    DurableResponsibilityCapability,
    MAX_QUESTIONS,
    ObservationalQuestion,
    PerceptionActivity,
    PerceptionBrief,
    PriorEvidence,
    SituationalCheck,
)


MAX_CHECK_INPUTS = 128
MAX_EXPLICIT_INPUTS = 8
MAX_PRIOR_INPUTS = 8
MAX_DEPENDENCIES = 16
MAX_INVALIDATIONS = 256
_PRIORITY = {"explicit_inspection": 0, "high": 1, "normal": 2, "low": 3}
_SCENE_CUE = (
    "Report one meaningful scene or activity transition outside the assigned "
    "questions; retain uncertainty and do not infer completed actions."
)


@dataclass(frozen=True, slots=True)
class PerceptionSelectionDiagnostic:
    kind: str
    reason: str
    candidate_id: UUID | None = None
    count: int = 1


@dataclass(frozen=True, slots=True)
class PerceptionComposition:
    brief: PerceptionBrief
    diagnostics: tuple[PerceptionSelectionDiagnostic, ...]
    semantic_key: str
    selection_round: int
    reused: bool
    selection_history: tuple[tuple[str, UUID, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _Candidate:
    question: ObservationalQuestion
    dependencies: tuple[SourceDependency, ...]


def _identity(source: SourceDependency) -> tuple[str, UUID]:
    return source.source_namespace, source.source_id


def _merge_dependencies(left, right):
    merged = {_identity(source): source for source in left}
    for source in right:
        key = _identity(source)
        if key in merged and merged[key] != source:
            return (), "conflicting_dependency_versions"
        merged[key] = source
        if len(merged) > MAX_DEPENDENCIES:
            return (), "dependency_budget"
    return tuple(merged[key] for key in sorted(merged, key=lambda key: (key[0], str(key[1])))), None


def _question_dependencies(question):
    return tuple(source for source in (
        question.check_ref, question.episode_or_object_ref, question.focus_ref,
    ) if source is not None)


def _check_reference(check):
    owner = check.canonical_owner_ref
    namespace = (
        "responsibility_check" if owner is not None and owner.source_namespace == "responsibility"
        else "visual_focus" if owner is not None and owner.source_namespace == "visual_focus"
        else "situational_check"
    )
    return SourceDependency(source_namespace=namespace, source_id=check.id, revision=check.revision)


def compose_perception_brief(
    snapshot: LiveSnapshot,
    *,
    now: datetime,
    scope_epoch: UUID,
    checks: tuple[SituationalCheck, ...] = (),
    explicit_questions: tuple[ObservationalQuestion, ...] = (),
    activity: PerceptionActivity | None = None,
    intent_ref: SourceDependency | None = None,
    episode_or_object_ref: SourceDependency | None = None,
    context_dependencies: tuple[SourceDependency, ...] = (),
    prior_evidence: tuple[PriorEvidence, ...] = (),
    dependency_eligible: Callable[[SourceDependency], bool] | None = None,
    invalidated_sources: tuple[SourceDependency, ...] = (),
    previous: PerceptionComposition | None = None,
) -> PerceptionComposition:
    """Select questions without changing supplied canonical rows or granting speech.

    Resupply current inputs, including original prior-evidence timestamps, each
    time. Explicit questions are ordered by priority, then caller-supplied FIFO;
    they precede generated check questions. Equal-priority checks rotate only on
    semantic publication or expiry, not on ordinary cache reads.

    The predicate must authorize the exact namespace/id/revision/generation in
    snapshot.scope, including check/episode/focus references and owner availability.
    Missing predicates fail closed for source-backed input. Invalidations deny an
    entire source identity, regardless of version, until the caller explicitly
    removes that denial after authoritative restoration. This is not source-order
    reconciliation or a replacement for downstream source/copy/output guards.

    Snapshot text entries are not parsed into goals, obligations or evidence.
    Unreferenced explicit questions/activity are trusted, already-scoped caller
    inputs. Their user provenance cannot be established by these schemas.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    same_scope = (
        previous is not None and previous.brief.scope == snapshot.scope
        and previous.brief.scope_epoch == scope_epoch
    )
    if same_scope and now < previous.brief.created_at:
        raise ValueError("composition time cannot move behind the previous brief")
    diagnostics: list[PerceptionSelectionDiagnostic] = []

    def note(kind, reason, candidate_id=None, count=1):
        diagnostics.append(PerceptionSelectionDiagnostic(kind, reason, candidate_id, count))

    def bounded(values, limit, kind):
        if len(values) > limit:
            note(kind, "input_budget", count=len(values) - limit)
        return values[:limit]

    invalidation_overflow = len(invalidated_sources) > MAX_INVALIDATIONS
    if invalidation_overflow:
        note("source", "invalidation_budget_fail_closed", count=len(invalidated_sources))
    denied = set() if invalidation_overflow else {_identity(source) for source in invalidated_sources}
    eligibility: dict[SourceDependency, str | None] = {}

    def admitted(dependencies):
        sources, reason = _merge_dependencies((), dependencies)
        if reason is not None:
            return (), reason
        for source in sources:
            if source not in eligibility:
                if invalidation_overflow or _identity(source) in denied:
                    eligibility[source] = "source_invalidated_or_retired"
                elif dependency_eligible is None:
                    eligibility[source] = "dependency_eligibility_unknown"
                else:
                    try:
                        eligibility[source] = None if dependency_eligible(source) is True else "dependency_ineligible"
                    except Exception:
                        eligibility[source] = "dependency_eligibility_error"
            if eligibility[source] is not None:
                return (), eligibility[source]
        return sources, None

    capability = DurableResponsibilityCapability(
        owner_namespace="responsibility", status=snapshot.capabilities.durable_responsibilities,
    )
    if snapshot.readiness != "ready":
        note("snapshot", snapshot.readiness)
    if len(context_dependencies) > MAX_DEPENDENCIES:
        base_dependencies, context_error = (), "dependency_budget"
    else:
        base_dependencies, context_error = admitted(context_dependencies + tuple(
            source for source in (intent_ref, episode_or_object_ref) if source is not None
        ))
    if context_error is not None:
        note("context", context_error)
        activity, intent_ref, episode_or_object_ref = None, None, None

    namespace = uuid5(scope_epoch, snapshot.scope.model_dump_json())
    check_inputs = bounded(checks, MAX_CHECK_INPUTS, "check")
    counts: dict[tuple[str, UUID], int] = {}
    for check in check_inputs:
        if check.scope == snapshot.scope:
            key = _identity(_check_reference(check))
            counts[key] = counts.get(key, 0) + 1
    eligible_checks: dict[tuple[str, UUID], _Candidate] = {}
    for check in check_inputs:
        check_ref = _check_reference(check)
        owner = check.canonical_owner_ref
        reason = None
        if check.scope != snapshot.scope:
            reason = "scope_mismatch"
        elif counts[_identity(check_ref)] != 1:
            reason = "ambiguous_check_projection"
        elif check.lifecycle not in (CheckLifecycle.CANDIDATE, CheckLifecycle.TRACKING):
            reason = "check_inactive"
        elif check.snoozed_until is not None and check.snoozed_until > now:
            reason = "check_snoozed"
        elif check.observation_window_expires_at is not None and check.observation_window_expires_at <= now:
            reason = "observation_window_expired"
        elif not check.relevance_now:
            reason = "check_not_relevant"
        elif owner is not None and owner.source_namespace not in ("responsibility", "visual_focus"):
            reason = "unsupported_canonical_owner"
        elif owner is not None and owner.source_namespace == "responsibility" and capability.status != "available":
            reason = "responsibility_owner_unavailable"
        elif owner is not None and owner.source_namespace == "visual_focus" and (
            owner.source_id != check.id or owner.revision != check.revision
        ):
            reason = "focus_owner_binding_mismatch"
        if reason is not None:
            note("check", reason, check.id)
            continue
        question = ObservationalQuestion(
            id=uuid5(namespace, f"{check_ref.source_namespace}:{check.id}:{check.revision}"),
            question=check.question,
            priority="high" if check.origin == "explicit_request" else "normal",
            check_ref=check_ref, episode_or_object_ref=check.episode_or_object_ref,
            focus_ref=owner if owner is not None and owner.source_namespace == "visual_focus" else None,
            request_grounding=check.request_grounding,
        )
        sources, reason = admitted(
            check.dependency_versions + _question_dependencies(question) + ((owner,) if owner is not None else ())
        )
        if reason is not None:
            note("check", reason, check.id)
            continue
        eligible_checks[_identity(check_ref)] = _Candidate(question, sources)

    explicit_inputs = bounded(explicit_questions, MAX_EXPLICIT_INPUTS, "explicit_question")
    explicit_counts: dict[UUID, int] = {}
    for question in explicit_inputs:
        explicit_counts[question.id] = explicit_counts.get(question.id, 0) + 1
    explicit: list[_Candidate] = []
    for question in explicit_inputs:
        if explicit_counts[question.id] != 1:
            note("explicit_question", "ambiguous_question_identity", question.id)
            continue
        dependencies = _question_dependencies(question)
        if question.check_ref is not None:
            current = eligible_checks.get(_identity(question.check_ref))
            if current is None:
                note("explicit_question", "check_inactive_or_missing", question.id)
                continue
            if (question.check_ref != current.question.check_ref
                    or question.episode_or_object_ref != current.question.episode_or_object_ref):
                note("explicit_question", "check_or_episode_version_mismatch", question.id)
                continue
            dependencies += current.dependencies
        sources, reason = admitted(dependencies)
        if reason is not None:
            note("explicit_question", reason, question.id)
            continue
        explicit.append(_Candidate(question, sources))
    explicit.sort(key=lambda item: _PRIORITY[item.question.priority])
    generated = sorted(eligible_checks.values(), key=lambda item: (
        _PRIORITY[item.question.priority], str(item.question.id),
    ))
    priors: list[tuple[PriorEvidence, tuple[SourceDependency, ...]]] = []
    for prior in bounded(prior_evidence, MAX_PRIOR_INPUTS, "prior_evidence"):
        if prior.reference.captured_at > now:
            note("prior_evidence", "future_capture", prior.reference.source.source_id)
            continue
        sources, reason = admitted((prior.reference.source,) + (
            (prior.reference.episode_or_object_ref,) if prior.reference.episode_or_object_ref is not None else ()
        ))
        if reason is not None:
            note("prior_evidence", reason, prior.reference.source.source_id)
            continue
        priors.append((prior, sources))
    priors.sort(key=lambda item: (-item[0].reference.captured_at.timestamp(), item[0].model_dump_json()))

    # Do not hash global revision/as_of, task entries, acknowledgements or read time.
    semantic = {
        "context_loading": snapshot.readiness == "warming",
        "scope": snapshot.scope.model_dump(mode="json"), "scope_epoch": str(scope_epoch),
        "state_epoch": str(snapshot.cursor.epoch), "capability": capability.model_dump(mode="json"),
        "activity": activity.model_dump(mode="json") if activity is not None else None,
        "intent": intent_ref.model_dump(mode="json") if intent_ref is not None else None,
        "episode": episode_or_object_ref.model_dump(mode="json") if episode_or_object_ref is not None else None,
        "context_dependencies": [source.model_dump(mode="json") for source in base_dependencies],
        "candidates": [
            [item.question.model_dump(mode="json"), [source.model_dump(mode="json") for source in item.dependencies]]
            for item in (*explicit, *generated)
        ],
        "explicit_count": len(explicit),
        "prior_evidence": [item.model_dump(mode="json") for item, _ in priors],
    }
    semantic_key = sha256(json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reused = bool(same_scope and previous.semantic_key == semantic_key and now < previous.brief.expires_at)
    selection_round = previous.selection_round if reused else previous.selection_round + 1 if same_scope else 0
    last_selected = ({(namespace, identity): round_number
                      for namespace, identity, round_number in previous.selection_history}
                     if same_scope else {})
    previous_order = {}
    if same_scope:
        for index, question in enumerate(previous.brief.questions):
            if question.check_ref is not None:
                key = _identity(question.check_ref)
                last_selected.setdefault(key, previous.selection_round)
                previous_order[key] = index
    # Retain only currently eligible checks: removed scopes/checks are not a
    # second responsibility store, and this projection stays bounded by input.
    last_selected = {key: value for key, value in last_selected.items() if key in eligible_checks}
    ordered_checks: list[_Candidate] = []
    for priority in _PRIORITY:
        group = [item for item in generated if item.question.priority == priority]
        group.sort(key=lambda item: (
            previous_order.get(_identity(item.question.check_ref), MAX_QUESTIONS) if reused
            else last_selected.get(_identity(item.question.check_ref), -1),
            str(item.question.id),
        ))
        ordered_checks.extend(group)

    selected: list[ObservationalQuestion] = []
    selected_checks: set[tuple[str, UUID]] = set()
    dependencies = base_dependencies
    for kind, candidates in (("explicit_question", explicit), ("check", ordered_checks)):
        for candidate in candidates:
            question = candidate.question
            candidate_id = question.check_ref.source_id if kind == "check" else question.id
            if kind == "check" and _identity(question.check_ref) in selected_checks:
                note(kind, "covered_by_explicit_question", candidate_id)
                continue
            if len(selected) >= MAX_QUESTIONS:
                note(kind, "question_budget", candidate_id)
                continue
            if any(item.id == question.id for item in selected):
                note(kind, "question_identity_conflict", candidate_id)
                continue
            merged, reason = _merge_dependencies(dependencies, candidate.dependencies)
            if reason is not None:
                note(kind, reason, candidate_id)
                continue
            selected.append(question)
            dependencies = merged
            if question.check_ref is not None:
                selected_checks.add(_identity(question.check_ref))
    selected_prior: list[PriorEvidence] = []
    for prior, sources in priors:
        merged, reason = _merge_dependencies(dependencies, sources)
        if reason is not None:
            note("prior_evidence", reason, prior.reference.source.source_id)
            continue
        selected_prior.append(prior)
        dependencies = merged

    if reused:
        composed = previous.brief
    else:
        composed = PerceptionBrief(
            id=previous.brief.id if same_scope else uuid5(namespace, "perception-brief"),
            version=previous.brief.version + 1 if same_scope else 1,
            scope=snapshot.scope, scope_epoch=scope_epoch,
            based_on_live_revision=snapshot.cursor.revision,
            activity=activity, intent_ref=intent_ref, episode_or_object_ref=episode_or_object_ref,
            questions=tuple(selected), prior_evidence=tuple(selected_prior),
            scene_change_cues=(_SCENE_CUE + (
                " Background context is still loading. Inspect the current scene; "
                "missing prior context is not evidence that an object or responsibility is absent."
                if snapshot.readiness == "warming" else ""
            ),), dependencies=dependencies,
            durable_responsibility=capability,
            created_at=now, fresh_until=now + timedelta(seconds=10),
            expires_at=now + timedelta(seconds=30),
        )
    if not reused:
        for key in selected_checks:
            last_selected[key] = selection_round
    history = tuple((key[0], key[1], last_selected[key])
                    for key in sorted(last_selected, key=lambda item: (item[0], str(item[1]))))
    return PerceptionComposition(composed, tuple(diagnostics), semantic_key, selection_round, reused, history)
