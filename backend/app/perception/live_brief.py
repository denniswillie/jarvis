"""Default-off bridge from the deterministic composer to the ONE fast analyzer.

Refresh from existing owner projections in background work, never from a frame
or analyzer completion callback. ``attach`` only reads the cached assignment and
a synchronous, scope-bound eligibility predicate; it performs no retrieval.

VisualAnalysisPipeline's optional live_brief_bridge_for lookup attaches before
ObservationPublication.prepare_inputs. Canonical capture records these fields,
locks and links all binding roots, then seals the final prepared request. The
shared prompt builder refuses uncaptured assignments before any provider call.

The caller still supplies the canonical owner-aware binder, background refresh,
and per-session bridge lifecycle. No unsupported source authority or unsafe
analyzer wrapper is supplied here. No resolver/disabled bridge preserves legacy
behavior; callers must not enable a bridge without an actual canonical binder.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
import json
from uuid import UUID

from app.live_state.models import GuardedPayloadBinding, LiveSnapshot, SessionScope, SourceDependency
from app.live_state.perception import PerceptionBrief
from app.live_state.perception_context import PerceptionComposition, compose_perception_brief
from app.perception.contracts import FastPerceptionRequest
from app.perception.applicability import PerceptionEffectGuard


MAX_BRIEF_CHARACTERS = 12_000
BriefBinder = Callable[[PerceptionBrief], Awaitable[GuardedPayloadBinding]]


def _utcnow():
    return datetime.now(UTC)


def perception_brief_document(brief: PerceptionBrief) -> dict:
    """The exact body the trusted canonical binder must materialize, not a summary."""
    document = brief.model_dump(mode="json")
    if len(json.dumps(document, ensure_ascii=True, separators=(",", ":"))) > MAX_BRIEF_CHARACTERS:
        raise ValueError("perception_brief_character_budget")
    return document


def _eligible(brief, predicate):
    if not brief.dependencies:
        return True
    if predicate is None:
        return False
    try:
        return all(predicate(source) is True for source in brief.dependencies)
    except Exception:
        return False


class LivePerceptionBriefBridge:
    """One session/epoch cache, not a check owner, scheduler or analyzer.

    ``bind`` must bind perception_brief_document(brief) through the existing
    canonical owners with exact dependency versions and removable source roots.
    A model/client-created receipt or a bare union of roots is not a binding.
    Unsupported owner namespaces must fail closed, never become memory sources.

    Callers own background refresh tasks. Closing invalidates in-flight results;
    it does not close borrowed owners. No model output is accepted by this API.
    """

    def __init__(self, *, scope: SessionScope, scope_epoch: UUID, enabled=False,
                 clock: Callable[[], datetime] = _utcnow, result_guard=None,
                 allow_warming_minimal: bool = False):
        self.scope, self.scope_epoch = scope, scope_epoch
        self.enabled, self.clock = enabled, clock
        self.allow_warming_minimal = allow_warming_minimal
        self._result_guard = result_guard
        self.composition: PerceptionComposition | None = None
        self.last_omission: str | None = None
        self._binding = None
        self._predicate = None
        self._generation = 0
        self._closed = False

    async def refresh(self, snapshot: LiveSnapshot, *, bind: BriefBinder,
                      dependency_eligible: Callable[[SourceDependency], bool] | None = None,
                      **inputs) -> PerceptionComposition | None:
        if self._closed:
            raise RuntimeError("A closed brief bridge cannot restart its epoch")
        if not self.enabled:
            return None
        self._generation += 1
        generation = self._generation
        previous_binding = self._binding
        self._binding = None
        self._predicate = dependency_eligible
        if snapshot.scope != self.scope:
            raise ValueError("perception_brief_scope_mismatch")
        warming_minimal = snapshot.readiness == "warming" and self.allow_warming_minimal
        if snapshot.readiness != "ready" and not warming_minimal:
            self.last_omission = "live_state_not_ready"
            return None
        if warming_minimal:
            # A slow memory producer must not prevent ordinary scene analysis.
            # Do not admit partial enrichment before it is ready. The minimal
            # brief still passes through the canonical scope and payload binder.
            inputs = {}
            dependency_eligible = None
            self._predicate = None
        previous = self.composition
        composition = compose_perception_brief(snapshot, now=self.clock(),
            scope_epoch=self.scope_epoch, previous=previous,
            dependency_eligible=dependency_eligible, **inputs)
        self.composition = composition
        brief = composition.brief
        perception_brief_document(brief)
        # An unchanged assignment remains usable while its exact existing roots
        # are revalidated. Do not create a generic-analysis window on every poll.
        # Changed semantics, expiry and ineligible dependencies still withdraw
        # immediately; capture/effect consumers retain their canonical guards.
        if (composition.reused and previous_binding is not None
                and previous_binding.scope == self.scope
                and set(previous_binding.dependencies) == set(brief.dependencies)
                and brief.created_at <= self.clock() < brief.expires_at
                and _eligible(brief, dependency_eligible)):
            self._binding = previous_binding
        # No per-frame bind. The callback is canonical background work only.
        try:
            binding = await bind(brief)
        except BaseException:
            if generation == self._generation:
                self._binding = None
            raise
        if self._closed or generation != self._generation:
            return composition
        self._binding = None
        if binding.scope != self.scope or set(binding.dependencies) != set(brief.dependencies):
            raise ValueError("perception_brief_binding_mismatch")
        if self.clock() >= brief.expires_at or not _eligible(brief, dependency_eligible):
            self.last_omission = "assignment_expired_or_dependency_changed"
            return composition
        self._binding = binding
        self.last_omission = None
        return composition

    def attach(self, request: FastPerceptionRequest) -> FastPerceptionRequest:
        """Call before canonical input capture; never call the composer or a model."""
        if not self.enabled or self._closed:
            return request
        if (request.owner_id, request.session_id) != (self.scope.owner_id, self.scope.session_id):
            raise ValueError("perception_brief_request_scope_mismatch")
        if request.perception_brief is not None:
            raise ValueError("perception_brief_already_attached")
        brief = self.composition.brief if self.composition is not None else None
        if brief is None or self._binding is None:
            self.last_omission = "assignment_unavailable"
            return request
        if not brief.created_at <= self.clock() < brief.expires_at:
            self.last_omission = "assignment_expired_or_future"
            return request
        if not _eligible(brief, self._predicate):
            self.last_omission = "assignment_dependency_changed"
            return request
        # Never bring post-frame context backward into the image interpretation.
        if (brief.created_at > request.primary_frame.captured_at
                or request.requested_at < request.primary_frame.captured_at
                or request.requested_at >= brief.expires_at):
            self.last_omission = "assignment_frame_time_mismatch"
            return request
        self.last_omission = None
        return replace(request, perception_brief=brief, perception_brief_binding=self._binding)

    def effect_guard_for(self, request, observation_payload_id):
        if (request.perception_brief is None or request.perception_brief_binding is None
                or (request.owner_id, request.session_id) != (self.scope.owner_id, self.scope.session_id)):
            raise ValueError("perception_effect_scope_mismatch")
        return PerceptionEffectGuard(request.perception_brief, request.perception_brief_binding,
            self._result_guard, lambda: self.enabled and not self._closed, observation_payload_id)

    def close(self):
        self._closed = True
        self._generation += 1
        self._binding = None
        self._predicate = None


def live_brief_input_fields(request: FastPerceptionRequest) -> dict:
    """Merge into canonical request_document; omit entirely on the legacy path."""
    if request.perception_brief is None:
        return {}
    # Request construction already enforces paired binding/scope/dependencies.
    return {
        "perception_brief": perception_brief_document(request.perception_brief),
        "perception_brief_binding": request.perception_brief_binding.model_dump(mode="json"),
    }


def live_brief_parent_payload_ids(request: FastPerceptionRequest) -> tuple[UUID, ...]:
    """Capture must lock and link ALL these parents, never just record their IDs."""
    return request.perception_brief_binding.payload_root_ids if request.perception_brief_binding else ()


def _input_digest(request):
    # Local imports also permit observation_inputs to import the two field/root
    # hooks without an initialization cycle. Both functions are deterministic.
    from app.knowledge.observation_inputs import request_document
    from app.tasks.models import digest
    return digest(request_document(request))


def seal_live_brief_input(request: FastPerceptionRequest, snapshot) -> FastPerceptionRequest:
    """After canonical capture, retain its exact fingerprint on the analyzer request.

    This checks contents, not DB authorization. Existing input validation must
    still lock/check the captured payload and its bound parents before analysis
    and again at observation publication. Never synthesize a snapshot to bypass
    those checks. The receipt is not sent to or accepted from the model.
    """
    if request.perception_brief is None:
        return request
    from app.knowledge.observation_inputs import ObservationInputSnapshot
    if (not isinstance(snapshot, ObservationInputSnapshot)
            or (snapshot.owner_id, snapshot.session_id) != (request.owner_id, request.session_id)
            or snapshot.request_digest != _input_digest(request)):
        raise ValueError("perception_brief_not_in_canonical_input")
    return replace(request, perception_brief_input_digest=snapshot.request_digest)


def render_live_perception_brief(request: FastPerceptionRequest) -> dict | None:
    """Called by the real prompt builder before the single provider invocation."""
    brief = request.perception_brief
    if brief is None:
        return None
    now = _utcnow()
    if (not brief.created_at <= request.requested_at <= now < brief.expires_at
            or brief.created_at > request.primary_frame.captured_at
            or any(frame.captured_at > request.requested_at for frame in request.frames)):
        raise ValueError("perception_brief_expired_or_frame_time_mismatch")
    if (request.perception_brief_input_digest is None
            or request.perception_brief_input_digest != _input_digest(request)):
        raise ValueError("perception_brief_not_in_canonical_input")
    # Keep exact versions/clocks/uncertainty; don't expose owner identity or a
    # binding as an instruction. Existing short frame/watch handles stay intact.
    return {key: value for key, value in perception_brief_document(brief).items() if key != "scope"}
