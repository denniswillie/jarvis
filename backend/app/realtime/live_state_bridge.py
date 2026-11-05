"""Opt-in application composition, with background-only context preparation.

The existing audio session owns provider/media/control transport. This attachment
owns only Live State and two bounded background consumers. It does not create a
second task authority, notification sink, voice model, or memory database.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
import json
import logging
from time import monotonic
from uuid import UUID, uuid4

from app.errors import JarvisHTTPError
from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.live_state.models import SessionScope
from app.live_state.projection import VoiceSelection, prepare_voice_context, select_voice_context
from app.live_state.refresh import REFRESH_SOURCE_ERROR_CODES
from app.live_state.runtime import create_session_live_state
from app.perception.evidence_session import create_session_perception_evidence
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.bridge import ContextBridgeStatus, OfferDisposition, utc_now
from app.voice_routes.contracts import NotificationOffer, RouteIdentity, RouteStart, VoiceContextUpdate


_VALIDITY_LOGGER = logging.getLogger(__name__)
_VALIDITY_LOG_LIMIT = 32
_MAX_UNWRITTEN_CONTEXT_RETRIES = 2
_OWNER_BUSY_RETRY_INTERVAL_SECONDS = 1.0
_VALIDITY_CODES = frozenset({
    "canonical_owner_busy",
    "voice_context_not_admitted", "voice_context_expired",
    "live_state_runtime_unavailable", "live_state_scope_unavailable",
    "live_state_scope_changed", "live_state_session_unavailable",
    "live_state_context_revoked", "live_state_context_expired",
    "live_state_source_scope", "live_state_source_version", "live_state_source_unsupported",
    "live_state_source_changed", "live_state_event_changed", "live_state_event_scope",
    "live_state_memory_scope", "live_state_canonical_owners_changed",
    "live_state_canonical_owners_unavailable", "live_state_reconstruction_required",
    "knowledge_area_unavailable", "knowledge_source_not_found", "knowledge_source_unavailable",
    "knowledge_source_changed", "knowledge_session_unavailable", "knowledge_payload_unavailable",
    "knowledge_source_deleted", "knowledge_source_forgotten", "knowledge_source_revision_conflict",
    "live_task_scope_changed", "live_task_context_unavailable", "live_task_request_changed",
    "live_task_time_unavailable", "live_task_context_oversized", "live_task_source_version",
    "live_task_snapshot_unavailable", "live_task_snapshot_changed", "live_task_context_expired",
    "live_episode_owner_unavailable", "live_episode_reference_invalid",
    "live_episode_source_changed", "live_episode_projection_unavailable",
    "live_episode_context_expired", "live_episode_cue_inapplicable",
})


class _ContextValidityCheck:
    """Preserve guard decisions while recording bounded, content-free rejection reasons."""

    def __init__(self, guard, live, clock):
        self.guard, self.live, self.clock = guard, live, clock
        self._reported = 0

    async def __call__(self, update):
        from app.live_state.task_context import TaskSnapshotControlChange, TaskSnapshotProgress
        try:
            async with self.guard(update):
                return True
        except (TaskSnapshotProgress, TaskSnapshotControlChange, CanonicalOwnerBusy) as error:
            self._rejected(update, error)
            # Preserve typed no-authority progress for the publication owner.
            # It must acquire a NEW projection, never authorize this snapshot.
            raise
        except (SourceError, TimeoutError) as error:
            self._rejected(update, error)
            return False

    def _rejected(self, update, error):
        try:
            if self._reported > _VALIDITY_LOG_LIMIT:
                return
            if self._reported == _VALIDITY_LOG_LIMIT:
                self._reported += 1
                record = {"stage": "truncated", "record_limit": _VALIDITY_LOG_LIMIT,
                          "absence_after_limit_is_inconclusive": True}
            else:
                self._reported += 1
                code = getattr(error, "code", None)
                worker = getattr(self.live, "worker", None)
                refresh_error = getattr(worker, "last_error", None)
                refresh_type = (None if refresh_error is None else
                    refresh_error if type(refresh_error) is str and refresh_error in {"TimeoutError", "SourceError"}
                    else "unrecognized_error")
                failed_stage = getattr(worker, "last_error_stage", None)
                refresh_stage = (None if failed_stage is None else
                    failed_stage if type(failed_stage) is str and failed_stage in {"read", "apply"}
                    else "unrecognized_stage")
                failed_code = getattr(worker, "last_error_code", None)
                refresh_code = (None if failed_code is None or refresh_type != "SourceError" else
                    failed_code if type(failed_code) is str and failed_code in REFRESH_SOURCE_ERROR_CODES
                    else "unrecognized_source_error")
                healthy = getattr(self.live, "_healthy", None)
                reconstruction = self.live.requires_reconstruction
                record = {
                    "stage": "validity_rejected", "route_id": str(update.route.route_id),
                    "route_generation": update.route.route_generation,
                    "update_id": str(update.update_id), "cursor_revision": update.cursor.revision,
                    "error_type": "TimeoutError" if isinstance(error, TimeoutError) else "SourceError",
                    "code": ("guard_timeout" if isinstance(error, TimeoutError) else
                             code if isinstance(code, str) and code in _VALIDITY_CODES
                             else "unrecognized_source_error"),
                    "guard_kind": "scope" if update.guard_binding is None else "payload_roots",
                    "expired_at_report": update.expires_at is not None and update.expires_at <= self.clock(),
                    "live_healthy": healthy if type(healthy) is bool else None,
                    "requires_reconstruction": reconstruction if type(reconstruction) is bool else None,
                    "refresh_error_type": refresh_type,
                    "refresh_error_stage": refresh_stage,
                    "refresh_error_code": refresh_code,
                }
            _VALIDITY_LOGGER.warning("voice_context_guard_diagnostic %s",
                                    json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Reporting must not change a rejection or expose raw exceptions.
            pass


def _selection_key(selection):
    # Cursor/expiry renewal alone must not materialize the same text repeatedly.
    return (selection.text, selection.dependencies, selection.parent_payload_root_ids,
            selection.selected_entry_ids, selection.observation_history)


_PREPARATION_LOGGER = logging.getLogger("app.realtime.live_state_bridge.preparation")
_PREPARATION_LOG_LIMIT = 64
_BINDING_LOGGER = logging.getLogger("app.realtime.live_state_bridge.binding")
_BINDING_SAMPLE_LIMIT = 32
_PREPARATION_CODES = frozenset({
    "binding_timeout", "binding_cancelled", "binding_unexpected_error",
    "preparation_expired", "attachment_closing", "reconstruction_required",
    "selection_changed_during_binding", "selection_expired_during_binding",
    "route_not_ready", "stale_cursor", "scope_mismatch", "route_identity_mismatch",
    "publication_deferred_active", "prepared_context_expired",
})


@dataclass(frozen=True, slots=True)
class _PreparedContext:
    selection: VoiceSelection
    key: tuple
    update: VoiceContextUpdate
    started: float
    binding_sequence: int | None = None
    binding_mode: str | None = None


@dataclass(slots=True)
class _BindingObservation:
    sequence: int
    selection: VoiceSelection
    started: float
    mode: str
    first_obsolete_monotonic_s: float | None = None
    first_obsolete_revision: int | None = None


class SessionLiveStateBridge:
    def __init__(self, session, live, route, *, clock=utc_now, native_sink=None, visual_pipeline=None,
                 selection_max_encoded_bytes: int | None = None,
                 preserve_observation_history: bool = False,
                 initial_selection: VoiceSelection | None = None):
        if type(preserve_observation_history) is not bool:
            raise ValueError("Observation history selection must be explicitly enabled.")
        self.session, self.live, self.route, self.clock = session, live, route, clock
        self._selection_max_encoded_bytes = selection_max_encoded_bytes
        self._preserve_observation_history = preserve_observation_history
        self.native_sink = native_sink
        self._visual_pipeline = visual_pipeline
        self._registered_perception_bridge = None
        self._perception_evidence = create_session_perception_evidence(live, visual_pipeline, clock=clock)
        self._registered_perception_evidence = None
        self.perception_unavailable_reason = (
            "canonical_visual_capture_unavailable" if visual_pipeline is None else None)
        self._wake = asyncio.Event()
        self._subscription = None
        self._tasks = ()
        self._closing = False
        self._close_task = None
        self._last_update = None
        self._last_key = None
        self._last_written_update_id = None
        self._unwritten_retry_count = 0
        self._owner_busy_retry_update_id = None
        self._owner_busy_retry_not_before = 0.0
        self.failure = None
        self.last_omission = None
        self._preparation_diagnostics = 0
        self._route_identity = route.request.route
        self._initial_selection = initial_selection
        if initial_selection is not None:
            initial = route.request.initial_context
            expiry = (initial_selection.hard_expires_at if initial_selection.observation_history
                      else initial_selection.expires_at)
            if (initial is None or initial.route != self._route_identity
                    or initial_selection.scope != self._route_identity.scope
                    or initial_selection.cursor != initial.cursor
                    or initial_selection.text != initial.text
                    or initial_selection.dependencies != initial.dependencies
                    or initial_selection.observation_history != initial.observation_history
                    or expiry != initial.expires_at):
                raise ValueError("Initial selection must be the exact prepared startup context.")
        self._prepared_context: _PreparedContext | None = None
        self._preparing = False
        self._binding_samples = 0
        self._binding_observation: _BindingObservation | None = None

    def registered(self):
        """Activate borrowed perception only after the audio manager admits us."""
        if self._closing:
            raise RuntimeError("A closing attachment cannot register perception")
        if self._visual_pipeline is None or self._registered_perception_bridge is not None:
            return
        bridge = self.live.perception_brief_bridge
        if bridge.scope != self.route.request.route.scope:
            raise RuntimeError("Perception and voice session scopes do not match")
        self._visual_pipeline.register_live_brief(bridge)
        self._registered_perception_bridge = bridge
        if self._perception_evidence is not None:
            self._visual_pipeline.register_perception_evidence_session(self._perception_evidence)
            self._registered_perception_evidence = self._perception_evidence
            self._perception_evidence.activate()

    def start(self):
        if self._tasks or self._closing:
            raise RuntimeError("A Live State voice attachment can start only once.")
        self._subscription = self.live.service.subscribe(capacity=8)
        self._tasks = (asyncio.create_task(self._listen(), name="voice-live-state-events"),
                       asyncio.create_task(self._publish(), name="voice-live-state-projection"))

    async def _fail(self, reason):
        self.failure = reason
        # The existing session closes provider transport and fences client output.
        # Do not call our own close here: the router joins these tasks afterward.
        await self.session.close(reason)

    async def _listen(self):
        try:
            while not self._closing:
                notice = await self._subscription.receive()
                if notice is None:
                    if not self._closing:
                        await self._fail("live_state_subscription_closed")
                    return
                if notice.resync_required or self.live.requires_reconstruction:
                    await self._fail("live_state_reconstruction_required")
                    return
                # No binding, provider call or database work in this consumer.
                # The publisher always reads the newest immutable snapshot.
                self._observe_binding_change("notice")
                self._wake.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail("live_state_subscription_failed")

    async def _publish(self):
        try:
            while not self._closing and not self.session._closed.is_set():
                try:
                    async with asyncio.timeout(0.1):
                        await self._wake.wait()
                except TimeoutError:
                    pass
                self._wake.clear()
                if self.live.requires_reconstruction:
                    await self._fail("live_state_reconstruction_required")
                    return
                await self.publish_latest()
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._fail("live_state_publication_failed")

    async def publish_latest(self):
        """Single background writer; preparation never runs in a voice handler."""
        # The normal loop is already serialized. Explicit callers must not add
        # a second binding job or an unbounded queue behind the owned worker.
        if self._preparing:
            self._wake.set()
            return
        self._preparing = True
        try:
            await self._prepare_latest()
        except asyncio.CancelledError:
            self._prepared_context = None
            raise
        finally:
            self._preparing = False

    def _publication_idle(self):
        return getattr(self.route, "context_publication_idle",
                       getattr(self.route, "context_publication_ready", self.route.ready))

    def _preparation_allowed(self):
        # Only runtimes explicitly exposing the separate seam opt in. Live and
        # older route implementations retain their existing idle-only behavior.
        return getattr(self.route, "context_preparation_ready", self._publication_idle())

    def _candidate_matches(self, candidate, selection):
        now = self.clock()
        expiry = candidate.update.expires_at
        return (candidate.update.route == self._route_identity == self.route.request.route
                and candidate.selection.scope == selection.scope == self._route_identity.scope
                and candidate.selection.cursor.epoch == selection.cursor.epoch
                and candidate.selection.cursor.revision <= selection.cursor.revision
                and candidate.key == _selection_key(selection)
                and (expiry is None or expiry > now)
                and (selection.expires_at is None or (
                    selection.expires_at > now and expiry is not None
                    and expiry <= selection.expires_at)))

    async def _prepare_latest(self):
        if (self._closing or not self._preparation_allowed()
                or self.live.requires_reconstruction):
            self._prepared_context = None
            return
        selection = select_voice_context(self.live.service.snapshot(), now=self.clock(),
            max_encoded_bytes=self._selection_max_encoded_bytes,
            preserve_observation_history=self._preserve_observation_history)
        if (self.route.request.route != self._route_identity
                or selection.scope != self._route_identity.scope):
            self._prepared_context = None
            self.last_omission = "route_identity_mismatch"
            return
        key = _selection_key(selection)
        previous = self._last_update
        publication = getattr(self.route, "context_publication_status", None)
        if (previous is None and self._initial_selection is not None
                and getattr(self.route, "context_publication_ready", False) is True
                and isinstance(publication, ContextBridgeStatus)
                and publication.identity == self._route_identity
                and publication.running and not publication.fenced):
            initial = self.route.request.initial_context
            original = self._initial_selection
            self._initial_selection = None
            candidate = _PreparedContext(original, _selection_key(original), initial, 0.0)
            if (publication.newest_cursor == initial.cursor
                    and publication.last_receipt is None
                    and self._candidate_matches(candidate, selection)):
                # ACTIVE follows the coordinator's confirmed start, which already
                # installed this exact context. Preserve that baseline, not a
                # fabricated publication receipt or provider comprehension ACK.
                self._last_update, self._last_key = initial, candidate.key
                previous = initial
        deferred = False
        if (previous is not None and isinstance(publication, ContextBridgeStatus)
                and publication.identity == previous.route):
            receipt = publication.last_receipt
            if (receipt is not None and receipt.route == previous.route
                    and receipt.update_id == previous.update_id and receipt.cursor == previous.cursor
                    and receipt.status in {"written", "provider_acknowledged"}
                    and self._last_written_update_id != receipt.update_id):
                # A local written receipt is not a provider ACK. It does settle
                # the offered-versus-written distinction for this exact update.
                self._last_written_update_id = receipt.update_id
                self._unwritten_retry_count = 0
            deferred = not publication.fenced and publication.retry_update_id == previous.update_id
        candidate = self._prepared_context
        if candidate is not None:
            if self._candidate_matches(candidate, selection):
                self._offer_prepared(candidate, selection, deferred=deferred)
                return
            self._prepared_context = None
            self.last_omission = "prepared_context_expired" if (
                candidate.update.expires_at is not None
                and candidate.update.expires_at <= self.clock()) else "selection_changed_during_binding"
        if deferred and previous.expires_at is not None and previous.expires_at <= self.clock():
            self.last_omission = "deferred_context_expired"
            if selection.cursor.revision <= previous.cursor.revision:
                return
        if previous is not None and key == self._last_key:
            if (isinstance(publication, ContextBridgeStatus)
                    and publication.refill_update_id == previous.update_id
                    and selection.cursor == previous.cursor
                    and self._candidate_matches(_PreparedContext(selection, key, previous, 0.0), selection)
                    and self._publication_idle()):
                # Same desired evidence, not a refreshed timestamp or revision.
                # Exact ACK-empty ownership permits a NEW guarded create only.
                result = self.route.offer_context(previous)
                self.last_omission = ("acknowledged_empty_refill_queued" if result.disposition in {
                    OfferDisposition.QUEUED, OfferDisposition.COALESCED} else result.reason)
                return
            if deferred:
                if (publication.reason == "context_owner_busy_awaiting_snapshot"
                        and (previous.expires_at is None or previous.expires_at > self.clock())):
                    # A try-lock miss is not a provider-write failure. Exhausting
                    # the write-retry budget here can strand unchanged context
                    # forever after a short transaction. Reconsider at a bounded
                    # cadence on the existing publisher, without another worker,
                    # queue, or a response-path wait. The exact no-write receipt
                    # and every canonical/provider guard are still required.
                    now = monotonic()
                    if self._owner_busy_retry_update_id != previous.update_id:
                        self._owner_busy_retry_update_id = previous.update_id
                        self._owner_busy_retry_not_before = now + _OWNER_BUSY_RETRY_INTERVAL_SECONDS
                    if now < self._owner_busy_retry_not_before or not self._publication_idle():
                        self.last_omission = "owner_busy_retry_wait"
                        return
                    self._owner_busy_retry_not_before = now + _OWNER_BUSY_RETRY_INTERVAL_SECONDS
                    result = self.route.offer_context(previous)
                    self.last_omission = ("owner_busy_retry_queued" if result.disposition in {
                        OfferDisposition.QUEUED, OfferDisposition.COALESCED} else result.reason)
                    return
                if self._unwritten_retry_count >= _MAX_UNWRITTEN_CONTEXT_RETRIES:
                    self.last_omission = "safe_context_retry_exhausted"
                    return
                if previous.expires_at is None or previous.expires_at > self.clock():
                    if not self._publication_idle():
                        return
                    # The existing 100ms publisher tick supplies progress even
                    # when no new source notice or semantic content arrives.
                    # Only the bridge's exact joined/no-write receipt permits
                    # this reoffer; the bridge and writer recheck eligibility.
                    result = self.route.offer_context(previous)
                    if result.disposition in {OfferDisposition.QUEUED, OfferDisposition.COALESCED}:
                        self._unwritten_retry_count += 1
                        self.last_omission = "safe_context_retry_queued"
                    elif result.disposition != OfferDisposition.DUPLICATE:
                        self.last_omission = result.reason
                    return
            fresh_snapshot_required = (
                isinstance(publication, ContextBridgeStatus)
                and publication.identity == previous.route
                and publication.running and not publication.fenced
                and getattr(self.route, "context_refresh_pending", False) is True)
            # Offered content is not necessarily installed content. In particular,
            # a confirmed delete followed by rejected creation leaves an empty
            # slot. A newer canonical snapshot must reach the existing guarded
            # preparation path even when its selected text is unchanged.
            if (not fresh_snapshot_required and (previous.expires_at is None
                    or previous.expires_at > self.clock() + timedelta(seconds=2))):
                return
        if previous is not None and selection.cursor.revision <= previous.cursor.revision:
            return

        preparation_started = asyncio.get_running_loop().time()
        bound_binding = None
        binding_observation = None
        self._report_preparation(selection, stage="binding_entry", outcome="started",
                                 started=preparation_started)

        async def bind(current):
            nonlocal bound_binding, binding_observation
            reuse = previous is not None and key == self._last_key and previous.guard_binding is not None
            binding_observation = self._begin_binding_observation(current, reuse=reuse)
            outcome = "failed"
            try:
                if reuse:
                    # Reuse only the exact committed copy. Coordinator validity and
                    # the actual writer still revalidate its original source roots.
                    bound_binding = previous.guard_binding
                else:
                    bound_binding = await self.live.bind(current)
                outcome = "returned"
                return bound_binding
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                self._finish_binding_observation(binding_observation, outcome)

        try:
            update = await prepare_voice_context(selection, self._route_identity,
                bind=bind, clock=self.clock)
        except (SourceError, TimeoutError) as error:
            self.last_omission = getattr(error, "code", "binding_timeout")
            self._report_preparation(selection, stage="binding_exit", outcome="rejected",
                started=preparation_started, code=self.last_omission, binding=bound_binding)
            return
        except asyncio.CancelledError:
            self._report_preparation(selection, stage="binding_exit", outcome="cancelled",
                started=preparation_started, code="binding_cancelled", binding=bound_binding)
            raise
        except Exception:
            self._report_preparation(selection, stage="binding_exit", outcome="failed",
                started=preparation_started, code="binding_unexpected_error", binding=bound_binding)
            raise
        if update is None:
            self._report_preparation(selection, stage="preparation_exit", outcome="omitted",
                started=preparation_started, code="preparation_expired", binding=bound_binding)
            return
        if self._closing or self.live.requires_reconstruction or not self._preparation_allowed():
            self._report_preparation(selection, stage="preparation_exit", outcome="omitted",
                started=preparation_started, update=update, binding=bound_binding,
                code=("attachment_closing" if self._closing else "reconstruction_required"
                      if self.live.requires_reconstruction else "route_not_ready"))
            return
        latest = select_voice_context(self.live.service.snapshot(), now=self.clock(),
            max_encoded_bytes=self._selection_max_encoded_bytes,
            preserve_observation_history=self._preserve_observation_history)
        candidate = _PreparedContext(selection, key, update, preparation_started,
            binding_sequence=binding_observation.sequence if binding_observation is not None else None,
            binding_mode=binding_observation.mode if binding_observation is not None else None)
        if (key != _selection_key(latest) or selection.scope != latest.scope
                or selection.cursor.epoch != latest.cursor.epoch
                or self.route.request.route != self._route_identity):
            self.last_omission = "selection_changed_during_binding"
            self._report_preparation(selection, stage="preparation_exit", outcome="omitted",
                started=preparation_started, code=self.last_omission, update=update,
                binding=bound_binding, latest=latest)
            self._wake.set()
            return
        if not self._candidate_matches(candidate, latest):
            self.last_omission = "selection_expired_during_binding"
            self._report_preparation(selection, stage="preparation_exit", outcome="omitted",
                started=preparation_started, code=self.last_omission, update=update,
                binding=bound_binding, latest=latest)
            return
        # bind() has returned and released its canonical transaction. Retain
        # only immutable data/identities, never a connection or an entered guard.
        self._prepared_context = candidate
        if not self._publication_idle():
            self.last_omission = None
            self._report_preparation(selection, stage="preparation_exit", outcome="prepared",
                started=preparation_started, update=update, binding=bound_binding,
                latest=latest, code="publication_deferred_active")
            return
        self._offer_prepared(candidate, latest, deferred=deferred)

    def _offer_prepared(self, candidate, latest, *, deferred):
        if not self._publication_idle():
            return
        if not self._candidate_matches(candidate, latest):
            self._prepared_context = None
            self._wake.set()
            return
        selection, key, update = candidate.selection, candidate.key, candidate.update
        self._prepared_context = None
        # No await separates the latest scoped-selection/idle checks from the
        # existing coordinator offer. Its preflight and final source guard still
        # authorize every actual provider mutation, including cached bindings.
        result = self.route.offer_context(update)
        if candidate.binding_sequence is not None:
            self._record_binding_observation(selection, sequence=candidate.binding_sequence,
                mode=candidate.binding_mode, stage="offer_return", update=update)
        self._report_preparation(selection, stage="offer_return", outcome=(
            result.disposition.value if isinstance(result.disposition, OfferDisposition) else "unrecognized"),
            started=candidate.started, code=result.reason, update=update,
            binding=update.guard_binding, latest=latest)
        if result.disposition in {OfferDisposition.QUEUED, OfferDisposition.COALESCED,
                                   OfferDisposition.DUPLICATE}:
            if key != self._last_key:
                self._unwritten_retry_count = 0
            elif deferred and result.disposition != OfferDisposition.DUPLICATE:
                self._unwritten_retry_count += 1
            self._last_update, self._last_key = update, key
            self.last_omission = None
        else:
            self.last_omission = result.reason

    def _begin_binding_observation(self, selection, *, reuse):
        """Sample existing binding work only; a binding return is not a write/ACK."""
        try:
            if self._binding_samples >= _BINDING_SAMPLE_LIMIT:
                if self._binding_samples == _BINDING_SAMPLE_LIMIT:
                    self._binding_samples += 1
                    _BINDING_LOGGER.warning("voice_context_binding_diagnostic %s", json.dumps({
                        "format": "jarvis.voice_context_binding_diagnostic.v1",
                        "stage": "truncated", "binding_sample_limit": _BINDING_SAMPLE_LIMIT,
                        "absence_after_limit_is_inconclusive": True,
                    }, sort_keys=True, separators=(",", ":")))
                return None
            self._binding_samples += 1
            observation = _BindingObservation(self._binding_samples, selection,
                asyncio.get_running_loop().time(),
                "exact_committed_binding_reuse" if reuse else "new_binding")
            self._binding_observation = observation
            self._record_binding_observation(selection, sequence=observation.sequence,
                mode=observation.mode, stage="binding_entry", started=observation.started)
            return observation
        except Exception:
            return None

    def _observe_binding_change(self, boundary):
        """First observed key change, not the source's unobserved mutation time.

        Called only by existing notices and binding completion. Selection reads
        the in-memory snapshot; it never binds, polls, cancels or authorizes work.
        """
        try:
            observation = self._binding_observation
            if observation is None or observation.first_obsolete_monotonic_s is not None:
                return
            original = observation.selection
            latest = select_voice_context(self.live.service.snapshot(), now=self.clock(),
                max_encoded_bytes=self._selection_max_encoded_bytes,
                preserve_observation_history=self._preserve_observation_history)
            if (self._closing or self.route.request.route != self._route_identity
                    or original.scope != latest.scope or original.cursor.epoch != latest.cursor.epoch
                    or _selection_key(original) == _selection_key(latest)):
                return
            observation.first_obsolete_monotonic_s = asyncio.get_running_loop().time()
            observation.first_obsolete_revision = latest.cursor.revision
            changes = dict(text=original.text != latest.text,
                dependencies=original.dependencies != latest.dependencies,
                parent_roots=original.parent_payload_root_ids != latest.parent_payload_root_ids,
                entry_ids=original.selected_entry_ids != latest.selected_entry_ids,
                observation_history=original.observation_history != latest.observation_history)
            self._record_binding_observation(original, sequence=observation.sequence,
                mode=observation.mode, stage="first_observed_obsolete", started=observation.started,
                first_obsolete=observation.first_obsolete_monotonic_s,
                first_obsolete_revision=observation.first_obsolete_revision,
                boundary=boundary, changes=changes)
        except Exception:
            # Missing diagnostic evidence never changes business eligibility.
            pass

    def _finish_binding_observation(self, observation, outcome):
        if observation is None:
            return
        try:
            if self._binding_observation is observation:
                self._observe_binding_change("binding_completion")
            self._record_binding_observation(observation.selection, sequence=observation.sequence,
                mode=observation.mode, stage="binding_exit", started=observation.started,
                outcome=outcome, first_obsolete=observation.first_obsolete_monotonic_s,
                first_obsolete_revision=observation.first_obsolete_revision)
        except Exception:
            pass
        finally:
            if self._binding_observation is observation:
                self._binding_observation = None

    def _record_binding_observation(self, selection, *, sequence, mode, stage, started=None,
                                    outcome=None, first_obsolete=None, first_obsolete_revision=None,
                                    boundary=None, changes=None, update=None):
        try:
            now, monotonic = self.clock(), asyncio.get_running_loop().time()
            route = self._route_identity
            record = {
                "format": "jarvis.voice_context_binding_diagnostic.v1",
                "diagnostic_only": True, "timing_scope": "same_process_monotonic",
                "stage": stage, "binding_sequence": sequence, "binding_mode": mode,
                "monotonic_s": monotonic,
                "elapsed_ms": None if started is None else round((monotonic - started) * 1000, 3),
                "route_id": str(route.route_id), "route_generation": route.route_generation,
                "cursor_epoch": str(selection.cursor.epoch), "cursor_revision": selection.cursor.revision,
                "update_id": None if update is None else str(update.update_id),
                "selection_remaining_freshness_ms": (None if selection.expires_at is None else
                    round((selection.expires_at - now).total_seconds() * 1000, 3)),
                "update_remaining_freshness_ms": (None if update is None or update.expires_at is None else
                    round((update.expires_at - now).total_seconds() * 1000, 3)),
                "first_obsolete_monotonic_s": first_obsolete,
                "first_obsolete_cursor_revision": first_obsolete_revision,
                "observation_boundary": boundary, "key_changes": changes,
                "binding_outcome": outcome,
            }
            _BINDING_LOGGER.warning("voice_context_binding_diagnostic %s",
                                   json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            pass

    def _report_preparation(self, selection, *, stage, outcome, started, code=None,
                            update=None, binding=None, latest=None):
        """Bounded pre-offer evidence, NOT a write/ACK or source-availability receipt."""
        try:
            count = self._preparation_diagnostics
            if count > _PREPARATION_LOG_LIMIT:
                return
            self._preparation_diagnostics = count + 1
            if count == _PREPARATION_LOG_LIMIT:
                record = {"stage": "truncated", "record_limit": _PREPARATION_LOG_LIMIT,
                          "absence_after_limit_is_inconclusive": True}
            else:
                def identity(value):
                    return str(value) if isinstance(value, UUID) else None

                route = self.route.request.route
                record = {
                    "stage": stage, "outcome": outcome,
                    "code": (None if code is None else code
                             if isinstance(code, str) and code in _PREPARATION_CODES | _VALIDITY_CODES
                             else "unrecognized_reason"),
                    "route_id": identity(route.route_id), "route_generation": route.route_generation,
                    "cursor_epoch": identity(selection.cursor.epoch),
                    "cursor_revision": selection.cursor.revision,
                    "update_id": identity(update.update_id) if update is not None else None,
                    "elapsed_ms": round((asyncio.get_running_loop().time() - started) * 1000, 3),
                    "selected_entry_count": len(selection.selected_entry_ids),
                    "omitted_entry_count": len(selection.omitted_entry_ids),
                    "source_count": len(selection.dependencies),
                    "observation_window_count": len(selection.observation_history),
                    "selection_expires_at": (selection.expires_at.isoformat()
                                             if selection.expires_at is not None else None),
                    "hard_expires_at": (selection.hard_expires_at.isoformat()
                                        if selection.hard_expires_at is not None else None),
                    "parent_payload_root_ids": [identity(root) for root in selection.parent_payload_root_ids[:64]],
                    "binding_returned": binding is not None,
                    "bound_payload_root_ids": ([] if binding is None else
                                               [identity(root) for root in binding.payload_root_ids[:64]]),
                    "latest_cursor_epoch": identity(latest.cursor.epoch) if latest is not None else None,
                    "latest_cursor_revision": latest.cursor.revision if latest is not None else None,
                    "key_changes": (None if latest is None else dict(
                        text=selection.text != latest.text,
                        dependencies=selection.dependencies != latest.dependencies,
                        parent_roots=selection.parent_payload_root_ids != latest.parent_payload_root_ids,
                        entry_ids=selection.selected_entry_ids != latest.selected_entry_ids,
                        observation_history=selection.observation_history != latest.observation_history)),
                }
            _PREPARATION_LOGGER.warning("voice_context_preparation_diagnostic %s",
                                       json.dumps(record, sort_keys=True, separators=(",", ":")))
        except Exception:
            # Logging cannot alter a guard, cancellation, offer or original error.
            pass

    async def close(self, reason="live_state_attachment_closed"):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(reason))
        await asyncio.shield(self._close_task)

    async def _close(self, reason):
        self._closing = True
        self._prepared_context = None
        self._binding_observation = None
        if self._perception_evidence is not None:
            self._perception_evidence.close()
        if self._registered_perception_evidence is not None:
            self._visual_pipeline.unregister_perception_evidence_session(self._registered_perception_evidence)
            self._registered_perception_evidence = None
        if self._registered_perception_bridge is not None:
            self._visual_pipeline.unregister_live_brief(self._registered_perception_bridge)
            self._registered_perception_bridge = None
        if self._subscription is not None:
            self._subscription.close()
        for task in self._tasks:
            task.cancel()
        try:
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            await self.session.close(reason)
        finally:
            try:
                if self.native_sink is not None:
                    await self.native_sink.close()
            finally:
                try:
                    if self._perception_evidence is not None:
                        if join := getattr(self._perception_evidence, "aclose", None):
                            await join()
                finally:
                    await self.live.close()


async def install_responsive_voice(session, services, settings, client_capabilities, *, clock=utc_now):
    """Install before manager registration; return None for untouched legacy clients.

    Explicit client opt-in never silently falls back to legacy audio. Server
    activation is deliberately off until notification and P2 qualification pass.
    """
    if client_capabilities is None:
        return None
    if not getattr(settings, "RESPONSIVE_VOICE_ENABLED", False):
        raise JarvisHTTPError(503, "responsive_voice_disabled", "The requested voice route is not enabled.")
    if not isinstance(client_capabilities, dict):
        raise JarvisHTTPError(400, "invalid_voice_route_capabilities", "Voice-route capabilities must be an object.")
    knowledge = getattr(services, "knowledge_runtime", None)
    if knowledge is None:
        raise JarvisHTTPError(503, "live_state_unavailable", "The canonical knowledge runtime is unavailable.")
    context = session.context
    live = None
    native_sink = None
    try:
        async with asyncio.timeout(2):
            selection = await knowledge.area_selections.read(context.owner_id, context.jarvis_session_id)
        if selection["area_status"] != "active":
            raise ValueError("The selected knowledge area is not active.")
        scope = SessionScope(owner_id=context.owner_id, session_id=context.jarvis_session_id,
                             area_id=UUID(selection["area_id"]))
        visual_pipeline = getattr(getattr(services, "visual_memory_services", None), "pipeline", None)
        if getattr(visual_pipeline, "supports_perception_briefs", False) is not True:
            visual_pipeline = None
        live = create_session_live_state(services, scope=scope, enabled=True, clock=clock,
                                        perception_brief_enabled=visual_pipeline is not None)
        await live.start()
        route = RouteIdentity(scope=scope, route_id=uuid4(), route_generation=1)
        initial_selection = select_voice_context(live.service.snapshot(), now=clock())
        initial = await prepare_voice_context(
            initial_selection, route, bind=live.bind, clock=clock)
        request = RouteStart(route=route, command_id=uuid4(), requested_at=clock(),
            # Reference the already-authenticated durable session, not a new grant.
            authorization_ref=context.jarvis_session_id, initial_context=initial)

        @asynccontextmanager
        async def guard(payload):
            if isinstance(payload, NotificationOffer) and payload.route == route and native_sink is not None:
                async with native_sink.guard(payload) as connection:
                    # Canonical notification guard already owns source/task locks.
                    # Check the selected session scope without another transaction.
                    await live.binder._session(connection)
                    yield connection
                return
            if not isinstance(payload, VoiceContextUpdate) or payload.route != route:
                raise SourceError("voice_context_not_admitted", "No canonical notification sink is attached.")
            if payload.expires_at is not None and payload.expires_at <= clock():
                raise SourceError("voice_context_expired", "The selected context expired.")
            # Startup keeps its existing admission policy. Later quiet context
            # updates must not queue behind long canonical publications and
            # exhaust their provider-mutation budget before a write is possible.
            options = {} if payload == initial else {"wait_for_owner": False}
            manager = (live.scope_guard(**options) if payload.guard_binding is None
                       else live.guard(payload.guard_binding, **options))
            async with manager as connection:
                if payload.expires_at is not None and payload.expires_at <= clock():
                    raise SourceError("voice_context_expired", "The selected context expired.")
                yield connection

        validity = _ContextValidityCheck(guard, live, clock)

        async def authorize(candidate):
            if candidate != request or session._closed.is_set():
                return False
            async with live.scope_guard():
                return True

        async def authorize_replacement(candidate):
            if candidate != request or session._closed.is_set():
                return False
            async with live.scope_guard(wait_for_owner=False):
                return True

        def roots(payload):
            if isinstance(payload, NotificationOffer) and payload.route == route and native_sink is not None:
                return (payload.attempt.source_payload_id,)
            if not isinstance(payload, VoiceContextUpdate):
                raise SourceError("voice_context_not_admitted", "Only canonical context is installed.")
            return payload.guard_binding.payload_root_ids if payload.guard_binding is not None else ()

        def on_event(event):
            if native_sink is not None:
                native_sink.on_event(event)

        attention_admission = session.try_attention
        runtime = RealtimeVoiceRouteRuntime(session, request, client_capabilities,
            atomic_guard=guard, payload_roots=roots, validity=validity,
            authorize_start=authorize, authorize_context_replacement=authorize_replacement,
            normal_turn_scope_guard=live.scope_guard,
            on_event=on_event, clock=clock)
        session.install_voice_route_runtime(runtime)
        attention = getattr(services, "attention_service", None)
        if attention is not None and not getattr(attention, "_closed", False):
            from app.attention.native_voice import NativeAttentionSink
            native_sink = NativeAttentionSink(session, runtime, attention.repository,
                attention_admission, clock=clock)
            native_sink.start()
        attachment = SessionLiveStateBridge(session, live, runtime, clock=clock, native_sink=native_sink,
                                            visual_pipeline=visual_pipeline,
                                            initial_selection=initial_selection)
        attachment.start()
        return attachment
    except BaseException as error:
        try:
            await session.close("responsive_voice_installation_failed")
        finally:
            try:
                if native_sink is not None:
                    await native_sink.close()
            finally:
                if live is not None:
                    await live.close()
        if isinstance(error, asyncio.CancelledError):
            raise
        raise JarvisHTTPError(503, "responsive_voice_unavailable",
            "The requested voice route could not be initialized; no fallback speaker was started.") from error
