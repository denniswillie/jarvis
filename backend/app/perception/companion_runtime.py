"""Small composition adapter: canonical inputs -> analyzer -> existing owners.

Call submit_settled_caption only after application settlement. This module does
not run another transcript poller, speech classifier, visual worker or scheduler.
Memory submission is bounded and separate from perception. The existing intake
serializes its durable native-operation receipts with forget; acceptance is not
finished extraction. There is no direct NativeMemory write fallback.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import UTC, datetime
import json
import logging
from time import monotonic
from typing import Awaitable, Callable

from pydantic import ValidationError

from app.live_state.companion import CompanionState
from app.live_state.human import HumanState, HumanStateContent
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.perception.contracts import FastPerceptionFrame
from app.perception.multimodal import (
    FACT_FIELDS, EventBatch, EvidenceOrigin, MultimodalEventAnalyzer, MultimodalResult,
    PerceptionContext, SettledDialogue, SituationalReminder, TriggeredReminder,
)
from app.shared_memory.client import MemoryDocument, NativeMemory, source_document_id
from app.shared_memory.ingestion import CanonicalCaption, ObservationDelta
from app.tasks.context_events import ContextRecord
from app.visual_memory.models import VisualFrameRecord


ReminderCallback = Callable[[TriggeredReminder, tuple[str, ...], int | None], Awaitable[None]]
LOGGER = logging.getLogger(__name__)


def _human_sources(content: HumanStateContent) -> set[str]:
    return {source for name in FACT_FIELDS
            for fact in ((content.setting,) if name == "setting" else getattr(content, name))
            if fact is not None for source in fact.sources}


class CompanionPerceptionRuntime:
    def __init__(self, *, state: CompanionState, model, on_reminder: ReminderCallback,
                 memory: NativeMemory | None = None,
                 intake=None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC),
                 memory_queue_size: int = 8, memory_timeout_seconds: float = 15,
                 frame_repository=None, caption_settle_seconds: float = 0.8):
        if memory is not None and memory.scope.owner_id != state.owner_id:
            raise ValueError("perception and native memory must belong to the same owner")
        if intake is not None:
            if intake.memory.scope.owner_id != state.owner_id or (
                    memory is not None and intake.memory.scope != memory.scope):
                raise ValueError("visual retention must use the same owner/bank intake")
            memory = intake.memory
        if memory_queue_size < 1 or memory_timeout_seconds <= 0 or caption_settle_seconds <= 0:
            raise ValueError("positive memory queue and timeout bounds required")
        self.state, self.memory, self.on_reminder = state, memory, on_reminder
        self.intake = intake
        self.clock, self.memory_timeout_seconds = clock, memory_timeout_seconds
        self.frame_repository = frame_repository
        self.analyzer = MultimodalEventAnalyzer(owner_id=state.owner_id, model=model,
            read_context=self.read_context, publish=self.publish, clock=clock,
            on_batch_finished=self._finish_frames)
        self.caption_settle_seconds = caption_settle_seconds
        self._captions = OrderedDict()
        self._caption_revisions = OrderedDict()
        self._caption_wake = asyncio.Event()
        self._caption_task = None
        self._closed = False
        self._memory_queue: asyncio.Queue = asyncio.Queue(maxsize=memory_queue_size)
        self._memory_task = None
        self.last_context_issue = None
        self.last_publication_issue = None
        self.last_reminder_error = None
        self.last_memory_error = "serialized_memory_writer_not_configured" if memory is not None and intake is None else None
        self.last_memory_status = None
        self.last_memory_document_id = None
        self.memory_submissions = 0

    async def read_current_view(self, *, frame_window, owner_id, session_id):
        """Wait for existing ingestion/analysis, never start or cancel model work."""
        requested_at = self.clock()

        def unavailable(reason):
            return {"status": "freshness_unavailable", "reason": reason,
                "requested_at": requested_at.isoformat(), "text":
                "A fresh current view could not be obtained. Do not substitute cached observations."}

        if owner_id != self.state.owner_id:
            raise ValueError("Current view owner mismatch")
        snapshot = self.state.read()
        if self._closed or not snapshot["ready"]:
            return unavailable("perception_unavailable")
        generation = snapshot["human_generation"]
        lease = frame_window.stream_lease(owner_id, session_id)
        target = frame_window.latest(owner_id, session_id, max_age_seconds=2)
        if (lease is None or target is None or not target.image_bytes
                or target.visual_session_id != lease.visual_session_id
                or not 0 <= (requested_at - target.captured_at).total_seconds() <= 2):
            return unavailable("no_recent_frame")
        result = self.analyzer.latest_visual_result
        try:
            async with asyncio.timeout(6):
                while True:
                    if (self._closed or not frame_window.owns_stream(lease)
                            or self.state.read()["human_generation"] != generation):
                        return unavailable("source_changed")
                    if result is not None and result.human_generation == generation:
                        origins = {origin.evidence_id for origin in result.origins
                                   if origin.session_id == str(session_id)}
                        # The requested frame is already held by this bounded
                        # lookup. Ring-buffer eviction while its analysis runs
                        # does not retract it; stream/generation checks above do.
                        eligible = {f"frame:{frame.frame_id}": frame
                            for frame in (target, *frame_window.window(owner_id, session_id))
                            if frame.visual_session_id == lease.visual_session_id
                            and frame.frame_sequence >= target.frame_sequence
                            and target.captured_at <= frame.captured_at <= self.clock()
                            and f"frame:{frame.frame_id}" in origins
                            and f"frame:{frame.frame_id}" in result.evidence_ids}
                        fact = result.current_view
                        sources = set(fact.sources) if fact is not None else set()
                        if fact is not None and sources and sources.issubset(eligible):
                            return {"status": "available", "requested_at": requested_at.isoformat(),
                                "target_frame_id": str(target.frame_id), "facts": [{
                                    "description": fact.description, "basis": "observed",
                                    "sources": sorted(sources), "evidence_at": max(
                                        eligible[source].captured_at for source in sources).isoformat()}],
                                "partial": False, "text":
                                "Current camera observations."}
                    result = await self.analyzer.wait_for_visual_result(result)
        except TimeoutError:
            return unavailable("deadline")
        except RuntimeError:
            return unavailable("perception_unavailable")

    async def read_context(self) -> PerceptionContext:
        snapshot = self.state.read()  # RAM-only; no Hindsight/SQL request here.
        if not snapshot["ready"]:
            raise RuntimeError("companion_state_not_initialized")
        envelope, human = snapshot.get("human"), None
        self.last_context_issue = None
        if envelope:
            try:
                content = HumanStateContent.model_validate(envelope["content"])
                human = HumanState(**content.model_dump(), revision=envelope["revision"],
                    generated_at=envelope["generated_at"], observed_through=envelope["last_observed_at"],
                    incorporated_evidence_ids=tuple(sorted(_human_sources(content)))[-64:])
            except (ValidationError, KeyError, TypeError):
                self.last_context_issue = "previous_human_schema_unavailable"
        reminders = []
        for row in snapshot["system"]["reminders"]:
            if (row.get("status") != "active" or row.get("remind_at")
                    or row.get("metadata", {}).get("background") or not row.get("condition")):
                continue
            try:
                # Carry the real updated_at token; do not invent a database revision.
                reminders.append(SituationalReminder(id=row["id"], revision=row["updated_at"],
                    condition=row["condition"], title=row.get("title", ""),
                    notification_status=self.state.reminder_notification_status(row["id"], row["updated_at"])))
            except (ValidationError, KeyError, TypeError):
                self.last_context_issue = "invalid_reminder_context"
        if len(reminders) > 32:
            self.last_context_issue = "reminder_context_truncated"
        return PerceptionContext(human, tuple(reminders[:32]), snapshot["human_generation"])

    def submit_frame(self, frame: VisualFrameRecord, image_bytes: bytes):
        """Borrow already-authenticated, camera-lease-validated ingestion input."""
        if frame.owner_id != self.state.owner_id or frame.raw_image_status != "available":
            raise ValueError("foreign or unavailable visual input")
        evidence = FastPerceptionFrame(frame.id, frame.frame_sequence, frame.captured_at,
                                        frame.source, frame.content_type, bytes(image_bytes))
        origin = EvidenceOrigin(f"frame:{frame.id}", str(frame.session_id), str(frame.id), frame.captured_at)
        return self.analyzer.submit(EventBatch(self.state.owner_id, frames=(evidence,), origins=(origin,)))

    async def enqueue_frame(self, frame: VisualFrameRecord, image_bytes: bytes) -> None:
        """Stored-frame path. Source storage is never rolled back by analysis."""
        if frame.owner_id != self.state.owner_id:
            raise ValueError("foreign visual input")
        repository = self.frame_repository
        if repository is not None:
            await repository.mark_analysis_pending(owner_id=frame.owner_id, frame_id=frame.id)
        try:
            dropped = self.submit_frame(frame, image_bytes)
        except Exception:
            if repository is not None:
                await repository.mark_analysis_failed(owner_id=frame.owner_id, frame_id=frame.id)
            raise
        if repository is not None:
            for item in dropped:
                await repository.mark_analysis_skipped(owner_id=frame.owner_id, frame_id=item.frame_id,
                                                       reason="multimodal_pending_window_coalesced")

    async def _finish_frames(self, batch, result, failure):
        repository = self.frame_repository
        if repository is None:
            return
        for frame in batch.frames:
            if failure == "cancelled":
                await repository.mark_analysis_skipped(owner_id=batch.owner_id, frame_id=frame.frame_id,
                                                       reason="multimodal_runtime_closed")
            elif failure:
                await repository.mark_analysis_failed(owner_id=batch.owner_id, frame_id=frame.frame_id)
            else:
                observation = result.memory_worthy_observation
                description = (observation.description if observation is not None
                               and f"frame:{frame.frame_id}" in observation.sources else None)
                await repository.apply_analysis_result(owner_id=batch.owner_id, frame_id=frame.frame_id,
                                                       frame_description=description, memory_tags=[])

    def observe_committed_record(self, record: ContextRecord) -> None:
        """Synchronous post-commit hook; temporarily holds canonical record references.

        No copied transcript store, additional SQL, model call or Hindsight wait.
        A quiet caption snapshot is application-settled, never provider-final.
        """
        if self._closed:
            return
        if record.owner_id != self.state.owner_id:
            raise ValueError("foreign canonical record")
        key = (record.source_namespace, record.source_event_id)
        payload = record.content or {}
        if (record.kind not in {"user_transcript", "assistant_output"}
                or payload.get("voice_route") != "live" or record.session_id is None):
            return
        if (not record.is_current or record.availability not in {"available", "not_finalized"}
                or record.finality not in {"final", "revised", "provisional"}):
            self._captions.pop(key, None)
            return
        current = self._captions.get(key)
        if record.revision <= self._caption_revisions.get(key, 0) or (
                current is not None and record.revision <= current[0].revision):
            return
        if key not in self._captions and len(self._captions) >= 16:
            self._captions.popitem(last=False)
            self.last_context_issue = "caption_settle_capacity_exceeded"
        delay = 0 if record.finality in {"final", "revised"} else self.caption_settle_seconds
        self._captions[key] = (record, monotonic() + delay, self.state.read()["human_generation"])
        self._caption_wake.set()
        if self._caption_task is None:
            self._caption_task = asyncio.create_task(self._settle_captions(), name="perception-caption-settle")

    async def _settle_captions(self):
        try:
            while self._captions:
                self._caption_wake.clear()
                remaining = min(item[1] for item in self._captions.values()) - monotonic()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(self._caption_wake.wait(), remaining)
                        continue
                    except TimeoutError:
                        pass
                for key, (record, deadline, generation) in tuple(self._captions.items()):
                    if deadline > monotonic():
                        continue
                    del self._captions[key]
                    if generation != self.state.read()["human_generation"]:
                        continue
                    caption = CanonicalCaption(record.sequence, str(record.session_id), record.source_namespace,
                        record.source_event_id, record.revision, record.kind, record.source_time,
                        record.recorded_at, record.availability, record.content)
                    try:
                        self.submit_settled_caption(caption, owner_id=record.owner_id)
                        self._caption_revisions[key] = record.revision
                        self._caption_revisions.move_to_end(key)
                        while len(self._caption_revisions) > 128:
                            self._caption_revisions.popitem(last=False)
                    except (ValueError, TypeError):
                        self.last_context_issue = "settled_caption_unusable"
        finally:
            self._caption_task = None

    def submit_settled_caption(self, caption: CanonicalCaption, *, owner_id: str) -> None:
        """Use whole canonical captions, never live-fragment observation rows."""
        if owner_id != self.state.owner_id or not caption.usable:
            raise ValueError("foreign or unusable canonical caption")
        observed = caption.source_time or caption.recorded_at
        # This is the revision's original approximate application observation,
        # not database commit, settlement, analysis or publication time. Older
        # captions lack it: retain their known time rather than invent freshness.
        revision_time = caption.content.get("observed_at")
        observed_until = datetime.fromisoformat(revision_time) if revision_time is not None else observed
        identity = json.dumps([caption.source_namespace, caption.source_event_id], separators=(",", ":"))
        document_id = source_document_id("live-caption", caption.session_id, identity)
        evidence = SettledDialogue(caption_id=document_id, revision=caption.revision,
            speaker=caption.content["speaker"], text=caption.content["text"],
            observed_from=observed, observed_until=observed_until, finality="application_settled",
            time_basis="application_ingress_approximate")
        origin = EvidenceOrigin(evidence.evidence_id, caption.session_id, identity, observed_until, document_id)
        self.analyzer.submit(EventBatch(self.state.owner_id, dialogue=(evidence,), origins=(origin,)))

    @staticmethod
    def _delta_document(result: MultimodalResult) -> MemoryDocument | None:
        observation = result.memory_worthy_observation
        if observation is None:
            return None
        origins = {origin.evidence_id: origin for origin in result.origins}
        selected = [origins[source] for source in observation.sources]
        frames = [origin for origin in selected if origin.evidence_id.startswith("frame:")]
        if not frames:
            return None  # Dialogue has its own ingestion owner.
        anchor = max(frames, key=lambda origin: (origin.observed_at, origin.source_id))
        # One delta document per canonical anchor frame. Re-evaluation replaces
        # that document, rather than minting an ID from wording or wall-clock time.
        return ObservationDelta(source_id=f"multimodal-frame:{anchor.source_id}",
            session_id=anchor.session_id, text=observation.description,
            observed_from=min(origin.observed_at for origin in selected),
            observed_until=max(origin.observed_at for origin in selected), analyzed_at=result.analyzed_at,
            frame_ids=tuple(origin.source_id for origin in frames)).document()

    def _pending_assistant_revisions(self, result):
        pending = dict(self.analyzer.pending_assistant_revisions())
        for record, _, generation in self._captions.values():
            if record.kind != "assistant_output" or generation != result.human_generation:
                continue
            identity = json.dumps([record.source_namespace, record.source_event_id], separators=(",", ":"))
            document = source_document_id("live-caption", str(record.session_id), identity)
            pending[document] = max(pending.get(document, 0), record.revision)
        consumed = dict(result.assistant_revisions)
        return tuple((key, revision) for key, revision in pending.items() if consumed.get(key, 0) < revision)

    async def publish(self, result: MultimodalResult) -> None:
        # One bounded summary per pass. Never log prompts, descriptions, quotes,
        # model output, images, or exception messages (which may contain input).
        diagnostic = {"pass_id": result.diagnostic_id, "stage": "publication",
            "generation": result.human_generation, "outcome": "started",
            "human": "not_requested", "deliveries": [], "released": [], "triggers": []}
        try:
            await self._publish(result, diagnostic)
        except BaseException as error:
            diagnostic["error_type"] = type(error).__name__
            raise
        finally:
            LOGGER.info("companion_perception_diagnostic=%s", json.dumps(diagnostic, separators=(",", ":")))

    async def _publish(self, result: MultimodalResult, diagnostic) -> None:
        if result.owner_id != self.state.owner_id:
            diagnostic["outcome"] = "foreign_owner"
            raise ValueError("foreign perception result")
        snapshot = self.state.read()
        diagnostic["current_generation"] = snapshot["human_generation"]
        self.last_publication_issue = None
        if result.human_generation != snapshot["human_generation"]:
            self.last_publication_issue = "discarded_pre_forget_result"
            diagnostic["outcome"] = self.last_publication_issue
            return
        origins = {origin.evidence_id: origin for origin in result.origins}
        if set(origins) != set(result.evidence_ids):
            self.last_publication_issue = "missing_canonical_source_binding"
            diagnostic["outcome"] = self.last_publication_issue
            return
        document = self._delta_document(result) if self.intake is not None else None
        can_queue = document is not None and not self._memory_queue.full()
        if document is not None and not can_queue:
            self.last_memory_error = "memory_queue_full_delta_not_submitted"

        def documents_for(evidence_ids):
            ids = {origins[source].document_id for source in evidence_ids
                   if source in origins and origins[source].document_id is not None}
            if can_queue and set(evidence_ids).intersection(result.memory_worthy_observation.sources):
                ids.add(document.document_id)
            return tuple(sorted(ids))

        human = result.updated_human_state
        if human is not None:
            prior = snapshot.get("human") or {}
            # Avoid replacing another writer's newer interpretation at the same
            # observation time. CompanionState also enforces generation/time.
            if prior.get("revision", 0) == result.expected_human_revision:
                sources = _human_sources(human.content())
                documents = set(documents_for(sources))
                if sources - set(origins):
                    # Prior facts survive this update; preserve their document links.
                    # Invalidation conservatively clears all human state, so bounded
                    # linkage is not relied upon as a complete deletion index.
                    documents.update(prior.get("source_document_ids", ()))
                applied = await self.state.update_human(human.content().model_dump(mode="json"),
                    observed_at=result.observed_through, source_document_ids=tuple(sorted(documents))[-128:],
                    generation=result.human_generation)
                diagnostic["human"] = "applied" if applied else "refused"
            else:
                self.last_publication_issue = "human_revision_changed"
                diagnostic["human"] = self.last_publication_issue
        for delivery in result.observed_reminder_deliveries:
            applied = await self.state.observe_reminder_delivery(delivery, generation=result.human_generation,
                                                                source_document_ids=documents_for(delivery.sources))
            diagnostic["deliveries"].append({"id": delivery.reminder_id,
                "revision": delivery.reminder_revision, "applied": applied,
                "notification_status": self.state.reminder_notification_status(
                    delivery.reminder_id, delivery.reminder_revision)})
        # Consume the captured barrier after recording any observed delivery.
        # Never drain this worker from its own publish callback.
        ready = await self.state.consume_assistant_revisions(result.assistant_revisions)
        diagnostic["consumed_assistant_revisions"] = list(result.assistant_revisions)
        for notice in ready:
            released = {"id": notice["reminder_id"], "notice_id": notice["id"], "outcome": "callback_started"}
            diagnostic["released"].append(released)
            match = TriggeredReminder(notice["reminder_id"], notice["reminder_revision"],
                notice.get("evidence_description", notice["text"]), (), datetime.fromisoformat(notice["evidence_at"]))
            await self.on_reminder(match, tuple(notice["source_document_ids"]), notice["generation"])
            released.update(outcome="callback_returned", notification_status=self.state.reminder_notification_status(
                notice["reminder_id"], notice["reminder_revision"]))
        waiting = self._pending_assistant_revisions(result)
        diagnostic["pending_assistant_revisions"] = list(waiting)
        for match in result.triggered_reminders:
            if self.state.read()["human_generation"] != result.human_generation:
                self.last_publication_issue = "discarded_pre_forget_result"
                diagnostic["outcome"] = self.last_publication_issue
                return
            trigger = {"id": match.reminder_id, "revision": match.reminder_revision,
                       "outcome": "started"}
            diagnostic["triggers"].append(trigger)
            try:
                # Main rechecks updated_at, eligibility, occurrence, quiet and
                # actual delivery. This callback does not imply a spoken notice.
                if waiting:
                    notice = await self.state.reserve_reminder_notice(reminder_id=match.reminder_id,
                        revision=match.reminder_revision, evidence_at=match.evidence_at,
                        generation=result.human_generation, source_document_ids=documents_for(match.sources),
                        waiting_for_assistant=waiting, evidence_description=match.description)
                    trigger["outcome"] = "reservation_created" if notice is not None else "reservation_refused"
                    if notice is not None:
                        trigger.update(notice_id=notice["id"], status=notice["status"],
                            expires_at=notice["expires_at"], waiting_for_assistant=notice["waiting_for_assistant"])
                else:
                    await self.on_reminder(match, documents_for(match.sources), result.human_generation)
                    trigger["outcome"] = "callback_returned"
                trigger["notification_status"] = self.state.reminder_notification_status(
                    match.reminder_id, match.reminder_revision)
            except Exception as error:
                self.last_reminder_error = type(error).__name__
                trigger.update(outcome="failed", error_type=self.last_reminder_error)
        if can_queue and self.state.read()["human_generation"] == result.human_generation:
            self._memory_queue.put_nowait((document, result.human_generation))
            if self._memory_task is None:
                self._memory_task = asyncio.create_task(self._submit_memory(), name="perception-memory-deltas")
        diagnostic["outcome"] = "finished"

    async def _submit_memory(self):
        try:
            while not self._memory_queue.empty():
                document, generation = self._memory_queue.get_nowait()
                try:
                    if generation != self.state.read()["human_generation"]:
                        self.last_memory_error = "discarded_pre_forget_delta"
                        continue
                    self.last_memory_document_id = document.document_id
                    async with asyncio.timeout(self.memory_timeout_seconds):
                        receipt = await self.intake.retain_observation(document,
                            is_current=lambda: generation == self.state.read()["human_generation"])
                    self.last_memory_status = receipt["status"]
                    self.memory_submissions += int(receipt["submitted"])
                    self.last_memory_error = None
                except asyncio.CancelledError:
                    self.last_memory_error = "submission_cancelled_outcome_unknown"
                    raise
                except Exception as error:
                    # Do not automatically retry an ambiguous paid submission.
                    self.last_memory_error = type(error).__name__
                finally:
                    self._memory_queue.task_done()
        finally:
            self._memory_task = None

    async def drain(self, *, include_memory=True):
        while self._caption_task is not None:
            await asyncio.shield(self._caption_task)
        await self.analyzer.drain()
        if include_memory:
            while self._memory_task is not None:
                await asyncio.shield(self._memory_task)

    async def close(self):
        self._closed = True
        if self._caption_task is not None:
            self._caption_task.cancel()
            await asyncio.gather(self._caption_task, return_exceptions=True)
        self._captions.clear()
        self._caption_revisions.clear()
        await self.analyzer.close()
        if self._memory_task is not None:
            self._memory_task.cancel()
            await asyncio.gather(self._memory_task, return_exceptions=True)
        while not self._memory_queue.empty():
            self._memory_queue.get_nowait()
            self._memory_queue.task_done()
        # Provider, NativeMemory and CompanionState are borrowed, not closed here.


class _LimitedTemporalModel:
    """Borrow the existing shared limiter without owning its client lifetime."""
    def __init__(self, model, limiter):
        self.model, self.limiter = model, limiter

    async def generate_temporal_json(self, **kwargs):
        async with self.limiter.limit():
            return await self.model.generate_temporal_json(**kwargs)


def initialize_companion_perception(*, state, visual_services, on_reminder, memory=None, intake=None,
                                    model=None, model_limiter=None, clock=None,
                                    caption_settle_seconds=0.8):
    """Explicit once-per-owner selection; does not start services or model work.

    Main owns CompanionState initialization and attaches observe_committed_record
    to each SessionContextCapture. Visual services close this runtime before the
    borrowed Gemini client. No parallel activation of the legacy frame analyzer.
    """
    existing = visual_services.companion_runtime
    if existing is not None:
        if existing.state is not state:
            raise ValueError("another companion owner is already selected")
        return existing
    if model is None:
        if visual_services.gemini_client is None:
            raise ValueError("configured Gemini client required for perception")
        model = GeminiFastVisualModel(visual_services.gemini_client)
    if model_limiter is not None:
        model = _LimitedTemporalModel(model, model_limiter)
    runtime = CompanionPerceptionRuntime(state=state, model=model, on_reminder=on_reminder,
        memory=memory, intake=intake, frame_repository=visual_services.repository,
        clock=clock or state.clock, caption_settle_seconds=caption_settle_seconds)
    visual_services.companion_runtime = runtime
    return runtime
