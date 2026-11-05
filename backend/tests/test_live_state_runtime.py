"""Offline behavioral tests for the opt-in Live State runtime, not DB adapters."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.live_state.models import (
    AggregateRef, CapabilitiesChanged, CheckEntry, ConversationView,
    EntryRetracted, EntryUpserted, GuardedPayloadBinding, LiveCapabilities,
    LiveSnapshot, ObservationEntry, ProducerCursor, SessionScope, SourceDependency,
    SourceStateChanged, SourceWatermark, StateCursor, TaskEntry, TurnRevised,
)
from app.live_state.projection import prepare_voice_context, select_voice_context
from app.live_state.reducer import (
    MAX_SEEN_EVENTS, MAX_SOURCE_WATERMARKS, dependency_available, expire,
    initial_state, reduce_event, register_producer,
)
from app.live_state.service import LiveStateService
from app.voice_routes.contracts import RouteIdentity


NOW = datetime(2026, 10, 30, 12, tzinfo=UTC)


def changed(model, **fields):
    return type(model)(**(model.model_dump() | fields))


class RuntimeFixtures:
    def setUp(self):
        self.scope = SessionScope(owner_id="runtime@example.test", session_id=UUID(int=1),
                                  area_id=UUID(int=2))
        self.source = SourceDependency(source_namespace="knowledge_source", source_id=UUID(int=3),
                                       revision=7, generation=3)
        self.epoch = UUID(int=4)
        self.aggregate_id = UUID(int=5)
        self.root = UUID(int=6)
        self.now = NOW
        self.state = register_producer(initial_state(self.scope, epoch=UUID(int=7), now=NOW),
                                       "fixture", self.epoch, now=NOW).state

    def fields(self, sequence, **changes):
        return dict(event_id=uuid4(), scope=self.scope, emitted_at=NOW,
            producer=ProducerCursor(producer_id="fixture", producer_epoch=self.epoch,
                aggregate=AggregateRef(namespace="projection", aggregate_id=self.aggregate_id,
                                       revision=sequence))) | changes

    def entry(self, *, sources=None, roots=None, guarded=True, **changes):
        sources = (self.source,) if sources is None else sources
        roots = (self.root,) if roots is None else roots
        fields = dict(entry_id="observation", scope=self.scope, received_at=NOW,
            observed_at=NOW - timedelta(seconds=2), evidence_kind="observed",
            summary="The display reads 42.", sources=sources)
        if guarded:
            fields["guard_binding"] = GuardedPayloadBinding(scope=self.scope,
                payload_root_ids=roots, dependencies=sources)
        return ObservationEntry(**(fields | changes))

    def upsert(self, sequence, entry=None, **fields):
        return EntryUpserted(**self.fields(sequence, **fields),
                             entry=entry if entry is not None else self.entry())

    def watermark(self, sequence, source=None, availability="active", **fields):
        return SourceStateChanged(**self.fields(sequence, **fields), watermark=SourceWatermark(
            source=source if source is not None else self.source,
            availability=availability, recorded_at=NOW))

    def apply(self, state, event, *, now=NOW):
        result = reduce_event(state, event, now=now)
        self.assertTrue(result.applied, result.reason)
        return result.state

    def active(self):
        return self.apply(self.state, self.watermark(1))

    def snapshot(self, *entries, **changes):
        return LiveSnapshot(**(dict(scope=self.scope, cursor=StateCursor(epoch=UUID(int=7), revision=5),
            as_of=NOW, readiness="ready", entries=entries) | changes))


class LiveStateReducerTests(RuntimeFixtures, unittest.TestCase):
    def test_events_cannot_register_their_own_producer_or_epoch(self):
        event = self.upsert(1, self.entry(sources=()))
        for producer in (changed(event.producer, producer_id="unknown"),
                         changed(event.producer, producer_epoch=uuid4())):
            with self.subTest(producer=producer):
                result = reduce_event(self.state, changed(event, producer=producer), now=NOW)
                self.assertFalse(result.applied)
                self.assertEqual(result.reason, "unregistered_or_retired_producer")
                self.assertIs(result.state, self.state)

    def test_scope_isolation_includes_owner_session_and_area(self):
        for scope in (changed(self.scope, owner_id="other"), changed(self.scope, session_id=uuid4()),
                      changed(self.scope, area_id=uuid4())):
            with self.subTest(scope=scope):
                event = self.watermark(1, scope=scope)
                self.assertEqual(reduce_event(self.state, event, now=NOW).reason, "scope_mismatch")

    def test_duplicate_and_out_of_order_events_do_not_advance_snapshot(self):
        event = self.upsert(10)
        state = self.apply(self.active(), event)
        for duplicate, reason in ((event, "duplicate_event"),
                                  (self.upsert(9), "outdated_projection_sequence"),
                                  (self.upsert(10), "outdated_projection_sequence")):
            with self.subTest(reason=reason):
                result = reduce_event(state, duplicate, now=NOW)
                self.assertFalse(result.applied)
                self.assertEqual(result.reason, reason)
                self.assertIs(result.state, state)

    def test_projection_sequence_is_independent_of_content_revision_and_wall_clock(self):
        state = self.apply(self.active(), self.upsert(2))
        event = self.upsert(3, self.entry(summary="Later canonical read."),
                            emitted_at=NOW - timedelta(days=1))
        state = self.apply(state, event)
        self.assertEqual(state.snapshot.entries[0].sources[0].revision, 7)
        self.assertEqual(state.snapshot.producer_cursors[0].aggregate.revision, 3)
        self.assertEqual(state.snapshot.entries[0].summary, "Later canonical read.")

    def test_each_aggregate_has_independent_sequence_ordering(self):
        state = self.apply(self.active(), self.upsert(50))
        event = self.upsert(1, self.entry(entry_id="other"))
        event = changed(event, producer=changed(event.producer,
            aggregate=changed(event.producer.aggregate, aggregate_id=uuid4())))
        state = self.apply(state, event)
        self.assertEqual({entry.entry_id for entry in state.snapshot.entries}, {"observation", "other"})

    def test_rotating_producer_epoch_rejects_old_inflight_work_and_resets_sequence(self):
        state = self.apply(self.active(), self.upsert(50))
        new_epoch = uuid4()
        state = register_producer(state, "fixture", new_epoch, now=NOW).state
        self.assertFalse(state.snapshot.producer_cursors)
        self.assertEqual(reduce_event(state, self.upsert(51), now=NOW).reason,
                         "unregistered_or_retired_producer")
        event = self.upsert(1, self.entry(summary="Reconciled."))
        state = self.apply(state, changed(event, producer=changed(event.producer, producer_epoch=new_epoch)))
        self.assertEqual(state.snapshot.entries[0].summary, "Reconciled.")

    def test_entry_requires_explicit_source_confirmation(self):
        result = reduce_event(self.state, self.upsert(1), now=NOW)
        self.assertEqual(result.reason, "dependency_not_confirmed")
        self.assertFalse(dependency_available(self.state, self.source))
        self.assertEqual(len(self.apply(self.active(), self.upsert(2)).snapshot.entries), 1)

    def test_historical_content_ids_require_individual_watermarks_at_current_generation(self):
        state = self.active()
        historical = changed(self.source, revision=2)
        self.assertFalse(dependency_available(state, historical))
        state = self.apply(state, self.watermark(2, historical))
        self.assertTrue(dependency_available(state, historical))
        self.assertTrue(dependency_available(state, self.source))
        state = self.apply(state, self.upsert(3, self.entry(sources=(historical,))))
        self.assertEqual(state.snapshot.entries[0].sources, (historical,))
        withdrawn = changed(self.source, generation=4)
        state = self.apply(state, self.watermark(4, withdrawn, "retracted"))
        self.assertFalse(state.snapshot.entries)
        self.assertFalse(dependency_available(state, historical))

    def test_unchanged_content_revision_withdrawal_blocks_late_resurrection(self):
        state = self.apply(self.active(), self.upsert(2))
        withdrawn = changed(self.source, generation=4)
        state = self.apply(state, self.watermark(3, withdrawn, "retracted"))
        self.assertEqual(withdrawn.revision, self.source.revision)
        self.assertFalse(state.snapshot.entries)
        self.assertEqual(reduce_event(state, self.watermark(4), now=NOW).reason, "old_source_generation")
        self.assertEqual(reduce_event(state, self.upsert(4), now=NOW).reason, "dependency_not_confirmed")
        self.assertEqual(reduce_event(state, self.watermark(4, withdrawn), now=NOW).reason,
                         "withdrawal_requires_new_authority_generation")
        restored = changed(withdrawn, generation=5)
        state = self.apply(state, self.watermark(5, restored))
        self.assertFalse(state.snapshot.entries, "Restoration must not recreate an old selection.")
        self.assertTrue(dependency_available(state, restored))

    def test_unknown_source_is_not_available_and_only_removes_dependent_entries(self):
        state = self.apply(self.active(), self.upsert(2))
        state = self.apply(state, self.upsert(3, self.entry(entry_id="independent", sources=())))
        state = self.apply(state, self.watermark(4, availability="unknown"))
        self.assertFalse(dependency_available(state, self.source))
        self.assertEqual([e.entry_id for e in state.snapshot.entries], ["independent"])
        state = self.apply(state, self.watermark(5))
        self.assertTrue(dependency_available(state, self.source))

    def test_source_without_lifecycle_orders_by_revision(self):
        source = changed(self.source, generation=None)
        state = self.apply(self.state, self.watermark(1, source))
        newer = changed(source, revision=8)
        state = self.apply(state, self.watermark(2, newer))
        self.assertFalse(dependency_available(state, source))
        self.assertTrue(dependency_available(state, newer))
        self.assertEqual(reduce_event(state, self.watermark(3, source), now=NOW).reason,
                         "old_source_generation")

    def test_source_version_domain_change_fences_instead_of_guessing(self):
        result = reduce_event(self.active(), self.watermark(2, changed(self.source, generation=None)), now=NOW)
        self.assertTrue(result.state.needs_resync)
        self.assertEqual(result.reason, "source_version_domain_changed")
        self.assertEqual(result.state.snapshot.readiness, "degraded")

    def test_retraction_tombstone_prevents_older_upsert(self):
        state = self.apply(self.active(), self.upsert(2))
        retract = EntryRetracted(**self.fields(4), entry_id="observation", reason="cancelled")
        state = self.apply(state, retract)
        self.assertFalse(state.snapshot.entries)
        result = reduce_event(state, self.upsert(3), now=NOW)
        self.assertEqual(result.reason, "outdated_projection_sequence")
        self.assertFalse(result.state.snapshot.entries)

    def test_expiry_boundary_removes_evidence_not_unresolved_durable_check(self):
        deadline = NOW + timedelta(seconds=10)
        entry = self.entry(fresh_until=deadline, expires_at=deadline + timedelta(seconds=20))
        check = CheckEntry(entry_id="check", scope=self.scope, received_at=NOW,
            evidence_kind="canonical_record", check_id=uuid4(), check_revision=1,
            lifecycle="tracking", question="Is shutdown confirmed?", durable=True,
            canonical_owner=AggregateRef(namespace="responsibility", aggregate_id=uuid4(), revision=1))
        state = self.apply(self.active(), self.upsert(2, entry))
        state = self.apply(state, self.upsert(3, check))
        before = state.snapshot
        self.assertFalse(expire(state, now=deadline - timedelta(microseconds=1)).applied)
        result = expire(state, now=deadline)
        self.assertTrue(result.critical)
        self.assertEqual(result.state.snapshot.entries, (check,))
        self.assertEqual(check.lifecycle, "tracking")
        self.assertEqual(len(before.entries), 2)

    def test_expired_upsert_is_rejected_without_consuming_sequence(self):
        state = self.active()
        expired = self.upsert(2, self.entry(expires_at=NOW))
        result = reduce_event(state, expired, now=NOW)
        self.assertEqual(result.reason, "expired_entry")
        self.assertIs(result.state, state)
        self.assertEqual(len(self.apply(state, self.upsert(2)).snapshot.entries), 1)

    def test_turn_changes_require_canonical_request_availability(self):
        event = TurnRevised(**self.fields(2), request_source=self.source,
            conversation=ConversationView(turn_id=uuid4(), turn_revision=2, topic="Current task"))
        self.assertEqual(reduce_event(self.state, event, now=NOW).reason, "turn_source_not_confirmed")
        self.assertEqual(self.apply(self.active(), event).snapshot.conversation, event.conversation)

    def test_identical_events_and_explicit_times_replay_deterministically(self):
        events = (self.watermark(1), self.upsert(2), self.watermark(3, availability="unknown"))
        left, right = self.state, self.state
        for event in events:
            left = self.apply(left, event)
            right = self.apply(right, event)
        self.assertEqual(left, right)

    def test_bounded_dedupe_keeps_sequence_protection_after_event_eviction(self):
        state = self.state
        first = None
        for sequence in range(1, MAX_SEEN_EVENTS + 3):
            event = CapabilitiesChanged(**self.fields(sequence), capabilities=LiveCapabilities())
            first = event if first is None else first
            state = self.apply(state, event)
        self.assertEqual(len(state.seen_events), MAX_SEEN_EVENTS)
        self.assertNotIn(first.event_id, state.seen_events)
        self.assertEqual(reduce_event(state, first, now=NOW).reason, "outdated_projection_sequence")

    def test_entry_capacity_overflow_is_explicit_and_fails_closed(self):
        state = self.state
        for index in range(128):
            state = self.apply(state, self.upsert(index + 1, self.entry(entry_id=f"e-{index}", sources=())))
        result = reduce_event(state, self.upsert(129, self.entry(entry_id="overflow", sources=())), now=NOW)
        self.assertEqual(result.reason, "entry_capacity")
        self.assertTrue(result.state.needs_resync)
        self.assertFalse(result.state.snapshot.entries)
        self.assertEqual(reduce_event(result.state, self.upsert(130), now=NOW).reason,
                         "resynchronization_required")

    def test_cursor_and_producer_capacity_fence_without_evicting_ordering(self):
        cursors = tuple(ProducerCursor(producer_id="fixture", producer_epoch=self.epoch,
            aggregate=AggregateRef(namespace="projection", aggregate_id=UUID(int=1000 + i), revision=1))
            for i in range(256))
        seeded = replace(self.state, snapshot=changed(self.state.snapshot, producer_cursors=cursors))
        result = reduce_event(seeded, self.upsert(1, self.entry(sources=())), now=NOW)
        self.assertEqual(result.reason, "cursor_capacity")
        self.assertTrue(result.state.needs_resync)
        state = self.state
        for i in range(63):
            state = register_producer(state, f"producer-{i}", uuid4(), now=NOW).state
        result = register_producer(state, "overflow", uuid4(), now=NOW)
        self.assertEqual(result.reason, "producer_capacity")
        self.assertTrue(result.state.needs_resync)

    def test_tombstone_budget_does_not_silently_forget_withdrawals(self):
        marks = tuple(SourceWatermark(source=changed(self.source, source_id=UUID(int=1000 + i)),
            availability="retracted", recorded_at=NOW) for i in range(MAX_SOURCE_WATERMARKS))
        state = replace(self.state, watermarks=marks)
        result = reduce_event(state, self.watermark(1), now=NOW)
        self.assertEqual(result.reason, "source_watermark_capacity")
        self.assertTrue(result.state.needs_resync)
        self.assertEqual(result.state.watermarks, marks)


class LiveStateServiceTests(RuntimeFixtures, unittest.IsolatedAsyncioTestCase):
    async def make_service(self, capacity=256):
        service = LiveStateService(self.scope, clock=lambda: self.now, queue_capacity=capacity)
        await service.start()
        self.addAsyncCleanup(service.close)
        service.register_producer("fixture", self.epoch)
        self.assertTrue(service.submit(self.watermark(1)).queued)
        await asyncio.wait_for(service.wait_idle(), 1)
        service.mark_ready()
        return service

    async def test_enqueue_is_not_acceptance_and_snapshots_remain_immutable(self):
        service = await self.make_service()
        before = service.snapshot()
        self.assertTrue(service.submit(self.upsert(2)).queued)
        self.assertIs(service.snapshot(), before)
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertFalse(before.entries)
        self.assertEqual(service.snapshot().entries, (self.entry(),))
        self.assertGreater(service.snapshot().cursor.revision, before.cursor.revision)

    async def test_queue_coalesces_only_adjacent_compatible_newer_observations(self):
        service = await self.make_service(1)
        service.submit(self.upsert(2))
        newer = self.entry(summary="New display value 43.")
        self.assertEqual(service.submit(self.upsert(3, newer)).reason, "coalesced")
        self.assertEqual(service.submit(self.upsert(2)).reason, "outdated_queued_update")
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertEqual(service.snapshot().entries, (newer,))

    async def test_overflow_fences_and_requires_reconciliation(self):
        service = await self.make_service(1)
        service.submit(self.upsert(2))
        receipt = service.submit(self.upsert(3, self.entry(entry_id="other")))
        self.assertFalse(receipt.queued)
        self.assertTrue(service.needs_resync)
        self.assertEqual(service.snapshot().readiness, "degraded")
        self.assertFalse(service.snapshot().entries)
        with self.assertRaises(RuntimeError):
            service.mark_ready()
        self.assertFalse(service.submit(self.upsert(4)).queued)
        await asyncio.wait_for(service.wait_idle(), 1)

    async def test_overflow_never_coalesces_across_withdrawal_or_retraction(self):
        for critical in (self.watermark(3, changed(self.source, generation=4), "retracted"),
                         EntryRetracted(**self.fields(3), entry_id="observation", reason="cancelled")):
            with self.subTest(kind=critical.kind):
                service = await self.make_service(2)
                service.submit(self.upsert(2))
                service.submit(critical)
                self.assertFalse(service.submit(self.upsert(4)).queued)
                self.assertTrue(service.needs_resync)
                self.assertFalse(service.snapshot().entries)

    async def test_source_dependency_change_is_not_replaceable_queue_coalescing(self):
        service = await self.make_service(1)
        service.submit(self.upsert(2))
        service.submit(self.upsert(3, self.entry(sources=(changed(self.source, generation=4),))))
        self.assertTrue(service.needs_resync)

    async def test_subscriber_overflow_latches_until_acknowledged_current_cursor(self):
        service = await self.make_service()
        subscription = service.subscribe(capacity=1)
        original = await subscription.receive()
        service.submit(self.upsert(2))
        await asyncio.wait_for(service.wait_idle(), 1)
        service.submit(self.watermark(3, availability="unknown"))
        await asyncio.wait_for(service.wait_idle(), 1)
        notice = await subscription.receive()
        self.assertTrue(notice.resync_required)
        self.assertTrue(notice.critical)
        self.assertFalse(notice.snapshot.entries)
        self.assertFalse(subscription.acknowledge_resync(original.snapshot.cursor))
        self.assertTrue(subscription.acknowledge_resync(service.snapshot().cursor))
        service.submit(self.watermark(4))
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertFalse((await subscription.receive()).resync_required)

    async def test_reconnect_with_old_cursor_requires_snapshot_resync(self):
        service = await self.make_service()
        for cursor in (StateCursor(epoch=uuid4(), revision=service.snapshot().cursor.revision),
                       changed(service.snapshot().cursor, revision=0)):
            with self.subTest(cursor=cursor):
                subscription = service.subscribe(after=cursor)
                self.assertTrue((await subscription.receive()).resync_required)
                subscription.close()

    async def test_reconciliation_rotates_state_epoch_and_rejects_old_producer(self):
        service = await self.make_service()
        before = service.snapshot().cursor
        service.submit(self.upsert(2))
        service.reset_for_reconciliation()
        self.assertNotEqual(service.snapshot().cursor.epoch, before.epoch)
        self.assertEqual(service.snapshot().readiness, "warming")
        self.assertFalse(service.snapshot().entries)
        self.assertFalse(service.submit(self.upsert(3)).queued)
        new_epoch = uuid4()
        service.register_producer("fixture", new_epoch)
        self.assertFalse(service.submit(self.upsert(4)).queued)
        event = self.upsert(1)
        service.submit(changed(event, producer=changed(event.producer, producer_epoch=new_epoch)))
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertEqual(service.last_rejection, "dependency_not_confirmed")
        self.assertFalse(service.snapshot().entries)

    async def test_epoch_rotation_fences_already_queued_old_work(self):
        service = await self.make_service()
        service.submit(self.upsert(2))
        service.register_producer("fixture", uuid4())
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertFalse(service.snapshot().entries)
        self.assertEqual(service.last_rejection, "unregistered_or_retired_producer")

    async def test_expiry_uses_injected_clock_without_real_sleep(self):
        service = await self.make_service()
        service.submit(self.upsert(2, self.entry(fresh_until=NOW + timedelta(seconds=1))))
        await asyncio.wait_for(service.wait_idle(), 1)
        self.now += timedelta(seconds=1)
        service.submit(CapabilitiesChanged(**self.fields(3), capabilities=LiveCapabilities()))
        await asyncio.wait_for(service.wait_idle(), 1)
        self.assertFalse(service.snapshot().entries)

    async def test_close_wakes_subscriber_and_joins_writer(self):
        service = await self.make_service()
        subscription = service.subscribe()
        await subscription.receive()
        waiting = asyncio.create_task(subscription.receive())
        await service.close()
        self.assertIsNone(await asyncio.wait_for(waiting, 1))
        self.assertEqual(service.snapshot().readiness, "degraded")
        self.assertFalse(service.submit(self.upsert(2)).queued)
        with self.assertRaises(RuntimeError):
            await service.start()
        self.assertTrue(service._worker.done())

    async def test_failed_reducer_fences_state_and_unblocks_idle_waiters(self):
        service = await self.make_service()
        with patch("app.live_state.service.reduce_event", side_effect=ValueError("synthetic failure")):
            service.submit(self.upsert(2))
            await asyncio.wait_for(service.wait_idle(), 1)
        self.assertTrue(service.needs_resync)
        self.assertEqual(service.snapshot().readiness, "degraded")
        self.assertFalse(service.snapshot().entries)
        self.assertIsInstance(service._worker.exception(), ValueError)

    async def test_mutation_before_start_and_subscription_limits_are_explicit(self):
        service = LiveStateService(self.scope, clock=lambda: NOW)
        with self.assertRaises(RuntimeError):
            service.register_producer("fixture", self.epoch)
        service = await self.make_service()
        subscriptions = [service.subscribe() for _ in range(64)]
        with self.assertRaises(RuntimeError):
            service.subscribe()
        subscriptions[0].close()
        service.subscribe().close()
        for capacity in (0, 33):
            with self.subTest(capacity=capacity), self.assertRaises(RuntimeError):
                service.subscribe(capacity=capacity)


class VoiceProjectionRuntimeTests(RuntimeFixtures, unittest.IsolatedAsyncioTestCase):
    def route(self):
        return RouteIdentity(scope=self.scope, route_id=UUID(int=9), route_generation=1)

    async def test_combined_text_gets_exact_new_binding_not_union_of_parent_roots(self):
        other = changed(self.source, source_id=uuid4())
        entries = (self.entry(entry_id="a"),
                   self.entry(entry_id="b", sources=(other,), roots=(UUID(int=10),)))
        selection = select_voice_context(self.snapshot(*entries), now=NOW)
        calls = []
        new_root = UUID(int=11)

        async def bind(value):
            calls.append(value)
            return GuardedPayloadBinding(scope=self.scope, payload_root_ids=(new_root,),
                                         dependencies=value.dependencies)

        update = await prepare_voice_context(selection, self.route(), bind=bind, clock=lambda: NOW)
        self.assertEqual(calls, [selection])
        self.assertEqual(update.text, selection.text)
        self.assertEqual(set(selection.parent_payload_root_ids), {self.root, UUID(int=10)})
        self.assertEqual(update.guard_binding.payload_root_ids, (new_root,))
        self.assertEqual(set(update.dependencies), {self.source, other})

    async def test_unbound_stale_retracted_entries_are_omitted_as_whole_facts(self):
        entries = (self.entry(entry_id="unbound", guarded=False),
                   self.entry(entry_id="stale", fresh_until=NOW),
                   self.entry(entry_id="retracted", validity="retracted"), self.entry(entry_id="usable"))
        selection = select_voice_context(self.snapshot(*entries), now=NOW)
        self.assertEqual(selection.selected_entry_ids, ("usable",))
        self.assertEqual(set(selection.omitted_entry_ids), {"unbound", "stale", "retracted"})

    async def test_source_version_conflict_never_drops_dependencies_from_admitted_text(self):
        first = self.entry(entry_id="a", summary="Accepted version.")
        second = self.entry(entry_id="b", summary="Conflicting version.",
                            sources=(changed(self.source, generation=4),))
        selection = select_voice_context(self.snapshot(first, second), now=NOW)
        self.assertEqual(selection.selected_entry_ids, ("a",))
        self.assertEqual(selection.dependencies, (self.source,))
        self.assertNotIn(second.summary, selection.text)

    async def test_projection_preserves_original_observation_time_and_earliest_deadline(self):
        entry = self.entry(observed_at=NOW - timedelta(minutes=10),
            fresh_until=NOW + timedelta(seconds=2), expires_at=NOW + timedelta(seconds=20))
        selection = select_voice_context(self.snapshot(entry), now=NOW)
        self.assertIn(entry.observed_at.isoformat(), selection.text)
        self.assertEqual(selection.expires_at, entry.fresh_until)

    async def test_projection_preserves_observation_uncertainty_in_voice_brief(self):
        entry = self.entry(summary="The control appears dark.", confidence=0.2,
                           unknowns=("The off marking is unreadable.",))
        selection = select_voice_context(self.snapshot(entry), now=NOW)
        self.assertIn(entry.unknowns[0], selection.text,
                      "A voice briefing must not strip the uncertainty from selected evidence.")

    async def test_projection_preserves_canonical_task_status(self):
        task = TaskEntry(entry_id="task", scope=self.scope, received_at=NOW,
            evidence_kind="canonical_record", summary="Checking the calculation.",
            task=AggregateRef(namespace="task", aggregate_id=uuid4(), revision=2), status="running",
            sources=(self.source,), guard_binding=self.entry().guard_binding)
        selection = select_voice_context(self.snapshot(task), now=NOW)
        self.assertIn("running", selection.text)
        self.assertNotIn("completed", selection.text)

    async def test_text_entry_dependency_and_root_budgets_omit_whole_optional_items(self):
        large = self.entry(entry_id="a", summary="x" * 1000)
        small = self.entry(entry_id="b", summary="Short.")
        selection = select_voice_context(self.snapshot(large, small), now=NOW, max_chars=128)
        self.assertEqual(selection.selected_entry_ids, ("b",))
        self.assertLessEqual(len(selection.text), 128)
        entries = tuple(self.entry(entry_id=f"e-{i}", sources=tuple(
            changed(self.source, source_id=UUID(int=1000 + i * 16 + j)) for j in range(16))) for i in range(5))
        selection = select_voice_context(self.snapshot(*entries), now=NOW)
        self.assertEqual(len(selection.dependencies), 64)
        self.assertEqual(selection.omitted_entry_ids, ("e-4",))
        entries = (self.entry(entry_id="a", roots=tuple(UUID(int=2000 + i) for i in range(40))),
                   self.entry(entry_id="b", roots=tuple(UUID(int=3000 + i) for i in range(40))))
        selection = select_voice_context(self.snapshot(*entries), now=NOW)
        self.assertEqual(selection.selected_entry_ids, ("a",))
        self.assertEqual(len(selection.parent_payload_root_ids), 40)

    async def test_deterministic_ordering_and_entry_budget(self):
        a, b = self.entry(entry_id="a"), self.entry(entry_id="b")
        left = select_voice_context(self.snapshot(a, b), now=NOW, max_entries=1)
        right = select_voice_context(self.snapshot(b, a), now=NOW, max_entries=1)
        self.assertEqual(left, right)
        self.assertEqual(left.selected_entry_ids, ("a",))

    async def test_unready_context_is_explicit_source_free_and_does_not_expose_entries(self):
        for readiness in ("warming", "degraded"):
            with self.subTest(readiness=readiness):
                selection = select_voice_context(self.snapshot(self.entry(), readiness=readiness), now=NOW)
                update = await prepare_voice_context(selection, self.route(), bind=None, clock=lambda: NOW)
                self.assertIn(readiness, update.text)
                self.assertFalse(update.dependencies)
                self.assertIsNone(update.guard_binding)
                self.assertNotIn(self.entry().summary, update.text)

    async def test_source_backed_and_root_only_selections_require_real_binder(self):
        for entry in (self.entry(), self.entry(sources=())):
            with self.subTest(sources=entry.sources), self.assertRaises(ValueError):
                await prepare_voice_context(select_voice_context(self.snapshot(entry), now=NOW),
                                            self.route(), bind=None, clock=lambda: NOW)

    async def test_wrong_scope_missing_extra_or_changed_generation_binding_is_rejected(self):
        selection = select_voice_context(self.snapshot(self.entry()), now=NOW)
        wrong_scope = changed(self.scope, area_id=uuid4())
        bindings = (
            GuardedPayloadBinding(scope=wrong_scope, payload_root_ids=(uuid4(),), dependencies=(self.source,)),
            GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),)),
            GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),),
                                  dependencies=(changed(self.source, generation=4),)),
            GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),),
                                  dependencies=(self.source, changed(self.source, source_id=uuid4()))),
        )
        for binding in bindings:
            async def bind(value):
                return binding
            with self.subTest(binding=binding), self.assertRaises(ValueError):
                await prepare_voice_context(selection, self.route(), bind=bind, clock=lambda: NOW)

    async def test_expiry_before_or_during_async_binding_cannot_produce_context(self):
        deadline = NOW + timedelta(seconds=1)
        selection = select_voice_context(self.snapshot(self.entry(fresh_until=deadline)), now=NOW)
        calls = []
        async def bind(value):
            calls.append(value)
            self.now = deadline
            return GuardedPayloadBinding(scope=self.scope, payload_root_ids=(uuid4(),),
                                         dependencies=value.dependencies)
        self.assertIsNone(await prepare_voice_context(selection, self.route(), bind=bind, clock=lambda: deadline))
        self.assertFalse(calls)
        self.assertIsNone(await prepare_voice_context(selection, self.route(), bind=bind, clock=lambda: self.now))
        self.assertEqual(calls, [selection])

    async def test_canonical_binder_refusal_propagates_instead_of_trusting_cached_roots(self):
        selection = select_voice_context(self.snapshot(self.entry()), now=NOW)
        async def bind(value):
            raise PermissionError("source retired while binding")
        with self.assertRaisesRegex(PermissionError, "retired"):
            await prepare_voice_context(selection, self.route(), bind=bind, clock=lambda: NOW)

    async def test_cross_route_scope_is_rejected_before_binder(self):
        selection = select_voice_context(self.snapshot(self.entry()), now=NOW)
        async def bind(value):
            self.fail("Cross-scope data reached the binder.")
        with self.assertRaises(ValueError):
            await prepare_voice_context(selection, changed(self.route(), scope=changed(self.scope, session_id=uuid4())),
                                        bind=bind, clock=lambda: NOW)
