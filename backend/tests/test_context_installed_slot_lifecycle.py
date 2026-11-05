"""Offline installed-slot lifecycle; no bridge, database, provider or phone I/O."""

import asyncio
import unittest
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from uuid import uuid4

from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.realtime_adapter import RealtimeRouteError, RealtimeVoiceRoute, SourceError
from test_realtime_voice_route import FakeControl, NOW, make_context, make_identity, make_start, stop_route


class ContextInstalledSlotLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW
        self.owned = True
        self.replacement_allowed = True
        self.guard_error = None
        self.expire_at_guard = False
        self.hold_exit = False
        self.exit_entered = asyncio.Event()
        self.exit_release = asyncio.Event()
        self.exit_holds = []
        self.replacements = []
        self.control = FakeControl()
        bindings = replace(self.control.bindings(), source_guard=self.source_guard,
            context_replacement_guard=self.replacement_guard,
            context_reconciliation_current=lambda: self.owned)
        self.route = RealtimeVoiceRoute(bindings, media_transport="webrtc",
            input_codec="opus", output_codec="opus", io_timeout=0.15,
            start_timeout=0.15, clock=lambda: self.now)
        self.control.item_event = lambda payload: self.route.observe_item_event(self.route.identity, payload)
        self.addAsyncCleanup(stop_route, self.route)
        request = make_start()
        self.initial = request.initial_context
        self.assertEqual((await self.route.start(request)).status, "completed")

    @asynccontextmanager
    async def source_guard(self, value):
        async with self.control.source_guard(value):
            if self.expire_at_guard:
                self.now = value.expires_at
            if self.guard_error is not None:
                raise self.guard_error
            yield

    @asynccontextmanager
    async def replacement_guard(self, update):
        self.replacements.append(update)
        try:
            yield self.replacement_allowed
        finally:
            self.exit_holds.append(self.route.context_reconciliation_pending)
            if self.hold_exit:
                self.exit_entered.set()
                await self.exit_release.wait()

    async def clear_initial(self):
        await self.route.clear_optional_context_for_response(current=lambda: self.owned)
        self.assertIsNone(self.route._context_item)
        self.assertTrue(self.route.context_reconciliation_pending)
        self.assertTrue(self.route.response_context_empty_proof[2].confirmed_deleted_without_create)

    async def assert_post_delete_hold(self, *, code, reason, expired=False):
        update = make_context(revision=1)
        self.guard_error = SourceError(code, "Offline fixture source failure.")
        self.expire_at_guard = expired
        self.hold_exit = True
        evidence = ContextWriteEvidence()
        publication = asyncio.create_task(self.route.publish_context_tracked(update, write_evidence=evidence))
        try:
            await asyncio.wait_for(self.exit_entered.wait(), 1)
            self.assertFalse(publication.done())
            self.assertFalse(evidence.work_joined)
            self.assertIsNone(self.route._context_item)
            self.assertTrue(self.route.context_reconciliation_pending)
            self.assertTrue(self.exit_holds[-1])
            self.assertFalse(self.route.can_refill_context(update))
        finally:
            self.exit_release.set()
            receipt = await publication
        self.assertEqual((receipt.status, receipt.reason), ("rejected", reason))
        self.assertTrue(evidence.confirmed_deleted_without_create)
        self.assertTrue(self.route.context_reconciliation_pending)
        self.assertEqual(self.route.can_refill_context(update), not expired)
        successor = make_context(revision=2, created_at=self.now,
                                 expires_at=self.now + timedelta(minutes=1))
        self.assertTrue(self.route.can_refill_context(successor))
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertEqual([event["type"] for event in self.control.writes],
                         ["conversation.item.create", "conversation.item.delete"])

    async def test_expired_post_delete_holds_before_tracked_return(self):
        await self.assert_post_delete_hold(code="voice_context_expired",
            reason="context_not_current_snapshot", expired=True)

    async def test_unavailable_post_delete_holds_before_tracked_return(self):
        await self.assert_post_delete_hold(code="live_state_runtime_unavailable",
            reason="context_source_unavailable")

    async def test_hold_clears_only_after_successful_replacement_write(self):
        update = make_context(revision=1)
        self.control.block_id = update.update_id
        evidence = ContextWriteEvidence()
        publication = asyncio.create_task(self.route.publish_context_tracked(update, write_evidence=evidence))
        try:
            await asyncio.wait_for(self.control.guard_entered.wait(), 1)
            self.assertFalse(publication.done())
            self.assertIsNone(self.route._context_item)
            self.assertTrue(self.route.context_reconciliation_pending)
            self.assertFalse(self.route.can_refill_context(update))
        finally:
            self.control.guard_release.set()
            receipt = await publication
        self.assertEqual(receipt.status, "written")
        self.assertFalse(self.route.context_reconciliation_pending)
        self.assertFalse(self.exit_holds[-1])
        self.assertTrue(self.route.can_retain_context(update))

    async def assert_exact_refill(self, *, tracked):
        await self.clear_initial()
        previous = self.control.writes[0]["item"]["id"]
        self.now += timedelta(minutes=1)
        self.assertTrue(self.route.can_refill_context(self.initial))
        guards_before = len(self.control.guarded)
        replacements_before = len(self.replacements)
        evidence = ContextWriteEvidence()
        if tracked:
            receipt = await self.route.publish_context_tracked(self.initial, write_evidence=evidence)
        else:
            receipt = await self.route.publish_context(self.initial)
        self.assertEqual(receipt.status, "written")
        self.assertEqual(receipt.update_id, self.initial.update_id)
        self.assertEqual(receipt.cursor, self.initial.cursor)
        self.assertIs(self.route._context, self.initial)
        self.assertEqual(self.route._context.created_at, NOW)
        self.assertEqual(self.route._context.expires_at, NOW + timedelta(minutes=5))
        self.assertIs(self.control.guarded[-1], self.initial)
        self.assertIs(self.replacements[-1], self.initial)
        self.assertEqual(len(self.control.guarded), guards_before + 1)
        self.assertEqual(len(self.replacements), replacements_before + 1)
        self.assertNotEqual(self.route._context_item, previous)
        self.assertFalse(self.route.context_reconciliation_pending)
        self.assertIsNone(self.route.response_context_empty_proof)
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertTrue(self.route.can_retain_context(self.initial))
        self.assertEqual([event["type"] for event in self.control.writes],
                         ["conversation.item.create", "conversation.item.delete", "conversation.item.create"])
        if tracked:
            self.assertTrue(evidence.work_joined)
            self.assertEqual(evidence.first_operation, "context_create")
            self.assertFalse(evidence.deleted_context_acknowledged)

    async def test_exact_same_update_tracked_refill_preserves_snapshot(self):
        await self.assert_exact_refill(tracked=True)

    async def test_exact_same_update_untracked_refill_still_uses_guards(self):
        await self.assert_exact_refill(tracked=False)

    async def test_successor_refills_without_repeating_acknowledged_delete(self):
        await self.clear_initial()
        update = make_context(revision=1)
        self.assertTrue(self.route.can_refill_context(update))
        evidence = ContextWriteEvidence()
        receipt = await self.route.publish_context_tracked(update, write_evidence=evidence)
        self.assertEqual(receipt.status, "written")
        self.assertEqual(evidence.first_operation, "context_create")
        self.assertFalse(self.route.context_reconciliation_pending)
        self.assertEqual(sum(event["type"] == "conversation.item.delete" for event in self.control.writes), 1)

    async def test_same_cursor_mutations_are_not_exact_refills(self):
        await self.clear_initial()
        before = len(self.control.writes)
        for changes in ({"update_id": uuid4()}, {"text": "Different facts."},
                        {"created_at": NOW + timedelta(seconds=1)},
                        {"expires_at": NOW + timedelta(minutes=6)}, {"mode": "delta"}):
            with self.subTest(changes=changes):
                update = self.initial.model_copy(update=changes)
                self.assertFalse(self.route.can_refill_context(update))
                self.assertEqual((await self.route.publish_context(update)).status, "rejected")
                self.assertTrue(self.route.can_refill_context(self.initial))
        self.assertFalse(self.route.can_refill_context(make_context(make_identity(2), revision=1)))
        wrong_epoch = self.initial.cursor.model_copy(update={"epoch": uuid4(), "revision": 1})
        self.assertFalse(self.route.can_refill_context(self.initial.model_copy(update={"cursor": wrong_epoch})))
        self.assertTrue(self.route.can_refill_context(self.initial))
        self.assertEqual(len(self.control.writes), before)

    async def test_expired_exact_update_rejected_but_fresh_successor_eligible(self):
        await self.clear_initial()
        self.now = self.initial.expires_at
        self.assertFalse(self.route.can_refill_context(self.initial))
        before = len(self.control.writes)
        receipt = await self.route.publish_context(self.initial)
        self.assertEqual((receipt.status, receipt.reason), ("rejected", "context_not_current_snapshot"))
        successor = make_context(revision=1, created_at=self.now,
                                 expires_at=self.now + timedelta(minutes=1))
        self.assertTrue(self.route.can_refill_context(successor))
        self.assertTrue(self.route.context_reconciliation_pending)
        self.assertEqual(len(self.control.writes), before)

    async def test_missing_empty_proof_is_unknown_not_written_or_retried(self):
        await self.clear_initial()
        self.route._empty_context_candidate = None
        self.assertFalse(self.route.can_refill_context(self.initial))
        before = len(self.control.writes)
        evidence = ContextWriteEvidence()
        receipt = await self.route.publish_context_tracked(self.initial, write_evidence=evidence)
        self.assertEqual((receipt.status, receipt.reason), ("unknown", "context_item_ownership_unknown"))
        self.assertTrue(evidence.proven_unwritten)
        self.assertEqual(len(self.control.writes), before)
        self.assertTrue(self.route.context_reconciliation_pending)

    async def test_unjoined_empty_proof_cannot_refill_or_be_consumed_by_query(self):
        await self.clear_initial()
        deletion = ContextWriteEvidence()
        deletion.begin_tracking()
        self.assertTrue(deletion.before_write("context_delete", item_id="fixture-deleted-item"))
        self.assertTrue(deletion.record_delete_acknowledged("fixture-deleted-item"))
        deletion.close()
        proof = (self.initial, self.route._source_epoch, deletion)
        self.route._empty_context_candidate = proof
        before = len(self.control.writes)
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertIs(self.route._empty_context_candidate, proof)
        receipt = await self.route.publish_context(self.initial)
        self.assertEqual(receipt.status, "unknown")
        self.assertEqual(len(self.control.writes), before)
        deletion.mark_joined()
        self.assertTrue(self.route.can_refill_context(self.initial))

    async def test_unacknowledged_delete_never_becomes_refill_ownership(self):
        self.control.item_event = None
        with self.assertRaises(RealtimeRouteError):
            await self.route.clear_optional_context_for_response(current=lambda: True)
        before = len(self.control.writes)
        self.assertTrue(self.route._response_clear_unresolved)
        self.assertIsNone(self.route.response_context_empty_proof)
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertFalse(self.route.can_refill_context(make_context(revision=1)))
        with self.assertRaises(RealtimeRouteError):
            await self.route.clear_optional_context_for_response(current=lambda: True)
        self.assertEqual(len(self.control.writes), before)

    async def test_withdrawn_owner_cannot_refill_or_restore_old_proof(self):
        await self.clear_initial()
        self.owned = False
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertIsNone(self.route._empty_context_candidate)
        self.owned = True
        self.assertFalse(self.route.can_refill_context(self.initial))
        before = len(self.control.writes)
        self.assertEqual((await self.route.publish_context(self.initial)).status, "unknown")
        self.assertEqual(len(self.control.writes), before)
        self.assertTrue(self.route.context_reconciliation_pending)

    async def test_source_withdrawal_still_vetoes_eligible_exact_refill(self):
        await self.clear_initial()
        self.assertTrue(self.route.can_refill_context(self.initial))
        self.control.valid = False
        before = len(self.control.writes)
        evidence = ContextWriteEvidence()
        receipt = await self.route.publish_context_tracked(self.initial, write_evidence=evidence)
        self.assertEqual(receipt.status, "unknown")
        self.assertTrue(evidence.proven_unwritten)
        self.assertFalse(self.route.can_refill_context(self.initial))
        self.assertEqual((await self.route.publish_context(self.initial)).status, "unknown")
        self.assertEqual(len(self.control.writes), before)
        self.assertTrue(self.route.context_reconciliation_pending)

    async def test_replacement_guard_still_vetoes_eligible_exact_refill(self):
        await self.clear_initial()
        self.replacement_allowed = False
        before = len(self.control.writes)
        receipt = await self.route.publish_context(self.initial)
        self.assertEqual((receipt.status, receipt.reason), ("rejected", "context_response_active"))
        self.assertEqual(len(self.control.writes), before)
        self.assertTrue(self.route.can_refill_context(self.initial))
        self.assertTrue(self.route.context_reconciliation_pending)

    async def test_write_budget_still_vetoes_eligible_exact_refill(self):
        await self.clear_initial()
        evidence = ContextWriteEvidence(write_budget=lambda: False)
        before = len(self.control.writes)
        receipt = await self.route.publish_context_tracked(self.initial, write_evidence=evidence)
        self.assertEqual(receipt.status, "unknown")
        self.assertTrue(evidence.proven_unwritten)
        self.assertEqual(len(self.control.writes), before)
        self.assertTrue(self.route.can_refill_context(self.initial))
        self.assertTrue(self.route.context_reconciliation_pending)

    async def test_duplicate_present_context_remains_idempotent(self):
        before = len(self.control.writes)
        guards_before = len(self.control.guarded)
        replacements_before = len(self.replacements)
        item = self.route._context_item
        self.assertFalse(self.route.can_refill_context(self.initial))
        evidence = ContextWriteEvidence()
        receipt = await self.route.publish_context_tracked(self.initial, write_evidence=evidence)
        self.assertEqual(receipt.status, "written")
        self.assertTrue(evidence.proven_unwritten)
        self.assertEqual(self.route._context_item, item)
        self.assertEqual(len(self.control.writes), before)
        self.assertEqual(len(self.control.guarded), guards_before)
        self.assertEqual(len(self.replacements), replacements_before)
        self.assertFalse(self.route.context_reconciliation_pending)


if __name__ == "__main__":
    unittest.main()
