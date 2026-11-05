"""ed19 task progress after delete ACK, before any replacement create.

Real publisher/session/guard/writer composition over bounded offline storage and
controlled provider receipts. This is not SQL, native playback or task execution.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import unittest
from unittest.mock import AsyncMock, patch

from app.attention.native_voice import NativeAttentionSink
from app.knowledge.models import SourceError
from app.live_state.task_context import TaskSnapshotProgress
from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.coordinator import CoordinationPhase
import test_task_progress_context_recovery as progress
from test_notification_ledger import CanonicalFixture


class EmptySlotProofTests(unittest.IsolatedAsyncioTestCase):
    setUp = progress.TaskProgressGuardTests.setUp
    project = progress.TaskProgressGuardTests.project
    until = progress.TaskProgressGuardTests.until
    update = progress.TaskProgressGuardTests.update

    async def setup_route(self):
        await progress.TaskProgressGuardTests.setup_route(self)
        self.pending = await self.update(130)
        self.owner_current = True

        @asynccontextmanager
        async def replacement(_):
            yield True

        self.adapter._bindings = replace(self.adapter._bindings,
            context_replacement_guard=replacement,
            context_reconciliation_current=lambda: self.owner_current)
        self.old_item = self.adapter._context_item
        self.proofs = []
        publish = self.adapter.publish_context_tracked

        async def capture(update, *, write_evidence):
            receipt = await publish(update, write_evidence=write_evidence)
            self.proofs.append((receipt, write_evidence))
            return receipt

        self.adapter.publish_context_tracked = capture

    def transition_at_delete(self, mutation=None):
        write = self.control._write

        async def changed(event):
            # Preflight passed. The canonical task completes before the owned
            # delete's ACK; its next validation is the replacement create guard.
            if not self.proofs:
                progress.finish_task(self.db, self.identity, self.now)
                if mutation is not None:
                    mutation()
            return await write(event)

        self.control._write = changed

    async def settle(self):
        self.owner.offer_context(self.pending)
        await self.until(lambda: self.bridge.status.fenced or self.bridge.awaiting_fresh_context)
        await self.until(lambda: bool(self.proofs) or self.bridge.status.fenced)
        # Allow the bridge to consume the sealed adapter receipt, not just the
        # adapter's conservative reconciliation latch.
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_task_progress_awaiting_snapshot')

    async def test_revision130_exact_ack_no_create_is_recoverable_not_unwritten(self):
        await self.setup_route()
        self.transition_at_delete()
        await self.settle()
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        receipt, proof = self.proofs[-1]
        self.assertEqual((receipt.status, receipt.reason), ('rejected', 'context_task_progress'))
        self.assertTrue(proof.confirmed_deleted_without_create)
        self.assertFalse(proof.proven_unwritten)
        self.assertTrue(proof.work_joined)
        self.assertFalse(proof.replacement_create_possible)
        self.assertIsNone(self.adapter._context_item)
        self.assertNotIn(self.old_item, self.adapter._items)
        self.assertTrue(self.adapter.context_reconciliation_pending)
        for stale in (self.installed, self.pending):
            with self.assertRaises(TaskSnapshotProgress):
                async with self.binder.guard(stale.guard_binding):
                    pass
        self.assertFalse(await self.bridge.wait_for_fresh_context(.01))
        fresh = await self.update(131)
        self.owner.offer_context(fresh)
        self.assertTrue(await self.bridge.wait_for_fresh_context(1))
        self.assertFalse(self.adapter.context_reconciliation_pending)
        self.assertEqual([row['type'] for row in self.control.writes],
                         ['conversation.item.create', 'conversation.item.delete', 'conversation.item.create'])
        self.owner.offer_context(fresh)
        await asyncio.sleep(.02)
        self.assertEqual(len(self.control.writes), 3)

    async def test_missing_foreign_ack_unjoined_and_replaced_ownership_stay_fenced(self):
        for boundary in ('missing_ack', 'foreign_ack', 'unjoined', 'owner', 'source_epoch'):
            with self.subTest(boundary=boundary):
                await self.setup_route()
                if boundary == 'missing_ack':
                    self.control.item_event = None
                elif boundary == 'foreign_ack':
                    self.control.item_event = lambda event: self.adapter.observe_item_event(
                        self.adapter.identity, {**event, 'item_id': 'not-the-deleted-item'})
                mutation = None
                if boundary == 'owner':
                    mutation = lambda: setattr(self, 'owner_current', False)
                elif boundary == 'source_epoch':
                    mutation = lambda: setattr(self.adapter, '_source_epoch', self.adapter._source_epoch + 1)
                self.transition_at_delete(mutation)
                with patch.object(ContextWriteEvidence, 'mark_joined',
                                  (lambda _: None) if boundary == 'unjoined' else ContextWriteEvidence.mark_joined):
                    await self.settle()
                self.assertTrue(self.bridge.status.fenced)
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertEqual(len([x for x in self.control.writes if x['type'] == 'conversation.item.create']), 1)
                await self.owner.close()

    async def test_same_code_without_verified_progress_and_possible_create_are_unknown(self):
        for possible_create in (False, True):
            with self.subTest(possible_create=possible_create):
                await self.setup_route()
                guard = self.adapter._bindings.source_guard

                @asynccontextmanager
                async def unverified(update):
                    if self.adapter._context_item is None and not possible_create:
                        raise SourceError('live_task_snapshot_changed', 'Not verified task progress')
                    async with guard(update):
                        yield

                self.adapter._bindings = replace(self.adapter._bindings, source_guard=unverified)
                if possible_create:
                    write = self.control._write

                    async def unknown_after_create(event):
                        result = await write(event)
                        if self.control.writes[-1]['type'] == 'conversation.item.create':
                            raise TaskSnapshotProgress()
                        return result

                    self.control._write = unknown_after_create
                await self.settle()
                self.assertTrue(self.bridge.status.fenced)
                receipt, proof = self.proofs[-1]
                self.assertEqual(receipt.status, 'unknown')
                self.assertEqual(proof.replacement_create_possible, possible_create)
                await self.owner.close()


class EmptySlotSessionTests(unittest.IsolatedAsyncioTestCase):
    cleanup = progress.TaskProgressSessionTests.cleanup
    until = progress.TaskProgressSessionTests.until
    rows = progress.TaskProgressSessionTests.rows
    mutations = progress.TaskProgressSessionTests.mutations
    creates = progress.TaskProgressSessionTests.creates
    observation = progress.TaskProgressSessionTests.observation
    expire_inside_adapter_guard = progress.TaskProgressSessionTests.expire_inside_adapter_guard
    followup = progress.TaskProgressSessionTests.followup
    join = progress.TaskProgressSessionTests.join
    hold_normal_scope = progress.TaskProgressSessionTests.hold_normal_scope

    async def asyncSetUp(self):
        # Native task/request DTO versions, not invented task revision fields.
        original = progress.task_fixture.Database.task

        def native_shape(db, **kwargs):
            identity = original(db, **kwargs)
            row = db.tasks[identity]
            row.update(revision=1, request_revision=2, fence=1, attempt_count=1)
            db.events[row['request_event_id']]['revision'] = 2
            return identity

        with patch.object(progress.task_fixture.Database, 'task', native_shape):
            await progress.TaskProgressSessionTests.asyncSetUp(self)
        self.bridge = self.runtime.coordinator._bridge
        self.return_states = []
        publish = self.runtime.adapter.publish_context_tracked

        async def capture(update, *, write_evidence):
            receipt = await publish(update, write_evidence=write_evidence)
            self.return_states.append((self.runtime.ready, receipt, write_evidence))
            return receipt

        self.runtime.adapter.publish_context_tracked = capture

    async def invalidate_after_delete(self, mutation=None):
        send = self.provider.send
        self.transitioned = False

        async def transition(raw):
            import json
            event = json.loads(raw)
            if event['type'] == 'conversation.item.delete' and not self.transitioned:
                self.transitioned = True
                progress.finish_task(self.task_db, self.task_id, self.now)
                if mutation is not None:
                    mutation()
            await send(raw)  # Exact controlled ACK through the real receive loop.

        self.provider.send = transition
        self.task_db.tasks[self.task_id]['last_event_sequence'] += 1
        await self.observation('pending-running-before-terminal-transition')
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_task_progress_awaiting_snapshot')
        self.assertTrue(self.transitioned)

    def assert_empty_hold(self):
        self.assertFalse(self.bridge.status.fenced)
        self.assertFalse(self.session._closed.is_set())
        self.assertFalse(self.runtime.ready)
        self.assertIsNone(self.runtime.adapter._context_item)
        ready, receipt, proof = self.return_states[-1]
        self.assertFalse(ready, 'No false-ready window between adapter return and bridge receipt consumption')
        self.assertEqual(receipt.reason, 'context_task_progress')
        self.assertTrue(proof.confirmed_deleted_without_create)
        self.assertTrue(proof.work_joined)
        self.assertFalse(proof.replacement_create_possible)
        self.assertEqual([row['type'] for row in self.mutations()],
                         ['conversation.item.create', 'conversation.item.delete'])

    async def test_real_publisher_owned_last_continuation_resumes_once(self):
        await self.invalidate_after_delete()
        self.assert_empty_hold()
        pending = self.followup()
        response = asyncio.create_task(self.session._prepare_and_send_response(pending, None, None))
        self.addAsyncCleanup(self.join, response)
        self.assertTrue(await asyncio.wait_for(response, 2))
        self.assertFalse(self.runtime.ready)
        self.assertTrue(pending.submitted)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.creates()[0]['event_id'], pending.request_id)
        self.assertEqual(self.creates()[0]['response']['instructions'].count('original-status-call'), 1)
        self.assertEqual([row['type'] for row in self.mutations()],
                         ['conversation.item.create', 'conversation.item.delete'])
        self.assertIn('Current Live State briefing is temporarily unavailable', self.creates()[0]['response']['instructions'])
        await self.observation('unchanged-canonical-result')
        await asyncio.sleep(.12)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.task_db.tasks[self.task_id]['attempt_count'], 1)
        self.assertFalse(self.session._closed.is_set())

    async def test_native_attention_admission_reopens_only_after_fresh_publication(self):
        await self.invalidate_after_delete()
        self.assert_empty_hold()
        # Actual sink + runtime readiness. The canonical claim scheduler below
        # is controlled, so this asserts eligibility only, not SQL/task delivery.
        repository, claim_marker = CanonicalFixture(), object()
        admission = AsyncMock(return_value=claim_marker)
        sink = NativeAttentionSink(self.session, self.runtime, repository, admission, clock=lambda: self.now)
        self.addAsyncCleanup(sink.close)
        options = dict(worker_id='offline-task-result', receipt_seconds=1)
        self.assertIsNone(await sink.try_attention(repository, **options))
        admission.assert_not_awaited()
        await self.observation('fresh-result-for-pending-notification')
        await self.until(lambda: self.runtime.ready)
        self.assertIs(await sink.try_attention(repository, **options), claim_marker)
        admission.assert_awaited_once()
        self.assertEqual(self.creates(), [])

    async def test_missing_fresh_snapshot_and_cancelled_waiter_never_create_response(self):
        await self.invalidate_after_delete()
        self.assert_empty_hold()
        self.assertFalse(await self.bridge.wait_for_fresh_context(.02))
        pending = self.followup()
        entered, release = self.hold_normal_scope()
        response = asyncio.create_task(self.session._prepare_and_send_response(pending, None, None))
        self.addAsyncCleanup(self.join, response)
        await asyncio.wait_for(entered.wait(), 2)
        response.cancel()
        await asyncio.gather(response, return_exceptions=True)
        release.set()
        await self.observation('fresh-but-caller-cancelled')
        await self.until(lambda: self.runtime.ready)
        self.assertEqual(self.creates(), [])
        self.assertFalse(pending.submitted)

    async def test_expired_installed_context_does_not_restore_readiness(self):
        await self.invalidate_after_delete()
        self.assert_empty_hold()
        self.now += timedelta(seconds=61)
        self.assertFalse(await self.runtime.wait_for_context_ready(.02))
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.creates(), [])
        self.assertEqual(len(self.mutations()), 2)

    async def test_correction_cancellation_replaced_attempt_and_withdrawal_after_ack_stay_fenced(self):
        # Each subcase gets a separate real composed session and source graph.
        # Other cases are separate test methods below to avoid sharing closed state.
        await self.invalidate_after_delete(lambda: self.task_db.tasks[self.task_id].update(revision=2))
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_cancelled_task_after_ack_is_not_ordinary_progress(self):
        await self.invalidate_after_delete(lambda: self.task_db.tasks[self.task_id].update(status='cancelled'))
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_replaced_attempt_after_ack_is_not_ordinary_progress(self):
        from uuid import uuid4
        await self.invalidate_after_delete(lambda: self.task_db.receipts[self.task_id].update(result_attempt_id=uuid4()))
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_request_withdrawal_after_ack_is_not_ordinary_progress(self):
        row = self.task_db.tasks[self.task_id]
        await self.invalidate_after_delete(lambda: self.task_db.events[row['request_event_id']].update(is_current=False))
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])
