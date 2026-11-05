"""Known deleted/never-created context recovery, not blind mutation retries."""
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import json
import unittest
from unittest.mock import patch

from app.knowledge.models import SourceError
from app.voice_routes.contracts import ContextWriteEvidence
from app.voice_routes.coordinator import CoordinationPhase
import test_voice_response_context_boundary as fixtures
from test_realtime_voice_route import NOW, make_context


class ConfirmedDeleteExpiryTests(unittest.IsolatedAsyncioTestCase):
    voice = fixtures.VoiceResponseContextBoundaryTests.voice
    route = fixtures.VoiceResponseContextBoundaryTests.route
    until = fixtures.VoiceResponseContextBoundaryTests.until

    async def setup_route(self, *, failure="expiry", enabled=True):
        self.now = NOW
        (*_, self.runtime, self.control, self.adapter, self.owner) = await self.route()
        self.bridge = self.owner._bridge
        self.bridge._clock = self.adapter._clock = lambda: self.now
        self.bridge._expiry_refresh_enabled = enabled
        self.old_item = self.control.writes[0]['item']['id']
        original = self.adapter._bindings.source_guard
        self.candidate_guards = 0

        @asynccontextmanager
        async def guard(update):
            async with original(update) as connection:
                if update.cursor.revision == 1:
                    self.candidate_guards += 1
                    if self.candidate_guards == 1 and failure != 'after_create':
                        # Bind the fault to actual provider-create admission,
                        # not the ordinal of internal authority checks.
                        self.assertIsNone(self.adapter._context_item)
                        self.assertNotIn(self.old_item, self.adapter._items)
                        self.now = NOW + timedelta(minutes=6)
                        if failure == 'withdrawal':
                            raise SourceError('source_withdrawn', 'private source')
                        if failure == 'timeout':
                            raise TimeoutError('private deadline')
                        if failure == 'fenced':
                            self.adapter._source_epoch += 1
                        if failure in ('expiry', 'fenced'):
                            raise SourceError('voice_context_expired', 'private expired source')
                yield connection

        self.adapter._bindings = replace(self.adapter._bindings, source_guard=guard)
        self.candidate = make_context(revision=1, expires_at=NOW + timedelta(minutes=1))
        logger = patch('app.voice_routes.bridge._PUBLICATION_LOGGER.warning')
        self.log = logger.start()
        self.addCleanup(logger.stop)

    def rows(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))['stage'] == stage]

    def writes(self, kind):
        return [row for row in self.control.writes if row['type'] == kind]

    async def settle(self):
        self.owner.offer_context(self.candidate)
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_expired_awaiting_snapshot')

    async def test_confirmed_delete_then_expiry_recovers_with_one_fresh_create_only(self):
        await self.setup_route()
        await self.settle()
        self.assertEqual(self.candidate_guards, 1)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(self.runtime.context_refresh_pending)
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertEqual([row['item_id'] for row in self.writes('conversation.item.delete')], [self.old_item])
        self.assertEqual(len(self.writes('conversation.item.create')), 1)
        row = self.rows('context_refresh_wait')[-1]
        self.assertEqual(row['outcome'], 'expired_after_acknowledged_delete')
        self.assertEqual(row['write_state'], 'write_possible')
        self.assertEqual(row['first_possible_operation'], 'context_delete')
        self.assertTrue(row['deleted_context_acknowledged'])
        self.assertFalse(row['replacement_create_possible'])
        self.assertTrue(row['owned_work_joined'])
        self.assertTrue(all(job.done() for job in self.bridge._jobs))
        self.assertFalse(await self.runtime.wait_for_context_ready(.01))
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        fresh = make_context(revision=2, expires_at=self.now + timedelta(seconds=10))
        self.owner.offer_context(fresh)
        self.assertTrue(await self.runtime.wait_for_context_ready(1))
        self.assertEqual(self.adapter._context, fresh)
        self.assertNotEqual(self.adapter._context_item, self.old_item)
        self.assertEqual(len(self.writes('conversation.item.delete')), 1)
        self.assertEqual(len(self.writes('conversation.item.create')), 2)
        self.assertNotIn('private', str(self.log.call_args_list))

    async def test_expired_factory_return_without_create_also_has_exact_empty_slot(self):
        await self.setup_route(failure=None)
        await self.settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertEqual(self.rows('context_refresh_wait')[-1]['outcome'], 'expired_after_acknowledged_delete')
        self.assertEqual(len(self.writes('conversation.item.create')), 1)

    async def test_missing_or_foreign_delete_ack_never_enters_recovery(self):
        for foreign in (False, True):
            with self.subTest(foreign=foreign):
                await self.setup_route()
                if foreign:
                    self.control.item_event = lambda event: self.adapter.observe_item_event(
                        self.adapter.identity, {**event, 'item_id':'foreign-item'})
                else:
                    self.control.item_event = None
                await self.settle()
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertEqual(self.candidate_guards, 0)
                self.assertEqual(self.rows('context_refresh_wait'), [])
                self.assertFalse(self.rows('adapter_return')[-1]['deleted_context_acknowledged'])
                self.assertEqual(len(self.writes('conversation.item.delete')), 1)
                await self.owner.close()

    async def test_actual_withdrawal_deadline_or_route_fence_are_not_expiry_recovery(self):
        for failure in ('withdrawal', 'timeout', 'fenced'):
            with self.subTest(failure=failure):
                await self.setup_route(failure=failure)
                await self.settle()
                self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
                self.assertEqual(self.rows('context_refresh_wait'), [])
                self.assertEqual(len(self.writes('conversation.item.create')), 1)
                await self.owner.close()

    async def test_possibly_created_replacement_cannot_claim_empty_context(self):
        await self.setup_route(failure='after_create')
        original = self.control._write

        async def uncertain_after_create(event):
            written = await original(event)
            if self.control.writes[-1]['type'] == 'conversation.item.create':
                self.now = NOW + timedelta(minutes=6)
                raise SourceError('voice_context_expired', 'private error after possible create')
            return written

        self.control._write = uncertain_after_create
        await self.settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        row = self.rows('adapter_return')[-1]
        self.assertTrue(row['deleted_context_acknowledged'])
        self.assertTrue(row['replacement_create_possible'])
        self.assertEqual(self.rows('context_refresh_wait'), [])
        self.assertEqual(len(self.writes('conversation.item.create')), 2)

    async def test_missing_owned_join_proof_prevents_recovery(self):
        await self.setup_route()
        with patch.object(ContextWriteEvidence, 'mark_joined', lambda _: None):
            await self.settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        row = self.rows('adapter_return')[-1]
        self.assertTrue(row['deleted_context_acknowledged'])
        self.assertFalse(row['owned_work_joined'])
        self.assertEqual(self.rows('context_refresh_wait'), [])

    async def test_nonoptin_route_keeps_fail_closed_behavior(self):
        await self.setup_route(enabled=False)
        await self.settle()
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual(self.rows('context_refresh_wait'), [])

    def test_delete_proof_is_exact_and_never_whole_attempt_unwritten(self):
        proof = ContextWriteEvidence()
        proof.begin_tracking()
        self.assertTrue(proof.before_write('context_delete', item_id='owned-item'))
        self.assertFalse(proof.record_delete_acknowledged('foreign-item'))
        self.assertFalse(proof.deleted_context_acknowledged)
        self.assertTrue(proof.record_delete_acknowledged('owned-item'))
        self.assertFalse(proof.confirmed_deleted_without_create)
        proof.close()
        self.assertFalse(proof.confirmed_deleted_without_create)
        proof.mark_joined()
        self.assertTrue(proof.confirmed_deleted_without_create)
        self.assertEqual(proof.state, 'write_possible')
        self.assertFalse(proof.proven_unwritten)
        self.assertFalse(proof.before_write('context_create'))

    def test_untracked_unmatched_multiple_and_possibly_created_evidence_is_not_recoverable(self):
        for fault in ('untracked', 'missing_id', 'foreign_ack', 'multiple_delete', 'create', 'late_ack'):
            with self.subTest(fault=fault):
                proof = ContextWriteEvidence()
                if fault != 'untracked': proof.begin_tracking()
                proof.before_write('context_delete', item_id=None if fault=='missing_id' else 'owned-item')
                if fault == 'late_ack': proof.close()
                proof.record_delete_acknowledged('foreign' if fault=='foreign_ack' else 'owned-item')
                if fault == 'multiple_delete': proof.before_write('context_delete', item_id='owned-item')
                if fault == 'create': proof.before_write('context_create')
                proof.close()
                proof.mark_joined()
                self.assertFalse(proof.confirmed_deleted_without_create)
                self.assertFalse(proof.proven_unwritten)


if __name__ == '__main__':
    unittest.main()
