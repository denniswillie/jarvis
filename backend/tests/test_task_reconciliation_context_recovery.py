"""ef131: reconciliation refreshes canonical context, not an output retry.

Real repository/projector/binder/validity/coordinator/adapter and session writer;
only canonical storage and provider transport are offline controlled fixtures.
No SQL locking, provider, native playback or cancellation acceptance is claimed.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4
import unittest

from app.knowledge.models import SourceError
from app.live_state.task_context import TaskSnapshotProgress
from app.voice_routes.coordinator import CoordinationPhase
import test_live_state_task_context as storage
import test_task_progress_context_recovery as progress


class ReconciliationConnection(storage.Connection):
    async def fetchval(self, sql, *args):
        if 'task_reconciliation_refresh' not in sql:
            if 'task_control_refresh' in sql:
                self.query(sql, args)
                return False  # This fixture has no source-bound cancel/update receipt.
            return await super().fetchval(sql, *args)
        normalized = self.query(sql, args)
        assert self.active
        for predicate in ('m.owner_id=a.owner_id', 'm.task_id=a.task_id',
                          'm.task_revision=a.task_revision', "m.status='active'",
                          'e.owner_id=a.owner_id', 'e.task_revision=a.task_revision',
                          'e.attempt_id=a.id', 'e.sequence=$6', 'a.fence >= $10',
                          'a.task_revision=$4', 'e.payload_id=$8', 'm.id=$9'):
            assert predicate in normalized
        owner, task, attempt, revision, fence, sequence, status, payload, manifest, old_fence = args
        a = self.db.attempts.get(attempt)
        e = self.db.task_events.get((task, sequence))
        if a is None or e is None:
            return False
        m = self.db.manifests.get(a['context_manifest_id'])
        if m is None or (a['owner_id'], a['task_id'], a['task_revision']) != (owner, task, revision):
            return False
        if (m['owner_id'], m['task_id'], m['task_revision'], m['status']) != (owner, task, revision, 'active'):
            return False
        if (e['owner_id'], e['task_id'], e['task_revision'], e['attempt_id']) != (owner, task, revision, attempt):
            return False
        if a['fence'] < old_fence or not (a['fence'] == fence or (
                a['fence'] + 1 == fence and a['status'] == 'outcome_unknown'
                and e['event_type'] == 'outcome_unknown')):
            return False
        if status == 'outcome_unknown':
            return (a['status'] in {'outcome_unknown', 'recovering'}
                    and a['provider_state'] in {'submitting', 'live'}
                    and e['event_type'] in {'outcome_unknown', 'reconciling', 'provider_running', 'provider_unknown'})
        return (status == 'succeeded' and a['status'] == 'succeeded'
                and a['provider_state'] == 'terminal' and e['event_type'] == 'succeeded'
                and e['payload_id'] == payload and a['context_manifest_id'] == manifest)


class ReconciliationDatabase(storage.Database):
    def __init__(self, scope):
        super().__init__(scope)
        self.attempts, self.task_events, self.manifests = {}, {}, {}

    def task(self, **kwargs):
        identity = super().task(**kwargs)
        self.tasks[identity].update(revision=1, fence=1, attempt_count=1)
        return identity

    @asynccontextmanager
    async def acquire(self):
        yield ReconciliationConnection(self)


def reconcile(db, identity, now, stage):
    """Stored rows follow abandon -> claim_recovery -> resolve_recovery shape."""
    row = db.tasks[identity]
    attempt = row['current_attempt_id']
    if attempt not in db.attempts:
        manifest = uuid4()
        db.manifests[manifest] = dict(owner_id=row['owner_id'], task_id=identity,
            task_revision=row['revision'], status='active')
        db.attempts[attempt] = dict(owner_id=row['owner_id'], task_id=identity,
            task_revision=row['revision'], context_manifest_id=manifest,
            fence=row['fence'], status='running', provider_state='live')
    a = db.attempts[attempt]
    if stage == 'unknown':
        # abandon increments the task fence, not the old attempt's fence.
        row.update(status='outcome_unknown', reason='worker_ResultHoldTimeout', fence=row['fence'] + 1,
                   updated_at=now, last_event_sequence=row['last_event_sequence'] + 1)
        a['status'] = 'outcome_unknown'
        kind = 'outcome_unknown'
    elif stage == 'recovering':
        row.update(fence=row['fence'] + 1, last_event_sequence=row['last_event_sequence'] + 1)
        a.update(fence=row['fence'], status='recovering')
        kind = 'reconciling'
    elif stage == 'succeeded':
        progress.finish_task(db, identity, now)
        row['reason'] = None
        db.receipts[identity].update(revision=row['revision'], manifest_id=a['context_manifest_id'])
        a.update(status='succeeded', provider_state='terminal')
        kind = 'succeeded'
    else:
        raise AssertionError(stage)
    db.task_events[(identity, row['last_event_sequence'])] = dict(owner_id=row['owner_id'],
        task_id=identity, task_revision=row['revision'], attempt_id=attempt,
        event_type=kind, payload_id=row['result_payload_id'])


class ReconciliationGuardTests(unittest.IsolatedAsyncioTestCase):
    project = progress.TaskProgressGuardTests.project
    update = progress.TaskProgressGuardTests.update
    until = progress.TaskProgressGuardTests.until

    def setUp(self):
        with patch.object(storage, 'Database', ReconciliationDatabase):
            storage.TaskContextTests.setUp(self)

    async def setup_route(self):
        await progress.TaskProgressGuardTests.setup_route(self)

    def advance(self, stage):
        reconcile(self.db, self.identity, self.now, stage)

    async def hold(self):
        self.owner.offer_context(self.pending)
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_task_progress_awaiting_snapshot')

    async def test_controlled_old_generic_rejection_reproduces_permanent_degradation(self):
        await self.setup_route()
        self.advance('unknown')
        self.advance('recovering')
        # Disable only the new classifier to reproduce the old path, including
        # failed retained-context validation and the real permanent fence.
        with patch('app.live_state.task_context._reconciled_task_progress', AsyncMock(return_value=False)):
            await self.hold()
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.DEGRADED)
        self.assertEqual(self.bridge.status.reason, 'context_validity_rejected')
        self.assertEqual(len(self.control.writes), 1)

    async def test_same_revision_fences_1_2_3_refresh_without_stale_replay(self):
        await self.setup_route()
        for stage in ('unknown', 'recovering', 'succeeded'):
            self.advance(stage)
            for old in (self.installed, self.pending):
                with self.assertRaises(TaskSnapshotProgress):
                    async with self.guard(old):
                        self.fail('Stale task metadata became authorized')
            self.assertFalse(self.db.lock.locked())
        await self.hold()
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual(self.owner.status.phase, CoordinationPhase.ACTIVE)
        self.assertFalse(await self.bridge.wait_for_fresh_context(.01))
        self.assertIsNone(self.bridge.status.retry_update_id)
        self.assertEqual(len(self.control.writes), 1)
        fresh = await self.update(114)
        self.owner.offer_context(fresh)
        self.assertTrue(await self.bridge.wait_for_fresh_context(1))
        self.assertNotEqual(fresh.guard_binding, self.pending.guard_binding)
        self.assertIn('succeeded', fresh.text)
        self.assertEqual(self.db.tasks[self.identity]['revision'], 1)
        self.assertEqual(self.db.tasks[self.identity]['fence'], 3)
        self.assertEqual([x['type'] for x in self.control.writes],
                         ['conversation.item.create', 'conversation.item.delete', 'conversation.item.create'])
        self.assertEqual(self.owner.offer_context(fresh).disposition.value, 'duplicate')
        self.assertEqual(len(self.control.writes), 3)

    async def test_unknown_projection_can_be_fresh_but_never_claims_completion(self):
        await self.setup_route()
        self.advance('unknown')
        await self.hold()
        self.assertFalse(self.bridge.status.fenced)
        fresh = await self.update(114)
        self.owner.offer_context(fresh)
        self.assertTrue(await self.bridge.wait_for_fresh_context(1))
        self.assertIn('outcome_unknown', fresh.text)
        self.assertNotIn('Worker-authored result', fresh.text)
        self.assertIsNone(self.db.tasks[self.identity]['result_payload_id'])

    async def test_missing_or_foreign_reconciliation_proof_never_means_progress(self):
        for fault in ('event_missing', 'event_attempt', 'event_revision', 'event_type',
                      'manifest_missing', 'manifest_owner', 'manifest_revoked',
                      'attempt_owner', 'attempt_revision', 'attempt_fence', 'provider_terminal',
                      'pending_cancel', 'replaced_attempt', 'no_event_advance', 'revision_changed'):
            with self.subTest(fault=fault):
                self.setUp()
                await self.setup_route()
                self.advance('unknown')
                self.advance('recovering')
                row = self.db.tasks[self.identity]
                attempt = self.db.attempts[row['current_attempt_id']]
                event = self.db.task_events[(self.identity, row['last_event_sequence'])]
                manifest = self.db.manifests[attempt['context_manifest_id']]
                if fault == 'event_missing': self.db.task_events.clear()
                elif fault == 'event_attempt': event['attempt_id'] = uuid4()
                elif fault == 'event_revision': event['task_revision'] += 1
                elif fault == 'event_type': event['event_type'] = 'cancel_requested'
                elif fault == 'manifest_missing': self.db.manifests.clear()
                elif fault == 'manifest_owner': manifest['owner_id'] = 'foreign'
                elif fault == 'manifest_revoked': manifest['status'] = 'revoked'
                elif fault == 'attempt_owner': attempt['owner_id'] = 'foreign'
                elif fault == 'attempt_revision': attempt['task_revision'] += 1
                elif fault == 'attempt_fence': attempt['fence'] -= 1
                elif fault == 'provider_terminal': attempt['provider_state'] = 'terminal'
                elif fault == 'pending_cancel': row['pending_status'] = 'cancelled'
                elif fault == 'replaced_attempt': row['current_attempt_id'] = uuid4()
                elif fault == 'no_event_advance': row['last_event_sequence'] -= 2
                elif fault == 'revision_changed': row['revision'] += 1
                with self.assertRaises(SourceError) as caught:
                    async with self.guard(self.pending):
                        self.fail('Unproved reconciliation became valid')
                self.assertNotIsInstance(caught.exception, TaskSnapshotProgress)
                await self.hold()
                self.assertTrue(self.bridge.status.fenced)
                self.assertEqual(len(self.control.writes), 1)
                self.assertFalse(self.db.lock.locked())
                await self.owner.close()

    async def test_withdrawn_source_or_scope_still_fences_even_with_matching_result(self):
        for fault in ('description', 'request', 'owner', 'session', 'area', 'result', 'manifest_payload'):
            with self.subTest(fault=fault):
                self.setUp()
                await self.setup_route()
                for stage in ('unknown', 'recovering', 'succeeded'):
                    self.advance(stage)
                row = self.db.tasks[self.identity]
                if fault == 'description': self.db.remove(row['description_payload_id'])
                elif fault == 'request': self.db.events[row['request_event_id']]['is_current'] = False
                elif fault == 'owner': row['owner_id'] = 'foreign'
                elif fault == 'session': row['session_id'] = uuid4()
                elif fault == 'area': row['knowledge_area_id'] = uuid4()
                elif fault == 'result': self.db.remove(row['result_payload_id'])
                elif fault == 'manifest_payload': self.db.remove(self.db.receipts[self.identity]['manifest_payload_id'])
                await self.hold()
                self.assertTrue(self.bridge.status.fenced)
                self.assertEqual(len(self.control.writes), 1)
                await self.owner.close()

    async def test_unchanged_context_needs_no_reconciliation_query(self):
        await self.setup_route()
        self.db.queries.clear()
        async with self.guard(self.pending):
            pass
        self.assertFalse(any('task_reconciliation_refresh' in sql for sql, _ in self.db.queries))

    async def test_polling_canonical_projection_observes_reconciled_result(self):
        self.identity = self.db.task()
        self.db.tasks[self.identity]['current_attempt_id'] = uuid4()
        live = await storage.TaskRuntimeTests.runtime(self)
        await self.until(lambda: live.worker.refresh_count > 0
                         and live.service.snapshot().readiness == 'ready'
                         and bool(live.service.snapshot().entries))
        selection = progress.select_voice_context(live.service.snapshot(), now=self.now)
        previous = await live.bind(selection)
        for stage in ('unknown', 'recovering', 'succeeded'):
            self.advance(stage)
        with self.assertRaises(TaskSnapshotProgress):
            async with live.guard(previous):
                self.fail('The previous polling snapshot became current')
        await self.until(lambda: any(getattr(e, 'status', None) == 'succeeded'
                                    for e in live.service.snapshot().entries))
        current = progress.select_voice_context(live.service.snapshot(), now=self.now)
        self.assertEqual(current.cursor.epoch, selection.cursor.epoch)
        self.assertGreater(current.cursor.revision, selection.cursor.revision)
        fresh = await live.bind(current)
        self.assertNotEqual(fresh, previous)
        async with live.guard(fresh):
            pass
        self.assertIn('succeeded', current.text)
        self.assertFalse(self.db.lock.locked())
        await live.close()
        self.assertTrue(live.worker._task.done())


class ReconciliationSessionTests(unittest.IsolatedAsyncioTestCase):
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
        with patch.object(storage, 'Database', ReconciliationDatabase):
            await progress.TaskProgressSessionTests.asyncSetUp(self)
        self.bridge = self.runtime.coordinator._bridge

    async def reject_stale_preflight(self):
        original = self.bridge._validity
        entered, release = asyncio.Event(), asyncio.Event()

        async def held(update):
            entered.set()
            await release.wait()
            return await original(update)

        self.bridge._validity = held
        self.task_db.tasks[self.task_id]['last_event_sequence'] += 1
        await self.observation('pending original task')
        await asyncio.wait_for(entered.wait(), 2)
        for stage in ('unknown', 'recovering', 'succeeded'):
            reconcile(self.task_db, self.task_id, self.now, stage)
        release.set()
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_task_progress_awaiting_snapshot')
        self.bridge._validity = original
        self.assertFalse(self.bridge.status.fenced)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.session._closed.is_set())
        self.assertEqual(self.creates(), [])

    async def test_real_owned_followup_writes_once_without_optional_projection(self):
        await self.reject_stale_preflight()
        request = self.followup()
        writing = asyncio.create_task(self.session._prepare_and_send_response(request, None, None))
        self.addAsyncCleanup(self.join, writing)
        self.assertTrue(await asyncio.wait_for(writing, 2))
        self.assertFalse(self.task_db.lock.locked())
        self.assertFalse(self.runtime.ready)
        self.assertIsNone(self.runtime.adapter._context_item)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.creates()[0]['event_id'], request.request_id)
        self.assertTrue(request.submitted)
        await self.observation('same result again')
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.task_db.tasks[self.task_id]['attempt_count'], 1)

    async def test_cancelled_waiter_is_joined_and_never_replayed_on_refresh(self):
        await self.reject_stale_preflight()
        request = self.followup()
        entered, release = self.hold_normal_scope()
        writing = asyncio.create_task(self.session._prepare_and_send_response(request, None, None))
        self.addAsyncCleanup(self.join, writing)
        await asyncio.wait_for(entered.wait(), 2)
        writing.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await writing
        release.set()
        self.assertTrue(writing.done())
        await self.observation('fresh canonical result after caller cancellation')
        await self.until(lambda: self.runtime.ready)
        self.assertEqual(self.creates(), [])
        self.assertFalse(request.submitted)
        self.assertFalse(self.task_db.lock.locked())


if __name__ == '__main__':
    unittest.main()
