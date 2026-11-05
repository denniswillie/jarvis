"""Canonical task controls plus real voice context composition, offline only.

The SQL execution boundary is an in-memory fixture. TaskRepository.mutate,
request authorization, revision/fence checks, stale completion, task projection,
source guards, publisher, adapter and session writer are production code.
This is not SQL, worker cancellation, or native playback qualification.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.voice_routes.contracts import ContextWriteEvidence
from test_task_progress_context_recovery import RealtimeToolFollowup, RealtimeToolReceipt, RealtimeToolRequest
import test_task_progress_context_recovery as progress
import test_task_progress_empty_slot_recovery as empty


class ControlDatabase(progress.task_fixture.Database):
    def __init__(self, scope):
        super().__init__(scope)
        self.now = progress.task_fixture.NOW
        self.requests, self.attempts, self.controls, self.actions = {}, {}, [], []

    @asynccontextmanager
    async def acquire(self):
        yield ControlConnection(self)


class ControlConnection(progress.task_fixture.Connection):
    async def execute(self, query, *args):
        sql = ' '.join(query.lower().split())
        db = self.db
        if sql.startswith('update jarvis_background_tasks set revision=revision+1'):
            owner, identity, status, pending, description, request, revision, watermark, reason = args
            row = db.tasks[identity]
            assert row['owner_id'] == owner and self.active
            row.update(revision=row['revision'] + 1, fence=row['fence'] + 1, status=status,
                pending_status=pending, description_payload_id=description, request_event_id=request,
                request_revision=revision, context_watermark=watermark, result_payload_id=None,
                reason=reason, updated_at=db.now)
            return 'UPDATE 1'
        if sql.startswith('update jarvis_task_attempts set lease_until=clock_timestamp()'):
            db.attempts[args[1]]['lease_valid'] = False
            return 'UPDATE 1'
        if sql.startswith("update jarvis_task_actions set status='revoked'"):
            for row in db.actions:
                if (row['owner_id'], row['task_id']) == args and row['status'] in ('prepared', 'authorized'):
                    row['status'] = 'revoked'
            return 'UPDATE 1'
        if sql.startswith('insert into jarvis_task_session_context'):
            return 'INSERT 1'
        if sql.startswith('insert into jarvis_task_requests'):
            owner, request, task, operation, digest, revision = args
            assert (owner, request) not in db.requests
            db.requests[owner, request] = dict(owner_id=owner, request_id=request, task_id=task,
                operation=operation, parameters_digest=digest, accepted_revision=revision)
            return 'INSERT 1'
        if sql.startswith('insert into jarvis_task_events'):
            owner, task, sequence, revision, attempt, kind, payload = args
            db.controls.append(dict(id=uuid4(), owner_id=owner, task_id=task, sequence=sequence,
                task_revision=revision, attempt_id=attempt, event_type=kind, payload_id=payload))
            return 'INSERT 1'
        if sql.startswith('update jarvis_background_tasks set active_seconds='):
            db.tasks[args[1]]['active_seconds'] += args[2]
            return 'UPDATE 1'
        if sql.startswith('update jarvis_task_reservations set status='):
            return 'UPDATE 0'
        if sql.startswith("update jarvis_task_attempts set status=$3,provider_state='terminal'"):
            db.attempts[args[1]].update(status=args[2], provider_state='terminal')
            return 'UPDATE 1'
        if sql.startswith('update jarvis_background_tasks set status=$3,reason=$4'):
            db.tasks[args[1]].update(status=args[2], reason=args[3], updated_at=db.now)
            return 'UPDATE 1'
        if sql.startswith('update jarvis_background_tasks set current_attempt_id=null'):
            db.tasks[args[1]].update(current_attempt_id=None, pending_status=None, result_payload_id=args[2])
            return 'UPDATE 1'
        return await super().execute(query, *args)

    async def fetchval(self, query, *args):
        sql = ' '.join(query.lower().split())
        db = self.db
        if 'task_control_refresh' in sql:
            owner, task, revision, source = args
            return any(r['owner_id'] == owner and r['task_id'] == task and r['accepted_revision'] == revision
                and r['operation'] in ('cancel', 'update') and any(
                    e['owner_id'] == owner and e['task_id'] == task and e['task_revision'] == revision
                    and e['event_type'] == r['operation'] + '_requested'
                    and source in db.parents.get(e['payload_id'], ())
                    and db.available(owner, e['payload_id']) for e in db.controls)
                for r in db.requests.values())
        if sql.startswith('update jarvis_background_tasks set last_event_sequence='):
            row = db.tasks[args[1]]
            row['last_event_sequence'] += 1
            row['updated_at'] = db.now
            return row['last_event_sequence']
        if sql.startswith('select exists(select 1 from jarvis_task_actions'):
            states = ('prepared', 'authorized') if "'prepared'" in sql else ('submitting', 'submitted', 'outcome_unknown')
            return any((x['owner_id'], x['task_id']) == args and x['status'] in states for x in db.actions)
        if sql.startswith('select exists(select 1 from jarvis_task_attempts'):
            row = db.attempts.get(args[2])
            return row is not None and (row['owner_id'], row['task_id']) == args[:2]
        if sql.startswith('update jarvis_task_attempts set charged_until='):
            return .01
        if sql.startswith('select count(*) from jarvis_background_tasks'):
            return 0
        return await super().fetchval(query, *args)

    async def fetchrow(self, query, *args):
        sql = ' '.join(query.lower().split())
        db = self.db
        if sql.startswith('select * from jarvis_task_requests'):
            return deepcopy(db.requests.get(tuple(args)))
        if sql.startswith('select id,revision,sequence,session_id,payload_id from jarvis_context_events'):
            row = db.events.get(args[1])
            return deepcopy(row) if row and (row['owner_id'], row['revision'], row['session_id']) == (args[0], args[2], args[3]) else None
        if 'lease_until>clock_timestamp() as lease_valid' in sql:
            row = db.attempts.get(args[1])
            return deepcopy(row) if row and row['owner_id'] == args[0] else None
        if 'from jarvis_task_reservations' in sql:
            return dict(input=0, output=0, steps=0)
        return await super().fetchrow(query, *args)


class TaskControlContextTests(unittest.IsolatedAsyncioTestCase):
    cleanup = empty.EmptySlotSessionTests.cleanup
    until = empty.EmptySlotSessionTests.until
    rows = empty.EmptySlotSessionTests.rows
    mutations = empty.EmptySlotSessionTests.mutations
    creates = empty.EmptySlotSessionTests.creates
    observation = empty.EmptySlotSessionTests.observation
    expire_inside_adapter_guard = empty.EmptySlotSessionTests.expire_inside_adapter_guard
    join = empty.EmptySlotSessionTests.join

    async def asyncSetUp(self):
        with patch.object(progress.task_fixture, 'Database', ControlDatabase):
            await empty.EmptySlotSessionTests.asyncSetUp(self)
        self.task_db.now = self.now
        self.repository_tasks = TaskRepository(self.task_db, TaskLimits())
        self.original_claim = await self.claim_fixture(self.task_id)
        self.unrelated_id = self.task_db.task(area=self.scope.area_id)
        self.task_db.tasks[self.unrelated_id]['current_attempt_id'] = uuid4()
        self.unrelated_claim = await self.claim_fixture(self.unrelated_id)
        self.old_binding = self.runtime.adapter._context.guard_binding

    async def claim_fixture(self, identity):
        task = await self.repository_tasks.get(self.scope.owner_id, identity)
        attempt = dict(id=task.current_attempt_id, owner_id=task.owner_id, task_id=task.id,
            task_revision=task.revision, fence=task.fence, status='running', lease_valid=True,
            lease_until=self.now + timedelta(seconds=30), charged_until=self.now,
            provider_state='live', provider_handle='controlled-existing-attempt', checkpoint_payload_id=None)
        self.task_db.attempts[attempt['id']] = attempt
        return self.repository_tasks._claim(task, attempt)

    def request_fixture(self, operation, *, finality='final'):
        original = self.task_db.events[self.task_db.tasks[self.task_id]['request_event_id']]
        identity, payload = uuid4(), self.task_db.payload({'text':
            'Cancel the workbench plan task; tell me its actual cancellation state.' if operation == 'cancel'
            else 'Replace the workbench objective with a compact reading desk, keep reasoning only.'})
        record = progress.task_fixture.ContextRecord.model_validate({**original,
            'id': identity, 'sequence': 20 + len(self.task_db.events), 'origin_sequence': 20 + len(self.task_db.events),
            'source_namespace': 'voice:offline-control', 'source_event_id': str(uuid4()),
            'revision': 1, 'supersedes_event_id': None, 'finality': finality,
            'source_time': self.now, 'recorded_at': self.now, 'payload_id': payload,
            'content': self.task_db.payloads[payload]['content'], 'content_digest': 'b' * 64, 'is_current': True})
        self.task_db.events[identity] = record.model_dump()
        return TaskVoiceSource(identity, record.revision, self.scope.session_id)

    async def control(self, operation, *, source=None):
        self.control_source = source or self.request_fixture(operation)
        self.control_request = TaskMutation(request_id=uuid4(),
            expected_revision=self.task_db.tasks[self.task_id]['revision'],
            message='' if operation == 'cancel' else 'Design a compact reading desk. Reason only; no external actions.')
        self.control_operation = operation
        self.control_result = await self.repository_tasks.mutate(self.scope.owner_id, self.task_id, operation,
            self.control_request, source=self.control_source)
        return self.control_result

    async def invalidate(self, operation, boundary, *, after_control=None):
        if boundary == 'preflight':
            original = self.bridge._validity
            entered, release = asyncio.Event(), asyncio.Event()

            async def validity(update):
                entered.set()
                await release.wait()
                return await original(update)

            self.bridge._validity = validity
        else:
            send = self.provider.send
            self.changed = False

            async def change(raw):
                if json.loads(raw)['type'] == 'conversation.item.delete' and not self.changed:
                    self.changed = True
                    await self.control(operation)
                    if after_control:
                        after_control()
                await send(raw)

            self.provider.send = change
        async with self.repository_tasks.transaction(self.scope.owner_id) as connection:
            task = await self.repository_tasks.load(connection, self.scope.owner_id, self.task_id)
            await self.repository_tasks.emit(connection, task, 'checkpoint', payload={'stage': 'controlled-before-user-control'})
        await self.observation('pending-old-task-context')
        if boundary == 'preflight':
            await asyncio.wait_for(entered.wait(), 2)
            await self.control(operation)
            if after_control:
                after_control()
            release.set()
        await self.until(lambda: self.bridge.status.fenced or
                         self.bridge.status.reason == 'context_task_control_awaiting_snapshot')
        if boundary == 'preflight':
            self.bridge._validity = original

    async def assert_recovery(self, operation, boundary):
        await self.invalidate(operation, boundary)
        expected = TaskStatus.CANCEL_REQUESTED if operation == 'cancel' else TaskStatus.PAUSED
        self.assertEqual(self.control_result.status, expected)
        self.assertEqual(self.control_result.pending_status, 'cancelled' if operation == 'cancel' else 'queued')
        self.assertEqual((self.control_result.revision, self.control_result.fence), (2, 2))
        self.assertFalse(self.bridge.status.fenced, self.bridge.status.reason)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.session._closed.is_set())
        with self.assertRaises(SourceError):
            async with self.live.binder.guard(self.old_binding):
                pass
        if boundary == 'after_ack':
            ready, receipt, proof = self.return_states[-1]
            self.assertFalse(ready)
            self.assertIsNone(self.runtime.adapter._context_item)
            self.assertTrue(proof.confirmed_deleted_without_create)
            self.assertFalse(proof.replacement_create_possible)
            self.assertEqual(receipt.reason, 'context_task_control_changed')
        # The actual old claim is fenced by the real repository; late outcomes
        # may be audited, but cannot become the current result or restart work.
        self.assertFalse(await self.repository_tasks.finish(self.original_claim,
            WorkerOutcome(TaskStatus.SUCCEEDED, {'answer': 'OBSOLETE WORKBENCH RESULT'})))
        self.assertIsNone((await self.repository_tasks.get(self.scope.owner_id, self.task_id)).result_payload_id)
        replay = await self.repository_tasks.mutate(self.scope.owner_id, self.task_id, operation,
            self.control_request, source=self.control_source)
        self.assertEqual(replay.revision, 2)
        self.assertEqual(len(self.task_db.requests), 1)
        self.assertEqual(sum(x['event_type'] == operation + '_requested' for x in self.task_db.controls), 1)
        # Unrelated work genuinely settles through the real finish path, rather
        # than merely asserting its old row is unchanged. Storage is controlled.
        self.assertTrue(await self.repository_tasks.finish(self.unrelated_claim,
            WorkerOutcome(TaskStatus.SUCCEEDED, {'answer': 'Independent task completed'})))
        unrelated = await self.repository_tasks.get(self.scope.owner_id, self.unrelated_id)
        self.assertEqual(unrelated.status, TaskStatus.SUCCEEDED)
        self.assertEqual(self.task_db.payloads[unrelated.result_payload_id]['content']['answer'], 'Independent task completed')
        reservation = self.session._reserve_response_locked('normal_user_turn', {})
        record = self.task_db.events[self.control_source.event_id]
        event = next(x for x in self.task_db.controls if x['event_type'] == operation + '_requested')
        reservation.tool_followup = RealtimeToolFollowup(
            RealtimeToolRequest(self.scope.owner_id, self.scope.session_id, reservation.turn_generation,
                record['id'], record['payload_id'], record['revision'], 'explicit-control-input'),
            (RealtimeToolReceipt('explicit-control-call', 'cancel_background_task' if operation == 'cancel'
                                 else 'update_background_task', 'success', event_id=event['id'], payload_id=event['payload_id']),))
        self.session._source_context.observe((record['payload_id'], event['payload_id']))
        preparing = asyncio.create_task(self.session._prepare_and_send_response(reservation, None, None))
        self.addAsyncCleanup(self.join, preparing)
        self.assertTrue(await asyncio.wait_for(preparing, 2))
        self.assertFalse(self.runtime.ready)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.creates()[0]['event_id'], reservation.request_id)
        self.assertIsNone(self.runtime.adapter._context_item)
        instructions = self.creates()[0]['response']['instructions']
        self.assertIn('Current Live State briefing is temporarily unavailable', instructions)
        self.assertEqual(instructions.count('explicit-control-call'), 1)
        self.assertNotIn('OBSOLETE WORKBENCH RESULT', instructions)
        await asyncio.sleep(.1)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(len(self.task_db.attempts), 2)

    async def test_explicit_cancel_preflight_keeps_fresh_conversation_available(self):
        await self.assert_recovery('cancel', 'preflight')

    async def test_explicit_cancel_after_ack_keeps_fresh_conversation_available(self):
        await self.assert_recovery('cancel', 'after_ack')

    async def test_explicit_replacement_preflight_fences_old_result_and_preserves_reply(self):
        await self.assert_recovery('update', 'preflight')

    async def test_explicit_replacement_after_ack_fences_old_result_and_preserves_reply(self):
        await self.assert_recovery('update', 'after_ack')

    async def test_missing_committed_control_proof_is_not_recoverable(self):
        await self.invalidate('cancel', 'after_ack', after_control=lambda: self.task_db.requests.clear())
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_actual_original_source_withdrawal_still_fences(self):
        original = self.task_db.tasks[self.task_id]['request_event_id']
        await self.invalidate('update', 'after_ack',
            after_control=lambda: self.task_db.events[original].update(is_current=False))
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_provisional_or_foreign_user_source_cannot_cancel(self):
        source = self.request_fixture('cancel', finality='provisional')
        for rejected in (source, TaskVoiceSource(source.event_id, source.revision, uuid4())):
            with self.subTest(source=rejected):
                with self.assertRaises(TaskError):
                    await self.control('cancel', source=rejected)
        self.assertEqual((await self.repository_tasks.get(self.scope.owner_id, self.task_id)).revision, 1)
        self.assertEqual(self.task_db.requests, {})

    async def test_confirmed_queued_task_cancellation_allows_a_new_ordinary_reply(self):
        # A queued task has no live execution to reconcile. Do not call a
        # running task's cancel_requested receipt confirmed cancellation.
        self.task_id = self.task_db.task(status='queued', area=self.scope.area_id)
        await self.observation('queued-task-before-explicit-cancel')
        await self.until(lambda: self.runtime.ready and str(self.task_id) in self.runtime.adapter._context.text)
        await self.invalidate('cancel', 'after_ack')
        self.assertEqual(self.control_result.status, TaskStatus.CANCELLED)
        self.assertIsNone(self.control_result.pending_status)
        self.assertFalse(self.bridge.status.fenced)
        await self.observation('confirmed-cancellation-context')
        await self.until(lambda: self.runtime.ready)
        self.assertIn('cancelled', self.runtime.adapter._context.text)
        pending = self.session._reserve_response_locked('normal_user_turn', {})
        self.assertTrue(await self.session._prepare_and_send_response(pending, None, None))
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual((await self.repository_tasks.get(self.scope.owner_id, self.task_id)).status, TaskStatus.CANCELLED)

    async def test_new_user_control_fences_old_owned_response_but_new_turn_can_answer(self):
        old = self.session._reserve_response_locked('normal_user_turn', {})
        generation = old.turn_generation
        item = 'explicit-user-control-' + uuid4().hex
        await self.session._handle_openai_event({'type': 'input_audio_buffer.speech_started',
                                                'item_id': item, 'audio_start_ms': 0})
        self.assertGreater(self.session._turn_generation, generation)
        await self.invalidate('update', 'preflight')
        self.assertFalse(self.bridge.status.fenced)
        self.assertFalse(await self.session._prepare_and_send_response(old, None, None))
        self.assertFalse(old.submitted)
        self.assertEqual(self.creates(), [])
        await self.observation('fresh-replacement-for-new-user-turn')
        await self.until(lambda: self.runtime.ready)
        await self.session._handle_openai_event({'type': 'input_audio_buffer.speech_stopped',
                                                'item_id': item, 'audio_end_ms': 100})
        await self.session._handle_openai_event({'type': 'input_audio_buffer.committed', 'item_id': item})
        await self.until(lambda: len(self.creates()) == 1)
        self.assertNotEqual(self.creates()[0]['event_id'], old.request_id)
        self.assertFalse(await self.session._prepare_and_send_response(old, None, None))
        self.assertEqual(len(self.creates()), 1)

    async def test_unacknowledged_delete_with_real_cancel_still_fences(self):
        inject = self.provider.inject

        async def missing_delete_ack(kind, **fields):
            if kind != 'conversation.item.deleted':
                await inject(kind, **fields)

        self.provider.inject = missing_delete_ack
        await self.invalidate('cancel', 'after_ack')
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.control_result.status, TaskStatus.CANCEL_REQUESTED)
        self.assertEqual(self.creates(), [])

    async def test_unjoined_control_replacement_never_recovers(self):
        with patch.object(ContextWriteEvidence, 'mark_joined', lambda _: None):
            await self.invalidate('update', 'after_ack')
        self.assertTrue(self.bridge.status.fenced)
        self.assertEqual(self.creates(), [])

    async def test_written_old_response_late_callback_cannot_resume_after_user_cancel(self):
        old = self.session._reserve_response_locked('normal_user_turn', {})
        self.assertTrue(await self.session._prepare_and_send_response(old, None, None))
        old_response = 'resp_offline_old_task_context'
        metadata = self.creates()[0]['response']['metadata']
        await self.provider.inject('response.created', response={
            'id': old_response, 'status': 'in_progress', 'metadata': metadata})
        await self.until(lambda: old_response in self.session._active_responses)
        item = 'cancel-current-task-' + uuid4().hex
        await self.session._handle_openai_event({'type': 'input_audio_buffer.speech_started',
                                                'item_id': item, 'audio_start_ms': 0})
        # Controlled external cancellation receipts. They establish neither
        # physical playback nor recall of already-forwarded audio.
        await self.provider.inject('response.done', response={
            'id': old_response, 'status': 'cancelled', 'metadata': metadata, 'output': []})
        await self.provider.inject('output_audio_buffer.cleared', response_id=old_response)
        await self.until(lambda: old_response not in self.session._active_responses)
        await self.invalidate('cancel', 'after_ack')
        self.assertFalse(self.bridge.status.fenced)
        started = sum(x['type'] == 'audio.webrtc.started' for x in self.app.events)
        await self.session._handle_openai_event({'type': 'output_audio_buffer.started', 'response_id': old_response})
        self.assertEqual(sum(x['type'] == 'audio.webrtc.started' for x in self.app.events), started)
        await self.observation('fresh-cancel-state-after-old-response-retirement')
        await self.until(lambda: self.runtime.ready)
        await self.session._handle_openai_event({'type': 'output_audio_buffer.started', 'response_id': old_response})
        self.assertEqual(sum(x['type'] == 'audio.webrtc.started' for x in self.app.events), started)
        await self.session._handle_openai_event({'type': 'input_audio_buffer.speech_stopped',
                                                'item_id': item, 'audio_end_ms': 100})
        await self.session._handle_openai_event({'type': 'input_audio_buffer.committed', 'item_id': item})
        await self.until(lambda: len(self.creates()) == 2)
        self.assertNotEqual(self.creates()[1]['event_id'], old.request_id)
        self.assertEqual(sum(x['event_id'] == old.request_id for x in self.creates()), 1)
        self.assertEqual((await self.repository_tasks.get(self.scope.owner_id, self.task_id)).status,
                         TaskStatus.CANCEL_REQUESTED)

    async def test_completed_task_cancel_is_refused_not_false_confirmation_or_result_loss(self):
        self.assertTrue(await self.repository_tasks.finish(self.original_claim,
            WorkerOutcome(TaskStatus.SUCCEEDED, {'answer': 'Completed before cancellation'})))
        before = await self.repository_tasks.get(self.scope.owner_id, self.task_id)
        with self.assertRaises(TaskError) as caught:
            await self.control('cancel')
        self.assertEqual(caught.exception.code, 'task_state_conflict')
        after = await self.repository_tasks.get(self.scope.owner_id, self.task_id)
        self.assertEqual((after.status, after.revision, after.fence, after.result_payload_id),
                         (before.status, before.revision, before.fence, before.result_payload_id))
        self.assertEqual(self.task_db.requests, {})
        self.assertEqual(self.task_db.payloads[after.result_payload_id]['content']['answer'],
                         'Completed before cancellation')

    async def test_cancel_revokes_only_undispatched_actions_and_never_resumes_uncertain_work(self):
        for status in ('prepared', 'authorized', 'submitting'):
            self.task_db.actions.append(dict(owner_id=self.scope.owner_id, task_id=self.task_id,
                                             status=status, identity=uuid4()))
        identities = [row['identity'] for row in self.task_db.actions]
        await self.invalidate('cancel', 'preflight')
        self.assertFalse(self.bridge.status.fenced)
        self.assertEqual([row['status'] for row in self.task_db.actions], ['revoked', 'revoked', 'submitting'])
        with self.assertRaises(TaskError) as caught:
            await self.repository_tasks.mutate(self.scope.owner_id, self.task_id, 'resume',
                TaskMutation(request_id=uuid4(), expected_revision=2), source=self.control_source)
        self.assertEqual(caught.exception.code, 'task_reconciliation_required')
        self.assertEqual([row['identity'] for row in self.task_db.actions], identities)
        self.assertEqual(self.task_db.actions[-1]['status'], 'submitting')
        self.assertEqual(len(self.task_db.requests), 1)
        await self.observation('cancellation-awaiting-reconciliation-not-a-rollback')
        await self.until(lambda: self.runtime.ready)
        self.assertIn('cancel_requested', self.runtime.adapter._context.text)
        self.assertIn('not confirmed', self.runtime.adapter._context.text)
