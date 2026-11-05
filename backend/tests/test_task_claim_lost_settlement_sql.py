"""Opt-in disposable-schema proof of ClaimLost settlement, not native/model work.

All task/claim/context/dispatch changes use canonical repositories. Controlled
dispatch intent represents possible submission, never actual provider I/O.
Dependent checks remain unreached if a prerequisite fails. Each fixture closes
its pool and independently audits its exact schema/public owner rows.
"""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.budgets import TaskBudgetRepository
from app.tasks.context import TaskContextBuilder
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.environ.get('JARVIS_SECOND_BRAIN_TEST_DB_URL'), 'explicit isolated SQL opt-in required')
class ClaimLostSettlementSQL(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def fixture(self, checks):
        directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'])
        receipt = {'format': 'jarvis.claim_lost_sql.v1', 'test': self._testMethodName,
                   'boundary': 'canonical_sql_controlled_dispatch_no_provider',
                   'checks': {name: 'unreached' for name in checks},
                   'storage': 'not_allocated', 'bank': 'not_allocated', 'status': 'running'}
        self.receipt = receipt
        path = directory.parent / (self._testMethodName + '.json')
        try:
            async with source_fixture(os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'], directory,
                                      test_name=self._testMethodName) as fixture:
                receipt.update(run_id=fixture.fixture.run_id, owners=list(fixture.fixture.owners),
                               schemas=list(fixture.fixture.manifest['schemas']))
                self.tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=120, active_seconds=300,
                    max_attempts=4, owner_concurrency=4, global_concurrency=4, shutdown_seconds=1))
                self.contexts = ContextRepository(self.tasks)
                self.builder = TaskContextBuilder(self.contexts)
                self.f = fixture
                yield fixture
            receipt['status'] = 'passed'
        except BaseException as error:
            receipt.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, 'w') as output:
                json.dump(receipt, output, indent=2)
                output.write('\n')

    def passed(self, name):
        self.receipt['checks'][name] = 'passed'

    async def create_claim(self):
        session = await SessionRepository(self.f.pool).create_session(self.f.owner, 'Claim settlement fixture')
        task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), title='Controlled reasoning',
            requested_outcome='Return controlled fixture evidence only; no external action.',
            session_id=session.id, knowledge_area_id=self.f.area), model='controlled-no-model')
        claim = await self.tasks.claim_next('fixture-worker', ('frontier',), owner_id=self.f.owner,
                                           provider_models={'frontier': 'controlled-no-model'})
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, task.id)
        return claim

    async def attach(self, claim):
        source = await self.contexts.append(self.f.owner, ContextIngress(session_id=claim.task.session_id,
            source_namespace='fixture-claim-context', source_event_id=uuid4().hex, receipt_id='initial',
            kind='observation', trust='sensor', finality='final', expected_revision=0,
            payload={'description': 'First controlled observation'}))
        manifest = await self.builder.build(claim, shared_context={})
        async with self.f.pool.acquire() as connection:
            bound = await connection.fetchval('select exists(select 1 from jarvis_task_context_sources '
                'where owner_id=$1 and manifest_id=$2 and event_id=$3 and event_revision=$4)',
                self.f.owner, manifest.id, source.id, source.revision)
        self.assertTrue(bound)
        return source, manifest

    async def invalidate(self, claim, source, manifest):
        revised = await self.contexts.append(self.f.owner, ContextIngress(session_id=claim.task.session_id,
            source_namespace=source.source_namespace, source_event_id=source.source_event_id,
            receipt_id='correction', kind='observation', trust='sensor', finality='revised',
            expected_revision=source.revision, payload={'description': 'Corrected controlled observation'}))
        async with self.f.pool.acquire() as connection:
            status = await connection.fetchval('select status from jarvis_task_context_manifests '
                                               'where owner_id=$1 and id=$2', self.f.owner, manifest.id)
        self.assertEqual(status, 'superseded')
        return revised

    async def attempt(self, claim):
        async with self.f.pool.acquire() as connection:
            return dict(await connection.fetchrow('select * from jarvis_task_attempts where owner_id=$1 and id=$2',
                                                   self.f.owner, claim.attempt_id))

    async def state(self, claim):
        async with self.f.pool.acquire() as connection:
            task = dict(await connection.fetchrow('select * from jarvis_background_tasks where owner_id=$1 and id=$2',
                                                  self.f.owner, claim.task.id))
            attempt = dict(await connection.fetchrow('select * from jarvis_task_attempts where owner_id=$1 and id=$2',
                                                     self.f.owner, claim.attempt_id))
            events = [dict(row) for row in await connection.fetch('select * from jarvis_task_events '
                'where owner_id=$1 and task_id=$2 order by sequence', self.f.owner, claim.task.id)]
        return task, attempt, events

    async def runner_call(self, claim, execute):
        provider = SimpleNamespace(model='controlled-no-model', execute=AsyncMock(side_effect=execute))
        runner = TaskRunner(self.tasks, {'frontier': provider}, owner_id=self.f.owner, worker_id='fixture-worker')
        await asyncio.wait_for(runner._execute(claim), 30)
        provider.execute.assert_awaited_once()
        self.assertFalse([task for task in asyncio.all_tasks()
                          if task.get_coro().__qualname__ == 'TaskRunner._heartbeat'])
        return provider

    async def test_invalid_manifest_settles_and_new_claim_rebuilds_fresh_context(self):
        async with self.fixture(['genuine_claim_manifest', 'revision_invalidates', 'registration_rejected',
                'settled_without_dispatch', 'new_attempt_no_attached_manifest', 'fresh_manifest_sources',
                'controlled_completion', 'old_manifest_remains_unavailable']):
            claim = await self.create_claim()
            source, manifest = await self.attach(claim)
            self.passed('genuine_claim_manifest')
            revised = await self.invalidate(claim, source, manifest)
            self.passed('revision_invalidates')
            rejected = []

            async def rejected_start(context):
                try:
                    await context.tasks.start_runtime(context.claim, 'controlled-never-launched', {'phase': 'starting'},
                                                      host_id='fixture-host')
                except ClaimLost:
                    rejected.append(True)
                    raise
                self.fail('Invalid attached manifest must reject runtime registration')

            await self.runner_call(claim, rejected_start)
            self.assertEqual(rejected, [True])
            self.passed('registration_rejected')
            settled = await self.tasks.get(self.f.owner, claim.task.id)
            old = await self.attempt(claim)
            self.assertEqual(settled.status, TaskStatus.QUEUED)
            self.assertIsNone(settled.current_attempt_id)
            self.assertGreater(settled.fence, claim.fence)
            self.assertEqual(old['status'], 'interrupted')
            self.assertEqual(old['provider_state'], 'not_started')
            self.assertIsNone(old['provider_handle'])
            self.assertIsNone(old['runtime_host_id'])
            self.assertIsNotNone(old['ended_at'])
            self.assertEqual(old['context_manifest_id'], manifest.id)
            self.passed('settled_without_dispatch')
            retry = await self.tasks.claim_next('fixture-worker-new', ('frontier',), owner_id=self.f.owner)
            self.assertIsNotNone(retry)
            self.assertEqual(retry.task.id, claim.task.id)
            self.assertNotEqual(retry.attempt_id, claim.attempt_id)
            self.assertEqual(retry.task_revision, claim.task_revision)
            self.assertIsNone((await self.attempt(retry))['context_manifest_id'])
            self.passed('new_attempt_no_attached_manifest')
            fresh = await self.builder.build(retry, shared_context={})
            self.assertNotEqual(fresh.id, manifest.id)
            async with self.f.pool.acquire() as connection:
                ids = {row['event_id'] for row in await connection.fetch('select event_id from '
                    'jarvis_task_context_sources where owner_id=$1 and manifest_id=$2', self.f.owner, fresh.id)}
            self.assertIn(revised.id, ids)
            self.assertNotIn(source.id, ids)
            self.passed('fresh_manifest_sources')

            async def complete(context):
                await context.tasks.start_runtime(context.claim, 'controlled-local-result', {'phase': 'controlled'},
                                                  host_id='fixture-host')
                return WorkerOutcome(TaskStatus.SUCCEEDED, {'controlled_result': 'No model or side effects'})

            await self.runner_call(retry, complete)
            result = await self.tasks.get(self.f.owner, claim.task.id)
            self.assertEqual(result.status, TaskStatus.SUCCEEDED)
            self.assertIsNotNone(result.result_payload_id)
            self.assertEqual(result.attempt_count, 2)
            self.passed('controlled_completion')
            with self.assertRaises(TaskError) as caught:
                await self.contexts.read_manifest(self.f.owner, claim.task.id, manifest.id)
            self.assertEqual(caught.exception.code, 'context_manifest_unavailable')
            self.assertEqual((await self.attempt(claim))['context_manifest_id'], manifest.id)
            self.passed('old_manifest_remains_unavailable')

    async def test_canonical_correction_and_cancel_make_stale_abandon_noop(self):
        async with self.fixture(['update_current_owner_untouched', 'cancel_current_owner_untouched']):
            for operation in ('update', 'cancel'):
                claim = await self.create_claim()
                await self.attach(claim)
                changed = await self.tasks.mutate(self.f.owner, claim.task.id, operation,
                    TaskMutation(request_id=uuid4(), expected_revision=claim.task_revision,
                                 message='Corrected instructions' if operation == 'update' else 'Cancel this task'))
                self.assertGreater(changed.revision, claim.task_revision)
                self.assertGreater(changed.fence, claim.fence)
                before = await self.state(claim)
                rejected = []

                async def old_start(context):
                    try:
                        await context.tasks.start_runtime(context.claim, 'must-not-bind', {'phase': 'starting'})
                    except ClaimLost:
                        rejected.append(True)
                        raise
                    self.fail('A replaced/cancelled claim must not start')

                await self.runner_call(claim, old_start)
                self.assertEqual(rejected, [True])
                self.assertEqual(await self.state(claim), before)
                self.assertIsNone((await self.tasks.get(self.f.owner, claim.task.id)).result_payload_id)
                self.passed(operation + '_current_owner_untouched')

    async def test_submitted_and_live_unknown_dispatch_are_not_retried(self):
        async with self.fixture(['submitting_preserved_unknown', 'submitting_no_fresh_claim',
                                 'live_preserved_unknown', 'live_no_fresh_claim']):
            for dispatch in ('submitting', 'live'):
                claim = await self.create_claim()
                source, manifest = await self.attach(claim)
                budgets = TaskBudgetRepository(self.tasks)
                reservation = await budgets.reserve(claim, 'controlled-dispatch-intent', output_tokens=1)
                if dispatch == 'live':
                    await self.tasks.start_runtime(claim, 'existing-controlled-handle', {'phase': 'controlled'},
                                                   host_id='fixture-host')
                    await budgets.submit(claim, reservation['id'], parent_provider_handle='existing-controlled-handle')
                else:
                    await budgets.submit(claim, reservation['id'], provider_dispatch=True)
                await self.invalidate(claim, source, manifest)
                rejected = []

                async def old_checkpoint(context):
                    try:
                        await context.tasks.checkpoint(context.claim, {'phase': 'must-not-commit'})
                    except ClaimLost:
                        rejected.append(True)
                        raise
                    self.fail('Invalid context must reject checkpoint')

                await self.runner_call(claim, old_checkpoint)
                self.assertEqual(rejected, [True])
                task, attempt, events = await self.state(claim)
                self.assertEqual(task['status'], 'outcome_unknown')
                self.assertEqual(task['current_attempt_id'], claim.attempt_id)
                self.assertIsNone(task['result_payload_id'])
                self.assertEqual(attempt['status'], 'outcome_unknown')
                self.assertEqual(attempt['provider_state'], dispatch)
                self.assertIsNone(attempt['ended_at'])
                self.assertEqual(attempt['provider_handle'], 'existing-controlled-handle' if dispatch == 'live' else None)
                async with self.f.pool.acquire() as connection:
                    rows = await connection.fetch('select status from jarvis_task_reservations where owner_id=$1 '
                                                  'and task_id=$2', self.f.owner, claim.task.id)
                self.assertEqual([row['status'] for row in rows], ['unknown'])
                self.passed(dispatch + '_preserved_unknown')
                before = await self.state(claim)
                await self.tasks.abandon(claim, reason='duplicate-stale-settlement')
                self.assertEqual(await self.state(claim), before)
                self.assertIsNone(await self.tasks.claim_next('must-not-retry', ('frontier',), owner_id=self.f.owner))
                self.assertEqual(attempt['attempt_number'], 1)
                self.assertEqual(sum(event['event_type'] == 'dispatch_prepared' for event in events), int(dispatch == 'submitting'))
                self.assertEqual(sum(event['event_type'] == 'runtime_started' for event in events), int(dispatch == 'live'))
                self.passed(dispatch + '_no_fresh_claim')
