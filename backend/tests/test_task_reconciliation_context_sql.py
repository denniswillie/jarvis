"""Isolated PostgreSQL proof for the same-attempt reconciliation EXISTS query.

Use evals.second_brain.run_tests --live-db with a NEW private manifest directory.
Real task transitions, context manifests, payload guards and DB clocks; controlled
runtime registration/result data only, no provider, worker, bank or storage.
"""
from contextlib import AsyncExitStack
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope
from app.live_state.task_context import (
    TaskContextProjector, TaskSnapshotControlChange, TaskSnapshotProgress,
    _read, _reconciled_task_progress, validate_task_dependency,
)
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context import TaskContextBuilder
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import Reconciliation, TaskCreate, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class ProofProbe:
    """Forward production SQL unchanged; optional negative argument substitution."""
    def __init__(self, connection, substitutions=None):
        self.connection = connection
        self.substitutions = substitutions or {}
        self.calls = 0
        self.results = []

    def __getattr__(self, name):
        return getattr(self.connection, name)

    async def fetchval(self, query, *arguments):
        targeted = '/* task_reconciliation_refresh */' in query
        if targeted:
            self.calls += 1
            arguments = tuple(self.substitutions.get(index, value)
                              for index, value in enumerate(arguments))
        result = await self.connection.fetchval(query, *arguments)
        if targeted:
            self.results.append(result)
        return result


class ProofProbeOfflineTests(unittest.IsolatedAsyncioTestCase):
    async def test_probe_forwards_real_result_and_only_substitutes_target_arguments(self):
        class Connection:
            def __init__(self): self.calls = []
            async def fetchval(self, query, *arguments):
                self.calls.append((query, arguments))
                return False
        connection = Connection()
        probe = ProofProbe(connection, {0: 'foreign'})
        self.assertIs(await probe.fetchval('select unrelated', 'original'), False)
        self.assertIs(await probe.fetchval('/* task_reconciliation_refresh */ select exists(...)',
                                          'original', 7), False)
        self.assertEqual(connection.calls[0][1], ('original',))
        self.assertEqual(connection.calls[1][1], ('foreign', 7))
        self.assertEqual((probe.calls, probe.results), (1, [False]))


@unittest.skipUnless(os.environ.get('JARVIS_SECOND_BRAIN_TEST_DB_URL')
                     and os.environ.get('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'requires existing isolated --live-db wrapper')
class TaskReconciliationContextSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']) / self._testMethodName
        self.evidence = dict(test=self.id(), status='running', checks=[], transitions=[],
            classification='real_SQL_controlled_runtime_metadata_no_provider',
            storage_used=False, hindsight_bank_used=False,
            clock_scope='real PostgreSQL transition/lease clock and real host UTC; no injected time',
            bounds_scope='30s binder/120s claim lease are isolated SQL test bounds, not production voice deadlines')
        self.addAsyncCleanup(self.write_evidence)
        stack = AsyncExitStack()
        self.addAsyncCleanup(stack.aclose)
        self.fixture = await stack.enter_async_context(source_fixture(
            os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'], self.directory, test_name=self.id()))
        self.owner = self.fixture.owner
        self.repo = TaskRepository(self.fixture.pool, TaskLimits(lease_seconds=120, active_seconds=300,
            owner_concurrency=4, global_concurrency=4))
        self.context = ContextRepository(self.repo)
        self.builder = TaskContextBuilder(self.context)
        self.session = await SessionRepository(self.fixture.pool).create_session(self.owner, 'Reconciliation SQL fixture')
        self.other_session = await SessionRepository(self.fixture.pool).create_session(
            self.fixture.other, 'Foreign SQL fixture')
        self.scope = SessionScope(owner_id=self.owner, session_id=self.session.id, area_id=self.fixture.area)
        self.binder = CanonicalContextBinder(self.fixture.sources, self.scope,
            clock=lambda: datetime.now(UTC), timeout_seconds=30)
        self.projector = TaskContextProjector(self.binder, self.repo)
        self.evidence.update(owner_id=self.owner, other_owner_id=self.fixture.other,
            schema=self.fixture.fixture.manifest['schemas'][0], manifest=str(self.fixture.fixture.path))

    async def write_evidence(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        fixture = getattr(self, 'fixture', None)
        if fixture is not None:
            self.evidence.update(cleanup=fixture.fixture.manifest.get('cleanup'),
                                 fixture_status=fixture.fixture.manifest['status'])
        path = self.directory / ('checks-' + uuid4().hex + '.json')
        with path.open('x') as output:
            json.dump(self.evidence, output, indent=2, default=str)
            output.write('\n')

    def plan(self, *names):
        self.evidence['checks'] = [dict(name=name, status='unreached') for name in names]

    def check(self, name, condition, **detail):
        row = next(x for x in self.evidence['checks'] if x['name'] == name)
        row.update(status='passed' if condition else 'failed', **detail)
        self.assertTrue(condition, name)

    async def create_running(self):
        key = uuid4().hex
        source = await self.context.append(self.owner, ContextIngress(session_id=self.session.id,
            source_namespace='fixture:task-reconciliation-sql', source_event_id=key, receipt_id=uuid4().hex,
            kind='user_transcript', trust='user_input', finality='final', expected_revision=0,
            payload={'text': 'Perform a controlled SQL-only task with no external actions.'}))
        task = await self.repo.create(self.owner, TaskCreate(request_id=uuid4(), title='SQL reconciliation fixture',
            requested_outcome='Record controlled fixture data; no external actions.',
            session_id=self.session.id, knowledge_area_id=self.fixture.area), model='controlled-no-provider',
            source=TaskVoiceSource(source.id, source.revision, source.session_id), mode='background')
        claim = await self.repo.claim_next('sql-reconciliation-worker', ('frontier',), owner_id=self.owner,
                                          provider_models={'frontier': 'controlled-no-provider'})
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, task.id)
        manifest = await self.builder.build(claim, shared_context={})
        # Canonical registration only. No provider object or worker is launched.
        await self.repo.start_runtime(claim, 'sql-controlled-existing-handle', {'fixture_only': True},
                                      host_id='sql-reconciliation-fixture')
        await self.record_transition(task.id, 'running')
        return task, claim, source, manifest

    async def record_transition(self, task_id, stage):
        async with self.fixture.pool.acquire() as connection:
            row = await connection.fetchrow('select clock_timestamp() as db_at,id,status,revision,fence,'
                'current_attempt_id,attempt_count,last_event_sequence,result_payload_id '
                'from jarvis_background_tasks where owner_id=$1 and id=$2', self.owner, task_id)
        self.evidence['transitions'].append(dict(row) | {'stage': stage, 'host_observed_at': datetime.now(UTC)})
        return row

    async def snapshot(self, task_id):
        now = datetime.now(UTC)
        entries = await self.projector.project((task_id,), received_at=now, expires_at=now + timedelta(minutes=5))
        self.assertEqual(len(entries), 1)
        async with self.binder._transaction() as connection:
            value = await _read(self.binder, connection, task_id)
        return entries[0], value

    async def reject_as_progress(self, entry):
        async with self.binder._transaction() as connection:
            self.assertTrue(connection.is_in_transaction())
            probe = ProofProbe(connection)
            with self.assertRaises(TaskSnapshotProgress):
                await validate_task_dependency(self.binder, probe, entry.sources[0])
            self.assertEqual((probe.calls, probe.results), (1, [True]),
                             'Must execute the new production PostgreSQL proof, not only a Python predicate')
        with self.assertRaises(TaskSnapshotProgress):
            async with self.binder.guard(entry.guard_binding):
                self.fail('Stale context cannot authorize publication')

    async def proof(self, old_entry, old, fresh, substitutions=None):
        async with self.binder._transaction() as connection:
            probe = ProofProbe(connection, substitutions)
            result = await _reconciled_task_progress(self.binder, probe, old.state, fresh, old_entry.sources[0])
            self.assertEqual(probe.calls, 1)
            self.assertEqual(probe.results, [result])
        return result

    async def reconcile_success(self, claim):
        await self.repo.abandon(claim, reason='worker_ResultHoldTimeout', target=TaskStatus.QUEUED)
        await self.record_transition(claim.task.id, 'outcome_unknown')
        recovery = await self.repo.claim_recovery('sql-reconciliation-recovery', owner_id=self.owner,
                                                 host_id='sql-reconciliation-fixture', backend_kinds=('frontier',))
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.attempt_id, claim.attempt_id)
        await self.record_transition(claim.task.id, 'reconciling')
        await self.repo.resolve_recovery(recovery, Reconciliation('terminal',
            WorkerOutcome(TaskStatus.SUCCEEDED, {'answer': 'Controlled SQL result; no model or external action.'})))
        await self.record_transition(claim.task.id, 'succeeded')
        return recovery

    async def test_real_same_attempt_1_2_3_fences_and_fresh_projection(self):
        self.plan('running_valid', 'unknown_proof', 'unknown_fresh_valid', 'reconciling_proof',
                  'success_proof', 'unknown_to_success_proof', 'success_fresh_valid', 'one_attempt_one_result')
        task, claim, _, _ = await self.create_running()
        old_entry, old = await self.snapshot(task.id)
        async with self.binder.guard(old_entry.guard_binding):
            self.check('running_valid', old.task.status == TaskStatus.RUNNING and old.task.fence == 1)
        await self.repo.abandon(claim, reason='worker_ResultHoldTimeout', target=TaskStatus.QUEUED)
        row = await self.record_transition(task.id, 'outcome_unknown')
        await self.reject_as_progress(old_entry)
        self.check('unknown_proof', row['status'] == 'outcome_unknown' and row['fence'] == 2
                   and row['current_attempt_id'] == claim.attempt_id)
        unknown_entry, unknown = await self.snapshot(task.id)
        async with self.binder.guard(unknown_entry.guard_binding):
            self.check('unknown_fresh_valid', unknown.task.result_payload_id is None
                       and 'outcome_unknown' in unknown_entry.summary)
        recovery = await self.repo.claim_recovery('sql-reconciliation-recovery', owner_id=self.owner,
                                                 host_id='sql-reconciliation-fixture', backend_kinds=('frontier',))
        self.assertIsNotNone(recovery)
        await self.record_transition(task.id, 'reconciling')
        await self.reject_as_progress(old_entry)
        self.check('reconciling_proof', recovery.attempt_id == claim.attempt_id and recovery.fence == 3
                   and recovery.task_revision == claim.task_revision)
        await self.repo.resolve_recovery(recovery, Reconciliation('terminal',
            WorkerOutcome(TaskStatus.SUCCEEDED, {'answer': 'Controlled SQL result; no model or external action.'})))
        await self.record_transition(task.id, 'succeeded')
        await self.reject_as_progress(old_entry)
        self.check('success_proof', True)
        await self.reject_as_progress(unknown_entry)
        self.check('unknown_to_success_proof', True)
        fresh_entry, fresh = await self.snapshot(task.id)
        async with self.binder.guard(fresh_entry.guard_binding):
            self.check('success_fresh_valid', fresh.task.status == TaskStatus.SUCCEEDED
                       and fresh.task.fence == 3 and fresh.task.revision == old.task.revision
                       and fresh_entry.guard_binding != old_entry.guard_binding
                       and fresh.state['receipt']['result_attempt_id'] == str(claim.attempt_id))
        async with self.fixture.pool.acquire() as connection:
            attempts = await connection.fetchval('select count(*) from jarvis_task_attempts where owner_id=$1 and task_id=$2',
                                                   self.owner, task.id)
            results = await connection.fetchval("select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type='succeeded'",
                                                  self.owner, task.id)
        self.check('one_attempt_one_result', attempts == results == fresh.task.attempt_count == 1)
        self.evidence['status'] = 'passed'

    async def test_query_rejects_foreign_identity_version_fence_and_receipt(self):
        names = ('owner', 'task', 'attempt', 'revision', 'fence', 'sequence', 'result', 'manifest')
        self.plan('baseline_proof', *names, 'foreign_source_guard')
        task, claim, _, _ = await self.create_running()
        old_entry, old = await self.snapshot(task.id)
        await self.reconcile_success(claim)
        _, fresh = await self.snapshot(task.id)
        self.check('baseline_proof', await self.proof(old_entry, old, fresh))
        # These probes execute the real EXISTS with wrong arguments, never
        # mutate canonical business state or substitute a fake SQL result.
        bad = {'owner': (0, self.fixture.other), 'task': (1, uuid4()), 'attempt': (2, uuid4()),
               'revision': (3, fresh.task.revision + 1), 'fence': (4, fresh.task.fence + 1),
               'sequence': (5, fresh.sequence + 1), 'result': (7, uuid4()), 'manifest': (8, uuid4())}
        for name, (index, value) in bad.items():
            with self.subTest(name=name):
                self.check(name, await self.proof(old_entry, old, fresh, {index: value}) is False)
        other_scope = SessionScope(owner_id=self.fixture.other, session_id=self.other_session.id)
        other = CanonicalContextBinder(self.fixture.sources, other_scope, clock=lambda: datetime.now(UTC), timeout_seconds=30)
        binding = type(old_entry.guard_binding)(scope=other_scope,
            payload_root_ids=old_entry.guard_binding.payload_root_ids, dependencies=old_entry.guard_binding.dependencies)
        with self.assertRaises(SourceError) as caught:
            async with other.guard(binding):
                self.fail('Foreign owner cannot borrow original source roots')
        self.check('foreign_source_guard', not isinstance(caught.exception, (TaskSnapshotProgress, TaskSnapshotControlChange)),
                   code=caught.exception.code)
        self.evidence['status'] = 'passed'

    async def test_revoked_manifest_corrected_and_deleted_request_stay_hard(self):
        faults = ('revoked_manifest', 'corrected_request', 'deleted_request')
        self.plan(*(fault + suffix for fault in faults for suffix in ('_proof_before', '_hard_rejection', '_fresh_no_old_answer')))
        for fault in faults:
            with self.subTest(fault=fault):
                task, claim, source, manifest = await self.create_running()
                old_entry, old = await self.snapshot(task.id)
                await self.reconcile_success(claim)
                fresh_entry, fresh = await self.snapshot(task.id)
                self.check(fault + '_proof_before', await self.proof(old_entry, old, fresh))
                if fault == 'corrected_request':
                    await self.context.append(self.owner, ContextIngress(session_id=self.session.id,
                        source_namespace=source.source_namespace, source_event_id=source.source_event_id,
                        receipt_id=uuid4().hex, kind='user_transcript', trust='user_input', finality='revised',
                        expected_revision=source.revision, payload={'text': 'Correction: the previous task request is withdrawn.'}))
                else:
                    # Explicit negative row perturbations in this isolated scope,
                    # not a claim to test the application's deletion/permission API.
                    async with self.repo.transaction(self.owner) as connection:
                        if fault == 'revoked_manifest':
                            await connection.execute("update jarvis_task_context_manifests set status='revoked' where owner_id=$1 and id=$2",
                                                     self.owner, manifest.id)
                        else:
                            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                                     self.owner, source.payload_id)
                for entry in (old_entry, fresh_entry):
                    with self.assertRaises(SourceError) as caught:
                        async with self.binder.guard(entry.guard_binding):
                            self.fail('Invalidated evidence cannot authorize publication')
                    self.assertNotIsInstance(caught.exception, (TaskSnapshotProgress, TaskSnapshotControlChange))
                self.check(fault + '_hard_rejection', True, code=caught.exception.code)
                now = datetime.now(UTC)
                projected = await self.projector.project((task.id,), received_at=now, expires_at=now + timedelta(minutes=5))
                self.check(fault + '_fresh_no_old_answer', all('Controlled SQL result' not in entry.summary for entry in projected),
                           projected_entries=len(projected))
        self.evidence['status'] = 'passed'


if __name__ == '__main__':
    unittest.main()
