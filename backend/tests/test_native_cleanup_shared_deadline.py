"""Offline cleanup composition tests. No SQL, HTTP, model or native acceptance."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from evals.responsive_voice import frontend_host_qualification as host
from evals.responsive_voice import native_replay_run as runner


RUN = 'a'*32
AREA = UUID('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb')
BANK = 'jarvis_'+'c'*32


def resources():
    return {
        'run_id': RUN, 'owner_id': 'direct-image-eval-second-brain-'+RUN+'-a',
        'area_id': str(AREA), 'database_schema': 'jarvis_sb_eval_'+RUN+'_0',
        'bank_name': BANK, 'hindsight_url': 'http://127.0.0.1:1',
        'hindsight_api_key_file': '/not-read', 'child_pid': None,
        'child_returncode': 0, 'bank_creation_attempted': True,
        'control_file': '/not-written',
    }


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    def budget(self, seconds=1, reserve=.1, persist=lambda: None):
        result = {}
        budget = host._CleanupDeadline(result, persist,
            deadline=asyncio.get_running_loop().time()+seconds, shutdown_seconds=reserve)
        return result, budget

    async def test_default_total_and_reserved_shutdown_are_one_budget(self):
        result = {}
        budget = host._CleanupDeadline(result, lambda: None)
        self.assertAlmostEqual(budget.deadline-budget.started, 180)
        self.assertEqual(budget.deadline-budget.work_deadline, 10)
        self.assertLessEqual(3*host.CLEANUP_CLOSE_SECONDS, host.CLEANUP_SHUTDOWN_SECONDS)
        self.assertEqual(result['timing']['clock'], 'same_process_event_loop_monotonic')

    async def test_invalid_extended_or_nonfinite_deadline_rejected(self):
        for deadline in [float('nan'), float('inf'), True, asyncio.get_running_loop().time()+181]:
            with self.subTest(deadline=deadline), self.assertRaises(host.QualificationFailure):
                host._CleanupDeadline({}, lambda: None, deadline=deadline)

    async def test_exhausted_budget_does_not_construct_io(self):
        result, budget = self.budget(seconds=-1)
        factory = AsyncMock()
        with self.assertRaises(TimeoutError):
            await budget.step('not_started_io', factory)
        factory.assert_not_called()
        self.assertEqual(result['timing']['failure_step']['status'], 'timed_out')

    async def test_started_receipt_precedes_side_effect(self):
        snapshots = []
        result = {}
        budget = host._CleanupDeadline(result, lambda: snapshots.append(json.loads(json.dumps(result))))
        async def action():
            self.assertEqual(snapshots[-1]['timing']['active_step']['status'], 'started')
            return 4
        self.assertEqual(await budget.step('query', action), 4)
        self.assertEqual(result['timing']['steps'][-1]['status'], 'completed')

    async def test_slow_step_is_bounded_by_shared_work_deadline(self):
        result, budget = self.budget(seconds=.04, reserve=.02)
        with self.assertRaises(TimeoutError):
            await budget.step('slow', lambda: asyncio.Event().wait(), limit=30)
        step = result['timing']['failure_step']
        self.assertEqual(step['deadline_monotonic'], budget.work_deadline)
        self.assertEqual(step['status'], 'timed_out')

    async def test_shutdown_uses_reserved_time_not_a_fresh_budget(self):
        result, budget = self.budget(seconds=.08, reserve=.06)
        await asyncio.sleep(.03)
        with self.assertRaises(TimeoutError):
            await budget.step('work', AsyncMock())
        closer = AsyncMock()
        await budget.step('close', closer, shutdown=True)
        closer.assert_awaited_once()
        self.assertEqual(result['timing']['steps'][-1]['deadline_monotonic'], budget.deadline)

    async def test_inflight_cancellation_retains_exact_step(self):
        result, budget = self.budget()
        entered = asyncio.Event()
        async def action():
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(budget.step('source_inspect', action, source_id='exact-source'))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(task.done())
        self.assertEqual(result['timing']['failure_step']['name'], 'source_inspect')
        self.assertEqual(result['timing']['failure_step']['source_id'], 'exact-source')
        self.assertEqual(result['timing']['failure_step']['status'], 'cancelled')

    async def test_cancellation_between_steps_prevents_next_io(self):
        result = {}
        def persist():
            timing = result.get('timing', {})
            if timing.get('steps') and timing['steps'][-1].get('name') == 'first' and timing['steps'][-1]['status'] == 'completed':
                asyncio.current_task().cancel()
        budget = host._CleanupDeadline(result, persist)
        next_io = AsyncMock()
        async def sequence():
            await budget.step('first', AsyncMock())
            await budget.step('second', next_io)
        task = asyncio.create_task(sequence())
        with self.assertRaises(asyncio.CancelledError):
            await task
        next_io.assert_not_called()
        self.assertEqual(result['timing']['failure_step']['name'], 'second')

    async def test_bounded_step_history_retains_total_count(self):
        result, budget = self.budget()
        for index in range(70):
            await budget.step('inspect', AsyncMock(), iteration=index)
        self.assertEqual(len(result['timing']['steps']), 64)
        self.assertEqual(result['timing']['step_count'], 70)
        self.assertEqual(result['timing']['dropped_steps'], 6)

    async def test_error_receipt_does_not_log_exception_payload(self):
        result, budget = self.budget()
        with self.assertRaises(ValueError):
            await budget.step('failure', AsyncMock(side_effect=ValueError('PRIVATE_CONTENT')))
        self.assertNotIn('PRIVATE_CONTENT', json.dumps(result))

    async def test_close_failure_does_not_skip_remaining_owned_joins(self):
        result, budget = self.budget()
        second = AsyncMock()
        with self.assertRaises(ValueError):
            await host._cleanup_close_all(budget,
                [('first_close', AsyncMock(side_effect=ValueError())), ('second_close', second)],
                primary_error=None)
        second.assert_awaited_once()
        self.assertTrue(budget.shutdown_failed)
        self.assertEqual(result['timing']['failure_step']['name'], 'first_close')

    async def test_close_timeout_has_separate_finite_cap_and_preserves_join_evidence(self):
        result, budget = self.budget()
        second = AsyncMock()
        with patch.object(host, 'CLEANUP_CLOSE_SECONDS', .01):
            with self.assertRaises(TimeoutError):
                await host._cleanup_close_all(budget,
                    [('slow_close', lambda: asyncio.Event().wait()), ('next_close', second)],
                    primary_error=None)
        second.assert_awaited_once()
        self.assertTrue(budget.shutdown_failed)
        self.assertEqual(result['timing']['steps'][-1]['status'], 'completed')


class Acquire:
    def __init__(self, connection):
        self.connection = connection
    async def __aenter__(self):
        return self.connection
    async def __aexit__(self, *args):
        return False


class SourceCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.row = {'id': UUID('dddddddd-dddd-4ddd-8ddd-dddddddddddd'),
                    'area_id': AREA, 'current_revision': 1, 'status': 'active'}
        self.connection = SimpleNamespace(fetch=AsyncMock(return_value=[self.row]))
        self.pool = SimpleNamespace(acquire=lambda: Acquire(self.connection), close=AsyncMock())
        self.client = SimpleNamespace(close=AsyncMock())
        self.deletions = SimpleNamespace(request=AsyncMock(),
            inspect=AsyncMock(return_value={'state': 'scoped_cleanup_recorded',
                'source_id': str(self.row['id']), 'revision': 1}))
        self.worker = SimpleNamespace(step=AsyncMock())
        for target, value in [
            ('evals.responsive_voice.frontend_host_qualification.asyncpg.create_pool', AsyncMock(return_value=self.pool)),
            ('evals.responsive_voice.frontend_host_qualification.HindsightClient', Mock(return_value=self.client)),
            ('evals.responsive_voice.frontend_host_qualification.secret_file', Mock(return_value='not-a-secret')),
            ('app.knowledge.sources.SourceRepository', Mock(return_value=SimpleNamespace(pool=self.pool))),
            ('app.knowledge.deletion.SourceDeletionService', Mock(return_value=self.deletions)),
            ('app.knowledge.outbox.OutboxRepository', Mock(return_value=object())),
            ('app.knowledge.ingestion.IngestionWorker', Mock(return_value=self.worker)),
        ]:
            self.stack.enter_context(patch(target, value))
        self.result = {}
        self.budget = host._CleanupDeadline(self.result, lambda: None)

    async def test_source_cleanup_uses_supplied_deadline_and_exact_originals(self):
        result = await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.assertEqual(result['status'], 'scoped_cleanup_recorded')
        self.assertFalse(result['model_submission'])
        self.deletions.request.assert_awaited_once_with(resources()['owner_id'], AREA, self.row['id'], expected_revision=1)
        self.worker.step.assert_not_awaited()
        self.client.close.assert_awaited_once()
        self.pool.close.assert_awaited_once()
        self.assertTrue(all(step['deadline_monotonic'] <= self.budget.deadline
                            for step in self.result['timing']['steps']))

    async def test_empty_source_inventory_still_joins_pool(self):
        self.connection.fetch.return_value = []
        result = await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.assertEqual(result['status'], 'no_sources')
        self.pool.close.assert_awaited_once()
        self.deletions.request.assert_not_awaited()

    async def test_source_scope_and_count_gates_unchanged(self):
        for rows in [[{**self.row, 'area_id': UUID(int=1)}], [self.row]*257]:
            with self.subTest(rows=len(rows)):
                self.connection.fetch.return_value = rows
                with self.assertRaises(host.QualificationFailure):
                    await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.deletions.request.assert_not_awaited()
        self.worker.step.assert_not_awaited()

    async def test_requires_attention_is_not_cleanup_success(self):
        self.deletions.inspect.return_value = {'state': 'requires_attention',
            'source_id': str(self.row['id']), 'revision': 1}
        with self.assertRaises(host.QualificationFailure):
            await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.worker.step.assert_not_awaited()
        self.pool.close.assert_awaited_once()

    async def test_inflight_source_inspect_cancellation_keeps_identity_and_joins(self):
        entered = asyncio.Event()
        async def inspect(*args):
            entered.set()
            await asyncio.Event().wait()
        self.deletions.inspect.side_effect = inspect
        task = asyncio.create_task(host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.client.close.assert_awaited_once()
        self.pool.close.assert_awaited_once()
        failure = self.result['timing']['failure_step']
        self.assertEqual((failure['name'], failure['source_id'], failure['revision']),
                         ('source_inspect', str(self.row['id']), 1))

    async def test_worker_step_timeout_does_not_become_false_pass(self):
        self.deletions.inspect.return_value = {'state': 'pending',
            'source_id': str(self.row['id']), 'revision': 1}
        async def slow(*args):
            await asyncio.Event().wait()
        self.worker.step.side_effect = slow
        self.budget = host._CleanupDeadline(self.result, lambda: None,
            deadline=asyncio.get_running_loop().time()+.05, shutdown_seconds=.02)
        with self.assertRaises(TimeoutError):
            await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.assertEqual(self.result['timing']['failure_step']['name'], 'source_worker_step')
        self.client.close.assert_awaited_once()
        self.pool.close.assert_awaited_once()

    async def test_source_client_close_timeout_still_joins_pool(self):
        async def slow():
            await asyncio.Event().wait()
        self.client.close.side_effect = slow
        with patch.object(host, 'CLEANUP_CLOSE_SECONDS', .01), self.assertRaises(TimeoutError):
            await host.cleanup_source_originals('postgresql://unused', resources(), budget=self.budget)
        self.pool.close.assert_awaited_once()
        self.assertEqual(self.result['timing']['failure_step']['name'], 'source_client_close')


class PreparedCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory/'resources.json'
        self.data = resources()
        self.key = self.directory/'operator.key'
        self.key.write_text('untouched-test-placeholder')
        self.bearer = self.directory/'bearer'
        self.bearer.write_text('private-test-placeholder')
        self.marker = self.directory/'admission.active.json'
        self.marker.write_text('preserve')
        self.data['bearer_token_file'] = str(self.bearer)
        self.path.write_text(json.dumps(self.data))
        self.fixture = SimpleNamespace(run_id=RUN, cleanup=AsyncMock(
            return_value={'remaining_schemas': 0, 'remaining_public_owner_rows': 0}))
        self.connection = SimpleNamespace(
            fetchval=AsyncMock(side_effect=[True, BANK]), execute=AsyncMock(), close=AsyncMock())
        self.storage = AsyncMock(return_value={'status': 'passed', 'remaining_keys': [],
                                                'independent_client_audit': True})
        self.originals = AsyncMock(return_value={'status': 'scoped_cleanup_recorded', 'source_count': 1})
        self.bank = AsyncMock(return_value={'native_bank_absent': True, 'independent_client_audit': True})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for target, value in [
            ('prepared_resources', Mock(side_effect=lambda *args: (self.path, json.loads(self.path.read_text()), self.fixture))),
            ('storage_inventory', AsyncMock(return_value={'objects': []})),
            ('cleanup_owned_storage', self.storage),
            ('cleanup_source_originals', self.originals),
            ('cleanup_owned_native_bank', self.bank),
        ]:
            self.stack.enter_context(patch.object(host, target, value))
        self.connect = self.stack.enter_context(patch.object(host.asyncpg, 'connect', AsyncMock(return_value=self.connection)))

    def reports(self):
        return [(path, json.loads(path.read_text())) for path in sorted(self.directory.glob('frontend-host-cleanup-*.json'))]

    async def test_success_keeps_all_gates_and_reuses_exact_budget(self):
        result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        self.assertEqual(result['status'], 'passed')
        budget = self.originals.await_args.kwargs['budget']
        self.assertIs(self.storage.await_args.kwargs['budget'], budget)
        self.assertIs(self.bank.await_args.kwargs['budget'], budget)
        self.fixture.cleanup.assert_awaited_once()
        self.assertTrue(result['cleanup']['native_bank_absent'])
        self.assertEqual(result['timing']['terminal_status'], 'passed')
        self.assertFalse(self.bearer.exists())
        self.assertTrue(self.key.exists())
        self.assertTrue(self.marker.exists())

    async def test_inflight_cancel_persists_terminal_receipt_and_manifest(self):
        entered = asyncio.Event()
        async def originals(*args, budget):
            async def blocked():
                entered.set()
                await asyncio.Event().wait()
            return await budget.step('source_worker_step', blocked)
        self.originals.side_effect = originals
        task = asyncio.create_task(host.cleanup_prepared_host(manifest=self.path, dsn='unused'))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        report = self.reports()[0][1]
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['failure_code'], 'cleanup_cancelled')
        self.assertEqual(report['timing']['failure_step']['name'], 'source_worker_step')
        self.assertEqual(json.loads(self.path.read_text())['status'], 'cleanup_incomplete')
        self.connection.close.assert_awaited_once()
        self.fixture.cleanup.assert_not_awaited()
        self.assertTrue(self.bearer.exists())
        self.assertTrue(self.marker.exists())

    async def test_exhausted_shared_budget_never_opens_sql(self):
        result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused',
            deadline=asyncio.get_running_loop().time()-1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_code'], 'cleanup_deadline_exhausted')
        self.connect.assert_not_called()
        self.assertTrue(self.marker.exists())

    async def test_each_existing_gate_failure_prevents_pass(self):
        for gate in [self.storage, self.originals, self.bank, self.fixture.cleanup]:
            with self.subTest(gate=[self.storage, self.originals, self.bank, self.fixture.cleanup].index(gate)):
                self.connection.fetchval.side_effect = [True, BANK]
                gate.side_effect = ValueError('not a pass')
                result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(self.bearer.exists())
                self.assertTrue(self.marker.exists())
                gate.side_effect = None

    async def test_later_recovery_is_a_separate_receipt_not_one_shot_success(self):
        self.originals.side_effect = TimeoutError()
        failed = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        failed_path = Path(failed['report_path'])
        before = failed_path.read_bytes()
        self.originals.side_effect = None
        self.connection.fetchval.side_effect = [True, BANK]
        recovered = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(recovered['status'], 'passed')
        self.assertNotEqual(failed['report_path'], recovered['report_path'])
        self.assertEqual(failed_path.read_bytes(), before)

    async def test_exact_canonical_bank_mismatch_keeps_scope_gate(self):
        self.connection.fetchval.side_effect = [True, 'jarvis_'+'e'*32]
        result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        self.assertEqual(result['status'], 'failed')
        self.storage.assert_not_awaited()
        self.originals.assert_not_awaited()
        self.fixture.cleanup.assert_not_awaited()

    async def test_absent_schema_recovery_still_requires_storage_bank_and_sql_audits(self):
        self.connection.fetchval.side_effect = [False]
        result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        self.assertEqual(result['status'], 'passed')
        self.originals.assert_not_awaited()
        self.storage.assert_awaited_once()
        self.bank.assert_awaited_once()
        self.fixture.cleanup.assert_awaited_once()

    async def test_host_boundary_supplies_single_absolute_deadline(self):
        mock = AsyncMock(return_value={'status': 'passed'})
        with patch.object(runner, 'cleanup_prepared_host', mock):
            before = asyncio.get_running_loop().time()
            await runner.HostBoundary('unused').cleanup(self.path)
            after = asyncio.get_running_loop().time()
        self.assertLessEqual(before+180, mock.await_args.kwargs['deadline'])
        self.assertLessEqual(mock.await_args.kwargs['deadline'], after+180)


    async def test_cancel_between_host_steps_stops_before_sql_and_records_terminal(self):
        original_save = host.save_json
        def save(path, value):
            original_save(path, value)
            steps = value.get('timing', {}).get('steps', [])
            if steps and steps[-1]['name'] == 'owned_host_lock' and steps[-1]['status'] == 'completed':
                asyncio.current_task().cancel()
        with patch.object(host, 'save_json', side_effect=save):
            task = asyncio.create_task(host.cleanup_prepared_host(manifest=self.path, dsn='unused'))
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.connect.assert_not_called()
        report = self.reports()[0][1]
        self.assertEqual(report['status'], 'cancelled')
        self.assertEqual(report['timing']['failure_step']['name'], 'cleanup_sql_connect')
        self.assertTrue(self.marker.exists())

    async def test_outer_connection_close_timeout_never_releases_cleanup(self):
        async def slow(*args, **kwargs):
            await asyncio.Event().wait()
        self.connection.close.side_effect = slow
        with patch.object(host, 'CLEANUP_CLOSE_SECONDS', .01):
            result = await host.cleanup_prepared_host(manifest=self.path, dsn='unused')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['timing']['failure_step']['name'], 'cleanup_sql_close')
        self.connection.close.assert_awaited_once()
        self.fixture.cleanup.assert_not_awaited()
        self.assertTrue(self.marker.exists())


class IndependentAuditGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_bank_presence_cannot_pass_and_both_clients_close(self):
        clients = [SimpleNamespace(_call=AsyncMock(), _bank=lambda bank: '/bank/'+bank, close=AsyncMock()),
                   SimpleNamespace(close=AsyncMock())]
        result = {}
        budget = host._CleanupDeadline(result, lambda: None)
        with patch.object(host, 'secret_file', return_value='placeholder'), \
                patch.object(host, 'native_bank_inventory', AsyncMock(return_value=[BANK])):
            with self.assertRaises(host.QualificationFailure):
                await host.cleanup_owned_native_bank(BANK, hindsight_url='http://127.0.0.1:1',
                    api_key_file='/not-read', client_factory=Mock(side_effect=clients), budget=budget)
        for client in clients:
            client.close.assert_awaited_once()
        self.assertEqual([step['name'] for step in result['timing']['steps']],
            ['bank_delete', 'bank_delete_client_close', 'bank_independent_audit', 'bank_audit_client_close'])

    async def test_storage_presence_cannot_pass_independent_audit(self):
        clients = [SimpleNamespace(request=AsyncMock()), SimpleNamespace(request=AsyncMock())]
        managers = [Acquire(client) for client in clients]
        data = resources()
        data['storage'] = {'bucket': 'files', 'prefix': 'owners/'+data['owner_id']+'/sessions/'}
        factory = Mock(side_effect=managers)
        result = {}
        budget = host._CleanupDeadline(result, lambda: None)
        with patch.object(host, 'list_owned_storage', AsyncMock(side_effect=[[], ['remaining-owned-key']])):
            with self.assertRaises(host.QualificationFailure):
                await host.cleanup_owned_storage(data, [], lambda: None,
                    config=object(), client_factory=factory, budget=budget)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(data['storage_cleanup']['status'], 'incomplete')
        self.assertEqual(data['storage_cleanup']['remaining_keys'], ['remaining-owned-key'])

    async def test_bank_audit_timeout_is_not_absence(self):
        clients = [SimpleNamespace(_call=AsyncMock(), _bank=lambda bank: '/bank/'+bank, close=AsyncMock()),
                   SimpleNamespace(close=AsyncMock())]
        result = {}
        budget = host._CleanupDeadline(result, lambda: None,
            deadline=asyncio.get_running_loop().time()+.05, shutdown_seconds=.02)
        async def slow(*args):
            await asyncio.Event().wait()
        with patch.object(host, 'secret_file', return_value='placeholder'), \
                patch.object(host, 'native_bank_inventory', side_effect=slow):
            with self.assertRaises(TimeoutError):
                await host.cleanup_owned_native_bank(BANK, hindsight_url='http://127.0.0.1:1',
                    api_key_file='/not-read', client_factory=Mock(side_effect=clients), budget=budget)
        self.assertEqual(result['timing']['failure_step']['name'], 'bank_independent_audit')
        for client in clients:
            client.close.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()

