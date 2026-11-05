"""Offline batching tests using the real source inspection and transaction code.

Only SQL row transport and worker settlements are scripted. These tests do not
claim actual SQL, Hindsight, native cleanup or remote erasure acceptance.
"""
import asyncio
from collections import Counter, deque
from contextlib import ExitStack
from datetime import UTC, datetime
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from app.knowledge.models import SourceError
from evals.responsive_voice import frontend_host_qualification as host
from tests.test_native_cleanup_shared_deadline import AREA, Acquire, resources

BATCH_METRICS = []
STAMP = datetime(2026, 10, 31, tzinfo=UTC)


class FastPollBudget(host._CleanupDeadline):
    """Count existing polling steps without wall sleeps in count-only fixtures."""
    def __init__(self, result, **kwargs):
        self.progress_snapshots = []
        super().__init__(result, lambda: None, **kwargs)

    def progress(self, **values):
        self.progress_snapshots.append(json.loads(json.dumps(values)))
        super().progress(**values)

    async def step(self, name, factory, **kwargs):
        if name == 'source_poll_interval':
            async def ready():
                return None
            factory = ready
        return await super().step(name, factory, **kwargs)


class Transaction:
    def __init__(self, store):
        self.store = store
    async def __aenter__(self):
        self.store.transactions += 1
        return self
    async def __aexit__(self, *args):
        self.store.transaction_exits += 1
        return False


class Rows:
    """Exact owner/source row responses, not an alternative cleanup state machine."""
    def __init__(self, count):
        self.owner = resources()['owner_id']
        self.ids = [UUID(int=index+1) for index in range(count)]
        self.rows = {identity: {
            'id': identity, 'area_id': AREA, 'current_revision': 1,
            'generation': 2, 'status': 'deleted', 'updated_at': STAMP,
            'namespace': 'test-canonical-source', 'source_key': str(identity),
        } for identity in self.ids}
        self.provider_states = {identity: 'pending' for identity in self.ids}
        self.lookups = Counter()
        self.queries = Counter()
        self.owners = []
        self.transactions = self.transaction_exits = 0

    def transaction(self):
        return Transaction(self)

    def _owner(self, owner):
        assert owner == self.owner
        self.owners.append(owner)

    async def execute(self, sql, *args):
        assert 'pg_advisory_xact_lock' in sql
        self._owner(args[0])
        self.queries['owner_lock'] += 1
        return 'SELECT 1'

    async def fetch(self, sql, *args):
        self._owner(args[0])
        if sql.startswith('select id,area_id,current_revision,status'):
            self.queries['inventory'] += 1
            return [dict(self.rows[identity]) for identity in self.ids]
        assert 'select distinct retained.id as source_id' in sql
        self.queries['shared_originals'] += 1
        return []

    async def fetchrow(self, sql, *args):
        self._owner(args[0])
        if 'select id,current_revision,generation,status,updated_at,namespace,source_key' in sql:
            self.queries['source_lookup'] += 1
            identity = args[2]
            self.lookups[identity] += 1
            row = self.rows.get(identity)
            return dict(row) if row is not None and args[1] == row['area_id'] else None
        identity = args[1]
        if 'count(*) as versions' in sql:
            self.queries['local_originals'] += 1
            return {'versions': 1, 'payloads': 1, 'remaining_content_payloads': 0,
                    'not_deleted_payloads': 0}
        assert 'count(*) as documents' in sql
        self.queries['provider_originals'] += 1
        state = self.provider_states[identity]
        return {
            'documents': 1, 'cleanup_recorded': int(state == 'done'),
            'pending': int(state in {'pending', 'leased', 'accepted', 'outcome_unknown'}),
            'failed': int(state == 'failed'),
            'unverified': int(state in {'missing', 'cancelled', 'searchable', 'done_without_inventory'}),
        }

    async def fetchval(self, sql, *args):
        assert sql == 'select clock_timestamp()' and not args
        self.queries['observed_clock'] += 1
        return STAMP


class BatchProgressTests(unittest.IsolatedAsyncioTestCase):
    def prepare(self, count=45, *, actions=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        store = Rows(count)
        pool = SimpleNamespace(acquire=lambda: Acquire(store), close=AsyncMock())
        client = SimpleNamespace(close=AsyncMock())
        script = deque(actions if actions is not None else
                       [(identity, 'done', 'done') for identity in store.ids])
        worker_calls = []
        effects = []
        hooks = {}

        async def step(owner, worker_id):
            assert owner == store.owner
            assert worker_id == 'frontend-cleanup-'+resources()['run_id']
            worker_calls.append((owner, worker_id))
            number = len(worker_calls)
            if number in hooks:
                answer = await hooks[number]()
                if answer is not NotImplemented:
                    return answer
            if not script:
                return None
            identity, state, outcome = script.popleft()
            if identity is not None:
                store.provider_states[identity] = state
                effects.append((identity, state))
            return outcome

        worker = SimpleNamespace(step=AsyncMock(side_effect=step))
        stack.enter_context(patch.object(host.asyncpg, 'create_pool', AsyncMock(return_value=pool)))
        stack.enter_context(patch.object(host, 'HindsightClient', Mock(return_value=client)))
        stack.enter_context(patch.object(host, 'secret_file', Mock(return_value='fixture-placeholder')))
        stack.enter_context(patch('app.knowledge.ingestion.IngestionWorker', Mock(return_value=worker)))
        # Non-legacy originals only; the actual inspect implementation still does
        # its owner lock and five scoped reads. Optional legacy SQL is excluded.
        stack.enter_context(patch('app.knowledge.deletion.inspect_imported_message_original',
                                  AsyncMock(return_value=None)))
        result = {}
        budget = FastPollBudget(result)
        return SimpleNamespace(store=store, pool=pool, client=client, worker=worker,
            script=script, calls=worker_calls, effects=effects, hooks=hooks,
            result=result, budget=budget)

    async def run_cleanup(self, fixture):
        return await host.cleanup_source_originals('postgresql://unused', resources(), budget=fixture.budget)

    def metrics(self, fixture, label):
        value = {'case': label, 'sources': len(fixture.store.ids),
                 'canonical_inspections': sum(fixture.store.lookups.values()),
                 'sql_statements_in_fake_transport': sum(fixture.store.queries.values()),
                 'queries_by_kind': dict(fixture.store.queries),
                 'transactions': fixture.store.transactions,
                 'worker_steps': len(fixture.calls), 'settlements': len(fixture.effects),
                 'inspection_passes': fixture.result['timing'].get('progress', {}).get('inspection_passes')}
        BATCH_METRICS.append(value)
        return value

    async def test_real_inspection_cost_is_six_statements_plus_transaction_boundaries(self):
        f = self.prepare(1)
        f.store.provider_states[f.store.ids[0]] = 'done'
        await self.run_cleanup(f)
        metric = self.metrics(f, 'single_already_settled')
        self.assertEqual(metric['canonical_inspections'], 1)
        self.assertEqual(metric['sql_statements_in_fake_transport'], 7)  # Six + initial inventory.
        self.assertEqual(f.store.transactions, f.store.transaction_exits)
        self.assertEqual(len(f.calls), 0)

    async def test_45_64_128_sources_have_bounded_audits_and_serial_work(self):
        for count in (45, 64, 128):
            with self.subTest(count=count):
                f = self.prepare(count)
                result = await self.run_cleanup(f)
                batches = math.ceil(count/host.CLEANUP_WORKER_BATCH_SIZE)
                expected_inspections = count*(batches+1)
                metric = self.metrics(f, 'settle_'+str(count))
                self.assertEqual(result['status'], 'scoped_cleanup_recorded')
                self.assertEqual(len(result['receipts']), count)
                self.assertEqual(metric['canonical_inspections'], expected_inspections)
                self.assertEqual(metric['sql_statements_in_fake_transport'], 1+6*expected_inspections)
                self.assertEqual(metric['worker_steps'], batches*host.CLEANUP_WORKER_BATCH_SIZE)
                self.assertEqual(len(f.effects), count)
                self.assertTrue(all(owner == f.store.owner for owner in f.store.owners))
                self.assertEqual(f.store.transactions, f.store.transaction_exits)
                self.assertEqual(f.result['timing']['progress']['worker_step_limit'], 180)
                f.pool.close.assert_awaited_once()
                f.client.close.assert_awaited_once()

    async def test_mixed_accepted_pending_unknown_are_not_completion(self):
        f = self.prepare()
        first = [(f.store.ids[index], state, state) for index, state in enumerate(
            ['accepted', 'outcome_unknown', 'pending', 'accepted', 'outcome_unknown', 'pending', 'accepted'])]
        f.script.clear()
        f.script.extend(first+[(None, None, None)]+[(identity, 'done', 'done') for identity in f.store.ids])
        result = await self.run_cleanup(f)
        self.assertEqual(result['status'], 'scoped_cleanup_recorded')
        self.assertEqual(f.budget.progress_snapshots[1]['source_states'], {'pending': 45})
        self.assertEqual(f.budget.progress_snapshots[2]['source_states'], {'pending': 45})
        self.assertTrue(all(receipt['provider']['pending'] == 0 for receipt in result['receipts']))
        self.assertEqual(len(f.calls), 56)
        self.assertEqual(self.metrics(f, 'mixed_then_terminal')['canonical_inspections'], 45*8)

    async def test_failed_worker_forces_immediate_full_scope_check_and_stops(self):
        f = self.prepare()
        f.script.clear()
        f.script.extend([(f.store.ids[0], 'failed', 'failed'),
                         (f.store.ids[1], 'done', 'done')])
        with self.assertRaises(host.QualificationFailure):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 1)
        self.assertEqual(f.store.provider_states[f.store.ids[1]], 'pending')
        self.assertEqual(sum(f.store.lookups.values()), 90)
        f.pool.close.assert_awaited_once()
        f.client.close.assert_awaited_once()

    async def test_all_nonterminal_states_keep_deleted_target_and_total_180_cap(self):
        for state in ('pending', 'accepted', 'outcome_unknown', 'leased'):
            with self.subTest(state=state):
                f = self.prepare()
                for identity in f.store.ids:
                    f.store.provider_states[identity] = 'done'
                target = f.store.ids[-1]
                f.store.provider_states[target] = state
                f.script.clear()
                with self.assertRaises(host.QualificationFailure):
                    await self.run_cleanup(f)
                self.assertEqual(len(f.calls), 180)
                self.assertEqual(f.store.lookups[target], 24)
                self.assertEqual(sum(f.store.lookups.values()), 45*24)
                self.assertEqual(f.result['timing']['progress']['source_states'],
                                 {'scoped_cleanup_recorded': 44, 'pending': 1})
                self.metrics(f, 'unresolved_'+state)

    async def test_181_targets_never_multiply_worker_limit_or_omit_last_target(self):
        f = self.prepare(181)
        with self.assertRaises(host.QualificationFailure):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 180)
        self.assertEqual(len(f.effects), 180)
        self.assertEqual(f.store.lookups[f.store.ids[-1]], 24)
        self.assertEqual(sum(f.store.lookups.values()), 181*24)
        self.assertEqual(f.result['timing']['progress']['source_states'],
                         {'scoped_cleanup_recorded': 180, 'pending': 1})
        self.metrics(f, '181_exceeds_existing_work_cap')

    async def test_worker_done_claim_cannot_replace_canonical_deletion_receipts(self):
        f = self.prepare()
        f.script.clear()
        f.script.extend([(None, None, 'done')]*180)
        with self.assertRaises(host.QualificationFailure):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 180)
        self.assertEqual(f.result['timing']['progress']['source_states'], {'pending': 45})

    async def test_missing_already_deleted_target_is_not_dropped(self):
        f = self.prepare()
        async def remove():
            del f.store.rows[f.store.ids[0]]
            return NotImplemented
        f.hooks[8] = remove
        with self.assertRaises(SourceError) as raised:
            await self.run_cleanup(f)
        self.assertEqual(raised.exception.code, 'knowledge_source_not_found')
        self.assertEqual(len(f.calls), 8)
        f.pool.close.assert_awaited_once()

    async def test_changed_revision_is_rejected_after_batch(self):
        f = self.prepare()
        async def revise():
            f.store.rows[f.store.ids[0]]['current_revision'] = 2
            return NotImplemented
        f.hooks[8] = revise
        with self.assertRaisesRegex(host.QualificationFailure, 'source_cleanup_target_changed'):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 8)
        f.pool.close.assert_awaited_once()

    async def test_changed_area_cannot_expand_target_scope(self):
        f = self.prepare()
        async def move():
            f.store.rows[f.store.ids[0]]['area_id'] = UUID(int=999)
            return NotImplemented
        f.hooks[8] = move
        with self.assertRaises(SourceError):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 8)

    async def test_final_audit_rechecks_earlier_done_target_after_late_unknown(self):
        f = self.prepare()
        async def regress():
            f.store.provider_states[f.store.ids[0]] = 'outcome_unknown'
            f.script.append((f.store.ids[0], 'done', 'done'))
            return None
        f.hooks[48] = regress
        result = await self.run_cleanup(f)
        self.assertEqual(result['status'], 'scoped_cleanup_recorded')
        states = [value.get('source_states') for value in f.budget.progress_snapshots]
        self.assertIn({'pending': 1, 'scoped_cleanup_recorded': 44}, states)
        self.assertEqual(len(f.calls), 56)
        self.assertEqual(f.store.lookups[f.store.ids[0]], 8)

    async def test_late_failed_original_never_passes_from_cached_done(self):
        f = self.prepare()
        async def fail():
            f.store.provider_states[f.store.ids[0]] = 'failed'
            return None
        f.hooks[48] = fail
        with self.assertRaises(host.QualificationFailure):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 48)

    async def test_worker_exception_stops_remaining_batch_and_joins(self):
        f = self.prepare()
        async def fail():
            raise ValueError('test-only failure')
        f.hooks[4] = fail
        with self.assertRaises(ValueError):
            await self.run_cleanup(f)
        self.assertEqual(len(f.calls), 4)
        self.assertEqual(f.result['timing']['failure_step']['batch_position'], 3)
        f.pool.close.assert_awaited_once()
        f.client.close.assert_awaited_once()

    async def test_cancelled_inflight_batch_and_later_recovery_are_separate(self):
        f = self.prepare()
        entered = asyncio.Event()
        async def block():
            f.store.provider_states[f.store.ids[0]] = 'leased'
            entered.set()
            await asyncio.Event().wait()
        f.hooks[1] = block
        task = asyncio.create_task(self.run_cleanup(f))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        old = json.dumps(f.result, sort_keys=True)
        self.assertEqual(f.result['timing']['failure_step']['status'], 'cancelled')
        self.assertEqual(f.store.provider_states[f.store.ids[0]], 'leased')
        f.hooks.clear()
        fresh = {}
        f.budget = FastPollBudget(fresh)
        recovered = await self.run_cleanup(f)
        self.assertEqual(recovered['status'], 'scoped_cleanup_recorded')
        self.assertEqual(json.dumps(f.result, sort_keys=True), old)
        self.assertEqual(f.pool.close.await_count, 2)
        self.assertEqual(f.client.close.await_count, 2)

    async def test_timeout_keeps_unknown_work_pending_until_separate_recovery(self):
        f = self.prepare()
        async def block():
            f.store.provider_states[f.store.ids[0]] = 'outcome_unknown'
            await asyncio.Event().wait()
        f.hooks[1] = block
        f.budget = FastPollBudget(f.result,
            deadline=asyncio.get_running_loop().time()+.3, shutdown_seconds=.1)
        with self.assertRaises(TimeoutError):
            await self.run_cleanup(f)
        old = json.dumps(f.result, sort_keys=True)
        self.assertEqual(f.result['timing']['failure_step']['name'], 'source_worker_step')
        self.assertEqual(f.result['timing']['failure_step']['status'], 'timed_out')
        self.assertEqual(f.store.provider_states[f.store.ids[0]], 'outcome_unknown')
        f.hooks.clear()
        fresh = {}
        f.budget = FastPollBudget(fresh)
        self.assertEqual((await self.run_cleanup(f))['status'], 'scoped_cleanup_recorded')
        self.assertEqual(json.dumps(f.result, sort_keys=True), old)

    async def test_invalid_remote_cleanup_inventory_still_requires_attention(self):
        for state in ('missing', 'cancelled', 'searchable', 'done_without_inventory'):
            with self.subTest(state=state):
                f = self.prepare(1)
                f.store.provider_states[f.store.ids[0]] = state
                with self.assertRaises(host.QualificationFailure):
                    await self.run_cleanup(f)
                self.assertEqual(len(f.calls), 0)


if __name__ == '__main__':
    unittest.main()
