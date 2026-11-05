"""Offline reproduction of the pre-runtime invalidation boundary.

Uses real TaskRunner, HermesTaskProvider, TaskRepository.start_runtime/claim/context
guards, and ResultHoldProvider. SQL transport, native launch/result, and upstream
transport are controlled. This is NOT SQL, model execution, or native acceptance.
The stale-manifest regressions require canonical fenced abandonment. The prior
stranded-claim counterexample remains in the original qualification artifacts.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider, _NativeRun
from app.tasks.models import ClaimLost, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner, WorkerContext
from app.tasks.worker import TaskWorkerRole
from evals.responsive_voice.native_task_result_hold import ResultHoldProvider


class ControlledConnection:
    def __init__(self, repository):
        self.repository = repository

    async def fetchrow(self, query, *args):
        r = self.repository
        if query.startswith('select *,lease_until>clock_timestamp()'):
            assert args == (r.task.owner_id, r.claim.attempt_id)
            return dict(r.attempt)
        if query.startswith('select m.status from jarvis_task_attempts'):
            return {'status': r.manifest_status}
        raise AssertionError('uncontrolled SQL row request')

    async def fetch(self, query, *args):
        assert query.startswith('select grant_id from jarvis_task_grants')
        return []

    async def execute(self, query, *args):
        r = self.repository
        assert "update jarvis_task_attempts set provider_state='live'" in query
        assert args[:2] == (r.task.owner_id, r.claim.attempt_id)
        r.attempt.update(provider_state='live', provider_handle=args[2], checkpoint_payload_id=args[3],
                         runtime_host_id=args[4])
        r.order.append('runtime_bound')


class ControlledRepository(TaskRepository):
    def __init__(self):
        super().__init__(None, SimpleNamespace(shutdown_seconds=.2))
        now, run = datetime.now(UTC), uuid4().hex
        self.task = SimpleNamespace(id=uuid4(), owner_id='direct-image-eval-second-brain-' + run + '-a',
            session_id=uuid4(), model='offline-configured-model', backend_kind='frontier',
            revision=1, fence=1, status=TaskStatus.RUNNING, active_seconds=0,
            pending_status=None, knowledge_area_id=None, title='Controlled reasoning task',
            requested_outcome='Reason only; no external actions.', request_event_id=uuid4(), request_revision=2,
            budget=SimpleNamespace(shutdown_seconds=5, active_seconds=120, lease_seconds=.15,
                                   poll_seconds=.01, request_output_tokens=64))
        self.claim = SimpleNamespace(task=self.task, attempt_id=uuid4(), task_revision=1, fence=1,
            recovering=False, provider_state='not_started', provider_handle=None,
            lease_until=now + timedelta(seconds=30))
        self.task.current_attempt_id = self.claim.attempt_id
        self.attempt = dict(id=self.claim.attempt_id, lease_valid=True, fence=1, status='running',
            provider_state='not_started', provider_handle=None, runtime_host_id=None,
            checkpoint_payload_id=None)
        self.manifest_status = 'active'
        self.connection = ControlledConnection(self)
        self.order, self.events, self.finishes, self.abandons = [], [], [], []
        self.renews = self.rollbacks = 0
        self.checkpoint = AsyncMock()

    @asynccontextmanager
    async def transaction(self, owner_id=None):
        assert owner_id == self.task.owner_id
        before = dict(self.attempt)
        try:
            yield self.connection
        except BaseException:
            self.attempt = before
            self.rollbacks += 1
            raise

    async def load(self, connection, owner_id, task_id, *, lock=False):
        assert connection is self.connection and (owner_id, task_id) == (self.task.owner_id, self.task.id)
        return self.task

    async def _source_available(self, connection, owner_id, event_id, revision, *, lock=True):
        assert (owner_id, event_id, revision) == (self.task.owner_id, self.task.request_event_id, 2)
        return True

    async def payload(self, connection, owner_id, payload):
        assert payload['host_id'] == self.expected_host
        assert payload['phase'] == 'starting'
        return uuid4()

    async def emit(self, connection, task, kind, **kwargs):
        self.events.append(kind)

    async def renew(self, claim):
        assert claim is self.claim
        self.renews += 1
        return True

    async def finish(self, claim, outcome):
        assert claim is self.claim
        self.finishes.append(outcome)
        self.order.append('settled_controlled_result')
        return True

    async def abandon(self, claim, **kwargs):
        self.abandons.append(kwargs)
        await super().abandon(claim, **kwargs)

    async def _charge_time(self, connection, task, attempt):
        return task.active_seconds

    async def has_unknown_actions(self, connection, task):
        return getattr(self, "unknown_action", False)

    async def transition(self, connection, task, status, *, reason=None):
        task.status = status

    async def _abandon_locked(self, connection, task, attempt, **kwargs):
        # Execute the real fenced settlement algorithm against controlled SQL.
        async def execute(query, *args):
            if query.startswith("update jarvis_task_attempts set status="):
                self.attempt.update(status=args[2], lease_valid=False,
                                    ended_at=None if args[3] else "controlled-ended")
            elif query.startswith("update jarvis_background_tasks set fence="):
                task.fence += 1
                task.current_attempt_id = args[2]
                if args[3]:
                    task.pending_status = "paused"
                elif not args[4] and not args[5]:
                    task.pending_status = None
            else:
                assert query.startswith(("update jarvis_task_actions set status=",
                                         "update jarvis_task_reservations set status="))
        with patch.object(connection, "execute", side_effect=execute):
            await super()._abandon_locked(connection, task, attempt, **kwargs)


class ControlledGateway:
    def __init__(self, context, upstream, *, run_id, **kwargs):
        self.run_id = run_id
        self.images, self.tool_receipts = {}, []

    def close(self):
        pass


def execution(tmp_path, *, wrapped, manifest_status='active', waiter=None):
    repository = ControlledRepository()
    root = tmp_path.resolve() / 'native'
    root.mkdir(mode=0o700)
    repository.expected_host = 'native-task-offline-runtime-host'
    context = WorkerContext(repository.claim, repository, None, None)
    prepared = SimpleNamespace(manifest=SimpleNamespace(id=uuid4(), owner_id=repository.task.owner_id,
        task_id=repository.task.id, task_revision=1),
        tools=SimpleNamespace(context=context, catalog=lambda: []))

    async def prepare(actual):
        assert actual is context
        repository.manifest_status = manifest_status
        repository.order.append('prepared_manifest')
        return prepared

    prepare.capabilities = ('live.readonly.v1',)
    configuration = HermesRuntimeConfiguration(Path(sys.executable), root, host_id=repository.expected_host)
    upstream = SimpleNamespace(aclose=AsyncMock())
    provider = HermesTaskProvider(SimpleNamespace(model=repository.task.model), configuration,
        prepare=prepare, upstream_factory=lambda: upstream)
    waiter = waiter or AsyncMock()
    decorated = ResultHoldProvider(provider, waiter, max_hold_seconds=1) if wrapped else provider
    runner = TaskRunner(repository, {'frontier': decorated}, worker_id='offline-worker',
        owner_id=repository.task.owner_id, host_id=repository.expected_host)
    runner.context = lambda claim: context
    return SimpleNamespace(repository=repository, context=context, provider=provider, decorated=decorated,
        runner=runner, waiter=waiter, root=root, upstream=upstream)


@asynccontextmanager
async def controlled_native(run):
    async def launch(native, prepared, frontier):
        assert run.repository.attempt['provider_state'] == 'live'
        assert run.repository.attempt['runtime_host_id'] == run.provider.configuration.host_id
        assert run.repository.events == ['runtime_started']
        run.repository.order.append('controlled_launch_boundary')

    # A deliberately incomplete result avoids claiming a real model answer. It
    # still exercises normal execute return/checkpoint/hold/runner settlement.
    result = AsyncMock(return_value={'completed': False})
    with patch('app.tasks.hermes.TaskModelGateway', ControlledGateway), \
         patch.object(_NativeRun, 'launch', autospec=True, side_effect=launch) as launching, \
         patch.object(_NativeRun, 'result', result), \
         patch('asyncio.create_subprocess_exec', side_effect=AssertionError('native launch forbidden')):
        yield launching, result


@pytest.mark.asyncio
@pytest.mark.parametrize('wrapped', [False, True])
async def test_runtime_identity_commits_before_controlled_launch_with_or_without_hold(tmp_path, wrapped):
    run = execution(tmp_path, wrapped=wrapped)
    async with controlled_native(run) as (launch, result):
        await run.runner._execute(run.repository.claim)
    assert run.repository.events == ['runtime_started']
    assert run.repository.attempt['runtime_host_id'] == run.provider.configuration.host_id
    assert run.repository.attempt['provider_handle'].startswith('hermes-run:')
    assert run.repository.order == ['prepared_manifest', 'runtime_bound', 'controlled_launch_boundary',
                                    'settled_controlled_result']
    assert len(run.repository.finishes) == 1 and not run.repository.abandons
    launch.assert_awaited_once()
    result.assert_awaited_once()
    assert run.waiter.await_count == int(wrapped)
    run.upstream.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('manifest_status', ['superseded', 'revoked'])
async def test_invalid_context_settles_current_claim_before_runtime(tmp_path, wrapped, manifest_status):
    run = execution(tmp_path, wrapped=wrapped, manifest_status=manifest_status)
    async with controlled_native(run) as (launch, result):
        await run.runner._execute(run.repository.claim)
    assert run.repository.order == ['prepared_manifest']
    assert run.repository.rollbacks == 1
    assert run.repository.task.status == TaskStatus.QUEUED
    assert run.repository.task.current_attempt_id is None
    assert run.repository.task.fence == 2
    assert run.repository.attempt['status'] == 'interrupted'
    assert run.repository.attempt['ended_at'] is not None
    assert run.repository.manifest_status == manifest_status
    assert run.repository.attempt['provider_state'] == 'not_started'
    assert run.repository.attempt['runtime_host_id'] is None
    assert run.repository.attempt['provider_handle'] is None
    assert run.repository.events == ['queued'] and not run.repository.finishes
    assert run.repository.abandons == [{'reason': 'worker_claim_lost'}]
    assert not [task for task in asyncio.all_tasks() if task.get_coro().__qualname__ == 'TaskRunner._heartbeat']
    launch.assert_not_awaited()
    result.assert_not_awaited()
    run.waiter.assert_not_awaited()
    markers = list(run.root.glob('*/run.json'))
    assert len(markers) == 1
    marker = json.loads(markers[0].read_text())
    assert marker['phase'] == 'exited' and marker['pid'] is None and marker['exit_code'] is None
    assert marker['private_home_cleanup']['state'] == 'complete'


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['revision', 'fence', 'cancel', 'lease', 'replacement'])
async def test_actual_ownership_loss_still_fences_runtime_and_hold(tmp_path, change):
    run = execution(tmp_path, wrapped=True)
    if change == 'revision':
        run.repository.task.revision = 2
    elif change == 'fence':
        run.repository.task.fence = 2
    elif change == 'lease':
        run.repository.attempt['lease_valid'] = False
    elif change == 'replacement':
        run.repository.task.current_attempt_id = uuid4()
    else:
        run.repository.task.status = TaskStatus.CANCEL_REQUESTED
    before_task, before_attempt = vars(run.repository.task).copy(), run.repository.attempt.copy()
    async with controlled_native(run) as (launch, result):
        await run.runner._execute(run.repository.claim)
    launch.assert_not_awaited()
    result.assert_not_awaited()
    run.waiter.assert_not_awaited()
    assert not run.repository.events and not run.repository.finishes
    assert run.repository.abandons == [{'reason': 'worker_claim_lost'}]
    assert vars(run.repository.task) == before_task
    assert run.repository.attempt == before_attempt
    assert run.repository.attempt['runtime_host_id'] is None


@pytest.mark.asyncio
async def test_hold_keeps_normal_heartbeat_and_delays_settlement_only_after_executor_return(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def wait(context):
        entered.set()
        await release.wait()

    run = execution(tmp_path, wrapped=True, waiter=wait)
    async with controlled_native(run):
        task = asyncio.create_task(run.runner._execute(run.repository.claim))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
                while not run.repository.renews:
                    await asyncio.sleep(.005)
            assert not task.done() and not run.repository.finishes
            assert run.repository.attempt['runtime_host_id'] == run.provider.configuration.host_id
            assert run.repository.events == ['runtime_started']
            release.set()
            await asyncio.wait_for(task, 1)
            assert len(run.repository.finishes) == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_actual_worker_role_sees_original_configuration_model_and_prepare_capabilities(tmp_path):
    run = execution(tmp_path, wrapped=True)
    role = TaskWorkerRole(run.repository, {'frontier': run.decorated}, owner_id=run.repository.task.owner_id,
        configuration=SimpleNamespace(host_id=run.repository.expected_host))
    assert run.decorated.configuration is run.provider.configuration
    assert run.decorated.prepare is run.provider.prepare
    assert run.decorated.model == run.provider.model
    assert run.decorated.reconcile.__self__ is run.provider
    assert role.runner.host_id == run.repository.expected_host
    assert role.capabilities == ('live.readonly.v1',)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['submitting', 'live', 'unknown_action'])
async def test_claim_lost_preserves_uncertain_effect_instead_of_requeue(state):
    repository = ControlledRepository()
    repository.unknown_action = state == 'unknown_action'
    if not repository.unknown_action:
        repository.attempt['provider_state'] = state
        repository.attempt['provider_handle'] = 'controlled-original-handle' if state == 'live' else None
    provider = SimpleNamespace(execute=AsyncMock(side_effect=ClaimLost()))
    runner = TaskRunner(repository, {'frontier': provider})
    await runner._execute(repository.claim)
    assert repository.task.status == TaskStatus.UNKNOWN
    assert repository.events == ['outcome_unknown'] and not repository.finishes
    assert repository.task.current_attempt_id == (None if repository.unknown_action else repository.claim.attempt_id)
    assert repository.attempt['status'] == ('interrupted' if repository.unknown_action else 'outcome_unknown')
    assert repository.attempt['provider_state'] == ('not_started' if repository.unknown_action else state)
    provider.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_claim_lost_recovery_keeps_existing_reconciliation_authority(caplog):
    repository = ControlledRepository()
    repository.claim.recovering = True
    repository.claim.provider_state = 'live'
    repository.task.pending_status = 'cancelled'
    provider = SimpleNamespace(reconcile=AsyncMock(side_effect=ClaimLost()))
    before = vars(repository.task).copy(), repository.attempt.copy()
    await TaskRunner(repository, {'frontier': provider})._execute(repository.claim)
    provider.reconcile.assert_awaited_once()
    assert provider.reconcile.await_args.kwargs == {'cancel': True}
    assert (vars(repository.task), repository.attempt) == before
    assert not repository.abandons and not repository.events
    assert 'recovery_owner_retained' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('lost', [False, True])
async def test_claim_lost_diagnostic_identifies_only_actual_allowlisted_branch(tmp_path, caplog, lost):
    run = execution(tmp_path, wrapped=True, manifest_status='superseded')
    if lost:
        run.repository.task.fence += 1
    async with controlled_native(run):
        await run.runner._execute(run.repository.claim)
    records = [r for r in caplog.records if r.name == 'app.tasks.runner']
    assert len(records) == 1
    evidence = json.loads(records[0].getMessage().split(' ', 1)[1])
    assert evidence['origin'] == ('canonical_claim_guard' if lost else 'runtime_registration')
    assert evidence['stage'] == 'provider_execute'
    assert evidence['disposition'] == 'fenced_abandon_returned'
    assert evidence['task_id'] == str(run.repository.task.id)
    for private in (run.repository.task.owner_id, run.repository.task.requested_outcome, str(tmp_path)):
        assert private not in records[0].getMessage()


@pytest.mark.asyncio
async def test_unknown_claim_lost_origin_redacts_exception_and_logging_failure_cannot_strand(tmp_path, caplog):
    error = ClaimLost()
    error.args = ('PRIVATE-EXCEPTION-CONTENT',)
    repository = ControlledRepository()
    provider = SimpleNamespace(execute=AsyncMock(side_effect=error))
    await TaskRunner(repository, {'frontier': provider})._execute(repository.claim)
    assert 'PRIVATE-EXCEPTION-CONTENT' not in caplog.text
    assert '"origin": "unavailable"' in caplog.text
    run = execution(tmp_path, wrapped=False, manifest_status='superseded')
    with patch('app.tasks.runner.LOGGER.warning', side_effect=RuntimeError('reporting unavailable')):
        async with controlled_native(run):
            await run.runner._execute(run.repository.claim)
    assert run.repository.task.status == TaskStatus.QUEUED
    assert run.repository.events == ['queued']


@pytest.mark.asyncio
async def test_abandon_failure_is_not_reported_as_returned_or_hidden(caplog):
    repository = ControlledRepository()
    repository.abandon = AsyncMock(side_effect=RuntimeError('controlled settlement failure'))
    provider = SimpleNamespace(execute=AsyncMock(side_effect=ClaimLost()))
    with pytest.raises(RuntimeError, match='controlled settlement failure'):
        await TaskRunner(repository, {'frontier': provider})._execute(repository.claim)
    assert 'fenced_abandon_failed' in caplog.text
    assert 'fenced_abandon_returned' not in caplog.text
    assert not repository.events


@pytest.mark.asyncio
async def test_heartbeat_claim_loss_fences_and_joins_running_work(caplog):
    repository = ControlledRepository()
    repository.attempt['provider_state'] = 'live'
    repository.renew = AsyncMock(return_value=False)
    entered, joined = asyncio.Event(), asyncio.Event()

    async def execute(context):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            joined.set()

    await asyncio.wait_for(TaskRunner(repository, {'frontier': SimpleNamespace(execute=execute)})._execute(repository.claim), 1)
    assert entered.is_set() and joined.is_set()
    assert repository.task.status == TaskStatus.UNKNOWN
    assert not repository.finishes
    assert '"origin": "heartbeat"' in caplog.text
