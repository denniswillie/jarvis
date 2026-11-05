"""Offline host/child handshake, not native, SQL, model or worker acceptance."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4
from unittest.mock import patch

import pytest

from app.runtime.feature_config import MemoryAuthority
from evals.responsive_voice import frontend_host as host_module
from evals.responsive_voice import frontend_host_qualification as qualification
from evals.responsive_voice.frontend_host import (
    FrontendReplayHost, FrontendTaskReplayControl, NativeTaskReplayMetadata,
    ReplayHostError, ReplayHostScope, read_task_replay_file, scoped_database_url,
)
from evals.responsive_voice.native_task_release_control import create_control, write_private


class Bridge:
    def __init__(self):
        self.scope = self.admission = self.claim = None
        self.configurations = []
        self.close_calls = 0
        self.joined = True
        self.close_entered = asyncio.Event()
        self.close_gate = None

    async def configure(self, **kwargs):
        self.configurations.append(kwargs)

    async def close(self):
        self.close_calls += 1
        self.close_entered.set()
        if self.close_gate is not None:
            await self.close_gate.wait()
        return {'joined': self.joined, 'hooks_restored': True}


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path.resolve()
    root.chmod(0o700)
    now = datetime.now(UTC)
    run, barrier = uuid4(), uuid4()
    owner = f'direct-image-eval-second-brain-{run.hex}-a'
    schema = f'jarvis_sb_eval_{run.hex}_0'
    scope = ReplayHostScope(run_id=run.hex, owner_id=owner, database_schema=schema,
        area_id=uuid4(), memory_bank_id='jarvis_' + run.hex, expires_at=now + timedelta(minutes=3))
    operator, staging, worker = root / 'operator', root / 'staged', root / 'worker'
    for directory in (operator, staging, worker):
        directory.mkdir(mode=0o700)
    inputs, controls = staging / 'inputs', staging / 'controls'
    inputs.mkdir(mode=0o700)
    controls.mkdir(mode=0o700)
    metadata = {'format': 'jarvis.native_task_replay.host.v1', 'barrier_id': str(barrier),
        'model': 'controlled-offline', 'configuration_path': str(operator / 'configuration.json'),
        'status_path': str(operator / 'status.json'), 'stop_path': str(operator / 'stop.json')}
    hold = create_control(worker, run_id=run, owner_id=owner, database_schema=schema,
        worker_host_id=f'native-task-{run.hex}-' + 'a' * 12, model=metadata['model'],
        expires_at=scope.expires_at - timedelta(seconds=5), max_hold_seconds=45)
    readiness = {'status': 'ready', 'registry_ready': True, 'run_id': run.hex,
        'owner_id': owner, 'database_schema': schema, 'model': metadata['model'],
        'worker_id': 'frontier-' + uuid4().hex, 'host_id': hold.scope.worker_host_id,
        'pid': 12345, 'host_expires_at': scope.expires_at.isoformat(),
        'result_hold_control': hold.readiness()}
    task_scope = {'format': 'jarvis.native_task_scope.v1', 'run_id': str(run),
        'barrier_id': str(barrier), 'owner_id': owner, 'database_schema': schema,
        'model': metadata['model'], 'worker_id': readiness['worker_id'],
        'runtime_host_id': readiness['host_id'], 'host_expires_at': scope.expires_at.isoformat()}
    manifest = {'version': 1, 'voice_route': 'realtime', 'run_id': str(run), 'owner_id': owner,
        'task_barrier': {'version': 1, 'scope_file': 'task-scope.json', 'request_input_ordinal': 0},
        'utterances': [{'trigger': {'kind': 'at_time'}},
                       {'trigger': {'kind': 'after_task_claim', 'after_input_ordinal': 0, 'timeout_seconds': 30}}]}
    write_private(inputs, 'replay.json', json.dumps(manifest).encode())
    write_private(inputs, 'task-scope.json', json.dumps(task_scope).encode())
    config = {'format': 'jarvis.native_task_replay.configuration.v1', 'run_id': str(run),
        'barrier_id': str(barrier), 'worker_readiness': readiness, 'control_root': str(controls),
        'manifest_file': str(inputs / 'replay.json'), 'scope_file': str(inputs / 'task-scope.json'),
        'input_sample_counts': [480, 960]}
    resources = {'run_id': run.hex, 'owner_id': owner, 'database_schema': schema,
        'capability_profile': qualification.NATIVE_MEDIA, 'native_task_replay': metadata,
        'native_options': {'workload': qualification.REASONING_REMINDER,
                           'frontier_model': metadata['model'], 'voice_route': 'realtime'}}
    return SimpleNamespace(root=root, operator=operator, inputs=inputs, controls=controls,
        scope=scope, metadata=metadata, config=config, readiness=readiness, task_scope=task_scope,
        manifest=manifest, hold=hold, resources=resources)


def publish_configuration(f, value=None):
    write_private(f.operator, 'configuration.json', json.dumps(value or f.config).encode())


def publish_stop(f, **overrides):
    value = {'format': 'jarvis.native_task_replay.stop.v1', 'run_id': f.scope.run_id,
        'barrier_id': f.metadata['barrier_id'], 'requested_at': datetime.now(UTC).isoformat(), **overrides}
    write_private(f.operator, 'stop.json', json.dumps(value).encode())


async def eventually(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(.005)


@asynccontextmanager
async def running(f, *, bridge=None, clock=None):
    bridge = bridge or Bridge()
    controller = FrontendTaskReplayControl(f.scope, f.metadata, **({'clock': clock} if clock else {}))
    with patch.object(host_module.NativeTaskBarrierHost, 'attach', return_value=bridge) as attach:
        await controller.start(SimpleNamespace())
        try:
            yield controller, bridge, attach
        finally:
            if bridge.close_gate is not None:
                bridge.close_gate.set()
            await controller.close()


@pytest.mark.asyncio
async def test_late_configuration_real_release_constructor_and_explicit_stop(fixture):
    f = fixture
    async with running(f) as (control, bridge, attach):
        assert not control.accepting_inputs
        assert control.status() is None
        attach.assert_called_once()
        assert attach.call_args.kwargs['run_id'] == UUID(f.scope.run_id)
        assert 'session_id' not in attach.call_args.kwargs
        publish_configuration(f)
        await eventually(lambda: control.accepting_inputs)
        receipt = json.loads(read_task_replay_file(Path(f.metadata['status_path'])))
        assert receipt['status'] == 'ready'
        assert not receipt['observer_joined'] and not receipt['release_loop_joined']
        assert control.release.config == f.hold
        assert control.release.status()['status'] == 'waiting'
        assert bridge.configurations == [{'worker_readiness': f.readiness, 'control_root': f.controls,
            'history_root': f.operator / 'task-barrier-history'}]
        assert not list(f.hold.control_root.glob('release-*.json'))
        publish_stop(f)
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'closed'
        assert receipt['observer_joined'] and receipt['release_loop_joined']
        assert receipt['close_reason'] == 'operator_stop' and receipt['failure_code'] is None
        assert bridge.close_calls == 1
        assert not control.accepting_inputs


@pytest.mark.asyncio
async def test_stop_before_configuration_joins_without_release_or_session(fixture):
    async with running(fixture) as (control, bridge, _):
        publish_stop(fixture)
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'closed' and not receipt['configured']
        assert receipt['observer_joined'] and receipt['release_loop_joined']
        assert control.release is None and not bridge.configurations and bridge.scope is None


@pytest.mark.asyncio
@pytest.mark.parametrize('defect', ['run', 'barrier', 'owner', 'schema', 'model', 'registry', 'worker',
    'expiry', 'hold', 'layout', 'samples', 'manifest', 'scope', 'missing_hold'])
async def test_configuration_failure_closes_and_retains_failure(fixture, defect):
    f = fixture
    if defect == 'run':
        f.config['run_id'] = str(uuid4())
    elif defect == 'barrier':
        f.config['barrier_id'] = str(uuid4())
    elif defect in {'owner', 'schema', 'model', 'worker'}:
        f.readiness[{'owner': 'owner_id', 'schema': 'database_schema', 'model': 'model',
                     'worker': 'worker_id'}[defect]] = 'foreign'
    elif defect == 'registry':
        f.readiness['registry_ready'] = False
    elif defect == 'expiry':
        f.task_scope['host_expires_at'] = (f.scope.expires_at + timedelta(seconds=1)).isoformat()
        write_private(f.inputs, 'task-scope.json', json.dumps(f.task_scope).encode())
    elif defect == 'hold':
        f.readiness['result_hold_control']['control_id'] = str(uuid4())
    elif defect == 'missing_hold':
        del f.readiness['result_hold_control']
    elif defect == 'layout':
        f.config['control_root'] = str(f.operator)
    elif defect == 'samples':
        f.config['input_sample_counts'][0] = True
    elif defect == 'manifest':
        f.manifest['owner_id'] = 'foreign'
        write_private(f.inputs, 'replay.json', json.dumps(f.manifest).encode())
    elif defect == 'scope':
        f.task_scope['barrier_id'] = str(uuid4())
        write_private(f.inputs, 'task-scope.json', json.dumps(f.task_scope).encode())
    async with running(f) as (control, bridge, _):
        publish_configuration(f)
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'closed'
        assert receipt['failure_code'] == 'task_replay_control_invalid'
        assert receipt['observer_joined'] and receipt['release_loop_joined']
        assert not control.accepting_inputs and not bridge.configurations
        publish_stop(f)  # Pre-app runner cleanup remains possible after configuration failure.
        assert (await control.close())['status'] == 'closed'


@pytest.mark.asyncio
@pytest.mark.parametrize('replacement', [False, True])
async def test_configuration_not_reloadable_or_removable(fixture, replacement):
    async with running(fixture) as (control, bridge, _):
        publish_configuration(fixture)
        await eventually(lambda: control.accepting_inputs)
        if replacement:
            fixture.config['input_sample_counts'][0] += 1
            publish_configuration(fixture)
        else:
            Path(fixture.metadata['configuration_path']).unlink()
        await eventually(lambda: control.task.done())
        assert (await control.close())['failure_code'] == 'task_replay_control_invalid'
        assert len(bridge.configurations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('defect', ['wrong_run', 'wrong_barrier', 'future', 'old', 'duplicate_key'])
async def test_stop_control_scope_and_time_fail_closed(fixture, defect):
    async with running(fixture) as (control, _, _):
        overrides = {}
        if defect == 'wrong_run':
            overrides['run_id'] = str(uuid4())
        elif defect == 'wrong_barrier':
            overrides['barrier_id'] = str(uuid4())
        elif defect in {'future', 'old'}:
            overrides['requested_at'] = (datetime.now(UTC) + timedelta(
                seconds=30 if defect == 'future' else -30)).isoformat()
        publish_stop(fixture, **overrides)
        if defect == 'duplicate_key':
            path = Path(fixture.metadata['stop_path'])
            data = read_task_replay_file(path)
            write_private(fixture.operator, path.name, data[:-1] + b',"run_id":"duplicate"}')
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'closed' and receipt['failure_code']
        assert receipt['observer_joined'] and receipt['release_loop_joined']


@pytest.mark.asyncio
async def test_expiry_wakes_unconfigured_loop_without_business_cancellation(fixture):
    now = [datetime.now(UTC)]
    async with running(fixture, clock=lambda: now[0]) as (control, bridge, _):
        now[0] = fixture.scope.expires_at
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['failure_code'] == 'host_scope_expired'
        assert receipt['status'] == 'closed' and not bridge.configurations


@pytest.mark.asyncio
async def test_closed_never_precedes_observer_and_release_loop_join(fixture):
    bridge = Bridge()
    bridge.close_gate = asyncio.Event()
    async with running(fixture, bridge=bridge) as (control, _, _):
        publish_configuration(fixture)
        await eventually(lambda: control.accepting_inputs)
        publish_stop(fixture)
        await bridge.close_entered.wait()
        assert not control.task.done()
        assert control.status()['status'] != 'closed'
        bridge.close_gate.set()
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'closed' and receipt['release_loop_joined']


@pytest.mark.asyncio
async def test_unjoined_observer_is_not_clean_shutdown(fixture):
    bridge = Bridge()
    bridge.joined = False
    async with running(fixture, bridge=bridge) as (control, _, _):
        publish_stop(fixture)
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['status'] == 'failed' and not receipt['observer_joined']
        assert receipt['release_loop_joined'] and receipt['failure_code'] == 'observer_join_pending'


@pytest.mark.asyncio
async def test_release_failure_does_not_restart_controller_or_authorize_again(fixture):
    async with running(fixture) as (control, _, _):
        publish_configuration(fixture)
        await eventually(lambda: control.accepting_inputs)
        release = control.release
        release.state, release.reason = 'failed', 'release_not_acknowledged'
        with patch.object(release, 'step', wraps=release.step) as step:
            await eventually(lambda: control.task.done())
            receipt = await control.close()
            assert step.call_count == 1
        assert control.release is release and receipt['failure_code'] == 'release_control_failed'
        assert receipt['release']['reason'] == 'release_not_acknowledged'
        assert receipt['status'] == 'closed'


@pytest.mark.asyncio
async def test_unknown_release_write_uses_same_controller_until_ack_not_reconstruction(fixture):
    async with running(fixture) as (control, _, _):
        publish_configuration(fixture)
        await eventually(lambda: control.accepting_inputs)
        release = control.release
        unknown = {'status': 'release_requested', 'reason': 'release_write_outcome_unknown',
            'release_command_id': str(uuid4()), 'semantic_acceptance': 'unmeasured'}
        with patch.object(release, 'step', return_value=unknown) as step, patch.object(release, 'status', return_value=unknown):
            await eventually(lambda: step.call_count >= 2)
            assert control.release is release and control.failure_code is None
            assert control.status()['release']['reason'] == 'release_write_outcome_unknown'
            publish_stop(fixture)
            await eventually(lambda: control.task.done())
            assert (await control.close())['status'] == 'closed'


@pytest.mark.parametrize('defect', ['symlink', 'permissions', 'oversized', 'hardlink'])
def test_bounded_private_file_reader(fixture, defect):
    path = fixture.operator / 'input.json'
    target = fixture.operator / 'target.json'
    write_private(fixture.operator, 'target.json', b'{}')
    if defect == 'symlink':
        path.symlink_to(target)
    elif defect == 'hardlink':
        os.link(target, path)
    else:
        write_private(fixture.operator, path.name, b'{}')
        if defect == 'permissions':
            path.chmod(0o644)
        else:
            with path.open('wb') as stream:
                stream.write(b'x' * 16385)
    with pytest.raises(Exception):
        read_task_replay_file(path)


@pytest.mark.asyncio
async def test_preexisting_operator_control_rejected_before_attach(fixture):
    publish_configuration(fixture)
    control = FrontendTaskReplayControl(fixture.scope, fixture.metadata)
    with patch.object(host_module.NativeTaskBarrierHost, 'attach') as attach:
        with pytest.raises(ReplayHostError, match='not_fresh'):
            await control.start(SimpleNamespace())
        attach.assert_not_called()


@pytest.mark.parametrize('defect', ['alias', 'relative', 'escape', 'foreign_parent'])
def test_operator_metadata_paths(fixture, defect):
    metadata = dict(fixture.metadata)
    if defect == 'alias':
        metadata['status_path'] = metadata['stop_path']
    elif defect == 'relative':
        metadata['status_path'] = 'status.json'
    elif defect == 'escape':
        metadata['status_path'] = str(fixture.operator / '..' / 'status.json')
    else:
        metadata['status_path'] = str(fixture.root / 'status.json')
    with pytest.raises(Exception):
        NativeTaskReplayMetadata.model_validate_json(json.dumps(metadata))


def test_parent_child_forward_frozen_metadata_without_config_or_session(fixture):
    f = fixture
    env = qualification.task_replay_child_environment(f.resources, f.scope, f.root)
    assert set(env) == {'JARVIS_FRONTEND_REPLAY_TASK_CONTROL'}
    env.update(JARVIS_FRONTEND_REPLAY_PROFILE=qualification.NATIVE_MEDIA,
        JARVIS_FRONTEND_REPLAY_WORKLOAD=qualification.REASONING_REMINDER,
        JARVIS_FRONTEND_REPLAY_VOICE_ROUTE='realtime', SECOND_BRAIN_FRONTIER_MODEL=f.metadata['model'])
    with patch.dict(os.environ, env, clear=True), patch.object(qualification, 'create_replay_application') as create:
        qualification.child_replay_application(f.scope)
    assert create.call_args.args == (f.scope,)
    assert create.call_args.kwargs['native_task_replay'].model_dump(mode='json') == f.metadata
    assert not Path(f.metadata['configuration_path']).exists()
    assert 'session_id' not in env['JARVIS_FRONTEND_REPLAY_TASK_CONTROL']


def test_default_parent_and_child_are_unchanged(fixture):
    del fixture.resources['native_task_replay']
    assert qualification.task_replay_child_environment(fixture.resources, fixture.scope, fixture.root) == {}
    with patch.dict(os.environ, {}, clear=True), patch.object(qualification, 'create_replay_application') as create:
        qualification.child_replay_application(fixture.scope)
    create.assert_called_once_with(fixture.scope)


@pytest.mark.parametrize('defect', ['owner', 'profile', 'workload', 'route', 'model', 'outside'])
def test_forwarder_rejects_unmanifested_scope_or_profile(fixture, defect):
    resources = fixture.resources
    if defect == 'owner':
        resources['owner_id'] = 'personal-owner'
    elif defect == 'profile':
        resources['capability_profile'] = qualification.NO_MEDIA
    elif defect == 'outside':
        for key in ('configuration_path', 'status_path', 'stop_path'):
            resources['native_task_replay'][key] = str(fixture.root.parent / Path(resources['native_task_replay'][key]).name)
    else:
        resources['native_options'][{'workload': 'workload', 'route': 'voice_route', 'model': 'frontier_model'}[defect]] = 'wrong'
    with pytest.raises(qualification.QualificationFailure):
        qualification.task_replay_child_environment(resources, fixture.scope, fixture.root)


@pytest.mark.asyncio
async def test_real_host_lifespan_attaches_before_input_and_joins_before_runtime_close(fixture):
    f = fixture
    bridge = Bridge()
    events = []
    settings = SimpleNamespace(JARVIS_OWNER_ID=f.scope.owner_id, JARVIS_STATIC_BEARER_TOKEN='offline-token-' * 3,
        SUPABASE_DB_URL=scoped_database_url('postgresql://unused/unused', f.scope.database_schema))
    flags = SimpleNamespace(push_enabled=False, mail_enabled=False, sandbox_enabled=False, artifacts_enabled=False,
        enabled=True, knowledge_enabled=True, tasks_enabled=True, attention_enabled=True,
        knowledge_area_id=f.scope.area_id, selected_memory_authority=lambda: MemoryAuthority.HINDSIGHT)

    class Connection:
        async def fetchval(self, sql):
            if 'current_schema()' in sql:
                return f.scope.database_schema
            if 'current_schemas' in sql:
                return [f.scope.database_schema, 'pg_catalog', 'extensions']
            return datetime.now(UTC)

        async def fetchrow(self, *args):
            return {'bank_name': f.scope.memory_bank_id, 'status': 'active'}

    @asynccontextmanager
    async def acquire():
        yield Connection()

    pool = SimpleNamespace(acquire=acquire)
    knowledge = SimpleNamespace(owner_id=f.scope.owner_id, configuration=SimpleNamespace(area_id=f.scope.area_id),
        sources=SimpleNamespace(pool=pool), context_repository=SimpleNamespace(tasks=SimpleNamespace(pool=pool)),
        _closed=False, task=SimpleNamespace(done=lambda: False))
    runtime = SimpleNamespace(settings=settings, memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=knowledge,
        db_pool=pool, attention_service=SimpleNamespace(push=None), _closed=False)

    @asynccontextmanager
    async def lifespan(app):
        events.append('runtime_open')
        try:
            yield
        finally:
            assert bridge.close_calls == 1
            events.append('runtime_closed')

    class App:
        def __init__(self):
            self.router = SimpleNamespace(lifespan_context=lifespan)
            self.state = SimpleNamespace(runtime_services=runtime)

        async def __call__(self, scope, receive, send):
            events.append('application_input')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'{}'})

    app = App()
    host = FrontendReplayHost(app, f.scope, settings=settings, flags=flags, native_task_replay=f.metadata)
    asgi = {'type': 'http', 'path': '/ordinary', 'method': 'GET', 'headers': [
        (b'x-jarvis-evaluation-run', f.scope.run_id.encode()),
        (b'authorization', ('Bearer ' + settings.JARVIS_STATIC_BEARER_TOKEN).encode())]}
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {'type': 'http.request', 'body': b''}

    with patch.object(host_module.NativeTaskBarrierHost, 'attach', return_value=bridge) as attach:
        async with app.router.lifespan_context(app):
            assert host.ready and attach.call_count == 1
            host.check_live()  # Child server readiness does not require late worker configuration.
            await host(asgi, receive, send)
            assert sent[0]['status'] == 403 and 'application_input' not in events
            publish_configuration(f)
            await eventually(lambda: host.task_replay.accepting_inputs)
            sent.clear()
            await host(asgi, receive, send)
            assert sent[0]['status'] == 200 and 'application_input' in events
    assert events == ['runtime_open', 'application_input', 'runtime_closed']
    assert host.task_replay.status()['status'] == 'closed' and not host.ready
