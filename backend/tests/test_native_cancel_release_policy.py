"""Offline held-pending policy, not voice cancellation or canonical settlement proof."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.responsive_voice import frontend_host as host
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.native_task_release_control import (
    BoundReleaseWaiter, read_hold, receipt_name, write_private,
)
from evals.responsive_voice.native_task_replay_control import (
    ALL_SDK_INPUTS, NO_AUTO_RELEASE, NativeTaskReplayRelease, ReplayControlError,
)
from evals.responsive_voice.native_task_result_hold import ResultHoldProvider
from test_native_task_replay_control import case, simulate_worker_ack
from test_native_task_replay_run import TaskFixture
from test_frontend_task_replay_host import (
    fixture, running, publish_configuration, publish_stop, eventually,
)


def held_controller(case, **kwargs):
    return NativeTaskReplayRelease(**{**case.arguments,
        'result_release_policy': NO_AUTO_RELEASE, 'scenario': 'explicit_cancel', **kwargs})


def policy_fixture(tmp_path, *, policy=NO_AUTO_RELEASE):
    # Grant/configuration fixture only. Existing case9 tests own full scenario admission.
    f = TaskFixture(tmp_path)
    f.options = replace(f.options, scenario='explicit_cancel', result_release_policy=policy)
    grant = runner.read_json(f.options.risk_authorization)
    grant.update(scenario='explicit_cancel', result_release_policy=policy)
    runner.save(f.options.risk_authorization, grant)
    return f


def test_default_release_and_legacy_receipt_shape_unchanged(case):
    result = case.controller.step(now=case.now)
    assert result['status'] == 'release_requested'
    assert 'result_release_policy' not in result
    simulate_worker_ack(case, case.calls[0])
    assert case.controller.step(now=case.now)['status'] == 'released'
    assert len(case.calls) == 1


def test_all_inputs_do_not_release_under_explicit_policy(case):
    controller = held_controller(case)
    for _ in range(3):
        result = controller.step(now=case.now)
        assert result['status'] == 'withheld'
        assert result['reason'] == 'automatic_release_disabled'
        assert result['release_command_id'] is None
        assert result['cancellation_qualification'] == result['semantic_acceptance'] == 'unmeasured'
    assert not case.calls
    assert not list(case.config.control_root.glob('release-*.json'))
    assert read_hold(case.config, attempt_id=case.held.identity.attempt_id,
                     fence=case.held.identity.fence).phase == 'entered'


@pytest.mark.parametrize('policy,scenario', [(None, 'explicit_cancel'), (True, 'explicit_cancel'),
    ('unknown', 'explicit_cancel'), (NO_AUTO_RELEASE, None), (NO_AUTO_RELEASE, 'pause_continue')])
def test_controller_policy_is_explicit_and_scoped(case, policy, scenario):
    with pytest.raises(ReplayControlError, match='invalid_result_release_policy'):
        held_controller(case, result_release_policy=policy, scenario=scenario)
    assert not case.calls


def test_hold_expiry_is_failed_not_cancelled(case):
    controller = held_controller(case)
    assert controller.step(now=case.now)['status'] == 'withheld'
    result = controller.step(now=case.held.expires_at)
    assert result['status'] == 'failed'
    assert result['reason'] == 'hold_bound_expired_without_cancel_proof'
    assert result['cancellation_qualification'] == 'unmeasured'
    assert not case.calls


@pytest.mark.parametrize('phase,expected', [('timeout', 'failed'), ('cancelled', 'hold_ended'),
                                          ('released', 'failed')])
def test_worker_terminal_receipt_does_not_fabricate_task_cancellation(case, phase, expected):
    controller = held_controller(case)
    controller.step(now=case.now)
    held = case.held.model_copy(update={'phase': phase, 'updated_at': case.now})
    write_private(case.config.control_root, receipt_name(held.identity), held.model_dump_json().encode())
    result = controller.step(now=case.now)
    assert result['status'] == expected
    assert result['cancellation_qualification'] == result['semantic_acceptance'] == 'unmeasured'
    assert result['release_command_id'] is None and not case.calls


def test_missing_stimulus_is_not_invented(case):
    (case.root / 'stimulus-completion.json').unlink()
    result = held_controller(case).step(now=case.now)
    assert result['status'] == 'waiting'
    assert result['reason'] == 'required_observation_missing'
    assert not case.calls


def test_incomplete_sdk_input_still_rejected(case):
    case.completion['inputs'][2]['accepted_samples'] -= 1
    case.write('stimulus-completion.json', case.completion)
    assert held_controller(case).step(now=case.now)['status'] == 'failed'
    assert not case.calls


@pytest.mark.asyncio
async def test_actual_waiter_cancel_joins_without_returning_or_releasing_result(case):
    config, identity = case.config, case.held.identity
    native = config.worker_root / 'native'
    native.mkdir(mode=0o700, exist_ok=True)
    write_private(native, 'worker.json', json.dumps({'format': 'jarvis.task_worker.v1',
        'worker_id': identity.worker_id, 'owner_digest': hashlib.sha256(config.scope.owner_id.encode()).hexdigest(),
        'host_id': config.scope.worker_host_id, 'pid': os.getpid()}).encode())
    receipt_path = config.control_root / receipt_name(identity)
    receipt_path.unlink()
    task = SimpleNamespace(owner_id=config.scope.owner_id, model=config.scope.model,
        backend_kind='frontier', session_id=identity.session_id, id=identity.task_id,
        current_attempt_id=identity.attempt_id, fence=identity.fence, revision=identity.task_revision)
    claim = SimpleNamespace(recovering=False, attempt_id=identity.attempt_id,
        fence=identity.fence, task_revision=identity.task_revision)
    context = SimpleNamespace(task=task, claim=claim)
    provider = SimpleNamespace(execute=AsyncMock(return_value=object()))
    execution = asyncio.create_task(ResultHoldProvider(provider, BoundReleaseWaiter(config),
        max_hold_seconds=config.max_hold_seconds).execute(context))
    try:
        async with asyncio.timeout(2):
            while not receipt_path.exists():
                await asyncio.sleep(0)
        controller = held_controller(case)
        assert controller.step(now=datetime.now(UTC))['status'] == 'withheld'
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert execution.done()
        result = controller.step(now=datetime.now(UTC))
        assert result['status'] == 'hold_ended'
        assert result['cancellation_qualification'] == 'unmeasured'
        assert not case.calls and not list(config.control_root.glob('release-*.json'))
        provider.execute.assert_awaited_once_with(context)
    finally:
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


def test_fresh_grant_required_and_default_grants_unchanged(tmp_path):
    f = policy_fixture(tmp_path)
    runner.validate_risk(f.options)
    grant = runner.read_json(f.options.risk_authorization)
    grant.pop('result_release_policy')
    runner.save(f.options.risk_authorization, grant)
    with pytest.raises(Exception, match='result_release_policy_authorization_mismatch'):
        runner.validate_risk(f.options)
    runner.validate_risk(replace(f.options, result_release_policy=ALL_SDK_INPUTS))
    assert f.prepares == f.launches == f.worker_starts == 0


@pytest.mark.parametrize('grant_policy', [None, True, 0, 'unknown', 'all_sdk_inputs', []])
def test_invalid_or_unauthorized_grant_policy_rejected(tmp_path, grant_policy):
    f = policy_fixture(tmp_path)
    grant = runner.read_json(f.options.risk_authorization)
    grant['result_release_policy'] = grant_policy
    runner.save(f.options.risk_authorization, grant)
    with pytest.raises(Exception, match='result_release_policy_authorization_mismatch'):
        runner.validate_risk(f.options)
    assert f.prepares == f.launches == 0


@pytest.mark.parametrize('override', [dict(scenario='pause_continue'), dict(cuts=2), dict(task_aware=False),
    dict(input_mode='baseline'), dict(workload='none'), dict(voice_route='live'),
    dict(result_hold_seconds=None), dict(result_hold_seconds=True), dict(result_hold_seconds=61),
    dict(result_hold_seconds=float('nan')), dict(result_hold_seconds=0)])
def test_policy_cannot_expand_existing_scope_or_hold_cap(tmp_path, override):
    f = policy_fixture(tmp_path)
    with pytest.raises(Exception, match='held_pending_cancel_scope_required'):
        runner.validate_risk(replace(f.options, **override))
    assert f.prepares == f.launches == 0


def test_runner_metadata_records_policy_without_touching_input_bytes(tmp_path):
    f = policy_fixture(tmp_path)
    before = f.options.input_manifest.read_bytes()
    metadata = runner.task_replay_metadata(f.options)
    assert metadata['result_release_policy'] == NO_AUTO_RELEASE
    assert metadata['scenario'] == 'explicit_cancel'
    typed = host._task_document(host.NativeTaskReplayMetadata, json.dumps(metadata).encode())
    assert typed.result_release_policy == NO_AUTO_RELEASE
    default = runner.task_replay_metadata(replace(f.options, result_release_policy=ALL_SDK_INPUTS))
    assert 'result_release_policy' not in default and 'scenario' not in default
    assert f.options.input_manifest.read_bytes() == before


@pytest.mark.parametrize('model,attribute', [
    (host.NativeTaskReplayMetadata, 'metadata'), (host._TaskReplayConfiguration, 'config')])
def test_exact_legacy_serialization_and_explicit_policy_binding(fixture, model, attribute):
    legacy = getattr(fixture, attribute)
    parsed = host._task_document(model, json.dumps(legacy).encode())
    assert parsed.model_dump(mode='json') == legacy
    assert json.loads(parsed.model_dump_json()) == legacy
    explicit_default = {**legacy, 'result_release_policy': ALL_SDK_INPUTS, 'scenario': None}
    parsed_default = host._task_document(model, json.dumps(explicit_default).encode())
    assert parsed_default.model_dump(mode='json') == legacy
    opted = {**legacy, 'result_release_policy': NO_AUTO_RELEASE, 'scenario': 'explicit_cancel'}
    parsed_opted = host._task_document(model, json.dumps(opted).encode())
    assert parsed_opted.model_dump(mode='json') == opted
    assert json.loads(parsed_opted.model_dump_json()) == opted


def test_opted_metadata_cannot_omit_cancellation_scope(fixture):
    invalid = {**fixture.metadata, 'result_release_policy': NO_AUTO_RELEASE}
    with pytest.raises(ValueError, match='held_pending_cancel_scope_required'):
        host._task_document(host.NativeTaskReplayMetadata, json.dumps(invalid).encode())


@pytest.mark.asyncio
async def test_host_policy_matches_authorized_metadata_before_configuration_io(fixture):
    f = fixture
    f.metadata.update(result_release_policy=NO_AUTO_RELEASE, scenario='explicit_cancel')
    control = host.FrontendTaskReplayControl(f.scope, f.metadata)
    with pytest.raises(host.ReplayHostError, match='replay_task_release_policy_mismatch'):
        await control._configure(json.dumps(f.config).encode())
    assert not control.configured and control.release is None


@pytest.mark.asyncio
async def test_host_uses_policy_and_joins_without_release_on_operator_stop(fixture):
    f = fixture
    f.metadata.update(result_release_policy=NO_AUTO_RELEASE, scenario='explicit_cancel')
    f.config.update(result_release_policy=NO_AUTO_RELEASE, scenario='explicit_cancel')
    async with running(f) as (control, bridge, _):
        publish_configuration(f)
        await eventually(lambda: control.accepting_inputs)
        assert control.release.result_release_policy == NO_AUTO_RELEASE
        assert control.release.status()['cancellation_qualification'] == 'unmeasured'
        publish_stop(f)
        await eventually(lambda: control.task.done())
        receipt = await control.close()
        assert receipt['observer_joined'] and receipt['release_loop_joined']
        assert receipt['failure_code'] is None
        assert receipt['semantic_acceptance'] == 'unmeasured'
        assert not list(f.hold.control_root.glob('release-*.json'))
