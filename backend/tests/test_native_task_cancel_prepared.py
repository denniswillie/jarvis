"""Offline admission/control composition for one explicit task cancellation."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
import hashlib
import json

import pytest

from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.native_capture_evidence import analyze_capture
from evals.responsive_voice.native_task_replay_control import NativeTaskReplayRelease
from evals.responsive_voice.native_task_release_control import receipt_name, write_private
import test_native_replay_prepared_bounds as prepared
import test_native_task_capture_controls as controls
from test_native_task_replay_control import case, simulate_worker_ack


def sequence():
    return [prepared.PreparedBoundsTests.at_time(7),
        dict(clip_id='question',trigger=dict(kind='after_task_claim',after_input_ordinal=0,timeout_seconds=30)),
        dict(clip_id='question',trigger=dict(kind='response_output_start',after_input_ordinal=1,
            delay_seconds=.3,timeout_seconds=10,max_pcm_age_seconds=.05,require_task_claim=True)),
        dict(clip_id='question',trigger=dict(kind='after_provider_input_finalized',after_input_ordinal=2,timeout_seconds=10))]


@pytest.fixture
def material():
    value=prepared.PreparedBoundsTests()
    value.setUp()
    try:
        yield value
    finally:
        value.doCleanups()


def options(material, *, inputs=None, cuts=1, duration=120):
    return replace(material.options(duration=duration,task_aware=True,scenario='explicit_cancel',cuts=cuts,
        utterances=sequence() if inputs is None else inputs),workload=runner.REASONING_REMINDER,
        result_hold_seconds=60,worker_model='offline-fixture-model',worker_reasoning_effort='medium')


def test_exact_cancel_plan_stages_without_post_cancel_claim_or_runtime_authority(material):
    configured=options(material)
    plan,records=material.stage(configured)
    assert plan['utterances']==sequence() and len(records)==2
    assert all('require_task_claim' not in row['trigger'] for row in plan['utterances'][3:])
    assert not {'owner_id','run_id','session_id','worker_id'} & plan.keys()
    assert not configured.risk_authorization.exists()
    assert not (configured.artifact_dir/'inputs/task-scope.json').exists()


@pytest.mark.parametrize('fault',['two_cuts','duration150','live','manual','no_worker','missing_claim',
    'later_claim','wrong_predecessor','missing_probe','timed_probe','unknown_field'])
def test_cancel_admission_does_not_broaden_other_profiles_or_drop_barriers(material,fault):
    inputs=sequence()
    if fault=='missing_claim': inputs[2]['trigger'].pop('require_task_claim')
    if fault=='later_claim': inputs[3]['trigger']=dict(kind='after_task_claim',after_input_ordinal=0,timeout_seconds=30)
    if fault=='wrong_predecessor': inputs[3]['trigger']['after_input_ordinal']=1
    if fault=='missing_probe': inputs.pop()
    if fault=='timed_probe': inputs[3]['trigger']=dict(kind='at_time',at_seconds=100)
    if fault=='unknown_field': inputs[3]['trigger']['tool_cancelled']=True
    configured=options(material,inputs=inputs,cuts=2 if fault=='two_cuts' else 1,
        duration=150 if fault=='duration150' else 120)
    if fault=='live':
        configured=replace(configured,voice_route='live')
        material.change_manifest(configured,voice_route='live')
    if fault=='manual': material.change_manifest(configured,capture_mode='manual')
    if fault=='no_worker': configured=replace(configured,workload=runner.NO_WORKLOAD)
    with pytest.raises(Exception): material.stage(configured)
    assert not (configured.artifact_dir/'inputs').exists()


def test_runtime_opt_in_reaches_worker_prerequisite_without_allocating_or_launching(material):
    with pytest.raises(Exception,match='prepared_workload_worker_configuration_required'):
        asyncio.run(runner._run_session(options(material),host=None,process=None))


def cancellation_controller(case):
    manifest=deepcopy(case.manifest)
    manifest['utterances']=sequence()
    data=json.dumps(manifest).encode()
    for value in (case.binding,case.completion):
        value['manifest_file_sha256']=hashlib.sha256(data).hexdigest()
    case.write('session-binding.json',case.binding)
    case.write('stimulus-completion.json',case.completion)
    return NativeTaskReplayRelease(**{**case.arguments,'manifest_bytes':data,'input_sample_counts':(48000,)*4})


def test_all_cancel_scenario_inputs_do_not_release_a_cancelled_or_ineligible_attempt(case):
    controller=cancellation_controller(case)
    case.current=case.current.model_copy(update={'status':'blocked','reason':'canonical_unavailable','claim':None})
    held=case.held.model_copy(update={'phase':'cancelled','updated_at':case.now})
    write_private(case.config.control_root,receipt_name(held.identity),held.model_dump_json().encode())
    status=controller.step(now=case.now)
    assert status['status']=='waiting' and status['reason']=='canonical_claim_not_eligible'
    assert status['semantic_acceptance']=='unmeasured' and not case.calls
    # Even a stale eligible observation cannot release the cancelled receipt.
    case.current=case.current.model_copy(update={'status':'eligible','reason':'active_claim_observed',
        'claim':case.bridge.claim})
    assert controller.step(now=case.now)['reason']=='hold_not_pending'
    assert not case.calls


def test_cancel_plan_still_requires_every_final_input_before_any_release(case):
    controller=cancellation_controller(case)
    case.completion['inputs'].pop()
    case.write('stimulus-completion.json',case.completion)
    assert controller.step(now=case.now)['reason']=='input_coverage_mismatch'
    assert not case.calls


def test_completion_before_cancellation_race_is_release_history_not_cancel_success(case):
    controller=cancellation_controller(case)
    assert controller.step(now=case.now)['status']=='release_requested'
    assert len(case.calls)==1
    case.current=case.current.model_copy(update={'status':'blocked','reason':'canonical_unavailable','claim':None})
    simulate_worker_ack(case,case.calls[0])
    result=controller.step(now=case.now)
    assert result['status']=='released' and result['semantic_acceptance']=='unmeasured'
    assert len(case.calls)==1


def cancel_capture(block_before_cancel=False):
    capture,pcm,docs=controls.task_fixture(2)
    capture['utterances']=capture['utterances'][:4]
    for ordinal,plan in enumerate(sequence()):
        capture['utterances'][ordinal]['trigger']=plan['trigger']
    docs['barrier_receipts']=docs['barrier_receipts'][:2]
    def manifest(value):
        value['duration_seconds']=120
        value['utterances']=[dict(clip_id=row['clip_id'],trigger=row['trigger']) for row in capture['utterances']]
    controls.update_manifest(docs,manifest)
    def stimulus(value):
        value['inputs']=value['inputs'][:4]
    controls.change(docs,'stimulus_completion_bytes',stimulus)
    last=json.loads(docs['barrier_receipts'][-1])
    observed=datetime.fromisoformat(last['observed_at'].replace('Z','+00:00'))+timedelta(
        seconds=.05 if block_before_cancel else 1)
    last.update(sequence=last['sequence']+1,status='blocked',reason='canonical_unavailable',claim=None,
        observed_at=observed.isoformat(),expires_at=(observed+timedelta(seconds=1)).isoformat())
    docs['barrier_receipts'].append(controls.encoded(last))
    return capture,pcm,docs


def test_reader_preserves_pre_cancel_claim_proof_but_does_not_infer_cancellation():
    capture,pcm,docs=cancel_capture()
    result=analyze_capture(capture,pcm,task_evidence=docs)
    assert result['task_control_evidence']['verdict']=='validated_control_metadata'
    assert result['task_control_evidence']['validated_claim_inputs']==2
    assert result['task_control_evidence']['task_continuity']=='unknown'
    assert result['provider_input_finalization_evidence']['validated_transactions']==0
    assert result['provider_input_finalization_evidence']['required_transactions']==1
    assert result['actual_cuts'] is None and not result['hearing_measured']


def test_reader_never_accepts_a_claim_already_withdrawn_before_cancel_enqueue():
    capture,pcm,docs=cancel_capture(True)
    result=analyze_capture(capture,pcm,task_evidence=docs)
    assert result['task_control_evidence']['verdict']=='unmeasured'
    assert 'task_claim_invalidated_by_blocked_receipt' in result['task_control_evidence']['blockers']
