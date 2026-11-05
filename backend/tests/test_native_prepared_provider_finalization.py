"""Offline prepared admission only; no runtime identities or grants are made."""
from dataclasses import replace
import hashlib
import json

import pytest

from evals.responsive_voice import native_replay_run as runner
import test_native_replay_prepared_bounds as prepared


@pytest.fixture
def material():
    fixture = prepared.PreparedBoundsTests()
    fixture.setUp()
    try:
        yield fixture
    finally:
        fixture.doCleanups()


def sequence(cuts=1):
    inputs = [prepared.PreparedBoundsTests.at_time(1), {
        "clip_id": "question", "trigger": {
            "kind": "after_task_claim", "after_input_ordinal": 0, "timeout_seconds": 1}}]
    for _ in range(cuts):
        inputs.append({"clip_id": "question", "trigger": {
            "kind": "response_output_start", "after_input_ordinal": len(inputs)-1,
            "delay_seconds": .1, "timeout_seconds": 1,
            "max_pcm_age_seconds": .05, "require_task_claim": True}})
        inputs.append({"clip_id": "question", "trigger": {
            "kind": "after_provider_input_finalized", "after_input_ordinal": len(inputs)-1,
            "timeout_seconds": 10}})
    return inputs


def options(material, *, cuts=1, duration=120, inputs=None):
    return replace(material.options(duration=duration, task_aware=True,
        scenario="pause_continue", cuts=cuts,
        utterances=sequence(cuts) if inputs is None else inputs),
        workload=runner.REASONING_REMINDER, result_hold_seconds=45)


@pytest.mark.parametrize("cuts", [1, 2, 3])
def test_distinct_provider_plan_stages_without_runtime_authority(material, cuts):
    configured = options(material, cuts=cuts)
    plan, records = material.stage(configured)
    assert plan["utterances"] == sequence(cuts)
    assert plan["duration_seconds"] == 120
    assert len(records) == 2
    assert not {"run_id", "owner_id", "area_id", "memory_bank_id", "session_id"} & plan.keys()
    assert not configured.risk_authorization.exists()
    assert not configured.ledger_dir.exists()
    assert not (configured.artifact_dir / "inputs/task-scope.json").exists()


@pytest.mark.parametrize("ordinal", [-1, 0, 1, 3, True, 2.0])
def test_finalization_requires_exact_immediate_predecessor(material, ordinal):
    inputs = sequence()
    inputs[-1]["trigger"]["after_input_ordinal"] = ordinal
    with pytest.raises(Exception, match="provider_input_finalization_requires_preceding_task_cut"):
        material.stage(options(material, inputs=inputs))


def test_predecessor_must_be_the_planned_task_cut_not_commentary_or_setup(material):
    inputs = sequence()
    inputs[-2]["trigger"] = {"kind": "after_input_complete", "after_input_ordinal": 1,
                             "delay_seconds": 0, "timeout_seconds": 1}
    with pytest.raises(Exception, match="provider_input_finalization_requires_preceding_task_cut"):
        material.stage(options(material, inputs=inputs))


@pytest.mark.parametrize("field", ["delay_seconds", "provider_item_id", "response_id", "connection_id"])
def test_finalization_rejects_unqualified_identity_or_delay_fields(material, field):
    inputs = sequence()
    inputs[-1]["trigger"][field] = "not-authority"
    with pytest.raises(Exception, match="unsupported_provider_input_finalization_fields"):
        material.stage(options(material, inputs=inputs))


def test_finalization_requires_complete_trigger_fields(material):
    inputs = sequence()
    inputs[-1]["trigger"].pop("timeout_seconds")
    with pytest.raises(Exception, match="unsupported_provider_input_finalization_fields"):
        material.stage(options(material, inputs=inputs))


@pytest.mark.parametrize("timeout", [0, -1, True, "10", 120.001])
def test_finalization_wait_keeps_finite_existing_bounds(material, timeout):
    inputs = sequence()
    inputs[-1]["trigger"]["timeout_seconds"] = timeout
    with pytest.raises(Exception, match="invalid_provider_input_finalization_timeout"):
        material.stage(options(material, inputs=inputs))


@pytest.mark.parametrize("fault", ["not_task", "live", "manual", "wrong_case", "no_worker_profile"])
def test_finalization_cannot_escalate_other_routes_or_profiles(material, fault):
    configured = options(material)
    if fault == "not_task":
        configured = replace(configured, task_aware=False)
    elif fault == "live":
        configured = replace(configured, voice_route="live")
        material.change_manifest(configured, voice_route="live")
    elif fault == "manual":
        material.change_manifest(configured, capture_mode="manual")
    elif fault == "wrong_case":
        configured = replace(configured, scenario="parallel_keep_working")
    else:
        configured = replace(configured, workload=runner.NO_WORKLOAD)
    with pytest.raises(Exception):
        material.stage(configured)
    assert not (configured.artifact_dir / "inputs").exists()


def test_entire_wait_and_clip_must_fit_original_scene_deadline(material):
    # Four four-sample clips: 1s start + 1s claim + 1s response + 10s
    # finalization + clip PCM + unchanged 5s scene reserve exceeds 18s.
    with pytest.raises(Exception, match="scenario_cannot_fit_deadline"):
        material.stage(options(material, duration=18))
    plan, _ = material.stage(options(material, duration=18.001))
    assert plan["duration_seconds"] == 18.001
    with pytest.raises(Exception, match="provider_input_finalization_scene_exceeds_task_bound"):
        material.stage(options(material, duration=120.001))


def test_prior_task_continue_and_capture_budget_unchanged(material):
    inputs = sequence()
    inputs[-1]["trigger"] = {"kind": "after_input_complete", "after_input_ordinal": 2,
                             "delay_seconds": 1, "timeout_seconds": 10}
    plan, _ = material.stage(options(material, inputs=inputs))
    assert plan["utterances"] == inputs
    budget = runner.capture_budget(120, 480, workload=runner.REASONING_REMINDER,
                                   post_media_capture_seconds=0)
    assert budget["minimum_lease_seconds"] == 480
    assert budget["capture_wait_seconds"] == 150
    assert budget["decoded_pcm_byte_limit"] == 14_400_000


def test_trigger_change_requires_exact_new_manifest_hash(material):
    configured = options(material)
    approved = hashlib.sha256(configured.input_manifest.read_bytes()).hexdigest()
    manifest = json.loads(configured.input_manifest.read_bytes())
    manifest["utterances"][-1]["trigger"]["timeout_seconds"] = 9
    prepared.save(configured.input_manifest, manifest)
    with pytest.raises(Exception, match="authorized_manifest_hash_mismatch"):
        runner.stage_media(configured, expected_manifest_sha256=approved)


def test_declared_three_cut_scene150_stages_without_widening_global_budget(material):
    configured = options(material, cuts=3, duration=150)
    plan, records = material.stage(configured)
    assert plan["duration_seconds"] == 150
    assert plan["utterances"] == sequence(3)
    assert len(records) == 2
    assert not configured.risk_authorization.exists()
    assert not configured.ledger_dir.exists()
    assert not (configured.artifact_dir / "inputs/task-scope.json").exists()
    assert runner.MAX_DURATION == 180
    assert runner.MAX_CAPTURE_PCM_BYTES == 20_160_000
    budget = runner.capture_budget(150, 510, workload=runner.REASONING_REMINDER,
                                   post_media_capture_seconds=0)
    assert budget["minimum_lease_seconds"] == 510
    assert budget["capture_wait_seconds"] == 180
    assert budget["decoded_pcm_byte_limit"] == 17_280_000
    with pytest.raises(Exception, match="lease_too_short_for_capture_and_export"):
        runner.capture_budget(150, 480, workload=runner.REASONING_REMINDER,
                              post_media_capture_seconds=0)


@pytest.mark.parametrize("cuts,duration", [(1,150),(2,150),(3,120.001),(3,149),(3,150.001),(3,151),(3,180)])
def test_scene150_does_not_relax_one_two_or_arbitrary_three_cut_bounds(material, cuts, duration):
    with pytest.raises(Exception, match="provider_input_finalization_scene_exceeds_task_bound"):
        material.stage(options(material, cuts=cuts, duration=duration))


def test_scene150_requires_all_three_distinct_provider_continuations(material):
    inputs = sequence(3)
    inputs[-1]["trigger"] = {"kind":"after_input_complete", "after_input_ordinal":6,
                             "delay_seconds":1, "timeout_seconds":10}
    with pytest.raises(Exception, match="provider_input_finalization_scene_exceeds_task_bound"):
        material.stage(options(material, cuts=3, duration=150, inputs=inputs))
