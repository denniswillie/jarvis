"""Offline runner/profile binding tests, separate from host configuration coverage."""
from dataclasses import replace
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from evals.responsive_voice import frontend_host_qualification as host
from evals.responsive_voice import native_replay_run as runner
from tests.test_native_replay_run import Fixture
from tests.test_native_retained_sequence import RetainedFixture, run as retained_run


def visual_fixture(root, *, retained=False):
    f = RetainedFixture(root) if retained else Fixture(root)
    f.options = replace(f.options, workload=runner.VISUAL_MEMORY)
    paths = ([Path(p["risk_authorization"]) for p in f.document["phases"]]
             if retained else [f.options.risk_authorization])
    for path in paths:
        grant = runner.read_json(path)
        grant["workload"] = runner.VISUAL_MEMORY
        runner.save(path, grant)
    if retained:
        f.authorize_sequence()
    return f


def test_exact_host_profile_without_frontier_configuration(tmp_path):
    f = visual_fixture(tmp_path)
    options = runner.host_options(f.options)
    assert options["workload"] == host.VISUAL_MEMORY == "visual-memory"
    assert options["voice_route"] == "realtime"
    assert options["frontier_model"] is options["frontier_reasoning_effort"] is None
    with pytest.raises(Exception):
        host.native_host_options(host.NO_MEDIA, workload=host.VISUAL_MEMORY)


def test_visual_memory_lease_has_no_worker_reserve():
    visual = runner.capture_budget(165, None, workload=runner.VISUAL_MEMORY,
                                   post_media_capture_seconds=15)
    baseline = runner.capture_budget(165, None, post_media_capture_seconds=15)
    assert visual == baseline
    assert visual["minimum_lease_seconds"] == 420
    assert visual["decoded_pcm_byte_limit"] == 20_160_000


def test_exact_visual_grant_is_accepted_without_new_authorization_schema(tmp_path):
    f = visual_fixture(tmp_path)
    identity, grant = runner.validate_risk(f.options)
    assert identity == grant["authorization_id"]
    assert grant["workload"] == "visual-memory"
    assert grant["schema_version"] == 1
    assert f.prepares == f.launches == 0


@pytest.mark.parametrize("workload", [None, "none", "reasoning-reminder", "visual-memory "])
def test_visual_profile_requires_exact_explicit_grant_workload(tmp_path, workload):
    f = visual_fixture(tmp_path)
    grant = runner.read_json(f.options.risk_authorization)
    if workload is None:
        grant.pop("workload")
    else:
        grant["workload"] = workload
    runner.save(f.options.risk_authorization, grant)
    with pytest.raises(Exception, match="workload_authorization_scope_mismatch"):
        runner.validate_risk(f.options)


@pytest.mark.parametrize("change", [
    {"voice_route": "live"}, {"task_aware": True}, {"result_hold_seconds": 1},
    {"worker_model": "not-authorized"}, {"worker_api_key_file": Path("/not-used.key")},
    {"worker_python": Path("/not-used-python")}, {"worker_reasoning_effort": "high"},
    {"live_model": "not-authorized"}, {"live_voice": "not-authorized"},
])
def test_visual_profile_rejects_route_task_worker_and_live_options(tmp_path, change):
    f = visual_fixture(tmp_path)
    options = replace(f.options, **change)
    for check in (runner.validate_risk, runner.host_options):
        with pytest.raises(Exception, match="visual_memory_requires_realtime_without_worker"):
            check(options)
    assert f.prepares == f.launches == 0


@pytest.mark.parametrize("key,value", [
    ("frontier_model", "not-authorized"), ("frontier_reasoning_effort", "low"),
    ("result_hold_seconds", 1),
])
def test_visual_grant_cannot_carry_worker_authority(tmp_path, key, value):
    f = visual_fixture(tmp_path)
    grant = runner.read_json(f.options.risk_authorization)
    grant[key] = value
    runner.save(f.options.risk_authorization, grant)
    with pytest.raises(Exception, match="visual_memory_grant_worker_escalation"):
        runner.validate_risk(f.options)


def test_unknown_profile_cannot_gain_authority_with_matching_grant(tmp_path):
    f = visual_fixture(tmp_path)
    grant = runner.read_json(f.options.risk_authorization)
    grant["workload"] = "visual-memory-expanded"
    runner.save(f.options.risk_authorization, grant)
    with pytest.raises(Exception, match="invalid_native_workload"):
        runner.validate_risk(replace(f.options, workload=grant["workload"]))


@pytest.mark.asyncio
async def test_actual_one_shot_runner_uses_visual_profile_without_worker(tmp_path):
    f = visual_fixture(tmp_path)
    worker = AsyncMock(side_effect=AssertionError("visual profile must not start a frontier worker"))
    with patch.object(f, "start_worker", worker, create=True):
        report = await runner.run_once(f.options, host=f, process=f)
    assert report["status"] == "lifecycle_complete", report
    assert report["native_options"]["workload"] == "visual-memory"
    assert (f.prepares, f.launches, f.cleanups) == (1, 1, 1)
    assert "worker_readiness" not in report and "worker" not in report["cleanup"]
    worker.assert_not_awaited()
    assert report["voice_qualification"]["status"] == "not_evaluated"


@pytest.mark.parametrize("first_zero_camera", [False, True])
@pytest.mark.asyncio
async def test_actual_retained_runner_preserves_fixture_and_separate_admissions(tmp_path, first_zero_camera):
    f = visual_fixture(tmp_path, retained=True)
    if first_zero_camera:
        path = Path(f.document["phases"][0]["input_manifest"])
        manifest = runner.read_json(path)
        manifest["frames"] = []
        runner.save(path, manifest)
        f.authorize_sequence()
    worker = AsyncMock(side_effect=AssertionError("visual profile must not start a frontier worker"))
    with patch.object(f, "start_worker", worker, create=True):
        report = await retained_run(f)
    assert report["status"] == "lifecycle_complete", report
    assert (f.prepares, f.launches, f.cleanups) == (1, 2, 1)
    assert f.scope_bindings[0] == f.scope_bindings[1]
    assert len({p["authorization_id"] for p in report["phases"]}) == 2
    assert all(p["native_options"]["workload"] == "visual-memory" for p in report["phases"])
    assert all("worker_readiness" not in p and "worker" not in p["cleanup"] for p in report["phases"])
    assert report["fixed_source_deadlines"] == "never_rebased"
    assert report["memory_semantics"] == "not_evaluated"
    worker.assert_not_awaited()


def test_retained_second_grant_cannot_downgrade_or_inherit_profile(tmp_path):
    f = visual_fixture(tmp_path, retained=True)
    path = Path(f.document["phases"][1]["risk_authorization"])
    grant = runner.read_json(path)
    grant.pop("workload")
    runner.save(path, grant)
    with pytest.raises(Exception, match="workload_authorization_scope_mismatch"):
        runner.load_retained_sequence(f.options)
    assert f.prepares == f.launches == 0


def test_visual_profile_keeps_ordinary_empty_camera_rejection(tmp_path):
    f = visual_fixture(tmp_path, retained=True)
    phase = f.document["phases"][1]
    options = replace(f.options, retained_sequence=None,
        input_manifest=Path(phase["input_manifest"]), media_root=Path(phase["media_root"]),
        speech_manifest=Path(phase["speech_manifest"]), artifact_dir=tmp_path/"staging")
    checksum = hashlib.sha256(options.input_manifest.read_bytes()).hexdigest()
    with pytest.raises(Exception, match="invalid_frames"):
        runner.stage_prepared_media(options, checksum)


def test_cli_enum_is_explicit_and_default_remains_none(tmp_path, capsys):
    assert Fixture(tmp_path).options.workload == runner.NO_WORKLOAD
    with pytest.raises(SystemExit) as stopped:
        runner.main(["--help"])
    assert stopped.value.code == 0
    assert "visual-memory" in capsys.readouterr().out
