"""Offline runner ordering with synthetic host/app receipts, not native acceptance."""
import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from evals.responsive_voice.native_replay_run import read_json, run_once, save
from test_native_replay_run import Fixture


class TaskFixture(Fixture):
    def __init__(self, root, cuts=1):
        super().__init__(root)
        compiled = self.prepared("pause_continue", cuts, duration=120)
        self.integrated()
        self.options = replace(self.options, task_aware=True, result_hold_seconds=45.0)
        sources = {item["source_clip"]: item["clip_id"] for item in compiled.expectations["occurrences"]}
        utterances = [
            {"clip_id": sources["setup/background_task_setup"], "trigger": {"kind": "at_time", "at_seconds": 7}},
            {"clip_id": sources["state/operation_status"], "trigger": {
                "kind": "after_task_claim", "after_input_ordinal": 0, "timeout_seconds": 30}},
        ]
        for _ in range(cuts):
            utterances.append({"clip_id": sources["conversation/speech_only_pause"], "trigger": {
                "kind": "response_output_start", "after_input_ordinal": len(utterances) - 1,
                "delay_seconds": .3, "timeout_seconds": 10, "max_pcm_age_seconds": .05,
                "require_task_claim": True}})
            utterances.append({"clip_id": sources["conversation/continue_answer"], "trigger": {
                "kind": "after_input_complete", "after_input_ordinal": len(utterances) - 1,
                "delay_seconds": 1, "timeout_seconds": 10}})
        manifest = read_json(self.options.input_manifest)
        manifest.update(utterances=utterances, task_barrier={"version": 1,
            "scope_file": "task-scope.json", "request_input_ordinal": 0})
        save(self.options.input_manifest, manifest)
        speech = read_json(self.options.speech_manifest)
        used = {item["clip_id"] for item in utterances}
        speech["clips"] = [clip for clip in speech["clips"] if clip["id"] in used]
        save(self.options.speech_manifest, speech)
        self.authorize_prepared()
        risk = read_json(self.options.risk_authorization)
        risk.update(task_aware=True, result_hold_seconds=45.0)
        save(self.options.risk_authorization, risk)
        self.control_ready_failure = self.control_bad_join = False
        self.configuration = None

    async def start_worker(self, options, resources):
        worker = await super().start_worker(options, resources)
        self.readiness_receipt.update(host_expires_at=self.host_expiry, result_hold_control={
            "format": "jarvis.native_task_result_hold.control.v1", "control_id": str(uuid4()),
            "config_path": str(self.root / "offline-hold-config.json"), "max_hold_seconds": 45.0})
        return worker

    async def serve(self, manifest, stop, ready):
        metadata = read_json(manifest)["native_task_replay"]

        def report(status, **extra):
            save(Path(metadata["status_path"]), {"format": "jarvis.native_task_replay.status.v1",
                "run_id": self.run, "barrier_id": metadata["barrier_id"], "status": status,
                "evidence_kind": "OFFLINE_HOST_CONTROL_FAKE_NOT_OBSERVER_ACCEPTANCE", **extra})

        async def controls():
            configured = False
            while not stop.is_set():
                if Path(metadata["stop_path"]).exists():
                    self.events.append("control_closed")
                    report("closed", observer_joined=not self.control_bad_join, release_loop_joined=True)
                    return
                if not configured and Path(metadata["configuration_path"]).exists():
                    self.configuration = read_json(metadata["configuration_path"])
                    configured = True
                    self.events.append("control_ready")
                    report("failed" if self.control_ready_failure else "ready")
                await asyncio.sleep(.001)

        async def record_ready(info):
            self.host_expiry = info["expires_at"]
            await ready(info)

        control_task = asyncio.create_task(controls())
        try:
            return await super().serve(manifest, stop, record_ready)
        finally:
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cuts", [1, 2, 3])
async def test_task_mode_stages_bound_scope_and_controls_before_launch(tmp_path, cuts):
    f = TaskFixture(tmp_path, cuts)
    result = await run_once(f.options, host=f, process=f)
    assert result["status"] == "lifecycle_complete", result.get("fatal")
    assert (f.prepares, f.worker_starts, f.launches, f.worker_closes, f.cleanups) == (1, 1, 1, 1, 1)
    assert not result["voice_pass"]
    inputs = Path(result["staged_inputs"])
    controls = Path(result["task_replay"]["control_root"])
    assert inputs.name == "inputs" and controls == inputs.parent / "controls"
    scope = read_json(inputs / "task-scope.json")
    assert UUID(scope["run_id"]) == UUID(f.run)
    assert scope["owner_id"] == f.resource["owner_id"]
    assert scope["worker_id"] == f.readiness_receipt["worker_id"]
    assert scope["runtime_host_id"] == f.readiness_receipt["host_id"]
    assert "session_id" not in scope
    assert f.configuration["manifest_file"] == str(inputs / "replay.json")
    assert f.configuration["scope_file"] == str(inputs / "task-scope.json")
    assert f.configuration["control_root"] == str(controls)
    assert f.configuration["input_sample_counts"] == [960] * (2 + 2 * cuts)
    assert len(f.plan["utterances"]) == 2 + 2 * cuts
    assert f.plan["utterances"][1]["trigger"]["kind"] == "after_task_claim"
    assert f.events.index("control_ready") < f.events.index("launch")
    assert f.events.index("terminate") < f.events.index("control_closed") < f.events.index("worker_closed")
    assert f.events.index("worker_closed") < f.events.index("canonical_export") < f.events.index("fixture_cleanup")
    assert all(not Path(path).exists() for path in result["token_files"])


@pytest.mark.asyncio
async def test_control_configuration_failure_does_not_launch_app(tmp_path):
    f = TaskFixture(tmp_path)
    f.control_ready_failure = True
    result = await run_once(f.options, host=f, process=f)
    assert result["status"] == "failed"
    assert f.launches == 0
    assert f.worker_closes == 1 and f.cleanups == 1
    assert result["local_cleanup_complete"]


@pytest.mark.asyncio
async def test_unconfirmed_control_join_blocks_export_and_fixture_deletion(tmp_path):
    f = TaskFixture(tmp_path)
    f.control_bad_join = True
    result = await run_once(f.options, host=f, process=f)
    assert result["status"] == "cleanup_pending"
    assert f.worker_closes == 1 and f.child_stopped
    assert f.cleanups == 0 and "canonical_export" not in f.events
    assert "task_control_join_not_confirmed" in result["canonical_export_error"]
    assert Path(result["active_ledger"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["task_aware", "result_hold_seconds"])
async def test_explicit_new_authorization_required_before_allocation(tmp_path, field):
    f = TaskFixture(tmp_path)
    risk = read_json(f.options.risk_authorization)
    risk.pop(field)
    save(f.options.risk_authorization, risk)
    with pytest.raises(Exception):
        await run_once(f.options, host=f, process=f)
    assert f.prepares == 0 and f.launches == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["future_input", "no_claim", "bad_delay", "no_gate", "scope_path"])
async def test_invalid_task_triggers_fail_before_host_allocation(tmp_path, fault):
    f = TaskFixture(tmp_path)
    manifest = read_json(f.options.input_manifest)
    if fault == "future_input":
        manifest["utterances"][1]["trigger"]["after_input_ordinal"] = 1
    elif fault == "no_claim":
        manifest["utterances"][2]["trigger"].pop("require_task_claim")
    elif fault == "bad_delay":
        manifest["utterances"][3]["trigger"]["delay_seconds"] = 11
    elif fault == "no_gate":
        manifest["utterances"][1]["trigger"] = {"kind": "after_input_complete", "after_input_ordinal": 0,
            "delay_seconds": 1, "timeout_seconds": 10}
    else:
        manifest["task_barrier"]["scope_file"] = "../other.json"
    save(f.options.input_manifest, manifest)
    f.authorize_prepared()
    result = await run_once(f.options, host=f, process=f)
    assert result["status"] == "failed"
    assert (f.prepares, f.worker_starts, f.launches) == (0, 0, 0)


@pytest.mark.asyncio
async def test_task_fields_cannot_silently_enable_legacy_mode(tmp_path):
    f = TaskFixture(tmp_path)
    f.options = replace(f.options, task_aware=False, result_hold_seconds=None)
    risk = read_json(f.options.risk_authorization)
    risk.pop("task_aware")
    risk.pop("result_hold_seconds")
    save(f.options.risk_authorization, risk)
    result = await run_once(f.options, host=f, process=f)
    assert result["status"] == "failed"
    assert f.prepares == 0 and f.launches == 0
