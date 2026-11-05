"""Exercise the actual native runner through its existing offline process/host ports.

These doubles are not native/model/SQL acceptance. No receipt-only controller is
used: the same _run_session code as run_once stages, launches, exports and joins.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from uuid import UUID, uuid4
from unittest.mock import patch

import pytest

from evals.responsive_voice import native_replay_run as runner
from tests.test_native_replay_run import Fixture


class RetainedFixture(Fixture):
    def __init__(self, root):
        super().__init__(root)
        self.prepared("baseline", 0, 45)
        self.capture_dirs = []
        self.scope_bindings = []
        self.accepted_sessions = []
        self.anchor_fault = self.capture_fault = self.serve_fault = None
        self.fail_second_launch = self.block_second_capture = False
        self.reuse_session = False
        self.phase_ready = asyncio.Event()
        self.anchor_reads = 0
        self.next_boot = str(uuid4())
        self.source_clock_sample = {"clock": "ProcessInfo.systemUptime",
            "boot_session_id": self.next_boot, "uptime_seconds": 146.0}
        self.sequence_path = root / "retained-sequence.json"
        second = root / "second-input"
        shutil.copytree(self.options.media_root, second)
        manifest = runner.read_json(second / self.options.input_manifest.name)
        manifest["frames"] = []
        manifest["utterances"][0]["trigger"] = {"kind": "at_time", "at_seconds": 8}
        runner.save(second / self.options.input_manifest.name, manifest)
        second_risk = root / "second-risk.json"
        risk = runner.read_json(self.options.risk_authorization)
        risk.update(authorization_id=str(uuid4()),
                    input_manifest_sha256=hashlib.sha256(
                        (second / self.options.input_manifest.name).read_bytes()).hexdigest())
        runner.save(second_risk, risk)
        self.document = {
            "format": "jarvis.native_replay.retained_sequence.v1",
            "sequence_id": str(uuid4()),
            "phases": [
                {"invocation_id": str(uuid4()), "phase_start_seconds": 0,
                 "input_manifest": str(self.options.input_manifest),
                 "media_root": str(self.options.media_root),
                 "speech_manifest": str(self.options.speech_manifest),
                 "risk_authorization": str(self.options.risk_authorization)},
                {"invocation_id": str(uuid4()), "phase_start_seconds": 45,
                 "input_manifest": str(second / self.options.input_manifest.name),
                 "media_root": str(second), "speech_manifest": str(second / self.options.speech_manifest.name),
                 "risk_authorization": str(second_risk)},
            ],
        }
        self.options = replace(self.options, retained_sequence=self.sequence_path)
        self.authorize_sequence()

    def authorize_sequence(self):
        runner.save(self.sequence_path, self.document)
        plan_sha = hashlib.sha256(self.sequence_path.read_bytes()).hexdigest()
        ids = [runner.read_json(Path(p["risk_authorization"]))["authorization_id"]
               for p in self.document["phases"]]
        for index, phase in enumerate(self.document["phases"]):
            path = Path(phase["risk_authorization"])
            risk = runner.read_json(path)
            manifest_path = Path(phase["input_manifest"])
            template = runner.read_json(manifest_path)
            risk["input_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            risk["retained_sequence"] = {
                "sequence_id": self.document["sequence_id"], "sequence_plan_sha256": plan_sha,
                "phase_index": index, "replay_invocation_id": phase["invocation_id"],
                "phase_start_seconds": phase["phase_start_seconds"],
                "post_media_capture_seconds": runner.effective_capture_tail(template),
                "previous_authorization_id": ids[index - 1] if index else None,
                "accept_unconfirmed_prior_provider_closure": bool(index),
            }
            runner.save(path, risk)

    async def serve(self, manifest, stop, ready):
        self.child_stopped = False
        self.scope_bindings.append(runner.retained_scope(runner.read_json(manifest)))

        async def observed_ready(info):
            if self.serve_fault == "scope" and len(self.scope_bindings) == 2:
                info = dict(info, memory_bank_id="jarvis_" + uuid4().hex)
            await ready(info)

        result = await super().serve(manifest, stop, observed_ready)
        session = self.accepted_sessions[0] if self.reuse_session and self.accepted_sessions else str(uuid4())
        self.accepted_sessions.append(session)
        trace = result["child"]["websocket_trace"][0]
        trace.update(accepted=True, session_id=[session])
        if self.serve_fault == "failed":
            result["status"] = "failed"
        if self.serve_fault == "missing_admission":
            trace["accepted"] = False
        if self.serve_fault == "provider_budget":
            result["child_output_drain"]["provider_diagnostics"].append(
                {"stage": "created", "call_id": "unexpected_second_call"})
        return result

    async def command(self, argv, log_path, timeout=20):
        if len(argv) > 2 and argv[2] == "launch":
            self.launches += 1
            self.events.append("launch")
            self.plan = runner.read_json(Path(argv[argv.index("--jarvis-replay-manifest") + 1]))
            self.launched.set()
            if self.fail_launch or self.fail_second_launch and self.launches == 2:
                return {"returncode": 1, "stdout": ""}
            invocation = self.plan["replay_invocation_id"]
            root = self.container / "Documents/Replay" / str(UUID(self.run)) / invocation
            root.mkdir(parents=True)
            self.capture_dirs.append(root)
            clock = self.plan["replay_source_clock"]
            if clock["anchor_file"] is None:
                anchor = {
                    "format": "jarvis.replay.source_clock.v1", "run_id": str(UUID(self.run)),
                    "origin_invocation_id": invocation, "monotonic_before_seconds": 100.0,
                    "monotonic_after_seconds": 100.001, "utc_unix_seconds": 1793448000.0,
                    "boot_session_id": self.next_boot,
                }
                if self.anchor_fault == "owner":
                    anchor["run_id"] = str(uuid4())
                elif self.anchor_fault == "origin":
                    anchor["origin_invocation_id"] = str(uuid4())
                elif self.anchor_fault == "boot":
                    anchor["boot_session_id"] = None
                elif self.anchor_fault == "bracket":
                    anchor["monotonic_after_seconds"] = 102.0
                runner.save(root / "source-clock.json", anchor)
            else:
                source = Path(argv[argv.index("--jarvis-replay-manifest") + 1]).parent / clock["anchor_file"]
                raw = source.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == clock["anchor_sha256"]
                (root / "source-clock.json").write_bytes(raw)
                if self.anchor_fault == "changed_second":
                    value = runner.read_json(root / "source-clock.json")
                    value["utc_unix_seconds"] += 1
                    runner.save(root / "source-clock.json", value)
            capture = {"schema_version": 2, "run_id": self.run, "replay_invocation_id": invocation,
                       "evidence_kind": "OFFLINE_DOUBLE_NOT_NATIVE_ACCEPTANCE",
                       "replay_source_clock": {"version": 1, "status": "bound",
                           "phase_start_seconds": clock["phase_start_seconds"],
                           "anchor_sha256": hashlib.sha256((root / "source-clock.json").read_bytes()).hexdigest()}}
            if self.capture_fault == "invocation":
                capture["replay_invocation_id"] = str(uuid4())
            if self.capture_fault == "scope":
                capture["run_id"] = uuid4().hex
            if self.capture_fault == "late_clock":
                capture["replay_source_clock"].update(status="failed", failure_code="fixed_source_trigger_missed")
            if self.capture_fault == "rebased_clock":
                capture["replay_source_clock"]["phase_start_seconds"] += 1
            runner.save(root / "capture.json", capture)
            (root / "decoded.pcm").write_bytes(b"")
            return {"returncode": 0, "stdout": ""}
        return await super().command(argv, log_path, timeout)

    async def wait_capture(self, path, seconds):
        if path.name == "source-clock.json":
            self.anchor_reads += 1
        if self.block_second_capture and self.launches == 2 and path.name == "capture.json":
            self.phase_ready.set()
            await asyncio.Event().wait()
        return await super().wait_capture(path, seconds)


def fixture(tmp_path):
    return RetainedFixture(tmp_path)


async def run(f):
    # The process port already supplies synthetic anchors. Never compare those
    # anchors to the test machine's boot or uptime; retain the real deadline code.
    with patch.object(runner, "sample_replay_source_clock", return_value=f.source_clock_sample):
        return await runner.run_once(f.options, host=f, process=f)


@pytest.mark.parametrize("change,code", [
    ({"uptime_seconds": 154.0}, "retained_stimulus_deadline_expired"),
    ({"uptime_seconds": 99.0}, "retained_source_clock_future_anchor"),
    ({"clock": "unknown"}, "retained_source_clock_unavailable"),
    ({"boot_session_id": "00000000-0000-4000-8000-000000000001"},
     "retained_source_clock_boot_mismatch"),
])
@pytest.mark.asyncio
async def test_invalid_or_expired_clock_blocks_second_launch_commit_and_joins(tmp_path, change, code):
    f = fixture(tmp_path)
    f.source_clock_sample.update(change)
    result = await run(f)
    assert result["status"] == "failed"
    assert (f.prepares, f.launches, f.cleanups) == (1, 1, 1)
    assert result["phases"][0]["status"] == "session_complete"
    second = result["phases"][1]
    assert second["launch_attempts"] == 0
    assert second["fatal"]["code"] == code
    assert result["local_cleanup_complete"] is True
    assert f.events.count("host_stopped") == 2
    assert f.events[-1] == "fixture_cleanup"
    if code == "retained_stimulus_deadline_expired":
        assert second["source_clock_prelaunch"]["status"] == "expired"


@pytest.mark.asyncio
async def test_two_real_runner_operations_one_fixture_two_existing_ledger_spends(tmp_path):
    f = fixture(tmp_path)
    result = await run(f)
    assert result["status"] == "lifecycle_complete", result
    assert (f.prepares, f.launches, f.cleanups) == (1, 2, 1)
    assert f.events.count("canonical_export") == f.events.count("host_stopped") == 2
    assert f.events[-1] == "fixture_cleanup"
    assert f.scope_bindings[0] == f.scope_bindings[1]
    assert len(set(f.accepted_sessions)) == 2
    assert len(set(f.capture_dirs)) == 2
    assert all(root.parent.name == str(UUID(f.run)) for root in f.capture_dirs)
    assert result["local_cleanup_complete"] is True
    assert result["voice_qualification"]["status"] == "not_evaluated"
    assert result["memory_semantics"] == "not_evaluated"
    assert result["calendar_simulation"] == "not_enabled"
    for phase in result["phases"]:
        assert phase["status"] == "session_complete"
        assert phase["session_cleanup_complete"] is True
        assert phase["local_cleanup_complete"] is False
        assert phase["cleanup"]["fixture"]["status"] == "retained_for_sequence"
        assert Path(phase["canonical_evidence"]).is_file()
        assert (f.options.ledger_dir / f"authorization-{phase['authorization_id']}.spent").is_file()
    active = f.options.ledger_dir / f"simulator-{str(UUID(f.options.simulator_id)).upper()}.active.json"
    assert not active.exists()
    assert not f.token.exists()


@pytest.mark.asyncio
async def test_exact_anchor_copy_zero_camera_no_oracle_or_frame_reuse(tmp_path):
    f = fixture(tmp_path)
    result = await run(f)
    first, second = result["phases"]
    first_anchor = Path(first["source_anchor"]["path"]).read_bytes()
    second_plan = runner.read_json(Path(second["staged_inputs"]) / "replay.json")
    assert second_plan["frames"] == []
    assert second_plan["utterances"][0]["trigger"]["at_seconds"] == 8
    assert second_plan["replay_source_clock"]["phase_start_seconds"] == 45
    assert second_plan["replay_source_clock"]["anchor_sha256"] == hashlib.sha256(first_anchor).hexdigest()
    assert (Path(second["staged_inputs"]) / "source-clock.json").read_bytes() == first_anchor
    assert "expectations" not in second_plan and "prior_context" not in second_plan
    assert f.anchor_reads == 2
    assert result["fixed_source_deadlines"] == "never_rebased"


@pytest.mark.parametrize("field,value", [
    ("sequence_plan_sha256", "0" * 64), ("phase_index", 0),
    ("phase_index", True), ("replay_invocation_id", str(uuid4())),
    ("phase_start_seconds", 46), ("post_media_capture_seconds", 0),
    ("previous_authorization_id", None), ("accept_unconfirmed_prior_provider_closure", False),
])
@pytest.mark.asyncio
async def test_separate_second_release_is_required_before_allocation(tmp_path, field, value):
    f = fixture(tmp_path)
    path = Path(f.document["phases"][1]["risk_authorization"])
    risk = runner.read_json(path)
    risk["retained_sequence"][field] = value
    runner.save(path, risk)
    with pytest.raises(Exception, match="retained_sequence_authorization_mismatch"):
        await run(f)
    assert (f.prepares, f.launches, f.cleanups) == (0, 0, 0)


@pytest.mark.parametrize("fault", ["same_grant", "expired", "same_invocation", "overlap", "relative", "domain_clock"])
@pytest.mark.asyncio
async def test_plan_rejects_reuse_invalid_scope_and_implicit_calendar(tmp_path, fault):
    f = fixture(tmp_path)
    a, b = f.document["phases"]
    if fault == "same_grant":
        b["risk_authorization"] = a["risk_authorization"]
    elif fault == "expired":
        path = Path(b["risk_authorization"])
        risk = runner.read_json(path)
        risk["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        runner.save(path, risk)
    elif fault == "same_invocation":
        b["invocation_id"] = a["invocation_id"]
    elif fault == "overlap":
        b["phase_start_seconds"] = 44
    elif fault == "relative":
        b["input_manifest"] = "not-absolute.json"
    else:
        f.document["domain_clock"] = {"day": 90}
    runner.save(f.sequence_path, f.document)
    with pytest.raises(Exception):
        await run(f)
    assert f.prepares == f.launches == 0


@pytest.mark.parametrize("fault", ["owner", "origin", "boot", "bracket", "changed_second"])
@pytest.mark.asyncio
async def test_no_invented_or_changed_historical_anchor(tmp_path, fault):
    f = fixture(tmp_path)
    f.anchor_fault = fault
    result = await run(f)
    assert result["status"] == "failed"
    assert f.launches == (2 if fault == "changed_second" else 1)
    assert f.cleanups == 1
    assert result["local_cleanup_complete"] is True


@pytest.mark.parametrize("fault", ["invocation", "scope"])
@pytest.mark.asyncio
async def test_capture_cannot_reuse_other_invocation_or_scope(tmp_path, fault):
    f = fixture(tmp_path)
    f.capture_fault = fault
    result = await run(f)
    assert result["status"] == "failed"
    assert f.launches == 1
    assert result["phases"][0]["fatal"]["code"] == "retained_capture_identity_mismatch"


@pytest.mark.parametrize("fault", ["failed", "missing_admission", "provider_budget", "scope"])
@pytest.mark.asyncio
async def test_host_readiness_and_actual_admission_evidence_gate_continuation(tmp_path, fault):
    f = fixture(tmp_path)
    f.serve_fault = fault
    result = await run(f)
    assert result["status"] != "lifecycle_complete"
    if fault == "failed":
        assert result["status"] == "cleanup_pending"
        assert f.cleanups == 0
    elif fault == "scope":
        assert f.launches == 1
        assert f.cleanups == 1
    else:
        assert f.launches == 1
        assert f.cleanups == 1


@pytest.mark.asyncio
async def test_old_session_identity_cannot_count_as_fresh_memory_recall(tmp_path):
    f = fixture(tmp_path)
    f.reuse_session = True
    result = await run(f)
    assert result["status"] == "failed"
    assert result["phases"][1]["fatal"]["code"] == "retained_session_reused"
    assert f.cleanups == 1


@pytest.mark.asyncio
async def test_second_launch_failure_spends_no_retry_and_cleans_once(tmp_path):
    f = fixture(tmp_path)
    f.fail_second_launch = True
    result = await run(f)
    assert result["status"] == "failed"
    assert (f.prepares, f.launches, f.cleanups) == (1, 2, 1)
    assert all((f.options.ledger_dir / f"authorization-{p['authorization_id']}.spent").exists()
               for p in result["phases"])


@pytest.mark.asyncio
async def test_cancellation_joins_second_session_before_final_cleanup(tmp_path):
    f = fixture(tmp_path)
    f.block_second_capture = True
    task = asyncio.create_task(run(f))
    await asyncio.wait_for(f.phase_ready.wait(), 2)
    task.cancel()
    result = await asyncio.wait_for(task, 2)
    assert result["status"] != "lifecycle_complete"
    assert f.events.count("host_stopped") == 2
    assert f.cleanups == 1
    assert result["local_cleanup_complete"] is True


@pytest.mark.asyncio
async def test_unexported_scope_preserved_and_no_second_admission(tmp_path):
    f = fixture(tmp_path)
    f.fail_export = True
    result = await run(f)
    assert result["status"] == "cleanup_pending"
    assert f.launches == 1 and f.cleanups == 0
    active = f.options.ledger_dir / f"simulator-{str(UUID(f.options.simulator_id)).upper()}.active.json"
    assert active.is_file()
    assert Path(result["resource_manifest"]).exists()


@pytest.mark.asyncio
async def test_existing_active_marker_blocks_allocation_and_is_not_overwritten(tmp_path):
    f = fixture(tmp_path)
    f.options.ledger_dir.mkdir()
    active = f.options.ledger_dir / f"simulator-{str(UUID(f.options.simulator_id)).upper()}.active.json"
    old = {"journal": "/unresolved/old-run.json"}
    runner.save(active, old)
    result = await run(f)
    assert result["status"] == "failed"
    assert f.prepares == f.launches == f.cleanups == 0
    assert runner.read_json(active) == old


def test_zero_camera_is_not_enabled_for_ordinary_prepared_input(tmp_path):
    f = fixture(tmp_path)
    phase = f.document["phases"][1]
    options = replace(f.options, input_manifest=Path(phase["input_manifest"]),
                      media_root=Path(phase["media_root"]), speech_manifest=Path(phase["speech_manifest"]),
                      retained_sequence=None, artifact_dir=tmp_path / "staging")
    options.artifact_dir.mkdir()
    checksum = hashlib.sha256(options.input_manifest.read_bytes()).hexdigest()
    with pytest.raises(Exception, match="invalid_frames"):
        runner.stage_prepared_media(options, checksum)


@pytest.mark.parametrize("tail", [-1, 21, True, float("inf"), "20"])
def test_tail_invalid_values_rejected(tmp_path, tail):
    f = fixture(tmp_path)
    value = runner.read_json(f.options.input_manifest)
    value["post_media_capture_seconds"] = tail
    with pytest.raises(Exception):
        runner.effective_capture_tail(value)


@pytest.mark.parametrize("duration,tail,wait", [(45, 20, 95), (165, 15, 210), (180, 0, 210)])
def test_tail_wait_reservation_and_pcm_limit(duration, tail, wait):
    assert runner.effective_capture_tail({"duration_seconds": duration}) == tail
    budget = runner.capture_budget(duration, None, post_media_capture_seconds=tail)
    assert budget["capture_wait_seconds"] == wait
    assert budget["minimum_lease_seconds"] == duration + tail + 240
    assert budget["decoded_pcm_byte_limit"] == (30 + duration + tail) * 48_000 * 2


def test_explicit_zero_tail_preserves_media_boundary_and_old_helper_default():
    assert runner.effective_capture_tail({"duration_seconds": 45, "post_media_capture_seconds": 0}) == 0
    assert runner.capture_budget(45, None)["capture_wait_seconds"] == 75


@pytest.mark.asyncio
async def test_default_one_shot_still_one_scope_one_attempt_old_paths(tmp_path):
    f = Fixture(tmp_path)
    result = await runner.run_once(f.options, host=f, process=f)
    assert result["status"] == "lifecycle_complete"
    assert (f.prepares, f.launches, f.cleanups) == (1, 1, 1)
    assert f.plan["duration_seconds"] == 45
    assert "replay_invocation_id" not in f.plan and "replay_source_clock" not in f.plan
    assert f.capture_wait == 95  # Capture-only tail; media and question deadlines unchanged.
    assert result["voice_qualification"]["status"] == "not_evaluated"


@pytest.mark.parametrize("fault", ["late_clock", "rebased_clock"])
@pytest.mark.asyncio
async def test_native_late_or_rebased_clock_is_not_a_completed_phase(tmp_path, fault):
    f = fixture(tmp_path)
    f.capture_fault = fault
    result = await run(f)
    assert result["status"] == "failed"
    assert f.launches == 1
    assert result["phases"][0]["fatal"]["code"] == "retained_native_source_clock_not_bound"
    assert result["phases"][0]["source_clock_evidence"]


@pytest.mark.asyncio
async def test_sequence_grant_cannot_be_used_as_standalone_fresh_fixture_grant(tmp_path):
    f = fixture(tmp_path)
    with pytest.raises(Exception, match="sequence_authorization_requires_retained_runner"):
        await runner.run_once(replace(f.options, retained_sequence=None), host=f, process=f)
    assert f.prepares == f.launches == f.cleanups == 0


def test_explicit_tail_cannot_be_silently_ignored_by_legacy_fixed_baseline(tmp_path):
    f = Fixture(tmp_path)
    template = runner.read_json(f.options.input_manifest)
    template["post_media_capture_seconds"] = 0
    runner.save(f.options.input_manifest, template)
    f.options.artifact_dir.mkdir()
    with pytest.raises(Exception, match="explicit_capture_tail_requires_prepared_input"):
        runner.stage_media(f.options)

