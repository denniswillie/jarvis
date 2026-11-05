"""Offline artifact bounds for the qualified client startup/scene/tail allocation.

Synthetic PCM tests prove byte acceptance/rejection, not native playback or hearing.
"""
import asyncio
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice import native_capture_evidence as reader
from evals.responsive_voice.test_native_capture_evidence import fixture as capture_fixture
from tests.test_native_replay_run import Fixture


@pytest.mark.parametrize("scene,tail,seconds", [
    (45, 20, 95), (67, 0, 97), (65, 20, 115),
    (160, 20, 210), (165, 15, 210), (180, 0, 210),
])
def test_exact_client_startup_scene_tail_allocation(scene, tail, seconds):
    budget = runner.capture_budget(scene, None, post_media_capture_seconds=tail)
    assert budget["decoded_pcm_byte_limit"] == seconds * 48_000 * 2
    assert budget["capture_wait_seconds"] == seconds
    assert budget["capture_json_byte_limit"] == 16 * 1024 * 1024
    assert runner.MAX_CAPTURE_PCM_BYTES == 20_160_000
    assert reader.MAX_PCM_BYTES == runner.MAX_CAPTURE_PCM_BYTES
    assert reader.MAX_DECODED_BLOCKS == 21_000
    assert reader.MAX_SCENE_DURATION == 180
    assert reader.MAX_JSON_BYTES == 16 * 1024 * 1024
    assert runner.MAX_DURATION == 180
    assert runner.MAX_TRIGGER_TIMEOUT == 120
    assert budget["minimum_lease_seconds"] == scene + tail + 240


def test_fractional_capacity_uses_client_sample_rounding_down():
    scene, tail = 45.123456, 0.125
    budget = runner.capture_budget(scene, None, post_media_capture_seconds=tail)
    assert budget["decoded_pcm_byte_limit"] == int((30 + scene + tail) * 48_000) * 2


@pytest.mark.parametrize("scene,tail", [(180, 1), (165, 16), (160, 21), (181, 0)])
def test_startup_does_not_expand_scene_or_tail_authority(scene, tail):
    with pytest.raises(Exception):
        runner.capture_budget(scene, None, post_media_capture_seconds=tail)


@pytest.mark.parametrize("overflow", [0, 1, 2])
def test_maximum_preroll_and_scene_tail_exact_copy_and_overflow(tmp_path, overflow):
    limit = runner.capture_budget(160, None, post_media_capture_seconds=20)["decoded_pcm_byte_limit"]
    source, target = tmp_path / "decoded.pcm", tmp_path / "retained.pcm"
    with source.open("wb") as stream:
        stream.truncate(20_160_000 + overflow)
    if overflow:
        with pytest.raises(Exception, match="artifact_byte_bound"):
            runner.copy_bounded(source, target, limit)
        assert not target.exists()
    else:
        receipt = runner.copy_bounded(source, target, limit)
        assert receipt["bytes"] == 20_160_000
        assert target.stat().st_size == 20_160_000


@pytest.mark.parametrize("overflow", [0, 1, 2])
def test_existing_capture_reader_accepts_maximum_storage_and_rejects_overflow(tmp_path, overflow):
    capture, _, _ = capture_fixture(0)
    # Existing real reader fixture, extended decoded storage. No semantic or
    # acoustic claim: this tests its file-byte boundary, not hearing or timing.
    size = 20_160_000 + overflow
    capture["decoded_pcm"].update(byte_count=size, sample_count=size // 2)
    reference = capture["clock_anchor"]["monotonic_reference_seconds"]
    capture["decoded_sample_map"] = [
        {"sample_offset": index * 480, "sample_count": 480,
         "callback_start_uptime_seconds": reference - 30 + index / 100,
         "callback_end_uptime_seconds": reference - 30 + index / 100 + .001,
         "local_output_epoch": 1} for index in range(21_000)]
    capture["decoded_blocks"] = 21_000
    capture["output_gates"] = [
        {"event_id": f"{capture['media_epoch_id']}/output-gate/{index}",
         "enabled": enabled, "changed": True, "requested_uptime_seconds": at,
         "local_output_epoch": 1, "last_admitted_active_sample": None}
        for index, (enabled, at) in enumerate([(True, reference - 30), (False, reference + 180)])]
    path = tmp_path / "capture.json"
    runner.save(path, capture)
    with (tmp_path / "decoded.pcm").open("wb") as stream:
        stream.truncate(size)
    if overflow:
        with pytest.raises(ValueError):
            reader.read_capture(path)
    else:
        assert isinstance(reader.read_capture(path), dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("overflow", [0, 2])
async def test_actual_runner_cleanup_uses_smaller_case_limit(tmp_path, overflow):
    f = Fixture(tmp_path)
    command = f.command
    limit = 95 * 48_000 * 2  # Legacy baseline: startup30 + media45 + tail20.

    async def with_pcm(argv, log_path, timeout=20):
        outcome = await command(argv, log_path, timeout)
        if len(argv) > 2 and argv[2] == "launch" and outcome["returncode"] == 0:
            pcm = f.container / "Documents/Replay" / str(UUID(f.run)) / "decoded.pcm"
            with pcm.open("wb") as stream:
                stream.truncate(limit + overflow)
        return outcome

    with patch.object(f, "command", side_effect=with_pcm):
        report = await runner.run_once(f.options, host=f, process=f)
    assert report["capture_budget"]["decoded_pcm_byte_limit"] == limit
    assert report["local_cleanup_complete"] is True
    assert (f.prepares, f.launches, f.cleanups) == (1, 1, 1)
    if overflow:
        assert report["native_output_error"]["code"] == "artifact_byte_bound"
        assert not (f.options.artifact_dir / "native-capture/decoded.pcm").exists()
    else:
        records = [row for row in report["native_output"] if Path(row["path"]).name == "decoded.pcm"]
        assert len(records) == 1 and records[0]["bytes"] == limit
    assert report["voice_qualification"]["status"] == "not_evaluated"
