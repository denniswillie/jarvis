"""Offline metadata/ownership/process tests. No ffprobe or media decoding runs."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.files.local_file import LocalFileError, VerifiedLocalFile, snapshot_file
from app.files.media_inspection import (MAX_STDERR_BYTES, MAX_STDOUT_BYTES,
                                        MediaInspectionError, inspect_mp4_file)
from app.files.transfer import (CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS,
                                CAPTURE_MAX_ENCODED_DURATION_MS, MP4_OUTPUT_MAX_BYTES,
                                FileTransferPolicy)


class FakeProcess:
    def __init__(self, stdout=b"", stderr=b"", *, running=False, returncode=0):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_data(stdout)
        self.stderr.feed_data(stderr)
        self.returncode = None if running else returncode
        self.kills = 0
        self.waits = 0
        self.done = asyncio.Event()
        if not running:
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.done.set()

    def kill(self):
        self.kills += 1
        self.returncode = -9
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.done.set()

    async def wait(self):
        self.waits += 1
        await self.done.wait()
        return self.returncode


class MediaInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.original = Path(self.directory.name).resolve() / "original.bin"
        self.content = b"offline fixture bytes, deliberately not an actual movie"
        self.original.write_bytes(self.content)
        self.local = await snapshot_file(self.original, max_bytes=1024)
        self.addCleanup(self.local.close)
        self.policy = FileTransferPolicy("mp4_output_v1", 32 * 1024**2)
        # This existing trusted path is never executed: subprocess creation is mocked.
        self.executable = str(Path(sys.executable).absolute())
        self.data = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "10.000000",
                                  "size": str(len(self.content)), "tags": {"major_brand": "isom"}},
                     "streams": [
                         {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1080,
                          "height": 1920, "duration": "10.000000", "disposition": {"attached_pic": 0}},
                         {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": "10.000000"}]}
        self.spawn = self.enterContext(patch("app.files.media_inspection.asyncio.create_subprocess_exec",
                                           new_callable=AsyncMock))

    def process(self, data=None, **kwargs):
        result = FakeProcess(json.dumps(self.data if data is None else data).encode(), **kwargs)
        self.spawn.return_value = result
        return result

    async def inspect(self, **kwargs):
        kwargs.setdefault("ffprobe_path", self.executable)
        kwargs.setdefault("timeout", 1.0)
        return await inspect_mp4_file(self.local, kwargs.pop("policy", self.policy), **kwargs)

    async def assert_rejected(self, data, code=None):
        self.process(data)
        with self.assertRaises(MediaInspectionError) as raised:
            await self.inspect()
        if code:
            self.assertEqual(raised.exception.code, code)

    async def test_exact_inspection_fixed_local_command_and_caller_ownership(self):
        process = self.process()
        result = await self.inspect()
        self.assertEqual(result.content_sha256, hashlib.sha256(self.content).hexdigest())
        self.assertEqual((result.size_bytes, result.duration_ms, result.width, result.height,
                          result.video_codec, result.audio_codec), (len(self.content), 10000, 1080, 1920, "h264", "aac"))
        args, kwargs = self.spawn.call_args
        self.assertEqual(args[0], self.executable)
        self.assertEqual(args[-2:], ("-i", str(self.local.path)))
        self.assertEqual(args[args.index("-protocol_whitelist") + 1], "file")
        self.assertEqual(args[args.index("-f") + 1], "mov")
        self.assertEqual(args[args.index("-enable_drefs") + 1], "0")
        self.assertEqual(args[args.index("-use_absolute_path") + 1], "0")
        self.assertEqual(kwargs["stdin"], asyncio.subprocess.DEVNULL)
        self.assertNotIn("shell", kwargs)
        self.assertNotIn("-select_streams", args)  # Unsupported tracks cannot be filtered away.
        self.assertFalse(self.local._closed)
        self.assertEqual(self.original.read_bytes(), self.content)
        self.assertGreaterEqual(process.waits, 1)
        self.assertEqual(process.kills, 0)

    async def test_silent_video_is_permitted_without_audio_claim(self):
        self.data["streams"].pop()
        self.process()
        self.assertIsNone((await self.inspect()).audio_codec)

    async def test_duration_rounds_up_using_longest_stream_without_precision_loss(self):
        self.data["streams"][1]["duration"] = "10.000000000000000000000000000001"
        self.process()
        self.assertEqual((await self.inspect()).duration_ms, 10001)

    async def test_capture_and_output_duration_limits_include_stream_duration(self):
        for target in [self.data["format"], *self.data["streams"]]:
            target["duration"] = f"{CAPTURE_MAX_DURATION_MS // 1000}.000001"
        await self.assert_rejected(self.data, "media_duration_exceeded")
        capture = FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES)
        self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()),
                                 FakeProcess(b"1799.980001,0.020000\n")]
        self.assertEqual((await self.inspect(policy=capture)).duration_ms, 1800001)
        self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()),
                                 FakeProcess(b"1800.100000,0.020000\n")]
        with self.assertRaisesRegex(MediaInspectionError, "media_duration_exceeded"):
            await self.inspect(policy=capture)

    async def test_capture_uses_packet_end_not_fragmented_container_estimate(self):
        self.data["format"]["duration"] = "61.165000"
        self.data["streams"][0]["duration"] = "60.975000"
        self.data["streams"][1]["duration"] = "60.992000"
        self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()),
            FakeProcess(b"0.190000,0.041667\n60.933333,0.041667\n60.970667,0.021333\n")]
        result = await self.inspect(policy=FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES))
        self.assertEqual(result.duration_ms, 60992)
        self.assertLessEqual(result.duration_ms, 60932 + 100)
        self.assertEqual(self.spawn.await_count, 2)
        self.assertIn("packet=pts_time,duration_time:packet_side_data=", self.spawn.call_args.args)

    async def test_capture_preserves_leading_gap_and_rejects_invalid_packet_times(self):
        capture = FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES)
        for packets, expected in ((b"5.000000,0.040000\n", 5040),
                                  (b"-0.021333,0.021333\n0.000000,0.020000\n", 20)):
            self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()), FakeProcess(packets)]
            self.assertEqual((await self.inspect(policy=capture)).duration_ms, expected)
        for packets in (b"", b"N/A,0.02\n", b"0.0,NaN\n", b"0.0,-0.02\n", b"0,0\n"):
            self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()), FakeProcess(packets)]
            with self.assertRaisesRegex(MediaInspectionError, "media_packet_timing_invalid"):
                await self.inspect(policy=capture)

    async def test_capture_packet_scan_uses_existing_timeout_and_cleanup(self):
        packets = FakeProcess(running=True)
        self.spawn.side_effect = [FakeProcess(json.dumps(self.data).encode()), packets]
        with self.assertRaisesRegex(MediaInspectionError, "media_probe_timeout"):
            await self.inspect(policy=FileTransferPolicy("mp4_capture_v1", CAPTURE_MAX_BYTES), timeout=0.02)
        self.assertEqual(packets.kills, 1)
        self.assertFalse(self.local._closed)

    async def test_nonfinite_nonpositive_or_missing_duration_rejected(self):
        for duration in (None, "NaN", "Infinity", "-Infinity", "N/A", "-1", "0", True, 10, "1e999999", " 1 "):
            with self.subTest(duration=duration):
                data = deepcopy(self.data)
                data["format"]["duration"] = duration
                await self.assert_rejected(data, "media_duration_invalid")
                data = deepcopy(self.data)
                data["streams"][0]["duration"] = duration
                await self.assert_rejected(data, "media_duration_invalid")

    async def test_missing_duplicate_or_unsupported_tracks_rejected(self):
        video, audio = self.data["streams"]
        cases = [[], [audio], [video, {**video, "index": 2}], [video, audio, {**audio, "index": 2}],
                 [video, {**audio, "index": 0}], [video, {**audio, "codec_name": "opus"}],
                 [{**video, "codec_name": "hevc"}], [video, {**audio, "codec_type": "subtitle"}],
                 [video, {**audio, "codec_type": "data"}], [{**video, "disposition": {"attached_pic": 1}}],
                 [{**video, "index": True}], [video, None]]
        for streams in cases:
            with self.subTest(streams=streams):
                await self.assert_rejected({**self.data, "streams": streams}, "media_streams_unsupported")

    async def test_quicktime_other_containers_unknown_brands_and_bad_dimensions_rejected(self):
        for key, value in (("format_name", "matroska,webm"), ("format_name", []),
                           ("tags", {"major_brand": "qt  "}), ("tags", {"major_brand": []}),
                           ("tags", {}), ("tags", {"major_brand": "3gp4"})):
            data = deepcopy(self.data)
            data["format"][key] = value
            await self.assert_rejected(data, "media_container_unsupported")
        for value in (None, True, "1080", 0, -1, 7681):
            data = deepcopy(self.data)
            data["streams"][0]["width"] = value
            await self.assert_rejected(data, "media_dimensions_invalid")

    async def test_reported_size_must_match_exact_bytes(self):
        for value in (None, "1000", "-1", True, len(self.content), "0" * 11):
            data = deepcopy(self.data)
            data["format"]["size"] = value
            await self.assert_rejected(data, "media_manifest_mismatch")

    async def test_byte_policy_and_invalid_input_fail_before_spawn(self):
        for policy in (FileTransferPolicy(), FileTransferPolicy("mp4_output_v1", len(self.content) - 1)):
            with self.assertRaisesRegex(MediaInspectionError, "media_size_exceeded"):
                await self.inspect(policy=policy)
        with self.assertRaisesRegex(MediaInspectionError, "media_inspection_invalid_input"):
            await inspect_mp4_file(object(), self.policy, ffprobe_path=self.executable)
        self.spawn.assert_not_called()

    async def test_hard_capture_and_output_byte_ceiling_cannot_be_widened(self):
        saved = self.local.size_bytes
        try:
            for name, bound in (("mp4_capture_v1", CAPTURE_MAX_BYTES), ("mp4_output_v1", MP4_OUTPUT_MAX_BYTES)):
                object.__setattr__(self.local, "size_bytes", bound + 1)
                with self.assertRaisesRegex(MediaInspectionError, "media_size_exceeded"):
                    await self.inspect(policy=FileTransferPolicy(name, CAPTURE_MAX_BYTES * 2))
        finally:
            object.__setattr__(self.local, "size_bytes", saved)
        self.spawn.assert_not_called()

    async def test_immutable_file_checked_before_and_after_process(self):
        checked = []
        original_iter = VerifiedLocalFile.iter_chunks
        async def observed(local):
            checked.append(local.path)
            async for block in original_iter(local):
                yield block
        self.process()
        with patch.object(VerifiedLocalFile, "iter_chunks", observed):
            await self.inspect()
        self.assertEqual(checked, [self.local.path, self.local.path])

    async def test_wrong_trusted_handle_digest_and_process_time_mutation_rejected(self):
        saved = self.local.content_sha256
        object.__setattr__(self.local, "content_sha256", "0" * 64)
        try:
            with self.assertRaisesRegex(MediaInspectionError, "media_manifest_mismatch"):
                await self.inspect()
        finally:
            object.__setattr__(self.local, "content_sha256", saved)
        self.spawn.assert_not_called()
        process = self.process()
        async def mutate(*args, **kwargs):
            os.chmod(self.local.path, 0o600)
            self.local.path.write_bytes(b"x" * len(self.content))
            return process
        self.spawn.side_effect = mutate
        with self.assertRaises(LocalFileError):
            await self.inspect()
        self.assertFalse(self.local._closed)
        self.assertEqual(self.original.read_bytes(), self.content)

    async def test_executable_and_timeout_require_explicit_bounded_inputs(self):
        for executable in ("ffprobe", "/missing/jarvis-ffprobe", self.directory.name, object(), "https://host/ffprobe"):
            with self.subTest(executable=executable), self.assertRaisesRegex(MediaInspectionError, "media_executable_invalid"):
                await self.inspect(ffprobe_path=executable)
        for timeout in (0, -1, 31, float("nan"), float("inf"), True, "1"):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(MediaInspectionError, "media_timeout_invalid"):
                await self.inspect(timeout=timeout)
        self.spawn.assert_not_called()

    async def test_invalid_json_duplicate_keys_and_nonzero_exit_are_content_free(self):
        for raw in (b"private input path", b'{"format":{},"format":{}}', b'{"bad":NaN}', b"\xff", b"[]"):
            self.spawn.return_value = FakeProcess(raw)
            with self.assertRaisesRegex(MediaInspectionError, "media_probe_invalid_metadata"):
                await self.inspect()
        for exitcode in (0, 1):
            self.process(stderr=b"secret path/provider error", returncode=exitcode)
            with self.assertRaisesRegex(MediaInspectionError, "^media_probe_failed$"):
                await self.inspect()

    async def test_start_failure_is_content_free(self):
        self.spawn.side_effect = OSError("private executable path")
        with self.assertRaisesRegex(MediaInspectionError, "^media_probe_start_failed$"):
            await self.inspect()

    async def test_stdout_and_stderr_overflow_kill_and_reap_once(self):
        for stdout, stderr in ((b"x" * (MAX_STDOUT_BYTES + 1), b""), (b"", b"x" * (MAX_STDERR_BYTES + 1))):
            process = FakeProcess(stdout, stderr, running=True)
            self.spawn.return_value = process
            with self.assertRaisesRegex(MediaInspectionError, "media_probe_output_exceeded"):
                await self.inspect()
            self.assertEqual(process.kills, 1)
            self.assertIsNotNone(process.returncode)
            self.assertGreaterEqual(process.waits, 1)
            self.assertTrue(process.stdout.at_eof())
            self.assertTrue(process.stderr.at_eof())
        self.assertEqual(self.spawn.await_count, 2)  # One process each; no retries.

    async def test_timeout_kills_reaps_and_preserves_caller_file(self):
        process = FakeProcess(running=True)
        self.spawn.return_value = process
        with self.assertRaisesRegex(MediaInspectionError, "media_probe_timeout"):
            await self.inspect(timeout=0.02)
        self.assertEqual(process.kills, 1)
        self.assertGreaterEqual(process.waits, 1)
        self.assertFalse(self.local._closed)

    async def test_cancel_kills_reaps_without_converting_cancellation_to_success(self):
        process = FakeProcess(running=True)
        self.spawn.return_value = process
        task = asyncio.create_task(self.inspect())
        while self.spawn.await_count == 0:
            await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(process.kills, 1)
        self.assertGreaterEqual(process.waits, 1)
        self.assertFalse(self.local._closed)

    async def test_pipe_error_still_reaps_and_does_not_expose_error_contents(self):
        process = FakeProcess(running=True)
        process.stdout.set_exception(OSError("private file path"))
        self.spawn.return_value = process
        with self.assertRaisesRegex(MediaInspectionError, "^media_probe_failed$"):
            await self.inspect()
        self.assertEqual(process.kills, 1)
        self.assertGreaterEqual(process.waits, 2)
        self.assertTrue(process.done.is_set())

    async def test_repeated_cancellation_cannot_abandon_reaping(self):
        cleanup_started, allow_reap = asyncio.Event(), asyncio.Event()
        class DelayedReap(FakeProcess):
            async def wait(self):
                self.waits += 1
                await self.done.wait()
                if self.returncode == -9:
                    cleanup_started.set()
                    await allow_reap.wait()
                return self.returncode
        process = DelayedReap(running=True)
        self.spawn.return_value = process
        task = asyncio.create_task(self.inspect())
        while self.spawn.await_count == 0:
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        allow_reap.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(process.kills, 1)
        self.assertGreaterEqual(process.waits, 1)
