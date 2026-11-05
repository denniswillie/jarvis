"""Provider doubles and fixed-program checks; no VM or actual codec invoked."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.sandbox.media_decode import (
    DECODE_MEDIA_INPUT, decode_request, parse_decode_receipt,
)
from app.sandbox.models import SandboxError
from tests import test_sandbox_media_materialization as materialization_tests


class MediaDecodeProviderTests(unittest.IsolatedAsyncioTestCase):
    # Reuse its existing SDK-double fixture, not its test methods/runs.
    setUp = materialization_tests.MediaMaterializationProviderTests.setUp

    def request(self):
        source = self.plan.recipe.recipe.sources[0].file
        size = self.plan.staging()["size_bytes"]
        return decode_request(self.handle, input_path=self.plan.input_path,
            content_sha256=source.content_sha256, size_bytes=size,
            binding=self.plan.binding, budget_seconds=57)

    def receipt(self):
        request = self.request()
        return {"format": "jarvis.media-decode.v1", **{key: request[key] for key in
            ("sandbox_id", "allocation_id", "input_path", "content_sha256", "size_bytes", "software_sha256")},
            "duration_ms": 30000, "video_streams": 1, "audio_streams": 1,
            "video_decoded": True, "audio_decoded": True, "video_frames": 720, "audio_duration_us": 30000000,
            "inspection_timing": {"inventory_ms": 39000, "metadata_ms": 20, "video_decode_ms": 2000, "audio_decode_ms": 50, "total_ms": 41070}}

    async def decode(self, **changes):
        request = self.request()
        kwargs = {key: request[key] for key in ("input_path", "content_sha256", "size_bytes")}
        kwargs.update(binding=self.plan.binding)
        kwargs.update(changes)
        return await self.provider.decode_media_render_input(self.handle, **kwargs)

    async def test_exact_typed_receipt_and_fixed_isolated_launch(self):
        self.result.stdout = json.dumps(self.receipt())
        result = await self.decode()
        self.assertEqual(result.model_dump(mode="json"), self.receipt())
        inner = shlex.split(shlex.split(self.run.call_args.args[0])[-1])
        self.assertEqual(inner[:5], ["exec", "/usr/bin/python3", "-I", "-S", "-c"])
        self.assertEqual(inner[5], DECODE_MEDIA_INPUT)
        self.assertEqual(json.loads(inner[6]), self.request())
        self.assertEqual(self.run.call_args.kwargs["timeout"], 60)
        self.assertEqual(self.run.call_args.kwargs["request_timeout"], 20)
        self.assertEqual(self.run.call_args.kwargs["user"], "root")
        self.run.assert_awaited_once()
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()

    async def test_absent_audio_is_explicit_not_a_successful_audio_decode(self):
        self.result.stdout = json.dumps(self.receipt() | {
            "audio_streams": 0, "audio_decoded": False, "audio_duration_us": 0})
        result = await self.decode()
        self.assertEqual(result.audio_streams, 0)
        self.assertFalse(result.audio_decoded)

    async def test_inspection_respects_smaller_admitted_policy_without_changing_control_timeout(self):
        self.policy = self.policy.model_copy(update={"command_seconds": 10})
        self.handle = self.handle.model_copy(update={"policy": self.policy})
        self.raw.metadata = self.scope.metadata(self.policy)
        receipt = self.receipt()
        receipt['inspection_timing'] = dict(inventory_ms=3000, metadata_ms=10,
            video_decode_ms=1000, audio_decode_ms=20, total_ms=4030)
        self.result.stdout = json.dumps(receipt)
        await self.decode()
        self.assertEqual(self.run.call_args.kwargs['timeout'], 10)
        self.assertEqual(self.run.call_args.kwargs['request_timeout'], 20)
        inner = shlex.split(shlex.split(self.run.call_args.args[0])[-1])
        self.assertEqual(json.loads(inner[6])['budget_seconds'], 7)

    async def test_bad_identity_path_limits_or_binding_precede_sdk_effects(self):
        for changes in ({"input_path": "/tmp/input.mp4"}, {"input_path": self.plan.input_path + "/../x"},
                        {"content_sha256": "x"}, {"size_bytes": True}, {"size_bytes": 67108865},
                        {"binding": None}, {"binding": self.plan.binding.model_copy(update={"physical_template_id": "other"})}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                await self.decode(**changes)
        self.api.get_info.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_actual_image_owner_and_network_checked_before_decode(self):
        for change in ({"template_id": "other"}, {"metadata": {}}, {"allow_internet_access": True}):
            before = {key: getattr(self.raw, key) for key in change}
            for key, value in change.items():
                setattr(self.raw, key, value)
            with self.subTest(change=change), self.assertRaises((SandboxError, ValueError)):
                await self.decode()
            for key, value in before.items():
                setattr(self.raw, key, value)
        self.api.connect.assert_not_awaited()
        self.run.assert_not_awaited()

    async def test_mismatched_or_fabricated_success_receipts_are_rejected(self):
        changes = [{"software_sha256": "0" * 64}, {"input_path": "/tmp/x"},
            {"sandbox_id": "other"}, {"allocation_id": "00000000-0000-0000-0000-000000000000"},
            {"size_bytes": True}, {"size_bytes": self.request()["size_bytes"] + 1},
            {"content_sha256": "0" * 64}, {"duration_ms": 60001}, {"video_frames": 0},
            {"video_decoded": False}, {"video_decoded": 1}, {"audio_decoded": False},
            {"audio_duration_us": 0}, {"audio_streams": 2}, {"extra": True}]
        invalid = [json.dumps(self.receipt() | change) for change in changes]
        invalid += [json.dumps(self.receipt() | {'inspection_timing': None}),
                    json.dumps(self.receipt() | {'inspection_timing': dict(
                        inventory_ms=58000, metadata_ms=0, video_decode_ms=0, audio_decode_ms=0, total_ms=58000)})]
        invalid += ["null", "{}", "x" * 8193, json.dumps(self.receipt())[:-1] + ',"video_decoded":true}']
        for raw in invalid:
            self.run.reset_mock()
            self.result.stdout = raw
            with self.subTest(raw=raw), self.assertRaises(SandboxError) as caught:
                await self.decode()
            self.assertEqual(caught.exception.code, "sandbox_media_decode_receipt_invalid")
            self.run.assert_awaited_once()

    async def test_nonzero_lost_ack_and_cancellation_never_retry_or_confirm(self):
        self.result.stdout = json.dumps(self.receipt())
        self.result.exit_code = 2
        with self.assertRaises(SandboxError):
            await self.decode()
        self.result.exit_code = 0
        for error in (TimeoutError("lost acknowledgement"), asyncio.CancelledError()):
            self.run.reset_mock()
            self.run.side_effect = error
            with self.assertRaises((SandboxError, asyncio.CancelledError)):
                await self.decode()
            self.run.assert_awaited_once()
        self.api.create.assert_not_awaited()
        self.api.kill.assert_not_awaited()


class MediaDecodeProgramTests(unittest.TestCase):
    def setUp(self):
        self.program = {"__name__": "fixed_program_test"}
        exec(compile(DECODE_MEDIA_INPUT, "<fixed-decode-program>", "exec"), self.program)
        self.request = {"format": "jarvis.media-decode-request.v1", "sandbox_id": "fixture-vm",
            "allocation_id": "00000000-0000-0000-0000-000000000001",
            "input_path": "/home/jarvis/inputs/" + "a" * 64,
            "content_sha256": hashlib.sha256(b"fixture-bytes").hexdigest(), "size_bytes": 13,
            "software_sha256": "b" * 64, "max_duration_ms": 60000, "budget_seconds": 17}
        self.meta = {"format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "size": "13",
                     "duration": "30.000", "tags": {"major_brand": "isom"}}, "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 720,
             "height": 1280, "duration": "30.000", "disposition": {"attached_pic": 0}},
            {"index": 1, "codec_type": "audio", "codec_name": "aac", "duration": "30.000"}]}
        self.calls = []

    def fake_process(self, args, deadline, limit=65536):
        self.calls.append(args)
        if args[0] == "/usr/bin/python3":
            return json.dumps({"software_sha256": self.request["software_sha256"], "render_qualified": False}).encode()
        if args[0].endswith("/ffprobe"):
            return json.dumps(self.meta).encode()
        if args[args.index("-map") + 1] == "0:v:0":
            return b"frame=720\nout_time_us=30000000\nprogress=end\n"
        return b"out_time_us=30000000\nprogress=end\n"

    def perform(self, run=None):
        return self.program["perform_decode"](self.request, time.monotonic() + 5, run or self.fake_process)

    def test_video_and_audio_must_each_run_full_decode(self):
        receipt = self.perform()
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(receipt["video_frames"], 720)
        self.assertEqual(receipt["audio_duration_us"], 30000000)
        self.assertEqual(parse_decode_receipt(json.dumps(receipt), self.request).duration_ms, 30000)
        for command in self.calls[2:]:
            self.assertIn("-xerror", command)
            self.assertIn("explode", command)
            self.assertEqual(command[command.index("-protocol_whitelist") + 1], "file")
            self.assertNotIn("-t", command)
            self.assertNotIn("-frames:v", command)
            self.assertEqual(command[-3:], ["-f", "null", "/dev/null"])

    def test_decode_uses_supported_raw_sinks_without_implicit_encoder_or_stream_copy(self):
        def limited_encoder_build(args, deadline, limit=65536):
            if args[0].endswith("/ffmpeg"):
                flag, codec = ("-c:v", "rawvideo") if "0:v:0" in args else ("-c:a", "pcm_s16le")
                # Reproduce the installed Remotion build's unavailable null
                # default; metadata or stream copy cannot satisfy this decode.
                if flag not in args or args[args.index(flag) + 1] != codec:
                    raise self.program["DecodeFailure"]("decode_process_failed")
            return self.fake_process(args, deadline, limit)
        receipt = self.perform(limited_encoder_build)
        self.assertTrue(receipt["video_decoded"])
        self.assertTrue(receipt["audio_decoded"])
        self.assertEqual(receipt["video_frames"], 720)
        self.assertEqual(receipt["audio_duration_us"], 30000000)

    def test_no_audio_runs_no_audio_process_and_returns_absence(self):
        self.meta["streams"].pop()
        receipt = self.perform()
        self.assertEqual(len(self.calls), 3)
        self.assertFalse(receipt["audio_decoded"])
        self.assertEqual(receipt["audio_duration_us"], 0)

    def test_wrong_actual_software_stops_before_probe_or_decode(self):
        def run(args, *_):
            self.calls.append(args)
            return json.dumps({"software_sha256": "c" * 64, "render_qualified": False}).encode()
        with self.assertRaisesRegex(ValueError, "software_identity_mismatch"):
            self.perform(run)
        self.assertEqual(len(self.calls), 1)

    def test_bad_metadata_prevents_decode(self):
        self.meta["streams"][1]["codec_name"] = "unknown"
        with self.assertRaises(ValueError):
            self.perform()
        self.assertEqual(len(self.calls), 2)

    def test_zero_frames_missing_end_or_audio_failure_cannot_be_success(self):
        for raw in (b"frame=0\nprogress=end\n", b"frame=12\nprogress=continue\n", b"frame=-1\nprogress=end\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.program["progress"](raw, "frame")
        def run(args, deadline, limit=65536):
            if "0:a:0" in args:
                raise self.program["DecodeFailure"]("decode_process_failed")
            return self.fake_process(args, deadline, limit)
        with self.assertRaises(ValueError):
            self.perform(run)

    def test_actual_input_hash_permissions_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ("a" * 64)
            path.write_bytes(b"fixture-bytes")
            path.chmod(0o444)
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                check = self.program["verify_input_at"]
                self.assertIsInstance(check(fd, self.request, time.monotonic() + 2, os.getuid()), tuple)
                with self.assertRaises(ValueError):
                    check(fd, self.request | {"content_sha256": "c" * 64}, time.monotonic() + 2, os.getuid())
                path.chmod(0o644)
                with self.assertRaises(ValueError):
                    check(fd, self.request, time.monotonic() + 2, os.getuid())
                path.rename(root / "original")
                path.symlink_to(root / "original")
                with self.assertRaises(OSError):
                    check(fd, self.request, time.monotonic() + 2, os.getuid())
                self.assertEqual((root / "original").read_bytes(), b"fixture-bytes")
            finally:
                os.close(fd)

    def test_production_entry_drops_authority_then_rechecks_exact_input(self):
        state = {"uid": 0, "gid": 0, "groups": [0]}
        events = []
        def set_value(key, value):
            events.append(key)
            state[key] = value
        self.program["os"] = SimpleNamespace(
            geteuid=lambda: state["uid"], getuid=lambda: state["uid"],
            getegid=lambda: state["gid"], getgid=lambda: state["gid"], getgroups=lambda: state["groups"],
            setgroups=lambda v: set_value("groups", v), setgid=lambda v: set_value("gid", v),
            setuid=lambda v: set_value("uid", v))
        self.program["pwd"] = SimpleNamespace(getpwnam=lambda _: SimpleNamespace(pw_uid=1001, pw_gid=1001))
        self.program["signal"] = SimpleNamespace(SIGTERM=15, SIGINT=2, signal=lambda *_: None)
        self.program["verify_inventory_entrypoint"] = lambda: events.append("entrypoint")
        observed = iter([("same",), ("same",)])
        def check_input(*_):
            events.append("source")
            return next(observed)
        self.program["verify_input"] = check_input
        def decode(*_):
            self.assertEqual(state, {"uid": 1001, "gid": 1001, "groups": []})
            events.append("decode")
            return {"fixture": True}
        self.program["perform_decode"] = decode
        self.assertEqual(self.program["main"](self.request), {"fixture": True})
        self.assertEqual(events, ["source", "entrypoint", "groups", "gid", "uid", "decode", "source"])
        state.update(uid=0, gid=0, groups=[0])
        observed = iter([("before",), ("changed",)])
        with self.assertRaisesRegex(ValueError, "input_changed"):
            self.program["main"](self.request)

    def test_actual_subprocess_output_bound_and_timeout_reap_owned_children(self):
        real_popen = subprocess.Popen
        processes = []
        def tracked(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(self.program["subprocess"], "Popen", tracked):
            with self.assertRaisesRegex(ValueError, "process_output_limit"):
                self.program["run_bounded"]([sys.executable, "-I", "-S", "-c", "print('x'*10000)"],
                                             time.monotonic() + 2, 32)
            with self.assertRaisesRegex(ValueError, "decode_timeout"):
                self.program["run_bounded"]([sys.executable, "-I", "-S", "-c", "import time;time.sleep(10)"],
                                             time.monotonic() + 0.1)
        self.assertEqual(len(processes), 2)
        for process in processes:
            self.assertIsNotNone(process.poll())
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)

    def test_actual_subprocess_exit_and_stderr_are_checked(self):
        for code in ("raise SystemExit(2)", "import sys;print('decoder error',file=sys.stderr)"):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "decode_process_failed"):
                self.program["run_bounded"]([sys.executable, "-I", "-S", "-c", code], time.monotonic() + 2)
        self.assertEqual(self.program["run_bounded"]([sys.executable, "-I", "-S", "-c", "print('done')"],
                                                    time.monotonic() + 2), b"done\n")
