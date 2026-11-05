"""Offline qualification-harness guards; no SQL, network, provider or codecs."""
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.local_file import snapshot_file
from app.files.storage import JarvisFileStorageClient
from app.plugins.media import MediaRendererBundle
from app.runtime.feature_config import ArtifactMediaConfiguration
from app.sandbox.models import SandboxError, SandboxPolicy
from app.sandbox.e2b_provider import E2BProvider
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_recipe import MediaRecipePolicy, RetainedRef, TemplatePin
from app.workspace.media_render import RendererBinding
from evals.second_brain.managed_media_qualification import (
    BYTE_LIMIT, OPERATION_KEY, STATUS_POLLS, OwnedCloudStorage, RunPlan, SingleVMProvider,
    audio_sample_evidence, bounded_process, drive_tools, execute, failure_location, inspect_output,
    output_metadata, recipe_for, retain, source_knowledge,
)


def audio_frames(samples=480000):
    half = samples // 2
    return ("#tb 0: 1/48000\n" +
            f"0, 0, 0, {half}, {half * 2}, " + "a" * 64 + "\n" +
            f"0, {half}, {half}, {samples - half}, {(samples - half) * 2}, " + "b" * 64 + "\n").encode()


class Probe:
    def __init__(self, root):
        self.root = root
        self.record = {"resources": [], "renderer": {"physical_template_id": "fixture-image"}}
        self.scopes = []
        self.provider = SimpleNamespace(create_bound=AsyncMock())
        self.persist_count = 0

    def persist(self):
        self.persist_count += 1


def metadata():
    return {"format": {"duration": "10.005"}, "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 720, "height": 1280,
         "avg_frame_rate": "24/1", "nb_read_frames": "240"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000"}]}


class ManagedMediaHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="managed-media-harness-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.root.chmod(0o700)
        self.binary = self.root / "never-executed"
        self.binary.write_bytes(b"offline fixture executable, never run")
        self.binary.chmod(0o700)
        self.binary_sha = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        receipt = self.root / "build.json"
        receipt.write_text(json.dumps({"status": "ready", "template_id": "fixture-image", "build_id": "fixture-build"}))
        self.template = TemplatePin(id="jarvis.clip.v1", version="fixture-v1", bundle_sha256="a" * 64)
        renderer = RendererBinding(template=self.template, physical_template_id="fixture-image", build_id="fixture-build",
            software_sha256="b" * 64, build_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            policy=MediaRecipePolicy(max_source_bytes=BYTE_LIMIT, max_source_duration_ms=60000, max_output_frames=720),
            max_output_bytes=BYTE_LIMIT)
        bundle = MediaRendererBundle(configuration_id=uuid4(), bundle_version="fixture-v1", bundle_sha256="c" * 64,
                                     renderer=renderer)
        guard = self.root / "guard.json"
        guard.write_text(json.dumps({"child_pid": os.getpid(), "child_pgid": os.getpgrp(), "parent_pid": os.getppid(),
                                     "hard_deadline_seconds": 300}))
        self.plan = RunPlan(directory=self.root, bundle=bundle,
            policy=SandboxPolicy(command_seconds=180, output_bytes=16384),
            media=ArtifactMediaConfiguration(storage_max_file_bytes=BYTE_LIMIT, ffprobe_path=str(self.binary),
                                            ffprobe_sha256=self.binary_sha),
            ffmpeg_path=self.binary, ffmpeg_sha256=self.binary_sha, build_receipt=receipt, watchdog_receipt=guard)
        self.recipe = recipe_for(RetainedRef(file_id=uuid4(), version_id="fixture-v1", content_sha256="d" * 64), self.template)

    async def local(self):
        path = self.root / "synthetic-bytes"
        path.write_bytes(b"not actual movie bytes")
        local = await snapshot_file(path, max_bytes=100)
        self.addCleanup(local.close)
        return local

    async def test_missing_opt_in_has_no_side_effects(self):
        with patch("evals.second_brain.managed_media_qualification.Probe") as probe:
            for flag in (False, None, 1, "yes"):
                with self.assertRaises(ValueError):
                    await execute(self.plan, execute_once=flag)
            probe.assert_not_called()
        self.assertFalse((self.root / "execute-once").exists())

    async def test_normal_worker_construction_requires_same_source_retrieval_port(self):
        pool = object()
        source = SimpleNamespace(owner='offline-fixture', area=uuid4(), sources=SimpleNamespace(pool=pool))
        context = SimpleNamespace(task=SimpleNamespace(owner_id=source.owner,
            knowledge_area_id=source.area, session_id=None), tasks=SimpleNamespace(pool=pool))
        knowledge = source_knowledge(source)
        with self.assertRaises(ValueError):
            knowledge.context_for('another-owner')
        prepared = AsyncMock(return_value=SimpleNamespace(tools='offline-end-boundary'))
        with patch('app.tasks.readonly_worker.is_readonly', AsyncMock(return_value=False)), \
             patch('app.tasks.worker.HermesContextFactory', return_value=prepared):
            # All ordinary WorkerContextFactory knowledge constructors run; only
            # the later database-dependent Hermes preparation is replaced.
            await WorkerContextFactory(pool, knowledge=knowledge)(context)
            prepared.assert_awaited_once()
            prepared.reset_mock()
            knowledge.retrieval = None
            try:
                await WorkerContextFactory(pool, knowledge=knowledge)(context)
            except AttributeError as error:
                self.assertEqual(error.name, 'sources')
                receipt = failure_location(error)
            else:
                self.fail('The original incomplete fixture must reproduce its constructor failure')
            prepared.assert_not_awaited()
        self.assertEqual(receipt['error_type'], 'AttributeError')
        self.assertEqual(receipt['frames'][-1]['file'], 'app/knowledge/memory_backend.py')
        self.assertEqual(receipt['origin'], 'project_exception_origin')
        self.assertNotIn('offline-fixture', json.dumps(receipt))

    def test_failure_locations_never_capture_messages_locals_or_external_paths(self):
        secret = 'PRIVATE-TOKEN-MUST-NOT-APPEAR'
        try:
            def fail(depth):
                if depth:
                    return fail(depth - 1)
                raise RuntimeError(secret)
            fail(40)
        except RuntimeError as error:
            receipt = failure_location(error)
        self.assertEqual(receipt['error_type'], 'RuntimeError')
        self.assertLessEqual(len(receipt['frames']), 8)
        self.assertEqual(receipt['frames_scanned'], 32)
        self.assertTrue(receipt['truncated'])
        self.assertNotIn(secret, json.dumps(receipt))
        self.assertNotIn('/Users/', json.dumps(receipt))

    async def test_fake_clients_cannot_be_called_a_live_run(self):
        with self.assertRaisesRegex(ValueError, "real configured clients"):
            await execute(self.plan, execute_once=True, provider=SimpleNamespace(), storage=SimpleNamespace(),
                          database_url="fixture", observe=AsyncMock())

    def test_reviewed_plan_limits_and_watchdog_are_explicit(self):
        self.plan.validate()
        for change in ({"internet_access": True}, {"command_seconds": 181}, {"lifetime_seconds": 301},
                       {"output_bytes": 16385}, {"template": "desktop"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(self.plan, policy=self.plan.policy.model_copy(update=change)).validate()
        guard = json.loads(self.plan.watchdog_receipt.read_text())
        guard["child_pid"] += 1
        self.plan.watchdog_receipt.write_text(json.dumps(guard))
        with self.assertRaisesRegex(ValueError, "watchdog"):
            self.plan.validate()

    def test_build_and_binary_identity_cannot_drift(self):
        self.binary.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "decoder changed"):
            self.plan.validate()
        self.plan.build_receipt.write_text('{}')
        with self.assertRaisesRegex(ValueError, "build receipt"):
            self.plan.validate()

    async def test_one_planned_create_is_consumed_even_if_reply_is_lost(self):
        probe = Probe(self.root)
        probe.provider.create_bound.side_effect = TimeoutError()
        provider = SingleVMProvider(probe)
        scope = SimpleNamespace(model_dump=lambda **_: {"allocation_id": str(uuid4())})
        with self.assertRaises(TimeoutError):
            await provider.create_bound(scope, self.plan.policy, template_id="fixture-image")
        self.assertEqual(probe.record["resources"][0]["phase"], "planned")
        with self.assertRaisesRegex(SandboxError, "single_vm_consumed"):
            await provider.create_bound(scope, self.plan.policy, template_id="fixture-image")
        probe.provider.create_bound.assert_awaited_once()
        self.assertEqual(len(probe.scopes), 1)

    async def test_unbound_and_wrong_image_never_allocate(self):
        probe = Probe(self.root)
        provider = SingleVMProvider(probe)
        with self.assertRaises(SandboxError):
            await provider.create(None, None)
        with self.assertRaises(SandboxError):
            await provider.create_bound(None, self.plan.policy, template_id="different")
        probe.provider.create_bound.assert_not_awaited()
        self.assertFalse(probe.record["resources"])

    async def test_provider_stage_is_recorded_before_effect_and_unknown_never_retried(self):
        probe = Probe(self.root)
        failure = SandboxError('sandbox_provider_outcome_uncertain')
        failure.diagnostic = {'stage':'privileged_command', 'outcome':'unknown', 'error_type':'TimeoutError',
                              'stdout':'PRIVATE-BYTES', 'stderr':'PRIVATE-TOKEN', 'unknown':'PRIVATE'}
        async def fail(_handle):
            self.assertEqual(probe.record['provider_stages'][0]['state'], 'started')
            raise failure
        probe.provider.prepare = AsyncMock(side_effect=fail)
        with self.assertRaises(SandboxError) as caught:
            await SingleVMProvider(probe).prepare(object())
        self.assertIs(caught.exception, failure)
        probe.provider.prepare.assert_awaited_once()
        stage = probe.record['provider_stages'][0]
        self.assertEqual(stage['state'], 'failed')
        self.assertEqual(stage['provider_diagnostic']['outcome'], 'unknown')
        self.assertNotIn('PRIVATE', json.dumps(stage))

    async def test_storage_manifests_before_effect_and_refuses_duplicate_or_foreign_key(self):
        probe = Probe(self.root)
        cloud = SimpleNamespace(bucket="fixture", upload_file=AsyncMock(side_effect=TimeoutError()), upload_bytes=AsyncMock())
        tracked = OwnedCloudStorage(cloud, probe, "fixture-owner")
        local = await self.local()
        path = "owners/fixture-owner/sessions/example/files/example/content"
        with self.assertRaises(TimeoutError):
            await tracked.upload_file(path=path, local_file=local, content_type="video/mp4")
        self.assertEqual(probe.record["storage_objects"][0]["phase"], "unknown")
        with self.assertRaises(ValueError):
            await tracked.upload_file(path=path, local_file=local, content_type="video/mp4")
        with self.assertRaises(ValueError):
            await tracked.upload_bytes(path="owners/other/content", content=b'{}', content_type="application/json")
        cloud.upload_file.assert_awaited_once()
        cloud.upload_bytes.assert_not_awaited()

    async def test_exact_private_artifact_is_preserved_and_never_overwritten(self):
        probe = Probe(self.root)
        local = await self.local()
        path = await retain(probe, "output.mp4", local)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), local.content_sha256)
        self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        with self.assertRaises(FileExistsError):
            await retain(probe, "output.mp4", local)
        self.assertFalse(local._closed)

    async def test_script_drives_normal_tools_once_then_same_key_replay(self):
        output = {"file": "exact-receipt"}
        responses = [self.result("submitted"), self.result("rendered"),
                     self.result("published", output=output), self.result("published", output=output)]
        bridge = SimpleNamespace(invoke=AsyncMock(side_effect=responses))
        record = []
        sleep = AsyncMock()
        await drive_tools(bridge, self.recipe, lambda *args: record.append(args), sleep=sleep)
        calls = bridge.invoke.await_args_list
        self.assertEqual([call.args[0] for call in calls], ["media_render", "media_status", "media_publish", "media_publish"])
        self.assertTrue(all(call.args[1]["operation_key"] == OPERATION_KEY for call in calls))
        self.assertEqual(len(record), 4)
        sleep.assert_awaited_once_with(5)

    async def test_unknown_render_never_restarts_and_stops_at_poll_bound(self):
        bridge = SimpleNamespace(invoke=AsyncMock(return_value=self.result("unknown")))
        with self.assertRaisesRegex(ValueError, "observation budget"):
            await drive_tools(bridge, self.recipe, lambda *_: None, sleep=AsyncMock())
        names = [call.args[0] for call in bridge.invoke.await_args_list]
        self.assertEqual(names.count("media_render"), 1)
        self.assertEqual(names.count("media_status"), STATUS_POLLS)
        self.assertNotIn("media_publish", names)

    async def test_unresolved_or_changed_publication_is_not_a_pass(self):
        cases = [[self.result("rendered"), self.result("accepted")],
                 [self.result("rendered"), self.result("published", output={"id": "one"}),
                  self.result("published", output={"id": "two"})]]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                await drive_tools(SimpleNamespace(invoke=AsyncMock(side_effect=values)), self.recipe, lambda *_: None)

    @staticmethod
    def result(state, **extra):
        return {"status": "success", "media_operation": {"state": state, **extra}}

    def test_output_contract_rejects_missing_audio_wrong_frames_fps_duration_and_dimensions(self):
        self.assertEqual(output_metadata(json.dumps(metadata())), {
            "video_frames": 240, "fps": 24, "width": 720, "height": 1280,
            "duration_seconds": "2001/200", "video_codec": "h264", "audio_codec": "aac", "sample_rate": 48000})
        for field, value in (("nb_read_frames", "239"), ("avg_frame_rate", "30/1"), ("width", 1080), ("codec_name", "hevc")):
            raw = metadata(); raw["streams"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                output_metadata(json.dumps(raw))
        for duration in ("9.99", "10.1", "NaN"):
            raw = metadata(); raw["format"]["duration"] = duration
            with self.assertRaises(ValueError):
                output_metadata(json.dumps(raw))
        raw = metadata(); raw["streams"].pop()
        with self.assertRaises(ValueError):
            output_metadata(json.dumps(raw))

    async def test_metadata_alone_never_qualifies_decode(self):
        local = await self.local()
        for tail in (b'frame=240\nout_time_us=10005000\nprogress=continue\n',
                     b'frame=239\nout_time_us=10005000\nprogress=end\n'):
            with patch('evals.second_brain.managed_media_qualification.bounded_process',
                       AsyncMock(side_effect=[json.dumps(metadata()).encode(), tail])) as run:
                with self.assertRaises(ValueError):
                    await inspect_output(self.plan, local)
                self.assertEqual(run.await_count, 2)

    async def test_decoder_arguments_and_exact_output_are_checked(self):
        local = await self.local()
        with patch('evals.second_brain.managed_media_qualification.bounded_process',
                   AsyncMock(side_effect=[json.dumps(metadata()).encode(),
                       b'frame=240\nout_time_us=10005000\nprogress=end\n', audio_frames()])) as run:
            result = await inspect_output(self.plan, local)
            self.assertTrue(result['full_video_audio_decode'])
            self.assertFalse(result['playback_or_content_review'])
            command = run.await_args_list[1].args[0]
            self.assertIn('-xerror', command)
            self.assertEqual(command[command.index('-protocol_whitelist') + 1], 'file')
            self.assertIn('0:v:0', command)
            self.assertIn('0:a:0', command)
            self.assertEqual(result['content_sha256'], local.content_sha256)
            audio_command = run.await_args_list[2].args[0]
            self.assertEqual(audio_command[audio_command.index('-map') + 1], '0:a:0')
            self.assertNotIn('0:v:0', audio_command)
            self.assertEqual(audio_command[audio_command.index('-af') + 1], 'asetpts=N/SR/TB')
            self.assertEqual(audio_command[audio_command.index('-c:a') + 1], 'pcm_s16le')
            self.assertEqual(result['decoded_audio_samples'], 480000)
            self.assertEqual(result['decoded_audio_duration_seconds'], '10')

    async def test_successful_video_and_audio_metadata_cannot_mask_empty_audio(self):
        local = await self.local()
        with patch('evals.second_brain.managed_media_qualification.bounded_process',
                   AsyncMock(side_effect=[json.dumps(metadata()).encode(),
                       b'frame=240\nout_time_us=10005000\nprogress=end\n', b'#tb 0: 1/48000\n'])) as run:
            with self.assertRaisesRegex(ValueError, 'Independent audio decoding'):
                await inspect_output(self.plan, local)
            self.assertEqual(run.await_count, 3)

    def test_audio_proof_rejects_gaps_wrong_clock_short_empty_and_bad_packets(self):
        self.assertEqual(audio_sample_evidence(audio_frames())['decoded_audio_packets'], 2)
        for raw in (b'', audio_frames(samples=1024), audio_frames(samples=600000),
                    audio_frames().replace(b'1/48000', b'1/24000'),
                    audio_frames().replace(b'240000, 240000', b'250000, 250000'),
                    audio_frames().replace(b'480000,', b'480002,'),
                    audio_frames() + b'#tb 0: 1/48000\n'):
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                audio_sample_evidence(raw)

    async def test_bounded_process_reaps_owned_child_after_timeout_and_overflow(self):
        pids = []
        spawn = asyncio.create_subprocess_exec
        async def observed(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            pids.append(process.pid)
            return process
        with patch('evals.second_brain.managed_media_qualification.asyncio.create_subprocess_exec', observed):
            with self.assertRaises(TimeoutError):
                await bounded_process([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(10)'], seconds=0.1)
            with self.assertRaisesRegex(ValueError, 'output exceeded'):
                await bounded_process([sys.executable, '-I', '-S', '-c',
                    'import sys; sys.stdout.write("x"*100000); sys.stdout.flush()'], seconds=2)
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    async def test_nonzero_decoder_and_stderr_do_not_pass(self):
        for code in ('raise SystemExit(2)', 'import sys; sys.stderr.write("error")'):
            with self.assertRaisesRegex(ValueError, 'decoder failed'):
                await bounded_process([sys.executable, '-I', '-S', '-c', code], seconds=2)

    async def test_failed_scenario_still_attempts_external_and_sql_cleanup_and_cannot_reexecute(self):
        # Real class identities, no configured/networked instances: all effect
        # boundaries are explicit doubles, so this cannot qualify integration.
        provider = object.__new__(E2BProvider)
        cloud = object.__new__(JarvisFileStorageClient)
        pool = SimpleNamespace()
        source = SimpleNamespace(pool=pool, owner='fixture-owner', area=uuid4(),
            sources=SimpleNamespace(pool=pool), fixture=SimpleNamespace(path=self.root / 'database.json'))
        manager = SimpleNamespace(__aenter__=AsyncMock(return_value=source), __aexit__=AsyncMock())
        snapshot = SimpleNamespace(close=lambda: None)
        module = 'evals.second_brain.managed_media_qualification.'
        with patch(module + 'source_fixture', return_value=manager), \
             patch(module + 'snapshot_file', AsyncMock(return_value=snapshot)), \
             patch(module + '_scenario', AsyncMock(side_effect=RuntimeError('scenario failed'))), \
             patch(module + 'cleanup_fixture', AsyncMock(side_effect=RuntimeError('cleanup failed'))) as native, \
             patch(module + 'OwnedCloudStorage.cleanup', AsyncMock()) as objects:
            with self.assertRaisesRegex(RuntimeError, 'scenario failed'):
                await execute(self.plan, execute_once=True, database_url='fixture-only',
                              provider=provider, storage=cloud, observe=AsyncMock())
            native.assert_awaited_once()
            objects.assert_awaited_once()
            manager.__aexit__.assert_awaited_once_with(None, None, None)
            manifests = list(self.root.glob('jarvis-e2b-*/manifest.json'))
            self.assertEqual(len(manifests), 1)
            record = json.loads(manifests[0].read_text())
            self.assertFalse(record['qualification']['passed'])
            self.assertIn('cleanup_failure', record)
            with self.assertRaises(FileExistsError):
                await execute(self.plan, execute_once=True, database_url='fixture-only',
                              provider=provider, storage=cloud, observe=AsyncMock())
            manager.__aenter__.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
