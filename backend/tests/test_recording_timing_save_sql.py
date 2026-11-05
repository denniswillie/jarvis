"""Real SQL/admission/publication and FFmpeg; bounded in-memory object storage."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.files.local_file import snapshot_file, _snapshot_chunks
from app.files.media_inspection import inspect_mp4_file
from app.files.models import StoredFileObject
from app.files.recording_ingress import RecordingIngress, RecordingUpload
from app.files.storage import JarvisFileStorageClient
from app.files.transfer import FileTransferPolicy
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission
from app.recordings.admission import RecordingAdmission
from app.recordings.timing import RecordingTimingOptions
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class ObjectStorage:
    """Object keys, unlike filesystem paths, may also be prefixes of other keys."""
    bucket = "timing-fixture"
    build_file_path = JarvisFileStorageClient.build_file_path

    def __init__(self):
        self.objects = {}
        self.uploads = 0

    async def upload_file(self, *, path, local_file, content_type):
        if path in self.objects:
            raise AssertionError("Unexpected overwrite")
        self.objects[path] = b"".join([chunk async for chunk in local_file.iter_chunks()])
        self.uploads += 1
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=local_file.size_bytes)

    async def read_object_file(self, *, bucket, path, max_bytes):
        assert bucket == self.bucket
        if path not in self.objects:
            return None
        async def chunks():
            yield self.objects[path]
        return await _snapshot_chunks(chunks(), max_bytes=max_bytes)

    async def delete_object(self, *, bucket, path):
        assert bucket == self.bucket
        self.objects.pop(path, None)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit isolated database fixture required")
class TimingSaveSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_corrected_save_recovery_single_file_and_fallback(self):
        temporary = tempfile.TemporaryDirectory(prefix="recording-timing-sql-")
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name).resolve() / "fixture.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-n",
            "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p",
            "-video_track_timescale", "600", "-c:a", "aac", "-avoid_negative_ts", "make_zero",
            str(source)], check=True, timeout=20)
        policy = FileTransferPolicy("mp4_capture_v1", 1_000_000)
        probe = str(Path(shutil.which("ffprobe")).resolve())
        async def inspector(local, selected):
            return await inspect_mp4_file(local, selected, ffprobe_path=probe, timeout=10)
        original = await snapshot_file(source, max_bytes=policy.max_bytes)
        self.addCleanup(original.close)
        measured = await inspector(original, policy)
        upload = RecordingUpload(filename="fixture.mp4", description="Isolated timing save test",
            size_bytes=original.size_bytes, content_sha256=original.content_sha256,
            client_report={"generation_id": str(uuid4()), "timeline_end_ms": measured.duration_ms,
                           "video_frames": 48, "source_audio_frames": 96000, "padded_audio_frames": 0})
        f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        tasks = TaskRepository(f.pool, TaskLimits())
        selections = SessionKnowledgeAreas(f.sources, tasks, f.area)
        admission = RecordingAdmission(f.sources, tasks, selections, owner_id=f.owner)
        storage = ObjectStorage()
        ingress = RecordingIngress(f.sources, storage, admission, policy=policy, inspect_file=inspector,
            timing_options=RecordingTimingOptions(video_delay_ms=1000, timeout_seconds=20))
        async with f.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)

        async def admitted_upload():
            command = CommandAdmission(owner_id=f.owner, session_id=session,
                provider_session_key="timing-sql", provider_call_id=uuid4().hex,
                tool_name="fixture", capability="recordings.start",
                normalized_arguments={"include_jarvis_audio": True})
            scope = (await admission.admit(command, area_id=f.area, policy=policy)).scope
            upload_id, _, raw_path = ingress._upload_target(scope)
            await storage.upload_file(path=raw_path, local_file=original, content_type="video/mp4")
            return scope, upload_id, raw_path

        scope, upload_id, raw_path = await admitted_upload()
        complete = admission.complete_in_transaction
        async def fail_after_completion(*args, **kwargs):
            await complete(*args, **kwargs)
            raise RuntimeError("Injected rollback after completion")
        args = dict(owner_id=f.owner, session_id=session, recording_id=scope.recording_id,
                    upload_id=upload_id, upload=upload)
        with patch.object(admission, "complete_in_transaction", side_effect=fail_after_completion):
            with self.assertRaisesRegex(RuntimeError, "Injected rollback"):
                await ingress.finalize_upload(**args)
        self.assertIn(raw_path, storage.objects, "Failed adoption must retain the raw upload")
        self.assertIn(raw_path + "/synced", storage.objects)
        async with f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files where owner_id=$1", f.owner), 0)
        ingress.timing_options = RecordingTimingOptions(video_delay_ms=0)
        saved = await ingress.finalize_upload(**args)
        self.assertEqual(saved["outcome"], "saved")
        self.assertEqual(saved["timing_processing"]["status"], "corrected")
        self.assertEqual(set(storage.objects), {raw_path + "/synced"})
        replayed = await ingress.finalize_upload(**args)
        self.assertEqual(saved["file_id"], replayed["file_id"])
        self.assertEqual(storage.uploads, 2, "Exactly one raw upload and one corrected upload")
        async with f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files where owner_id=$1", f.owner), 1)
            row = await connection.fetchrow("select storage_path,size_bytes from jarvis_files where owner_id=$1 and id=$2",
                                           f.owner, UUID(saved["file_id"]))
        self.assertEqual(row["storage_path"], raw_path + "/synced")
        self.assertEqual(row["size_bytes"], len(storage.objects[raw_path + "/synced"]))
        corrected = Path(temporary.name).resolve() / "corrected.mp4"
        corrected.write_bytes(storage.objects[raw_path + "/synced"])
        def audio(path):
            return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path),
                "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "-"], timeout=20)
        self.assertEqual(audio(source), audio(corrected))

        fallback, fallback_id, fallback_path = await admitted_upload()
        ingress.timing_options = RecordingTimingOptions(video_delay_ms=1000, ffmpeg="/missing/ffmpeg")
        unchanged = await ingress.finalize_upload(owner_id=f.owner, session_id=session,
            recording_id=fallback.recording_id, upload_id=fallback_id, upload=upload)
        self.assertEqual(unchanged["outcome"], "saved")
        self.assertEqual(unchanged["timing_processing"]["reason"], "executable_unavailable")
        self.assertEqual(storage.objects[fallback_path], source.read_bytes())
        self.assertEqual(len(storage.objects), 2, "One object for each of two independent recordings")
        async with f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files where owner_id=$1", f.owner), 2)
        receipt = {"corrected_saved": True, "one_gallery_file_per_recording": True,
                   "rollback_recovery": True, "retry_without_rerender": True,
                   "audio_samples_identical": True, "fallback_saved_original": True,
                   "corrected_sha256": hashlib.sha256(corrected.read_bytes()).hexdigest(),
                   "storage": "bounded in-memory object store", "database": "isolated schema"}
        evidence = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / "timing-save-result.json"
        evidence.write_text(json.dumps(receipt, indent=2) + "\n")
