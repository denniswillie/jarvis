"""One Core-only manifested media-input integration case; no live sandbox.

The exact synthetic MP4 is retained by real recording admission/ingress/Files,
then staged through the real task claim, memory evidence and SandboxToolBridge.
The provider double only consumes verified file chunks: it does not qualify
E2B transport/installer permissions, media editing, physical capture/hearing,
native Phone behavior or either complete demo. Revocation is explicit fixture
authority on disposable data. No forged source original or canned save receipt.
"""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from uuid import UUID, uuid4

from app.files.local_file import CHUNK_BYTES, VerifiedLocalFile, snapshot_file
from app.files.media_inspection import inspect_mp4_file
from app.files.recording_ingress import RecordingClientReport, RecordingIngress, RecordingUpload
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy
from app.knowledge.files import SourceFileRepository
from app.knowledge.product_files import RECORDING_ORIGINAL_FORMAT
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission
from app.migrations.runner import load_migrations
from app.recordings.admission import RecordingAdmission
from app.sandbox.file_transfer import MAX_TRANSFER_BYTES
from app.sandbox.tools import SandboxToolBridge
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import ClaimLost
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from test_recording_ingress_sql import EXPECTED_SHA256, EXPECTED_SIZE, FIXTURE_LIMIT, TrackedDiskStorage, file_sha256
from test_sandbox_inputs import FileProvider


class StreamInputProvider(FileProvider):
    """Existing allocation double plus a bounded, actually consumed media stream."""
    def __init__(self, case):
        super().__init__()
        self.case = case
        self.transfers = []
        self.private_inputs = []
        self.revoke_during_transfer = None
        self.bytes_calls = 0

    async def stage_input(self, *args, **kwargs):
        self.bytes_calls += 1
        raise AssertionError("A retained MP4 must not use the whole-bytes document path")

    async def stage_input_file(self, handle, identity, local_file, *, policy):
        self.case.assertIsInstance(local_file, VerifiedLocalFile)
        self.case.assertEqual(policy, self.case.policy)
        self.case.assertEqual(handle.scope.task_id, self.case.f.task.id)
        self.case.assertEqual(handle.scope.attempt_id, self.case.f.claim.attempt_id)
        self.private_inputs.append(local_file)
        checksum, size, chunks = hashlib.sha256(), 0, 0
        async for block in local_file.iter_chunks():
            self.case.assertLessEqual(len(block), CHUNK_BYTES)
            size += len(block); chunks += 1; checksum.update(block)
            if self.revoke_during_transfer is not None:
                revoke, self.revoke_during_transfer = self.revoke_during_transfer, None
                await revoke()
        self.case.assertEqual((size, checksum.hexdigest()), (EXPECTED_SIZE, EXPECTED_SHA256))
        self.case.assertGreater(chunks, 128)
        receipt = {"path": "/home/jarvis/inputs/" + identity, "size_bytes": size, "sha256": checksum.hexdigest()}
        self.transfers.append({"identity": identity, "allocation_id": handle.scope.allocation_id,
                               "sandbox_id": handle.sandbox_id, "receipt": receipt})
        # No actual sandbox path or permissions are created by this provider.
        self.inputs[identity] = receipt
        return receipt


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested Core database execution lane required")
class SandboxMediaInputSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_recording_source_task_staging_replay_and_revocation(self):
        names = ("JARVIS_MEDIA_FIXTURE_FILE", "JARVIS_MEDIA_FIXTURE_SHA256",
                 "JARVIS_MEDIA_FFPROBE_PATH", "JARVIS_MEDIA_FFPROBE_SHA256")
        self.assertTrue(all(os.getenv(name) for name in names), "All four pinned media/probe inputs are mandatory")
        original, probe = Path(os.environ[names[0]]), Path(os.environ[names[2]])
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.assertTrue(directory.is_absolute())
        for path in (original, probe):
            self.assertTrue(path.is_absolute() and path.is_file() and not path.is_symlink())
        self.assertEqual(os.environ[names[1]], EXPECTED_SHA256)
        self.assertEqual((original.stat().st_size, file_sha256(original)), (EXPECTED_SIZE, EXPECTED_SHA256))
        probe_sha = os.environ[names[3]]
        self.assertEqual(file_sha256(probe), probe_sha)
        self.assertIn((48, "recording_admission"), {(item.version, item.name) for item in load_migrations()})
        self.assertEqual(MAX_TRANSFER_BYTES, 8 * 1024**2, "The document input bound must remain unchanged")
        self.assertGreater(EXPECTED_SIZE, MAX_TRANSFER_BYTES)
        self.policy = FileTransferPolicy("mp4_capture_v1", FIXTURE_LIMIT)
        inspected = []

        async def inspector(local, policy):
            self.assertEqual(file_sha256(probe), probe_sha)
            inspected.append(local.path)
            return await inspect_mp4_file(local, policy, ffprobe_path=probe, timeout=10)

        local = await snapshot_file(original, max_bytes=FIXTURE_LIMIT,
                                    expected_size=EXPECTED_SIZE, expected_sha256=EXPECTED_SHA256)
        try:
            measured = await inspector(local, self.policy)
            self.assertEqual((measured.video_codec, measured.audio_codec), ("h264", "aac"))
        finally:
            local.close()
        # Actual input qualification precedes creation of any disposable schema.
        self.f = await self.enterAsyncContext(sandbox_task_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            directory, test_name=self.id(), with_session=True,
            requested_outcome="Fixture: stage an exact retained synthetic recording for authorized computation."))
        owner, session = self.f.source.owner, self.f.task.session_id
        temporary = tempfile.TemporaryDirectory(prefix="jarvis-sandbox-media-sql-")
        self.addCleanup(temporary.cleanup)
        storage = TrackedDiskStorage(Path(temporary.name).resolve() / "objects")
        self.addCleanup(storage.close_handles)
        context = TaskRunner(self.f.tasks, {}).context(self.f.claim)
        await WorkerContextFactory(self.f.source.pool)(context)
        selections = SessionKnowledgeAreas(self.f.source.sources, self.f.tasks, self.f.source.area)
        admissions = RecordingAdmission(self.f.source.sources, self.f.tasks, selections, owner_id=owner)
        ingress = RecordingIngress(self.f.source.sources, storage, admissions, policy=self.policy, inspect_file=inspector)
        # Fixture-constructed trusted command; this is not a voice admission test.
        admitted = await admissions.admit(CommandAdmission(owner_id=owner, session_id=session,
            provider_session_key="fixture-media-staging", provider_call_id="start-" + uuid4().hex,
            tool_name="fixture_recording", capability="recordings.start", normalized_arguments={"include_jarvis_audio": True}),
            area_id=self.f.source.area, policy=self.policy)
        await admissions.stop(CommandAdmission(owner_id=owner, session_id=session,
            provider_session_key="fixture-media-staging", provider_call_id="stop-" + uuid4().hex,
            tool_name="fixture_recording", capability="recordings.stop",
            normalized_arguments={"recording_id": str(admitted.scope.recording_id)}))

        async def recording_bytes():
            with original.open("rb") as stream:
                while block := stream.read(CHUNK_BYTES):
                    yield block

        descriptor = RecordingUpload(filename="synthetic-staging-recording.mp4",
            description="Exact synthetic client-reported recording; no physical capture claim.",
            size_bytes=EXPECTED_SIZE, content_sha256=EXPECTED_SHA256,
            client_report=RecordingClientReport(generation_id=uuid4(), timeline_end_ms=measured.duration_ms,
                video_frames=1, source_audio_frames=0, padded_audio_frames=0))
        saved = await ingress.finalize(owner_id=owner, session_id=session, recording_id=admitted.scope.recording_id,
                                       upload=descriptor, chunks=recording_bytes())
        self.assertEqual(saved["outcome"], "saved")
        file_id, source_id = UUID(saved["file_id"]), UUID(saved["source_id"])
        source = await self.f.source.sources.read(owner, self.f.source.area, source_id)
        self.assertEqual((source.source_type, source.trust), ("recording_file", "user_input"))
        self.assertEqual(source.content["format"], RECORDING_ORIGINAL_FORMAT)
        self.assertEqual(source.content["media_origin"], "client_reported")
        self.assertEqual(source.content["content_sha256"], EXPECTED_SHA256)
        self.assertEqual(source.document_id, saved["version_id"])
        evidence = TaskMemoryEvidence(context, self.f.source.sources, area_id=self.f.source.area)
        files = SourceFileRepository(self.f.source.sources, storage, owner_id=owner, area_id=self.f.source.area,
            task_evidence=evidence, media_policy=self.policy, media_inspector=inspector)
        base = await TaskToolBridge.from_modules(context, (FileToolModule(files, storage),),
            read_tools=("list_files", "get_file_metadata"), memory_evidence=evidence)
        provider = StreamInputProvider(self)
        self.addCleanup(lambda: [local.close() for local in provider.private_inputs])
        bridge = await SandboxToolBridge.attach(base, provider, files=files)
        self.assertIsInstance(bridge, SandboxToolBridge)

        async def stage():
            return await bridge.invoke("sandbox_stage_file", {"file_id": str(file_id)},
                                       provider_call_id="fixture-stage-" + uuid4().hex)

        first, repeated = await stage(), await stage()
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(first, repeated)
        self.assertEqual(len(provider.resources), 1)
        self.assertEqual(len(provider.inputs), 1)
        self.assertEqual(provider.bytes_calls, 0)
        self.assertEqual(len(provider.transfers), 2)
        self.assertEqual(provider.transfers[0], provider.transfers[1])
        copied = first["sandbox_input"]
        self.assertEqual((copied["size_bytes"], copied["sha256"]), (EXPECTED_SIZE, EXPECTED_SHA256))
        self.assertEqual(copied["source"]["source"]["document_id"], source.document_id)
        citation = copied["source"]["task_evidence"]
        self.assertEqual(citation, {"source_id": str(source_id), "revision": source.revision,
            "generation": source.generation, "content_digest": source.content_digest, "source_usage": "current"})
        async with self.f.source.pool.acquire() as connection:
            rows = await connection.fetch("select * from jarvis_sandbox_allocations where owner_id=$1 and task_id=$2",
                                          owner, self.f.task.id)
            self.assertEqual(len(rows), 1)
            allocation_id = rows[0]["id"]
            self.assertEqual(rows[0]["phase"], "ready")
            self.assertEqual(rows[0]["grant_id"], self.f.grant)
            self.assertEqual(provider.transfers[0]["allocation_id"], allocation_id)
            bindings = await connection.fetch("""select m.* from jarvis_task_memory_sources m
                join jarvis_task_context_manifests c on c.owner_id=m.owner_id and c.id=m.manifest_id
                where c.owner_id=$1 and c.task_id=$2 and m.source_id=$3""", owner, self.f.task.id, source_id)
            self.assertEqual(len(bindings), 1)
            self.assertEqual((bindings[0]["payload_id"], bindings[0]["content_digest"], bindings[0]["source_revision"]),
                             (source.payload_id, source.content_digest, source.revision))
        self.assertTrue(all(not local.path.exists() for local in provider.private_inputs))

        async def revoke():
            # Delete only this disposable canonical source, using its real owner.
            # The provider has already consumed a chunk of the exact private copy.
            await self.f.source.sources.delete_source(owner, self.f.source.area, source_id, expected_revision=source.revision)

        provider.revoke_during_transfer = revoke
        result = None
        try:
            result = await stage()
        except ClaimLost:
            pass  # Exact source dependencies may invalidate the whole task claim.
        self.assertTrue(result is None or result.get("status") == "failed", result)
        self.assertEqual(len(provider.transfers), 3, "Revocation must occur inside the actual transfer callback")
        self.assertEqual({item["allocation_id"] for item in provider.transfers}, {allocation_id})
        self.assertEqual(len(provider.resources), 1)
        self.assertEqual(len(bridge.inputs._staged_revalidators), 2, "Failed staging must not register another input")
        self.assertTrue(all(not local.path.exists() for local in provider.private_inputs))
        self.assertTrue(all(not local.path.exists() for local in storage.returned))
        self.assertTrue(all(not path.exists() for path in inspected))
        async with self.f.source.pool.acquire() as connection:
            rows = await connection.fetch("select id,phase from jarvis_sandbox_allocations where owner_id=$1 and task_id=$2",
                                          owner, self.f.task.id)
            self.assertEqual([(row["id"], row["phase"]) for row in rows], [(allocation_id, "cleanup_requested")])
        # Cleanup intent is authoritative; this double cannot prove VM destruction.
        self.assertEqual(storage.uploads, 1)
        self.assertEqual(file_sha256(original), EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
