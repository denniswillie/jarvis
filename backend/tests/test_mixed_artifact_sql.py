"""Actual serialized same-task mixed-output quotas, using disposable SQL only."""
import asyncio
import os
import unittest

from app.files.generated import ArtifactPolicy, ArtifactRequest, ArtifactSnapshot
from app.files.storage import StoredFileObject
from app.files.transfer import FileTransferPolicy
from app.tasks.models import TaskError, json_object
from test_file_backed_publication_sql import DiskStorage
from test_generated_media_source_sql import GeneratedMediaSourceSQLTests, LIMIT


class MixedDiskStorage(DiskStorage):
    async def upload_bytes(self, *, path, content, content_type):
        self.uploads += 1
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=len(content))

    async def read_object(self, *, bucket, path, max_bytes):
        if bucket != self.bucket:
            raise AssertionError("wrong fixture bucket")
        target = self.root / path
        if not target.exists():
            return None
        with target.open("rb") as stream:
            data = stream.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("fixture read exceeds bound")
        return data


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Manifested SQL lane required")
class MixedArtifactSQLTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = GeneratedMediaSourceSQLTests.asyncSetUp
    storage_class = MixedDiskStorage

    def publication_allowance(self):
        return ArtifactPolicy(max_files=3, max_file_bytes=1024, max_total_bytes=self.local.size_bytes + 100,
                              formats=("text", "mp4"), mp4_max_file_bytes=LIMIT)

    def document(self, key):
        return ArtifactRequest(operation_key=key, filename=key + ".txt", format="text", description="Synthetic summary")

    async def test_same_task_document_and_mp4_persist_concrete_limits_and_replay(self):
        document = self.document("summary")
        first = await self.publisher.publish(document, b"A synthetic summary.", provider_call_id="mixed-doc")
        movie = await self.publisher.publish(self.request, self.local, provider_call_id="mixed-movie")
        self.assertEqual((first["status"], movie["status"]), ("success", "success"))
        self.assertNotEqual(first["file"]["id"], movie["file"]["id"])
        self.assertEqual(self.storage.uploads, 2)
        async with self.f.source.pool.acquire() as connection:
            rows = await connection.fetch("select transfer_policy,transfer_limit_bytes,phase from jarvis_file_publications")
        self.assertEqual({(row["transfer_policy"], row["transfer_limit_bytes"], row["phase"]) for row in rows},
                         {("document_v1", 1024, "available"), ("mp4_output_v1", LIMIT, "available")})
        self.files.media_policy = FileTransferPolicy("mp4_output_v1", 1)
        for request, saved in ((document, first), (self.request, movie)):
            replay = await self.publisher.publish(request, provider_call_id="mixed-replay-" + request.format)
            self.assertEqual(replay["file"], saved["file"])
            self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(self.storage.uploads, 2)

    async def test_competing_outputs_share_quota_and_per_format_limit(self):
        # Setup already reserves the real movie's exact snapshot. Only100bytes
        # remain in this SAME parent allowance, not a fresh budget per format.
        with self.assertRaises(TaskError):
            await self.publisher.prepare(self.document("oversized"), b"x" * 1025)
        results = await asyncio.gather(
            self.publisher.prepare(self.document("left"), b"l" * 60),
            self.publisher.prepare(self.document("right"), b"r" * 60), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, tuple) for result in results), 1, results)
        errors = [result for result in results if isinstance(result, TaskError)]
        self.assertEqual(len(errors), 1, results)
        self.assertEqual(errors[0].code, "artifact_allowance_exhausted")
        async with self.f.source.pool.acquire() as connection:
            rows = await connection.fetch("""select p.content from jarvis_task_grants t
                join jarvis_operation_grants g on g.owner_id=t.owner_id and g.id=t.grant_id
                join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                where t.owner_id=$1 and t.task_id=$2 and g.capability='files.publish_snapshot'""",
                self.f.source.owner, self.f.task.id)
        snapshots = [ArtifactSnapshot.model_validate(json_object(row["content"])) for row in rows]
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(sum(item.inspection.size_bytes for item in snapshots), self.local.size_bytes + 60)
        self.assertEqual(self.storage.uploads, 0)
