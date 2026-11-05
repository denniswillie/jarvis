"""Real claim filtering with synthetic grants in manifested disposable SQL.

This proves worker selection, not cloud media execution or physical recording.
"""
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.files.generated import ArtifactPolicy
from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested Core database lane required")
class ArtifactMediaClaimSQLTests(unittest.IsolatedAsyncioTestCase):
    async def qualify(self, policy, media):
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()) as fixture:
            tasks = TaskRepository(fixture.pool, TaskLimits(lease_seconds=300))
            async with fixture.pool.acquire() as connection:
                payload = await tasks.payload(connection, fixture.owner, {"text": "Synthetic publication request."})
                event = await connection.fetchval("""insert into jarvis_context_events
                    (owner_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                    values($1,'fixture-artifact-media',$2,'user_request','final','user_input',$3) returning id""",
                    fixture.owner, str(uuid4()), payload)
                scope = await tasks.payload(connection, fixture.owner, policy.model_dump(mode="json"))
                grant = await connection.fetchval("""insert into jarvis_operation_grants
                    (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                    values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                    fixture.owner, str(uuid4()), event, scope)
            task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(), title="Synthetic media admission",
                requested_outcome="Check worker capability selection only.", knowledge_area_id=fixture.area,
                grant_ids=(grant,)), model="fixture-no-model")
            capabilities = ("files.publish_generated",)
            claim = await tasks.claim_next("fixture-document-worker", ("frontier",),
                                          owner_id=fixture.owner, worker_capabilities=capabilities)
            if media:
                self.assertIsNone(claim)
                async with fixture.pool.acquire() as connection:
                    self.assertEqual(await connection.fetchval(
                        "select attempt_count from jarvis_background_tasks where owner_id=$1 and id=$2",
                        fixture.owner, task.id), 0)
                claim = await tasks.claim_next("fixture-media-worker", ("frontier",), owner_id=fixture.owner,
                    worker_capabilities=capabilities + ("files.publish_generated_mp4",))
            self.assertIsNotNone(claim)
            self.assertEqual(claim.task.id, task.id)

    async def test_mixed_allowance_cannot_be_claimed_by_document_only_worker(self):
        await self.qualify(ArtifactPolicy(formats=("text", "mp4"), mp4_max_file_bytes=20000000), True)

    async def test_legacy_document_scope_needs_no_media_capability(self):
        await self.qualify(ArtifactPolicy(), False)

    async def test_legacy_mp4_scope_also_requires_media_capability(self):
        await self.qualify(ArtifactPolicy(formats=("mp4",), transfer_policy="mp4_output_v1",
                                          max_file_bytes=20000000), True)
