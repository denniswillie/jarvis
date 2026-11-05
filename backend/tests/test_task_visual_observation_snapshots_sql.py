"""Real isolated SQL; controlled stored JPEGs, no model/storage/provider calls.

Reuse the existing manifested frame fixture. Tasks are genuinely admitted and
claimed, never seeded RUNNING. The one permission fault archives only this
fixture's owned area; it is not presented as a user permission-handler test.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.images import ImageUnavailableError
from app.knowledge.images import SourceImageService
from app.knowledge.publication import CanonicalSourcePublication
from app.tasks.context import TaskContextBuilder
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_images import TaskContextImages
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.visual_memory.models import ValidatedVisualFrame, VisualFrameStorageObject
from app.visual_memory.repository import VisualFrameRepository
import test_second_brain_stored_images as stored_tests


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit isolated SQL required")
class TaskVisualSnapshotSQL(unittest.IsolatedAsyncioTestCase):
    reference = stored_tests.StoredImageDatabaseTests.reference

    async def asyncSetUp(self):
        # This established fixture owns its exact schema, two generated owners,
        # pool close, and independent cleanup. Its storage client is in-memory.
        await stored_tests.StoredImageDatabaseTests.asyncSetUp(self)
        self.history = ContextRepository(self.tasks)
        self.capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="fixture-visual-snapshots", write_timeout=10)
        self.addAsyncCleanup(self.capture.close)
        self.receipt = {"format": "jarvis.task_visual_snapshots_sql.v1", "test": self._testMethodName,
            "run_id": self.f.fixture.run_id, "owners": list(self.f.fixture.owners),
            "schemas": list(self.f.fixture.manifest["schemas"]), "checks": {},
            "boundary": "canonical SQL and controlled stored pixels; no models or actual transport",
            "external_storage": "not_allocated", "bank": "not_allocated"}
        self.addAsyncCleanup(self.write_receipt)

    async def write_receipt(self):
        path = self.root.parent / (self._testMethodName + ".json")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(self.receipt, output, indent=2)
            output.write("\n")

    def plan(self, *checks):
        self.receipt["checks"] = {name: "unreached" for name in checks}

    def passed(self, check):
        self.receipt["checks"][check] = "passed"

    async def observation(self, delivery_id, state):
        snapshot = await self.history.snapshot(self.f.owner, self.session, max_messages=512, max_pins=32)
        rows = [row for row in snapshot["recent"] if row.kind == "observation" and row.content
                and row.content.get("delivery_id") == delivery_id and row.content.get("delivery_state") == state]
        self.assertEqual(len(rows), 1)
        return rows[0]

    async def admitted(self, image, delivery_id):
        identity = "request-" + uuid4().hex
        self.capture.input_final(identity, "Reason about this exact captured snapshot, not the current scene.")
        self.capture.bind_call(identity, identity)
        self.capture.image_delivery(image, "socket_written", delivery_id)
        self.assertTrue(await self.capture.flush())
        source = await self.capture.request_for_call(identity)
        self.assertIsNotNone(source)
        task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(), title="Snapshot reasoning",
            requested_outcome="Describe only the captured snapshot and its time.", session_id=self.session,
            knowledge_area_id=self.f.area), model="controlled-no-model", mode="background",
            source=TaskVoiceSource(source.id, source.revision, source.session_id))
        claim = await self.tasks.claim_next("fixture-worker", ("frontier",), owner_id=self.f.owner,
                                           provider_models={"frontier": "controlled-no-model"})
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, task.id)
        manifest = await TaskContextBuilder(self.history).build(claim)
        observation = await self.observation(delivery_id, "socket_written")
        async with self.f.pool.acquire() as connection:
            usage = await connection.fetchval("select source_usage from jarvis_task_context_sources "
                "where owner_id=$1 and manifest_id=$2 and event_id=$3", self.f.owner, manifest.id, observation.id)
        self.assertEqual(usage, "current")
        return claim, manifest, observation

    async def status(self, manifest):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("select status from jarvis_task_context_manifests "
                "where owner_id=$1 and id=$2", self.f.owner, manifest.id)

    async def assert_fenced(self, claim, manifest):
        with self.assertRaises(ClaimLost):
            await self.tasks.checkpoint(claim, {"must_not_commit": True})
        with self.assertRaises(TaskError) as caught:
            await self.history.read_manifest(self.f.owner, claim.task.id, manifest.id)
        self.assertEqual(caught.exception.code, "context_manifest_unavailable")
        self.assertIsNone((await self.tasks.get(self.f.owner, claim.task.id)).result_payload_id)

    async def test_ack_new_frame_and_unrelated_observation_preserve_exact_consumed_snapshot(self):
        self.plan("normal_admission_and_current_binding", "ack_is_additive_not_revision", "queued_pixels_still_valid",
                  "new_frame_and_unrelated_observation_do_not_invalidate", "same_attempt_completes_without_recovery")
        image = self.reference(current=True)
        claim, manifest, original = await self.admitted(image, "first-image")
        self.passed("normal_admission_and_current_binding")
        images = TaskContextImages(self.history, stored=self.service)
        selection = await images.select(claim, original.id)
        resolved = await images.resolve(claim, selection)
        self.assertEqual(resolved.image_bytes, self.storage.image)
        self.assertEqual(selection["image_view"], "captured_snapshot")
        self.assertFalse(selection["current_view_verified"])
        for _ in range(3):
            self.capture.image_delivery(image, "provider_acknowledged", "first-image")
            self.assertTrue(await self.capture.flush())
        ack = await self.observation("first-image", "provider_acknowledged")
        self.assertNotEqual(ack.source_event_id, original.source_event_id)
        self.assertEqual((original.revision, ack.revision), (1, 1))
        self.assertEqual(original.content["images"], ack.content["images"])
        self.assertEqual(original.source_time, ack.source_time)
        self.assertFalse((await self.history.get(self.f.owner, original.id)).content["provider_acknowledged"])
        self.assertEqual(await self.status(manifest), "active")
        self.passed("ack_is_additive_not_revision")
        await resolved.revalidate()
        self.assertEqual(await images.select(claim, original.id), selection)
        self.passed("queued_pixels_still_valid")
        now = datetime.now(UTC)
        path = self.storage.build_storage_path(owner_id=self.f.owner, session_id=self.session,
            visual_session_id=image.visual_session_id, frame_sequence=1)
        frames = VisualFrameRepository(self.f.pool)
        pending = await frames.create_pending_frame(owner_id=self.f.owner, session_id=self.session,
            audio_session_id=None, visual_session_id=image.visual_session_id,
            frame=ValidatedVisualFrame(1, now, "glasses", "image/jpeg", 640, 220, self.storage.image),
            server_received_at=now, storage_bucket=self.storage.bucket, storage_path=path,
            raw_image_expires_at=now + timedelta(minutes=5))
        later = await frames.mark_frame_available(owner_id=self.f.owner, frame_id=pending.id,
            stored_object=VisualFrameStorageObject(self.storage.bucket, path, len(self.storage.image)))
        self.capture.image_delivery(self.reference(later, current=True), "socket_written", "second-image")
        self.assertTrue(await self.capture.flush())
        await self.history.append(self.f.owner, ContextIngress(session_id=self.session, source_namespace="fixture-extra",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="observation", trust="sensor",
            payload={"description": "Unrelated subsequent observation"}))
        self.assertEqual(await self.status(manifest), "active")
        await resolved.revalidate()
        await self.tasks.checkpoint(claim, {"captured_at": image.captured_at.isoformat(), "current_view_verified": False})
        self.passed("new_frame_and_unrelated_observation_do_not_invalidate")
        self.assertTrue(await self.tasks.finish(claim, WorkerOutcome(TaskStatus.SUCCEEDED,
            {"controlled_result": "Snapshot only", "current_view_verified": False})))
        completed = await self.tasks.get(self.f.owner, claim.task.id)
        self.assertEqual(completed.status, TaskStatus.SUCCEEDED)
        self.assertEqual(completed.attempt_count, 1)
        self.passed("same_attempt_completes_without_recovery")

    async def test_real_source_correction_and_goal_replacement_still_fence(self):
        self.plan("source_correction_supersedes_manifest", "goal_replacement_rejects_old_claim")
        image = self.reference(current=True)
        claim, manifest, original = await self.admitted(image, "corrected-image")
        await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace=original.source_namespace, source_event_id=original.source_event_id,
            receipt_id="actual-correction", expected_revision=original.revision, kind=original.kind,
            trust=original.trust, finality="revised", source_time=original.source_time,
            payload={**original.content, "evidence_gap": "Original observation was corrected; do not rely on it."}))
        self.assertEqual(await self.status(manifest), "superseded")
        await self.assert_fenced(claim, manifest)
        self.passed("source_correction_supersedes_manifest")
        await self.tasks.abandon(claim, target=TaskStatus.PAUSED)
        claim, manifest, _ = await self.admitted(image, "goal-image")
        updated = await self.tasks.mutate(self.f.owner, claim.task.id, "update",
            TaskMutation(request_id=uuid4(), expected_revision=claim.task_revision, message="Use a different goal."))
        self.assertGreater(updated.revision, claim.task_revision)
        with self.assertRaises(ClaimLost):
            await self.tasks.checkpoint(claim, {"must_not_commit": True})
        self.assertIsNone((await self.tasks.get(self.f.owner, claim.task.id)).result_payload_id)
        self.passed("goal_replacement_rejects_old_claim")

    async def test_source_deletion_and_scoped_permission_withdrawal_still_fence(self):
        self.plan("source_bound_original_resolves", "source_deletion_revokes_receipts_and_manifest",
                  "late_ack_cannot_restore_deleted_source", "area_permission_withdrawal_blocks_old_claim")
        publishing = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        event = await publishing.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="fixture-original-image", source_event_id=uuid4().hex, receipt_id=uuid4().hex,
            kind="observation", trust="sensor", source_time=self.frame.captured_at,
            payload={"images": [self.reference(current=True).to_metadata()]}))
        async with self.f.pool.acquire() as connection:
            identity = await connection.fetchval("select source_id from jarvis_knowledge_context_sources "
                "where owner_id=$1 and event_id=$2", self.f.owner, event.id)
        source = await self.f.sources.read(self.f.owner, self.f.area, identity)
        source_images = SourceImageService(self.f.sources, self.service, owner_id=self.f.owner, area_id=self.f.area)
        image = await source_images.select_image(owner_id=self.f.owner, area_id=self.f.area, source_id=source.source_id)
        claim, manifest, original = await self.admitted(image, "owned-source-image")
        images = TaskContextImages(self.history, stored=self.service, source_historical=source_images)
        selected = await images.select(claim, original.id)
        resolved = await images.resolve(claim, selected)
        self.assertEqual(resolved.image_bytes, self.storage.image)
        self.passed("source_bound_original_resolves")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=source.revision)
        self.assertEqual(await self.status(manifest), "revoked")
        await self.assert_fenced(claim, manifest)
        with self.assertRaises(ClaimLost):
            await resolved.revalidate()
        self.passed("source_deletion_revokes_receipts_and_manifest")
        self.capture.image_delivery(image, "provider_acknowledged", "owned-source-image")
        await self.capture.flush()
        snapshot = await self.history.snapshot(self.f.owner, self.session, max_messages=512, max_pins=32)
        self.assertFalse(any(row.content and row.content.get("delivery_id") == "owned-source-image"
            and row.content.get("provider_acknowledged") for row in snapshot["recent"]))
        self.assertEqual(await self.status(manifest), "revoked")
        self.passed("late_ack_cannot_restore_deleted_source")
        await self.tasks.abandon(claim, target=TaskStatus.PAUSED)
        claim, manifest, _ = await self.admitted(self.reference(current=True), "permission-image")
        # Explicit fixture fault: canonical area guard, not a fabricated claim
        # and not qualification of an application permission-change handler.
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.f.area)
        async with self.tasks.transaction(self.f.owner) as connection:
            task = await self.tasks.load(connection, self.f.owner, claim.task.id)
            self.assertEqual(await self.tasks.context_problem(connection, task), "knowledge_area_unavailable")
        await self.assert_fenced(claim, manifest)
        self.passed("area_permission_withdrawal_blocks_old_claim")
