"""Actual publisher prepare with fake transactions/grants and real private files.

The inspector is a deterministic fixture, not ffprobe or real MP4 qualification.
No SQL, provider, upload, runtime enablement or native work occurs here.
"""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.generated import ArtifactInspection, ArtifactPolicy, ArtifactRequest, ArtifactSnapshot
from app.files.local_file import snapshot_file
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.tasks.generated_artifacts import GeneratedArtifactPublisher, SNAPSHOT_CAPABILITY
from app.tasks.models import TaskError


def media_policy(**changes):
    return ArtifactPolicy(**(dict(max_files=2, max_file_bytes=20_000_000, max_total_bytes=32_000_000,
                                 formats=("mp4",), transfer_policy="mp4_output_v1") | changes))


class Store:
    """Only the SQL shapes used by the real _parent/_existing/prepare methods."""
    def __init__(self, policy):
        self.policy, self.parent_id, self.area_id = policy, uuid4(), uuid4()
        self.task = SimpleNamespace(id=uuid4(), owner_id="fixture-owner", session_id=uuid4())
        self.parent = dict(id=self.parent_id, capability="files.publish_generated", scope_payload_id=uuid4(),
            request_event_id=uuid4(), request_revision=1, expires_at=datetime.now(UTC) + timedelta(hours=1))
        self.active, self.entries, self.revoked = False, 0, False
        self.grants, self.payloads, self.links, self.created = {}, {}, [], []
        self.rows_available = True

    @asynccontextmanager
    async def transaction(self, owner):
        assert owner == self.task.owner_id and not self.active
        self.active = True
        self.entries += 1
        try:
            yield self
        finally:
            self.active = False

    def is_in_transaction(self):
        return self.active

    async def lock_owner(self, connection, owner):
        assert connection is self and self.active and owner == self.task.owner_id

    async def lock_claim(self, connection, claim):
        assert connection is self and self.active
        return self.task, {}

    async def context_problem(self, connection, task):
        assert self.active
        return None

    async def validate_grant(self, connection, owner, identity):
        assert connection is self and self.active and owner == self.task.owner_id
        if self.revoked:
            raise TaskError("task_grant_unavailable", "Fixture grant revoked", 403)
        if identity == self.parent_id:
            return self.parent | {"scope": self.policy.model_dump(mode="json")}
        return self.grants[identity]

    async def fetchval(self, query, *args):
        assert self.active
        if "select exists(select 1 from jarvis_task_grants" in query:
            assert args == (self.task.owner_id, self.task.id, self.parent_id)
            return True
        if "insert into jarvis_operation_grants" in query:
            owner, capability, key, event, revision, payload, expires = args
            assert owner == self.task.owner_id and capability == SNAPSHOT_CAPABILITY
            identity = uuid4()
            self.grants[identity] = dict(id=identity, capability=capability, operation_key=key,
                request_event_id=event, request_revision=revision, scope_payload_id=payload,
                expires_at=expires, scope=self.payloads[payload])
            self.created.append(identity)
            return identity
        raise AssertionError("Unexpected fixture fetchval")

    async def fetchrow(self, query, *args):
        assert self.active and "select g.id from jarvis_operation_grants" in query
        owner, task_id, key = args
        assert (owner, task_id) == (self.task.owner_id, self.task.id)
        matches = [grant for grant in self.grants.values() if grant["operation_key"] == key]
        return {"id": matches[0]["id"]} if matches else None

    async def fetch(self, query, *args):
        assert self.active and "select p.content,p.availability" in query
        assert args == (self.task.owner_id, self.task.id, SNAPSHOT_CAPABILITY)
        return [{"content": grant["scope"], "availability": "available" if self.rows_available else "deleted"}
                for grant in self.grants.values()]

    async def execute(self, query, *args):
        assert self.active and "insert into jarvis_task_grants" in query
        assert args[:2] == (self.task.owner_id, self.task.id) and args[2] in self.grants

    async def payload(self, connection, owner, content, *, source_payload_ids):
        assert connection is self and self.active and owner == self.task.owner_id
        identity = uuid4()
        self.payloads[identity] = content
        self.links.append(source_payload_ids)
        return identity


class GeneratedMediaPrepareTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="jarvis-media-prepare-test-")
        self.root = Path(self.directory.name).resolve()
        self.addCleanup(self.directory.cleanup)
        self.local = await self.local_file(b"synthetic fixture, not a decoded movie" * 100)
        self.request = ArtifactRequest(operation_key="output", filename="output.mp4", format="mp4",
                                       description="Explicit synthetic offline output.")
        self.evidence_id = uuid4()

        async def freeze(connection, context, sources, area):
            self.assertIs(connection, sources)
            self.assertTrue(connection.active)
            return self.evidence_id

        self.freeze = AsyncMock(side_effect=freeze)
        patcher = patch("app.tasks.generated_artifacts.freeze_artifact_evidence", self.freeze)
        patcher.start(); self.addCleanup(patcher.stop)
        self.configure()

    async def local_file(self, content):
        path = self.root / uuid4().hex
        path.write_bytes(content)
        result = await snapshot_file(path, max_bytes=len(content))
        self.addCleanup(result.close)
        return result

    def configure(self, policy=None):
        self.store = Store(policy or media_policy())
        self.context = SimpleNamespace(tasks=self.store, task=self.store.task, claim=object())
        self.after_inspection = None

        async def inspect(local, policy):
            self.assertFalse(self.store.active, "Inspection must not hold a transaction")
            self.assertIs(local, self.local)
            measured = MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                duration_ms=1000, width=720, height=1280, video_codec="h264", audio_codec="aac")
            if self.after_inspection:
                self.after_inspection()
            return measured

        self.inspector = AsyncMock(side_effect=inspect)
        self.files = SimpleNamespace(sources=self.store, area_id=self.store.area_id,
            task_evidence=SimpleNamespace(context=self.context), media_policy=FileTransferPolicy("mp4_output_v1", 20_000_000),
            media_inspector=self.inspector)
        self.publisher = GeneratedArtifactPublisher(SimpleNamespace(context=self.context, actions={}),
                                                    self.files, self.store.parent_id)

    def seed_snapshot(self, *, request=None):
        request = request or self.request.model_copy(update={"operation_key": "earlier"})
        inspection = ArtifactInspection(format="mp4", size_bytes=self.local.size_bytes,
            content_sha256=self.local.content_sha256, media_inspection=MP4Inspection(size_bytes=self.local.size_bytes,
                content_sha256=self.local.content_sha256, duration_ms=1000, width=720, height=1280,
                video_codec="h264", audio_codec="aac"))
        snapshot = ArtifactSnapshot(request=request, inspection=inspection, parent_grant_id=self.store.parent_id,
            task_id=self.store.task.id, evidence_payload_id=uuid4())
        identity = uuid4()
        self.store.grants[identity] = self.store.parent | dict(id=identity, capability=SNAPSHOT_CAPABILITY,
            operation_key=self.publisher._key(request), scope=snapshot.model_dump(mode="json"))
        return snapshot

    async def test_fresh_large_media_inspects_outside_transaction_and_freezes_exact_dependencies(self):
        self.local = await self.local_file(b"x" * (8 * 1024 * 1024 + 1))
        action, snapshot = await self.publisher.prepare(self.request, self.local)
        self.inspector.assert_awaited_once_with(self.local, self.files.media_policy)
        self.freeze.assert_awaited_once()
        self.assertEqual(self.store.entries, 2)
        self.assertEqual((snapshot.inspection.size_bytes, snapshot.inspection.content_sha256),
                         (self.local.size_bytes, self.local.content_sha256))
        self.assertEqual(snapshot.evidence_payload_id, self.evidence_id)
        self.assertEqual(self.store.links, [(self.store.parent["scope_payload_id"], self.evidence_id)])
        self.assertEqual(action.parameters, snapshot.model_dump(mode="json"))
        self.assertEqual(action.request_event_id, self.store.parent["request_event_id"])
        self.assertEqual(len(self.store.created), 1)
        self.assertTrue(self.local.path.exists(), "The caller retains local-file ownership")

    async def test_missing_mismatched_configuration_and_wrong_content_fail_before_inspection(self):
        for case in ("missing_policy", "missing_inspector", "capture", "document", "parent_document", "bytes", "narrow", "filename"):
            with self.subTest(case=case):
                self.configure()
                content, request = self.local, self.request
                if case == "missing_policy": self.files.media_policy = None
                elif case == "missing_inspector": self.files.media_inspector = None
                elif case == "capture": self.files.media_policy = FileTransferPolicy("mp4_capture_v1", 20_000_000)
                elif case == "document": self.files.media_policy = FileTransferPolicy()
                elif case == "parent_document": self.store.policy = ArtifactPolicy()
                elif case == "bytes": content = b"not a verified file"
                elif case == "narrow": self.files.media_policy = FileTransferPolicy("mp4_output_v1", self.local.size_bytes - 1)
                elif case == "filename": request = self.request.model_copy(update={"filename": "output.txt"})
                with self.assertRaises(TaskError) as caught:
                    await self.publisher.prepare(request, content)
                self.assertEqual(caught.exception.code, "artifact_publication_invalid")
                self.inspector.assert_not_awaited()
                self.freeze.assert_not_awaited()
                self.assertEqual(self.store.created, [])

    async def test_parent_repository_quota_and_revocation_are_rechecked_after_inspection(self):
        for case in ("parent_narrow", "repository_narrow", "parent_kind", "revoked", "count", "total", "withdrawn", "inspector"):
            with self.subTest(case=case):
                self.configure()

                def change():
                    if case == "parent_narrow": self.store.policy = media_policy(max_file_bytes=self.local.size_bytes - 1)
                    elif case == "repository_narrow":
                        self.files.media_policy = FileTransferPolicy("mp4_output_v1", self.local.size_bytes - 1)
                    elif case == "parent_kind": self.store.policy = ArtifactPolicy()
                    elif case == "revoked": self.store.revoked = True
                    elif case == "inspector": self.files.media_inspector = AsyncMock()
                    else:
                        self.seed_snapshot()
                        if case == "count": self.store.policy = media_policy(max_files=1)
                        elif case == "total": self.store.policy = media_policy(max_total_bytes=self.local.size_bytes)
                        else: self.store.rows_available = False

                self.after_inspection = change
                with self.assertRaises(TaskError):
                    await self.publisher.prepare(self.request, self.local)
                self.inspector.assert_awaited_once()
                self.freeze.assert_not_awaited()
                self.assertEqual(self.store.created, [])
                self.assertTrue(self.local.path.exists())

    async def test_prior_snapshot_replays_without_original_or_current_inspector(self):
        expected = await self.publisher.prepare(self.request, self.local)
        self.inspector.reset_mock(); self.freeze.reset_mock()
        self.files.media_policy = None
        self.files.media_inspector = None
        self.local.close()
        replay = await self.publisher.prepare(self.request)
        self.assertEqual(replay, expected)
        self.inspector.assert_not_awaited()
        self.freeze.assert_not_awaited()
        self.assertEqual(len(self.store.created), 1)

    async def test_prior_snapshot_rejects_conflicting_original_or_request_without_inspection(self):
        await self.publisher.prepare(self.request, self.local)
        self.inspector.reset_mock(); self.freeze.reset_mock()
        changed = await self.local_file(b"different exact file")
        for request, content in ((self.request, changed),
                (self.request.model_copy(update={"filename": "renamed.mp4"}), None)):
            with self.subTest(content=content is not None), self.assertRaises(TaskError) as caught:
                await self.publisher.prepare(request, content)
            self.assertEqual(caught.exception.code, "artifact_operation_conflict")
        self.inspector.assert_not_awaited()
        self.freeze.assert_not_awaited()
        self.assertEqual(len(self.store.created), 1)

    async def test_concurrent_accepted_snapshot_wins_without_duplicate_freeze(self):
        accepted = []
        self.after_inspection = lambda: accepted.append(self.seed_snapshot(request=self.request))
        _, snapshot = await self.publisher.prepare(self.request, self.local)
        self.assertEqual(snapshot, accepted[0])
        self.freeze.assert_not_awaited()
        self.assertEqual(self.store.created, [])

    async def test_mismatched_inspection_is_rejected_without_freezing(self):
        self.inspector.side_effect = None
        self.inspector.return_value = MP4Inspection(size_bytes=self.local.size_bytes, content_sha256="0" * 64,
            duration_ms=1000, width=720, height=1280, video_codec="h264", audio_codec=None)
        with self.assertRaises(TaskError) as caught:
            await self.publisher.prepare(self.request, self.local)
        self.assertEqual(caught.exception.code, "artifact_publication_invalid")
        self.freeze.assert_not_awaited()
        self.assertTrue(self.local.path.exists())

    async def test_document_prepare_and_serialized_scopes_remain_unchanged(self):
        self.configure(ArtifactPolicy())
        self.files.media_policy = None
        self.files.media_inspector = None
        request = ArtifactRequest(operation_key="document", filename="report.csv", format="csv",
                                  description="Synthetic document.")
        _, snapshot = await self.publisher.prepare(request, b"name,count\nfixture,2\n")
        self.assertEqual(snapshot.inspection.text_excerpt, "name,count\nfixture,2\n")
        self.assertTrue(snapshot.inspection.text_complete)
        self.assertNotIn("transfer_policy", self.store.policy.model_dump(mode="json"))
        self.assertNotIn("media_inspection", snapshot.inspection.model_dump(mode="json"))
        self.inspector.assert_not_awaited()
        self.freeze.assert_awaited_once()
        with self.assertRaises(TaskError):
            await self.publisher.prepare(request.model_copy(update={"operation_key": "too-large"}),
                                         b"x" * (8 * 1024 * 1024 + 1))
        self.assertEqual(len(self.store.created), 1)


if __name__ == "__main__":
    unittest.main()
