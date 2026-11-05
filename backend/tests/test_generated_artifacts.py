"""Generated originals use real action/source SQL and deterministic storage effects."""
import asyncio
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.files.generated import ArtifactInspection, ArtifactPolicy, ArtifactRequest, inspect_artifact
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.tasks.artifacts import result_artifacts
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.sandbox_fixture import sandbox_task_fixture


def request(key="report", kind="csv", **changes):
    return ArtifactRequest(**({"operation_key": key, "filename": "report." + ("txt" if kind == "text" else kind),
        "format": kind, "description": "Synthetic generated report."} | changes))


# Opaque publication fixtures, intentionally not usable PDF/XLSX documents.
# Native document acceptance separately checks real files and their contents.
BINARY_ORIGINALS = (("pdf", b"%PDF-1.7\nnot a parsed document\n"),
                    ("xlsx", b"PK\x03\x04opaque workbook with =SUM(B2:B3)"))


class ArtifactFormatTests(unittest.TestCase):
    def test_text_json_and_csv_retain_exact_hash_and_explicit_excerpt(self):
        cases = [("text", b"Generated text.\n"), ("json", b'{"synthetic":true}\n'),
                 ("csv", b'label,value\nsynthetic,42\n')]
        for kind, content in cases:
            with self.subTest(kind=kind):
                result = inspect_artifact(request(kind=kind), content, ArtifactPolicy())
                self.assertEqual(result.content_sha256, hashlib.sha256(content).hexdigest())
                self.assertTrue(result.text_complete)
                self.assertEqual(result.text_excerpt.encode(), content)
        result = inspect_artifact(request(kind="text"), b"a" * 12001, ArtifactPolicy())
        self.assertFalse(result.text_complete)
        self.assertEqual(len(result.text_excerpt), 12000)

    def test_format_and_filename_bounds_fail_before_storage(self):
        for filename in ("/tmp/a.txt", "../a.txt", "a\\b.txt", "..", "a.txt\n"):
            with self.subTest(filename=filename), self.assertRaises(ValidationError):
                request(kind="text", filename=filename)
        for selected, content, policy in [
            (request(kind="text"), b"", ArtifactPolicy()),
            (request(kind="pdf", filename="report.xlsx"), b"opaque", ArtifactPolicy()),
            (request(kind="text", filename="page.html"), b"text", ArtifactPolicy()),
            (request(kind="text"), b"abc", ArtifactPolicy(max_file_bytes=2)),
            (request(kind="text"), b"text", ArtifactPolicy(formats=("csv",))),
        ]:
            with self.subTest(format=selected.format, content=content), self.assertRaises(ValueError):
                inspect_artifact(selected, content, policy)

    def test_originals_are_not_parsed_or_claimed_verified(self):
        import builtins
        original_import = builtins.__import__
        def without_document_parsers(name, *args, **kwargs):
            if name.split(".")[0] in {"pypdf", "openpyxl", "defusedxml"}:
                raise AssertionError("Publication must not import a document parser")
            return original_import(name, *args, **kwargs)
        cases = BINARY_ORIGINALS + (("json", b"{broken}"), ("csv", b'"unfinished CSV'))
        with patch("builtins.__import__", side_effect=without_document_parsers):
            for kind, content in cases:
                with self.subTest(kind=kind):
                    result = inspect_artifact(request(kind=kind), content, ArtifactPolicy())
                    self.assertEqual(result.content_sha256, hashlib.sha256(content).hexdigest())
                    self.assertEqual(result.size_bytes, len(content))
                    self.assertIsNone(result.item_count)
                    self.assertIn("format is declared, not parsed", result.meaning)
                    if kind in {"pdf", "xlsx"}:
                        self.assertIsNone(result.text_excerpt)
                        self.assertFalse(result.text_complete)

    def test_undecodable_or_nul_text_is_deliverable_without_an_extraction_claim(self):
        for content in (b"\xff", b"nul\x00data"):
            with self.subTest(content=content):
                result = inspect_artifact(request(kind="text"), content, ArtifactPolicy())
                self.assertIsNone(result.text_excerpt)
                self.assertFalse(result.text_complete)

    def test_historical_inspection_metadata_round_trips_without_relabelling(self):
        old = {"format": "pdf", "content_sha256": "a" * 64, "size_bytes": 20,
               "text_excerpt": None, "text_complete": False, "item_count": 1,
               "meaning": "Structural format inspection only; not verification of factual claims or a malware scan."}
        self.assertEqual(ArtifactInspection.model_validate(old).model_dump(), old)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "isolated database fixture required")
class GeneratedArtifactDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(sandbox_task_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(), with_session=True))
        self.context = TaskRunner(self.f.tasks, {}).context(self.f.claim)
        await WorkerContextFactory(self.f.source.pool)(self.context)
        self.storage = MemoryStorage()
        self.evidence = TaskMemoryEvidence(self.context, self.f.source.sources, area_id=self.f.source.area)
        self.files = SourceFileRepository(self.f.source.sources, self.storage, owner_id=self.f.source.owner,
                                          area_id=self.f.source.area, task_evidence=self.evidence)
        self.foreground = SourceFileRepository(self.f.source.sources, self.storage,
                                               owner_id=self.f.source.owner, area_id=self.f.source.area)
        self.bridge = await TaskToolBridge.from_modules(self.context, (FileToolModule(self.files, self.storage),),
            read_tools=("list_files", "get_file_metadata"), memory_evidence=self.evidence,
            receipt_revalidator=self.files.revalidate_receipt)
        self.parent = await self.grant(ArtifactPolicy(max_files=3, max_total_bytes=100000))
        self.publisher = GeneratedArtifactPublisher(self.bridge, self.files, self.parent)

    async def grant(self, policy):
        async with self.f.tasks.transaction(self.f.source.owner) as connection:
            scope = await self.f.tasks.payload(connection, self.f.source.owner, policy.model_dump(mode="json"))
            identity = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.f.source.owner, str(uuid4()), self.f.event, scope)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     self.f.source.owner, self.f.task.id, identity)
            return identity

    async def publish(self, selected=None, content=b"name,value\nsynthetic,42\n"):
        return await self.publisher.publish(selected or request(), content, provider_call_id="fixture-" + uuid4().hex)

    async def test_exact_original_survives_producer_loss_and_appears_in_inbox(self):
        first = await self.publish()
        second = await self.publish(content=None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["file"], first["file"])
        self.assertTrue(second["task_operation"]["replayed"])
        self.assertEqual(self.storage.uploads, 1)
        file_id = UUID(first["file"]["id"])
        self.assertEqual(await self.foreground.read_content(owner_id=self.f.source.owner, file_id=file_id),
                         b"name,value\nsynthetic,42\n")
        output = SimpleNamespace(expert_available=True, content=SimpleNamespace(
            completed_operation_refs=[first["task_operation"]["operation_ref"]]))
        async with self.f.source.pool.acquire() as connection:
            linked = await result_artifacts(connection, self.f.tasks, self.f.task, output)
            self.assertEqual([item["file_id"] for item in linked], [str(file_id)])
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_actions"), 1)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_memory_sources"), 1)

    async def test_key_conflicts_and_per_parent_file_budget(self):
        await self.publish()
        for selected, content in [(request(), b"new,bytes\n"), (request(filename="other.csv"), None)]:
            with self.subTest(selected=selected), self.assertRaises(TaskError):
                await self.publish(selected, content)
        await self.publish(request("two"))
        await self.publish(request("three"))
        with self.assertRaises(TaskError):
            await self.publish(request("four"))
        self.assertEqual(self.storage.uploads, 3)

    async def test_total_allowance_and_ungranted_capability_do_not_upload(self):
        other = GeneratedArtifactPublisher(self.bridge, self.files, self.f.grant)
        with self.assertRaises(TaskError):
            await other.publish(request(), b"data\n", provider_call_id="fixture-wrong-capability")
        narrow = GeneratedArtifactPublisher(self.bridge, self.files,
            await self.grant(ArtifactPolicy(max_total_bytes=3)))
        with self.assertRaises(TaskError):
            await narrow.publish(request(), b"four", provider_call_id="fixture-over-budget")
        self.assertEqual(self.storage.uploads, 0)

    async def test_generated_allowance_cannot_move_to_a_second_task(self):
        import asyncpg
        with self.assertRaises(asyncpg.CheckViolationError):
            await self.f.tasks.create(self.f.source.owner, TaskCreate(request_id=uuid4(),
                title="Other synthetic task", requested_outcome="No cross-task publication.",
                knowledge_area_id=self.f.source.area, session_id=self.f.task.session_id,
                grant_ids=(self.parent,)), model="fixture-no-model")

    async def test_invalid_call_identity_cannot_consume_a_snapshot_allowance(self):
        with self.assertRaises(ValueError):
            await self.publisher.publish(request(), b"synthetic", provider_call_id="")
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_operation_grants where capability='files.publish_snapshot'"), 0)

    async def test_unknown_upload_recovers_same_object_without_original_bytes(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.storage.mode = None
        recovered = await self.publish(content=None)
        self.assertEqual(recovered["status"], "success", recovered)
        self.assertEqual(recovered["task_operation"]["state"], "accepted")
        self.assertEqual(self.storage.uploads, 1)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 1)

    async def test_absent_unknown_upload_is_not_resubmitted(self):
        self.storage.mode = "fail_before"
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.storage.mode = None
        with self.assertRaises(SourceError):
            await self.publish(content=None)
        self.assertEqual(self.storage.uploads, 1)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)

    async def test_unsubmitted_snapshot_needs_original_bytes_without_crossing_effect_boundary(self):
        await self.publisher.prepare(request(), b"initial\n")
        with self.assertRaises(TaskError):
            await self.publish(content=None)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_actions"), 0)

    async def test_correction_invalidates_artifact_without_a_manifest_dependency_cycle(self):
        source = await self.f.source.publish("Synthetic progress: not completed.")
        await self.evidence.bind([source.original_evidence()])
        saved = await self.publish()
        await self.f.source.publish("Synthetic correction: completed.", expected_revision=1)
        with self.assertRaises(SourceError):
            await self.foreground.read_content(owner_id=self.f.source.owner, file_id=UUID(saved["file"]["id"]))

    async def test_context_correction_invalidates_the_generated_original(self):
        saved = await self.publish()
        repo = ContextRepository(self.f.tasks)
        async with self.f.source.pool.acquire() as connection:
            source = await repo._record(connection, self.f.source.owner, self.f.task.request_event_id)
        await repo.append(self.f.source.owner, ContextIngress(
            session_id=source.session_id, source_namespace=source.source_namespace, source_event_id=source.source_event_id,
            receipt_id="fixture-correction", kind=source.kind, trust=source.trust, finality="revised",
            payload={"text": "Corrected synthetic request."}, expected_revision=source.revision))
        with self.assertRaises(SourceError):
            await self.foreground.read_content(owner_id=self.f.source.owner, file_id=UUID(saved["file"]["id"]))

    async def test_revoked_parent_during_upload_cannot_be_adopted(self):
        self.storage.mode = "pause_after"
        pending = asyncio.create_task(self.publish())
        await asyncio.wait_for(self.storage.entered.wait(), 10)
        try:
            async with self.f.source.pool.acquire() as connection:
                await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.parent)
        finally:
            self.storage.release.set()
        with self.assertRaises((ClaimLost, TaskError)):
            await pending
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "outcome_unknown")

    async def test_source_deletion_removes_derived_original_and_prevents_receipt_replay(self):
        source = await self.f.source.publish("Synthetic private fact.")
        await self.evidence.bind([source.original_evidence()])
        saved = await self.publish()
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     source.payload_id)
        with self.assertRaises(SourceError):
            await self.foreground.read_content(owner_id=self.f.source.owner, file_id=UUID(saved["file"]["id"]))
        with self.assertRaises((ClaimLost, TaskError)):
            await self.publish(content=None)
        self.assertEqual(self.storage.uploads, 1)

    async def test_binary_pdf_and_xlsx_are_exact_downloads_with_assistant_origin(self):
        for kind, content in BINARY_ORIGINALS:
            with self.subTest(kind=kind):
                saved = await self.publish(request(kind, kind=kind), content)
                file_id = UUID(saved["file"]["id"])
                self.assertEqual(await self.foreground.read_content(owner_id=self.f.source.owner, file_id=file_id), content)
                self.assertEqual(saved["file"]["memory_source"]["trust"], "assistant_output")
                self.assertEqual(saved["file"]["memory_source"]["source_type"], "generated_file")
        self.assertEqual(self.storage.uploads, 2)

    def recovery(self):
        from app.tasks.file_recovery import TaskFileRecovery
        return TaskFileRecovery(self.f.source.sources, self.f.tasks, self.storage,
            owner_id=self.f.source.owner, area_id=self.f.source.area)

    async def stopped_upload(self, mode="fail_after"):
        self.storage.mode = mode
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.storage.mode = None
        await self.f.tasks.abandon(self.f.claim)
        self.assertEqual((await self.f.tasks.get(self.f.source.owner, self.f.task.id)).status, TaskStatus.UNKNOWN)

    async def test_recovery_preserves_original_lineage_and_replays_in_a_new_attempt(self):
        await self.stopped_upload()
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual([item["phase"] for item in result], ["available"], result)
        paused = await self.f.tasks.get(self.f.source.owner, self.f.task.id)
        self.assertEqual(paused.status, TaskStatus.PAUSED)
        await self.f.tasks.mutate(self.f.source.owner, self.f.task.id, "resume",
            TaskMutation(request_id=uuid4(), expected_revision=paused.revision, message="Continue the synthetic report."))
        claim = await self.f.tasks.claim_next("fixture-restarted", ("frontier",),
            worker_capabilities=("sandbox.execute", "files.publish_generated"))
        self.assertNotEqual(claim.attempt_id, self.f.claim.attempt_id)
        context = TaskRunner(self.f.tasks, {}).context(claim)
        await WorkerContextFactory(self.f.source.pool)(context)
        evidence = TaskMemoryEvidence(context, self.f.source.sources, area_id=self.f.source.area)
        files = SourceFileRepository(self.f.source.sources, self.storage, owner_id=self.f.source.owner,
                                    area_id=self.f.source.area, task_evidence=evidence)
        bridge = await TaskToolBridge.from_modules(context, (FileToolModule(files, self.storage),),
            memory_evidence=evidence, receipt_revalidator=files.revalidate_receipt)
        saved = await GeneratedArtifactPublisher(bridge, files, self.parent).publish(
            request(), provider_call_id="fixture-fresh-attempt")
        self.assertEqual(saved["status"], "success", saved)
        self.assertTrue(saved["task_operation"]["replayed"])
        self.assertIsNotNone(saved["file"]["task_evidence"])
        original = await self.foreground.get_file(owner_id=self.f.source.owner, file_id=UUID(saved["file"]["id"]))
        self.assertEqual(original.metadata["task_artifact"]["attempt_id"], str(self.f.claim.attempt_id))
        self.assertEqual(self.storage.uploads, 1)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_actions"), 1)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 1)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_reservations where status in ('submitted','unknown')"), 0)

    async def test_recovery_waits_for_a_live_origin_and_never_uses_generic_adoption(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.publish()
        self.storage.mode = None
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "awaiting_task", result)
        generic = await self.foreground.publications.reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(generic[0]["phase"], "unresolved")
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)

    async def test_recovery_missing_unknown_object_remains_unknown_without_upload(self):
        await self.stopped_upload("fail_before")
        for _ in range(2):
            result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
            self.assertEqual(result[0]["phase"], "outcome_unknown", result)
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual((await self.f.tasks.get(self.f.source.owner, self.f.task.id)).status, TaskStatus.UNKNOWN)

    async def test_recovery_cancellation_cleans_existing_upload_without_adoption(self):
        await self.stopped_upload()
        current = await self.f.tasks.get(self.f.source.owner, self.f.task.id)
        await self.f.tasks.mutate(self.f.source.owner, current.id, "cancel",
            TaskMutation(request_id=uuid4(), expected_revision=current.revision, message="Stop."))
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "deleted", result)
        self.assertEqual((await self.f.tasks.get(self.f.source.owner, current.id)).status, TaskStatus.CANCELLED)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "rejected")
        self.assertEqual(self.storage.uploads, 1)

    async def test_recovery_removed_source_cannot_be_adopted(self):
        await self.stopped_upload()
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where id=(select scope_payload_id from jarvis_operation_grants where id=$1)""", self.parent)
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "deleted", result)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "rejected")

    async def test_recovery_revocation_during_storage_read_fences_adoption(self):
        await self.stopped_upload()
        read = self.storage.read_object
        async def revoke(**arguments):
            content = await read(**arguments)
            async with self.f.source.pool.acquire() as connection:
                await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.parent)
            return content
        self.storage.read_object = revoke
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "unresolved", result)
        self.storage.read_object = read
        cleaned = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(cleaned[0]["phase"], "deleted", cleaned)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)

    async def test_recovery_owner_boundary_and_recorded_area_not_default_area(self):
        await self.stopped_upload()
        from app.tasks.file_recovery import TaskFileRecovery
        recovery = TaskFileRecovery(self.f.source.sources, self.f.tasks, self.storage,
            owner_id=self.f.source.owner, area_id=uuid4())
        with self.assertRaises(SourceError):
            await recovery.reconcile_pending(owner_id=self.f.source.other)
        result = await recovery.reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "available", result)
        self.assertEqual(self.storage.uploads, 1)

    async def test_recovery_unknown_missing_upload_is_not_closed_on_cancel(self):
        await self.stopped_upload("fail_before")
        current = await self.f.tasks.get(self.f.source.owner, self.f.task.id)
        await self.f.tasks.mutate(self.f.source.owner, current.id, "cancel",
            TaskMutation(request_id=uuid4(), expected_revision=current.revision, message="Stop."))
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "delete_pending", result)
        self.assertEqual((await self.f.tasks.get(self.f.source.owner, current.id)).status, TaskStatus.UNKNOWN)
        self.assertEqual(self.storage.uploads, 1)

    async def test_recovery_finishes_accepted_file_deletion_after_grant_revocation(self):
        saved = await self.publish()
        file_id = UUID(saved["file"]["id"])
        remove = self.storage.delete_object
        async def unavailable(**arguments):
            raise RuntimeError("Synthetic interrupted cleanup")
        self.storage.delete_object = unavailable
        with self.assertRaises(RuntimeError):
            await self.foreground.delete_source_file(owner_id=self.f.source.owner, file_id=file_id)
        async with self.f.source.pool.acquire() as connection:
            await connection.execute("update jarvis_operation_grants set status='revoked' where id=$1", self.parent)
        self.storage.delete_object = remove
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "deleted", result)
        self.assertEqual(await self.recovery().reconcile_pending(owner_id=self.f.source.owner), [])
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "accepted")
            self.assertEqual(await connection.fetchval("select phase from jarvis_file_publications"), "deleted")

    async def test_recovery_closes_prepared_publication_without_an_upload(self):
        original = self.files.publications.publish
        async def stop_before_upload(**arguments):
            await self.files.publications.prepare(**arguments)
            raise RuntimeError("Synthetic stop before upload")
        self.files.publications.publish = stop_before_upload
        try:
            with self.assertRaises(RuntimeError):
                await self.publish()
        finally:
            self.files.publications.publish = original
        await self.f.tasks.abandon(self.f.claim)
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "deleted", result)
        self.assertEqual(self.storage.uploads, 0)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "rejected")
        self.assertEqual((await self.f.tasks.get(self.f.source.owner, self.f.task.id)).status, TaskStatus.PAUSED)

    async def test_recovery_exact_text_uses_its_original_grant_and_receipt(self):
        scope = {"filename": "fixture.txt", "content": "Exact synthetic text.", "description": "Synthetic text file."}
        async with self.f.tasks.transaction(self.f.source.owner) as connection:
            payload = await self.f.tasks.payload(connection, self.f.source.owner, scope)
            grant = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.create_text',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.f.source.owner, str(uuid4()), self.f.event, payload)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     self.f.source.owner, self.f.task.id, grant)
        bridge = await TaskToolBridge.from_modules(self.context, (FileToolModule(self.files, self.storage),),
            capability_tools={"files.create_text": "create_text_file"}, memory_evidence=self.evidence,
            receipt_revalidator=self.files.revalidate_receipt)
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await bridge.invoke("create_text_file", {"operation_ref": "op_" + grant.hex},
                                provider_call_id="fixture-exact-text-crash")
        self.storage.mode = None
        await self.f.tasks.abandon(self.f.claim)
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "available", result)
        async with self.f.source.pool.acquire() as connection:
            file_id = await connection.fetchval("select id from jarvis_files where filename='fixture.txt'")
        self.assertEqual(await self.foreground.read_content(owner_id=self.f.source.owner, file_id=file_id),
                         scope["content"].encode())
        self.assertEqual(self.storage.uploads, 1)

    async def test_recovery_wrong_storage_bytes_do_not_create_a_success_receipt(self):
        await self.stopped_upload()
        original = self.storage.read_object
        async def corrupt(**arguments):
            await original(**arguments)
            return b"Different synthetic bytes"
        self.storage.read_object = corrupt
        result = await self.recovery().reconcile_pending(owner_id=self.f.source.owner)
        self.assertEqual(result[0]["phase"], "unresolved", result)
        async with self.f.source.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_files"), 0)
            self.assertEqual(await connection.fetchval("select status from jarvis_task_actions"), "outcome_unknown")
        self.storage.read_object = original
        self.assertEqual((await self.recovery().reconcile_pending(owner_id=self.f.source.owner))[0]["phase"], "available")
        self.assertEqual(self.storage.uploads, 1)
