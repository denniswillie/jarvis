"""Bounded source cleanup recovery, using real owned database state."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.deletion import SourceDeletionService
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle_tools import (
    SourceLifecycleToolModule, audit_source_lifecycle_arguments, audit_source_lifecycle_result,
)
from app.knowledge.models import SourceError
from app.knowledge.outbox import IngestionClaimLost, OutboxRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class CleanupRecoveryInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_bounds_scope_and_foreground_authority(self):
        pool = object()
        module = SourceLifecycleToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool),
                                           "fixture", uuid4())
        tools = {t.name: t for t in module.build_tools(ToolBuildContext(None, "fixture", uuid4()))}
        tool = tools["retry_memory_source_cleanup"]
        valid = {"source_id": str(uuid4()), "expected_generation": 1}
        for change in ({"expected_generation": True}, {"expected_generation": "1"},
                       {"expected_generation": 0}, {"limit": 0}, {"limit": 129}, {"limit": False},
                       {"source_id": "private prose"}, {"owner_id": "foreign"}, {"area_id": str(uuid4())},
                       {"expected_revision": 1}, {"request_event_id": str(uuid4())}):
            with self.subTest(change=change):
                self.assertEqual((await tool.handler(valid | change))["error"]["code"], "invalid_tool_input")
        self.assertEqual((await tool.handler(valid))["error"]["code"], "knowledge_request_not_finalized")
        with bind_tool_invocation(provider_call_id="fixture", request_event_id=uuid4(), request_revision=1,
                                  task=TaskToolInvocation(uuid4(), 1, uuid4())):
            self.assertEqual((await tool.handler(valid))["error"]["code"], "knowledge_request_not_finalized")
        self.assertNotIn("retry_memory_source_cleanup",
                         {t.name for t in module.build_tools(ToolBuildContext(None, "fixture", None))})
        args = audit_source_lifecycle_arguments(valid | {"replacement": "private prose"})
        result = audit_source_lifecycle_result({"status": "success", "source_state": {
            "source_id": valid["source_id"], "generation": 1, "requeued": 2,
            "remaining_retryable": 3, "blocked_current_originals": 1, "text": "private prose"}})
        self.assertEqual(args["expected_generation"], 1)
        self.assertEqual(result["requeued"], 2)
        self.assertNotIn("private prose", json.dumps([args, result]))


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class CleanupRecoveryDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        self.deletions = SourceDeletionService(self.f.sources)
        self.repository = OutboxRepository(self.f.sources, retry_seconds=1, max_failures=1)
        self.sessions = {}
        async with self.f.pool.acquire() as connection:
            for owner in (self.f.owner, self.f.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
        self.context = ToolBuildContext(None, self.f.owner, self.sessions[self.f.owner])
        self.tools = self.build_tools()

    def build_tools(self, access=None):
        module = SourceLifecycleToolModule(self.f.sources, self.tasks, self.f.owner, self.f.area, access)
        return {t.name: t for t in module.build_tools(self.context)}

    async def request(self, *, owner=None, previous=None, finality="final", kind="user_transcript"):
        owner = owner or self.f.owner
        return await self.history.append(owner, ContextIngress(
            session_id=self.sessions[owner], source_namespace="voice:cleanup-recovery",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            expected_revision=previous.revision if previous else 0, kind=kind, finality=finality,
            trust="assistant_output" if kind == "assistant_output" else "user_input",
            source_time=datetime.now(UTC), payload={"text": "Finish the existing source cleanup."}))

    async def invoke(self, name, source, *, request=None, tools=None, **args):
        request = request or await self.request()
        with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request.id,
                                  request_revision=request.revision):
            return await (tools or self.tools)[name].handler({"source_id": str(source.source_id), **args})

    async def retry(self, source, generation, **kwargs):
        return await self.invoke("retry_memory_source_cleanup", source, expected_generation=generation, **kwargs)

    async def failed(self, source, *, inventory=()):
        # Explicit fixture fault injection. Never fabricate another owner or delete a current real document.
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""insert into jarvis_memory_outbox
                (owner_id,document_id,generation,operation,status,failure_count,last_error,cleanup_operation_ids)
                values($1,$2,$3,'delete','failed',8,'fixture_failure',$4::uuid[])
                on conflict(owner_id,document_id,operation) do update set
                status='failed',failure_count=8,last_error='fixture_failure',
                cleanup_operation_ids=excluded.cleanup_operation_ids returning *""",
                self.f.owner, source.document_id, source.generation, list(inventory))

    async def outbox(self, source, operation="delete"):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select * from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2 and operation=$3""",
                self.f.owner, source.document_id, operation)

    async def test_bounded_retry_preserves_identity_inventory_dispatch_and_fences_old_attempts(self):
        versions = [await self.f.publish("Version one")]
        for revision in (1, 2, 3):
            versions.append(await self.f.publish("Next version", expected_revision=revision))
        before = [await self.failed(source, inventory=(uuid4(), uuid4())) for source in versions[:-1]]
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_memory_outbox set dispatch_count=3
                where operation='delete'""")
        first = (await self.retry(versions[-1], 4, limit=2))["source_state"]
        self.assertEqual((first["requeued"], first["remaining_failed"], first["remaining_retryable"]), (2, 1, 1))
        self.assertEqual(first["blocked_current_originals"], 0)
        self.assertFalse(first["provider_contacted"])
        self.assertFalse(first["cleanup_completion_verified"])
        for source, old in zip(versions[:2], before[:2]):
            new = await self.outbox(source)
            self.assertEqual((new["id"], new["cleanup_operation_ids"], new["dispatch_count"]),
                             (old["id"], old["cleanup_operation_ids"], 3))
            self.assertEqual((new["status"], new["generation"], new["fence"], new["failure_count"]),
                             ("pending", 4, old["fence"] + 1, 0))
            self.assertIsNone(new["last_error"])
            self.assertIsNone(new["worker_id"])
            self.assertIsNone(new["lease_until"])
        second = (await self.retry(versions[-1], 4, limit=2))["source_state"]
        self.assertEqual((second["requeued"], second["remaining_failed"]), (1, 0))
        self.assertEqual((await self.retry(versions[-1], 4))["source_state"]["requeued"], 0)

    async def test_current_original_stays_protected_even_in_archived_area(self):
        original = await self.f.publish()
        before = await self.failed(original)
        for archived in (False, True):
            if archived:
                async with self.f.pool.acquire() as connection:
                    await connection.execute("""update jarvis_knowledge_areas
                        set status='archived',revision=revision+1 where id=$1""", self.f.area)
            state = (await self.retry(original, 1))["source_state"]
            self.assertEqual((state["requeued"], state["remaining_failed"], state["remaining_retryable"],
                              state["blocked_current_originals"]), (0, 1, 0, 1))
            self.assertEqual(await self.outbox(original), before)
        # The worker independently applies the same guard even if a failed row is manually requeued.
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_memory_outbox set status='pending' where id=$1", before["id"])
        provider = SimpleNamespace(delete_document=AsyncMock(), erase_operations=AsyncMock())
        self.assertEqual(await IngestionWorker(self.repository, provider).step(self.f.owner, "fixture"), "failed")
        provider.delete_document.assert_not_awaited()
        self.assertEqual((await self.outbox(original))["last_error"], "current_source_cleanup_refused")

    async def test_obsolete_and_forgotten_cleanup_does_not_require_original_deletion(self):
        original = await self.f.publish()
        corrected = await self.invoke("correct_memory_source", original, expected_revision=1,
                                      replacement="A corrected synthetic statement.")
        self.assertEqual(corrected["status"], "success")
        current = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        await self.failed(original)
        self.assertEqual((await self.retry(current, current.generation))["source_state"]["requeued"], 1)
        with self.assertRaises(SourceError) as error:
            await self.deletions.retry_failed(self.f.owner, self.f.area, current.source_id,
                                              expected_generation=current.generation)
        self.assertEqual(error.exception.code, "knowledge_source_not_deleted")
        forgotten = await self.invoke("forget_memory_source", current, expected_revision=2)
        await self.failed(current)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1",
                                     self.f.area)
        state = (await self.retry(current, forgotten["source_state"]["generation"]))["source_state"]
        self.assertEqual((state["requeued"], state["state"]), (1, "forgotten"))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payloads
                where id=any($1::uuid[]) and content is not null and availability='available'""",
                [original.payload_id, current.payload_id]), 2)

    async def test_deleted_source_supports_both_legacy_and_foreground_retry(self):
        original = await self.f.publish()
        await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id, expected_revision=1)
        await self.failed(original)
        state = await self.deletions.inspect(self.f.owner, self.f.area, original.source_id)
        result = await self.deletions.retry_failed(
            self.f.owner, self.f.area, original.source_id, expected_generation=state["generation"])
        self.assertEqual(result["requeued"], 1)
        await self.failed(original)
        self.assertEqual((await self.retry(original, state["generation"]))["source_state"]["requeued"], 1)

    async def test_scope_and_generation_conflicts_leave_failed_rows_unchanged(self):
        original = await self.f.publish()
        current = await self.f.publish("Revised", expected_revision=1)
        before = await self.failed(original)
        result = await self.retry(current, 1)
        self.assertEqual(result["error"]["code"], "knowledge_source_generation_conflict")
        for owner, area in ((self.f.other, self.f.area), (self.f.owner, uuid4())):
            with self.assertRaises(SourceError) as error:
                await self.deletions.retry_failed_cleanup(owner, area, original.source_id, expected_generation=2)
            self.assertEqual(error.exception.code, "knowledge_source_not_found")
        self.assertEqual(await self.outbox(original), before)

    async def test_no_existing_failure_means_no_new_intent_or_lease_reset(self):
        original = await self.f.publish()
        self.assertEqual((await self.retry(original, 1))["source_state"]["requeued"], 0)
        self.assertIsNone(await self.outbox(original))
        current = await self.f.publish("Revised", expected_revision=1)
        claim = await self.repository.claim(self.f.owner, "in-progress")
        self.assertEqual(claim.operation, "delete")
        before = await self.outbox(original)
        self.assertEqual((await self.retry(current, 2))["source_state"]["requeued"], 0)
        self.assertEqual(await self.outbox(original), before)

    async def test_concurrent_retries_and_stale_worker_cannot_repeat_effect(self):
        original = await self.f.publish()
        current = await self.f.publish("Revised", expected_revision=1)
        claim = await self.repository.claim(self.f.owner, "old-worker")
        await self.repository.prepare_cleanup(claim, ())
        await self.repository.failure(claim, "fixture_failure", retryable=False)
        request = await self.request()
        results = await asyncio.gather(self.retry(current, 2, request=request), self.retry(current, 2, request=request))
        self.assertEqual(sorted(r["source_state"]["requeued"] for r in results), [0, 1])
        with self.assertRaises(IngestionClaimLost):
            await self.repository.settle(claim, "done")
        provider = SimpleNamespace(delete_document=AsyncMock(), erase_operations=AsyncMock(return_value=True))
        self.assertEqual(await IngestionWorker(self.repository, provider).step(self.f.owner, "new-worker"), "done")
        self.assertEqual((await self.outbox(original))["id"], claim.id)
        self.assertEqual(provider.delete_document.await_count, 2)
        self.assertEqual((await self.retry(current, 2))["source_state"]["requeued"], 0)

    async def test_stale_removed_non_user_and_foreign_requests_cannot_retry(self):
        original = await self.f.publish()
        current = await self.f.publish("Revised", expected_revision=1)
        before = await self.failed(original)
        stale = await self.request()
        await self.request(previous=stale, finality="revised")
        removed = await self.request()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     removed.payload_id)
        for request in (stale, removed, await self.request(owner=self.f.other),
                        await self.request(finality="provisional"), await self.request(kind="assistant_output")):
            result = await self.retry(current, 2, request=request)
            self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertEqual(await self.outbox(original), before)

    async def test_runtime_closure_rolls_back_the_retry(self):
        original = await self.f.publish()
        current = await self.f.publish("Revised", expected_revision=1)
        before = await self.failed(original)
        calls = 0
        def access(owner):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise SourceError("knowledge_intake_closed", "Fixture runtime closed", 503)
        tools = self.build_tools(access)
        calls = 0
        result = await self.retry(current, 2, tools=tools)
        self.assertEqual(result["error"]["code"], "knowledge_intake_closed")
        self.assertEqual(await self.outbox(original), before)

    async def test_transactional_recovery_refuses_autocommit(self):
        original = await self.f.publish()
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(RuntimeError):
                await self.deletions.retry_failed_cleanup_in_transaction(
                    connection, self.f.owner, self.f.area, original.source_id, expected_generation=1)
