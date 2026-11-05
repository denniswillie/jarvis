"""Correction and logical-forgetting acceptance with real scoped source/task state."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule, audit_source_lifecycle_result
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.outbox import OutboxRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture
from test_second_brain_ingestion_integration import Provider


class SourceLifecycleInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_mutations_refuse_bad_arguments_and_untrusted_invocations(self):
        pool = object()
        module = SourceLifecycleToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool),
                                           "fixture", uuid4())
        tools = {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, "fixture", uuid4()))}
        base = {"source_id": str(uuid4()), "expected_revision": 1}
        for name in ("correct_memory_source", "forget_memory_source"):
            valid = base | ({"replacement": "An explicit correction"} if name.startswith("correct") else {})
            for change in ({"expected_revision": True}, {"owner_id": "foreign"}, {"source_id": "private prose"}):
                with self.subTest(name=name, change=change):
                    result = await tools[name].handler(valid | change)
                    self.assertEqual(result["error"]["code"], "invalid_tool_input")
            result = await tools[name].handler(valid)
            self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
            with bind_tool_invocation(provider_call_id="fixture", request_event_id=uuid4(), request_revision=1,
                                      task=TaskToolInvocation(uuid4(), 1, uuid4())):
                result = await tools[name].handler(valid)
            self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        for replacement in ("", " ", "\x00", "x" * 8001, 4):
            result = await tools["correct_memory_source"].handler(base | {"replacement": replacement})
            self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_lifecycle_audit_omits_replacement_and_preserved_originals(self):
        result = audit_source_lifecycle_result({"status": "success", "source_state": {
            "source_id": str(uuid4()), "revision": 2, "generation": 3, "state": "forgotten",
            "replacement": "Private original", "local_originals": {"text": "Private original"}}})
        self.assertEqual(result["state"], "forgotten")
        self.assertNotIn("Private original", json.dumps(result))


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class SourceLifecycleDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks)
        self.publisher = ContextRepository(
            self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.lifecycle = SourceLifecycleService(self.f.sources, self.tasks)
        self.when = datetime(2026, 9, 15, 12, tzinfo=UTC)
        self.sessions = {}
        async with self.f.pool.acquire() as connection:
            for owner in (self.f.owner, self.f.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
        self.context = ToolBuildContext(None, self.f.owner, self.sessions[self.f.owner])
        self.module = SourceLifecycleToolModule(self.f.sources, self.tasks, self.f.owner, self.f.area)
        self.tools = {tool.name: tool for tool in self.module.build_tools(self.context)}
        self.provider = Provider()
        self.worker = IngestionWorker(OutboxRepository(self.f.sources, retry_seconds=1), self.provider)

    async def request(self, text="Apply the explicit fixture source operation.", *, owner=None, previous=None,
                      finality="final", kind="user_transcript", publish=False):
        owner = owner or self.f.owner
        event = ContextIngress(
            session_id=self.sessions[owner], source_namespace="voice:source-lifecycle",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            kind=kind, trust="assistant_output" if kind == "assistant_output" else "user_input",
            finality=finality, source_time=self.when, expected_revision=previous.revision if previous else 0,
            payload={"text": text},
        )
        return await (self.publisher if publish else self.history).append(owner, event)

    async def invoke(self, operation, source, *, request=None, revision=None, tools=None, **arguments):
        request = request or await self.request()
        with bind_tool_invocation(provider_call_id=str(uuid4()), request_event_id=request.id,
                                 request_revision=request.revision):
            return await (tools or self.tools)[operation + "_memory_source"].handler({
                "source_id": str(source.source_id),
                "expected_revision": source.revision if revision is None else revision, **arguments})

    async def row(self, source):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow(
                "select * from jarvis_knowledge_sources where owner_id=$1 and id=$2", self.f.owner, source.source_id)

    async def payload(self, identity):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.f.owner, identity)

    async def mapped(self, record):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval(
                "select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                self.f.owner, record.id)

    async def settle(self):
        for _ in range(20):
            self.provider.complete()
            await self.f.ready_jobs()
            if await self.worker.step(self.f.owner, "source-lifecycle-worker") is None:
                return
        self.fail("Synthetic source lifecycle outbox did not settle")

    async def test_correction_keeps_historical_original_and_exact_request_provenance(self):
        original = await self.f.publish("The robot is complete.", trust="sensor")
        request = await self.request("It is not complete; motor calibration remains.")
        result = await self.invoke("correct", original, request=request,
                                   replacement="The robot still needs motor calibration.")
        self.assertEqual(result["status"], "success")
        corrected = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        self.assertEqual((corrected.revision, corrected.generation, corrected.trust.value), (2, 2, "derived"))
        self.assertEqual(corrected.source_time, request.source_time)
        self.assertEqual(corrected.content["request"], {"event_id": str(request.id), "revision": request.revision})
        self.assertEqual(corrected.content["replaces"]["document_id"], original.document_id)
        historical = await self.f.sources.read(self.f.owner, self.f.area, original.source_id, revision=1)
        self.assertFalse(historical.is_current)
        self.assertEqual((historical.content, historical.source_time, historical.trust),
                         (original.content, original.source_time, original.trust))
        with self.assertRaises(SourceError):
            await self.f.sources.revalidate(original)
        async with self.f.pool.acquire() as connection:
            parents = await connection.fetchval("""select array_agg(source_payload_id)
                from jarvis_context_payload_dependencies where owner_id=$1 and dependent_payload_id=$2""",
                self.f.owner, corrected.payload_id)
        self.assertEqual(set(parents), {original.payload_id, request.payload_id})
        self.assertEqual(corrected.context_event_id, UUID(result["source_state"]["correction_event_id"]))

    async def test_concurrent_correction_and_stale_retry_create_only_one_version(self):
        original = await self.f.publish()
        request = await self.request()
        results = await asyncio.gather(*[
            self.invoke("correct", original, request=request, replacement="One corrected statement.") for _ in range(2)])
        self.assertEqual(sorted(item["status"] for item in results), ["failed", "success"])
        failure = next(item for item in results if item["status"] == "failed")
        self.assertEqual(failure["error"]["code"], "knowledge_source_revision_conflict")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_source_versions where source_id=$1", original.source_id), 2)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_events where source_namespace='knowledge:source-correction'"), 1)

    async def test_late_asr_revision_cannot_replace_a_user_authored_correction(self):
        spoken = await self.request("The robot is complete.", publish=True)
        original = await self.f.sources.read(self.f.owner, self.f.area, await self.mapped(spoken))
        result = await self.invoke("correct", original, replacement="Motor calibration remains.")
        self.assertEqual(result["status"], "success")
        corrected = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        revised = await self.request("The robot was complete.", previous=spoken, finality="revised", publish=True)
        self.assertTrue(revised.authorizes_request)
        self.assertIsNone(await self.mapped(revised))
        current = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        self.assertEqual((current.document_id, current.content), (corrected.document_id, corrected.content))

    async def test_forgetting_preserves_originals_and_shared_evidence_but_blocks_source_replay(self):
        original = await self.f.publish("Keep the actual original, not its automatic recall.")
        sibling = await self.f.sources.publish(self.f.owner, SourcePublication(
            area_id=self.f.area, namespace="fixture", source_key="shared", payload_id=original.payload_id,
            source_type=original.source_type, trust=original.trust, source_time=original.source_time,
            time_basis=original.time_basis, expected_revision=0))
        result = await self.invoke("forget", original)
        self.assertEqual(result["source_state"]["state"], "forgotten")
        self.assertFalse(result["source_state"]["originals_deleted"])
        self.assertIsNotNone((await self.payload(original.payload_id))["content"])
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, sibling.source_id)).content, sibling.content)
        for revision in (None, 1):
            with self.assertRaises(SourceError):
                await self.f.sources.read(self.f.owner, self.f.area, original.source_id, revision=revision)
        with self.assertRaises(SourceError) as caught:
            await self.f.publish("Automatic paraphrase must not restore the same source.", expected_revision=1)
        self.assertEqual(caught.exception.code, "knowledge_source_forgotten")
        repeated = await self.invoke("forget", original)
        self.assertFalse(repeated["source_state"]["changed"])
        self.assertEqual(repeated["source_state"]["generation"], result["source_state"]["generation"])
        with self.assertRaises(SourceError):
            await self.f.sources.delete_source(self.f.owner, self.f.area, original.source_id, expected_revision=1)

    async def test_forgotten_transcript_can_be_revised_without_capture_failure_or_republication(self):
        spoken = await self.request("A detail to stop recalling.", publish=True)
        original = await self.f.sources.read(self.f.owner, self.f.area, await self.mapped(spoken))
        self.assertEqual((await self.invoke("forget", original))["status"], "success")
        revised = await self.request("An ASR correction of that detail.", previous=spoken,
                                     finality="revised", publish=True)
        self.assertTrue(revised.authorizes_request)
        self.assertIsNone(await self.mapped(revised))
        self.assertEqual((await self.row(original))["status"], "forgotten")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_knowledge_source_versions where source_id=$1", original.source_id), 1)

    async def test_unavailable_transition_and_area_reopen_do_not_remove_forgetting_marker(self):
        original = await self.f.publish()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where id=$1",
                                     self.f.area)
        self.assertEqual((await self.invoke("forget", original))["status"], "success")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='expired',content=null where id=$1",
                                     original.payload_id)
            await connection.execute("update jarvis_knowledge_areas set status='active',revision=revision+1 where id=$1",
                                     self.f.area)
        self.assertEqual((await self.row(original))["status"], "forgotten")
        with self.assertRaises(SourceError) as caught:
            await self.f.publish("Upstream refresh", expected_revision=1)
        self.assertEqual(caught.exception.code, "knowledge_source_forgotten")

    async def test_database_prevents_automatic_unforget_but_explicit_deletion_still_works(self):
        original = await self.f.publish()
        await self.invoke("forget", original)
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("""update jarvis_knowledge_sources
                    set status='active',current_revision=current_revision+1,generation=generation+1 where id=$1""",
                    original.source_id)
        result = await self.invoke("correct", original, replacement="Cannot unforget by correction.")
        self.assertEqual(result["error"]["code"], "knowledge_source_forgotten")
        deleted = await self.invoke("delete", original)
        self.assertEqual(deleted["status"], "success")
        self.assertEqual((await self.row(original))["status"], "deleted")
        self.assertIsNone((await self.payload(original.payload_id))["content"])

    async def test_late_provider_retention_correction_and_forgetting_cleanup_preserve_originals(self):
        original = await self.f.publish()
        self.assertEqual(await self.worker.step(self.f.owner, "fixture"), "accepted")
        await self.invoke("correct", original, replacement="A different wording of the corrected observation.")
        corrected = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        await self.settle()
        self.assertNotIn(original.document_id, self.provider.documents)
        self.assertIn(corrected.document_id, self.provider.documents)
        await self.invoke("forget", corrected)
        await self.settle()
        self.assertNotIn(corrected.document_id, self.provider.documents)
        for source in (original, corrected):
            self.assertIsNotNone((await self.payload(source.payload_id))["content"])
        state = await self.tools["inspect_memory_source_state"].handler({"source_id": str(original.source_id)})
        self.assertEqual(state["source_state"]["state"], "forgotten")
        self.assertEqual(state["source_state"]["provider_cleanup"]["cleanup_recorded"], 2)
        before = list(self.provider.sent)
        await self.invoke("forget", corrected)
        await self.settle()
        self.assertEqual(self.provider.sent, before)

    async def test_untrusted_stale_removed_and_foreign_requests_cannot_mutate(self):
        original = await self.f.publish()
        stale = await self.request()
        await self.request(previous=stale, finality="revised")
        removed = await self.request()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     removed.payload_id)
        requests = [stale, removed, await self.request(owner=self.f.other),
                    await self.request(finality="provisional"), await self.request(kind="assistant_output")]
        for request in requests:
            for operation in ("correct", "forget"):
                with self.subTest(request=request.id, operation=operation):
                    args = {"replacement": "Unaccepted correction"} if operation == "correct" else {}
                    result = await self.invoke(operation, original, request=request, **args)
                    self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertEqual((await self.row(original))["generation"], 1)

    async def test_runtime_closure_rolls_back_correction_and_original_suppression(self):
        original = await self.f.publish()
        for operation in ("correct", "forget"):
            calls = 0
            def access(owner):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise SourceError("knowledge_intake_closed", "Fixture shutdown", 503)
            module = SourceLifecycleToolModule(self.f.sources, self.tasks, self.f.owner, self.f.area, access)
            tools = {tool.name: tool for tool in module.build_tools(self.context)}
            calls = 0
            args = {"replacement": "Do not commit this correction."} if operation == "correct" else {}
            result = await self.invoke(operation, original, tools=tools, **args)
            self.assertEqual(result["error"]["code"], "knowledge_intake_closed")
            self.assertEqual((await self.row(original))["generation"], 1)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_events where source_namespace='knowledge:source-correction'"), 0)

    async def task_invalidation(self, operation):
        original = await self.f.publish("The robot is ready.")
        request = await self.request("Review the robot project.")
        task = await self.tasks.create(self.f.owner, TaskCreate(
            request_id=uuid4(), title="Robot review", requested_outcome="Explain remaining work",
            session_id=self.context.session_id, knowledge_area_id=self.f.area),
            model="fixture-model", source_event_id=request.id)
        claim = await self.tasks.claim_next("fixture-task", (task.backend_kind,), owner_id=self.f.owner)
        self.assertIsNotNone(claim)
        manifest = await self.history.store_manifest(
            claim, watermark=request.sequence, records=[request],
            content={"summary": "All assembly is finished; this is a differently worded derived conclusion."})
        async with self.f.pool.acquire() as connection:
            await connection.execute("""insert into jarvis_task_memory_sources(
                owner_id,manifest_id,area_id,source_id,source_revision,source_generation,payload_id,content_digest,source_usage)
                values($1,$2,$3,$4,$5,$6,$7,$8,'current')""",
                self.f.owner, manifest.id, self.f.area, original.source_id, original.revision,
                original.generation, original.payload_id, original.content_digest)
        args = {"replacement": "Calibration remains."} if operation == "correct" else {}
        self.assertEqual((await self.invoke(operation, original, **args))["status"], "success")
        with self.assertRaises(TaskError):
            await self.history.read_manifest(self.f.owner, task.id, manifest.id)
        self.assertTrue(await self.tasks.finish(
            claim, WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "The obsolete conclusion must not be published."})))
        settled = await self.tasks.get(self.f.owner, task.id)
        self.assertEqual(settled.status, TaskStatus.PAUSED)
        self.assertIsNone(settled.result_payload_id)
        self.assertIsNotNone((await self.payload(original.payload_id))["content"])
        async with self.f.pool.acquire() as connection:
            status = await connection.fetchval("select status from jarvis_task_context_manifests where id=$1", manifest.id)
            self.assertFalse(await connection.fetchval("""select exists(select 1 from jarvis_context_payloads
                where owner_id=$1 and content->>'answer'='The obsolete conclusion must not be published.')""",
                self.f.owner))
        self.assertEqual(status, "superseded" if operation == "correct" else "revoked")
        if operation == "forget":
            self.assertIsNone((await self.payload(manifest.payload_id))["content"])

    async def test_correction_fences_actual_task_with_old_source_bound_conclusion(self):
        await self.task_invalidation("correct")

    async def test_forgetting_revokes_actual_task_and_erases_derived_copy_not_original(self):
        await self.task_invalidation("forget")

    async def test_removed_correction_request_invalidates_derived_replacement(self):
        original = await self.f.publish()
        request = await self.request()
        await self.invoke("correct", original, request=request, replacement="A statement dependent on this request.")
        corrected = await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     request.payload_id)
        self.assertIsNone((await self.payload(corrected.payload_id))["content"])
        self.assertIsNotNone((await self.payload(original.payload_id))["content"])
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, original.source_id)
