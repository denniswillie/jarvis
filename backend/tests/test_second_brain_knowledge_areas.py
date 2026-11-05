"""Reversible area visibility with actual scoped DB state and simulated provider I/O."""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from pydantic import SecretStr

from app.knowledge.areas import KnowledgeAreaService
from app.knowledge.area_tools import KnowledgeAreaToolModule, audit_area_arguments, audit_area_result
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.models import SourceError
from app.knowledge.outbox import OutboxRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import KnowledgeConfiguration, TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate
from app.tasks.repository import TaskRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture
from test_second_brain_ingestion_integration import Provider


class AreaToolContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session = "fixture", uuid4()
        self.module = KnowledgeAreaToolModule(SimpleNamespace(), self.owner)
        self.tools = {tool.name: tool for tool in self.module.build_tools(
            ToolBuildContext(None, self.owner, self.session))}

    async def test_explicit_reversible_tools_do_not_accept_owner_or_deletion_authority(self):
        self.assertEqual(set(self.tools), {"list_knowledge_areas", "get_knowledge_area", "set_knowledge_area_state",
                                          "create_knowledge_areas", "update_knowledge_area",
                                          "get_knowledge_area_history", "list_knowledge_area_sources"})
        self.assertEqual([tool.name for tool in self.tools.values() if tool.mutates_state],
                         ["set_knowledge_area_state", "create_knowledge_areas", "update_knowledge_area"])
        for tool in self.tools.values():
            self.assertFalse(tool.input_schema["additionalProperties"])
            self.assertNotIn("owner_id", tool.input_schema["properties"])
            self.assertNotIn("request_event_id", tool.input_schema["properties"])
        readonly = self.module.build_tools(ToolBuildContext(None, self.owner, None))
        self.assertEqual(len(readonly), 4)
        with self.assertRaises(SourceError):
            self.module.build_tools(ToolBuildContext(None, "foreign", self.session))

    async def test_bad_arguments_are_refused_before_service_access(self):
        valid = {"area_id": str(uuid4()), "state": "archived", "expected_revision": 1}
        for args in ({**valid, "owner_id": "foreign"}, {**valid, "state": "deleted"},
                     {**valid, "expected_revision": True}, {**valid, "expected_revision": "1"},
                     {**valid, "area_id": "private invalid ID"}):
            with self.subTest(args=args):
                result = await self.tools["set_knowledge_area_state"].handler(args)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        for args in ({"limit": 0}, {"limit": 65}, {"include_archived": "yes"}):
            result = await self.tools["list_knowledge_areas"].handler(args)
            self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_missing_user_source_and_worker_context_cannot_mutate_area(self):
        args = {"area_id": str(uuid4()), "state": "archived", "expected_revision": 1}
        result = await self.tools["set_knowledge_area_state"].handler(args)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        with bind_tool_invocation(provider_call_id="fixture", request_event_id=uuid4(), request_revision=1,
                                  task=TaskToolInvocation(uuid4(), 1, uuid4())):
            result = await self.tools["set_knowledge_area_state"].handler(args)
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")

    async def test_audits_do_not_copy_names_domains_or_invalid_input_prose(self):
        secret = "S23 private area name"
        args = audit_area_arguments({"area_id": secret, "domain": secret, "state": secret})
        result = audit_area_result({"status": "success", "area": {
            "id": str(uuid4()), "revision": 2, "status": "archived", "name": secret, "domain": secret}})
        self.assertNotIn(secret, json.dumps([args, result]))
        self.assertTrue(result["content_omitted"])

    async def test_service_pool_identity_and_bounds_are_enforced_before_database(self):
        pool = object()
        with self.assertRaises(ValueError):
            KnowledgeAreaService(SimpleNamespace(pool=pool), SimpleNamespace(pool=object()))
        service = KnowledgeAreaService(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool))
        for kwargs in ({"limit": True}, {"limit": 65}, {"after_id": "not-a-uuid"}, {"include_archived": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await service.list(self.owner, **kwargs)


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class KnowledgeAreaDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.service = KnowledgeAreaService(self.f.sources, self.tasks)
        self.sessions = {}
        async with self.f.pool.acquire() as connection:
            for owner in (self.f.owner, self.f.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
        self.context = ToolBuildContext(None, self.f.owner, self.sessions[self.f.owner])
        self.tools = {tool.name: tool for tool in KnowledgeAreaToolModule(
            self.service, self.f.owner).build_tools(self.context)}
        self.provider = Provider()
        self.outbox = OutboxRepository(self.f.sources, retry_seconds=1)
        self.worker = IngestionWorker(self.outbox, self.provider)

    async def request(self, *, owner=None, previous=None, trust="user_input", kind="user_request", finality="final"):
        owner = owner or self.f.owner
        async with self.tasks.transaction(owner) as connection:
            payload = await self.tasks.payload(connection, owner, {"text": "Change the explicitly selected fixture area's visibility"})
            return await connection.fetchrow("""insert into jarvis_context_events(
                owner_id,session_id,source_namespace,source_event_id,revision,supersedes_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-area-command',$3,$4,$5,$6,$7,$8,$9) returning *""",
                owner, self.sessions[owner], previous["source_event_id"] if previous else str(uuid4()),
                previous["revision"] + 1 if previous else 1, previous["id"] if previous else None,
                kind, finality, trust, payload)

    async def invoke(self, name, arguments, request=None):
        with bind_tool_invocation(provider_call_id=str(uuid4()),
                                 request_event_id=request["id"] if request else None,
                                 request_revision=request["revision"] if request else None):
            return await self.tools[name].handler(arguments)

    async def set_state(self, state, *, revision=None, area=None, request=None):
        area = area or self.f.area
        if revision is None:
            revision = (await self.service.read(self.f.owner, area))["revision"]
        return await self.invoke("set_knowledge_area_state", {
            "area_id": str(area), "state": state, "expected_revision": revision,
        }, request or await self.request())

    async def step(self):
        await self.f.ready_jobs()
        return await self.worker.step(self.f.owner, "area-fixture-worker")

    async def row(self, document_id):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select * from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2 and operation='retain'""", self.f.owner, document_id)

    async def content(self, payload):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval(
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2", self.f.owner, payload)

    async def capture(self, text, *, previous=None, finality="final", kind="user_transcript"):
        event = ContextIngress(
            session_id=self.context.session_id, source_namespace="voice:archive-regression",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            kind=kind, trust="assistant_output" if kind == "assistant_output" else "user_input",
            finality=finality, expected_revision=previous.revision if previous else 0, payload={"text": text},
        )
        return event, await self.history.append(self.f.owner, event)

    async def memory_mapping(self, event):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select cs.source_id,cs.source_revision,v.payload_id,v.document_id
                from jarvis_knowledge_context_sources cs join jarvis_knowledge_source_versions v
                on v.owner_id=cs.owner_id and v.source_id=cs.source_id and v.revision=cs.source_revision
                where cs.owner_id=$1 and cs.event_id=$2 and cs.event_revision=$3""",
                self.f.owner, event.id, event.revision)

    async def test_archived_capture_can_authorize_reactivation_without_backfilling_paused_messages(self):
        _, before = await self.capture("Archive this project's knowledge for now.")
        original = await self.memory_mapping(before)
        self.assertEqual(original["payload_id"], before.payload_id)
        archived = await self.set_state("archived", request={"id": before.id, "revision": before.revision})
        self.assertEqual(archived["area"]["status"], "archived")

        ingress, reopen = await self.capture("Reactivate this project's knowledge.")
        _, assistant = await self.capture("I can reactivate that area.", kind="assistant_output")
        self.assertTrue(reopen.authorizes_request)
        self.assertFalse(assistant.authorizes_request)
        for event in (reopen, assistant):
            self.assertEqual((await self.history.get(self.f.owner, event.id)).content, event.content)
            self.assertIsNone(await self.memory_mapping(event))
        active = await self.set_state("active", request={"id": reopen.id, "revision": reopen.revision})
        self.assertEqual(active["area"]["status"], "active")
        # Retrying an archived-interval receipt after reopening must not publish it.
        retry = await self.history.append(self.f.owner, ingress)
        self.assertEqual(retry.id, reopen.id)
        self.assertIsNone(await self.memory_mapping(retry))
        _, later = await self.capture("Continue recording this project's new decisions.")
        later_source = await self.memory_mapping(later)
        self.assertEqual(later_source["payload_id"], later.payload_id)
        self.assertEqual((await self.memory_mapping(before))["document_id"], original["document_id"])
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_memory_outbox where owner_id=$1 and operation='retain'", self.f.owner), 2)

    async def test_archived_transcript_revision_preserves_correction_and_rejects_stale_memory(self):
        _, first = await self.capture("The robot is complete.")
        original = await self.memory_mapping(first)
        await self.set_state("archived")
        _, corrected = await self.capture(
            "The robot is not complete; reactivate this area.", previous=first, finality="revised")
        self.assertTrue(corrected.authorizes_request)
        self.assertEqual(corrected.supersedes_event_id, first.id)
        self.assertFalse((await self.history.get(self.f.owner, first.id)).authorizes_request)
        self.assertIsNone(await self.memory_mapping(corrected))
        active = await self.set_state("active", request={"id": corrected.id, "revision": corrected.revision})
        self.assertEqual(active["area"]["status"], "active")
        with self.assertRaises(SourceError) as caught:
            await self.f.sources.read(self.f.owner, self.f.area, original["source_id"])
        self.assertEqual(caught.exception.code, "knowledge_source_unavailable")
        self.assertIsNotNone(await self.content(first.payload_id))
        self.assertEqual((await self.history.get(self.f.owner, corrected.id)).content, corrected.content)
        self.assertEqual((await self.memory_mapping(first))["document_id"], original["document_id"])

    async def test_archived_provisional_capture_needs_final_transcript_to_reactivate(self):
        await self.set_state("archived")
        _, provisional = await self.capture("Reactivate", finality="provisional")
        self.assertFalse(provisional.authorizes_request)
        refused = await self.set_state("active", request={"id": provisional.id, "revision": provisional.revision})
        self.assertEqual(refused["error"]["code"], "task_request_evidence_missing")
        _, final = await self.capture("Reactivate this area.", previous=provisional)
        self.assertTrue(final.authorizes_request)
        active = await self.set_state("active", request={"id": final.id, "revision": final.revision})
        self.assertEqual(active["area"]["status"], "active")
        self.assertIsNone(await self.memory_mapping(provisional))
        self.assertIsNone(await self.memory_mapping(final))

    async def test_archive_capture_exception_does_not_accept_foreign_missing_or_deleted_areas(self):
        async with self.f.pool.acquire() as connection:
            foreign = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain,status)
                values($1,'Foreign archive','test','archived') returning id""", self.f.other)
            await connection.execute("""update jarvis_knowledge_areas set status='deleted',revision=revision+1
                where owner_id=$1 and id=$2""", self.f.owner, self.f.area)
        for area in (foreign, uuid4(), self.f.area):
            with self.subTest(area=area):
                history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, area))
                event = ContextIngress(
                    session_id=self.context.session_id, source_namespace="voice:invalid-area-regression",
                    source_event_id=str(uuid4()), receipt_id=str(uuid4()), kind="user_transcript",
                    trust="user_input", payload={"text": "Do not silently accept a misconfigured scope."})
                with self.assertRaises(SourceError) as caught:
                    await history.append(self.f.owner, event)
                self.assertEqual(caught.exception.code, "knowledge_area_unavailable")
                async with self.f.pool.acquire() as connection:
                    self.assertFalse(await connection.fetchval("""select exists(select 1 from jarvis_context_events
                        where owner_id=$1 and source_namespace=$2 and source_event_id=$3)""",
                        self.f.owner, event.source_namespace, event.source_event_id))

    async def test_archive_and_reactivate_keep_originals_and_journal_visibility_changes(self):
        source = await self.f.publish("S23 retained project decision")
        request = await self.request()
        async with self.f.pool.acquire() as connection:
            before = await connection.fetchval(
                "select last_sequence from jarvis_source_journal_streams where owner_id=$1", self.f.owner)
        archived = await self.set_state("archived", request=request)
        self.assertEqual((archived["area"]["status"], archived["area"]["revision"]), ("archived", 2))
        self.assertFalse(archived["originals_deleted"])
        for operation in (self.f.sources.read(self.f.owner, self.f.area, source.source_id),
                          self.f.sources.resolve_document(self.f.owner, self.f.area, source.document_id),
                          self.f.sources.current_snapshot(self.f.owner, self.f.area)):
            with self.assertRaises(SourceError):
                await operation
        self.assertIsNotNone(await self.content(source.payload_id))
        async with self.f.pool.acquire() as connection:
            after = await connection.fetchval(
                "select last_sequence from jarvis_source_journal_streams where owner_id=$1", self.f.owner)
        self.assertGreater(after, before)
        active = await self.set_state("active")
        self.assertEqual(active["area"]["revision"], 3)
        restored = await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual((restored.payload_id, restored.document_id, restored.generation),
                         (source.payload_id, source.document_id, source.generation))
        self.assertEqual(restored.content, source.content)

    async def test_listing_pages_are_owned_and_include_archives_without_erasing_them(self):
        async with self.f.pool.acquire() as connection:
            other = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Second owned area','test') returning id",
                self.f.owner)
            foreign = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Foreign private name','test') returning id",
                self.f.other)
        await self.set_state("archived", area=other)
        seen, cursor = set(), None
        while True:
            page = await self.invoke("list_knowledge_areas", {"limit": 1, "after_id": cursor})
            self.assertNotIn("Foreign private name", json.dumps(page))
            seen.update(UUID(area["id"]) for area in page["areas"])
            cursor = page["next_after_id"]
            if cursor is None:
                break
        self.assertEqual(seen, {self.f.area, other})
        active = await self.invoke("list_knowledge_areas", {"include_archived": False})
        self.assertEqual([area["id"] for area in active["areas"]], [str(self.f.area)])
        result = await self.invoke("list_knowledge_areas", {"after_id": str(foreign)})
        self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")
        result = await self.set_state("archived", area=foreign, revision=1)
        self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")

    async def test_stale_retry_and_same_state_do_not_create_another_revision(self):
        await self.set_state("archived", revision=1)
        stale = await self.set_state("archived", revision=1)
        self.assertEqual(stale["error"]["code"], "knowledge_area_revision_conflict")
        same = await self.set_state("archived", revision=2)
        self.assertFalse(same["changed"])
        self.assertEqual(same["area"]["revision"], 2)
        old_reopen = await self.set_state("active", revision=1)
        self.assertEqual(old_reopen["error"]["code"], "knowledge_area_revision_conflict")
        self.assertEqual((await self.service.read(self.f.owner, self.f.area))["status"], "archived")

    async def test_untrusted_superseded_removed_and_foreign_requests_cannot_archive(self):
        superseded = await self.request()
        await self.request(previous=superseded, finality="revised")
        removed = await self.request()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                     self.f.owner, removed["payload_id"])
        requests = [
            superseded, removed, await self.request(owner=self.f.other),
            await self.request(trust="external_content", kind="observation"),
            await self.request(trust="assistant_output", kind="assistant_output"),
            await self.request(finality="provisional"),
        ]
        for request in requests:
            with self.subTest(request=request["id"]):
                result = await self.set_state("archived", revision=1, request=request)
                self.assertEqual(result["error"]["code"], "task_request_evidence_missing")
        self.assertEqual((await self.service.read(self.f.owner, self.f.area))["revision"], 1)

    async def test_archiving_pauses_new_dispatch_and_reactivation_reuses_the_same_intent(self):
        source = await self.f.publish()
        before = await self.row(source.document_id)
        await self.set_state("archived")
        self.assertIsNone(await self.step())
        paused = await self.row(source.document_id)
        self.assertEqual((paused["id"], paused["status"], paused["dispatch_count"]),
                         (before["id"], "pending", 0))
        self.assertEqual(self.provider.sent, [])
        await self.set_state("active")
        self.assertEqual(await self.step(), "accepted")
        self.assertEqual(self.provider.sent, [str(before["id"])])

    async def test_expired_predispatch_lease_is_released_without_discarding_original(self):
        source = await self.f.publish()
        claim = await self.outbox.claim(self.f.owner, "before-archive")
        await self.set_state("archived")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second' where id=$1",
                                     claim.id)
        self.assertEqual(await self.step(), "pending")
        self.assertEqual((await self.row(source.document_id))["id"], claim.id)
        self.assertEqual((await self.outbox.diagnostics(self.f.owner))["native_admissions"]["reserved"], 0)
        self.assertIsNotNone(await self.content(source.payload_id))
        self.assertIsNone(await self.step())

    async def test_archive_between_initial_read_and_dispatch_does_not_send_original(self):
        source = await self.f.publish()
        entered, release = asyncio.Event(), asyncio.Event()
        async def wait_at_bank(bank):
            entered.set()
            await release.wait()
        with patch.object(self.provider, "ensure_bank", side_effect=wait_at_bank):
            running = asyncio.create_task(self.step())
            try:
                await asyncio.wait_for(entered.wait(), 5)
                await self.set_state("archived")
                release.set()
                self.assertEqual(await asyncio.wait_for(running, 5), "pending")
            finally:
                release.set()
                await asyncio.gather(running, return_exceptions=True)
        self.assertEqual(self.provider.sent, [])
        self.assertEqual((await self.row(source.document_id))["dispatch_count"], 0)

    async def test_dispatched_retention_can_finish_while_archived_without_public_read_access(self):
        source = await self.f.publish()
        self.assertEqual(await self.step(), "accepted")
        identity = self.provider.sent[0]
        await self.set_state("archived")
        self.provider.complete()
        self.assertEqual(await self.step(), "searchable")
        self.assertEqual(self.provider.sent, [identity])
        self.assertIn(source.document_id, self.provider.documents)
        self.assertIn(identity, self.provider.operations)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, source.source_id)
        self.assertEqual((await self.outbox.diagnostics(self.f.owner))["native_admissions"]["reserved"], 0)

    async def test_missing_dispatched_receipt_stays_unknown_and_is_not_resent_while_archived(self):
        source = await self.f.publish()
        self.assertEqual(await self.step(), "accepted")
        before = await self.row(source.document_id)
        await self.set_state("archived")
        self.provider.operations.clear()
        self.assertEqual(await self.step(), "outcome_unknown")
        after = await self.row(source.document_id)
        self.assertEqual((after["id"], after["dispatch_count"]), (before["id"], 1))
        self.assertEqual(self.provider.sent, [str(before["id"])])
        self.assertEqual((await self.outbox.diagnostics(self.f.owner))["native_admissions"]["reserved"], 1)

    async def test_archive_alone_is_not_authority_to_delete_a_current_native_document(self):
        source = await self.f.publish()
        await self.step()
        self.provider.complete()
        await self.step()
        await self.set_state("archived")
        async with self.f.pool.acquire() as connection:
            await connection.execute("""insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
                values($1,$2,$3,'delete')""", self.f.owner, source.document_id, source.generation)
        self.assertEqual(await self.step(), "failed")
        self.assertIn(source.document_id, self.provider.documents)
        self.assertIsNotNone(await self.content(source.payload_id))

    async def test_superseded_document_cleanup_continues_without_indexing_archived_replacement(self):
        old = await self.f.publish("Old project decision")
        await self.step()
        current = await self.f.publish("Corrected project decision", expected_revision=1)
        await self.set_state("archived")
        self.provider.complete()
        self.assertEqual(await self.step(), "done")
        self.assertNotIn(old.document_id, self.provider.documents)
        # The previously submitted retain still needs terminal bookkeeping.
        # Allow only that cancellation/cleanup, not dispatch of the replacement.
        for _ in range(4):
            status = await self.step()
            if status is None:
                break
            self.assertIn(status, ("cancelled", "done"))
        else:
            self.fail("Obsolete operation bookkeeping did not settle")
        old_intent = await self.row(old.document_id)
        self.assertEqual(old_intent["status"], "cancelled")
        self.assertEqual(self.provider.sent, [str(old_intent["id"])])
        replacement = await self.row(current.document_id)
        self.assertEqual((replacement["status"], replacement["dispatch_count"]), ("pending", 0))
        self.assertIsNotNone(await self.content(old.payload_id))
        self.assertIsNotNone(await self.content(current.payload_id))

    async def test_actual_original_removal_still_cleans_provider_while_archived(self):
        source = await self.f.publish()
        await self.step()
        await self.set_state("archived")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                                     self.f.owner, source.payload_id)
        self.provider.complete()
        self.assertEqual(await self.step(), "done")
        self.assertNotIn(source.document_id, self.provider.documents)
        await self.set_state("active")
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.f.area, source.source_id)

    async def test_area_archive_gates_task_context_without_cancelling_task_or_deleting_history(self):
        request = await self.request()
        task = await self.tasks.create(self.f.owner, TaskCreate(
            request_id=uuid4(), title="Existing project task", requested_outcome="Keep the accepted objective",
            session_id=self.context.session_id, knowledge_area_id=self.f.area,
        ), model="fixture-model", source_event_id=request["id"])
        await self.set_state("archived")
        current = await self.tasks.get(self.f.owner, task.id)
        self.assertEqual(current.status, task.status)
        async with self.tasks.transaction(self.f.owner) as connection:
            self.assertEqual(await self.tasks.context_problem(connection, current), "knowledge_area_unavailable")
        await self.set_state("active")
        async with self.tasks.transaction(self.f.owner) as connection:
            self.assertIsNone(await self.tasks.context_problem(connection, current))

    async def test_runtime_can_restart_archived_for_management_but_cannot_reopen_deleted_area(self):
        await self.set_state("archived")
        client = SimpleNamespace(close=AsyncMock(), diagnostics=lambda: {})
        configuration = KnowledgeConfiguration(base_url="http://127.0.0.1:19999",
            api_key=SecretStr("fixture-only-key"), area_id=self.f.area)
        with patch("app.knowledge.runtime.HindsightClient", return_value=client):
            runtime = await KnowledgeRuntime.create(pool=self.f.pool, owner_id=self.f.owner,
                                                     configuration=configuration, tasks=self.tasks)
        try:
            module = runtime.area_tool_module_for(self.f.owner)
            registry = {tool.name: tool for tool in module.build_tools(self.context)}
            read = await registry["get_knowledge_area"].handler({"area_id": str(self.f.area)})
            self.assertEqual(read["area"]["status"], "archived")
            with self.assertRaises(SourceError):
                await runtime.sources.current_snapshot(self.f.owner, self.f.area)
        finally:
            await runtime.close()
        client.close.assert_awaited_once()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='deleted',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.f.area)
        result = await self.set_state("active", revision=3)
        self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")

    async def test_area_change_and_transcript_revision_use_the_same_lock_order(self):
        request = await self.request()
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.tasks._voice_source
        async def hold_request(*args, **kwargs):
            result = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return result
        with patch.object(self.tasks, "_voice_source", side_effect=hold_request):
            changing = asyncio.create_task(self.set_state("archived", revision=1, request=request))
            revising = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                revising = asyncio.create_task(self.request(previous=request, finality="revised"))
                with self.assertRaises(asyncio.TimeoutError):
                    await asyncio.wait_for(asyncio.shield(revising), 0.1)
                release.set()
                result = await asyncio.wait_for(changing, 5)
                await asyncio.wait_for(revising, 5)
                self.assertTrue(result["changed"])
            finally:
                release.set()
                await asyncio.gather(changing, *([revising] if revising else []), return_exceptions=True)
