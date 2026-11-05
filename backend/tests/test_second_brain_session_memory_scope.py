"""Session-scoped memory consumers and durable cross-switch receipt identity."""
from datetime import UTC, datetime
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.conversation_writes import ConversationMemoryWriteBackend, NAMESPACE
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.tools import MemoryToolModule, create_memory_tools
from app.runtime.feature_config import TaskLimits
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


def recalled(sources=()):
    return {"status": "available", "sources": list(sources), "rejected_candidates": 0,
            "filtered_candidates": 0, "unscanned_candidates": 0, "omitted_sources": 0,
            "omitted_content": 0, "vendor_usage": None, "unscanned_documents": 0,
            "missing_supporting_facts": 0, "omitted_supporting_facts": 0,
            "source_facts_truncated": False, "elapsed_ms": 0}


class SessionMemoryContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.session = "scope-fixture-owner", uuid4(), uuid4()
        self.project, self.other_session = uuid4(), uuid4()
        self.sources = SimpleNamespace(pool=object(), read=AsyncMock())
        self.retrieval = SimpleNamespace(sources=self.sources, recall=AsyncMock(return_value=recalled()))
        self.selection = {"area_id": str(self.project), "area_status": "active",
                          "selection_revision": 1, "selection_event_id": str(uuid4()),
                          "origin": "session_selection"}

        async def selected(owner, session):
            if session == self.session:
                return dict(self.selection)
            return self.selection | {"area_id": str(self.area), "selection_revision": 0,
                                     "selection_event_id": None, "origin": "configured_default"}

        self.selections = SimpleNamespace(sources=self.sources, default_area_id=self.area,
                                         read=AsyncMock(side_effect=selected))
        self.backend = SourceMemoryReadBackend(self.sources, self.retrieval,
            owner_id=self.owner, area_id=self.area, selections=self.selections)

    def tools(self, session):
        return {t.name: t for t in MemoryToolModule(read_backend=self.backend).build_tools(
            ToolBuildContext(None, self.owner, session))}

    async def test_session_binding_is_independent_of_search_session_filter(self):
        first, other = self.tools(self.session), self.tools(self.other_session)
        result = await first["search_memory"].handler({"query": "project context"})
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        self.assertIsNone(self.retrieval.recall.call_args.kwargs["scope"].session_id)
        self.assertEqual(self.retrieval.recall.call_args.args[1], self.project)
        await other["search_memory"].handler({"query": "other conversation"})
        self.assertEqual(self.retrieval.recall.call_args.args[1], self.area)
        await first["search_memory"].handler({"query": "this session", "current_session_only": True})
        self.assertEqual(self.retrieval.recall.call_args.kwargs["scope"].session_id, self.session)
        self.assertEqual(self.backend.area_id, self.area)
        self.assertIsNone(self.backend.session_id)

    async def test_existing_tool_catalog_resolves_new_selection_per_operation(self):
        tools = self.tools(self.session)
        first = await tools["search_memory"].handler({"query": "first"})
        later_area = uuid4()
        self.selection.update(area_id=str(later_area), selection_revision=2)
        second = await tools["search_memory"].handler({"query": "second"})
        self.assertEqual(first["knowledge_scope"]["area_id"], str(self.project))
        self.assertEqual(second["knowledge_scope"]["area_id"], str(later_area))

    async def test_inflight_search_keeps_the_scope_used_for_retrieval(self):
        async def during_recall(*args, **kwargs):
            self.selection.update(area_id=str(self.area), selection_revision=2)
            return recalled()
        self.retrieval.recall.side_effect = during_recall
        result = await self.tools(self.session)["search_memory"].handler({"query": "in flight"})
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        self.assertEqual(result["knowledge_scope"]["selection_revision"], 1)

    async def test_explanation_reads_only_the_selected_area(self):
        now = datetime.now(UTC)
        source = SourceVersion(owner_id=self.owner, area_id=self.project, source_id=uuid4(),
            revision=1, generation=1, payload_id=uuid4(), document_id="fixture-source",
            bank_name="fixture-bank", content_digest="a" * 64, source_type="user_transcript",
            trust="user_input", source_time=now, time_basis="synthetic fixture",
            recorded_at=now, content={"text": "Original project fact."},
            is_current=True, ingestion_status="pending")
        self.sources.read.return_value = source
        result = await self.tools(self.session)["explain_memory"].handler({"memory_id": str(source.source_id)})
        self.sources.read.assert_awaited_once_with(self.owner, self.project, source.source_id, revision=None)
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        self.assertIn("Original project fact.", result["content_page"]["text"])

    async def test_unavailable_selected_area_never_falls_back_or_contacts_provider(self):
        for status in ("archived", "deleted"):
            self.selection["area_status"] = status
            result = await self.tools(self.session)["search_memory"].handler({"query": "missing scope"})
            self.assertEqual(result["error"]["code"], "knowledge_area_unavailable")
        self.retrieval.recall.assert_not_awaited()

    async def test_model_cannot_supply_a_session_or_area_override(self):
        tools = create_memory_tools(owner_id=self.owner, session_id=self.session, read_backend=self.backend)
        search = next(t for t in tools if t.name == "search_memory")
        for extra in ({"session_id": str(self.other_session)}, {"area_id": str(self.area)},
                      {"knowledge_scope": self.selection}):
            result = await search.handler({"query": "fixture", **extra})
            self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.selections.read.assert_not_awaited()

    async def test_result_capture_uses_returned_scope_without_resolving_current_selection(self):
        backend = self.backend.for_session(self.session)
        context = SimpleNamespace(capture=SimpleNamespace(owner_id=self.owner, session_id=self.session,
            repository=SimpleNamespace(pool=self.sources.pool)), tool_name="search_memory", call_id="scope-search")
        result = {"results": [], "knowledge_scope": dict(self.selection)}
        self.selection["area_id"] = str(self.area)
        with patch("app.knowledge.read_provenance.SourceReadProvenance.bind", new_callable=AsyncMock) as bind:
            await backend.capture_tool_result(result, context)
            self.assertEqual(bind.call_args.args[:3], (self.owner, self.project, []))
        self.selections.read.assert_not_awaited()

    async def test_save_receipt_capture_preserves_original_saved_area(self):
        backend = self.backend.for_session(self.session)
        context = SimpleNamespace(capture=SimpleNamespace(owner_id=self.owner, session_id=self.session,
            repository=SimpleNamespace(pool=self.sources.pool)), tool_name="remember", call_id="scope-save")
        result = {"source": {"area_id": str(self.area)}}
        with patch("app.knowledge.read_provenance.SourceReadProvenance.bind", new_callable=AsyncMock) as bind:
            await backend.capture_tool_result(result, context)
            self.assertEqual(bind.call_args.args[1], self.area)
        self.selections.read.assert_not_awaited()

    async def test_cross_session_or_mixed_scope_receipts_are_rejected(self):
        backend = self.backend.for_session(self.session)
        capture = SimpleNamespace(owner_id=self.owner, session_id=self.other_session,
                                  repository=SimpleNamespace(pool=self.sources.pool))
        context = SimpleNamespace(capture=capture, tool_name="search_memory", call_id="bad-scope")
        result = {"results": [], "knowledge_scope": dict(self.selection)}
        with self.assertRaises(SourceError):
            await backend.capture_tool_result(result, context)
        capture.session_id = self.session
        result["results"] = [{"area_id": str(self.area)}]
        with self.assertRaises(SourceError) as caught:
            await backend.capture_tool_result(result, context)
        self.assertEqual(caught.exception.code, "knowledge_result_scope_mismatch")

    def test_reader_composition_bounds_and_fixed_worker_scope(self):
        for changes in ({"sources": SimpleNamespace(pool=object())}, {"default_area_id": uuid4()}):
            wrong = SimpleNamespace(**(vars(self.selections) | changes))
            with self.assertRaises(ValueError):
                SourceMemoryReadBackend(self.sources, self.retrieval,
                    owner_id=self.owner, area_id=self.area, selections=wrong)
        with self.assertRaises(ValueError):
            self.backend.for_session("not-a-session-uuid")
        fixed = SourceMemoryReadBackend(self.sources, self.retrieval, owner_id=self.owner, area_id=self.area)
        self.assertIs(fixed.for_session(self.session), fixed)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class SessionMemoryDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.selections = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area, self.selections))
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Bench','robotics') returning id", self.f.owner)
            self.fitness = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Training','fitness') returning id", self.f.owner)
        self.capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="scoped-memory-fixture", write_timeout=10)
        self.retrieval = SimpleNamespace(sources=self.f.sources, recall=AsyncMock(return_value=recalled()))
        self.reader = SourceMemoryReadBackend(self.f.sources, self.retrieval, owner_id=self.f.owner,
            area_id=self.f.area, selections=self.selections).for_session(self.session)
        self.writer = ConversationMemoryWriteBackend(self.f.sources, self.tasks,
            owner_id=self.f.owner, area_id=self.f.area, selections=self.selections)

    async def request(self, text):
        return await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:scoped-memory-test", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": text}))

    async def select(self, area):
        request = await self.request("Select a synthetic area.")
        selected = await self.selections.read(self.f.owner, self.session)
        await self.selections.select(self.f.owner, area,
            request=TaskVoiceSource(request.id, request.revision, self.session),
            expected_revision=selected["selection_revision"])

    async def mapping(self, event_id):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select s.id,s.area_id from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2""", self.f.owner, event_id)

    async def save(self, request, call_id, *, writer=None, content="Finish the synthetic robot calibration."):
        with bind_tool_invocation(provider_call_id=call_id, request_event_id=request.id, request_revision=request.revision):
            return await (writer or self.writer).remember(owner_id=self.f.owner, session_id=self.session,
                parameters={"memory_type": "goal", "content": content})

    async def test_delayed_search_and_empty_receipts_keep_scope_and_source_lifecycle(self):
        await self.select(self.project)
        event = await self.request("A synthetic robotics source.")
        mapping = await self.mapping(event.id)
        source = await self.f.sources.read(self.f.owner, self.project, mapping["id"])

        async def change_scope(*args, **kwargs):
            await self.select(self.fitness)
            return recalled([source.original_evidence()])
        self.retrieval.recall.side_effect = change_scope
        result = await self.reader.search(owner_id=self.f.owner, request=MemoryRetrievalQuery(query="robotics"))
        captured = await self.reader.capture_tool_result(result, SimpleNamespace(
            capture=self.capture, call_id="delayed-search", tool_name="search_memory"))
        self.assertEqual((await self.mapping(captured.event_id))["area_id"], self.project)
        self.assertEqual(result["knowledge_scope"]["area_id"], str(self.project))
        with self.assertRaises(SourceError):
            await self.reader.explain(owner_id=self.f.owner, memory_id=source.source_id)

        await self.select(self.project)
        self.retrieval.recall.side_effect = None
        self.retrieval.recall.return_value = recalled()
        empty = await self.reader.search(owner_id=self.f.owner, request=MemoryRetrievalQuery(query="not present"))
        await self.select(self.fitness)
        empty_capture = await self.reader.capture_tool_result(empty, SimpleNamespace(
            capture=self.capture, call_id="empty-search", tool_name="search_memory"))
        self.assertEqual((await self.mapping(empty_capture.event_id))["area_id"], self.project)
        await self.f.sources.delete_source(self.f.owner, self.project, source.source_id, expected_revision=1)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.f.owner, captured.payload_id), "deleted")
        with self.assertRaises(SourceError):
            await self.reader.capture_tool_result(result, SimpleNamespace(
                capture=self.capture, call_id="deleted-read", tool_name="search_memory"))

    async def test_cross_area_request_save_and_retry_do_not_move_or_duplicate_memory(self):
        request = await self.request("Remember the synthetic calibration goal.")
        await self.select(self.project)
        first, second = await asyncio.gather(self.save(request, "first"), self.save(request, "second"))
        self.assertEqual(first["memory_id"], second["memory_id"])
        self.assertEqual({first["replayed"], second["replayed"]}, {False, True})
        self.assertEqual(first["source"]["area_id"], str(self.project))
        await self.select(self.fitness)
        replay = await self.save(request, "retry-after-switch")
        self.assertEqual(replay["memory_id"], first["memory_id"])
        captured = await self.reader.capture_tool_result(replay, SimpleNamespace(
            capture=self.capture, call_id="saved-replay", tool_name="remember"))
        self.assertEqual((await self.mapping(captured.event_id))["area_id"], self.project)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_events where owner_id=$1 and source_namespace=$2",
                self.f.owner, NAMESPACE), 1)
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.fitness)
        self.assertEqual((await self.save(request, "retry-while-current-archived"))["memory_id"], first["memory_id"])
        unavailable_request = await self.request("A request captured without semantic publication.")
        with self.assertRaises(SourceError):
            await self.save(unavailable_request, "unavailable")
        fixed = ConversationMemoryWriteBackend(self.f.sources, self.tasks, owner_id=self.f.owner, area_id=self.f.area)
        fixed_result = await self.save(request, "fixed", writer=fixed, content="A distinct fixed-scope statement.")
        self.assertEqual(fixed_result["source"]["area_id"], str(self.f.area))
        original = await self.mapping(request.id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, original["id"], expected_revision=1)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.project, UUID(first["memory_id"]))

    async def test_publication_scope_is_part_of_receipt_identity_and_cannot_rebind_a_stream(self):
        event = ContextIngress(session_id=self.session, source_namespace="voice:scoped-receipt",
            source_event_id="scoped-result", receipt_id="first", kind="tool_outcome",
            trust="tool_receipt", payload={"result": "Synthetic project result."})
        async with self.history.publication.transaction(self.tasks, self.f.owner) as connection:
            original = await self.history.append_in_transaction(connection, self.f.owner, event,
                                                                  publication_area_id=self.project)
        self.assertEqual((await self.mapping(original.id))["area_id"], self.project)
        with self.assertRaises(TaskError) as caught:
            async with self.history.publication.transaction(self.tasks, self.f.owner) as connection:
                await self.history.append_in_transaction(connection, self.f.owner, event, publication_area_id=self.fitness)
        self.assertEqual(caught.exception.code, "context_receipt_conflict")
        revised = event.model_copy(update={"receipt_id": "second", "expected_revision": 1})
        with self.assertRaises(SourceError) as caught:
            async with self.history.publication.transaction(self.tasks, self.f.owner) as connection:
                await self.history.append_in_transaction(connection, self.f.owner, revised, publication_area_id=self.fitness)
        self.assertEqual(caught.exception.code, "knowledge_context_scope_conflict")
        self.assertTrue((await self.history.get(self.f.owner, original.id)).is_current)
        fixed = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        with self.assertRaises(SourceError):
            async with fixed.publication.transaction(self.tasks, self.f.owner) as connection:
                await fixed.append_in_transaction(connection, self.f.owner,
                    event.model_copy(update={"source_event_id": "fixed-conflict"}), publication_area_id=self.project)
        with self.assertRaises(TaskError):
            async with self.tasks.transaction(self.f.owner) as connection:
                await ContextRepository(self.tasks).append_in_transaction(
                    connection, self.f.owner, event, publication_area_id=self.project)

    async def test_version_receipt_keeps_selected_scope_and_historical_removal_lineage(self):
        await self.select(self.project)
        old = await self.f.publish("Original synthetic setting", area_id=self.project)
        await self.f.publish("Corrected synthetic setting", area_id=self.project, expected_revision=1)
        result = await self.reader.versions(owner_id=self.f.owner, memory_id=old.source_id)
        self.assertEqual([row["revision"] for row in result["versions"]], [2, 1])
        await self.select(self.fitness)
        captured = await self.reader.capture_tool_result(result, SimpleNamespace(
            capture=self.capture, call_id="version-history", tool_name="list_memory_versions"))
        self.assertEqual((await self.mapping(captured.event_id))["area_id"], self.project)
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where id=$1", old.payload_id)
            self.assertEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where id=$1", captured.payload_id), "deleted")
        with self.assertRaises(SourceError):
            await self.reader.capture_tool_result(result, SimpleNamespace(
                capture=self.capture, call_id="version-history-replay", tool_name="list_memory_versions"))
