"""Selected memory exposes area management without selecting a new source scope."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from pydantic import SecretStr

from app.knowledge.area_tools import KnowledgeAreaToolModule
from app.knowledge.areas import KnowledgeAreaService
from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.memory.tools import MemoryToolModule
from app.runtime.feature_config import KnowledgeConfiguration, MemoryAuthority, TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


class AreaRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area = str(uuid4()), uuid4()
        pool = object()
        sources, requests = SimpleNamespace(pool=pool), SimpleNamespace(pool=pool)
        self.area_service = KnowledgeAreaService(sources, requests)
        self.area_service.list = AsyncMock(return_value={"areas": [], "next_after_id": None})
        self.module = KnowledgeAreaToolModule(self.area_service, self.owner)
        runtime = SimpleNamespace(
            sources=sources, context_repository=SimpleNamespace(tasks=requests),
            configuration=SimpleNamespace(area_id=self.area), context_for=Mock(),
            area_selections=SessionKnowledgeAreas(sources, requests, self.area),
            memory_read_backend_for=Mock(return_value=Mock()),
            area_tool_module_for=Mock(return_value=self.module),
        )
        self.services = SimpleNamespace(
            memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=runtime,
            knowledge_issue=None, settings=SimpleNamespace(JARVIS_OWNER_ID=self.owner),
            tool_modules=(), assistant_context_providers=(),
        )

    async def test_selected_conversation_exposes_owned_area_management_once(self):
        configure_selected_memory(self.services)
        configure_selected_memory(self.services)
        modules = [m for m in self.services.tool_modules if isinstance(m, KnowledgeAreaToolModule)]
        self.assertEqual(modules, [self.module])
        tools = {tool.name: tool for tool in self.module.build_tools(
            ToolBuildContext(None, self.owner, uuid4()))}
        self.assertEqual(set(tools), {"list_knowledge_areas", "get_knowledge_area",
            "set_knowledge_area_state", "create_knowledge_areas", "update_knowledge_area",
            "get_knowledge_area_history", "list_knowledge_area_sources"})
        result = await tools["list_knowledge_areas"].handler({})
        self.assertNotEqual(result.get("status"), "failed")
        self.assertEqual(self.area_service.list.await_args.args, (self.owner,))
        self.assertEqual(self.services.knowledge_runtime.configuration.area_id, self.area)
        self.services.knowledge_runtime.area_tool_module_for.assert_called_with(self.owner)
        memory = next(m for m in self.services.tool_modules if isinstance(m, MemoryToolModule))
        self.assertIs(memory.write_backend.selections, self.services.knowledge_runtime.area_selections)

    def test_sessionless_catalog_does_not_gain_mutations(self):
        configure_selected_memory(self.services)
        tools = self.module.build_tools(ToolBuildContext(None, self.owner, None))
        self.assertEqual({tool.name for tool in tools}, {"list_knowledge_areas", "get_knowledge_area",
                                                       "get_knowledge_area_history", "list_knowledge_area_sources"})

    def test_legacy_and_unavailable_selection_remove_stale_management_modules(self):
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                self.services.memory_authority = MemoryAuthority.HINDSIGHT
                configure_selected_memory(self.services)
                self.assertIn(self.module, self.services.tool_modules)
                if unavailable:
                    self.services.knowledge_runtime = None
                else:
                    self.services.memory_authority = MemoryAuthority.LEGACY
                configure_selected_memory(self.services)
                self.assertFalse(any(isinstance(m, (KnowledgeAreaToolModule, SourceLifecycleToolModule))
                    for m in self.services.tool_modules))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class AreaRoutingDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_runtime_catalog_creates_and_recovers_an_owned_area(self):
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()) as f:
            tasks = TaskRepository(f.pool, TaskLimits())
            async with f.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
            client = SimpleNamespace(close=AsyncMock())
            configuration = KnowledgeConfiguration(base_url="http://127.0.0.1:19999",
                api_key=SecretStr("fixture-only-key"), area_id=f.area)
            with patch("app.knowledge.runtime.HindsightClient", return_value=client):
                runtime = await KnowledgeRuntime.create(pool=f.pool, owner_id=f.owner,
                    configuration=configuration, tasks=tasks)
            try:
                services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,
                    knowledge_runtime=runtime, knowledge_issue=None,
                    settings=SimpleNamespace(JARVIS_OWNER_ID=f.owner),
                    tool_modules=(), assistant_context_providers=())
                configure_selected_memory(services)
                module = next(m for m in services.tool_modules if isinstance(m, KnowledgeAreaToolModule))
                tools = {tool.name: tool for tool in module.build_tools(ToolBuildContext(None, f.owner, session))}
                request = await ContextRepository(tasks).append(f.owner, ContextIngress(
                    session_id=session, source_namespace="voice:area-routing-fixture",
                    source_event_id="create-bench", receipt_id="request-1", expected_revision=0,
                    kind="user_request", finality="final", trust="user_input",
                    payload={"text": "Create a knowledge area called Bench for my robotics work."}))
                arguments = {"areas": [{"name": "Bench", "domain": "robotics",
                    "description": "Synthetic robot bench notes."}]}
                results = []
                for _ in range(2):
                    with bind_tool_invocation(provider_call_id=str(uuid4()),
                        request_event_id=request.id, request_revision=request.revision):
                        result = await tools["create_knowledge_areas"].handler(arguments)
                    self.assertNotEqual(result.get("status"), "failed", result)
                    results.append(result)
                self.assertFalse(results[0]["replayed"])
                self.assertTrue(results[1]["replayed"])
                area = results[0]["areas"][0]
                self.assertEqual(results[1]["areas"][0]["id"], area["id"])
                read = await tools["get_knowledge_area"].handler({"area_id": area["id"]})
                self.assertEqual(read["area"]["description"], arguments["areas"][0]["description"])
                self.assertEqual(runtime.configuration.area_id, f.area)
                async with f.pool.acquire() as connection:
                    self.assertEqual(await connection.fetchval(
                        "select count(*) from jarvis_knowledge_areas where owner_id=$1 and name='Bench'",
                        f.owner), 1)
            finally:
                await runtime.close()
            client.close.assert_awaited_once()
