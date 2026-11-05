"""One selected memory authority; unused/broken integrations never select legacy."""
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.knowledge.selection import configure_selected_memory, MemoryAuthorityContextProvider
from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.tools import MemoryToolModule
from app.runtime import JarvisRuntimeServices
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import MemoryAuthority, SecondBrainSettings
from app.realtime.session_manager import RealtimeSessionManager
from app.settings import JarvisSettings
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.second_brain.source_fixture import source_fixture


class MemoryAuthorityTests(unittest.IsolatedAsyncioTestCase):
    def test_explicit_selection_is_independent_of_shadow_ingestion(self):
        flags = SecondBrainSettings(_env_file=None,enabled="true",knowledge_enabled="true")
        self.assertEqual(flags.selected_memory_authority(),MemoryAuthority.LEGACY)
        self.assertEqual(SecondBrainSettings(_env_file=None,memory_authority="hindsight").selected_memory_authority(),
                         MemoryAuthority.HINDSIGHT)
        with self.assertRaisesRegex(ValueError,"invalid_memory_authority"):
            SecondBrainSettings(_env_file=None,memory_authority="guess").selected_memory_authority()

    async def test_unavailable_selected_memory_has_no_legacy_fallback(self):
        legacy = MemoryToolModule(repository=Mock(),entity_repository=Mock())
        keep = SimpleNamespace(name="unrelated")
        services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,knowledge_runtime=None,
            knowledge_issue="knowledge_disabled",tool_modules=(legacy,keep),assistant_context_providers=())
        configure_selected_memory(services)
        memory = next(module for module in services.tool_modules if isinstance(module,MemoryToolModule))
        self.assertIsNone(memory.repository)
        self.assertIsNone(memory.service)
        self.assertIn(keep,services.tool_modules)
        tools = {t.name:t for t in memory.build_tools(ToolBuildContext(None,"fixture-owner",uuid4()))}
        self.assertEqual(set(tools),{"search_memory","remember","explain_memory"})
        result = await tools["remember"].handler({"memory_type":"fact","content":"Synthetic statement"})
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["error"]["code"],"selected_memory_unavailable")
        legacy.repository.assert_not_called()
        configure_selected_memory(services)
        self.assertEqual(sum(isinstance(m,MemoryToolModule) for m in services.tool_modules),1)
        self.assertEqual(sum(isinstance(p,MemoryAuthorityContextProvider) for p in services.assistant_context_providers),1)


    async def test_selected_delegation_owns_correction_even_without_a_ready_worker(self):
        from app.knowledge.lifecycle_tools import SourceLifecycleToolModule
        pool, area = object(), uuid4()
        runtime = SimpleNamespace(
            sources=SimpleNamespace(pool=pool),
            context_repository=SimpleNamespace(tasks=SimpleNamespace(pool=pool)),
            configuration=SimpleNamespace(area_id=area), area_selections=None,
            context_for=Mock(), area_tool_module_for=Mock(return_value=SimpleNamespace(name="knowledge_areas")),
            memory_read_backend_for=Mock(return_value=object()))
        for delegated in (False, True):
            with self.subTest(delegated=delegated):
                services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,
                    knowledge_runtime=runtime, knowledge_issue=None,
                    settings=SimpleNamespace(JARVIS_OWNER_ID="fixture-owner"),
                    background_task_service=SimpleNamespace() if delegated else None,
                    tool_modules=(), assistant_context_providers=())
                configure_selected_memory(services)
                configure_selected_memory(services)
                modules = [m for m in services.tool_modules if isinstance(m, SourceLifecycleToolModule)]
                self.assertEqual(len(modules), 1)
                names = {tool.name for tool in modules[0].build_tools(
                    ToolBuildContext(None, "fixture-owner", uuid4()))}
                self.assertEqual("correct_memory_source" in names, not delegated)
                self.assertIn("forget_memory_source", names)
                section = await MemoryAuthorityContextProvider(services).build_section(
                    SimpleNamespace(requested_at_utc=None))
                self.assertEqual("start_background_task" in str(section), delegated)
                if delegated:
                    self.assertIn("memory_edit", str(section))
                    self.assertIn("simple question", str(section))
                    self.assertIn("do not bypass", str(section))

    def test_registry_requires_explicit_mutation_receipt_capture(self):
        tool = ToolDefinition(name="fixture",description="Fixture",input_schema={"type":"object"},
            handler=AsyncMock(),mutates_state=True,result_capture=AsyncMock())
        with self.assertRaises(ValueError):
            JarvisToolRegistry().register(tool)
        registry = JarvisToolRegistry()
        registry.register(replace(tool,capture_after_commit=True))
        self.assertTrue(registry.get("fixture").mutates_state)
        for invalid in (replace(tool,capture_after_commit=True,mutates_state=False),
                        replace(tool,capture_after_commit=True,result_capture=None),
                        replace(tool,capture_after_commit=True,image_resolver=AsyncMock())):
            with self.assertRaises(ValueError):
                JarvisToolRegistry().register(invalid)

    def test_legacy_selection_preserves_the_existing_modules(self):
        modules = (SimpleNamespace(name="memory"),)
        services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY,tool_modules=modules)
        configure_selected_memory(services)
        self.assertIs(services.tool_modules,modules)


    def test_shadow_capture_keeps_legacy_as_the_only_conversational_authority(self):
        keep = SimpleNamespace(name="memory")
        source_tools = SimpleNamespace(name="source_lifecycle")
        provider = SimpleNamespace(key="knowledge_sources")
        ordinary = SimpleNamespace(key="current_time")
        services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY,
            tool_modules=(keep,source_tools),assistant_context_providers=(provider,ordinary))
        configure_selected_memory(services)
        self.assertEqual(services.tool_modules,(keep,))
        self.assertEqual(services.assistant_context_providers,(ordinary,))

    def test_populated_legacy_runtime_cannot_be_mislabeled_as_selected_hindsight(self):
        services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT,memory_service=object())
        with self.assertRaisesRegex(RuntimeError,"legacy_memory_constructed"):
            configure_selected_memory(services)

    async def test_selected_prospective_context_does_not_read_legacy_goal_projection(self):
        focus = SimpleNamespace(get_active_visual_focus=AsyncMock(return_value=None))
        reminders = SimpleNamespace(list_active_reminders=AsyncMock(return_value=[]))
        working = SimpleNamespace(get_owner_working_memory=AsyncMock(return_value=None))
        builder = ProspectiveContextBuilder(focus_reader=focus,reminder_reader=reminders,goal_reader=None,
            owner_working_memory_reader=working,max_conditional_reminders=5,max_goals=5,max_commitments=5)
        result = await builder.build(owner_id="fixture-owner")
        self.assertEqual(result.goals,())
        self.assertEqual(focus.get_active_visual_focus.await_count,1)
        self.assertEqual(reminders.list_active_reminders.await_count,1)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class SelectedMemoryStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_startup_does_not_construct_legacy_memory_readers_or_workers(self):
        import app.main as main
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),test_name=self.id()) as f:
            settings = JarvisSettings(_env_file=None,SUPABASE_DB_URL="postgresql://fixture.invalid/test",
                JARVIS_OWNER_ID=f.owner,OPENAI_API_KEY="",MEMORY_JOB_RUNNER_ENABLED=True,
                REMINDER_SCHEDULER_ENABLED=False)
            services = JarvisRuntimeServices(settings=settings,realtime_session_manager=RealtimeSessionManager(),
                sensory_frame_window=SimpleNamespace(),latest_visual_frame_buffer=SimpleNamespace(),
                domain_event_bus=DomainEventBus(),model_limiter=PriorityAsyncLimiter(2))
            flags = SecondBrainSettings(_env_file=None,memory_authority="hindsight")
            constructors = (
                "MemoryRepository","MemoryProjectionService","MemoryForgettingService","EpisodicMemoryRepository",
                "MemoryJobRepository","MemoryJobRunner","HierarchicalMemoryRetrievalService","MemoryApplicationService",
                "OpenAIStructuredOutputClient","GeminiMultimodalEmbeddingModel",
            )
            from contextlib import ExitStack
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(main,"get_second_brain_settings",return_value=flags))
                    stack.enter_context(patch.object(main,"create_pool",new=AsyncMock(return_value=f.pool)))
                    stack.enter_context(patch.object(main,"get_supabase_storage_config",
                        side_effect=JarvisHTTPError(503,"fixture_storage_disabled","Fixture has no external storage.")))
                    for name in constructors:
                        stack.enter_context(patch.object(main,name,side_effect=AssertionError("legacy construction: "+name)))
                    await main._initialize_runtime_services(services)
                self.assertEqual(services.memory_authority,MemoryAuthority.HINDSIGHT)
                for name in ("memory_service","memory_projection_service","memory_forgetting_service",
                             "memory_retrieval_service","memory_job_runner_task","memory_maintenance_scheduler_task"):
                    self.assertIsNone(getattr(services,name),name)
                self.assertIsNotNone(services.working_memory_service)
                self.assertIsNotNone(services.prospective_service)
                self.assertFalse(any(isinstance(m,MemoryToolModule) for m in services.tool_modules))
            finally:
                await services.close()
