"""Offline task configuration, advertisement, and unchanged admission boundaries."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.realtime.gemini.instructions import build_gemini_instructions
from app.realtime.gemini.operations import native_declarations
from app.realtime.instructions import build_realtime_instructions
from app.runtime.feature_config import Capability, TaskLimits
from app.tasks.service import initialize_task_runtime
from app.tasks.voice_tools import TASK_TOOL_NAMES
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry

import test_second_brain_voice_tools as fixtures


CONTROLS = TASK_TOOL_NAMES - {"start_background_task"}


class TaskCapabilityAdvertisementTests(unittest.IsolatedAsyncioTestCase):
    setUp = fixtures.VoiceToolTests.setUp
    invoke = fixtures.VoiceToolTests.invoke

    def rebuild(self):
        self.registry = create_tool_registry(
            modules=(self.module,), context=ToolBuildContext(None, self.owner, self.session))
        return {tool.name for tool in self.registry.list()}

    def reference(self):
        return {"task_ref": self.repository.reference(self.task.id), "expected_revision": 1}

    def test_absent_frontier_configuration_hides_only_new_task(self):
        self.service.registry = SimpleNamespace(ready=AsyncMock(), require=AsyncMock())
        for providers in ({}, {"frontier": None}, {"other": self.provider}):
            with self.subTest(providers=tuple(providers)):
                self.service.providers = providers
                self.assertFalse(self.service.provider_configured("frontier"))
                self.assertEqual(self.rebuild(), CONTROLS)
        self.service.registry.ready.assert_not_awaited()
        self.service.registry.require.assert_not_awaited()
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()
        self.provider.execute.assert_not_awaited()

    def test_worker_health_does_not_change_static_advertisement(self):
        for ready in (False, True):
            with self.subTest(worker_ready=ready):
                registry = SimpleNamespace(ready=AsyncMock(return_value=ready), require=AsyncMock())
                self.service.registry = registry
                self.assertTrue(self.service.provider_configured("frontier"))
                self.assertEqual(self.rebuild(), TASK_TOOL_NAMES)
                registry.ready.assert_not_awaited()
                registry.require.assert_not_awaited()
        self.service.registry.ready.side_effect = RuntimeError("fixture registry unavailable")
        self.assertEqual(self.rebuild(), TASK_TOOL_NAMES)
        self.service.registry.ready.assert_not_awaited()

    async def test_configured_but_warming_status_is_not_admission(self):
        self.service.registry = SimpleNamespace(ready=AsyncMock(return_value=False), require=AsyncMock())
        status = await self.service.provider_status(self.owner)
        self.assertTrue(status[0]["configured"])
        self.assertFalse(status[0]["worker_ready"])
        self.assertEqual(self.rebuild(), TASK_TOOL_NAMES)
        self.service.registry.ready.assert_awaited_once_with(self.owner, "frontier", self.provider.model)
        self.repository.create.assert_not_awaited()

    def test_existing_control_schemas_and_descriptions_are_unchanged(self):
        before = {tool.name: (tool.description, tool.input_schema, tool.mutates_state)
                  for tool in self.registry.list() if tool.name in CONTROLS}
        self.service.providers = {}
        self.rebuild()
        after = {tool.name: (tool.description, tool.input_schema, tool.mutates_state)
                 for tool in self.registry.list()}
        self.assertEqual(after, before)
        self.assertEqual(self.module.build_tools(ToolBuildContext(None, self.owner, None)), [])

    def test_both_provider_declarations_and_prompts_follow_filtered_registry(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                self.service.providers = {"frontier": self.provider} if configured else {}
                self.rebuild()
                openai = self.registry.openai_realtime_tools()
                gemini = native_declarations(self.registry)
                expected = TASK_TOOL_NAMES if configured else CONTROLS
                self.assertEqual({tool["name"] for tool in openai}, expected)
                self.assertEqual({tool["name"] for tool in gemini}, expected)
                self.assertEqual({t["name"]: t["parameters"] for t in openai},
                                 {t["name"]: t["parametersJsonSchema"] for t in gemini})
                enabled = any(tool["name"] == "start_background_task" for tool in openai)
                for prompt in (build_realtime_instructions(context=None, tasks_enabled=enabled),
                               build_gemini_instructions(tasks_enabled=enabled)):
                    self.assertEqual("## Background experts" in prompt, configured)
                    self.assertEqual("Use start_background_task" in prompt, configured)
                    self.assertEqual("Background expert delegation is not available" in prompt, not configured)

    async def test_disabled_provider_retains_reads_and_authorized_cancellation(self):
        self.service.providers = {}
        self.rebuild()
        self.repository.voice_detail.return_value = (self.task, {
            "result": None, "result_unavailable_reason": None, "last_milestone": None})
        listing = await self.invoke("list_background_tasks", {}, source=False)
        detail = await self.invoke("get_background_task", self.reference(), source=False)
        self.assertEqual(listing["tasks"][0]["task_ref"], self.reference()["task_ref"])
        self.assertEqual(detail["task"]["revision"], 1)
        self.repository.mutate.assert_not_awaited()
        cancelled = await self.invoke("cancel_background_task", self.reference())
        self.assertTrue(cancelled["accepted"])
        call = self.repository.mutate.await_args
        self.assertEqual(call.args[:3], (self.owner, self.task.id, "cancel"))
        self.assertEqual(call.kwargs["source"].event_id, self.source.id)
        self.assertNotIn("admission", call.kwargs)
        self.repository.create.assert_not_awaited()
        self.provider.execute.assert_not_awaited()

    async def test_retained_cancellation_still_requires_finalized_source(self):
        self.service.providers = {}
        self.rebuild()
        result = await self.invoke("cancel_background_task", self.reference(), source=False)
        self.assertEqual(result["error"]["code"], "task_request_not_finalized")
        self.source = self.source.model_copy(update={"availability": "deleted"})
        result = await self.invoke("cancel_background_task", self.reference())
        self.assertEqual(result["error"]["code"], "task_request_not_finalized")
        self.repository.mutate.assert_not_awaited()

    async def test_retained_resume_does_not_bypass_provider_admission(self):
        self.service.providers = {}
        self.rebuild()
        self.task.backend_kind = "frontier"

        async def mutate(owner, task_id, operation, request, **kwargs):
            self.assertEqual(operation, "resume")
            await kwargs["admission"](None, self.task)
            self.fail("An unconfigured provider must not authorize resume")

        self.repository.mutate.side_effect = mutate
        result = await self.invoke("resume_background_task", self.reference())
        self.assertEqual(result["error"]["code"], "task_provider_not_configured")
        self.assertNotIn("accepted", result)
        self.provider.execute.assert_not_awaited()

    async def test_previously_built_start_handler_remains_fail_closed(self):
        self.service.providers = {}
        result = await self.invoke("start_background_task", self.args)
        self.assertEqual(result["error"]["code"], "task_provider_not_configured")
        self.assertNotIn("accepted", result)
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_disabled_configuration_does_not_hide_prior_accepted_replay(self):
        self.service.providers = {}
        self.repository.replay_create.return_value = self.task
        result = await self.invoke("start_background_task", self.args)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["task"]["task_ref"], self.reference()["task_ref"])
        self.repository.create.assert_not_awaited()
        self.provider.execute.assert_not_awaited()
        self.assertEqual(self.rebuild(), CONTROLS)

    def test_rebuilding_after_configuration_change_does_not_change_tasks(self):
        self.assertEqual(self.rebuild(), TASK_TOOL_NAMES)
        self.service.providers = {}
        self.assertEqual(self.rebuild(), CONTROLS)
        self.service.providers = {"frontier": self.provider}
        self.assertEqual(self.rebuild(), TASK_TOOL_NAMES)
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_runtime_registers_controls_even_without_valid_frontier_configuration(self):
        connection = SimpleNamespace(fetchval=AsyncMock(return_value="offline_fixture"))

        class Pool:
            @asynccontextmanager
            async def acquire(self):
                yield connection

        for selection in (None, ValueError("fixture invalid configuration"), self.provider):
            with self.subTest(selection=type(selection).__name__):
                flags = SimpleNamespace(
                    resolve=lambda capability, **kwargs: SimpleNamespace(
                        enabled=True, configured=True, issue=None,
                        configuration=TaskLimits() if capability == Capability.TASKS else None))

                def frontier_selection():
                    if isinstance(selection, ValueError):
                        raise selection
                    return selection

                flags.frontier_selection = frontier_selection
                services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="unused-offline"),
                    db_pool=Pool(), tool_modules=(), assistant_context_providers=())
                with patch("app.tasks.service.get_second_brain_settings", return_value=flags), patch(
                    "app.tasks.service.read_schema_readiness", AsyncMock(return_value=SimpleNamespace(ready=True))):
                    await initialize_task_runtime(services)
                self.assertIsNone(services.background_task_issue)
                self.assertEqual(len(services.tool_modules), 1)
                registry = create_tool_registry(modules=services.tool_modules,
                    context=ToolBuildContext(None, self.owner, self.session))
                configured = selection is self.provider
                self.assertEqual({tool.name for tool in registry.list()},
                                 TASK_TOOL_NAMES if configured else CONTROLS)
                self.assertEqual(services.background_task_provider_issue,
                    "frontier_configuration_invalid" if isinstance(selection, ValueError) else None)
