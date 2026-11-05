"""Registered conversational quiet tools, executor provenance and post-commit hooks."""
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.quiet import OwnerQuietRepository
from app.attention.service import AttentionService, initialize_attention_runtime
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.runtime.feature_config import Capability, SecondBrainSettings
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import ToolInvocationContext
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from test_attention_quiet import LedgerFixture
from test_attention_quiet_propagation import ControlledDelivery


def attention_only_flags():
    configuration = SecondBrainSettings(_env_file=None, enabled="true", attention_enabled="true").resolve(
        Capability.ATTENTION, database_url="fixture").configuration

    def resolve(capability, **_):
        enabled = capability == Capability.ATTENTION
        return SimpleNamespace(enabled=enabled, configured=enabled, issue=None,
                               configuration=configuration if enabled else None)

    return SimpleNamespace(resolve=resolve)


class QuietToolEntryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = LedgerFixture()
        self.owner, self.session = "quiet-tool-entry-fixture", uuid4()
        self.quiet = OwnerQuietRepository(self.db)
        self.record = SimpleNamespace(id=uuid4(), revision=1,
            source_namespace="fixture", source_event_id="finalized-request")
        self.db.authorize(self.owner, self.session, ToolInvocationContext(
            request_event_id=self.record.id, request_revision=self.record.revision))

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.owner)
            async with self.db.transaction() as connection:
                yield connection

        self.service = AttentionService(SimpleNamespace(tasks=SimpleNamespace(transaction=transaction)))
        self.module = PreferenceToolModule(PreferenceRepository(self.db))
        # Production builds modules before attention readiness. A prebuilt
        # registry must observe the later runtime binding as well.
        self.registry = create_tool_registry(modules=(self.module,),
            context=ToolBuildContext(settings=None, owner_id=self.owner, session_id=self.session))
        self.capture = SimpleNamespace(request_for_call=AsyncMock(return_value=self.record), tool_outcome=Mock())
        self.executor = JarvisToolExecutor(self.registry, owner_id=self.owner, session_id=self.session,
                                           context_capture=self.capture)
        self.module.bind_attention_service(self.service)

    async def asyncTearDown(self):
        self.service._deliveries.clear()
        await self.service.close()

    async def execute(self, action="activate", revision=0, call="fixture-call"):
        return await self.executor.execute("set_attention_quiet",
            {"action": action, "expected_revision": revision}, provider_call_id=call)

    async def test_registered_schema_and_real_executor_apply_after_commit(self):
        schema = next(tool for tool in self.registry.openai_realtime_tools()
                      if tool["name"] == "set_attention_quiet")
        self.assertFalse(schema["parameters"]["additionalProperties"])
        self.assertNotIn("owner_id", schema["parameters"]["properties"])
        delivery = ControlledDelivery(self.owner)

        async def stop():
            self.assertFalse(self.db.lock.locked())
            self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)

        delivery.stop_hook = stop
        self.service._deliveries.add(delivery)
        result = await self.execute()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["delivery_propagation"]["state"], "checked")
        self.assertEqual(result["delivery_propagation"]["stop_requests"], 1)
        self.assertFalse(result["delivery_propagation"]["playback_stop_confirmed"])
        self.assertEqual(delivery.reasons, ["owner_quiet_quiet"])
        self.capture.request_for_call.assert_awaited_once_with("fixture-call")

    async def test_new_provider_ids_for_same_request_do_not_duplicate_episode_or_stop(self):
        delivery = ControlledDelivery(self.owner)
        self.service._deliveries.add(delivery)
        first = await self.execute()
        for call in ("retry-2", "retry-3"):
            replay = await self.execute(call=call)
            self.assertTrue(replay["request_replayed"])
            self.assertEqual(replay["quiet"]["episode"], first["quiet"]["episode"])
            self.assertEqual(replay["delivery_propagation"]["stop_requests"], 0)
        self.assertEqual((await self.quiet.get(self.owner)).revision, 1)
        self.assertEqual(delivery.reasons, ["owner_quiet_quiet"])

    async def test_read_for_direct_question_never_resumes_or_invokes_fence(self):
        await self.execute()
        with patch.object(self.service, "enforce_owner_quiet", new_callable=AsyncMock) as enforce:
            result = await self.executor.execute("get_attention_quiet", {}, provider_call_id="question")
        self.assertEqual(result["quiet"]["effective_state"], "quiet")
        self.assertEqual(result["quiet"]["revision"], 1)
        enforce.assert_not_awaited()

    async def test_missing_authority_and_extra_identity_fields_cannot_commit_or_fence(self):
        self.capture.request_for_call.return_value = None
        with patch.object(self.service, "enforce_owner_quiet", new_callable=AsyncMock) as enforce:
            result = await self.execute()
            self.assertEqual(result["error"]["code"], "request_evidence_required")
            invalid = await self.executor.execute("set_attention_quiet",
                {"action": "activate", "expected_revision": 0, "owner_id": self.owner})
            self.assertEqual(invalid["error"]["code"], "invalid_tool_input")
        enforce.assert_not_awaited()
        self.assertEqual((await self.quiet.get(self.owner)).revision, 0)

    async def test_post_commit_failure_is_not_reported_as_a_failed_preference_write(self):
        with patch.object(self.service, "enforce_owner_quiet", new_callable=AsyncMock,
                          side_effect=RuntimeError("private transport detail")):
            result = await self.execute()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["accepted_revision"], 1)
        self.assertEqual(result["delivery_propagation"]["state"], "unconfirmed")
        self.assertNotIn("private transport detail", str(result))
        self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)

    async def test_unbound_or_closed_service_is_explicitly_unconfirmed(self):
        self.module.bind_attention_service(None)
        first = await self.execute()
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["delivery_propagation"]["state"], "unconfirmed")
        self.module.bind_attention_service(self.service)
        await self.service.close()
        replay = await self.execute(call="closed-retry")
        self.assertTrue(replay["request_replayed"])
        self.assertEqual(replay["delivery_propagation"]["state"], "unconfirmed")

    async def test_runtime_readiness_binds_existing_module_and_unready_unbinds(self):
        connection = SimpleNamespace(fetchval=AsyncMock(return_value="fixture_schema"))

        @asynccontextmanager
        async def acquire():
            yield connection

        pool = SimpleNamespace(acquire=acquire)
        module = PreferenceToolModule(PreferenceRepository(pool))
        runtime = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="fixture", OPENAI_API_KEY="",
            JARVIS_OWNER_ID=self.owner), db_pool=pool, tool_modules=(module,),
            background_task_service=SimpleNamespace(repository=SimpleNamespace(pool=pool)),
            realtime_session_manager=SimpleNamespace(), attention_service=None)
        with patch("app.attention.service.get_second_brain_settings", return_value=attention_only_flags()), \
             patch("app.attention.service.read_schema_readiness", new_callable=AsyncMock,
                   return_value=SimpleNamespace(ready=True)) as readiness, \
             patch.object(AttentionService, "start_maintenance") as start, \
             patch("app.reminders.duty_runtime.initialize_responsibilities", new_callable=AsyncMock,
                   return_value=None):
            await initialize_attention_runtime(runtime)
            self.assertIsNone(runtime.attention_issue)
            self.assertIs(module._attention_service, runtime.attention_service)
            start.assert_called_once_with(owner=self.owner)
            await runtime.attention_service.close()
            readiness.return_value = SimpleNamespace(ready=False)
            await initialize_attention_runtime(runtime)
            self.assertEqual(runtime.attention_issue, "attention_schema_not_ready")
            self.assertIsNone(module._attention_service)
