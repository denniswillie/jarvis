"""Local snapshot reads must not touch persistence or request-source lookup."""
import asyncio
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import UUID

from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import bind_tool_invocation, current_tool_invocation
from app.tools.registry import JarvisToolRegistry, ToolDefinition


SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class ForbiddenExternalAccess:
    def __getattr__(self, name):
        raise AssertionError(f"Local snapshot touched external dependency: {name}")


def definition(handler, **changes):
    return ToolDefinition(name="live_briefing", description="Read the local briefing.",
                          input_schema=SCHEMA, handler=handler, **changes)


class LocalSnapshotRegistrationTests(unittest.TestCase):
    def test_external_materialization_or_mutation_cannot_opt_in(self):
        for changes in ({"mutates_state": True}, {"result_capture": AsyncMock()},
                        {"image_resolver": object()}, {"capture_after_commit": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                JarvisToolRegistry().register(definition(
                    AsyncMock(), local_snapshot_read=True, **changes))

    def test_policy_requires_actual_boolean(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JarvisToolRegistry().register(definition(AsyncMock(), local_snapshot_read=value))

    def test_execution_policy_is_not_a_model_argument(self):
        registry = JarvisToolRegistry()
        registry.register(definition(AsyncMock(), local_snapshot_read=True))
        exposed = registry.openai_realtime_tools()[0]
        self.assertEqual(exposed["parameters"], SCHEMA)
        self.assertNotIn("local_snapshot_read", exposed)


class LocalSnapshotExecutionTests(unittest.IsolatedAsyncioTestCase):
    def executor(self, handler, *, local=True, **options):
        registry = JarvisToolRegistry()
        registry.register(definition(handler, local_snapshot_read=local))
        return JarvisToolExecutor(registry, **options)

    async def test_snapshot_does_not_touch_any_optional_external_service(self):
        expected = {"status": "success", "outcome": "available",
                    "effect_committed": False, "summary": "A local observation."}
        handler = AsyncMock(return_value=expected)
        forbidden = ForbiddenExternalAccess()
        observer = Mock(side_effect=AssertionError("Request evidence observer was called"))
        executor = self.executor(handler, execution_repository=forbidden,
            experience_event_service=forbidden, context_capture=forbidden,
            owner_id="opaque-owner", session_id=UUID(int=1),
            require_source_tracking=True, on_request_evidence=observer)
        result = await asyncio.wait_for(executor.execute(
            "live_briefing", {}, provider_call_id="call-local"), timeout=1)
        self.assertEqual(result, expected)
        handler.assert_awaited_once_with({})
        observer.assert_not_called()

    async def test_local_call_preserves_wire_id_without_inheriting_request_authority(self):
        async def handler(arguments):
            invocation = current_tool_invocation()
            self.assertEqual(invocation.provider_call_id, "call-local")
            self.assertIsNone(invocation.request_event_id)
            self.assertIsNone(invocation.request_revision)
            self.assertIsNone(invocation.request_identity)
            self.assertIsNone(invocation.task)
            return {"status": "success"}

        executor = self.executor(handler)
        with bind_tool_invocation(provider_call_id="outer", request_event_id=UUID(int=2), request_revision=3):
            await executor.execute("live_briefing", {}, provider_call_id="call-local")
            self.assertEqual(current_tool_invocation().provider_call_id, "outer")
            self.assertEqual(current_tool_invocation().request_revision, 3)

    async def test_local_read_still_counts_against_tool_budget(self):
        handler = AsyncMock(return_value={"status": "success"})
        budget = Mock()
        await self.executor(handler).execute("live_briefing", {}, budget=budget)
        budget.record.assert_called_once_with(tool_name="live_briefing", arguments={})

    async def test_local_failure_reports_no_effect_without_audit_dependency(self):
        executor = self.executor(AsyncMock(side_effect=ValueError("Invalid cursor")),
                                 context_capture=ForbiddenExternalAccess())
        with self.assertLogs("app.tools.executor", level="ERROR"):
            result = await executor.execute("live_briefing", {}, provider_call_id="call-local")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["outcome"], "failed")
        self.assertIs(result["effect_committed"], False)

    async def test_default_reader_keeps_existing_request_capture(self):
        capture = Mock()
        capture.request_for_call = AsyncMock(return_value=None)
        observer = Mock()
        executor = self.executor(AsyncMock(return_value={"status": "success"}), local=False,
                                 context_capture=capture, on_request_evidence=observer)
        await executor.execute("live_briefing", {}, provider_call_id="call-legacy")
        capture.request_for_call.assert_awaited_once_with("call-legacy")
        capture.tool_outcome.assert_called_once()
        observer.assert_called_once_with("call-legacy", None)

    async def test_default_reader_keeps_existing_durable_dispatch(self):
        handler = AsyncMock(return_value={"status": "success"})
        executor = self.executor(handler, local=False, execution_repository=object(),
                                 owner_id="opaque-owner", session_id=UUID(int=1))
        executor._execute_durable = AsyncMock(return_value={"status": "success"})
        await executor.execute("live_briefing", {}, provider_call_id="call-legacy")
        executor._execute_durable.assert_awaited_once()
        handler.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
