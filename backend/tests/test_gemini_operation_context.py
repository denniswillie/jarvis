from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.gemini_live.interruption import Barrier, Trace
from evals.gemini_live.interruption_runner import INTERRUPT, SAVE, STATUS, Provider
from evals.gemini_live.operation_context import OperationContext, operation_receipt


class OperationContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session_id, self.operation_id = "fixture-owner", uuid4(), uuid4()
        self.row = {
            "id": self.operation_id, "provider_call_id": "original-call", "tool_name": "remember",
            "mutates_state": True, "arguments": '{"memory_type":"fact","content":"Amber Finch"}',
            "status": "started", "result": None, "error": None,
            "started_at": datetime(2026, 9, 11, tzinfo=UTC), "completed_at": None,
        }
        self.fixtures = SimpleNamespace(snapshot=AsyncMock(return_value={"executions": [self.row], "memories": []}))
        self.context = OperationContext(self.fixtures, self.owner, self.session_id)
        self.handler = AsyncMock(return_value={"status": "success"})
        self.registry = JarvisToolRegistry()
        self.remember = ToolDefinition(name="remember", description="Remember a fact",
                                       input_schema={"type": "object"}, handler=self.handler, mutates_state=True)
        self.registry.register(self.remember)
        self.context.register(self.registry)

    def provider(self, *, candidate=True, admitted=True):
        trace = Trace()
        provider = Provider("gemini", {}, self.registry, JarvisToolExecutor(self.registry),
                            Barrier(None, trace), trace,
                            operation_context=self.context if candidate else None)
        provider.wire = SimpleNamespace(send=AsyncMock())
        provider.calls["original-call"] = {"id": "original-call", "name": "remember",
                                            "args": {"memory_type": "fact", "content": "Amber Finch"}}
        if admitted:
            provider.lifecycle.session._handled_tool_call_ids.add("original-call")
        return provider

    async def test_receipt_preserves_real_pending_operation_identity(self):
        context = await self.context.capture(self.provider())
        receipt = context["operations"][0]
        self.assertTrue(receipt["status_ref"].startswith("op_"))
        self.assertEqual(len(receipt["status_ref"]), 15)
        self.assertNotIn(str(self.operation_id), json.dumps(receipt))
        self.assertEqual(receipt["state"], "started")
        self.assertIsNone(receipt["result"])
        self.fixtures.snapshot.assert_awaited_once_with(self.owner, self.session_id)

    async def test_completed_receipt_comes_from_committed_outcome(self):
        self.row.update(status="completed", result='{"status":"success","memory_id":"real-result"}',
                        completed_at=datetime(2026, 9, 11, 1, tzinfo=UTC))
        context = await self.context.capture(self.provider())
        self.assertEqual(context["operations"][0]["result"], {"status": "success"})
        self.assertNotIn("real-result", json.dumps(context))
        self.assertEqual(context["operations"][0]["state"], "completed")

    async def test_failed_operation_is_not_reported_as_success(self):
        self.row.update(status="failed", error='{"code":"save_failed"}')
        result = await self.context.read_status({})
        self.assertEqual(result["operations"][0]["state"], "failed")
        self.assertEqual(result["operations"][0]["error"], {"code": "save_failed"})

    async def test_unadmitted_request_is_distinguished_from_unknown_outcome(self):
        self.fixtures.snapshot.return_value = {"executions": []}
        not_started = await self.context.capture(self.provider(admitted=False))
        unknown = await self.context.capture(self.provider(admitted=True))
        self.assertEqual(not_started["operations"][0]["state"], "not_started")
        self.assertEqual(unknown["operations"][0]["state"], "unknown")
        self.assertEqual(unknown["operations"][0]["status_ref"], not_started["operations"][0]["status_ref"])

    async def test_read_status_gets_current_state_without_repeating_mutation(self):
        first = await self.context.read_status({})
        self.row["status"] = "completed"
        self.row["result"] = {"status": "success"}
        second = await self.context.read_status({})
        self.assertEqual(first["operations"][0]["state"], "started")
        self.assertEqual(second["operations"][0]["state"], "completed")
        self.handler.assert_not_awaited()

    async def test_unknown_or_other_owner_id_does_not_return_any_receipt(self):
        result = await self.context.read_status({"status_ref": "op_not_a_known_reference"})
        self.assertEqual(result["error"]["code"], "operation_not_found")
        self.assertNotIn("operations", result)
        self.fixtures.snapshot.assert_awaited_once_with(self.owner, self.session_id)

    async def test_malformed_id_and_scope_override_rejected_before_database_read(self):
        for arguments in ({"operation_id": str(self.operation_id)}, {"status_ref": ""}, {"status_ref": 12},
                          {"status_ref": None}, {"status_ref": "x" * 33},
                          {"status_ref": "op_example", "owner_id": "someone-else"}):
            result = await self.context.read_status(arguments)
            self.assertEqual(result["error"]["code"], "invalid_tool_arguments")
        self.fixtures.snapshot.assert_not_awaited()

    async def test_status_tool_is_read_only_and_remember_is_unchanged(self):
        self.assertIs(self.registry.get("remember"), self.remember)
        self.assertFalse(self.registry.get("get_operation_status").mutates_state)
        executor = JarvisToolExecutor(self.registry)
        arguments = {"content": "same content"}
        await executor.execute("remember", arguments, provider_call_id="first-request")
        await executor.execute("remember", arguments, provider_call_id="second-request")
        self.assertEqual(self.handler.await_count, 2)

    async def test_read_only_calls_are_not_presented_as_pending_mutations(self):
        provider = self.provider()
        provider.calls["status-call"] = {"id": "status-call", "name": "get_operation_status",
                                          "args": {}}
        context = await self.context.capture(provider)
        self.assertEqual(len(context["operations"]), 1)
        self.assertEqual(context["operations"][0]["tool_name"], "remember")

    async def test_receipt_and_unchanged_user_request_sent_in_one_input(self):
        provider = self.provider()
        await provider.text(SAVE, "save")
        self.assertEqual(provider.wire.send.await_args.args[0]["realtimeInput"]["text"], SAVE)
        await provider.text(INTERRUPT, "interruption", interrupt=True)
        envelope = json.loads(provider.wire.send.await_args.args[0]["realtimeInput"]["text"])
        self.assertEqual(envelope["user_message"], INTERRUPT)
        self.assertEqual(envelope["application_operation_context"]["operations"][0]["state"], "started")
        self.assertEqual(provider.wire.send.await_count, 2)
        await provider.text(STATUS, "status")
        self.assertEqual(provider.wire.send.await_args.args[0]["realtimeInput"]["text"], STATUS)
        self.fixtures.snapshot.assert_awaited_once()

    async def test_native_baseline_does_not_inject_context(self):
        provider = self.provider(candidate=False)
        await provider.text(INTERRUPT, "interruption", interrupt=True)
        self.assertEqual(provider.wire.send.await_args.args[0]["realtimeInput"]["text"], INTERRUPT)
        self.fixtures.snapshot.assert_not_awaited()

    async def test_context_read_failure_is_visible_not_a_fabricated_receipt(self):
        self.fixtures.snapshot.side_effect = RuntimeError("database unavailable")
        provider = self.provider()
        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            await provider.text(INTERRUPT, "interruption", interrupt=True)
        provider.wire.send.assert_not_awaited()

    def test_json_objects_and_database_json_strings_have_same_receipt(self):
        encoded = operation_receipt(self.row, status_ref="op_example")
        self.row["arguments"] = json.loads(self.row["arguments"])
        self.assertEqual(encoded, operation_receipt(self.row, status_ref="op_example"))

    async def test_not_started_status_lookup_succeeds_without_any_database_id(self):
        self.fixtures.snapshot.return_value = {"executions": []}
        receipt = (await self.context.capture(self.provider(admitted=False)))["operations"][0]
        for arguments in ({}, {"status_ref": receipt["status_ref"]}):
            result = await self.context.read_status(arguments)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["operations"][0]["state"], "not_started")
            self.assertEqual(result["operations"][0]["reason"], "interrupted_before_admission")
        self.handler.assert_not_awaited()

    async def test_reference_survives_fresh_context_for_same_owner_session_and_call(self):
        original = (await self.context.read_status({}))["operations"][0]
        restored = OperationContext(self.fixtures, self.owner, self.session_id)
        result = await restored.read_status({"status_ref": original["status_ref"]})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["operations"][0], original)

    async def test_reference_is_scoped_to_owner_and_session(self):
        reference = (await self.context.read_status({}))["operations"][0]["status_ref"]
        for owner, session in (("different-owner", self.session_id), (self.owner, uuid4())):
            other = OperationContext(self.fixtures, owner, session)
            result = await other.read_status({"status_ref": reference})
            self.assertEqual(result["error"]["code"], "operation_not_found")
            self.fixtures.snapshot.assert_awaited_with(owner, session)

    async def test_receipts_hide_provider_database_and_saved_memory_identifiers(self):
        memory_id = str(uuid4())
        self.row["result"] = {"status": "success", "memory": {"id": memory_id, "session_id": str(self.session_id)}}
        context = await self.context.capture(self.provider())
        serialized = json.dumps(context)
        for identifier in (str(self.operation_id), self.row["provider_call_id"], memory_id, str(self.session_id)):
            self.assertNotIn(identifier, serialized)
        self.assertEqual(context["operations"][0]["result"], {"status": "success"})

    async def test_list_and_select_multiple_operations_without_assuming_latest(self):
        second = {**self.row, "id": uuid4(), "provider_call_id": "second-call", "arguments": '{"content":"teal"}'}
        self.fixtures.snapshot.return_value["executions"].append(second)
        result = await self.context.read_status({})
        self.assertEqual(len(result["operations"]), 2)
        references = {operation["status_ref"] for operation in result["operations"]}
        self.assertEqual(len(references), 2)
        for operation in result["operations"]:
            selected = await self.context.read_status({"status_ref": operation["status_ref"]})
            self.assertEqual(selected["operations"], [operation])

    async def test_missing_previously_admitted_record_is_unknown_not_not_started(self):
        receipt = (await self.context.read_status({}))["operations"][0]
        self.fixtures.snapshot.return_value = {"executions": []}
        result = await self.context.read_status({"status_ref": receipt["status_ref"]})
        self.assertEqual(result["operations"][0]["state"], "unknown")

    async def test_lookup_failure_does_not_change_original_operation_state(self):
        failed = await self.context.read_status({"status_ref": "original-call"})
        self.assertEqual(failed["status"], "failed")
        result = await self.context.read_status({})
        self.assertEqual(result["operations"][0]["state"], "started")
        self.handler.assert_not_awaited()

    async def test_recovery_receipts_keep_unstarted_request_and_new_durable_save_distinct(self):
        self.fixtures.snapshot.return_value = {"executions": []}
        original = (await self.context.capture(self.provider(admitted=False)))["operations"][0]
        followup = {**self.row, "provider_call_id": "followup-call", "status": "completed"}
        recovery = self.context.describe_snapshot({"executions": [followup]})
        states = {receipt["status_ref"]: receipt["state"] for receipt in recovery["operations"]}
        self.assertEqual(states[original["status_ref"]], "not_started")
        self.assertEqual(sorted(states.values()), ["completed", "not_started"])

    async def test_read_only_execution_errors_do_not_appear_as_failed_saves(self):
        failed_lookup = {**self.row, "provider_call_id": "lookup-call", "tool_name": "get_operation_status",
                         "mutates_state": False, "status": "failed"}
        self.fixtures.snapshot.return_value["executions"].append(failed_lookup)
        result = await self.context.read_status({})
        self.assertEqual(len(result["operations"]), 1)
        self.assertEqual(result["operations"][0]["state"], "started")


if __name__ == "__main__":
    unittest.main()
