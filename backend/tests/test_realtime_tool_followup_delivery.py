"""Real executor/continuation/source writer, with offline SQL and socket fixtures.

These tests qualify source-bound response construction, not provider obedience,
receipt ACKs, spoken correctness, indexing, or hearing.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.assistant.base_prompt import build_base_prompt
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.instructions import build_tool_followup_instructions, TOOL_FOLLOWUP_MAX_BYTES
from app.realtime.models import (
    RealtimeSessionContext, RealtimeToolFollowup, RealtimeToolReceipt, RealtimeToolRequest,
)
from app.realtime.source_context import LiveContextError
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tools.executor import JarvisToolExecutor
from app.tools.provenance import SourceBoundToolResult
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from app.voice_routes.realtime_adapter import RealtimeVoiceRoute
import test_realtime_response_admission_lane as admission_fixtures
import test_realtime_shared_view as provider_fixtures


class _Sources(admission_fixtures._SourcePool):
    def __init__(self):
        super().__init__()
        self.withdrawn = set()

    async def fetchval(self, sql, *args):
        result = await super().fetchval(sql, *args)
        if "lock_jarvis_copy_sources" in sql:
            return result and not self.withdrawn.intersection(args[-1])
        return result


def _record(capture, *, kind="user_transcript", content=None):
    content = content or {"provider_item_id": "item_original", "text": "PRIVATE_REQUEST_TEXT",
                          "input_id": "commit:original"}
    if kind == "user_transcript":
        content = {"capture_input_version": 1, **content}
    return ContextRecord(id=uuid4(), owner_id=capture.owner_id, session_id=capture.session_id,
        sequence=1, origin_sequence=1, source_namespace=capture.namespace,
        source_event_id=capture._source("input", "commit:original"), revision=1,
        supersedes_event_id=None, kind=kind,
        trust="user_input" if kind == "user_transcript" else "tool_receipt",
        finality="final", source_time=datetime.now(UTC), recorded_at=datetime.now(UTC),
        payload_id=uuid4(), availability="available", content_digest="fixture", is_current=True,
        content=content)


class ToolFollowupDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = RealtimeSessionContext(audio_session_id="followup-fixture", owner_id="owner",
            jarvis_session_id=uuid4(), client_kind="rayban_audio", session=SimpleNamespace())
        self.repository = SimpleNamespace(append=AsyncMock(), request_evidence=AsyncMock())
        self.capture = SessionContextCapture(self.repository, owner_id=self.context.owner_id,
            session_id=self.context.jarvis_session_id, provider="openai", audio_session_id=self.context.audio_session_id)
        self.original = _record(self.capture)
        self.repository.request_evidence.return_value = self.original
        self.session = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()), context=self.context,
            settings=JarvisSettings(_env_file=None, OPENAI_API_KEY="offline-fixture-not-a-provider-key"),
            repository=SimpleNamespace(create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
                                       get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=(), context_capture=self.capture)
        self.session._realtime_context_ready.set()
        self.session._voice_route_runtime = SimpleNamespace(
            ready=True, before_session_close=AsyncMock(), after_session_close=AsyncMock())
        self.session._send_app_event = AsyncMock(return_value=True)
        self.pool = _Sources()
        self.session._source_context.pool = self.pool
        self.wire = provider_fixtures._Wire(self.session)
        self.session.openai._websocket = self.wire
        self.receiver = asyncio.create_task(self.receive())
        self.waiters = []
        self.writer_held = False
        self.tool_record = _record(self.capture, kind="tool_outcome", content={"private": "RESULT_BODY"})
        self.saved = SourceBoundToolResult({"status": "success", "memory_status": "stored",
            "source": {"ingestion_status": "pending"}, "private": "PRIVATE_RESULT_TEXT"}, self.tool_record)
        self.success = self.register("record_fact", self.saved)
        self.failure = self.register("query_status", {"status": "failed", "error": {
            "code": "fixture_unavailable", "message": "PRIVATE_ERROR_TEXT"}})

    def register(self, name, result):
        handler = AsyncMock(return_value=result)
        self.session.tool_registry.register(ToolDefinition(name, "Offline tool", {}, handler))
        return handler

    async def asyncTearDown(self):
        if self.writer_held:
            self.session.openai._send_lock.release()
        for task in self.waiters:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.waiters, return_exceptions=True)
        await self.session.close("fixture_complete")
        self.receiver.cancel()
        await asyncio.gather(self.receiver, return_exceptions=True)
        await self.capture.close()
        self.assertEqual(self.pool.depth, 0)

    async def receive(self):
        self.session._provider_receive_task = asyncio.current_task()
        while True:
            event = await self.wire.incoming.get()
            try:
                await self.session._handle_openai_event(event)
            finally:
                self.wire.incoming.task_done()

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                if self.receiver.done():
                    await self.receiver
                await asyncio.sleep(0)

    async def begin(self):
        self.capture.admit_openai_commit("original", 1)
        self.capture.openai_committed("item_original")
        self.capture.openai_transcript("item_original", "PRIVATE_REQUEST_TEXT")
        self.session._turn_generation = 1
        self.session._normal_turn_pending = True
        self.assertTrue(await self.session._request_normal_response(turn_generation=1))
        await self.until(lambda: "response-1" in self.session._active_responses)
        return self.session._active_responses["response-1"]

    async def call(self, state, name="record_fact", call_id="call_success"):
        await self.session._handle_function_call_item({"type": "function_call", "name": name,
            "call_id": call_id, "arguments": "{}"}, response_id=state.response_id)
        await self.until(lambda: not state.pending_tool_call_ids)

    async def finish(self, state, *, expect_response=True):
        count = len(self.wire.responses)
        await self.session._finish_provider_response(state.response_id,
            {"id": state.response_id, "status": "completed", "output": []})
        if expect_response:
            await self.until(lambda: len(self.wire.responses) == count + 1
                and f"response-{count + 1}" in self.session._active_responses)
            return self.session._active_responses[f"response-{count + 1}"]

    async def quiet_update(self):
        root = uuid4()
        self.session._source_context.observe((root,))
        await self.session.openai.send_event(RealtimeVoiceRoute._text_item(
            "quiet_snapshot", "context_snapshot", "PRIVATE_QUIET_SNAPSHOT", {"revision": 5}),
            write_guard=self.session._source_context.context_guard)
        return root

    def delivery(self):
        instructions = self.wire.responses[-1]["event"]["response"]["instructions"]
        section = instructions.split("\n\nCurrent tool-followup delivery:\n", 1)[1]
        return instructions, json.loads(section[section.index('{"request":'):])

    async def test_mixed_written_receipts_survive_newer_quiet_snapshot(self):
        state = await self.begin()
        await self.call(state, "query_status", "call_failure")
        await self.call(state)
        quiet = await self.quiet_update()
        await self.finish(state)
        instructions, delivery = self.delivery()
        self.assertIn(build_base_prompt(), instructions)
        self.assertEqual(delivery["request"], {"provider_item_id": "item_original",
            "canonical_event_id": str(self.original.id), "revision": 1,
            "quoted_user_content": {"speaker": "USER", "text": "PRIVATE_REQUEST_TEXT", "status": "available"}})
        self.assertEqual([(r["call_id"], r["status"]) for r in delivery["receipts"]],
            [("call_failure", "failed"), ("call_success", "success")])
        self.assertEqual(delivery["receipts"][1]["canonical_event_id"], str(self.tool_record.id))
        self.assertEqual(delivery["receipts"][1]["owner_outcome"],
            {"memory_status": "stored", "source_ingestion_status": "pending"})
        for marker in ("PRIVATE_RESULT_TEXT", "PRIVATE_ERROR_TEXT", "PRIVATE_QUIET_SNAPSHOT"):
            self.assertNotIn(marker, instructions)
        self.assertNotIn("PRIVATE_REQUEST_TEXT", repr(state.tool_followup))
        self.assertEqual(self.repository.request_evidence.await_count, 2)
        self.success.assert_awaited_once()
        self.failure.assert_awaited_once()
        snapshot = self.session._source_context.snapshot_for(
            self.wire.responses[-1]["event"]["response"]["metadata"]["jarvis_request_id"])
        self.assertTrue({self.original.payload_id, self.tool_record.payload_id, quiet}.issubset(snapshot.sources))
        self.assertTrue(any(set(snapshot.sources).issubset(roots) for roots in self.pool.locks))
        events = self.wire.events
        quiet_index = next(i for i, event in enumerate(events) if event.get("item", {}).get("id") == "quiet_snapshot")
        self.assertLess(quiet_index, len(events) - 1)
        self.assertEqual(events[-1]["type"], "response.create")
        self.assertNotIn("tool_choice", events[-1]["response"])
        self.assertNotIn("input", events[-1]["response"])

    async def test_receipts_accumulate_across_actual_sequential_continuations(self):
        state = await self.begin()
        state.task_instructions = "Preserve the existing delivery policy."
        await self.call(state, "query_status", "first_call")
        state = await self.finish(state)
        await self.call(state, call_id="second_call")
        await self.quiet_update()
        await self.finish(state)
        instructions, delivery = self.delivery()
        self.assertIn("Preserve the existing delivery policy.", instructions)
        self.assertEqual([r["call_id"] for r in delivery["receipts"]], ["first_call", "second_call"])
        self.assertEqual(self.repository.request_evidence.await_count, 2)
        self.success.assert_awaited_once()
        self.failure.assert_awaited_once()

    async def test_request_deleted_after_output_blocks_followup_under_existing_guard(self):
        state = await self.begin()
        await self.call(state)
        self.pool.withdrawn.add(self.original.payload_id)
        with self.assertRaises(LiveContextError) as rejected:
            await self.finish(state, expect_response=False)
        self.assertEqual(rejected.exception.code, "live_context_revoked")
        self.assertEqual(len(self.wire.responses), 1)
        self.assertTrue(self.session._context_revoked)
        self.success.assert_awaited_once()

    async def test_success_receipt_withdrawal_blocks_followup(self):
        state = await self.begin()
        await self.call(state)
        self.pool.withdrawn.add(self.tool_record.payload_id)
        with self.assertRaises(LiveContextError) as rejected:
            await self.finish(state, expect_response=False)
        self.assertEqual(rejected.exception.code, "live_context_revoked")
        self.assertEqual(len(self.wire.responses), 1)
        self.assertTrue(self.session._context_revoked)

    async def test_request_revision_conflict_cannot_merge_or_replay_effects(self):
        state = await self.begin()
        await self.call(state)
        self.repository.request_evidence.return_value = self.original.model_copy(update={
            "id": uuid4(), "payload_id": uuid4(), "revision": 2, "finality": "revised"})
        await self.call(state, "query_status", "revised_call")
        self.assertTrue(state.tool_followup.binding_conflict)
        self.failure.assert_not_awaited()
        await self.finish(state, expect_response=False)
        self.assertEqual(len(self.wire.responses), 1)
        self.success.assert_awaited_once()

    async def test_new_user_turn_cancels_old_delivery_without_replaying_mutation(self):
        state = await self.begin()
        await self.call(state)
        async with self.session._response_admission_lock:
            await self.session._begin_user_turn_locked()
        await self.finish(state, expect_response=False)
        self.assertEqual(len(self.wire.responses), 1)
        self.success.assert_awaited_once()

    async def test_scope_change_after_preparation_drops_delivery_at_actual_writer(self):
        state = await self.begin()
        await self.call(state)
        await self.session.openai._send_lock.acquire()
        self.writer_held = True
        task = asyncio.create_task(self.finish(state, expect_response=False))
        self.waiters.append(task)
        await self.until(lambda: self.session._pending_response is not None and
            self.session._source_context.snapshot_for(self.session._pending_response.request_id) is not None)
        self.session.context = replace(self.context, owner_id="replacement-owner")
        self.session.openai._send_lock.release()
        self.writer_held = False
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.wire.responses), 1)
        self.success.assert_awaited_once()

    async def test_original_and_unbound_ordinary_paths_have_no_delivery_directive(self):
        state = await self.begin()
        self.assertNotIn("Current tool-followup delivery", self.wire.responses[0]["event"]["response"]["instructions"])
        self.repository.request_evidence.return_value = None
        await self.call(state, "query_status", "unbound_call")
        await self.finish(state)
        self.assertNotIn("Current tool-followup delivery", self.wire.responses[-1]["event"]["response"]["instructions"])

    async def test_receipt_capture_failure_preserves_reported_effect_uncertainty(self):
        state = await self.begin()
        handler = self.register("local_commit", {"status": "failed", "error": {
            "code": "tool_receipt_unavailable_after_commit", "local_effect_reported": True,
            "message": "PRIVATE_LOCAL_EFFECT_MESSAGE"}})
        await self.call(state, "local_commit", "local_call")
        await self.finish(state)
        instructions, delivery = self.delivery()
        self.assertTrue(delivery["receipts"][0]["local_effect_reported"])
        self.assertNotIn("PRIVATE_LOCAL_EFFECT_MESSAGE", instructions)
        handler.assert_awaited_once()

    async def test_generic_status_or_wrong_shape_does_not_report_a_local_effect(self):
        state = await self.begin()
        payloads = (
            {"status": "success"},
            {"status": "failed", "local_effect_reported": True},
            {"status": "failed", "error": {"local_effect_reported": "true"}},
            {"status": "failed", "error": {"local_effect_reported": False}},
            {"status": "failed", "error": "PRIVATE_NOT_A_RECEIPT"},
        )
        for index, payload in enumerate(payloads):
            name = f"effect_shape_{index}"
            self.register(name, payload)
            await self.call(state, name, f"effect_call_{index}")
        await self.finish(state)
        instructions, delivery = self.delivery()
        self.assertEqual(len(delivery["receipts"]), len(payloads))
        self.assertTrue(all(receipt["local_effect_reported"] is False
                            for receipt in delivery["receipts"]))
        self.assertNotIn("PRIVATE_NOT_A_RECEIPT", instructions)


class FollowupContractTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return RealtimeToolRequest("owner", uuid4(), 1, uuid4(), uuid4(), 1, "item_original")

    def test_complete_entry_budget_and_duplicates(self):
        delivery = RealtimeToolFollowup(request=self.request())
        for index in range(20):
            delivery = delivery.append(RealtimeToolReceipt(f"call_{index}_" + "x" * 145,
                "t" * 80, "success" if index % 2 else "failed", uuid4(), uuid4()))
        self.assertEqual(len(delivery.receipts), 16)
        self.assertEqual(delivery.omitted_receipts, 4)
        self.assertEqual(delivery.append(delivery.receipts[0]), delivery)
        wire = build_tool_followup_instructions(delivery)
        self.assertLessEqual(len(wire.encode()), TOOL_FOLLOWUP_MAX_BYTES)
        parsed = json.loads(wire[wire.index('{"request":'):])
        self.assertEqual(len(parsed["receipts"]) + parsed["omitted_receipts"], 20)
        self.assertEqual(parsed["receipts"][0]["call_id"], delivery.receipts[0].call_id)

    def test_request_scope_finality_and_identity_are_not_inferred_from_prose(self):
        capture = SessionContextCapture(SimpleNamespace(), owner_id="owner", session_id=uuid4(),
            provider="openai", audio_session_id="audio")
        record = _record(capture)
        kwargs = dict(owner_id="owner", session_id=capture.session_id, turn_generation=1,
                      source_namespace=capture.namespace)
        self.assertIsNotNone(RealtimeToolRequest.from_record(record, **kwargs))
        for change in ({"owner_id": "foreign"}, {"session_id": uuid4()}, {"is_current": False},
                       {"availability": "removed"}, {"finality": "provisional"},
                       {"source_namespace": "foreign"}, {"content": {"provider_item_id": "bad\nPRIVATE"}}):
            with self.subTest(change=tuple(change)):
                self.assertIsNone(RealtimeToolRequest.from_record(record.model_copy(update=change), **kwargs))

    async def test_observer_reuses_one_lookup_and_generic_default_is_unchanged(self):
        registry = JarvisToolRegistry()
        handler = AsyncMock(return_value={"status": "success"})
        registry.register(ToolDefinition("inspect_status", "Offline", {}, handler))
        capture = SimpleNamespace(request_for_call=AsyncMock(return_value=None), tool_outcome=lambda *a, **kw: None)
        observed = []
        executor = JarvisToolExecutor(registry, context_capture=capture,
            on_request_evidence=lambda call, request: observed.append((call, request)))
        self.assertEqual(await executor.execute("inspect_status", {}, provider_call_id="call_1"), {"status": "success"})
        capture.request_for_call.assert_awaited_once_with("call_1")
        self.assertEqual(observed, [("call_1", None)])
        self.assertEqual(await JarvisToolExecutor(registry).execute("inspect_status", {}), {"status": "success"})
        self.assertEqual(len(observed), 1)

    async def test_lookup_cancellation_does_not_invoke_handler_or_observer(self):
        registry = JarvisToolRegistry()
        handler = AsyncMock(return_value={"status": "success"})
        registry.register(ToolDefinition("inspect_status", "Offline", {}, handler))
        entered = asyncio.Event()
        async def lookup(call_id):
            entered.set()
            await asyncio.Event().wait()
        capture = SimpleNamespace(request_for_call=lookup)
        observed = []
        executor = JarvisToolExecutor(registry, context_capture=capture,
            on_request_evidence=lambda *args: observed.append(args))
        task = asyncio.create_task(executor.execute("inspect_status", {}, provider_call_id="call_1"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            handler.assert_not_awaited()
            self.assertEqual(observed, [])
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
