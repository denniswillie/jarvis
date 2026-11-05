"""Normal quiet tools through executor, continuation, wait and actual JSON writer.

Only storage/provider I/O is controlled. No SQL, provider adoption, playback or
historical c05a at-write evidence is claimed by these tests.
"""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.quiet import OwnerQuietRepository, QuietCommand
from app.preferences.repository import PreferenceRepository
from app.preferences.tools import PreferenceToolModule
from app.tools.invocation import ToolInvocationContext
from app.tools.module import ToolBuildContext
import test_attention_quiet as quiet_fixture
import test_realtime_tool_followup_wire_boundary as wire_fixture


class QuietFollowupWireTests(unittest.IsolatedAsyncioTestCase):
    # Helpers only, never inherit or collect the other suite's test cases.
    cleanup = wire_fixture.ToolFollowupWireBoundaryTests.cleanup
    until = wire_fixture.ToolFollowupWireBoundaryTests.until
    rows = wire_fixture.ToolFollowupWireBoundaryTests.rows
    observation = wire_fixture.ToolFollowupWireBoundaryTests.observation
    expire_inside_adapter_guard = wire_fixture.ToolFollowupWireBoundaryTests.expire_inside_adapter_guard
    mutations = wire_fixture.ToolFollowupWireBoundaryTests.mutations
    creates = wire_fixture.ToolFollowupWireBoundaryTests.creates
    finish_tool_response = wire_fixture.ToolFollowupWireBoundaryTests.finish_tool_response
    deferred_quiet_replacement = wire_fixture.ToolFollowupWireBoundaryTests.deferred_quiet_replacement

    async def asyncSetUp(self):
        await wire_fixture.ToolFollowupWireBoundaryTests.asyncSetUp(self)
        self.input_text = "You may interrupt again now; keep doing the background work."
        self.quiet_storage = quiet_fixture.LedgerFixture()
        self.quiet = OwnerQuietRepository(self.quiet_storage)
        source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.quiet_storage.authorize(self.scope.owner_id, self.scope.session_id, source)
        await self.quiet.save(self.scope.owner_id, self.scope.session_id, source,
            QuietCommand(action="activate", expected_revision=0))
        module = PreferenceToolModule(PreferenceRepository(self.quiet_storage))
        self.enforce = AsyncMock(return_value=0)
        module.bind_attention_service(SimpleNamespace(_closed=False, enforce_owner_quiet=self.enforce))
        for tool in module.build_tools(ToolBuildContext(settings=None, owner_id=self.scope.owner_id,
                                                       session_id=self.scope.session_id)):
            self.session.tool_registry.register(tool)

    async def tool_turn(self, *, mixed=False):
        await self.provider.inject("input_audio_buffer.speech_started", item_id="original-input")
        await self.provider.inject("input_audio_buffer.speech_stopped", item_id="original-input")
        await self.provider.inject("input_audio_buffer.committed", item_id="original-input")
        await self.provider.inject("conversation.item.input_audio_transcription.completed",
                                   item_id="original-input", transcript=self.input_text)
        await self.until(lambda: len(self.creates()) == 1)
        await self.provider.inject("response.created", response={"id": "tool-response",
            "metadata": self.creates()[0]["response"]["metadata"]})
        await self.until(lambda: "tool-response" in self.session._active_responses
                         and any(row.kind == "user_transcript" for row in self.canonical.records))
        record = next(row for row in self.canonical.records if row.kind == "user_transcript")
        self.quiet_storage.authorize(self.scope.owner_id, self.scope.session_id,
            ToolInvocationContext(request_event_id=record.id, request_revision=record.revision))
        self.tool_items = []
        names = ("set_attention_quiet", "unavailable") if mixed else ("set_attention_quiet",)
        for name in names:
            item = {"id": "item-" + name, "type": "function_call", "name": name, "call_id": "call-" + name,
                "arguments": json.dumps({"action": "resume", "expected_revision": 1}) if name == "set_attention_quiet" else "{}"}
            self.tool_items.append(item)
            await self.provider.inject("response.output_item.done", response_id="tool-response", item=item)
        await self.until(lambda: len(self.session._active_responses["tool-response"].tool_followup.receipts) == len(names)
                         and not self.session._ordinary_tool_tasks)
        self.original = self.session._active_responses["tool-response"].tool_followup.request
        self.assertIsNotNone(self.original)
        self.assertEqual(self.executions.records["call-set_attention_quiet"].status, "completed")
        self.assertEqual(len(self.quiet_storage.rows), 2)
        self.assertEqual(len(self.canonical.lookups), len(names))

    async def assert_wire(self, pending=None):
        await self.until(lambda: len(self.creates()) == 2 or self.session._closed.is_set())
        self.assertFalse(self.session._closed.is_set(), self.session._close_reason)
        final = self.creates()[1]
        raw, guarded, reservation = next(row for row in self.wire if json.loads(row[0]).get("event_id") == final["event_id"])
        self.assertTrue(guarded)
        if pending is not None:
            self.assertIs(reservation, pending)
        instructions = json.loads(raw)["response"]["instructions"]
        anchor = instructions.split("Current tool-followup delivery:\n", 1)[1]
        data = json.loads(anchor[anchor.index('{"request":'):])
        self.assertEqual(data["request"]["quoted_user_content"],
            {"speaker": "USER", "text": self.input_text, "status": "available"})
        self.assertEqual(data["request"]["canonical_event_id"], str(self.original.event_id))
        self.assertEqual(data["request"]["revision"], self.original.revision)
        result = data["receipts"][0]
        self.assertEqual(result["owner_outcome"], {"quiet_effective_state": "resumed", "quiet_revision": 2,
            "quiet_suppresses_unsolicited": False, "quiet_accepted_revision": 2, "replayed": False,
            "quiet_delivery_propagation": "checked", "quiet_playback_stop_confirmed": False})
        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["canonical_event_id"])  # No invented bound-result identity.
        output_index = next(i for i, event in enumerate(self.provider.writes)
            if event.get("item", {}).get("call_id") == "call-set_attention_quiet"
            and event["item"]["type"] == "function_call_output")
        self.assertLess(output_index, self.provider.writes.index(final))
        self.assertIn(self.original.payload_id, self.session._source_context.snapshot_for(final["event_id"]).sources)
        self.assertEqual(self.enforce.await_count, 1)
        self.assertEqual(len(self.quiet_storage.rows), 2)
        self.assertNotIn(self.input_text, repr(self.executions.records))
        self.assertNotIn(self.input_text, repr(reservation))
        self.assertNotIn(self.input_text, repr(self.trace_events.call_args_list))
        self.assertNotIn("tool_choice", final["response"])
        diagnostics = [call.kwargs for call in self.trace_events.call_args_list
                       if call.args == ("tool_followup_write",) and call.kwargs["request_id"] == final["event_id"]]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]["instructions_sha256"], hashlib.sha256(instructions.encode()).hexdigest())
        self.assertTrue(diagnostics[0]["anchor_present"])
        self.assertEqual(diagnostics[0]["outcome_fields"], sorted({key for row in data["receipts"] for key in row["owner_outcome"]}))
        return data

    async def test_normal_direct_resume_without_get_through_serialized_writer(self):
        with patch.object(OwnerQuietRepository, "get", new_callable=AsyncMock,
                          side_effect=AssertionError("no read-first tool")) as get:
            await self.tool_turn()
            await self.finish_tool_response()
            await self.assert_wire()
        get.assert_not_awaited()

    async def test_success_and_failure_survive_wait_newer_quiet_and_rebuilt_candidate(self):
        await self.tool_turn(mixed=True)
        pending, release = await self.deferred_quiet_replacement(replace_candidate=True)
        release.set()
        data = await self.assert_wire(pending)
        self.assertEqual(data["receipts"][1]["status"], "failed")
        self.assertEqual(data["receipts"][1]["owner_outcome"], {})
        self.assertEqual(len(self.canonical.lookups), 2)
        quiet = [event for event in self.provider.writes if event["type"] == "conversation.item.create"
                 and event.get("item", {}).get("type") == "message"]
        self.assertFalse(any("quiet-candidate-newest" in json.dumps(event) for event in quiet))
        self.assertFalse(any("quiet-candidate-one" in json.dumps(event) for event in quiet))
        self.assertIn("Current Live State briefing is temporarily unavailable", self.creates()[1]["response"]["instructions"])

    async def test_source_addition_refreshes_same_request_without_replaying_resume(self):
        await self.tool_turn()
        original_guard = self.session._source_context.guard
        snapshots, requests = [], []

        @asynccontextmanager
        async def add_once(snapshot):
            snapshots.append(snapshot)
            requests.append(self.session._pending_response)
            if len(snapshots) == 1:
                self.session._source_context.observe((uuid4(),))
            async with original_guard(snapshot):
                yield

        self.session._source_context.guard = add_once
        await self.finish_tool_response()
        await self.assert_wire()
        self.assertGreaterEqual(len(snapshots), 2)
        self.assertTrue(all(request is requests[0] for request in requests))
        self.assertEqual(len(self.canonical.lookups), 1)

    async def test_withdrawn_original_after_commit_never_exposes_followup(self):
        await self.tool_turn()
        pending, release = await self.deferred_quiet_replacement()
        self.storage.rejected.add(self.original.payload_id)
        release.set()
        await self.until(lambda: self.session._context_revoked or self.session._closed.is_set())
        self.assertFalse(pending.submitted)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(len(self.quiet_storage.rows), 2)  # Already-committed effect is not undone or replayed.

    async def test_new_user_generation_cancels_pending_resume_followup(self):
        await self.tool_turn()
        pending, release = await self.deferred_quiet_replacement()
        generation = self.session._turn_generation
        await self.provider.inject("input_audio_buffer.speech_started", item_id="replacement-input")
        await self.until(lambda: self.session._turn_generation > generation)
        release.set()
        await self.until(lambda: getattr(self.session, "_context_waiting_request_id", None) is None)
        self.assertFalse(pending.submitted)
        self.assertEqual(len(self.creates()), 1)
        self.assertEqual(self.enforce.await_count, 1)
