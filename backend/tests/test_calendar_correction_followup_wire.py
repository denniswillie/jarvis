"""Calendar A2 wire regression, not model obedience or native memory acceptance.

Reuse the maintained real capture/executor/continuation/writer harness. SQL and
provider transports remain offline fixtures; the successful remember result is
a fixture, not a new persistence operation or an indexing/hearing claim.
"""
import asyncio
from dataclasses import replace
import json
import unittest
from uuid import uuid4

from app.realtime.source_context import LiveContextError
from app.tools.provenance import SourceBoundToolResult
from app.voice_routes.realtime_adapter import RealtimeVoiceRoute
import test_realtime_tool_followup_delivery as followup_fixtures


CORRECTION = (
    "Correction to the dated report I just asked you to remember, for August 1, "
    "2026, at 10 a.m. UTC, the cedar calibration code was seventy-one, not "
    "seventeen. Keep the earlier report as history and remember this correction."
)
QUIET_HISTORY = (
    "Historical source, not a current user request: on August 1, 2026 at "
    "10:00 AM UTC, the earlier cedar calibration code report was 17."
)
CALL_ID = "call_calendar_correction"
COMMIT_ID = "calendar-correction"
ITEM_ID = "item_calendar_correction"


class CalendarCorrectionFollowupWireTests(unittest.IsolatedAsyncioTestCase):
    # Reuse harness operations without inheriting/discovering its test methods.
    register = followup_fixtures.ToolFollowupDeliveryTests.register
    receive = followup_fixtures.ToolFollowupDeliveryTests.receive
    until = followup_fixtures.ToolFollowupDeliveryTests.until
    call = followup_fixtures.ToolFollowupDeliveryTests.call
    finish = followup_fixtures.ToolFollowupDeliveryTests.finish
    delivery = followup_fixtures.ToolFollowupDeliveryTests.delivery
    asyncTearDown = followup_fixtures.ToolFollowupDeliveryTests.asyncTearDown

    async def asyncSetUp(self):
        await followup_fixtures.ToolFollowupDeliveryTests.asyncSetUp(self)
        self.original = followup_fixtures._record(self.capture, content={
            "provider_item_id": ITEM_ID, "text": CORRECTION,
            "input_id": "commit:" + COMMIT_ID, "transcript_status": "finalized",
        }).model_copy(update={
            "revision": 2,
            "source_event_id": self.capture._source("input", "commit:" + COMMIT_ID),
        })
        self.repository.request_evidence.return_value = self.original
        source_id = str(uuid4())
        result = {
            "status": "success", "memory_status": "stored", "memory_id": source_id,
            "replayed": False,
            "source": {
                "source_id": source_id, "owner_id": self.context.owner_id,
                "session_id": str(self.context.jarvis_session_id),
                "revision": 1, "generation": 1, "is_current": True,
                "source_time": "2026-08-01T10:00:00Z", "ingestion_status": "pending",
            },
            "meaning": (
                "Local source storage and queued ingestion, not semantic-search "
                "readiness or independent truth."
            ),
        }
        self.tool_record = followup_fixtures._record(self.capture,
            kind="tool_outcome", content={
                "provider_call_id": CALL_ID, "tool_name": "remember", "result": result,
                "input_id": "commit:" + COMMIT_ID,
            }).model_copy(update={
                "source_event_id": self.capture._source("bound-tool", CALL_ID),
            })
        self.saved = SourceBoundToolResult(result, self.tool_record)
        self.remember = self.register("remember", self.saved)

    async def begin_correction(self):
        self.capture.admit_openai_commit(COMMIT_ID, 2)
        self.capture.openai_committed(ITEM_ID)
        self.capture.openai_transcript(ITEM_ID, CORRECTION)
        self.session._turn_generation = 2
        self.session._normal_turn_pending = True
        self.assertTrue(await self.session._request_normal_response(turn_generation=2))
        await self.until(lambda: "response-1" in self.session._active_responses)
        state = self.session._active_responses["response-1"]
        await self.call(state, "remember", CALL_ID)
        self.assertEqual(state.tool_followup.request.event_id, self.original.id)
        self.assertEqual(state.tool_followup.request.turn_generation, 2)
        return state

    async def quiet_history(self):
        root = uuid4()
        self.session._source_context.observe((root,))
        await self.session.openai.send_event(RealtimeVoiceRoute._text_item(
            "calendar_quiet_snapshot", "context_snapshot", QUIET_HISTORY, {"revision": 5}),
            write_guard=self.session._source_context.context_guard)
        return root

    def assert_single_original_function_output(self):
        outputs = [event for event in self.wire.events
            if event.get("item", {}).get("type") == "function_call_output"
            and event["item"].get("call_id") == CALL_ID]
        self.assertEqual(len(outputs), 1)
        output = json.loads(outputs[0]["item"]["output"])
        self.assertEqual(output, dict(self.saved))
        self.assertNotIn("source_bound_tool_result", output)
        self.assertEqual(output["memory_status"], "stored")
        self.assertEqual(output["source"]["ingestion_status"], "pending")
        self.remember.assert_awaited_once()
        return outputs[0]

    async def test_correction_and_stored_pending_result_survive_quiet_snapshot(self):
        state = await self.begin_correction()
        quiet = await self.quiet_history()
        await self.finish(state)
        instructions, delivery = self.delivery()
        self.assertEqual(delivery["request"], {
            "provider_item_id": ITEM_ID, "canonical_event_id": str(self.original.id),
            "revision": 2,
            "quoted_user_content": {"speaker": "USER", "text": CORRECTION, "status": "available"},
        })
        self.assertEqual(len(delivery["receipts"]), 1)
        receipt = delivery["receipts"][0]
        self.assertEqual((receipt["call_id"], receipt["tool"], receipt["status"]),
            (CALL_ID, "remember", "success"))
        self.assertEqual(receipt["canonical_event_id"], str(self.tool_record.id))
        self.assertEqual(receipt["owner_outcome"], {
            "memory_status": "stored", "source_ingestion_status": "pending", "replayed": False,
        })
        self.assertFalse(receipt["local_effect_reported"])
        self.assertEqual(delivery["omitted_receipts"], 0)
        self.assertEqual(delivery["evidence_boundary"],
            "tool_outputs_written_not_provider_adoption_or_hearing")
        self.assertNotIn(QUIET_HISTORY, instructions)
        self.assertNotIn(CORRECTION, repr(state.tool_followup))
        function_output = self.assert_single_original_function_output()
        quiet_event = next(event for event in self.wire.events
            if event.get("item", {}).get("id") == "calendar_quiet_snapshot")
        self.assertIn(QUIET_HISTORY, json.dumps(quiet_event))
        self.assertLess(self.wire.events.index(function_output), self.wire.events.index(quiet_event))
        self.assertEqual(self.wire.events[-1]["type"], "response.create")
        self.assertLess(self.wire.events.index(quiet_event), len(self.wire.events) - 1)
        response = self.wire.responses[-1]["event"]["response"]
        self.assertNotIn("input", response)
        self.assertNotIn("tool_choice", response)
        self.assertEqual(response["metadata"]["jarvis_turn_generation"], "2")
        snapshot = self.session._source_context.snapshot_for(response["metadata"]["jarvis_request_id"])
        self.assertTrue({self.original.payload_id, self.tool_record.payload_id, quiet}.issubset(snapshot.sources))
        self.assertTrue(any(set(snapshot.sources).issubset(roots) for roots in self.pool.locks))
        self.assertEqual(self.repository.request_evidence.await_count, 1)

    async def assert_withdrawn_root_blocks_followup(self, which):
        state = await self.begin_correction()
        await self.quiet_history()
        root = self.original.payload_id if which == "request" else self.tool_record.payload_id
        self.pool.withdrawn.add(root)
        with self.assertRaises(LiveContextError) as rejected:
            await self.finish(state, expect_response=False)
        self.assertEqual(rejected.exception.code, "live_context_revoked")
        self.assertTrue(self.session._context_revoked)
        self.assertEqual(len(self.wire.responses), 1)
        self.assert_single_original_function_output()

    async def test_withdrawn_correction_root_blocks_followup(self):
        await self.assert_withdrawn_root_blocks_followup("request")

    async def test_withdrawn_result_root_blocks_followup(self):
        await self.assert_withdrawn_root_blocks_followup("result")

    async def hold_prepared_followup(self, state):
        await self.session.openai._send_lock.acquire()
        self.writer_held = True
        task = asyncio.create_task(self.finish(state, expect_response=False))
        self.waiters.append(task)
        await self.until(lambda: self.session._pending_response is not None and
            self.session._source_context.snapshot_for(self.session._pending_response.request_id) is not None)
        return task, self.session._pending_response

    def release_writer(self):
        self.session.openai._send_lock.release()
        self.writer_held = False

    async def assert_mismatched_root_blocks_at_writer(self, which):
        state = await self.begin_correction()
        await self.quiet_history()
        task, reservation = await self.hold_prepared_followup(state)
        delivery = reservation.tool_followup
        if which == "request":
            reservation.tool_followup = replace(delivery,
                request=replace(delivery.request, payload_id=uuid4()))
        else:
            reservation.tool_followup = replace(delivery,
                receipts=(replace(delivery.receipts[0], payload_id=uuid4()),))
        self.release_writer()
        with self.assertRaises(LiveContextError) as rejected:
            await asyncio.wait_for(task, 2)
        self.assertEqual(rejected.exception.code, "live_context_source_invalid")
        self.assertEqual(len(self.wire.responses), 1)
        self.assert_single_original_function_output()

    async def test_mismatched_correction_root_after_preparation_blocks_writer(self):
        await self.assert_mismatched_root_blocks_at_writer("request")

    async def test_mismatched_result_root_after_preparation_blocks_writer(self):
        await self.assert_mismatched_root_blocks_at_writer("result")

    async def test_owner_change_after_preparation_cannot_adopt_quiet_request(self):
        state = await self.begin_correction()
        await self.quiet_history()
        task, _ = await self.hold_prepared_followup(state)
        self.session.context = replace(self.context, owner_id="different-owner")
        self.release_writer()
        await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.wire.responses), 1)
        self.assert_single_original_function_output()
