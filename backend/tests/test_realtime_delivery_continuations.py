from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import test_realtime_output_phases as phase_fixtures


class DeliveryContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.case = phase_fixtures.RealtimeOutputPhaseTests()
        await self.case.asyncSetUp()
        self.session = self.case.session
        self.fixture = self.case.fixture

    async def asyncTearDown(self):
        await self.case.asyncTearDown()

    async def start_delivery(self, kind="reminder"):
        await self.case.start_durable(kind)
        self.session.tool_executor.execute = AsyncMock(return_value={"status": "success"})
        initial = self.fixture.creates()[-1]["response"]["instructions"]
        return initial.split("\n\nCurrent response task:\n", 1)[1]

    async def finish_from_provider(self, response_id="response-1"):
        # Production schedules preparation without blocking provider reception.
        previous = self.session._provider_receive_task
        self.session._provider_receive_task = asyncio.current_task()
        try:
            await phase_fixtures.finish(self.session, response_id)
        finally:
            self.session._provider_receive_task = previous
        return self.session._response_preparation_task

    async def drain(self, preparation=None):
        if preparation is not None:
            await asyncio.gather(preparation, return_exceptions=True)
        await asyncio.sleep(0)
        while self.session._continuation_settlement_tasks:
            await asyncio.gather(*tuple(self.session._continuation_settlement_tasks))
            await asyncio.sleep(0)

    def reminder_events(self):
        return [
            call.kwargs["event_type"]
            for call in self.session.reminder_repository.mark_event_delivery.await_args_list
        ]

    async def assert_task_preserved(self, kind):
        original_task = await self.start_delivery(kind)
        await self.session._send_complete_session_update(SimpleNamespace(
            as_prompt_payload=lambda: {"working_memory": {"marker": "NEW_CONTEXT"}},
        ))
        await self.case.tool()
        await self.drain(await self.finish_from_provider())
        created = self.fixture.creates()[-1]
        instructions = created["response"]["instructions"]
        self.assertEqual(instructions.count("\n\nCurrent response task:\n"), 1)
        self.assertEqual(instructions.split("\n\nCurrent response task:\n", 1)[1], original_task)
        self.assertIn("NEW_CONTEXT", instructions)
        self.assertNotIn(original_task, json.dumps(created["response"]["metadata"]))
        await self.fixture.acknowledge(response_id="continuation")
        self.assertEqual(
            self.session._active_responses["continuation"].task_instructions,
            original_task,
        )
        await phase_fixtures.output_item(self.session, "continuation", "answer", "final_answer")
        await phase_fixtures.speech(self.session, "continuation", "answer", "The requested message.")
        await phase_fixtures.finish(self.session, "continuation")

    async def test_reminder_continuation_keeps_task_and_refreshes_context(self):
        await self.assert_task_preserved("reminder")
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "delivered_to_session"])

    async def test_prospective_continuation_keeps_task_and_refreshes_context(self):
        await self.assert_task_preserved("prospective")
        self.case.outcome.assert_awaited_once_with("delivered", "continuation")

    async def test_visual_interrupt_continuation_keeps_task_and_refreshes_context(self):
        await self.assert_task_preserved("visual_interrupt")
        self.session.visual_repository.mark_visual_interrupt_spoken.assert_awaited_once()

    async def test_task_survives_multiple_tool_rounds_without_reusing_stale_context(self):
        original_task = await self.start_delivery()
        await self.case.tool()
        await self.drain(await self.finish_from_provider())
        await self.fixture.acknowledge(response_id="continuation")
        await self.session._send_complete_session_update(SimpleNamespace(
            as_prompt_payload=lambda: {"working_memory": {"marker": "LATEST_CONTEXT"}},
        ))
        await self.case.tool(response_id="continuation", call_id="second-tool")
        await self.drain(await self.finish_from_provider("continuation"))
        self.assertEqual(len(self.fixture.creates()), 3)
        instructions = self.fixture.creates()[-1]["response"]["instructions"]
        self.assertEqual(instructions.split("\n\nCurrent response task:\n", 1)[1], original_task)
        self.assertIn("LATEST_CONTEXT", instructions)
        self.assertEqual(self.reminder_events(), ["dispatched_to_session"])

    async def test_preparation_failure_settles_reminder_without_closing_socket(self):
        await self.start_delivery()
        await self.case.tool()
        self.session.prepare_visual_response = AsyncMock(side_effect=RuntimeError("preparation unavailable"))
        await self.drain(await self.finish_from_provider())
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "failed"])
        self.assertFalse(self.session._pending_reminder_events)
        self.assertFalse(self.session._closed.is_set())
        self.assertIsNone(self.session._pending_response)
        self.assertEqual(len(self.fixture.creates()), 1)
        metadata = self.session.reminder_repository.mark_event_delivery.await_args.kwargs["metadata"]
        self.assertEqual(metadata["reason"], "continuation_preparation_failed")

    async def test_preparation_failure_settles_prospective_callback(self):
        await self.start_delivery("prospective")
        await self.case.tool()
        self.session.prepare_visual_response = AsyncMock(side_effect=RuntimeError("preparation unavailable"))
        await self.drain(await self.finish_from_provider())
        self.case.outcome.assert_awaited_once_with("failed", None)
        self.assertFalse(self.session._pending_prospective_deliveries)
        self.assertFalse(self.session._closed.is_set())

    async def test_cancel_before_preparation_coroutine_starts_is_reconciled(self):
        await self.start_delivery()
        await self.case.tool()
        self.session.prepare_visual_response = AsyncMock(return_value=None)
        preparation = await self.finish_from_provider()
        request = self.session._pending_response
        self.assertFalse(request.submitted)
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.drain(preparation)
        self.session.prepare_visual_response.assert_not_awaited()
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "interrupted"])
        self.assertFalse(self.session._pending_reminder_events)
        self.assertIn(request.request_id, self.session._aborted_continuation_requests)
        self.assertEqual(len(self.fixture.creates()), 1)

    async def test_direct_cancellation_before_preparation_starts_releases_reservation(self):
        await self.start_delivery("prospective")
        await self.case.tool()
        preparation = await self.finish_from_provider()
        request = self.session._pending_response
        self.assertFalse(request.submitted)
        preparation.cancel()
        await self.drain(preparation)
        self.assertIsNone(self.session._pending_response)
        self.case.outcome.assert_awaited_once_with("interrupted", None)

    async def test_barge_in_while_waiting_for_visual_preparation_settles_old_delivery(self):
        await self.start_delivery("prospective")
        await self.case.tool()
        entered = asyncio.Event()

        async def prepare(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.session.prepare_visual_response = AsyncMock(side_effect=prepare)
        preparation = await self.finish_from_provider()
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await self.fixture.append()
            await self.drain(preparation)
            self.case.outcome.assert_awaited_once_with("interrupted", None)
            self.assertTrue(self.session._normal_turn_pending)
            self.assertTrue(self.session._user_audio_pending)
            self.assertFalse(self.session._pending_prospective_deliveries)
        finally:
            if not preparation.done():
                preparation.cancel()
            await asyncio.gather(preparation, return_exceptions=True)

    async def test_barge_in_between_parent_done_and_continuation_admission_is_reconciled(self):
        await self.start_delivery()
        await self.case.tool()

        async def app_event(kind, **payload):
            await self.fixture._record_event(kind, **payload)
            if kind == "response.done" and payload.get("response_id") == "response-1":
                await self.fixture.append()

        self.session._send_app_event = AsyncMock(side_effect=app_event)
        await self.drain(await self.finish_from_provider())
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "interrupted"])
        self.assertTrue(self.session._user_audio_pending)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(len(self.fixture.creates()), 1)

    async def test_correlated_rejection_settles_once_even_with_duplicate_error_and_late_ack(self):
        await self.start_delivery("prospective")
        await self.case.tool()
        await self.drain(await self.finish_from_provider())
        created = self.fixture.creates()[-1]
        error = {
            "type": "error",
            "error": {"event_id": created["event_id"], "code": "invalid_request_error"},
        }
        await self.session._handle_openai_event(error)
        await self.session._handle_openai_event(error)
        self.case.outcome.assert_awaited_once_with("failed", None)
        self.assertIsNone(self.session._pending_response)
        await self.fixture.acknowledge(event=created, response_id="late")
        await phase_fixtures.finish(self.session, "late", status="cancelled")
        self.case.outcome.assert_awaited_once()
        self.assertFalse(self.session._pending_prospective_deliveries)

    async def test_old_request_rejection_does_not_clear_a_newer_user_reservation(self):
        await self.start_delivery()
        await self.case.tool()
        await self.drain(await self.finish_from_provider())
        old_request = self.fixture.creates()[-1]
        await self.fixture.start_user_response()
        current = self.session._pending_response
        await self.session._handle_openai_event({
            "type": "error",
            "error": {"event_id": old_request["event_id"], "code": "invalid_request_error"},
        })
        self.assertIs(self.session._pending_response, current)
        self.assertEqual(self.session._latest_response_request_id, current.request_id)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "failed"])

    async def test_submitted_cancel_waits_for_late_provider_settlement(self):
        original_task = await self.start_delivery("prospective")
        await self.case.tool()
        await self.drain(await self.finish_from_provider())
        created = self.fixture.creates()[-1]
        request = self.session._pending_response
        self.assertTrue(request.submitted)
        await self.session._handle_client_event({"type": "response.cancel"})
        self.case.outcome.assert_not_awaited()
        self.assertIn(request.request_id, self.session._unacknowledged_response_requests)
        await self.fixture.acknowledge(event=created, response_id="late")
        state = self.session._active_responses["late"]
        self.assertTrue(state.interrupted)
        self.assertEqual(state.task_instructions, original_task)
        await phase_fixtures.finish(self.session, "late", status="cancelled")
        self.case.outcome.assert_awaited_once_with("interrupted", "late")

    async def test_submitted_send_exception_remains_unknown_until_ack(self):
        original_task = await self.start_delivery()
        await self.case.tool()
        self.fixture.provider.fail_create = True
        await self.drain(await self.finish_from_provider())
        pending = self.session._pending_response
        self.assertIsNotNone(pending)
        self.assertTrue(pending.submitted)
        self.assertEqual(self.reminder_events(), ["dispatched_to_session"])
        self.fixture.provider.fail_create = False
        await self.fixture.acknowledge(response_id="accepted-after-send-error")
        self.assertEqual(
            self.session._active_responses["accepted-after-send-error"].task_instructions,
            original_task,
        )
        await phase_fixtures.output_item(self.session, "accepted-after-send-error", "answer", "final_answer")
        await phase_fixtures.speech(self.session, "accepted-after-send-error", "answer", "The reminder.")
        await phase_fixtures.finish(self.session, "accepted-after-send-error")
        self.assertEqual(self.reminder_events(), ["dispatched_to_session", "delivered_to_session"])

    async def test_shutdown_joins_continuation_settlement_before_close_reconciliation(self):
        await self.start_delivery()
        await self.case.tool()
        entered, release = asyncio.Event(), asyncio.Event()

        async def persist(**kwargs):
            if kwargs["event_type"] == "failed":
                self.assertFalse(self.session._response_admission_lock.locked())
                entered.set()
                await release.wait()

        self.session.reminder_repository.mark_event_delivery.side_effect = persist
        self.session.prepare_visual_response = AsyncMock(side_effect=RuntimeError("preparation unavailable"))
        preparation = await self.finish_from_provider()
        close_task = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            close_task = asyncio.create_task(self.session.close("client_closed"))
            await asyncio.sleep(0)
            self.assertFalse(close_task.done())
            release.set()
            await asyncio.wait_for(close_task, 1)
            self.assertEqual(self.reminder_events(), ["dispatched_to_session", "failed"])
            self.assertFalse(self.session._pending_reminder_events)
            self.assertFalse(self.session._continuation_settlement_tasks)
        finally:
            release.set()
            await asyncio.gather(preparation, return_exceptions=True)
            if close_task is not None:
                await asyncio.gather(close_task, return_exceptions=True)
