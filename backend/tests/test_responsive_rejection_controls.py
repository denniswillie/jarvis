"""Actual rejection handler/settlement ownership with controlled persistence."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import test_generic_visual_speech as fixtures


class RejectionControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = fixtures.GenericVisualSpeechTests()
        await self.h.asyncSetUp()
        self.session = self.h.session
        self.session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(return_value=True),
            after_session_close=AsyncMock(return_value=True))
        self.send = AsyncMock(wraps=self.session.openai.send_event)
        self.session.openai.send_event = self.send
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.reader = None

    async def asyncTearDown(self):
        self.release.set()
        if self.reader is not None:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        await self.h.asyncTearDown()

    async def submitted_reminder(self):
        event_id = str(uuid4())
        self.session._pending_reminder_events[event_id] = str(uuid4())
        async with self.session._response_admission_lock:
            reservation = self.session._reserve_response_locked("reminder", {"reminder_event_id": event_id})
        await self.session._send_reserved_response(reservation)
        self.assertTrue(reservation.submitted)
        return reservation

    def error(self, request_id):
        return {"type": "error", "event_id": "unrelated-outer-id",
                "error": {"event_id": request_id, "code": "fixture_create_rejected",
                          "message": "Fixture response rejected."}}

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                if self.reader is not None and self.reader.done():
                    await self.reader
                await asyncio.sleep(0.001)

    def cancellations(self):
        return [call.args[0] for call in self.send.await_args_list
                if isinstance(call.args[0], dict) and call.args[0].get("type") == "response.cancel"]

    async def test_real_reader_advances_during_rejection_persistence_and_late_ack_targets_only_old(self):
        old = await self.submitted_reminder()
        old_metadata = dict(self.h.creates()[-1]["response"]["metadata"])
        await self.session._handle_client_event({"type": "response.cancel"})
        self.assertTrue(self.session._response_controls.busy)

        async def persist(**kwargs):
            self.entered.set()
            await self.release.wait()
            return True

        self.session._persist_reminder_event_delivery = AsyncMock(side_effect=persist)
        incoming = asyncio.Queue()
        self.session.openai.receive_event = incoming.get
        self.reader = asyncio.create_task(self.session._openai_to_client_loop())
        self.session._provider_receive_task = self.reader
        await incoming.put(self.error(old.request_id))
        await asyncio.wait_for(self.entered.wait(), 1)
        await asyncio.wait_for(self.session._response_controls.wait_idle(), 1)
        self.assertFalse(self.cancellations())
        self.assertIn(old.request_id, self.session._aborted_continuation_requests)

        async with self.session._response_admission_lock:
            newer = self.session._reserve_response_locked("normal_user_turn", {})
        await self.session._send_reserved_response(newer)
        new_metadata = dict(self.h.creates()[-1]["response"]["metadata"])
        await incoming.put({"type": "response.created", "response": {"id": "new", "metadata": new_metadata}})
        await self.until(lambda: "new" in self.session._active_responses)
        self.assertFalse(self.release.is_set())
        self.assertFalse(self.session._active_responses["new"].interrupted)

        await incoming.put({"type": "response.created", "response": {"id": "old", "metadata": old_metadata}})
        await self.until(lambda: bool(self.cancellations()))
        self.assertEqual([event["response_id"] for event in self.cancellations()], ["old"])
        self.assertEqual(self.session._current_response_id, "new")
        self.assertFalse(self.session._active_responses["new"].interrupted)
        self.release.set()
        await self.until(lambda: not self.session._continuation_settlement_tasks)
        self.assertEqual(self.session._persist_reminder_event_delivery.await_count, 1)

    async def test_uncorrelated_error_does_not_release_waiting_request(self):
        old = await self.submitted_reminder()
        await self.session._handle_client_event({"type": "response.cancel"})
        error = self.error("not-the-request")
        error["event_id"] = old.request_id
        await self.session._handle_openai_event(error)
        self.assertTrue(self.session._response_controls.busy)
        self.assertIn(old.request_id, self.session._unacknowledged_response_requests)
        self.assertFalse(self.cancellations())

    async def test_duplicate_rejection_schedules_one_canonical_settlement(self):
        old = await self.submitted_reminder()
        settlement = AsyncMock()
        self.session._settle_unstarted_continuation = settlement
        await self.session._handle_openai_event(self.error(old.request_id))
        await self.session._handle_openai_event(self.error(old.request_id))
        await self.until(lambda: not self.session._continuation_settlement_tasks)
        settlement.assert_awaited_once()

    async def test_shutdown_joins_already_started_settlement(self):
        async def settle(*args, **kwargs):
            self.entered.set()
            await self.release.wait()
        self.session._settle_unstarted_continuation = AsyncMock(side_effect=settle)
        await self.session._settle_rejected_continuation({})
        await self.entered.wait()
        closing = asyncio.create_task(self.session.close("fixture_close"))
        try:
            await asyncio.sleep(0.01)
            self.assertFalse(closing.done())
            self.release.set()
            await asyncio.wait_for(closing, 1)
            self.assertFalse(self.session._continuation_settlement_tasks)
        finally:
            self.release.set()
            await asyncio.gather(closing, return_exceptions=True)

    async def test_settlement_capacity_is_bounded_and_failure_closes(self):
        async def settle(*args, **kwargs):
            await self.release.wait()
        self.session._settle_unstarted_continuation = AsyncMock(side_effect=settle)
        for _ in range(33):
            await self.session._settle_rejected_continuation({})
        self.assertLessEqual(len(self.session._continuation_settlement_tasks), 32)
        self.assertEqual(self.session._close_reason, "response_rejection_settlement_queue_exhausted")

    async def test_settlement_error_is_not_replayed_or_reported_as_success(self):
        self.session._settle_unstarted_continuation = AsyncMock(side_effect=RuntimeError("fixture failure"))
        await self.session._settle_rejected_continuation({})
        await self.until(lambda: self.session._closed.is_set())
        self.assertEqual(self.session._close_reason, "response_rejection_settlement_failed")
        self.assertEqual(self.session._settle_unstarted_continuation.await_count, 1)

    async def test_legacy_settlement_remains_inline(self):
        self.session._voice_route_runtime = None
        async def settle(*args, **kwargs):
            self.entered.set()
            await self.release.wait()
        self.session._settle_unstarted_continuation = AsyncMock(side_effect=settle)
        pending = asyncio.create_task(self.session._settle_rejected_continuation({}))
        await self.entered.wait()
        self.assertFalse(pending.done())
        self.release.set()
        await pending
        self.assertFalse(self.session._continuation_settlement_tasks)
