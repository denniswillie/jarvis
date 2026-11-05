"""Explicit provider rejection must settle initial speech, not just continuations."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import test_generic_visual_speech as gpt_fixture


class InitialResponseRejectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.h = gpt_fixture.GenericVisualSpeechTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)
        self.session = self.h.session
        self.event_id = uuid4()
        self.outcome = AsyncMock()
        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
        )

    async def start(self, kind):
        if kind == "reminder":
            return await self.session.speak_reminder(
                message="Fixture reminder.", reminder_id=uuid4(), reminder_event_id=self.event_id,
            )
        if kind == "prospective":
            return await self.session.speak_prospective_interrupt(
                message="The fixture keys are visible.", event_id=self.event_id,
                subject_type="visual_focus", subject_id=uuid4(), on_outcome=self.outcome,
            )
        return await self.h.candidate()

    async def reject(self, request_id):
        await self.session._handle_openai_event({
            "type": "error",
            "error": {"type": "invalid_request_error", "code": "fixture_rejection",
                      "event_id": request_id, "message": "Fixture request rejected."},
        })

    def failures(self, kind):
        name = {"visual_interrupt": "visual.interrupt.done",
                "reminder": "reminder.failed", "prospective": "prospective.failed"}[kind]
        return [payload for event, payload in self.h.app_events if event == name]

    async def check_rejection(self, kind):
        await self.start(kind)
        request = self.h.creates()[-1]
        await self.reject(request["event_id"])
        self.assertIsNone(self.session._pending_response)
        self.assertEqual(len(self.failures(kind)), 1)
        self.assertFalse(self.session._pending_reminder_events)
        self.assertFalse(self.session._pending_prospective_deliveries)
        if kind == "prospective":
            self.outcome.assert_awaited_once_with("failed", None)
        if kind == "reminder":
            self.assertEqual(
                [call.kwargs["event_type"] for call in
                 self.session.reminder_repository.mark_event_delivery.await_args_list],
                ["dispatched_to_session", "failed"],
            )
        # Duplicate errors and contradictory late acceptance must not reopen delivery.
        await self.reject(request["event_id"])
        await self.session._handle_openai_event({
            "type": "response.created",
            "response": {"id": "late-rejected", "metadata": request["response"]["metadata"]},
        })
        self.assertTrue(self.session._active_responses["late-rejected"].interrupted)
        await self.h.finish("late-rejected")
        self.assertEqual(len(self.failures(kind)), 1)
        self.assertFalse(any(event.endswith(".response_created") for event, _ in self.h.app_events))
        self.session.reminder_repository.set_event_provider_response_id.assert_not_awaited()

    async def test_initial_reminder_rejection_settles_once(self):
        await self.check_rejection("reminder")

    async def test_initial_prospective_rejection_settles_once(self):
        await self.check_rejection("prospective")

    async def test_initial_visual_rejection_settles_once(self):
        await self.check_rejection("visual_interrupt")

    async def check_early_rejection(self, kind):
        send = self.session._send_reserved_response

        async def reject_before_return(reservation, **kwargs):
            result = await send(reservation, **kwargs)
            await self.reject(reservation.request_id)
            return result

        self.session._send_reserved_response = reject_before_return
        result = await self.start(kind)
        self.assertEqual(len(self.failures(kind)), 1)
        self.assertFalse(any(event.endswith(".started") for event, _ in self.h.app_events))
        if kind != "reminder":
            self.assertFalse(result)

    async def test_reminder_rejection_before_sender_resumes_does_not_announce_started(self):
        await self.check_early_rejection("reminder")

    async def test_prospective_rejection_before_sender_resumes_does_not_announce_started(self):
        await self.check_early_rejection("prospective")

    async def test_visual_rejection_before_sender_resumes_does_not_announce_started(self):
        await self.check_early_rejection("visual_interrupt")

    async def test_unrelated_error_does_not_settle_pending_reminder(self):
        await self.start("reminder")
        pending = self.session._pending_response
        await self.reject("unrelated-request")
        self.assertIs(self.session._pending_response, pending)
        self.assertIn(str(self.event_id), self.session._pending_reminder_events)
        self.assertFalse(self.failures("reminder"))
