from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.perception.session_analysis_coordinator import SessionAnalysisCoordinator
from test_generic_visual_speech import GenericVisualSpeechTests


class ResponsePreparationCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reuse the established admission fixture without inheriting/rerunning
        # its test cases; these tests exercise additional cancellation seams.
        self.fixture = GenericVisualSpeechTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

        async def prepare(**kwargs):
            self.entered.set()
            await self.release.wait()
            return None

        self.session.prepare_visual_response = AsyncMock(side_effect=prepare)

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def wait_preparing(self):
        await asyncio.wait_for(self.entered.wait(), 1)

    async def test_barge_in_rejects_unsent_candidate_without_cancelling_producer(self):
        candidate = asyncio.create_task(self.fixture.candidate())
        await self.wait_preparing()
        await self.fixture.append()
        self.assertFalse(await asyncio.wait_for(candidate, 1))
        self.assertEqual(candidate.cancelling(), 0)
        self.assertFalse(self.fixture.creates())

    async def test_close_rejects_unsent_candidate_without_cancelling_producer(self):
        candidate = asyncio.create_task(self.fixture.candidate())
        await self.wait_preparing()
        await self.session.close("client_stop")
        self.assertFalse(await candidate)
        self.assertEqual(candidate.cancelling(), 0)
        self.assertFalse(self.fixture.creates())

    async def test_real_producer_cancellation_still_propagates(self):
        candidate = asyncio.create_task(self.fixture.candidate())
        await self.wait_preparing()
        candidate.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await candidate
        self.assertGreater(candidate.cancelling(), 0)
        self.assertIsNone(self.session._pending_response)

    async def test_cancelled_visual_speech_does_not_drop_analysis_tail_or_pending_frame(self):
        completed = []

        async def process(sequence):
            if sequence == 1:
                self.assertFalse(await self.fixture.candidate())
            completed.append(sequence)

        coordinator = SessionAnalysisCoordinator(process)
        try:
            await coordinator.submit(self.session.context.jarvis_session_id, 1)
            await self.wait_preparing()
            await coordinator.submit(self.session.context.jarvis_session_id, 2)
            await self.fixture.append()
            await asyncio.wait_for(coordinator.drain(), 1)
            self.assertEqual(completed, [1, 2])
        finally:
            await coordinator.close()

    async def test_submitted_durable_speech_keeps_unknown_outcome_until_lifecycle(self):
        self.release.set()
        provider = self.fixture.provider
        provider.create_release = asyncio.Event()
        outcome = AsyncMock()
        event_id = uuid4()
        speech = asyncio.create_task(self.session.speak_prospective_interrupt(
            message="Remember to floss.", event_id=event_id,
            subject_type="reminder", subject_id=uuid4(), on_outcome=outcome,
        ))
        await asyncio.wait_for(provider.create_started.wait(), 1)
        reservation = self.session._pending_response
        self.assertTrue(reservation.submitted)
        # Session close uses this same cancellation while a provider write is
        # blocked. Exercise it alone so that lifecycle reconciliation remains visible.
        self.session._response_preparation_task.cancel()
        self.assertTrue(await asyncio.wait_for(speech, 1))
        self.assertIn(str(event_id), self.session._pending_prospective_deliveries)
        outcome.assert_not_awaited()
        provider.create_release.set()
        await self.fixture.acknowledge(response_id="durable-response")
        await self.fixture.finish(response_id="durable-response", status="cancelled")
        self.assertNotIn(str(event_id), self.session._pending_prospective_deliveries)
        self.assertEqual(outcome.await_count, 1)
