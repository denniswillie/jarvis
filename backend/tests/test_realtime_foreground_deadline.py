"""Bounded foreground writes with truthful prewrite versus unknown outcomes."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session_manager import RealtimeSessionManager
from app.settings import JarvisSettings
import test_generic_visual_speech as gpt_fixture


class ForegroundWriteDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def test_timeout_configuration_must_be_positive_and_finite(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JarvisSettings(_env_file=None, OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=value)

    def connection(self, send):
        connection = OpenAIRealtimeConnection(
            settings=SimpleNamespace(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=.02),
            owner_id="isolated-deadline-fixture",
        )
        connection._websocket = SimpleNamespace(send=send, close=AsyncMock())
        self.addAsyncCleanup(connection.close)
        return connection

    async def completed_with_deadline(self, task):
        try:
            done, _ = await asyncio.wait({task}, timeout=.3)
            self.assertIn(task, done, "Foreground write remained unbounded")
            self.assertIsInstance(task.exception(), TimeoutError)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_queued_write_times_out_without_evaluating_its_payload(self):
        connection = self.connection(AsyncMock())
        factory = Mock(return_value={"type": "response.create"})
        await connection._send_lock.acquire()
        try:
            task = asyncio.create_task(connection.send_event(factory))
            await self.completed_with_deadline(task)
            factory.assert_not_called()
            connection._websocket.send.assert_not_awaited()
            self.assertEqual(connection._foreground_waiters, 0)
        finally:
            connection._send_lock.release()

    async def test_started_socket_write_has_a_deadline(self):
        started = asyncio.Event()

        async def stalled(_):
            started.set()
            await asyncio.Future()

        connection = self.connection(stalled)
        task = asyncio.create_task(connection.send_event({"type": "response.create"}))
        await asyncio.wait_for(started.wait(), .3)
        await self.completed_with_deadline(task)
        self.assertFalse(connection._send_lock.locked())
        self.assertEqual(connection._foreground_waiters, 0)

    async def test_successful_factory_still_runs_once_under_the_writer_lock(self):
        send = AsyncMock()
        connection = self.connection(send)
        calls = []

        def factory():
            calls.append(connection._send_lock.locked())
            return {"type": "response.create", "event_id": "fixture"}

        self.assertTrue(await connection.send_event(factory))
        self.assertEqual(calls, [True])
        self.assertEqual(json.loads(send.await_args.args[0])["event_id"], "fixture")
        self.assertEqual(connection._foreground_waiters, 0)


class GPTWriteOutcomeTests(ForegroundWriteDeadlineTests):
    async def asyncSetUp(self):
        self.h = gpt_fixture.GenericVisualSpeechTests()
        await self.h.asyncSetUp()
        self.addAsyncCleanup(self.h.asyncTearDown)
        self.session = self.h.session
        await self.session._handle_client_event({
            "type": "audio.input.append", "audio_base64": "AAA=",
        })
        await self.session._handle_client_event({"type": "audio.input.commit"})

    async def test_prewrite_timeout_releases_the_never_submitted_turn(self):
        connection = self.connection(AsyncMock())
        self.session.openai = connection
        self.session._client_receive_task = asyncio.current_task()
        await connection._send_lock.acquire()
        try:
            await self.session._handle_client_event({"type": "response.create"})
            await self.completed_with_deadline(self.session._response_preparation_task)
            self.assertFalse(self.session._normal_turn_pending)
            self.assertIsNone(self.session._pending_response)
            connection._websocket.send.assert_not_awaited()
        finally:
            connection._send_lock.release()

    async def test_started_timeout_preserves_correlation_for_a_late_ack(self):
        submitted = []

        async def stalled(payload):
            submitted.append(json.loads(payload))
            await asyncio.Future()

        connection = self.connection(stalled)
        self.session.openai = connection
        self.session._client_receive_task = asyncio.current_task()
        await self.session._handle_client_event({"type": "response.create"})
        await self.completed_with_deadline(self.session._response_preparation_task)
        reservation = self.session._pending_response
        self.assertIsNotNone(reservation)
        self.assertTrue(reservation.submitted)
        self.assertTrue(any(
            kind == "error" and payload.get("code") == "response_write_outcome_unknown"
            for kind, payload in self.h.app_events
        ))
        await self.session._handle_client_event({"type": "response.create"})
        self.assertEqual(len(submitted), 1, "Unknown speech must not be replayed")
        await self.session._handle_openai_event({
            "type": "response.created",
            "response": {"id": "late-response", "metadata": submitted[0]["response"]["metadata"]},
        })
        self.assertIsNone(self.session._pending_response)
        self.assertEqual(self.session._current_response_id, "late-response")
        self.assertFalse(self.session._active_responses["late-response"].interrupted)

    async def test_started_reminder_timeout_remains_owned_instead_of_reporting_failure(self):
        submitted = []

        async def stalled(payload):
            submitted.append(json.loads(payload))
            await asyncio.Future()

        connection = self.connection(stalled)
        self.session.openai = connection
        repository = SimpleNamespace(mark_event_delivery=AsyncMock())
        self.session.reminder_repository = repository
        manager = RealtimeSessionManager()
        await manager.register(
            audio_session_id=self.session.context.audio_session_id,
            owner_id=self.session.context.owner_id,
            jarvis_session_id=self.session.context.jarvis_session_id,
            session=self.session,
        )
        self.addAsyncCleanup(
            manager.unregister, audio_session_id=self.session.context.audio_session_id,
            owner_id=self.session.context.owner_id,
            jarvis_session_id=self.session.context.jarvis_session_id,
        )
        event_id = uuid4()
        task = asyncio.create_task(manager.speak_reminder(
            owner_id=self.session.context.owner_id,
            preferred_jarvis_session_id=self.session.context.jarvis_session_id,
            message="Fixture reminder.", reminder_id=uuid4(), reminder_event_id=event_id,
        ))
        try:
            done, _ = await asyncio.wait({task}, timeout=.3)
            self.assertIn(task, done)
            outcomes = await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(outcomes[0], self.session.context.jarvis_session_id)
            self.assertEqual(
                [call.kwargs["event_type"] for call in repository.mark_event_delivery.await_args_list],
                ["dispatched_to_session"],
            )
            self.assertIn(str(event_id), self.session._pending_reminder_events)
            self.assertTrue(self.session._pending_response.submitted)
            self.assertFalse(any(kind == "reminder.failed" for kind, _ in self.h.app_events))
            await self.session._handle_openai_event({
                "type": "response.created",
                "response": {"id": "late-reminder", "metadata": submitted[0]["response"]["metadata"]},
            })
            self.assertEqual(self.session._current_response_id, "late-reminder")
            self.assertIn(str(event_id), self.session._pending_reminder_events)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
