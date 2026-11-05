from __future__ import annotations

import asyncio
import base64
import json
import unittest
from contextlib import AsyncExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from app.assistant.proactive_policy import (
    GenericSpeechActivity,
    ProactiveInterruptGate,
    generic_visual_speech_rejection,
)
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.visual_context import VisualFrameMetadata
from app.settings import JarvisSettings
from app.visual_memory.interrupts import VisualInterruptService
from test_attention_quiet import LedgerFixture


NOW = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


class _Clock:
    seconds = 0.0

    def now(self):
        return NOW + timedelta(seconds=self.seconds)

    def monotonic(self):
        return 100 + self.seconds


class _Provider:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.close = AsyncMock()
        self.fail_create = False
        self.create_started = asyncio.Event()
        self.create_release: asyncio.Event | None = None

    async def send_event(self, event, *, write_guard=None):
        async with AsyncExitStack() as stack:
            if write_guard is not None:
                await stack.enter_async_context(write_guard())
            event = event() if callable(event) else event
            if event is None:
                return False
            self.events.append(event)
            if event["type"] == "response.create":
                self.create_started.set()
                if self.create_release is not None:
                    await self.create_release.wait()
                if self.fail_create:
                    raise RuntimeError("expected send failure")
            return True


class GenericSpeechPolicyTests(unittest.TestCase):
    def test_busy_quiet_source_and_independent_observation_policy(self) -> None:
        idle = GenericSpeechActivity(False, True, False, False, False, NOW, 100)
        def reason(activity=idle, source=NOW + timedelta(seconds=6), seconds=6):
            return generic_visual_speech_rejection(
                activity, source_matches=True, source_received_at=source,
                now=NOW + timedelta(seconds=seconds), monotonic_now=100 + seconds,
            )
        self.assertIsNone(reason())
        self.assertEqual(reason(seconds=4), "source_time_unavailable")
        self.assertEqual(reason(source=NOW, seconds=4), "user_turn_quiet_interval")
        self.assertEqual(reason(source=NOW), "superseded_observation")
        self.assertEqual(reason(source=None), "source_time_unavailable")
        self.assertEqual(reason(source=NOW.replace(tzinfo=None)), "source_time_unavailable")
        self.assertEqual(reason(replace(idle, user_audio_pending=True)), "user_turn_busy")
        self.assertEqual(reason(replace(idle, normal_turn_pending=True)), "user_turn_busy")
        self.assertEqual(reason(replace(idle, speech_pending_or_active=True)), "speech_busy")
        self.assertEqual(reason(replace(idle, closed=True)), "session_unavailable")


class GenericVisualSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.clock = _Clock()
        self.provider = _Provider()
        self.app_events = []
        quiet_pool = LedgerFixture()
        quiet_pool.is_in_transaction = lambda: quiet_pool.lock.locked()
        self.repository = SimpleNamespace(
            _pool=quiet_pool,
            create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
            get_recent_messages=AsyncMock(return_value=[]),
        )
        self.session = RealtimeAudioSession(
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(
                audio_session_id="audio", owner_id="owner", jarvis_session_id=uuid4(),
                client_kind="rayban_audio", session=SimpleNamespace(),
            ),
            settings=JarvisSettings(_env_file=None),
            repository=self.repository,
            tool_modules=(), context_providers=(),
            now=self.clock.now, monotonic=self.clock.monotonic,
        )
        self.session.openai = self.provider
        self.session._realtime_context_ready.set()
        self.session._send_app_event = AsyncMock(side_effect=self._record_event)

    async def asyncTearDown(self) -> None:
        await self.session.close("test_complete")

    async def _record_event(self, event_type, **payload):
        self.app_events.append((event_type, payload))

    def creates(self):
        return [event for event in self.provider.events if event["type"] == "response.create"]

    async def candidate(self, *, source=None, message="The window has opened."):
        return await self.session.speak_visual_interrupt(
            owner_id="owner", jarvis_session_id=self.session.context.jarvis_session_id,
            source_received_at=source or self.clock.now(),
            message=message, interrupt_id=uuid4(),
        )

    async def append(self):
        await self.session._handle_client_event({
            "type": "audio.input.append",
            "audio_base64": base64.b64encode(b"\x01\x00" * 240).decode(),
        })

    async def start_user_response(self):
        await self.append()
        await self.session._handle_client_event({"type": "audio.input.commit"})
        await self.session._handle_client_event({"type": "response.create"})
        return self.creates()[-1]

    async def acknowledge(self, event=None, response_id="response-1"):
        event = event or self.creates()[-1]
        await self.session._handle_openai_event({
            "type": "response.created",
            "response": {"id": response_id, "metadata": event["response"]["metadata"]},
        })
        return response_id

    async def finish(self, response_id="response-1", status="completed"):
        # A completed provider envelope alone is not proof of forwarded speech.
        # Lifecycle fixtures that model a successful answer also deliver audio.
        if status == "completed":
            for event in (
                {"type": "response.output_audio.delta", "delta": "AA=="},
                {"type": "response.output_audio.done"},
            ):
                await self.session._handle_openai_event({
                    **event, "response_id": response_id, "item_id": f"{response_id}-speech",
                })
        await self.session._handle_openai_event({
            "type": "response.done", "response": {"id": response_id, "status": status},
        })


    def block_reminder_dispatch(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def persist(**kwargs):
            if kwargs["event_type"] == "dispatched_to_session":
                entered.set()
                await release.wait()

        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(side_effect=persist),
            set_event_provider_response_id=AsyncMock(),
        )
        return entered, release

    async def cancel_test_task(self, task):
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_user_audio_pending_ack_and_normal_output_reject_without_cancelling(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=1)
        await self.append()
        self.assertFalse(await self.candidate())
        await self.session._handle_client_event({"type": "audio.input.commit"})
        await self.session._handle_client_event({"type": "response.create"})
        self.assertFalse(await self.candidate(message="Differently worded offer."))
        await self.acknowledge()
        self.assertFalse(await self.candidate())
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(any(event["type"] == "response.cancel" for event in self.provider.events))
        self.assertFalse(any(kind == "visual.interrupt.started" for kind, _ in self.app_events))
        self.assertTrue(self.session._proactive_interrupt_gate.try_begin())

    async def test_quiet_interval_and_delayed_observation_then_fresh_independent_candidate(self):
        await self.start_user_response()
        await self.acknowledge()
        self.clock.seconds = 1
        await self.finish()
        self.clock.seconds = 5
        self.assertFalse(await self.candidate())
        self.clock.seconds = 7
        self.assertFalse(await self.candidate(source=NOW))
        self.assertTrue(await self.candidate())
        self.assertEqual(len(self.creates()), 2)

    async def test_two_concurrent_candidates_only_reserve_one_response(self):
        self.provider.create_release = asyncio.Event()
        first = asyncio.create_task(self.candidate())
        await self.provider.create_started.wait()
        second = asyncio.create_task(self.candidate(message="A different wording."))
        self.provider.create_release.set()
        results = await asyncio.gather(first, second)
        self.assertEqual(results, [True, False])
        self.assertEqual(len(self.creates()), 1)

    async def test_barge_in_cancels_generic_and_old_lifecycle_cannot_clear_new_turn(self):
        self.assertTrue(await self.candidate())
        generic = self.creates()[-1]
        normal = await self.start_user_response()
        self.assertTrue(any(event["type"] == "response.cancel" for event in self.provider.events))
        await self.acknowledge(normal, "new-normal")
        await self.acknowledge(generic, "late-generic")
        self.assertEqual(self.session._current_response_id, "new-normal")
        self.assertTrue(self.session._active_responses["late-generic"].interrupted)
        await self.finish("late-generic", "cancelled")
        self.assertTrue(self.session._normal_turn_pending)
        self.assertFalse(await self.candidate())

    async def test_old_normal_done_does_not_settle_new_normal_generation(self):
        first = await self.start_user_response()
        await self.acknowledge(first, "old-normal")
        second = await self.start_user_response()
        await self.acknowledge(second, "new-normal")
        await self.finish("old-normal", "cancelled")
        self.assertTrue(self.session._normal_turn_pending)
        self.assertEqual(self.session._current_response_id, "new-normal")

    async def test_tool_work_blocks_generic_and_continuation_waits_for_parent_done(self):
        await self.start_user_response()
        await self.acknowledge()
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"status": "success"}
        self.session.tool_executor = SimpleNamespace(execute=execute)
        task = asyncio.create_task(self.session._handle_function_call_item(
            {"call_id": "call-1", "name": "example", "arguments": "{}"},
            response_id="response-1",
        ))
        await entered.wait()
        self.assertFalse(await self.candidate())
        release.set()
        await task
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(await self.candidate())
        await self.finish()
        self.assertEqual(len(self.creates()), 2)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertFalse(await self.candidate())
        await self.acknowledge(response_id="followup")
        await self.finish("followup")
        self.assertFalse(self.session._normal_turn_pending)

    async def test_cancelled_tool_and_future_sync_waiter_cannot_restart_old_turn(self):
        await self.start_user_response()
        await self.acknowledge()
        generation = self.session._turn_generation
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(*args, **kwargs):
            entered.set()
            await release.wait()
            return {"status": "success"}
        self.session.tool_executor = SimpleNamespace(execute=execute)
        task = asyncio.create_task(self.session._handle_function_call_item(
            {"call_id": "call-1", "name": "example", "arguments": "{}"},
            response_id="response-1",
        ))
        await entered.wait()
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.start_user_response()
        release.set()
        await task
        await self.finish("response-1", "cancelled")
        self.assertFalse(await self.session._request_normal_response(
            turn_generation=generation, continuation=True,
        ))
        self.assertEqual(len(self.creates()), 2)

    async def test_started_send_failure_preserves_unknown_reservation_and_chain(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=1)
        self.provider.fail_create = True
        self.assertFalse(await self.candidate())
        self.assertTrue(self.session._pending_response.submitted)
        self.provider.fail_create = False
        self.assertFalse(await self.candidate(message="Retry with other words."))
        self.assertEqual(len(self.creates()), 1)
        self.assertFalse(any(kind == "visual.interrupt.started" for kind, _ in self.app_events))

    async def test_prewrite_failure_releases_reservation_without_resetting_chain(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=1)
        original = self.session.prepare_visual_response
        self.session.prepare_visual_response = AsyncMock(side_effect=RuntimeError("prewrite fixture failure"))
        self.assertFalse(await self.candidate())
        self.assertIsNone(self.session._pending_response)
        self.session.prepare_visual_response = original
        self.assertFalse(await self.candidate(message="Retry with other words."))
        self.assertFalse(self.creates())

    async def test_correlated_rejection_releases_pending_but_old_error_does_not(self):
        original = await self.start_user_response()
        await self.session._handle_openai_event({
            "type": "error", "error": {"event_id": original["event_id"], "message": "Rejected"},
        })
        self.assertIsNone(self.session._pending_response)
        self.assertFalse(self.session._normal_turn_pending)
        current = await self.start_user_response()
        await self.session._handle_openai_event({
            "type": "error", "error": {"event_id": original["event_id"], "message": "Late"},
        })
        self.assertEqual(self.session._pending_response.request_id, current["event_id"])

    async def test_unknown_response_start_keeps_reservation_and_accepts_late_ack(self):
        request = await self.start_user_response()
        with patch("app.realtime.session.asyncio.sleep", new=AsyncMock()):
            await self.session._watch_response_start(request["event_id"])
        self.assertIsNotNone(self.session._pending_response)
        self.assertFalse(await self.candidate())
        await self.acknowledge(request)
        self.assertIsNone(self.session._pending_response)
        self.assertTrue(self.session._normal_turn_pending)

    async def test_passive_images_and_context_refresh_do_not_reset_speech_state(self):
        await self.start_user_response()
        generation = self.session._turn_generation
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=1)
        self.assertTrue(self.session._proactive_interrupt_gate.try_begin())
        visual = self.session._visual_context
        stream_id = uuid4()
        visual.activate_visual_stream(stream_id)
        frame = VisualFrameMetadata(
            visual.connection, stream_id, uuid4(), 1, "glasses", NOW, NOW, NOW, 100,
        )
        self.assertTrue(visual.offer_frame(frame))
        item = visual.reserve_create(now=NOW, monotonic_now=100)
        await self.session._handle_openai_event({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item.item_id, "transcript": "Not a spoken user turn.",
        })
        self.repository.create_message.assert_not_awaited()
        self.assertEqual(self.session._turn_generation, generation)
        self.assertFalse(self.session._proactive_interrupt_gate.try_begin())
        self.assertEqual(len(self.creates()), 1)
        self.session._context_refresh = SimpleNamespace(
            notify=AsyncMock(), stop=Mock(), close=AsyncMock(),
        )
        await self.session.notify_working_memory_updated(3)
        self.session._context_refresh.notify.assert_awaited_once_with(3)

    async def test_timed_reminder_keeps_delivery_and_completion_outcomes(self):
        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
        )
        await self.start_user_response()
        await self.acknowledge()
        reminder_id, event_id = uuid4(), uuid4()
        await self.session.speak_reminder(
            message="Take a break.", reminder_id=reminder_id, reminder_event_id=event_id,
        )
        self.assertFalse(await self.candidate())
        await self.acknowledge(response_id="reminder-response")
        await self.finish("reminder-response")
        types = [
            call.kwargs["event_type"]
            for call in self.session.reminder_repository.mark_event_delivery.await_args_list
        ]
        self.assertEqual(types, ["dispatched_to_session", "delivered_to_session"])
        self.assertNotIn(str(event_id), self.session._pending_reminder_events)

    async def test_reminder_database_wait_does_not_hold_response_lock(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def persist(**kwargs):
            if kwargs["event_type"] == "dispatched_to_session":
                entered.set()
                await release.wait()
        self.session.reminder_repository = SimpleNamespace(mark_event_delivery=AsyncMock(side_effect=persist))
        task = asyncio.create_task(self.session.speak_reminder(
            message="Break.", reminder_id=uuid4(), reminder_event_id=uuid4(),
        ))
        await entered.wait()
        await self.append()
        release.set()
        with self.assertRaises(RuntimeError):
            await task
        self.assertEqual(self.creates(), [])
        self.assertTrue(self.session._normal_turn_pending)

    async def test_prospective_delivery_keeps_callback_outcome(self):
        outcome = AsyncMock()
        self.assertTrue(await self.session.speak_prospective_interrupt(
            message="Your keys are on the desk.", event_id=uuid4(),
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
        ))
        self.assertFalse(await self.candidate())
        await self.acknowledge()
        await self.finish()
        outcome.assert_awaited_once_with("delivered", "response-1")


    async def test_run_awaits_cleanup_when_provider_exit_cancels_client_close(self):
        receive_started = asyncio.Event()
        provider_closed = asyncio.Event()
        release_close = asyncio.Event()
        client_cancelled = asyncio.Event()
        ready_sent = False

        async def receive_event():
            nonlocal ready_sent
            if not ready_sent:
                ready_sent = True
                return {"type": "session.updated"}
            receive_started.set()
            await provider_closed.wait()
            raise ConnectionClosedOK(Close(1000, ""), Close(1000, ""), True)

        async def close_provider():
            provider_closed.set()
            await release_close.wait()

        async def messages():
            # Ensure the provider receive loop is running before requesting close.
            await receive_started.wait()
            yield json.dumps({"type": "session.close"})
            raise AssertionError("client loop continued after session.close")

        original_client_loop = self.session._client_to_openai_loop

        async def client_loop():
            try:
                await original_client_loop()
            except asyncio.CancelledError:
                client_cancelled.set()
                raise

        self.provider.connect = AsyncMock()
        self.provider.receive_event = receive_event
        self.provider.close.side_effect = close_provider
        self.session.websocket.iter_text = messages
        self.session._client_to_openai_loop = client_loop
        self.session._send_session_update = AsyncMock(return_value=None)
        self.session._context_refresh = SimpleNamespace(
            initialize=lambda revision: None, stop=Mock(), close=AsyncMock(),
        )
        self.session.reminder_repository = SimpleNamespace(mark_event_delivery=AsyncMock())
        reminder_event_id = uuid4()
        self.session._pending_reminder_events[str(reminder_event_id)] = str(uuid4())
        outcome = AsyncMock()
        self.assertTrue(await self.session.speak_prospective_interrupt(
            message="Your keys are here.", event_id=uuid4(),
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
        ))
        response_watchdog = self.session._response_start_task
        expiry_task = asyncio.create_task(asyncio.Event().wait())
        self.session._working_memory_expiry_task = expiry_task
        run_task = asyncio.create_task(self.session.run())

        try:
            await asyncio.wait_for(client_cancelled.wait(), timeout=1)
            self.assertFalse(run_task.done())
            self.assertFalse(self.session._close_task.cancelled())
            self.session.websocket.close.assert_not_awaited()
            outcome.assert_not_awaited()
            release_close.set()
            await asyncio.wait_for(run_task, timeout=1)
        finally:
            release_close.set()
            await asyncio.wait_for(
                asyncio.gather(run_task, return_exceptions=True), timeout=1,
            )

        self.provider.close.assert_awaited_once()
        self.session.websocket.close.assert_awaited_once()
        self.session._context_refresh.close.assert_awaited_once()
        self.assertEqual(self.session._close_reason, "client_closed")
        self.assertIsNone(self.session._pending_response)
        self.assertIsNone(self.session._response_start_task)
        self.assertTrue(response_watchdog.cancelled())
        self.assertTrue(expiry_task.cancelled())
        outcome.assert_awaited_once_with("interrupted", None)
        self.assertEqual(self.session._pending_prospective_deliveries, {})
        self.assertEqual(self.session._pending_reminder_events, {})
        self.assertEqual(
            self.session.reminder_repository.mark_event_delivery.await_args.kwargs["event_type"],
            "interrupted",
        )
        self.assertIn(("session.closed", {"reason": "client_closed"}), self.app_events)

    async def test_cancelled_close_waiter_does_not_cancel_shared_cleanup(self):
        close_started, release_close, second_waiting = (
            asyncio.Event(), asyncio.Event(), asyncio.Event()
        )

        async def close_provider():
            close_started.set()
            await release_close.wait()

        async def close_again():
            second_waiting.set()
            await self.session.close("second_reason")

        self.provider.close.side_effect = close_provider
        first = asyncio.create_task(self.session.close("first_reason"))
        second = None
        try:
            await asyncio.wait_for(close_started.wait(), timeout=1)
            second = asyncio.create_task(close_again())
            await asyncio.wait_for(second_waiting.wait(), timeout=1)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first

            self.assertFalse(self.session._close_task.cancelled())
            self.assertFalse(second.done())
            self.assertEqual(self.session._close_reason, "first_reason")
            release_close.set()
            await asyncio.wait_for(second, timeout=1)
        finally:
            release_close.set()
            tasks = [first] + ([second] if second is not None else [])
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=1,
            )

        await self.session.close("already_finished")
        self.provider.close.assert_awaited_once()
        self.session.websocket.close.assert_awaited_once()
        self.assertEqual(
            [(kind, payload) for kind, payload in self.app_events if kind == "session.closed"],
            [("session.closed", {"reason": "first_reason"})],
        )

    async def test_close_interrupts_provider_write_without_waiting_for_admission_lock(self):
        self.provider.create_release = asyncio.Event()

        async def close_provider():
            self.provider.fail_create = True
            self.provider.create_release.set()

        self.provider.close.side_effect = close_provider
        speech = asyncio.create_task(self.candidate())
        self.addAsyncCleanup(self.cancel_test_task, speech)
        await asyncio.wait_for(self.provider.create_started.wait(), timeout=1)

        await asyncio.wait_for(self.session.close("client_stop"), timeout=1)

        self.assertFalse(await asyncio.wait_for(speech, timeout=1))
        self.provider.close.assert_awaited_once()
        self.session.websocket.close.assert_awaited_once()
        self.assertIsNone(self.session._pending_response)
        self.assertIsNone(self.session._response_start_task)
        self.assertFalse(await self.candidate())
        await self.session.close("duplicate_stop")
        self.provider.close.assert_awaited_once()

    async def test_generic_notification_failures_do_not_reject_accepted_speech(self):
        async def notify(kind, **payload):
            if kind == "visual.interrupt.started":
                await self.acknowledge(response_id="accepted-visual")
            if kind.startswith("visual.interrupt."):
                raise RuntimeError("app notification unavailable")
            await self._record_event(kind, **payload)

        self.session._send_app_event.side_effect = notify
        self.assertTrue(await self.candidate())
        self.assertEqual(self.session._current_response_id, "accepted-visual")
        self.assertFalse(self.session._active_responses["accepted-visual"].interrupted)
        self.assertFalse(await self.candidate())
        await self.finish("accepted-visual")
        self.assertIsNone(self.session._current_response_id)

    async def test_timed_reminder_notification_failures_preserve_provider_outcomes(self):
        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
        )

        async def notify(kind, **payload):
            if kind == "reminder.started":
                await self.acknowledge(response_id="accepted-reminder")
            if kind.startswith("reminder."):
                raise RuntimeError("app notification unavailable")
            await self._record_event(kind, **payload)

        self.session._send_app_event.side_effect = notify
        event_id = uuid4()
        await self.session.speak_reminder(
            message="Take a break.", reminder_id=uuid4(), reminder_event_id=event_id,
        )
        self.assertIn(str(event_id), self.session._pending_reminder_events)
        self.assertEqual(self.session._current_response_id, "accepted-reminder")
        delivery = self.session.reminder_repository.mark_event_delivery
        self.assertEqual(
            [call.kwargs["event_type"] for call in delivery.await_args_list],
            ["dispatched_to_session"],
        )
        await self.finish("accepted-reminder")
        self.assertEqual(
            [call.kwargs["event_type"] for call in delivery.await_args_list],
            ["dispatched_to_session", "delivered_to_session"],
        )
        self.assertNotIn(str(event_id), self.session._pending_reminder_events)

    async def test_prospective_notification_failures_preserve_delivery_callback(self):
        outcome = AsyncMock()

        async def notify(kind, **payload):
            if kind == "prospective.started":
                await self.acknowledge(response_id="accepted-prospective")
            if kind.startswith("prospective."):
                raise RuntimeError("app notification unavailable")
            await self._record_event(kind, **payload)

        self.session._send_app_event.side_effect = notify
        event_id = uuid4()
        self.assertTrue(await self.session.speak_prospective_interrupt(
            message="Your keys are on the desk.", event_id=event_id,
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
        ))
        self.assertIn(str(event_id), self.session._pending_prospective_deliveries)
        outcome.assert_not_awaited()
        self.assertEqual(self.session._current_response_id, "accepted-prospective")
        await self.finish("accepted-prospective")
        outcome.assert_awaited_once_with("delivered", "accepted-prospective")
        self.assertNotIn(str(event_id), self.session._pending_prospective_deliveries)

    async def test_cancelled_unsent_reminder_releases_reservation_and_reconciles_failure(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=2)
        entered, _ = self.block_reminder_dispatch()
        event_id = uuid4()
        task = asyncio.create_task(self.session.speak_reminder(
            message="Take a break.", reminder_id=uuid4(), reminder_event_id=event_id,
        ))
        self.addAsyncCleanup(self.cancel_test_task, task)
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNone(self.session._pending_response)
        self.assertIsNone(self.session._latest_response_request_id)
        self.assertIsNone(self.session._response_start_task)
        self.assertNotIn(str(event_id), self.session._pending_reminder_events)
        self.assertEqual(self.creates(), [])
        self.assertEqual(
            self.session.reminder_repository.mark_event_delivery.await_args.kwargs["metadata"],
            {"delivery_route": "realtime", "reason": "response_create_cancelled"},
        )
        self.assertTrue(await self.candidate())

    async def test_cancelled_unsent_reminder_does_not_reset_proactive_budget(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=1)
        entered, _ = self.block_reminder_dispatch()
        task = asyncio.create_task(self.session.speak_reminder(
            message="Take a break.", reminder_id=uuid4(), reminder_event_id=uuid4(),
        ))
        self.addAsyncCleanup(self.cancel_test_task, task)
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNone(self.session._pending_response)
        self.assertFalse(await self.candidate())
        self.assertEqual(self.creates(), [])

    async def test_cancelled_old_reminder_preserves_new_reservation_and_watchdog(self):
        self.session._proactive_interrupt_gate = ProactiveInterruptGate(max_chain_length=2)
        entered, _ = self.block_reminder_dispatch()
        old_event_id = uuid4()
        old = asyncio.create_task(self.session.speak_reminder(
            message="Take a break.", reminder_id=uuid4(), reminder_event_id=old_event_id,
        ))
        self.addAsyncCleanup(self.cancel_test_task, old)
        await asyncio.wait_for(entered.wait(), timeout=1)
        outcome = AsyncMock()
        new_event_id = uuid4()
        self.assertTrue(await self.session.speak_prospective_interrupt(
            message="Your keys are here.", event_id=new_event_id,
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
        ))
        new_reservation = self.session._pending_response
        new_watchdog = self.session._response_start_task

        old.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await old

        self.assertIs(self.session._pending_response, new_reservation)
        self.assertIs(self.session._response_start_task, new_watchdog)
        self.assertEqual(self.session._latest_response_request_id, new_reservation.request_id)
        self.assertFalse(self.session._proactive_interrupt_gate.try_begin())
        self.assertNotIn(str(old_event_id), self.session._pending_reminder_events)
        self.assertIn(str(new_event_id), self.session._pending_prospective_deliveries)
        outcome.assert_not_awaited()
        await self.acknowledge(response_id="new-prospective")
        await self.finish("new-prospective")
        outcome.assert_awaited_once_with("delivered", "new-prospective")

    async def test_cancelled_reminder_write_keeps_unknown_outcome_until_late_ack(self):
        self.session.reminder_repository = SimpleNamespace(
            mark_event_delivery=AsyncMock(), set_event_provider_response_id=AsyncMock(),
        )
        self.provider.create_release = asyncio.Event()
        event_id = uuid4()
        task = asyncio.create_task(self.session.speak_reminder(
            message="Take a break.", reminder_id=uuid4(), reminder_event_id=event_id,
        ))
        self.addAsyncCleanup(self.cancel_test_task, task)
        await asyncio.wait_for(self.provider.create_started.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(self.session._pending_response.submitted)
        self.assertIsNotNone(self.session._response_start_task)
        self.assertIn(str(event_id), self.session._pending_reminder_events)
        self.assertFalse(await self.candidate())
        delivery = self.session.reminder_repository.mark_event_delivery
        self.assertEqual(
            [call.kwargs["event_type"] for call in delivery.await_args_list],
            ["dispatched_to_session"],
        )
        await self.acknowledge(response_id="late-reminder")
        self.assertFalse(self.session._active_responses["late-reminder"].interrupted)
        await self.finish("late-reminder")
        self.assertEqual(
            [call.kwargs["event_type"] for call in delivery.await_args_list],
            ["dispatched_to_session", "delivered_to_session"],
        )

    async def test_cancelled_app_notification_keeps_accepted_prospective_delivery(self):
        entered, release = asyncio.Event(), asyncio.Event()
        outcome = AsyncMock()

        async def notify(kind, **payload):
            if kind == "prospective.started":
                await self.acknowledge(response_id="accepted-before-cancellation")
                entered.set()
                await release.wait()
            await self._record_event(kind, **payload)

        self.session._send_app_event.side_effect = notify
        event_id = uuid4()
        task = asyncio.create_task(self.session.speak_prospective_interrupt(
            message="Your keys are here.", event_id=event_id,
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
        ))
        self.addAsyncCleanup(self.cancel_test_task, task)
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIn(str(event_id), self.session._pending_prospective_deliveries)
        outcome.assert_not_awaited()
        await self.finish("accepted-before-cancellation")
        outcome.assert_awaited_once_with("delivered", "accepted-before-cancellation")

    async def test_prewrite_failure_still_fails_timed_reminder(self):
        self.session.reminder_repository = SimpleNamespace(mark_event_delivery=AsyncMock())
        self.session.prepare_visual_response = AsyncMock(
            side_effect=RuntimeError("prewrite fixture failure"),
        )
        event_id = uuid4()
        with self.assertRaises(RuntimeError):
            await self.session.speak_reminder(
                message="Take a break.", reminder_id=uuid4(), reminder_event_id=event_id,
            )
        self.assertIsNone(self.session._pending_response)
        self.assertIsNone(self.session._response_start_task)
        self.assertNotIn(str(event_id), self.session._pending_reminder_events)
        self.assertEqual(
            [call.kwargs["event_type"] for call in
             self.session.reminder_repository.mark_event_delivery.await_args_list],
            ["dispatched_to_session", "failed"],
        )

    async def test_ambiguous_provider_write_retains_prospective_until_close(self):
        self.provider.fail_create = True
        event_id, outcome = uuid4(), AsyncMock()
        with self.assertRaises(RuntimeError):
            await self.session.speak_prospective_interrupt(
                message="Your keys are here.", event_id=event_id,
                subject_type="visual_focus", subject_id=uuid4(), on_outcome=outcome,
            )
        self.assertTrue(self.session._pending_response.submitted)
        self.assertIsNotNone(self.session._response_start_task)
        self.assertIn(str(event_id), self.session._pending_prospective_deliveries)
        outcome.assert_not_awaited()
        await self.session.close("provider_closed")
        outcome.assert_awaited_once_with("interrupted", None)

    async def test_close_releases_pending_and_never_starts_fallback_speech(self):
        await self.candidate()
        await self.session.close("closed")
        self.assertIsNone(self.session._pending_response)
        self.assertFalse(await self.candidate())
        self.assertEqual(len(self.creates()), 1)


class GenericVisualAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_candidate_is_an_audit_not_delivered_speech_and_dedupe_stays(self):
        record = SimpleNamespace(id=uuid4())
        repository = SimpleNamespace(
            recent_visual_interrupt_exists=AsyncMock(return_value=False),
            create_visual_interrupt=AsyncMock(return_value=record),
        )
        manager = SimpleNamespace(speak_visual_interrupt=AsyncMock(return_value=False))
        service = VisualInterruptService(
            repository=repository, realtime_session_manager=manager, dedupe_seconds=10,
        )
        arguments = dict(
            owner_id="owner", session_id=uuid4(), audio_session_id="audio",
            visual_frame_id=uuid4(), source_received_at=NOW,
            message="The door is open.", reason="Visible change.", confidence=0.95,
        )
        self.assertIs(await service.maybe_interrupt(**arguments), record)
        self.assertEqual(manager.speak_visual_interrupt.await_args.kwargs["source_received_at"], NOW)
        self.assertEqual(manager.speak_visual_interrupt.await_args.kwargs["owner_id"], "owner")
        repository.recent_visual_interrupt_exists.return_value = True
        self.assertIsNone(await service.maybe_interrupt(**arguments))
        manager.speak_visual_interrupt.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
