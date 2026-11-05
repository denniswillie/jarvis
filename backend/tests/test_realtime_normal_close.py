from __future__ import annotations

import asyncio
import json
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import uvicorn
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from base64 import b64encode
from uuid import uuid4

from app.realtime.models import PendingRealtimeResponse, RealtimeResponseState, RealtimeSessionContext
from app.realtime.session import MAX_RECENT_CANCELLATIONS, RealtimeAudioSession
from app.realtime.telemetry import RealtimeTrace
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.settings import JarvisSettings


class RealtimeNormalCloseTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_close_stops_before_requesting_another_message(self) -> None:
        requested_messages = 0
        session = object.__new__(RealtimeAudioSession)
        session._closed = asyncio.Event()
        session._trace = RealtimeTrace(provider="openai", audio_session_id="fixture", jarvis_session_id=uuid4())

        async def messages():
            nonlocal requested_messages
            requested_messages += 1
            yield json.dumps({"type": "session.close"})
            requested_messages += 1
            raise AssertionError("client loop requested a message after closing")

        async def close_session(_: dict[str, object]) -> None:
            session._closed.set()

        session.websocket = SimpleNamespace(iter_text=messages)
        session._enforce_client_message_size = lambda _: None
        session._handle_client_event = close_session

        await session._client_to_openai_loop()

        self.assertEqual(requested_messages, 1)

    async def test_expected_close_after_client_shutdown_is_not_an_error(self) -> None:
        session = object.__new__(RealtimeAudioSession)
        session._closed = asyncio.Event()
        session._trace = RealtimeTrace(provider="openai", audio_session_id="fixture", jarvis_session_id=uuid4())
        session._closed.set()
        session._close_reason = "client_closed"
        session.openai = SimpleNamespace(
            receive_event=AsyncMock(
                side_effect=ConnectionClosedOK(
                    Close(1000, ""), Close(1000, ""), True
                )
            )
        )

        await session._openai_to_client_loop()

        self.assertEqual(session._close_reason, "client_closed")

    async def test_provider_normal_close_uses_nonfailure_reason(self) -> None:
        session = object.__new__(RealtimeAudioSession)
        session._closed = asyncio.Event()
        session._trace = RealtimeTrace(provider="openai", audio_session_id="fixture", jarvis_session_id=uuid4())
        session._close_reason = "unknown"
        session.openai = SimpleNamespace(
            receive_event=AsyncMock(
                side_effect=ConnectionClosedOK(
                    Close(1000, ""), Close(1000, ""), True
                )
            )
        )

        await session._openai_to_client_loop()

        self.assertEqual(session._close_reason, "openai_closed")


def _transport_session(websocket):
    session = RealtimeAudioSession(
        websocket=websocket,
        context=RealtimeSessionContext(
            audio_session_id="audio", owner_id="owner", jarvis_session_id=uuid4(),
            client_kind="rayban_audio", session=SimpleNamespace(),
        ),
        settings=JarvisSettings(_env_file=None),
        repository=SimpleNamespace(
            create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[]),
        ),
        tool_modules=(), context_providers=(),
    )
    # Isolate lifecycle behavior from database context building and live providers.
    session._send_session_update = AsyncMock(return_value=None)
    return session


class RealtimeSessionShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app_events = []
        self.ready = asyncio.Event()
        self.client_input = asyncio.Queue()
        self.provider_input = asyncio.Queue()
        self.provider_input.put_nowait({"type": "session.updated"})

        async def messages():
            while True:
                event = await self.client_input.get()
                if event is None:
                    return
                if isinstance(event, Exception):
                    raise event
                yield event

        async def send_text(raw):
            event = json.loads(raw)
            self.app_events.append(event)
            if event["type"] == "session.created":
                self.ready.set()

        async def receive_event():
            event = await self.provider_input.get()
            if isinstance(event, Exception):
                raise event
            return event

        self.websocket = SimpleNamespace(
            iter_text=messages, send_text=AsyncMock(side_effect=send_text), close=AsyncMock(),
        )
        self.session = _transport_session(self.websocket)
        self.provider = SimpleNamespace(
            connect=AsyncMock(), send_event=AsyncMock(return_value=True),
            receive_event=AsyncMock(side_effect=receive_event), close=AsyncMock(),
        )
        self.session.openai = self.provider
        self.run_task = None

    async def asyncTearDown(self):
        await asyncio.wait_for(self.session.close("test_complete"), 1)
        if self.run_task is not None:
            await asyncio.wait_for(asyncio.gather(self.run_task, return_exceptions=True), 1)

    async def start_session(self):
        self.run_task = asyncio.create_task(self.session.run())
        await asyncio.wait_for(self.ready.wait(), 1)

    async def test_departure_during_audio_output_settles_all_relays_without_errors(self):
        with self.assertNoLogs("app.realtime.session", level="ERROR"):
            await self.start_session()
            self.websocket.send_text.side_effect = WebSocketDisconnect(code=1006)
            self.provider_input.put_nowait({
                "type": "response.output_audio.delta", "delta": "AA==", "response_id": "response",
            })
            await asyncio.wait_for(self.run_task, 1)

        self.assertEqual(self.session._close_reason, "client_disconnect")
        self.assertTrue(self.session._client_receive_task.done())
        self.assertTrue(self.session._provider_receive_task.done())
        self.provider.close.assert_awaited_once()
        self.websocket.close.assert_not_awaited()

    async def test_departure_under_admission_lock_does_not_await_cleanup(self):
        await self.start_session()
        self.websocket.send_text.side_effect = WebSocketDisconnect()
        async with self.session._response_admission_lock:
            await asyncio.wait_for(self.session._send_app_event("memory.updated"), 1)
            self.assertTrue(self.session._closed.is_set())
        await asyncio.wait_for(self.run_task, 1)
        self.assertEqual(self.session._close_reason, "client_disconnect")

    async def test_client_iterator_exhaustion_marks_the_peer_closed(self):
        with self.assertNoLogs("app.realtime.session", level="ERROR"):
            await self.start_session()
            self.client_input.put_nowait(None)
            await asyncio.wait_for(self.run_task, 1)
        self.assertEqual(self.session._close_reason, "client_disconnect")
        self.websocket.close.assert_not_awaited()

    async def test_relay_failure_cancels_sibling_and_duration_timer(self):
        timer_started, timer_stopped = asyncio.Event(), asyncio.Event()

        async def timer():
            timer_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                timer_stopped.set()

        self.session._max_duration_timer = timer
        with self.assertLogs("app.realtime.session", level="ERROR") as logs:
            await self.start_session()
            await asyncio.wait_for(timer_started.wait(), 1)
            self.provider_input.put_nowait(RuntimeError("PRIVATE_PROVIDER_FAILURE"))
            await asyncio.wait_for(self.run_task, 1)

        self.assertTrue(timer_stopped.is_set())
        self.assertTrue(self.session._client_receive_task.cancelled())
        self.assertEqual(self.session._close_reason, "realtime_session_failed")
        self.assertIn("realtime_session_failed", str(self.app_events))
        self.assertNotIn("PRIVATE_PROVIDER_FAILURE", str(logs.output) + str(self.app_events))

    async def test_abnormal_provider_departure_is_still_an_actionable_failure(self):
        with self.assertLogs("app.realtime.session", level="ERROR"):
            await self.start_session()
            self.provider_input.put_nowait(ConnectionClosedError(
                Close(1011, "provider failure"), None, None,
            ))
            await asyncio.wait_for(self.run_task, 1)
        self.assertEqual(self.session._close_reason, "realtime_session_failed")

    async def test_setup_failure_preserves_specific_code_and_redaction(self):
        self.provider.connect.side_effect = RuntimeError("PRIVATE_SETUP_FAILURE")
        with self.assertLogs("app.realtime.session", level="ERROR") as logs:
            await self.session.run()
        self.assertEqual(self.session._close_reason, "openai_realtime_connect_failed")
        self.assertIn("openai_realtime_connect_failed", str(self.app_events))
        self.assertNotIn("PRIVATE_SETUP_FAILURE", str(logs.output) + str(self.app_events))
        self.provider.close.assert_awaited_once()

    async def test_close_cancels_pending_provider_setup(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def connect():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        self.provider.connect.side_effect = connect
        self.run_task = asyncio.create_task(self.session.run())
        await asyncio.wait_for(entered.wait(), 1)
        with self.assertNoLogs("app.realtime.session", level="ERROR"):
            await self.session.close("client_closed")
            await asyncio.wait_for(self.run_task, 1)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(any(event["type"] == "session.created" for event in self.app_events))
        self.assertEqual(self.session._close_reason, "client_closed")

    async def test_close_waits_for_late_handshake_before_releasing_provider(self):
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        socket = SimpleNamespace(close=AsyncMock())
        provider = OpenAIRealtimeConnection(
            settings=JarvisSettings(_env_file=None), owner_id="shutdown-test",
        )
        original_close = provider.close
        provider.close = AsyncMock(wraps=original_close)
        self.session.openai = provider

        async def connect(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model a handshake that finishes acquiring its socket as stop
                # arrives. Cleanup must wait for that acquisition, then close it.
                cancelled.set()
                await release.wait()
                return socket

        close_task = None
        with patch(
            "app.realtime.openai_client.websockets.connect", side_effect=connect,
        ):
            try:
                with self.assertNoLogs("app.realtime.session", level="ERROR"):
                    self.run_task = asyncio.create_task(self.session.run())
                    await asyncio.wait_for(entered.wait(), 1)
                    close_task = asyncio.create_task(self.session.close("client_closed"))
                    await asyncio.wait_for(cancelled.wait(), 1)
                    provider.close.assert_not_awaited()
                    self.assertFalse(close_task.done())
                    release.set()
                    await asyncio.wait_for(asyncio.gather(close_task, self.run_task), 1)

                provider.close.assert_awaited_once()
                socket.close.assert_awaited_once()
                self.assertIsNone(provider._websocket)
                self.assertIsNone(self.session._provider_connect_task)
                self.session._send_session_update.assert_not_awaited()
                self.assertFalse(any(event["type"] == "session.created" for event in self.app_events))
                self.assertEqual(self.session._close_reason, "client_closed")
            finally:
                release.set()
                await self.session.close("test_cleanup")
                if close_task is not None:
                    await asyncio.gather(close_task, return_exceptions=True)
                if self.run_task is not None:
                    await asyncio.gather(self.run_task, return_exceptions=True)
                await original_close()

    async def test_closed_session_never_starts_provider_setup(self):
        await self.session.close("client_closed")
        await self.session._initialize_openai_session()
        self.provider.connect.assert_not_awaited()
        self.session._send_session_update.assert_not_awaited()

    async def test_refresh_stops_before_provider_close_and_does_not_retry(self):
        entered, provider_closed = asyncio.Event(), asyncio.Event()
        self.session._realtime_context_ready.set()
        self.session._build_assistant_context = AsyncMock(return_value=SimpleNamespace(
            working_memory={"available": True, "session": None, "owner": None},
        ))

        async def stalled_send(context):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await provider_closed.wait()
                raise RuntimeError("transport retired during refresh")

        async def close_provider():
            coordinator = self.session._context_refresh
            self.assertTrue(coordinator._closed)
            self.assertGreater(coordinator._task.cancelling(), 0)
            provider_closed.set()

        self.session._send_complete_session_update = AsyncMock(side_effect=stalled_send)
        self.provider.close.side_effect = close_provider
        try:
            with self.assertNoLogs("app.realtime.context_refresh", level="ERROR"):
                await self.session._context_refresh.invalidate()
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.wait_for(self.session.close("client_closed"), 1)
            self.session._send_complete_session_update.assert_awaited_once()
            self.assertIsNone(self.session._context_refresh._task)
        finally:
            provider_closed.set()


class RealtimeSocketIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_peer_departure_settles_audio_session(self):
        sessions = []
        finished = asyncio.Event()

        async def app(scope, receive, send):
            websocket = WebSocket(scope, receive, send)
            await websocket.accept()
            session = _transport_session(websocket)
            incoming = asyncio.Queue()
            incoming.put_nowait({"type": "session.updated"})
            incoming.put_nowait({"type": "response.output_audio.delta", "delta": "AA=="})
            session.openai = SimpleNamespace(
                connect=AsyncMock(), receive_event=incoming.get,
                send_event=AsyncMock(return_value=True), close=AsyncMock(),
            )
            sessions.append(session)
            try:
                await session.run()
            finally:
                finished.set()

        with self.assertNoLogs("app.realtime.session", level="ERROR"):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
                server = uvicorn.Server(uvicorn.Config(
                    app, log_config=None, access_log=False, lifespan="off", ws="auto",
                ))
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                try:
                    async with asyncio.timeout(5):
                        while not server.started:
                            if server_task.done():
                                await server_task
                                self.fail("WebSocket test server stopped before startup")
                            await asyncio.sleep(0.01)
                        async with websockets.connect(f"ws://127.0.0.1:{port}") as peer:
                            self.assertEqual(json.loads(await peer.recv())["type"], "session.created")
                            self.assertEqual(json.loads(await peer.recv())["type"], "audio.output.delta")
                        await finished.wait()
                finally:
                    server.should_exit = True
                    await asyncio.wait_for(server_task, 5)
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]._close_reason, "client_disconnect")
        sessions[0].openai.close.assert_awaited_once()


class RealtimeCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.session = RealtimeAudioSession(
            websocket=SimpleNamespace(send_text=AsyncMock(), close=AsyncMock()),
            context=RealtimeSessionContext(
                audio_session_id="audio", owner_id="owner", jarvis_session_id=uuid4(),
                client_kind="rayban_audio", session=SimpleNamespace(),
            ),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(
                create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[]),
            ),
            tool_modules=(), context_providers=(),
        )
        self.session.openai = self.provider
        self.session._realtime_context_ready.set()
        self.session._send_app_event = AsyncMock()

    async def asyncTearDown(self):
        await self.session.close("test_complete")

    def activate_response(self):
        state = RealtimeResponseState(
            response_id="old-response", kind="normal_user_turn",
            metadata={"jarvis_request_id": "old-create", "jarvis_turn_generation": "0"},
        )
        self.session._active_responses[state.response_id] = state
        self.session._current_response_id = state.response_id
        self.session._latest_response_request_id = "old-create"
        return state

    def events(self, event_type):
        return [
            call.args[0] for call in self.provider.send_event.await_args_list
            if isinstance(call.args[0], dict) and call.args[0]["type"] == event_type
        ]

    async def append_audio(self):
        await self.session._handle_client_event({
            "type": "audio.input.append", "audio_base64": b64encode(b"\x01\x00" * 240).decode(),
        })

    async def test_idle_and_unsubmitted_preparation_do_not_cancel_provider(self):
        await self.session._cancel_response_locked()
        self.session._pending_response = PendingRealtimeResponse("not-sent", "normal_user_turn", 0)
        await self.session._cancel_response_locked()
        self.provider.send_event.assert_not_awaited()

    async def test_submitted_pending_request_has_correlated_default_cancel(self):
        self.session._pending_response = PendingRealtimeResponse(
            "pending-create", "normal_user_turn", 0, submitted=True,
        )
        await self.session._cancel_response_locked()
        event = self.events("response.cancel")[0]
        self.assertIn("event_id", event)
        self.assertNotIn("response_id", event)
        delivery = SimpleNamespace(settle_response=Mock(), close=AsyncMock())
        self.session._visual_delivery = delivery
        delivery.handle_provider_event = lambda payload: False
        await self.session._handle_openai_event({
            "type": "error", "error": {"event_id": event["event_id"], "code": "response_cancel_not_active"},
        })
        delivery.settle_response.assert_called_once_with(request_id="pending-create", response_id=None)
        self.session._send_app_event.assert_not_awaited()

    async def test_completion_race_preserves_new_audio_turn_and_truncation(self):
        state = self.activate_response()
        await self.append_audio()
        cancel = self.events("response.cancel")[0]
        self.assertEqual(cancel["response_id"], "old-response")
        self.assertTrue(state.interrupted)
        generation = self.session._turn_generation
        await self.session._handle_openai_event({
            "type": "response.done", "response": {"id": "old-response", "status": "completed"},
        })
        self.session._send_app_event.reset_mock()
        await self.session._handle_openai_event({
            "type": "error",
            "error": {
                "code": "response_cancel_not_active", "event_id": cancel["event_id"],
                "message": "PRIVATE_PROVIDER_MESSAGE",
            },
        })
        self.assertEqual(self.session._turn_generation, generation)
        self.assertTrue(self.session._normal_turn_pending)
        self.assertTrue(self.session._user_audio_pending)
        self.assertEqual(len(self.events("input_audio_buffer.append")), 1)
        self.session._send_app_event.assert_not_awaited()
        await self.session._handle_client_event({
            "type": "playback.truncated", "item_id": "old-output", "content_index": 0, "audio_end_ms": 120,
        })
        self.assertEqual(self.events("conversation.item.truncate"), [{
            "type": "conversation.item.truncate", "item_id": "old-output", "content_index": 0, "audio_end_ms": 120,
        }])
        await self.session._handle_client_event({"type": "audio.input.commit"})
        self.assertEqual(len(self.events("input_audio_buffer.commit")), 1)

    async def test_only_matching_cancel_code_and_request_are_benign(self):
        for code, correlation in (
            ("invalid_request_error", "matching"),
            ("response_cancel_not_active", "unknown"),
            ("response_cancel_not_active", "missing"),
            ("response_cancel_not_active", "top_level_only"),
        ):
            with self.subTest(code=code, correlation=correlation):
                await self.session._send_response_cancel_locked(response_id="old", response_request_id="create")
                request_id = self.events("response.cancel")[-1]["event_id"]
                error = {"code": code, "message": "PRIVATE_SECRET_AND_IMAGE"}
                if correlation in {"matching", "unknown"}:
                    error["event_id"] = request_id if correlation == "matching" else "unknown"
                payload = {"type": "error", "error": error}
                if correlation == "top_level_only":
                    payload["event_id"] = request_id
                self.session._send_app_event.reset_mock()
                with self.assertLogs("app.realtime.session", level="WARNING") as logs:
                    await self.session._handle_openai_event(payload)
                self.session._send_app_event.assert_awaited_once()
                args = self.session._send_app_event.await_args
                self.assertEqual(args.args[0], "error")
                self.assertEqual(args.kwargs["code"], "openai_realtime_error")
                self.assertNotIn("PRIVATE_SECRET_AND_IMAGE", str(args) + str(logs.output))

    async def test_late_cancel_error_cannot_settle_a_new_response(self):
        self.activate_response()
        await self.append_audio()
        cancel_id = self.events("response.cancel")[0]["event_id"]
        newer = PendingRealtimeResponse("new-create", "normal_user_turn", self.session._turn_generation, submitted=True)
        self.session._pending_response = newer
        self.session._latest_response_request_id = newer.request_id
        self.session._audio_commit_request = ("new-commit", newer.turn_generation)
        await self.session._handle_openai_event({
            "type": "error", "error": {"code": "response_cancel_not_active", "event_id": cancel_id},
        })
        self.assertIs(self.session._pending_response, newer)
        self.assertEqual(self.session._audio_commit_request, ("new-commit", newer.turn_generation))
        self.assertEqual(self.session._latest_response_request_id, newer.request_id)

    async def test_late_created_response_uses_the_same_cancel_correlation(self):
        await self.session._register_response({
            "id": "late-response", "metadata": {
                "jarvis_request_id": "old-create", "jarvis_turn_generation": "-1",
            },
        })
        cancel = self.events("response.cancel")[0]
        self.assertEqual(cancel["response_id"], "late-response")
        await self.session._handle_openai_event({
            "type": "error", "error": {"code": "response_cancel_not_active", "event_id": cancel["event_id"]},
        })
        self.session._send_app_event.assert_not_awaited()

    async def test_cancel_tracking_is_bounded_and_close_ignores_late_provider_events(self):
        for _ in range(MAX_RECENT_CANCELLATIONS + 3):
            await self.session._send_response_cancel_locked(response_id="old", response_request_id="create")
        self.assertEqual(len(self.session._recent_cancellations), MAX_RECENT_CANCELLATIONS)
        await asyncio.gather(self.session.close("client_closed"), self.session.close("client_closed"))
        self.provider.close.assert_awaited_once()
        self.assertEqual(self.session._recent_cancellations, {})
        self.session._send_app_event.reset_mock()
        with self.assertNoLogs("app.realtime.session", level="WARNING"):
            await self.session._handle_openai_event({
                "type": "error", "error": {
                    "event_id": "retired-image-request", "code": "invalid_value",
                    "message": "PRIVATE_BASE64_AND_AUTH",
                },
            })
            await self.session._handle_openai_event({
                "type": "response.created", "response": {"id": "late"},
            })
        self.session._send_app_event.assert_not_awaited()
        self.assertNotIn("late", self.session._active_responses)

if __name__ == "__main__":
    unittest.main()
