from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from fastapi import WebSocketDisconnect

from app.realtime.socket_sender import WebSocketEventSender
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.visual_memory.latest_frame_buffer import LatestVisualFrameBuffer
from app.visual_memory.pipeline import _best_effort_send
from app.visual_memory.router import visual_realtime_websocket
from app.visual_memory.services import VisualMemoryServices
from app.visual_memory.storage import VisualFrameStorageClient
from app.visual_memory.validation import VisualFrameValidationError


class VisualSocketSenderTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_and_json_share_the_send_and_close_lock(self):
        entered, release = asyncio.Event(), asyncio.Event()
        events = []

        async def send_text(payload):
            events.append(("text", payload))
            entered.set()
            await release.wait()

        socket = SimpleNamespace(
            send_text=AsyncMock(side_effect=send_text),
            send_json=AsyncMock(side_effect=lambda payload: events.append(("json", payload))),
            close=AsyncMock(side_effect=lambda **kwargs: events.append(("close", kwargs))),
        )
        sender = WebSocketEventSender(socket)
        text_task = asyncio.create_task(sender.send_text('{"type":"audio.output.delta"}'))
        await asyncio.wait_for(entered.wait(), 1)
        json_task = asyncio.create_task(sender.send_event({"type": "diagnostic"}))
        close_task = asyncio.create_task(sender.close())
        release.set()
        await asyncio.wait_for(asyncio.gather(text_task, json_task, close_task), 1)
        await sender.send_text("too late")

        self.assertEqual([event[0] for event in events], ["text", "json", "close"])
        socket.send_text.assert_awaited_once_with('{"type":"audio.output.delta"}')

    async def test_text_send_retires_expected_departures_but_not_arbitrary_errors(self):
        for error in (
            WebSocketDisconnect(code=1006),
            RuntimeError('Cannot call "send" once a close message has been sent.'),
        ):
            with self.subTest(error_type=type(error).__name__):
                socket = SimpleNamespace(send_text=AsyncMock(side_effect=error), close=AsyncMock())
                retired = Mock()
                sender = WebSocketEventSender(socket, on_closed=retired)
                await sender.send_text("first")
                await sender.send_text("later")
                await sender.close()
                retired.assert_called_once_with()
                socket.send_text.assert_awaited_once_with("first")
                socket.close.assert_not_awaited()

        sender = WebSocketEventSender(SimpleNamespace(
            send_text=AsyncMock(side_effect=RuntimeError("unexpected text send bug")),
        ))
        with self.assertRaisesRegex(RuntimeError, "unexpected text send bug"):
            await sender.send_text("first")
        self.assertFalse(sender.closed)

    async def test_one_departure_retires_concurrent_diagnostics_and_close(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def departing_send(payload):
            entered.set()
            await release.wait()
            raise WebSocketDisconnect(code=1006)

        socket = SimpleNamespace(send_json=AsyncMock(side_effect=departing_send), close=AsyncMock())
        retired = Mock()
        sender = WebSocketEventSender(socket, on_closed=retired)
        first = asyncio.create_task(sender.send_event({"type": "first"}))
        await asyncio.wait_for(entered.wait(), 1)
        later = asyncio.create_task(sender.send_event({"type": "later"}))
        closing = asyncio.create_task(sender.close())
        release.set()
        await asyncio.wait_for(asyncio.gather(first, later, closing), 1)
        await sender.close()
        sender.mark_closed()
        self.assertTrue(sender.closed)
        self.assertEqual(socket.send_json.await_count, 1)
        socket.close.assert_not_awaited()
        retired.assert_called_once_with()

    async def test_only_known_closed_runtime_errors_are_benign(self):
        for error in (
            RuntimeError('Cannot call "send" once a close message has been sent.'),
            RuntimeError("Unexpected ASGI message 'websocket.send', after sending 'websocket.close' or response already completed."),
            OSError("peer left"),
        ):
            with self.subTest(error_type=type(error).__name__):
                socket = SimpleNamespace(send_json=AsyncMock(side_effect=error))
                sender = WebSocketEventSender(socket)
                await sender.send_event({"type": "first"})
                await sender.send_event({"type": "later"})
                self.assertTrue(sender.closed)
                self.assertEqual(socket.send_json.await_count, 1)
        socket = SimpleNamespace(send_json=AsyncMock(side_effect=RuntimeError("serializer bug")))
        sender = WebSocketEventSender(socket)
        with self.assertRaisesRegex(RuntimeError, "serializer bug"):
            await sender.send_event({"type": "diagnostic"})
        self.assertFalse(sender.closed)

    async def test_close_is_serialized_idempotent_and_keeps_unexpected_errors_visible(self):
        socket = SimpleNamespace(close=AsyncMock(), send_json=AsyncMock())
        sender = WebSocketEventSender(socket)
        await asyncio.gather(sender.close(), sender.close())
        await sender.send_event({"type": "too_late"})
        socket.close.assert_awaited_once_with(code=1000, reason="")
        socket.send_json.assert_not_awaited()

        socket.close = AsyncMock(side_effect=RuntimeError("unexpected close bug"))
        sender = WebSocketEventSender(socket)
        with self.assertRaisesRegex(RuntimeError, "unexpected close bug"):
            await sender.close()

    async def test_diagnostic_failures_do_not_own_analysis_results(self):
        with self.assertNoLogs("app.visual_memory.pipeline"):
            await _best_effort_send(AsyncMock(side_effect=WebSocketDisconnect()), {"type": "done"})
        with self.assertLogs("app.visual_memory.pipeline", level="WARNING") as logs:
            await _best_effort_send(
                AsyncMock(side_effect=RuntimeError("PRIVATE_MEDIA")), {"type": "done"},
            )
        self.assertIn("RuntimeError", str(logs.output))
        self.assertNotIn("PRIVATE_MEDIA", str(logs.output))
        with self.assertRaises(asyncio.CancelledError):
            await _best_effort_send(AsyncMock(side_effect=asyncio.CancelledError()), {})


class _IngressHarness:
    """Mock external persistence; exercise the real router and transport boundary."""

    def __init__(self, *, departure=None, failure=None):
        now = datetime.now(UTC)
        self.record = SimpleNamespace(id=uuid4(), raw_image_expires_at=now + timedelta(minutes=5))
        self.frame = SimpleNamespace(frame_sequence=1, image_bytes=b"synthetic-frame", content_type="image/jpeg", source="glasses")
        self.repository = SimpleNamespace(
            create_pending_frame=AsyncMock(return_value=self.record),
            mark_frame_available=AsyncMock(return_value=self.record),
            mark_frame_failed=AsyncMock(),
        )
        self.storage = Mock(spec=VisualFrameStorageClient)
        self.storage.bucket = "files"
        self.storage.build_storage_path.return_value = "synthetic.jpg"
        self.storage.upload_frame = AsyncMock(return_value=SimpleNamespace(bucket="files", path="synthetic.jpg"))
        self.storage.delete_objects = AsyncMock()
        self.window = Mock(spec=RecentVisualFrameWindow)
        self.window.owns_stream.return_value = True
        self.window.put.return_value = True
        self.services = Mock(spec=VisualMemoryServices)
        self.services.pipeline = SimpleNamespace(enqueue=AsyncMock())
        self.session_repository = SimpleNamespace(get_session=AsyncMock(return_value=SimpleNamespace(status="active")))
        self.validator = Mock(return_value=self.frame)
        self.events = []
        self.departure = departure
        self.socket = SimpleNamespace(
            accept=AsyncMock(), close=AsyncMock(),
            receive_text=AsyncMock(side_effect=[json.dumps({"frame_sequence": 1}), WebSocketDisconnect()]),
            send_json=AsyncMock(side_effect=self._send),
            headers={"authorization": "Bearer fake-test-token"},
            query_params={"session_id": str(uuid4()), "visual_session_id": str(uuid4())},
            app=SimpleNamespace(state=SimpleNamespace(runtime_services=SimpleNamespace(
                db_pool=object(), visual_frame_storage=self.storage,
                latest_visual_frame_buffer=Mock(spec=LatestVisualFrameBuffer),
                sensory_frame_window=self.window, visual_memory_services=self.services,
                domain_event_bus=SimpleNamespace(publish=AsyncMock()),
            ))),
        )
        if failure == "validation":
            self.validator.side_effect = VisualFrameValidationError("invalid_frame", "Invalid synthetic frame.")
        elif failure == "upload":
            self.storage.upload_frame.side_effect = RuntimeError("upload failed")
        elif failure == "database":
            self.repository.mark_frame_available.side_effect = RuntimeError("database failed")
        elif failure == "session":
            self.session_repository.get_session.return_value = None

    async def _send(self, payload):
        self.events.append(payload)
        if self.departure is not None and payload["type"] in {"visual.frame.accepted", "error"}:
            raise self.departure

    async def run(self):
        prefix = "app.visual_memory.router."
        with ExitStack() as stack:
            for name, value in (
                ("get_settings", lambda: SimpleNamespace(
                    PHONE_CAMERA_DEBUG_ENABLED=False,
                    JARVIS_REALTIME_MAX_CLIENT_MESSAGE_BYTES=10000,
                    VISUAL_RAW_FRAME_TTL_SECONDS=300,
                )),
                ("authenticate_static_bearer", lambda **kwargs: SimpleNamespace(owner_id="owner")),
                ("VisualFrameRepository", lambda pool: self.repository),
                ("SessionRepository", lambda pool: self.session_repository),
                ("validate_visual_frame_event", self.validator),
                ("buffered_frame_from_record", lambda record: record),
            ):
                stack.enter_context(patch(prefix + name, value))
            stack.enter_context(patch.object(SensoryVisualFrame, "from_buffered_frame", return_value=object()))
            await visual_realtime_websocket(self.socket)


class VisualSocketIngressTests(unittest.IsolatedAsyncioTestCase):
    async def test_ack_departure_preserves_accepted_storage_and_analysis(self):
        for departure in (
            WebSocketDisconnect(),
            RuntimeError('Cannot call "send" once a close message has been sent.'),
        ):
            with self.subTest(error_type=type(departure).__name__):
                harness = _IngressHarness(departure=departure)
                with self.assertNoLogs("app.visual_memory.router", level="ERROR"):
                    await harness.run()
                harness.repository.mark_frame_available.assert_awaited_once()
                harness.storage.delete_objects.assert_not_awaited()
                harness.repository.mark_frame_failed.assert_not_awaited()
                harness.services.pipeline.enqueue.assert_awaited_once()
                harness.window.end_stream.assert_called_once()
                self.assertEqual([event["type"] for event in harness.events], [
                    "visual.session.created", "visual.frame.accepted",
                ])
                # The queued analysis receives the same already-retired sender.
                diagnostic = harness.services.pipeline.enqueue.call_args.kwargs["send_diagnostic"]
                await diagnostic({"type": "visual.frame.analyzed"})
                self.assertEqual(harness.socket.send_json.await_count, 2)

    async def test_unexpected_ack_bug_is_not_relabelled_as_ingestion_failure(self):
        harness = _IngressHarness(departure=RuntimeError("unexpected sender bug"))
        with self.assertRaisesRegex(RuntimeError, "unexpected sender bug"):
            await harness.run()
        harness.storage.delete_objects.assert_not_awaited()
        harness.repository.mark_frame_available.assert_awaited_once()
        self.assertNotIn("visual.frame.dropped", [event["type"] for event in harness.events])

    async def test_real_ingestion_failures_retain_cleanup_and_classification(self):
        for failure in ("validation", "upload", "database"):
            with self.subTest(failure=failure):
                harness = _IngressHarness(failure=failure)
                if failure == "validation":
                    await harness.run()
                else:
                    with self.assertLogs("app.visual_memory.router", level="ERROR"):
                        await harness.run()
                dropped = [event for event in harness.events if event["type"] == "visual.frame.dropped"]
                self.assertEqual(len(dropped), 1)
                self.assertEqual(dropped[0]["code"], "invalid_frame" if failure == "validation" else "visual_frame_create_failed")
                harness.services.pipeline.enqueue.assert_not_awaited()
                self.assertEqual(harness.storage.delete_objects.await_count, int(failure == "database"))

    async def test_departure_while_reporting_session_error_does_not_close_twice(self):
        harness = _IngressHarness(failure="session", departure=WebSocketDisconnect())
        await harness.run()
        self.assertEqual(harness.events[0]["code"], "session_not_found")
        harness.socket.close.assert_not_awaited()
        harness.window.end_stream.assert_not_called()
