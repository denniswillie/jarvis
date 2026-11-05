"""P0 controlled evidence: handler delay is not network-arrival or hearing evidence."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.session import RealtimeAudioSession
from app.realtime.telemetry import RealtimeTrace


def trace():
    return RealtimeTrace(provider="openai", audio_session_id="p0-fixture", jarvis_session_id=uuid4())


def rows(logs):
    return [json.loads(record.getMessage().split("realtime_trace ", 1)[1]) for record in logs.records]


class ControlTimingTests(unittest.TestCase):
    def test_clock_boundaries_and_content_redaction(self):
        recorder = trace()
        payload = dict(type="response.created", response={"id": "r1", "text": "PRIVATE"},
                       event_id="e1", transcript="PRIVATE", arguments="PRIVATE", audio="PRIVATE")
        with patch("app.realtime.telemetry.time.monotonic", side_effect=[10, 10, 10.25, 10.25]):
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                with recorder.control_handler(payload, origin="provider_control", dequeued_at=9.995,
                                              receive_started_at=9):
                    pass
        start, finish = rows(logs)
        self.assertEqual(start["dispatch_id"], finish["dispatch_id"])
        self.assertEqual(start["response_id"], "r1")
        self.assertAlmostEqual(finish["duration_ms"], 250)
        self.assertAlmostEqual(start["decode_dispatch_ms"], 5)
        self.assertAlmostEqual(start["receive_wait_ms"], 995)
        self.assertEqual(start["boundary"], "application_dequeue_not_wire_arrival")
        self.assertNotIn("PRIVATE", str(rows(logs)))
        self.assertNotIn("heard", str(rows(logs)))

    def test_fast_media_does_not_create_per_chunk_logs(self):
        with patch("app.realtime.telemetry.time.monotonic", side_effect=[10, 10.001]):
            with self.assertNoLogs("app.realtime.telemetry", "INFO"):
                with trace().control_handler({"type": "response.output_audio.delta", "delta": "PRIVATE"},
                                             origin="provider_control", dequeued_at=10):
                    pass

    def test_slow_media_is_measured_without_payload(self):
        with patch("app.realtime.telemetry.time.monotonic", side_effect=[10, 10.3, 10.3]):
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                with trace().control_handler({"type": "audio.input.append", "audio_base64": "PRIVATE"},
                                             origin="client_control", dequeued_at=10):
                    pass
        result = rows(logs)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["duration_ms"], 300)
        self.assertNotIn("PRIVATE", str(result))

    def test_failure_and_cancellation_propagate_without_exception_text(self):
        for error, status in ((RuntimeError("PRIVATE"), "failed"), (asyncio.CancelledError(), "cancelled")):
            with self.subTest(status=status), self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                with self.assertRaises(type(error)):
                    with trace().control_handler({"type": "response.cancel"}, origin="client_control",
                                                 dequeued_at=RealtimeTrace.control_clock()):
                        raise error
            result = rows(logs)
            self.assertEqual(result[-1]["status"], status)
            self.assertEqual(result[-1]["error_type"], type(error).__name__)
            self.assertNotIn("PRIVATE", str(result))


class SerialControlEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_loop_waits_for_handler_before_next_dequeue(self):
        entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        events = [{"type": "response.function_call_arguments.done", "event_id": "tool"},
                  {"type": "input_audio_buffer.speech_started", "event_id": "barge-in"}]

        async def handler(event):
            if event["event_id"] == "tool":
                entered.set()
                await release.wait()
            else:
                closed.set()

        receive = AsyncMock(side_effect=events)
        session = SimpleNamespace(_closed=closed, _trace=trace(), openai=SimpleNamespace(receive_event=receive),
                                  _handle_openai_event=handler)
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            task = asyncio.create_task(RealtimeAudioSession._openai_to_client_loop(session))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                await asyncio.sleep(0.03)
                self.assertEqual(receive.await_count, 1)
                release.set()
                await asyncio.wait_for(task, 1)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        evidence = rows(logs)
        finished = next(row for row in evidence if row["event_id"] == "tool"
                        and row["event"] == "control_handler_finished")
        self.assertGreaterEqual(finished["duration_ms"], 20)
        later = next(row for row in evidence if row["event_id"] == "barge-in"
                     and row["event"] == "control_handler_started")
        self.assertGreaterEqual(later["dequeued_monotonic_s"], finished["monotonic_s"])
        self.assertEqual(receive.await_count, 2)

    async def test_client_loop_keeps_error_and_close_semantics(self):
        closed = asyncio.Event()

        async def messages():
            yield json.dumps({"type": "response.cancel", "client_event_id": "cancel-1"})

        async def handler(event):
            self.assertEqual(event["type"], "response.cancel")
            closed.set()

        session = SimpleNamespace(_closed=closed, _trace=trace(), websocket=SimpleNamespace(iter_text=messages),
            _enforce_client_message_size=lambda raw: None, _handle_client_event=handler)
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await RealtimeAudioSession._client_to_openai_loop(session)
        self.assertEqual([row["event"] for row in rows(logs)],
                         ["control_handler_started", "control_handler_finished"])
        self.assertTrue(closed.is_set())
