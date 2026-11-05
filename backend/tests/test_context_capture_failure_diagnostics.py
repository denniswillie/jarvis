"""Offline capture-failure classification; no database, provider or audio I/O."""

import asyncio
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch
from uuid import uuid4

from app.tasks.context_capture import SessionContextCapture
from app.tasks.models import TaskError


class CaptureFailureDiagnosticsTests(IsolatedAsyncioTestCase):
    def capture(self, append, *, timeout=0.2):
        return SessionContextCapture(SimpleNamespace(append=append),
            owner_id="PRIVATE_OWNER", session_id=uuid4(), provider="openai",
            audio_session_id="PRIVATE_CONNECTION", write_timeout=timeout)

    async def failure(self, error=None, *, stall=False):
        async def append(owner, event):
            if event.kind == "context_gap":
                return None
            if stall:
                await asyncio.Event().wait()
            raise error

        capture = self.capture(append, timeout=0.01 if stall else 0.2)
        with self.assertLogs("app.tasks.context_capture", "WARNING") as logs:
            capture.input_final("PRIVATE_INPUT", "PRIVATE_SPEECH sk-private-credential")
            await asyncio.wait_for(capture._writer, 0.5)
            await capture.close()
        records = [json.loads(message.split("diagnostic=", 1)[1])
            for message in logs.output if "write diagnostic=" in message]
        self.assertEqual(len(records), 1)
        self.assertEqual(capture.issue, "context_capture_persistence_failed")
        combined = "\n".join(logs.output)
        for private in ("PRIVATE_OWNER", "PRIVATE_CONNECTION", "PRIVATE_INPUT",
                        "PRIVATE_SPEECH", "sk-private-credential", "PRIVATE_ERROR"):
            self.assertNotIn(private, combined)
        self.assertLess(len(combined), 4000)
        self.assertLessEqual(len(records[0]["application_frames"]), 8)
        return records[0]

    async def test_generic_error_is_classified_without_private_exception_or_payload(self):
        record = await self.failure(ValueError("PRIVATE_ERROR PRIVATE_SPEECH"))
        self.assertEqual(record["error_type"], "ValueError")
        self.assertIsNone(record["cause_type"])
        self.assertFalse(record["deadline_exceeded"])
        self.assertEqual(record["kind"], "user_transcript")
        self.assertEqual(record["finality"], "final")
        self.assertFalse(record["is_gap"])

    async def test_capture_deadline_is_distinguished_from_inner_timeout(self):
        deadline = await self.failure(stall=True)
        self.assertTrue(deadline["deadline_exceeded"])
        self.assertEqual(deadline["error_type"], "TimeoutError")
        self.assertEqual(deadline["cause_type"], "CancelledError")
        self.assertGreaterEqual(deadline["elapsed_ms"], 0)
        inner = await self.failure(TimeoutError("PRIVATE_ERROR"))
        self.assertFalse(inner["deadline_exceeded"])
        self.assertEqual(inner["error_type"], "TimeoutError")

    async def test_task_error_code_preserves_guard_reason_not_message(self):
        record = await self.failure(TaskError("context_finality_conflict", "PRIVATE_ERROR"))
        self.assertEqual(record["task_error_code"], "context_finality_conflict")
        self.assertFalse(record["deadline_exceeded"])

    async def test_source_removal_remains_a_normal_tombstone_without_failure(self):
        async def append(owner, event):
            raise TaskError("context_source_removed", "PRIVATE_ERROR")

        capture = self.capture(append)
        with patch("app.tasks.context_capture.LOGGER.warning") as warning:
            capture.input_final("input", "PRIVATE_SPEECH")
            await asyncio.wait_for(capture._writer, 0.5)
            await capture.close()
        warning.assert_not_called()
        self.assertIsNone(capture.issue)
        self.assertEqual(capture.failed_events, 0)

    async def test_external_cancellation_propagates_without_claiming_write_failure(self):
        entered = asyncio.Event()

        async def append(owner, event):
            entered.set()
            await asyncio.Event().wait()

        capture = self.capture(append)
        with patch("app.tasks.context_capture.LOGGER.warning") as warning:
            capture.input_final("input", "PRIVATE_SPEECH")
            await asyncio.wait_for(entered.wait(), 0.5)
            capture._writer.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await capture._writer
            await capture.close()
        warning.assert_not_called()
        self.assertIsNone(capture.issue)
