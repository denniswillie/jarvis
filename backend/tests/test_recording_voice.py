"""Recording command coordination with explicit domain/device substitutes."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileTransferPolicy
from app.knowledge.models import SourceError
from app.recordings.voice import LeanRecordingDispatcher


class RecordingVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.owner, self.session, self.generation = "fixture-recording", uuid4(), uuid4()
        self.events, self.scopes, self.closed = [], {}, set()
        self.saved = None

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, self.owner)
            yield object()

        async def admit(command, area_id, policy):
            if command.operation_id not in self.scopes:
                self.scopes[command.operation_id] = SimpleNamespace(recording_id=command.operation_id,
                    include_jarvis_audio=command.normalized_arguments["include_jarvis_audio"],
                    max_bytes=policy.max_bytes, max_duration_ms=policy.max_duration_ms)
            scope = self.scopes[command.operation_id]
            return SimpleNamespace(scope=scope, outcome="admitted", capture_allowed=command.operation_id not in self.closed)

        async def stop(command):
            from uuid import UUID
            identity = UUID(command.normalized_arguments["recording_id"])
            self.closed.add(identity)
            return SimpleNamespace(scope=self.scopes[identity], outcome="capture_closed", capture_allowed=False)

        async def require(connection, owner, session, identity, *, for_capture=False):
            if for_capture and identity in self.closed:
                raise SourceError("recording_capture_closed", "Closed.", 403)
            return self.scopes[identity]

        self.original_admit = admit
        self.admissions = SimpleNamespace(sources=SimpleNamespace(transaction=transaction),
            tasks=SimpleNamespace(lock_owner=AsyncMock()), selections=SimpleNamespace(read=AsyncMock(
                return_value={"area_id": str(uuid4()), "area_status": "active"})),
            admit=AsyncMock(side_effect=admit), stop=AsyncMock(side_effect=stop),
            require_admitted_recording=AsyncMock(side_effect=require), get_completion=AsyncMock(return_value=None))

        def enqueue(payload):
            self.events.append(payload)
            return True

        self.dispatch = LeanRecordingDispatcher(owner_id=self.owner, session_id=self.session,
            admissions=self.admissions, policy=FileTransferPolicy("mp4_capture_v1", 40000000),
            provider_session_key=lambda: "fixture-provider", is_active=lambda: True, on_command=enqueue)
        self.device = {"type": "recording.capabilities", "version": 1, "generation_id": str(self.generation),
            "available": True, "include_jarvis_audio": True, "max_bytes": 30000000, "max_duration_ms": CAPTURE_MAX_DURATION_MS}

    async def start(self, call="start", include=True):
        return await self.dispatch.execute({"action": "start", "include_jarvis_audio": include}, provider_call_id=call)

    def report(self, sequence, state, **extra):
        return {"type": "recording.report", "version": 1, "generation_id": str(self.generation),
                "recording_id": str(self.dispatch.current_id), "sequence": sequence, "state": state} | extra

    async def test_unoffered_device_or_unspecified_audio_policy_never_admits(self):
        self.assertEqual((await self.start())["outcome"], "unavailable")
        self.dispatch.configure_device(self.device)
        result = await self.dispatch.execute({"action": "start"}, provider_call_id="missing-policy")
        self.assertEqual(result["outcome"], "rejected")
        self.admissions.admit.assert_not_awaited()

    async def test_start_is_admission_not_capture_or_save_and_uses_lower_device_limit(self):
        self.dispatch.configure_device(self.device)
        result = await self.start()
        self.assertEqual(result["outcome"], "admitted")
        self.assertEqual(result["max_bytes"], 30000000)
        self.assertTrue(result["device_command_enqueued"])
        self.assertNotIn("saved", result)
        self.assertEqual(self.events[0]["action"], "start")
        self.assertEqual(self.events[0]["generation_id"], str(self.generation))

    async def test_confirmed_command_waits_for_actual_phone_start(self):
        self.dispatch.configure_device(self.device)
        task = asyncio.create_task(self.dispatch.execute_confirmed(
            {"action": "start", "include_jarvis_audio": True}, provider_call_id="confirmed"))
        for _ in range(100):
            if self.events:
                break
            await asyncio.sleep(0)
        self.assertTrue(self.events)
        self.assertFalse(task.done())
        self.dispatch.accept_report(self.report(1, "started"))
        result = await asyncio.wait_for(task, 1)
        self.assertEqual(result["outcome"], "capture_started")
        self.assertTrue(result["capture_confirmed"])
        self.assertFalse(result["last_client_report"]["saved"])

    async def test_confirmed_command_reports_phone_failure_and_timeout_truthfully(self):
        self.dispatch.configure_device(self.device)
        def fail_on_enqueue(payload):
            self.events.append(payload)
            self.dispatch.accept_report(self.report(1, "failed"))
            return True
        self.dispatch.on_command = fail_on_enqueue
        failed = await self.dispatch.execute_confirmed(
            {"action": "start", "include_jarvis_audio": True}, provider_call_id="failed")
        self.assertEqual(failed["outcome"], "capture_failed")
        self.assertFalse(failed["capture_confirmed"])
        self.dispatch.reports.clear()
        unconfirmed = await self.dispatch.execute_confirmed(
            {"action": "start", "include_jarvis_audio": True}, provider_call_id="failed",
            confirmation_timeout=0.001)
        self.assertFalse(unconfirmed["capture_confirmed"])
        self.assertIn("not confirmed", unconfirmed["summary"])
        self.assertEqual(len(self.events), 1)

    async def test_confirmed_start_does_not_block_stop_or_accept_wrong_generation(self):
        self.dispatch.configure_device(self.device)
        starting = asyncio.create_task(self.dispatch.execute_confirmed(
            {"action": "start", "include_jarvis_audio": True}, provider_call_id="start"))
        for _ in range(100):
            if self.events:
                break
            await asyncio.sleep(0)
        self.assertTrue(self.events)
        self.assertIsNone(self.dispatch.accept_report(self.report(1, "started", generation_id=str(uuid4()))))
        self.assertFalse(starting.done())
        stop = await self.dispatch.execute({"action": "stop"}, provider_call_id="stop")
        self.assertEqual(stop["outcome"], "capture_closed")
        self.dispatch.accept_report(self.report(1, "stopped"))
        result = await asyncio.wait_for(starting, 1)
        self.assertEqual(result["outcome"], "capture_stopped")
        self.assertFalse(result["capture_confirmed"])

    async def test_replayed_call_does_not_enqueue_another_capture(self):
        self.dispatch.configure_device(self.device)
        first, second = await self.start(), await self.start()
        self.assertEqual(first["recording_id"], second["recording_id"])
        self.assertEqual(len(self.scopes), 1)
        self.assertEqual(len(self.events), 1)

    async def test_new_call_cannot_start_a_second_open_recording(self):
        self.dispatch.configure_device(self.device)
        first = await self.start()
        second = await self.start("duplicate-intent")
        self.assertEqual(second["recording_id"], first["recording_id"])
        self.admissions.admit.assert_awaited_once()

    async def test_interruption_before_admission_is_rejected(self):
        self.dispatch.configure_device(self.device)
        result = await self.dispatch.execute({"action": "start", "include_jarvis_audio": False},
            provider_call_id="interrupted", admit_if=lambda: False)
        self.assertEqual(result["outcome"], "rejected")
        self.admissions.admit.assert_not_awaited()

    async def test_explicit_stop_during_admission_suppresses_late_start_hint(self):
        self.dispatch.configure_device(self.device)
        entered, finish = asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            entered.set()
            await finish.wait()
            return await self.original_admit(*args, **kwargs)

        self.admissions.admit.side_effect = delayed
        starting = asyncio.create_task(self.start())
        await asyncio.wait_for(entered.wait(), 1)
        stopping = asyncio.create_task(self.dispatch.execute({"action": "stop"}, provider_call_id="stop"))
        await asyncio.sleep(0)
        finish.set()
        first, last = await asyncio.gather(starting, stopping)
        self.assertFalse(first["device_command_enqueued"])
        self.assertEqual(last["outcome"], "capture_closed")
        self.assertEqual([event["action"] for event in self.events], ["stop"])

    async def test_queue_failure_does_not_erase_durable_admission(self):
        self.dispatch.configure_device(self.device)
        self.dispatch.on_command = lambda payload: False
        result = await self.start()
        self.assertEqual(result["outcome"], "admitted")
        self.assertTrue(result["effect_committed"])
        self.assertFalse(result["device_command_enqueued"])

    async def test_reports_are_generation_fenced_ordered_and_never_saved_receipts(self):
        self.dispatch.configure_device(self.device)
        await self.start()
        self.assertIsNone(self.dispatch.accept_report(self.report(1, "started", generation_id=str(uuid4()))))
        report = self.dispatch.accept_report(self.report(1, "started"))
        self.assertEqual(report["basis"], "client_report")
        self.assertFalse(report["saved"])
        self.assertIsNone(self.dispatch.accept_report(self.report(1, "started")))
        self.assertIsNotNone(self.dispatch.accept_report(self.report(2, "stopped")))
        self.assertIsNone(self.dispatch.accept_report(self.report(3, "started")))
        self.assertIsNone(self.dispatch.accept_report(self.report(4, "saved")))

    async def test_status_requires_actual_domain_saved_receipt(self):
        self.dispatch.configure_device(self.device)
        await self.start()
        self.dispatch.accept_report(self.report(1, "started"))
        result = await self.dispatch.execute({"action": "status"}, provider_call_id="status")
        self.assertEqual(result["outcome"], "admitted")
        self.admissions.get_completion.return_value = SimpleNamespace(
            file_id=uuid4(), source_id=uuid4(), version_id="exact-version", receipt_event_id=uuid4())
        result = await self.dispatch.execute({"action": "status"}, provider_call_id="saved-status")
        self.assertEqual(result["outcome"], "saved")
        self.assertEqual(result["saved"]["version_id"], "exact-version")

    async def test_invalid_capability_update_revokes_local_availability(self):
        self.dispatch.configure_device(self.device)
        self.assertFalse(self.dispatch.configure_device(self.device | {"max_bytes": True}))
        self.assertEqual((await self.start())["outcome"], "unavailable")

    async def test_missing_include_audio_capability_does_not_silently_drop_it(self):
        self.dispatch.configure_device(self.device | {"include_jarvis_audio": False})
        self.assertEqual((await self.start())["outcome"], "unavailable")
        self.assertEqual((await self.start(include=False))["outcome"], "admitted")
