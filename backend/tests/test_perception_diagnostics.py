"""Bounded metadata diagnostics, with all provider/persistence I/O offline."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.live_state.companion import CompanionState
from app.live_state.companion_notices import CompanionNotices
from app.perception.companion_runtime import CompanionPerceptionRuntime
from app.perception.multimodal import EventBatch, EvidenceOrigin, MultimodalRequest, parse_multimodal_output
from app.shared_memory.ingestion import CanonicalCaption
from test_companion_observed_deliveries import delivery_json
from test_companion_state import Memory, Reminders
from test_multimodal_perception import NOW, dialogue, response


PRIVATE_TEXT = "Private synthetic words never included in diagnostics."


class PerceptionDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW + timedelta(seconds=10)
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders(),
                                    clock=lambda: self.now)
        await self.state.initialize()
        reminder = await self.state.manage_reminder(action="create", title=PRIVATE_TEXT, condition=PRIVATE_TEXT)
        self.reminder_id = reminder["id"]
        self.state.reminders.rows[self.reminder_id]["updated_at"] = NOW.isoformat()
        await self.state.refresh_reminders()
        self.manager = NS(send_companion_notice=AsyncMock(return_value={"status": "provider_received"}))
        self.notices = CompanionNotices(self.state, self.manager)
        self.model = NS(generate_temporal_json=AsyncMock(return_value=response()))
        self.runtime = CompanionPerceptionRuntime(state=self.state, model=self.model,
            on_reminder=self.notices.on_reminder, clock=lambda: self.now)
        self.caption = CanonicalCaption(1, str(uuid4()), "canonical", "assistant", 1, "assistant_output",
            NOW, self.now, "not_finalized", {"voice_route": "live", "speaker": "assistant",
                "caption_id": "assistant", "text": PRIVATE_TEXT,
                "observed_at": (NOW + timedelta(seconds=1)).isoformat()})

    async def asyncTearDown(self):
        await self.runtime.close()

    def records(self, logs):
        text = "\n".join(logs.output)
        self.assertNotIn(PRIVATE_TEXT, text)
        rows = [json.loads(line.split("_diagnostic=", 1)[1]) for line in logs.output
                if "_diagnostic=" in line]
        self.assertLess(len(text), 12_000)
        return {row["stage"]: row for row in rows}

    async def run_logged(self):
        with self.assertLogs("app.perception", level="INFO") as logs:
            self.runtime.submit_settled_caption(self.caption, owner_id="fixture")
            await self.runtime.drain()
        return self.records(logs)

    async def reserve_captured_caption(self):
        return await self.state.reserve_reminder_notice(reminder_id=self.reminder_id,
            revision=NOW.isoformat(), evidence_at=NOW, generation=0,
            waiting_for_assistant=((self.caption.document_id, self.caption.revision),))

    async def test_absent_finding_is_distinct_from_rejection(self):
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["delivery_ids"], [])
        self.assertEqual(rows["parsed"]["reminder_ids"], [])
        self.assertEqual(rows["parsed"]["rejected"], [])
        self.assertEqual(rows["publication"]["outcome"], "finished")
        self.assertEqual(rows["publication"]["deliveries"], [])
        self.assertEqual(rows["input"]["dialogue"][0]["source_id"], self.caption.document_id)
        self.assertEqual(rows["input"]["dialogue"][0]["revision"], 1)
        self.assertEqual(rows["input"]["pass_id"], rows["parsed"]["pass_id"])
        self.assertEqual(rows["parsed"]["pass_id"], rows["publication"]["pass_id"])

    async def test_optional_rejection_is_logged_without_quote(self):
        self.model.generate_temporal_json.return_value = delivery_json(PRIVATE_TEXT, act="promise")
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["delivery_ids"], [])
        self.assertEqual(rows["parsed"]["rejected"], ["deliveries.0:non_delivery_speech_act"])
        self.assertEqual(rows["publication"]["deliveries"], [])

    async def test_successful_state_recording_is_logged_without_text(self):
        await self.reserve_captured_caption()
        self.model.generate_temporal_json.return_value = delivery_json(PRIVATE_TEXT)
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["delivery_ids"], [self.reminder_id])
        outcome = rows["publication"]["deliveries"][0]
        self.assertTrue(outcome["applied"])
        self.assertEqual(outcome["notification_status"], "assistant_output_observed")

    async def test_refused_delivery_write_is_not_called_success(self):
        await self.reserve_captured_caption()
        async def generate(**kwargs):
            self.state.reminders.rows[self.reminder_id]["updated_at"] = (NOW + timedelta(seconds=5)).isoformat()
            await self.state.refresh_reminders()
            return delivery_json(PRIVATE_TEXT)

        self.model.generate_temporal_json.side_effect = generate
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["delivery_ids"], [self.reminder_id])
        self.assertFalse(rows["publication"]["deliveries"][0]["applied"])
        self.assertEqual(rows["publication"]["deliveries"][0]["notification_status"], "deferred")

    async def test_valid_model_finding_without_occurrence_is_logged_as_refused(self):
        self.model.generate_temporal_json.return_value = delivery_json(PRIVATE_TEXT)
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["rejected"], [])
        self.assertEqual(rows["parsed"]["delivery_ids"], [self.reminder_id])
        self.assertFalse(rows["publication"]["deliveries"][0]["applied"])
        self.assertIsNone(rows["publication"]["deliveries"][0]["notification_status"])

    async def test_generation_refusal_is_explicit(self):
        async def generate(**kwargs):
            await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=(self.caption.document_id,)))
            return delivery_json(PRIVATE_TEXT)

        self.model.generate_temporal_json.side_effect = generate
        rows = await self.run_logged()
        self.assertEqual(rows["parsed"]["delivery_ids"], [self.reminder_id])
        self.assertEqual(rows["publication"]["outcome"], "discarded_pre_forget_result")
        self.assertEqual((rows["publication"]["generation"], rows["publication"]["current_generation"]), (0, 1))

    async def test_fixed_barrier_reservation_is_logged(self):
        speech = dialogue(caption_id="user-document", text=PRIVATE_TEXT)
        batch = EventBatch("fixture", dialogue=(speech,), origins=(
            EvidenceOrigin(speech.evidence_id, self.caption.session_id, "user-source", NOW, speech.caption_id),))
        request = MultimodalRequest(batch, await self.runtime.read_context(), self.now)
        result = parse_multimodal_output(response(reminders=[{
            "reminder_id": "reminder:0", "description": PRIVATE_TEXT, "sources": ["dialogue:0"]}]),
            request, analyzed_at=self.now)
        result = replace(result, diagnostic_id="direct-publish-fixture")
        with patch.object(self.runtime.analyzer, "pending_assistant_revisions", return_value=(("assistant-document", 7),)):
            with self.assertLogs("app.perception", level="INFO") as logs:
                await self.runtime.publish(result)
        row = self.records(logs)["publication"]
        self.assertEqual(row["pending_assistant_revisions"], [["assistant-document", 7]])
        self.assertEqual(row["triggers"][0]["outcome"], "reservation_created")
        self.assertEqual(row["triggers"][0]["status"], "deferred")
        self.assertEqual(row["triggers"][0]["waiting_for_assistant"], {"assistant-document": 7})
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_model_failure_logs_phase_not_exception_message(self):
        self.model.generate_temporal_json.side_effect = RuntimeError(PRIVATE_TEXT)
        rows = await self.run_logged()
        self.assertEqual(rows["failed"]["phase"], "model")
        self.assertEqual(rows["failed"]["error_type"], "RuntimeError")
        self.assertNotIn("parsed", rows)
        self.assertNotIn("publication", rows)
        self.assertEqual(self.runtime.analyzer.last_error, "RuntimeError")

    async def test_processed_frame_ids_and_count_are_logged_without_image_bytes(self):
        frame_id = uuid4()
        frame = NS(id=frame_id, owner_id="fixture", session_id=self.caption.session_id,
            frame_sequence=1, captured_at=NOW, source="glasses", content_type="image/jpeg",
            raw_image_status="available")
        with self.assertLogs("app.perception", level="INFO") as logs:
            self.runtime.submit_frame(frame, PRIVATE_TEXT.encode())
            await self.runtime.drain()
        row = self.records(logs)["input"]
        self.assertEqual(row["frames"], [str(frame_id)])
        self.assertEqual(row["frame_count"], 1)
        self.assertEqual(row["omitted_frames"], 0)
