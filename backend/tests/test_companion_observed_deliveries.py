"""Offline single-pass delivery bookkeeping; no provider or hearing claims."""
import asyncio
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.companion import CompanionState
from app.live_state.companion_notices import CompanionNotices
from app.perception.companion_runtime import CompanionPerceptionRuntime
from app.perception.multimodal import (
    EventBatch, MultimodalRequest, ObservedReminderDelivery, PerceptionContext,
    SituationalReminder, parse_multimodal_output,
)
from app.shared_memory.ingestion import CanonicalCaption
from test_companion_state import Memory, Reminders
from test_multimodal_perception import NOW, dialogue, response


def delivery_json(quote="Take your umbrella.", *, source="dialogue:0", act="reminder"):
    return json.dumps({"observed_reminder_deliveries": [{
        "reminder_id": "reminder:0", "description": "Umbrella reminder",
        "sources": [source], "speech_act": act, "quote": quote}]})


class DeliveryContractTests(unittest.TestCase):
    def parse(self, speech, raw):
        req = MultimodalRequest(EventBatch("fixture", dialogue=(speech,)),
            PerceptionContext(reminders=(SituationalReminder(id="r", revision=1,
                title="Take your umbrella", condition="Leaving"),)), NOW + timedelta(seconds=20))
        return parse_multimodal_output(raw, req, analyzed_at=NOW + timedelta(seconds=21))

    def test_only_exact_new_assistant_reminder_quote_is_accepted(self):
        speech = dialogue(speaker="assistant", text="Take your umbrella.")
        self.assertEqual(len(self.parse(speech, delivery_json()).observed_reminder_deliveries), 1)
        for changed, raw in (
            (speech.model_copy(update={"speaker": "user"}), delivery_json()),
            (speech, delivery_json(source="dialogue:9")),
            (speech, delivery_json(quote="Take your keys.")),
            (speech, delivery_json(act="other")),
            (speech.model_copy(update={"text": "Take your umbrella. New topic.",
                                       "new_text_start": len("Take your umbrella.")}), delivery_json()),
        ):
            with self.subTest(speech=changed, raw=raw):
                result = self.parse(changed, raw)
                self.assertEqual(result.observed_reminder_deliveries, ())
                self.assertTrue(result.rejected)

    def test_promise_does_not_record_delivery(self):
        text = "I will remind you to take your umbrella later."
        result = self.parse(dialogue(speaker="assistant", text=text), delivery_json(text, act="promise"))
        self.assertEqual(result.observed_reminder_deliveries, ())


class ObservedDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW + timedelta(seconds=10)
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders(),
                                    clock=lambda: self.now)
        await self.state.initialize()
        row = await self.state.manage_reminder(action="create", title="Take your umbrella", condition="Leaving")
        self.reminder_id = row["id"]
        self.state.reminders.rows[row["id"]]["updated_at"] = NOW.isoformat()
        await self.state.refresh_reminders()
        self.manager = NS(send_companion_notice=AsyncMock(return_value={"status": "provider_received"}))
        self.notices = CompanionNotices(self.state, self.manager)
        self.model = NS(generate_temporal_json=AsyncMock(return_value=response()))
        self.runtime = CompanionPerceptionRuntime(state=self.state, model=self.model,
            on_reminder=self.notices.on_reminder, clock=lambda: self.now, caption_settle_seconds=0.01)
        self.session = str(uuid4())
        self.match = NS(reminder_id=self.reminder_id, reminder_revision=NOW.isoformat(),
                        evidence_at=NOW, description="Leaving")

    async def asyncTearDown(self):
        await self.runtime.close()

    def caption(self, speaker, text, *, revision=1, second=1):
        return CanonicalCaption(revision, self.session, "native-live", speaker, revision,
            "assistant_output" if speaker == "assistant" else "user_transcript", NOW, self.now,
            "not_finalized", {"voice_route": "live", "speaker": speaker, "caption_id": speaker,
                "text": text, "observed_at": (NOW + timedelta(seconds=second)).isoformat()})

    def delivery(self):
        return ObservedReminderDelivery(self.reminder_id, NOW.isoformat(), "Umbrella reminder",
            ("dialogue:assistant:1",), NOW + timedelta(seconds=2), "Take your umbrella.")

    def saved(self):
        return self.state._section["notices"][self.reminder_id]

    async def defer(self, waiting=(("assistant", 2),)):
        return await self.state.reserve_reminder_notice(reminder_id=self.reminder_id,
            revision=NOW.isoformat(), evidence_at=NOW, generation=0,
            source_document_ids=("user-document",), waiting_for_assistant=waiting)

    async def run_pending_caption(self, *, says_reminder):
        user_entered, user_release = asyncio.Event(), asyncio.Event()
        assistant_entered, assistant_release = asyncio.Event(), asyncio.Event()
        prompts = []

        async def generate(**kwargs):
            prompt = json.loads(kwargs["prompt"].split("Inputs follow:\n", 1)[1])
            prompts.append(prompt)
            if prompt["dialogue"][0]["speaker"] == "user":
                user_entered.set()
                await user_release.wait()
                return response(reminders=[{"reminder_id": "reminder:0", "description": "Leaving",
                                             "sources": ["dialogue:0"]}])
            assistant_entered.set()
            await assistant_release.wait()
            return delivery_json() if says_reminder else delivery_json(
                "I'll remind you later.", act="promise")

        self.model.generate_temporal_json.side_effect = generate
        self.runtime.submit_settled_caption(self.caption("user", "I am leaving."), owner_id="fixture")
        await asyncio.wait_for(user_entered.wait(), 1)
        caption = self.caption("assistant", "Take your umbrella." if says_reminder else "I'll remind you later.", second=2)
        # Exercise the canonical post-commit settle buffer, not only analyzer._pending.
        record = NS(owner_id="fixture", source_namespace=caption.source_namespace,
            source_event_id=caption.source_event_id, kind=caption.kind, content=caption.content,
            session_id=caption.session_id, is_current=True, availability="not_finalized",
            finality="provisional", revision=caption.revision, sequence=caption.sequence,
            source_time=caption.source_time, recorded_at=caption.recorded_at)
        self.runtime.observe_committed_record(record)
        user_release.set()
        try:
            await asyncio.wait_for(assistant_entered.wait(), 1)
            self.assertEqual(self.saved()["status"], "deferred")
            self.assertEqual(self.saved()["waiting_for_assistant"], {caption.document_id: 1})
            self.manager.send_companion_notice.assert_not_awaited()
        finally:
            assistant_release.set()
        await asyncio.wait_for(self.runtime.drain(), 1)
        self.assertEqual(prompts[-1]["active_situational_reminders"][0]["title"], "Take your umbrella")
        self.assertEqual(prompts[-1]["active_situational_reminders"][0]["notification_status"], "deferred")

    async def test_pending_assistant_delivery_suppresses_older_trigger(self):
        await self.run_pending_caption(says_reminder=True)
        self.manager.send_companion_notice.assert_not_awaited()
        self.assertEqual(self.saved()["status"], "assistant_output_observed")
        self.assertEqual(self.saved()["notified_at"], (NOW + timedelta(seconds=2)).isoformat())
        self.assertEqual(self.saved()["notification_source"]["quote"], "Take your umbrella.")
        self.assertEqual(self.state.read()["system"]["reminders"][0]["status"], "active")

    async def test_pending_promise_releases_valid_trigger(self):
        await self.run_pending_caption(says_reminder=False)
        self.manager.send_companion_notice.assert_awaited_once()
        self.assertEqual(self.saved()["status"], "provider_received")
        self.assertNotIn("notified_at", self.saved())
        self.assertEqual(self.saved()["evidence_description"], "Leaving")
        self.assertIn("Observed (", self.saved()["text"])
        self.assertTrue(self.saved()["text"].endswith(": Leaving"))

    async def test_fresh_trigger_without_assistant_wait_sends_normally(self):
        self.model.generate_temporal_json.return_value = response(reminders=[{
            "reminder_id": "reminder:0", "description": "Leaving", "sources": ["dialogue:0"]}])
        self.runtime.submit_settled_caption(self.caption("user", "I am leaving."), owner_id="fixture")
        await self.runtime.drain()
        self.manager.send_companion_notice.assert_awaited_once()

    async def test_old_quote_in_growing_caption_cannot_become_new_delivery(self):
        first = self.caption("assistant", "Take your umbrella.")
        self.runtime.submit_settled_caption(first, owner_id="fixture")
        await self.runtime.drain()
        self.model.generate_temporal_json.return_value = delivery_json()
        later = replace(first, revision=2, content={**first.content,
            "text": first.content["text"] + " Now let's discuss dinner.",
            "observed_at": (NOW + timedelta(seconds=3)).isoformat()})
        self.runtime.submit_settled_caption(later, owner_id="fixture")
        await self.runtime.drain()
        self.assertIsNone(self.state.reminder_notification_status(self.reminder_id, NOW.isoformat()))
        self.manager.send_companion_notice.assert_not_awaited()
        prompt = json.loads(self.model.generate_temporal_json.call_args.kwargs["prompt"].split("Inputs follow:\n", 1)[1])
        self.assertEqual(prompt["dialogue"][0]["new_text_start"], len(first.content["text"]))

    async def test_wait_target_does_not_grow_with_unrelated_changes(self):
        original = await self.defer()
        await self.state.update_human({"unrelated": True}, observed_at=self.now)
        self.now += timedelta(seconds=1)
        self.assertIsNone(await self.defer((("new-assistant", 9),)))
        self.assertEqual(self.saved()["id"], original["id"])
        self.assertEqual(self.saved()["expires_at"], original["expires_at"])
        self.assertEqual(self.saved()["waiting_for_assistant"], {"assistant": 2})
        self.assertEqual(await self.state.consume_assistant_revisions((("new-assistant", 99),)), [])
        ready = await self.state.consume_assistant_revisions((("assistant", 2),))
        self.assertEqual(len(ready), 1)
        await self.notices.on_reminder(self.match, (), 0)
        self.manager.send_companion_notice.assert_awaited_once()

    async def test_deferred_notice_expires_without_extending_its_ttl(self):
        await self.defer()
        self.now += timedelta(seconds=31)
        self.assertEqual(await self.state.consume_assistant_revisions((("assistant", 2),)), [])
        await self.notices.on_reminder(self.match, (), 0)
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_readiness_and_invalidation_fence_deferred_notices(self):
        await self.defer()
        self.state._ready = False
        self.assertEqual(await self.state.consume_assistant_revisions((("assistant", 2),)), [])
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("user-document",)))
        self.state._ready = True
        await self.notices.on_reminder(self.match, (), 0)
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_observed_speech_preserves_occurrence_and_late_provider_receipt(self):
        notice = await self.defer((("assistant", 1),))
        self.assertFalse(self.state.notice_current(notice))
        self.assertTrue(await self.state.observe_reminder_delivery(self.delivery(), generation=0,
                                                                  source_document_ids=("assistant-document",)))
        self.assertFalse(self.state.notice_current(notice))
        await self.state.finish_reminder_notice(notice, "provider_received")
        self.assertEqual(self.saved()["status"], "assistant_output_observed")
        self.assertEqual(self.saved()["provider_status"], "provider_received")
        self.now += timedelta(seconds=31)
        await self.notices.on_reminder(self.match, (), 0)
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_recorded_four_without_reservation_cannot_create_delivery(self):
        # Preserve the historical speech regression, not its removed quiet policy.
        self.assertIsNone(self.state.reminder_notification_status(self.reminder_id, NOW.isoformat()))
        prefix = " Sure! I'll set that up. Do Not Disturb is on. I\u2019ll check back in 20 seconds."
        self.assertEqual(len(prefix), 77)
        speech = dialogue(2, caption_id="assistant", revision=18, speaker="assistant",
                          text=prefix + " Four.", new_text_start=77)
        req = MultimodalRequest(EventBatch("fixture", dialogue=(speech,)),
                                await self.runtime.read_context(), self.now)
        result = parse_multimodal_output(delivery_json("Four."), req, analyzed_at=self.now)
        # A valid reference/quote is not semantic proof; state must not invent an occurrence.
        self.assertEqual(result.rejected, ())
        self.assertEqual(len(result.observed_reminder_deliveries), 1)
        before = self.state.working_memory.row.revision
        self.assertFalse(await self.state.observe_reminder_delivery(
            result.observed_reminder_deliveries[0], generation=0))
        self.assertEqual(self.state.working_memory.row.revision, before)
        self.assertIsNone(self.state.reminder_notification_status(self.reminder_id, NOW.isoformat()))
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_no_reservation_cannot_create_delivery(self):
        before = self.state.working_memory.row.revision
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.assertEqual(self.state.working_memory.row.revision, before)

    async def test_reminder05_deferred_revision25_records_existing_occurrence(self):
        self.state.reminders.rows[self.reminder_id].update(
            title="Put on your headphones", condition="When you say you are starting the music")
        await self.state.refresh_reminders()
        notice = await self.defer((("assistant", 25),))
        delivery = replace(self.delivery(), sources=("dialogue:assistant:25",),
                           quote="Hey, put on your headphones.")
        self.assertTrue(await self.state.observe_reminder_delivery(
            delivery, generation=0, source_document_ids=("assistant",)))
        saved = self.saved()
        self.assertEqual(saved["id"], notice["id"])
        self.assertEqual(saved["attempted_at"], notice["attempted_at"])
        self.assertEqual(saved["expires_at"], notice["expires_at"])
        self.assertEqual(saved["notified_at"], delivery.evidence_at.isoformat())
        self.assertEqual(saved["notification_source"]["quote"], delivery.quote)
        self.assertEqual(saved["status"], "assistant_output_observed")
        self.assertEqual(saved["waiting_for_assistant"], {})
        self.assertEqual(await self.state.consume_assistant_revisions((("assistant", 25),)), [])
        self.assertEqual(self.state.read()["system"]["reminders"][0]["status"], "active")
        self.assertFalse(self.state.notice_current(notice))
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_unmatched_captured_source_or_revision_cannot_consume_occurrence(self):
        notice = await self.defer((("assistant", 25),))
        for sources in (("dialogue:assistant:24",), ("dialogue:assistant:26",),
                        ("dialogue:other:25",), ("frame:assistant:25",), (),
                        ("dialogue:assistant:25", "dialogue:assistant:25")):
            with self.subTest(sources=sources):
                self.assertFalse(await self.state.observe_reminder_delivery(
                    replace(self.delivery(), sources=sources), generation=0))
                self.assertEqual(self.saved(), notice)

    async def test_pending_without_captured_revision_cannot_be_inferred_delivered(self):
        notice = await self.defer(())
        self.assertTrue(self.state.notice_current(notice))
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.assertEqual(self.saved(), notice)

    async def test_stale_or_future_delivery_and_expired_occurrence_are_refused(self):
        notice = await self.defer((("assistant", 1),))
        for evidence_at in (self.now - timedelta(seconds=61), self.now + timedelta(seconds=1)):
            self.assertFalse(await self.state.observe_reminder_delivery(
                replace(self.delivery(), evidence_at=evidence_at), generation=0))
            self.assertEqual(self.saved(), notice)
        self.now += timedelta(seconds=31)
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.assertEqual(self.saved(), notice)

    async def test_replacement_reminder_cannot_consume_older_occurrence(self):
        notice = await self.defer((("assistant", 1),))
        revision = (NOW + timedelta(seconds=1)).isoformat()
        self.state.reminders.rows[self.reminder_id]["updated_at"] = revision
        await self.state.refresh_reminders()
        self.assertFalse(await self.state.observe_reminder_delivery(
            replace(self.delivery(), reminder_revision=revision), generation=0))
        self.assertEqual(self.saved(), notice)

    async def test_new_generation_cannot_consume_older_occurrence(self):
        notice = await self.defer((("assistant", 1),))
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=()))
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=1))
        self.assertEqual(self.saved(), notice)

    async def test_readiness_loss_and_expired_occurrence_are_refused(self):
        notice = await self.defer((("assistant", 1),))
        self.state._ready = False
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.now += timedelta(seconds=31)
        self.state._ready = True
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.assertEqual(self.saved(), notice)

    async def test_already_dispatched_occurrence_is_not_reclassified(self):
        notice = await self.defer((("assistant", 1),))
        await self.state.finish_reminder_notice(notice, "provider_received")
        self.assertFalse(await self.state.observe_reminder_delivery(self.delivery(), generation=0))
        self.assertEqual(self.saved()["status"], "provider_received")
        self.assertNotIn("notified_at", self.saved())
