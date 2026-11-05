import json
import unittest

from app.perception.multimodal import (
    EventBatch, MultimodalRequest, PerceptionContext, SituationalReminder, parse_multimodal_output,
)
from test_multimodal_perception import NOW, dialogue


class FindingReasonTests(unittest.TestCase):
    def parse(self, item, *, kind="observed_reminder_deliveries", speech=None):
        speech = speech or dialogue(speaker="assistant", text="Sit up straight.")
        request = MultimodalRequest(EventBatch("fixture", dialogue=(speech,)),
            PerceptionContext(reminders=(SituationalReminder(id="actual-reminder", revision=1,
                title="Sit up straight", condition="At the desk"),)), NOW)
        return parse_multimodal_output(json.dumps({kind: [item]}), request, analyzed_at=NOW)

    def item(self, **updates):
        return {"reminder_id": "reminder:0", "description": "A reminder", "sources": ["dialogue:0"],
                "speech_act": "reminder", "quote": "Sit up straight.", **updates}

    def test_short_and_canonical_handle_mapping_is_valid(self):
        self.assertEqual(len(self.parse(self.item()).observed_reminder_deliveries), 1)
        item = self.item(reminder_id="actual-reminder", sources=["dialogue:caption-one:1"])
        self.assertEqual(len(self.parse(item).observed_reminder_deliveries), 1)

    def test_exact_delivery_reasons_preserve_rejection(self):
        cases = [
            (self.item(reminder_id="reminder_0"), None, "unknown_reminder_handle"),
            (self.item(sources=["dialogue_0"]), None, "unknown_source_handle"),
            (self.item(speech_act="promise"), None, "non_delivery_speech_act"),
            (self.item(), dialogue(text="Sit up straight."), "wrong_speaker"),
            (self.item(quote="Different text"), None, "quote_mismatch"),
            (self.item(quote=" "), None, "blank_quote"),
            (self.item(), dialogue(speaker="assistant", text="Sit up straight. Later.", new_text_start=16), "quote_not_new"),
            (self.item(), dialogue(speaker="assistant", text="Sit up straight. Sit up straight."), "ambiguous_quote"),
        ]
        for item, speech, reason in cases:
            with self.subTest(reason=reason):
                result = self.parse(item, speech=speech)
                self.assertEqual(result.observed_reminder_deliveries, ())
                self.assertEqual(result.rejected, (f"deliveries.0:{reason}",))

    def test_paired_unknown_reminder_is_named_for_each_kind(self):
        item = self.item(reminder_id="reminder_0")
        delivery = self.parse(item)
        trigger = self.parse({k: v for k, v in item.items() if k not in {"speech_act", "quote"}},
                             kind="triggered_reminders")
        self.assertEqual(delivery.rejected, ("deliveries.0:unknown_reminder_handle",))
        self.assertEqual(trigger.rejected, ("reminders.0:unknown_reminder_handle",))

    def test_shape_diagnostic_does_not_echo_private_fields_or_values(self):
        item = self.item()
        del item["sources"]
        item["PRIVATE_UNEXPECTED_FIELD"] = "PRIVATE_CONTENT"
        result = self.parse(item)
        self.assertEqual(result.rejected, ("deliveries.0:invalid_shape:sources:missing",))
        self.assertNotIn("PRIVATE", str(result.rejected))
        item = self.item()
        item["PRIVATE_UNEXPECTED_FIELD"] = "PRIVATE_CONTENT"
        self.assertEqual(self.parse(item).rejected, ("deliveries.0:invalid_shape:item:extra_forbidden",))
