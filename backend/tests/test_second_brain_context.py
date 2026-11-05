from datetime import datetime
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.tasks.context_events import ContextIngress, ContextWindow
from app.tools.invocation import ToolInvocationContext, bind_tool_invocation, current_tool_invocation


class CanonicalContextContractTests(unittest.TestCase):
    def event(self, **changes):
        return ContextIngress(**({"session_id": uuid4(), "source_namespace": "fixture", "source_event_id": "utterance",
            "receipt_id": "receipt", "kind": "user_transcript", "trust": "user_input", "payload": {"text": "Fixture"}} | changes))

    def test_source_text_cannot_claim_user_authority(self):
        for changes in ({"trust": "assistant_output"}, {"trust": "external_content"}, {"kind": "tool_outcome"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.event(**changes)

    def test_event_identity_time_and_window_are_bounded(self):
        for changes in ({"receipt_id": " "}, {"source_time": datetime(2026, 1, 1)}, {"payload": {"bad": float("nan")}}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                self.event(**changes)
        for changes in ({"max_messages": 0}, {"max_characters": 1}, {"max_pins": 1000}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                ContextWindow(**changes)

    def test_missing_text_cannot_become_a_final_user_request(self):
        for text in (None, "", " ", "bad\x00text"):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                self.event(payload={"text": text}, finality="final")
        self.event(payload={"text": None}, finality="unavailable")

    def test_request_invocation_identity_is_scoped_and_not_implicitly_inherited(self):
        event_id = uuid4()
        with self.assertRaises(ValueError):
            ToolInvocationContext(request_event_id=event_id)
        with bind_tool_invocation(provider_call_id="outer", request_event_id=event_id, request_revision=2):
            self.assertEqual(current_tool_invocation().request_event_id, event_id)
            with bind_tool_invocation(provider_call_id="unrelated"):
                self.assertIsNone(current_tool_invocation().request_event_id)
            self.assertEqual(current_tool_invocation().request_revision, 2)
        self.assertIsNone(current_tool_invocation().request_event_id)
