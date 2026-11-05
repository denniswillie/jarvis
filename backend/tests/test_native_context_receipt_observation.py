"""Preserve exact typed context receipt outcomes, never their source contents."""
from datetime import datetime, timezone
import json
import unittest
from uuid import uuid4

from app.live_state.models import SessionScope, StateCursor
from app.voice_routes.contracts import ContextReceipt, RouteIdentity
from evals.responsive_voice.frontend_host_qualification import native_route_observation


class NativeContextReceiptObservationTests(unittest.TestCase):
    def receipt(self, status="unknown", reason="context_write_failed"):
        return ContextReceipt(route=RouteIdentity(
            scope=SessionScope(owner_id="PRIVATE_OWNER", session_id=uuid4(), area_id=None),
            route_id=uuid4(), route_generation=1), update_id=uuid4(),
            cursor=StateCursor(epoch=uuid4(), revision=36), observed_at=datetime.now(timezone.utc),
            status=status, reason=reason, provider_receipt_ref="PRIVATE_PROVIDER_REFERENCE")

    def test_exact_receipt_identity_and_status_survive_without_private_fields(self):
        for status in ("written", "provider_acknowledged", "rejected", "failed", "unknown"):
            with self.subTest(status=status):
                receipt = self.receipt(status)
                payload = {"type":"voice_route.event", "payload":receipt.model_dump(mode="json")}
                original = json.dumps(payload, sort_keys=True)
                row = native_route_observation(payload)
                self.assertEqual(row["context_receipt"], {
                    "route_id":str(receipt.route.route_id), "route_generation":1,
                    "update_id":str(receipt.update_id), "cursor_epoch":str(receipt.cursor.epoch),
                    "cursor_revision":36, "observed_at":receipt.observed_at.isoformat(),
                    "status":status, "reason":"context_write_failed"})
                self.assertNotIn("PRIVATE", json.dumps(row))
                self.assertEqual(json.dumps(payload, sort_keys=True), original)

    def test_unknown_private_reason_is_redacted_and_malformed_receipt_is_not_accepted(self):
        receipt = self.receipt(reason="sk-PRIVATE_REASON")
        row = native_route_observation({"type":"voice_route.event", "payload":receipt.model_dump(mode="json")})
        self.assertEqual(row["context_receipt"]["reason"], "unrecognized_code")
        self.assertNotIn("PRIVATE", json.dumps(row))
        invalid = receipt.model_dump(mode="json")
        invalid["source_text"] = "PRIVATE_TRANSCRIPT"
        row = native_route_observation({"type":"voice_route.event", "payload":invalid})
        self.assertEqual(row["context_receipt_invalid"], True)
        self.assertNotIn("context_receipt", row)
        self.assertNotIn("PRIVATE", json.dumps(row))

    def test_other_route_events_do_not_acquire_context_evidence(self):
        row = native_route_observation({"type":"voice_route.event", "payload":{
            "kind":"delivery_evidence", "content":"PRIVATE_TRANSCRIPT"}})
        self.assertEqual(set(row), {"type", "code", "observed_at"})
        self.assertNotIn("PRIVATE", json.dumps(row))
