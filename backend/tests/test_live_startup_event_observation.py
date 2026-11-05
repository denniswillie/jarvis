"""Content-free startup event evidence; no provider, database or native calls."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import unittest
from uuid import UUID

from app.live_state.models import SessionScope, StateCursor
from app.voice_routes.contracts import (
    CommandReceipt, ContextReceipt, RouteConnectionEvent, RouteError, RouteIdentity,
    TranscriptRevision,
)
from evals.responsive_voice.frontend_host_qualification import native_route_observation


def startup_wire_fixtures():
    """Synthetic canonical model serialization, explicitly not retained run data."""
    identity = RouteIdentity(scope=SessionScope(owner_id="PRIVATE_FIXTURE_OWNER",
        session_id=UUID("11111111-1111-4111-8111-111111111111")),
        route_id=UUID("22222222-2222-4222-8222-222222222222"), route_generation=1)
    observed = datetime(2026, 10, 31, 16, 6, 43, 709174, tzinfo=timezone.utc)
    cursor = StateCursor(epoch=UUID("33333333-3333-4333-8333-333333333333"), revision=4)
    update_id = UUID("44444444-4444-4444-8444-444444444444")
    events = {
        "starting": RouteConnectionEvent(route=identity, observed_at=observed, state="starting"),
        "ready": RouteConnectionEvent(route=identity, observed_at=observed, state="ready"),
        "route_error": RouteError(route=identity, observed_at=observed, code="live_reader_failed",
            message="PRIVATE_FAILURE_TEXT", recoverable=False),
        "command_completed": CommandReceipt(route=identity, observed_at=observed,
            command_id=UUID("66666666-6666-4666-8666-666666666666"), operation="start", status="completed"),
        "provisional_caption": TranscriptRevision(route=identity, observed_at=observed,
            transcript_id=UUID("77777777-7777-4777-8777-777777777777"), revision=1,
            role="assistant", text="PRIVATE_SYNTHETIC_CAPTION", final=False),
    }
    for status in ("queued", "written", "provider_acknowledged"):
        events["context_" + status] = ContextReceipt(route=identity, observed_at=observed,
            update_id=update_id, cursor=cursor, status=status,
            provider_receipt_ref="PRIVATE_PROVIDER_REFERENCE" if status == "provider_acknowledged" else None)
    return {name: {"type": "voice_route.event", "protocol_version": 1,
        "audio_session_id": "55555555-5555-4555-8555-555555555555",
        "payload": event.model_dump(mode="json")} for name, event in events.items()}


class LiveStartupEventObservationTests(unittest.TestCase):
    def test_typed_startup_kinds_and_identity_survive_without_contents(self):
        for name, wire in startup_wire_fixtures().items():
            with self.subTest(name=name):
                before = deepcopy(wire)
                row = native_route_observation(wire)
                detail = row["route_event"]
                self.assertTrue(detail["schema_valid"])
                self.assertEqual(detail["kind"], wire["payload"]["kind"])
                self.assertEqual(detail["route_id"], wire["payload"]["route"]["route_id"])
                self.assertEqual(detail["route_generation"], 1)
                self.assertEqual(detail["event_observed_at"], "2026-10-31T16:06:43.709174+00:00")
                self.assertNotIn("PRIVATE", json.dumps(row))
                self.assertEqual(wire, before)

    def test_canonical_cursor_is_a_structured_epoch_revision_pair(self):
        for status in ("queued", "written", "provider_acknowledged"):
            with self.subTest(status=status):
                wire = startup_wire_fixtures()["context_" + status]
                self.assertEqual(wire["payload"]["cursor"], {
                    "epoch": "33333333-3333-4333-8333-333333333333", "revision": 4})
                typed = ContextReceipt.model_validate(wire["payload"])
                row = native_route_observation(wire)
                self.assertEqual(row["context_receipt"]["cursor_epoch"], str(typed.cursor.epoch))
                self.assertEqual(row["context_receipt"]["cursor_revision"], 4)
                self.assertEqual(row["context_receipt"]["status"], status)

    def test_failure_and_connection_codes_are_static_not_identifier_shaped_private_text(self):
        fixtures = startup_wire_fixtures()
        for name, field in (("route_error", "code"), ("starting", "reason"), ("command_completed", "reason")):
            for value in ("PRIVATE_LOOKS_LIKE_CODE", "sk-PRIVATE", "PRIVATE user exception content"):
                with self.subTest(name=name, value=value):
                    wire = deepcopy(fixtures[name])
                    wire["payload"][field] = value
                    row = native_route_observation(wire)
                    self.assertTrue(row["route_event"]["schema_valid"])
                    self.assertEqual(row["route_event"][field], "unrecognized_code")
                    self.assertNotIn("PRIVATE", json.dumps(row))
        row = native_route_observation(fixtures["route_error"])
        self.assertEqual(row["route_event"]["code"], "live_reader_failed")
        self.assertFalse(row["route_event"]["recoverable"])

    def test_missing_timestamp_is_invalid_for_every_supported_timed_event(self):
        for name, wire in startup_wire_fixtures().items():
            with self.subTest(name=name):
                del wire["payload"]["observed_at"]
                row = native_route_observation(wire)
                self.assertEqual(row["route_event"], {"kind": wire["payload"]["kind"], "schema_valid": False})
                self.assertNotIn("PRIVATE", json.dumps(row))
                if wire["payload"]["kind"] == "context_receipt":
                    self.assertTrue(row["context_receipt_invalid"])
                    self.assertNotIn("context_receipt", row)

    def test_malformed_known_events_never_echo_validation_messages_or_claim_identity(self):
        mutations = (("extra", "PRIVATE_EXTRA"), ("revision", 0),
            ("transcript_id", "PRIVATE_NOT_UUID"), ("observed_at", "PRIVATE_NOT_TIME"))
        for field, value in mutations:
            with self.subTest(field=field):
                wire = startup_wire_fixtures()["provisional_caption"]
                wire["payload"][field] = value
                row = native_route_observation(wire)
                self.assertEqual(row["route_event"], {"kind": "transcript_revision", "schema_valid": False})
                self.assertNotIn("PRIVATE", json.dumps(row))

    def test_unknown_or_unhashable_kinds_do_not_expand_the_projection(self):
        for kind in ("PRIVATE_KIND", [], {}, None, "delivery_evidence"):
            with self.subTest(kind=kind):
                row = native_route_observation({"type": "voice_route.event", "payload": {
                    "kind": kind, "text": "PRIVATE_TEXT"}})
                self.assertEqual(set(row), {"type", "code", "observed_at"})
                self.assertNotIn("PRIVATE", json.dumps(row))

    def test_large_caption_retains_only_bounded_shape_not_speech_or_attribution(self):
        wire = startup_wire_fixtures()["provisional_caption"]
        wire["payload"]["text"] = "X" * 4096
        row = native_route_observation(wire)
        detail = row["route_event"]
        self.assertEqual(detail["character_count"], 4096)
        self.assertEqual(detail["revision"], 1)
        self.assertEqual(detail["role"], "assistant")
        self.assertFalse(detail["final"])
        self.assertNotIn("text", detail)
        self.assertNotIn("output", detail)
        self.assertLess(len(json.dumps(row)), 1000)

    def test_command_outcome_is_not_readiness_or_playback_proof(self):
        row = native_route_observation(startup_wire_fixtures()["command_completed"])
        detail = row["route_event"]
        self.assertEqual(detail["operation"], "start")
        self.assertEqual(detail["status"], "completed")
        self.assertIsNone(detail["output_fenced"])
        self.assertIsNone(detail["provider_stopped"])
        self.assertNotIn("ready", detail)
        self.assertNotIn("heard", detail)
