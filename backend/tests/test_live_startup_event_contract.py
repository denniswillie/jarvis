"""Offline Live startup event contracts, not native/provider acceptance.

The real session, route, reader and event writer run against existing controlled
HTTP, socket and canonical-store fixtures. No simulated caption is user intent,
provider finality, media readiness or evidence of hearing.
"""
import asyncio
from datetime import datetime, timezone
import json
import unittest
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.live_state.models import SessionScope
from app.voice_routes.contracts import RouteError, RouteIdentity, TranscriptRevision
import test_live_session as session_fixtures
from test_live_voice_adapter import LiveFixture


class LiveStartupEventSchemaTests(unittest.TestCase):
    def test_route_error_requires_its_canonical_timestamp(self):
        identity = RouteIdentity(scope=SessionScope(owner_id="fixture", session_id=uuid4()),
            route_id=uuid4(), route_generation=1)
        event = RouteError(route=identity, observed_at=datetime.now(timezone.utc),
            code="live_reader_failed", message="Route requires replacement.", recoverable=False)
        wire = event.model_dump(mode="json")
        self.assertEqual(RouteError.model_validate(wire), event)
        self.assertIsInstance(wire["observed_at"], str)
        without_timestamp = dict(wire)
        del without_timestamp["observed_at"]
        with self.assertRaises(ValidationError):
            RouteError.model_validate(without_timestamp)


class LiveStartupFailureContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_failure_emits_timestamp_and_fences_once(self):
        fixture = LiveFixture()
        try:
            self.assertEqual((await fixture.start()).status, "completed")
            fixture.route._fail("live_reader_failed")
            fixture.route._fail("live_reader_failed")
            await fixture.until(lambda: fixture.route._closed)
            await fixture.pump
            errors = [event for event in fixture.events if isinstance(event, RouteError)]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0].code, "live_reader_failed")
            self.assertEqual(RouteError.model_validate(errors[0].model_dump(mode="json")), errors[0])
            self.assertTrue(fixture.route.requires_fenced_replacement)
            self.assertEqual(len(fixture.fences), 1)
            self.assertEqual(len(fixture.wire.requests), 1)
            self.assertTrue(fixture.wire.socket.closed)
        finally:
            await fixture.close()


class LiveStartupCaptionCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reuse setup/teardown, not inherited test cases or another session loop.
        self.fixture = session_fixtures.LiveSessionTests(
            "test_actual_composition_gates_media_and_joins_single_readers")
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_actual_reader_can_forward_provisional_caption_before_activation(self):
        fixture = self.fixture
        fixture.phone.auto_prepare = False
        session = await fixture.make()
        self.assertTrue(await fixture.manager.register(
            audio_session_id=fixture.context.audio_session_id,
            owner_id=fixture.context.owner_id,
            jarvis_session_id=fixture.context.jarvis_session_id, session=session))
        fixture.run_task = asyncio.create_task(session.run())
        await fixture.until(lambda: any(message["type"] == "live.media.prepare"
            for message in fixture.phone.messages))

        fixture.wire.socket.incoming.put_nowait(json.dumps({
            "type": "session.output_transcript.delta", "event_id": "fixture-early-caption",
            "delta": "Synthetic provisional caption.", "start_ms": 0, "end_ms": 100,
        }))

        def captions():
            return [message for message in fixture.phone.messages
                if message["type"] == "voice_route.event"
                and message["payload"]["kind"] == "transcript_revision"]

        await fixture.until(lambda: bool(captions()))
        first = captions()[0]
        self.assertEqual(set(first), {"type", "protocol_version", "audio_session_id", "payload"})
        self.assertEqual(first["audio_session_id"], fixture.context.audio_session_id)
        self.assertEqual(first["protocol_version"], 1)
        event = TranscriptRevision.model_validate(first["payload"])
        self.assertEqual(event.route, session.adapter.identity)
        self.assertIsInstance(event.transcript_id, UUID)
        self.assertEqual(event.revision, 1)
        self.assertEqual(event.role, "assistant")
        self.assertFalse(event.final)
        self.assertIsNone(event.output)
        self.assertIsInstance(first["payload"]["observed_at"], str)
        self.assertFalse(session.ready)
        self.assertFalse(any(message["type"] == "live.media.activate"
            for message in fixture.phone.messages))
        self.assertFalse(any(message["type"] == "voice_route.event"
            and message["payload"]["kind"] == "connection_state"
            and message["payload"]["state"] == "ready" for message in fixture.phone.messages))
        captured = [item for item in fixture.captured if item.kind == "assistant_output"]
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].finality, "provisional")
        self.assertIsNone(captured[0].payload["output_identity"])
        self.assertIsNone(captured[0].payload["heard_transcript"])

        # A caption did not replace actual matching media readiness.
        prepare = next(message for message in fixture.phone.messages
            if message["type"] == "live.media.prepare")
        fixture.phone.incoming.put_nowait(
            {key: value for key, value in prepare.items() if key != "answer_sdp"} | {
                "type": "live.media.receipt", "phase": "prepare", "receipt_id": str(uuid4()),
                "provider_started": True, "output_gated": True,
            })
        await fixture.until(lambda: session.ready)
        self.assertEqual([message["type"] for message in fixture.phone.messages
            if message["type"].startswith("live.media.")], ["live.media.prepare", "live.media.activate"])
        await session.close("fixture_complete")
        await fixture.run_task
        self.assertTrue(session.shutdown_confirmed)
        self.assertFalse(session._tasks)
