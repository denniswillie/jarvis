"""Actual Live publication/transport emitters with external services mocked."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.models import GuardedPayloadBinding, SourceDependency
from app.realtime.source_context import LiveContextError
from app.voice_routes.bridge import OfferDisposition
from app.voice_routes.live_transport import LiveTransportError
import test_live_session as session_fixtures
from test_live_voice_adapter import LiveFixture, NOW
from test_live_voice_transport import TransportFixture


LOGGER = "app.voice_routes.live_adapter.lifecycle"


def rows(captured, prefix):
    return [json.loads(record.getMessage().split(prefix, 1)[1])
        for record in captured.records if prefix in record.getMessage()]


class LivePublicationOriginCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = session_fixtures.LiveSessionTests(
            "test_actual_composition_gates_media_and_joins_single_readers")
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def publication(self, mode):
        session = await self.fixture.start()
        now = datetime.now(timezone.utc)
        initial = session.request.initial_context
        update = initial.model_copy(update={"update_id": uuid4(),
            "cursor": initial.cursor.model_copy(update={"revision": 1000}),
            "created_at": now, "expires_at": now + timedelta(seconds=60)})
        original_guard = session.adapter._b.source_guard

        @asynccontextmanager
        async def guard(payload):
            async with original_guard(payload):
                target = getattr(payload, "update_id", None) == update.update_id
                if target and mode == "withdrawal":
                    raise LiveContextError("live_context_revoked")
                if target and mode == "expiry":
                    session.adapter._clock = lambda: update.expires_at + timedelta(seconds=1)
                yield
                if target and mode == "guard_exit":
                    raise SourceError("source_payload_unavailable", "PRIVATE_SOURCE_TEXT")

        session.adapter._b = replace(session.adapter._b, source_guard=guard)
        if mode == "socket":
            original_send = self.fixture.wire.socket.send

            async def send(text):
                value = json.loads(text)
                if value.get("event_id") == "ctx_" + update.update_id.hex:
                    raise OSError("PRIVATE_SOCKET_ERROR")
                await original_send(text)

            self.fixture.wire.socket.send = send

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            offered = session.offer_context(update)
            self.assertNotEqual(offered.disposition, OfferDisposition.REJECTED)
            await asyncio.wait_for(self.fixture.run_task, 4)
        failures = rows(captured, "live_append_failure_diagnostic=")
        matching = [row for row in failures if row.get("payload_id") == str(update.update_id)]
        self.assertEqual(len(matching), 1)
        retirement, = rows(captured, "live_retirement_diagnostic=")
        self.assertEqual(retirement["reason"], "append_write_unconfirmed")
        self.assertTrue(session.shutdown_confirmed)
        self.assertFalse(session._tasks)
        self.assertEqual(len(self.fixture.wire.requests), 1)
        self.assertNotIn("PRIVATE", json.dumps(matching + [retirement]))
        messages = [m for m in self.fixture.phone.messages if m["type"] == "voice_route.close"]
        self.assertTrue(messages)
        self.assertEqual(messages[0]["payload"]["command_id"], retirement["command_id"])
        self.assertEqual(messages[0]["payload"]["reason"], retirement["reason"])
        return matching[0]

    async def test_actual_canonical_guard_rejection_is_known_before_handoff(self):
        row = await self.publication("withdrawal")
        self.assertFalse(row["payload_taken"])
        self.assertFalse(row["freshness_expired"])
        self.assertEqual(row["transport"], {"stage": "guard_entry", "outcome": "error",
            "socket_handoff": False, "socket_send_completed": False,
            "error_type": "LiveContextError", "error_code": "live_context_revoked"})

    async def test_expiry_inside_guard_is_distinct_without_weakening_existing_fence(self):
        row = await self.publication("expiry")
        self.assertTrue(row["freshness_expired"])
        self.assertFalse(row["payload_taken"])
        self.assertEqual(row["transport"]["stage"], "pre_factory_current")
        self.assertEqual(row["transport"]["outcome"], "not_written")
        self.assertFalse(row["transport"]["socket_handoff"])
        self.assertIsNone(row["transport"]["error_code"])

    async def test_actual_socket_exception_preserves_possible_handoff(self):
        row = await self.publication("socket")
        self.assertTrue(row["payload_taken"])
        self.assertEqual(row["transport"]["stage"], "socket_send")
        self.assertTrue(row["transport"]["socket_handoff"])
        self.assertFalse(row["transport"]["socket_send_completed"])
        self.assertEqual(row["transport"]["error_type"], "OSError")

    async def test_guard_exit_after_send_never_claims_unwritten(self):
        row = await self.publication("guard_exit")
        self.assertTrue(row["payload_taken"])
        self.assertEqual(row["transport"]["stage"], "guard_exit")
        self.assertTrue(row["transport"]["socket_handoff"])
        self.assertTrue(row["transport"]["socket_send_completed"])
        self.assertEqual(row["transport"]["error_code"], "source_payload_unavailable")


class LiveRetirementEmitterTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = LiveFixture()

    async def asyncTearDown(self):
        await self.fixture.close()

    async def test_real_expiry_monitor_emits_original_retirement_once(self):
        await self.fixture.start()
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            self.fixture.now = NOW + timedelta(minutes=6)
            await self.fixture.until(lambda: self.fixture.route._closed)
            await self.fixture.close()
        row, = rows(captured, "live_retirement_diagnostic=")
        self.assertEqual(row["reason"], "remote_context_expired_requires_replacement")
        self.assertTrue(row["current_context_expired"])
        self.assertFalse(rows(captured, "live_append_failure_diagnostic="))
        self.assertEqual(len(self.fixture.fences), 1)

    async def test_source_revision_replacement_is_not_reported_as_routine_expiry(self):
        source = SourceDependency(source_namespace="context_event", source_id=uuid4(), revision=1)
        binding = GuardedPayloadBinding(scope=self.fixture.identity.scope,
            payload_root_ids=(uuid4(),), dependencies=(source,))
        initial = self.fixture.context(dependencies=(source,), guard_binding=binding)
        await self.fixture.start(self.fixture.start_request(initial_context=initial))
        changed = source.model_copy(update={"revision": 2})
        update = self.fixture.context(1, dependencies=(changed,),
            guard_binding=binding.model_copy(update={"dependencies": (changed,)}))
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            receipt = await self.fixture.route.publish_context(update)
            await self.fixture.until(lambda: self.fixture.route._closed)
        self.assertEqual(receipt.status, "rejected")
        row, = rows(captured, "live_retirement_diagnostic=")
        self.assertEqual(row["reason"], "context_replacement_requires_fenced_reconnect")
        self.assertFalse(row["current_context_expired"])
        self.assertEqual(sum(m["type"] == "session.thinking.append" for m in self.fixture.wire.socket.sent), 1)

    async def test_diagnostic_logger_failure_does_not_prevent_real_expiry_fence(self):
        await self.fixture.start()
        with patch("app.voice_routes.live_adapter._LIFECYCLE_LOGGER.warning", side_effect=RuntimeError("PRIVATE")):
            self.fixture.now = NOW + timedelta(minutes=6)
            await self.fixture.until(lambda: self.fixture.route._closed)
        self.assertEqual(len(self.fixture.fences), 1)
        self.assertTrue(self.fixture.wire.socket.closed)


class LiveSendBoundaryEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = TransportFixture()
        await self.fixture.start()

    async def asyncTearDown(self):
        await self.fixture.close()

    async def test_private_error_codes_are_redacted_before_observation(self):
        @asynccontextmanager
        async def guard():
            raise SourceError("PRIVATE_CODE", "PRIVATE_CONTENT")
            yield
        evidence = []
        with self.assertRaisesRegex(LiveTransportError, "sideband_write_unconfirmed"):
            await self.fixture.transport.send(lambda: {"type": "never"}, guard=guard, observe=evidence.append)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].error_code, "unrecognized_code")
        self.assertFalse(evidence[0].socket_handoff)
        self.assertNotIn("PRIVATE", repr(evidence))
        self.assertFalse(self.fixture.socket.sent)

    async def test_callback_failure_does_not_change_success_or_add_send(self):
        def broken(_):
            raise RuntimeError("PRIVATE")
        self.assertTrue(await self.fixture.transport.send(lambda: {"type": "fixture"}, observe=broken))
        self.assertEqual(self.fixture.socket.sent, [{"type": "fixture"}])

    async def test_cancellation_after_socket_handoff_remains_cancelled_and_unknown(self):
        self.fixture.socket.block_type = "blocked"
        evidence = []
        task = asyncio.create_task(self.fixture.transport.send(lambda: {"type": "blocked"}, observe=evidence.append))
        await self.fixture.socket.blocked.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].outcome, "cancelled")
        self.assertTrue(evidence[0].socket_handoff)
        self.assertFalse(evidence[0].socket_send_completed)
        self.assertEqual(len(self.fixture.socket.sent), 1)
