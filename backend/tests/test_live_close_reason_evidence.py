"""First owner-close reason and exact bounded command metadata, offline only."""
import asyncio
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.realtime.source_context import LiveContextError
from app.voice_routes.contracts import CloseRouteCommand, InterruptSpeechCommand, RouteIdentity
from evals.responsive_voice.frontend_host_qualification import native_route_observation
import test_live_session as session_fixtures
from test_live_startup_event_observation import startup_wire_fixtures


LOGGER = "app.realtime.live_session.close"


def records(captured):
    return [json.loads(record.getMessage().split("live_close_diagnostic=", 1)[1])
        for record in captured.records if "live_close_diagnostic=" in record.getMessage()]


class LiveOwnerCloseDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = session_fixtures.LiveSessionTests(
            "test_actual_composition_gates_media_and_joins_single_readers")
        await self.fixture.asyncSetUp()

    async def asyncTearDown(self):
        await self.fixture.asyncTearDown()

    async def test_actual_route_failure_retains_first_reason_before_normal_fence_and_join(self):
        session = await self.fixture.start()
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            session.adapter._fail("remote_context_expired_requires_replacement")
            await asyncio.wait_for(self.fixture.run_task, 3)
        rows = records(captured)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason"], "remote_context_expired_requires_replacement")
        self.assertEqual(rows[0]["route_id"], str(session.adapter.identity.route_id))
        self.assertTrue(rows[0]["startup_completed"])
        self.assertTrue(rows[0]["adapter_retired"])
        self.assertIsNone(rows[0]["cause_type"])
        self.assertTrue(session.shutdown_confirmed)
        self.assertFalse(session._tasks)
        self.assertTrue(any(message["type"] == "voice_route.close" for message in self.fixture.phone.messages))

    async def test_source_watch_preserves_error_code_without_exception_text(self):
        session = await self.fixture.make()
        error = LiveContextError("live_context_revoked")
        session._source_context = SimpleNamespace(wait_for_invalidation=AsyncMock(side_effect=error))
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            await session._watch_sources()
        row, = records(captured)
        self.assertTrue(session._closed.is_set())
        self.assertEqual(session._close_reason, "live_source_context_revoked")
        self.assertEqual(row["cause_type"], "LiveContextError")
        self.assertEqual(row["cause_code"], "live_context_revoked")
        self.assertFalse(row["startup_completed"])
        self.assertIsNone(row["route_id"])

    async def test_actual_client_stop_preserves_explicit_reason_and_finalization_logger(self):
        session = await self.fixture.start()
        with self.assertLogs(LOGGER, level="WARNING") as close_records, \
                self.assertLogs("app.realtime.live_session.finalization", level="INFO") as final_records:
            self.fixture.phone.incoming.put_nowait({"type": "session.stop"})
            await asyncio.wait_for(self.fixture.run_task, 3)
        row, = records(close_records)
        self.assertEqual(row["reason"], "client_requested_close")
        self.assertIsNone(row["cause_type"])
        self.assertIsNone(row["cause_code"])
        self.assertEqual(len(final_records.records), 1)
        self.assertIn("live_finalization_diagnostic=", final_records.records[0].getMessage())
        self.assertTrue(session.shutdown_confirmed)
        self.assertFalse(session._tasks)

    async def test_known_callsite_codes_remain_distinct_without_claiming_unknown_ownership(self):
        session = await self.fixture.make()
        known = (
            "client_requested_close", "client_disconnected", "source_retirement_requires_reconnect",
            "live_control_capacity_exceeded", "live_interrupt_requires_fenced_replacement",
            "live_readonly_remote_source_revoked", "live_source_context_revoked",
            "live_context_requires_reconstruction", "remote_context_expired_requires_replacement",
            "context_replacement_requires_fenced_reconnect", "live_session_worker_failed",
        )
        for reason in known:
            with self.subTest(reason=reason):
                # Pure first-close recorder fixture: no route/authority claim.
                owner = SimpleNamespace(_close_reason=None, _closed=asyncio.Event(), context=session.context)
                with self.assertLogs(LOGGER, level="WARNING") as captured:
                    type(session)._mark_closing(owner, reason)
                row, = records(captured)
                self.assertEqual(row["reason"], reason)
                self.assertIsNone(row["cause_code"])
                self.assertIsNone(row["cause_type"])
                self.assertIsNone(row["route_id"])
                self.assertNotIn("ownership_lost", row)
                self.assertTrue(owner._closed.is_set())

    async def test_source_timeout_remains_failure_not_revocation_proof(self):
        session = await self.fixture.make()
        session._source_context = SimpleNamespace(wait_for_invalidation=AsyncMock(side_effect=TimeoutError("PRIVATE")))
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            await session._watch_sources()
        row, = records(captured)
        self.assertEqual(row["reason"], "live_source_context_revoked")
        self.assertEqual(row["cause_type"], "TimeoutError")
        self.assertIsNone(row["cause_code"])
        self.assertNotIn("PRIVATE", json.dumps(row))
        self.assertTrue(session._closed.is_set())

    async def test_source_watch_cancellation_still_propagates_without_close_record(self):
        session = await self.fixture.make()
        session._source_context = SimpleNamespace(wait_for_invalidation=AsyncMock(side_effect=asyncio.CancelledError()))
        with patch("app.realtime.live_session._CLOSE_LOGGER.warning") as emit:
            with self.assertRaises(asyncio.CancelledError):
                await session._watch_sources()
        emit.assert_not_called()
        self.assertFalse(session._closed.is_set())

    async def test_worker_failure_preserves_safe_type_and_existing_close_policy(self):
        session = await self.fixture.make()

        async def failing():
            raise ValueError("PRIVATE_WORKER_ERROR")

        with self.assertLogs(LOGGER, level="WARNING") as captured:
            task = session._spawn(failing(), "fixture-failing-worker")
            with self.assertRaises(ValueError):
                await task
            await self.fixture.until(lambda: session._closed.is_set())
        row, = records(captured)
        self.assertEqual(session.failure, "live_session_worker_failed")
        self.assertEqual(row["reason"], session.failure)
        self.assertEqual(row["cause_type"], "ValueError")
        self.assertNotIn("PRIVATE", json.dumps(row))
        self.assertFalse(session._tasks)

    async def test_first_close_only_and_private_values_redacted_without_mutating_reason(self):
        session = await self.fixture.make()
        error = SourceError("PRIVATE_SOURCE_CODE", "PRIVATE_EXCEPTION_TEXT")
        with self.assertLogs(LOGGER, level="WARNING") as captured:
            session._mark_closing("PRIVATE_REASON", cause=error)
            for _ in range(100):
                session._mark_closing("live_context_requires_reconstruction")
        row, = records(captured)
        self.assertEqual(session._close_reason, "PRIVATE_REASON")
        self.assertEqual(row["reason"], "unrecognized_code")
        self.assertEqual(row["cause_code"], "unrecognized_code")
        self.assertEqual(row["cause_type"], "SourceError")
        self.assertNotIn("PRIVATE", json.dumps(row))
        self.assertLess(len(json.dumps(row)), 1200)

    async def test_diagnostic_failure_cannot_prevent_close_or_overwrite_first_reason(self):
        session = await self.fixture.make()
        with patch("app.realtime.live_session._CLOSE_LOGGER.warning", side_effect=RuntimeError("PRIVATE")) as emit:
            session._mark_closing("live_context_requires_reconstruction")
            session._mark_closing("live_session_worker_failed")
        self.assertEqual(emit.call_count, 1)
        self.assertTrue(session._closed.is_set())
        self.assertEqual(session._close_reason, "live_context_requires_reconstruction")


class LiveCloseControlProjectionTests(unittest.TestCase):
    def command(self, model=CloseRouteCommand, reason="remote_context_expired_requires_replacement"):
        route = RouteIdentity.model_validate(startup_wire_fixtures()["starting"]["payload"]["route"])
        return model(route=route, command_id=uuid4(), requested_at=datetime.now(timezone.utc), reason=reason)

    def test_close_and_interrupt_retain_exact_typed_command_not_delivery_proof(self):
        for kind, model in (("voice_route.close", CloseRouteCommand), ("voice_route.interrupt", InterruptSpeechCommand)):
            with self.subTest(kind=kind):
                command = self.command(model)
                payload = {"type": kind, "payload": command.model_dump(mode="json")}
                before = json.dumps(payload, sort_keys=True)
                row = native_route_observation(payload)
                self.assertEqual(row["route_control"], {
                    "schema_valid": True, "route_id": str(command.route.route_id), "route_generation": 1,
                    "command_id": str(command.command_id), "requested_at": command.requested_at.isoformat(),
                    "reason": "remote_context_expired_requires_replacement"})
                self.assertEqual(json.dumps(payload, sort_keys=True), before)
                self.assertNotIn("PRIVATE", json.dumps(row))
                self.assertNotIn("output_fenced", row["route_control"])
                self.assertNotIn("provider_stopped", row["route_control"])

    def test_unknown_reason_and_malformed_payload_are_not_echoed(self):
        command = self.command(reason="PRIVATE_IDENTIFIER_SHAPED_REASON")
        row = native_route_observation({"type": "voice_route.close", "payload": command.model_dump(mode="json")})
        self.assertEqual(row["route_control"]["reason"], "unrecognized_code")
        self.assertNotIn("PRIVATE", json.dumps(row))
        for value in (None, [], {"PRIVATE": "PRIVATE"}, command.model_dump(mode="json") | {"extra": "PRIVATE"}):
            with self.subTest(value_type=type(value).__name__):
                row = native_route_observation({"type": "voice_route.close", "payload": value})
                self.assertEqual(row["route_control"], {"schema_valid": False})
                self.assertNotIn("PRIVATE", json.dumps(row))
