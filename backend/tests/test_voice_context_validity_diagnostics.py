"""Offline rejection diagnostics, not canonical DB or native audio qualification."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from app.knowledge.models import SourceError
from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency, StateCursor
from app.realtime import live_state_bridge as bridge
from app.voice_routes.contracts import RouteIdentity, VoiceContextUpdate


NOW = datetime(2026, 10, 31, 8, tzinfo=timezone.utc)
PRIVATE = "PRIVATE_SECRET_Bearer_transcript_sdp"
FIELDS = {"stage", "route_id", "route_generation", "update_id", "cursor_revision",
          "error_type", "code", "guard_kind", "expired_at_report", "live_healthy",
          "requires_reconstruction", "refresh_error_type", "refresh_error_stage", "refresh_error_code"}


class VoiceContextValidityDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="fixture-owner", session_id=UUID(int=1), area_id=UUID(int=2))
        self.update = VoiceContextUpdate(
            route=RouteIdentity(scope=self.scope, route_id=UUID(int=3), route_generation=7),
            update_id=UUID(int=4), cursor=StateCursor(epoch=UUID(int=5), revision=63),
            mode="snapshot", text=PRIVATE, created_at=NOW, expires_at=NOW + timedelta(seconds=10))
        self.binding = GuardedPayloadBinding(scope=self.scope, payload_root_ids=(UUID(int=6),),
            dependencies=(SourceDependency(source_namespace="context_event", source_id=UUID(int=7), revision=1),))
        self.live = SimpleNamespace(_healthy=True, requires_reconstruction=False,
                                    worker=SimpleNamespace(last_error=None))
        self.warning_patch = patch.object(bridge._VALIDITY_LOGGER, "warning")
        self.warning = self.warning_patch.start()
        self.addCleanup(self.warning_patch.stop)
        self.entered = []
        self.exited = []

    def guard(self, error=None, *, after_yield=False):
        @asynccontextmanager
        async def guarded(update):
            self.entered.append(update)
            try:
                if error is not None and not after_yield:
                    raise error
                yield
                if error is not None:
                    raise error
            finally:
                self.exited.append(update)
        return guarded

    def check(self, error=None, *, guard=None, live=None, clock=None):
        return bridge._ContextValidityCheck(
            self.guard(error) if guard is None else guard,
            self.live if live is None else live, (lambda: NOW) if clock is None else clock)

    def records(self):
        records = []
        for call in self.warning.call_args_list:
            self.assertEqual(call.args[0], "voice_context_guard_diagnostic %s")
            self.assertEqual(len(call.args), 2)
            self.assertEqual(call.kwargs, {})
            self.assertNotIn(PRIVATE, call.args[1])
            records.append(json.loads(call.args[1]))
        return records

    def rejection(self):
        record = self.records()[-1]
        self.assertEqual(set(record), FIELDS)
        self.assertEqual(record["stage"], "validity_rejected")
        self.assertEqual(record["route_id"], str(self.update.route.route_id))
        self.assertEqual(record["route_generation"], 7)
        self.assertEqual(record["update_id"], str(self.update.update_id))
        self.assertEqual(record["cursor_revision"], 63)
        return record

    async def test_valid_guard_true_no_rejection_log_and_same_update(self):
        self.assertIs(await self.check()(self.update), True)
        self.assertEqual(self.entered, [self.update])
        self.assertEqual(self.exited, [self.update])
        self.warning.assert_not_called()

    async def test_actual_source_errors_preserve_false_and_static_codes(self):
        for code in ("voice_context_expired", "knowledge_source_changed", "live_state_reconstruction_required"):
            with self.subTest(code=code):
                self.assertIs(await self.check(SourceError(code, PRIVATE))(self.update), False)
                record = self.rejection()
                self.assertEqual(record["error_type"], "SourceError")
                self.assertEqual(record["code"], code)
                self.assertFalse(record["expired_at_report"])
        self.assertEqual(self.entered, self.exited)

    async def test_source_rejection_on_guard_exit_cannot_return_true(self):
        guard = self.guard(SourceError("voice_context_expired", PRIVATE), after_yield=True)
        self.assertIs(await self.check(guard=guard)(self.update), False)
        self.assertEqual(self.rejection()["code"], "voice_context_expired")
        self.assertEqual(self.entered, self.exited)

    async def test_actual_inner_deadline_preserves_false_and_timeout_code(self):
        @asynccontextmanager
        async def timed_guard(update):
            async with asyncio.timeout(0):
                await asyncio.sleep(0)
            yield

        self.assertIs(await self.check(guard=timed_guard)(self.update), False)
        record = self.rejection()
        self.assertEqual(record["error_type"], "TimeoutError")
        self.assertEqual(record["code"], "guard_timeout")

    async def test_timeout_message_is_not_logged(self):
        self.assertIs(await self.check(TimeoutError(PRIVATE))(self.update), False)
        self.assertEqual(self.rejection()["code"], "guard_timeout")

    async def test_actual_caller_cancellation_propagates_and_releases_guard(self):
        entered, exited = asyncio.Event(), asyncio.Event()

        @asynccontextmanager
        async def waiting_guard(update):
            entered.set()
            try:
                await asyncio.Event().wait()
                yield
            finally:
                exited.set()

        task = asyncio.create_task(self.check(guard=waiting_guard)(self.update))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(exited.is_set())
            self.warning.assert_not_called()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_unexpected_guard_errors_propagate_on_entry_and_exit(self):
        for after_yield in (False, True):
            with self.subTest(after_yield=after_yield):
                error = RuntimeError(PRIVATE)
                with self.assertRaises(RuntimeError) as caught:
                    await self.check(guard=self.guard(error, after_yield=after_yield))(self.update)
                self.assertIs(caught.exception, error)
        self.assertEqual(self.entered, self.exited)
        self.warning.assert_not_called()

    async def test_unknown_source_codes_are_redacted_not_syntactically_trusted(self):
        for code in (PRIVATE, "knowledge_source_changed_" + PRIVATE, "future_source_error"):
            with self.subTest(code_kind=code == "future_source_error"):
                self.assertIs(await self.check(SourceError(code, PRIVATE))(self.update), False)
                self.assertEqual(self.rejection()["code"], "unrecognized_source_error")

    async def test_health_and_refresh_error_types_are_bounded_and_explicit(self):
        class PrivateRefreshFailure(Exception):
            def __str__(self):
                raise AssertionError("Diagnostic must not stringify a refresh exception")

        cases = (
            (True, False, None, True, False, None),
            (False, True, "TimeoutError", False, True, "TimeoutError"),
            (False, True, "SourceError", False, True, "SourceError"),
            (False, False, PRIVATE, False, False, "unrecognized_error"),
            (PRIVATE, 1, PrivateRefreshFailure(), None, None, "unrecognized_error"),
        )
        for healthy, reconstruct, refresh, expected_health, expected_reconstruct, expected_refresh in cases:
            with self.subTest(refresh=expected_refresh):
                live = SimpleNamespace(_healthy=healthy, requires_reconstruction=reconstruct,
                                       worker=SimpleNamespace(last_error=refresh))
                self.assertIs(await self.check(SourceError("live_state_scope_changed", PRIVATE), live=live)(self.update), False)
                record = self.rejection()
                self.assertIs(record["live_healthy"], expected_health)
                self.assertIs(record["requires_reconstruction"], expected_reconstruct)
                self.assertEqual(record["refresh_error_type"], expected_refresh)

    async def test_refresh_stage_and_source_code_are_static_allowlisted_metadata(self):
        cases = (
            ("TimeoutError", "read", None, "read", None),
            ("TimeoutError", "apply", "knowledge_source_deleted", "apply", None),
            ("SourceError", "apply", "knowledge_source_deleted", "apply", "knowledge_source_deleted"),
            ("SourceError", "read", "live_state_projection_unavailable", "read", "live_state_projection_unavailable"),
            ("SourceError", PRIVATE, PRIVATE, "unrecognized_stage", "unrecognized_source_error"),
            ("SourceError", [PRIVATE], [PRIVATE], "unrecognized_stage", "unrecognized_source_error"),
            ("SourceError", None, None, None, None),
        )
        for kind, stage, code, expected_stage, expected_code in cases:
            with self.subTest(kind=kind, expected_stage=expected_stage, expected_code=expected_code):
                live = SimpleNamespace(_healthy=False, requires_reconstruction=False,
                    worker=SimpleNamespace(last_error=kind, last_error_stage=stage, last_error_code=code))
                self.assertIs(await self.check(SourceError("live_state_runtime_unavailable", PRIVATE), live=live)(self.update), False)
                record = self.rejection()
                self.assertEqual(record["refresh_error_stage"], expected_stage)
                self.assertEqual(record["refresh_error_code"], expected_code)

    async def test_missing_optional_worker_health_is_unknown_not_healthy(self):
        live = SimpleNamespace(requires_reconstruction=False)
        self.assertIs(await self.check(SourceError("live_state_scope_unavailable", PRIVATE), live=live)(self.update), False)
        record = self.rejection()
        self.assertIsNone(record["live_healthy"])
        self.assertIsNone(record["refresh_error_type"])
        self.assertIsNone(record["refresh_error_stage"])
        self.assertIsNone(record["refresh_error_code"])

    async def test_expiry_and_scope_vs_payload_roots_are_reported_not_redecided(self):
        for binding, kind in ((None, "scope"), (self.binding, "payload_roots")):
            for expiry, expired in ((None, False), (NOW + timedelta(seconds=1), False),
                                    (NOW, True), (NOW - timedelta(seconds=1), True)):
                with self.subTest(kind=kind, expired=expired, no_expiry=expiry is None):
                    # Diagnostic-only inputs: the supplied guard remains the decision owner.
                    update = self.update.model_copy(update={"guard_binding": binding, "expires_at": expiry})
                    self.assertIs(await self.check(SourceError("live_state_source_changed", PRIVATE))(update), False)
                    record = self.rejection()
                    self.assertEqual(record["guard_kind"], kind)
                    self.assertIs(record["expired_at_report"], expired)
        self.warning.reset_mock()
        expired = self.update.model_copy(update={"expires_at": NOW - timedelta(seconds=1)})
        self.assertIs(await self.check()(expired), True)
        self.warning.assert_not_called()

    async def test_cap_is_32_rejections_plus_one_explicit_truncation_marker(self):
        check = self.check(SourceError("live_state_context_revoked", PRIVATE))
        for _ in range(70):
            self.assertIs(await check(self.update), False)
        records = self.records()
        self.assertEqual(len(records), 33)
        self.assertTrue(all(set(row) == FIELDS for row in records[:32]))
        self.assertEqual(records[-1], {"stage": "truncated", "record_limit": 32,
                                      "absence_after_limit_is_inconclusive": True})
        other = self.check(SourceError("live_state_context_revoked", PRIVATE))
        self.assertIs(await other(self.update), False)
        self.assertEqual(self.records()[-1]["stage"], "validity_rejected")

    async def test_logging_failure_does_not_change_rejection_or_bypass_cap(self):
        self.warning.side_effect = RuntimeError(PRIVATE)
        check = self.check(SourceError("knowledge_source_deleted", PRIVATE))
        for _ in range(70):
            self.assertIs(await check(self.update), False)
        self.assertEqual(self.warning.call_count, 33)
        self.assertEqual(self.records()[-1]["stage"], "truncated")

    async def test_diagnostic_clock_failure_does_not_change_rejection(self):
        def failed_clock():
            raise RuntimeError(PRIVATE)

        self.assertIs(await self.check(SourceError("voice_context_expired", PRIVATE), clock=failed_clock)(self.update), False)
        self.warning.assert_not_called()
