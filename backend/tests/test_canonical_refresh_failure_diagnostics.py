"""Offline real-worker failure metadata; no canonical DB or provider calls."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from app.knowledge.models import SourceError
from app.live_state.models import SessionScope, StateCursor
from app.live_state.refresh import CanonicalRefreshWorker
from app.realtime import live_state_bridge as bridge
from app.runtime.events import DomainEventBus, DomainEventName
from app.voice_routes.contracts import RouteIdentity, VoiceContextUpdate


PRIVATE = "PRIVATE_REFRESH_BODY_Bearer_transcript_sdp"
NOW = datetime(2026, 10, 31, 23, tzinfo=timezone.utc)


class CanonicalRefreshFailureDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="fixture-owner", session_id=UUID(int=1), area_id=UUID(int=2))
        self.update = VoiceContextUpdate(
            route=RouteIdentity(scope=self.scope, route_id=UUID(int=3), route_generation=1),
            update_id=UUID(int=4), cursor=StateCursor(epoch=UUID(int=5), revision=1),
            mode="snapshot", text=PRIVATE, created_at=NOW, expires_at=NOW + timedelta(seconds=10))

    def worker(self, read, apply, invalidate, **options):
        return CanonicalRefreshWorker(scope=self.scope, bus=DomainEventBus(),
            event_names=(DomainEventName.OWNER_WORKING_MEMORY_UPDATED,),
            read=read, apply=apply, invalidate=invalidate,
            **({"interval_seconds": 60, "timeout_seconds": 1} | options))

    async def rejected_record(self, worker):
        @asynccontextmanager
        async def guard(update):
            raise SourceError("live_state_runtime_unavailable", PRIVATE)
            yield

        live = SimpleNamespace(_healthy=False, requires_reconstruction=False, worker=worker)
        check = bridge._ContextValidityCheck(guard, live, lambda: NOW)
        with patch.object(bridge._VALIDITY_LOGGER, "warning") as warning:
            self.assertIs(await check(self.update), False)
        self.assertEqual(warning.call_count, 1)
        raw = warning.call_args.args[1]
        self.assertNotIn(PRIVATE, raw)
        return json.loads(raw)

    async def failure_case(self, stage, error=None, *, deadline=False):
        invalidated, released = asyncio.Event(), asyncio.Event()
        calls, reasons = [], []
        value = object()

        async def fail():
            try:
                if deadline:
                    await asyncio.Event().wait()
                raise error
            finally:
                released.set()

        async def read():
            calls.append("read")
            if stage == "read":
                await fail()
            return value

        async def apply(actual):
            calls.append("apply")
            self.assertIs(actual, value)
            if stage == "apply":
                await fail()

        async def invalidate(reason):
            reasons.append(reason)
            invalidated.set()

        worker = self.worker(read, apply, invalidate, timeout_seconds=0.02 if deadline else 1)
        worker.start()
        try:
            await asyncio.wait_for(invalidated.wait(), 1)
            self.assertTrue(released.is_set())
            self.assertEqual(reasons, ["canonical_refresh_unavailable"])
            self.assertEqual(calls, ["read"] if stage == "read" else ["read", "apply"])
            self.assertEqual(worker.failure_count, 1)
            self.assertEqual(worker.refresh_count, 0)
            self.assertEqual(worker.last_error_stage, stage)
            return worker.last_error, worker.last_error_code, await self.rejected_record(worker)
        finally:
            await worker.close()
            self.assertTrue(worker._task.done())

    async def test_actual_read_and_apply_failures_reach_existing_validity_diagnostic(self):
        cases = (
            (TimeoutError(PRIVATE), "TimeoutError", "TimeoutError", None),
            (SourceError("live_state_context_revoked", PRIVATE), "SourceError", "SourceError", "live_state_context_revoked"),
            (SourceError(PRIVATE, PRIVATE), "SourceError", "SourceError", "unrecognized_source_error"),
            (ValueError(PRIVATE), "ValueError", "unrecognized_error", None),
        )
        for stage in ("read", "apply"):
            for error, stored_type, diagnostic_type, code in cases:
                with self.subTest(stage=stage, kind=stored_type, code=code):
                    actual_type, actual_code, record = await self.failure_case(stage, error)
                    self.assertEqual(actual_type, stored_type)
                    self.assertEqual(actual_code, code)
                    self.assertEqual(record["refresh_error_type"], diagnostic_type)
                    self.assertEqual(record["refresh_error_stage"], stage)
                    self.assertEqual(record["refresh_error_code"], code)
                    self.assertEqual(record["code"], "live_state_runtime_unavailable")
                    self.assertFalse(record["live_healthy"])
                    self.assertFalse(record["expired_at_report"])

    async def test_real_deadline_records_read_or_apply_without_changing_invalidation(self):
        for stage in ("read", "apply"):
            with self.subTest(stage=stage):
                kind, code, record = await self.failure_case(stage, deadline=True)
                self.assertEqual(kind, "TimeoutError")
                self.assertIsNone(code)
                self.assertEqual(record["refresh_error_type"], "TimeoutError")
                self.assertEqual(record["refresh_error_stage"], stage)

    async def test_failure_code_accessor_cannot_interrupt_invalidation(self):
        class HostileSourceError(SourceError):
            def __getattribute__(self, name):
                if name == "code":
                    raise RuntimeError(PRIVATE)
                return super().__getattribute__(name)

        kind, code, record = await self.failure_case("read", HostileSourceError("ignored", PRIVATE))
        self.assertEqual(kind, "HostileSourceError")
        self.assertEqual(code, "unrecognized_source_error")
        self.assertEqual(record["refresh_error_type"], "unrecognized_error")
        self.assertIsNone(record["refresh_error_code"])

    async def test_successful_refresh_clears_previous_failure_metadata(self):
        second_read, release, applied = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls, reasons = [], []

        async def read():
            calls.append("read")
            if len(calls) == 1:
                raise SourceError("live_state_event_changed", PRIVATE)
            if len(calls) == 2:
                second_read.set()
                await release.wait()
                return 42
            await asyncio.Event().wait()

        async def apply(value):
            self.assertEqual(value, 42)
            applied.set()

        async def invalidate(reason):
            reasons.append(reason)

        worker = self.worker(read, apply, invalidate, interval_seconds=0.001)
        worker.start()
        try:
            await asyncio.wait_for(second_read.wait(), 1)
            self.assertEqual((worker.last_error, worker.last_error_stage, worker.last_error_code),
                ("SourceError", "read", "live_state_event_changed"))
            release.set()
            await asyncio.wait_for(applied.wait(), 1)
            self.assertEqual(worker.refresh_count, 1)
            self.assertEqual(worker.failure_count, 1)
            self.assertEqual(reasons, ["canonical_refresh_unavailable"])
            self.assertEqual((worker.last_error, worker.last_error_stage, worker.last_error_code), (None, None, None))
        finally:
            await worker.close()
            self.assertTrue(worker._task.done())

    async def test_close_cancellation_is_joined_and_not_mislabeled_as_refresh_failure(self):
        for stage in ("read", "apply"):
            with self.subTest(stage=stage):
                entered, released = asyncio.Event(), asyncio.Event()
                reasons = []

                async def pause():
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        released.set()

                async def read():
                    if stage == "read":
                        await pause()
                    return 42

                async def apply(value):
                    if stage == "apply":
                        await pause()

                async def invalidate(reason):
                    reasons.append(reason)

                worker = self.worker(read, apply, invalidate)
                worker.start()
                try:
                    await asyncio.wait_for(entered.wait(), 1)
                finally:
                    await worker.close()
                self.assertTrue(worker._task.done())
                self.assertTrue(released.is_set())
                self.assertEqual(reasons, [])
                self.assertEqual(worker.failure_count, 0)
                self.assertEqual((worker.last_error, worker.last_error_stage, worker.last_error_code), (None, None, None))
