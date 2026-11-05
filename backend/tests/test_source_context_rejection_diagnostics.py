"""Offline source-lock boundary evidence; fake SQL, no predicate reimplementation."""
import asyncio
from contextlib import asynccontextmanager
import json
import unittest
from unittest.mock import patch
from uuid import UUID

from app.realtime import source_context as module
from app.realtime.source_context import LiveContextError, LiveResponseSources, ResponseSourceSnapshot


PRIVATE = "PRIVATE_OWNER_Bearer_secret_transcript"
ROOTS = (UUID(int=10), UUID(int=20))
FIELDS = {"session_id", "audio_session_id", "stage", "code", "reason", "attempted_root_ids",
          "attempted_root_count", "retained_root_count", "root_limit", "omitted_root_count", "roots_truncated"}


class Connection:
    def __init__(self, available=False, error=None):
        self.available, self.error = available, error
        self.calls = []
        self.transactions_entered = self.transactions_exited = 0
        self.before_return = None

    def is_in_transaction(self):
        return True

    async def fetchval(self, query, owner, roots):
        self.calls.append((query, owner, tuple(roots)))
        if self.before_return is not None:
            await self.before_return()
        if self.error is not None:
            raise self.error
        return self.available

    @asynccontextmanager
    async def transaction(self):
        self.transactions_entered += 1
        try:
            yield
        finally:
            self.transactions_exited += 1


class Pool:
    def __init__(self, connection):
        self.connection = connection
        self.acquired = self.released = 0

    @asynccontextmanager
    async def acquire(self):
        self.acquired += 1
        try:
            yield self.connection
        finally:
            self.released += 1


class SourceContextRejectionDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = Connection()
        self.pool = Pool(self.connection)
        self.sources = LiveResponseSources(self.pool, owner_id=PRIVATE, session_id=UUID(int=1),
                                           audio_session_id=str(UUID(int=2)))
        self.warning_patch = patch.object(module._LOCK_LOGGER, "warning")
        self.warning = self.warning_patch.start()
        self.addCleanup(self.warning_patch.stop)

    def records(self):
        result = []
        for call in self.warning.call_args_list:
            self.assertEqual(call.args[0], "source_context_lock_rejection_diagnostic %s")
            self.assertEqual(len(call.args), 2)
            self.assertEqual(call.kwargs, {})
            self.assertNotIn(PRIVATE, call.args[1])
            self.assertLess(len(call.args[1]), 6500)
            result.append(json.loads(call.args[1]))
        return result

    def assert_attempt(self, roots):
        self.assertEqual(self.connection.calls, [
            ("select lock_jarvis_copy_sources($1,$2::uuid[])", PRIVATE, tuple(roots))])
        record = self.records()[-1]
        self.assertEqual(set(record), FIELDS)
        self.assertEqual(record["reason"], "lock_predicate_false")
        self.assertEqual(record["code"], "live_context_revoked")
        self.assertEqual(record["attempted_root_ids"], [str(root) for root in roots])
        self.assertEqual(record["attempted_root_count"], len(roots))
        self.assertEqual(record["retained_root_count"], len(roots))
        self.assertEqual(record["omitted_root_count"], 0)
        self.assertFalse(record["roots_truncated"])
        self.assertEqual(record["session_id"], str(UUID(int=1)))
        self.assertEqual(record["audio_session_id"], str(UUID(int=2)))

    async def test_false_lock_logs_exact_attempt_without_extra_query_and_rejects(self):
        self.sources.observe(ROOTS[:1])
        with self.assertRaises(LiveContextError) as caught:
            async with self.sources.context_guard(connection=self.connection, extra_sources=ROOTS[1:]):
                self.fail("Rejected guard must not admit a provider write")
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assert_attempt(ROOTS)
        self.assertEqual(self.pool.acquired, 0)
        self.assertEqual(self.sources._sources, set(ROOTS[:1]))

    async def test_normal_true_lock_uses_same_single_query_and_logs_nothing(self):
        self.connection.available = True
        self.sources.observe(ROOTS)
        async with self.sources.context_guard(connection=self.connection):
            self.assertEqual(len(self.connection.calls), 1)
        self.assertEqual(self.connection.calls[0],
            ("select lock_jarvis_copy_sources($1,$2::uuid[])", PRIVATE, ROOTS))
        self.warning.assert_not_called()

    async def test_empty_context_still_yields_without_query_or_diagnostic(self):
        async with self.sources.context_guard():
            pass
        self.assertEqual(self.pool.acquired, 0)
        self.assertEqual(self.connection.calls, [])
        self.warning.assert_not_called()

    async def test_source_watcher_preserves_revocation_and_transaction_release(self):
        self.sources.observe(ROOTS)
        with self.assertRaises(LiveContextError) as caught:
            await self.sources.wait_for_invalidation(interval=0.01, timeout=1)
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assert_attempt(ROOTS)
        self.assertEqual((self.pool.acquired, self.pool.released), (1, 1))
        self.assertEqual((self.connection.transactions_entered, self.connection.transactions_exited), (1, 1))

    async def test_prepare_rejection_never_writes_payload_or_response_receipt(self):
        self.sources.observe(ROOTS)
        with self.assertRaises(LiveContextError):
            await self.sources.prepare("fixture-request")
        self.assert_attempt(ROOTS)
        self.assertIsNone(self.sources.source_for("fixture-request"))
        self.assertEqual(self.pool.released, 1)

    async def test_response_guard_includes_materialized_payload_and_reuses_transaction(self):
        self.sources.max_sources = 2
        self.sources.observe(ROOTS)
        payload = UUID(int=30)
        snapshot = ResponseSourceSnapshot(payload, ROOTS, self.sources._version)
        async with self.sources.use_transaction(self.connection):
            with self.assertRaises(LiveContextError):
                async with self.sources.guard(snapshot):
                    self.fail("Rejected response must not be forwarded")
        self.assert_attempt((*ROOTS, payload))
        self.assertEqual(self.records()[-1]["root_limit"], 3)
        self.assertEqual(self.pool.acquired, 0)
        self.assertIsNone(self.sources._current_connection())

    async def test_default_full_budget_plus_response_root_is_not_truncated(self):
        roots = tuple(UUID(int=index) for index in range(1, 129))
        self.sources.observe(roots)
        payload = UUID(int=129)
        snapshot = ResponseSourceSnapshot(payload, roots, self.sources._version)
        with self.assertRaises(LiveContextError):
            async with self.sources.guard(snapshot):
                self.fail("Rejected response must not be forwarded")
        self.assert_attempt((*roots, payload))

    async def test_larger_configured_budget_is_not_changed_and_omission_is_explicit(self):
        self.sources.max_sources = 256
        roots = tuple(UUID(int=index) for index in range(1, 201))
        self.sources.observe(roots)
        with self.assertRaises(LiveContextError):
            async with self.sources.context_guard(connection=self.connection):
                self.fail("Rejected guard admitted")
        self.assertEqual(len(self.sources._sources), 200)
        self.assertEqual(self.connection.calls[0][2], roots)
        record = self.records()[-1]
        self.assertEqual(record["attempted_root_count"], 200)
        self.assertEqual(record["retained_root_count"], 129)
        self.assertEqual(record["omitted_root_count"], 71)
        self.assertTrue(record["roots_truncated"])
        self.assertEqual(record["attempted_root_ids"], [str(root) for root in roots[:129]])

    async def test_concurrent_observe_does_not_replace_attempted_root_evidence(self):
        self.sources.observe(ROOTS[:1])

        async def addition():
            self.sources.observe(ROOTS[1:])

        self.connection.before_return = addition
        with self.assertRaises(LiveContextError) as caught:
            async with self.sources.context_guard(connection=self.connection):
                self.fail("False lock cannot become a benign-addition retry")
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assert_attempt(ROOTS[:1])
        self.assertEqual(self.sources._sources, set(ROOTS))

    async def test_preexisting_version_mismatch_and_source_limit_do_not_log_false_sql(self):
        self.sources.max_sources = 1
        with self.assertRaises(LiveContextError) as caught:
            self.sources.observe(ROOTS)
        self.assertEqual(caught.exception.code, "live_context_source_limit")
        self.sources.observe(ROOTS[:1])
        stale = ResponseSourceSnapshot(UUID(int=30), ROOTS[:1], self.sources._version - 1)
        with self.assertRaises(LiveContextError) as caught:
            async with self.sources.guard(stale):
                self.fail("Stale snapshot admitted")
        self.assertEqual(caught.exception.code, "live_context_changed")
        self.assertEqual(self.connection.calls, [])
        self.warning.assert_not_called()

    async def test_fetch_exception_or_cancellation_is_not_reclassified_as_false(self):
        for error in (RuntimeError(PRIVATE), TimeoutError(PRIVATE), asyncio.CancelledError()):
            with self.subTest(error_type=type(error).__name__):
                connection = Connection(error=error)
                with self.assertRaises(type(error)) as caught:
                    await self.sources._lock(connection, ROOTS)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(connection.calls), 1)
        self.warning.assert_not_called()

    async def test_null_remains_fail_closed_without_fabricating_boolean_false_receipt(self):
        self.connection.available = None
        with self.assertRaises(LiveContextError) as caught:
            await self.sources._lock(self.connection, ROOTS)
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.warning.assert_not_called()

    async def test_cap_is_eight_records_plus_one_marker_and_all_attempts_still_reject(self):
        for _ in range(25):
            with self.assertRaises(LiveContextError):
                await self.sources._lock(self.connection, ROOTS)
        records = self.records()
        self.assertEqual(len(records), 9)
        self.assertTrue(all(set(record) == FIELDS for record in records[:8]))
        self.assertEqual(records[-1], {"session_id": str(UUID(int=1)), "audio_session_id": str(UUID(int=2)),
            "stage": "truncated", "record_limit": 8, "absence_after_limit_is_inconclusive": True})
        self.assertEqual(len(self.connection.calls), 25)

    async def test_logging_failure_cannot_replace_revocation_or_disable_cap(self):
        self.warning.side_effect = RuntimeError(PRIVATE)
        for _ in range(25):
            with self.assertRaises(LiveContextError) as caught:
                await self.sources._lock(self.connection, ROOTS)
            self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assertEqual(self.warning.call_count, 9)
        self.assertEqual(len(self.connection.calls), 25)
        self.records()

    async def test_non_uuid_identity_and_unsafe_internal_root_are_never_logged(self):
        self.sources.session_id = PRIVATE
        self.sources.audio_session_id = PRIVATE
        with self.assertRaises(LiveContextError):
            await self.sources._lock(self.connection, [ROOTS[0], PRIVATE])
        record = self.records()[-1]
        self.assertIsNone(record["session_id"])
        self.assertIsNone(record["audio_session_id"])
        self.assertEqual(record["attempted_root_ids"], [str(ROOTS[0])])
        self.assertEqual(record["omitted_root_count"], 1)
        self.assertTrue(record["roots_truncated"])

    async def test_single_use_source_iterable_is_materialized_once_for_sql_and_diagnostic(self):
        with self.assertRaises(LiveContextError):
            await self.sources._lock(self.connection, iter(ROOTS))
        self.assert_attempt(ROOTS)
