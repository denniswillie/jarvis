"""Offline timing evidence at the real source/writer boundaries, not playback.

External socket/storage are controlled. No PostgreSQL predicate, provider ACK,
native media, useful-answer latency or user hearing is qualified here.
"""

import asyncio
from contextlib import asynccontextmanager
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.realtime.telemetry import RealtimeTrace, latency_event
from test_response_preparation_source_race import Connection, Pool, Socket


class TransactionConnection(Connection):
    def is_in_transaction(self):
        return self.entered > self.exited


class RealtimeWriteLatencyBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        capture = patch("app.realtime.telemetry.LOGGER.info")
        self.logger = capture.start()
        self.addCleanup(capture.stop)
        self.tasks = []
        self.writers = []
        self.manual_locks = []

    async def asyncTearDown(self):
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for lock in self.manual_locks:
            if lock.locked():
                lock.release()
        for writer in self.writers:
            self.assertEqual(writer._foreground_waiters, 0)
            self.assertFalse(writer._send_lock.locked())
        self.assertTrue(all(task.done() for task in self.tasks))

    def trace(self):
        return RealtimeTrace(provider="openai", audio_session_id=str(uuid4()),
                             jarvis_session_id=uuid4())

    def writer(self, socket=None):
        writer = OpenAIRealtimeConnection(
            settings=SimpleNamespace(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=5),
            owner_id="offline-latency-boundary")
        writer._websocket = socket if socket is not None else Socket()
        writer.bind_trace(self.trace())
        self.writers.append(writer)
        return writer

    def start(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.append(task)
        return task

    def rows(self, event=None, phase=None, status=None):
        rows = [json.loads(call.args[1]) for call in self.logger.call_args_list
                if len(call.args) > 1 and call.args[0] == "realtime_trace %s"]
        return [row for row in rows if (event is None or row.get("event") == event)
                and (phase is None or row.get("phase") == phase)
                and (status is None or row.get("status") == status)]

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    def terminal(self):
        rows = [row for row in self.rows("provider_write_latency", "operation")
                if row["status"] != "started"]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def payload(self):
        return {"type": "response.create", "event_id": "latency-request",
                "response": {"instructions": "PRIVATE-PROMPT-NOT-TELEMETRY",
                             "metadata": {"jarvis_request_id": "latency-request"}}}

    async def test_controlled_waits_preserve_queue_guard_factory_send_and_cleanup_order(self):
        releases = {name: asyncio.Event() for name in ("guard", "factory", "send", "exit")}
        entered = {name: asyncio.Event() for name in releases}
        calls = []

        class HeldSocket(Socket):
            async def send(inner, raw):
                inner.events.append(json.loads(raw))
                calls.append("send")
                entered["send"].set()
                await releases["send"].wait()

        writer = self.writer(HeldSocket())

        @asynccontextmanager
        async def guard():
            calls.append("guard")
            entered["guard"].set()
            await releases["guard"].wait()
            try:
                yield
            finally:
                calls.append("exit")
                entered["exit"].set()
                await releases["exit"].wait()

        async def factory():
            calls.append("factory")
            entered["factory"].set()
            await releases["factory"].wait()
            return self.payload()

        await writer._send_lock.acquire()
        self.manual_locks.append(writer._send_lock)
        task = self.start(writer.send_event(factory, write_guard=guard))
        await self.until(lambda: self.rows("provider_write_latency", "queue", "started"))
        self.assertEqual(calls, [])
        self.assertFalse(self.rows("provider_write_latency", "queue", "completed"))
        writer._send_lock.release()
        self.manual_locks.remove(writer._send_lock)

        for phase, following in (("guard", "factory"), ("factory", "send"), ("send", "exit")):
            await self.until(entered[phase].is_set)
            self.assertFalse(entered[following].is_set())
            self.assertFalse(task.done())
            releases[phase].set()
        await self.until(entered["exit"].is_set)
        self.assertFalse(task.done())
        self.assertTrue(writer._send_lock.locked())
        releases["exit"].set()
        self.assertIs(await task, True)
        self.assertEqual(calls, ["guard", "factory", "send", "exit"])
        self.assertEqual(writer._websocket.events, [self.payload()])

        rows = self.rows("provider_write_latency")
        self.assertEqual(len({row["dispatch_id"] for row in rows}), 1)
        self.assertEqual([(row["phase"], row["status"]) for row in rows
                          if row["phase"] != "operation"],
                         [(phase, status) for phase in
                          ("queue", "guard_enter", "factory", "encode", "send", "guard_exit")
                          for status in ("started", "completed")])
        for row in rows:
            self.assertTrue(math.isfinite(row["monotonic_s"]))
            if row["status"] != "started":
                self.assertTrue(math.isfinite(row["duration_ms"]))
                self.assertGreaterEqual(row["duration_ms"], 0)
        finished = self.terminal()
        self.assertEqual(finished["status"], "completed")
        self.assertTrue(finished["write_possible"])
        self.assertTrue(finished["send_returned"])
        self.assertTrue(finished["guard_present"])
        self.assertEqual(finished["event_type"], "response.create")
        self.assertNotIn("PRIVATE-PROMPT-NOT-TELEMETRY", json.dumps(rows))

    async def test_failure_and_cancellation_preserve_prewrite_and_possible_write_boundaries(self):
        cases = [("failed", phase) for phase in ("guard", "factory", "send", "exit")]
        cases += [("cancelled", phase) for phase in ("queue", "guard", "factory", "send", "exit")]
        for status, phase in cases:
            with self.subTest(status=status, phase=phase):
                self.logger.reset_mock()
                entered, never = asyncio.Event(), asyncio.Event()
                error = RuntimeError("PRIVATE-ERROR-NOT-TELEMETRY")
                factory_calls = []

                async def boundary(name):
                    if name != phase:
                        return
                    entered.set()
                    if status == "failed":
                        raise error
                    await never.wait()

                class BoundarySocket(Socket):
                    async def send(inner, raw):
                        inner.events.append(json.loads(raw))
                        await boundary("send")

                writer = self.writer(BoundarySocket())

                @asynccontextmanager
                async def guard():
                    await boundary("guard")
                    try:
                        yield
                    finally:
                        await boundary("exit")

                async def factory():
                    factory_calls.append(True)
                    await boundary("factory")
                    return self.payload()

                if phase == "queue":
                    await writer._send_lock.acquire()
                    self.manual_locks.append(writer._send_lock)
                task = self.start(writer.send_event(factory, write_guard=guard))
                if phase == "queue":
                    await self.until(lambda: self.rows("provider_write_latency", "queue", "started"))
                else:
                    await self.until(entered.is_set)
                if status == "cancelled":
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    with self.assertRaises(RuntimeError) as caught:
                        await task
                    self.assertIs(caught.exception, error)
                if phase == "queue":
                    self.assertTrue(writer._send_lock.locked())
                    writer._send_lock.release()
                    self.manual_locks.remove(writer._send_lock)
                self.assertEqual(len(factory_calls), 0 if phase in {"queue", "guard"} else 1)
                self.assertEqual(len(writer._websocket.events), 1 if phase in {"send", "exit"} else 0)
                terminal = self.terminal()
                self.assertEqual(terminal["status"], status)
                self.assertIs(terminal["write_possible"], phase in {"send", "exit"})
                self.assertIs(terminal["send_returned"], phase == "exit")
                self.assertNotIn("PRIVATE-ERROR-NOT-TELEMETRY", json.dumps(self.rows()))
                self.assertEqual(writer._foreground_waiters, 0)
                self.assertFalse(writer._send_lock.locked())

    async def test_no_payload_is_local_false_not_a_submission_or_approval(self):
        writer = self.writer()
        factory = Mock(return_value=None)
        self.assertIs(await writer.send_event(factory), False)
        factory.assert_called_once_with()
        self.assertEqual(writer._websocket.events, [])
        self.assertFalse(self.rows("provider_write_latency", "send"))
        terminal = self.terminal()
        self.assertEqual(terminal["status"], "completed")
        self.assertEqual(terminal["reason"], "no_payload")
        self.assertFalse(terminal["write_possible"])
        self.assertFalse(terminal["send_returned"])

    async def test_source_prepare_and_borrowed_guard_report_real_ownership_without_extra_acquire(self):
        connection = TransactionConnection()
        pool = Pool(connection)
        sources = LiveResponseSources(pool, owner_id="offline-latency-boundary",
            session_id=uuid4(), audio_session_id=str(uuid4()))
        sources.bind_trace(self.trace())
        sources.observe((uuid4(),))
        snapshot = await sources.prepare("source-latency-request")
        completed = self.rows("response_source_latency", status="completed")
        for phase in ("prepare", "pool_acquire", "transaction_enter", "source_lock",
                      "payload_insert", "dependency_insert", "transaction_exit", "pool_release"):
            self.assertTrue(any(row["phase"] == phase for row in completed), phase)
        prepared = self.rows("response_source_latency", "prepare", "completed")[-1]
        self.assertEqual(prepared["request_id"], "source-latency-request")
        self.assertEqual(prepared["source_snapshot_id"], str(snapshot.payload_id))

        self.logger.reset_mock()
        async with pool.acquire() as borrowed, borrowed.transaction(), sources.use_transaction(borrowed):
            acquired = pool.entered
            async with sources.guard(snapshot):
                self.assertEqual(pool.entered, acquired)
                self.assertTrue(borrowed.is_in_transaction())
        rows = self.rows("response_source_latency")
        self.assertTrue(self.rows("response_source_latency", "transaction_reuse", "reused"))
        self.assertFalse(self.rows("response_source_latency", "pool_acquire"))
        guarded = self.rows("response_source_latency", "guard", "completed")[-1]
        self.assertEqual(guarded["source_snapshot_id"], str(snapshot.payload_id))
        self.assertTrue(all("root_ids" not in row and "content" not in row for row in rows))
        self.assertEqual(pool.entered, pool.exited)
        self.assertEqual(connection.entered, connection.exited)

    async def test_logging_failure_cannot_change_success_or_real_source_denial(self):
        writer = self.writer()
        connection = TransactionConnection()
        pool = Pool(connection)
        sources = LiveResponseSources(pool, owner_id="offline-latency-boundary",
            session_id=uuid4(), audio_session_id=str(uuid4()))
        sources.bind_trace(writer.latency_trace)
        root = uuid4()
        sources.observe((root,))
        snapshot = await sources.prepare("guard-denial-request")
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=RuntimeError("logging broken")):
            self.assertIs(await writer.send_event(self.payload()), True)
            connection.rejected.add(root)
            factory = Mock(return_value=self.payload())
            with self.assertRaises(LiveContextError) as caught:
                await writer.send_event(factory, write_guard=lambda: sources.guard(snapshot))
            self.assertEqual(caught.exception.code, "live_context_revoked")
            factory.assert_not_called()
        self.assertEqual(writer._websocket.events, [self.payload()])
        self.assertEqual(pool.entered, pool.exited)
        self.assertEqual(connection.entered, connection.exited)

    def test_latency_records_are_content_free_capped_and_diagnostic_only(self):
        trace = self.trace()
        for _ in range(2055):
            latency_event(trace, "provider_write_latency", phase="queue", status="started",
                dispatch_id="bounded-latency-probe", duration_ms=float("nan"),
                content="PRIVATE-CONTENT-NOT-TELEMETRY", instructions="PRIVATE-INSTRUCTIONS",
                error_message="PRIVATE-ERROR-NOT-TELEMETRY", token="PRIVATE-TOKEN")
        rows = self.rows()
        self.assertEqual(len(rows), 2049)
        self.assertEqual(len(self.rows("latency_diagnostics_truncated")), 1)
        for row in rows:
            self.assertNotIn("duration_ms", row)
        encoded = json.dumps(rows)
        self.assertNotIn("PRIVATE-", encoded)
        # Diagnostics never grant delivery, approval, acknowledgement or playback.
        self.assertFalse(any(row.get("status") in {"accepted", "acknowledged", "heard", "played"}
                             for row in rows))
