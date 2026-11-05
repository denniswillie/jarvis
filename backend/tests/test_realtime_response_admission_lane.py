"""Real direct-session handlers and ordered provider writer; offline barriers.

SQL replies and the provider socket are fixtures, not live services. In particular
these checks prove control liveness/ownership, not audible or network latency.
"""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.quiet import OWNER_LOCK, OWNER_SHARED_LOCK
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.source_context import LiveContextError
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime, _Dispatch
from app.settings import JarvisSettings
import test_realtime_shared_view as provider_fixtures


class _SourcePool:
    """Exercise real source preparation/guards with explicitly fake SQL results."""

    def __init__(self):
        self.acquisitions = 0
        self.depth = 0
        self.available = True
        self.locks = []
        self.copies = []
        self.edges = []
        self.owner_locks = []

    @asynccontextmanager
    async def acquire(self):
        self.acquisitions += 1
        yield self

    @asynccontextmanager
    async def transaction(self):
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    def is_in_transaction(self):
        return self.depth > 0

    async def execute(self, sql, *args):
        if sql not in (OWNER_LOCK, OWNER_SHARED_LOCK) or len(args) != 1 or not self.is_in_transaction():
            raise AssertionError("Unexpected or unguarded owner-lock SQL in offline fixture")
        self.owner_locks.append(args[0])
        return "SELECT 1"

    async def fetchval(self, sql, *args):
        if "lock_jarvis_copy_sources" in sql:
            self.locks.append(tuple(args[-1]))
            return self.available
        if "insert into jarvis_context_payloads" in sql:
            identity = uuid4()
            self.copies.append(identity)
            return identity
        raise AssertionError("Unexpected SQL in offline source fixture")

    async def executemany(self, sql, rows):
        self.edges.extend(rows)


class ResponseAdmissionLaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(
                audio_session_id="admission-fixture", owner_id="owner",
                jarvis_session_id=uuid4(), client_kind="rayban_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())),
                                       get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=(),
        )
        self.session._realtime_context_ready.set()
        # Only the opt-in lane gate is stubbed. Direct speech handling, response
        # admission, source preparation, WebRTC send wrapper and provider writer
        # are production code. Full route/sink composition has separate tests.
        self.session._voice_route_runtime = SimpleNamespace(
            ready=True, before_session_close=AsyncMock(), after_session_close=AsyncMock())
        self.session._send_app_event = AsyncMock(return_value=True)
        self.provider = self.session.openai
        self.wire = provider_fixtures._Wire(self.session)
        self.provider._websocket = self.wire
        self.pool = _SourcePool()
        self.session._source_context.pool = self.pool
        self.barriers = []
        self.waiters = []
        self.writer_held = False
        self.receiver = asyncio.create_task(self.receive())

    async def asyncTearDown(self):
        for barrier in self.barriers:
            barrier.set()
        self.release_writer()
        await self.session.close("fixture_complete")
        self.receiver.cancel()
        for task in self.waiters:
            if not task.done():
                task.cancel()
        await asyncio.gather(self.receiver, *self.waiters, return_exceptions=True)

    async def receive(self):
        self.session._provider_receive_task = asyncio.current_task()
        while True:
            event = await self.wire.incoming.get()
            try:
                await self.session._handle_openai_event(event)
            finally:
                self.wire.incoming.task_done()

    def task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.waiters.append(task)
        return task

    def barrier(self):
        barrier = asyncio.Event()
        self.barriers.append(barrier)
        return barrier

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                if self.receiver.done():
                    await self.receiver
                await asyncio.sleep(0)

    async def emit(self, kind, **payload):
        self.wire.incoming.put_nowait({"type": kind, **payload})
        await asyncio.wait_for(self.wire.incoming.join(), 1)
        if self.receiver.done():
            await self.receiver

    async def ask(self, item="first"):
        await self.emit("input_audio_buffer.speech_started", item_id=item)
        await self.emit("input_audio_buffer.committed", item_id=item)

    def source(self):
        root = uuid4()
        self.session._source_context.observe((root,))
        return root

    def blocked_source(self, *, ignore_cancel=False):
        self.source()
        entered, release, cancelled = self.barrier(), self.barrier(), self.barrier()
        original = self.session._source_context.prepare

        async def prepare(request_id):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                if not ignore_cancel:
                    raise
                await release.wait()
            return await original(request_id)

        self.session._source_context.prepare = AsyncMock(side_effect=prepare)
        return entered, release, cancelled

    async def hold_writer(self):
        await self.provider._send_lock.acquire()
        self.writer_held = True

    def release_writer(self):
        if self.writer_held:
            self.writer_held = False
            self.provider._send_lock.release()

    async def test_speech_preempts_blocked_source_and_stale_result_cannot_replace_new_turn(self):
        entered, release, cancelled = self.blocked_source(ignore_cancel=True)
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        old = self.session._pending_response
        old_task = self.session._response_preparation_task
        self.assertFalse(self.session._response_admission_lock.locked())
        await self.emit("input_audio_buffer.speech_started", item_id="newer")
        await asyncio.wait_for(cancelled.wait(), 1)
        self.assertFalse(release.is_set())
        self.assertFalse(old_task.done())
        self.assertTrue(self.session._user_audio_pending)
        await self.emit("input_audio_buffer.committed", item_id="newer")
        await self.until(lambda: self.session._pending_response is not None)
        newer = self.session._pending_response
        self.assertIsNot(newer, old)
        release.set()
        await self.until(lambda: len(self.wire.responses) == 1 and not self.session._direct_tasks)
        self.assertFalse(old.submitted)
        self.assertTrue(old_task.cancelled())
        with self.assertRaises(asyncio.CancelledError):
            old_task.result()
        self.assertEqual(self.wire.responses[0]["event"]["event_id"], newer.request_id)
        self.assertNotIn(old.request_id, self.session._unacknowledged_response_requests)
        self.assertIsNone(self.session._tool_followup)

    async def test_duplicate_response_requests_share_single_reservation_while_source_waits(self):
        entered, release, _ = self.blocked_source()
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        reservation = self.session._pending_response
        self.assertFalse(await self.session._request_normal_response(
            turn_generation=self.session._turn_generation))
        await self.emit("input_audio_buffer.committed", item_id="first")
        self.assertIs(self.session._pending_response, reservation)
        self.assertFalse(release.is_set())
        release.set()
        await self.until(lambda: len(self.wire.responses) == 1 and not self.session._direct_tasks)
        self.assertEqual(self.session._source_context.prepare.await_count, 1)
        self.assertEqual(self.wire.responses[0]["event"]["event_id"], reservation.request_id)

    async def test_queued_provider_write_does_not_block_speech_start(self):
        await self.hold_writer()
        await self.ask()
        await self.until(lambda: self.provider._foreground_waiters == 1)
        old = self.session._pending_response
        await self.emit("input_audio_buffer.speech_started", item_id="newer")
        self.assertTrue(self.writer_held)
        self.assertFalse(old.submitted)
        self.assertTrue(self.session._user_audio_pending)
        await self.emit("input_audio_buffer.committed", item_id="newer")
        await self.until(lambda: self.session._pending_response is not None)
        newer = self.session._pending_response
        self.release_writer()
        await self.until(lambda: len(self.wire.responses) == 1 and not self.session._direct_tasks)
        self.assertEqual(self.wire.responses[0]["event"]["event_id"], newer.request_id)
        self.assertNotEqual(newer.request_id, old.request_id)

    async def test_at_write_generation_check_rejects_cancel_resistant_source_guard(self):
        self.source()
        entered, release, cancelled = self.barrier(), self.barrier(), self.barrier()
        original = self.session._source_context.guard

        @asynccontextmanager
        async def guard(snapshot):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            async with original(snapshot):
                yield

        self.session._source_context.guard = guard
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        old = self.session._pending_response
        await self.emit("input_audio_buffer.speech_started", item_id="newer")
        await asyncio.wait_for(cancelled.wait(), 1)
        self.assertFalse(release.is_set())
        release.set()
        await self.until(lambda: not self.session._direct_tasks)
        self.assertFalse(old.submitted)
        self.assertEqual(self.wire.responses, [])
        self.assertTrue(self.session._normal_turn_pending)
        self.assertTrue(self.session._user_audio_pending)
        self.assertIsNone(self.session._response_start_task)

    async def test_started_write_is_cancelled_without_replay_and_late_ack_is_interrupted(self):
        entered, release, cancelled = self.barrier(), self.barrier(), self.barrier()
        original = self.wire.send
        submitted = []

        async def send(raw):
            event = json.loads(raw)
            if event["type"] == "response.create" and not submitted:
                submitted.append(event)
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            await original(raw)

        self.wire.send = send
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        old = self.session._pending_response
        self.assertTrue(old.submitted)
        self.assertFalse(self.session._response_admission_lock.locked())
        await self.emit("input_audio_buffer.speech_started", item_id="newer")
        self.assertTrue(cancelled.is_set())
        self.assertFalse(release.is_set())
        self.assertIn(old.request_id, self.session._unacknowledged_response_requests)
        self.assertEqual(self.wire.responses, [])
        await self.emit("response.created", response={
            "id": "late-old", "metadata": submitted[0]["response"]["metadata"]})
        self.assertTrue(self.session._active_responses["late-old"].interrupted)
        self.assertIsNone(self.session._current_response_id)
        await self.emit("response.done", response={"id": "late-old", "status": "completed"})
        await self.until(lambda: not self.session._provider_completion_tasks)
        self.assertTrue(self.session._user_audio_pending)
        self.assertIsNone(self.session._tool_followup)
        await self.emit("input_audio_buffer.committed", item_id="newer")
        await self.until(lambda: len(self.wire.responses) == 1 and not self.session._direct_tasks)
        self.assertNotEqual(self.wire.responses[0]["event"]["event_id"], old.request_id)

    async def test_source_withdrawal_at_writer_still_rejects_response(self):
        self.source()
        self.session._begin_context_rebuild = Mock()
        await self.hold_writer()
        await self.ask()
        await self.until(lambda: self.provider._foreground_waiters == 1)
        old = self.session._pending_response
        self.pool.available = False
        self.release_writer()
        await self.until(lambda: not self.session._direct_tasks)
        self.assertFalse(old.submitted)
        self.assertEqual(self.wire.responses, [])
        self.session._begin_context_rebuild.assert_called_once()

    async def test_native_notification_keeps_guard_transaction_and_metadata(self):
        root = self.source()
        offer = SimpleNamespace(offer_id=uuid4(), attempt=SimpleNamespace(attempt_id=uuid4()))
        self.session._attention_delivery = SimpleNamespace(
            accepts_offer=lambda candidate: candidate is offer, stop=AsyncMock(), task=None)
        checks = []

        @asynccontextmanager
        async def canonical_guard():
            checks.append("canonical")
            async with self.pool.transaction(), self.session._source_context.use_transaction(self.pool):
                yield

        dispatch = _Dispatch(offer=offer, guard=canonical_guard, current=lambda: True,
                             request_id="notification-dispatch")
        runtime = SimpleNamespace(
            ready=True, session=self.session,
            _dispatch_context=ContextVar("test_dispatch", default=dispatch),
            _owner_provider_send=self.provider.send_event,
            _report_unavailable_status=AsyncMock(),
            request=SimpleNamespace(route=SimpleNamespace(route_id=uuid4(), route_generation=1)))
        self.provider.send_event = lambda event, **kwargs: RealtimeVoiceRouteRuntime._provider_send(
            runtime, event, **kwargs)

        def at_write():
            self.assertFalse(self.session._response_admission_lock.locked())
            self.assertTrue(self.pool.is_in_transaction())
            self.assertEqual(self.pool.depth, 1)
            self.assertEqual(self.pool.owner_locks, [self.session.context.owner_id])
            self.assertEqual(self.pool.locks[-1], (root, self.pool.copies[-1]))
            checks.append("write")

        self.wire.before_response = at_write
        # This fixture qualifies transaction/metadata composition, not policy
        # persistence. The real quiet guard still checks this explicit decision.
        decision = AsyncMock(return_value=SimpleNamespace(allowed=True))
        with patch("app.attention.quiet_delivery.quiet_delivery_decision", decision):
            self.assertTrue(await self.session.request_route_notification(
                offer, turn_generation=self.session._turn_generation))
        decision.assert_awaited_once_with(self.pool, self.session.context.owner_id,
                                         kind="unsolicited", channel="voice")
        self.assertEqual(checks, ["canonical", "write"])
        self.assertEqual(self.pool.acquisitions, 1)  # Prepare only; writer borrows canonical transaction.
        self.assertEqual(self.pool.depth, 0)
        self.assertTrue(dispatch.written)
        metadata = self.wire.responses[0]["event"]["response"]["metadata"]
        self.assertEqual(metadata["voice_route_offer_id"], str(offer.offer_id))
        self.assertEqual(metadata["attention_attempt_id"], str(offer.attempt.attempt_id))
        self.assertEqual(metadata["voice_route_request_id"], dispatch.request_id)
        self.assertEqual(dispatch.response_request_id,
                         self.wire.responses[0]["event"]["event_id"])
        self.assertEqual(metadata["jarvis_request_id"], dispatch.response_request_id)
        self.assertEqual(metadata["jarvis_kind"], "prospective")

    async def test_notification_source_wait_is_preempted_without_manufactured_user_turn(self):
        entered, release, cancelled = self.blocked_source()
        offer = SimpleNamespace(offer_id=uuid4(), attempt=SimpleNamespace(attempt_id=uuid4()))
        delivery = SimpleNamespace(accepts_offer=lambda candidate: candidate is offer,
                                   stop=AsyncMock(), task=None)
        self.session._attention_delivery = delivery
        notification = self.task(self.session.request_route_notification(offer, turn_generation=0))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(self.session._turn_generation, 0)
        await self.emit("input_audio_buffer.speech_started", item_id="newer")
        self.assertFalse(await notification)
        self.assertTrue(cancelled.is_set())
        self.assertFalse(release.is_set())
        self.assertEqual(self.session._turn_generation, 1)
        self.assertEqual(self.wire.responses, [])
        delivery.stop.assert_awaited_once_with("user_interrupted")

    async def test_shutdown_joins_source_preparation_without_sending(self):
        entered, release, cancelled = self.blocked_source()
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        preparing = self.session._response_preparation_task
        await asyncio.wait_for(self.session.close("source_wait_shutdown"), 1)
        self.assertTrue(preparing.done())
        self.assertTrue(cancelled.is_set())
        self.assertFalse(release.is_set())
        self.assertEqual(self.session._direct_tasks, set())
        self.assertEqual(self.wire.responses, [])
        self.assertIsNone(self.session._pending_response)

    async def test_shutdown_joins_writer_wait_without_releasing_test_writer(self):
        await self.hold_writer()
        await self.ask()
        await self.until(lambda: self.provider._foreground_waiters == 1)
        preparing = self.session._response_preparation_task
        await asyncio.wait_for(self.session.close("writer_wait_shutdown"), 1)
        self.assertTrue(self.writer_held)
        self.assertTrue(preparing.done())
        self.assertEqual(self.session._direct_tasks, set())
        self.assertEqual(self.wire.responses, [])
        self.wire.close.assert_awaited_once()

    async def test_default_legacy_source_wait_keeps_existing_lock_behavior(self):
        self.session._voice_route_runtime = None
        entered, release, _ = self.blocked_source()
        await self.ask()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertTrue(self.session._response_admission_lock.locked())
        release.set()
        await self.until(lambda: len(self.wire.responses) == 1 and not self.session._direct_tasks)
        self.assertFalse(self.session._response_admission_lock.locked())
