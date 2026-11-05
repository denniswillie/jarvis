"""Real app/native-transcript writer and quiet/source guards, offline only.

Canonical quiet reads and storage/socket I/O are controlled fixtures. This is
not SQL, provider, native playback, or retrospective native-stack proof.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from app.attention.quiet_delivery import quiet_response_guard
from app.realtime.source_context import LiveContextError
import test_app_output_source_addition_race as output_fixtures


class QuietAppWriterAdapterTests(unittest.IsolatedAsyncioTestCase):
    voice = output_fixtures.AppOutputSourceAdditionRaceTests.voice

    async def fixture(self, *, kind="prospective", native=True, with_sources=True, metadata=None):
        voice, sources, connection, pool, socket, sent = self.voice(
            native=native, with_sources=with_sources)
        state = voice._active_responses["resp_fixture"]
        state.delivery_kind = kind
        state.metadata = metadata or {}
        pin = await sources.prepare("accepted-request") if with_sources else None
        state.source_snapshot = pin
        order = []
        original_execute = connection.execute

        async def owner_lock(query, *args):
            result = await original_execute(query, *args)
            order.append("owner")
            return result

        async def decision(owned, owner, *, kind, channel):
            self.assertIs(owned, connection)
            self.assertIs(sources._current_connection(), connection)
            self.assertTrue(connection.is_in_transaction())
            self.assertFalse(callable(connection))
            self.assertEqual((owner, kind, channel), ("fixture-owner", "unsolicited", "voice"))
            order.append("quiet")
            return SimpleNamespace(allowed=True)

        async def source_lock(roots):
            self.assertTrue(connection.is_in_transaction())
            order.append("sources")

        async def send(raw):
            if state.delivery_kind != "normal_user_turn":
                self.assertIs(sources._current_connection(), connection)
                self.assertTrue(connection.is_in_transaction())
            order.append("send")
            sent.append(json.loads(raw))

        connection.execute = owner_lock
        connection.on_lock = source_lock
        socket.send_text.side_effect = send
        policy = AsyncMock(side_effect=decision)
        capture = patch("app.attention.quiet_delivery.quiet_delivery_decision", policy)
        capture.start()
        self.addCleanup(capture.stop)
        return SimpleNamespace(voice=voice, sources=sources, connection=connection,
            pool=pool, socket=socket, sent=sent, pin=pin, order=order, policy=policy)

    async def send(self, fixture, kind="transcript.delta"):
        return await fixture.voice._send_app_event(kind, response_id="resp_fixture",
            item_id="item_fixture", text="PRIVATE_TRANSCRIPT", audio_base64="AAAAAA==")

    def assert_released(self, fixture):
        self.assertIsNone(fixture.sources._current_connection())
        self.assertEqual(fixture.pool.entered, fixture.pool.exited)
        self.assertEqual(fixture.connection.entered, fixture.connection.exited)

    async def test_old_direct_guard_wiring_raises_only_after_one_send(self):
        f = await self.fixture()
        raw = json.dumps({"type": "transcript.delta", "response_id": "resp_fixture"})
        # Exact former app-boundary wiring, without copying the app method or
        # replacing either the native wrapper, quiet guard, or actual writer.
        with self.assertRaises(TypeError):
            await f.voice._app_sender.send_text(raw, write_guard=lambda: quiet_response_guard(
                f.sources, origin="prospective", source_guard=lambda: f.sources.output_guard(f.pin)))
        self.assertEqual(f.order, ["owner", "quiet", "sources", "send"])
        f.socket.send_text.assert_awaited_once()
        self.assertEqual(len(f.sent), 1)
        f.voice._trace.forwarded.assert_not_called()
        f.voice._trace.event.assert_called_with("socket_write_failed",
            reason="write_outcome_unknown", error_type="TypeError")
        self.assert_released(f)

    async def test_actual_app_boundary_delivers_unsolicited_transcripts_once(self):
        for origin in ("prospective", "reminder", "visual_interrupt", "live_readonly_result"):
            for event_type in ("transcript.delta", "transcript.done"):
                with self.subTest(origin=origin, event_type=event_type):
                    f = await self.fixture(kind=origin)
                    acquired = f.pool.entered
                    self.assertTrue(await self.send(f, event_type))
                    self.assertEqual(f.order, ["owner", "quiet", "sources", "send"])
                    self.assertEqual(f.pool.entered, acquired + 1)
                    self.assertEqual(f.connection.locks[-1], (*f.pin.sources, f.pin.payload_id))
                    f.socket.send_text.assert_awaited_once()
                    self.assertEqual(len(f.sent), 1)
                    f.voice._trace.forwarded.assert_called_once()
                    f.voice._begin_context_rebuild.assert_not_called()
                    self.assert_released(f)

    async def test_continuation_keeps_trusted_unsolicited_lineage(self):
        f = await self.fixture(kind="tool_continuation", metadata={"jarvis_delivery_kind": "reminder"})
        self.assertTrue(await self.send(f))
        f.policy.assert_awaited_once()
        self.assertEqual(f.order, ["owner", "quiet", "sources", "send"])
        self.assert_released(f)

    async def test_no_source_unsolicited_hint_still_checks_quiet_and_sends_once(self):
        f = await self.fixture(with_sources=False)
        self.assertTrue(await self.send(f, "prospective.started"))
        self.assertEqual(f.order, ["owner", "quiet", "send"])
        f.socket.send_text.assert_awaited_once()
        self.assert_released(f)

    async def test_legacy_pcm_writer_uses_same_adapter_without_second_native_speaker(self):
        f = await self.fixture(native=False)
        self.assertTrue(await self.send(f, "audio.output.delta"))
        self.assertEqual(f.order, ["owner", "quiet", "sources", "send"])
        f.socket.send_text.assert_awaited_once()
        self.assertGreater(f.voice._native_playback["item_fixture"]["samples"], 0)
        self.assert_released(f)

    async def test_denied_or_unavailable_quiet_never_reaches_source_or_socket(self):
        for failure in ("denied", "unavailable"):
            with self.subTest(failure=failure):
                f = await self.fixture()
                f.policy.side_effect = RuntimeError("PRIVATE_READ") if failure == "unavailable" else None
                f.policy.return_value = SimpleNamespace(allowed=False, reason="owner_quiet_active")
                self.assertFalse(await self.send(f))
                self.assertEqual(f.order, ["owner"])
                f.policy.assert_awaited_once()
                f.socket.send_text.assert_not_awaited()
                f.voice._trace.forwarded.assert_not_called()
                f.voice._begin_context_rebuild.assert_not_called()
                self.assert_released(f)

    async def test_direct_answer_does_not_enter_unsolicited_quiet_guard(self):
        f = await self.fixture(kind="normal_user_turn")
        f.policy.side_effect = None
        f.policy.return_value = SimpleNamespace(allowed=False, reason="owner_quiet_active")
        self.assertTrue(await self.send(f))
        f.policy.assert_not_awaited()
        self.assertEqual(f.order, ["sources", "send"])
        f.socket.send_text.assert_awaited_once()
        self.assert_released(f)

    async def test_actual_source_withdrawal_still_blocks_and_requests_rebuild(self):
        f = await self.fixture()
        f.connection.rejected.add(UUID(int=1))
        self.assertFalse(await self.send(f))
        self.assertEqual(f.order, ["owner", "quiet", "sources"])
        f.socket.send_text.assert_not_awaited()
        f.voice._begin_context_rebuild.assert_called_once()
        f.voice._trace.forwarded.assert_not_called()
        self.assert_released(f)

    async def test_borrowed_transaction_stays_owned_and_is_not_reacquired(self):
        f = await self.fixture()
        async with f.pool.acquire() as connection, connection.transaction():
            async with f.sources.use_transaction(connection):
                acquired, entered = f.pool.entered, connection.entered
                self.assertTrue(await self.send(f))
                self.assertEqual((f.pool.entered, connection.entered), (acquired, entered))
                self.assertTrue(connection.is_in_transaction())
                self.assertIs(f.sources._current_connection(), connection)
        self.assertEqual(f.order, ["owner", "quiet", "sources", "send"])
        self.assert_released(f)

    async def test_cancellation_before_write_joins_authority_without_send(self):
        f = await self.fixture()
        entered, joined = asyncio.Event(), asyncio.Event()

        async def blocked_policy(*args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                joined.set()

        f.policy.side_effect = blocked_policy
        task = asyncio.create_task(self.send(f))
        try:
            await asyncio.wait_for(entered.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.done() and task.cancelled())
        self.assertTrue(joined.is_set())
        f.socket.send_text.assert_not_awaited()
        f.voice._trace.forwarded.assert_not_called()
        self.assert_released(f)

    async def test_cancel_after_handoff_is_unknown_and_never_resends(self):
        f = await self.fixture()
        handed_off = asyncio.Event()
        original_send = f.socket.send_text.side_effect

        async def pending_send(raw):
            await original_send(raw)
            handed_off.set()
            await asyncio.Event().wait()

        f.socket.send_text.side_effect = pending_send
        task = asyncio.create_task(self.send(f))
        try:
            await asyncio.wait_for(handed_off.wait(), .5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.done() and task.cancelled())
        self.assertEqual(len(f.sent), 1)
        f.socket.send_text.assert_awaited_once()
        f.voice._trace.forwarded.assert_not_called()
        f.voice._trace.event.assert_called_with("socket_write_failed",
            reason="write_outcome_unknown", error_type="CancelledError")
        self.assert_released(f)

    async def test_unknown_writer_typeerror_is_not_swallowed_or_retried(self):
        f = await self.fixture()
        original_send = f.socket.send_text.side_effect

        async def broken_send(raw):
            await original_send(raw)
            raise TypeError("PRIVATE_TRANSPORT_FAILURE")

        f.socket.send_text.side_effect = broken_send
        with self.assertRaises(TypeError):
            await self.send(f)
        f.socket.send_text.assert_awaited_once()
        self.assertEqual(len(f.sent), 1)
        f.voice._trace.forwarded.assert_not_called()
        self.assert_released(f)

    async def test_source_failure_after_send_is_not_replayed_or_reported_forwarded(self):
        f = await self.fixture()
        original = f.sources.output_guard

        @asynccontextmanager
        async def reject_after_send(snapshot):
            async with original(snapshot):
                yield
                raise LiveContextError("live_context_revoked")

        f.sources.output_guard = reject_after_send
        self.assertFalse(await self.send(f))
        f.socket.send_text.assert_awaited_once()
        self.assertEqual(len(f.sent), 1)
        f.voice._trace.forwarded.assert_not_called()
        f.voice._begin_context_rebuild.assert_called_once()
        self.assert_released(f)

    async def test_valid_forwarding_callback_still_runs_after_send_inside_guard(self):
        f = await self.fixture(kind="normal_user_turn", with_sources=False)
        marker = Mock(side_effect=lambda: f.order.append("marked"))

        @asynccontextmanager
        async def marked_guard():
            f.order.append("enter")
            try:
                yield marker
            finally:
                f.order.append("exit")

        raw = json.dumps({"type": "transcript.delta", "response_id": "resp_fixture"})
        self.assertTrue(await f.voice._app_sender.send_text(raw, write_guard=marked_guard))
        self.assertEqual(f.order, ["enter", "send", "marked", "exit"])
        marker.assert_called_once_with()
        f.socket.send_text.assert_awaited_once()
        f.voice._trace.forwarded.assert_called_once()
