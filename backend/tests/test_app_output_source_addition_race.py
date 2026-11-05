"""Actual app-output writer/source guards over fake SQL and a local mock socket.

This reproduces native transcript delivery, not native audio or provider proof.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.attention.quiet_delivery import QuietDeliverySuppressed
from app.realtime.session import RealtimeAudioSession
from app.realtime.socket_sender import WebSocketEventSender
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.settings import JarvisSettings
import test_response_preparation_source_race as source_fixtures


class Connection(source_fixtures.Connection):
    def is_in_transaction(self):
        return self.entered > self.exited

    async def execute(self, query, *args):
        from app.attention.quiet import OWNER_LOCK, OWNER_SHARED_LOCK
        if query not in (OWNER_LOCK, OWNER_SHARED_LOCK) or not self.is_in_transaction():
            raise AssertionError("Unexpected or unlocked fake SQL")
        return "SELECT 1"


class AppOutputSourceAdditionRaceTests(unittest.IsolatedAsyncioTestCase):
    def voice(self, *, native=True, with_sources=True):
        voice = object.__new__(RealtimeAudioSession)
        voice.settings = JarvisSettings(_env_file=None, OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=0.1)
        voice._closed = asyncio.Event()
        voice._context_revoked = False
        voice._active_responses = {"resp_fixture": SimpleNamespace(
            metadata={}, delivery_kind="normal_user_turn")}
        voice._prospective_attempt_for = Mock(return_value=None)
        voice._native_playback = {}
        voice._trace = Mock()
        connection = Connection()
        pool = source_fixtures.Pool(connection)
        sources = LiveResponseSources(pool, owner_id="fixture-owner", session_id=uuid4(),
                                      audio_session_id=str(uuid4()))
        if with_sources:
            sources.observe((UUID(int=1),))
        voice._source_context = sources

        def retire():
            voice._context_revoked = True
        voice._begin_context_rebuild = Mock(side_effect=retire)
        sent = []

        async def send(raw):
            sent.append(json.loads(raw))
        socket = SimpleNamespace(send_text=AsyncMock(side_effect=send))
        sender = WebSocketEventSender(socket, trace=voice._trace)
        voice._app_sender = sender
        if native:
            # Real runtime wrapper: transcript output passes through the same
            # app writer, not a replacement native audio source.
            runtime = object.__new__(RealtimeVoiceRouteRuntime)
            runtime._owner_send_text = sender.send_text
            sender.send_text = runtime._app_send_text

        async def cleanup():
            self.assertEqual(pool.entered, pool.exited)
            self.assertEqual(connection.entered, connection.exited)
        self.addAsyncCleanup(cleanup)
        return voice, sources, connection, pool, socket, sent

    def refreshes(self, voice):
        return [call.kwargs for call in voice._trace.event.call_args_list
                if call.args == ("app_output_source_refresh",)]

    async def send(self, voice, kind="transcript.delta"):
        return await voice._send_app_event(kind, response_id="resp_fixture", item_id="item_fixture",
            text="PRIVATE_TRANSCRIPT", audio_base64="AAAAAA==")

    async def test_one_or_two_additions_during_lock_wait_write_same_output_once(self):
        for count in (1, 2):
            for kind in ("transcript.delta", "transcript.done"):
                with self.subTest(additions=count, kind=kind):
                    voice, sources, connection, pool, socket, sent = self.voice()
                    additions = [UUID(int=i + 2) for i in range(count)]

                    async def add(_):
                        if additions:
                            sources.observe((additions.pop(0),))
                    connection.on_lock = add
                    self.assertTrue(await self.send(voice, kind))
                    socket.send_text.assert_awaited_once()
                    self.assertEqual(len(sent), 1)
                    self.assertEqual(sent[0]["response_id"], "resp_fixture")
                    self.assertEqual(sent[0]["item_id"], "item_fixture")
                    self.assertEqual(connection.locks[-1], tuple(UUID(int=i + 1) for i in range(count + 1)))
                    self.assertEqual(pool.entered, count + 1)
                    voice._begin_context_rebuild.assert_not_called()
                    self.assertEqual([row["status"] for row in self.refreshes(voice)],
                                     [f"refresh_{i + 1}" for i in range(count)])
                    self.assertNotIn("PRIVATE", repr(self.refreshes(voice)))
                    voice._trace.forwarded.assert_called_once()

    async def test_sustained_additions_exhaust_at_three_lock_attempts_without_write(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        count = 0

        async def add(_):
            nonlocal count
            count += 1
            sources.observe((UUID(int=100 + count),))
        connection.on_lock = add
        self.assertFalse(await self.send(voice))
        self.assertEqual(count, 3)
        self.assertEqual(len(self.refreshes(voice)), 2)
        socket.send_text.assert_not_awaited()
        voice._trace.forwarded.assert_not_called()
        voice._begin_context_rebuild.assert_called_once()

    async def test_actual_withdrawal_on_initial_or_refreshed_guard_still_retires(self):
        for after_addition in (False, True):
            with self.subTest(after_addition=after_addition):
                voice, sources, connection, pool, socket, sent = self.voice()
                count = 0

                async def mutate(_):
                    nonlocal count
                    count += 1
                    if after_addition and count == 1:
                        sources.observe((UUID(int=2),))
                    else:
                        connection.rejected.add(UUID(int=1))
                connection.on_lock = mutate
                self.assertFalse(await self.send(voice))
                self.assertEqual(count, 2 if after_addition else 1)
                socket.send_text.assert_not_awaited()
                voice._begin_context_rebuild.assert_called_once()
                self.assertEqual(len(self.refreshes(voice)), int(after_addition))

    async def test_same_error_after_guard_yield_cannot_replay_a_client_write(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        original = sources.context_guard

        @asynccontextmanager
        async def after_yield():
            async with original():
                yield
                raise LiveContextError("live_context_changed")
        sources.context_guard = after_yield
        self.assertFalse(await self.send(voice))
        socket.send_text.assert_awaited_once()
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.refreshes(voice), [])
        voice._begin_context_rebuild.assert_called_once()
        # A transport side effect is not a completed guarded forwarding receipt.
        voice._trace.forwarded.assert_not_called()

    async def test_same_error_from_socket_is_unknown_not_a_guard_retry(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        socket.send_text.side_effect = LiveContextError("live_context_changed")
        self.assertFalse(await self.send(voice))
        socket.send_text.assert_awaited_once()
        self.assertEqual(self.refreshes(voice), [])
        voice._begin_context_rebuild.assert_called_once()

    async def test_unknown_socket_failure_propagates_without_repeat(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        socket.send_text.side_effect = RuntimeError("PRIVATE_SOCKET_FAILURE")
        with self.assertRaises(RuntimeError):
            await self.send(voice)
        socket.send_text.assert_awaited_once()
        self.assertEqual(self.refreshes(voice), [])
        voice._trace.forwarded.assert_not_called()

    async def test_deadline_during_refreshed_guard_remains_fail_closed(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        count = 0

        async def hold(_):
            nonlocal count
            count += 1
            if count == 1:
                sources.observe((UUID(int=2),))
            else:
                await asyncio.Event().wait()
        connection.on_lock = hold
        self.assertFalse(await self.send(voice))
        self.assertEqual(count, 2)
        socket.send_text.assert_not_awaited()
        voice._begin_context_rebuild.assert_called_once()

    async def test_cancellation_joins_guard_and_never_writes(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        entered = asyncio.Event()

        async def hold(_):
            entered.set()
            await asyncio.Event().wait()
        connection.on_lock = hold
        task = asyncio.create_task(self.send(voice))
        try:
            await asyncio.wait_for(entered.wait(), 0.5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            socket.send_text.assert_not_awaited()
            voice._begin_context_rebuild.assert_not_called()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_close_or_revocation_while_guard_waits_is_rechecked_before_write(self):
        for change in ("closed", "revoked"):
            with self.subTest(change=change):
                voice, sources, connection, pool, socket, sent = self.voice()

                async def change_state(_):
                    if change == "closed":
                        voice._closed.set()
                    else:
                        voice._context_revoked = True
                connection.on_lock = change_state
                self.assertFalse(await self.send(voice))
                socket.send_text.assert_not_awaited()
                voice._trace.forwarded.assert_not_called()

    async def test_borrowed_owner_transaction_is_not_reacquired_for_refresh(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        additions = [UUID(int=2), UUID(int=3)]

        async def add(_):
            if additions:
                sources.observe((additions.pop(0),))
        connection.on_lock = add
        async with pool.acquire() as borrowed, borrowed.transaction():
            async with sources.use_transaction(borrowed):
                self.assertTrue(await self.send(voice))
                self.assertIs(sources._current_connection(), borrowed)
        self.assertEqual(pool.entered, 1)
        self.assertEqual(connection.entered, 1)
        self.assertEqual(connection.locks[-1], (UUID(int=1), UUID(int=2), UUID(int=3)))
        socket.send_text.assert_awaited_once()
        self.assertIsNone(sources._current_connection())

    async def test_quiet_policy_still_rejects_before_source_refresh(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        voice._active_responses["resp_fixture"].delivery_kind = "visual_interrupt"
        with patch("app.attention.quiet_delivery.assert_unsolicited_allowed", AsyncMock(
                side_effect=QuietDeliverySuppressed("owner_quiet_active", "visual_interrupt"))):
            self.assertFalse(await self.send(voice))
        self.assertEqual(connection.locks, [])
        self.assertEqual(self.refreshes(voice), [])
        socket.send_text.assert_not_awaited()
        voice._begin_context_rebuild.assert_not_called()

    async def test_no_source_and_legacy_output_paths_keep_existing_behavior(self):
        for source_bound in (False, True):
            for kind in ("transcript.delta", "transcript.done", "audio.output.delta", "audio.output.done"):
                with self.subTest(sources=source_bound, kind=kind):
                    voice, sources, connection, pool, socket, sent = self.voice(native=False,
                                                                              with_sources=source_bound)
                    self.assertTrue(await self.send(voice, kind))
                    socket.send_text.assert_awaited_once()
                    self.assertEqual(len(connection.locks), int(source_bound))
                    self.assertEqual(self.refreshes(voice), [])
                    voice._begin_context_rebuild.assert_not_called()

    async def test_logging_failure_cannot_change_guard_refresh_or_forwarding(self):
        voice, sources, connection, pool, socket, sent = self.voice()
        additions = [UUID(int=2)]

        async def add(_):
            if additions:
                sources.observe((additions.pop(0),))
        connection.on_lock = add
        voice._trace.event.side_effect = RuntimeError("PRIVATE_LOGGING_FAILURE")
        self.assertTrue(await self.send(voice))
        socket.send_text.assert_awaited_once()
        voice._begin_context_rebuild.assert_not_called()
