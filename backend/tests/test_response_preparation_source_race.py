"""Actual guarded response writer with fake storage/socket, no external calls."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.settings import JarvisSettings


class Connection:
    def __init__(self):
        self.locks, self.payloads, self.dependencies = [], [], []
        self.on_lock = None
        self.rejected = set()
        self.materialized = asyncio.Event()
        self.entered = self.exited = 0

    @asynccontextmanager
    async def transaction(self):
        self.entered += 1
        try:
            yield
        finally:
            self.exited += 1

    async def fetchval(self, query, owner, value):
        if query.startswith("select lock_jarvis_copy_sources"):
            self.locks.append(tuple(value))
            if self.on_lock is not None:
                await self.on_lock(tuple(value))
            return not self.rejected.intersection(value)
        if query.startswith("insert into jarvis_context_payloads"):
            payload = UUID(int=1000 + len(self.payloads))
            self.payloads.append((payload, json.loads(value)))
            return payload
        raise AssertionError("Unexpected fake SQL operation")

    async def executemany(self, query, rows):
        if not query.startswith("insert into jarvis_context_payload_dependencies"):
            raise AssertionError("Unexpected dependency operation")
        self.dependencies.append(tuple(rows))
        self.materialized.set()


class Pool:
    def __init__(self, connection):
        self.connection = connection
        self.entered = self.exited = 0

    @asynccontextmanager
    async def acquire(self):
        self.entered += 1
        try:
            yield self.connection
        finally:
            self.exited += 1


class Socket:
    def __init__(self):
        self.events = []
        self.error = None

    async def send(self, raw):
        self.events.append(json.loads(raw))
        if self.error is not None:
            raise self.error


class SourcePreparationRaceTests(unittest.IsolatedAsyncioTestCase):
    def sources(self, *, max_responses=512):
        connection = Connection()
        pool = Pool(connection)
        sources = LiveResponseSources(pool, owner_id="offline-owner", session_id=uuid4(),
                                      audio_session_id=str(uuid4()), max_responses=max_responses)
        return sources, connection, pool

    def voice(self, *, with_sources=True):
        voice = object.__new__(RealtimeAudioSession)
        voice.settings = JarvisSettings(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=0.2)
        voice._source_context, connection, pool = self.sources()
        if with_sources:
            voice._source_context.observe((UUID(int=1),))
        voice._closed = asyncio.Event()
        voice._context_revoked = False
        voice._turn_generation = 1
        voice._pending_response = None
        voice._voice_route_runtime = SimpleNamespace(ready=True)
        voice._response_controls = SimpleNamespace(busy=False, wait_idle=AsyncMock())
        voice._prospective_attempt_for = Mock(return_value=None)
        voice._trace = Mock()
        voice._assistant_context = None
        voice._visual_prompt_metadata = Mock(return_value=None)
        voice.tool_registry = SimpleNamespace(get=Mock(return_value=None))
        voice.tool_executor = SimpleNamespace(execute=AsyncMock())
        voice.prepare_visual_response = AsyncMock()
        voice.context = SimpleNamespace(audio_session_id=str(uuid4()), jarvis_session_id=uuid4())
        voice._response_start_task = None
        voice._cancel_response_start_timer = Mock()
        voice._watch_response_start = AsyncMock()

        def retire():
            voice._context_revoked = True
        voice._begin_context_rebuild = Mock(side_effect=retire)
        voice.openai = OpenAIRealtimeConnection(settings=voice.settings, owner_id="offline-owner")
        socket = Socket()
        voice.openai._websocket = socket
        reservation = voice._reserve_response_locked("normal_user_turn", {})

        async def cleanup():
            task = voice._response_start_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(pool.entered, pool.exited)
            self.assertEqual(connection.entered, connection.exited)
            voice.tool_executor.execute.assert_not_awaited()
            voice.prepare_visual_response.assert_not_awaited()
        self.addAsyncCleanup(cleanup)
        return voice, reservation, connection, socket

    def setUp(self):
        instructions = patch("app.realtime.session.build_realtime_instructions", return_value="Offline fixture")
        instructions.start()
        self.addCleanup(instructions.stop)

    def trace_statuses(self, voice):
        return [call.kwargs["status"] for call in voice._trace.event.call_args_list
                if call.args == ("response_source_preparation",)]

    async def test_refresh_is_explicit_immutable_and_does_not_consume_unique_request_budget(self):
        sources, connection, _ = self.sources(max_responses=1)
        sources.observe((UUID(int=1),))
        first = await sources.prepare("same-request")
        sources.observe((UUID(int=2),))
        self.assertIs(await sources.prepare("same-request"), first)
        second = await sources.prepare("same-request", refresh_stale=True)
        self.assertNotEqual(first.payload_id, second.payload_id)
        self.assertEqual(first.sources, (UUID(int=1),))
        self.assertEqual(second.sources, (UUID(int=1), UUID(int=2)))
        self.assertEqual(sources.source_for("same-request"), second.payload_id)
        self.assertIs(await sources.prepare("same-request", refresh_stale=True), second)
        self.assertEqual(len(connection.payloads), 2)
        self.assertEqual([row[1]["request_id"] for row in connection.payloads], ["same-request"] * 2)
        self.assertEqual([row[1] for row in connection.dependencies[0]], [UUID(int=1)])
        with self.assertRaises(LiveContextError) as caught:
            await sources.prepare("new-request")
        self.assertEqual(caught.exception.code, "live_context_response_limit")

    async def test_failed_refresh_keeps_original_cache_and_checks_all_accumulated_roots(self):
        sources, connection, _ = self.sources()
        sources.observe((UUID(int=1),))
        first = await sources.prepare("request")
        sources.observe((UUID(int=2),))
        connection.rejected.add(UUID(int=1))
        with self.assertRaises(LiveContextError) as caught:
            await sources.prepare("request", refresh_stale=True)
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assertEqual(connection.locks[-1], (UUID(int=1), UUID(int=2)))
        self.assertEqual(sources.source_for("request"), first.payload_id)
        self.assertEqual(len(connection.payloads), 1)

    async def test_refresh_failure_or_cancellation_never_replaces_cached_identity(self):
        for error in (TimeoutError(), asyncio.CancelledError(), RuntimeError("fixture")):
            with self.subTest(error=type(error).__name__):
                sources, connection, pool = self.sources()
                sources.observe((UUID(int=1),))
                first = await sources.prepare("request")
                sources.observe((UUID(int=2),))
                async def fail(_):
                    raise error
                connection.on_lock = fail
                with self.assertRaises(type(error)):
                    await sources.prepare("request", refresh_stale=True)
                self.assertEqual(sources.source_for("request"), first.payload_id)
                self.assertEqual(len(connection.payloads), 1)
                self.assertEqual(pool.entered, pool.exited)

    async def test_one_and_two_addition_races_refresh_same_request_and_write_only_once(self):
        for races in (1, 2):
            with self.subTest(races=races):
                voice, reservation, connection, socket = self.voice()
                remaining = races
                async def add_after_preparation(roots):
                    nonlocal remaining
                    if any(root.int >= 1000 for root in roots) and remaining:
                        voice._source_context.observe((UUID(int=2 + races - remaining),))
                        remaining -= 1
                connection.on_lock = add_after_preparation
                self.assertTrue(await voice._send_reserved_response_locked(reservation))
                self.assertEqual(self.trace_statuses(voice), ["retry_prewrite"] * races)
                self.assertEqual(len(socket.events), 1)
                self.assertEqual(socket.events[0]["event_id"], reservation.request_id)
                self.assertEqual(socket.events[0]["response"]["metadata"]["jarvis_request_id"], reservation.request_id)
                self.assertEqual(len(connection.payloads), races + 1)
                self.assertEqual({row[1]["request_id"] for row in connection.payloads}, {reservation.request_id})
                self.assertTrue(reservation.submitted)
                voice._begin_context_rebuild.assert_not_called()

    async def test_cached_snapshot_refreshes_after_real_writer_queue_race(self):
        voice, reservation, connection, socket = self.voice()
        lock = voice.openai._send_lock
        await lock.acquire()
        sending = asyncio.create_task(voice._send_reserved_response_locked(reservation))
        try:
            await asyncio.wait_for(connection.materialized.wait(), 0.5)
            self.assertFalse(reservation.submitted)
            voice._source_context.observe((UUID(int=2),))
        finally:
            lock.release()
        self.assertTrue(await asyncio.wait_for(sending, 0.5))
        self.assertEqual(len(socket.events), 1)
        self.assertEqual(self.trace_statuses(voice), ["retry_prewrite"])
        self.assertEqual(connection.locks[-1][:-1], (UUID(int=1), UUID(int=2)))

    async def test_new_sources_while_source_free_writer_waits_get_lineage(self):
        voice, reservation, connection, socket = self.voice(with_sources=False)
        lock = voice.openai._send_lock
        await lock.acquire()
        sending = asyncio.create_task(voice._send_reserved_response_locked(reservation))
        try:
            await asyncio.sleep(0)
            self.assertEqual(voice.openai._foreground_waiters, 1)
            voice._source_context.observe((UUID(int=1),))
        finally:
            lock.release()
        self.assertTrue(await asyncio.wait_for(sending, 0.5))
        self.assertEqual(self.trace_statuses(voice), ["retry_prewrite"])
        self.assertEqual(len(socket.events), 1)
        self.assertEqual(len(connection.payloads), 1)

    async def test_retry_exhaustion_is_explicit_failure_and_retirement_not_success_or_replay(self):
        voice, reservation, connection, socket = self.voice()
        next_root = 2
        async def always_add(roots):
            nonlocal next_root
            if any(root.int >= 1000 for root in roots):
                voice._source_context.observe((UUID(int=next_root),))
                next_root += 1
        connection.on_lock = always_add
        with self.assertRaises(LiveContextError) as caught:
            await voice._send_reserved_response_locked(reservation)
        self.assertEqual(caught.exception.code, "live_context_changed")
        self.assertEqual(self.trace_statuses(voice), ["retry_prewrite", "retry_prewrite", "retry_exhausted"])
        self.assertEqual(len(connection.payloads), 3)
        self.assertEqual(socket.events, [])
        self.assertFalse(reservation.submitted)
        voice._begin_context_rebuild.assert_called_once_with()
        self.assertTrue(voice._context_revoked)

    async def test_source_withdrawal_after_change_still_retires_without_writing(self):
        voice, reservation, connection, socket = self.voice()
        async def withdraw_and_add(roots):
            if any(root.int >= 1000 for root in roots):
                voice._source_context.observe((UUID(int=2),))
                connection.rejected.add(UUID(int=1))
        connection.on_lock = withdraw_and_add
        with self.assertRaises(LiveContextError) as caught:
            await voice._send_reserved_response_locked(reservation)
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assertEqual(self.trace_statuses(voice), [])
        self.assertEqual(socket.events, [])
        voice._begin_context_rebuild.assert_called_once_with()

    async def test_withdrawal_discovered_on_refresh_is_not_bypassed(self):
        voice, reservation, connection, socket = self.voice()
        original = voice._source_context.guard
        @asynccontextmanager
        async def changed(snapshot):
            voice._source_context.observe((UUID(int=2),))
            connection.rejected.add(UUID(int=1))
            async with original(snapshot):
                yield
        voice._source_context.guard = changed
        with self.assertRaises(LiveContextError) as caught:
            await voice._send_reserved_response_locked(reservation)
        self.assertEqual(caught.exception.code, "live_context_revoked")
        self.assertEqual(self.trace_statuses(voice), ["retry_prewrite"])
        self.assertEqual(socket.events, [])
        voice._begin_context_rebuild.assert_called_once_with()

    async def test_explicit_false_submission_required_for_changed_retry(self):
        for submitted in (True, None):
            with self.subTest(submitted=submitted):
                voice, reservation, connection, socket = self.voice()
                reservation.submitted = submitted
                async def changed(_):
                    raise LiveContextError("live_context_changed")
                connection.on_lock = changed
                with self.assertRaises(LiveContextError):
                    await voice._send_reserved_response_locked(reservation)
                self.assertEqual(len(connection.locks), 1)
                self.assertEqual(self.trace_statuses(voice), [])
                self.assertEqual(socket.events, [])

    async def test_error_after_actual_send_is_never_retried(self):
        for error in (LiveContextError("live_context_changed"), TimeoutError(), RuntimeError("unknown write")):
            with self.subTest(error=type(error).__name__):
                voice, reservation, connection, socket = self.voice()
                socket.error = error
                with self.assertRaises(type(error)):
                    await voice._send_reserved_response_locked(reservation)
                self.assertTrue(reservation.submitted)
                self.assertEqual(len(socket.events), 1)
                self.assertEqual(len(connection.payloads), 1)
                self.assertEqual(self.trace_statuses(voice), [])
                voice._begin_context_rebuild.assert_called_once_with()

    async def test_guard_exit_error_after_write_does_not_replay(self):
        voice, reservation, connection, socket = self.voice()
        original = voice._source_context.guard
        @asynccontextmanager
        async def fail_exit(snapshot):
            async with original(snapshot):
                yield
                raise LiveContextError("live_context_changed")
        voice._source_context.guard = fail_exit
        with self.assertRaises(LiveContextError):
            await voice._send_reserved_response_locked(reservation)
        self.assertTrue(reservation.submitted)
        self.assertEqual(len(socket.events), 1)
        self.assertEqual(self.trace_statuses(voice), [])

    async def test_replacement_cancel_closed_or_revoked_turn_does_not_refresh_or_write(self):
        for change in ("generation", "reservation", "route", "provider", "closed", "revoked", "prospective"):
            with self.subTest(change=change):
                voice, reservation, connection, socket = self.voice()
                async def replace(roots):
                    if not any(root.int >= 1000 for root in roots):
                        return
                    voice._source_context.observe((UUID(int=2),))
                    if change == "generation":
                        voice._turn_generation += 1
                    elif change == "reservation":
                        voice._pending_response = None
                    elif change == "route":
                        voice._voice_route_runtime = SimpleNamespace()
                    elif change == "provider":
                        voice.openai = SimpleNamespace()
                    elif change == "closed":
                        voice._closed.set()
                    elif change == "revoked":
                        voice._context_revoked = True
                    else:
                        voice._prospective_attempt_for.return_value = SimpleNamespace(revoked=True)
                connection.on_lock = replace
                self.assertFalse(await voice._send_reserved_response_locked(reservation))
                self.assertFalse(reservation.submitted)
                self.assertEqual(len(connection.payloads), 1)
                self.assertEqual(socket.events, [])
                self.assertEqual(self.trace_statuses(voice), [])
                voice._begin_context_rebuild.assert_not_called()

    async def test_initial_ineligible_turn_never_prepares(self):
        voice, reservation, connection, socket = self.voice()
        voice._turn_generation += 1
        self.assertFalse(await voice._send_reserved_response_locked(reservation))
        self.assertEqual(connection.locks, [])
        self.assertEqual(socket.events, [])

    async def test_refresh_uses_original_deadline_and_cancels_owned_wait(self):
        voice, reservation, connection, socket = self.voice()
        voice.settings = voice.settings.model_copy(update={"OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS": 0.04})
        original_timeout_at = asyncio.timeout_at
        deadlines, joined = [], asyncio.Event()
        original_prepare = voice._source_context.prepare
        async def prepare(request_id, **kwargs):
            if kwargs.get("refresh_stale"):
                try:
                    await asyncio.Event().wait()
                finally:
                    joined.set()
            return await original_prepare(request_id, **kwargs)
        voice._source_context.prepare = prepare
        async def add(roots):
            if any(root.int >= 1000 for root in roots):
                voice._source_context.observe((UUID(int=2),))
        connection.on_lock = add
        def timeout_at(deadline):
            deadlines.append(deadline)
            return original_timeout_at(deadline)
        with patch("app.realtime.session.asyncio.timeout_at", side_effect=timeout_at):
            with self.assertRaises(TimeoutError):
                await voice._send_reserved_response_locked(reservation)
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(deadlines[0], deadlines[1])
        self.assertTrue(joined.is_set())
        self.assertFalse(reservation.submitted)
        self.assertEqual(socket.events, [])
        voice._begin_context_rebuild.assert_called_once_with()

    async def test_cancel_during_refresh_joins_without_detached_retry(self):
        voice, reservation, connection, socket = self.voice()
        entered, joined = asyncio.Event(), asyncio.Event()
        original_prepare = voice._source_context.prepare
        async def prepare(request_id, **kwargs):
            if kwargs.get("refresh_stale"):
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    joined.set()
            return await original_prepare(request_id, **kwargs)
        voice._source_context.prepare = prepare
        async def add(roots):
            if any(root.int >= 1000 for root in roots):
                voice._source_context.observe((UUID(int=2),))
        connection.on_lock = add
        sending = asyncio.create_task(voice._send_reserved_response_locked(reservation))
        await asyncio.wait_for(entered.wait(), 0.5)
        sending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await sending
        self.assertTrue(joined.is_set())
        self.assertFalse(reservation.submitted)
        self.assertEqual(socket.events, [])
        self.assertEqual(len(connection.payloads), 1)
        voice._begin_context_rebuild.assert_not_called()

    async def test_response_controls_still_precede_retried_response(self):
        voice, reservation, connection, socket = self.voice()
        original_guard = voice._source_context.guard
        first = True
        @asynccontextmanager
        async def pending(snapshot):
            nonlocal first
            async with original_guard(snapshot):
                if first:
                    first = False
                    voice._response_controls.busy = True
                yield
        voice._source_context.guard = pending
        async def idle():
            if voice._response_controls.busy:
                await voice.openai.send_event({"type": "output_audio_buffer.clear"})
                voice._response_controls.busy = False
        voice._response_controls.wait_idle.side_effect = idle
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        self.assertEqual([event["type"] for event in socket.events], ["output_audio_buffer.clear", "response.create"])
        self.assertEqual(len(connection.payloads), 1)
        self.assertEqual(self.trace_statuses(voice), [])

    async def test_legacy_and_source_free_paths_still_submit_once(self):
        for with_sources in (False, True):
            with self.subTest(with_sources=with_sources):
                voice, reservation, connection, socket = self.voice(with_sources=with_sources)
                voice._voice_route_runtime = None
                self.assertTrue(await voice._send_reserved_response_locked(reservation))
                self.assertEqual(len(socket.events), 1)
                self.assertEqual(len(connection.payloads), int(with_sources))
                voice._response_controls.wait_idle.assert_not_awaited()
                self.assertEqual(self.trace_statuses(voice), [])
