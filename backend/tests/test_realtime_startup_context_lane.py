"""Actual direct-session initialization and provider writer, offline transport/SQL.

Connection readiness and the provider session.updated ACK are distinct barriers.
These tests do not establish phone playback readiness or production latency.
"""
import asyncio
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.assistant.context_providers import (
    BehaviorContextProvider, CurrentTimeContextProvider, RecentMessagesContextProvider,
    SessionContextProvider,
)
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.sessions.models import JarvisMessage, JarvisSession
from app.settings import JarvisSettings
from app.tasks.context_repository import ContextRepository
from app.tools.module import FunctionToolModule
from app.tools.registry import ToolDefinition
from test_realtime_response_admission_lane import _SourcePool
from test_realtime_shared_view import _Wire


class StartupContextLaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        now, identity = datetime.now(UTC), uuid4()
        self.session_record = JarvisSession(id=identity, title="Synthetic startup", status="active",
            created_at=now, updated_at=now, closed_at=None, last_activity_at=now,
            behavior_instructions="Use short answers.")
        self.history_entered, self.history_release = asyncio.Event(), asyncio.Event()
        self.history_cancelled = asyncio.Event()
        self.connect_entered, self.connect_release = asyncio.Event(), asyncio.Event()
        self.connect_release.set()
        self.messages = [self.message("Prior conversation detail")]
        self.repository = SimpleNamespace(get_recent_messages=AsyncMock(side_effect=self.load_messages),
            create_message=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
        self.behavior_repository = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
            behavior_instructions="Keep my measurements exact.")))
        providers = (CurrentTimeContextProvider(), SessionContextProvider(),
            BehaviorContextProvider(self.behavior_repository), RecentMessagesContextProvider(max_items=20))
        module = FunctionToolModule("startup-fixture", lambda context: (
            ToolDefinition(name="startup_fixture_tool", description="Synthetic tool availability.",
                input_schema={"type": "object", "properties": {}}, handler=AsyncMock()),))
        self.session = DirectRealtimeAudioSession(offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="startup-fixture", owner_id="owner",
                jarvis_session_id=identity, client_kind="phone_audio", session=self.session_record),
            settings=JarvisSettings(_env_file=None), repository=self.repository,
            tool_modules=(module,), context_providers=providers)
        self.session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(), after_session_close=AsyncMock())
        self.app_events = []
        async def app_event(kind, **fields):
            self.app_events.append((kind, fields))
            return True
        self.session._send_app_event = AsyncMock(side_effect=app_event)
        self.provider = self.session.openai
        self.wire = _Wire(self.session)
        self.auto_ack = True
        original_send = self.wire.send
        async def send(raw):
            await original_send(raw)
            if json.loads(raw)["type"] == "session.update" and self.auto_ack:
                self.wire.incoming.put_nowait({"type": "session.updated", "session": {}})
        async def recv():
            return json.dumps(await self.wire.incoming.get())
        self.wire.send, self.wire.recv = send, recv
        async def connect():
            self.connect_entered.set()
            await self.connect_release.wait()
            self.provider._websocket = self.wire
            self.provider.answer = "v=0\r\n"
        self.provider.connect = AsyncMock(side_effect=connect)
        self.pool = _SourcePool()
        self.session._source_context.pool = self.pool
        self.tasks = []
        self.writer_held = False
        self.extra_barriers = []

    async def asyncTearDown(self):
        self.history_release.set()
        self.connect_release.set()
        for event in self.extra_barriers:
            event.set()
        self.release_writer()
        await self.session.close("startup_fixture_complete")
        for task in self.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def message(self, text, *, source=None):
        return JarvisMessage(id=uuid4(), session_id=self.session_record.id, role="user", modality="audio",
            content=text, provider_item_id=None, provider_response_id=None,
            created_at=datetime.now(UTC), source_payload_id=source)

    async def load_messages(self, *args, **kwargs):
        self.history_entered.set()
        try:
            await self.history_release.wait()
        except asyncio.CancelledError:
            self.history_cancelled.set()
            raise
        return self.messages

    def task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    def updates(self):
        return [event for event in self.wire.events if event["type"] == "session.update"]

    async def start(self):
        task = self.task(self.session._initialize_openai_session())
        await asyncio.wait_for(asyncio.shield(task), 1)
        return task

    async def joined_context(self):
        await asyncio.wait_for(asyncio.shield(self.session._direct_startup_context_task), 1)

    async def hold_writer(self):
        await self.provider._send_lock.acquire()
        self.writer_held = True

    def release_writer(self):
        if self.writer_held:
            self.writer_held = False
            self.provider._send_lock.release()

    async def test_ready_and_simple_response_do_not_wait_for_recent_messages(self):
        await self.start()
        self.assertTrue(self.session._realtime_context_ready.is_set())
        self.assertTrue(self.history_entered.is_set())
        self.assertFalse(self.history_release.is_set())
        self.assertFalse(self.session._direct_startup_context_task.done())
        first = self.updates()[0]["session"]
        self.assertIn('"availability": "warming"', first["instructions"])
        self.assertIn("Use short answers.", first["instructions"])
        self.assertIn("startup_fixture_tool", [item["name"] for item in first["tools"]])
        self.assertEqual(first["audio"]["input"]["turn_detection"]["create_response"], False)
        await self.session._handle_openai_event({"type": "input_audio_buffer.speech_started", "item_id": "new"})
        await self.session._handle_openai_event({"type": "input_audio_buffer.committed", "item_id": "new"})
        await self.until(lambda: len(self.wire.responses) == 1)
        self.assertFalse(self.history_release.is_set())
        self.assertIn('"availability": "warming"', self.wire.responses[0]["event"]["response"]["instructions"])
        self.history_release.set()
        await self.joined_context()
        self.assertIn("Prior conversation detail", self.updates()[1]["session"]["instructions"])
        self.assertIn("Keep my measurements exact.", self.updates()[1]["session"]["instructions"])
        self.assertEqual(len(self.wire.responses), 1)  # Enrichment is quiet, not another answer.
        self.assertEqual(self.session._turn_generation, 1)
        self.assertEqual(self.session._direct_startup_context_status, "submitted")

    async def test_neither_connection_nor_initial_provider_ack_is_faked(self):
        self.connect_release.clear()
        self.auto_ack = False
        startup = self.task(self.session._initialize_openai_session())
        await asyncio.wait_for(self.connect_entered.wait(), 1)
        self.assertFalse(self.session._realtime_context_ready.is_set())
        self.assertEqual(self.app_events, [])
        self.connect_release.set()
        await self.until(lambda: len(self.updates()) == 1)
        self.assertIn("audio.webrtc.answer", [event[0] for event in self.app_events])
        self.assertFalse(self.session._realtime_context_ready.is_set())
        self.assertFalse(startup.done())
        self.wire.incoming.put_nowait({"type": "session.updated"})
        await asyncio.wait_for(startup, 1)
        self.assertTrue(self.session._realtime_context_ready.is_set())
        self.assertFalse(self.history_release.is_set())

    async def test_optional_context_provider_is_also_off_readiness_path(self):
        self.history_release.set()
        entered, release = asyncio.Event(), asyncio.Event()
        self.extra_barriers.append(release)
        async def behavior(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(behavior_instructions="Late preferences")
        self.behavior_repository.get.side_effect = behavior
        await self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertTrue(self.session._realtime_context_ready.is_set())
        self.assertFalse(release.is_set())
        release.set()
        await self.joined_context()
        self.assertIn("Late preferences", self.updates()[1]["session"]["instructions"])

    async def test_canonical_history_loads_later_with_original_source_roots(self):
        self.history_release.set()
        entered, release = asyncio.Event(), asyncio.Event()
        self.extra_barriers.append(release)
        root = uuid4()
        record = SimpleNamespace(id=uuid4(), kind="user_transcript", availability="available", is_current=True,
            origin_sequence=1, content={"item_id": "prior-item"}, payload_id=root,
            as_context=lambda: {"id": "retained", "origin_sequence": 1,
                "content": {"text": "Canonical retained detail"}})
        async def history(*args):
            entered.set()
            await release.wait()
            return {"total": 1, "records": [record]}
        repository = ContextRepository(SimpleNamespace(pool=None))
        repository.voice_history = AsyncMock(side_effect=history)
        self.session._context_capture = SimpleNamespace(repository=repository, close=AsyncMock())
        await self.start()
        await asyncio.wait_for(entered.wait(), 1)
        self.assertFalse(self.session._source_context.has_sources)
        release.set()
        await self.joined_context()
        self.assertIn("Canonical retained detail", self.updates()[1]["session"]["instructions"])
        self.assertIn(root, self.session._source_context._sources)
        self.assertIn((root,), self.pool.locks)
        self.assertEqual(self.session._assistant_context.voice_continuity_source_ids, (root,))

    async def test_newer_full_refresh_supersedes_slow_startup_snapshot(self):
        await self.start()
        old = self.messages
        async def messages(*args, **kwargs):
            return [self.message("Corrected current context")]
        self.repository.get_recent_messages.side_effect = messages
        await self.session._send_session_update()
        self.assertEqual(len(self.updates()), 2)
        self.messages = old
        self.history_release.set()
        await self.joined_context()
        self.assertEqual(len(self.updates()), 2)
        self.assertIn("Corrected current context", self.updates()[-1]["session"]["instructions"])
        self.assertEqual(self.session._direct_startup_context_status, "superseded")

    async def test_queued_history_rechecks_refresh_identity_at_actual_writer(self):
        root = uuid4()
        self.messages = [self.message("Obsolete source", source=root)]
        await self.start()
        await self.hold_writer()
        self.history_release.set()
        await self.until(lambda: self.provider._foreground_waiters == 1)
        self.repository.get_recent_messages.side_effect = AsyncMock(return_value=[self.message("New snapshot")])
        refreshed = self.task(self.session._send_session_update())
        await self.until(lambda: self.provider._foreground_waiters == 2)
        self.release_writer()
        await asyncio.wait_for(refreshed, 1)
        await self.joined_context()
        self.assertEqual(len(self.updates()), 2)
        self.assertIn("New snapshot", self.updates()[-1]["session"]["instructions"])
        self.assertNotIn(root, self.session._source_context._sources)

    async def test_queued_history_cannot_cross_source_retirement(self):
        self.messages = [self.message("Old context", source=uuid4())]
        await self.start()
        await self.hold_writer()
        self.history_release.set()
        await self.until(lambda: self.provider._foreground_waiters == 1)
        self.session._context_revoked = True
        self.release_writer()
        await self.joined_context()
        self.assertEqual(len(self.updates()), 1)
        self.assertFalse(self.session._source_context.has_sources)

    async def test_withdrawn_history_is_not_sent_or_enrolled(self):
        self.messages = [self.message("Withdrawn history", source=uuid4())]
        await self.start()
        self.pool.available = False
        self.history_release.set()
        await self.joined_context()
        self.assertFalse(self.session._source_context.has_sources)
        self.assertNotIn("Withdrawn history", " ".join(update["session"]["instructions"] for update in self.updates()))
        self.assertEqual(self.session._direct_startup_context_status, "unavailable")
        self.assertIn('"availability": "unavailable"', self.updates()[-1]["session"]["instructions"])

    async def test_bounded_optional_load_timeout_reports_unavailable_not_empty_success(self):
        with patch("app.realtime.direct_session.STARTUP_CONTEXT_LOAD_TIMEOUT_SECONDS", 0.02):
            await self.start()
            await self.joined_context()
        self.assertTrue(self.history_cancelled.is_set())
        self.assertTrue(self.session._realtime_context_ready.is_set())
        self.assertEqual(self.session._direct_startup_context_status, "unavailable")
        self.assertIn('"availability": "unavailable"', self.updates()[-1]["session"]["instructions"])

    async def test_close_cancels_and_joins_background_history(self):
        await self.start()
        worker = self.session._direct_startup_context_task
        await asyncio.wait_for(self.session.close("closing_during_history"), 1)
        self.assertTrue(worker.done())
        self.assertTrue(self.history_cancelled.is_set())
        self.assertEqual(len(self.updates()), 1)
        self.wire.close.assert_awaited_once()

    async def test_connection_failure_joins_context_loader_without_claiming_ready(self):
        async def fail_connect():
            await self.history_entered.wait()
            raise RuntimeError("synthetic connection failure")
        self.provider.connect.side_effect = fail_connect
        with self.assertRaisesRegex(RuntimeError, "synthetic connection failure"):
            await self.start()
        self.assertFalse(self.session._realtime_context_ready.is_set())
        self.assertTrue(self.session._direct_startup_context_task.done())
        self.assertTrue(self.history_cancelled.is_set())
        self.assertEqual(self.updates(), [])

    async def test_legacy_keeps_full_context_before_provider_readiness(self):
        self.session._voice_route_runtime = None
        startup = self.task(self.session._initialize_openai_session())
        await asyncio.wait_for(self.history_entered.wait(), 1)
        await self.until(lambda: any(kind == "audio.webrtc.answer" for kind, _ in self.app_events))
        self.assertEqual(self.updates(), [])
        self.assertFalse(self.session._realtime_context_ready.is_set())
        self.history_release.set()
        await asyncio.wait_for(startup, 1)
        self.assertEqual(len(self.updates()), 1)
        self.assertIn("Prior conversation detail", self.updates()[0]["session"]["instructions"])
        self.assertIsNone(self.session._direct_startup_context_task)
