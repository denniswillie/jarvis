from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.context_refresh import RealtimeContextRefreshCoordinator
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveResponseSources
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.socket_sender import WebSocketEventSender
from fastapi import WebSocketDisconnect
from app.runtime.events import (
    DomainEvent,
    DomainEventName,
    MemoryItemChangedPayload,
    OwnerWorkingMemoryUpdatedPayload,
    WorkingMemoryUpdatedPayload,
)


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


class _RealtimeSession:
    def __init__(self) -> None:
        self.working_revisions: list[int] = []
        self.owner_revisions: list[int] = []
        self.memory_events: list[DomainEvent] = []

    async def notify_working_memory_updated(self, revision: int) -> None:
        self.working_revisions.append(revision)

    async def notify_memory_item_changed(self, event: DomainEvent) -> None:
        self.memory_events.append(event)

    async def notify_owner_working_memory_updated(self, revision: int) -> None:
        self.owner_revisions.append(revision)


class _PromptContext:
    def __init__(self, working_memory, marker: str) -> None:
        self.working_memory = working_memory
        self._marker = marker

    def as_prompt_payload(self):
        return {"marker": self._marker, "working_memory": self.working_memory}


class _SerialWebSocket:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.active_sends = 0
        self.max_active_sends = 0

    async def send_text(self, message: str) -> None:
        self.active_sends += 1
        self.max_active_sends = max(self.max_active_sends, self.active_sends)
        try:
            await asyncio.sleep(0)
            self.messages.append(message)
        finally:
            self.active_sends -= 1


class RealtimeContextRefreshTests(unittest.IsolatedAsyncioTestCase):
    async def test_coalesces_revisions_and_never_applies_older_state(self) -> None:
        refreshed: list[int] = []

        async def refresh(revision: int) -> int:
            refreshed.append(revision)
            return revision

        coordinator = RealtimeContextRefreshCoordinator(
            refresh,
            minimum_interval_seconds=0,
        )
        await coordinator.notify(1)
        await coordinator.notify(3)
        await coordinator.notify(2)
        await coordinator.wait_idle()
        await coordinator.notify(2)
        await coordinator.wait_idle()

        self.assertEqual(refreshed, [3])
        self.assertEqual(coordinator.applied_revision, 3)

    async def test_refresh_failure_isolated_and_later_revision_can_apply(self) -> None:
        attempts: list[int] = []

        async def refresh(revision: int) -> int:
            attempts.append(revision)
            if len(attempts) == 1:
                raise RuntimeError("provider unavailable")
            return revision

        coordinator = RealtimeContextRefreshCoordinator(
            refresh,
            minimum_interval_seconds=0,
            retry_delay_seconds=0,
        )
        await coordinator.notify(1)
        await coordinator.wait_idle()
        await coordinator.notify(2)
        await coordinator.wait_idle()

        self.assertEqual(attempts, [1, 1, 2])
        self.assertEqual(coordinator.applied_revision, 2)

    async def test_initial_refresh_timestamp_throttles_next_update(self) -> None:
        now = 10.0
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            nonlocal now
            sleeps.append(seconds)
            now += seconds

        coordinator = RealtimeContextRefreshCoordinator(
            AsyncMock(return_value=2),
            minimum_interval_seconds=2,
            monotonic=lambda: now,
            sleep=sleep,
        )
        coordinator.initialize(1)
        await coordinator.notify(2)
        await coordinator.wait_idle()

        self.assertEqual(sleeps, [2.0])

    async def test_non_revision_invalidation_rebuilds_after_latest_revision(self) -> None:
        requested: list[int | None] = []

        async def refresh(revision: int | None) -> int | None:
            requested.append(revision)
            return 4

        coordinator = RealtimeContextRefreshCoordinator(
            refresh,
            minimum_interval_seconds=0,
        )
        coordinator.initialize(4)
        await coordinator.invalidate()
        await coordinator.wait_idle()

        self.assertEqual(requested, [None])
        self.assertEqual(coordinator.applied_revision, 4)

    async def test_manager_routes_working_and_durable_memory_events(self) -> None:
        manager = RealtimeSessionManager()
        session_id = uuid4()
        memory_id = uuid4()
        session = _RealtimeSession()
        await manager.register(
            audio_session_id="audio",
            owner_id="owner",
            jarvis_session_id=session_id,
            session=session,
        )
        await manager.handle_working_memory_updated(
            DomainEvent(
                name=DomainEventName.WORKING_MEMORY_UPDATED,
                payload=WorkingMemoryUpdatedPayload(
                    owner_id="owner",
                    session_id=session_id,
                    revision=4,
                ),
            )
        )
        memory_event = DomainEvent(
            name=DomainEventName.MEMORY_ITEM_CHANGED,
            payload=MemoryItemChangedPayload(
                owner_id="owner",
                session_id=session_id,
                memory_item_id=memory_id,
                memory_type="semantic_fact",
                change_type="updated",
                reason="user_correction",
            ),
            occurred_at=NOW,
        )
        await manager.handle_memory_item_changed(memory_event)
        await manager.handle_owner_working_memory_updated(
            DomainEvent(
                name=DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
                payload=OwnerWorkingMemoryUpdatedPayload(
                    owner_id="owner",
                    revision=6,
                ),
            )
        )

        self.assertEqual(session.working_revisions, [4])
        self.assertEqual(session.owner_revisions, [6])
        self.assertEqual(session.memory_events, [memory_event])

    async def test_manager_rejects_session_scoped_memory_event_for_wrong_owner(self) -> None:
        manager = RealtimeSessionManager()
        session_id = uuid4()
        session = _RealtimeSession()
        await manager.register(
            audio_session_id="audio",
            owner_id="owner",
            jarvis_session_id=session_id,
            session=session,
        )

        await manager.handle_memory_item_changed(
            DomainEvent(
                name=DomainEventName.MEMORY_ITEM_CHANGED,
                payload=MemoryItemChangedPayload(
                    owner_id="different-owner",
                    session_id=session_id,
                    memory_item_id=uuid4(),
                    change_type="updated",
                ),
            )
        )

        self.assertEqual(session.memory_events, [])

    async def test_realtime_refresh_sends_complete_instructions_and_serialized_hints(self) -> None:
        working_memory = _working_memory_payload(
            revision=2,
            expires_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        session = _audio_session([_PromptContext(working_memory, "complete-context")])
        try:
            applied = await session._refresh_realtime_context(2)
            await asyncio.gather(
                session._send_app_event("memory.updated", change_id="one"),
                session._send_app_event("memory.updated", change_id="two"),
            )

            self.assertEqual(applied, 2)
            update = session.openai.send_event.await_args.args[0]
            self.assertEqual(update["type"], "session.update")
            self.assertIn("complete-context", update["session"]["instructions"])
            self.assertIn("Runtime context", update["session"]["instructions"])
            self.assertEqual(update["session"]["tools"], [{"type": "function"}])
            self.assertEqual(session.websocket.max_active_sends, 1)
            event_types = [json.loads(message)["type"] for message in session.websocket.messages]
            self.assertEqual(
                event_types,
                ["working_memory.updated", "memory.updated", "memory.updated"],
            )
        finally:
            await _stop_audio_session_tasks(session)

    async def test_expiry_invalidates_context_without_a_new_revision(self) -> None:
        empty_memory = {"available": True, "session": None, "owner": None}
        session = _audio_session([_PromptContext(empty_memory, "expired-cleared")])
        session._schedule_working_memory_expiry(
            _working_memory_payload(
                revision=3,
                expires_at=datetime.now(UTC) - timedelta(seconds=1),
            )
        )
        try:
            expiry_task = session._working_memory_expiry_task
            self.assertIsNotNone(expiry_task)
            await expiry_task
            await session._context_refresh.wait_idle()

            update = session.openai.send_event.await_args.args[0]
            self.assertIn("expired-cleared", update["session"]["instructions"])
            self.assertIn('"session": null', update["session"]["instructions"])
            self.assertFalse(session._closed.is_set())
        finally:
            await _stop_audio_session_tasks(session)

    async def test_refresh_send_failure_does_not_close_audio(self) -> None:
        session = _audio_session(
            [
                _PromptContext(
                    {"available": True, "session": None, "owner": None},
                    "failure",
                ),
                _PromptContext(
                    {"available": True, "session": None, "owner": None},
                    "failure-retry",
                ),
            ]
        )
        session.openai.send_event.side_effect = RuntimeError("provider disconnected")
        try:
            await session._context_refresh.invalidate()
            await session._context_refresh.wait_idle()

            self.assertFalse(session._closed.is_set())
        finally:
            await _stop_audio_session_tasks(session)

    async def test_unavailable_refresh_preserves_current_prompt_and_expiry(self) -> None:
        unavailable = {"available": False, "session": None, "owner": None}
        session = _audio_session([_PromptContext(unavailable, "unavailable")])
        existing_expiry = asyncio.create_task(asyncio.sleep(60))
        session._working_memory_expiry_task = existing_expiry
        try:
            with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                await session._refresh_realtime_context(None)

            session.openai.send_event.assert_not_awaited()
            self.assertIs(session._working_memory_expiry_task, existing_expiry)
            self.assertFalse(existing_expiry.cancelled())
        finally:
            await _stop_audio_session_tasks(session)

    async def test_close_during_context_build_does_not_emit_late_updates(self) -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        context = _PromptContext(
            _working_memory_payload(revision=2, expires_at=datetime.now(UTC) + timedelta(minutes=5)),
            "late-context",
        )
        session = _audio_session([])

        async def build():
            entered.set()
            await release.wait()
            return context

        session._build_assistant_context = AsyncMock(side_effect=build)
        task = asyncio.create_task(session._refresh_realtime_context(2))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            session._mark_closing("client_closed")
            release.set()
            self.assertIsNone(await asyncio.wait_for(task, 1))
            session.openai.send_event.assert_not_awaited()
            self.assertEqual(session.websocket.messages, [])
            self.assertIsNone(session._working_memory_expiry_task)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await _stop_audio_session_tasks(session)

    async def test_departure_during_hint_prevents_provider_refresh(self) -> None:
        session = _audio_session([_PromptContext(
            _working_memory_payload(revision=2, expires_at=datetime.now(UTC) + timedelta(minutes=5)),
            "departed",
        )])
        session.websocket.send_text = AsyncMock(side_effect=WebSocketDisconnect())
        try:
            with self.assertNoLogs("app.realtime.session", level="WARNING"):
                self.assertIsNone(await session._refresh_realtime_context(2))
            self.assertTrue(session._closed.is_set())
            self.assertEqual(session._close_reason, "client_disconnect")
            session.openai.send_event.assert_not_awaited()
            self.assertIsNone(session._working_memory_expiry_task)
        finally:
            await _stop_audio_session_tasks(session)

    async def test_closed_session_does_not_wait_for_context_readiness(self) -> None:
        session = _audio_session([])
        session._realtime_context_ready.clear()
        session._mark_closing("client_closed")
        try:
            self.assertIsNone(await asyncio.wait_for(session._refresh_realtime_context(None), 1))
            session._build_assistant_context.assert_not_awaited()
        finally:
            await _stop_audio_session_tasks(session)

    async def test_stop_disables_notifications_before_joining_cancelled_refresh(self) -> None:
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def refresh(revision):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                raise RuntimeError("transport closed after cancellation")

        callback = AsyncMock(side_effect=refresh)
        coordinator = RealtimeContextRefreshCoordinator(
            callback, minimum_interval_seconds=0, retry_delay_seconds=0,
        )
        close_task = None
        try:
            with self.assertNoLogs("app.realtime.context_refresh", level="ERROR"):
                await coordinator.notify(1)
                await asyncio.wait_for(entered.wait(), 1)
                coordinator.stop()
                await asyncio.wait_for(cancelled.wait(), 1)
                self.assertFalse(await coordinator.notify(2))
                self.assertFalse(await coordinator.invalidate())
                close_task = asyncio.create_task(coordinator.close())
                release.set()
                await asyncio.wait_for(close_task, 1)
            callback.assert_awaited_once_with(1)
            self.assertIsNone(coordinator._task)
        finally:
            release.set()
            await coordinator.close()
            if close_task is not None:
                await asyncio.gather(close_task, return_exceptions=True)


def _working_memory_payload(*, revision: int, expires_at: datetime):
    return {
        "available": True,
        "session": {
            "revision": revision,
            "observed_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
            "expires_at": expires_at.isoformat(),
            "state": {"scene": {"label": "desk"}},
        },
        "owner": None,
    }


def _audio_session(contexts):
    session = object.__new__(RealtimeAudioSession)
    session.context = SimpleNamespace(owner_id="owner", jarvis_session_id=uuid4())
    session._source_context = LiveResponseSources(None, owner_id="owner",
        session_id=session.context.jarvis_session_id, audio_session_id="fixture")
    session.settings = SimpleNamespace(
        OPENAI_REALTIME_MODEL="gpt-realtime",
        OPENAI_REALTIME_VOICE="alloy",
        OPENAI_REALTIME_TRANSCRIPTION_MODEL="",
        JARVIS_AUDIO_FORMAT="pcm16",
        JARVIS_AUDIO_SAMPLE_RATE=24_000,
    )
    session.tool_registry = SimpleNamespace(
        openai_realtime_tools=lambda: [{"type": "function"}]
    )
    session.openai = SimpleNamespace(send_event=AsyncMock())
    session.websocket = _SerialWebSocket()
    session._app_sender = WebSocketEventSender(
        session.websocket, on_closed=lambda: session._mark_closing("client_disconnect"),
    )
    session._closed = asyncio.Event()
    session._realtime_context_ready = asyncio.Event()
    session._realtime_context_ready.set()
    session._working_memory_expiry_task = None
    session._build_assistant_context = AsyncMock(side_effect=contexts)
    session._context_refresh = RealtimeContextRefreshCoordinator(
        session._refresh_realtime_context,
        minimum_interval_seconds=0,
        retry_delay_seconds=0,
    )
    return session


async def _stop_audio_session_tasks(session) -> None:
    expiry_task = session._working_memory_expiry_task
    session._working_memory_expiry_task = None
    if expiry_task is not None:
        expiry_task.cancel()
        try:
            await expiry_task
        except asyncio.CancelledError:
            pass
    await session._context_refresh.close()


if __name__ == "__main__":
    unittest.main()
