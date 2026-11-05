"""Bounded response persistence and extra-source admission, without network I/O."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import test_realtime_image_tools as fixtures
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveResponseSources
from app.tools.provenance import SourceBoundToolResult


class SourceEnrollmentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.current = True
        self.root = uuid4()
        self.sources = LiveResponseSources(None, owner_id="owner", session_id=uuid4(), audio_session_id="audio")
        self.result = SourceBoundToolResult({"status": "success", "value": "synthetic"},
            SimpleNamespace(payload_id=self.root, id=uuid4()))
        self.connection = SimpleNamespace(is_in_transaction=lambda: True, fetchval=AsyncMock(return_value=True))
        self.sent = []
        async def send(event, *, write_guard):
            async with self.sources.use_transaction(self.connection):
                async with write_guard():
                    value = event()
                    if value is None:
                        return False
                    self.sent.append(value)
                    return True
        self.session = SimpleNamespace(_closed=asyncio.Event(), _context_revoked=False,
            _source_context=self.sources, _begin_context_rebuild=Mock(),
            _tool_output_event=RealtimeAudioSession._tool_output_event,
            openai=SimpleNamespace(send_event=AsyncMock(side_effect=send)))

    async def write(self):
        return await RealtimeAudioSession._write_tool_output(self.session, "call", self.result,
                                                            is_current=lambda: self.current)

    async def test_obsolete_result_never_enrolls_or_queries_its_root(self):
        self.current = False
        self.assertFalse(await self.write())
        self.assertFalse(self.sources.has_sources)
        self.connection.fetchval.assert_not_awaited()
        self.session.openai.send_event.assert_not_awaited()

    async def test_result_obsoleted_during_source_check_does_not_enroll(self):
        async def lock(*args):
            self.assertFalse(self.sources.has_sources)
            self.current = False
            return True
        self.connection.fetchval.side_effect = lock
        self.assertFalse(await self.write())
        self.assertFalse(self.sources.has_sources)
        self.assertEqual(self.sent, [])
        self.session._begin_context_rebuild.assert_not_called()

    async def test_withdrawn_obsolete_extra_source_does_not_poison_current_session(self):
        async def lock(*args):
            self.current = False
            return False
        self.connection.fetchval.side_effect = lock
        self.assertFalse(await self.write())
        self.assertFalse(self.sources.has_sources)
        self.session._begin_context_rebuild.assert_not_called()

    async def test_live_result_enrolls_only_after_source_lock(self):
        async def lock(*args):
            self.assertFalse(self.sources.has_sources)
            self.assertEqual(args[-1], [self.root])
            return True
        self.connection.fetchval.side_effect = lock
        self.assertTrue(await self.write())
        self.assertTrue(self.sources.has_sources)
        self.assertEqual(len(self.sent), 1)

    async def test_current_source_failure_still_requires_context_rebuild(self):
        self.connection.fetchval.return_value = False
        self.assertFalse(await self.write())
        self.assertFalse(self.sources.has_sources)
        self.session._begin_context_rebuild.assert_called_once()


class CompletionLaneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = fixtures.RealtimeImageToolTests()
        await self.fixture.asyncSetUp()
        self.session = self.fixture.session
        await self.fixture.start()
        self.session._voice_route_runtime = SimpleNamespace(
            before_session_close=AsyncMock(return_value=True), after_session_close=AsyncMock(return_value=True))
        self.persisting, self.release = asyncio.Event(), asyncio.Event()
        async def persist(**kwargs):
            self.persisting.set()
            await self.release.wait()
        self.session._persist_assistant_transcript = AsyncMock(side_effect=persist)

    async def asyncTearDown(self):
        self.release.set()
        await self.fixture.asyncTearDown()

    async def test_database_wait_does_not_block_later_provider_control(self):
        seen = asyncio.Event()
        original = self.session._handle_openai_event
        async def handle(event):
            await original(event)
            if event.get("event_id") == "control-after-persistence":
                seen.set()
        self.session._handle_openai_event = handle
        self.fixture.done()
        await asyncio.wait_for(self.persisting.wait(), 1)
        self.fixture.wire.incoming.put_nowait({"type": "input_audio_buffer.committed",
                                             "event_id": "control-after-persistence", "item_id": "input"})
        await asyncio.wait_for(seen.wait(), 0.25)
        self.assertFalse(self.release.is_set())
        self.release.set()
        await self.fixture.until(lambda: not self.session._provider_completion_tasks)

    async def test_duplicate_terminal_events_have_one_persistence_owner(self):
        self.fixture.done()
        self.fixture.done()
        await asyncio.wait_for(self.persisting.wait(), 1)
        await asyncio.wait_for(self.fixture.wire.incoming.join(), 1)
        self.assertEqual(len(self.session._provider_completion_tasks), 1)
        self.release.set()
        await self.fixture.until(lambda: not self.session._provider_completion_tasks)
        self.session._persist_assistant_transcript.assert_awaited_once()

    async def test_barge_in_while_persisting_cannot_resume_old_turn(self):
        self.fixture.done()
        await asyncio.wait_for(self.persisting.wait(), 1)
        await self.session._handle_client_event({"type": "response.cancel"})
        self.release.set()
        await self.fixture.until(lambda: not self.session._provider_completion_tasks)
        self.assertIsNone(self.session._tool_followup)
        self.assertFalse(self.session._normal_turn_pending)

    async def test_close_joins_pending_persistence(self):
        self.fixture.done()
        await asyncio.wait_for(self.persisting.wait(), 1)
        await self.session.close("fixture_shutdown")
        self.assertFalse(self.session._provider_completion_tasks)

    async def test_completion_capacity_is_bounded(self):
        started, release = asyncio.Event(), asyncio.Event()
        active = peak = 0
        async def finish(*args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                started.set()
            try:
                await release.wait()
            finally:
                active -= 1
        self.session._finish_provider_response = finish
        for index in range(32):
            self.session._schedule_provider_completion(f"response-cap-{index}", {"status": "completed"})
        await asyncio.wait_for(started.wait(), 1)
        self.assertEqual(active, 2)
        self.session._schedule_provider_completion("response-overflow", {"status": "completed"})
        self.assertTrue(self.session._closed.is_set())
        self.assertEqual(self.session._close_reason, "provider_completion_queue_exhausted")
        self.assertEqual(len(self.session._provider_completion_tasks), 32)
        release.set()
        await self.fixture.until(lambda: not self.session._provider_completion_tasks)
        self.assertEqual(peak, 2)
