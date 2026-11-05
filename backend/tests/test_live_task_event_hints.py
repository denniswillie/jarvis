import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.task_events import TaskEventObserver


class TaskEventHintTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner = "task-event-fixture"
        self.task_id = uuid4()
        self.heads = [{"id": self.task_id, "last_event_sequence": 3}]
        self.connection = SimpleNamespace(fetch=AsyncMock(side_effect=lambda *args: self.heads))
        @asynccontextmanager
        async def acquire():
            yield self.connection
        self.events = [dict(id=uuid4(), sequence=i, event_type=kind) for i, kind in
            [(1, "created"), (2, "action_accepted"), (3, "succeeded")]]
        self.repository = SimpleNamespace(pool=SimpleNamespace(acquire=acquire),
            events=AsyncMock(return_value=self.events))
        self.consume = AsyncMock()
        self.observer = TaskEventObserver(self.repository, self.owner, self.consume)

    async def test_one_commit_hint_without_speech_or_task_execution(self):
        await self.observer.refresh()
        self.consume.assert_awaited_once_with(self.events[1]["id"])
        self.repository.events.assert_awaited_once_with(self.owner, self.task_id, after=0, limit=16)
        self.assertEqual(self.connection.fetch.call_args.args[1], self.owner)

    async def test_unchanged_head_does_not_reread_or_duplicate(self):
        await self.observer.refresh()
        await self.observer.refresh()
        self.assertEqual(self.repository.events.await_count, 1)
        self.assertEqual(self.consume.await_count, 1)

    async def test_failed_consumer_preserves_cursor_for_next_refresh(self):
        self.consume.side_effect = RuntimeError("temporarily unavailable")
        await self.observer.refresh()
        self.assertEqual(self.observer._cursors[self.task_id], 1)
        self.consume.side_effect = None
        await self.observer.refresh()
        self.assertEqual(self.observer._cursors[self.task_id], 3)
        self.assertEqual(self.consume.await_count, 2)

    async def test_new_commit_after_an_older_timestamp_is_not_skipped(self):
        await self.observer.refresh()
        new = dict(id=uuid4(), sequence=4, event_type="action_accepted", created_at="earlier")
        self.heads[0]["last_event_sequence"] = 4
        self.repository.events.return_value = [new]
        await self.observer.refresh()
        self.consume.assert_awaited_with(new["id"])

    async def test_concurrent_refreshes_share_cursor(self):
        await asyncio.gather(self.observer.refresh(), self.observer.refresh())
        self.consume.assert_awaited_once()

    async def test_bounded_pages_and_cursor_storage(self):
        self.heads = [{"id": uuid4(), "last_event_sequence": 1} for _ in range(32)]
        self.repository.events.return_value = [dict(id=uuid4(), sequence=1, event_type="created")]
        self.observer._cursors = {uuid4(): 1 for _ in range(64)}
        await self.observer.refresh()
        self.assertEqual(self.repository.events.await_count, 4)
        self.assertEqual(len(self.observer._cursors), 4)

    async def test_cancellation_is_not_swallowed_or_marked_consumed(self):
        self.consume.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.observer.refresh()
        self.assertEqual(self.observer._cursors[self.task_id], 1)


if __name__ == "__main__":
    unittest.main()
