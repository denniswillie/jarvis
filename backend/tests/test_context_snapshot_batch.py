"""Projection query shape and decoding; real SQL semantics have a separate gate."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.context_events import ContextRecord
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError


class ContextSnapshotBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session = "snapshot-owner", uuid4()
        self.record = ContextRecord(id=uuid4(), owner_id=self.owner, session_id=self.session,
            sequence=9, origin_sequence=2, source_namespace="fixture", source_event_id="one",
            revision=2, supersedes_event_id=uuid4(), kind="user_transcript", trust="user_input",
            finality="revised", source_time=None, recorded_at=datetime.now(UTC), payload_id=uuid4(),
            availability="available", content_digest=None, content={"text": "Corrected context"}, is_current=True)
        self.result = dict(watermark=9, total=5, pinned_total=3,
            recent=json.dumps([self.record.model_dump(mode="json")]), pinned="[]",
            selected_pinned_ids=[self.record.id])
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value=True),
            fetchrow=AsyncMock(side_effect=lambda *args: self.result))
        self.enters = 0

        @asynccontextmanager
        async def read_transaction():
            self.enters += 1
            yield self.connection

        self.tasks = SimpleNamespace(pool=object(), read_transaction=read_transaction,
            load=AsyncMock(return_value=object()))
        self.context = ContextRepository(self.tasks)

    async def test_single_bounded_contents_query_preserves_typed_result(self):
        result = await self.context.snapshot(self.owner, self.session, max_messages=4, max_pins=1)
        self.assertEqual(result["recent"], [self.record])
        self.assertEqual(result["selected_pinned_ids"], {self.record.id})
        self.assertEqual((result["watermark"], result["total"], result["pinned_total"]), (9, 5, 3))
        self.assertEqual(self.enters, 1)
        self.connection.fetchval.assert_awaited_once()
        self.connection.fetchrow.assert_awaited_once()
        query, *args = self.connection.fetchrow.call_args.args
        self.assertEqual(args, [self.owner, [self.session], 4, 1])
        self.assertIn("limit $3", query)
        self.assertIn("limit $4", query)
        self.assertIn("e.id in (select id from selected_ids)", query)
        self.assertNotIn("pg_advisory", query)

    async def test_empty_scope_keeps_owner_watermark_and_decodes_lists(self):
        self.result = dict(watermark=12, total=0, pinned_total=0, recent=[], pinned=[], selected_pinned_ids=[])
        result = await self.context.snapshot(self.owner, None, max_messages=1, max_pins=1)
        self.assertEqual((result["watermark"], result["total"], result["scope_session_count"]), (12, 0, 0))
        self.assertEqual(result["recent"], [])
        self.connection.fetchval.assert_not_awaited()
        self.assertEqual(self.connection.fetchrow.call_args.args[2], [])

    async def test_foreign_session_rejected_before_contents(self):
        self.connection.fetchval.return_value = False
        with self.assertRaises(TaskError) as error:
            await self.context.snapshot(self.owner, self.session, max_messages=1, max_pins=1)
        self.assertEqual(error.exception.code, "context_session_not_found")
        self.connection.fetchrow.assert_not_awaited()

    async def test_task_session_scope_and_sources_preserved(self):
        extra, task_id = uuid4(), uuid4()
        self.context._task_sessions = AsyncMock(return_value=([self.session, extra], [self.record]))
        result = await self.context.snapshot(self.owner, self.session, max_messages=2, max_pins=1, task_id=task_id)
        self.tasks.load.assert_awaited_once_with(self.connection, self.owner, task_id)
        self.assertEqual(result["scope_sources"], [self.record])
        self.assertEqual(result["scope_session_count"], 2)
        self.assertEqual(self.connection.fetchrow.call_args.args[2], [self.session, extra])

    async def test_invalid_bounds_do_not_acquire_connection(self):
        for messages, pins in ((0, 1), (513, 1), (1, 0), (1, 129)):
            with self.subTest(messages=messages, pins=pins), self.assertRaises(ValueError):
                await self.context.snapshot(self.owner, self.session, max_messages=messages, max_pins=pins)
        self.assertEqual(self.enters, 0)
