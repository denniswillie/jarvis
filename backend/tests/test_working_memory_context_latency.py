from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from app.assistant.memory_context import WorkingMemoryContextProvider


class WorkingMemoryContextLatencyTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return SimpleNamespace(owner_id="owner", session=SimpleNamespace(id=uuid4()),
                               requested_at_utc=datetime.now(UTC))

    async def test_session_and_owner_reads_run_concurrently(self):
        started = set()
        both_started = asyncio.Event()

        async def read(name):
            started.add(name)
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            return None

        service = SimpleNamespace(
            get_session_working_memory=lambda **_: read("session"),
            get_owner_working_memory=lambda **_: read("owner"),
        )
        section = await WorkingMemoryContextProvider(service, read_timeout_seconds=0.2).build_section(self.request())
        self.assertEqual(started, {"session", "owner"})
        self.assertTrue(section.payload["available"])

    async def test_slow_optional_read_is_cancelled_and_returns_unavailable(self):
        owner_cancelled = asyncio.Event()

        async def session(**_):
            return None

        async def owner(**_):
            try:
                await asyncio.Future()
            finally:
                owner_cancelled.set()

        service = SimpleNamespace(get_session_working_memory=session, get_owner_working_memory=owner)
        section = await WorkingMemoryContextProvider(service, read_timeout_seconds=0.01).build_section(self.request())
        self.assertEqual(section.payload, {"available": False, "session": None, "owner": None})
        self.assertTrue(owner_cancelled.is_set())

    async def test_context_cancellation_joins_both_database_reads(self):
        started = set()
        cancelled = set()
        both_started = asyncio.Event()

        async def read(name):
            started.add(name)
            if len(started) == 2:
                both_started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.add(name)

        service = SimpleNamespace(
            get_session_working_memory=lambda **_: read("session"),
            get_owner_working_memory=lambda **_: read("owner"),
        )
        task = asyncio.create_task(WorkingMemoryContextProvider(service).build_section(self.request()))
        await both_started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(cancelled, {"session", "owner"})

    async def test_database_failure_cancels_other_read_instead_of_leaking_it(self):
        owner_started = asyncio.Event()
        owner_cancelled = asyncio.Event()

        async def session(**_):
            await owner_started.wait()
            raise RuntimeError("isolated database fixture failure")

        async def owner(**_):
            owner_started.set()
            try:
                await asyncio.Future()
            finally:
                owner_cancelled.set()

        service = SimpleNamespace(get_session_working_memory=session, get_owner_working_memory=owner)
        with self.assertRaises(ExceptionGroup):
            await WorkingMemoryContextProvider(service).build_section(self.request())
        self.assertTrue(owner_cancelled.is_set())
