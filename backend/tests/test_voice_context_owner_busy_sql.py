"""Actual same-owner lock contention and recovery in one manifested fixture."""
from datetime import UTC, datetime
import os
from pathlib import Path
import unittest

from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.models import SessionScope
from app.live_state.runtime import _SessionBinder
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
    os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database fixture required")
class OwnerBusySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_owner_locks_defer_without_authority_then_revalidate_sources(self):
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
                Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()) as f:
            tasks = TaskRepository(f.pool, TaskLimits())
            async with f.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
            binder = _SessionBinder(f.sources, SessionScope(owner_id=f.owner, session_id=session, area_id=f.area),
                clock=lambda: datetime.now(UTC), access_check=lambda owner: None,
                area_selections=SessionKnowledgeAreas(f.sources, tasks, f.area))
            record = await ContextRepository(tasks).append(f.owner, ContextIngress(
                session_id=session, source_namespace="fixture:owner-busy", source_event_id="request",
                receipt_id="first", kind="user_request", trust="user_input", payload={"text": "Synthetic request"}))
            binding = await binder.bind_record(record, "Synthetic guarded request")
            for kind in ("memory", "task"):
                with self.subTest(lock=kind):
                    owner = f.sources if kind == "memory" else tasks
                    async with owner.transaction(f.owner):
                        with self.assertRaises(CanonicalOwnerBusy):
                            async with binder.guard(binding, wait_for_owner=False):
                                self.fail("A busy owner cannot authorize any payload")
                        if kind == "task":
                            async with f.pool.acquire() as connection:
                                self.assertIs(await connection.fetchval(
                                    "select pg_try_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-memory-owner:' || $1,0))",
                                    f.owner), True, "Failed task-lock entry must release its earlier memory lock")
                    async with binder.guard(binding, wait_for_owner=False) as connection:
                        self.assertTrue(connection.is_in_transaction())
            async with f.sources.transaction(f.owner) as connection:
                await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2",
                                         f.owner, record.payload_id)
            with self.assertRaises(SourceError) as raised:
                async with binder.guard(binding, wait_for_owner=False):
                    self.fail("Lock availability is not source validity")
            self.assertNotIsInstance(raised.exception, CanonicalOwnerBusy)


if __name__ == "__main__":
    unittest.main()
