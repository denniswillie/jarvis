"""Committed scope reads do not block canonical writers or become write grants."""
import asyncio
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class SessionAreaReadSnapshotSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_observes_committed_scope_without_waiting_for_owner(self):
        async with source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
            test_name=self.id(),
        ) as f:
            tasks = TaskRepository(f.pool, TaskLimits())
            selections = SessionKnowledgeAreas(f.sources, tasks, f.area)
            async with f.pool.acquire() as connection:
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
            async with f.sources.transaction(f.owner) as connection:
                await connection.execute("""update jarvis_knowledge_areas set status='archived',revision=revision+1
                    where owner_id=$1 and id=$2""", f.owner, f.area)
                # A held mutation lock and uncommitted area UPDATE must not
                # block this one-statement snapshot or leak uncommitted state.
                async with asyncio.timeout(2):
                    before = await selections.read(f.owner, session)
                self.assertEqual(before, {
                    "area_id": str(f.area), "area_status": "active",
                    "selection_revision": 0, "selection_event_id": None,
                    "origin": "configured_default",
                })
            after = await selections.read(f.owner, session)
            self.assertEqual(after["area_status"], "archived")
            self.assertEqual(await selections.read(f.owner, None), after)
            # No fallback when the exact session or configured area is foreign
            # or missing. Being able to read metadata does not reopen an area.
            for owner, identity in [(f.other, session), (f.owner, uuid4())]:
                with self.subTest(owner_is_current=owner == f.owner):
                    with self.assertRaises(SourceError) as error:
                        await selections.read(owner, identity)
                    self.assertEqual(error.exception.code, "knowledge_session_unavailable")
            missing = SessionKnowledgeAreas(f.sources, tasks, uuid4())
            with self.assertRaises(SourceError) as error:
                await missing.read(f.owner, session)
            self.assertEqual(error.exception.code, "knowledge_area_unavailable")
            with self.assertRaises(ValueError):
                await selections.read(f.owner, str(session))
            async with f.sources.transaction(f.owner) as connection:
                with self.assertRaises(SourceError) as error:
                    await f.sources._area(connection, f.owner, f.area)
                self.assertEqual(error.exception.code, "knowledge_area_unavailable")
