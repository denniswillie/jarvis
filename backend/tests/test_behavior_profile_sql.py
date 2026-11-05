"""Opt-in PostgreSQL CAS proof through Core's existing disposable-schema lane."""
import asyncio
import os
from pathlib import Path
import unittest

from app.assistant.models import BehaviorProfileRevisionConflict, BehaviorProfileUpdate
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.assistant.tools import create_behavior_tools
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class BehaviorProfileSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_initialization_edits_and_preserved_profile_fields(self):
        fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        events = []
        repository = BehaviorProfileRepository(fixture.pool, on_committed=events.append)
        other_repository = BehaviorProfileRepository(fixture.pool, on_committed=events.append)
        self.assertIsNone(await repository.get(owner_id=fixture.owner))

        initialized = await asyncio.gather(*(
            repo.set(owner_id=fixture.owner, behavior_instructions=text, expected_revision=0)
            for repo, text in ((repository, "First creator"), (other_repository, "Second creator"))
        ), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, BehaviorProfileRevisionConflict) for result in initialized), 1)
        self.assertEqual(len(events), 1)
        current = await repository.get(owner_id=fixture.owner)
        self.assertEqual(current.revision, 1)

        _, voice_set = create_behavior_tools(repository=other_repository, owner_id=fixture.owner)
        app = BehaviorProfileService(repository)
        app_result, voice_result = await asyncio.gather(
            app.update(owner_id=fixture.owner, update=BehaviorProfileUpdate(
                behavior_instructions="App winner", expected_revision=1)),
            voice_set.handler({"behavior_instructions": "Voice winner", "expected_revision": 1}),
            return_exceptions=True,
        )
        app_won = isinstance(app_result, dict)
        self.assertEqual(voice_result["status"], "failed" if app_won else "success")
        if app_won:
            self.assertEqual(voice_result["error"]["code"], "behavior_profile_revision_conflict")
        else:
            self.assertIsInstance(app_result, BehaviorProfileRevisionConflict)
        persisted = await repository.get(owner_id=fixture.owner)
        self.assertEqual(persisted.revision, 2)
        self.assertEqual(persisted.created_at, current.created_at)
        self.assertEqual(persisted.behavior_instructions, "App winner" if app_won else "Voice winner")
        self.assertEqual([event.revision for event in events], [1, 2])
        self.assertIsNone(await repository.get(owner_id=fixture.other))

        # Existing-style rows default to revision one without changing text or
        # timestamps. This is not a live production migration exercise.
        async with fixture.pool.acquire() as connection:
            old = await connection.fetchrow(
                "insert into jarvis_behavior_profiles(owner_id,behavior_instructions) "
                "values($1,'Existing instructions') returning created_at,updated_at,revision",
                fixture.other)
        preserved = await repository.get(owner_id=fixture.other)
        self.assertEqual(preserved.revision, 1)
        self.assertEqual(preserved.behavior_instructions, "Existing instructions")
        self.assertEqual((preserved.created_at, preserved.updated_at), (old["created_at"], old["updated_at"]))
        cleared = await repository.set(owner_id=fixture.other, behavior_instructions="", expected_revision=1)
        self.assertEqual(cleared.revision, 2)
        self.assertEqual(cleared.created_at, preserved.created_at)


if __name__ == "__main__":
    unittest.main()
