"""Session scope changes never retag earlier canonical streams or other sessions."""
import asyncio
from datetime import UTC, datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.current_context import CurrentSourceContext, SourceContextProvider
from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError, TaskVoiceSource
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class SessionAreaContractTests(unittest.IsolatedAsyncioTestCase):
    def test_publication_and_selection_cannot_use_different_scope_stores(self):
        pool, area = object(), uuid4()
        sources = SimpleNamespace(pool=pool)
        scopes = SessionKnowledgeAreas(sources, SimpleNamespace(pool=pool), area)
        CanonicalSourcePublication(sources, area, scopes)
        for source_store, default in ((SimpleNamespace(pool=object()), area), (sources, uuid4())):
            with self.assertRaises(ValueError):
                CanonicalSourcePublication(source_store, default, scopes)

    def test_pool_and_default_identity_are_explicit(self):
        pool = object()
        with self.assertRaises(ValueError):
            SessionKnowledgeAreas(SimpleNamespace(pool=pool), SimpleNamespace(pool=object()), uuid4())
        with self.assertRaises(ValueError):
            SessionKnowledgeAreas(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), "guessed")

    async def test_bad_selection_bounds_do_not_touch_storage(self):
        pool = object()
        service = SessionKnowledgeAreas(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), uuid4())
        request = TaskVoiceSource(uuid4(), 1, uuid4())
        for area, revision in (("name", 0), (uuid4(), True), (uuid4(), -1)):
            with self.subTest(area=area, revision=revision), self.assertRaises(ValueError):
                await service.select("fixture", area, request=request, expected_revision=revision)

    def test_unknown_selected_scope_is_not_reported_as_default_scope(self):
        provider = SourceContextProvider(SimpleNamespace(), uuid4(), selections=object())
        result = provider.fallback_section(SimpleNamespace(requested_at_utc=datetime.now(UTC)))
        self.assertIsNone(result.payload["area_id"])
        self.assertEqual(result.payload["status"], "unavailable")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class SessionAreaDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.scopes = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area, self.scopes))
        async with self.f.pool.acquire() as connection:
            self.first = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.second = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.other)
            self.robotics = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                values($1,'Bench','robotics') returning id""", self.f.owner)
            self.fitness = await connection.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                values($1,'Training','fitness') returning id""", self.f.owner)

    async def capture(self, text, *, session=None, previous=None, finality="final", history=None):
        event = ContextIngress(session_id=session or self.first, source_namespace="voice:session-scope-fixture",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", finality=finality,
            expected_revision=previous.revision if previous else 0, payload={"text": text})
        return await (history or self.history).append(self.f.owner, event)

    async def select(self, area, *, request=None, expected=None, session=None, access_check=None):
        request = request or await self.capture("Select this synthetic knowledge area.", session=session)
        if expected is None:
            expected = (await self.scopes.read(self.f.owner, request.session_id))["selection_revision"]
        return await self.scopes.select(self.f.owner, area,
            request=TaskVoiceSource(request.id, request.revision, request.session_id),
            expected_revision=expected, access_check=access_check)

    async def mapping(self, record):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("""select s.id,s.area_id,cs.source_revision from jarvis_knowledge_context_sources cs
                join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where cs.owner_id=$1 and cs.event_id=$2 and cs.event_revision=$3""",
                self.f.owner, record.id, record.revision)

    async def state(self, area, status):
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_knowledge_areas set status=$3,revision=revision+1
                where owner_id=$1 and id=$2""", self.f.owner, area, status)

    async def test_selection_persists_and_only_routes_new_sources_in_its_session(self):
        before = await self.capture("Existing default-area evidence.")
        await self.select(self.robotics)
        after = await self.capture("New robotics evidence.")
        other = await self.capture("Another conversation remains independent.", session=self.second)
        self.assertEqual((await self.mapping(before))["area_id"], self.f.area)
        self.assertEqual((await self.mapping(after))["area_id"], self.robotics)
        self.assertEqual((await self.mapping(other))["area_id"], self.f.area)
        restarted = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.assertEqual((await restarted.read(self.f.owner, self.first))["area_id"], str(self.robotics))
        self.assertEqual((await restarted.read(self.f.owner, self.second))["selection_revision"], 0)
        self.assertEqual(restarted.default_area_id, self.f.area)

    async def test_provisional_stream_and_late_corrections_keep_original_scope(self):
        partial = await self.capture("Started before the switch", finality="provisional")
        self.assertIsNone(await self.mapping(partial))
        await self.select(self.robotics)
        final = await self.capture("Finished after the switch.", previous=partial)
        initial_source = await self.mapping(final)
        self.assertEqual(initial_source["area_id"], self.f.area)
        corrected = await self.capture("Corrected earlier utterance.", previous=final, finality="revised")
        corrected_source = await self.mapping(corrected)
        self.assertEqual(corrected_source["id"], initial_source["id"])
        self.assertEqual(corrected_source["area_id"], self.f.area)
        self.assertEqual(corrected_source["source_revision"], 2)

    async def test_preexisting_publication_identity_outranks_changed_default(self):
        old_history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        old = await self.capture("Published before scope binding existed.", history=old_history)
        original = await self.mapping(old)
        changed_default = SessionKnowledgeAreas(self.f.sources, self.tasks, self.robotics)
        new_history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.robotics, changed_default))
        updated = await self.capture("Correction after a configuration change.", previous=old,
            finality="revised", history=new_history)
        self.assertEqual((await self.mapping(updated))["id"], original["id"])
        self.assertEqual((await self.mapping(updated))["area_id"], self.f.area)

    async def test_unbound_provisional_history_uses_original_sequence_not_finalization_time(self):
        old_history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        partial = await self.capture("Unbound partial before selection.", finality="provisional", history=old_history)
        await self.select(self.robotics)
        final = await self.capture("Final after selection.", previous=partial)
        self.assertEqual((await self.mapping(final))["area_id"], self.f.area)

    async def test_retry_does_not_reselect_old_area_after_a_later_selection(self):
        request = await self.capture("Use the robotics area.")
        results = await asyncio.gather(*(self.select(self.robotics, request=request, expected=0) for _ in range(2)))
        self.assertEqual(sum(result["replayed"] for result in results), 1)
        await self.select(self.fitness)
        replay = await self.select(self.robotics, request=request, expected=0)
        self.assertEqual(replay["applied_area_id"], str(self.robotics))
        self.assertEqual(replay["applied_revision"], 1)
        self.assertEqual(replay["selection"]["area_id"], str(self.fitness))
        self.assertEqual(replay["selection"]["selection_revision"], 2)
        with self.assertRaisesRegex(SourceError, "already selected"):
            await self.select(self.fitness, request=request, expected=0)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_session_knowledge_selections where owner_id=$1", self.f.owner), 2)

    async def test_concurrent_different_requests_do_not_overwrite_the_same_revision(self):
        requests = [await self.capture("First selection."), await self.capture("Second selection.")]
        results = await asyncio.gather(self.select(self.robotics, request=requests[0], expected=0),
            self.select(self.fitness, request=requests[1], expected=0), return_exceptions=True)
        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        self.assertEqual(sum(isinstance(result, SourceError) for result in results), 1)
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["selection_revision"], 1)

    async def test_older_request_cannot_replace_a_newer_selection_even_after_asr_revision(self):
        older = await self.capture("Earlier request for the default area.")
        newer = await self.capture("Later request for robotics.")
        await self.select(self.robotics, request=newer, expected=0)
        with self.assertRaises(SourceError) as caught:
            await self.select(self.f.area, request=older, expected=1)
        self.assertEqual(caught.exception.code, "knowledge_selection_request_stale")
        older_revised = await self.capture("Late ASR correction of the earlier request.", previous=older, finality="revised")
        with self.assertRaises(SourceError) as caught:
            await self.select(self.f.area, request=older_revised, expected=1)
        self.assertEqual(caught.exception.code, "knowledge_selection_request_stale")
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["area_id"], str(self.robotics))
        latest_corrected = await self.capture("Correction to the latest request means fitness.", previous=newer, finality="revised")
        selected = await self.select(self.fitness, request=latest_corrected, expected=1)
        self.assertEqual(selected["selection"]["area_id"], str(self.fitness))

    async def test_archived_or_deleted_selection_preserves_capture_without_fallback(self):
        await self.select(self.robotics)
        for status in ("archived", "deleted"):
            await self.state(self.robotics, status)
            captured = await self.capture("Capture continues while selected scope is " + status)
            self.assertTrue(captured.authorizes_request)
            self.assertIsNone(await self.mapping(captured))
            scope = await self.scopes.read(self.f.owner, self.first)
            self.assertEqual(scope["area_id"], str(self.robotics))
            self.assertEqual(scope["area_status"], status)
        await self.select(self.f.area, request=captured)
        self.assertEqual((await self.mapping(await self.capture("New default-scope source.")))["area_id"], self.f.area)

    async def test_archived_target_is_not_selected_and_archived_default_can_be_left(self):
        await self.state(self.robotics, "archived")
        with self.assertRaises(SourceError):
            await self.select(self.robotics)
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["selection_revision"], 0)
        await self.state(self.f.area, "archived")
        await self.select(self.fitness)
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["area_id"], str(self.fitness))

    async def test_invalid_current_request_and_foreign_session_cannot_change_selection(self):
        partial = await self.capture("Incomplete request.", finality="provisional")
        with self.assertRaises((SourceError, TaskError)):
            await self.select(self.robotics, request=partial, expected=0)
        final = await self.capture("Final request.", previous=partial)
        await self.capture("Revised request.", previous=final, finality="revised")
        with self.assertRaises((SourceError, TaskError)):
            await self.select(self.robotics, request=final, expected=0)
        with self.assertRaises(SourceError):
            await self.scopes.read(self.f.owner, self.other_session)
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["selection_revision"], 0)

    async def test_receipt_failure_and_access_revocation_roll_back_selection(self):
        request = await self.capture("Select robotics atomically.")
        with patch.object(ContextRepository, "append_in_transaction", side_effect=RuntimeError("fixture receipt failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture receipt"):
                await self.select(self.robotics, request=request, expected=0)
        calls = 0
        def check(_owner):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise SourceError("fixture_revoked", "Fixture access revoked.")
        with self.assertRaises(SourceError):
            await self.select(self.robotics, request=request, expected=0, access_check=check)
        self.assertEqual((await self.scopes.read(self.f.owner, self.first))["selection_revision"], 0)
        result = await self.select(self.robotics, request=request, expected=0)
        self.assertFalse(result["replayed"])

    async def test_binding_and_selection_history_are_immutable(self):
        await self.select(self.robotics)
        for table in ("jarvis_session_knowledge_selections", "jarvis_knowledge_context_scopes"):
            for operation in ("delete from " + table, "update " + table + " set area_id=area_id"):
                async with self.f.pool.acquire() as connection:
                    with self.assertRaises(asyncpg.CheckViolationError):
                        await connection.execute(operation + " where owner_id=$1", self.f.owner)

    async def test_conflicting_preexisting_area_mappings_fail_without_guessing(self):
        old_history = ContextRepository(self.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        record = await self.capture("Fixture with inconsistent legacy scope.", history=old_history)
        other_publisher = CanonicalSourcePublication(self.f.sources, self.robotics)
        async with other_publisher.transaction(self.tasks, self.f.owner) as connection:
            await other_publisher.publish(connection, self.f.owner, record)
        with self.assertRaisesRegex(SourceError, "disagree"):
            await self.capture("Do not guess the area.", previous=record, finality="revised")
        self.assertTrue((await self.history.get(self.f.owner, record.id)).is_current)

    async def test_context_uses_selection_and_rejects_a_mid_build_scope_change(self):
        await self.select(self.robotics)
        await self.capture("Robot evidence for the selected context.")
        service = CurrentSourceContext(self.f.sources, timeout_seconds=10)
        provider = SourceContextProvider(service, self.f.area, selections=self.scopes)
        request = SimpleNamespace(owner_id=self.f.owner, session=SimpleNamespace(id=self.first),
                                  requested_at_utc=datetime.now(UTC))
        section = await provider.build_section(request)
        self.assertEqual(section.payload["area_id"], str(self.robotics))
        self.assertEqual(section.payload["selection"]["selection_revision"], 1)
        switching = await self.capture("Switch scope during the synthetic context build.")
        original = service.describe
        async def change_after_read(*args, **kwargs):
            result = await original(*args, **kwargs)
            await self.select(self.fitness, request=switching)
            return result
        with patch.object(service, "describe", side_effect=change_after_read):
            with self.assertRaisesRegex(SourceError, "scope changed"):
                await provider.build_section(request)
