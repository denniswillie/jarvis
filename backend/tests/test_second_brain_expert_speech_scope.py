"""Expert speech is scoped by owned delivery evidence, never today's topic."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg

from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource, json_object
from app.tasks.runner import TaskRunner


class ExpertScopeContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.project, self.session = "expert-scope-fixture", uuid4(), uuid4(), uuid4()
        self.intent, self.attempt, self.payload = uuid4(), uuid4(), uuid4()
        self.sources = SimpleNamespace(pool=object(), publish_in_transaction=AsyncMock())
        self.selections = SimpleNamespace(sources=self.sources, default_area_id=self.area,
            bind_context=AsyncMock(return_value=self.project))
        self.publisher = CanonicalSourcePublication(self.sources, self.area, self.selections)
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value=self.project))

    def record(self, **changes):
        return SimpleNamespace(**(dict(owner_id=self.owner, session_id=self.session, payload_id=self.payload,
            is_current=True, availability="available", finality="provisional", kind="assistant_output",
            trust="assistant_output", content={"authored_by": "frontier_expert",
                "intent_id": str(self.intent), "attempt_id": str(self.attempt), "delivery_generation": 2}) | changes))

    async def test_lookup_binds_owned_attempt_receiving_session_and_actual_parent(self):
        self.assertEqual(await self.publisher._expert_area(self.connection, self.owner, self.record()), self.project)
        self.assertEqual(self.connection.fetchval.call_args.args[1:],
                         (self.owner, self.attempt, self.intent, 2, self.session, self.payload))

    async def test_provisional_speech_already_binds_original_area(self):
        self.assertIsNone(await self.publisher.publish(self.connection, self.owner, self.record()))
        self.selections.bind_context.assert_awaited_once_with(self.connection, self.owner,
            unittest.mock.ANY, requested_area_id=self.project)
        self.sources.publish_in_transaction.assert_not_awaited()

    async def test_explicit_override_cannot_replace_verified_task_area(self):
        with self.assertRaises(SourceError) as caught:
            await self.publisher.publish(self.connection, self.owner, self.record(), requested_area_id=self.area)
        self.assertEqual(caught.exception.code, "knowledge_context_scope_conflict")
        self.selections.bind_context.assert_not_awaited()

    async def test_unverified_or_legacy_area_does_not_bind_a_default(self):
        self.connection.fetchval.return_value = None
        self.assertIsNone(await self.publisher.publish(self.connection, self.owner, self.record()))
        self.selections.bind_context.assert_not_awaited()
        self.sources.publish_in_transaction.assert_not_awaited()

    async def test_malformed_delivery_provenance_is_not_a_scope_capability(self):
        original = self.record().content
        for changes in ({"intent_id": "bad"}, {"attempt_id": None}, {"delivery_generation": True},
                        {"delivery_generation": 0}, {"delivery_generation": "2"}):
            self.assertIsNone(await self.publisher._expert_area(
                self.connection, self.owner, self.record(content=original | changes)))
        self.connection.fetchval.assert_not_awaited()

    async def test_user_prose_and_nonexpert_notifications_keep_normal_scope(self):
        for changes in ({"kind": "user_transcript", "trust": "user_input"},
                        {"content": {"authored_by": "notification_generator"}}):
            self.assertIsNone(await self.publisher.publish(self.connection, self.owner, self.record(**changes)))
        self.connection.fetchval.assert_not_awaited()
        self.assertTrue(all(call.kwargs["requested_area_id"] is None
                            for call in self.selections.bind_context.await_args_list))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class ExpertScopeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from tests.test_attention_voice import AttentionVoiceDatabaseTests
        await AttentionVoiceDatabaseTests.asyncSetUp(self)
        self.scopes = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area, self.scopes))
        self.capture = SessionContextCapture(self.history, owner_id=self.f.owner, session_id=self.session,
            provider="openai", audio_session_id="expert-scope", write_timeout=10)
        self.voice._context_capture = self.capture
        async with self.f.pool.acquire() as connection:
            self.project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Expert bench','robotics') returning id", self.f.owner)
            self.foreign_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Foreign bench','robotics') returning id", self.f.other)
            self.receiving_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)

    async def select(self, area, *, session=None):
        session = session or self.session
        event = await self.history.append(self.f.owner, ContextIngress(session_id=session,
            source_namespace="voice:expert-area-selection", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": "Select a synthetic knowledge area."}))
        state = await self.scopes.read(self.f.owner, session)
        await self.scopes.select(self.f.owner, area, request=TaskVoiceSource(event.id, event.revision, session),
                                 expected_revision=state["selection_revision"])

    async def complete(self, *, area, owner=None, session=None):
        owner, session = owner or self.f.owner, session or self.session
        task = await self.tasks.create(owner, TaskCreate(request_id=uuid4(), title="Synthetic expert",
            requested_outcome="Only reason about fixture evidence.", session_id=session, knowledge_area_id=area),
            model="fixture-model")
        claim = await self.tasks.claim_next("expert-scope-worker", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        context = TaskRunner(self.tasks, {}).context(claim)
        outcome = await HermesTaskResult(status="succeeded", answer="Synthetic expert answer.",
            speech=[{"text": "Synthetic expert answer."}, {"text": "Check the fixture evidence."}],
            evidence=[], completed_operation_refs=[], uncertainties=[]).verified(context, SimpleNamespace())
        self.assertTrue(await self.tasks.finish(claim, outcome))
        return await self.tasks.get(owner, task.id)

    async def claim(self, *, owner=None, session=None):
        result = await self.inbox.claim_next(owner or self.f.owner, session or self.session, "expert-scope-delivery")
        self.assertIsNotNone(result)
        return result

    def speech_event(self, claim, *, previous=None, parents=None, session=None, **changes):
        payload = {"authored_by": "frontier_expert", "audio_provider": "openai_tts",
            "text": "Synthetic expert answer.", "intent_id": str(claim.intent_id),
            "attempt_id": str(claim.attempt_id), "delivery_generation": claim.generation,
            "heard_transcript": None, "playback_status": "unknown"} | changes
        return ContextIngress(session_id=session or self.session, source_namespace="voice:expert-scope-fixture",
            source_event_id=previous.source_event_id if previous else str(uuid4()), receipt_id=str(uuid4()),
            expected_revision=previous.revision if previous else 0, kind="assistant_output",
            trust="assistant_output", payload=payload,
            source_payload_ids=(claim.source_payload_id,) if parents is None else parents)

    async def rows(self, namespace="voice:expert-scope-fixture"):
        async with self.f.pool.acquire() as connection:
            return await connection.fetch("""select e.id,e.source_event_id,e.revision,e.source_time,
                p.content,p.availability,scope.area_id as bound_area,s.id as source_id,s.area_id,cs.source_revision
                from jarvis_context_events e join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                left join jarvis_knowledge_context_scopes scope on scope.owner_id=e.owner_id
                    and scope.source_namespace=e.source_namespace and scope.source_event_id=e.source_event_id
                left join jarvis_knowledge_context_sources cs on cs.owner_id=e.owner_id and cs.event_id=e.id
                left join jarvis_knowledge_sources s on s.owner_id=cs.owner_id and s.id=cs.source_id
                where e.owner_id=$1 and e.source_namespace=$2 and e.kind='assistant_output'
                order by e.sequence""", self.f.owner, namespace)

    async def test_actual_voice_and_playback_receipts_keep_the_task_area_after_switching(self):
        from tests.test_attention_voice import AttentionVoiceDatabaseTests
        await self.select(self.project)
        await self.complete(area=self.project)
        await self.select(self.f.area)
        delivery, speech = await AttentionVoiceDatabaseTests.deliver(self)
        await asyncio.wait_for(self.generated.wait(), 10)
        for index in range(2):
            await self.voice._handle_client_event({"type": "playback.completed", "item_id": delivery.item_id(index),
                "response_id": delivery.response_id, "played_samples": 480, "content_index": 0})
        await asyncio.wait_for(delivery.task, 10)
        self.assertTrue(await self.capture.flush(), self.capture.issue)
        rows = await self.rows(self.capture.namespace)
        self.assertTrue(rows)
        self.assertEqual({r["bound_area"] for r in rows}, {self.project})
        self.assertEqual({r["area_id"] for r in rows if r["source_id"] is not None}, {self.project})
        self.assertEqual(len({r["source_id"] for r in rows if r["source_id"] is not None}), 2)
        self.assertTrue(all(json_object(r["content"])["heard_transcript"] is None for r in rows))
        self.assertEqual(len(speech.requests), 2)

    async def test_other_receiving_session_and_later_progress_keep_original_scope(self):
        await self.complete(area=self.project)
        claim = await self.claim(session=self.receiving_session)
        event = await self.history.append(self.f.owner, self.speech_event(claim, session=self.receiving_session))
        await self.select(self.f.area, session=self.receiving_session)
        revised = await self.history.append(self.f.owner, self.speech_event(claim, previous=event,
            session=self.receiving_session, playback_status="completed", played_samples=480))
        rows = await self.rows()
        self.assertEqual({r["bound_area"] for r in rows}, {self.project})
        self.assertEqual({r["area_id"] for r in rows}, {self.project})
        self.assertEqual(revised.source_time, event.source_time)

    async def test_provenance_requires_owned_lineage_and_verified_publication_scope(self):
        await self.complete(area=self.project)
        claim = await self.claim()
        foreign = await self.complete(area=self.foreign_area, owner=self.f.other, session=self.other_session)
        foreign_claim = await self.claim(owner=self.f.other, session=self.other_session)
        rejected = [self.speech_event(claim, parents=()), self.speech_event(claim, attempt_id="bad"),
            self.speech_event(claim, intent_id=str(foreign_claim.intent_id), attempt_id=str(foreign_claim.attempt_id))]
        for event in rejected:
            with self.assertRaises(asyncpg.CheckViolationError):
                await self.history.append(self.f.owner, event)
        events = [self.speech_event(claim, delivery_generation=True),
            self.speech_event(claim, session=self.receiving_session),
            self.speech_event(claim, intent_id=str(foreign_claim.intent_id))]
        for event in events:
            record = await self.history.append(self.f.owner, event)
            self.assertEqual(record.availability, "available")
        with self.assertRaises(TaskError):
            await self.history.append(self.f.owner, self.speech_event(claim, parents=(foreign.result_payload_id,)))
        with self.assertRaises(SourceError) as caught:
            async with self.history.publication.transaction(self.tasks, self.f.owner) as connection:
                await self.history.append_in_transaction(connection, self.f.owner, self.speech_event(claim),
                                                         publication_area_id=self.f.area)
        self.assertEqual(caught.exception.code, "knowledge_context_scope_conflict")
        rows = await self.rows()
        self.assertEqual(len(rows), len(events))
        self.assertTrue(all(r["bound_area"] is None and r["source_id"] is None for r in rows))

    async def test_legacy_task_without_area_is_not_guessed_from_current_selection(self):
        await self.complete(area=None)
        claim = await self.claim()
        await self.select(self.project)
        event = await self.history.append(self.f.owner, self.speech_event(claim))
        await self.select(self.f.area)
        await self.history.append(self.f.owner, self.speech_event(claim, previous=event, played_samples=480))
        rows = await self.rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["availability"] == "available" and r["bound_area"] is None
                            and r["source_id"] is None for r in rows))

    async def test_deleted_result_prevents_late_copy_and_invalidates_its_source(self):
        task = await self.complete(area=self.project)
        claim = await self.claim()
        event = await self.history.append(self.f.owner, self.speech_event(claim))
        first = (await self.rows())[0]
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2",
                                     self.f.owner, task.result_payload_id)
        with self.assertRaises(TaskError):
            await self.history.append(self.f.owner, self.speech_event(claim, previous=event, played_samples=480))
        self.assertEqual(len(await self.rows()), 1)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.project, first["source_id"])

    async def test_archived_area_keeps_canonical_capture_and_immutable_scope(self):
        await self.complete(area=self.project)
        claim = await self.claim()
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.project)
        event = await self.history.append(self.f.owner, self.speech_event(claim))
        first = (await self.rows())[0]
        self.assertEqual(first["bound_area"], self.project)
        self.assertIsNone(first["source_id"])
        self.assertEqual(first["availability"], "available")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='active',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.project)
        await self.history.append(self.f.owner, self.speech_event(claim, previous=event, played_samples=480))
        latest = (await self.rows())[-1]
        self.assertEqual((latest["bound_area"], latest["area_id"]), (self.project, self.project))

    async def test_forgotten_speech_source_is_not_resurrected_by_late_telemetry(self):
        from app.knowledge.lifecycle import SourceLifecycleService
        await self.complete(area=self.project)
        claim = await self.claim()
        event = await self.history.append(self.f.owner, self.speech_event(claim))
        first = (await self.rows())[0]
        request = await self.history.append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:forget-expert-fixture", source_event_id=str(uuid4()), receipt_id=str(uuid4()),
            kind="user_transcript", trust="user_input", payload={"text": "Forget this synthetic expert speech."}))
        await SourceLifecycleService(self.f.sources, self.tasks).forget(self.f.owner, self.project, first["source_id"],
            expected_revision=first["source_revision"], request=TaskVoiceSource(request.id, request.revision, self.session))
        await self.history.append(self.f.owner, self.speech_event(claim, previous=event, played_samples=480))
        latest = (await self.rows())[-1]
        self.assertEqual(latest["bound_area"], self.project)
        self.assertIsNone(latest["source_id"])
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.project, first["source_id"])
        async with self.f.pool.acquire() as connection:
            status = await connection.fetchval("select status from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                                              self.f.owner, first["source_id"])
        self.assertEqual(status, "forgotten")

    async def test_deleted_speech_source_rejects_late_telemetry(self):
        await self.complete(area=self.project)
        claim = await self.claim()
        event = await self.history.append(self.f.owner, self.speech_event(claim))
        first = (await self.rows())[0]
        await self.f.sources.delete_source(self.f.owner, self.project, first["source_id"],
                                          expected_revision=first["source_revision"])
        with self.assertRaises(TaskError) as caught:
            await self.history.append(self.f.owner, self.speech_event(claim, previous=event, played_samples=480))
        self.assertEqual(caught.exception.code, "context_source_removed")
        self.assertEqual(len(await self.rows()), 1)
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner, self.project, first["source_id"])
        async with self.f.pool.acquire() as connection:
            status = await connection.fetchval("select status from jarvis_knowledge_sources where owner_id=$1 and id=$2",
                                              self.f.owner, first["source_id"])
        self.assertEqual(status, "deleted")
