"""Tasks keep the request's knowledge area across delayed admission and steering."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.runtime import KnowledgeRuntime
from app.knowledge.selection import configure_selected_memory
from app.knowledge.session_areas import SessionKnowledgeAreas
from app.knowledge.sources import SourceRepository
from app.runtime.feature_config import MemoryAuthority, TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule, audit_task_result, task_payload
from evals.second_brain.source_fixture import source_fixture


class TaskAreaContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session, self.area, self.project = "task-area-fixture", uuid4(), uuid4(), uuid4()
        pool = object()
        self.scopes = SessionKnowledgeAreas(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), self.area)
        self.request = TaskCreate(request_id=uuid4(), title="Synthetic task", requested_outcome="Reason about the fixture.",
                                  session_id=self.session)
        self.event = {"id": uuid4()}
        self.record = SimpleNamespace(id=self.event["id"], revision=1, session_id=self.session, authorizes_request=True,
            source_namespace="voice:fixture", source_event_id=str(uuid4()), origin_sequence=17)
        self.connection = SimpleNamespace(is_in_transaction=lambda: True,
            fetchval=AsyncMock(return_value=self.project), fetch=AsyncMock(return_value=[]))

    async def resolve(self, request=None):
        with patch.object(ContextRepository, "_record", AsyncMock(return_value=self.record)):
            return await self.scopes.task_area_in_transaction(self.connection, self.owner,
                                                              request or self.request, self.event)

    async def test_canonical_binding_wins_without_reading_current_selection(self):
        self.connection.fetchval.side_effect = [None, self.project]
        with patch.object(self.scopes, "read_in_transaction", AsyncMock(side_effect=AssertionError("Current selection used"))):
            self.assertEqual(await self.resolve(), self.project)
        self.connection.fetch.assert_not_awaited()

    async def test_exact_request_selection_precedes_original_binding(self):
        self.assertEqual(await self.resolve(), self.project)
        self.connection.fetchval.assert_awaited_once()
        self.assertEqual(self.connection.fetchval.await_args.args[1:],
                         (self.owner, self.record.id, self.record.revision, self.session))
        self.connection.fetch.assert_not_awaited()

    async def test_explicit_assignment_is_returned_for_repository_owner_validation(self):
        self.assertEqual(await self.resolve(self.request.model_copy(update={"knowledge_area_id": self.area})), self.area)
        self.connection.fetchval.assert_not_awaited()

    async def test_api_scope_uses_request_origin_not_a_later_selection(self):
        self.record.source_namespace = "task_api"
        self.connection.fetchval.return_value = None
        with patch.object(self.scopes, "read_in_transaction", AsyncMock(return_value={"area_id": str(self.area)})) as read:
            self.assertEqual(await self.resolve(), self.area)
        read.assert_awaited_once_with(self.connection, self.owner, self.session, through_sequence=17)

    async def test_missing_or_ambiguous_old_scope_never_falls_back(self):
        self.connection.fetchval.return_value = None
        for rows in ([], [{"area_id": self.area}, {"area_id": self.project}]):
            self.connection.fetch.return_value = rows
            with self.assertRaises(SourceError) as caught:
                await self.resolve()
            self.assertEqual(caught.exception.code, "knowledge_task_scope_unavailable")
        self.connection.fetch.return_value = [{"area_id": self.project}]
        self.assertEqual(await self.resolve(), self.project)

    async def test_transaction_and_exact_request_session_are_required(self):
        self.connection.is_in_transaction = lambda: False
        with self.assertRaises(SourceError):
            await self.resolve()
        self.connection.is_in_transaction = lambda: True
        self.record.session_id = uuid4()
        with self.assertRaises(SourceError):
            await self.resolve()

    async def test_service_defers_scope_to_atomic_admission_and_replays_removed_provider(self):
        repository = SimpleNamespace(create=AsyncMock(return_value="accepted"), replay_create=AsyncMock(return_value="old"))
        resolver = AsyncMock(side_effect=AssertionError("Resolved outside transaction"))
        service = TaskService(repository, {"frontier": SimpleNamespace(model="fixture-model")},
                              knowledge_area_resolver=resolver)
        self.assertEqual(await service.create(self.owner, self.request), "accepted")
        self.assertIs(repository.create.call_args.kwargs["area_resolver"], resolver)
        resolver.assert_not_awaited()
        service.providers = {}
        self.assertEqual(await service.create(self.owner, self.request), "old")
        resolver.assert_not_awaited()

    def test_legacy_authority_removes_selected_scope_hook(self):
        service = SimpleNamespace(knowledge_area_resolver=object())
        configure_selected_memory(SimpleNamespace(memory_authority=MemoryAuthority.LEGACY,
            background_task_service=service, tool_modules=(), assistant_context_providers=()))
        self.assertIsNone(service.knowledge_area_resolver)

    def test_status_and_audit_report_persisted_area_without_copying_expert_answer(self):
        task = SimpleNamespace(id=uuid4(), revision=1, title="Synthetic", mode="background",
            status=SimpleNamespace(value="queued"), reason=None, pending_status=None,
            session_id=self.session, knowledge_area_id=self.project, updated_at=datetime.now(UTC))
        result = task_payload(SimpleNamespace(reference=lambda _: "task_0123456789ab"), task)
        self.assertEqual(result["knowledge_area_id"], str(self.project))
        audit = audit_task_result({"status": "success", "task": result})
        self.assertEqual(audit["task"]["knowledge_area_id"], str(self.project))
        self.assertNotIn("title", audit["task"])


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested source fixtures required")
class TaskAreaDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits())
        self.scopes = SessionKnowledgeAreas(self.f.sources, self.tasks, self.f.area)
        self.history = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area, self.scopes))
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            self.project = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Task bench','robotics') returning id", self.f.owner)
            self.foreign = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Foreign bench','robotics') returning id", self.f.other)
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-model")})
        self.runtime = SimpleNamespace(sources=self.f.sources, context_repository=self.history,
            configuration=SimpleNamespace(area_id=self.f.area), context_for=self.access,
            memory_read_backend_for=lambda owner: SimpleNamespace(), area_selections=self.scopes)
        self.runtime.area_tool_module_for = lambda owner: KnowledgeRuntime.area_tool_module_for(self.runtime, owner)
        self.services = SimpleNamespace(memory_authority=MemoryAuthority.HINDSIGHT, knowledge_runtime=self.runtime,
            settings=SimpleNamespace(JARVIS_OWNER_ID=self.f.owner), background_task_service=self.service,
            tool_modules=(BackgroundTaskToolModule(self.service),), assistant_context_providers=())
        configure_selected_memory(self.services)

    def access(self, owner):
        if owner != self.f.owner:
            raise SourceError("knowledge_owner_mismatch", "The fixture owner does not match.", 403)
        return self.history

    async def capture(self, text="Synthetic delegated request.", *, previous=None, history=None):
        return await (history or self.history).append(self.f.owner, ContextIngress(session_id=self.session,
            source_namespace="voice:task-area-fixture", source_event_id=previous.source_event_id if previous else str(uuid4()),
            receipt_id=str(uuid4()), expected_revision=previous.revision if previous else 0,
            kind="user_transcript", trust="user_input", payload={"text": text}))

    async def select(self, area):
        event = await self.capture("Select a synthetic knowledge area.")
        before = await self.scopes.read(self.f.owner, self.session)
        await self.scopes.select(self.f.owner, area,
            request=TaskVoiceSource(event.id, event.revision, self.session),
            expected_revision=before["selection_revision"])

    def request(self, **changes):
        return TaskCreate(**(dict(request_id=uuid4(), title="Synthetic task",
            requested_outcome="Reason about the synthetic request.", session_id=self.session) | changes))

    async def create(self, event, request=None, service=None):
        return await (service or self.service).create(self.f.owner, request or self.request(),
            source=TaskVoiceSource(event.id, event.revision, self.session), mode="background")

    async def count(self):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("select count(*) from jarvis_background_tasks where owner_id=$1", self.f.owner)

    async def test_delayed_voice_admission_and_unavailable_runtime_replay_keep_original_area(self):
        await self.select(self.project)
        event = await self.capture()
        request = self.request()
        await self.select(self.f.area)
        task = await self.create(event, request)
        self.assertEqual(task.knowledge_area_id, self.project)
        self.services.knowledge_runtime = None
        configure_selected_memory(self.services)
        replay = await self.create(event, request)
        self.assertEqual((replay.id, replay.knowledge_area_id), (task.id, self.project))
        with self.assertRaises(TaskError) as caught:
            await self.create(event)
        self.assertEqual(caught.exception.code, "task_knowledge_scope_unavailable")
        self.assertEqual(await self.count(), 1)

    async def test_corrected_transcript_retains_its_original_stream_area(self):
        await self.select(self.project)
        original = await self.capture()
        await self.select(self.f.area)
        revised = await self.capture("Corrected synthetic request.", previous=original)
        self.assertEqual((await self.create(revised)).knowledge_area_id, self.project)
        with self.assertRaises(TaskError):
            await self.create(original)
        self.assertEqual(await self.count(), 1)

    async def test_api_creation_and_replay_are_session_scoped_and_other_sessions_are_independent(self):
        await self.select(self.project)
        request = self.request()
        task = await self.service.create(self.f.owner, request)
        self.assertEqual(task.knowledge_area_id, self.project)
        await self.select(self.f.area)
        self.assertEqual((await self.service.create(self.f.owner, request)).knowledge_area_id, self.project)
        other = await self.service.create(self.f.owner, self.request(session_id=self.other_session))
        self.assertEqual(other.knowledge_area_id, self.f.area)
        self.assertEqual(await self.count(), 2)

    async def test_explicit_owned_assignment_is_allowed_but_foreign_and_archived_areas_are_not(self):
        event = await self.capture()
        self.assertEqual((await self.create(event, self.request(knowledge_area_id=self.project))).knowledge_area_id, self.project)
        with self.assertRaises(TaskError) as caught:
            await self.create(event, self.request(knowledge_area_id=self.foreign))
        self.assertEqual(caught.exception.code, "task_area_unavailable")
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.project)
        with self.assertRaises(TaskError) as caught:
            await self.create(event, self.request(knowledge_area_id=self.project))
        self.assertEqual(caught.exception.code, "task_area_unavailable")
        self.assertEqual(await self.count(), 1)

    async def test_steering_and_cancellation_do_not_move_the_task_to_the_new_topic(self):
        await self.select(self.project)
        task = await self.create(await self.capture())
        await self.select(self.f.area)
        for operation in ("update", "cancel"):
            event = await self.capture("Update or stop the existing synthetic task.")
            task = await self.service.mutate(self.f.owner, task.id, operation,
                TaskMutation(request_id=uuid4(), expected_revision=task.revision, message="Revised synthetic objective."),
                source=TaskVoiceSource(event.id, event.revision, self.session))
            self.assertEqual(task.knowledge_area_id, self.project)

    async def test_unpublished_voice_request_has_no_guessed_current_area(self):
        await self.select(self.project)
        event = await self.capture(history=ContextRepository(self.tasks))
        with self.assertRaises(TaskError) as caught:
            await self.create(event)
        self.assertEqual(caught.exception.code, "knowledge_task_scope_unavailable")
        self.assertEqual(await self.count(), 0)

    async def test_admission_uses_one_connection_without_nested_source_transaction(self):
        await self.select(self.project)
        event = await self.capture()
        await self.select(self.f.area)
        permit = asyncio.Semaphore(1)
        @asynccontextmanager
        async def acquire():
            async with permit, self.f.pool.acquire() as connection:
                yield connection
        pool = SimpleNamespace(acquire=acquire)
        tasks = TaskRepository(pool, TaskLimits())
        scopes = SessionKnowledgeAreas(SourceRepository(pool), tasks, self.f.area)
        service = TaskService(tasks, {"frontier": SimpleNamespace(model="fixture-model")},
                              knowledge_area_resolver=scopes.task_area_in_transaction)
        async with asyncio.timeout(10):
            task = await self.create(event, service=service)
        self.assertEqual(task.knowledge_area_id, self.project)

    async def test_request_scope_precedes_configured_artifact_fallback(self):
        await self.select(self.project)
        event = await self.capture()
        await self.select(self.f.area)
        task = await self.tasks.create(self.f.owner, self.request(), model="fixture-model",
            source=TaskVoiceSource(event.id, event.revision, self.session), mode="background",
            default_knowledge_area_id=self.f.area, area_resolver=self.service.knowledge_area_resolver)
        self.assertEqual(task.knowledge_area_id, self.project)
