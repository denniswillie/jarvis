"""Explicit retention authority, replay, dependency holds and atomic expiry."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
from datetime import UTC, datetime, timedelta, timezone
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from pydantic import ValidationError

from app.knowledge.areas import KnowledgeAreaService
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.retention import SourceRetentionService
from app.knowledge.retention_tools import RetentionChange, SourceRetentionToolModule
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from evals.second_brain.source_fixture import source_fixture


class RetentionContractTests(unittest.IsolatedAsyncioTestCase):
    def arguments(self):
        return dict(area_id=str(uuid4()), source_id=str(uuid4()),
                    expected_source_revision=1, expected_source_generation=1,
                    expected_policy_revision=0, expires_at="2026-10-01T12:00:00+00:00")

    def test_dates_revisions_and_extra_fields_are_explicit(self):
        self.assertIsNotNone(RetentionChange.model_validate(self.arguments()).expires_at.tzinfo)
        for changes in ({"expires_at": "2026-10-01T12:00:00"}, {"expected_policy_revision": True},
                        {"expected_source_revision": "1"}, {"future_sources": True},
                        {"clock": "2027-01-01T00:00:00Z"}, {"as_of": "2027-01-01T00:00:00Z"}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                RetentionChange.model_validate(self.arguments() | changes)
        self.assertIsNone(RetentionChange.model_validate(self.arguments() | {"expires_at": None}).expires_at)

    async def test_tool_requires_owned_finalized_foreground_request_and_session(self):
        pool = object()
        module = SourceRetentionToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), "owner")
        context = SimpleNamespace(owner_id="owner", session_id=uuid4())
        tools = {tool.name: tool for tool in module.build_tools(context)}
        result = await tools["set_memory_retention"].handler(self.arguments())
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        with bind_tool_invocation(provider_call_id="fixture", request_event_id=uuid4(), request_revision=1,
                                  task=TaskToolInvocation(task_id=uuid4(), task_revision=1, attempt_id=uuid4())):
            result = await tools["set_memory_retention"].handler(self.arguments())
        self.assertEqual(result["error"]["code"], "knowledge_request_not_finalized")
        self.assertTrue(tools["set_memory_retention"].mutates_state)
        self.assertFalse(tools["inspect_memory_retention"].mutates_state)
        self.assertEqual([t.name for t in module.build_tools(
            SimpleNamespace(owner_id="owner", session_id=None))], ["inspect_memory_retention"])
        with self.assertRaises(SourceError):
            module.build_tools(SimpleNamespace(owner_id="foreign", session_id=context.session_id))

    async def test_rejected_private_arguments_are_redacted_by_both_registered_tools(self):
        from app.tasks.models import digest
        pool = object()
        module = SourceRetentionToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), "owner")
        tools = module.build_tools(SimpleNamespace(owner_id="owner", session_id=uuid4()))
        for tool in tools:
            with self.subTest(tool=tool.name):
                arguments = self.arguments() | {
                    "unexpected_source_text": "synthetic-private-original-marker",
                    "expires_at": "synthetic-private-date-marker"}
                result = await tool.handler(arguments)
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
                audit = tool.audit_arguments(arguments)
                self.assertEqual(audit["input_digest"], digest(arguments))
                self.assertEqual(audit["source_id"], arguments["source_id"])
                self.assertNotIn("synthetic-private", json.dumps(audit))
                self.assertNotIn("expires_at", audit)
                self.assertNotIn("synthetic-private", json.dumps(tool.audit_result(result)))
                self.assertNotEqual(audit["input_digest"], tool.audit_arguments(
                    arguments | {"unexpected_source_text": "a changed rejected value"})["input_digest"])

    async def test_actual_executor_persists_only_redacted_arguments_and_replays_error_code(self):
        from app.tools.executor import JarvisToolExecutor
        from app.tools.registry import JarvisToolRegistry
        pool, session = object(), uuid4()
        module = SourceRetentionToolModule(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool), "owner")
        registry = JarvisToolRegistry()
        registry.register_many(module.build_tools(SimpleNamespace(owner_id="owner", session_id=session)))
        repository = SimpleNamespace(
            start_execution_with_status=AsyncMock(return_value=(SimpleNamespace(status="started"), True)),
            finalize_execution=AsyncMock(return_value=SimpleNamespace(status="failed")))
        executor = JarvisToolExecutor(registry, execution_repository=repository,
                                      owner_id="owner", session_id=session)
        arguments = self.arguments() | {"unexpected_source_text": "synthetic-private-original-marker"}
        result = await executor.execute("set_memory_retention", arguments, provider_call_id="retention-call")
        self.assertEqual(result["error"]["code"], "invalid_tool_input")
        recorded = repository.start_execution_with_status.call_args.kwargs["arguments"]
        self.assertNotIn("synthetic-private-original-marker", json.dumps(recorded))
        self.assertIn("input_digest", recorded)
        error = repository.finalize_execution.call_args.kwargs["error"]
        self.assertEqual(error["code"], "invalid_tool_input")
        self.assertNotIn("synthetic-private-original-marker", json.dumps(error))
        repository.start_execution_with_status.return_value = (SimpleNamespace(status="failed", error=error), False)
        replay = await executor.execute("set_memory_retention", arguments, provider_call_id="retention-call")
        self.assertEqual(replay["error"]["code"], result["error"]["code"])
        repository.finalize_execution.assert_awaited_once()

    async def test_legacy_selection_removes_retention_tools(self):
        from app.knowledge.selection import configure_selected_memory
        from app.runtime.feature_config import MemoryAuthority
        keep = SimpleNamespace(name="fixture_ordinary")
        services = SimpleNamespace(memory_authority=MemoryAuthority.LEGACY,
            tool_modules=(keep, SimpleNamespace(name="source_retention")), assistant_context_providers=())
        configure_selected_memory(services)
        self.assertEqual(services.tool_modules, (keep,))

    async def test_invalid_service_bounds_and_naive_dates_are_rejected_before_io(self):
        pool = object()
        service = SourceRetentionService(SimpleNamespace(pool=pool), SimpleNamespace(pool=pool))
        for limit in (0, 129, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                await service.process_due("owner", limit=limit)
        with self.assertRaises(ValueError):
            service._deadline(datetime(2026, 10, 1))
        with self.assertRaises(ValueError):
            SourceRetentionService(SimpleNamespace(pool=object()), SimpleNamespace(pool=object()))

    async def test_domain_clock_is_frozen_for_selection_and_each_guarded_expiry(self):
        pool = object()
        candidate = {"source_id": uuid4(), "policy_revision": 1}
        db = SimpleNamespace(fetch=AsyncMock(return_value=[candidate, candidate]))

        @asynccontextmanager
        async def transaction(owner):
            self.assertEqual(owner, "owner")
            yield db

        instant = datetime(2027, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
        clock = Mock(return_value=instant)
        service = SourceRetentionService(SimpleNamespace(pool=pool, transaction=transaction),
                                         SimpleNamespace(pool=pool), clock=clock)
        service._expire = AsyncMock(return_value=None)
        self.assertEqual(await service.process_due("owner"), [])
        clock.assert_called_once_with()
        expected = instant.astimezone(UTC)
        self.assertEqual(db.fetch.await_args.args[-1], expected)
        self.assertEqual(service._expire.await_count, 2)
        for call in service._expire.await_args_list:
            self.assertEqual(call.kwargs, {"as_of": expected})

    async def test_default_clock_leaves_eligibility_to_database_time(self):
        pool = object()
        db = SimpleNamespace(fetch=AsyncMock(return_value=[]))

        @asynccontextmanager
        async def transaction(owner):
            yield db

        service = SourceRetentionService(SimpleNamespace(pool=pool, transaction=transaction),
                                         SimpleNamespace(pool=pool))
        self.assertEqual(await service.process_due("owner"), [])
        self.assertIsNone(db.fetch.await_args.args[-1])
        self.assertIn("coalesce($3::timestamptz,clock_timestamp())", db.fetch.await_args.args[0])

    async def test_invalid_domain_clocks_fail_before_database_io(self):
        pool = object()
        sources, requests = SimpleNamespace(pool=pool), SimpleNamespace(pool=pool)
        with self.assertRaises(ValueError):
            SourceRetentionService(sources, requests, clock=True)
        for value in (None, True, "2027-01-01", datetime(2027, 1, 1)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service = SourceRetentionService(sources, requests, clock=lambda: value)
                await service.process_due("owner")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class SourceRetentionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.history = ContextRepository(self.tasks)
        self.service = SourceRetentionService(self.f.sources, self.tasks)
        async with self.f.pool.acquire() as db:
            self.session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",
                                             self.f.owner)
        self.request = await self.new_request()
        self.original = await self.f.publish("Synthetic disposable source, explicitly selected for expiry.")
        self.other = await self.f.publish("Keep this unrelated durable decision.", source_key="keep-source")
        self.past = datetime.now(UTC) - timedelta(days=1)

    async def new_request(self):
        event = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="voice:retention-fixture", source_event_id=uuid4().hex,
            receipt_id=uuid4().hex, kind="user_transcript", trust="user_input",
            payload={"text": "Set the explicitly selected source retention policy."}))
        return TaskVoiceSource(event.id, event.revision, self.session)

    async def schedule(self, source=None, **changes):
        source = source or self.original
        parameters = dict(expected_source_revision=source.revision,
            expected_source_generation=source.generation, expected_policy_revision=0,
            expires_at=self.past, request=self.request) | changes
        return await self.service.set_policy(self.f.owner, self.f.area, source.source_id, **parameters)

    async def inspect(self, source=None):
        return await self.service.inspect(self.f.owner, self.f.area, (source or self.original).source_id)

    async def test_default_future_and_cancelled_policies_do_not_delete_memories(self):
        self.assertEqual((await self.inspect())["retention"]["status"], "not_scheduled")
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        first = await self.schedule(expires_at=datetime.now(UTC) + timedelta(days=10))
        self.assertFalse(first["replayed"])
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        second_request = await self.new_request()
        cleared = await self.schedule(expires_at=None, expected_policy_revision=1, request=second_request)
        self.assertEqual(cleared["retention"]["status"], "disabled")
        replay = await self.schedule(expires_at=None, expected_policy_revision=1, request=second_request)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["applied_policy_revision"], 2)
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,self.original.source_id)).revision, 1)

    async def test_expiry_reuses_deletion_intents_and_keeps_non_content_tombstone(self):
        await self.schedule()
        results = await self.service.process_due(self.f.owner)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "expired")
        self.assertFalse(results[0]["full_erasure_verified"])
        with self.assertRaises(SourceError):
            await self.f.sources.read(self.f.owner,self.f.area,self.original.source_id)
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,self.other.source_id)).revision, 1)
        state = await self.inspect()
        self.assertEqual(state["source_state"], "deleted")
        self.assertIsNone(state["tombstone_retention"]["expires_at"])
        self.assertFalse(state["tombstone_retention"]["original_content_required"])
        async with self.f.pool.acquire() as db:
            self.assertIsNone(await db.fetchval("select content from jarvis_context_payloads where id=$1",
                                                self.original.payload_id))
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_memory_outbox
                where owner_id=$1 and document_id=$2 and operation='delete'""",
                self.f.owner,self.original.document_id), 1)
        self.assertEqual(await self.service.process_due(self.f.owner), [])
        replay = await self.schedule()
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["source_state"], "deleted")

    async def test_old_replay_never_reinstalls_a_newer_policy(self):
        await self.schedule()
        await self.schedule(expires_at=None, expected_policy_revision=1, request=await self.new_request())
        replay = await self.schedule()
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["applied_policy_revision"], 1)
        self.assertEqual(replay["retention"]["policy_revision"], 2)
        self.assertEqual(replay["retention"]["status"], "disabled")
        with self.assertRaises(SourceError) as caught:
            await self.schedule(expires_at=None)
        self.assertEqual(caught.exception.code, "retention_request_conflict")

    async def test_concurrent_expiry_is_one_effect_and_stale_candidate_is_inert(self):
        await self.schedule()
        outcomes = await asyncio.gather(self.service.process_due(self.f.owner),
                                       self.service.process_due(self.f.owner))
        self.assertEqual(sum(len(rows) for rows in outcomes), 1)
        await self.schedule(self.other)
        candidate = {"source_id": self.other.source_id, "policy_revision": 1}
        await self.schedule(self.other, expires_at=None, expected_policy_revision=1, request=await self.new_request())
        self.assertIsNone(await self.service._expire(self.f.owner, candidate))
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,self.other.source_id)).revision, 1)

    async def test_referenced_evidence_is_held_without_starving_other_expiry(self):
        async with self.tasks.transaction(self.f.owner) as db:
            dependent = await self.tasks.payload(db,self.f.owner,{"text":"Retained evidence copy."},
                                                source_payload_ids=(self.original.payload_id,))
        await self.schedule()
        held = await self.service.process_due(self.f.owner, limit=1)
        self.assertEqual(held[0]["reason"], "retained_dependents")
        await self.schedule(self.other)
        next_pass = await self.service.process_due(self.f.owner, limit=1)
        self.assertEqual(next_pass[0]["source_id"], str(self.other.source_id))
        self.assertEqual(next_pass[0]["status"], "expired")
        async with self.f.sources.transaction(self.f.owner) as db:
            await db.execute("update jarvis_context_payloads set content=null,availability='deleted' where id=$1",dependent)
            await db.execute("update jarvis_source_retention set next_check_at=clock_timestamp() where source_id=$1",
                             self.original.source_id)
        self.assertEqual((await self.service.process_due(self.f.owner))[0]["status"], "expired")

    async def test_source_correction_requires_new_retention_authority(self):
        await self.schedule()
        corrected = await SourceLifecycleService(self.f.sources,self.tasks).correct(
            self.f.owner,self.f.area,self.original.source_id,expected_revision=1,
            replacement="A corrected fact that must not inherit an old expiry.",request=await self.new_request())
        self.assertEqual(corrected["revision"], 2)
        result = (await self.service.process_due(self.f.owner))[0]
        self.assertEqual((result["status"],result["reason"]), ("requires_review","source_changed"))
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,self.original.source_id)).revision, 2)

    async def test_removed_authority_and_foreign_requests_cannot_expire_a_source(self):
        await self.schedule()
        async with self.f.sources.transaction(self.f.owner) as db:
            await db.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where id=(select payload_id from jarvis_context_events where owner_id=$1 and id=$2)""",
                self.f.owner,self.request.event_id)
        result = (await self.service.process_due(self.f.owner))[0]
        self.assertEqual(result["reason"], "request_evidence_unavailable")
        with self.assertRaises((SourceError,TaskError)):
            await self.service.set_policy("foreign-fixture-owner",
                self.f.area,self.original.source_id,expected_source_revision=1,expected_source_generation=1,
                expected_policy_revision=1,expires_at=None,request=self.request)
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,self.original.source_id)).revision,1)

    async def test_shared_original_is_not_implicitly_enrolled_or_removed(self):
        sibling = await self.f.sources.publish(self.f.owner,SourcePublication(
            area_id=self.f.area,namespace="fixture:shared",source_key=uuid4().hex,
            payload_id=self.original.payload_id,source_type=self.original.source_type,trust=self.original.trust,
            source_time=self.original.source_time,time_basis="Shared original fixture.",expected_revision=0))
        await self.schedule()
        result = (await self.service.process_due(self.f.owner))[0]
        self.assertEqual(result["reason"], "knowledge_source_shared_original")
        self.assertEqual(result["status"], "scheduled")
        self.assertEqual((await self.f.sources.read(self.f.owner,self.f.area,sibling.source_id)).revision,1)
        self.assertEqual((await self.inspect())["source_state"],"active")

    async def test_expiry_remains_explicit_and_available_in_an_archived_area(self):
        await self.schedule()
        async with self.f.pool.acquire() as db:
            revision = await db.fetchval("select revision from jarvis_knowledge_areas where id=$1",self.f.area)
        await KnowledgeAreaService(self.f.sources,self.tasks).set_status(
            self.f.owner,self.f.area,"archived",expected_revision=revision,request=await self.new_request())
        self.assertEqual((await self.service.process_due(self.f.owner))[0]["status"],"expired")
        self.assertEqual((await self.inspect())["source_state"],"deleted")

    async def test_current_task_memory_reference_holds_its_original(self):
        task = await self.tasks.create(self.f.owner,TaskCreate(
            request_id=uuid4(),title="Retention reader",requested_outcome="Inspect the synthetic source.",
            session_id=self.session,knowledge_area_id=self.f.area),model="fixture",
            source=self.request,mode="background")
        claim = await self.tasks.claim_next("retention-fixture",("frontier",))
        self.assertEqual(claim.task.id,task.id)
        knowledge = SimpleNamespace(sources=self.f.sources,configuration=SimpleNamespace(area_id=self.f.area),
            retrieval=SimpleNamespace(sources=self.f.sources),context_for=lambda owner: None)
        prepared = await WorkerContextFactory(self.f.pool,knowledge=knowledge)(TaskRunner(self.tasks,{}).context(claim))
        result = await prepared.tools.invoke("explain_memory",{"memory_id":str(self.original.source_id)},
                                            provider_call_id=uuid4().hex)
        self.assertEqual(result["status"],"success")
        await self.schedule()
        self.assertEqual((await self.service.process_due(self.f.owner))[0]["reason"],"retained_dependents")

    async def test_virtual_days_expire_only_explicitly_scheduled_sources_across_service_instances(self):
        async with self.f.pool.acquire() as db:
            start = await db.fetchval("select clock_timestamp()")
        sources = {1: self.original}
        for day in (7, 30, 90):
            sources[day] = await self.f.publish(
                f"Synthetic explicit retention for day {day}.", source_key=f"day-{day}")
        for day, source in sources.items():
            await self.schedule(source, expires_at=start + timedelta(days=day))
        for day in (0, 1, 7, 30, 90):
            with self.subTest(day=day):
                instant = start + timedelta(days=day)
                service = SourceRetentionService(self.f.sources, self.tasks, clock=lambda: instant)
                result = await service.process_due(self.f.owner, limit=128)
                self.assertEqual([row["source_id"] for row in result],
                                 [] if day == 0 else [str(sources[day].source_id)])
                self.assertTrue(all(row["status"] == "expired" for row in result))
                self.assertEqual(await service.process_due(self.f.owner), [])
                self.assertEqual((await self.f.sources.read(
                    self.f.owner, self.f.area, self.other.source_id)).revision, 1)
                for expiry_day, source in sources.items():
                    state = await service.inspect(self.f.owner, self.f.area, source.source_id)
                    self.assertEqual(state["source_state"], "deleted" if expiry_day <= day else "active")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_memory_outbox
                where owner_id=$1 and operation='delete' and document_id=any($2::text[])""",
                self.f.owner, [source.document_id for source in sources.values()]), 4)
            audit = await db.fetchrow("""select checked_at,next_check_at,clock_timestamp() as real_now
                from jarvis_source_retention where owner_id=$1 and source_id=$2""",
                self.f.owner, sources[90].source_id)
            self.assertLessEqual(audit["checked_at"], audit["real_now"])
            self.assertLess(audit["checked_at"], start + timedelta(days=1))
            self.assertEqual(audit["next_check_at"], start + timedelta(days=90, minutes=1))

    async def test_virtual_retention_backoff_preserves_referenced_evidence_until_released(self):
        async with self.f.pool.acquire() as db:
            start = await db.fetchval("select clock_timestamp()")
        async with self.tasks.transaction(self.f.owner) as db:
            dependent = await self.tasks.payload(db, self.f.owner, {"text": "Synthetic retained evidence."},
                                                source_payload_ids=(self.original.payload_id,))
        instant = start + timedelta(days=7)
        service = SourceRetentionService(self.f.sources, self.tasks, clock=lambda: instant)
        await self.schedule(expires_at=instant)
        first = await service.process_due(self.f.owner)
        self.assertEqual(first[0]["reason"], "retained_dependents")
        self.assertEqual((await self.inspect())["source_state"], "active")
        self.assertEqual(await service.process_due(self.f.owner), [])
        instant += timedelta(seconds=59)
        self.assertEqual(await service.process_due(self.f.owner), [])
        instant += timedelta(seconds=1)
        self.assertEqual((await service.process_due(self.f.owner))[0]["reason"], "retained_dependents")
        async with self.f.sources.transaction(self.f.owner) as db:
            await db.execute("""update jarvis_context_payloads set content=null,availability='deleted'
                where id=$1""", dependent)
        instant += timedelta(minutes=1)
        self.assertEqual((await service.process_due(self.f.owner))[0]["status"], "expired")
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, self.other.source_id)).revision, 1)
