"""Conversation admission and real action receipts for frontier-selected corrections."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.knowledge.lifecycle_tools import CorrectSourceArguments
from app.runtime.feature_config import Capability, SecondBrainSettings, TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.memory_editing import MEMORY_EDIT_CAPABILITY, TaskMemoryEditingBridge
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule, StartArguments
from app.tasks.worker import WorkerContextFactory
from app.tools.invocation import bind_tool_invocation
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


def observed_search_originals(trace, phase):
    """Require the real search wire shape; missing evidence is not an empty recall."""
    responses = [event.get("result") for event in trace
                 if event.get("phase") == phase and event.get("name") == "search_memory"]
    if not responses:
        raise AssertionError("The native phase did not observe a search response")
    originals = []
    for response in responses:
        if not isinstance(response, dict) or response.get("status") != "success":
            raise AssertionError("A failed search does not establish memory absence")
        rows = response.get("results")
        if not isinstance(rows, list):
            raise AssertionError("The successful search response must contain a results list")
        originals.extend(rows)
    return originals


class MemoryEditingContractTests(unittest.TestCase):
    def test_native_search_evidence_uses_results_and_the_requested_phase(self):
        original = {"source_id": str(uuid4()), "revision": 2}
        trace = [
            {"phase": "initial", "name": "search_memory",
             "result": {"status": "success", "results": [{"source_id": "unrelated"}]}},
            {"phase": "after_correction", "name": "search_memory",
             "result": {"status": "success", "results": [original]}},
            {"phase": "after_correction", "name": "explain_memory",
             "result": {"status": "success", "source": {"source_id": "another"}}},
        ]
        self.assertEqual(observed_search_originals(trace, "after_correction"), [original])

    def test_native_search_evidence_rejects_missing_malformed_and_failed_responses(self):
        responses = [
            None, {"status": "success", "sources": []},
            {"status": "success", "results": None},
            {"status": "success", "results": {}},
            {"status": "failed", "results": []},
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(AssertionError):
                observed_search_originals(
                    [{"phase": "after_deletion", "name": "search_memory", "result": response}],
                    "after_deletion")
        with self.assertRaises(AssertionError):
            observed_search_originals([], "after_deletion")

    def test_native_empty_recall_is_distinct_from_missing_response(self):
        trace = [{"phase": "after_deletion", "name": "search_memory",
                  "result": {"status": "success", "results": []}}]
        self.assertEqual(observed_search_originals(trace, "after_deletion"), [])

    def test_capability_is_explicit_and_default_requests_keep_their_fingerprint(self):
        self.assertNotIn("memory_edit", StartArguments(title="Recall", requested_outcome="Read only").requested_capabilities)
        request = TaskCreate(request_id=uuid4(), title="Recall", requested_outcome="Read only")
        self.assertNotIn("edit_memory", TaskRepository._create_parameters(request))
        with self.assertRaises(ValidationError):
            TaskCreate(request_id=uuid4(), title="Edit", requested_outcome="Correct it", edit_memory=True)
        self.assertEqual(WorkerContextFactory(None).capabilities, ())
        self.assertEqual(WorkerContextFactory(None, knowledge=object()).capabilities, (MEMORY_EDIT_CAPABILITY,))

    def test_memory_edit_lifetime_resolves_defaults_overrides_and_invalid_values(self):
        for value, expected in (("3600", 3600), ("7200", 7200), ("0", None), ("not-a-duration", None)):
            with self.subTest(value=value):
                settings = SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true",
                                               memory_edit_grant_seconds=value)
                resolution = settings.resolve(Capability.TASKS, database_url="configured")
                self.assertEqual(resolution.configured, expected is not None)
                if expected is not None:
                    self.assertEqual(resolution.configuration.memory_edit_grant_seconds, expected)
        inactive = SecondBrainSettings(_env_file=None, enabled="false", memory_edit_grant_seconds="invalid")
        self.assertFalse(inactive.resolve(Capability.TASKS, database_url="configured").enabled)

    def test_memory_edit_lifetime_uses_the_existing_settings_environment_prefix(self):
        with patch.dict(os.environ, {"SECOND_BRAIN_MEMORY_EDIT_GRANT_SECONDS": "7200"}):
            settings = SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true")
            self.assertEqual(settings.resolve(Capability.TASKS, database_url="configured")
                             .configuration.memory_edit_grant_seconds, 7200)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class FrontierMemoryEditingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.history = ContextRepository(self.tasks)
        self.original = await self.f.publish("Robot Cedar version one uses firmware 2.3 and two stepper motors.")
        self.other = await self.f.publish("Robot Cedar version two uses firmware 2.3 and four servo motors.",
                                          source_key="version-two")
        self.parameters = {"source_id": str(self.original.source_id), "expected_revision": 1,
            "replacement": "Robot Cedar version one uses firmware 2.4 and two stepper motors."}
        self.knowledge = SimpleNamespace(sources=self.f.sources,
            configuration=SimpleNamespace(area_id=self.f.area), retrieval=SimpleNamespace(sources=self.f.sources),
            context_for=lambda owner: self.assertEqual(owner, self.f.owner))
        async with self.f.pool.acquire() as db:
            self.session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
        self.request_text = "Correct only Cedar version one's firmware to 2.4; preserve all other facts."
        self.request = None
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-model")},
                                   knowledge_area_resolver=self.resolve_area)

    async def resolve_area(self, connection, owner, request, event):
        return request.knowledge_area_id or self.f.area

    async def utterance(self, text, *, session_id=None):
        return await self.history.append(self.f.owner, ContextIngress(
            session_id=session_id or self.session, source_namespace="voice:frontier-edit-fixture", source_event_id=uuid4().hex,
            receipt_id=uuid4().hex, kind="user_transcript", trust="user_input", payload={"text": text}))

    async def start(self, *, editing=True, slot=1):
        if self.request is None:
            self.request = await self.utterance(self.request_text)
        module = BackgroundTaskToolModule(self.service)
        tools = module.build_tools(ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=self.session))
        tool = next(tool for tool in tools if tool.name == "start_background_task")
        with bind_tool_invocation(provider_call_id=uuid4().hex,
                                  request_event_id=self.request.id, request_revision=self.request.revision):
            return await tool.handler({"title": "Correct retained Cedar knowledge" if editing else "Recall retained Cedar knowledge",
                "requested_outcome": self.request_text, "mode": "consult", "request_slot": slot,
                "requested_capabilities": ["reasoning", "session_context"] + (["memory_edit"] if editing else [])})

    async def prepare(self, result):
        identity = await self.tasks.resolve_reference(self.f.owner, result["task"]["task_ref"])
        claim = await self.tasks.claim_next("frontier-memory-edit-fixture", ("frontier",))
        self.assertEqual(claim.task.id, identity)
        ctx = TaskRunner(self.tasks, {}).context(claim)
        return ctx, await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(ctx)

    async def read(self, prepared, source=None):
        result = await prepared.tools.invoke("explain_memory",
            {"memory_id": str((source or self.original).source_id)}, provider_call_id=uuid4().hex)
        self.assertEqual(result["status"], "success")
        return result

    async def correct(self, prepared, parameters=None):
        return await prepared.tools.invoke("correct_memory_source", parameters or self.parameters,
                                           provider_call_id=uuid4().hex)

    async def current(self, source=None):
        return await self.f.sources.read(self.f.owner, self.f.area, (source or self.original).source_id)

    async def test_voice_admission_replays_and_frontier_prepares_exact_effect(self):
        first, retry = await self.start(), await self.start()
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["task"]["task_ref"], retry["task"]["task_ref"])
        self.assertIn("memory_edit", first["capabilities"])
        ctx, prepared = await self.prepare(first)
        self.assertIsInstance(prepared.tools, TaskMemoryEditingBridge)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_operation_grants"), 1)
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)
        await self.read(prepared)
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual((await self.current()).content["text"], self.parameters["replacement"])
        self.assertEqual((await self.current(self.other)).content, self.other.content)
        again = await self.correct(prepared)
        self.assertTrue(again["task_operation"]["replayed"])
        self.assertEqual(again["task_operation"]["operation_ref"], result["task_operation"]["operation_ref"])
        referenced = await prepared.tools.invoke("correct_memory_source",
            {"operation_ref": result["task_operation"]["operation_ref"]}, provider_call_id=uuid4().hex)
        self.assertEqual(referenced["status"], "success")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)
            self.assertEqual(await db.fetchval("select count(*) from jarvis_operation_grants"), 2)

    async def test_missing_original_read_is_recoverable_before_submission(self):
        _, prepared = await self.prepare(await self.start())
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "task_memory_evidence_not_read")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)
            self.assertEqual(await db.fetchval("select count(*) from jarvis_operation_grants"), 1)
        await self.read(prepared)
        self.assertEqual((await self.correct(prepared))["status"], "success")

    async def test_readonly_task_has_no_correction_authority(self):
        _, prepared = await self.prepare(await self.start(editing=False))
        self.assertNotIn("correct_memory_source", {tool["name"] for tool in prepared.tools.catalog()})
        self.assertEqual((await self.correct(prepared))["status"], "failed")
        self.assertEqual((await self.current()).revision, 1)

    async def test_worker_selection_requires_the_admitted_memory_capability(self):
        result = await self.start()
        denied = await self.tasks.claim_next("incapable-worker", ("frontier",), worker_capabilities=())
        self.assertIsNone(denied)
        claim = await self.tasks.claim_next("capable-worker", ("frontier",),
                                           worker_capabilities=(MEMORY_EDIT_CAPABILITY,))
        self.assertEqual(claim.task.id,
                         await self.tasks.resolve_reference(self.f.owner, result["task"]["task_ref"]))

    async def test_readonly_task_does_not_require_memory_editing_worker_capability(self):
        result = await self.start(editing=False)
        claim = await self.tasks.claim_next("reasoning-worker", ("frontier",), worker_capabilities=())
        self.assertEqual(claim.task.id,
                         await self.tasks.resolve_reference(self.f.owner, result["task"]["task_ref"]))

    async def test_not_configured_and_missing_worker_cannot_admit_editing(self):
        self.service.knowledge_area_resolver = None
        self.assertEqual((await self.start())["status"], "failed")
        self.service.knowledge_area_resolver = self.resolve_area
        self.service.registry = SimpleNamespace(require=AsyncMock(side_effect=TaskError("worker_missing", "No worker.", 503)))
        self.assertEqual((await self.start())["status"], "failed")
        self.assertIn(MEMORY_EDIT_CAPABILITY,
                      self.service.registry.require.await_args.kwargs["required_capabilities"])
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_background_tasks"), 0)
            self.assertEqual(await db.fetchval("select count(*) from jarvis_operation_grants"), 0)

    async def test_new_call_ids_concurrently_reuse_one_operation(self):
        _, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        results = await asyncio.gather(self.correct(prepared), self.correct(prepared))
        self.assertTrue(all(result["status"] in {"success", "outcome_unknown"} for result in results))
        self.assertEqual((await self.correct(prepared))["status"], "success")
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)

    async def test_same_request_cannot_rewrite_again_under_new_revision(self):
        _, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        self.assertEqual((await self.correct(prepared))["status"], "success")
        result = await self.correct(prepared, self.parameters | {"expected_revision": 2, "replacement": "Changed again."})
        self.assertEqual(result["error"]["code"], "memory_edit_operation_conflict")
        self.assertEqual((await self.current()).revision, 2)

    async def test_two_tasks_for_same_request_cannot_duplicate_the_effect(self):
        first, second = await self.start(), await self.start(slot=2)
        _, one = await self.prepare(first)
        await self.read(one)
        self.assertEqual((await self.correct(one))["status"], "success")
        _, two = await self.prepare(second)
        result = await self.correct(two, self.parameters | {"expected_revision": 2})
        self.assertEqual(result["error"]["code"], "memory_edit_already_owned")
        self.assertEqual((await self.current()).revision, 2)

    async def test_missing_source_or_stale_revision_creates_no_action(self):
        _, prepared = await self.prepare(await self.start())
        result = await self.correct(prepared, self.parameters | {"source_id": str(uuid4())})
        self.assertEqual(result["status"], "failed")
        result = await self.correct(prepared, self.parameters | {"expected_revision": 99})
        self.assertEqual(result["error"]["code"], "knowledge_source_revision_conflict")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)

    async def test_actual_foreign_owner_and_other_area_sources_cannot_be_corrected(self):
        _, prepared = await self.prepare(await self.start())
        for owner in (self.f.other, self.f.owner):
            with self.subTest(owner=owner):
                async with self.f.pool.acquire() as db:
                    area = await db.fetchval("""insert into jarvis_knowledge_areas(owner_id,name,domain)
                        values($1,'Separate scope','test') returning id""", owner)
                foreign = replace(self.f, owner=owner, area=area)
                original = await foreign.publish("Separate original.", source_key=uuid4().hex)
                result = await self.correct(prepared, self.parameters | {"source_id": str(original.source_id)})
                self.assertEqual(result["status"], "failed")
                current = await foreign.sources.read(owner, area, original.source_id)
                self.assertEqual(current.revision, 1)
                self.assertEqual(current.content, original.content)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)

    async def invalidate_parent(self, status):
        _, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        async with self.f.pool.acquire() as db:
            await db.execute("""update jarvis_operation_grants set status=$2
                where owner_id=$1 and capability=$3""", self.f.owner, status, MEMORY_EDIT_CAPABILITY)
        with self.assertRaises(ClaimLost):
            await self.correct(prepared)
        self.assertEqual((await self.current()).revision, 1)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)

    async def test_revoked_parent_cannot_issue_or_submit_a_correction(self):
        await self.invalidate_parent("revoked")

    async def test_expired_parent_cannot_issue_or_submit_a_correction(self):
        await self.invalidate_parent("expired")

    async def test_prepared_operations_consume_the_bounded_source_allowance(self):
        self.tasks.limits = TaskLimits(lease_seconds=300, tool_steps=2)
        _, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        await self.read(prepared, self.other)
        await prepared.tools.prepare(CorrectSourceArguments(**self.parameters))
        await prepared.tools.prepare(CorrectSourceArguments(**(
            self.parameters | {"source_id": str(self.other.source_id)})))
        with self.assertRaises(TaskError) as caught:
            await prepared.tools.prepare(CorrectSourceArguments(**(
                self.parameters | {"source_id": str(uuid4())})))
        self.assertEqual(caught.exception.code, "memory_edit_allowance_exhausted")
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)
        self.assertEqual((await self.current()).revision, 1)

    async def test_fresh_worker_context_replays_the_durable_receipt(self):
        ctx, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        result = await self.correct(prepared)
        rebuilt = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(
            TaskRunner(self.tasks, {}).context(ctx.claim))
        replay = await self.correct(rebuilt)
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(replay["task_operation"]["operation_ref"], result["task_operation"]["operation_ref"])
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)

    async def test_resume_does_not_authorize_a_second_edit_of_the_same_source(self):
        ctx, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        self.assertEqual((await self.correct(prepared))["status"], "success")
        self.assertTrue(await self.tasks.finish(ctx.claim, WorkerOutcome(TaskStatus.PAUSED, {"checkpoint": "effect accepted"})))
        resume = await self.utterance("Resume that task.")
        current = await self.tasks.get(self.f.owner, ctx.task.id)
        await self.service.mutate(self.f.owner, current.id, "resume", TaskMutation(
            request_id=uuid4(), expected_revision=current.revision),
            source=TaskVoiceSource(resume.id, resume.revision, self.session))
        claim = await self.tasks.claim_next("resumed-memory-edit", ("frontier",))
        context = TaskRunner(self.tasks, {}).context(claim)
        resumed = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(context)
        await self.read(resumed)
        result = await self.correct(resumed, self.parameters | {"expected_revision": 2})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "memory_edit_operation_conflict")
        replay = await self.correct(resumed)
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual(replay["authorization"]["request_event_id"], str(self.request.id))
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)

    async def test_repeated_cross_session_resumes_reuse_original_authority_and_receipt(self):
        ctx, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        initial = await self.correct(prepared)
        for _ in range(2):
            self.assertTrue(await self.tasks.finish(ctx.claim, WorkerOutcome(TaskStatus.PAUSED, {"checkpoint": "effect accepted"})))
            async with self.f.pool.acquire() as db:
                session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            resume = await self.utterance("Resume that task.", session_id=session)
            current = await self.tasks.get(self.f.owner, ctx.task.id)
            await self.service.mutate(self.f.owner, current.id, "resume", TaskMutation(
                request_id=uuid4(), expected_revision=current.revision),
                source=TaskVoiceSource(resume.id, resume.revision, session))
            claim = await self.tasks.claim_next("cross-session-memory-edit", ("frontier",),
                                               worker_capabilities=(MEMORY_EDIT_CAPABILITY,))
            ctx = TaskRunner(self.tasks, {}).context(claim)
            rebuilt = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(ctx)
            replay = await self.correct(rebuilt)
            self.assertEqual(replay["status"], "success")
            self.assertTrue(replay["task_operation"]["replayed"])
            self.assertEqual(replay["task_operation"]["operation_ref"], initial["task_operation"]["operation_ref"])
            self.assertEqual(replay["authorization"]["request_event_id"], str(self.request.id))
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)

    async def test_explicit_updated_instructions_can_authorize_a_new_correction(self):
        ctx, prepared = await self.prepare(await self.start())
        await self.read(prepared)
        first = await self.correct(prepared)
        self.assertTrue(await self.tasks.finish(ctx.claim, WorkerOutcome(TaskStatus.WAITING, {"message": "Ready for follow-up"})))
        request = await self.utterance("Actually set version one's firmware to 2.5; preserve the rest.")
        current = await self.tasks.get(self.f.owner, ctx.task.id)
        await self.service.mutate(self.f.owner, current.id, "update", TaskMutation(
            request_id=uuid4(), expected_revision=current.revision,
            message="Correct only Cedar version one's firmware to 2.5; preserve all other facts."),
            source=TaskVoiceSource(request.id, request.revision, self.session))
        claim = await self.tasks.claim_next("updated-memory-edit", ("frontier",))
        ctx = TaskRunner(self.tasks, {}).context(claim)
        revised = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(ctx)
        await self.read(revised)
        result = await self.correct(revised, self.parameters | {"expected_revision": 2,
            "replacement": "Robot Cedar version one uses firmware 2.5 and two stepper motors."})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["authorization"]["request_event_id"], str(request.id))
        self.assertNotEqual(result["task_operation"]["operation_ref"], first["task_operation"]["operation_ref"])
        self.assertEqual((await self.current()).revision, 3)
        self.assertEqual((await self.current()).content["text"],
                         "Robot Cedar version one uses firmware 2.5 and two stepper motors.")
        self.assertEqual((await self.current(self.other)).content, self.other.content)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 2)

    async def test_scenario_request_is_the_only_seeded_user_utterance(self):
        self.request_text = "Correct Cedar's firmware to 2.4; preserve everything else."
        await self.start()
        await self.assert_only_scenario_request()

    async def assert_only_scenario_request(self):
        async with self.f.pool.acquire() as db:
            records = await db.fetch("""select e.id,p.content->>'text' as text
                from jarvis_context_events e join jarvis_context_payloads p
                  on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.session_id=$2 and e.kind='user_transcript'
                order by e.sequence""", self.f.owner, self.session)
        self.assertEqual([(row["id"], row["text"]) for row in records],
                         [(self.request.id, self.request_text)])
        return [str(row["id"]) for row in records]

    async def test_clarification_followup_uses_new_canonical_request(self):
        ctx, prepared = await self.prepare(await self.start())
        self.assertTrue(await self.tasks.finish(ctx.claim, WorkerOutcome(TaskStatus.WAITING, {"question": "Which version?"})))
        followup = await self.utterance("Version one only. Change its firmware to 2.4.")
        current = await self.tasks.get(self.f.owner, ctx.task.id)
        changed = await self.service.mutate(self.f.owner, current.id, "update", TaskMutation(
            request_id=uuid4(), expected_revision=current.revision, message=self.request_text),
            source=TaskVoiceSource(followup.id, followup.revision, self.session))
        claim = await self.tasks.claim_next("frontier-memory-edit-followup", ("frontier",))
        self.assertEqual(claim.task.id, changed.id)
        context = TaskRunner(self.tasks, {}).context(claim)
        revised = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(context)
        await self.read(revised)
        result = await self.correct(revised)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["authorization"]["request_event_id"], str(followup.id))
        self.assertEqual((await self.current()).revision, 2)

    @unittest.skipUnless(os.getenv("JARVIS_TEST_NATIVE_MEMORY_EDITING") == "1", "Explicit real native editing acceptance required")
    async def test_native_frontier_selects_and_corrects_from_conversation(self):
        await self.native_case(ambiguous=False)

    @unittest.skipUnless(os.getenv("JARVIS_TEST_NATIVE_MEMORY_EDITING") == "1", "Explicit real native editing acceptance required")
    async def test_native_ambiguous_request_clarifies_without_editing(self):
        await self.native_case(ambiguous=True)

    async def native_case(self, *, ambiguous):
        import time
        from app.knowledge.runtime import KnowledgeRuntime
        from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, KnowledgeConfiguration
        from app.settings import JarvisSettings
        from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
        from evals.second_brain.hindsight_semantic_trace import drain
        from evals.second_brain.hindsight_service import save
        label = "ambiguous" if ambiguous else "precise"
        model = os.getenv("JARVIS_S04_MODEL", DEFAULT_FRONTIER_MODEL)
        root = Path(os.environ["JARVIS_NATIVE_MEMORY_EDITING_ROOT"])
        report = {"case": label, "model": model, "worker_events": [], "tools": [],
                  "scope": "real_conversation_tool_admission_and_native_frontier_not_live_audio_or_playback"}
        phase = "initial"
        self.knowledge = await KnowledgeRuntime.create(pool=self.f.pool, owner_id=self.f.owner, tasks=self.tasks,
            configuration=KnowledgeConfiguration(base_url=os.environ["JARVIS_S04_MEMORY_URL"],
                api_key=os.environ["JARVIS_S04_MEMORY_TOKEN"], area_id=self.f.area))
        self.service.providers["frontier"] = SimpleNamespace(model=model)
        try:
            if ambiguous:
                self.request_text = "Correct Cedar's firmware to 2.4; preserve everything else."
            await drain(self.f, self.knowledge.worker, report, label="native-frontier-edit-seed")
            admission = await self.start()
            report["admission"] = admission
            self.assertEqual(admission["status"], "success")
            report["seeded_user_request_ids"] = await self.assert_only_scenario_request()
            claim = await self.tasks.claim_next("native-frontier-memory-edit", ("frontier",))
            ctx = TaskRunner(self.tasks, {}).context(claim)
            factory = WorkerContextFactory(self.f.pool, knowledge=self.knowledge)
            async def prepare(context):
                prepared = await factory(context)
                invoke = prepared.tools.invoke
                async def observe(name, arguments, *, provider_call_id):
                    event = {"name": name, "arguments": arguments, "phase": phase}
                    report["tools"].append(event)
                    try:
                        result = await invoke(name, arguments, provider_call_id=provider_call_id)
                        event["result"] = result
                        return result
                    except Exception as error:
                        event["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
                        raise
                prepared.tools.invoke = observe
                return prepared
            provider = HermesTaskProvider(FrontierConfiguration(model=model,
                reasoning_effort="medium", api_key=JarvisSettings().OPENAI_API_KEY),
                HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]), root / ("hermes-" + label)),
                prepare=prepare)
            started = time.monotonic()
            outcome = await asyncio.wait_for(provider.execute(ctx), 240)
            report.update(worker_seconds=time.monotonic()-started,
                outcome={"status": outcome.status.value, "payload": outcome.payload},
                usage=await ctx.budgets.usage(self.f.owner, ctx.task.id))
            self.assertTrue(await self.tasks.finish(ctx.claim, outcome))
            original, other = await self.current(), await self.current(self.other)
            report["originals_after"] = [original.original_evidence(), other.original_evidence()]
            self.assertEqual(other.content, self.other.content)
            if ambiguous:
                self.assertEqual(outcome.status, TaskStatus.WAITING)
                self.assertEqual(original.revision, 1)
                async with self.f.pool.acquire() as db:
                    self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)
                # Actual user clarification through the foreground tool boundary;
                # neither the source UUID nor replacement body enters admission.
                followup = await self.utterance("Version one only.")
                current = await self.tasks.get(self.f.owner, ctx.task.id)
                tools = BackgroundTaskToolModule(self.service).build_tools(
                    ToolBuildContext(settings=None, owner_id=self.f.owner, session_id=self.session))
                update = next(tool for tool in tools if tool.name == "update_background_task")
                with bind_tool_invocation(provider_call_id=uuid4().hex, request_event_id=followup.id,
                                          request_revision=followup.revision):
                    changed = await update.handler({"task_ref": admission["task"]["task_ref"],
                        "expected_revision": current.revision,
                        "message": "Correct only Cedar version one's firmware to 2.4; preserve everything else."})
                self.assertEqual(changed["status"], "success")
                phase = "clarification"
                claim = await self.tasks.claim_next("native-clarified-memory-edit", ("frontier",),
                                                   worker_capabilities=(MEMORY_EDIT_CAPABILITY,))
                context = TaskRunner(self.tasks, {}).context(claim)
                began = time.monotonic()
                continued = await asyncio.wait_for(provider.execute(context), 240)
                report["continuation"] = {"admission": changed, "request_event_id": str(followup.id),
                    "worker_seconds": time.monotonic()-began,
                    "outcome": {"status": continued.status.value, "payload": continued.payload}}
                self.assertTrue(await self.tasks.finish(context.claim, continued))
                corrected = await self.current()
                report["continuation"]["original_after"] = corrected.original_evidence()
                self.assertEqual(continued.status, TaskStatus.SUCCEEDED)
                self.assertEqual(corrected.revision, 2)
                self.assertEqual(corrected.content["text"], self.parameters["replacement"])
                self.assertEqual(corrected.content["request"]["event_id"], str(followup.id))
                self.assertEqual((await self.current(self.other)).content, self.other.content)
                self.assertEqual(len(continued.payload["completed_operation_refs"]), 1)
                self.assertTrue(any(row.get("source_id") == str(corrected.source_id) and row["revision"] == 2
                                    for row in continued.payload["evidence"]))
            else:
                self.assertEqual(outcome.status, TaskStatus.SUCCEEDED)
                self.assertEqual(original.revision, 2)
                self.assertEqual(original.content["text"], self.parameters["replacement"])
                self.assertEqual(len(outcome.payload["completed_operation_refs"]), 1)
                self.assertTrue(any(row.get("source_id") == str(original.source_id) and row["revision"] == 2
                                    for row in outcome.payload["evidence"]))

            # A local receipt is not proof of searchable, cross-session memory.
            # Exercise the actual outbox and native provider, then ask a fresh
            # task which has neither the correction transcript nor edit authority.
            report["roundtrip"] = {"semantic_review": "not_automatically_established"}
            await drain(self.f, self.knowledge.worker, report, label="native-frontier-correction-indexing")
            indexed = await self.current()
            self.assertEqual(indexed.ingestion_status, "searchable")
            report["roundtrip"]["indexed_original"] = indexed.original_evidence()

            async def recall_in_new_session(query, stage):
                nonlocal phase
                async with self.f.pool.acquire() as db:
                    self.session = await db.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
                self.request_text = query
                self.request = await self.utterance(query)
                admitted = await self.start(editing=False)
                self.assertEqual(admitted["status"], "success")
                phase = stage
                recalled_claim = await self.tasks.claim_next(
                    "native-memory-" + stage, ("frontier",), worker_capabilities=())
                self.assertIsNotNone(recalled_claim)
                self.assertEqual(recalled_claim.task.id, await self.tasks.resolve_reference(
                    self.f.owner, admitted["task"]["task_ref"]))
                recalled_context = TaskRunner(self.tasks, {}).context(recalled_claim)
                began = time.monotonic()
                recalled = await asyncio.wait_for(provider.execute(recalled_context), 240)
                report["roundtrip"][stage] = {
                    "admission": admitted, "session_id": str(self.session), "query": query,
                    "worker_seconds": time.monotonic()-began,
                    "outcome": {"status": recalled.status.value, "payload": recalled.payload},
                    "usage": await recalled_context.budgets.usage(self.f.owner, recalled_claim.task.id),
                }
                self.assertTrue(await self.tasks.finish(recalled_claim, recalled))
                self.assertIn("search_memory", {row["name"] for row in recalled.payload.get("tool_receipts", ())})
                self.assertFalse(recalled.payload.get("completed_operation_refs"))
                return recalled_claim.task.id, recalled

            recall_task, recalled = await recall_in_new_session(
                "What firmware and motors do Cedar version one and version two currently use?", "after_correction")
            self.assertEqual(recalled.status, TaskStatus.SUCCEEDED)
            cited = {(row.get("source_id"), row.get("revision"), row.get("source_usage"))
                     for row in recalled.payload["evidence"]}
            self.assertIn((str(self.original.source_id), 2, "current"), cited)
            self.assertIn((str(self.other.source_id), 1, "current"), cited)
            self.assertNotIn((str(self.original.source_id), 1, "current"), cited)
            recalled_originals = observed_search_originals(report["tools"], "after_correction")
            retrieved = {(row["source_id"], row["revision"]) for row in recalled_originals}
            self.assertIn((str(self.original.source_id), 2), retrieved)
            self.assertIn((str(self.other.source_id), 1), retrieved)
            self.assertNotIn((str(self.original.source_id), 1), retrieved)
            self.assertEqual((await self.current()).revision, 2)
            self.assertEqual((await self.current(self.other)).content, self.other.content)
            async with self.f.pool.acquire() as db:
                self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)

            # Delete through the real source lifecycle, not a mocked recall.
            # Saved derived answers must be withdrawn, and a later session must
            # not reconstruct the deleted version from a similar surviving one.
            await self.f.sources.delete_source(self.f.owner, self.f.area, self.original.source_id,
                                               expected_revision=2)
            self.assertIsNone((await self.tasks.get(self.f.owner, recall_task)).result_payload_id)
            await drain(self.f, self.knowledge.worker, report, label="native-frontier-correction-deletion")
            _, missing = await recall_in_new_session(
                "What firmware and motors does Cedar version one use?", "after_deletion")
            self.assertIn(missing.status, (TaskStatus.WAITING, TaskStatus.SUCCEEDED))
            self.assertTrue(missing.payload.get("uncertainties"))
            self.assertNotIn(str(self.original.source_id),
                             {row.get("source_id") for row in missing.payload.get("evidence", ())})
            remaining = {row["source_id"] for row in observed_search_originals(
                report["tools"], "after_deletion")}
            self.assertNotIn(str(self.original.source_id), remaining)
            self.assertIn(str(self.other.source_id), remaining)
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
            raise
        finally:
            await self.knowledge.close()
            save(root / ("native-" + label + ".json"), report)
