"""Manifested disposable SQL acceptance for task-only lean commands.

Run with evals.second_brain.run_tests --live-db --manifest-dir PRIVATE_DIR
--pattern test_lean_task_admission_sql.py. No provider, worker, voice transport,
storage service, or attention suppression is started. Closing the durable app
session below is a source-lifetime probe, not native disconnect qualification.
"""

import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.knowledge.session_areas import SessionKnowledgeAreas
from app.live_state.admission import CommandAdmission, command_parameters, task_command_source
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, digest
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from evals.second_brain.source_fixture import source_fixture


class FixtureRollback(Exception):
    """Deliberate test-only failure after actual acceptance SQL has executed."""


class RollbackAfterAcceptedEvent(TaskRepository):
    observed_uncommitted_acceptance = False

    async def emit(self, connection, task, kind, **kwargs):
        await super().emit(connection, task, kind, **kwargs)
        if kind == "accepted":
            self.observed_uncommitted_acceptance = connection.is_in_transaction() and await connection.fetchval(
                """select exists(select 1 from jarvis_background_tasks t
                    join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id
                    join jarvis_context_events c on c.owner_id=t.owner_id and c.id=t.request_event_id
                    join jarvis_task_events e on e.owner_id=t.owner_id and e.task_id=t.id
                    where t.owner_id=$1 and t.id=$2 and r.operation='create'
                        and c.source_namespace='voice_command' and c.kind='command_request'
                        and e.event_type='accepted')""", task.owner_id, task.id)
            raise FixtureRollback()


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Requires an explicit manifested disposable SQL fixture",
)
class LeanTaskAdmissionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / self._testMethodName
        self.evidence = {
            "test": self.id(), "classification": "real_SQL_command_admission_no_provider_no_worker",
            "provider_used": False, "worker_started": False, "voice_transport_started": False,
            "storage_used": False, "attention_suppression_tested": False, "checks": [],
        }
        self.addAsyncCleanup(self.write_evidence)
        self.stack = AsyncExitStack()
        self.addAsyncCleanup(self.stack.aclose)
        self.fixture = await self.stack.enter_async_context(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], self.directory, test_name=self.id()))
        self.owner, self.other = self.fixture.owner, self.fixture.other
        self.repo = TaskRepository(self.fixture.pool, TaskLimits())
        self.context = ContextRepository(self.repo)
        self.session = await SessionRepository(self.fixture.pool).create_session(
            self.owner, "Synthetic lean command SQL fixture")
        self.service = self.service_for(self.repo)
        self.evidence.update(owner_id=self.owner, other_owner_id=self.other,
            schema=self.fixture.fixture.manifest["schemas"][0], manifest=str(self.fixture.fixture.path))

    def service_for(self, repository):
        areas = SessionKnowledgeAreas(self.fixture.sources, repository, self.fixture.area)
        # A configuration descriptor only: no execute/reconcile or provider client.
        return TaskService(repository, {"frontier": SimpleNamespace(model="fixture-no-provider")},
                           knowledge_area_resolver=areas.task_area_in_transaction)

    async def write_evidence(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        fixture = getattr(self, "fixture", None)
        if fixture is not None:
            self.evidence.update(cleanup=fixture.fixture.manifest.get("cleanup"),
                                 fixture_status=fixture.fixture.manifest["status"])
        with (self.directory / ("checks-" + uuid4().hex + ".json")).open("x") as output:
            json.dump(self.evidence, output, indent=2, default=str)

    def plan(self, *names):
        self.evidence["checks"] = [{"name": name, "status": "unreached"} for name in names]

    def check(self, name, condition, **details):
        row = next(row for row in self.evidence["checks"] if row["name"] == name)
        row.update(status="passed" if condition else "failed", **details)
        self.assertTrue(condition, name)

    def command(self, operation="create", *, arguments=None):
        return CommandAdmission(owner_id=self.owner, session_id=self.session.id,
            provider_session_key=" fixture-provider-session \u00e9 ", provider_call_id=" call:" + uuid4().hex + " ",
            tool_name="fixture_task_" + operation, capability="tasks." + operation,
            normalized_arguments=arguments if arguments is not None else {
                "goal": "Compare synthetic desk layouts without external actions."})

    def request(self, command, **changes):
        values = dict(request_id=command.operation_id, session_id=self.session.id,
                      title="Synthetic desk comparison", requested_outcome="Compare two fictional desk layouts.")
        values.update(changes)
        return TaskCreate(**values)

    async def create(self):
        command = self.command()
        request = self.request(command)
        task = await self.service.create(self.owner, request, command_admission=command, mode="background")
        return task, command, request

    async def owner_counts(self):
        tables = (
            "jarvis_background_tasks", "jarvis_task_requests", "jarvis_context_events",
            "jarvis_context_ingress_receipts", "jarvis_context_payloads",
            "jarvis_context_payload_dependencies", "jarvis_task_events", "jarvis_task_session_context",
        )
        # Fixed fixture-owned identifiers only; values remain query parameters.
        query = "select " + ",".join(
            f"(select count(*) from {table} where owner_id=$1) as {table}" for table in tables)
        async with self.fixture.pool.acquire() as connection:
            return dict(await connection.fetchrow(query, self.owner))

    async def test_create_source_task_receipt_and_context_share_one_sql_transaction(self):
        self.plan("joined_acceptance", "single_transaction", "exact_source_and_receipt",
                  "task_context_binding", "no_worker_or_grants")
        task, command, _ = await self.create()
        async with self.fixture.pool.acquire() as connection:
            row = await connection.fetchrow("""select
                    t.xmin::text as task_xid,c.xmin::text as context_xid,p.xmin::text as source_payload_xid,
                    d.xmin::text as description_xid,r.xmin::text as receipt_xid,
                    i.xmin::text as ingress_xid,b.xmin::text as binding_xid,
                    e.xmin::text as event_xid,lineage.xmin::text as dependency_xid,
                    r.operation,r.accepted_revision,r.parameters_digest,
                    c.kind,c.trust,c.source_namespace,c.source_event_id,c.sequence,c.payload_id,
                    b.session_id,b.request_event_id,b.request_revision
                from jarvis_background_tasks t
                join jarvis_context_events c on c.owner_id=t.owner_id and c.id=t.request_event_id
                    and c.revision=t.request_revision
                join jarvis_context_payloads p on p.owner_id=c.owner_id and p.id=c.payload_id
                join jarvis_context_payloads d on d.owner_id=t.owner_id and d.id=t.description_payload_id
                join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id and r.request_id=$3
                join jarvis_context_ingress_receipts i on i.owner_id=c.owner_id and i.event_id=c.id and i.receipt_id=$4
                join jarvis_task_session_context b on b.owner_id=t.owner_id and b.task_id=t.id and b.session_id=t.session_id
                join jarvis_task_events e on e.owner_id=t.owner_id and e.task_id=t.id
                    and e.event_type='accepted' and e.payload_id=t.description_payload_id
                join jarvis_context_payload_dependencies lineage on lineage.owner_id=t.owner_id
                    and lineage.source_payload_id=c.payload_id and lineage.dependent_payload_id=t.description_payload_id
                where t.owner_id=$1 and t.id=$2""", self.owner, task.id, command.operation_id, command.journal_call_key)
            activity = await connection.fetchrow("""select
                (select count(*) from jarvis_task_attempts where owner_id=$1 and task_id=$2) as attempts,
                (select count(*) from jarvis_task_grants where owner_id=$1 and task_id=$2) as grants,
                (select count(*) from jarvis_task_actions where owner_id=$1 and task_id=$2) as actions""", self.owner, task.id)
        self.check("joined_acceptance", row is not None, task_id=task.id, operation_id=command.operation_id)
        xids = {key: row[key] for key in row.keys() if key.endswith("_xid")}
        self.check("single_transaction", len(xids) == 9 and len(set(xids.values())) == 1, insertion_xids=xids)
        source = await self.context.get(self.owner, task.request_event_id)
        self.check("exact_source_and_receipt",
            (row["source_namespace"], row["kind"], row["trust"]) == ("voice_command", "command_request", "derived")
            and row["source_event_id"] == str(command.operation_id) and row["operation"] == "create"
            and row["accepted_revision"] == task.revision == 1
            and row["parameters_digest"] == digest(command_parameters(command))
            and source.content["command"] == command.parameters() and not source.authorizes_request)
        async with self.repo.read_transaction() as connection:
            sessions, sources = await self.context._task_sessions(connection, task)
        self.check("task_context_binding", row["session_id"] == self.session.id
            and row["request_event_id"] == source.id and row["request_revision"] == source.revision
            and task.context_watermark == source.sequence and self.session.id in sessions
            and [record.id for record in sources] == [source.id])
        self.check("no_worker_or_grants", dict(activity) == {"attempts": 0, "grants": 0, "actions": 0})

    async def test_failure_after_actual_acceptance_sql_rolls_back_all_copies_and_receipts(self):
        self.plan("acceptance_observed_before_failure", "all_owner_rows_rolled_back")
        repository = RollbackAfterAcceptedEvent(self.fixture.pool, TaskLimits())
        service = self.service_for(repository)
        command = self.command()
        before = await self.owner_counts()
        with self.assertRaises(FixtureRollback):
            await service.create(self.owner, self.request(command), command_admission=command, mode="background")
        self.check("acceptance_observed_before_failure", repository.observed_uncommitted_acceptance)
        after = await self.owner_counts()
        self.check("all_owner_rows_rolled_back", after == before, before=before, after=after)

    async def test_create_replay_pins_area_before_any_new_resolution(self):
        self.plan("replay_skips_resolver", "original_scope_and_payload_retained", "no_duplicate_rows")
        task, command, request = await self.create()
        source = await self.context.get(self.owner, task.request_event_id)
        before = await self.owner_counts()

        async def forbidden_resolver(*args, **kwargs):
            self.fail("A replay must not resolve a new session area")

        self.service.knowledge_area_resolver = forbidden_resolver
        # Deliberately nonexistent new area: a fresh acceptance would fail SQL scope validation.
        replay = await self.service.create(self.owner,
            request.model_copy(update={"knowledge_area_id": uuid4()}), command_admission=command, mode="background")
        self.check("replay_skips_resolver", replay.id == task.id and replay.revision == task.revision)
        reread = await self.context.get(self.owner, task.request_event_id)
        self.check("original_scope_and_payload_retained", replay.knowledge_area_id == self.fixture.area
            and reread == source and source.content["area_id"] == str(self.fixture.area)
            and source.content["task_id"] == str(task.id) and source.content["base_revision"] == 0)
        self.check("no_duplicate_rows", await self.owner_counts() == before)

    async def test_concurrent_cancel_and_later_replay_do_not_repeat_revision_or_retarget(self):
        self.plan("parallel_single_acceptance", "original_target_and_base_pinned", "one_cancel_receipt",
                  "resume_keeps_original_cancel_historical")
        task, _, _ = await self.create()
        command = self.command("cancel", arguments={"target": "current fixture task"})
        request = TaskMutation(request_id=command.operation_id, expected_revision=task.revision)
        outcomes = await asyncio.gather(*(
            self.service.mutate(self.owner, task.id, "cancel", request, command_admission=command)
            for _ in range(2)), return_exceptions=True)
        self.check("parallel_single_acceptance", all(not isinstance(value, BaseException) for value in outcomes)
            and all(value.id == task.id and value.revision == task.revision + 1
                    and value.status == TaskStatus.CANCELLED for value in outcomes),
            outcome_types=[type(value).__name__ for value in outcomes])
        cancelled = outcomes[0]
        source = await self.context.get(self.owner, cancelled.request_event_id)
        replay = await self.service.mutate(self.owner, uuid4(), "cancel",
            request.model_copy(update={"expected_revision": 999}), command_admission=command)
        self.check("original_target_and_base_pinned", replay.id == task.id and replay.revision == cancelled.revision
            and replay.fence == cancelled.fence and source.content["task_id"] == str(task.id)
            and source.content["base_revision"] == task.revision
            and source.content["area_id"] == str(task.knowledge_area_id))
        async with self.fixture.pool.acquire() as connection:
            counts = await connection.fetchrow("""select
                (select count(*) from jarvis_task_requests where owner_id=$1 and request_id=$3) as receipts,
                (select count(*) from jarvis_context_events where owner_id=$1
                    and source_namespace='voice_command' and source_event_id=$4) as sources,
                (select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2
                    and event_type='cancel_requested') as cancellations""",
                self.owner, task.id, command.operation_id, str(command.operation_id))
        self.check("one_cancel_receipt", dict(counts) == {"receipts": 1, "sources": 1, "cancellations": 1})
        resume = self.command("resume", arguments={"target": str(task.id)})
        resumed = await self.service.mutate(self.owner, task.id, "resume",
            TaskMutation(request_id=resume.operation_id, expected_revision=cancelled.revision), command_admission=resume)
        before = await self.owner_counts()
        historical = await self.service.mutate(self.owner, uuid4(), "cancel", request, command_admission=command)
        async with self.fixture.pool.acquire() as connection:
            accepted_revision = await connection.fetchval(
                "select accepted_revision from jarvis_task_requests where owner_id=$1 and request_id=$2",
                self.owner, command.operation_id)
        self.check("resume_keeps_original_cancel_historical", historical.id == resumed.id
            and historical.status == TaskStatus.QUEUED and historical.revision == resumed.revision
            and accepted_revision == cancelled.revision and await self.owner_counts() == before)

    async def test_closed_app_session_does_not_revoke_durable_command_source(self):
        self.plan("session_closed", "command_still_valid", "task_still_queued", "replay_without_live_session")
        task, command, request = await self.create()
        async with self.fixture.pool.acquire() as connection:
            status = await connection.fetchval(
                "update jarvis_sessions set status='closed' where owner_id=$1 and id=$2 returning status",
                self.owner, self.session.id)
        self.check("session_closed", status == "closed", boundary="durable_session_row_not_native_transport")
        async with self.repo.transaction(self.owner) as connection:
            current = await self.repo.load(connection, self.owner, task.id)
            source = await task_command_source(self.repo, connection, current)
            problem = await self.repo.context_problem(connection, current)
        self.check("command_still_valid", source.id == task.request_event_id and problem is None
            and not source.authorizes_request)
        self.check("task_still_queued", current.status == TaskStatus.QUEUED
            and current.revision == task.revision and current.current_attempt_id is None)
        before = await self.owner_counts()
        replay = await self.service.create(self.owner, request, command_admission=command, mode="background")
        self.check("replay_without_live_session", replay.id == task.id and await self.owner_counts() == before)

    async def test_source_removal_or_supersession_rejects_authority_and_replay_without_recreation(self):
        names = ("initially_valid", "source_invalidated", "authority_rejected", "replay_rejected", "no_recreation")
        self.plan(*(mode + "_" + name for mode in ("deleted", "superseded") for name in names))
        for mode in ("deleted", "superseded"):
            with self.subTest(mode=mode):
                task, command, request = await self.create()
                source = await self.context.get(self.owner, task.request_event_id)
                async with self.repo.transaction(self.owner) as connection:
                    valid = await task_command_source(self.repo, connection, task)
                self.check(mode + "_initially_valid", valid.id == source.id)
                if mode == "deleted":
                    async with self.repo.transaction(self.owner) as connection:
                        await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                            where owner_id=$1 and id=$2""", self.owner, source.payload_id)
                else:
                    # A canonical correction is real SQL, not a modified Python snapshot.
                    await self.context.append(self.owner, ContextIngress(
                        session_id=source.session_id, source_namespace=source.source_namespace,
                        source_event_id=source.source_event_id, receipt_id="fixture-invalidation:" + uuid4().hex,
                        kind="command_request", trust="derived", finality="revised", expected_revision=1,
                        payload=source.content))
                after_source = await self.context.get(self.owner, source.id)
                self.check(mode + "_source_invalidated", after_source.availability == "deleted"
                    if mode == "deleted" else not after_source.is_current)
                async with self.repo.transaction(self.owner) as connection:
                    with self.assertRaises(TaskError) as caught:
                        await task_command_source(self.repo, connection, task)
                    problem = await self.repo.context_problem(connection, task)
                self.check(mode + "_authority_rejected", caught.exception.code == "task_command_source_unavailable"
                    and problem == "task_command_source_unavailable", code=caught.exception.code)
                before = await self.owner_counts()
                with self.assertRaises(TaskError) as replay_error:
                    await self.service.create(self.owner, request, command_admission=command, mode="background")
                self.check(mode + "_replay_rejected", replay_error.exception.code == "task_command_source_unavailable")
                after_task = await self.repo.get(self.owner, task.id)
                self.check(mode + "_no_recreation", await self.owner_counts() == before
                    and after_task.revision == task.revision and after_task.request_event_id == source.id)

    async def test_foreign_owner_cannot_resolve_the_committed_command_source(self):
        self.plan("foreign_sql_lookup_rejected", "original_owner_unchanged")
        task, _, _ = await self.create()
        before = await self.owner_counts()
        async with self.repo.transaction(self.other) as connection:
            with self.assertRaises(TaskError) as caught:
                await task_command_source(self.repo, connection, task.model_copy(update={"owner_id": self.other}))
        self.check("foreign_sql_lookup_rejected", caught.exception.code == "task_command_source_unavailable")
        async with self.repo.transaction(self.owner) as connection:
            valid = await task_command_source(self.repo, connection, task)
        self.check("original_owner_unchanged", valid.owner_id == self.owner and await self.owner_counts() == before)
