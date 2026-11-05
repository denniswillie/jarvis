"""Real SQL/action/tool boundaries; intentionally blocked without an explicit DB.

No substitute state machine and no model/native agent. Existing manifested
isolated schemas supply real transactions and independent cleanup audits.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.notes.repository import NoteRepository
from legacy_note_tool_fixture import NoteToolModule
from app.runtime.feature_config import TaskLimits
from app.tasks.actions import TaskActionRepository
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, Reconciliation, TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome, json_object
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.tool_bridge import TaskToolBridge
from app.tools.invocation import bind_tool_invocation
from evals.second_brain.source_fixture import source_fixture


class EffectBarrierNotes(NoteRepository):
    def __init__(self, pool, *, boundary):
        super().__init__(pool)
        self.boundary = boundary
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.committed_id = None
        self.handler_calls = 0

    async def hold(self):
        self.entered.set()
        await asyncio.wait_for(self.release.wait(), 15)

    async def create_note(self, **arguments):
        self.handler_calls += 1
        if self.boundary == "tool_dispatch":
            await self.hold()
        note = await super().create_note(**arguments)
        self.committed_id = note
        if self.boundary == "effect_committed":
            await self.hold()
        return note


class BeforeSubmissionActions(TaskActionRepository):
    """Pause before, never inside or instead of, the canonical submission owner."""
    def __init__(self, tasks, barrier):
        super().__init__(tasks)
        self.barrier = barrier

    async def begin_submission(self, claim, action_id):
        await self.barrier.hold()
        return await super().begin_submission(claim, action_id)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "BLOCKED: explicit isolated PostgreSQL and manifested cleanup required")
class TaskInterruptionEffectSQLTests(unittest.IsolatedAsyncioTestCase):
    @asynccontextmanager
    async def isolated_case(self, kind, count):
        # Each burst variant gets its own manifested schema and SAME-owner
        # primary/control tasks. A failed variant cannot leave queue entries
        # for a later scheduler claim, even when unittest continues subtests.
        async with source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]),
            test_name=self.id() + "/" + kind + "/burst-" + str(count),
        ) as fixture:
            self.fixture = fixture
            self.tasks = TaskRepository(self.fixture.pool, TaskLimits(lease_seconds=60))
            self.sources = ContextRepository(self.tasks)
            self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-no-provider")})
            self.runner = TaskRunner(self.tasks, {})
            self.jobs, self.gates = [], []
            try:
                async with self.fixture.pool.acquire() as connection:
                    self.session = await connection.fetchval(
                        "insert into jarvis_sessions(owner_id) values($1) returning id", self.fixture.owner)
                yield
            finally:
                # Join the controlled worker before pool/schema teardown. The
                # existing fixture then independently audits committed removal.
                await self.join_workers()

    async def join_workers(self):
        for gate in self.gates:
            gate.release.set()
        for job in self.jobs:
            if not job.done():
                job.cancel()
        if self.jobs:
            await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 10)
            self.assertTrue(all(job.done() for job in self.jobs))

    async def source(self, text):
        record = await self.sources.append(self.fixture.owner, ContextIngress(
            session_id=self.session, source_namespace="fixture-interruption-confirmed",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="user_request",
            trust="user_input", payload={"text": text}))
        self.assertTrue(record.authorizes_request)
        return record, TaskVoiceSource(record.id, record.revision, self.session)

    async def create(self, label, *, effect=False):
        event, source = await self.source("Explicit synthetic request: " + label)
        grants = ()
        parameters = {"title": "Synthetic interruption " + label, "body": "Exact approved fixture content."}
        if effect:
            async with self.tasks.transaction(self.fixture.owner) as connection:
                scope = await self.tasks.payload(connection, self.fixture.owner, parameters,
                                                source_payload_ids=(event.payload_id,))
                grant = await connection.fetchval("""insert into jarvis_operation_grants
                    (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                    values($1,'notes.create',$2,$3,$4,1,$5,clock_timestamp()+interval '1 hour') returning id""",
                    self.fixture.owner, uuid4().hex, event.id, event.revision, scope)
                grants = (grant,)
        return await self.service.create(self.fixture.owner, TaskCreate(request_id=uuid4(),
            title=label, requested_outcome="Perform only the explicitly approved fixture work.",
            session_id=self.session, knowledge_area_id=self.fixture.area, grant_ids=grants),
            source=source, mode="background")

    async def claim(self, task):
        claim = await self.tasks.claim_next("burst-worker-" + uuid4().hex, ("frontier",), owner_id=self.fixture.owner)
        self.assertIsNotNone(claim)
        self.assertEqual(claim.task.id, task.id)
        return self.runner.context(claim)

    async def bridge(self, context, notes):
        return await TaskToolBridge.from_modules(context, (NoteToolModule(notes),),
            read_tools=(), capability_tools={"notes.create": "create_note"})

    async def count(self):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchval("select count(*) from jarvis_notes where owner_id=$1", self.fixture.owner)

    async def action_state(self, task):
        async with self.fixture.pool.acquire() as connection:
            rows = await connection.fetch("""select a.*,p.content as receipt from jarvis_task_actions a
                left join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                where a.owner_id=$1 and a.task_id=$2""", self.fixture.owner, task.id)
        self.assertEqual(len(rows), 1)
        return dict(rows[0])

    async def assert_committed_effect(self, task, context, notes, token, before):
        action = await self.action_state(task)
        self.assertEqual(action["status"], "accepted")
        self.assertEqual(action["submission_token"], token)
        self.assertIsNotNone(token)
        self.assertIsNotNone(action["receipt_payload_id"])
        self.assertIsNotNone(action["receipt_digest"])
        self.assertEqual(action["task_revision"], context.claim.task_revision)
        receipt = json_object(action["receipt"])
        self.assertEqual(receipt["tool"], "create_note")
        self.assertEqual(receipt["result"]["status"], "success")
        self.assertEqual(receipt["result"]["note"]["id"], str(notes.committed_id.id))
        self.assertEqual(notes.handler_calls, 1)
        self.assertEqual(await self.count(), before + 1)
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_notes where owner_id=$1 and id=$2",
                self.fixture.owner, notes.committed_id.id), 1)
            for event_type in ("action_submitting", "action_accepted"):
                self.assertEqual(await connection.fetchval("""select count(*) from jarvis_task_events
                    where owner_id=$1 and task_id=$2 and event_type=$3""",
                    self.fixture.owner, task.id, event_type), 1)
        self.assertEqual((await context.budgets.usage(self.fixture.owner, task.id))["steps"], 1)
        return action

    async def burst(self, task, count, *, operation="cancel", message=""):
        event, source = await self.source("Explicit " + operation + " " + task.title)
        request = TaskMutation(request_id=uuid4(), expected_revision=task.revision, message=message)
        values = []
        for index in range(count):
            with bind_tool_invocation(provider_call_id="new-provider-id-" + str(index),
                                      request_event_id=event.id, request_revision=event.revision):
                values.append(await self.service.mutate(self.fixture.owner, task.id, operation, request, source=source))
        self.assertEqual({value.revision for value in values}, {task.revision + 1})
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_task_requests
                where owner_id=$1 and task_id=$2 and request_id=$3""",
                self.fixture.owner, task.id, request.request_id), 1)
        return values[-1]

    async def recover_joined(self, context, *, answer):
        # Caller has joined the controlled local worker and checked actual
        # effect receipt/state. Cancellation alone is never cleanup proof.
        claims = await asyncio.gather(*[self.tasks.claim_recovery(
            "recovery-" + str(index), owner_id=self.fixture.owner, backend_kinds=("frontier",)) for index in (1, 2)])
        owned = [claim for claim in claims if claim is not None]
        self.assertEqual(len(owned), 1)
        self.assertEqual(owned[0].attempt_id, context.claim.attempt_id)
        self.assertFalse(await self.tasks.finish(context.claim,
            WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "late obsolete callback"})))
        await self.tasks.resolve_recovery(owned[0], Reconciliation("terminal",
            WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": answer})))
        self.assertIsNone(await self.tasks.claim_recovery("third-recovery", owner_id=self.fixture.owner,
                                                        backend_kinds=("frontier",)))

    async def run_case(self, kind, count):
        async with asyncio.timeout(90):
            before = await self.count()
            task = await self.create(kind + str(count), effect=kind != "replace_running")
            context = await self.claim(task) if kind != "cancel_queued" else None
            unrelated = await self.create("unrelated-" + kind + str(count))
            # create() returns the admission snapshot before its final event
            # bookkeeping. Compare fresh committed state, including updated_at.
            unrelated = await self.tasks.get(self.fixture.owner, unrelated.id)
            self.assertEqual(unrelated.owner_id, task.owner_id)
            if kind == "cancel_queued":
                await self.burst(task, count)
                current = await self.tasks.get(self.fixture.owner, task.id)
                self.assertEqual((current.status, current.current_attempt_id), (TaskStatus.CANCELLED, None))
            elif kind == "replace_running":
                current = task
                for index in range(count):
                    current = await self.burst(current, 1, operation="update", message="Replacement scope " + str(index))
                self.assertEqual(current.revision, count + 1)
                self.assertEqual(current.status, TaskStatus.PAUSED)
                await self.recover_joined(context, answer="Old scope completed before revision; not the replacement answer.")
                current = await self.tasks.get(self.fixture.owner, task.id)
                self.assertEqual(current.status, TaskStatus.QUEUED)
                self.assertIsNone(current.result_payload_id)
                self.assertEqual(current.requested_outcome, "Replacement scope " + str(count - 1))
                # Settle the preserved control separately so the next actual
                # scheduler claim belongs to this corrected task.
                self.assertEqual(await self.tasks.get(self.fixture.owner, unrelated.id), unrelated)
                await self.burst(unrelated, 1)
                new_context = await self.claim(current)
                self.assertEqual(new_context.claim.task_revision, count + 1)
                self.assertTrue(await self.tasks.finish(new_context.claim,
                    WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": current.requested_outcome})))
                self.assertEqual(await self.count(), before)
                return
            elif kind == "cancel_after_result":
                bridge = await self.bridge(context, NoteRepository(self.fixture.pool))
                reference = bridge.authorized_operations()[0]["operation_ref"]
                result = await bridge.invoke("create_note", {"operation_ref": reference}, provider_call_id="first")
                self.assertEqual(result["status"], "success")
                self.assertTrue(await self.tasks.finish(context.claim,
                    WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "The authorized note was already saved."})))
                persisted = await self.tasks.get(self.fixture.owner, task.id)
                for index in range(count):
                    with self.assertRaises(TaskError) as caught:
                        await self.burst(persisted, 1)
                    self.assertEqual(caught.exception.code, "task_state_conflict")
                    self.assertEqual(await self.tasks.get(self.fixture.owner, task.id), persisted)
                self.assertEqual(await self.count(), before + 1)
            else:
                boundary = ("before_submission" if kind == "cancel_dispatch" else
                            "tool_dispatch" if kind == "cancel_post_submission" else "effect_committed")
                notes = EffectBarrierNotes(self.fixture.pool, boundary=boundary)
                self.gates.append(notes)
                if boundary == "before_submission":
                    context = replace(context, actions=BeforeSubmissionActions(self.tasks, notes))
                bridge = await self.bridge(context, notes)
                reference = bridge.authorized_operations()[0]["operation_ref"]
                job = asyncio.create_task(bridge.invoke("create_note", {"operation_ref": reference}, provider_call_id="first"))
                self.jobs.append(job)
                await asyncio.wait_for(notes.entered.wait(), 10)
                self.assertEqual(await self.count(), before + (boundary == "effect_committed"))
                dispatch = await self.action_state(task)
                if boundary == "before_submission":
                    self.assertEqual(dispatch["status"], "authorized")
                    self.assertIsNone(dispatch["submission_token"])
                    self.assertEqual(notes.handler_calls, 0)
                else:
                    self.assertEqual(dispatch["status"], "submitting")
                    self.assertIsNotNone(dispatch["submission_token"])
                    self.assertIsNone(dispatch["receipt_payload_id"])
                    self.assertEqual(notes.handler_calls, 1)
                if kind == "effect_retry":
                    for index in range(count):
                        replay = await bridge.invoke("create_note", {"operation_ref": reference},
                                                     provider_call_id="retry-pending-" + str(index))
                        self.assertEqual(replay["status"], "outcome_unknown")
                        self.assertEqual(await self.count(), before + 1)
                    notes.release.set()
                    result = await asyncio.wait_for(job, 10)
                    self.assertEqual(result["status"], "success")
                    for index in range(count):
                        replay = await bridge.invoke("create_note", {"operation_ref": reference},
                                                     provider_call_id="retry-settled-" + str(index))
                        self.assertEqual(replay["note"]["id"], result["note"]["id"])
                        self.assertTrue(replay["task_operation"]["replayed"])
                    self.assertTrue(await self.tasks.finish(context.claim,
                        WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Exactly one authorized note was saved."})))
                    await self.assert_committed_effect(task, context, notes, dispatch["submission_token"], before)
                else:
                    await self.burst(task, count)
                    notes.release.set()
                    with self.assertRaises(ClaimLost):
                        await asyncio.wait_for(job, 10)
                    action = await self.action_state(task)
                    if boundary == "before_submission":
                        self.assertEqual(await self.count(), before)
                        self.assertEqual(notes.handler_calls, 0)
                        self.assertIsNone(notes.committed_id)
                        self.assertEqual(action["status"], "revoked")
                        self.assertIsNone(action["submission_token"])
                        self.assertIsNone(action["receipt_payload_id"])
                        self.assertIsNone(action["receipt_digest"])
                        self.assertEqual((await context.budgets.usage(self.fixture.owner, task.id))["steps"], 0)
                        answer = "The joined worker never submitted or entered the effect handler."
                    else:
                        # Durable submission already happened, even if INSERT
                        # had not. Inspect proof rather than inventing rollback.
                        action = await self.assert_committed_effect(
                            task, context, notes, dispatch["submission_token"], before)
                        await context.actions.record_receipt(self.fixture.owner, task.id, action["id"],
                            dispatch["submission_token"], "outcome_unknown", {"error": "late uncertain callback"})
                        after = await self.assert_committed_effect(
                            task, context, notes, dispatch["submission_token"], before)
                        self.assertEqual(after["receipt_digest"], action["receipt_digest"])
                        answer = "The dispatched operation committed its authorized note; cancellation did not undo it."
                    await self.recover_joined(context, answer=answer)
                    current = await self.tasks.get(self.fixture.owner, task.id)
                    self.assertEqual((current.status, current.current_attempt_id, current.pending_status),
                                     (TaskStatus.CANCELLED, None, None))
                    self.assertIsNone(current.result_payload_id)
                    for index in range(count):
                        with self.assertRaises(ClaimLost):
                            await bridge.invoke("create_note", {"operation_ref": reference},
                                                provider_call_id="late-new-id-" + str(index))
                    self.assertEqual(await self.count(), before + (boundary != "before_submission"))
                    self.assertEqual(notes.handler_calls, int(boundary != "before_submission"))
                async with self.fixture.pool.acquire() as connection:
                    self.assertEqual(await connection.fetchval(
                        "select count(*) from jarvis_task_actions where owner_id=$1 and task_id=$2", self.fixture.owner, task.id), 1)
            self.assertEqual(await self.tasks.get(self.fixture.owner, unrelated.id), unrelated)
            await self.burst(unrelated, 1)  # Separate explicit fixture cleanup, after preservation proof.


def sql_case(kind):
    async def test(self):
        for count in (1, 2, 3):
            with self.subTest(boundary=kind, interruptions=count):
                async with self.isolated_case(kind, count):
                    await self.run_case(kind, count)
    test.__name__ = "test_" + kind + "_one_two_three"
    return test


for _kind in ("cancel_queued", "replace_running", "cancel_dispatch", "effect_retry",
              "cancel_effect_committed", "cancel_after_result", "cancel_post_submission"):
    _test = sql_case(_kind)
    setattr(TaskInterruptionEffectSQLTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
