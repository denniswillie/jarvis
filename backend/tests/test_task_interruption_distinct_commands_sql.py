"""Distinct finalized cancel commands plus actual unrelated task completion.

Opt-in isolated SQL only. These are controlled task/action boundaries, not
native interruptions, acoustic cuts, model execution, or semantic acceptance.
Every count is a new source event AND canonical request ID, not a transport
retry. Existing manifested fixtures own schema creation and independent cleanup.
"""

import asyncio
from dataclasses import replace
import os
import unittest
from uuid import uuid4

from legacy_note_tool_fixture import NoteToolModule
from app.tasks.models import ClaimLost, TaskError, TaskMutation, TaskStatus, WorkerOutcome, json_object
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tools.invocation import bind_tool_invocation
import test_task_interruption_effect_bursts_sql as effect_fixtures


class DistinctCommandBarrierNotes(effect_fixtures.EffectBarrierNotes):
    async def hold(self):
        self.entered.set()
        # Leave room for distinct source admissions and the unrelated SQL
        # completion; the whole case still has its own finite 90-second bound.
        await asyncio.wait_for(self.release.wait(), 75)


class ControlledCompletionWorker:
    """One real TaskRunner attempt, with only an authorized fixture SQL effect."""

    model = "fixture-no-provider"

    def __init__(self, task_id, notes):
        self.task_id, self.notes = task_id, notes
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls = 0
        self.result = None

    async def execute(self, context):
        if context.task.id != self.task_id:
            raise AssertionError("Controlled worker received the wrong task")
        self.calls += 1
        await context.checkpoint({"stage": "unrelated controlled worker entered"})
        self.entered.set()
        await asyncio.wait_for(self.release.wait(), 75)
        bridge = await TaskToolBridge.from_modules(
            context, (NoteToolModule(self.notes),), read_tools=(),
            capability_tools={"notes.create": "create_note"},
        )
        operations = bridge.authorized_operations()
        if len(operations) != 1:
            raise AssertionError("Expected exactly the unrelated task's authorized note")
        self.result = await bridge.invoke(
            "create_note", {"operation_ref": operations[0]["operation_ref"]},
            provider_call_id="unrelated-effect-" + uuid4().hex,
        )
        if self.result["status"] != "success":
            raise AssertionError("Unrelated task's controlled effect did not succeed")
        return WorkerOutcome(TaskStatus.SUCCEEDED, {
            "answer": "Unrelated controlled task actually completed.",
            "note_id": self.result["note"]["id"],
        })

    async def reconcile(self, context, *, cancel):
        raise AssertionError("This controlled unrelated attempt must finish without recovery")


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "BLOCKED: explicit isolated PostgreSQL and manifested cleanup required",
)
class TaskInterruptionDistinctCommandsSQLTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only fixture helpers, without inheriting/collecting the old cases.
    isolated_case = effect_fixtures.TaskInterruptionEffectSQLTests.isolated_case
    join_workers = effect_fixtures.TaskInterruptionEffectSQLTests.join_workers
    source = effect_fixtures.TaskInterruptionEffectSQLTests.source
    create = effect_fixtures.TaskInterruptionEffectSQLTests.create
    claim = effect_fixtures.TaskInterruptionEffectSQLTests.claim
    bridge = effect_fixtures.TaskInterruptionEffectSQLTests.bridge
    count = effect_fixtures.TaskInterruptionEffectSQLTests.count
    action_state = effect_fixtures.TaskInterruptionEffectSQLTests.action_state
    assert_committed_effect = effect_fixtures.TaskInterruptionEffectSQLTests.assert_committed_effect
    recover_joined = effect_fixtures.TaskInterruptionEffectSQLTests.recover_joined

    async def start_unrelated(self):
        task = await self.create("unrelated-actually-finishes", effect=True)
        context = await self.claim(task)
        notes = effect_fixtures.EffectBarrierNotes(self.fixture.pool, boundary="unblocked")
        worker = ControlledCompletionWorker(task.id, notes)
        runner = TaskRunner(self.tasks, {"frontier": worker}, owner_id=self.fixture.owner)
        self.gates.append(worker)
        # Real claim, heartbeat, provider interface and durable finish path;
        # not a task-service process or a synthetic call to repository.finish.
        job = asyncio.create_task(runner._execute(context.claim))
        self.jobs.append(job)
        await asyncio.wait_for(worker.entered.wait(), 10)
        self.assertFalse(job.done())
        self.assertEqual((await self.tasks.get(self.fixture.owner, task.id)).status, TaskStatus.RUNNING)
        return task, context, notes, worker, job

    async def finish_unrelated(self, control):
        task, context, notes, worker, job = control
        before = await self.count()
        worker.release.set()
        await asyncio.wait_for(job, 20)
        self.assertFalse(job.cancelled())
        self.assertEqual(worker.calls, 1)
        current = await self.tasks.get(self.fixture.owner, task.id)
        self.assertEqual(current.status, TaskStatus.SUCCEEDED)
        self.assertEqual(current.attempt_count, 1)
        self.assertIsNone(current.current_attempt_id)
        self.assertIsNone(current.pending_status)
        self.assertIsNotNone(current.result_payload_id)
        async with self.fixture.pool.acquire() as connection:
            result = json_object(await connection.fetchval(
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.fixture.owner, current.result_payload_id,
            ))
        self.assertEqual(result["answer"], "Unrelated controlled task actually completed.")
        self.assertEqual(result["note_id"], str(notes.committed_id.id))
        self.assertEqual(worker.result["note"]["id"], result["note_id"])
        action = await self.action_state(task)
        await self.assert_committed_effect(task, context, notes, action["submission_token"], before)
        return current, result

    async def assert_unrelated_persisted(self, control, settled):
        task, _context, notes, worker, job = control
        current, result = settled
        self.assertTrue(job.done())
        self.assertFalse(job.cancelled())
        self.assertEqual(worker.calls, 1)
        self.assertEqual(notes.handler_calls, 1)
        self.assertEqual(await self.tasks.get(self.fixture.owner, task.id), current)
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(json_object(await connection.fetchval(
                "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                self.fixture.owner, current.result_payload_id,
            )), result)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_notes where owner_id=$1 and id=$2",
                self.fixture.owner, notes.committed_id.id,
            ), 1)

    async def distinct_cancels(self, task, count, control, *, already_succeeded):
        identities = []
        settled = None
        initial = await self.tasks.get(self.fixture.owner, task.id)
        for index in range(count):
            current = await self.tasks.get(self.fixture.owner, task.id)
            event, source = await self.source(
                f"Independent finalized cancellation {index + 1} of fixture task {task.id}"
            )
            request = TaskMutation(request_id=uuid4(), expected_revision=current.revision)
            provider_id = "independent-cancel-" + uuid4().hex
            identities.append((event.id, request.request_id, provider_id))
            with bind_tool_invocation(provider_call_id=provider_id,
                                      request_event_id=event.id, request_revision=event.revision):
                if already_succeeded:
                    with self.assertRaises(TaskError) as caught:
                        await self.service.mutate(self.fixture.owner, task.id, "cancel", request, source=source)
                    self.assertEqual(caught.exception.code, "task_state_conflict")
                else:
                    updated = await self.service.mutate(
                        self.fixture.owner, task.id, "cancel", request, source=source,
                    )
                    self.assertEqual(updated.revision, current.revision + 1)
                    self.assertEqual(updated.fence, current.fence + 1)
                    self.assertEqual(updated.request_event_id, event.id)
                    self.assertEqual(updated.request_revision, event.revision)
                    self.assertEqual(updated.current_attempt_id, current.current_attempt_id)
                    self.assertEqual(updated.status,
                        TaskStatus.CANCEL_REQUESTED if current.current_attempt_id else TaskStatus.CANCELLED)
            if already_succeeded:
                self.assertEqual(await self.tasks.get(self.fixture.owner, task.id), initial)
            async with self.fixture.pool.acquire() as connection:
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2 and request_id=$3",
                    self.fixture.owner, task.id, request.request_id,
                ), 0 if already_succeeded else 1)
            if index == 0:
                # Complete real unrelated work after the first control, while
                # any primary effect barrier is still held. Later commands must
                # preserve this actual completion, not just an untouched queue row.
                settled = await self.finish_unrelated(control)
            await self.assert_unrelated_persisted(control, settled)
        for position in range(3):
            self.assertEqual(len({identity[position] for identity in identities}), count)
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2 and request_id=any($3::uuid[])",
                self.fixture.owner, task.id, [identity[1] for identity in identities],
            ), 0 if already_succeeded else count)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type='cancel_requested'",
                self.fixture.owner, task.id,
            ), 0 if already_succeeded else count)
        final = await self.tasks.get(self.fixture.owner, task.id)
        self.assertEqual(final.revision, initial.revision + (0 if already_succeeded else count))
        return settled

    async def run_distinct_case(self, boundary, count):
        async with asyncio.timeout(90):
            before = await self.count()
            control = await self.start_unrelated()
            task = await self.create("distinct-" + boundary, effect=boundary != "queued")
            self.assertEqual(control[0].owner_id, task.owner_id)
            context = await self.claim(task) if boundary != "queued" else None
            bridge = notes = job = dispatch = None
            if boundary != "queued":
                note_boundary = {
                    "before_submission": "before_submission",
                    "after_submission": "tool_dispatch",
                    "effect_committed": "effect_committed",
                    "after_result": "unblocked",
                }[boundary]
                notes = DistinctCommandBarrierNotes(self.fixture.pool, boundary=note_boundary)
                self.gates.append(notes)
                if boundary == "before_submission":
                    context = replace(context, actions=effect_fixtures.BeforeSubmissionActions(self.tasks, notes))
                bridge = await self.bridge(context, notes)
                reference = bridge.authorized_operations()[0]["operation_ref"]
                if boundary == "after_result":
                    result = await bridge.invoke("create_note", {"operation_ref": reference}, provider_call_id="primary-effect")
                    self.assertEqual(result["status"], "success")
                    self.assertTrue(await self.tasks.finish(context.claim, WorkerOutcome(
                        TaskStatus.SUCCEEDED, {"answer": "Primary result was already committed."},
                    )))
                else:
                    job = asyncio.create_task(bridge.invoke(
                        "create_note", {"operation_ref": reference}, provider_call_id="primary-effect",
                    ))
                    self.jobs.append(job)
                    await asyncio.wait_for(notes.entered.wait(), 10)
                dispatch = await self.action_state(task)
                if boundary == "before_submission":
                    self.assertEqual(dispatch["status"], "authorized")
                    self.assertIsNone(dispatch["submission_token"])
                    self.assertEqual(notes.handler_calls, 0)
                elif boundary != "after_result":
                    self.assertEqual(dispatch["status"], "submitting")
                    self.assertIsNotNone(dispatch["submission_token"])
                    self.assertIsNone(dispatch["receipt_payload_id"])
                    self.assertEqual(notes.handler_calls, 1)
            self.assertEqual(await self.count(), before + int(boundary in {"effect_committed", "after_result"}))
            primary_before = await self.tasks.get(self.fixture.owner, task.id)
            settled = await self.distinct_cancels(
                task, count, control, already_succeeded=boundary == "after_result",
            )
            if job is not None:
                notes.release.set()
                with self.assertRaises(ClaimLost):
                    await asyncio.wait_for(job, 10)
                if boundary == "before_submission":
                    action = await self.action_state(task)
                    self.assertEqual(action["status"], "revoked")
                    for key in ("submission_token", "receipt_payload_id", "receipt_digest"):
                        self.assertIsNone(action[key])
                    self.assertEqual(notes.handler_calls, 0)
                    self.assertIsNone(notes.committed_id)
                    self.assertEqual((await context.budgets.usage(self.fixture.owner, task.id))["steps"], 0)
                else:
                    accepted = await self.assert_committed_effect(
                        task, context, notes, dispatch["submission_token"], before + 1,
                    )
                    await context.actions.record_receipt(self.fixture.owner, task.id, accepted["id"],
                        dispatch["submission_token"], "outcome_unknown", {"error": "late uncertain callback"})
                    self.assertEqual((await self.action_state(task))["receipt_digest"], accepted["receipt_digest"])
                await self.recover_joined(context, answer="Controlled primary worker joined; inspect the durable effect receipt.")
                for index in range(count):
                    with self.assertRaises(ClaimLost):
                        await bridge.invoke("create_note", {"operation_ref": reference},
                                            provider_call_id="late-independent-" + str(index))
            if boundary == "after_result":
                await self.assert_committed_effect(task, context, notes, dispatch["submission_token"], before + 1)
                self.assertFalse(await self.tasks.finish(context.claim,
                    WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "obsolete callback must not replace result"})))
                current = await self.tasks.get(self.fixture.owner, task.id)
                # A rejected stale finish emits an audit event, which advances
                # updated_at. No other task field or committed effect may change.
                self.assertEqual(current.model_dump(exclude={"updated_at"}),
                                 primary_before.model_dump(exclude={"updated_at"}))
                self.assertGreaterEqual(current.updated_at, primary_before.updated_at)
                self.assertEqual(await self.action_state(task), dispatch)
                async with self.fixture.pool.acquire() as connection:
                    self.assertEqual(json_object(await connection.fetchval(
                        "select content from jarvis_context_payloads where owner_id=$1 and id=$2",
                        self.fixture.owner, current.result_payload_id,
                    )), {"answer": "Primary result was already committed."})
                    self.assertEqual(await connection.fetchval(
                        "select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type='stale_attempt_outcome'",
                        self.fixture.owner, task.id,
                    ), 1)
            else:
                current = await self.tasks.get(self.fixture.owner, task.id)
                self.assertEqual((current.status, current.current_attempt_id, current.pending_status),
                                 (TaskStatus.CANCELLED, None, None))
                self.assertIsNone(current.result_payload_id)
                self.assertEqual(current.revision, primary_before.revision + count)
            self.assertEqual(await self.count(), before + 1 + int(boundary in {
                "after_submission", "effect_committed", "after_result",
            }))
            async with self.fixture.pool.acquire() as connection:
                self.assertEqual(await connection.fetchval(
                    "select count(*) from jarvis_task_actions where owner_id=$1 and task_id=$2",
                    self.fixture.owner, task.id,
                ), 0 if boundary == "queued" else 1)
            await self.assert_unrelated_persisted(control, settled)


def distinct_sql_case(boundary, count):
    async def test(self):
        async with self.isolated_case("distinct-" + boundary, count):
            await self.run_distinct_case(boundary, count)
    test.__name__ = f"test_distinct_{boundary}_{count}"
    return test


for _boundary in ("queued", "before_submission", "after_submission", "effect_committed", "after_result"):
    for _count in (1, 2, 3):
        _test = distinct_sql_case(_boundary, _count)
        setattr(TaskInterruptionDistinctCommandsSQLTests, _test.__name__, _test)


if __name__ == "__main__":
    unittest.main()
