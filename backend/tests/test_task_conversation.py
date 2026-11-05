import os
import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.conversation import INPUT, TaskConversation, bounded_content
from app.tasks.models import BudgetExceeded, ClaimLost, TaskError, TaskStatus, WorkerOutcome
import test_second_brain_tasks_integration as task_fixture


class WorkMessageContractTests(unittest.IsolatedAsyncioTestCase):
    def test_original_text_is_not_rewritten(self):
        self.assertEqual(bounded_content({"text": "  Actually, use EUR.\n"}),
                         {"text": "  Actually, use EUR.\n"})

    def test_input_is_bounded(self):
        with self.assertRaises(TaskError):
            bounded_content({"text": "x" * 65537})

    async def test_invalid_cursors_fail_before_storage(self):
        mailbox = TaskConversation(SimpleNamespace(pool=None))
        for value in (-1, True, "1"):
            with self.assertRaises(ValueError):
                await mailbox.read_inputs(None, after=value)
            with self.assertRaises(ValueError):
                await mailbox.finish(None, None, through_sequence=value)

    async def test_progress_requires_actual_source_roots(self):
        mailbox = TaskConversation(SimpleNamespace(pool=None))
        with self.assertRaises(ValueError):
            await mailbox.publish_progress(None, "Still working", source_payload_ids=())


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit second-brain database fixture configuration required")
class WorkConversationDatabaseTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the isolated-schema lifecycle, not the inherited task test suite.
    asyncSetUp = task_fixture.TaskDatabaseTests.asyncSetUp
    request = task_fixture.TaskDatabaseTests.request
    create = task_fixture.TaskDatabaseTests.create
    claim = task_fixture.TaskDatabaseTests.claim
    mutation = task_fixture.TaskDatabaseTests.mutation

    async def user_message(self, text, *, owner=None, finality="final"):
        return await ContextRepository(self.repository).append(owner or self.owner, ContextIngress(
            session_id=None, source_namespace="work-mailbox-fixture", source_event_id=str(uuid4()),
            receipt_id=str(uuid4()), expected_revision=0, kind="user_request", trust="user_input",
            finality=finality, payload={"text": text}))

    async def test_busy_followup_completion_race_idle_continuation_and_scope(self):
        mailbox = TaskConversation(self.repository)
        task = await self.create()
        claim = await self.claim()
        original = await self.repository.get(self.owner, task.id)
        first = await self.user_message("  Group by month instead.\n")
        sent = await mailbox.append_user(self.owner, task.id, first.id)
        replay = await mailbox.append_user(self.owner, task.id, first.id)
        self.assertEqual(sent.sequence, replay.sequence)
        current = await self.repository.get(self.owner, task.id)
        for field in ("revision", "fence", "current_attempt_id", "request_event_id", "description_payload_id"):
            self.assertEqual(getattr(current, field), getattr(original, field), field)
        self.assertEqual(current.status, TaskStatus.RUNNING)
        messages = await mailbox.read_inputs(claim)
        self.assertEqual([message.content["text"] for message in messages], ["  Group by month instead.\n"])

        foreign = await self.user_message("Foreign context", owner=self.other)
        with self.assertRaises(TaskError):
            await mailbox.append_user(self.owner, task.id, foreign.id)
        with self.assertRaises(TaskError):
            await mailbox.append_user(self.other, task.id, first.id)
        provisional = await self.user_message("Unfinished ordinary transcript", finality="provisional")
        with self.assertRaises(TaskError):
            await mailbox.append_user(self.owner, task.id, provisional.id)

        await mailbox.publish_progress(claim, "I am preparing the grouped summary.",
                                       source_payload_ids=(sent.payload_id,))
        late = await self.user_message("Also include a total.")
        late_message = await mailbox.append_user(self.owner, task.id, late.id)
        outcome = WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "First attempt result."})
        await mailbox.finish(claim, outcome, through_sequence=sent.sequence)
        queued = await self.repository.get(self.owner, task.id)
        self.assertEqual(queued.status, TaskStatus.QUEUED)
        self.assertIsNone(queued.current_attempt_id)
        self.assertIsNone(queued.result_payload_id)
        second = await self.claim()
        self.assertEqual(second.task.id, task.id)
        self.assertNotEqual(second.attempt_id, claim.attempt_id)
        self.assertEqual(len(await mailbox.read_inputs(second)), 2)
        await mailbox.finish(second, outcome, through_sequence=late_message.sequence)
        self.assertEqual((await self.repository.get(self.owner, task.id)).status, TaskStatus.SUCCEEDED)
        with self.assertRaises(ClaimLost):
            await mailbox.publish_progress(claim, "Stale writer", source_payload_ids=(sent.payload_id,))

        later = await self.user_message("Continue the same work with a chart.")
        again = await mailbox.append_user(self.owner, task.id, later.id)
        queued = await self.repository.get(self.owner, task.id)
        self.assertEqual(queued.status, TaskStatus.QUEUED)
        self.assertEqual(queued.request_event_id, task.request_event_id)
        third = await self.claim()
        self.assertEqual([item.sequence for item in await mailbox.read_inputs(third, after=late_message.sequence)],
                         [again.sequence])
        await mailbox.finish(third, outcome, through_sequence=again.sequence)
        exhausted = await self.user_message("Another turn cannot renew exhausted resources.")
        async with self.repository.transaction(self.owner) as connection:
            await connection.execute("update jarvis_background_tasks set attempt_count=$3 where owner_id=$1 and id=$2",
                                     self.owner, task.id, queued.budget.max_attempts)
        with self.assertRaises(BudgetExceeded):
            await mailbox.append_user(self.owner, task.id, exhausted.id)

        other_task = await self.create()
        cancelled = await self.mutation(other_task, "cancel")
        with self.assertRaises(TaskError):
            await mailbox.append_user(self.owner, cancelled.id, first.id)
        self.assertEqual([event["event_type"] for event in await self.repository.events(self.owner, cancelled.id)],
                         ["accepted", "cancel_requested"])
        inputs = [event for event in await self.repository.events(self.owner, task.id) if event["event_type"] == INPUT]
        self.assertEqual(len(inputs), 3)
