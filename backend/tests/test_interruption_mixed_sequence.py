"""Mixed speech-stop, goal-correction and task-cancel policy, offline only.

Real direct-session controls, registered task handlers, TaskService,
TaskRepository, claim fences and TaskRunner completion use existing controlled
transport/in-memory storage fixtures. Initial tasks/claims are fixture-seeded,
not scheduler or task-creation acceptance. No DB, model, native audio or external
effect runs. Persistence means retained fixture state, not SQL durability.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import ClaimLost, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.service import TaskService
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.voice_tools import BackgroundTaskToolModule
from app.tools.invocation import CanonicalRequestIdentity, bind_tool_invocation
from app.tools.module import ToolBuildContext
import test_realtime_auto_vad_request_evidence as voice_fixtures
from test_task_interruption_bursts import BurstStorageBoundary


class MixedSequenceStorage(BurstStorageBoundary):
    """Only the general voice-mutation queries missing from the read-only fake.

    Production code still decides authority, transitions, replay and fencing.
    Unsupported queries fail closed in the inherited storage fixture.
    """

    async def fetch(self, sql, *args):
        query = " ".join(sql.split()).lower()
        if "'task_' || left(replace(id::text" in query:
            return [{"id": row["id"]} for row in self.tasks.values()
                    if row["owner_id"] == args[0]
                    and TaskRepository.reference(row["id"]) == args[1]][:2]
        return await super().fetch(sql, *args)

    async def fetchrow(self, sql, *args):
        query = " ".join(sql.split()).lower()
        if query.startswith("select id,revision,sequence,session_id,payload_id from jarvis_context_events"):
            row = self.record(args[1])
            return row if row and (row["owner_id"], row["revision"], row["session_id"]) == (
                args[0], args[2], args[3]) else None
        return await super().fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        if "last_event_sequence=last_event_sequence+1" in sql:
            self.tasks[args[1]]["updated_at"] = datetime.now(timezone.utc)
        return await super().fetchval(sql, *args)

    async def execute(self, sql, *args):
        query = " ".join(sql.split()).lower()
        if query.startswith("update jarvis_background_tasks set revision=revision+1,fence=fence+1,status=$3"):
            task = self.tasks[args[1]]
            assert task["owner_id"] == args[0]
            task.update(revision=task["revision"] + 1, fence=task["fence"] + 1,
                status=args[2], pending_status=args[3], description_payload_id=args[4],
                request_event_id=args[5], request_revision=args[6], context_watermark=args[7],
                result_payload_id=None, reason=args[8], updated_at=datetime.now(timezone.utc))
            return
        if query.startswith("insert into jarvis_task_session_context"):
            assert self.tasks[args[1]]["owner_id"] == args[0]
            assert (args[0], args[2]) in self.sessions
            # Keep this test-only binding inside the inherited transaction's
            # copied task state so rollback does not retain a failed mutation.
            self.tasks[args[1]].setdefault("fixture_session_bindings", {})[args[2]] = args[3:5]
            return
        if query.startswith("update jarvis_task_actions set status='revoked'"):
            assert not self.grants  # These reasoning-only tasks own no actions.
            return
        return await super().execute(sql, *args)


class ControlledReasoningCompletion:
    model = "fixture-no-provider"

    def __init__(self):
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.calls = 0

    async def execute(self, context):
        self.calls += 1
        self.entered.set()
        await asyncio.wait_for(self.release.wait(), 2)
        return WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Unrelated reasoning completed."})


class InterruptionMixedSequenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.voice = voice_fixtures.NativeAutoVADRequestEvidenceTests()
        await self.voice.asyncSetUp()
        self.addAsyncCleanup(self.voice.asyncTearDown)
        self.owner = self.voice.session.context.owner_id
        self.session_id = self.voice.session.context.jarvis_session_id
        self.db = MixedSequenceStorage()
        self.db.sessions.add((self.owner, self.session_id))
        self.tasks = TaskRepository(self.db, TaskLimits())
        self.sources = ContextRepository(self.tasks)
        self.service = TaskService(self.tasks, {"frontier": SimpleNamespace(model="fixture-no-provider")})
        for tool in BackgroundTaskToolModule(self.service).build_tools(
                ToolBuildContext(None, self.owner, self.session_id)):
            self.voice.session.tool_registry.register(tool)
        self.jobs, self.releases = [], []
        self.addAsyncCleanup(self.join_jobs)

    async def join_jobs(self):
        for release in self.releases:
            release.set()
        for job in self.jobs:
            if not job.done():
                job.cancel()
        if self.jobs:
            await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 2)
        self.assertEqual(self.db.depth, 0)
        self.assertFalse(self.db.lock.locked())

    async def source(self, text):
        identity = uuid4().hex
        source = await self.sources.append(self.owner, ContextIngress(
            session_id=self.session_id, source_namespace="fixture-finalized-mixed-command",
            source_event_id=identity, receipt_id=identity, kind="user_request",
            trust="user_input", payload={"text": text}))
        self.assertTrue(source.authorizes_request)
        return source

    async def seed_running(self, label):
        source = await self.source("Reason about " + label + "; do not perform external actions.")
        # Same explicit initial-row/claim boundary as legacy_worker_context;
        # this is not a mock replacement for the mutation or finish handlers.
        async with self.tasks.transaction(self.owner) as connection:
            description = await self.tasks.payload(connection, self.owner, {
                "title": label, "requested_outcome": "Reason about " + label,
                "mode": "background"}, source_payload_ids=(source.payload_id,))
            identity = await connection.fetchval("insert into jarvis_background_tasks returning id",
                self.owner, self.session_id, None, "frontier", "fixture-no-provider",
                description, source.id, source.revision, source.sequence, self.tasks.limits.model_dump_json())
        task = await self.tasks.get(self.owner, identity)
        attempt_id = self.db.running(task)
        task = await self.tasks.get(self.owner, identity)
        return task, TaskRunner(self.tasks, {}).context(self.tasks._claim(task, self.db.attempts[attempt_id]))

    async def invoke(self, name, arguments, source=None):
        with bind_tool_invocation(provider_call_id="independent-provider-delivery-" + uuid4().hex,
                request_event_id=source.id if source else None,
                request_revision=source.revision if source else None,
                request_identity=CanonicalRequestIdentity.from_record(source) if source else None):
            return await self.voice.session.tool_registry.get(name).handler(arguments)

    @staticmethod
    def semantic_state(task):
        # Emitting a stale-attempt audit event legitimately updates updated_at.
        return task.model_dump(exclude={"updated_at"})

    def events(self, task_id, kind):
        return [event for event in self.db.task_events if event[1] == task_id and event[5] == kind]

    async def assert_retry_is_not_a_new_mutation(self, name, args, source, task_id):
        before = await self.tasks.get(self.owner, task_id)
        requests, events = len(self.db.requests), len(self.db.task_events)
        retry = await self.invoke(name, args, source)
        self.assertEqual(retry["status"], "success")
        self.assertEqual(retry["task"]["revision"], before.revision)
        self.assertEqual(await self.tasks.get(self.owner, task_id), before)
        self.assertEqual((len(self.db.requests), len(self.db.task_events)), (requests, events))

    async def scenario(self, rounds, callback_boundary):
        async with asyncio.timeout(8):
            primary, old_context = await self.seed_running("primary")
            unrelated, other_context = await self.seed_running("unrelated")
            bridge = await TaskToolBridge.from_modules(old_context, (), read_tools=(), capability_tools={})
            denied = await bridge.invoke("create_note", {}, provider_call_id="ungranted-effect")
            self.assertEqual(denied["error"]["code"], "task_tool_not_allowed")
            controlled = ControlledReasoningCompletion()
            self.releases.append(controlled.release)
            worker = asyncio.create_task(TaskRunner(self.tasks, {"frontier": controlled}, owner_id=self.owner)
                                         ._execute(other_context.claim))
            self.jobs.append(worker)
            await asyncio.wait_for(controlled.entered.wait(), 1)
            old_release = asyncio.Event()
            self.releases.append(old_release)

            async def delayed_old_callback():
                await asyncio.wait_for(old_release.wait(), 2)
                return await self.tasks.finish(old_context.claim,
                    WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Obsolete original goal answer."}))

            delayed = asyncio.create_task(delayed_old_callback())
            self.jobs.append(delayed)
            sources = []
            for index in range(rounds):
                await self.voice.start(f"mixed-input-{index}")
                response_id = f"mixed-response-{index}"
                await self.voice.commit_and_ack_response(f"mixed-input-{index}", response_id)
                before = {task.id: await self.tasks.get(self.owner, task.id) for task in (primary, unrelated)}
                requests = len(self.db.requests)
                await self.voice.session._handle_client_event({"type": "response.cancel",
                    "client_event_id": "speech-only-stop-" + uuid4().hex})
                self.assertTrue(self.voice.session._active_responses[response_id].interrupted)
                self.assertIsNone(self.voice.session._current_response_id)
                for identity, state in before.items():
                    self.assertEqual(await self.tasks.get(self.owner, identity), state)
                self.assertEqual(len(self.db.requests), requests)
                self.assertFalse(worker.done())
                self.assertFalse(delayed.done())

                message = f"Revised primary goal {index + 1}; do not perform external actions."
                source = await self.source(message)
                sources.append(source)
                args = {"task_ref": self.tasks.reference(primary.id),
                        "expected_revision": before[primary.id].revision, "message": message}
                reply = await self.invoke("update_background_task", args, source)
                self.assertEqual(reply["status"], "success")
                corrected = await self.tasks.get(self.owner, primary.id)
                self.assertEqual(corrected.requested_outcome, message)
                self.assertEqual(corrected.revision, primary.revision + index + 1)
                self.assertEqual((corrected.status, corrected.pending_status), (TaskStatus.PAUSED, "queued"))
                self.assertEqual(corrected.current_attempt_id, old_context.claim.attempt_id)
                self.assertIsNone(corrected.result_payload_id)
                self.assertEqual(reply["task"]["delivery_state"], "not_established_by_task_status")
                await self.assert_retry_is_not_a_new_mutation("update_background_task", args, source, primary.id)
                with self.assertRaises(ClaimLost):
                    await bridge.invoke("create_note", {}, provider_call_id="stale-effect-" + str(index))
                self.assertEqual(await self.tasks.get(self.owner, unrelated.id), before[unrelated.id])

            if callback_boundary == "after_correction":
                before_callback = self.semantic_state(await self.tasks.get(self.owner, primary.id))
                old_release.set()
                self.assertFalse(await delayed)
                self.assertEqual(self.semantic_state(await self.tasks.get(self.owner, primary.id)), before_callback)

            current = await self.tasks.get(self.owner, primary.id)
            cancel_source = await self.source("Cancel only the primary task; keep the unrelated task working.")
            sources.append(cancel_source)
            args = {"task_ref": self.tasks.reference(primary.id), "expected_revision": current.revision}
            reply = await self.invoke("cancel_background_task", args, cancel_source)
            self.assertEqual(reply["status"], "success")
            cancelled = await self.tasks.get(self.owner, primary.id)
            self.assertEqual((cancelled.status, cancelled.pending_status), (TaskStatus.CANCEL_REQUESTED, "cancelled"))
            self.assertEqual(cancelled.revision, primary.revision + rounds + 1)
            self.assertEqual(cancelled.requested_outcome, current.requested_outcome)
            self.assertIsNone(cancelled.result_payload_id)
            await self.assert_retry_is_not_a_new_mutation("cancel_background_task", args, cancel_source, primary.id)
            self.assertFalse(worker.done())
            old_release.set()
            self.assertFalse(await delayed)
            self.assertFalse(await self.tasks.finish(old_context.claim,
                WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Duplicate obsolete callback."})))
            with self.assertRaises(ClaimLost):
                await bridge.invoke("create_note", {}, provider_call_id="post-cancel-effect")

            # An independently finalized correction is not an implicit resume
            # while the cancelled attempt still requires reconciliation.
            refused = await self.invoke("update_background_task", {
                "task_ref": self.tasks.reference(primary.id), "expected_revision": cancelled.revision,
                "message": "A later correction must not silently resume cancelled work."},
                await self.source("Change the cancelled task before reconciliation."))
            self.assertEqual(refused["error"]["code"], "task_reconciliation_required")

            controlled.release.set()
            await asyncio.wait_for(worker, 1)
            finished = await self.tasks.get(self.owner, unrelated.id)
            self.assertEqual(finished.status, TaskStatus.SUCCEEDED)
            self.assertEqual(controlled.calls, 1)
            self.assertIsNone(finished.current_attempt_id)
            self.assertEqual(self.db.payloads[finished.result_payload_id]["content"],
                             {"answer": "Unrelated reasoning completed."})
            self.assertEqual(len(self.events(unrelated.id, "succeeded")), 1)
            self.assertFalse(await self.tasks.finish(other_context.claim,
                WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Duplicate unrelated callback."})))
            reconstructed = TaskRepository(self.db, self.tasks.limits)
            self.assertEqual(self.semantic_state(await reconstructed.get(self.owner, unrelated.id)),
                             self.semantic_state(finished))
            self.assertEqual(self.semantic_state(await reconstructed.get(self.owner, primary.id)),
                             self.semantic_state(cancelled))
            self.assertEqual(len(self.db.tasks), 2)
            self.assertEqual(len(self.db.requests), rounds + 1)
            self.assertEqual(len({source.id for source in sources}), rounds + 1)
            self.assertEqual(len({source.source_event_id for source in sources}), rounds + 1)
            self.assertEqual(len(self.events(primary.id, "update_requested")), rounds)
            self.assertEqual(len(self.events(primary.id, "cancel_requested")), 1)
            self.assertFalse(self.db.grants)

    async def test_unfinalized_correction_and_cancel_do_not_mutate_tasks(self):
        primary, _ = await self.seed_running("primary")
        unrelated, _ = await self.seed_running("unrelated")
        for name in ("update_background_task", "cancel_background_task"):
            reply = await self.invoke(name, {"task_ref": self.tasks.reference(primary.id),
                "expected_revision": primary.revision, "message": "Ambiguous stop without finalized authority."})
            self.assertEqual(reply["error"]["code"], "task_request_not_finalized")
        for task in (primary, unrelated):
            self.assertEqual(await self.tasks.get(self.owner, task.id), task)
        self.assertFalse(self.db.requests)


def mixed_case(rounds, boundary):
    async def test(self):
        await self.scenario(rounds, boundary)
    test.__name__ = f"test_stop_correct_cancel_{rounds}_independent_corrections_callback_{boundary}"
    return test


for _rounds in (1, 2, 3):
    for _boundary in ("after_correction", "after_cancel"):
        _test = mixed_case(_rounds, _boundary)
        setattr(InterruptionMixedSequenceTests, _test.__name__, _test)
