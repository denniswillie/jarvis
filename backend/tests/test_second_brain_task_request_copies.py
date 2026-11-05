"""Exact request-copy lifecycle over manifested PostgreSQL, without model I/O."""
import os
from pathlib import Path
import unittest
from uuid import uuid4

import asyncpg

from app.runtime.feature_config import TaskLimits
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource, WorkerOutcome, digest
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain fixtures required",
)
class TaskRequestCopyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.owner, self.other = self.fixture.owner, self.fixture.other
        self.tasks = TaskRepository(self.fixture.pool, TaskLimits())
        self.sessions = {}
        async with self.fixture.pool.acquire() as connection:
            for owner in (self.owner, self.other):
                self.sessions[owner] = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner,
                )

    async def source(self, *, owner=None, text="Synthetic exact request"):
        owner = owner or self.owner
        async with self.tasks.transaction(owner) as connection:
            payload = await self.tasks.payload(connection, owner, {"text":text})
            return await connection.fetchrow("""insert into jarvis_context_events(
                owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-task-copy',$3,'user_request','final','user_input',$4)
                returning id,revision,payload_id,session_id""",
                owner, self.sessions[owner], str(uuid4()), payload)

    async def create(self, *, owner=None, source=None, voice=False, request=None):
        owner = owner or self.owner
        request = request or TaskCreate(
            request_id=uuid4(), title="Synthetic private title", requested_outcome="Synthetic private instructions",
            session_id=self.sessions[owner],
        )
        arguments = {}
        if source is not None:
            if voice:
                arguments = {"source":TaskVoiceSource(source["id"], source["revision"], source["session_id"]),
                             "mode":"background"}
            else:
                arguments = {"source_event_id":source["id"]}
        task = await self.tasks.create(owner, request, model="fixture-model", **arguments)
        return task, request

    async def request_payload(self, task):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchval(
                "select payload_id from jarvis_context_events where owner_id=$1 and id=$2 and revision=$3",
                task.owner_id, task.request_event_id, task.request_revision,
            )

    async def remove(self, payload, *, owner=None, state="deleted"):
        owner = owner or self.owner
        async with self.tasks.transaction(owner) as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability=$3,content=null where owner_id=$1 and id=$2",
                owner, payload, state,
            )

    async def parents(self, payload, *, owner=None):
        async with self.fixture.pool.acquire() as connection:
            return set(await connection.fetchval(
                "select array_agg(source_payload_id) from jarvis_context_payload_dependencies "
                "where owner_id=$1 and dependent_payload_id=$2", owner or self.owner, payload,
            ) or ())

    async def body(self, payload, *, owner=None):
        async with self.fixture.pool.acquire() as connection:
            return await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2",
                owner or self.owner, payload,
            )

    async def mutate(self, task, operation, message):
        return await self.tasks.mutate(task.owner_id, task.id, operation, TaskMutation(
            request_id=uuid4(), expected_revision=task.revision, message=message,
        ))

    async def test_api_explicit_and_voice_requests_bind_description_and_accepted_history(self):
        for route in ("api", "explicit", "voice"):
            with self.subTest(route=route):
                source = None if route == "api" else await self.source()
                task, _ = await self.create(source=source, voice=route == "voice")
                original = await self.request_payload(task)
                self.assertEqual(await self.parents(task.description_payload_id), {original})
                before = await self.tasks.events(self.owner, task.id)
                self.assertEqual(before[0]["payload_id"], task.description_payload_id)
                self.assertEqual(before[0]["content"]["title"], "Synthetic private title")
                await self.remove(original)
                current = await self.tasks.get(self.owner, task.id)
                self.assertIsNone(current.title)
                self.assertIsNone(current.requested_outcome)
                history = await self.tasks.events(self.owner, task.id)
                self.assertEqual(history[0]["id"], before[0]["id"])
                self.assertEqual(history[0]["event_type"], "accepted")
                self.assertIsNone(history[0]["content"])
                self.assertEqual(history[0]["availability"], "deleted")

    async def test_unavailable_and_expired_sources_remove_their_copies_too(self):
        for state in ("unavailable", "expired"):
            with self.subTest(state=state):
                task, _ = await self.create()
                await self.remove(await self.request_payload(task), state=state)
                current = await self.tasks.get(self.owner, task.id)
                self.assertIsNone(current.requested_outcome)
                self.assertIsNone((await self.body(task.description_payload_id))["content"])

    async def test_update_links_its_message_and_inherited_description_to_distinct_sources(self):
        original, _ = await self.create()
        updated = await self.mutate(original, "update", "Synthetic replacement instructions")
        request_payload = await self.request_payload(updated)
        self.assertEqual(await self.parents(updated.description_payload_id),
                         {request_payload, original.description_payload_id})
        event = (await self.tasks.events(self.owner, updated.id))[-1]
        self.assertEqual(event["event_type"], "update_requested")
        self.assertEqual(await self.parents(event["payload_id"]), {request_payload})
        await self.remove(request_payload)
        current = await self.tasks.get(self.owner, updated.id)
        self.assertIsNone(current.title)
        self.assertIsNone(current.requested_outcome)
        self.assertIsNone((await self.tasks.events(self.owner, updated.id))[-1]["content"])
        self.assertEqual((await self.body(original.description_payload_id))["availability"], "available")

    async def test_deleting_inherited_title_cascades_but_preserves_the_new_original_request(self):
        original, _ = await self.create()
        updated = await self.mutate(original, "update", "Synthetic independent new instructions")
        async with self.tasks.transaction(self.owner) as connection:
            mirror = await self.tasks.payload(connection, self.owner, {"text":"Synthetic derived copy"},
                                              source_payload_ids=(updated.description_payload_id,))
        await self.remove(await self.request_payload(original))
        self.assertIsNone((await self.body(updated.description_payload_id))["content"])
        self.assertIsNone((await self.body(mirror))["content"])
        self.assertEqual((await self.body(await self.request_payload(updated)))["availability"], "available")
        event = (await self.tasks.events(self.owner, updated.id))[-1]
        self.assertEqual(event["content"], {"message":"Synthetic independent new instructions"})
        repaired = await self.mutate(await self.tasks.get(self.owner, updated.id),
                                     "update", "Fresh instructions after forgetting the old title")
        self.assertEqual(repaired.title, "Updated task")
        self.assertEqual(repaired.requested_outcome, "Fresh instructions after forgetting the old title")
        self.assertEqual(await self.parents(repaired.description_payload_id), {await self.request_payload(repaired)})

    async def test_cancel_and_resume_messages_do_not_rebind_the_existing_description(self):
        original, _ = await self.create()
        cancelled = await self.mutate(original, "cancel", "Synthetic private cancellation reason")
        resumed = await self.mutate(cancelled, "resume", "Synthetic private resumption reason")
        self.assertEqual(resumed.description_payload_id, original.description_payload_id)
        self.assertEqual(await self.parents(resumed.description_payload_id), {await self.request_payload(original)})
        await self.remove(await self.request_payload(cancelled))
        events = await self.tasks.events(self.owner, original.id)
        self.assertIsNone(next(event for event in events if event["event_type"] == "cancel_requested")["content"])
        self.assertEqual(next(event for event in events if event["event_type"] == "resume_requested")["content"],
                         {"message":"Synthetic private resumption reason"})
        await self.remove(await self.request_payload(resumed))
        self.assertIsNone((await self.tasks.events(self.owner, original.id))[-1]["content"])
        self.assertEqual((await self.body(original.description_payload_id))["availability"], "available")

    async def test_voice_update_message_uses_the_exact_utterance(self):
        original, _ = await self.create()
        source = await self.source(text="Synthetic voice update")
        changed = await self.tasks.mutate(self.owner, original.id, "update", TaskMutation(
            request_id=uuid4(), expected_revision=original.revision, message="Synthetic voice-derived instructions",
        ), source=TaskVoiceSource(source["id"], source["revision"], source["session_id"]))
        event = (await self.tasks.events(self.owner, original.id))[-1]
        self.assertEqual(await self.parents(event["payload_id"]), {source["payload_id"]})
        await self.remove(source["payload_id"])
        self.assertIsNone((await self.tasks.get(self.owner, changed.id)).requested_outcome)
        self.assertIsNone((await self.tasks.events(self.owner, changed.id))[-1]["content"])

    async def test_resume_requires_fresh_instructions_after_original_removal(self):
        task, _ = await self.create()
        await self.remove(await self.request_payload(task))
        cancelled = await self.mutate(await self.tasks.get(self.owner, task.id), "cancel", "Stop the removed task")
        before = await self.tasks.events(self.owner, task.id)
        with self.assertRaises(TaskError) as raised:
            await self.mutate(cancelled, "resume", "Resume without reconstructing private instructions")
        self.assertEqual(raised.exception.code, "task_context_removed")
        self.assertEqual(await self.tasks.events(self.owner, task.id), before)
        repaired = await self.mutate(cancelled, "update", "Use these new instructions instead")
        self.assertEqual(repaired.status, TaskStatus.QUEUED)
        claim = await self.tasks.claim_next("fixture-repaired-task", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        self.assertEqual(claim.task.requested_outcome, "Use these new instructions instead")

    async def test_replay_cannot_recreate_a_removed_description(self):
        task, request = await self.create()
        await self.remove(await self.request_payload(task))
        before = await self.tasks.events(self.owner, task.id)
        replay, _ = await self.create(request=request)
        self.assertEqual(replay.id, task.id)
        self.assertIsNone(replay.requested_outcome)
        self.assertEqual(await self.tasks.events(self.owner, task.id), before)

    async def test_deletion_does_not_touch_another_task_or_owner(self):
        removed, _ = await self.create()
        other_task, _ = await self.create()
        foreign_task, _ = await self.create(owner=self.other)
        await self.remove(await self.request_payload(removed))
        self.assertEqual((await self.tasks.get(self.owner, other_task.id)).requested_outcome, other_task.requested_outcome)
        self.assertEqual((await self.tasks.get(self.other, foreign_task.id)).requested_outcome, foreign_task.requested_outcome)
        self.assertEqual((await self.body(foreign_task.description_payload_id, owner=self.other))["availability"], "available")

    async def test_removed_or_foreign_parent_refuses_late_copy_without_orphan_payload(self):
        original = await self.source()
        foreign = await self.source(owner=self.other)
        await self.remove(original["payload_id"])
        async with self.fixture.pool.acquire() as connection:
            before = await connection.fetchval("select count(*) from jarvis_context_payloads where owner_id=$1", self.owner)
        for source in (original, foreign):
            with self.subTest(source=source["id"]), self.assertRaises(asyncpg.CheckViolationError):
                async with self.tasks.transaction(self.owner) as connection:
                    await self.tasks.payload(connection, self.owner, {"text":"Must not survive"},
                                             source_payload_ids=(source["payload_id"],))
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_context_payloads where owner_id=$1", self.owner), before)
        self.assertEqual((await self.body(foreign["payload_id"], owner=self.other))["availability"], "available")

    async def test_source_bound_payload_requires_transaction_and_deduplicates_parents(self):
        source = await self.source()
        async with self.fixture.pool.acquire() as connection:
            with self.assertRaises(ValueError):
                await self.tasks.payload(connection, self.owner, {"text":"Not admitted outside a transaction"},
                                         source_payload_ids=(source["payload_id"],))
        async with self.tasks.transaction(self.owner) as connection:
            child = await self.tasks.payload(connection, self.owner, {"text":"Synthetic copy"},
                                            source_payload_ids=(source["payload_id"], source["payload_id"]))
        self.assertEqual(await self.parents(child), {source["payload_id"]})

    async def test_unknown_action_identity_and_submission_token_survive_request_removal(self):
        task, _ = await self.create()
        claim = await self.tasks.claim_next("fixture-request-copy", (task.backend_kind,), owner_id=self.owner)
        self.assertIsNotNone(claim)
        parameters = {"body":"Synthetic authorized action content"}
        token, operation = uuid4(), "synthetic-action-" + uuid4().hex
        async with self.tasks.transaction(self.owner) as connection:
            scope = await self.tasks.payload(connection, self.owner, parameters)
            grant = await connection.fetchval("""insert into jarvis_operation_grants(
                owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'notes.create',$2,$3,$4,1,$5,clock_timestamp()+interval '1 hour') returning id""",
                self.owner, operation, task.request_event_id, task.request_revision, scope)
            await connection.execute("insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)",
                                     self.owner, task.id, grant)
            action = await connection.fetchval("""insert into jarvis_task_actions(
                owner_id,task_id,task_revision,operation_key,grant_id,scope_revision,
                parameters_payload_id,parameters_digest,status,submission_token)
                values($1,$2,$3,$4,$5,1,$6,$7,'outcome_unknown',$8) returning id""",
                self.owner, task.id, task.revision, operation, grant, scope, digest(parameters), token)
        self.assertTrue(await self.tasks.finish(
            claim, WorkerOutcome(TaskStatus.PAUSED, {"fixture": "uncertain action remains"})))
        self.assertEqual((await self.tasks.get(self.owner, task.id)).status, TaskStatus.UNKNOWN)
        await self.remove(await self.request_payload(task))
        current = await self.tasks.get(self.owner, task.id)
        self.assertIsNone(current.requested_outcome)
        cancelled = await self.mutate(current, "cancel", "Stop without repeating the unknown action")
        self.assertEqual(cancelled.status, TaskStatus.UNKNOWN)
        async with self.fixture.pool.acquire() as connection:
            rows = await connection.fetch(
                "select id,operation_key,status,submission_token from jarvis_task_actions where owner_id=$1 and task_id=$2",
                self.owner, task.id,
            )
        self.assertEqual([tuple(row.values()) for row in rows], [(action, operation, "outcome_unknown", token)])
        with self.assertRaises(TaskError) as raised:
            await self.mutate(cancelled, "resume", "Do not silently repeat the operation")
        self.assertEqual(raised.exception.code, "task_reconciliation_required")
