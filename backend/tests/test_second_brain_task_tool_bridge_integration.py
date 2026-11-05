"""Real task/action ledger plus Jarvis's existing note handlers in isolated schemas."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.notes.repository import NoteRepository
from legacy_note_tool_fixture import NoteToolModule
from app.runtime.feature_config import TaskLimits
from app.tasks.models import BudgetExceeded, ClaimLost, TaskCreate, TaskError, TaskMutation
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import GrantedToolAction, TaskToolBridge
from app.tools.invocation import current_tool_invocation
from evals.second_brain.source_fixture import source_fixture


class PausedNotes(NoteRepository):
    def __init__(self, pool):
        super().__init__(pool)
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def create_note(self, **arguments):
        note = await super().create_note(**arguments)
        self.entered.set()
        await self.release.wait()
        return note


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class TaskToolBridgeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(lease_seconds=300))
        self.parameters = {"title":"Synthetic robotics progress","body":"Both search-bound updates remain unfinished."}
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id",self.f.owner,
            )
            request_payload = await self.tasks.payload(connection,self.f.owner,{"text":"Create this exact synthetic note."})
            self.event = await connection.fetchval(
                "insert into jarvis_context_events(owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id) "
                "values($1,$2,'fixture-tool-bridge',$3,'user_request','final','user_input',$4) returning id",
                self.f.owner,self.session,str(uuid4()),request_payload,
            )
            scope = await self.tasks.payload(connection,self.f.owner,self.parameters)
            self.grant = await connection.fetchval(
                "insert into jarvis_operation_grants(owner_id,capability,operation_key,request_event_id,request_revision,"
                "scope_revision,scope_payload_id,expires_at) values($1,'notes.create',$2,$3,1,1,$4,"
                "clock_timestamp()+interval '1 hour') returning id",self.f.owner,str(uuid4()),self.event,scope,
            )
        self.task = await self.tasks.create(
            self.f.owner,TaskCreate(request_id=uuid4(),title="Synthetic authorized note",
                requested_outcome="Create the authorized note once.",session_id=self.session,
                knowledge_area_id=self.f.area,grant_ids=(self.grant,)),
            model="fixture-no-model",
        )
        self.claim = await self.tasks.claim_next("fixture-tool-worker",("frontier",))
        self.context = TaskRunner(self.tasks,{}).context(self.claim)
        self.notes = NoteRepository(self.f.pool)
        self.bridge = await self.make_bridge()
        self.reference = self.bridge.authorized_operations()[0]["operation_ref"]

    async def make_bridge(self, notes=None):
        return await TaskToolBridge.from_modules(
            self.context,(NoteToolModule(notes or self.notes),),read_tools=("search_notes",),
            capability_tools={"notes.create":"create_note"},
        )

    async def call(self, bridge=None, identity="native-one"):
        return await (bridge or self.bridge).invoke("create_note",{"operation_ref":self.reference},
                                                   provider_call_id=identity)

    async def count(self):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("select count(*) from jarvis_notes where owner_id=$1",self.f.owner)

    async def test_real_write_handler_is_bound_to_its_separate_authorizing_message(self):
        seen = []
        class RecordingNotes(NoteRepository):
            async def create_note(self, **arguments):
                seen.append(current_tool_invocation())
                return await super().create_note(**arguments)
        result = await self.call(await self.make_bridge(RecordingNotes(self.f.pool)))
        self.assertEqual(result["status"],"success")
        self.assertNotEqual(self.task.request_event_id,self.event)
        self.assertEqual(seen[0].request_event_id,self.event)
        self.assertEqual(seen[0].request_revision,1)
        self.assertEqual(seen[0].task.task_id,self.task.id)
        self.assertEqual(seen[0].task.action_id,UUID(result["task_operation"]["action_id"]))
        self.assertEqual(await self.count(),1)

    async def test_new_provider_id_replays_the_real_shared_handler_receipt_once(self):
        first = await self.call()
        second = await self.call(identity="new-native-id")
        self.assertEqual(first["status"],"success")
        self.assertEqual(first["note"]["id"],second["note"]["id"])
        self.assertTrue(second["task_operation"]["replayed"])
        self.assertEqual(await self.count(),1)
        self.assertEqual((await self.context.budgets.usage(self.f.owner,self.task.id))["steps"],1)
        read = await self.bridge.invoke("search_notes",{},provider_call_id="read-native")
        self.assertEqual(read["notes"][0]["id"],first["note"]["id"])

    async def test_concurrent_native_calls_cannot_duplicate_a_pending_effect(self):
        notes = PausedNotes(self.f.pool)
        bridge = await self.make_bridge(notes)
        first = asyncio.create_task(self.call(bridge))
        try:
            await asyncio.wait_for(notes.entered.wait(),10)
            second = await self.call(await self.make_bridge(),identity="concurrent-native")
            self.assertEqual(second["status"],"outcome_unknown")
            self.assertEqual(await self.count(),1)
            notes.release.set()
            completed = await first
            replay = await self.call(identity="later-native")
            self.assertEqual(replay["note"]["id"],completed["note"]["id"])
            self.assertEqual(await self.count(),1)
        finally:
            notes.release.set()
            await asyncio.gather(first,return_exceptions=True)

    async def test_cancellation_after_effect_does_not_authorize_a_second_write(self):
        notes = PausedNotes(self.f.pool)
        first = asyncio.create_task(self.call(await self.make_bridge(notes)))
        try:
            await asyncio.wait_for(notes.entered.wait(),10)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            replay = await self.call(await self.make_bridge(),identity="after-interrupt")
            self.assertEqual(replay["status"],"outcome_unknown")
            self.assertEqual(await self.count(),1)
        finally:
            notes.release.set()
            await asyncio.gather(first,return_exceptions=True)

    async def test_lost_receipt_persistence_keeps_existing_effect_uncertain(self):
        with patch.object(self.context.actions,"record_receipt",new=AsyncMock(side_effect=RuntimeError("synthetic loss"))):
            with self.assertRaises(RuntimeError):
                await self.call()
        replay = await self.call(await self.make_bridge(),identity="after-lost-receipt")
        self.assertEqual(replay["status"],"outcome_unknown")
        self.assertEqual(await self.count(),1)

    async def test_verified_reconciliation_is_reported_as_known_without_resubmission(self):
        notes = PausedNotes(self.f.pool)
        first = asyncio.create_task(self.call(await self.make_bridge(notes)))
        try:
            await asyncio.wait_for(notes.entered.wait(),10)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            async with self.f.pool.acquire() as connection:
                action = await connection.fetchval(
                    "select id from jarvis_task_actions where owner_id=$1 and task_id=$2",self.f.owner,self.task.id,
                )
                note = await connection.fetchval("select id from jarvis_notes where owner_id=$1",self.f.owner)
            await self.context.actions.reconcile(self.f.owner,self.task.id,action,outcome="accepted",
                proof={"note_id":str(note),"method":"independent fixture database read"})
            result = await self.call(await self.make_bridge(),identity="after-reconciliation")
            self.assertEqual(result["status"],"success")
            self.assertEqual(result["task_operation"]["reconciled_outcome"],"accepted")
            self.assertEqual(result["reconciliation"]["note_id"],str(note))
            self.assertNotIn("note",result)
            self.assertEqual(await self.count(),1)
        finally:
            notes.release.set()
            await asyncio.gather(first,return_exceptions=True)

    async def test_module_construction_uses_the_current_owner_not_another_tasks_identity(self):
        await self.call()
        async with self.f.pool.acquire() as connection:
            session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",self.f.other)
        other = await self.tasks.create(self.f.other,TaskCreate(
            request_id=uuid4(),title="Other owner",requested_outcome="Search only my own notes.",session_id=session,
        ),model="fixture-no-model")
        claim = await self.tasks.claim_next("foreign-owner-worker",("frontier",))
        self.assertEqual(claim.task.id,other.id)
        bridge = await TaskToolBridge.from_modules(
            TaskRunner(self.tasks,{}).context(claim),(NoteToolModule(self.notes),),read_tools=("search_notes",),
            capability_tools={"notes.create":"create_note"},
        )
        result = await bridge.invoke("search_notes",{},provider_call_id="other-owner-read")
        self.assertEqual(result["notes"],[])
        self.assertFalse(bridge.authorized_operations())
        denied = await self.call(bridge,identity="foreign-operation-ref")
        self.assertEqual(denied["status"],"failed")
        self.assertEqual(await self.count(),1)

    async def test_model_cannot_edit_granted_parameters_or_use_ungranted_delete(self):
        changed = await self.bridge.invoke("create_note",{"operation_ref":self.reference,"body":"different"},
                                           provider_call_id="changed")
        denied = await self.bridge.invoke("delete_note",{"operation_ref":self.reference},provider_call_id="delete")
        self.assertEqual(changed["status"],"failed")
        self.assertEqual(denied["status"],"failed")
        self.assertEqual(await self.count(),0)

    async def test_removed_receipt_is_not_recreated_by_replaying_the_action(self):
        first = await self.call()
        async with self.f.pool.acquire() as connection:
            payload = await connection.fetchval(
                "select receipt_payload_id from jarvis_task_actions where owner_id=$1 and id=$2",
                self.f.owner,UUID(first["task_operation"]["action_id"]),
            )
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                self.f.owner,payload,
            )
        with self.assertRaises(TaskError) as caught:
            await self.call(identity="after-removal")
        self.assertEqual(caught.exception.code,"task_operation_receipt_unavailable")
        self.assertEqual(await self.count(),1)

    async def test_task_cancellation_fences_future_tool_calls(self):
        await self.tasks.mutate(self.f.owner,self.task.id,"cancel",
                                TaskMutation(request_id=uuid4(),expected_revision=self.task.revision))
        with self.assertRaises(ClaimLost):
            await self.call()
        self.assertEqual(await self.count(),0)

    async def test_task_without_attached_grant_cannot_use_another_tasks_operation(self):
        other = await self.tasks.create(self.f.owner,TaskCreate(
            request_id=uuid4(),title="Read only",requested_outcome="Do not write.",session_id=self.session,
        ),model="fixture-no-model")
        claim = await self.tasks.claim_next("other-worker",("frontier",))
        self.assertEqual(claim.task.id,other.id)
        context = TaskRunner(self.tasks,{}).context(claim)
        copied = GrantedToolAction(self.grant,"notes.create","create_note",self.parameters,self.event,1)
        bridge = TaskToolBridge(context,self.bridge.registry,read_tools=(),actions=(copied,))
        with self.assertRaises(TaskError) as caught:
            await self.call(bridge)
        self.assertEqual(caught.exception.code,"task_tool_grant_not_attached")
        self.assertEqual(await self.count(),0)
