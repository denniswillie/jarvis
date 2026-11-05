"""Task bridge contracts; provider/model execution is not simulated acceptance."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.models import ClaimLost
from app.tasks.tool_bridge import GrantedToolAction, TaskToolBridge
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation, current_tool_invocation
from app.tools.registry import JarvisToolRegistry, ToolDefinition


def worker():
    task = SimpleNamespace(id=uuid4(), owner_id="fixture-owner", session_id=uuid4(),
                           request_event_id=uuid4(), request_revision=3)
    return SimpleNamespace(
        task=task, claim=SimpleNamespace(task=task, task_revision=2, attempt_id=uuid4()),
        tasks=object(), actions=SimpleNamespace(prepare=AsyncMock()),
        budgets=SimpleNamespace(submit=AsyncMock()), reserve=AsyncMock(return_value={"id":uuid4()}),
        settle=AsyncMock(),
    )


class TaskInvocationTests(unittest.TestCase):
    def test_typed_task_context_restores_outer_invocation(self):
        identity = TaskToolInvocation(uuid4(), 2, uuid4())
        before = current_tool_invocation()
        with bind_tool_invocation(provider_call_id="native", request_event_id=uuid4(), request_revision=1, task=identity):
            self.assertIs(current_tool_invocation().task, identity)
            with bind_tool_invocation(provider_call_id="foreground"):
                self.assertIsNone(current_tool_invocation().task)
            self.assertIs(current_tool_invocation().task, identity)
        self.assertIs(current_tool_invocation(), before)

    def test_untrusted_or_partial_task_identities_are_rejected(self):
        for kwargs in ({"task_id":"not-a-uuid"}, {"task_revision":True}, {"task_revision":0},
                       {"action_id":uuid4()}, {"operation_key":"orphan"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TaskToolInvocation(**(dict(task_id=uuid4(), task_revision=1, attempt_id=uuid4()) | kwargs))
        with self.assertRaises(ValueError):
            with bind_tool_invocation(provider_call_id="fixture", task={"owner_id":"other"}):
                pass


class TaskToolBridgeTests(unittest.IsolatedAsyncioTestCase):
    def bridge(self, *, read=None):
        self.context = worker()
        self.registry = JarvisToolRegistry()
        self.reader = read or AsyncMock(return_value={"status":"success","text":"literal fixture"})
        self.writer = AsyncMock(return_value={"status":"success"})
        for name, handler, mutates in (
            ("read_fixture",self.reader,False),("write_fixture",self.writer,True),("ungranted",self.writer,True),
        ):
            self.registry.register(ToolDefinition(name,"Fixture",{"type":"object"},handler,mutates))
        self.action = GrantedToolAction(uuid4(),"fixture.write","write_fixture",{"text":"exact granted body"},uuid4(),4)
        bridge = TaskToolBridge(self.context,self.registry,read_tools=("read_fixture",),actions=(self.action,))
        bridge._check = AsyncMock()
        return bridge

    async def test_catalog_hides_ungranted_tools_and_write_parameters_are_not_model_editable(self):
        bridge = self.bridge()
        schemas = {entry["name"]:entry for entry in bridge.catalog()}
        self.assertEqual(set(schemas), {"read_fixture","write_fixture"})
        self.assertEqual(set(schemas["write_fixture"]["parameters"]["properties"]), {"operation_ref"})
        self.assertEqual(schemas["write_fixture"]["parameters"]["properties"]["operation_ref"]["enum"], [self.action.reference])
        operations = bridge.authorized_operations()
        operations[0]["parameters"]["text"] = "changed catalog copy"
        self.assertEqual(bridge.authorized_operations()[0]["parameters"]["text"], "exact granted body")

    async def test_read_only_registration_cannot_smuggle_a_mutation(self):
        self.bridge()
        with self.assertRaises(ValueError):
            TaskToolBridge(self.context,self.registry,read_tools=("write_fixture",),actions=())

    async def test_wrong_reference_extra_parameters_and_ungranted_tool_never_prepare_action(self):
        bridge = self.bridge()
        for name, args in (
            ("write_fixture", {"operation_ref":"made-up"}),
            ("write_fixture", {"operation_ref":self.action.reference,"text":"new body"}),
            ("ungranted", {"operation_ref":self.action.reference}),
        ):
            with self.subTest(name=name,args=args):
                result = await bridge.invoke(name,args,provider_call_id="native")
                self.assertEqual(result["status"],"failed")
        self.context.actions.prepare.assert_not_awaited()
        self.writer.assert_not_awaited()

    async def test_read_handler_receives_trusted_request_and_task_not_argument_identity(self):
        seen = []
        async def read(arguments):
            seen.append(current_tool_invocation())
            return {"status":"success","content":"precise \u03bb text"}
        bridge = self.bridge(read=read)
        result = await bridge.invoke("read_fixture",{"task_id":"not authority"},provider_call_id="native-call")
        self.assertEqual(json.loads(json.dumps(result))["content"],"precise \u03bb text")
        self.assertEqual(seen[0].request_event_id,self.context.task.request_event_id)
        self.assertEqual(seen[0].task.task_id,self.context.task.id)
        self.assertEqual(seen[0].task.attempt_id,self.context.claim.attempt_id)
        self.context.budgets.submit.assert_awaited_once()
        self.context.settle.assert_awaited_once()
        self.assertIsNone(current_tool_invocation().task)

    async def test_effect_context_uses_the_grant_source_not_the_task_start_message(self):
        seen = []
        async def write(arguments):
            seen.append(current_tool_invocation())
            return {"status":"success"}
        bridge = self.bridge()
        tool = ToolDefinition("write_fixture","Fixture",{"type":"object"},write,True)
        await bridge._handler(tool,self.action.parameters,"native-write",action=self.action,
                              action_id=uuid4(),operation_key="fixture-stable-operation")
        self.assertEqual(seen[0].request_event_id,self.action.request_event_id)
        self.assertEqual(seen[0].request_revision,4)
        self.assertNotEqual(seen[0].request_event_id,self.context.task.request_event_id)
        self.assertEqual(seen[0].task.task_id,self.context.task.id)

    async def test_stale_claim_propagates_instead_of_becoming_a_recoverable_model_error(self):
        bridge = self.bridge(read=AsyncMock(side_effect=ClaimLost()))
        with self.assertRaises(ClaimLost):
            await bridge.invoke("read_fixture",{},provider_call_id="native")
        self.context.settle.assert_awaited_once()

    async def test_read_failure_is_not_a_successful_empty_result(self):
        bridge = self.bridge(read=AsyncMock(side_effect=RuntimeError("PRIVATE EXCEPTION")))
        result = await bridge.invoke("read_fixture",{},provider_call_id="native")
        self.assertEqual(result["status"],"failed")
        self.assertEqual(result["error"]["code"],"task_tool_failed")
        self.assertNotIn("PRIVATE",str(result))

    async def test_parallel_coroutines_do_not_share_task_invocation_identity(self):
        observed = []
        ready = asyncio.Event()
        async def inspect(arguments):
            initial = current_tool_invocation()
            ready.set()
            await asyncio.sleep(0)
            observed.append((initial.task.task_id,current_tool_invocation().task.task_id))
            return {"status":"success"}
        first = self.bridge(read=inspect)
        first_id = first.context.task.id
        second = self.bridge(read=inspect)
        await asyncio.gather(
            first.invoke("read_fixture",{},provider_call_id="first"),
            second.invoke("read_fixture",{},provider_call_id="second"),
        )
        self.assertEqual({a for a,b in observed},{first_id,second.context.task.id})
        self.assertTrue(all(a==b for a,b in observed))
