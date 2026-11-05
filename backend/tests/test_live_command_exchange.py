"""Offline exchange/observer checks; these do not prove actual SQL or IPC latency."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.live_state.models import SessionScope
from app.live_state.task_events import TaskEventObserver
from app.tasks.live_command_exchange import (
    LiveCommandExchange, LiveCommandReceipt, REQUEST, STARTED, REPLIED,
)
from app.tasks.models import ClaimLost, TaskError
from app.voice_routes.contracts import RouteIdentity


class Repository:
    def __init__(self, task):
        self.task, self.rows, self.roots = task, [], {}
        self.mutex = asyncio.Lock()
        self.pool = SimpleNamespace(acquire=self.acquire)
        self.context_problem = AsyncMock(return_value=None)

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self, owner):
        assert owner == self.task.owner_id
        async with self.mutex:
            yield self

    async def load(self, connection, owner, task_id):
        assert owner == self.task.owner_id and task_id == self.task.id
        return self.task

    async def lock_claim(self, connection, claim):
        if (claim.task.owner_id != self.task.owner_id or claim.task.id != self.task.id
                or claim.attempt_id != self.task.current_attempt_id or claim.fence != self.task.fence
                or claim.task_revision != self.task.revision):
            raise ClaimLost()
        return self.task, {}

    async def fetchrow(self, query, *args):
        for row in self.rows:
            if row["owner_id"] != args[0]:
                continue
            if "e.id=$2" in query:
                if row["id"] == args[1]:
                    return row
            elif (row["task_id"], row["event_type"], row["content"]["call_key"]) == tuple(args[1:]):
                return row
        return None

    async def fetch(self, query, *args):
        assert "'work_input','live_command_replied'" in query
        return []

    async def emit(self, connection, task, kind, *, payload, attempt_id, source_payload_ids):
        assert self.mutex.locked()
        row = {"id": uuid4(), "payload_id": uuid4(), "owner_id": task.owner_id,
               "task_id": task.id, "sequence": len(self.rows) + 1,
               "task_revision": task.revision, "attempt_id": attempt_id,
               "event_type": kind, "availability": "available", "content": payload}
        self.rows.append(row)
        self.roots[row["id"]] = source_payload_ids


class LiveCommandExchangeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.route = RouteIdentity(scope=SessionScope(owner_id="synthetic-command-owner",
            session_id=uuid4()), route_id=uuid4(), route_generation=1)
        self.task = SimpleNamespace(id=uuid4(), owner_id=self.route.scope.owner_id,
            session_id=self.route.scope.session_id, revision=1, fence=2, current_attempt_id=uuid4())
        self.claim = SimpleNamespace(task=self.task, task_revision=1, fence=2,
                                     attempt_id=self.task.current_attempt_id)
        self.context = SimpleNamespace(task=self.task, claim=self.claim)
        self.repository = Repository(self.task)
        self.exchange = LiveCommandExchange(self.repository)
        self.source = SimpleNamespace(payload_id=uuid4(), content={"parameters": {
            "route": self.route.model_dump(mode="json"),
            "command_catalog": [{"name": "recording"}]}})
        self.source_check = AsyncMock(return_value=self.source)
        patcher = patch("app.tasks.live_command_exchange.live_command_source", self.source_check)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.execute = AsyncMock(return_value=LiveCommandReceipt({"status": "success",
            "outcome": "accepted", "effect_committed": False, "operation_id": str(uuid4())}))

    async def request(self, arguments=None, call="actual-native-call"):
        return await self.exchange.request(self.context, "recording", arguments or {"action": "start"},
                                           provider_call_id=call)

    async def dispatch(self, ticket, **options):
        return await self.exchange.dispatch(ticket.event_id, route=options.get("route", self.route),
            is_current=options.get("is_current", lambda: True), execute=options.get("execute", self.execute))

    async def test_request_replay_preserves_actual_tool_identity_and_immutable_arguments(self):
        args = {"action": "start"}
        first = await self.request(args)
        args["action"] = "stop"
        again = await self.request()
        self.assertEqual(first, again)
        self.assertEqual(len(self.repository.rows), 1)
        self.assertEqual(self.repository.rows[0]["content"]["arguments"], {"action": "start"})
        self.assertEqual(self.repository.rows[0]["content"]["provider_call_id"], "actual-native-call")
        self.assertEqual(self.repository.roots[first.event_id], (self.source.payload_id,))
        with self.assertRaises(TaskError) as caught:
            await self.request({"action": "stop"})
        self.assertEqual(caught.exception.code, "live_command_call_conflict")

    async def test_unoffered_tool_profile_and_invalid_identity_are_rejected(self):
        with self.assertRaises(TaskError):
            await self.exchange.request(self.context, "arbitrary_shell", {}, provider_call_id="call")
        with self.assertRaises(TaskError):
            await self.request(call=" ")
        self.source_check.return_value = None
        with self.assertRaises(TaskError):
            await self.request()
        self.assertEqual(self.repository.rows, [])

    async def test_dispatch_and_reply_use_one_effect_and_no_owner_lock_during_execution(self):
        ticket = await self.request()
        async def execute(name, arguments, **options):
            self.assertFalse(self.repository.mutex.locked())
            self.assertEqual(name, "recording")
            self.assertEqual(options["provider_call_id"], "actual-native-call")
            self.assertEqual(options["backend_session_key"], f"hermes:{self.task.id}:{self.claim.attempt_id}")
            self.assertTrue(options["admit_if"]())
            return LiveCommandReceipt({"status": "success", "outcome": "accepted"}, (self.source.payload_id,))
        self.execute.side_effect = execute
        result = await self.dispatch(ticket)
        self.assertEqual(await self.dispatch(ticket), result)
        self.assertEqual(await self.exchange.wait(self.context, ticket), result)
        self.execute.assert_awaited_once()
        self.assertEqual([r["event_type"] for r in self.repository.rows], [REQUEST, STARTED, REPLIED])
        reply = self.repository.rows[-1]
        self.assertIn(self.source.payload_id, self.repository.roots[reply["id"]])

    async def test_concurrent_dispatch_never_repeats_a_pending_effect(self):
        ticket = await self.request()
        entered, release = asyncio.Event(), asyncio.Event()
        async def execute(*args, **kwargs):
            entered.set()
            await release.wait()
            return LiveCommandReceipt({"status": "success"})
        self.execute.side_effect = execute
        work = asyncio.create_task(self.dispatch(ticket))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertIsNone(await self.dispatch(ticket))
        finally:
            release.set()
            await work
        self.execute.assert_awaited_once()

    async def test_foreign_or_closed_route_and_stale_claim_do_not_execute(self):
        ticket = await self.request()
        other = self.route.model_copy(update={"route_id": uuid4()})
        self.assertIsNone(await self.dispatch(ticket, route=other))
        self.assertIsNone(await self.dispatch(ticket, is_current=lambda: False))
        self.task.fence += 1
        with self.assertRaises(ClaimLost):
            await self.dispatch(ticket)
        self.execute.assert_not_awaited()
        self.assertEqual(len(self.repository.rows), 1)

    async def test_route_closing_after_acceptance_is_rejected_without_effect(self):
        ticket = await self.request()
        states = iter((True, False))
        result = await self.dispatch(ticket, is_current=lambda: next(states))
        self.assertEqual(result["outcome"], "rejected")
        self.assertIs(result["effect_committed"], False)
        self.execute.assert_not_awaited()

    async def test_failure_after_dispatch_is_unknown_not_permission_to_retry(self):
        ticket = await self.request()
        self.execute.side_effect = RuntimeError("synthetic unknown effect")
        result = await self.dispatch(ticket)
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertIsNone(result["effect_committed"])
        self.assertEqual(await self.dispatch(ticket), result)
        self.execute.assert_awaited_once()

    async def test_cancelled_dispatch_retains_unknown_marker_and_does_not_redispatch(self):
        ticket = await self.request()
        self.execute.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.dispatch(ticket)
        self.assertIsNone(await self.dispatch(ticket))
        self.assertIsNone(await self.exchange.response(ticket))
        self.execute.assert_awaited_once()

    async def test_observation_timeout_keeps_request_and_never_claims_no_effect(self):
        ticket = await self.request()
        with self.assertRaises(TaskError) as caught:
            await self.exchange.wait(self.context, ticket, timeout=.02, poll_seconds=.01)
        self.assertEqual(caught.exception.code, "live_command_reply_unconfirmed")
        self.assertEqual(await self.request(), ticket)
        self.assertEqual(len(self.repository.rows), 1)

    async def test_withdrawn_source_or_changed_canonical_route_does_not_execute(self):
        ticket = await self.request()
        self.source_check.side_effect = TaskError("source_withdrawn", "Synthetic source removed")
        with self.assertRaises(TaskError):
            await self.dispatch(ticket)
        self.source_check.side_effect = None
        self.source.content["parameters"]["route"] = {"wrong": "scope"}
        with self.assertRaises(TaskError):
            await self.dispatch(ticket)
        self.execute.assert_not_awaited()

    async def test_withdrawn_reply_is_not_returned(self):
        ticket = await self.request()
        await self.dispatch(ticket)
        self.repository.rows[-1]["availability"] = "deleted"
        with self.assertRaises(TaskError):
            await self.exchange.response(ticket)


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_stays_action_only_and_opt_in_consumer_gets_commands(self):
        task_id, command_id, action_id = uuid4(), uuid4(), uuid4()
        connection = SimpleNamespace(fetch=AsyncMock(return_value=[{"id": task_id, "last_event_sequence": 2}]))
        @asynccontextmanager
        async def acquire():
            yield connection
        repository = SimpleNamespace(pool=SimpleNamespace(acquire=acquire), events=AsyncMock(return_value=[
            {"id": command_id, "sequence": 1, "event_type": REQUEST},
            {"id": action_id, "sequence": 2, "event_type": "action_accepted"}]))
        default, selected = AsyncMock(), AsyncMock()
        await TaskEventObserver(repository, "owner", default).refresh()
        watcher = TaskEventObserver(repository, "owner", selected, event_types=(REQUEST,))
        await watcher.refresh()
        await watcher.refresh()
        default.assert_awaited_once_with(action_id)
        selected.assert_awaited_once_with(command_id)

    def test_event_selection_cannot_be_an_unbounded_or_accidental_string(self):
        for kinds in ("action_accepted", (), ("",), ("x" * 81,)):
            with self.subTest(kinds=kinds), self.assertRaises(ValueError):
                TaskEventObserver(None, "owner", None, event_types=kinds)
