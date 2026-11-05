"""Offline application-contract tests; SQL publication is separately database-gated."""
from contextlib import asynccontextmanager
from copy import deepcopy
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.attention.models import ACTIVE_PHASES, AttentionError, DeliveryClaim, InboxMutation
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService
from app.tasks.models import TaskError, TaskMutation, TaskStatus, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule, CancelArguments, audit_task_result


class Connection:
    """Bounded in-memory SQL contract fixture, never a database connection."""

    def __init__(self, fixture):
        self.f = fixture
        self.queries = []

    async def fetchrow(self, sql, *args):
        self.queries.append((sql, args))
        f = self.f
        if "select id,revision,sequence,session_id,payload_id from jarvis_context_events" in sql:
            if args == (f.owner, f.source.event_id, f.source.revision, f.source.session_id):
                return {"id": f.source.event_id, "revision": f.source.revision,
                        "sequence": 7, "session_id": f.session, "payload_id": f.request_payload}
            return None
        if "select e.id,p.availability,e.finality,e.trust,e.kind" in sql:
            if args != (f.owner, f.source.event_id, f.source.revision) or not f.source_current:
                return None
            return {"id": f.source.event_id, "availability": f.availability,
                    "finality": f.finality, "trust": f.trust, "kind": "user_transcript"}
        if "select * from jarvis_attention_requests" in sql:
            return deepcopy(f.requests.get(args))
        if "select * from jarvis_attention_intents" in sql:
            row = f.intents.get(args[1])
            return deepcopy(row) if row and row["owner_id"] == args[0] else None
        if "select id from jarvis_context_payloads" in sql:
            return {"id": args[1]} if args[0] == f.owner else None
        if "select * from jarvis_attention_attempts" in sql:
            row = f.attempts.get(args[1])
            if row and (row["owner_id"], row["intent_id"], row["generation"],
                        row["session_id"], row["worker_id"]) == args[:1] + args[2:6]:
                return deepcopy(row) if row["phase"] in args[6] else None
            return None
        raise AssertionError("Unexpected fetchrow contract: " + sql)

    async def fetchval(self, sql, *args):
        self.queries.append((sql, args))
        if "jarvis_context_streams" in sql:
            return 1
        if "jarvis_sessions" in sql:
            return args == (self.f.owner, self.f.session)
        raise AssertionError("Unexpected fetchval contract: " + sql)

    async def fetch(self, sql, *args):
        self.queries.append((sql, args))
        if "select id,generation from jarvis_attention_intents" not in sql:
            raise AssertionError("Unexpected fetch contract: " + sql)
        assert "source_namespace='task_result'" in sql
        assert "source_payload_id=$4" in sql and "source_id=$4" in sql
        assert "task_revision=$3" in sql and "source_revision=$3" in sql
        assert "order by" not in sql.lower() and "limit 2" in sql
        owner, task_id, revision, payload = args
        rows = [{"id": row["id"], "generation": row["generation"]}
                for row in self.f.intents.values()
                if row["owner_id"] == owner and row["task_id"] == task_id
                and row["task_revision"] == revision and row["source_namespace"] == "task_result"
                and row["source_payload_id"] == payload and row["source_id"] == payload
                and row["source_revision"] == revision][:2]
        if self.f.selection_hook is not None:
            self.f.selection_hook()
        return rows

    async def execute(self, sql, *args):
        self.queries.append((sql, args))
        f = self.f
        if "update jarvis_attention_attempts set phase='revoked'" in sql:
            for row in f.attempts.values():
                if row["owner_id"] == args[0] and row["intent_id"] == args[1] and row["phase"] in args[2]:
                    row.update(phase="revoked", reason="inbox_dismissed", ended_at="after")
        elif "update jarvis_attention_intents set state=$3,generation=$4" in sql:
            row = f.intents[args[1]]
            assert row["owner_id"] == args[0]
            row.update(state=args[2], generation=args[3], reason=None)
        elif "insert into jarvis_attention_requests" in sql:
            owner, request, intent, operation, signature, generation = args
            assert (owner, request) not in f.requests
            f.requests[owner, request] = {"intent_id": intent, "operation": operation,
                "parameters_digest": signature, "accepted_generation": generation}
        else:
            raise AssertionError("Unexpected write contract: " + sql)


class Fixture:
    _voice_source = TaskRepository._voice_source
    _source_available = TaskRepository._source_available
    reference = staticmethod(TaskRepository.reference)

    def __init__(self):
        self.owner, self.session = "owned-test", uuid4()
        self.source = TaskVoiceSource(uuid4(), 2, self.session)
        self.request_payload = uuid4()
        self.source_current = True
        self.availability, self.finality, self.trust = "available", "final", "user_input"
        self.task = SimpleNamespace(id=uuid4(), owner_id=self.owner, session_id=self.session,
            status=TaskStatus.SUCCEEDED, revision=3, result_payload_id=uuid4(), title="Exact task",
            mode="background", reason=None, pending_status=None, knowledge_area_id=None,
            updated_at=SimpleNamespace(isoformat=lambda: "2026-11-01T00:00:00+00:00"))
        self.intent = uuid4()
        self.intents = {self.intent: {
            "id": self.intent, "owner_id": self.owner, "task_id": self.task.id,
            "task_revision": self.task.revision, "source_namespace": "task_result",
            "source_id": self.task.result_payload_id, "source_revision": self.task.revision,
            "source_payload_id": self.task.result_payload_id, "generation": 5,
            "state": "pending", "attempt_count": 0, "reason": None,
        }}
        self.requests, self.attempts = {}, {}
        self.selection_hook = None
        self.lock = asyncio.Lock()
        self.pool = object()
        self.connection = Connection(self)
        self.mutate = AsyncMock(side_effect=TaskError("task_state_conflict", "Already completed"))
        self.resolve_reference = AsyncMock(return_value=self.task.id)
        self.attention = AttentionRepository(self)
        self.coordinator = AttentionService(self.attention)
        self.service = TaskService(self, {})
        self.service.attention_service = self.coordinator
        self.request = TaskMutation(request_id=uuid4(), expected_revision=3, message="Cancel it and do not announce its result")

    @asynccontextmanager
    async def transaction(self, owner=None):
        async with self.lock:
            before = deepcopy((self.intents, self.attempts, self.requests, self.task))
            try:
                yield self.connection
            except BaseException:
                self.intents, self.attempts, self.requests, self.task = before
                raise

    async def load(self, connection, owner, task_id, *, lock=False):
        if owner != self.owner or task_id != self.task.id:
            raise TaskError("task_not_found", "No owned task", 404)
        return deepcopy(self.task)

    async def dismiss(self, request=None, source=None):
        return await self.attention.dismiss_task_result(
            self.owner, self.task.id, request or self.request, source=source or self.source)

    def claim(self, *, phase="forwarding"):
        identity = uuid4()
        row = self.intents[self.intent]
        row["state"] = "claimed"
        claim = DeliveryClaim(self.owner, self.intent, identity, row["generation"],
            self.session, "delivery", row["source_payload_id"])
        self.attempts[identity] = {"id": identity, "owner_id": self.owner, "intent_id": self.intent,
            "generation": claim.generation, "session_id": self.session, "worker_id": "delivery",
            "phase": phase, "forwarded_samples": 48000, "played_samples": 12000,
            "submitted_at": "before", "ended_at": None}
        return claim


class Delivery:
    def __init__(self, claim, fixture):
        self.claim, self.fixture = claim, fixture
        self.task, self.alive, self.stops = None, True, []
        self.fail = False

    def current(self):
        return self.alive

    async def stop(self, reason):
        assert not self.fixture.lock.locked(), "Transport must run after commit"
        if self.fail:
            raise RuntimeError("fixture transport failure")
        self.stops.append(reason)
        self.alive = False


class TaskResultSuppressionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = Fixture()

    async def test_completed_work_and_result_are_preserved(self):
        f = self.f
        original = deepcopy(f.task.__dict__)
        task, work, receipt = await f.service.cancel_with_result_suppression(
            f.owner, f.task.id, f.request, source=f.source)
        self.assertEqual(f.task.__dict__, original)
        self.assertEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertEqual(work["status"], "already_completed")
        self.assertFalse(work["cancellation_requested"])
        self.assertFalse(work["completed_work_cancelled"])
        self.assertEqual(receipt["status"], "dismissed")
        self.assertTrue(receipt["durable_suppression"])
        self.assertEqual(f.intents[f.intent]["generation"], 6)
        self.assertEqual(f.intents[f.intent]["state"], "dismissed")
        self.assertEqual(len(f.requests), 1)

    async def test_missing_intent_is_not_a_future_suppression_promise(self):
        f = self.f
        saved = f.intents.pop(f.intent)
        _, receipt = await f.dismiss()
        self.assertFalse(receipt["durable_suppression"])
        self.assertEqual(receipt["reason"], "exact_result_intent_missing")
        self.assertFalse(f.requests)
        f.intents[f.intent] = saved
        self.assertEqual(f.intents[f.intent]["state"], "pending")
        self.assertFalse(receipt["durable_suppression"])

    async def test_ambiguous_intents_are_not_dismissed(self):
        f = self.f
        other = uuid4()
        f.intents[other] = deepcopy(f.intents[f.intent]) | {"id": other}
        _, receipt = await f.dismiss()
        self.assertEqual(receipt["reason"], "exact_result_intent_ambiguous")
        self.assertFalse(f.requests)
        self.assertTrue(all(row["state"] == "pending" for row in f.intents.values()))

    async def test_wrong_result_revision_namespace_or_owner_never_matches(self):
        for field, value in (("task_revision", 2), ("source_revision", 2),
                             ("source_payload_id", uuid4()), ("source_id", uuid4()),
                             ("owner_id", "foreign"), ("task_id", uuid4()),
                             ("source_namespace", "reminder_event")):
            with self.subTest(field=field):
                f = Fixture()
                f.intents[f.intent][field] = value
                _, receipt = await f.dismiss()
                self.assertFalse(receipt["durable_suppression"])
                self.assertFalse(f.requests)

    async def test_trusted_request_source_must_be_current_final_owned_and_available(self):
        for field, value in (("source_current", False), ("availability", "removed"),
                             ("finality", "provisional"), ("trust", "assistant_output")):
            with self.subTest(field=field):
                f = Fixture()
                setattr(f, field, value)
                with self.assertRaises(TaskError) as error:
                    await f.dismiss()
                self.assertEqual(error.exception.code, "task_request_evidence_missing")
                self.assertFalse(f.requests)
        f = self.f
        for source in (None, TaskVoiceSource(uuid4(), 2, f.session),
                       TaskVoiceSource(f.source.event_id, 1, f.session),
                       TaskVoiceSource(f.source.event_id, 2, uuid4())):
            with self.subTest(source=source):
                with self.assertRaises(TaskError):
                    await f.attention.dismiss_task_result(f.owner, f.task.id, f.request, source=source)
                self.assertFalse(f.requests)

    async def test_foreign_owner_cannot_act_on_known_ids(self):
        f = self.f
        with self.assertRaises(TaskError):
            await f.attention.dismiss_task_result("foreign", f.task.id, f.request, source=f.source)
        self.assertEqual(f.intents[f.intent]["state"], "pending")

    async def test_wrong_task_reference_and_revision_reject(self):
        f = self.f
        with self.assertRaises(TaskError) as error:
            await f.attention.dismiss_task_result(f.owner, uuid4(), f.request, source=f.source)
        self.assertEqual(error.exception.code, "task_not_found")
        with self.assertRaises(TaskError) as error:
            await f.dismiss(f.request.model_copy(update={"expected_revision": 2}))
        self.assertEqual(error.exception.code, "task_revision_conflict")

    async def test_noncompleted_result_adapter_cannot_cancel_work(self):
        for state in (TaskStatus.RUNNING, TaskStatus.UNKNOWN, TaskStatus.CANCEL_REQUESTED, TaskStatus.FAILED):
            with self.subTest(state=state):
                f = Fixture()
                f.task.status = state
                with self.assertRaises(TaskError):
                    await f.dismiss()
                self.assertEqual(f.task.status, state)
                self.assertFalse(f.requests)

    async def test_generation_change_at_lock_boundary_rejects(self):
        f = self.f
        f.selection_hook = lambda: f.intents[f.intent].update(generation=6)
        with self.assertRaises(AttentionError) as error:
            await f.dismiss()
        self.assertEqual(error.exception.code, "attention_generation_conflict")
        self.assertFalse(f.requests)

    async def test_result_change_at_lock_boundary_rejects(self):
        f = self.f
        f.selection_hook = lambda: setattr(f.task, "result_payload_id", uuid4())
        with self.assertRaises(AttentionError) as error:
            await f.dismiss()
        self.assertEqual(error.exception.code, "attention_source_changed")
        self.assertFalse(f.requests)

    async def test_duplicate_request_does_not_repeat_dismissal(self):
        f = self.f
        _, first = await f.dismiss()
        _, second = await f.dismiss()
        self.assertTrue(second["replayed_request"])
        self.assertEqual(first["generation"], second["generation"])
        self.assertEqual(f.intents[f.intent]["generation"], 6)
        self.assertEqual(len(f.requests), 1)

    async def test_conflicting_retry_does_not_mutate(self):
        f = self.f
        await f.dismiss()
        with self.assertRaises(AttentionError) as error:
            await f.dismiss(f.request.model_copy(update={"message": "different instruction"}))
        self.assertEqual(error.exception.code, "attention_request_conflict")
        self.assertEqual(f.intents[f.intent]["generation"], 6)

    async def test_same_request_cannot_target_another_intent(self):
        f = self.f
        await f.dismiss()
        original = f.intents.pop(f.intent)
        replacement = uuid4()
        f.intents[replacement] = original | {"id": replacement, "generation": 1, "state": "pending"}
        with self.assertRaises(AttentionError) as error:
            await f.dismiss()
        self.assertEqual(error.exception.code, "attention_request_conflict")
        self.assertEqual(f.intents[replacement]["state"], "pending")

    async def test_later_task_revision_is_unaffected_by_old_request(self):
        f = self.f
        await f.dismiss()
        f.task.revision += 1
        f.task.result_payload_id = uuid4()
        with self.assertRaises(TaskError) as error:
            await f.dismiss()
        self.assertEqual(error.exception.code, "task_revision_conflict")
        self.assertEqual(f.task.revision, 4)

    async def test_explicit_replay_generation_is_not_silently_redismissed(self):
        f = self.f
        await f.dismiss()
        f.intents[f.intent].update(generation=7, state="pending")
        _, receipt = await f.dismiss()
        self.assertEqual(receipt["status"], "historical_dismissal")
        self.assertFalse(receipt["durable_suppression"])
        self.assertEqual(f.intents[f.intent]["generation"], 7)

    async def test_source_correction_revokes_retry_authority(self):
        f = self.f
        await f.dismiss()
        f.source_current = False
        with self.assertRaises(TaskError):
            await f.dismiss()
        self.assertEqual(len(f.requests), 1)

    async def test_unrelated_intent_and_already_forwarded_evidence_are_preserved(self):
        f = self.f
        claim = f.claim()
        other = uuid4()
        f.intents[other] = deepcopy(f.intents[f.intent]) | {"id": other, "task_id": uuid4()}
        before = deepcopy(f.intents[other])
        _, receipt = await f.dismiss()
        self.assertEqual(f.intents[other], before)
        attempt = f.attempts[claim.attempt_id]
        self.assertEqual(attempt["phase"], "revoked")
        self.assertEqual(attempt["forwarded_samples"], 48000)
        self.assertEqual(attempt["played_samples"], 12000)
        self.assertEqual(attempt["submitted_at"], "before")
        self.assertEqual(receipt["hearing"], "not_established")
        self.assertEqual(receipt["prior_delivery"], "not_retracted")

    async def test_dismissal_serializes_with_write_and_fences_next_write(self):
        f = self.f
        claim = f.claim()
        entered, release = asyncio.Event(), asyncio.Event()
        f.attention._source = AsyncMock(return_value=(SimpleNamespace(speech_ready=True), None))

        async def writer():
            async with f.attention.guard(claim):
                entered.set()
                await release.wait()

        with patch("app.attention.repository.quiet_delivery_decision",
                   AsyncMock(return_value=SimpleNamespace(allowed=True))), patch(
                   "app.attention.repository.coalesce_ended_episode", AsyncMock(return_value=True)):
            writing = asyncio.create_task(writer())
            await asyncio.wait_for(entered.wait(), 1)
            dismissing = asyncio.create_task(f.dismiss())
            try:
                await asyncio.sleep(0)
                self.assertFalse(dismissing.done())
            finally:
                release.set()
                await asyncio.gather(writing, dismissing)
            with self.assertRaises(AttentionError) as error:
                async with f.attention.guard(claim):
                    self.fail("Revoked generation reached writer")
            self.assertEqual(error.exception.code, "attention_claim_lost")

    async def test_completion_publication_lock_prevents_partial_intent_observation(self):
        f = self.f
        intent = f.intents.pop(f.intent)
        f.task.status = TaskStatus.RUNNING
        entered, release = asyncio.Event(), asyncio.Event()

        async def complete_and_publish():
            # Emulates the existing AFTER UPDATE trigger's single transaction;
            # this is an offline lock-contract test, not a live SQL qualification.
            async with f.transaction(f.owner):
                f.task.status = TaskStatus.SUCCEEDED
                entered.set()
                await release.wait()
                f.intents[f.intent] = intent

        publishing = asyncio.create_task(complete_and_publish())
        await asyncio.wait_for(entered.wait(), 1)
        dismissing = asyncio.create_task(f.dismiss())
        try:
            await asyncio.sleep(0)
            self.assertFalse(dismissing.done())
        finally:
            release.set()
            await publishing
            _, receipt = await dismissing
        self.assertTrue(receipt["durable_suppression"])

    async def test_active_cancel_and_unknown_effects_keep_existing_outcomes(self):
        for state in (TaskStatus.CANCELLED, TaskStatus.CANCEL_REQUESTED, TaskStatus.UNKNOWN):
            with self.subTest(state=state):
                f = Fixture()
                f.task.status = state
                f.mutate = AsyncMock(return_value=f.task)
                task, work, delivery = await f.service.cancel_with_result_suppression(
                    f.owner, f.task.id, f.request, source=f.source)
                f.mutate.assert_awaited_once_with(f.owner, f.task.id, "cancel", f.request, source=f.source)
                self.assertEqual(task.status, state)
                self.assertEqual(work["status"], state.value)
                self.assertTrue(work["cancellation_requested"])
                self.assertEqual(work["external_effects"], "not_undone")
                self.assertFalse(delivery["durable_suppression"])
                self.assertFalse(f.requests)

    async def test_default_completed_cancel_still_conflicts(self):
        f = self.f
        with self.assertRaises(TaskError) as error:
            await f.service.mutate(f.owner, f.task.id, "cancel", f.request, source=f.source)
        self.assertEqual(error.exception.code, "task_state_conflict")
        self.assertFalse(f.requests)

    async def test_non_state_error_is_not_converted_to_suppression(self):
        f = self.f
        f.mutate.side_effect = TaskError("task_revision_conflict", "Changed")
        with self.assertRaises(TaskError) as error:
            await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=f.source)
        self.assertEqual(error.exception.code, "task_revision_conflict")
        self.assertFalse(f.requests)

    async def test_missing_attention_or_trusted_source_never_claims_suppression(self):
        f = self.f
        f.service.attention_service = None
        with self.assertRaises(TaskError) as error:
            await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=f.source)
        self.assertEqual(error.exception.code, "task_announcement_control_unavailable")
        f.mutate.reset_mock()
        with self.assertRaises(TaskError):
            await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=None)
        f.mutate.assert_not_called()

    async def test_post_commit_stop_is_exact_and_duplicate_safe(self):
        f = self.f
        claim = f.claim()
        wanted = Delivery(claim, f)
        foreign = Delivery(DeliveryClaim("foreign", claim.intent_id, uuid4(), claim.generation,
            f.session, "foreign", claim.source_payload_id), f)
        later = Delivery(DeliveryClaim(f.owner, claim.intent_id, uuid4(), claim.generation + 1,
            f.session, "later", claim.source_payload_id), f)
        wrong_source = Delivery(DeliveryClaim(f.owner, claim.intent_id, uuid4(), claim.generation,
            f.session, "wrong", uuid4()), f)
        f.coordinator._deliveries.update((wanted, foreign, later, wrong_source))
        await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=f.source)
        await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=f.source)
        self.assertEqual(wanted.stops, ["task_result_dismissed"])
        self.assertFalse(foreign.stops or later.stops or wrong_source.stops)

    async def test_transport_failure_does_not_erase_durable_dismissal_or_claim_buffer_clear(self):
        f = self.f
        delivery = Delivery(f.claim(), f)
        delivery.fail = True
        f.coordinator._deliveries.add(delivery)
        _, _, receipt = await f.service.cancel_with_result_suppression(f.owner, f.task.id, f.request, source=f.source)
        self.assertTrue(receipt["durable_suppression"])
        self.assertEqual(receipt["transport_stop"]["status"], "unconfirmed")
        self.assertEqual(receipt["prior_delivery"], "not_retracted")

    async def test_existing_inbox_dismissal_remains_retry_stable(self):
        f = self.f
        request = InboxMutation(request_id=uuid4(), expected_generation=5)
        first = await f.attention.mutate(f.owner, f.intent, "dismiss", request)
        second = await f.attention.mutate(f.owner, f.intent, "dismiss", request)
        self.assertEqual(first["generation"], 6)
        self.assertTrue(second["replayed_request"])


class TaskSuppressionVoiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = Fixture()
        f = self.f
        self.tool = next(tool for tool in BackgroundTaskToolModule(f.service).build_tools(
            SimpleNamespace(owner_id=f.owner, session_id=f.session)) if tool.name == "cancel_background_task")
        self.arguments = {"task_ref": f.reference(f.task.id), "expected_revision": 3}
        self.invocation = SimpleNamespace(task=None, request_event_id=f.source.event_id,
            request_revision=f.source.revision, request_identity=SimpleNamespace(
                namespace="voice:trusted", event_id="input:original"))

    def test_flag_defaults_false_and_is_strict(self):
        self.assertFalse(CancelArguments(**self.arguments).suppress_result_announcement)
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                CancelArguments(**self.arguments, suppress_result_announcement=value)

    async def test_default_voice_path_does_not_dismiss_completed_result(self):
        with patch("app.tasks.voice_tools.current_tool_invocation", return_value=self.invocation):
            result = await self.tool.handler(self.arguments)
        self.assertEqual(result["error"]["code"], "task_state_conflict")
        self.assertFalse(self.f.requests)
        self.assertEqual(self.f.task.status, TaskStatus.SUCCEEDED)

    async def test_opt_in_uses_trusted_request_and_preserves_audit_outcomes(self):
        with patch("app.tasks.voice_tools.current_tool_invocation", return_value=self.invocation):
            result = await self.tool.handler(self.arguments | {"suppress_result_announcement": True})
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["accepted"])
        self.assertEqual(result["work"]["status"], "already_completed")
        self.assertTrue(result["delivery"]["durable_suppression"])
        self.assertEqual(audit_task_result(result)["work"], result["work"])
        self.assertEqual(audit_task_result(result)["delivery"], result["delivery"])
        self.assertEqual(self.f.mutate.await_args.kwargs["source"], self.f.source)

    async def test_missing_intent_voice_receipt_is_not_accepted_suppression(self):
        self.f.intents.clear()
        with patch("app.tasks.voice_tools.current_tool_invocation", return_value=self.invocation):
            result = await self.tool.handler(self.arguments | {"suppress_result_announcement": True})
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["work"]["status"], "already_completed")
        self.assertFalse(result["delivery"]["durable_suppression"])

    async def test_unfinalized_or_worker_invocation_cannot_dismiss(self):
        for changes in ({"request_event_id": None}, {"task": object()}):
            with self.subTest(changes=tuple(changes)):
                invocation = SimpleNamespace(**(self.invocation.__dict__ | changes))
                with patch("app.tasks.voice_tools.current_tool_invocation", return_value=invocation):
                    result = await self.tool.handler(self.arguments | {"suppress_result_announcement": True})
                self.assertEqual(result["error"]["code"], "task_request_not_finalized")
                self.assertFalse(self.f.requests)

