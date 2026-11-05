"""Real SQL for cancelled-empty retention; controlled transport, no worker/model."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskStatus, TaskVoiceSource
import test_realtime_unwritten_request_sources_sql as base
import test_realtime_unselected_request_continuation_sql as continuation


SELECTOR = "jarvis_request_source"


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain SQL fixture required; no default DSN",
)
class CancelledEmptyRequestSQLTests(unittest.IsolatedAsyncioTestCase):
    # Reuse lifecycle/helpers, not TestCase inheritance or its existing tests.
    asyncSetUp = base.UnwrittenRequestSourceSQLTests.asyncSetUp
    open_case = base.UnwrittenRequestSourceSQLTests.open_case
    until = base.UnwrittenRequestSourceSQLTests.until
    event = base.UnwrittenRequestSourceSQLTests.event
    creates = base.UnwrittenRequestSourceSQLTests.creates
    commit = base.UnwrittenRequestSourceSQLTests.commit
    final = base.UnwrittenRequestSourceSQLTests.final
    assert_no_mutation = base.UnwrittenRequestSourceSQLTests.assert_no_mutation
    call = continuation.UnselectedContinuationSQLTests.call
    cancel_on = continuation.UnselectedContinuationSQLTests.cancel_on
    assert_latest_unchanged = continuation.UnselectedContinuationSQLTests.assert_latest_unchanged

    def receipt(self, **fields):
        path = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        path = path / (self.fixture.run_id + "-cancelled-empty.json")
        path.write_text(json.dumps({**self.details, **fields}, indent=2, default=str) + "\n",
                        encoding="utf-8")

    async def written_empty_then_current_get(self):
        case = self.case
        await self.event(case, "input_audio_buffer.speech_started", item_id="old-cancel")
        await self.commit(case, "old-cancel")
        await self.until(case.route.entered.is_set)
        await self.final(case, "old-cancel",
                         "Cancel Workbench Plan. Do not continue it or announce its result.")
        original = case.voice._pending_response
        self.assertIsNotNone(original)
        case.route.release.set()
        await self.until(lambda: len(self.creates(case)) == 1)
        await asyncio.gather(*tuple(case.voice._direct_tasks))
        self.assertTrue(original.submitted)
        first_wire = self.creates(case)[0]
        self.assertEqual(first_wire["response"]["metadata"]["jarvis_request_id"], original.request_id)

        # The real submitted request is interrupted BEFORE its delayed creation ACK.
        await self.event(case, "input_audio_buffer.speech_started", item_id="current-question")
        await self.event(case, "response.created", response={
            "id": "resp_sql_written_old", "metadata": first_wire["response"]["metadata"],
        })
        interrupted = case.voice._active_responses["resp_sql_written_old"]
        self.assertTrue(interrupted.interrupted)
        self.assertNotIn(interrupted.response_id, case.voice._responses_with_tool_calls)
        self.assertFalse(interrupted.pending_tool_call_ids)
        await self.event(case, "response.done", response={
            "id": interrupted.response_id, "status": "cancelled", "output": [],
        })
        await self.until(lambda: case.capture.diagnostics()["cancelled_empty_requests"] == 1)
        self.assertEqual(case.capture.diagnostics()["unwritten_requests"], 0)
        self.assertEqual(len(self.creates(case)), 1)  # Retention never sends a response itself.
        await self.assert_no_mutation()

        case.route.release.clear()
        await self.commit(case, "current-question")
        await self.final(case, "current-question",
                         "What is Workbench Plan's status? Separately, what is two plus three?")
        case.route.release.set()
        await self.until(lambda: len(self.creates(case)) == 2)
        await asyncio.gather(*tuple(case.voice._direct_tasks))
        current_wire = self.creates(case)[-1]
        await self.event(case, "response.created", response={
            "id": "resp_sql_current", "metadata": current_wire["response"]["metadata"],
        })
        state = case.voice._active_responses["resp_sql_current"]
        self.assertFalse(state.interrupted)
        old = next(r for r in state.request_sources.requests if r.kind == "retained_cancelled_empty")
        current = next(r for r in state.request_sources.requests if r.kind == "current")
        for source in (old, current):
            exact = await self.context.request_evidence(self.owner, self.session,
                source_namespace=source.namespace, source_event_id=source.source_event_id)
            self.assertEqual((exact.id, exact.revision, exact.payload_id),
                             (source.record.id, source.record.revision, source.record.payload_id))
            self.assertEqual((exact.owner_id, exact.session_id, exact.finality),
                             (self.owner, self.session, "final"))
        self.assertNotEqual(old.record.id, current.record.id)
        self.assertEqual(state.request_sources.generation, 2)
        self.assertIn(old.record.content["text"], current_wire["response"]["instructions"])

        result = await self.call(case, state.response_id, "get_background_task", "call_current_get", {
            "task_ref": self.tasks.reference(self.task.id), "expected_revision": self.task.revision,
            SELECTOR: current.reference,
        })
        self.assertEqual((result["status"], result["task"]["state"]), ("success", "queued"))
        selected = await case.capture.request_for_call("call_current_get")
        self.assertEqual((selected.id, selected.revision), (current.record.id, current.record.revision))
        await self.until(lambda: not state.pending_tool_call_ids)
        await self.assert_no_mutation()
        await self.event(case, "response.done", response={
            "id": state.response_id, "status": "completed", "output": [],
        })
        await self.until(lambda: len(self.creates(case)) == 3)
        await asyncio.gather(*tuple(case.voice._direct_tasks))
        wire = self.creates(case)[-1]
        await self.event(case, "response.created", response={
            "id": "resp_sql_natural_followup", "metadata": wire["response"]["metadata"],
        })
        followup = case.voice._active_responses["resp_sql_natural_followup"]
        self.assertFalse(followup.interrupted)
        fresh_old = next(r for r in followup.request_sources.requests if r.input_id == old.input_id)
        fresh_current = next(r for r in followup.request_sources.requests if r.input_id == current.input_id)
        self.assertEqual(fresh_old.kind, "retained_cancelled_empty")
        self.assertEqual(fresh_old.record, old.record)
        self.assertEqual(fresh_current.record, current.record)
        self.assertNotIn(fresh_old.reference, {old.reference, current.reference})
        self.assertNotIn(fresh_current.reference, {old.reference, current.reference})
        self.assertEqual(followup.tool_followup.call_request_ids, (("call_current_get", current.record.id),))
        self.assertEqual(followup.tool_followup.request.event_id, current.record.id)
        self.assertEqual(followup.tool_followup.conversation_request.event_id, current.record.id)
        self.assertIn(old.record.content["text"], wire["response"]["instructions"])
        self.assertIn(current.record.content["text"], wire["response"]["instructions"])
        cancel = next(t for t in wire["response"]["tools"] if t["name"] == "cancel_background_task")
        self.assertEqual(set(cancel["parameters"]["properties"][SELECTOR]["enum"]),
                         {fresh_old.reference, fresh_current.reference})
        self.details = {
            "format": "jarvis.cancelled_empty.sql_evidence.v1", "test": self.id(),
            "fixture": self.fixture.run_id, "owner": self.owner, "session_id": str(self.session),
            "task_id": str(self.task.id), "initial_request_id": original.request_id,
            "initial_response_id": interrupted.response_id, "initial_submitted": original.submitted,
            "late_ack_after_new_speech": True, "terminal_status": "cancelled", "terminal_output": [],
            "retention_kind": fresh_old.kind, "response_id": followup.response_id,
            "old_event_id": str(old.record.id), "old_revision": old.record.revision,
            "old_payload_id": str(old.record.payload_id), "old_input_id": old.input_id,
            "current_event_id": str(current.record.id), "current_revision": current.record.revision,
            "current_input_id": current.input_id, "old_response_reference": old.reference,
            "fresh_old_reference": fresh_old.reference, "fresh_current_reference": fresh_current.reference,
            "response_create_count_before_cancel": len(self.creates(case)),
            "boundary": "real SQL/session/writer/capture/executor/registered tools; controlled socket/readiness; queued task, no worker/model",
        }
        self.receipt(status="followup_ready")
        return old, current, followup, fresh_old

    async def test_written_empty_current_get_then_old_cancel_is_source_bound_and_idempotent(self):
        old, current, followup, fresh_old = await self.written_empty_then_current_get()
        for call_id in ("call_old_cancel_one", "call_old_cancel_two"):
            result = await self.cancel_on(followup, fresh_old.reference, call_id)
            self.assertEqual(result["status"], "success")
            selected = await self.case.capture.request_for_call(call_id)
            self.assertEqual((selected.id, selected.revision), (old.record.id, old.record.revision))
        self.assertEqual(len(self.mutations), 2)
        self.assertEqual(self.mutations[0][3].request_id, self.mutations[1][3].request_id)
        self.assertTrue(all(m[4] == TaskVoiceSource(old.record.id, old.record.revision, self.session)
                            for m in self.mutations))
        task = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual((task.status, task.revision, task.fence),
                         (TaskStatus.CANCELLED, self.task.revision + 1, self.task.fence + 1))
        self.assertEqual((task.request_event_id, task.request_revision), (old.record.id, old.record.revision))
        async with self.pool.acquire() as connection:
            joined = await connection.fetch("""select r.request_id,r.accepted_revision,
                e.id as user_event_id,e.revision as user_revision,e.session_id,e.finality,e.trust,
                e.source_namespace,e.source_event_id,e.payload_id as user_payload,
                te.task_revision,te.payload_id as control_payload,d.source_payload_id
                from jarvis_task_requests r
                join jarvis_background_tasks t on t.owner_id=r.owner_id and t.id=r.task_id
                join jarvis_context_events e on e.owner_id=t.owner_id
                    and e.id=t.request_event_id and e.revision=t.request_revision
                join jarvis_task_events te on te.owner_id=t.owner_id and te.task_id=t.id
                    and te.event_type='cancel_requested' and te.task_revision=r.accepted_revision
                join jarvis_context_payload_dependencies d on d.owner_id=e.owner_id
                    and d.source_payload_id=e.payload_id and d.dependent_payload_id=te.payload_id
                where r.owner_id=$1 and r.task_id=$2 and r.operation='cancel'""", self.owner, self.task.id)
            self.assertEqual(len(joined), 1)
            row = joined[0]
            self.assertEqual((row["user_event_id"], row["user_revision"], row["session_id"],
                              row["user_payload"], row["source_payload_id"], row["accepted_revision"]),
                             (old.record.id, old.record.revision, self.session,
                              old.record.payload_id, old.record.payload_id, task.revision))
            self.assertEqual((row["finality"], row["trust"]), ("final", "user_input"))
            counts = {table: await connection.fetchval(f"select count(*) from {table} where owner_id=$1", self.owner)
                      for table in ("jarvis_background_tasks", "jarvis_task_attempts", "jarvis_task_actions")}
            self.assertEqual(counts, {"jarvis_background_tasks": 1, "jarvis_task_attempts": 0, "jarvis_task_actions": 0})
        executions = await self.executions.get_executions_by_call_ids(owner_id=self.owner,
            session_id=self.session, provider_call_ids=["call_old_cancel_one", "call_old_cancel_two"])
        self.assertEqual({e.provider_call_id for e in executions}, {"call_old_cancel_one", "call_old_cancel_two"})
        self.assertTrue(all(e.status == "completed" for e in executions))
        self.assertEqual(followup.tool_followup.conversation_request.event_id, current.record.id)
        await self.assert_latest_unchanged(current)
        self.receipt(status="passed", task_state=task.status, task_revision=task.revision, task_fence=task.fence,
                     canonical_control_join=[dict(row) for row in joined], counts=counts,
                     provider_call_ids=sorted(e.provider_call_id for e in executions), latest_preserved=True)

    async def test_written_empty_corrected_canonical_source_and_old_selector_do_not_cancel(self):
        old, current, followup, fresh_old = await self.written_empty_then_current_get()
        stale_selector = await self.cancel_on(followup, old.reference, "call_old_response_selector")
        self.assertEqual(stale_selector["status"], "failed")
        self.assertEqual(stale_selector["error"]["code"], "tool_request_source_invalid")
        await self.assert_no_mutation()
        corrected = await self.context.append(self.owner, ContextIngress(
            session_id=self.session, source_namespace=old.namespace, source_event_id=old.source_event_id,
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input", finality="revised",
            expected_revision=old.record.revision,
            payload={**old.record.content, "text": "Correction: do not cancel Workbench Plan. Leave it unchanged."},
        ))
        self.assertEqual(corrected.revision, old.record.revision + 1)
        self.assertEqual(self.case.capture.cached_request(old.input_id), old.record)
        result = await self.cancel_on(followup, fresh_old.reference, "call_corrected_old_source")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "tool_execution_failed")
        await self.assert_no_mutation()
        await self.assert_latest_unchanged(current)
        exact = await self.context.request_evidence(self.owner, self.session,
            source_namespace=old.namespace, source_event_id=old.source_event_id)
        self.assertEqual(exact.id, corrected.id)
        self.receipt(status="passed", old_selector_error=stale_selector["error"]["code"],
                     correction_error=result["error"]["code"], corrected_event_id=str(corrected.id),
                     corrected_revision=corrected.revision, mutation_count=0, latest_preserved=True)
