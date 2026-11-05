"""Opt-in SQL extension of the original four selected-request attribution cases."""
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import unittest
from uuid import uuid4

from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskError, TaskStatus
from test_realtime_unwritten_request_sources_sql import UnwrittenRequestSourceSQLTests


SELECTOR = "jarvis_request_source"


class UnselectedContinuationSQLTests(UnwrittenRequestSourceSQLTests):
    async def call(self, case, response_id, name, call_id, arguments):
        await self.event(case, "response.output_item.done", response_id=response_id, item={
            "type": "function_call", "id": "item_" + call_id, "call_id": call_id,
            "name": name, "arguments": json.dumps(arguments), "status": "completed",
        })

        def matching():
            return [event["item"] for event in case.socket.events
                    if event.get("type") == "conversation.item.create"
                    and event.get("item", {}).get("type") == "function_call_output"
                    and event["item"].get("call_id") == call_id]

        await self.until(lambda: bool(matching()))
        self.assertTrue(await case.capture.flush())
        return json.loads(matching()[-1]["output"])

    async def current_get_then_followup(self):
        state, old, current = await self.offered_pair(self.case)
        result = await self.call(self.case, state.response_id, "get_background_task", "call_current_read", {
            "task_ref": self.tasks.reference(self.task.id), "expected_revision": self.task.revision,
            SELECTOR: current.reference,
        })
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task"]["state"], "queued")
        resolved = await self.case.capture.request_for_call("call_current_read")
        self.assertEqual((resolved.id, resolved.revision), (current.record.id, current.record.revision))
        await self.assert_no_mutation()
        before = len(self.creates(self.case))
        await self.event(self.case, "response.done", response={
            "id": state.response_id, "status": "completed", "output": [],
        })
        await self.until(lambda: len(self.creates(self.case)) == before + 1)
        wire = self.creates(self.case)[-1]
        await self.event(self.case, "response.created", response={
            "id": "resp_sql_followup", "metadata": wire["response"]["metadata"],
        })
        followup = self.case.voice._active_responses["resp_sql_followup"]
        self.assertFalse(followup.interrupted)
        self.assertIsNotNone(followup.request_sources)
        fresh_old = next(r for r in followup.request_sources.requests if r.input_id == old.input_id)
        fresh_current = next(r for r in followup.request_sources.requests if r.input_id == current.input_id)
        self.assertEqual(fresh_old.kind, "retained_unwritten")
        self.assertEqual(fresh_old.record, old.record)
        self.assertNotIn(fresh_old.reference, {old.reference, current.reference})
        self.assertNotIn(fresh_current.reference, {old.reference, current.reference})
        self.assertEqual(followup.tool_followup.call_request_ids, (("call_current_read", current.record.id),))
        self.assertEqual(followup.tool_followup.request.event_id, current.record.id)
        self.assertEqual(followup.tool_followup.conversation_request.event_id, current.record.id)
        self.assertIn(old.record.content["text"], wire["response"]["instructions"])
        self.assertIn(current.record.content["text"], wire["response"]["instructions"])
        cancel = next(tool for tool in wire["response"]["tools"] if tool["name"] == "cancel_background_task")
        self.assertEqual(set(cancel["parameters"]["properties"][SELECTOR]["enum"]),
                         {fresh_old.reference, fresh_current.reference})
        self.assertFalse(self.case.capture._unwritten_inputs)
        self.details = {
            "format": "jarvis.unselected_continuation.sql_evidence.v1", "test": self.id(),
            "fixture": self.fixture.run_id, "owner": self.owner, "session_id": str(self.session),
            "task_id": str(self.task.id), "response_id": followup.response_id,
            "request_id": wire["response"]["metadata"]["jarvis_request_id"],
            "old_event_id": str(old.record.id), "old_revision": old.record.revision,
            "current_event_id": str(current.record.id), "current_revision": current.record.revision,
            "old_provider_item_id": old.provider_item_id, "current_provider_item_id": current.provider_item_id,
            "old_response_reference": old.reference, "fresh_old_reference": fresh_old.reference,
            "fresh_current_reference": fresh_current.reference,
            "get_result_status": result["status"], "get_task_state": result["task"]["state"],
            "boundary": "real SQL/session/guarded wire/executor/registered tools; controlled socket/readiness; no worker/model",
        }
        return state, old, current, followup, fresh_old, fresh_current

    async def cancel_on(self, response, reference, call_id):
        arguments = {"task_ref": self.tasks.reference(self.task.id), "expected_revision": self.task.revision}
        if reference is not None:
            arguments[SELECTOR] = reference
        return await self.call(self.case, response.response_id, "cancel_background_task", call_id, arguments)

    async def assert_latest_unchanged(self, current):
        actual = await self.context.request_evidence(self.owner, self.session,
            source_namespace=current.namespace, source_event_id=current.source_event_id)
        self.assertEqual((actual.id, actual.revision, actual.payload_id),
                         (current.record.id, current.record.revision, current.record.payload_id))
        self.assertEqual(self.case.capture._generations[self.case.voice._turn_generation], current.input_id)

    def receipt(self, **fields):
        path = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / (self.fixture.run_id + "-continuation.json")
        path.write_text(json.dumps({**self.details, **fields}, indent=2, default=str) + "\n", encoding="utf-8")

    async def test_continuation_current_read_then_old_cancel_commits_once(self):
        state, old, current, followup, fresh_old, fresh_current = await self.current_get_then_followup()
        first = await self.cancel_on(followup, fresh_old.reference, "call_old_cancel_one")
        second = await self.cancel_on(followup, fresh_old.reference, "call_old_cancel_two")
        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        task = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertEqual(task.revision, self.task.revision + 1)
        self.assertEqual(task.fence, self.task.fence + 1)
        self.assertEqual((task.request_event_id, task.request_revision), (old.record.id, old.record.revision))
        for call_id in ("call_old_cancel_one", "call_old_cancel_two"):
            exact = await self.case.capture.request_for_call(call_id)
            self.assertEqual((exact.id, exact.revision), (old.record.id, old.record.revision))
        self.assertEqual(followup.tool_followup.conversation_request.event_id, current.record.id)
        async with self.pool.acquire() as connection:
            requests = await connection.fetch("select * from jarvis_task_requests where owner_id=$1 and task_id=$2 and operation='cancel'", self.owner, self.task.id)
            events = await connection.fetch("select * from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type='cancel_requested'", self.owner, self.task.id)
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(events), 1)
            self.assertEqual(requests[0]["accepted_revision"], task.revision)
            self.assertTrue(await connection.fetchval(
                "select exists(select 1 from jarvis_context_payload_dependencies where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)",
                self.owner, old.record.payload_id, events[0]["payload_id"]))
            executions = await connection.fetch("select provider_call_id,status from jarvis_tool_executions where owner_id=$1 and tool_name='cancel_background_task' order by provider_call_id", self.owner)
            self.assertEqual({row["provider_call_id"] for row in executions}, {"call_old_cancel_one", "call_old_cancel_two"})
            self.assertTrue(all(row["status"] == "completed" for row in executions))
            counts = {table: await connection.fetchval(f"select count(*) from {table} where owner_id=$1", self.owner)
                      for table in ("jarvis_background_tasks", "jarvis_task_attempts", "jarvis_task_actions")}
            self.assertEqual(counts, {"jarvis_background_tasks": 1, "jarvis_task_attempts": 0, "jarvis_task_actions": 0})
        await self.assert_latest_unchanged(current)
        self.receipt(status="passed", task_state=task.status, task_revision=task.revision, fence=task.fence,
                     cancel_requests=[dict(row) for row in requests], cancel_events=[dict(row) for row in events],
                     executions=[dict(row) for row in executions], counts=counts, latest_preserved=True)

    async def test_continuation_missing_foreign_and_old_response_selectors_have_no_effect(self):
        state, old, current, followup, fresh_old, fresh_current = await self.current_get_then_followup()
        async with self.pool.acquire() as connection:
            other_session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", self.other)
        other = await self.open_case(self.other, other_session)
        foreign_state, foreign_old, foreign_current = await self.offered_pair(other)
        rejections = []
        for name, reference, expected in (
            ("missing", None, "tool_request_source_required"),
            ("old_response", old.reference, "tool_request_source_invalid"),
            ("foreign", foreign_old.reference, "tool_request_source_invalid"),
        ):
            result = await self.cancel_on(followup, reference, "call_rejected_" + name)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error"]["code"], expected)
            await self.assert_no_mutation()
            rejections.append({"kind": name, "code": expected})
        with self.assertRaises(TaskError):
            await self.context.get(self.owner, foreign_old.record.id)
        await self.assert_latest_unchanged(current)
        self.receipt(status="passed", rejections=rejections, foreign_owner=self.other,
                     foreign_event_id=str(foreign_old.record.id), mutation_count=0, latest_preserved=True)

    async def test_continuation_corrected_canonical_source_rejects_old_cache_and_selector(self):
        state, old, current, followup, fresh_old, fresh_current = await self.current_get_then_followup()
        revised = await self.context.append(self.owner, ContextIngress(
            session_id=self.session, source_namespace=old.namespace, source_event_id=old.source_event_id,
            receipt_id=str(uuid4()), kind="user_transcript", trust="user_input", finality="revised",
            expected_revision=old.record.revision, source_time=datetime.now(UTC),
            payload={**old.record.content, "text": "Withdraw my earlier cancellation request. Leave this task unchanged."},
        ))
        self.assertEqual(revised.revision, old.record.revision + 1)
        self.assertEqual(self.case.capture.cached_request(old.input_id), old.record)
        result = await self.cancel_on(followup, fresh_old.reference, "call_corrected_source")
        self.assertEqual(result["status"], "failed")
        await self.assert_no_mutation()
        await self.assert_latest_unchanged(current)
        exact = await self.context.request_evidence(self.owner, self.session,
            source_namespace=old.namespace, source_event_id=old.source_event_id)
        self.assertEqual(exact.id, revised.id)
        self.receipt(status="passed", error_code=result["error"]["code"], new_event_id=str(revised.id),
                     new_revision=revised.revision, mutation_count=0, latest_preserved=True)

    async def test_continuation_withdrawn_canonical_payload_cannot_cancel(self):
        state, old, current, followup, fresh_old, fresh_current = await self.current_get_then_followup()
        # Exact fixture-only withdrawal exercises real dependency/deletion SQL.
        async with self.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2", self.owner, old.record.payload_id)
        self.assertEqual(self.case.capture.cached_request(old.input_id), old.record)
        result = await self.cancel_on(followup, fresh_old.reference, "call_withdrawn_source")
        self.assertEqual(result["status"], "failed")
        await self.assert_no_mutation()
        await self.assert_latest_unchanged(current)
        with self.assertRaises(TaskError):
            await self.context.request_evidence(self.owner, self.session,
                source_namespace=old.namespace, source_event_id=old.source_event_id)
        self.receipt(status="passed", error_code=result["error"]["code"], withdrawn_payload_id=str(old.record.payload_id),
                     mutation_count=0, latest_preserved=True)


def load_tests(loader, tests, pattern):
    # Extend, do not rerun/replace the four preserved original SQL cases.
    return unittest.TestSuite(UnselectedContinuationSQLTests(name) for name in (
        "test_continuation_current_read_then_old_cancel_commits_once",
        "test_continuation_missing_foreign_and_old_response_selectors_have_no_effect",
        "test_continuation_corrected_canonical_source_rejects_old_cache_and_selector",
        "test_continuation_withdrawn_canonical_payload_cannot_cancel",
    ))
