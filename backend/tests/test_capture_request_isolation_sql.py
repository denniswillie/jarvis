"""Manifested real SQL/task-tool qualification; capture faults are controlled."""
import os
import unittest

from app.tasks.models import TaskStatus
import test_realtime_unwritten_request_sources_sql as base


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested SQL fixture required",
)
class RequestCaptureIsolationSQLTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = base.UnwrittenRequestSourceSQLTests.asyncSetUp
    open_case = base.UnwrittenRequestSourceSQLTests.open_case

    async def cancel(self, call):
        return await self.case.voice.tool_executor.execute("cancel_background_task", {
            "task_ref": self.tasks.reference(self.task.id), "expected_revision": self.task.revision,
        }, provider_call_id=call)

    async def test_unrelated_gap_preserves_exact_durable_cancel_authority(self):
        capture = self.case.capture
        capture.write_timeout = 3.0
        append = self.context.append

        async def fail_optional(owner, event):
            if event.kind == "observation":
                raise OSError("Controlled optional capture failure before SQL")
            return await append(owner, event)

        self.context.append = fail_optional
        capture._put("optional-view", kind="observation", trust="sensor", payload={"text": "fixture view"})
        self.assertFalse(await capture.flush())
        capture.input_final("cancel-input", "Cancel Workbench Plan.")
        capture.bind_call("cancel-call", "cancel-input")
        request = await capture.request_for_call("cancel-call")
        self.assertIsNotNone(request)
        self.assertEqual(request.content["capture_input_version"], 1)
        self.assertEqual(request.owner_id, self.owner)
        self.assertEqual(request.session_id, self.session)
        self.assertIsNotNone(capture.issue)
        self.assertIsNone(capture.request_authority_issue)
        result = await self.cancel("cancel-call")
        self.assertEqual(result["status"], "success")
        current = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual(current.status, TaskStatus.CANCELLED)
        self.assertEqual((current.request_event_id, current.request_revision), (request.id, request.revision))
        self.assertEqual(len(self.mutations), 1)
        await capture.flush()
        async with self.pool.acquire() as connection:
            self.assertGreater(await connection.fetchval(
                "select count(*) from jarvis_context_events where owner_id=$1 and session_id=$2 and kind='context_gap'",
                self.owner, self.session), 0)

    async def test_failed_a_b_a_cannot_cancel_using_old_committed_a(self):
        capture = self.case.capture
        capture.write_timeout = 3.0
        capture.input_final("input", "Cancel Workbench Plan.")
        capture.bind_call("unused-original-call", "input")
        original = await capture.request_for_call("unused-original-call")
        self.assertIsNotNone(original)
        append = self.context.append

        async def fail_corrections(owner, event):
            if event.kind == "user_transcript":
                raise OSError("Controlled failed correction before SQL")
            return await append(owner, event)

        self.context.append = fail_corrections
        for text in ("Do not cancel Workbench Plan.", "Cancel Workbench Plan."):
            capture.input_final("input", text)
            self.assertFalse(await capture.flush())
        durable = await self.context.request_evidence(self.owner, self.session,
            source_namespace=capture.namespace, source_event_id=capture._source("input", "input"))
        self.assertEqual((durable.id, durable.revision), (original.id, original.revision))
        self.assertEqual(durable.content["capture_input_version"], 1)
        self.assertEqual(capture._input_versions["input"], 3)
        capture.bind_call("stale-call", "input")
        self.assertIsNone(await capture.request_for_call("stale-call"))
        result = await self.cancel("stale-call")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "task_request_not_finalized")
        current = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual((current.status, current.revision), (self.task.status, self.task.revision))
        self.assertEqual(self.mutations, [])


if __name__ == "__main__":
    unittest.main()
