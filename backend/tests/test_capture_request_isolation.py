"""Exact request authority must not require a healthy unrelated capture queue."""
import asyncio
import unittest
from uuid import uuid4

from app.tasks.context_capture import SessionContextCapture
from app.tasks.models import TaskError
from test_realtime_unwritten_request_sources import CanonicalStore, Sources


class RequestCaptureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = CanonicalStore(Sources())
        self.capture = SessionContextCapture(self.store, owner_id="isolated-request", session_id=uuid4(),
            provider="openai", audio_session_id="request-isolation", write_timeout=0.3)
        self.addAsyncCleanup(self.capture.close)

    def observation(self, name):
        self.capture._put(name, kind="observation", trust="sensor", payload={"text": name})

    def request(self, text="Actual user request", identity="input", call="call"):
        self.capture.input_final(identity, text)
        self.capture.bind_call(call, identity)

    async def test_unrelated_storage_gap_does_not_revoke_durable_request(self):
        append = self.store.append

        async def fail_observation(owner, event):
            if event.kind == "observation":
                raise OSError("controlled optional capture failure")
            return await append(owner, event)

        self.store.append = fail_observation
        self.observation("failed-view")
        self.assertFalse(await self.capture.flush())
        self.request()
        record = await self.capture.request_for_call("call")
        self.assertTrue(record.authorizes_request)
        self.assertIsNotNone(self.capture.issue)
        self.assertIsNone(self.capture.request_authority_issue)
        self.assertEqual(self.capture.cached_request("input"), record)
        await self.capture.flush()
        self.assertTrue(any(row.kind == "context_gap" for row in self.store.rows.values()))

    async def test_final_input_precedes_optional_backlog_and_waits_only_for_itself(self):
        entered, first_release, optional_entered, optional_release = (asyncio.Event() for _ in range(4))
        append = self.store.append
        order = []

        async def blocked(owner, event):
            if event.source_event_id == "first-view":
                entered.set()
                await first_release.wait()
            elif event.kind == "observation":
                optional_entered.set()
                await optional_release.wait()
            order.append(event.kind)
            return await append(owner, event)

        self.store.append = blocked
        self.observation("first-view")
        await entered.wait()
        for i in range(4):
            self.observation(f"optional-{i}")
        self.request()
        waiter = asyncio.create_task(self.capture.request_for_call("call"))
        try:
            first_release.set()
            record = await asyncio.wait_for(waiter, 0.2)
            await asyncio.wait_for(optional_entered.wait(), 0.2)
            self.assertTrue(record.authorizes_request)
            self.assertEqual(order[:2], ["observation", "user_transcript"])
            self.assertFalse(optional_release.is_set())
        finally:
            first_release.set()
            optional_release.set()
            await asyncio.gather(waiter, return_exceptions=True)

    async def test_request_write_failure_does_not_authorize_or_retry(self):
        append = self.store.append
        writes = []

        async def fail_request(owner, event):
            if event.kind == "user_transcript":
                writes.append(event)
                raise OSError("controlled request failure")
            return await append(owner, event)

        self.store.append = fail_request
        self.request()
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.assertEqual(len(writes), 1)

    async def test_failed_a_b_a_revisions_cannot_reuse_original_a(self):
        self.request("A")
        original = await self.capture.request_for_call("call")
        self.assertIsNotNone(original)
        append = self.store.append

        async def fail_revisions(owner, event):
            if event.kind == "user_transcript":
                raise OSError("controlled correction failure")
            return await append(owner, event)

        self.store.append = fail_revisions
        for text in ("B", "A"):
            self.capture.input_final("input", text)
            await self.capture.flush()
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.assertIsNone(self.capture.cached_request("input"))

    async def test_identity_conflict_stays_fenced_after_later_io_failure(self):
        self.request()
        self.assertIsNotNone(await self.capture.request_for_call("call"))
        self.capture.bind_call("call", "different-input")
        self.capture._failure("context_capture_persistence_failed")
        self.assertEqual(self.capture.request_authority_issue, "context_call_identity_conflict")
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.assertIsNone(self.capture.cached_request("input"))

    async def test_repository_finality_error_is_not_downgraded_to_storage_gap(self):
        append = self.store.append

        async def conflict(owner, event):
            if event.kind == "user_transcript":
                raise TaskError("context_finality_conflict", "Controlled finality failure")
            return await append(owner, event)

        self.store.append = conflict
        self.request()
        self.assertIsNone(await self.capture.request_for_call("call"))
        self.assertEqual(self.capture.request_authority_issue, "context_finality_conflict")

    async def test_cancelling_waiter_does_not_cancel_owned_writer(self):
        entered, release = asyncio.Event(), asyncio.Event()
        append = self.store.append

        async def blocked(owner, event):
            if event.kind == "user_transcript":
                entered.set()
                await release.wait()
            return await append(owner, event)

        self.store.append = blocked
        self.request()
        await entered.wait()
        waiter = asyncio.create_task(self.capture.request_for_call("call"))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        self.assertFalse(self.capture._writer.done())
        release.set()
        self.assertIsNotNone(await self.capture.request_for_call("call"))

    async def test_revision_change_during_lookup_cannot_change_call_authority(self):
        self.request("Original")
        self.assertIsNotNone(await self.capture.request_for_call("call"))
        lookup = self.store.request_evidence

        async def revise(*args, **kwargs):
            record = await lookup(*args, **kwargs)
            self.capture.input_final("input", "Correction")
            return record

        self.store.request_evidence = revise
        self.assertIsNone(await self.capture.request_for_call("call"))

    async def test_foreign_or_unversioned_durable_record_is_rejected(self):
        self.request()
        original = await self.capture.request_for_call("call")
        for changes in ({"owner_id": "foreign"}, {"session_id": uuid4()},
                        {"source_namespace": "foreign"}, {"is_current": False},
                        {"content": {**original.content, "capture_input_version": True}},
                        {"content": {k: v for k, v in original.content.items() if k != "capture_input_version"}}):
            with self.subTest(changes=changes.keys()):
                async def wrong(*args, **kwargs):
                    return original.model_copy(update=changes)
                self.store.request_evidence = wrong
                self.assertIsNone(await self.capture.request_for_call("call"))


if __name__ == "__main__":
    unittest.main()
