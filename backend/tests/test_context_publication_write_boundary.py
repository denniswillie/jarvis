"""Deterministic control boundaries; no provider, native, service or DB calls."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.voice_routes.bridge import OfferDisposition, VoiceContextBridge
from app.voice_routes.contracts import ContextWriteEvidence
from test_realtime_voice_route import NOW, FakeControl, make_adapter, make_context, make_start, stop_route


class ContextPublicationWriteBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.control = FakeControl()
        self.route = make_adapter(self.control)
        self.request = make_start()
        self.assertEqual((await self.route.start(self.request)).status, "completed")
        self.previous_valid = self.candidate_valid = True
        self.validated, self.failures = [], []
        self.bridges = []
        logger = patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning")
        self.log = logger.start()
        self.addCleanup(logger.stop)

    async def asyncTearDown(self):
        self.control.guard_release.set()
        for bridge in self.bridges:
            await asyncio.wait_for(bridge.close(), 1)
        await stop_route(self.route)

    async def validity(self, update):
        self.validated.append(update.cursor.revision)
        return self.previous_valid if update == self.request.initial_context else self.candidate_valid

    async def begin(self, *, route=None, validity=None):
        bridge = VoiceContextBridge(route or self.route, self.request.initial_context,
            validity=validity or self.validity, publish_timeout=0.03, clock=lambda: NOW,
            on_failure=self.failures.append)
        self.bridges.append(bridge)
        await bridge.start()
        return bridge

    async def until(self, predicate):
        async with asyncio.timeout(1):
            while not predicate():
                await asyncio.sleep(0)

    def writes(self, kind):
        return [event for event in self.control.writes if event["type"] == kind]

    def diagnostics(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    async def test_evidence_defaults_unknown_and_possible_write_is_irreversible(self):
        evidence = ContextWriteEvidence()
        evidence.close()
        evidence.mark_joined()
        self.assertEqual(evidence.state, "unknown")
        self.assertFalse(evidence.proven_unwritten)
        self.assertFalse(evidence.before_write("context_create"))
        evidence = ContextWriteEvidence()
        evidence.begin_tracking()
        self.assertEqual(evidence.state, "prewrite")
        self.assertTrue(evidence.before_write("context_delete"))
        evidence.close()
        evidence.mark_joined()
        evidence.begin_tracking()
        self.assertEqual(evidence.state, "write_possible")
        self.assertEqual(evidence.first_operation, "context_delete")
        self.assertFalse(evidence.proven_unwritten)
        self.assertFalse(evidence.before_write("context_create"))

    async def test_cancelled_writer_is_joined_before_same_update_can_retry(self):
        entered, cancelled, release, finished = (asyncio.Event() for _ in range(4))
        original = self.control.send_event

        async def sender(event, **kwargs):
            entered.set()
            try:
                return await original(event, **kwargs)
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                raise
            finally:
                finished.set()

        self.control.send_event = sender
        await self.control.lock.acquire()
        bridge = await self.begin()
        update = make_context(revision=1)
        try:
            bridge.offer(update)
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(cancelled.wait(), 1)
            self.assertFalse(finished.is_set())
            self.assertEqual(bridge.offer(update).disposition, OfferDisposition.DUPLICATE)
            self.assertEqual(len(self.control.writes), 1)
            release.set()
            await self.until(lambda: bridge.status.reason == "context_publication_deferred_unwritten")
            self.assertTrue(finished.is_set())
            self.assertFalse(any(not task.done() for task in self.route._data_tasks))
            self.assertTrue(self.route.can_retain_context(self.request.initial_context))
            self.assertFalse(bridge.status.fenced)
            self.assertIn(0, self.validated)
            self.control.lock.release()
            self.assertEqual(bridge.offer(update).disposition, OfferDisposition.QUEUED)
            await self.until(lambda: bridge.status.last_receipt is not None)
            self.assertEqual(bridge.status.last_receipt.status, "written")
            self.assertEqual(len(self.writes("conversation.item.delete")), 1)
            self.assertEqual(len(self.writes("conversation.item.create")), 2)
            self.assertEqual(bridge.offer(update).disposition, OfferDisposition.DUPLICATE)
            self.assertEqual(self.failures, [])
        finally:
            release.set()
            if self.control.lock.locked():
                self.control.lock.release()

    async def test_three_explicit_unwritten_retries_have_only_one_final_replacement(self):
        bridge = await self.begin()
        update = make_context(revision=1)
        await self.control.lock.acquire()
        try:
            for attempt in range(3):
                self.assertEqual(bridge.offer(update).disposition, OfferDisposition.QUEUED)
                await self.until(lambda: len(self.diagnostics("publication_deferred")) == attempt + 1)
                self.assertEqual(len(self.control.writes), 1)
                self.assertFalse(bridge.status.fenced)
            self.control.lock.release()
            bridge.offer(update)
            await self.until(lambda: bridge.status.last_receipt is not None)
            self.assertEqual(len(self.writes("conversation.item.delete")), 1)
            self.assertEqual(len(self.writes("conversation.item.create")), 2)
        finally:
            if self.control.lock.locked():
                self.control.lock.release()

    async def test_superseded_unwritten_publication_coalesces_to_latest_only(self):
        bridge = await self.begin()
        await self.control.lock.acquire()
        try:
            bridge.offer(make_context(revision=1))
            await self.until(lambda: bool(self.route._data_tasks))
            bridge.offer(make_context(revision=2))
            latest = make_context(revision=3)
            bridge.offer(latest)
            await self.until(lambda: 3 in self.validated)
            self.control.lock.release()
            await self.until(lambda: bridge.status.last_receipt is not None)
            self.assertEqual(bridge.status.last_receipt.cursor, latest.cursor)
            self.assertEqual(len(self.writes("conversation.item.delete")), 1)
            self.assertEqual(len(self.writes("conversation.item.create")), 2)
        finally:
            if self.control.lock.locked():
                self.control.lock.release()

    async def test_source_change_in_preflight_defers_only_with_valid_previous_context(self):
        bridge = await self.begin()
        self.candidate_valid = False
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.reason == "context_publication_deferred_unwritten")
        self.assertEqual(len(self.control.writes), 1)
        self.assertFalse(bridge.status.fenced)
        self.previous_valid = False
        bridge.offer(make_context(revision=2))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.control.writes), 1)
        self.assertEqual(self.failures, ["retained_context_unavailable_before_retry"])

    async def test_changed_source_during_guard_after_delete_must_fence(self):
        update = make_context(revision=1)
        self.control.block_id = update.update_id
        bridge = await self.begin()
        bridge.offer(update)
        await asyncio.wait_for(self.control.guard_entered.wait(), 1)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.control.valid = False
        self.control.guard_release.set()
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.writes("conversation.item.create")), 1)
        self.assertEqual(bridge.offer(make_context(revision=2)).disposition, OfferDisposition.REJECTED)
        self.assertEqual(self.diagnostics("publication_deferred"), [])
        row = self.diagnostics("adapter_return")[-1]
        self.assertEqual(row["write_state"], "write_possible")
        self.assertEqual(row["first_possible_operation"], "context_delete")

    async def test_delete_ack_loss_keeps_fence_and_never_replays_delete(self):
        self.control.item_event = None
        bridge = await self.begin()
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(len(self.writes("conversation.item.create")), 1)
        self.assertIsNone(bridge.status.last_receipt)
        self.assertEqual(self.diagnostics("publication_deferred"), [])
        self.assertEqual(self.diagnostics("publication_deadline")[0]["write_state"], "write_possible")

    async def test_local_create_return_is_not_provider_ack_and_cannot_justify_deferral(self):
        def only_delete_ack(event):
            if event["type"] == "conversation.item.deleted":
                self.route.observe_item_event(self.route.identity, event)
        self.control.item_event = only_delete_ack
        bridge = await self.begin()
        update = make_context(revision=1)
        bridge.offer(update)
        await self.until(lambda: bridge.status.last_receipt is not None)
        self.assertEqual(bridge.status.last_receipt.status, "written")
        self.assertFalse(self.route.can_retain_context(update))
        bridge.offer(make_context(revision=2))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(len(self.writes("conversation.item.create")), 2)
        self.assertEqual(self.diagnostics("publication_deferred"), [])

    async def test_lost_local_return_after_create_is_unknown_not_retryable(self):
        original = self.control._write
        async def lost_return(event):
            result = await original(event)
            if self.control.writes[-1]["type"] == "conversation.item.create":
                raise OSError("PRIVATE_WRITE_ERROR")
            return result
        self.control._write = lost_return
        bridge = await self.begin()
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        self.assertIsNone(bridge.status.last_receipt)
        self.assertEqual(len(self.writes("conversation.item.create")), 2)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(self.diagnostics("publication_deferred"), [])
        self.assertNotIn("PRIVATE_WRITE_ERROR", str(self.log.call_args_list))

    async def test_unjoined_child_fences_and_its_late_factory_cannot_write(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.control.send_event
        async def stubborn(event, **kwargs):
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return await original(event, **kwargs)
        self.control.send_event = stubborn
        bridge = await self.begin()
        try:
            bridge.offer(make_context(revision=1))
            await asyncio.wait_for(entered.wait(), 1)
            await self.until(lambda: bridge.status.fenced)
            self.assertEqual(len(self.control.writes), 1)
            self.assertEqual(self.diagnostics("publication_deferred"), [])
            self.assertTrue(any(not task.done() for task in self.route._data_tasks))
            release.set()
            await self.until(lambda: not any(not task.done() for task in self.route._data_tasks))
            self.assertEqual(len(self.control.writes), 1)
        finally:
            release.set()

    async def test_uninstrumented_adapter_timeout_retains_unknown_fail_closed(self):
        async def publish(_):
            await asyncio.Event().wait()
        route = SimpleNamespace(identity=self.route.identity, publish_context=publish)
        bridge = await self.begin(route=route)
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(self.diagnostics("publication_deadline")[0]["write_state"], "unknown")
        self.assertEqual(self.diagnostics("publication_deferred"), [])

    async def test_real_control_writer_marks_only_after_guard_and_before_socket(self):
        events, entered, release = [], asyncio.Event(), asyncio.Event()
        class Socket:
            async def send(self, data):
                events.append(("wire", json.loads(data)["type"]))
        writer = OpenAIRealtimeConnection(
            settings=SimpleNamespace(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=1), owner_id="fixture")
        writer._foreground_waiters = 0
        writer._send_lock = asyncio.Lock()
        writer._websocket = Socket()
        evidence = ContextWriteEvidence()
        evidence.begin_tracking()
        @asynccontextmanager
        async def guard():
            entered.set()
            await release.wait()
            yield
        def payload():
            if not evidence.before_write("context_create"):
                return None
            events.append(("possible", evidence.state))
            return {"type": "conversation.item.create"}
        task = asyncio.create_task(writer.send_event(payload, write_guard=guard))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(evidence.state, "prewrite")
        evidence.close()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        evidence.mark_joined()
        self.assertTrue(evidence.proven_unwritten)
        self.assertEqual(events, [])
        self.assertEqual(writer._foreground_waiters, 0)
        evidence = ContextWriteEvidence()
        evidence.begin_tracking()
        release.set()
        self.assertTrue(await writer.send_event(payload, write_guard=guard))
        self.assertEqual(events, [("possible", "write_possible"), ("wire", "conversation.item.create")])
        evidence.close()
        evidence.mark_joined()
        self.assertFalse(evidence.proven_unwritten)


if __name__ == "__main__":
    unittest.main()
