"""Real bridge/adapter, controlled ACKs; no provider, database or native calls."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.knowledge.models import SourceError
from app.voice_routes.bridge import OfferDisposition, VoiceContextBridge
from app.voice_routes.contracts import ContextWriteEvidence
from test_realtime_voice_route import NOW, FakeControl, make_adapter, make_context, make_start, stop_route


class ContextPublicationPhaseBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = NOW
        self.control = FakeControl()
        self.route = make_adapter(self.control, clock=lambda: self.now)
        self.route._io_timeout = 1
        self.request = make_start()
        await self.route.start(self.request)
        self.bridges, self.ack_tasks, self.failures = [], [], []
        self.previous_valid = self.candidate_valid = True
        self.validity_delay = 0
        self.log = self.enterContext(patch("app.voice_routes.bridge._PUBLICATION_LOGGER.warning"))

    async def asyncTearDown(self):
        self.control.guard_release.set()
        for bridge in self.bridges:
            await asyncio.wait_for(bridge.close(), 2)
        for task in self.ack_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.ack_tasks, return_exceptions=True)
        await asyncio.wait_for(stop_route(self.route), 2)

    async def validity(self, update):
        if update == self.request.initial_context:
            return self.previous_valid
        await asyncio.sleep(self.validity_delay)
        return self.candidate_valid

    async def begin(self, *, timeout=.15, validity=None, route=None):
        bridge = VoiceContextBridge(route or self.route, self.request.initial_context,
            validity=validity or self.validity, publish_timeout=timeout,
            clock=lambda: self.now, on_failure=self.failures.append, preflight_expiry_refresh=True)
        self.bridges.append(bridge)
        await bridge.start()
        return bridge

    async def until(self, predicate):
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(0)

    def records(self, stage):
        return [row for call in self.log.call_args_list
                if (row := json.loads(call.args[1]))["stage"] == stage]

    def writes(self, kind):
        return [event for event in self.control.writes if event["type"] == kind]

    def delayed_delete_ack(self, delay, *, before_ack=None):
        def observe(event):
            if event["type"] != "conversation.item.deleted":
                self.route.observe_item_event(self.route.identity, event)
                return

            async def later():
                await asyncio.sleep(delay)
                if before_ack is not None:
                    before_ack()
                self.route.observe_item_event(self.route.identity, event)
            self.ack_tasks.append(asyncio.create_task(later()))
        self.control.item_event = observe

    async def test_slow_preparation_and_queue_do_not_spend_mutation_ack_budget(self):
        self.validity_delay = .06

        @asynccontextmanager
        async def preparation(_):
            await asyncio.sleep(.025)
            yield True
        self.route._bindings = replace(self.route._bindings, context_replacement_guard=preparation)
        original_send = self.control.send_event

        async def queued(event, **kwargs):
            await asyncio.sleep(.025)
            return await original_send(event, **kwargs)
        self.control.send_event = queued
        self.delayed_delete_ack(.075)
        bridge = await self.begin()
        candidate = make_context(revision=1)
        bridge.offer(candidate)
        await self.until(lambda: bridge.status.last_receipt is not None or bridge.status.fenced)
        self.assertFalse(bridge.status.fenced)
        self.assertEqual(bridge.status.last_receipt.status, "written")
        self.assertEqual(self.route._context, candidate)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(len(self.writes("conversation.item.create")), 2)
        self.assertEqual(len(self.records("mutation_budget_entry")), 1)
        row = self.records("adapter_return")[-1]
        self.assertGreater(row["elapsed_ms"], row["preparation_budget_ms"])
        self.assertLess(row["elapsed_ms"], row["preparation_budget_ms"] + row["mutation_budget_ms"])
        self.assertTrue(row["deleted_context_acknowledged"])
        self.assertTrue(row["owned_work_joined"])
        self.assertEqual(self.records("publication_deadline"), [])
        self.assertEqual(bridge.offer(candidate).disposition, OfferDisposition.DUPLICATE)

    async def test_hung_preflight_keeps_original_bound_and_joins_without_write(self):
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def validity(update):
            if update == self.request.initial_context:
                return True
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        bridge = await self.begin(timeout=.03, validity=validity)
        bridge.offer(make_context(revision=1))
        await asyncio.wait_for(entered.wait(), 1)
        await self.until(lambda: bridge.status.reason == "context_publication_deferred_unwritten")
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(self.control.writes), 1)
        self.assertFalse(bridge.status.fenced)
        self.assertEqual(self.records("mutation_budget_entry"), [])
        self.assertEqual(self.records("publication_deadline")[0]["budget_phase"], "preparation")
        self.assertFalse(any(not job.done() for job in bridge._jobs))

    async def test_missing_ack_is_bounded_unknown_and_never_replayed(self):
        self.validity_delay = .02
        self.control.item_event = None
        bridge = await self.begin(timeout=.05)
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        row = self.records("publication_fenced")[-1]
        self.assertEqual(row["budget_phase"], "mutation")
        self.assertEqual(row["write_state"], "write_possible")
        self.assertTrue(row["owned_work_joined"])
        self.assertFalse(row["deleted_context_acknowledged"])
        self.assertFalse(row["replacement_create_possible"])
        self.assertIsNone(bridge.status.last_receipt)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(len(self.writes("conversation.item.create")), 1)
        item = self.writes("conversation.item.delete")[0]["item_id"]
        self.route.observe_item_event(self.route.identity, {"type": "conversation.item.deleted", "item_id": item})
        self.assertTrue(bridge.status.fenced)
        self.assertEqual(bridge.offer(make_context(revision=2)).disposition, OfferDisposition.REJECTED)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)

    async def test_delete_ack_does_not_reset_budget_for_blocked_create_guard(self):
        self.delayed_delete_ack(.035)
        candidate = make_context(revision=1)
        self.control.block_id = candidate.update_id
        bridge = await self.begin(timeout=.06)
        bridge.offer(candidate)
        await asyncio.wait_for(self.control.guard_entered.wait(), 1)
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.records("mutation_budget_entry")), 1)
        row = self.records("publication_fenced")[-1]
        self.assertEqual(row["budget_phase"], "mutation")
        self.assertTrue(row["deleted_context_acknowledged"])
        self.assertFalse(row["replacement_create_possible"])
        self.assertEqual(len(self.writes("conversation.item.create")), 1)

    async def test_candidate_expiring_in_writer_queue_stays_unwritten(self):
        original = self.control.send_event

        async def queued(event, **kwargs):
            self.now = NOW + timedelta(seconds=2)
            return await original(event, **kwargs)
        self.control.send_event = queued
        bridge = await self.begin()
        bridge.offer(make_context(revision=1, expires_at=NOW + timedelta(seconds=1)))
        await self.until(lambda: bridge.status.reason == "context_expired_awaiting_snapshot" or bridge.status.fenced)
        self.assertFalse(bridge.status.fenced)
        self.assertEqual(self.records("mutation_budget_entry"), [])
        self.assertEqual(len(self.control.writes), 1)
        self.assertEqual(self.records("adapter_return")[-1]["write_state"], "unwritten")
        self.assertTrue(self.records("adapter_return")[-1]["owned_work_joined"])

    async def test_expiry_during_preflight_never_starts_mutation_budget(self):
        async def validity(_):
            self.now = NOW + timedelta(seconds=2)
            return True
        bridge = await self.begin(validity=validity)
        bridge.offer(make_context(revision=1, expires_at=NOW + timedelta(seconds=1)))
        await self.until(lambda: bridge.awaiting_fresh_context)
        self.assertEqual(self.records("mutation_budget_entry"), [])
        self.assertEqual(len(self.control.writes), 1)

    async def test_correction_or_withdrawal_before_delete_still_fences(self):
        for code in ("live_state_source_changed", "live_state_context_revoked"):
            with self.subTest(code=code):
                @asynccontextmanager
                async def guard(_):
                    raise SourceError(code, "private withdrawn source")
                    yield
                self.route._bindings = replace(self.route._bindings, context_replacement_guard=guard)
                self.previous_valid = False
                bridge = await self.begin()
                bridge.offer(make_context(revision=1))
                await self.until(lambda: bridge.status.fenced)
                self.assertEqual(len(self.control.writes), 1)
                self.assertEqual(self.records("mutation_budget_entry"), [])
                await bridge.close()

    async def test_withdrawal_after_delete_ack_cannot_authorize_create(self):
        self.delayed_delete_ack(.02, before_ack=lambda: setattr(self.control, "valid", False))
        bridge = await self.begin()
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        self.assertEqual(len(self.writes("conversation.item.delete")), 1)
        self.assertEqual(len(self.writes("conversation.item.create")), 1)
        self.assertTrue(self.records("adapter_return")[-1]["deleted_context_acknowledged"])
        self.assertFalse(self.records("adapter_return")[-1]["replacement_create_possible"])

    async def test_explicit_close_joins_ack_wait_and_prevents_late_factory(self):
        self.control.item_event = None
        bridge = await self.begin()
        bridge.offer(make_context(revision=1))
        await self.until(lambda: len(self.writes("conversation.item.delete")) == 1)
        await asyncio.wait_for(bridge.close(), 1)
        self.assertFalse(any(not job.done() for job in bridge._jobs))
        self.assertFalse(any(not task.done() for task in self.route._data_tasks))
        self.assertEqual(len(self.writes("conversation.item.create")), 1)
        self.assertIsNone(bridge.status.last_receipt)

    async def test_uninstrumented_path_has_only_original_deadline(self):
        async def publish(_):
            await asyncio.Event().wait()
        route = SimpleNamespace(identity=self.route.identity, publish_context=publish)
        bridge = await self.begin(route=route, timeout=.03)
        bridge.offer(make_context(revision=1))
        await self.until(lambda: bridge.status.fenced)
        row = self.records("publication_deadline")[-1]
        self.assertEqual(row["budget_phase"], "preparation")
        self.assertEqual(row["write_state"], "unknown")
        self.assertIsNone(row["mutation_started_monotonic_s"])

    def test_budget_veto_cannot_erase_prior_uncertain_write_or_reopen_closed_work(self):
        allowed = True
        evidence = ContextWriteEvidence(write_budget=lambda: allowed)
        evidence.begin_tracking()
        self.assertTrue(evidence.before_write("context_delete", item_id="old"))
        allowed = False
        self.assertFalse(evidence.before_write("context_create", item_id="new"))
        evidence.mark_joined()
        self.assertEqual(evidence.state, "write_possible")
        self.assertFalse(evidence.deleted_context_acknowledged)
        self.assertFalse(evidence.proven_unwritten)
        allowed = True
        self.assertFalse(evidence.before_write("context_create", item_id="late"))
        unwritten = ContextWriteEvidence(write_budget=lambda: False)
        unwritten.begin_tracking()
        self.assertFalse(unwritten.before_write("context_delete"))
        unwritten.mark_joined()
        self.assertTrue(unwritten.proven_unwritten)
