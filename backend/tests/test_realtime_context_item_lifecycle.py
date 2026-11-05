"""Offline owned-item receipts: no provider, app, database or native execution."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import SourceRetirementCommand
from tests.test_realtime_voice_route import (
    FakeControl, NOW, make_adapter, make_context, make_identity, make_offer,
    make_start, stop_route,
)


class ContextItemLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.control = FakeControl()
        self.route = make_adapter(self.control)
        self.control.item_event = None
        self.assertEqual((await self.route.start(make_start())).status, "completed")
        self.initial = self.creates()[0]

    async def asyncTearDown(self):
        self.control.guard_release.set()
        await stop_route(self.route)

    def creates(self):
        return [e for e in self.control.writes if e["type"] == "conversation.item.create"]

    def deletes(self):
        return [e for e in self.control.writes if e["type"] == "conversation.item.delete"]

    def observe(self, event):
        self.route.observe_item_event(self.route.identity, event)

    def ack(self, event, kind="conversation.item.done"):
        self.observe({"type": kind, "item": event["item"]})

    def reject(self, event, code="string_above_max_length"):
        self.observe({"type": "error", "error": {
            "event_id": event["event_id"], "code": code,
            "message": "sensitive arbitrary text must not be logged"}})

    def auto_ack(self, event):
        self.observe(event)

    async def wait_until(self, predicate):
        async with asyncio.timeout(0.5):
            while not predicate():
                await asyncio.sleep(0)

    async def test_context_and_offer_ids_use_native_32_character_shape(self):
        await self.route.offer_notification(make_offer())
        ids = [event["item"]["id"] for event in self.creates()]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)
        for item_id in ids:
            self.assertRegex(item_id, r"^[0-9a-f]{32}$")

    async def test_missing_ack_never_speculatively_deletes_or_adds_replacement(self):
        receipt = await self.route.publish_context(make_context(revision=1))
        self.assertEqual(receipt.status, "unknown")
        self.assertEqual(self.deletes(), [])
        self.assertEqual(len(self.creates()), 1)

    async def test_added_is_not_final_and_foreign_events_do_not_acknowledge(self):
        self.ack(self.initial, "conversation.item.added")
        self.ack({"item": {**self.initial["item"], "id": "foreign"}})
        self.ack({"item": {**self.initial["item"], "role": "assistant"}})
        self.observe({"type": "error", "error": {"event_id": "foreign"}})
        with self.assertRaises(Exception):
            self.route.observe_item_event(make_identity(2), {
                "type": "conversation.item.done", "item": self.initial["item"]})
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "unknown")
        self.assertFalse(self.deletes())

    async def test_rejected_create_is_not_deleted_and_replacement_can_be_written(self):
        with self.assertLogs("app.voice_routes.realtime_adapter", "WARNING") as logs:
            self.reject(self.initial)
        self.assertNotIn("sensitive", " ".join(logs.output))
        self.assertIn(self.initial["event_id"], " ".join(logs.output))
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "written")
        self.assertFalse(self.deletes())
        self.assertEqual(len(self.creates()), 2)

    async def test_late_final_ack_releases_background_wait_without_duplicate_delete(self):
        publication = asyncio.create_task(self.route.publish_context(make_context(revision=1)))
        await asyncio.sleep(0)
        self.assertFalse(self.deletes())
        self.ack(self.initial)
        await self.wait_until(lambda: len(self.deletes()) == 1)
        self.ack(self.initial)  # A late duplicate must not reset delete_pending.
        self.observe({"type": "conversation.item.deleted", "item_id": "foreign"})
        self.assertFalse(publication.done())
        deleted = {"type": "conversation.item.deleted", "item_id": self.initial["item"]["id"]}
        self.observe(deleted)
        self.observe(deleted)
        self.assertEqual((await publication).status, "written")
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(len(self.creates()), 2)

    async def test_legacy_created_ack_is_supported(self):
        self.ack(self.initial, "conversation.item.created")
        self.control.item_event = self.auto_ack
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "written")
        self.assertEqual(len(self.deletes()), 1)

    async def test_delete_rejection_is_not_success_or_blindly_retried(self):
        self.ack(self.initial)
        publication = asyncio.create_task(self.route.publish_context(make_context(revision=1)))
        await self.wait_until(lambda: bool(self.deletes()))
        self.reject(self.deletes()[0], "item_delete_invalid_item_id")
        self.assertEqual((await publication).status, "unknown")
        self.assertEqual((await self.route.publish_context(make_context(revision=2))).status, "unknown")
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(len(self.creates()), 1)

    async def test_missing_delete_ack_is_not_retried_or_assumed_complete(self):
        self.ack(self.initial)
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "unknown")
        self.assertEqual((await self.route.publish_context(make_context(revision=2))).status, "unknown")
        self.assertEqual(len(self.deletes()), 1)
        self.assertEqual(len(self.creates()), 1)

    async def test_guard_rejection_does_not_reserve_a_phantom_item(self):
        self.reject(self.initial)
        self.control.valid = False
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "unknown")
        self.assertFalse(self.route._items)
        self.assertFalse(self.deletes())

    async def test_ambiguous_create_retains_identity_and_requires_real_ack(self):
        self.reject(self.initial)
        original = self.control.send_event

        async def lost_write_receipt(event, **kwargs):
            await original(event, **kwargs)
            raise OSError("write outcome unknown")

        self.control.send_event = lost_write_receipt
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "unknown")
        self.control.send_event = original
        self.assertEqual((await self.route.publish_context(make_context(revision=2))).status, "unknown")
        self.assertEqual(len(self.creates()), 2)
        self.assertFalse(self.deletes())
        self.ack(self.creates()[-1])
        self.control.item_event = self.auto_ack
        self.assertEqual((await self.route.publish_context(make_context(revision=3))).status, "written")
        self.assertEqual(len(self.deletes()), 1)

    async def test_retirement_does_not_delete_nonacked_or_rejected_items(self):
        await self.route.offer_notification(make_offer())
        self.reject(self.creates()[-1])
        receipt = await self.route.retire_context(SourceRetirementCommand(
            route=self.route.identity, command_id=uuid4(), requested_at=NOW,
            cursor=make_context().cursor, sources=(make_offer().attempt.source,), reason="fixture"))
        self.assertEqual(receipt.status, "reconnect_required")
        self.assertFalse(self.deletes())

    async def test_immediate_provider_rejection_does_not_report_written(self):
        self.reject(self.initial)

        def reject_new(event):
            if event["type"] == "conversation.item.done":
                self.reject(self.creates()[-1])

        self.control.item_event = reject_new
        self.assertEqual((await self.route.publish_context(make_context(revision=1))).status, "rejected")
        self.assertFalse(self.deletes())

    async def test_runtime_observes_item_ack_before_owner_without_response_admission(self):
        owner = AsyncMock()
        runtime = SimpleNamespace(adapter=self.route, request=make_start(),
            session=SimpleNamespace(_active_responses={}), _owner_provider_handler=owner,
            _response_offers={}, _response_dispatch=lambda _: None)
        ack = {"type": "conversation.item.done", "item": self.initial["item"]}

        async def check_observed(payload):
            self.assertEqual(self.route._items[self.initial["item"]["id"]].state, "created")

        owner.side_effect = check_observed
        await RealtimeVoiceRouteRuntime._provider_event(runtime, ack)
        owner.assert_awaited_once_with(ack)

    async def test_runtime_preserves_exact_errors_for_existing_handler(self):
        owner = AsyncMock()
        runtime = SimpleNamespace(adapter=self.route, request=make_start(),
            session=SimpleNamespace(_active_responses={}), _owner_provider_handler=owner,
            _response_offers={}, _response_dispatch=lambda _: None)
        event = {"type": "error", "error": {
            "event_id": self.initial["event_id"], "code": "string_above_max_length"}}
        await RealtimeVoiceRouteRuntime._provider_event(runtime, event)
        owner.assert_awaited_once_with(event)
        self.assertIs(owner.call_args.args[0], event)
        self.assertEqual(self.route._items[self.initial["item"]["id"]].state, "create_rejected")


if __name__ == "__main__":
    unittest.main()
