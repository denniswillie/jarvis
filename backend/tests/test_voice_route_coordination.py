"""Offline coordinator/bridge qualification; earlier production modules stay unchanged."""

import asyncio
import unittest
from datetime import timedelta
from uuid import uuid4

from app.voice_routes.bridge import OfferDisposition, VoiceContextBridge
from app.voice_routes.coordinator import CoordinationPhase, RouteCoordinationError, VoiceRouteCoordinator
from app.voice_routes.contracts import (
    CloseRouteCommand, CommandReceipt, ContextReceipt, InterruptSpeechCommand,
    SourceRetirementCommand, SourceRetirementReceipt,
)
from test_realtime_voice_route import NOW, FakeControl, make_adapter, make_context, make_identity, make_offer, make_start


async def valid(_):
    return True


class LifecycleRoute:
    def __init__(self, identity=None):
        self.identity = identity or make_identity()
        self.capabilities = make_adapter(FakeControl(self.identity)).capabilities
        self.started = 0
        self.closed = 0
        self.fenced = True
        self.close_hook = None
        self.publish_entered = asyncio.Event()
        self.publish_release = asyncio.Event()
        self.block_publish = False
        self.published = []

    async def start(self, request):
        self.started += 1
        return CommandReceipt(route=self.identity, command_id=request.command_id,
                              operation="start", observed_at=NOW, status="completed")

    async def publish_context(self, update):
        self.publish_entered.set()
        if self.block_publish:
            await self.publish_release.wait()
        self.published.append(update)
        return ContextReceipt(route=self.identity, update_id=update.update_id, cursor=update.cursor,
                              observed_at=NOW, status="provider_acknowledged")

    async def interrupt_speech(self, command):
        return CommandReceipt(route=self.identity, command_id=command.command_id, operation="interrupt_speech",
            observed_at=NOW, status="completed", output_fenced=True)

    async def retire_context(self, command):
        return SourceRetirementReceipt(route=self.identity, command_id=command.command_id,
            cursor=command.cursor, observed_at=NOW, status="provider_acknowledged")

    async def close(self, command):
        self.closed += 1
        if self.close_hook is not None:
            self.close_hook()
        return CommandReceipt(route=self.identity, command_id=command.command_id, operation="close",
            observed_at=NOW, status="completed" if self.fenced else "unknown", output_fenced=self.fenced)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_replacement_requires_fence_and_never_installs_failed_candidate(self):
        old, new = LifecycleRoute(), LifecycleRoute(make_identity(2))
        owner = VoiceRouteCoordinator(old.identity.scope, validity=valid, clock=lambda: NOW)
        try:
            await owner.start(old, make_start())
            old.fenced = False
            with self.assertRaises(RouteCoordinationError):
                await owner.replace(new, make_start(new.identity))
            self.assertEqual(new.started, 0)
            self.assertEqual(owner.status.phase, CoordinationPhase.DEGRADED)
            self.assertEqual(owner.status.identity, old.identity)
            old.fenced = True
            await owner.replace(new, make_start(new.identity))
            self.assertEqual(new.started, 1)
            self.assertEqual(owner.status.identity, new.identity)
        finally:
            old.fenced = True
            await owner.close()

    async def test_identity_generation_reuse_is_rejected_before_old_close(self):
        old = LifecycleRoute()
        owner = VoiceRouteCoordinator(old.identity.scope, validity=valid, clock=lambda: NOW)
        try:
            await owner.start(old, make_start())
            with self.assertRaises(RouteCoordinationError):
                await owner.replace(LifecycleRoute(old.identity), make_start(old.identity))
            self.assertEqual(old.closed, 0)
        finally:
            await owner.close()

    async def test_interruption_not_blocked_by_background_publication(self):
        route = LifecycleRoute()
        route.block_publish = True
        owner = VoiceRouteCoordinator(route.identity.scope, validity=valid, clock=lambda: NOW)
        try:
            await owner.start(route, make_start())
            owner.offer_context(make_context(revision=1))
            await asyncio.wait_for(route.publish_entered.wait(), 0.5)
            receipt = await asyncio.wait_for(owner.interrupt_speech(InterruptSpeechCommand(route=route.identity,
                command_id=uuid4(), requested_at=NOW, reason="barge_in")), 0.1)
            self.assertTrue(receipt.output_fenced)
            self.assertFalse(route.publish_release.is_set())
        finally:
            route.publish_release.set()
            await owner.close()

    async def test_provider_retirement_ack_never_reopens_publication(self):
        route = LifecycleRoute()
        owner = VoiceRouteCoordinator(route.identity.scope, validity=valid, clock=lambda: NOW)
        try:
            await owner.start(route, make_start())
            command = SourceRetirementCommand(route=route.identity, command_id=uuid4(), requested_at=NOW,
                cursor=make_context().cursor, sources=(make_offer().attempt.source,), reason="withdrawn")
            receipt = await owner.retire_context(command)
            self.assertEqual(receipt.status, "provider_acknowledged")
            self.assertEqual(owner.status.phase, CoordinationPhase.DEGRADED)
            self.assertTrue(owner.status.reconnect_required)
            self.assertEqual(owner.offer_context(make_context(revision=1)).disposition, OfferDisposition.REJECTED)
        finally:
            await owner.close()

    async def test_replacement_rechecks_expiry_before_calling_candidate_start(self):
        # A failed assertion here identifies an earlier coordinator boundary;
        # this slice deliberately does not modify that production module.
        clock = [NOW]
        old, new = LifecycleRoute(), LifecycleRoute(make_identity(2))
        owner = VoiceRouteCoordinator(old.identity.scope, validity=valid, clock=lambda: clock[0])
        try:
            await owner.start(old, make_start())
            request = make_start(new.identity, expires_at=NOW + timedelta(seconds=1))
            old.close_hook = lambda: clock.__setitem__(0, NOW + timedelta(seconds=2))
            with self.assertRaises(RouteCoordinationError):
                await owner.replace(new, request)
            self.assertEqual(new.started, 0, "Expired initial context reached candidate.start after old-route shutdown.")
        finally:
            await owner.close()


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_newest_pending_snapshot_and_duplicate_cursor_handling(self):
        route = LifecycleRoute()
        entered, release, latest_written = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def validity(update):
            if update.cursor.revision == 1:
                entered.set()
                await release.wait()
            return True

        original_publish = route.publish_context

        async def publish(update):
            result = await original_publish(update)
            latest_written.set()
            return result

        route.publish_context = publish
        bridge = VoiceContextBridge(route, make_context(), validity=validity, clock=lambda: NOW)
        try:
            await bridge.start()
            first, newest = make_context(revision=1), make_context(revision=2)
            self.assertEqual(bridge.offer(first).disposition, OfferDisposition.QUEUED)
            await asyncio.wait_for(entered.wait(), 0.5)
            bridge.offer(newest)
            self.assertEqual(bridge.offer(newest).disposition, OfferDisposition.DUPLICATE)
            self.assertEqual(bridge.offer(first).disposition, OfferDisposition.REJECTED)
            self.assertEqual(bridge.offer(make_context(revision=3, mode="delta")).disposition, OfferDisposition.REJECTED)
            release.set()
            await asyncio.wait_for(latest_written.wait(), 0.5)
            self.assertEqual([update.cursor.revision for update in route.published], [2])
        finally:
            release.set()
            await bridge.close()

    async def test_late_validity_completion_cannot_reopen_fenced_bridge(self):
        route = LifecycleRoute()
        entered, release = asyncio.Event(), asyncio.Event()

        async def validity(_):
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            return True

        bridge = VoiceContextBridge(route, make_context(), validity=validity, clock=lambda: NOW)
        await bridge.start()
        bridge.offer(make_context(revision=1))
        await asyncio.wait_for(entered.wait(), 0.5)
        bridge.fence("source_retirement")
        release.set()
        await asyncio.wait_for(bridge.close(), 0.5)
        self.assertEqual(route.published, [])
        self.assertTrue(bridge.status.fenced)
        self.assertIsNone(bridge.status.last_receipt)

    async def test_deadline_fences_and_reports_failure_without_ack(self):
        route = LifecycleRoute()
        route.block_publish = True
        failed = asyncio.Event()
        reasons = []

        def failure(reason):
            reasons.append(reason)
            failed.set()

        bridge = VoiceContextBridge(route, make_context(), validity=valid, publish_timeout=0.02,
                                    clock=lambda: NOW, on_failure=failure)
        try:
            await bridge.start()
            bridge.offer(make_context(revision=1))
            await asyncio.wait_for(failed.wait(), 0.5)
            self.assertEqual(reasons, ["context_publication_deadline"])
            self.assertTrue(bridge.status.fenced)
            self.assertIsNone(bridge.status.last_receipt)
        finally:
            route.publish_release.set()
            await bridge.close()


if __name__ == "__main__":
    unittest.main()
