"""Owner-scoped propagation on existing maintenance, not native playback proof."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.quiet import OwnerQuietRepository, QuietCommand, quiet_delivery_decision
from app.attention.service import AttentionService
from app.tools.invocation import ToolInvocationContext
from test_attention_quiet import LedgerFixture


class ControlledDelivery:
    def __init__(self, owner):
        self.claim = SimpleNamespace(owner_id=owner)
        self.task = None
        self.revoked = False
        self.owns_output = True
        self.reasons = []
        self.stop_hook = None

    def current(self):
        return not self.revoked and self.owns_output

    async def stop(self, reason):
        self.revoked = True
        self.reasons.append(reason)
        if self.stop_hook:
            await self.stop_hook()


class QuietPropagationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = LedgerFixture()
        self.owner, self.session = "quiet-propagation-fixture", uuid4()
        self.quiet = OwnerQuietRepository(self.db)
        @asynccontextmanager
        async def transaction(owner):
            async with self.db.transaction():
                yield self.db
        self.repository = SimpleNamespace(tasks=SimpleNamespace(transaction=transaction),
            _recover=AsyncMock(), reconcile_notifications=AsyncMock(), claim_next=AsyncMock())
        self.service = AttentionService(self.repository)

    async def asyncTearDown(self):
        self.service._deliveries.clear()
        await self.service.close()

    async def command(self, action="activate"):
        source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.db.authorize(self.owner, self.session, source)
        return await self.quiet.save(self.owner, self.session, source, QuietCommand(
            action=action, expected_revision=(await self.quiet.get(self.owner)).revision))

    async def test_no_delivery_uses_no_new_poll_or_policy_read(self):
        before = list(self.db.calls)
        self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)
        self.assertEqual(self.db.calls, before)
        self.assertIsNone(self.service._task)
        self.assertIsNone(self.service._maintenance_task)
        self.repository.claim_next.assert_not_awaited()

    async def test_existing_maintenance_propagates_only_to_owned_notification(self):
        owned, foreign, direct = (ControlledDelivery(owner) for owner in (self.owner, "another-owner", self.owner))
        self.service._deliveries.update((owned, foreign))
        await self.command()
        async def outside_transaction():
            self.assertFalse(self.db.lock.locked())
        owned.stop_hook = outside_transaction
        await self.service._maintenance_cycle(self.owner)
        self.assertEqual(owned.reasons, ["owner_quiet_quiet"])
        self.assertEqual(foreign.reasons, [])
        self.assertEqual(direct.reasons, [])
        self.repository._recover.assert_awaited_once()
        self.repository.reconcile_notifications.assert_awaited_once_with(self.owner)
        self.repository.claim_next.assert_not_awaited()

    async def test_repeated_quiet_does_not_repeat_fence_or_implicitly_resume(self):
        delivery = ControlledDelivery(self.owner)
        self.service._deliveries.add(delivery)
        for count in (1, 2, 3):
            with self.subTest(repeats=count):
                for _ in range(count):
                    result = await self.command()
                    await self.service.enforce_owner_quiet(self.owner)
                    self.assertTrue((await quiet_delivery_decision(None, self.owner,
                        kind="solicited_reply", channel="voice")).allowed)
                    self.assertEqual((await self.quiet.get(self.owner)).revision, result["accepted_revision"])
                    self.assertTrue((await self.quiet.get(self.owner)).suppresses_unsolicited)
        self.assertEqual(delivery.reasons, ["owner_quiet_quiet"])
        self.repository.claim_next.assert_not_awaited()

    async def test_resume_during_fence_does_not_deadlock_or_stop_a_new_speaker(self):
        delivery = ControlledDelivery(self.owner)
        self.service._deliveries.add(delivery)
        await self.command()
        entered, release = asyncio.Event(), asyncio.Event()
        async def fence():
            self.assertFalse(self.db.lock.locked())
            entered.set()
            await release.wait()
        delivery.stop_hook = fence
        pending = asyncio.create_task(self.service.enforce_owner_quiet(self.owner))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(self.command("resume"), 1)
            new = ControlledDelivery(self.owner)
            self.service._deliveries.add(new)
            self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)
        finally:
            release.set()
            self.assertEqual(await asyncio.wait_for(pending, 1), 1)
        self.assertEqual(new.reasons, [])
        self.assertFalse((await self.quiet.get(self.owner)).suppresses_unsolicited)

    async def test_stale_delivery_identity_cannot_interrupt_current_direct_output(self):
        old = ControlledDelivery(self.owner)
        old.owns_output = False
        self.service._deliveries.add(old)
        await self.command()
        self.assertEqual(await self.service.enforce_owner_quiet(self.owner), 0)
        self.assertEqual(old.reasons, [])

    async def test_failed_authority_read_fences_only_known_notifications_and_surfaces_failure(self):
        owned, foreign = ControlledDelivery(self.owner), ControlledDelivery("another-owner")
        self.service._deliveries.update((owned, foreign))
        with patch("app.attention.service.quiet_delivery_decision", AsyncMock(side_effect=RuntimeError("private fixture detail"))):
            with self.assertRaises(AttentionError) as error:
                await self.service.enforce_owner_quiet(self.owner)
        self.assertEqual(error.exception.code, "owner_quiet_unavailable")
        self.assertEqual(owned.reasons, ["owner_quiet_unavailable"])
        self.assertEqual(foreign.reasons, [])
        self.assertFalse(self.db.lock.locked())
        self.assertNotIn("private fixture detail", str(error.exception))
