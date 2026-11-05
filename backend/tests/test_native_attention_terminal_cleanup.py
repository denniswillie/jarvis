"""A later stop cannot cancel successful notification cleanup midway through."""
import asyncio
import unittest

import test_native_attention_delivery as fixtures


class NativeTerminalCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_during_successful_cleanup_preserves_owner_release(self):
        fixture = fixtures.NativeDeliveryTests()
        await fixture.asyncSetUp()
        entered, release = asyncio.Event(), asyncio.Event()
        delivery = fixture.sink.create_delivery(fixture.f, None, fixture.f.claim,
                                                fixture.session, receipt_seconds=1)
        async def heartbeat():
            try:
                await asyncio.Future()
            finally:
                entered.set()
                await release.wait()
        delivery._heartbeat = heartbeat
        fixture.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(delivery.output_finished)
            await delivery.stop("native_attention_closed")
            release.set()
            result = await asyncio.gather(delivery.task, return_exceptions=True)
            self.assertNotIsInstance(result[0], asyncio.CancelledError)
            self.assertIsNone(fixture.session._attention_delivery)
            self.assertEqual(fixture.f.attempt["phase"], "unknown")
        finally:
            release.set()
            await asyncio.gather(delivery.task, return_exceptions=True)
            await fixture.asyncTearDown()
