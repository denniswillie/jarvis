from __future__ import annotations

import asyncio
import unittest

from app.runtime.concurrency import ModelCallPriority, PriorityAsyncLimiter


class PriorityAsyncLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_high_priority_runs_before_queued_normal_work(self) -> None:
        limiter = PriorityAsyncLimiter(1)
        order: list[str] = []
        await limiter.acquire()

        async def worker(label: str, priority: ModelCallPriority) -> None:
            async with limiter.limit(priority):
                order.append(label)
                await asyncio.sleep(0)

        normal_one = asyncio.create_task(worker("normal-1", ModelCallPriority.NORMAL))
        await asyncio.sleep(0)
        high = asyncio.create_task(worker("high", ModelCallPriority.HIGH))
        normal_two = asyncio.create_task(worker("normal-2", ModelCallPriority.NORMAL))
        await asyncio.sleep(0)
        await limiter.release()
        await asyncio.gather(normal_one, high, normal_two)

        self.assertEqual(order, ["high", "normal-1", "normal-2"])
        self.assertEqual(limiter.active_count, 0)
        self.assertEqual(limiter.waiting_count, 0)

    async def test_cancelled_waiter_does_not_leak_a_permit(self) -> None:
        limiter = PriorityAsyncLimiter(1)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire(ModelCallPriority.HIGH))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter

        self.assertEqual(limiter.waiting_count, 0)
        await limiter.release()
        await asyncio.wait_for(limiter.acquire(), timeout=0.2)
        self.assertEqual(limiter.active_count, 1)
        await limiter.release()

    async def test_active_call_is_not_preempted(self) -> None:
        limiter = PriorityAsyncLimiter(1)
        await limiter.acquire(ModelCallPriority.NORMAL)
        high = asyncio.create_task(limiter.acquire(ModelCallPriority.HIGH))
        await asyncio.sleep(0)

        self.assertFalse(high.done())
        self.assertEqual(limiter.active_count, 1)
        await limiter.release()
        await high
        await limiter.release()
