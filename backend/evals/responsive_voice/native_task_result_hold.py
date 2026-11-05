"""Opt-in result-boundary control; no replacement worker or synthetic result.

The supplied release waiter must bind the actual claim and experiment scope.
This decorator only withholds a successful real executor result. It does not
claim that the model is still computing, or create/extend a lease. TaskRunner's
normal cancellation, heartbeat and settlement remain authoritative.
"""
from __future__ import annotations

import asyncio
import math


class ResultHoldTimeout(TimeoutError):
    """The controlled boundary was not released within its experiment budget."""


class ResultHoldProvider:
    def __init__(self, provider, wait_for_release, *, max_hold_seconds=30.0):
        if (type(max_hold_seconds) not in (int, float)
                or not math.isfinite(max_hold_seconds)
                or not 0 < max_hold_seconds <= 60):
            raise ValueError("invalid_result_hold_budget")
        if not callable(wait_for_release):
            raise ValueError("invalid_result_release_waiter")
        self._provider = provider
        self._wait_for_release = wait_for_release
        self._max_hold_seconds = max_hold_seconds

    def __getattr__(self, name):
        # Preserve the real provider's model, prepare capabilities, recovery and
        # reconciliation. Only successful execute return is decorated.
        return getattr(self._provider, name)

    async def execute(self, context):
        result = await self._provider.execute(context)
        deadline = asyncio.timeout(self._max_hold_seconds)
        try:
            async with deadline:
                await self._wait_for_release(context)
        except TimeoutError as error:
            if deadline.expired():
                raise ResultHoldTimeout("native_result_hold_timeout") from error
            raise
        return result
