"""Offline qualification of the real-result boundary, not task continuity."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from evals.responsive_voice.native_task_result_hold import (
    ResultHoldProvider, ResultHoldTimeout,
)


@pytest.mark.asyncio
async def test_real_result_waits_once_and_identity_is_preserved():
    result, context = object(), object()
    entered, release = asyncio.Event(), asyncio.Event()
    provider = SimpleNamespace(execute=AsyncMock(return_value=result))

    async def wait(actual):
        assert actual is context
        provider.execute.assert_awaited_once_with(context)
        entered.set()
        await release.wait()

    execution = asyncio.create_task(ResultHoldProvider(provider, wait).execute(context))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert not execution.done()
        release.set()
        assert await execution is result
        provider.execute.assert_awaited_once_with(context)
    finally:
        execution.cancel()
        await asyncio.gather(execution, return_exceptions=True)


@pytest.mark.asyncio
async def test_executor_error_never_enters_result_hold():
    failure = RuntimeError("executor_failed")
    provider = SimpleNamespace(execute=AsyncMock(side_effect=failure))
    waiter = AsyncMock()
    with pytest.raises(RuntimeError) as caught:
        await ResultHoldProvider(provider, waiter).execute(object())
    assert caught.value is failure
    waiter.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_during_real_execution_propagates_without_hold():
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def execute(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    waiter = AsyncMock()
    task = asyncio.create_task(ResultHoldProvider(SimpleNamespace(execute=execute), waiter).execute(object()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        waiter.assert_not_awaited()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_joins_release_waiter_without_returning_result():
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def wait(_):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    provider = SimpleNamespace(execute=AsyncMock(return_value=object()))
    task = asyncio.create_task(ResultHoldProvider(provider, wait).execute(object()))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        provider.execute.assert_awaited_once()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_hold_budget_ends_wait_without_fabricating_success():
    cancelled = asyncio.Event()

    async def wait(_):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    provider = SimpleNamespace(execute=AsyncMock(return_value=object()))
    with pytest.raises(ResultHoldTimeout, match="native_result_hold_timeout"):
        await ResultHoldProvider(provider, wait, max_hold_seconds=.01).execute(object())
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_waiter_own_timeout_is_not_mislabeled_as_hold_deadline():
    failure = TimeoutError("release_observer_failed")
    provider = SimpleNamespace(execute=AsyncMock(return_value=object()))
    with pytest.raises(TimeoutError) as caught:
        await ResultHoldProvider(provider, AsyncMock(side_effect=failure)).execute(object())
    assert caught.value is failure
    assert not isinstance(caught.value, ResultHoldTimeout)


def test_other_provider_contracts_are_delegated_unchanged():
    provider = SimpleNamespace(prepare=object(), model="configured-model", reconcile=AsyncMock())
    decorated = ResultHoldProvider(provider, AsyncMock())
    assert decorated.prepare is provider.prepare
    assert decorated.model == provider.model
    assert decorated.reconcile is provider.reconcile


@pytest.mark.parametrize("budget", [0, -1, 61, float("inf"), float("nan"), True, "1", None])
def test_invalid_hold_budget_rejected(budget):
    with pytest.raises(ValueError, match="invalid_result_hold_budget"):
        ResultHoldProvider(object(), AsyncMock(), max_hold_seconds=budget)


def test_release_waiter_required():
    with pytest.raises(ValueError, match="invalid_result_release_waiter"):
        ResultHoldProvider(object(), None)
