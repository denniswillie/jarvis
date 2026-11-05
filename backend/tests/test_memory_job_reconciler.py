from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Any

from app.memory.jobs.reconciler import (
    DuplicateMemoryJobRecoveryRuleError,
    MemoryJobReconciler,
)
from app.memory.jobs.models import MemoryJobEnqueueResult


class _Context:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        if hasattr(self.value, "active"):
            self.value.active = True  # type: ignore[attr-defined]
        return self.value

    async def __aexit__(self, *_: object) -> None:
        if hasattr(self.value, "active"):
            self.value.active = False  # type: ignore[attr-defined]


class _Connection:
    active = False

    def transaction(self) -> _Context:
        return _Context(self)


class _Pool:
    def __init__(self) -> None:
        self.connection = _Connection()

    def acquire(self) -> _Context:
        return _Context(self.connection)


class _Rule:
    def __init__(self, name: str, recovered: int = 1, *, fails: bool = False) -> None:
        self.name = name
        self.recovered = recovered
        self.fails = fails
        self.saw_transaction = False

    async def reconcile(self, **kwargs: Any) -> int:
        self.saw_transaction = kwargs["connection"].active
        if self.fails:
            raise RuntimeError("failed")
        return self.recovered


class MemoryJobReconcilerTests(unittest.IsolatedAsyncioTestCase):
    async def test_rules_run_deterministically_in_owned_transactions(self) -> None:
        second = _Rule("b", recovered=2)
        first = _Rule("a", recovered=1)
        reconciler = MemoryJobReconciler(
            pool=_Pool(),  # type: ignore[arg-type]
            repository=object(),  # type: ignore[arg-type]
            rules=(second, first),
        )

        results = await reconciler.run_once(now=datetime.now(UTC))

        self.assertEqual(results, {"a": 1, "b": 2})
        self.assertTrue(first.saw_transaction)
        self.assertTrue(second.saw_transaction)

    async def test_rule_failure_is_isolated_and_recorded(self) -> None:
        reconciler = MemoryJobReconciler(
            pool=_Pool(),  # type: ignore[arg-type]
            repository=object(),  # type: ignore[arg-type]
            rules=(_Rule("broken", fails=True), _Rule("healthy")),
        )

        results = await reconciler.run_once(now=datetime.now(UTC))

        self.assertEqual(results, {"healthy": 1})
        self.assertEqual(reconciler.last_errors, {"broken": "RuntimeError"})

    def test_duplicate_rule_name_fails_startup(self) -> None:
        with self.assertRaises(DuplicateMemoryJobRecoveryRuleError):
            MemoryJobReconciler(
                pool=_Pool(),  # type: ignore[arg-type]
                repository=object(),  # type: ignore[arg-type]
                rules=(_Rule("same"), _Rule("same")),
            )

    async def test_dead_dedupe_marker_is_not_recreated(self) -> None:
        dead_result = object()

        class _DeadRepository:
            calls = 0

            async def enqueue(self, **_: Any) -> MemoryJobEnqueueResult:
                self.calls += 1
                return MemoryJobEnqueueResult(  # type: ignore[arg-type]
                    job=dead_result,
                    created=False,
                )

        class _RecoveryRule:
            name = "recover"

            async def reconcile(self, **kwargs: Any) -> int:
                result = await kwargs["repository"].enqueue(
                    owner_id="owner",
                    session_id=None,
                    job_type="test",
                    dedupe_key="stable",
                    payload={},
                    available_at=kwargs["now"],
                    connection=kwargs["connection"],
                )
                return int(result.created)

        repository = _DeadRepository()
        reconciler = MemoryJobReconciler(
            pool=_Pool(),  # type: ignore[arg-type]
            repository=repository,  # type: ignore[arg-type]
            rules=(_RecoveryRule(),),
        )

        result = await reconciler.run_once(now=datetime.now(UTC))

        self.assertEqual(result, {"recover": 0})
        self.assertEqual(repository.calls, 1)
