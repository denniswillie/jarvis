from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

import asyncpg

from app.memory.jobs.repository import MemoryJobRepository


LOGGER = logging.getLogger(__name__)


class MemoryJobRecoveryRule(Protocol):
    """Idempotently reconstructs missing non-dead work from durable state."""

    name: str

    async def reconcile(
        self,
        *,
        repository: MemoryJobRepository,
        connection: asyncpg.Connection,
        now: datetime,
    ) -> int: ...


class DuplicateMemoryJobRecoveryRuleError(ValueError):
    pass


class MemoryJobReconciler:
    def __init__(
        self,
        *,
        pool: asyncpg.Pool,
        repository: MemoryJobRepository,
        rules: Iterable[MemoryJobRecoveryRule] = (),
    ) -> None:
        self._pool = pool
        self._repository = repository
        self._rules: dict[str, MemoryJobRecoveryRule] = {}
        self.last_run_at: datetime | None = None
        self.last_results: dict[str, int] = {}
        self.last_errors: dict[str, str] = {}
        for rule in rules:
            self.register(rule)

    def register(self, rule: MemoryJobRecoveryRule) -> None:
        name = rule.name.strip()
        if not name:
            raise ValueError("memory job recovery rule name cannot be blank")
        if name in self._rules:
            raise DuplicateMemoryJobRecoveryRuleError(
                f"memory job recovery rule already registered: {name}"
            )
        self._rules[name] = rule

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))

    async def run_once(self, *, now: datetime) -> dict[str, int]:
        results: dict[str, int] = {}
        errors: dict[str, str] = {}
        for name in self.rule_names:
            rule = self._rules[name]
            try:
                async with self._pool.acquire() as connection:
                    async with connection.transaction():
                        recovered = await rule.reconcile(
                            repository=self._repository,
                            connection=connection,
                            now=now,
                        )
                results[name] = max(0, int(recovered))
            except Exception as error:
                errors[name] = type(error).__name__
                LOGGER.exception(
                    "memory job reconciliation failed rule=%s",
                    name,
                )
        self.last_run_at = now
        self.last_results = results
        self.last_errors = errors
        return results
