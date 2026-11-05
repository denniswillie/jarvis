from __future__ import annotations

import unittest
from datetime import UTC, datetime

from app.memory.jobs.runner import MemoryJobRunner, MemoryJobRunnerConfig
from evals.perception_memory.metrics import assess_report
from evals.perception_memory.new_architecture_runner import run_architecture_evaluation


class _RecoveryRepository:
    def __init__(self) -> None:
        self.recovered = 0
        self.requeued = 0

    async def recover_stale_claims(self, **_: object) -> int:
        self.recovered += 1
        return 1

    async def requeue_due_failed(self) -> int:
        self.requeued += 1
        return 1

    async def claim_next(self, **_: object) -> None:
        return None


class _EmptyRegistry:
    job_types: tuple[str, ...] = ()


class _FlakyReconciler:
    rule_names: tuple[str, ...] = ("test",)
    last_run_at = None
    last_results: dict[str, object] = {}
    last_errors: dict[str, object] = {}

    def __init__(self) -> None:
        self.calls = 0

    async def run_once(self, *, now: datetime) -> dict[str, int]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("injected reconciliation failure")
        self.last_run_at = now
        return {"test": 1}


class MemoryFailureRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_runner_retries_recovery_after_transient_reconciliation_failure(self) -> None:
        repository = _RecoveryRepository()
        reconciler = _FlakyReconciler()
        runner = MemoryJobRunner(
            pool=object(),  # type: ignore[arg-type]
            repository=repository,  # type: ignore[arg-type]
            registry=_EmptyRegistry(),  # type: ignore[arg-type]
            reconciler=reconciler,  # type: ignore[arg-type]
            config=MemoryJobRunnerConfig(
                poll_interval_seconds=1,
                max_concurrency=1,
                claim_timeout_seconds=1,
                reconcile_interval_seconds=1,
            ),
            now=lambda: datetime.now(UTC),
        )

        with self.assertRaisesRegex(RuntimeError, "injected reconciliation"):
            await runner.run_cycle()
        await runner.run_cycle()

        self.assertEqual(reconciler.calls, 2)
        self.assertEqual(repository.recovered, 2)
        self.assertEqual(repository.requeued, 2)

    def test_duplicate_interruptions_fail_architecture_gate(self) -> None:
        evaluation = run_architecture_evaluation()
        regressed = evaluation.report.model_copy(deep=True)
        regressed.aggregate.duplicate_interruptions = 1
        assessment = assess_report(regressed)

        self.assertFalse(assessment.passed)
        self.assertFalse(
            next(
                gate
                for gate in assessment.gates
                if gate.name == "duplicate_interruptions"
            ).passed
        )
