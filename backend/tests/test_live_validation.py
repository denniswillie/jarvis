from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from evals.perception_memory.baseline_runner import run_baseline
from evals.perception_memory.models import EvalMode
from evals.perception_memory.new_architecture_runner import (
    run_architecture_evaluation,
)
from scripts.smoke_focused_release import CHECKS, focused_payload, run_smoke_series


class LiveArchitectureRunnerTests(unittest.TestCase):
    def test_live_mode_forwards_explicit_trace_without_legacy_simulation(self) -> None:
        report = run_baseline(simulate_legacy_gaps=False)
        trace_path = Path("/tmp/disposable-live-observations.json")

        with patch(
            "evals.perception_memory.new_architecture_runner.run_baseline",
            return_value=report,
        ) as mocked_runner:
            evaluation = run_architecture_evaluation(
                mode=EvalMode.LIVE,
                live_results_path=trace_path,
                baseline_path=None,
            )

        kwargs = mocked_runner.call_args.kwargs
        self.assertEqual(kwargs["mode"], EvalMode.LIVE)
        self.assertEqual(kwargs["live_results_path"], trace_path)
        self.assertFalse(kwargs["simulate_legacy_gaps"])
        self.assertEqual(evaluation.report.metadata["validation_mode"], "live")


class FocusedSmokeSeriesTests(unittest.TestCase):
    def test_smoke_retries_transient_unavailability_without_leaking_token(self) -> None:
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            self.assertEqual(request.headers["ngrok-skip-browser-warning"], "true")
            attempt = request_count // len(CHECKS)
            request_count += 1
            if attempt == 0:
                raise httpx.ConnectError("backend restarting", request=request)
            payload = {
                "/diagnostics/config": {"status": "ready", "background_jobs": {
                    "memory_job_runner": False, "memory_maintenance_scheduler": False}},
                "/tasks/providers": [{"backend_kind": "frontier", "configured": True, "worker_ready": True}],
                "/attention/push-status": {"state": "unavailable", "reason": "push_disabled",
                                          "inbox_available": True, "provider_access": "unverified"},
            }.get(request.url.path, {"status": "ready"})
            return httpx.Response(200, json=payload)

        snapshots = asyncio.run(
            run_smoke_series(
                base_url="https://jarvis.example",
                token="must-not-appear",
                allow_degraded=False,
                attempts=2,
                interval_seconds=0,
                transport=httpx.MockTransport(handler),
            )
        )

        self.assertEqual(len(snapshots), 2)
        self.assertFalse(snapshots[0]["passed"])
        self.assertTrue(snapshots[1]["passed"])
        self.assertNotIn("must-not-appear", str(snapshots))

    def test_retired_writers_and_unready_worker_fail_even_if_http_is_ready(self):
        for jobs in ({}, {"memory_job_runner": True, "memory_maintenance_scheduler": False}):
            self.assertFalse(focused_payload("/diagnostics/config",
                {"status": "ready", "background_jobs": jobs})[0])
        self.assertFalse(focused_payload("/tasks/providers",
            [{"backend_kind": "frontier", "configured": True, "worker_ready": False}])[0])

    def test_disabled_push_is_explicit_and_can_be_required(self):
        payload = {"state": "unavailable", "reason": "push_disabled", "provider_access": "unverified"}
        passed, warnings = focused_payload("/attention/push-status", payload)
        self.assertTrue(passed)
        self.assertIn("phone_push_not_running", warnings)
        self.assertFalse(focused_payload("/attention/push-status", payload, require_push=True)[0])
        self.assertFalse(focused_payload("/attention/push-status", {"state": "unavailable"})[0])

    def test_smoke_rejects_invalid_retry_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "attempts"):
            asyncio.run(
                run_smoke_series(
                    base_url="https://jarvis.example",
                    token="token",
                    allow_degraded=False,
                    attempts=0,
                    interval_seconds=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
