"""Measurement integrity checks; small samples never claim a P2 p95 gate."""
import unittest

from evals.responsive_voice.control_benchmark import distribution, run_benchmark, snapshot_samples


class ControlBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def test_nearest_rank_statistics_keep_missing_measurements_explicit(self):
        result = distribution([*range(1, 21), None, float("nan"), -1, True])
        self.assertEqual(result, dict(count=20, missing=4, p50_ms=10, p95_ms=19, maximum_ms=20))
        self.assertEqual(distribution([None])["p95_ms"], None)

    async def test_real_snapshot_projection_has_usable_evidence_and_no_external_io(self):
        report = await snapshot_samples(20)
        self.assertEqual(report["summary"]["count"], 20)
        self.assertGreater(report["selected_entries"], 0)
        self.assertEqual(report["intercepted_file_and_network_calls"], 0)

    async def test_real_loops_report_each_held_boundary_without_claiming_hardware(self):
        report = await run_benchmark(samples=1, hold_ms=1, deadline_ms=100)
        self.assertFalse(report["qualified_population"])
        self.assertFalse(report["gates"]["passed"])
        self.assertEqual(len(report["workloads"]), 3)
        self.assertIn("human hearing", report["exclusions"])
        for case in report["workloads"]:
            self.assertEqual(case["dispatch"]["count"] + case["dispatch"]["missing"], 4)
            run, = case["samples"]
            self.assertTrue(run["external_work_still_held_at_measurement"])
            self.assertEqual(run["canonical_execution_count"], 1)
            self.assertEqual(run["canonical_finalization_count"], 1)
            self.assertEqual(run["tool_calls"], 1)
            self.assertEqual(len(run["controls"]), 4)
            if case["unfinished_or_failed"] or case["dispatch"]["missing"]:
                self.assertFalse(report["gates"]["control_p95_under_25ms_and_no_stalls"])


if __name__ == "__main__":
    unittest.main()
