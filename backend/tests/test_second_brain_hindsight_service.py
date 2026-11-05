"""Owned native fixture capacity must leave room for retention."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals.second_brain.hindsight_service import POSTGRES_ARGUMENTS, WORKER_ENV, configured_worker_allocation, wait_for_postgres


class NativePostgresBudgetTests(unittest.TestCase):
    def test_fixture_bounds_wal_without_disabling_durability(self):
        self.assertEqual(POSTGRES_ARGUMENTS[0], "postgres")
        self.assertEqual(POSTGRES_ARGUMENTS[1::2], ("-c",) * 4)
        self.assertEqual(dict(value.split("=", 1) for value in POSTGRES_ARGUMENTS[2::2]), {
            "max_wal_size": "64MB", "min_wal_size": "32MB",
            "checkpoint_timeout": "30s", "checkpoint_completion_target": "0.5",
        })


class NativeWorkerCapacityTests(unittest.TestCase):
    def probe(self, value):
        with patch("evals.second_brain.hindsight_service.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps(value),
        )):
            return configured_worker_allocation("/fixture", WORKER_ENV)

    def test_fixture_reserves_one_of_two_slots_and_reports_native_shared_capacity(self):
        self.assertEqual(WORKER_ENV["HINDSIGHT_API_WORKER_MAX_SLOTS"], "2")
        self.assertEqual(WORKER_ENV["HINDSIGHT_API_WORKER_CONSOLIDATION_RESERVED_SLOTS"], "1")
        self.assertEqual(self.probe({"max_slots": 2, "reservations": {"consolidation": 1}})["shared_slots"], 1)

    def test_zero_capacity_and_unbounded_or_malformed_configuration_fail_closed(self):
        for value in (
            {"max_slots": 2, "reservations": {"consolidation": 2}},
            {"max_slots": 10, "reservations": {}},
            {"max_slots": 2, "reservations": {"consolidation": -1}},
            {"max_slots": 2, "reservations": {"consolidation": True}},
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.probe(value)


class NativePostgresReadinessTests(unittest.TestCase):
    def test_readiness_requires_final_tcp_listener_not_initialization_socket(self):
        with patch("evals.second_brain.hindsight_service.docker", return_value="accepting connections") as run:
            wait_for_postgres("jarvis-m02-fixture")
        run.assert_called_once_with("exec", "jarvis-m02-fixture", "pg_isready", "-h", "127.0.0.1",
                                    "-U", "postgres", "-d", "hindsight")

    def test_initial_server_rejection_is_retried_before_tcp_readiness(self):
        with patch("evals.second_brain.hindsight_service.docker",
                   side_effect=[RuntimeError("not_ready"), "accepting connections"]) as run, \
             patch("evals.second_brain.hindsight_service.time.monotonic", side_effect=[0, 1]), \
             patch("evals.second_brain.hindsight_service.time.sleep") as sleep:
            wait_for_postgres("jarvis-m02-fixture")
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_readiness_deadline_does_not_allow_startup_to_continue(self):
        with patch("evals.second_brain.hindsight_service.docker", side_effect=RuntimeError("not_ready")), \
             patch("evals.second_brain.hindsight_service.time.monotonic", side_effect=[0, 46]), \
             patch("evals.second_brain.hindsight_service.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "not_ready"):
                wait_for_postgres("jarvis-m02-fixture")
        sleep.assert_not_called()
