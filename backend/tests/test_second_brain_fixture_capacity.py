"""Resource bounds for source-schema suites; no Docker resource is created here."""
import unittest
from unittest.mock import patch

from evals.second_brain.postgres_recovery import PostgresRecoveryFixture


class SourceFixtureCapacityTests(unittest.TestCase):
    def test_source_capacity_is_bounded_before_creating_a_manifest(self):
        with patch("evals.second_brain.postgres_recovery.Path.mkdir") as mkdir:
            for value in (0, 255, 2049, True, "1024"):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    PostgresRecoveryFixture("/tmp/jarvis-s11-restore.invalid-capacity", data_mebibytes=value)
            mkdir.assert_not_called()

    def test_source_capacity_and_memory_overhead_are_recorded_and_applied(self):
        with patch("evals.second_brain.postgres_recovery.Path.mkdir"), patch.object(PostgresRecoveryFixture, "save"):
            default = PostgresRecoveryFixture("/tmp/jarvis-s11-restore.default-capacity")
            expanded = PostgresRecoveryFixture("/tmp/jarvis-s11-restore.expanded-capacity", data_mebibytes=1024)
        for fixture, size in ((default, 512), (expanded, 1024)):
            with self.subTest(size=size):
                args = fixture._run_args("source", readonly=False)
                self.assertEqual(args[args.index("--memory") + 1], str(size + 256) + "m")
                self.assertEqual(args[args.index("--tmpfs") + 1], "/var/lib/postgresql/data:rw,size=" + str(size) + "m")
                self.assertEqual(fixture.manifest["limits"]["data_mebibytes"], size)
                self.assertEqual(fixture.manifest["limits"]["memory_mebibytes"], size + 256)
