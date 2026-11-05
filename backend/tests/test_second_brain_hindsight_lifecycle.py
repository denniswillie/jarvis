"""Native lifecycle fixture contracts; no external service is needed here."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from evals.second_brain.hindsight_lifecycle_trace import (
    background_idle, bank_snapshot, erasure_findings, native_read, sql_identifier, sql_literal,
)
from evals.second_brain.hindsight_service import PRIVACY_ENV, PRIVACY_FIELDS, configured_privacy_policy, hindsight_service


class AuditContractTests(unittest.TestCase):
    def test_zero_pending_count_does_not_hide_processing_native_jobs(self):
        stats = {"pending_operations": 0, "failed_operations": 0, "pending_consolidation": 0,
                 "failed_consolidation": 0, "operations_by_status": {"completed": 3}}
        self.assertTrue(background_idle(stats))
        self.assertFalse(background_idle(stats | {"operations_by_status": {"completed": 3, "processing": 1}}))
        for changes in ({"operations_by_status": {"unknown": 1}}, {"operations_by_status": {"failed": 1}},
                        {"pending_consolidation": None}, {"pending_operations": False},
                        {"operations_by_status": {"completed": -1}}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                background_idle(stats | changes)

    def test_positive_control_is_required_and_new_residue_is_not_ignored(self):
        before = {"documents": {"target_rows": 1}, "memory_units": {"target_rows": 1},
                  "observation_history": {"target_rows": 0}}
        after = {name: {"target_rows": 0} for name in before}
        self.assertTrue(erasure_findings(before, after)["positive_surfaces_cleared"])
        after["observation_history"]["target_rows"] = 2
        result = erasure_findings(before, after)
        self.assertFalse(result["positive_surfaces_cleared"])
        self.assertEqual(result["remaining_target_rows"], {"observation_history": 2})
        self.assertFalse(erasure_findings({}, {})["positive_surfaces_cleared"])

    def test_privacy_configuration_disables_payload_histories_in_native_runtime(self):
        expected = dict.fromkeys(PRIVACY_FIELDS, False)
        self.assertTrue(all(value == "false" for value in PRIVACY_ENV.values()))
        for value, accepted in ((expected, True), ({**expected, "llm_trace_enabled": True}, False),
                                ({key: flag for key, flag in expected.items() if key != "audit_log_enabled"}, False),
                                ({**expected, "audit_log_enabled": 0}, False)):
            with self.subTest(value=value), patch("evals.second_brain.hindsight_service.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=json.dumps(value))):
                if accepted:
                    self.assertEqual(configured_privacy_policy("/fixture", PRIVACY_ENV), expected)
                else:
                    with self.assertRaises(RuntimeError):
                        configured_privacy_policy("/fixture", PRIVACY_ENV)

    def test_sql_quotes_and_database_transaction_are_read_only(self):
        self.assertEqual(sql_literal("bank'one"), "'bank''one'")
        self.assertEqual(sql_identifier('a"b'), '"a""b"')
        completed = SimpleNamespace(returncode=0, stdout='BEGIN\n{"rows":0}\nCOMMIT\n')
        with patch("evals.second_brain.hindsight_lifecycle_trace.subprocess.run", return_value=completed) as run:
            self.assertEqual(native_read("jarvis-m02-owned", "select 1"), {"rows": 0})
        command = run.call_args.args[0]
        self.assertEqual(command[-1], "BEGIN READ ONLY; select 1; COMMIT")
        with self.assertRaises(ValueError):
            native_read("production-container", "select 1")
        with self.assertRaises(ValueError):
            native_read("jarvis-m02-owned", "delete from documents")

    def test_audit_errors_do_not_echo_database_output(self):
        with patch("evals.second_brain.hindsight_lifecycle_trace.subprocess.run",
                   return_value=SimpleNamespace(returncode=1, stdout="", stderr="private-diagnostic")):
            with self.assertRaisesRegex(RuntimeError, "^native_read_only_audit_failed$"):
                native_read("jarvis-m02-owned", "select 1")

    def test_all_discovered_bank_tables_are_audited_including_new_history_surfaces(self):
        queries = []
        def read(container, query):
            queries.append(query)
            if "information_schema.columns" in query:
                return ["documents", "new_history_table"]
            return {"rows": 1, "target_rows": 0, "control_rows": 0, "target_ids": []}
        with patch("evals.second_brain.hindsight_lifecycle_trace.native_read", side_effect=read):
            result = bank_snapshot("jarvis-m02-owned", "bank", "target", "control")
        self.assertEqual(set(result), {"documents", "new_history_table", "unit_entities", "entity_cooccurrences"})
        self.assertTrue(any('public."new_history_table"' in query for query in queries))


class OwnedRestartTests(unittest.TestCase):
    def test_restart_cleanup_targets_latest_process_even_when_probe_raises(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                root = Path("/tmp") / ("jarvis-m02-native.unit-" + uuid4().hex)
                processes = []
                def popen(*args, **kwargs):
                    process = Mock(pid=100000 + len(processes))
                    process.returncode = None
                    process.poll.side_effect = lambda: process.returncode
                    def terminate():
                        process.returncode = 0
                    process.terminate.side_effect = terminate
                    processes.append(process)
                    return process
                def docker(*arguments, **kwargs):
                    return "127.0.0.1:15432" if arguments[0] == "port" else "ready"
                try:
                    with patch("evals.second_brain.hindsight_service.docker", side_effect=docker), \
                         patch("evals.second_brain.hindsight_service.configured_worker_allocation",
                               return_value={"max_slots": 2, "shared_slots": 1}), \
                         patch("evals.second_brain.hindsight_service.configured_privacy_policy",
                               return_value=dict.fromkeys(PRIVACY_FIELDS, False)), \
                         patch("evals.second_brain.hindsight_service.subprocess.Popen", side_effect=popen), \
                         patch("evals.second_brain.hindsight_service.subprocess.run",
                               return_value=SimpleNamespace(returncode=1, stdout="", stderr="")), \
                         patch("evals.second_brain.hindsight_service.httpx.Client") as client:
                        client.return_value.__enter__.return_value.get.return_value.status_code = 200
                        try:
                            with hindsight_service(root, runtime="/not-used", api_key="fixture-key") as service:
                                service["stop_owned"]()
                                service["start_owned"]()
                                self.assertEqual(len(processes), 2)
                                if fail:
                                    raise LookupError("fixture-failure")
                        except LookupError:
                            self.assertTrue(fail)
                    self.assertTrue(all(process.poll() == 0 for process in processes))
                    self.assertEqual(processes[-1].terminate.call_count, 1)
                    manifest = json.loads((root/"service-resources.json").read_text())
                    self.assertEqual(manifest["process_history"], [process.pid for process in processes])
                    self.assertEqual(manifest["restart_count"], 1)
                    self.assertNotIn("fixture-key", json.dumps(manifest))
                    self.assertEqual(manifest["status"], "failed_or_cleanup_unverified" if fail else "cleaned")
                finally:
                    if root.exists():
                        shutil.rmtree(root)
