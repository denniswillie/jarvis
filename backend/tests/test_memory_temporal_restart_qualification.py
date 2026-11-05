import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.knowledge.retrieval import SourceRecallScope
from evals.responsive_voice.memory_lifecycle_qualification import SOURCE_DAY_ZERO, QualificationFailure
from evals.responsive_voice import memory_temporal_restart_qualification as temporal
from evals.responsive_voice.memory_temporal_restart_qualification import (
    CHILD_MODULE, DomainClock, STAGES, StageTrace, launch_child, new_child_report,
)
from evals.responsive_voice.memory_temporal_transport import PersistentControlledMemory


class TemporalHarnessTests(unittest.TestCase):
    def test_child_metadata_describes_process_and_clock_scope_without_acceptance(self):
        report = new_child_report("day7", DomainClock(7), 6 * 86400)
        self.assertFalse(report["accepted"])
        self.assertFalse(report["native_semantic_quality_qualified"])
        self.assertEqual(report["format"], "jarvis.temporal_memory_restart.child.v1")
        self.assertEqual(report["day"], 7)
        self.assertIn("working-memory expiry", report["clock"]["virtual_scope"])
        self.assertFalse(report["clock"]["scheduler_time_travel"])
        self.assertNotIn("in one process", " ".join(report["limitations"]))
        self.assertIn("not crash recovery", " ".join(report["limitations"]))

    def test_named_failure_preserves_exception_and_bounded_diagnostics_without_messages(self):
        with TemporaryDirectory() as directory:
            report, path = {}, Path(directory) / "trace.json"
            trace = StageTrace(report, path, lambda: {"readiness": "warming"}, limit=2)
            with self.assertRaises(TimeoutError):
                with trace.boundary("selection.await_expected_sources"):
                    raise TimeoutError("private error text must not be retained")
            recorded = json.loads(path.read_text())
            self.assertEqual(recorded["failed_boundary"], "selection.await_expected_sources")
            self.assertEqual(recorded["boundaries"][0]["state"], {"readiness": "warming"})
            self.assertTrue(recorded["boundaries"][0]["frames"])
            self.assertNotIn("private error text", json.dumps(recorded))
            with trace.boundary("teardown"):
                pass
            with self.assertRaises(QualificationFailure):
                with trace.boundary("exceeds_budget"):
                    self.fail("diagnostic budget was bypassed")

    def test_domain_clock_advances_without_replacing_monotonic_or_wall_clocks(self):
        import time
        from datetime import UTC, datetime
        monotonic, wall = time.monotonic(), datetime.now(UTC)
        clock = DomainClock()
        for day in (0, 1, 7, 30, 90):
            clock.advance(day)
            self.assertEqual(clock.now(), SOURCE_DAY_ZERO + timedelta(days=day))
        self.assertLess(time.monotonic() - monotonic, 1)
        self.assertLess((datetime.now(UTC) - wall).total_seconds(), 1)
        with self.assertRaises(QualificationFailure):
            clock.advance(7)

    def test_source_date_scope_is_explicit_not_calendar_scheduler(self):
        scope = SourceRecallScope(start_time=SOURCE_DAY_ZERO,
                                 end_time=SOURCE_DAY_ZERO + timedelta(days=7))
        for day, expected in ((0, True), (1, True), (7, False), (30, False), (90, False)):
            item = SimpleNamespace(context_session_id=None, source_time=SOURCE_DAY_ZERO + timedelta(days=day))
            self.assertEqual(scope.matches(item), expected)

    def test_controlled_bank_persists_ingested_documents_across_real_child_process(self):
        with TemporaryDirectory() as directory:
            path, bank = Path(directory) / "transport.json", "jarvis_temporal_eval_" + uuid4().hex
            transport = PersistentControlledMemory(path, bank, create=True)
            def request(method, tail="", body=None):
                return transport(httpx.Request(method, "http://127.0.0.1:19999/v1/default/banks/" + bank + tail,
                                              json=body))
            request("PUT")
            request("POST", "/memories", {"operation_id": "operation", "items": [
                {"document_id": "document", "content": "Training log 30 kg", "metadata": {}}]})
            program = ("import json,sys,os; from evals.responsive_voice.memory_temporal_transport import "
                "PersistentControlledMemory; s=PersistentControlledMemory(sys.argv[1],sys.argv[2]).read(); "
                "print(json.dumps({'pid':os.getpid(),'documents':list(s['banks'][sys.argv[2]]['documents'])}))")
            completed = subprocess.run([sys.executable, "-c", program, str(path), bank],
                check=True, capture_output=True, text=True, timeout=10)
            result = json.loads(completed.stdout)
            self.assertNotEqual(result["pid"], os.getpid())
            self.assertEqual(result["documents"], ["document"])
            request("DELETE", "/documents/document")
            self.assertFalse(PersistentControlledMemory(path, bank).read()["banks"][bank]["documents"])
            request("DELETE")
            self.assertFalse(PersistentControlledMemory(path, bank).read()["banks"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_controlled_transport_rejects_foreign_banks_and_origins(self):
        with TemporaryDirectory() as directory:
            transport = PersistentControlledMemory(Path(directory) / "state.json",
                "jarvis_temporal_eval_" + uuid4().hex, create=True)
            for url in ("https://api.openai.com/v1/realtime/calls",
                        "http://127.0.0.1:19999/v1/default/banks/foreign"):
                with self.subTest(url=url), self.assertRaises(QualificationFailure):
                    transport(httpx.Request("PUT", url))


class TemporalLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_launch_uses_importable_module_even_from_main(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "day0.json").write_text(json.dumps({"pid": 4321, "status": "passed", "accepted": True}))
            process = SimpleNamespace(pid=4321, returncode=0, communicate=AsyncMock(return_value=(b"{}", None)))
            with patch.object(temporal, "__name__", "__main__"), \
                 patch.object(asyncio, "create_subprocess_exec", AsyncMock(return_value=process)) as launch:
                receipt = await launch_child(root / "child-input.json", "day0", "fixture-dsn")
            self.assertEqual(launch.await_args.args[:3], (sys.executable, "-m", CHILD_MODULE))
            self.assertEqual(receipt["pid"], 4321)
            self.assertNotIn("fixture-dsn", (root / "day0-process.json").read_text())


@unittest.skipUnless(os.getenv("JARVIS_MEMORY_TEMPORAL_SQL_TEST") == "1", "Opt-in isolated real PostgreSQL")
class TemporalRestartSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_clean_processes_with_real_ingress_domain_expiry_and_cleanup(self):
        root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]) / ("temporal-" + uuid4().hex)
        # Exercise the real CLI parent as well as all six real child processes.
        process = await asyncio.create_subprocess_exec(sys.executable, "-m", CHILD_MODULE,
            "--controlled", "--report-dir", str(root),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        output, _ = await asyncio.wait_for(process.communicate(), 690)
        (root / "parent-process.json").write_text(json.dumps({"pid": process.pid,
            "returncode": process.returncode, "output": output.decode(errors="replace")}, indent=2))
        report = json.loads((root / "qualification.json").read_text())
        print("temporal_restart_report=" + str(root / "qualification.json"))
        self.assertTrue(report["accepted"], json.dumps(report, indent=2))
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["parent_pid"], process.pid)
        self.assertNotEqual(process.pid, os.getpid())
        self.assertEqual(tuple(report["stages"]), STAGES)
        self.assertTrue(all(item["status"] == "passed" for item in report["stages"].values()))
        self.assertEqual(len({item["pid"] for item in report["children"]}), 6)
        self.assertTrue(all(item["accepted"] for item in report["children"]))
        self.assertEqual([item["day"] for item in report["children"]], [0, 1, 7, 30, 90, 90])
        self.assertTrue(report["children"][0]["current_evidence"]["fresh_observation_projected"])
        self.assertTrue(report["children"][1]["current_evidence"]["reader_expired"])
        self.assertEqual(len(report["children"][3]["dated_windows"]), 2)
        self.assertFalse(report["clock_scope"]["scheduler_time_travel"])
        self.assertLessEqual(report["transport"]["recalls"], 12)
        self.assertEqual(report["cleanup"]["database"]["remaining_schemas"], 0)
        self.assertEqual(report["cleanup"]["database"]["remaining_public_owner_rows"], 0)
        self.assertTrue(report["cleanup"]["controlled_bank"]["independent_client_audit"])
