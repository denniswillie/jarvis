"""Offline CLI/pipe diagnostics. No worker startup, SQL or provider execution."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.knowledge.ingestion import IngestionWorker
from app.knowledge.outbox import OutboxRepository
from app.tasks import worker
from evals.responsive_voice import native_task_worker as supervisor


class TaskWorkerStartupDiagnosticsTests(unittest.TestCase):
    def failure(self, function):
        try:
            function()
        except Exception as error:
            return error
        self.fail("Fixture must raise")

    def compiled(self, filename, source, **scope):
        exec(compile(source, filename, "exec"), scope)
        return scope

    def test_real_numeric_guard_origin_without_db(self):
        for function, stage, source in (
            (lambda: OutboxRepository(None, lease_seconds=0), "knowledge_outbox", "outbox.py"),
            (lambda: IngestionWorker(SimpleNamespace(lease_seconds=60), None, request_deadline=60),
             "knowledge_ingestion", "ingestion.py"),
        ):
            with self.subTest(stage=stage):
                error = self.failure(function)
                terminal = error.__traceback__
                while terminal.tb_next:
                    terminal = terminal.tb_next
                receipt = worker.startup_failure_receipt(error)
                self.assertEqual(receipt["diagnostic"], {"version": 1, "stage": stage,
                    "frame": {"source": source, "function": "__init__", "line": terminal.tb_lineno},
                    "basis": "exception_origin", "truncated": False})
                self.assertNotIn(str(error), json.dumps(receipt))

    def test_private_origin_reports_only_allowlisted_caller(self):
        private = self.compiled("/private/SECRET/runtime.py",
            "def PRIVATE_FUNCTION():\n    raise ValueError('SECRET bearer transcript')\n")["PRIVATE_FUNCTION"]
        caller = self.compiled(worker.__file__, "def run_command():\n    private()\n", private=private)["run_command"]
        receipt = worker.startup_failure_receipt(self.failure(caller))
        self.assertEqual(receipt["diagnostic"], {"version": 1, "stage": "worker",
            "frame": {"source": "worker.py", "function": "run_command", "line": 2},
            "basis": "filtered_caller", "truncated": False})
        for secret in ("SECRET", "PRIVATE_FUNCTION", "/private", "bearer", "transcript"):
            self.assertNotIn(secret, json.dumps(receipt))

    def test_unrecognized_path_or_function_cannot_look_like_origin(self):
        for filename, name in (("/private/SECRET/worker.py", "run_command"),
                               (worker.__file__, "PRIVATE_FUNCTION")):
            with self.subTest(name=name):
                call = self.compiled(filename, f"def {name}():\n    raise ValueError('SECRET')\n")[name]
                diagnostic = worker.startup_failure_receipt(self.failure(call))["diagnostic"]
                self.assertEqual(diagnostic, {"version": 1, "stage": "unavailable", "frame": None,
                    "basis": "unavailable", "truncated": False})

    def test_bounded_traceback_does_not_claim_origin(self):
        call = self.compiled(worker.__file__,
            "def run_command(count):\n    if count:\n        return run_command(count-1)\n"
            "    raise ValueError('SECRET')\n")["run_command"]
        receipt = worker.startup_failure_receipt(self.failure(lambda: call(90)))
        self.assertTrue(receipt["diagnostic"]["truncated"])
        self.assertEqual(receipt["diagnostic"]["basis"], "filtered_caller")
        self.assertLess(len(json.dumps(receipt).encode()), 512)

    def test_no_traceback_and_private_type_code_message_are_safe(self):
        class SECRET_ERROR(Exception):
            code = "SECRET_CODE"

            def __str__(self):
                raise AssertionError("Do not stringify")

        receipt = worker.startup_failure_receipt(SECRET_ERROR())
        self.assertEqual(receipt["error_type"], "unrecognized_error_type")
        self.assertEqual(receipt["code"], "unrecognized_code")
        self.assertEqual(receipt["diagnostic"]["basis"], "unavailable")
        self.assertNotIn("SECRET", json.dumps(receipt))

    def test_diagnostic_does_not_follow_chained_secret_causes(self):
        source = "def run_command():\n    try:\n        raise RuntimeError('SECRET')\n"
        source += "    except RuntimeError as cause:\n        raise ValueError('PRIVATE') from cause\n"
        call = self.compiled(worker.__file__, source)["run_command"]
        receipt = worker.startup_failure_receipt(self.failure(call))
        self.assertEqual(receipt["diagnostic"]["basis"], "exception_origin")
        self.assertEqual(receipt["diagnostic"]["frame"]["line"], 5)
        self.assertNotIn("SECRET", json.dumps(receipt))
        self.assertNotIn("PRIVATE", json.dumps(receipt))

    def test_cli_emits_one_safe_receipt_and_same_failure_exit(self):
        async def failing_command(command):
            raise ValueError("SECRET")

        output = io.StringIO()
        with patch.object(worker, "run_command", failing_command), patch("sys.argv", ["worker", "run"]), redirect_stdout(output):
            result = worker.main()
        self.assertEqual(result, 1)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        receipt = json.loads(output.getvalue())
        self.assertFalse(receipt["ready"])
        self.assertEqual(receipt["error_type"], "ValueError")
        self.assertEqual(receipt["diagnostic"]["basis"], "filtered_caller")
        self.assertEqual(receipt["diagnostic"]["frame"]["function"], "main")
        self.assertNotIn("SECRET", output.getvalue())

    def test_all_wire_profiles_match_and_fit_unchanged_line_bound(self):
        expected = {(stage, Path(relative).name): functions
                    for relative, (stage, functions) in worker._STARTUP_FRAME_SOURCES.items()}
        self.assertEqual(expected, supervisor._STARTUP_DIAGNOSTIC_SOURCES)
        self.assertEqual(worker._STARTUP_FAILURE_TYPES, supervisor._STARTUP_FAILURE_TYPES)
        self.assertEqual(worker._STARTUP_FAILURE_CODES, supervisor._STARTUP_FAILURE_CODES)
        for (stage, source), functions in expected.items():
            for function in functions:
                receipt = {"ready": False, "error_type": "ValidationError", "code": "sandbox_cleanup_worker_stopped",
                    "diagnostic": {"version": 1, "stage": stage,
                        "frame": {"source": source, "function": function, "line": 1_000_000},
                        "basis": "exception_origin", "truncated": False}}
                self.assertTrue(supervisor._valid_startup_diagnostic(receipt["diagnostic"]))
                self.assertLessEqual(len(json.dumps(receipt).encode()), supervisor._STARTUP_FAILURE_MAX_LINE_BYTES)

    def test_actual_cli_receipt_survives_supervisor_stream(self):
        error = self.failure(lambda: OutboxRepository(None, lease_seconds=0))
        expected = worker.startup_failure_receipt(error)

        async def run():
            stream = asyncio.StreamReader()
            data = (json.dumps(expected) + "\n").encode()
            stream.feed_data(data[:23])
            stream.feed_data(data[23:])
            stream.feed_eof()
            drain = supervisor._PipeDrain(stream, 8)
            await drain.run()
            drain.tail.clear()
            return drain.receipt()

        self.assertEqual(asyncio.run(run())["startup_failure"], expected)
