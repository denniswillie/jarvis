"""Offline oracles plus opt-in REAL PostgreSQL / CONTROLLED Hindsight lifecycle.

No answer fixture is injected into retrieval: the HTTP stub retains submitted
documents and returns their identities, exercising the real runtime, ingestion,
source resolution and command APIs. It does NOT qualify native semantic ranking.
"""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.parse import unquote
from uuid import uuid4

import httpx

from app.knowledge.hindsight import HindsightClient
from app.runtime.feature_config import KnowledgeConfiguration
from evals.responsive_voice.memory_lifecycle_qualification import (
    Bounds, LifecycleCase, QualificationFailure, STAGES, assert_recall,
    failure, new_report, run_qualification, save,
)


class MemoryLifecycleOracleTests(unittest.TestCase):
    def setUp(self):
        self.original = SimpleNamespace(source_id=uuid4(), revision=2, content_digest="a" * 64,
            content={"text": "A synthetic input, not a seeded answer"}, trust="derived",
            source_time=datetime(2026, 1, 5, tzinfo=UTC))
        self.item = {"source_id": str(self.original.source_id), "revision": 2,
            "content_digest": self.original.content_digest, "content": self.original.content,
            "trust": "derived", "is_current": True, "source_time": self.original.source_time.isoformat()}

    def test_source_bound_original_is_required_not_just_plausible_prose(self):
        assert_recall({"status": "available", "sources": [self.item]}, (self.original,))
        for changes in ({"revision": 1}, {"is_current": False}, {"trust": "sensor"},
                        {"content_digest": "b" * 64}, {"content": {"text": "invented"}},
                        {"source_time": "2099-01-01T00:00:00+00:00"}, {"source_id": str(uuid4())}):
            with self.subTest(changes=changes), self.assertRaises(QualificationFailure):
                assert_recall({"status": "available", "sources": [self.item | changes]}, (self.original,))

    def test_outage_empty_recall_and_duplicate_original_do_not_pass_positive_control(self):
        for result in ({"status": "unavailable", "sources": []},
                       {"status": "available", "sources": []},
                       {"status": "available", "sources": [self.item, self.item]}):
            with self.subTest(result=result), self.assertRaises(QualificationFailure):
                assert_recall(result, (self.original,))

    def test_timestamp_equivalence_requires_the_same_timezone_aware_instant(self):
        for timestamp in ("2026-01-05T00:00:00Z", "2026-01-05T00:00:00+00:00",
                          "2026-01-05T01:00:00+01:00"):
            with self.subTest(timestamp=timestamp):
                assert_recall({"status": "available", "sources": [self.item | {"source_time": timestamp}]},
                              (self.original,))
        for timestamp in (None, "not-a-date", "2026-01-05T00:00:00", "2026-01-06T00:00:00Z"):
            with self.subTest(timestamp=timestamp), self.assertRaises(QualificationFailure):
                assert_recall({"status": "available", "sources": [self.item | {"source_time": timestamp}]},
                              (self.original,))

    def test_deletion_checks_both_identity_and_content(self):
        result = {"status": "available", "sources": [self.item]}
        with self.assertRaises(QualificationFailure):
            assert_recall(result, (), absent=(self.original.source_id,))
        with self.assertRaises(QualificationFailure):
            assert_recall(result, (), forbidden=(self.original.content["text"],))

    def test_controlled_evidence_and_calendar_labels_never_mean_native_acceptance(self):
        report = new_report("controlled_hindsight_real_postgres")
        self.assertFalse(report["native_semantic_quality_qualified"])
        self.assertFalse(report["clock"]["scheduler_time_travel"])
        self.assertFalse(report["clock"]["database_clock_changed"])
        self.assertEqual(set(report["stages"]), set(STAGES))
        self.assertTrue(all(stage["status"] == "unreached" for stage in report["stages"].values()))

    def test_failures_do_not_serialize_credentials(self):
        report = failure(RuntimeError("postgresql://private:secret@host"))
        self.assertEqual(report, {"type": "RuntimeError"})

    def test_budget_validation(self):
        for changes in ({"max_steps": True}, {"max_steps": 361}, {"drain_seconds": 0},
                        {"lifecycle_seconds": 1201}, {"cleanup_seconds": 601}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Bounds(**changes)

    def test_report_is_private(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            save(path, {"status": "fixture"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class MemoryLifecycleBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_drain_is_bounded_without_moving_database_time(self):
        class Outbox:
            async def diagnostics(self, owner):
                return {"pending_operations": 1, "recorded_failures": 0}
        class Worker:
            calls = 0
            async def step(self, owner, identity):
                self.calls += 1
        case = object.__new__(LifecycleCase)
        case.owner, case.bounds, case.report = "fixture", Bounds(max_steps=1), {"drains": []}
        case.runtime = SimpleNamespace(outbox=Outbox(), worker=Worker(), worker_id="fixture")
        with self.assertRaisesRegex(QualificationFailure, "step_budget"):
            await case.drain()
        self.assertEqual(case.runtime.worker.calls, 1)

    async def test_failed_stage_keeps_later_stages_unreached(self):
        case = object.__new__(LifecycleCase)
        case.report = new_report("controlled_hindsight_real_postgres")
        case.checkpoint = lambda: None
        async def fail():
            raise QualificationFailure("controlled_failure")
        with self.assertRaises(QualificationFailure):
            await case.stage(STAGES[0], fail)
        self.assertEqual(case.report["stages"][STAGES[0]]["status"], "failed")
        self.assertTrue(all(case.report["stages"][stage]["status"] == "unreached" for stage in STAGES[1:]))


class ControlledHindsightHTTP:
    """Deterministic transport fixture, no semantic scoring or final answers."""
    def __init__(self):
        self.banks = {}
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        parts = [unquote(part) for part in request.url.path.split("/") if part]
        body = json.loads(request.content) if request.content else {}
        def reply(value, status=200):
            return httpx.Response(status, json=value)
        if parts == ["v1", "default", "banks"]:
            return reply({"banks": [{"bank_id": key} for key in self.banks]})
        if parts[:3] != ["v1", "default", "banks"] or len(parts) < 4:
            raise AssertionError("Unexpected controlled HTTP path")
        bank, tail = parts[3], parts[4:]
        if not tail and request.method == "PUT":
            self.banks.setdefault(bank, {"documents": {}, "operations": {}})
            return reply({"bank_id": bank})
        if bank not in self.banks:
            return reply({"detail": "not found"}, 404)
        state = self.banks[bank]
        if not tail and request.method == "DELETE":
            del self.banks[bank]
            return reply({"success": True})
        if tail == ["config"]:
            return reply({"bank_id": bank, "config": {"audit_log_enabled": False, "store_document_text": True}})
        if tail == ["mental-models"]:
            return reply({"total": 0, "items": []})
        if tail == ["memories"] and request.method == "POST":
            identity = body["operation_id"]
            state["operations"][identity] = {"operation_id": identity, "status": "completed",
                "child_operations": [], "next_retry_at": None}
            for item in body["items"]:
                state["documents"][item["document_id"]] = {
                    "id": item["document_id"], "bank_id": bank,
                    "original_text": item["content"], "document_metadata": item["metadata"],
                }
            return reply({"async": True, "operation_id": identity})
        if tail == ["memories", "recall"]:
            return reply({"results": [{"document_id": identity} for identity in state["documents"]]})
        if tail[0] == "documents":
            if request.method == "DELETE":
                state["documents"].pop(tail[1], None)
                return reply({"success": True})
            item = state["documents"].get(tail[1])
            return reply(item if item else {"detail": "not found"}, 200 if item else 404)
        if tail[0] == "operations":
            if len(tail) == 3 and tail[2] == "delete" and request.method == "DELETE":
                state["operations"].pop(tail[1], None)
                return reply({"success": True, "operation_id": tail[1]})
            item = state["operations"].get(tail[1])
            return reply(item if item else {"detail": "not found"}, 200 if item else 404)
        raise AssertionError("Unexpected controlled HTTP operation")

    def client(self, *args, **kwargs):
        return HindsightClient(*args, **kwargs, transport=httpx.MockTransport(self))


@unittest.skipUnless(os.getenv("JARVIS_MEMORY_LIFECYCLE_SQL_TEST") == "1",
                     "Opt-in disposable SQL fixture; no native Hindsight acceptance")
class MemoryLifecycleControlledSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_runtime_lifecycle_with_controlled_http_and_independent_cleanup(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        service = ControlledHindsightHTTP()
        configuration = KnowledgeConfiguration(base_url="http://127.0.0.1:19999",
            api_key="controlled-offline-key", area_id=uuid4())
        directory = root / ("memory-lifecycle-controlled-" + uuid4().hex)
        with patch("app.knowledge.runtime.HindsightClient", side_effect=service.client), \
             patch("evals.responsive_voice.memory_lifecycle_qualification.HindsightClient", side_effect=service.client):
            report = await run_qualification(dsn, configuration, directory,
                mode="controlled_hindsight_real_postgres")
        print("memory_lifecycle_report=" + str(directory / "qualification.json"))
        self.assertTrue(report["accepted"], json.dumps(report, indent=2))
        self.assertFalse(report["native_semantic_quality_qualified"])
        self.assertTrue(all(item["status"] == "passed" for item in report["stages"].values()))
        self.assertEqual(len(report["runtime_instances"]), 6)
        self.assertEqual(report["cleanup"]["database"]["remaining_schemas"], 0)
        self.assertEqual(report["cleanup"]["database"]["remaining_public_owner_rows"], 0)
        self.assertTrue(report["cleanup"]["memory"]["native_bank_absent"])
        self.assertFalse(service.banks)

