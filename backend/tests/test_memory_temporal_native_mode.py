"""Offline readiness only: every native client uses a controlled HTTP wire."""
import asyncio
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

from app.knowledge.hindsight import HindsightError
from app.runtime.feature_config import KnowledgeConfiguration
from evals.responsive_voice.memory_lifecycle_qualification import QualificationFailure, save
from evals.responsive_voice import memory_temporal_native as native
from evals.responsive_voice import memory_temporal_restart_qualification as runner
from evals.responsive_voice.memory_temporal_transport import PersistentControlledMemory


class NativeTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bank = "jarvis_temporal_eval_" + uuid4().hex
        self.origin = "http://127.0.0.1:19999"
        self.config = KnowledgeConfiguration(base_url=self.origin, api_key="offline-secret", area_id=uuid4())
        self.budget = native.PersistentActualMemory(self.root / "native-budget.json", self.bank, self.origin, create=True)
        self.seen = []

    async def client(self, response=None):
        def wire(request):
            self.seen.append((request.method, request.url.path))
            return response(request) if response else httpx.Response(200, json={"results": []})
        client = self.budget.client(self.origin, "offline-secret", transport=httpx.MockTransport(wire))
        self.addAsyncCleanup(client.close)
        return client

    async def test_real_client_wrapper_counts_failed_attempts_before_dispatch(self):
        self.budget.admit_bank()
        client = await self.client(lambda request: httpx.Response(503, json={"detail": "private response"}))
        with self.assertRaises(HindsightError):
            await client._call("GET", client._bank(self.bank) + "/config")
        state = native.PersistentActualMemory(self.budget.path, self.bank, self.origin).read()
        self.assertEqual(state["main_http"], 1)
        self.assertEqual(state["failed_http"], 1)
        self.assertNotIn("offline-secret", self.budget.path.read_text())
        self.assertNotIn("private response", self.budget.path.read_text())

    async def test_origin_bank_operation_and_global_recall_guards_precede_wire(self):
        self.budget.admit_bank()
        client = await self.client()
        for path in ("https://api.openai.com/v1/realtime/calls", "/v1/default/banks/foreign/config",
                     client._bank(self.bank) + "/reflect"):
            with self.subTest(path=path), self.assertRaises(QualificationFailure):
                await client._call("POST", path)
        self.assertFalse(self.seen)
        for index in range(12):
            await client._call("POST", client._bank(self.bank) + "/memories/recall", body={"query": "synthetic"})
        with self.assertRaises(QualificationFailure):
            await client._call("POST", client._bank(self.bank) + "/memories/recall", body={"query": "synthetic"})
        self.assertEqual(len(self.seen), 12)
        self.assertEqual(self.budget.read()["recalls"], 12)
        with self.assertRaises(QualificationFailure):
            self.budget.client("http://127.0.0.1:19998", "offline-secret")

    async def test_retain_payload_and_http_caps_persist_without_logging_content(self):
        self.budget.admit_bank()
        client = await self.client()
        await client._call("POST", client._bank(self.bank) + "/memories",
            body={"items": [{"content": "PRIVATE_SYNTHETIC_CONTENT"}]})
        state = self.budget.read()
        self.assertEqual(state["retain_requests"], 1)
        self.assertEqual(state["retain_characters"], len("PRIVATE_SYNTHETIC_CONTENT"))
        self.assertNotIn("PRIVATE_SYNTHETIC_CONTENT", self.budget.path.read_text())
        state["retain_requests"] = native.RETAIN_LIMIT
        save(self.budget.path, state)
        with self.assertRaises(QualificationFailure):
            await client._call("POST", client._bank(self.bank) + "/memories", body={"items": [{"content": "next"}]})
        state["main_http"] = native.MAIN_HTTP_LIMIT
        save(self.budget.path, state)
        with self.assertRaises(QualificationFailure):
            await client._call("GET", client._bank(self.bank) + "/config")
        self.assertEqual(len(self.seen), 1)
        self.budget.begin_cleanup()
        await client._call("DELETE", client._bank(self.bank))
        await client._call("GET", "/v1/default/banks")
        self.assertEqual(self.budget.read()["cleanup_http"], 2)
        with self.assertRaises(QualificationFailure):
            await client._call("POST", client._bank(self.bank) + "/memories/recall")
        state = self.budget.read()
        state["cleanup_http"] = native.CLEANUP_HTTP_LIMIT
        save(self.budget.path, state)
        with self.assertRaises(QualificationFailure):
            await client._call("GET", "/v1/default/banks")

    async def test_unknown_failed_recall_is_not_retried_and_clocks_remain_monotonic(self):
        self.budget.admit_bank()
        client = await self.client(lambda request: httpx.Response(503, json={}))
        with self.assertRaises(HindsightError):
            await client._call("POST", client._bank(self.bank) + "/memories/recall")
        self.assertEqual(self.budget.read()["recalls"], 1)
        self.assertEqual(len(self.seen), 1)
        state = self.budget.read()
        state["main_deadline_monotonic"] = 0
        save(self.budget.path, state)
        with self.assertRaises(QualificationFailure):
            await client._call("GET", client._bank(self.bank) + "/config")
        self.assertEqual(len(self.seen), 1)

    async def test_preflight_requires_actual_inventory_receipt_and_nonexistent_bank(self):
        client = await self.client(lambda request: httpx.Response(200, json={"banks": [{"bank_id": self.bank}]}))
        with patch.object(self.budget, "client", return_value=client):
            with self.assertRaises(QualificationFailure):
                await self.budget.preflight(self.config)
        self.assertFalse(self.budget.read()["bank_admitted"])
        self.assertEqual(self.seen, [("GET", "/v1/default/banks")])

    async def test_successful_preflight_and_independent_cleanup_clients_share_budget(self):
        first = await self.client(lambda request: httpx.Response(200, json={"banks": []}))
        with patch.object(self.budget, "client", return_value=first):
            await self.budget.preflight(self.config)
        self.assertTrue(self.budget.read()["bank_admitted"])
        self.budget.begin_cleanup()
        second = await self.client(lambda request: httpx.Response(200, json={"banks": []}))
        self.assertIsNot(first, second)
        inventory = await second._call("GET", "/v1/default/banks")
        self.assertEqual(inventory, {"banks": []})
        self.assertEqual(self.budget.read()["main_http"], 1)
        self.assertEqual(self.budget.read()["cleanup_http"], 1)
        self.assertIsNone(self.budget.report()["provider_tokens"])
        self.assertIsNone(self.budget.report()["dollar_cost"])


class NativeModeReadinessTests(unittest.TestCase):
    def test_actual_release_fails_before_configuration_or_io(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(native, "configured_inputs") as configuration:
            with self.assertRaisesRegex(QualificationFailure, "release_required"):
                native.actual_inputs()
            configuration.assert_not_called()

    def test_file_credentials_are_required_and_not_serialized_into_child_metadata(self):
        with TemporaryDirectory() as directory:
            key = Path(directory) / "api-key"
            key.write_text("offline-secret")
            config = KnowledgeConfiguration(base_url="http://127.0.0.1:19999", api_key="offline-secret", area_id=uuid4())
            with patch.dict(os.environ, {native.RELEASE_ENV: "1", "SECOND_BRAIN_HINDSIGHT_API_KEY_FILE": str(key)}, clear=True), \
                 patch.object(native, "configured_inputs", return_value=("fixture-dsn", SimpleNamespace(configured=True, configuration=config))):
                self.assertEqual(native.actual_inputs(), ("fixture-dsn", config))
                os.environ["SECOND_BRAIN_HINDSIGHT_API_KEY"] = "inline-secret"
                with self.assertRaisesRegex(QualificationFailure, "inline_key_forbidden"):
                    native.actual_inputs()
            report = runner.new_child_report("day0", runner.DomainClock(), 0, mode="actual")
            self.assertIn("actual_hindsight", report["evidence_mode"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["clock"]["scheduler_time_travel"])
            self.assertNotIn("Controlled lexical", " ".join(report["limitations"]))
            self.assertNotIn("offline-secret", json.dumps(report))

    def test_controlled_factory_remains_controlled_even_with_release_present(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {native.RELEASE_ENV: "1"}), \
             patch.object(runner, "PersistentActualMemory") as actual:
            transport = runner.transport_for(Path(directory), "jarvis_temporal_eval_" + uuid4().hex,
                "controlled", None, create=True)
            self.assertIsInstance(transport, PersistentControlledMemory)
            actual.assert_not_called()

    def test_six_actual_budget_process_claims_and_no_stage_replay(self):
        with TemporaryDirectory() as directory:
            path, bank = Path(directory) / "budget.json", "jarvis_temporal_eval_" + uuid4().hex
            origin = "http://127.0.0.1:19999"
            budget = native.PersistentActualMemory(path, bank, origin, create=True)
            budget.admit_bank()
            program = ("import sys; from evals.responsive_voice.memory_temporal_native import PersistentActualMemory; "
                "b=PersistentActualMemory(sys.argv[1],sys.argv[2],sys.argv[3]); "
                "b.claim_child(sys.argv[4],int(sys.argv[5])); "
                "b._reserve('GET','/v1/default/banks/'+sys.argv[2]+'/config',None)")
            for position, stage in enumerate(runner.STAGES):
                subprocess.run([sys.executable, "-c", program, str(path), bank, origin, stage, str(position)],
                    check=True, capture_output=True, text=True, timeout=10)
            state = budget.read()
            self.assertEqual(state["main_http"], 6)
            self.assertEqual(len({item["pid"] for item in state["children"]}), 6)
            self.assertEqual([item["stage"] for item in state["children"]], list(runner.STAGES))
            with self.assertRaises(QualificationFailure):
                budget.claim_child("day0", 0)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)


class NativeRunnerGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_programmatic_entry_is_blocked_before_fixture_creation(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
             patch.object(runner, "DatabaseFixture") as fixture, patch.object(runner.asyncpg, "connect") as connect:
            target = Path(directory) / "must-not-exist"
            with self.assertRaisesRegex(QualificationFailure, "release_required"):
                await runner.run_actual("fixture-dsn", None, target)
            self.assertFalse(target.exists())
            fixture.assert_not_called()
            connect.assert_not_called()
