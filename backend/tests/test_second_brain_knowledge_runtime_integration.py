"""Owned worker and real capture/DB integration with a simulated memory service."""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.hindsight import HindsightClient, HindsightError, document_metadata, original_document
from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime, initialize_knowledge_runtime
from app.realtime.router import _create_context_capture
from app.runtime.feature_config import KnowledgeConfiguration, SecondBrainSettings
from evals.second_brain.source_fixture import source_fixture


class SimulatedMemoryService:
    verify_document = staticmethod(HindsightClient.verify_document)

    def __init__(self):
        self.documents, self.operations = {}, {}
        self.calls, self.closed = 0, 0
        self.unavailable = False
        self.gate = None
        self.entered = asyncio.Event()

    async def ensure_bank(self, bank):
        self.calls += 1
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.unavailable:
            raise HindsightError("synthetic_outage", retryable=True)

    async def operation(self, bank, operation_id):
        return self.operations.get(str(operation_id))

    async def retain(self, source, operation_id):
        self.operations[str(operation_id)] = {"operation_id": str(operation_id), "status": "completed"}
        self.documents[source.document_id] = dict(
            id=source.document_id, bank_id=source.bank_name, original_text=original_document(source),
            document_metadata=document_metadata(source),
        )
        return {"async": True, "operation_id": str(operation_id)}

    async def document(self, bank, document_id):
        return self.documents.get(document_id)

    async def delete_document(self, bank, document_id):
        self.documents.pop(document_id, None)

    def diagnostics(self):
        return {"simulated": True, "calls": self.calls, "provider_token_usage": None}

    async def close(self):
        self.closed += 1


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class KnowledgeRuntimeDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.provider = SimulatedMemoryService()
        self.options = SecondBrainSettings(
            _env_file=None, enabled="true", knowledge_enabled="true",
            hindsight_url="http://127.0.0.1:19999", hindsight_api_key="synthetic-memory-key",
            knowledge_area_id=str(self.f.area), knowledge_poll_seconds="0.02",
            knowledge_shutdown_seconds="0.05",
        )
        self.legacy = object()
        self.services = SimpleNamespace(
            settings=SimpleNamespace(SUPABASE_DB_URL=self.f.fixture.dsn, JARVIS_OWNER_ID=self.f.owner),
            db_pool=self.f.pool, knowledge_runtime=None, knowledge_issue=None,
            background_task_service=None, assistant_context_providers=(),
            memory_service=self.legacy, tool_modules=("unchanged-legacy-tools",),
        )
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner,
            )

    async def initialize(self):
        with patch("app.knowledge.runtime.get_second_brain_settings", return_value=self.options), \
             patch("app.knowledge.runtime.HindsightClient", return_value=self.provider):
            await initialize_knowledge_runtime(self.services)
        runtime = self.services.knowledge_runtime
        self.assertIsNotNone(runtime, self.services.knowledge_issue)
        self.addAsyncCleanup(runtime.close)
        return runtime

    async def capture(self, provider="openai"):
        capture = _create_context_capture(
            self.services, owner_id=self.f.owner, session_id=self.session,
            provider=provider, audio_session_id="runtime-" + provider,
        )
        self.addAsyncCleanup(capture.close)
        capture.input_final("input", "Synthetic source from " + provider)
        self.assertTrue(await capture.flush(), capture.diagnostics())
        return capture

    async def wait_for_sources(self, count):
        async with asyncio.timeout(20):
            while True:
                snapshot = await self.f.sources.current_snapshot(self.f.owner, self.f.area)
                if len(snapshot["sources"]) == count and all(
                    item.ingestion_status == "searchable" for item in snapshot["sources"]
                ):
                    return snapshot
                await asyncio.sleep(0.1)

    async def test_initializer_is_idempotent_and_does_not_replace_legacy_tools_or_memory(self):
        runtime = await self.initialize()
        await initialize_knowledge_runtime(self.services)
        self.assertIs(self.services.knowledge_runtime, runtime)
        self.assertEqual(len(self.services.assistant_context_providers), 1)
        self.assertIs(self.services.memory_service, self.legacy)
        self.assertEqual(self.services.tool_modules, ("unchanged-legacy-tools",))
        status = await runtime.diagnostics()
        self.assertTrue(status["worker_running"])
        self.assertEqual(status["queue"]["pending_operations"], 0)
        self.assertEqual(self.provider.calls, 0)
        self.assertIsNone(status["transport"]["provider_token_usage"])

    async def test_router_capture_helper_and_owned_worker_preserve_both_provider_originals(self):
        runtime = await self.initialize()
        await self.capture("openai")
        await self.capture("gemini")
        snapshot = await self.wait_for_sources(2)
        self.assertEqual({item.content["audio_provider"] for item in snapshot["sources"]},
                         {"openai", "gemini"})
        self.assertTrue(all(item.context_event_id is not None for item in snapshot["sources"]))
        self.assertTrue(all(item.context_session_id == self.session for item in snapshot["sources"]))
        self.assertEqual(len(self.provider.documents), 2)
        self.assertEqual(len(self.provider.operations), 2)
        status = await runtime.diagnostics()
        self.assertEqual(status["queue"]["pending_operations"], 0)
        self.assertIsNone(status["issue"])

    async def test_provider_outage_leaves_current_sources_readable_and_queue_state_truthful(self):
        self.provider.unavailable = True
        runtime = await self.initialize()
        await self.capture()
        async with asyncio.timeout(15):
            while True:
                status = await runtime.diagnostics()
                if status["queue"]["recorded_failures"]:
                    break
                await asyncio.sleep(0.1)
        local = await runtime.current_context.describe(self.f.owner, self.f.area, include_content=True)
        self.assertEqual(local["sources"][0]["content"]["text"], "Synthetic source from openai")
        self.assertEqual(local["not_searchable_sources"], 1)
        self.assertFalse(local["provider_contacted"])
        self.assertEqual(status["queue"]["pending_operations"], 1)
        self.assertTrue(status["worker_running"])
        self.assertEqual(len(self.provider.documents), 0)

    async def test_cancelled_inflight_step_is_recoverable_under_a_new_runtime(self):
        self.provider.gate = asyncio.Event()
        runtime = await self.initialize()
        capture = await self.capture()
        await asyncio.wait_for(self.provider.entered.wait(), timeout=15)
        await capture.close()
        await runtime.close()
        async with self.f.pool.acquire() as connection:
            before = await connection.fetchrow("select id,status,fence from jarvis_memory_outbox")
            self.assertEqual(before["status"], "leased")
            await connection.execute(
                "update jarvis_memory_outbox set lease_until=clock_timestamp()-interval '1 second'",
            )
        replacement = SimulatedMemoryService()
        with patch("app.knowledge.runtime.HindsightClient", return_value=replacement):
            restarted = await KnowledgeRuntime.create(
                pool=self.f.pool, owner_id=self.f.owner, configuration=runtime.configuration,
            )
        self.addAsyncCleanup(restarted.close)
        restarted.start()
        await self.wait_for_sources(1)
        async with self.f.pool.acquire() as connection:
            after = await connection.fetchrow("select id,status,fence from jarvis_memory_outbox")
        self.assertEqual(after["id"], before["id"])
        self.assertGreater(after["fence"], before["fence"])
        self.assertEqual(after["status"], "searchable")
        self.assertEqual(set(replacement.operations), {str(before["id"])})
        self.assertEqual(self.provider.closed, 1)

    async def test_foreign_area_is_denied_before_creating_a_memory_client(self):
        async with self.f.pool.acquire() as connection:
            other_area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other','test') returning id",
                self.f.other,
            )
        config = KnowledgeConfiguration(
            base_url="http://127.0.0.1:19999", api_key="synthetic-memory-key", area_id=other_area,
        )
        with patch("app.knowledge.runtime.HindsightClient") as client, self.assertRaises(SourceError):
            await KnowledgeRuntime.create(pool=self.f.pool, owner_id=self.f.owner, configuration=config)
        client.assert_not_called()
