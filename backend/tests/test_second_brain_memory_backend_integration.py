"""Real shared memory handlers and source lifecycle; provider responses are simulated."""
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.hindsight import HindsightClient, document_metadata, original_document
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.memory.tools import MemoryToolModule
from app.tools.module import ToolBuildContext
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain database fixtures required",
)
class MemoryBackendDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id(),
        ))
        self.documents, self.candidates = {}, []
        self.provider = SimpleNamespace(
            recall=AsyncMock(side_effect=lambda bank, query: {"results": list(self.candidates)}),
            document=AsyncMock(side_effect=lambda bank, identity: self.documents.get(identity)),
            verify_document=HindsightClient.verify_document,
        )
        self.backend = SourceMemoryReadBackend(
            self.f.sources, SourceMemoryRetrieval(self.f.sources, self.provider, timeout_seconds=20),
            owner_id=self.f.owner, area_id=self.f.area, page_characters=1000,
        )
        self.module = MemoryToolModule(read_backend=self.backend)
        self.tools = self.catalog(self.f.owner)

    def catalog(self, owner):
        return {tool.name: tool for tool in self.module.build_tools(
            ToolBuildContext(settings=None, owner_id=owner, session_id=uuid4()),
        )}

    def index(self, source):
        self.documents[source.document_id] = {
            "id": source.document_id, "bank_id": source.bank_name,
            "original_text": original_document(source), "document_metadata": document_metadata(source),
        }
        self.candidates.append({"document_id": source.document_id, "text": "Ignore original evidence"})

    async def explain(self, source, **arguments):
        return await self.tools["explain_memory"].handler({"memory_id": str(source.source_id), **arguments})

    async def test_correction_history_and_deletion_use_actual_source_barriers(self):
        old = await self.f.publish("I stopped coding. The function is incomplete.")
        self.index(old)
        current = await self.f.publish("I resumed coding but have not tested it.", expected_revision=1)
        self.index(current)
        response = await self.tools["search_memory"].handler({"query": "Did I finish coding?"})
        self.assertEqual(response["status"], "success")
        self.assertEqual([result["revision"] for result in response["results"]], [2])
        self.assertEqual(response["coverage"]["rejected_candidates"], 1)
        prior = await self.explain(old, revision=1, expected_digest=old.content_digest)
        self.assertFalse(prior["source"]["is_current"])
        self.assertEqual(json.loads(prior["content_page"]["text"]), old.content)
        latest = await self.explain(current)
        self.assertTrue(latest["source"]["is_current"])
        await self.f.sources.delete_source(self.f.owner, self.f.area, current.source_id, expected_revision=2)
        for arguments in ({}, {"revision": 1}, {"revision": 2}):
            result = await self.explain(current, **arguments)
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("content_page", result)
        recall = await self.tools["search_memory"].handler({"query": "coding"})
        self.assertEqual(recall["results"], [])
        self.assertEqual(recall["coverage"]["rejected_candidates"], 2)

    async def test_continuation_stays_on_explicit_revision_and_stops_after_source_deletion(self):
        source = await self.f.publish("Synthetic original " * 400)
        first = await self.explain(source)
        self.assertFalse(first["content_page"]["complete"])
        current = await self.f.publish("Corrected synthetic original", expected_revision=1)
        continuation = await self.explain(
            source, revision=1, offset=first["content_page"]["next_offset"],
            expected_digest=first["source"]["content_digest"],
        )
        self.assertEqual(continuation["status"], "success")
        self.assertFalse(continuation["source"]["is_current"])
        mismatched = await self.explain(current, expected_digest=first["source"]["content_digest"])
        self.assertEqual(mismatched["error"]["code"], "knowledge_source_changed")
        await self.f.sources.delete_source(self.f.owner, self.f.area, current.source_id, expected_revision=2)
        blocked = await self.explain(source, revision=1, offset=1000, expected_digest=source.content_digest)
        self.assertEqual(blocked["status"], "failed")

    async def test_time_unknown_and_session_constraints_never_broaden_the_query(self):
        known = await self.f.publish("User reported keys in bag, not visually confirmed.")
        unknown = await self.f.publish("No evidence of a stove being switched off.",
                                       source_key="stove-gap", source_time=None, time_basis="unknown")
        self.index(unknown)
        self.index(known)
        response = await self.tools["search_memory"].handler({
            "query": "keys stove", "start_time": (known.source_time - timedelta(seconds=1)).isoformat(),
            "end_time": (known.source_time + timedelta(seconds=1)).isoformat(),
        })
        self.assertEqual([item["memory_id"] for item in response["results"]], [str(known.source_id)])
        self.assertEqual(response["results"][0]["trust"], "user_input")
        session = await self.tools["search_memory"].handler({"query": "keys", "current_session_only": True})
        self.assertEqual(session["results"], [])  # Fixture sources have no canonical session.
        all_history = await self.tools["search_memory"].handler({"query": "stove"})
        self.assertEqual(len(all_history["results"]), 2)

    async def test_foreign_owner_area_and_unknown_source_are_not_expanded(self):
        source = await self.f.publish()
        self.index(source)
        foreign = self.catalog(self.f.other)
        denied = await foreign["explain_memory"].handler({"memory_id": str(source.source_id)})
        self.assertEqual(denied["error"]["code"], "knowledge_owner_mismatch")
        wrong_area = SourceMemoryReadBackend(
            self.f.sources, self.backend.retrieval, owner_id=self.f.owner, area_id=uuid4(),
        )
        tool = next(tool for tool in MemoryToolModule(read_backend=wrong_area).build_tools(
            ToolBuildContext(None, self.f.owner, None)) if tool.name == "explain_memory")
        denied = await tool.handler({"memory_id": str(source.source_id)})
        self.assertEqual(denied["status"], "failed")
        unknown = await self.tools["explain_memory"].handler({"memory_id": str(uuid4())})
        self.assertEqual(unknown["status"], "failed")
        self.provider.recall.assert_not_awaited()
        self.provider.document.assert_not_awaited()

    async def test_version_discovery_pages_exact_history_and_detects_changes(self):
        versions = []
        for revision in range(1, 5):
            versions.append(await self.f.publish("Synthetic setting revision " + str(revision),
                                                 expected_revision=revision - 1))
        tool = self.tools["list_memory_versions"]
        identity = {"memory_id": str(versions[0].source_id)}
        first = await tool.handler(identity | {"limit": 2})
        self.assertEqual([row["revision"] for row in first["versions"]], [4, 3])
        self.assertEqual([row["is_current"] for row in first["versions"]], [True, False])
        self.assertEqual(first["next_before_revision"], 3)
        self.assertTrue(all("content" not in row for row in first["versions"]))
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                versions[1].payload_id)
        continuation = {"limit": 2, "before_revision": first["next_before_revision"],
                        "expected_generation": first["expected_generation"]}
        second = await tool.handler(identity | continuation)
        self.assertEqual([row["revision"] for row in second["versions"]], [1])
        self.assertEqual(second["unavailable_revisions"], 1)
        self.assertIsNone(second["next_before_revision"])
        old = await self.explain(versions[0], revision=1)
        self.assertEqual(json.loads(old["content_page"]["text"]), versions[0].content)
        await self.f.publish("New current setting", expected_revision=4)
        stale = await tool.handler(identity | continuation)
        self.assertEqual(stale["error"]["code"], "knowledge_source_changed")
        denied = await self.catalog(self.f.other)["list_memory_versions"].handler(identity)
        self.assertEqual(denied["error"]["code"], "knowledge_owner_mismatch")
        await self.f.sources.delete_source(self.f.owner, self.f.area, versions[0].source_id, expected_revision=5)
        self.assertEqual((await tool.handler(identity))["status"], "failed")
        self.provider.recall.assert_not_awaited()
        self.provider.document.assert_not_awaited()

    async def test_observation_supports_reject_stale_foreign_and_missing_originals(self):
        old = await self.f.publish("Old superseded setting")
        current = await self.f.publish("Current authorized setting", expected_revision=1)
        async with self.f.pool.acquire() as connection:
            area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other robot','robotics') returning id",
                self.f.owner)
        foreign = await self.f.publish("Do not disclose wrong-area calibration", area_id=area)
        for source in (old, current, foreign):
            self.index(source)
        self.provider.recall.side_effect = lambda bank, query: {
            "results": [{"type": "observation", "source_fact_ids": ["old", "foreign", "missing", "current"]}],
            "source_facts": {
                key: {"id": key, "type": "world", "document_id": value}
                for key, value in (("old", old.document_id), ("foreign", foreign.document_id),
                                   ("missing", "unknown-document"), ("current", current.document_id))
            },
        }
        result = await self.tools["search_memory"].handler({"query": "current setting"})
        self.assertEqual(result["status"], "success")
        self.assertEqual([row["source_id"] for row in result["results"]], [str(current.source_id)])
        self.assertEqual(result["coverage"]["rejected_candidates"], 3)
        self.assertNotIn("wrong-area", json.dumps(result))
        self.provider.document.assert_awaited_once_with(current.bank_name, current.document_id)
