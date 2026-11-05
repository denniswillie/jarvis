"""Shared-tool source reads, without vendor services or legacy coercion."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.hindsight import HindsightClient, HindsightError, document_metadata, original_document
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.retrieval import SourceMemoryRetrieval, SourceRecallScope
from app.knowledge.sources import SourceRepository
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.tools import MemoryToolModule, create_memory_tools
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry


class SourceMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.area, self.session = "fixture-owner", uuid4(), uuid4()
        self.when = datetime(2026, 9, 14, 12, tzinfo=UTC)
        self.originals = {}
        self.candidates = []
        self.connection = SimpleNamespace(fetchval=AsyncMock(return_value="fixture-bank"))

        @asynccontextmanager
        async def transaction(owner):
            yield self.connection

        async def resolve(owner, area, document_id):
            source = self.originals.get(document_id)
            if source is None or not source.is_current:
                raise SourceError("knowledge_source_unavailable", "Source unavailable")
            return source

        async def read(owner, area, identity, *, revision=None):
            for source in self.originals.values():
                if source.source_id == identity and (revision is None or source.revision == revision):
                    return source
            raise SourceError("knowledge_source_not_found", "Source unavailable", 404)

        async def revalidate(source):
            return await resolve(source.owner_id, source.area_id, source.document_id)

        async def recall(bank, query):
            return {"results": list(self.candidates)}

        async def document(bank, identity):
            source = self.originals[identity]
            return {"id": identity, "bank_id": bank, "original_text": original_document(source),
                    "document_metadata": document_metadata(source)}

        self.sources = SimpleNamespace(
            transaction=transaction, _area=AsyncMock(), resolve_document=AsyncMock(side_effect=resolve),
            revalidate=AsyncMock(side_effect=revalidate), read=AsyncMock(side_effect=read),
        )
        self.provider = SimpleNamespace(
            recall=AsyncMock(side_effect=recall), document=AsyncMock(side_effect=document),
            verify_document=HindsightClient.verify_document,
        )
        self.retrieval = SourceMemoryRetrieval(self.sources, self.provider)
        self.backend = SourceMemoryReadBackend(
            self.sources, self.retrieval, owner_id=self.owner, area_id=self.area,
            page_characters=1000, now=lambda: self.when,
        )
        self.tools = self.catalog(self.owner)

    def catalog(self, owner, session=True):
        registry = JarvisToolRegistry()
        registry.register_many(MemoryToolModule(read_backend=self.backend).build_tools(
            ToolBuildContext(settings=None, owner_id=owner, session_id=self.session if session else None),
        ))
        return {tool.name: tool for tool in registry.list()}

    def add_source(self, **changes):
        source = SourceVersion(**(dict(
            owner_id=self.owner, area_id=self.area, source_id=uuid4(), revision=1, generation=1,
            payload_id=uuid4(), document_id="fixture-" + uuid4().hex, bank_name="fixture-bank",
            content_digest="a" * 64, source_type="user_transcript", trust="user_input",
            source_time=self.when, time_basis="original report", recorded_at=self.when,
            content={"text": "I stopped coding; I did not complete the function."},
            is_current=True, ingestion_status="searchable", context_session_id=self.session,
        ) | changes))
        self.originals[source.document_id] = source
        self.candidates.append({"document_id": source.document_id, "text": "UNSUPPORTED: completed coding"})
        return source

    async def search(self, **arguments):
        return await self.tools["search_memory"].handler({"query": "What did I do?", **arguments})

    async def explain(self, source, **arguments):
        return await self.tools["explain_memory"].handler({"memory_id": str(source.source_id), **arguments})

    def test_source_catalog_is_read_only_and_legacy_catalog_is_unchanged(self):
        self.assertEqual(set(self.tools), {"search_memory", "explain_memory", "list_memory_versions"})
        self.assertTrue(all(not tool.mutates_state for tool in self.tools.values()))
        schema = self.tools["search_memory"].input_schema["properties"]
        for unsupported in ("memory_types", "include_superseded", "include_evidence_metadata"):
            self.assertNotIn(unsupported, schema)
        legacy = create_memory_tools(repository=object(), entity_repository=object(),
                                     service=object(), owner_id=self.owner)
        self.assertEqual({tool.name for tool in legacy},
                         {"get_semantic_fact_history", "search_memory", "remember", "explain_memory"})
        self.assertNotIn("revision", next(tool for tool in legacy if tool.name == "explain_memory").input_schema["properties"])
        for key in ("repository", "entity_repository", "service", "forgetting_service",
                    "projection_service", "image_service"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                create_memory_tools(owner_id=self.owner, read_backend=self.backend, **{key: object()})

    async def test_search_returns_original_evidence_without_vendor_prose_or_fake_confidence(self):
        source = self.add_source()
        response = await self.search()
        self.assertEqual(response["status"], "success")
        result = response["results"][0]
        self.assertEqual(result["memory_id"], str(source.source_id))
        self.assertEqual(result["content"], source.content)
        self.assertEqual(result["trust"], "user_input")
        self.assertEqual(result["source_time"], source.model_dump(mode="json")["source_time"])
        self.assertNotIn("UNSUPPORTED", json.dumps(response))
        for key in ("confidence", "importance", "occurred_at", "record_type", "bank_name"):
            self.assertNotIn(key, result)
        self.assertIsNone(response["constraints"]["start_time"])
        self.assertIsNone(response["vendor_usage"])

    async def test_time_and_session_filter_before_limit_and_do_not_use_ingestion_time(self):
        self.retrieval.max_sources = 1
        self.add_source(context_session_id=uuid4())
        self.add_source(source_time=None)
        self.add_source(source_time=self.when - timedelta(days=1))
        target = self.add_source(source_time=self.when - timedelta(seconds=5))
        self.add_source(source_time=self.when)  # Exclusive end bound.
        response = await self.search(seconds_before_now=10, current_session_only=True)
        self.assertEqual([item["memory_id"] for item in response["results"]], [str(target.source_id)])
        self.assertEqual(response["coverage"]["filtered_candidates"], 4)
        self.provider.document.assert_awaited_once_with("fixture-bank", target.document_id)

    async def test_unknown_source_time_is_preserved_without_an_explicit_time_filter(self):
        self.add_source(source_time=None, time_basis="unknown")
        response = await self.search()
        self.assertIsNone(response["results"][0]["source_time"])
        self.assertEqual(response["results"][0]["time_basis"], "unknown")

    async def test_invalid_and_unsupported_filters_fail_before_provider_calls(self):
        cases = [
            {"owner_id": "foreign"}, {"memory_types": ["semantic_fact"]}, {"include_superseded": True},
            {"start_time": self.when.isoformat()}, {"seconds_before_now": True},
            {"seconds_before_now": 10, "end_time": self.when.isoformat()},
            {"seconds_before_now": 10 ** 100}, {"query": "x" * 8001},
            {"query": "\u00e9" * 5000},
        ]
        for arguments in cases:
            with self.subTest(arguments=str(arguments)[:100]):
                response = await self.search(**arguments)
                self.assertEqual(response["status"], "failed")
                self.assertEqual(response["error"]["code"], "invalid_tool_input")
        self.provider.recall.assert_not_awaited()
        no_session = self.catalog(self.owner, session=False)
        self.assertEqual((await no_session["search_memory"].handler(
            {"query": "keys", "current_session_only": True}))["status"], "failed")

    async def test_foreign_owner_is_denied_before_database_or_provider_access(self):
        source = self.add_source()
        tools = self.catalog("foreign-owner")
        for name, arguments in (("search_memory", {"query": "keys"}),
                                ("explain_memory", {"memory_id": str(source.source_id)})):
            result = await tools[name].handler(arguments)
            self.assertEqual(result["error"]["code"], "knowledge_owner_mismatch")
        self.connection.fetchval.assert_not_awaited()
        self.sources.read.assert_not_awaited()
        self.provider.recall.assert_not_awaited()

    async def test_outage_is_not_reported_as_empty_memory_but_local_expansion_still_works(self):
        source = self.add_source(ingestion_status="accepted")
        self.provider.recall.side_effect = HindsightError("synthetic_outage", retryable=True)
        response = await self.search()
        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["error"]["code"], "memory_provider_unavailable")
        self.assertNotIn("results", response)
        original = await self.explain(source)
        self.assertEqual(original["status"], "success")
        self.assertEqual(original["source"]["ingestion_status"], "accepted")
        self.assertFalse(original["provider_contacted"])

    async def test_large_source_has_a_handle_and_pages_reconstruct_the_exact_original(self):
        source = self.add_source(content={"text": "literal \\ slash, quote \" and unicode \u00e9; " * 2000})
        self.retrieval.max_characters = 2500
        response = await self.search()
        found = response["results"][0]
        self.assertFalse(found["content_included"])
        self.assertNotIn("content", found)
        self.assertEqual(response["coverage"]["omitted_content"], 1)
        chunks, offset = [], 0
        while True:
            page = await self.explain(source, revision=source.revision, offset=offset,
                                      expected_digest=source.content_digest)
            self.assertEqual(page["status"], "success")
            chunks.append(page["content_page"]["text"])
            self.assertLessEqual(len(chunks[-1]), 1000)
            offset = page["content_page"]["next_offset"]
            if offset is None:
                break
        self.assertEqual(json.loads("".join(chunks)), source.content)
        self.assertFalse(page["content_page"]["complete"])

    async def test_invalid_page_identity_and_unknown_source_do_not_release_content(self):
        source = self.add_source()
        cases = [
            {"offset": 1}, {"offset": 1, "revision": 1}, {"offset": True},
            {"offset": -1}, {"revision": True}, {"revision": 0}, {"offset": None},
            {"expected_digest": "b" * 64}, {"expected_digest": "not-a-digest"},
            {"offset": 100000, "revision": 1, "expected_digest": source.content_digest},
            {"bank_name": "model-selected"}, {"memory_id": str(uuid4())},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                response = await self.explain(source, **arguments)
                self.assertEqual(response["status"], "failed")
                self.assertNotIn("content_page", response)

    async def test_unknown_candidates_and_post_network_revocation_do_not_release_evidence(self):
        first = self.add_source()
        second = self.add_source()
        self.candidates.insert(0, {"document_id": "unmapped"})
        self.candidates.append({"text": "missing ID"})
        self.candidates.append(self.candidates[1])
        original_document_call = self.provider.document.side_effect

        async def deleting_document(bank, identity):
            result = await original_document_call(bank, identity)
            if identity == second.document_id:
                self.originals.pop(first.document_id)
            return result

        self.provider.document.side_effect = deleting_document
        response = await self.search()
        self.assertEqual([item["memory_id"] for item in response["results"]], [str(second.source_id)])
        self.assertEqual(response["coverage"]["rejected_candidates"], 3)

    async def test_mismatched_document_rejects_only_that_candidate_not_valid_originals(self):
        mismatched = self.add_source()
        valid = self.add_source()
        get_document = self.provider.document.side_effect

        async def substituted_document(bank, identity):
            document = await get_document(bank, identity)
            if identity == mismatched.document_id:
                document["original_text"] = "Vendor substituted a different original."
            return document

        self.provider.document.side_effect = substituted_document
        response = await self.search()
        self.assertEqual(response["status"], "success")
        self.assertEqual([item["memory_id"] for item in response["results"]], [str(valid.source_id)])
        self.assertEqual(response["coverage"]["rejected_candidates"], 1)
        self.assertNotIn("substituted", json.dumps(response))

    async def test_document_transport_failure_is_still_a_search_outage(self):
        self.add_source()
        self.provider.document.side_effect = HindsightError("hindsight_transport_failure", retryable=True)
        response = await self.search()
        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["error"]["code"], "memory_provider_unavailable")
        self.assertNotIn("results", response)

    async def test_source_scope_is_rechecked_after_network_wait(self):
        source = self.add_source()
        self.sources.revalidate.side_effect = None
        self.sources.revalidate.return_value = source.model_copy(update={"context_session_id": uuid4()})
        result = await self.search(current_session_only=True)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["coverage"]["filtered_candidates"], 1)

    async def test_lifecycle_guard_is_rechecked_before_releasing_async_results(self):
        source = self.add_source()
        calls = 0
        def guard(owner):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise SourceError("knowledge_intake_closed", "Runtime closed", 503)
        self.backend.access_check = guard
        response = await self.explain(source)
        self.assertEqual(response["error"]["code"], "knowledge_intake_closed")
        self.assertNotIn("content_page", response)

    async def test_candidate_scan_and_response_budget_are_bounded(self):
        for index in range(70):
            self.add_source(content={"text": "x" * 5000})
        self.retrieval.max_characters = 2500
        result = await self.retrieval.recall(self.owner, self.area, "fixture")
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=True)), 2500)
        self.assertEqual(result["unscanned_candidates"], 6)
        self.assertLessEqual(self.provider.document.await_count, self.retrieval.max_sources)
        self.assertGreater(result["omitted_sources"], 0)

    def test_historical_context_revision_is_not_mislabeled_current(self):
        source = self.add_source()
        row = source.model_dump() | {
            "source_status": "active", "availability": "available", "current_revision": 1,
            "current_generation": 1, "context_is_current": False,
        }
        self.assertFalse(SourceRepository._version(row, historical=True).is_current)
        with self.assertRaises(SourceError):
            SourceRepository._version(row)
        for kwargs in (
            {"start_time": self.when}, {"start_time": self.when, "end_time": self.when},
            {"start_time": datetime(2026, 1, 1), "end_time": self.when}, {"session_id": "foreign"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SourceRecallScope(**kwargs)

    async def test_native_observation_expands_only_linked_verified_originals(self):
        first, second, unlinked = self.add_source(), self.add_source(), self.add_source()
        self.provider.recall.side_effect = None
        self.provider.recall.return_value = {
            "results": [{"id": "observation", "type": "observation", "text": "UNSUPPORTED inference",
                         "source_fact_ids": ["first", "second", "first"],
                         "scores": {"semantic": 0.2, "final": 0.7, "text": "private", "keyword": True}}],
            "source_facts": {
                "first": {"id": "first", "type": "world", "document_id": first.document_id},
                "second": {"id": "second", "type": "experience", "document_id": second.document_id},
                "unlinked": {"id": "unlinked", "type": "world", "document_id": unlinked.document_id},
            },
        }
        result = await self.search()
        self.assertEqual([item["source_id"] for item in result["results"]],
                         [str(first.source_id), str(second.source_id)])
        self.assertEqual(self.provider.document.await_count, 2)
        for item in result["results"]:
            self.assertEqual(item["retrieval_match"], {
                "candidate_rank": 1, "basis": "observation_support",
                "native_scores": {"semantic": 0.2, "final": 0.7},
            })
        self.assertNotIn("UNSUPPORTED", json.dumps(result))
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(result["coverage"]["rejected_candidates"], 0)
        self.assertGreaterEqual(result["retrieval"]["elapsed_ms"], 0)

    async def test_malformed_cyclic_and_missing_supports_are_not_evidence(self):
        source = self.add_source()
        self.provider.recall.side_effect = None
        self.provider.recall.return_value = {
            "results": [{"type": "observation",
                         "source_fact_ids": ["cycle", "absent", "mismatch", {}, "nested", "good"],
                         "scores": {"semantic": float("nan"), "final": 10 ** 1000}}],
            "source_facts_truncated": True,
            "source_facts": {
                "cycle": {"id": "cycle", "type": "observation", "source_fact_ids": ["cycle"],
                          "document_id": source.document_id},
                "mismatch": {"id": "other", "type": "world", "document_id": source.document_id},
                "nested": {"id": "nested", "type": "world", "source_fact_ids": ["good"],
                           "document_id": source.document_id},
                "good": {"id": "good", "type": "world", "document_id": source.document_id},
            },
        }
        result = await self.search()
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["coverage"]["missing_supporting_facts"], 5)
        self.assertTrue(result["coverage"]["source_facts_truncated"])
        self.assertEqual(result["results"][0]["retrieval_match"]["native_scores"], {})
        self.provider.document.assert_awaited_once()

    async def test_support_expansion_is_bounded_and_retains_native_order(self):
        sources = [self.add_source() for _ in range(80)]
        facts = {str(i): {"id": str(i), "type": "world", "document_id": source.document_id}
                 for i, source in enumerate(sources)}
        self.provider.recall.side_effect = None
        self.provider.recall.return_value = {
            "results": [{"type": "observation", "source_fact_ids": [str(i) for i in range(start, start + 16)]}
                        for start in range(0, 80, 16)] +
                       [{"type": "observation", "source_fact_ids": list(facts)}],
            "source_facts": facts,
        }
        result = await self.search()
        self.assertEqual([row["source_id"] for row in result["results"]],
                         [str(source.source_id) for source in sources[:8]])
        self.assertEqual(result["coverage"]["unscanned_documents"], 16)
        self.assertEqual(result["coverage"]["omitted_supporting_facts"], 64)
        self.assertEqual(self.sources.resolve_document.await_count, 64)
        self.assertEqual(self.provider.document.await_count, 8)

    async def test_support_expansion_rechecks_sources_after_network_wait(self):
        first, second = self.add_source(), self.add_source()
        self.provider.recall.side_effect = None
        self.provider.recall.return_value = {
            "results": [{"type": "observation", "source_fact_ids": ["first", "second"]}],
            "source_facts": {
                "first": {"id": "first", "type": "world", "document_id": first.document_id},
                "second": {"id": "second", "type": "world", "document_id": second.document_id},
            },
        }
        original = self.provider.document.side_effect
        async def removing(bank, identity):
            value = await original(bank, identity)
            if identity == second.document_id:
                self.originals.pop(first.document_id)
            return value
        self.provider.document.side_effect = removing
        result = await self.search()
        self.assertEqual([row["source_id"] for row in result["results"]], [str(second.source_id)])
        self.assertEqual(result["coverage"]["rejected_candidates"], 1)

    async def test_version_navigation_uses_local_scoped_reader_and_safe_audit(self):
        source = self.add_source()
        metadata = source.original_evidence()
        metadata.pop("content")
        self.sources.list_versions = AsyncMock(return_value={
            "memory_id": str(source.source_id), "versions": [metadata], "expected_generation": 1,
            "current_revision": 1, "unavailable_revisions": 0, "next_before_revision": None,
        })
        tool = self.tools["list_memory_versions"]
        result = await tool.handler({"memory_id": str(source.source_id)})
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["provider_contacted"])
        self.assertNotIn("content", result["versions"][0])
        self.assertIsNotNone(tool.result_capture)
        self.assertEqual(set(tool.audit_arguments({"memory_id": str(source.source_id)})), {"arguments_digest"})
        denied = await self.catalog("foreign")["list_memory_versions"].handler({"memory_id": str(source.source_id)})
        self.assertEqual(denied["error"]["code"], "knowledge_owner_mismatch")
        self.sources.list_versions.assert_awaited_once_with(
            self.owner, self.area, source.source_id, limit=8, before_revision=None, expected_generation=None)
        self.provider.recall.assert_not_awaited()

    async def test_version_cursor_bounds_are_validated_without_database_access(self):
        for parameters in ({"limit": True}, {"limit": 0}, {"limit": 17},
                           {"before_revision": 2}, {"before_revision": True},
                           {"expected_generation": 0}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                await SourceRepository(None).list_versions(self.owner, self.area, uuid4(), **parameters)

    async def test_successful_retrieval_reports_measured_elapsed_time(self):
        self.add_source()
        with patch("app.knowledge.retrieval.time") as clock:
            clock.monotonic.side_effect = [10.0, 10.25]
            result = await self.search()
        self.assertEqual(result["retrieval"]["elapsed_ms"], 250)
