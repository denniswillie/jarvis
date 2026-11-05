"""The S09 oracle grades observations; it never manufactures answer evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from evals.second_brain.retrieval_acceptance import load_corpus, originals_searchable, structural_checks, validate_corpus


class RetrievalAcceptanceOracleTests(unittest.TestCase):
    def test_native_limits_match_production_token_budgets(self):
        from app.runtime.feature_config import TaskLimits
        from evals.second_brain.retrieval_acceptance import acceptance_limits
        actual, production = acceptance_limits(), TaskLimits()
        for name in ("request_output_tokens", "task_output_tokens", "task_input_tokens", "tool_steps"):
            self.assertEqual(getattr(actual, name), getattr(production, name))

    def setUp(self):
        self.corpus, self.digest = load_corpus()
        self.bindings = {row["key"]: SimpleNamespace(source_id=uuid4(), revision=1)
                         for row in self.corpus["sources"]}

    def test_frozen_cases_cover_links_old_session_uncertainty_and_stale_index(self):
        self.assertEqual(self.corpus["version"], 4)
        self.assertEqual(len(self.digest), 64)
        self.assertEqual(len(self.corpus["cases"]), 5)
        self.assertTrue(any(case.get("delete_before") for case in self.corpus["cases"]))
        self.assertTrue(any(case.get("require_omitted_context") for case in self.corpus["cases"]))
        self.assertTrue(any(row.get("origin") == "session" for row in self.corpus["sources"]))
        self.assertTrue(any(row.get("links") for row in self.corpus["sources"]))

    def test_invalid_oracle_links_and_conflicting_expectations_are_rejected(self):
        for change in ("duplicate_source", "missing_link", "missing_oracle", "conflicting_oracle", "invalid_wire_status"):
            corpus = deepcopy(self.corpus)
            if change == "duplicate_source":
                corpus["sources"].append(corpus["sources"][0])
            elif change == "missing_link":
                corpus["sources"][0]["links"] = ["unknown"]
            elif change == "missing_oracle":
                corpus["cases"][0]["required_candidates"] = ["unknown"]
            elif change == "conflicting_oracle":
                corpus["cases"][0]["forbidden_evidence"] = corpus["cases"][0]["required_evidence"]
            else:
                corpus["cases"][0]["expected_statuses"] = ["waiting"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_corpus(corpus)

    def test_fluent_answer_does_not_repair_missing_candidates_or_citations(self):
        case = self.corpus["cases"][0]
        checks = structural_checks(case, self.bindings, {"status": "available", "sources": []},
            {"answer": case["semantic_oracle"], "evidence": [], "tool_receipts": [],
             "completed_operation_refs": []}, "succeeded", {})
        self.assertFalse(checks["candidate_recall"])
        self.assertFalse(checks["required_original_citations"])
        self.assertFalse(checks["required_tools"])

    def test_empty_unknown_answer_is_not_a_false_positive_for_retrieval_failure(self):
        case = self.corpus["cases"][3]
        checks = structural_checks(case, self.bindings, {"status": "unavailable", "sources": []},
            {"evidence": [], "tool_receipts": [{"name": "search_memory"}],
             "uncertainties": ["Unknown"], "completed_operation_refs": []}, "waiting", {})
        self.assertFalse(checks["candidate_recall"])
        self.assertTrue(checks["uncertainty_disclosed"])

    def test_old_session_gate_requires_actual_omission_and_absent_original(self):
        case = self.corpus["cases"][1]
        source = self.bindings["old-session-clock"]
        recalled = {"status": "available", "sources": [{"source_id": str(source.source_id)}]}
        payload = {"evidence": [{"source_id": str(source.source_id), "revision": 1, "source_usage": "current"}],
                   "tool_receipts": [{"name": "search_memory"}], "completed_operation_refs": []}
        for omitted, included, expected in ((0, False, False), (1, True, False), (1, False, True)):
            checks = structural_checks(case, self.bindings, recalled, payload, "succeeded",
                {"omitted_count": omitted, "old_session_original_included": included})
            self.assertEqual(checks["bounded_old_session_handoff"], expected)

    def test_authorized_distractors_and_deleted_candidates_have_different_oracles(self):
        case = self.corpus["cases"][0]
        rows = [{"source_id": str(self.bindings[key].source_id)}
                for key in [*case["required_candidates"], *case["distractor_sources"]]]
        checks = structural_checks(case, self.bindings, {"status": "available", "sources": rows},
            {"evidence": [], "tool_receipts": [], "completed_operation_refs": []}, "succeeded", {})
        self.assertTrue(checks["candidate_recall"])
        self.assertTrue(checks["no_forbidden_candidates"])
        deleted = self.corpus["cases"][-1]
        source_id = str(self.bindings[deleted["delete_before"]].source_id)
        checks = structural_checks(deleted, self.bindings,
            {"status": "available", "sources": [{"source_id": source_id}]},
            {"evidence": [{"source_id": source_id, "revision": 1, "source_usage": "current"}],
             "tool_receipts": [], "completed_operation_refs": []}, "waiting_for_input", {})
        self.assertFalse(checks["no_forbidden_candidates"])
        self.assertFalse(checks["no_forbidden_citations"])

    def test_full_original_search_is_sufficient_but_metadata_alone_is_not(self):
        case = self.corpus["cases"][0]
        rows = [{"source_id": str(self.bindings[key].source_id), "revision": 1}
                for key in case["required_evidence"]]
        recalled = {"status": "available", "sources": rows}
        payload = {"evidence": [row | {"source_usage": "current"} for row in rows],
                   "tool_receipts": [{"name": "search_memory"}], "completed_operation_refs": []}
        for complete in (False, True):
            checks = structural_checks(case, self.bindings, recalled, payload, "succeeded", {},
                original_reads=[{"tool": "search_memory", "sources": [
                    row | {"content_complete": complete} for row in rows]}])
            self.assertEqual(checks["required_original_contents_seen"], complete)
            self.assertTrue(checks["required_tools"])


class RetrievalAcceptanceReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def test_readiness_awaits_every_canonical_original(self):
        originals = [SimpleNamespace(source_id=uuid4()), SimpleNamespace(source_id=uuid4())]
        sources = SimpleNamespace(read=AsyncMock(side_effect=[
            SimpleNamespace(ingestion_status="searchable"), SimpleNamespace(ingestion_status="pending")]))
        self.assertFalse(await originals_searchable(sources, "fixture-owner", uuid4(), originals))
        self.assertEqual(sources.read.await_count, 2)
