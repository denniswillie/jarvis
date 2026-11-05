from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from evals.glasses_reliability.memory_pipeline_probe import assess_state_sequence, assess_independent_tasks, GROUNDING_CASES
from evals.glasses_reliability.memory_pipeline_runtime import FixtureLedger, PipelineRuntime


def state_runs():
    old = {"id": "old", "memory_type": "semantic_fact", "metadata": {
        "subject_entity_id": "pouch", "predicate": "is", "value": "missing", "temporal_scope": "state",
    }}
    new = {"id": "new", "memory_type": "semantic_fact", "metadata": {**old["metadata"], "value": "found"}}
    runs = []
    for name in ("missing", "uncertain_match", "found", "late_missing", "repeat_found"):
        changed = name in {"found", "late_missing", "repeat_found"}
        runs.append({
            "case": name, "consolidation": {"semantic_fact_candidates": [
                {"change_intent": "transition", "previous_value": "missing"},
            ] if name == "found" else []},
            "snapshot": {"memories": [old, new] if changed else [old],
                         "current": {"results": [{"id": "new" if changed else "old"}]},
                         "history": {"results": [{"id": "old", "temporal_status": "historical" if changed else "current",
                                                  "valid_to": "2026-09-01T00:00:00Z" if changed else None}]}},
        })
    return runs


class MemoryPipelineOracleTests(unittest.TestCase):
    def test_shared_owner_fixture_explicitly_distinguishes_the_checklists(self):
        case = next(item for item in GROUNDING_CASES if item.name == "interrupted_approval")
        self.assertIn("separate packing checklist", case.user_text)
        self.assertIn("not the travel checklist", case.user_text)
        self.assertTrue(case.interrupted)

    def test_independent_task_oracle_rejects_retirement_and_conflicting_new_fact(self):
        old = {
            "id": "checked", "source_experience_event_id": "first", "memory_type": "semantic_fact",
            "status": "active", "valid_from": "before", "valid_to": None, "content": "fully checked",
            "metadata": {"subject_entity_id": "travel", "predicate": "is", "value": "fully checked"},
        }
        runs = [
            {"case": "supported_completion", "event_id": "first", "snapshot": {"memories": [deepcopy(old)]}},
            {"case": "interrupted_approval", "event_id": "second", "snapshot": {"memories": [deepcopy(old)]}},
        ]
        self.assertTrue(all(assess_independent_tasks(runs).values()))
        runs[1]["snapshot"]["memories"][0]["status"] = "superseded"
        self.assertFalse(assess_independent_tasks(runs)["earlier_completion_retained"])
        runs[1]["snapshot"]["memories"] = [deepcopy(old), {
            **old, "id": "wrong", "source_experience_event_id": "second",
            "metadata": {**old["metadata"], "value": "medication unchecked"},
        }]
        self.assertFalse(assess_independent_tasks(runs)["separate_task_does_not_change_prior_entity_state"])
        self.assertFalse(all(assess_independent_tasks([]).values()))

    def test_complete_transition_history_and_reinforcement(self):
        self.assertTrue(all(assess_state_sequence(state_runs()).values()))

    def test_empty_or_partial_run_cannot_pass(self):
        self.assertFalse(all(assess_state_sequence([]).values()))
        self.assertFalse(all(assess_state_sequence(state_runs()[:-1]).values()))

    def test_uncertain_late_and_duplicate_regressions_are_detected(self):
        for index, key in ((1, "possible_match_does_not_retire_missing"),
                           (3, "late_evidence_does_not_resurrect_missing"),
                           (4, "repeat_confirmation_does_not_duplicate_state")):
            with self.subTest(case=index):
                runs = deepcopy(state_runs())
                runs[index]["snapshot"]["current"]["results"] = []
                self.assertFalse(assess_state_sequence(runs)[key])

    def test_historical_label_and_validity_are_both_required(self):
        runs = state_runs()
        runs[2]["snapshot"]["history"]["results"][0]["valid_to"] = None
        self.assertFalse(assess_state_sequence(runs)["missing_retained_as_history"])

    def test_ambiguous_initial_state_requires_review_not_a_guessed_match(self):
        runs = deepcopy(state_runs())
        runs[0]["snapshot"]["memories"].append({**runs[0]["snapshot"]["memories"][0], "id": "duplicate"})
        self.assertFalse(all(assess_state_sequence(runs).values()))


class MemoryPipelineFixtureSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_reusing_manifest_refuses_to_overwrite_previous_owners(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            original = FixtureLedger(path)
            with self.assertRaises(FileExistsError):
                FixtureLedger(path)
            original.data["status"] = "cleaned"
            original.persist()
            self.assertEqual(original.data["owners"], sorted(original.owners))

    async def test_fresh_owners_are_manifested_and_foreign_claim_fails_before_database_access(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            ledger = FixtureLedger(path)
            self.assertTrue(path.exists())
            self.assertEqual(len(ledger.owners), 2)
            self.assertTrue(all(owner.startswith("direct-image-eval-memory-pipeline-") for owner in ledger.owners))
            runtime = object.__new__(PipelineRuntime)
            runtime.ledger = ledger
            runtime.pool = SimpleNamespace(acquire=AsyncMock())
            with self.assertRaisesRegex(ValueError, "fresh fixture manifest"):
                await runtime.claim_owned_job("personal-owner")
            runtime.pool.acquire.assert_not_called()

    async def test_foreign_event_seed_fails_before_session_creation(self):
        with TemporaryDirectory() as directory:
            runtime = object.__new__(PipelineRuntime)
            runtime.ledger = FixtureLedger(Path(directory) / "fixture.json")
            runtime.sessions = SimpleNamespace(create_session=AsyncMock())
            with self.assertRaises(ValueError):
                await runtime.seed_event(owner="personal-owner", case=None, ended_at=None)
            runtime.sessions.create_session.assert_not_called()
