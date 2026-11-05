"""The native proposal experiment must fail incorrect scope and text, not grade fluency."""
from copy import deepcopy
import json
import unittest

from evals.second_brain.frontier_memory_editing import (
    ASR_REQUESTS, EditingProposal, REHEARSAL_INSTRUCTIONS, observed_originals, proposal_checks, requests,
)
from evals.second_brain.native_memory_lifecycle import load_cases


class FrontierMemoryEditingOracleTests(unittest.TestCase):
    def setUp(self):
        cases, _ = load_cases()
        self.cases = {case["id"]: case for case in cases}
        self.bindings = {key: {"source_id": key + "-synthetic-id", "revision": 1}
                         for key in ("cedar-v1", "cedar-v2")}
        self.original = self.cases["precise_correction"]["sources"][0]["text"]
        self.proposal = {"action": "correct", "edits": [{"source_id": "cedar-v1-synthetic-id",
            "revision": 1, "replacement": self.original.replace("firmware 2.3", "firmware 2.4")}],
            "question": None, "explanation": "Proposed only, nothing changed."}

    def grade(self, proposal, case="precise_correction", status="succeeded"):
        return proposal_checks(self.cases[case], self.bindings, json.dumps(proposal), status)[0]

    def test_population_keeps_both_original_and_previously_failing_asr(self):
        entries = requests(list(self.cases.values()))
        self.assertEqual(len(entries), 4)
        self.assertEqual(len({entry["id"] for entry in entries}), 4)
        for entry in entries:
            expected = (entry["case"]["utterance"] if entry["variant"] == "original"
                        else ASR_REQUESTS[entry["case"]["id"]]["text"])
            self.assertEqual(entry["utterance"], expected)
        self.assertIn("I have to see the prototypes.", entries[-1]["utterance"])

    def test_shared_instructions_do_not_reveal_expected_entities_or_targets(self):
        for forbidden in ("Cedar", "Seeder", "2.4", "firmware", "version one", "version two"):
            self.assertNotIn(forbidden, REHEARSAL_INSTRUCTIONS)

    def test_exact_proposal_passes_and_unrelated_spelling_change_fails(self):
        self.assertTrue(all(self.grade(self.proposal).values()))
        changed = deepcopy(self.proposal)
        changed["edits"][0]["replacement"] = changed["edits"][0]["replacement"].replace("Cedar", "Seeder")
        self.assertFalse(self.grade(changed)["exact_target_and_text"])

    def test_rehearsal_declares_the_actual_strict_oracle_schema(self):
        schema = json.loads(REHEARSAL_INSTRUCTIONS.split("Written-answer JSON schema: ", 1)[1])
        self.assertEqual(schema, EditingProposal.model_json_schema())
        edit = schema["$defs"]["ProposedEdit"]
        self.assertEqual(set(edit["required"]), {"source_id", "revision", "replacement"})
        self.assertIs(edit["additionalProperties"], False)
        for incorrect_key in ("text", "replacement_text"):
            with self.subTest(incorrect_key=incorrect_key):
                proposal = deepcopy(self.proposal)
                proposal["edits"][0][incorrect_key] = proposal["edits"][0].pop("replacement")
                self.assertFalse(self.grade(proposal)["proposal_schema"])

    def test_wrong_extra_duplicate_and_stale_targets_fail(self):
        for kind in ("wrong", "extra", "duplicate", "stale"):
            with self.subTest(kind=kind):
                proposal = deepcopy(self.proposal)
                if kind == "wrong":
                    proposal["edits"][0]["source_id"] = "cedar-v2-synthetic-id"
                elif kind == "extra":
                    proposal["edits"].append(proposal["edits"][0] | {"source_id": "cedar-v2-synthetic-id"})
                elif kind == "duplicate":
                    proposal["edits"].append(dict(proposal["edits"][0]))
                else:
                    proposal["edits"][0]["revision"] = 2
                self.assertFalse(self.grade(proposal)["exact_target_and_text"])

    def test_ambiguity_requires_question_no_edits_and_waiting_status(self):
        proposal = {"action": "clarify", "edits": [], "question": "Which version?",
                    "explanation": "The originals describe different prototypes; nothing changed."}
        self.assertTrue(all(self.grade(proposal, "ambiguous_correction", "waiting_for_input").values()))
        self.assertFalse(self.grade(proposal, "ambiguous_correction")["appropriate_task_status"])
        proposal["edits"] = self.proposal["edits"]
        self.assertFalse(self.grade(proposal, "ambiguous_correction", "waiting_for_input")["exact_target_and_text"])
        proposal["edits"], proposal["question"] = [], " "
        self.assertFalse(self.grade(proposal, "ambiguous_correction", "waiting_for_input")["exact_target_and_text"])

    def test_schema_failure_cannot_become_pass_from_plausible_prose(self):
        for answer in ("I corrected that.", "null", '{"action":"correct"}', "[]"):
            checks, parsed = proposal_checks(self.cases["precise_correction"], self.bindings, answer, "succeeded")
            self.assertFalse(any(checks.values()))
            self.assertIsNone(parsed)

    def test_truncated_or_failed_memory_reads_are_not_original_evidence(self):
        row = {"source_id": "original", "revision": 1, "content_included": True, "content": {"text": "Actual original"}}
        tool = {"name": "search_memory", "result": {"status": "success", "results": [row]}}
        self.assertEqual(observed_originals([tool]), {("original", 1)})
        for result in ({"status": "failed"}, {"status": "success", "results": [row | {"content_included": False}]}):
            self.assertEqual(observed_originals([tool | {"result": result}]), set())
