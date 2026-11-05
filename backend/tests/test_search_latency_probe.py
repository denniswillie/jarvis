"""Negative controls for owner-safe, correlated native search evaluation."""
from copy import deepcopy
import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from PIL import Image

from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.references import PerceptionReferenceError

from evals.glasses_reliability.search_latency_probe import (
    aggregate, assess, load_sequence, observation_timings, runtime_pending, schedule,
)
from evals.glasses_reliability.search_runtime import NativeTurnJournal, ObservedAnalyzer


class SearchAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.owner, self.session, self.focus = "direct-image-eval-fixture", uuid4(), uuid4()
        self.captured = [{"id": "frame", "at": 10, "eligible_target": True}]
        self.timeline = [{"type": "audio.output.delta", "at": 13, "response_id": "answer"}]
        self.snapshot = {
            "deliveries": [{"id": "delivery", "visual_focus_id": self.focus, "event_type": "delivered_to_session",
                            "provider_response_id": "answer", "owner_id": self.owner, "session_id": self.session,
                            "evidence": {"supporting_frame_ids": ["frame"]}}],
            "focus": {"id": self.focus, "mode": "find_once", "completed_at": "complete", "cleared_at": None},
            "frames": [{"id": "frame", "owner_id": self.owner, "session_id": self.session}],
            "observations": [{"id": "observation"}], "working_memory": [{"revision": 1}], "executions": [],
        }

    def grade(self):
        return assess(self.snapshot, owner=self.owner, session_id=self.session, focus_id=self.focus,
                      captured=self.captured, timeline=self.timeline)

    def test_only_correlated_forwarded_audio_supplies_latency(self):
        result = self.grade()
        self.assertTrue(result["infrastructure_passed"])
        self.assertEqual(result["capture_to_first_forwarded_seconds"], 3)
        self.timeline[0]["response_id"] = "unrelated"
        self.assertIsNone(self.grade()["capture_to_first_forwarded_seconds"])
        self.assertFalse(self.grade()["infrastructure_passed"])

    def test_missing_detection_is_not_zero_latency(self):
        self.snapshot["deliveries"] = []
        self.assertIsNone(self.grade()["capture_to_first_forwarded_seconds"])
        self.assertFalse(self.grade()["infrastructure_passed"])

    def test_cancelled_search_is_not_completed_search(self):
        self.snapshot["focus"].update(completed_at=None, cleared_at="cleared")
        self.assertFalse(self.grade()["checks"]["completed_not_cancelled"])

    def test_duplicates_and_foreign_owners_fail(self):
        self.snapshot["deliveries"].append(deepcopy(self.snapshot["deliveries"][0]))
        self.assertFalse(self.grade()["checks"]["one_tracked_delivery"])
        self.snapshot["frames"][0]["owner_id"] = "personal-owner"
        self.assertFalse(self.grade()["checks"]["frames_and_deliveries_scoped"])

    def test_false_positive_before_target_is_not_a_negative_fast_latency(self):
        self.timeline[0]["at"] = 9
        self.assertIsNone(self.grade()["capture_to_first_forwarded_seconds"])
        self.assertFalse(self.grade()["checks"]["no_audio_before_target"])

    def test_late_second_audio_response_cannot_hide_behind_one_delivery(self):
        self.timeline.append({"type": "audio.output.delta", "at": 30, "response_id": "duplicate"})
        self.assertFalse(self.grade()["checks"]["one_intended_audio_response"])

    def test_lost_frame_persistence_or_new_domain_mutation_fails(self):
        self.snapshot["frames"] = []
        self.snapshot["executions"] = [{"mutates_state": True}]
        self.assertFalse(self.grade()["checks"]["all_captured_frames_persisted"])
        self.assertFalse(self.grade()["checks"]["no_audio_tool_mutations"])

    def test_negative_frame_cannot_support_a_delayed_false_positive(self):
        self.captured.append({"id": "negative", "at": 9, "eligible_target": False})
        self.snapshot["deliveries"][0]["evidence"] = {"supporting_frame_ids": ["negative"]}
        self.assertFalse(self.grade()["checks"]["delivery_has_target_evidence"])
        self.snapshot["deliveries"][0]["evidence"] = json.dumps({"supporting_frame_ids": ["frame", "negative"]})
        self.assertTrue(self.grade()["checks"]["delivery_has_target_evidence"])

    def test_unknown_malformed_or_missing_evidence_fails(self):
        for evidence in (None, "not json", [], {"supporting_frame_ids": ["foreign"]},
                         {"supporting_frame_ids": ["frame", "foreign"]}, {"supporting_frame_ids": [42]}):
            with self.subTest(evidence=evidence):
                self.snapshot["deliveries"][0]["evidence"] = evidence
                self.assertFalse(self.grade()["checks"]["delivery_has_target_evidence"])

    def test_duplicate_frame_rows_and_unpopulated_working_memory_fail(self):
        self.snapshot["frames"] *= 2
        self.snapshot["working_memory"][0]["revision"] = 0
        self.assertFalse(self.grade()["checks"]["all_captured_frames_persisted"])
        self.assertFalse(self.grade()["checks"]["real_observations_and_working_state"])


class SearchAggregateTests(unittest.TestCase):
    def test_duplicate_unknown_or_out_of_range_trials_are_rejected(self):
        row = {"provider": "openai", "repeat": 1, "infrastructure_passed": True,
               "capture_to_first_forwarded_seconds": 2}
        for rows in ([row, row], [{**row, "provider": "other"}], [{**row, "repeat": 0}],
                     [{**row, "repeat": True}], [{**row, "repeat": 11}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                aggregate(rows, expected_pairs=10)

    def test_invalid_latency_and_failed_infrastructure_cannot_pass(self):
        for value in (-1, float("nan"), float("inf"), True, None, "fast"):
            row = {"provider": "openai", "repeat": 1, "infrastructure_passed": True,
                   "capture_to_first_forwarded_seconds": value}
            with self.subTest(value=value):
                report = aggregate([row], expected_pairs=1)["openai"]
                self.assertFalse(report["five_second_median_gate"])
                self.assertEqual(report["missing_measurements"], 1)
        row.update(capture_to_first_forwarded_seconds=1, infrastructure_passed=False)
        self.assertFalse(aggregate([row], expected_pairs=1)["openai"]["five_second_median_gate"])

    def test_partial_or_missing_measurements_cannot_pass(self):
        report = aggregate([{"provider": "openai", "repeat": 1, "infrastructure_passed": True,
                             "capture_to_first_forwarded_seconds": 1}], expected_pairs=10)
        self.assertFalse(report["openai"]["five_second_median_gate"])
        self.assertFalse(report["gemini"]["infrastructure_passed"])
        self.assertIsNone(report["gemini"]["median_seconds"])

    def test_p95_and_outliers_are_retained_even_when_median_is_fast(self):
        rows = [{"provider": "openai", "repeat": i + 1, "infrastructure_passed": True,
                 "capture_to_first_forwarded_seconds": value} for i, value in enumerate([2] * 9 + [12])]
        report = aggregate(rows, expected_pairs=10)["openai"]
        self.assertEqual(report["median_seconds"], 2)
        self.assertEqual(report["p95_nearest_rank_seconds"], 12)
        self.assertEqual(report["over_ten_seconds"], [{"repeat": 10, "seconds": 12}])

    def test_single_pair_cannot_claim_ten_pair_acceptance(self):
        rows = [{"provider": "openai", "repeat": 1, "infrastructure_passed": True,
                 "capture_to_first_forwarded_seconds": 2}]
        self.assertFalse(aggregate(rows, expected_pairs=1)["openai"]["ten_pair_acceptance_eligible"])

    def test_schedule_is_declared_before_model_output(self):
        self.assertEqual(schedule()[:4], [(float(i), "negative") for i in range(4)])
        self.assertEqual(schedule()[4], (4.0, "positive"))
        self.assertEqual(len(schedule()), 13)


class SearchObservationTests(unittest.TestCase):
    def test_invalid_scope_is_recorded_without_calling_the_model(self):
        now = datetime.now(UTC)
        frame = FastPerceptionFrame(uuid4(), 1, now, "glasses", "image/jpeg", b"fixture")
        request = FastPerceptionRequest(
            owner_id="fixture-owner", session_id=uuid4(), window_id=uuid4(), frames=(frame,),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="", active_focus={"id": str(uuid4()), "owner_id": "foreign"},
            active_reminders=(), recent_interrupts=(), requested_at=now,
        )
        delegate = SimpleNamespace(provider="fixture", model="fixture", analyze=AsyncMock())
        observer = ObservedAnalyzer(delegate)
        with self.assertRaises(PerceptionReferenceError):
            asyncio.run(observer.analyze(request))
        delegate.analyze.assert_not_awaited()
        self.assertEqual(observer.calls[0]["error_type"], "PerceptionReferenceError")
        self.assertIn("finished_at", observer.calls[0])

    def test_gpt_session_without_gemini_diagnostics_uses_native_journal(self):
        journal = NativeTurnJournal()
        runtime = SimpleNamespace(native_turns=journal, service=SimpleNamespace(_pending_deliveries={}), session=object())
        self.assertFalse(runtime_pending(runtime))
        journal.request()
        self.assertTrue(runtime_pending(runtime))
        journal.terminal("answer")
        self.assertFalse(runtime_pending(runtime))

    def test_generic_and_queued_native_alerts_must_settle(self):
        journal = NativeTurnJournal()
        proactive = {"pending": 0, "owned_tasks": 0}
        runtime = SimpleNamespace(native_turns=journal, service=SimpleNamespace(_pending_deliveries={}),
                                  session=SimpleNamespace(diagnostics=lambda: {"proactive": proactive}))
        self.assertFalse(runtime_pending(runtime))
        journal.request()
        self.assertTrue(runtime_pending(runtime))
        journal.terminal("first")
        journal.terminal("first")
        self.assertFalse(runtime_pending(runtime))
        self.assertTrue(journal.diagnostics()["balanced"])
        proactive["owned_tasks"] = 1
        self.assertTrue(runtime_pending(runtime))
        proactive.update(owned_tasks=0, pending=1)
        self.assertTrue(runtime_pending(runtime))

    def test_unmatched_terminal_does_not_count_as_balanced(self):
        journal = NativeTurnJournal()
        journal.terminal()
        self.assertFalse(journal.diagnostics()["balanced"])

    def test_valid_and_persisted_boundaries_use_their_own_primary_capture(self):
        rows = observation_timings(
            [{"id": "old", "at": 2, "eligible_target": False}, {"id": "new", "at": 5, "eligible_target": True}],
            [{"primary_frame_id": "new", "valid_output_at": 7}, {"primary_frame_id": "old", "error_type": "failure"}],
            [{"primary_frame_id": "new", "persisted_at": 8}],
        )
        self.assertEqual([row["capture_to_boundary_seconds"] for row in rows], [2, 3])
        self.assertEqual([row["boundary"] for row in rows], ["native_valid_output", "persisted_observation"])
        self.assertTrue(all(row["eligible_target"] for row in rows))


class SearchManifestTests(unittest.TestCase):
    def test_metadata_and_image_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            for role, color in (("negative", "white"), ("positive", "black")):
                data = BytesIO()
                Image.new("RGB", (32, 32), color).save(data, format="JPEG")
                (root / (role + ".jpg")).write_bytes(data.getvalue())
                entries[role] = {"path": role + ".jpg", "sha256": sha256(data.getvalue()).hexdigest()}
            manifest = {"schema": 1, "stratum": "synthetic_photograph_controlled_replay",
                        "target": {"name": "fixture", "description": "fixture", "purpose": "fixture"},
                        "images": entries, "ground_truth": "Parser fixture only, not a recognition case.",
                        "original_audio_model_input": "unproven"}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(set(load_sequence(path)[1]), {"negative", "positive"})
            manifest["original_audio_model_input"] = "exact"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_sequence(path)
            manifest["original_audio_model_input"] = "unproven"
            entries["positive"]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_sequence(path)
