"""Scene compatibility only; no provider, task authority or native acceptance."""
from copy import deepcopy
import unittest

from evals.responsive_voice import native_capture_evidence as reader
from evals.responsive_voice.test_native_provider_input_finalization import analyze, barrier_fixture


def task_fixture(cuts=3, duration=150, before_sdk_end=True):
    capture, pcm, manifest = barrier_fixture(cuts * 2 + 1, before_sdk_end)
    for ordinal, row in enumerate(capture["utterances"]):
        if ordinal == 0:
            continue
        if ordinal == 1:
            row["trigger"] = dict(kind="after_task_claim", after_input_ordinal=0, timeout_seconds=30)
        elif ordinal % 2 == 0:
            row["trigger"] = dict(kind="response_output_start", after_input_ordinal=ordinal - 1,
                delay_seconds=.3, timeout_seconds=10, max_pcm_age_seconds=.05, require_task_claim=True)
        else:
            row["provider_input_finalization"]["scene_deadline_uptime_seconds"] = (
                capture["clock_anchor"]["monotonic_reference_seconds"] + duration)
            continue
        row.pop("provider_input_finalization", None)
    manifest.update(duration_seconds=duration,
        task_barrier=dict(version=1, scope_file="task-scope.json", request_input_ordinal=0),
        utterances=[dict(clip_id=row["clip_id"], trigger=deepcopy(row["trigger"])) for row in capture["utterances"]])
    return capture, pcm, manifest


class ProviderInputScene150Tests(unittest.TestCase):
    def test_exact_three_cut_scene_accepts_finalization_before_and_after_sdk_completion(self):
        for before in (True, False):
            with self.subTest(before=before):
                capture, pcm, manifest = task_fixture(before_sdk_end=before)
                originals = deepcopy((capture, manifest))
                report = analyze(capture, pcm, manifest)
                self.assertEqual(report["provider_input_finalization_evidence"]["validated_transactions"], 3)
                self.assertEqual(report["task_control_evidence"]["verdict"], "unmeasured")
                self.assertIsNone(report["actual_cuts"])
                self.assertFalse(report["hearing_measured"])
                self.assertFalse(report["answer_quality_judged"])
                self.assertEqual((capture, manifest), originals)

    def test_one_two_and_default_three_cut_scene120_remain_accepted(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts):
                capture, pcm, manifest = task_fixture(cuts, 120)
                self.assertEqual(analyze(capture, pcm, manifest)["provider_input_finalization_evidence"]
                    ["validated_transactions"], cuts)

    def test_no_other_cut_count_or_duration_is_extended(self):
        for cuts, duration in ((1, 150), (2, 150), (3, 120.001), (3, 149), (3, 150.001), (3, 180)):
            with self.subTest(cuts=cuts, duration=duration):
                with self.assertRaisesRegex(reader.EvidenceError, "invalid_provider_scene_duration"):
                    analyze(*task_fixture(cuts, duration))

    def test_extension_requires_exact_task_scope_and_all_three_claim_gated_cuts(self):
        for fault in ("missing_barrier", "bool_version", "bool_request", "foreign_scope_file",
                      "no_claim", "false_claim", "wrong_predecessor", "other_continue"):
            with self.subTest(fault=fault):
                capture, pcm, manifest = task_fixture()
                if fault == "missing_barrier":
                    manifest.pop("task_barrier")
                elif fault == "bool_version":
                    manifest["task_barrier"]["version"] = True
                elif fault == "bool_request":
                    manifest["task_barrier"]["request_input_ordinal"] = False
                elif fault == "foreign_scope_file":
                    manifest["task_barrier"]["scope_file"] = "foreign.json"
                else:
                    for row in (capture["utterances"][4], manifest["utterances"][4]):
                        if fault == "no_claim": row["trigger"].pop("require_task_claim")
                        elif fault == "false_claim": row["trigger"]["require_task_claim"] = False
                        elif fault == "wrong_predecessor": row["trigger"]["after_input_ordinal"] = 1
                    if fault == "other_continue":
                        for row in (capture["utterances"][5], manifest["utterances"][5]):
                            row["trigger"] = dict(kind="after_input_complete", after_input_ordinal=4,
                                timeout_seconds=10, delay_seconds=1)
                        capture["utterances"][5].pop("provider_input_finalization")
                with self.assertRaisesRegex(reader.EvidenceError, "invalid_provider_scene_duration"):
                    analyze(capture, pcm, manifest)

    def test_extension_does_not_accept_old_scene_deadline_or_fabricate_missing_proof(self):
        capture, pcm, manifest = task_fixture()
        capture["utterances"][3]["provider_input_finalization"]["scene_deadline_uptime_seconds"] -= 30
        with self.assertRaisesRegex(reader.EvidenceError, "provider_input_deadline_anchor_mismatch"):
            analyze(capture, pcm, manifest)
        capture, pcm, manifest = task_fixture()
        capture["utterances"][3].pop("provider_input_finalization")
        report = analyze(capture, pcm, manifest)
        self.assertEqual(report["provider_input_finalization_evidence"]["validated_transactions"], 2)
        self.assertEqual(report["occurrences"][3]["provider_input_finalization_verdict"], "missing")
        self.assertIsNone(report["actual_cuts"])
