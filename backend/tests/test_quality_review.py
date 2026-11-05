"""Scoring controls do not substitute for semantic review of native outputs."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import wave

from evals.glasses_reliability.quality_cases import BY_NAME, definition_manifest
from evals.glasses_reliability.quality_probe import select_matrix
from evals.glasses_reliability.quality_review import final_audio_files, indexed, review_matrix, score_verified


class ReviewIdentityTests(unittest.TestCase):
    def test_missing_duplicate_and_foreign_trials_are_rejected(self):
        expected = {("partial_ret", 1, "openai"), ("partial_ret", 1, "gemini")}
        rows = [{"case": name, "repeat": repeat, "provider": provider} for name, repeat, provider in expected]
        self.assertEqual(set(indexed(rows, expected)), expected)
        for invalid in (rows[:1], rows + [rows[0]], [rows[0], {**rows[1], "provider": "other"}],
                        [rows[0], {**rows[1], "repeat": True}]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                indexed(invalid, expected)

    def test_audio_is_selected_by_final_response_not_filename_order(self):
        artifact = {"turns": [{"response_ids": ["first"]}, {"response_ids": ["final"]}],
                    "phone_timeline": [{"type": "audio.output.delta", "response_id": "first", "item_id": "a"},
                                       {"type": "audio.output.delta", "response_id": "final", "item_id": "b"}],
                    "audio_files": [{"item_id": "b", "file": "final.wav"}, {"item_id": "a", "file": "first.wav"}]}
        self.assertEqual(final_audio_files(artifact), [{"item_id": "b", "file": "final.wav"}])
        artifact["turns"][-1]["response_ids"] = []
        self.assertEqual(final_audio_files(artifact), [])


class ReviewScoringTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"case": case.name, "repeat": repeat, "provider": provider,
                      "judgment": "pass", "unsupported_confident_approval": False,
                      "audio_judgment": "pass" if case.family == "continuity" else "not_applicable",
                      "infrastructure_passed": True}
                     for case, repeat, provider in select_matrix(list(BY_NAME), 3)]

    def test_full_denominators_and_separate_provider_gates(self):
        for report in score_verified(self.rows).values():
            self.assertEqual(report["reviewed_runs"], 45)
            self.assertEqual(report["clear_code_runs"], 24)
            self.assertEqual(report["continuity_runs"], 9)
            self.assertEqual(report["uncertainty_runs"], 12)
            self.assertTrue(report["provider_quality_acceptance"])

    def test_one_confident_trap_approval_fails_despite_high_average(self):
        row = next(row for row in self.rows if row["case"] == "wrong_indentation" and row["provider"] == "openai")
        row.update(judgment="fail", unsupported_confident_approval=True)
        report = score_verified(self.rows)
        self.assertGreater(report["openai"]["useful_correct_clear_code_fraction"], .8)
        self.assertFalse(report["openai"]["clear_code_gate"])
        self.assertTrue(report["gemini"]["provider_quality_acceptance"])

    def test_unknown_audio_is_neither_a_pass_nor_a_claim_of_incorrect_speech(self):
        row = next(row for row in self.rows if row["case"] == "continuity_reset" and row["provider"] == "openai")
        row["audio_judgment"] = "unverified"
        report = score_verified(self.rows)["openai"]
        self.assertEqual(report["unverified_judgments"], 1)
        self.assertFalse(report["continuity_gate"])
        self.assertFalse(report["provider_quality_acceptance"])

    def test_missing_correct_guidance_uncertainty_or_infrastructure_cannot_be_hidden(self):
        code = [row for row in self.rows if row["provider"] == "gemini" and BY_NAME[row["case"]].family == "code"]
        for row in code[:5]:
            row["judgment"] = "fail"
        self.assertFalse(score_verified(self.rows)["gemini"]["clear_code_gate"])
        for field, value in (("judgment", "unverified"), ("infrastructure_passed", False)):
            rows = deepcopy(self.rows)
            row = next(row for row in rows if row["provider"] == "openai" and row["case"] == "vision_cropped")
            row[field] = value
            self.assertFalse(score_verified(rows)["openai"]["provider_quality_acceptance"])


class ReviewArtifactTests(unittest.TestCase):
    """File-integrity fixtures with invented judgments, never native quality evidence."""
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.review_path = self.root / "manual.json"
        self.audio_path = self.root / "asr.json"
        self.manifest = {**definition_manifest(), "selected_cases": list(BY_NAME), "repeats": 3,
                         "providers": ["openai", "gemini"], "source_revision": "fixture-only"}
        self.matrix = {"expected_runs": 90, "recorded_runs": 90, "cleanup": {"remaining_rows": 0}, "results": []}
        self.reviews = {"schema": 1, "reviews": []}
        self.audio = {"prompt_or_expected_transcript_supplied": False, "results": []}
        self.artifacts = []
        for index, (case, repeat, provider) in enumerate(select_matrix(list(BY_NAME), 3), 1):
            key = {"case": case.name, "repeat": repeat, "provider": provider}
            artifact = {**key, "infrastructure_passed": True, "checks": {"fixture_boundary": True},
                        "cleanup": {"remaining_rows": 0}}
            grade = {**key, "judgment": "pass", "unsupported_confident_approval": False,
                     "notes": "Invented parser fixture; does not assess model behavior.", "audio_judgment": "not_applicable"}
            if case.family == "continuity":
                filename = f"{case.name}-{repeat}-{provider}.wav"
                pcm = bytes((index, 0)) * 240
                digest = sha256(pcm).hexdigest()
                with wave.open(str(self.root / filename), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(24000)
                    output.writeframes(pcm)
                artifact.update(turns=[{"response_ids": ["final-response"]}],
                    phone_timeline=[{"type": "audio.output.delta", "response_id": "final-response", "item_id": "final-item"}],
                    audio_files=[{"file": filename, "item_id": "final-item", "pcm_sha256": digest}])
                grade.update(audio_judgment="pass", audio_pcm_sha256s=[digest])
                self.audio["results"].append({"pcm_sha256": digest, "status": "transcribed", "transcript": "Fixture transcript."})
            path = self.root / f"{case.name}-{repeat}-{provider}.json"
            self.write(path, artifact)
            digest = sha256(path.read_bytes()).hexdigest()
            self.matrix["results"].append({**key, "infrastructure_passed": True, "artifact_sha256": digest})
            self.reviews["reviews"].append({**grade, "artifact_sha256": digest})
            self.artifacts.append(path)
        self.write(self.root / "run-manifest.json", self.manifest)
        self.write(self.root / "matrix.json", self.matrix)
        self.write_audio_and_grades()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + "\n")

    def write_audio_and_grades(self):
        self.write(self.audio_path, self.audio)
        self.reviews["audio_review_sha256"] = sha256(self.audio_path.read_bytes()).hexdigest()
        self.write(self.review_path, self.reviews)

    def review(self):
        return review_matrix(self.root, self.review_path, self.audio_path)

    def test_complete_bound_fixture_can_be_scored(self):
        report = self.review()
        self.assertEqual(report["reviewed_runs"], 90)
        self.assertTrue(report["cleanup_zero"])
        self.assertTrue(all(row["provider_quality_acceptance"] for row in report["providers"].values()))

    def test_changed_native_artifact_is_rejected(self):
        self.artifacts[0].write_text(self.artifacts[0].read_text() + " ")
        with self.assertRaisesRegex(ValueError, "output changed"):
            self.review()

    def test_grade_bound_to_old_output_is_rejected(self):
        self.reviews["reviews"][0]["artifact_sha256"] = "0" * 64
        self.write_audio_and_grades()
        with self.assertRaisesRegex(ValueError, "manual grade"):
            self.review()

    def test_wrong_final_audio_evidence_is_rejected(self):
        grade = next(row for row in self.reviews["reviews"] if row["audio_judgment"] == "pass")
        grade["audio_pcm_sha256s"] = ["0" * 64]
        self.write_audio_and_grades()
        with self.assertRaisesRegex(ValueError, "final response"):
            self.review()

    def test_empty_transcription_is_not_independent_evidence(self):
        self.audio["results"][0]["transcript"] = ""
        self.write_audio_and_grades()
        with self.assertRaisesRegex(ValueError, "lacks independent"):
            self.review()

    def test_hint_prompted_audio_is_not_independent(self):
        self.audio["prompt_or_expected_transcript_supplied"] = True
        self.write_audio_and_grades()
        with self.assertRaisesRegex(ValueError, "prompted"):
            self.review()

    def test_missing_review_or_incomplete_matrix_is_rejected(self):
        removed = self.reviews["reviews"].pop()
        self.write_audio_and_grades()
        with self.assertRaisesRegex(ValueError, "Missing trials"):
            self.review()
        self.reviews["reviews"].append(removed)
        self.write_audio_and_grades()
        self.matrix["recorded_runs"] = 89
        self.write(self.root / "matrix.json", self.matrix)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.review()

    def test_changed_rubric_is_rejected(self):
        self.manifest["cases"][0]["expected"] = ["Always approve."]
        self.write(self.root / "run-manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "frozen quality rubric"):
            self.review()

    def test_unknown_cleanup_cannot_produce_provider_acceptance(self):
        self.matrix["cleanup"] = {}
        self.write(self.root / "matrix.json", self.matrix)
        report = self.review()
        self.assertFalse(report["cleanup_zero"])
        self.assertFalse(any(row["provider_quality_acceptance"] for row in report["providers"].values()))

    def test_boolean_false_is_not_a_zero_row_count(self):
        self.matrix["cleanup"] = {"remaining_rows": False}
        self.write(self.root / "matrix.json", self.matrix)
        self.assertFalse(self.review()["cleanup_zero"])

    def test_reported_success_cannot_override_a_failed_boundary_check(self):
        path = self.artifacts[0]
        artifact = json.loads(path.read_text())
        artifact["checks"]["fixture_boundary"] = False
        self.write(path, artifact)
        digest = sha256(path.read_bytes()).hexdigest()
        self.matrix["results"][0]["artifact_sha256"] = digest
        self.reviews["reviews"][0]["artifact_sha256"] = digest
        self.write(self.root / "matrix.json", self.matrix)
        self.write_audio_and_grades()
        report = self.review()["providers"][artifact["provider"]]
        self.assertFalse(report["infrastructure_passed"])
        self.assertFalse(report["provider_quality_acceptance"])
