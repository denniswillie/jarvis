"""Offline contract tests. No synthesis, SQL, browser, provider or subprocess."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from uuid import UUID
import wave

from evals.responsive_voice import lean_note_replay as replay


def identity(number):
    return str(UUID(int=number))


class LeanNoteReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run"

    @staticmethod
    def speech(output, utterances):
        output.mkdir(mode=0o700)
        clips = []
        for name, text in utterances:
            path = output / (name + ".wav")
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(replay.base.RATE)
                stream.writeframes(b"\x01\x00" * 480)
            path.chmod(0o600)
            clips.append({"id": name, "text": text, "relative_path": path.name,
                "sha256": replay.base.digest(path.read_bytes()), "sample_count": 480,
                "first_active_sample": 0, "last_active_sample_exclusive": 480})
        replay.base.save(output / "speech.json", {"format": "jarvis.speech_assets.v1", "clips": clips})

    def bundle(self):
        return replay.prepare(self.root, model="offline-audio", voice="offline-voice",
                              worker_model="offline-worker", speech_builder=self.speech)

    def test_prepare_freezes_english_inputs_and_trace_boundary(self):
        manifest = self.bundle()
        loaded, clips = replay.validate_bundle(self.root)
        self.assertEqual(manifest, loaded)
        self.assertEqual([c["id"] for c in clips], ["note", "unrelated", "correct"])
        self.assertEqual(manifest["trace_scope"]["worker_request_response_wire"], "NOT CAPTURED")
        self.assertTrue(manifest["normal_worker_timing"])
        self.assertFalse(manifest["automatic_runner_retry"])
        self.assertEqual(self.root.stat().st_mode & 0o077, 0)

    def test_no_old_s01_contract_or_oracle_enters_speech(self):
        self.bundle()
        speech = json.loads((self.root / "speech/speech.json").read_text())
        self.assertEqual([c["text"] for c in speech["clips"]], [row[1] for row in replay.SCRIPT])
        self.assertNotIn("oracle", speech)

    def test_prepare_does_not_reuse_run_directory(self):
        self.bundle()
        with self.assertRaises(FileExistsError):
            self.bundle()

    def test_manifest_limit_change_rejected(self):
        manifest = self.bundle()
        manifest["limits"] = {**manifest["limits"], "system_seconds": 121}
        replay.base.save(self.root / "manifest.json", manifest)
        with self.assertRaisesRegex(replay.base.ReplayError, "frozen_note_contract_changed"):
            replay.validate_bundle(self.root)

    def test_audio_tampering_rejected(self):
        self.bundle()
        with (self.root / "speech/note.wav").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(replay.base.ReplayError, "speech_hash_mismatch"):
            replay.validate_bundle(self.root)

    def test_nonprivate_input_rejected(self):
        self.bundle()
        (self.root / "speech/note.wav").chmod(0o644)
        with self.assertRaisesRegex(replay.base.ReplayError, "private_file_required"):
            replay.validate_bundle(self.root)

    def test_model_configuration_is_bounded(self):
        with self.assertRaises(replay.base.ReplayError):
            replay.validate_models("model\nsecret", "voice", "worker")

    def test_attempt_is_consumed_without_retry(self):
        self.bundle()
        replay.base.claim(self.root)
        with self.assertRaises(FileExistsError):
            replay.base.claim(self.root)

    def test_execute_requires_explicit_opt_in_before_any_setup(self):
        with self.assertRaisesRegex(replay.base.ReplayError, "execute_once_required"):
            asyncio.run(replay.execute(self.root, execute_once=False, hindsight_url="unused",
                api_key_file="unused", openai_key_file="unused", hermes_python="unused"))

    @staticmethod
    def media():
        return {"audio_frame": 20000, "turns": [{"last_output_sample": 5000}],
            "responses": {"native-result": {"status": "completed", "created_frame": 3000,
                "output_observed": True, "has_function_calls": False, "function_call_ids": [],
                "buffer_started": True, "buffer_stopped": True}},
            "notifications": {"native-result": {"response_id": "native-result", "created_frame": 3000,
                "metadata": {"jarvis_offer_id": identity(1), "jarvis_attempt_id": identity(2),
                             "jarvis_request_id": "real-request"}}}}

    def test_real_metadata_terminal_audio_selects_notification(self):
        self.assertEqual(replay.notification_finished(self.media(), after_frame=2000)["response_id"], "native-result")

    def test_ack_without_notification_metadata_is_not_late_result(self):
        state = self.media()
        state["notifications"] = {}
        self.assertIsNone(replay.notification_finished(state, after_frame=2000))

    def test_ack_with_pending_tool_is_not_late_spoken_result(self):
        state = self.media()
        state["responses"]["native-result"].update(has_function_calls=True, function_call_ids=["provider-call"])
        self.assertIsNone(replay.notification_finished(state, after_frame=2000))

    def test_stale_or_duplicate_notification_not_accepted(self):
        self.assertIsNone(replay.notification_finished(self.media(), after_frame=3000))
        self.assertIsNone(replay.notification_finished(self.media(), after_frame=2000, excluded={"native-result"}))

    def test_response_done_without_decoded_signal_is_insufficient(self):
        state = self.media()
        state["turns"][0]["last_output_sample"] = 2500
        self.assertIsNone(replay.notification_finished(state, after_frame=2000))

    def test_stopped_without_started_is_insufficient(self):
        state = self.media()
        state["responses"]["native-result"]["buffer_started"] = False
        self.assertIsNone(replay.notification_finished(state, after_frame=2000))

    def test_observer_failure_cannot_be_hidden(self):
        state = self.media()
        state["observer_error"] = "observer_bound"
        with self.assertRaises(replay.base.ReplayError):
            replay.notification_finished(state, after_frame=0)

    def test_optional_correction_cannot_extend_deadline(self):
        self.assertTrue(replay.correction_fits(100, now=55))
        self.assertFalse(replay.correction_fits(100, now=55.01))

    @staticmethod
    def worker_receipt():
        return {"status": "closed", "worker_joined": True, "readiness_inactive": True,
            "pipes_complete": True, "secrets_removed": True, "forced_kill": False,
            "errors": [], "returncode": 0}

    def test_returned_worker_cleanup_failure_is_failure(self):
        receipt = self.worker_receipt()
        self.assertTrue(replay.worker_cleanup_confirmed(receipt))
        for field, value in (("forced_kill", True), ("readiness_inactive", False),
                             ("status", "failed"), ("returncode", 1), ("errors", ["join_unconfirmed"])):
            self.assertFalse(replay.worker_cleanup_confirmed({**receipt, field: value}))

    @staticmethod
    def canonical_case():
        report = {"session_id": identity(10), "late_results": [{"response_id": "native-result",
            "metadata": {"jarvis_attempt_id": identity(2)}}],
            "turns": [{}, {"input_utc": "2026-11-02T04:00:10+00:00"}],
            "note_reads": [{"artifact": {"artifact_id": identity(20), "version_id": "saved-v1"}}]}
        tables = {
            "jarvis_attention_attempts": [{"id": identity(2), "intent_id": identity(3)}],
            "jarvis_attention_intents": [{"id": identity(3), "source_namespace": "task_result", "task_id": identity(4)}],
            "jarvis_background_tasks": [{"id": identity(4), "session_id": identity(10), "status": "failed"}],
            "jarvis_task_actions": [{"id": identity(5), "task_id": identity(4), "status": "accepted"}],
            "jarvis_note_operations": [{"operation_id": identity(5), "artifact_id": identity(20),
                                        "result_version_id": "saved-v1", "outcome": "completed"}],
            "jarvis_task_attempts": [{"task_id": identity(4), "started_at": "2026-11-02T04:00:05+00:00",
                                      "ended_at": "2026-11-02T04:00:20+00:00"}]}
        return {"tables": tables}, report

    def test_receipt_wins_without_relabeling_worker_failure(self):
        canonical, report = self.canonical_case()
        evidence = replay.evidence_summary(canonical, report)
        self.assertTrue(evidence["publication_linked"])
        self.assertTrue(evidence["worker_overlap"])
        self.assertEqual(evidence["task_state"], "failed")
        self.assertEqual(evidence["semantic_acceptance"], "not_evaluated")

    def test_accepted_action_without_domain_operation_is_not_save(self):
        canonical, report = self.canonical_case()
        canonical["tables"]["jarvis_note_operations"] = []
        self.assertFalse(replay.evidence_summary(canonical, report)["publication_linked"])

    def test_other_task_or_version_cannot_prove_publication(self):
        canonical, report = self.canonical_case()
        canonical["tables"]["jarvis_task_actions"][0]["task_id"] = identity(99)
        self.assertFalse(replay.evidence_summary(canonical, report)["publication_linked"])
        canonical, report = self.canonical_case()
        report["note_reads"][0]["artifact"]["version_id"] = "newer-version"
        self.assertFalse(replay.evidence_summary(canonical, report)["publication_linked"])

    def test_early_worker_completion_is_not_concurrency_evidence(self):
        canonical, report = self.canonical_case()
        canonical["tables"]["jarvis_task_attempts"][0]["ended_at"] = "2026-11-02T04:00:09+00:00"
        self.assertFalse(replay.evidence_summary(canonical, report)["worker_overlap"])

    def test_http_snapshot_reads_note_and_history_not_task_status(self):
        self.bundle()
        calls = []
        artifact = {"artifact_id": identity(20), "version_id": "v2"}

        class HTTP:
            async def get(self, path, params=None):
                calls.append((path, params))
                body = {"note": {"artifact": artifact, "content": {}}} if path == "/workspace/note" else {"items": []}
                return SimpleNamespace(status_code=200, content=b"{}", json=lambda: body)

        report = {"session_id": identity(10), "note_reads": []}
        asyncio.run(replay.note_snapshot(HTTP(), self.root, report, "offline",
                    original={"artifact_id": identity(20), "version_id": "v1"}))
        self.assertEqual([path for path, _ in calls], ["/workspace/note",
            "/workspace/artifacts/" + identity(20) + "/versions", "/workspace/artifacts/" + identity(20)])
        self.assertEqual(calls[-1][1]["version_id"], "v1")
        self.assertEqual((self.root / "offline-http.json").stat().st_mode & 0o077, 0)

    def test_collection_requires_worker_cleanup_and_proven_overlap(self):
        report = {"sequence_completed": True, "decoded_signal": 1, "canonical_status": "captured",
            "cleanup": {"status": "passed"}, "host_shutdown": {"status": "passed"},
            "wiretrace": {"incomplete": False}, "worker_cleanup": self.worker_receipt(),
            "browser_cleanup": {"media": {"peer": "closed", "context": "closed"},
                "loopback_server_stopped": True, "process_group_gone": True, "chrome_returncode": -15},
            "evidence": {"publication_linked": True, "worker_overlap": True}}
        self.assertTrue(replay.collection_complete(report))
        broken = deepcopy(report)
        broken["worker_cleanup"]["status"] = "failed"
        self.assertFalse(replay.collection_complete(broken))
        report["evidence"]["worker_overlap"] = False
        self.assertFalse(replay.collection_complete(report))


if __name__ == "__main__":
    unittest.main()
