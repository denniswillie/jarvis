"""Offline frozen provider-input barrier, not a semantic/native acceptance test."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.responsive_voice import native_capture_evidence as reader
from evals.responsive_voice import native_interruption_scenarios as compiler
from evals.responsive_voice import test_native_interruption_scenarios as scenario_fixtures
from evals.responsive_voice.test_native_capture_evidence import fixture


CONNECTION = "33333333-3333-4333-8333-333333333333"


def event(kind, item, time, serial):
    return dict(stage="provider_data_channel_handled", fields=dict(event_type=kind, item_id=item,
        provider_event_id=f"event_{serial}", connection_id=CONNECTION,
        boundary="ordered_current_provider_channel"), received_monotonic_s=time,
        handled_monotonic_s=time + 0.00001, callback_to_handler_seconds=0.00001,
        bound_response_identity=None)


def barrier_fixture(count=1, before_sdk_end=True):
    capture, pcm, _ = fixture(count + 1)
    controls = [event("session.created", None, 99.05, "session")]
    capture.update(control_events=controls, control_events_truncated=False, control_event_timing_invalid=False)
    for ordinal in range(1, count + 1):
        previous, current = capture["utterances"][ordinal - 1:ordinal + 1]
        at = previous["first_sdk_callback_start_uptime_seconds"]
        clean = len(controls)
        indices = []
        for step, kind in enumerate(reader._PROVIDER_INPUT_STEPS):
            indices.append(len(controls))
            stamp = at + ((step + 1) * 0.00015 if before_sdk_end else 0.01 + step * 0.01)
            controls.append(event(kind, f"item_{ordinal}", stamp, f"{ordinal}_{step}"))
        current["trigger"] = dict(kind="after_provider_input_finalized", after_input_ordinal=ordinal - 1,
                                  timeout_seconds=10)
        current["provider_input_finalization"] = dict(schema_version=1, after_input_ordinal=ordinal - 1,
            correlation_basis="replay_exclusive_provider_transaction", canonical_input_binding="unknown",
            status="validated_provider_transaction", failure_code=None, connection_id=CONNECTION,
            provider_item_id=f"item_{ordinal}", transaction_reserved_uptime_seconds=at - 0.1,
            clean_control_event_count=clean, validation_control_event_count=len(controls),
            session_created_control_event_index=0, **dict(zip(reader._PROVIDER_INPUT_INDEX_FIELDS, indices)),
            input_complete_uptime_seconds=previous["last_sdk_callback_end_uptime_seconds"],
            scene_deadline_uptime_seconds=capture["clock_anchor"]["monotonic_reference_seconds"] + 120,
            deadline_uptime_seconds=previous["last_sdk_callback_end_uptime_seconds"] + 10,
            enqueue_revalidated_uptime_seconds=current["enqueue_requested_uptime_seconds"])
    manifest = dict(version=1, run_id=capture["run_id"], voice_route="realtime", duration_seconds=120,
        utterances=[dict(clip_id=row["clip_id"], trigger=deepcopy(row["trigger"])) for row in capture["utterances"]])
    return capture, pcm, manifest


def analyze(capture, pcm, manifest):
    return reader.analyze_capture(capture, pcm, replay_manifest_bytes=json.dumps(manifest).encode())


class ProviderInputEvidenceTests(unittest.TestCase):
    def test_one_two_three_before_and_after_sdk_end_have_only_transaction_proof(self):
        for count in (1, 2, 3):
            for before in (True, False):
                with self.subTest(count=count, before=before):
                    capture, pcm, manifest = barrier_fixture(count, before)
                    report = analyze(capture, pcm, manifest)
                    self.assertEqual(report["provider_input_finalization_evidence"]["validated_transactions"], count)
                    for row in report["occurrences"][1:]:
                        self.assertEqual(row["provider_input_finalization_verdict"], "validated_provider_transaction")
                        self.assertEqual(row["canonical_input_binding"], "unknown")
                    self.assertIsNone(report["actual_cuts"])
                    self.assertEqual(report["provider_response_attribution"], "unavailable")
                    self.assertFalse(report["hearing_measured"])

    def test_finalization_before_sdk_end_never_allows_early_enqueue(self):
        capture, pcm, manifest = barrier_fixture()
        target = capture["utterances"][1]
        target["provider_input_finalization"]["enqueue_revalidated_uptime_seconds"] = 100.0007
        target["enqueue_requested_uptime_seconds"] = 100.0008
        with self.assertRaisesRegex(reader.EvidenceError, "sdk_completion_or_reservation_order"):
            analyze(capture, pcm, manifest)

    def test_old_item_and_events_before_reservation_or_sdk_start_fail_closed(self):
        for change in ("old_item", "before_reserve", "before_sdk"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture(2)
                proof = capture["utterances"][2]["provider_input_finalization"]
                row = capture["control_events"][proof["speech_started_control_event_index"]]
                if change == "old_item":
                    row["fields"]["item_id"] = "item_1"
                else:
                    row["received_monotonic_s"] = 101.8 if change == "before_reserve" else 101.99
                    row["handled_monotonic_s"] = row["received_monotonic_s"] + 0.00001
                with self.assertRaises(reader.EvidenceError):
                    analyze(capture, pcm, manifest)

    def test_split_foreign_connection_reset_and_transcription_failure_reject(self):
        for change in ("split", "foreign", "reset", "transcription_failed", "response_done"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                row = capture["control_events"][3]
                if change == "split":
                    row["fields"]["item_id"] = "item_competing"
                elif change == "foreign":
                    row["fields"]["connection_id"] = "44444444-4444-4444-8444-444444444444"
                elif change == "reset":
                    row["stage"] = "connection_reset"
                elif change == "response_done":
                    capture["control_events"][4]["fields"]["event_type"] = "response.done"
                else:
                    capture["control_events"][4]["fields"]["event_type"] = "conversation.item.input_audio_transcription.failed"
                with self.assertRaises(reader.EvidenceError):
                    analyze(capture, pcm, manifest)

    def test_pending_old_prefix_and_replaced_session_cannot_be_clean(self):
        for change in ("pending", "session"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                capture["control_events"].insert(1, event("input_audio_buffer.speech_started", "old", 99.2, "old"))
                proof = capture["utterances"][1]["provider_input_finalization"]
                proof["clean_control_event_count"] += 1
                proof["validation_control_event_count"] += 1
                for field in reader._PROVIDER_INPUT_INDEX_FIELDS:
                    proof[field] += 1
                if change == "session":
                    capture["control_events"][1] = event("session.created", None, 99.2, "session2")
                with self.assertRaises(reader.EvidenceError):
                    analyze(capture, pcm, manifest)

    def test_exact_duplicate_is_idempotent_but_conflict_and_reordering_reject(self):
        for change in ("duplicate", "conflict", "reorder"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                duplicate = deepcopy(capture["control_events"][-1])
                if change == "conflict":
                    duplicate["fields"]["item_id"] = "other"
                elif change == "reorder":
                    duplicate["received_monotonic_s"] = 100.0001
                capture["control_events"].append(duplicate)
                capture["utterances"][1]["provider_input_finalization"]["validation_control_event_count"] += 1
                if change == "duplicate":
                    self.assertEqual(analyze(capture, pcm, manifest)["provider_input_finalization_evidence"]["validated_transactions"], 1)
                else:
                    with self.assertRaises(reader.EvidenceError):
                        analyze(capture, pcm, manifest)

    def test_later_events_after_validation_prefix_do_not_rewrite_history(self):
        capture, pcm, manifest = barrier_fixture()
        capture["control_events"].append(event("input_audio_buffer.speech_started", "next_item", 102.1, "next"))
        self.assertEqual(analyze(capture, pcm, manifest)["provider_input_finalization_evidence"]["validated_transactions"], 1)

    def test_session_index_requires_actual_ordered_session_created_and_handled_prefix(self):
        for change in ("wrong_stage", "wrong_type", "missing_handled", "not_ordered"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                session = capture["control_events"][0]
                if change == "wrong_stage": session["stage"] = "offer_created"
                elif change == "wrong_type": session["fields"]["event_type"] = "response.created"
                elif change == "missing_handled": session["handled_monotonic_s"] = None
                else: session["fields"]["boundary"] = "direct_unverified"
                with self.assertRaises(reader.EvidenceError):
                    analyze(capture, pcm, manifest)

    def test_uuid_spelling_is_not_identity_and_raw_evidence_is_not_normalized_in_place(self):
        capture, pcm, manifest = barrier_fixture()
        connection = "abcdefab-abcd-4abc-8abc-abcdefabcdef"
        capture["utterances"][1]["provider_input_finalization"]["connection_id"] = connection.upper()
        for row in capture["control_events"]:
            row["fields"]["connection_id"] = connection
        original = deepcopy(capture)
        self.assertEqual(analyze(capture, pcm, manifest)["provider_input_finalization_evidence"]["validated_transactions"], 1)
        self.assertEqual(capture, original)

    def test_timeout_or_close_is_not_reached_with_nullable_unavailable_evidence(self):
        for code in ("trigger_outputWaitExpired", "trigger_providerInputFinalizationUnavailable",
                     "enqueue_providerInputFinalizationUnavailable", "run_closed_before_injection",
                     "provider_input_ambiguous_items", "provider_input_competing_sdk_input"):
            with self.subTest(code=code):
                capture, pcm, manifest = barrier_fixture()
                row = capture["utterances"][1]
                row.update(status="not_started", callbacks=[], attempted_clip_samples=0, sdk_accepted_clip_samples=0)
                for key in ("triggered_uptime_seconds", "enqueue_requested_uptime_seconds", "first_sdk_callback_start_uptime_seconds",
                            "last_sdk_callback_end_uptime_seconds", "first_successful_activity_uptime_seconds", "last_successful_activity_uptime_seconds"):
                    row[key] = None
                proof = row["provider_input_finalization"]
                proof.update(status="not_reached", failure_code=code, connection_id=None, provider_item_id=None)
                for key in (*reader._PROVIDER_INPUT_COUNTS, *reader._PROVIDER_INPUT_TIMES):
                    proof[key] = None
                result = analyze(capture, pcm, manifest)
                self.assertEqual(result["occurrences"][1]["provider_input_finalization_verdict"], "not_reached")
                self.assertEqual(result["occurrences"][1]["input_verdict"], "not_started")
                self.assertEqual(result["provider_input_finalization_evidence"]["validated_transactions"], 0)

    def test_manifest_trace_flags_sdk_failure_and_missing_proof_never_validate(self):
        for change in ("manifest", "controls", "control_events_truncated", "control_event_timing_invalid",
                       "input_mapping_truncated", "callback_failure", "sdk_failure", "proof"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                if change == "manifest":
                    report = reader.analyze_capture(capture, pcm)
                else:
                    if change == "controls":
                        del capture["control_events"]
                    elif change == "callback_failure":
                        capture[change] = -1
                    elif change == "sdk_failure":
                        previous = capture["utterances"][0]
                        previous.update(status="failed", sdk_accepted_clip_samples=0, first_successful_activity_uptime_seconds=None,
                                        last_successful_activity_uptime_seconds=None)
                        previous["callbacks"][0]["sdk_status"] = -1
                    elif change == "proof":
                        del capture["utterances"][1]["provider_input_finalization"]
                    else:
                        capture[change] = True
                    report = analyze(capture, pcm, manifest)
                self.assertEqual(report["provider_input_finalization_evidence"]["validated_transactions"], 0)

    def test_wrong_profile_scene_deadline_index_units_unknown_fields_and_private_error_reject(self):
        for change in ("live", "manual", "scene", "timeout", "index", "milliseconds", "extra", "private", "ordinal", "early_prefix"):
            with self.subTest(change=change):
                capture, pcm, manifest = barrier_fixture()
                proof = capture["utterances"][1]["provider_input_finalization"]
                if change == "live": manifest["voice_route"] = "live"
                elif change == "manual": manifest["capture_mode"] = "manual"
                elif change == "scene": proof["scene_deadline_uptime_seconds"] += 1
                elif change == "timeout": proof["deadline_uptime_seconds"] -= 1
                elif change == "index": proof["committed_control_event_index"] = 1
                elif change == "milliseconds": proof["transaction_reserved_uptime_seconds"] *= 1000
                elif change == "extra": proof["transcript"] = "private fixture content"
                elif change == "private": proof["failure_code"] = "private fixture content"
                elif change == "ordinal": proof["after_input_ordinal"] = 1
                else: proof["clean_control_event_count"] = 2
                with self.assertRaises(reader.EvidenceError) as caught:
                    analyze(capture, pcm, manifest)
                self.assertNotIn("private fixture content", str(caught.exception))

    def test_no_pcm_does_not_claim_answer_hearing_or_actual_cut(self):
        capture, _, manifest = barrier_fixture()
        capture.update(decoded_sample_map=[], decoded_blocks=0)
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        report = analyze(capture, b"", manifest)
        self.assertIsNone(report["first_decoded_onset"]["sample"])
        self.assertIsNone(report["actual_cuts"])
        self.assertFalse(report["answer_quality_judged"])

    def test_reader_cli_uses_exact_retained_manifest_without_rewriting_artifacts(self):
        capture, pcm, manifest = barrier_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"capture.json": json.dumps(capture).encode(), "decoded.pcm": pcm,
                        "replay.json": json.dumps(manifest).encode()}
            for name, data in original.items(): (root / name).write_bytes(data)
            output = io.StringIO()
            with redirect_stdout(output):
                code = reader.main([str(root / "capture.json"), "--replay-manifest", str(root / "replay.json")])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["provider_input_finalization_evidence"]["validated_transactions"], 1)
            self.assertEqual({name: (root / name).read_bytes() for name in original}, original)


class ProviderInputCompilerTests(unittest.TestCase):
    setUp = scenario_fixtures.ScenarioTests.setUp
    compile = scenario_fixtures.ScenarioTests.compile

    def test_explicit_one_two_three_version_and_unchanged_old_default(self):
        for cuts in (1, 2, 3):
            with self.subTest(cuts=cuts):
                old = self.compile(cuts=cuts, task_aware=True)
                self.assertEqual(old, self.compile(cuts=cuts, task_aware=True, distinct_provider_turns=False))
                result = self.compile(cuts=cuts, task_aware=True, distinct_provider_turns=True)
                self.assertEqual(result.expectations["case_revision"], 5)
                self.assertEqual(result.manifest["frames"], old.manifest["frames"])
                for ordinal in range(3, len(result.manifest["utterances"]), 2):
                    self.assertEqual(result.manifest["utterances"][ordinal]["trigger"],
                        dict(kind="after_provider_input_finalized", after_input_ordinal=ordinal - 1, timeout_seconds=10.0))
                self.assertEqual(result.speech, old.speech)
                self.assertFalse(result.expectations["task_scope_staged"])
                self.assertIsNone(result.expectations["actual_cuts"])

    def test_opt_in_rejects_wrong_case_route_flag_and_insufficient_deadline(self):
        for values in (dict(task_aware=False), dict(case="baseline", cuts=0),
                       dict(binding={**self.binding, "voice_route": "live"}), dict(distinct_provider_turns=1), dict(duration=45)):
            with self.subTest(values=values):
                options = dict(task_aware=True, distinct_provider_turns=True)
                options.update(values)
                with self.assertRaises(compiler.ScenarioError): self.compile(**options)

    def test_cli_opt_in_emits_no_identity_or_oracle_and_never_stages_scope(self):
        binding, frames, output = self.root / "binding.json", self.root / "frames.json", self.root / "new"
        binding.write_text(json.dumps(self.binding))
        frames.write_text(json.dumps(self.frames))
        with patch.object(compiler, "load_bundles", return_value=self.clips), redirect_stdout(io.StringIO()):
            code = compiler.main(["--case", "pause_continue", "--cuts", "1", "--task-aware", "--distinct-provider-turns",
                                  "--binding", str(binding), "--frames", str(frames), "--output", str(output)])
        self.assertEqual(code, 0)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["utterances"][3]["trigger"]["kind"], "after_provider_input_finalized")
        self.assertFalse((output / "task-scope.json").exists())
        self.assertNotIn("provider_item_id", json.dumps(manifest))
        self.assertNotIn("semantic", json.dumps(manifest))

    def test_accounting_requires_independent_barrier_verdict_not_completed_input_alone(self):
        compiled = self.compile(task_aware=True, distinct_provider_turns=True)
        capture, pcm = scenario_fixtures.ScenarioTests.capture(self, compiled)
        for verdict in ("missing", "not_reached", "unmeasured", "validated_provider_transaction"):
            with self.subTest(verdict=verdict):
                rows = [dict(ordinal=i, occurrence_id=row["occurrence_id"], input_verdict="sdk_completed_active_input",
                    response_trigger_verdict="validated_control_binding", provider_input_finalization_verdict=verdict)
                    for i, row in enumerate(compiled.expectations["occurrences"])]
                with patch.object(compiler, "analyze_capture", return_value=dict(occurrences=rows, timing_blockers=[])):
                    result = compiler.account_execution(compiled, capture, pcm)
                self.assertEqual(result["input_execution"], "complete" if verdict == "validated_provider_transaction" else "incomplete")
                self.assertIsNone(result["actual_cuts"])
                self.assertFalse(result["semantic_assertions_evaluated"])


if __name__ == "__main__":
    unittest.main()
