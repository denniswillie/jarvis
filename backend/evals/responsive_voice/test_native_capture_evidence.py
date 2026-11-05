"""Synthetic offline v2 fixtures; never starts a provider, app or service."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from uuid import UUID

from evals.responsive_voice.native_capture_evidence import (
    EvidenceError, RATE, analyze_capture, main, read_capture,
)


RUN = UUID("11111111-1111-4111-8111-111111111111")
MEDIA = UUID("22222222-2222-4222-8222-222222222222")


def fixture(count=1):
    capture = {
        "schema_version": 2, "run_id": str(RUN), "media_epoch_id": str(MEDIA),
        "provenance": "frontend_replay", "boundary": "custom_native_adm_pcm_callback",
        "physical_playback_measured": False, "provider_closure_confirmed": False, "useful_answer_onset": None,
        "input_threshold_abs_pcm16": 1, "decoded_output_threshold_abs_pcm16": 17,
        "legacy_input_activity_boundary": "render_preparation_before_sdk_call_not_delivery_confirmation",
        "truncated": False, "decoded_mapping_truncated": False, "input_mapping_truncated": False,
        "output_gate_history_truncated": False, "callback_failure": None,
        "clock_anchor": {
            "monotonic_before_seconds": 99.0, "monotonic_after_seconds": 99.001,
            "monotonic_reference_seconds": 99.0005, "sampling_bracket_seconds": 0.001,
            "utc_unix_seconds": 1_700_000_000.5,
            "utc_iso8601": datetime.fromtimestamp(1_700_000_000.5, timezone.utc).isoformat(),
        },
        "utterances": [], "decoded_sample_map": [], "output_gates": [],
        "input_first_non_silent_uptime": [], "input_last_non_silent_uptime": [], "output_starts": [],
        "input_blocks": count, "decoded_blocks": count * 2, "gated_decoded_blocks": 0,
        "final_ui_status": "fixture", "qualification": "synthetic offline fixture, not actual provider evidence",
    }
    samples, annotations = [], []
    epoch = 1

    def gate(time, enabled, current_epoch):
        capture["output_gates"].append({
            "event_id": f"{MEDIA}/output-gate/{len(capture['output_gates'])}",
            "enabled": enabled, "changed": True, "requested_uptime_seconds": time,
            "local_output_epoch": current_epoch, "last_admitted_active_sample": None,
        })

    def block(time, current_epoch):
        offset = len(samples)
        samples.extend([0, 100, 200, 0])
        capture["decoded_sample_map"].append({
            "sample_offset": offset, "sample_count": 4, "callback_start_uptime_seconds": time,
            "callback_end_uptime_seconds": time + 0.001, "local_output_epoch": current_epoch,
        })
        return offset

    def annotate(ordinal, kind, offset, current_epoch):
        annotations.append({
            "annotation_id": f"review/{ordinal}/{kind}", "occurrence_id": f"{RUN.hex}/utterance/{ordinal}",
            "kind": kind, "sample_offset": offset, "local_output_epoch": current_epoch,
            "provenance": {"method": "human_audio_review", "review_id": "fixture-review", "reference": "explicit synthetic annotation"},
        })

    gate(99.1, True, epoch)
    for ordinal in range(count):
        at = 100.0 + ordinal * 2
        identity = f"{RUN.hex}/utterance/{ordinal}"
        capture["utterances"].append({
            "ordinal": ordinal, "occurrence_id": identity, "clip_id": "same-clip",
            "trigger_id": identity + "/trigger", "trigger": {"kind": "at_time", "at_seconds": ordinal * 2},
            "status": "sdk_input_complete", "failure_code": None, "sample_count": 4,
            "pcm16_sha256": hashlib.sha256(struct.pack("<4h", 0, 128, 128, 0)).hexdigest(),
            "all_samples_zero": False, "activity_threshold_abs_pcm16": 128,
            "activity_comparison": ">=", "activity_threshold_origin": "speech_manifest",
            "source_first_active_sample": 1, "source_last_active_sample_exclusive": 3,
            "trigger_armed_uptime_seconds": 99.1, "triggered_uptime_seconds": at - 0.1,
            "enqueue_requested_uptime_seconds": at - 0.09,
            "first_sdk_callback_start_uptime_seconds": at, "last_sdk_callback_end_uptime_seconds": at + 0.001,
            "first_successful_activity_uptime_seconds": at + 1 / RATE,
            "last_successful_activity_uptime_seconds": at + 2 / RATE,
            "stop_requested_uptime_seconds": None, "attempted_clip_samples": 4, "sdk_accepted_clip_samples": 4,
            "callbacks": [{"source_sample_offset": 0, "clip_sample_count": 4, "transport_frame_count": 480,
                           "callback_start_uptime_seconds": at, "callback_end_uptime_seconds": at + 0.001, "sdk_status": 0}],
        })
        old_offset = block(at + 0.02, epoch)
        annotate(ordinal, "old_answer_tail", old_offset + 2, epoch)
        gate(at + 0.04, False, epoch)
        epoch += 1
        gate(at + 0.15, True, epoch)
        useful_offset = block(at + 0.2, epoch)
        annotate(ordinal, "useful_onset", useful_offset + 1, epoch)
        # This remains open into the next cut, representing resumed output.
    gate(100 + (count - 1) * 2 + 0.4, False, epoch)
    pcm = struct.pack("<" + "h" * len(samples), *samples)
    capture["decoded_pcm"] = {"filename": "decoded.pcm", "sample_rate_hz": RATE, "channels": 1,
                              "encoding": "signed_pcm16_little_endian", "sample_count": len(samples), "byte_count": len(pcm)}
    document = {"schema_version": 1, "capture_run_id": str(RUN), "media_epoch_id": str(MEDIA),
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "annotations": annotations}
    return capture, pcm, document


class NativeCaptureEvidenceTests(unittest.TestCase):
    def test_one_two_three_occurrences_keep_distinct_identity_and_separate_intervals(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                capture, pcm, annotations = fixture(count)
                result = analyze_capture(capture, pcm, annotations)
                self.assertEqual(len({item["occurrence_id"] for item in result["occurrences"]}), count)
                for ordinal, item in enumerate(result["occurrences"]):
                    self.assertEqual(item["clip_id"], "same-clip")
                    self.assertEqual(item["input_verdict"], "sdk_completed_active_input")
                    tail = item["speech_onset_to_annotated_old_tail"]
                    useful = item["input_end_to_annotated_useful_onset"]
                    self.assertAlmostEqual(tail["delta_seconds"], 0.02 + 1 / RATE, places=8)
                    self.assertAlmostEqual(useful["delta_seconds"], 0.2 - 2 / RATE, places=8)
                    self.assertEqual(tail["location"]["local_output_epoch"], ordinal + 1)
                    self.assertEqual(useful["location"]["local_output_epoch"], ordinal + 2)
                    self.assertFalse(useful["semantic_label_verified"])

    def test_first_decoded_observation_is_not_useful_or_attributed_response_onset(self):
        capture, pcm, annotations = fixture()
        result = analyze_capture(capture, pcm, annotations)
        energy = result["occurrences"][0]["first_decoded_non_silent_after_input_end"]
        useful = result["occurrences"][0]["input_end_to_annotated_useful_onset"]
        self.assertEqual(energy["sample"]["sample_offset"], 1)
        self.assertNotEqual(energy["sample"]["local_output_epoch"], useful["location"]["local_output_epoch"])
        self.assertFalse(energy["response_onset_established"])
        self.assertFalse(result["first_decoded_onset"]["useful_speech_established"])

    def test_missing_annotations_never_fall_back_to_energy_or_cursors(self):
        capture, pcm, _ = fixture()
        capture["output_starts"] = [{"local_cursor": 1, "uptime": 100.02}]
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["verdict"], "not_annotated")
        self.assertEqual(result["occurrences"][0]["speech_onset_to_annotated_old_tail"]["verdict"], "not_annotated")

    def test_hold_on_without_output_is_not_a_quality_failure(self):
        capture, _, _ = fixture()
        capture["decoded_sample_map"] = []
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        result = analyze_capture(capture, b"")
        self.assertIsNone(result["first_decoded_onset"]["sample"])
        self.assertEqual(result["occurrences"][0]["input_verdict"], "sdk_completed_active_input")
        self.assertEqual(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["verdict"], "not_annotated")
        self.assertFalse(result["answer_quality_judged"])

    def test_gap_is_preserved_instead_of_using_file_offset_as_continuous_time(self):
        capture, pcm, annotations = fixture()
        capture["decoded_sample_map"][1].update(callback_start_uptime_seconds=100.8, callback_end_uptime_seconds=100.801)
        capture["output_gates"][-1]["requested_uptime_seconds"] = 100.9
        result = analyze_capture(capture, pcm, annotations)
        self.assertAlmostEqual(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["delta_seconds"], 0.8 - 2 / RATE, places=8)

    def test_negative_old_tail_interval_is_preserved_not_clamped(self):
        capture, pcm, annotations = fixture()
        capture["decoded_sample_map"][0].update(callback_start_uptime_seconds=99.5, callback_end_uptime_seconds=99.501)
        tail = analyze_capture(capture, pcm, annotations)["occurrences"][0]["speech_onset_to_annotated_old_tail"]
        self.assertTrue(tail["negative_interval"])
        self.assertAlmostEqual(tail["delta_seconds"], -0.5 + 1 / RATE, places=8)

    def test_silent_and_unstarted_input_do_not_gain_fabricated_speech_boundaries(self):
        for mode in ("silent", "not_started", "failed"):
            with self.subTest(mode=mode):
                capture, pcm, annotations = fixture()
                item = capture["utterances"][0]
                item.update(first_successful_activity_uptime_seconds=None, last_successful_activity_uptime_seconds=None)
                if mode == "silent":
                    item.update(all_samples_zero=True, source_first_active_sample=None, source_last_active_sample_exclusive=None)
                    expected = "silent_input"
                elif mode == "not_started":
                    item.update(status="not_started", attempted_clip_samples=0, sdk_accepted_clip_samples=0, callbacks=[],
                                first_sdk_callback_start_uptime_seconds=None, last_sdk_callback_end_uptime_seconds=None)
                    expected = "not_started"
                else:
                    item.update(status="failed", sdk_accepted_clip_samples=0)
                    item["callbacks"][0]["sdk_status"] = -50
                    expected = "incomplete_or_failed_input"
                result = analyze_capture(capture, pcm, annotations)["occurrences"][0]
                self.assertEqual(result["input_verdict"], expected)
                self.assertIsNone(result["speech_onset_uptime_seconds"])
                self.assertIsNone(result["input_end_to_annotated_useful_onset"]["delta_seconds"])

    def test_completed_label_requires_actual_complete_sdk_acceptance(self):
        capture, pcm, _ = fixture()
        capture["utterances"][0]["sdk_accepted_clip_samples"] = 3
        with self.assertRaises(EvidenceError):
            analyze_capture(capture, pcm)

    def test_truncation_and_callback_failure_block_annotation_timing(self):
        for field, value in [("truncated", True), ("decoded_mapping_truncated", True),
                             ("input_mapping_truncated", True), ("output_gate_history_truncated", True), ("callback_failure", -50)]:
            with self.subTest(field=field):
                capture, pcm, annotations = fixture()
                capture[field] = value
                result = analyze_capture(capture, pcm, annotations)
                self.assertEqual(result["capture_verdict"], "limited_evidence")
                self.assertEqual(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["verdict"], "timing_unqualified")

    def test_pcm_length_odd_bytes_and_unmapped_samples_are_rejected(self):
        capture, pcm, _ = fixture()
        for changed_pcm in (pcm[:-1], pcm + b"\0\0"):
            with self.subTest(length=len(changed_pcm)), self.assertRaises(EvidenceError):
                analyze_capture(capture, changed_pcm)
        capture["decoded_sample_map"][1]["sample_offset"] += 1
        with self.assertRaisesRegex(EvidenceError, "contiguous"):
            analyze_capture(capture, pcm)

    def test_map_reversal_and_millisecond_domain_mix_are_rejected(self):
        capture, pcm, _ = fixture()
        capture["decoded_sample_map"][1].update(callback_start_uptime_seconds=100.01, callback_end_uptime_seconds=100.011)
        with self.assertRaisesRegex(EvidenceError, "nonmonotonic"):
            analyze_capture(capture, pcm)
        capture, pcm, _ = fixture()
        capture["utterances"][0]["callbacks"][0]["callback_start_uptime_seconds"] *= 1000
        with self.assertRaisesRegex(EvidenceError, "monotonic_seconds_domain"):
            analyze_capture(capture, pcm)

    def test_activity_sample_offsets_and_seconds_must_agree(self):
        capture, pcm, _ = fixture()
        capture["utterances"][0]["last_successful_activity_uptime_seconds"] += 1 / RATE
        with self.assertRaisesRegex(EvidenceError, "sample_mapping"):
            analyze_capture(capture, pcm)

    def test_missing_clock_is_limited_evidence_not_calibrated_latency(self):
        capture, pcm, annotations = fixture()
        capture["clock_anchor"] = None
        result = analyze_capture(capture, pcm, annotations)
        self.assertIn("clock_anchor_missing", result["timing_blockers"])
        self.assertIsNone(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["delta_seconds"])

    def test_bad_utc_pair_and_nonfinite_numbers_are_rejected(self):
        capture, pcm, _ = fixture()
        capture["clock_anchor"]["utc_unix_seconds"] += 10
        with self.assertRaises(EvidenceError):
            analyze_capture(capture, pcm)
        capture, pcm, _ = fixture()
        capture["decoded_sample_map"][0]["callback_start_uptime_seconds"] = float("nan")
        with self.assertRaises(EvidenceError):
            analyze_capture(capture, pcm)

    def test_annotations_require_exact_file_and_capture_identity(self):
        for field, value in [("pcm_sha256", "0" * 64), ("capture_run_id", str(MEDIA)), ("media_epoch_id", str(RUN))]:
            with self.subTest(field=field):
                capture, pcm, annotations = fixture()
                annotations[field] = value
                with self.assertRaisesRegex(EvidenceError, "identity_mismatch"):
                    analyze_capture(capture, pcm, annotations)

    def test_out_of_range_negative_boolean_and_wrong_epoch_annotations_are_invalid(self):
        for field, value in [("sample_offset", 8), ("sample_offset", -1), ("sample_offset", True), ("local_output_epoch", 99)]:
            with self.subTest(field=field, value=value):
                capture, pcm, annotations = fixture()
                annotations["annotations"][1][field] = value
                result = analyze_capture(capture, pcm, annotations)
                self.assertEqual(result["occurrences"][0]["input_end_to_annotated_useful_onset"]["verdict"], "invalid_annotation")

    def test_annotation_provenance_cannot_be_energy_or_cursor_inference(self):
        capture, pcm, annotations = fixture()
        annotations["annotations"][1]["provenance"]["method"] = "energy_threshold"
        value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["input_end_to_annotated_useful_onset"]
        self.assertEqual(value["verdict"], "invalid_annotation")

    def test_multiple_annotations_are_ambiguous_not_automatically_selected(self):
        capture, pcm, annotations = fixture()
        extra = deepcopy(annotations["annotations"][1])
        extra.update(annotation_id="second-review", sample_offset=6)
        annotations["annotations"].append(extra)
        value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["input_end_to_annotated_useful_onset"]
        self.assertEqual(value["verdict"], "ambiguous_annotations")
        self.assertIsNone(value["delta_seconds"])

    def test_missing_or_crossed_epoch_boundaries_do_not_guess_attribution(self):
        for mode in ("missing", "crossed", "before_open"):
            with self.subTest(mode=mode):
                capture, pcm, annotations = fixture()
                if mode == "missing":
                    capture["output_gates"] = []
                elif mode == "crossed":
                    capture["decoded_sample_map"][1]["callback_end_uptime_seconds"] = 100.4
                else:
                    capture["decoded_sample_map"][1].update(callback_start_uptime_seconds=100.14, callback_end_uptime_seconds=100.141)
                value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["input_end_to_annotated_useful_onset"]
                self.assertEqual(value["verdict"], "timing_unqualified")

    def test_samples_claiming_an_already_closed_epoch_are_not_qualified(self):
        capture, pcm, annotations = fixture()
        capture["decoded_sample_map"][0].update(callback_start_uptime_seconds=100.045, callback_end_uptime_seconds=100.046)
        value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["speech_onset_to_annotated_old_tail"]
        self.assertEqual(value["verdict"], "timing_unqualified")

    def test_preclose_admission_allows_nominal_sample_tail_after_close(self):
        capture, pcm, annotations = fixture()
        capture["decoded_sample_map"][0].update(
            callback_start_uptime_seconds=100.03998, callback_end_uptime_seconds=100.03999)
        value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["speech_onset_to_annotated_old_tail"]
        self.assertEqual(value["verdict"], "externally_annotated_nominal_timing")
        self.assertGreater(value["location"]["nominal_uptime_seconds"], 100.04)
        self.assertFalse(value["semantic_label_verified"])

    def test_callback_end_at_or_after_close_and_old_epoch_after_reopen_are_unqualified(self):
        for start, end in ((100.039, 100.04), (100.039, 100.041), (100.17, 100.171)):
            with self.subTest(start=start, end=end):
                capture, pcm, annotations = fixture()
                capture["decoded_sample_map"][0].update(
                    callback_start_uptime_seconds=start, callback_end_uptime_seconds=end)
                value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["speech_onset_to_annotated_old_tail"]
                self.assertEqual(value["verdict"], "timing_unqualified")
                self.assertIsNone(value["delta_seconds"])

    def test_unknown_close_time_does_not_establish_epoch_admission(self):
        capture, pcm, annotations = fixture()
        capture["output_gates"][1]["requested_uptime_seconds"] = None
        value = analyze_capture(capture, pcm, annotations)["occurrences"][0]["speech_onset_to_annotated_old_tail"]
        self.assertEqual(value["verdict"], "timing_unqualified")
        self.assertIn("local_epoch_history_missing", value["reasons"])

    def test_null_callback_list_is_a_validation_error_not_a_typeerror(self):
        for callbacks in (None, {}, "not-a-list"):
            with self.subTest(callbacks=callbacks):
                capture, pcm, _ = fixture()
                capture["utterances"][0]["callbacks"] = callbacks
                with self.assertRaisesRegex(EvidenceError, "^invalid_input_callbacks$"):
                    analyze_capture(capture, pcm)

    def test_cli_null_callbacks_reports_controlled_error(self):
        capture, pcm, _ = fixture()
        capture["utterances"][0]["callbacks"] = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture.json").write_text(json.dumps(capture))
            (root / "decoded.pcm").write_bytes(pcm)
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(root / "capture.json")]), 2)
            self.assertEqual(json.loads(output.getvalue()), {
                "verdict": "invalid_artifact", "error_code": "invalid_input_callbacks"})

    def test_reversed_occurrence_callback_order_is_rejected(self):
        capture, pcm, _ = fixture(2)
        first, second = capture["utterances"]
        for key in ("first_sdk_callback_start_uptime_seconds", "last_sdk_callback_end_uptime_seconds",
                    "first_successful_activity_uptime_seconds", "last_successful_activity_uptime_seconds"):
            first[key], second[key] = second[key], first[key]
        first["callbacks"], second["callbacks"] = second["callbacks"], first["callbacks"]
        with self.assertRaises(EvidenceError):
            analyze_capture(capture, pcm)

    def test_cross_occurrence_callback_overlap_is_rejected_but_touching_is_valid(self):
        for end, reject in ((102.001, True), (102.0, False)):
            with self.subTest(end=end):
                capture, pcm, annotations = fixture(2)
                capture["utterances"][0]["callbacks"][0]["callback_end_uptime_seconds"] = end
                capture["utterances"][0]["last_sdk_callback_end_uptime_seconds"] = end
                if reject:
                    with self.assertRaisesRegex(EvidenceError, "^nonmonotonic_cross_occurrence_input_callbacks$"):
                        analyze_capture(capture, pcm, annotations)
                else:
                    result = analyze_capture(capture, pcm, annotations)
                    self.assertEqual(len(result["occurrences"]), 2)
                    self.assertEqual(result["provider_response_attribution"], "unavailable")

    def test_unstarted_occurrence_does_not_reset_callback_chronology(self):
        capture, pcm, _ = fixture(3)
        first, middle, last = capture["utterances"]
        middle.update(status="not_started", callbacks=[], attempted_clip_samples=0, sdk_accepted_clip_samples=0,
                      first_sdk_callback_start_uptime_seconds=None, last_sdk_callback_end_uptime_seconds=None,
                      first_successful_activity_uptime_seconds=None, last_successful_activity_uptime_seconds=None)
        self.assertEqual(analyze_capture(capture, pcm)["occurrences"][1]["input_verdict"], "not_started")
        for key in ("first_sdk_callback_start_uptime_seconds", "last_sdk_callback_end_uptime_seconds",
                    "first_successful_activity_uptime_seconds", "last_successful_activity_uptime_seconds", "callbacks"):
            first[key], last[key] = last[key], first[key]
        with self.assertRaisesRegex(EvidenceError, "^nonmonotonic_cross_occurrence_input_callbacks$"):
            analyze_capture(capture, pcm)

    def test_failed_startup_with_no_clock_or_io_has_no_latency_or_answer_claim(self):
        capture, _, _ = fixture()
        capture.update(clock_anchor=None, input_blocks=0, decoded_blocks=0, decoded_sample_map=[], output_starts=[],
                       final_ui_status="The requested voice route is not enabled.")
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        capture["output_gates"] = [dict(event_id=f"{MEDIA}/output-gate/{index}", enabled=False, changed=False,
                                       local_output_epoch=0, requested_uptime_seconds=44390.39 + index / 10,
                                       last_admitted_active_sample=None) for index in range(3)]
        item = capture["utterances"][0]
        item.update(status="not_started", failure_code="run_closed_before_injection", sample_count=72004,
                    source_first_active_sample=147, source_last_active_sample_exclusive=70211,
                    attempted_clip_samples=0, sdk_accepted_clip_samples=0, callbacks=[],
                    trigger_armed_uptime_seconds=None, triggered_uptime_seconds=None, enqueue_requested_uptime_seconds=None,
                    first_sdk_callback_start_uptime_seconds=None, last_sdk_callback_end_uptime_seconds=None,
                    first_successful_activity_uptime_seconds=None, last_successful_activity_uptime_seconds=None,
                    stop_requested_uptime_seconds=44390.645158)
        result = analyze_capture(capture, b"")
        self.assertEqual(result["capture_verdict"], "limited_evidence")
        self.assertEqual(result["timing_blockers"], ["clock_anchor_missing"])
        self.assertIsNone(result["first_decoded_onset"]["sample"])
        self.assertFalse(result["first_decoded_onset"]["useful_speech_established"])
        self.assertFalse(result["answer_quality_judged"])
        self.assertFalse(result["hearing_measured"])
        self.assertFalse(result["aggregate_percentile_qualification"]["supported"])
        occurrence = result["occurrences"][0]
        self.assertEqual(occurrence["input_verdict"], "not_started")
        self.assertIsNone(occurrence["speech_onset_uptime_seconds"])
        self.assertIsNone(occurrence["input_end_exclusive_uptime_seconds"])
        self.assertIsNone(occurrence["first_decoded_non_silent_after_input_end"]["sample"])
        for name in ("speech_onset_to_annotated_old_tail", "input_end_to_annotated_useful_onset"):
            self.assertEqual(occurrence[name]["verdict"], "not_annotated")
            self.assertIsNone(occurrence[name]["delta_seconds"])

    def test_twenty_occurrences_do_not_establish_percentile_qualification(self):
        capture, pcm, annotations = fixture(20)
        result = analyze_capture(capture, pcm, annotations)
        self.assertFalse(result["aggregate_percentile_qualification"]["supported"])
        self.assertFalse(result["hearing_measured"])
        self.assertFalse(result["answer_quality_judged"])

    def test_reader_and_cli_preserve_raw_artifacts_and_write_only_stdout(self):
        capture, pcm, annotations = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "capture.json", root / "decoded.pcm", root / "annotations.json"]
            contents = [json.dumps(capture).encode(), pcm, json.dumps(annotations).encode()]
            for path, content in zip(paths, contents):
                path.write_bytes(content)
            result = read_capture(paths[0], annotations_path=paths[2])
            self.assertEqual(result["pcm_sha256"], hashlib.sha256(pcm).hexdigest())
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(paths[0]), "--annotations", str(paths[2])]), 0)
            self.assertFalse(json.loads(output.getvalue())["hearing_measured"])
            self.assertEqual([path.read_bytes() for path in paths], contents)
            self.assertEqual({path.name for path in root.iterdir()}, {path.name for path in paths})

    def test_duplicate_json_keys_fail_without_echoing_raw_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text('{"schema_version":2,"schema_version":"sensitive-fixture"}')
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main([str(path)]), 2)
            self.assertIn("duplicate_json_key", output.getvalue())
            self.assertNotIn("sensitive-fixture", output.getvalue())


CONNECTION = UUID("33333333-3333-4333-8333-333333333333")


def response_fixture(cuts=1):
    capture, pcm, _ = fixture(cuts + 1)
    capture.update(response_trigger_schema_version=1, response_correlation_basis="authorized_gate_interval",
        response_semantic_causality="unknown", response_evidence_truncated=False,
        response_evidence_invalid=False, uncorrelated_decoded_blocks=0,
        local_interruptions=[], response_stop_measurements=[])

    def identity(epoch):
        return dict(connection_id=str(CONNECTION), response_id=f"resp_fixture{epoch}",
                    item_id=f"item_fixture{epoch}", local_output_epoch=epoch)

    def point(block, relative):
        return dict(sample_offset=block["sample_offset"] + relative,
            nominal_uptime_seconds=block["callback_start_uptime_seconds"] + relative / RATE,
            local_output_epoch=block["local_output_epoch"])

    for block in capture["decoded_sample_map"]:
        block["response_identity"] = identity(block["local_output_epoch"])
    for gate in capture["output_gates"]:
        gate.update(response_identity=identity(gate["local_output_epoch"]), local_interruption_event_id=None)
    for item in capture["utterances"]:
        item["response_trigger"] = None
    for ordinal in range(1, cuts + 1):
        item, previous = capture["utterances"][ordinal], capture["utterances"][ordinal - 1]
        at = 100 + ordinal * 2
        block = capture["decoded_sample_map"][2 * ordinal - 1]
        block.update(callback_start_uptime_seconds=at - 0.1, callback_end_uptime_seconds=at - 0.099)
        selected, revalidated = at - 0.05, at - 0.04
        item.update(trigger={"kind": "response_output_start", "after_input_ordinal": ordinal - 1,
                            "delay_seconds": 0, "timeout_seconds": 3, "max_pcm_age_seconds": 0.2},
            trigger_armed_uptime_seconds=previous["last_sdk_callback_end_uptime_seconds"],
            triggered_uptime_seconds=selected, enqueue_requested_uptime_seconds=revalidated)
        owner = identity(ordinal + 1)
        item["response_trigger"] = dict(identity=owner, after_input_ordinal=ordinal - 1,
            input_complete_uptime_seconds=previous["last_sdk_callback_end_uptime_seconds"],
            response_admitted_uptime_seconds=capture["output_gates"][2 * ordinal]["requested_uptime_seconds"],
            first_retained_active_sample=point(block, 1), latest_retained_active_sample=point(block, 2),
            latest_decoded_callback_end_uptime_seconds=block["callback_end_uptime_seconds"],
            selected_uptime_seconds=selected, enqueue_revalidated=True,
            enqueue_revalidated_uptime_seconds=revalidated,
            pcm_age_seconds=revalidated - block["callback_end_uptime_seconds"],
            failure_code=None, semantic_causality="unknown")
        interrupt_id = f"{MEDIA}/local-interruption/{ordinal - 1}"
        requested = at + 0.03
        capture["local_interruptions"].append(dict(event_id=interrupt_id, identity=owner,
            cause="client_interrupt_method", requested_uptime_seconds=requested))
        gate = capture["output_gates"][2 * ordinal + 1]
        gate["local_interruption_event_id"] = interrupt_id
        onset = item["first_successful_activity_uptime_seconds"]
        tail = capture["decoded_sample_map"][2 * ordinal]
        capture["response_stop_measurements"].append(dict(occurrence_id=item["occurrence_id"], identity=owner,
            interruption_event_id=interrupt_id, output_gate_event_id=gate["event_id"],
            input_first_successful_activity_uptime_seconds=onset,
            local_interruption_requested_uptime_seconds=requested,
            local_gate_closed_uptime_seconds=gate["requested_uptime_seconds"],
            input_to_local_interruption_seconds=requested - onset,
            input_to_local_gate_close_seconds=gate["requested_uptime_seconds"] - onset,
            last_retained_active_minus_input_seconds=tail["callback_start_uptime_seconds"] + 2 / RATE - onset))
    return capture, pcm


class ResponseControlEvidenceTests(unittest.TestCase):
    def test_legacy_captures_keep_existing_verdicts_and_unmeasured_cuts(self):
        capture, pcm, _ = fixture()
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["response_control_evidence"]["verdict"], "not_recorded")
        self.assertEqual(result["occurrences"][0]["response_trigger_verdict"], "not_applicable")
        self.assertIsNone(result["occurrences"][0]["response_control_stop"])
        self.assertIsNone(result["actual_cuts"])

    def test_one_two_three_distinct_response_occurrences_with_repeated_clip(self):
        for count in (1, 2, 3):
            with self.subTest(count=count):
                capture, pcm = response_fixture(count)
                saved = deepcopy(capture)
                result = analyze_capture(capture, pcm)
                evidence = result["response_control_evidence"]
                self.assertEqual(evidence["control_correlated_stop_count"], count)
                self.assertEqual(evidence["verdict"], "validated_control_evidence")
                for item in result["occurrences"][1:]:
                    self.assertEqual(item["clip_id"], "same-clip")
                    self.assertEqual(item["response_trigger_verdict"], "validated_control_binding")
                    self.assertEqual(item["response_control_stop"]["verdict"], "validated_control_correlated_stop")
                self.assertEqual(capture, saved)
                self.assertIsNone(result["actual_cuts"])
                self.assertEqual(result["provider_response_attribution"], "unavailable")
                self.assertEqual(evidence["semantic_causality"], "unknown")
                self.assertFalse(evidence["usefulness_measured"])
                self.assertFalse(result["hearing_measured"])

    def test_missing_trigger_is_not_a_completed_binding(self):
        capture, pcm = response_fixture()
        capture["utterances"][1]["response_trigger"] = None
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["occurrences"][1]["response_trigger_verdict"], "missing")
        self.assertIsNone(result["occurrences"][1]["response_control_stop"])
        self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])

    def test_zero_pcm_cannot_pass_strong_evidence_even_with_completed_input(self):
        capture, _ = response_fixture()
        capture["decoded_sample_map"] = []
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        capture["utterances"][1]["response_trigger"] = None
        capture["response_stop_measurements"] = []
        result = analyze_capture(capture, b"")
        self.assertEqual(result["response_control_evidence"]["verdict"], "unqualified_evidence")
        self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])
        self.assertIsNone(result["first_decoded_onset"]["sample"])

    def test_truncated_invalid_and_missing_retained_identity_never_qualify(self):
        for mode in ("response_evidence_truncated", "response_evidence_invalid", "missing_identity"):
            with self.subTest(mode=mode):
                capture, pcm = response_fixture()
                if mode == "missing_identity":
                    capture["decoded_sample_map"][1]["response_identity"] = None
                    capture["uncorrelated_decoded_blocks"] = 1
                else:
                    capture[mode] = True
                result = analyze_capture(capture, pcm)
                self.assertEqual(result["occurrences"][1]["response_trigger_verdict"], "unmeasured")
                self.assertIsNone(result["occurrences"][1]["response_control_stop"])
                self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])

    def test_stale_at_enqueue_does_not_pass_with_fresh_selection_age(self):
        capture, pcm = response_fixture()
        item = capture["utterances"][1]
        item["trigger"]["max_pcm_age_seconds"] = 0.05
        proof = item["response_trigger"]
        proof["pcm_age_seconds"] = proof["selected_uptime_seconds"] - proof["latest_decoded_callback_end_uptime_seconds"]
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["occurrences"][1]["response_trigger_verdict"], "unmeasured")
        self.assertIn("response_pcm_stale_at_enqueue", result["response_control_evidence"]["trigger_diagnostics"][0]["reasons"])

    def closed_gate_fixture(self, closed_at):
        capture, pcm = response_fixture()
        # Keep only callbacks actually admitted before the earlier natural
        # closure. Fresh retained PCM alone must not authorize either trigger
        # boundary, even though the exported revalidation flag says true.
        removed = capture["decoded_sample_map"].pop(2)
        begin, count = removed["sample_offset"], removed["sample_count"]
        pcm = pcm[:begin * 2] + pcm[(begin + count) * 2:]
        for block in capture["decoded_sample_map"][2:]:
            block["sample_offset"] -= count
        capture["decoded_pcm"].update(sample_count=len(pcm) // 2, byte_count=len(pcm))
        capture["decoded_blocks"] -= 1
        capture["output_gates"][3].update(requested_uptime_seconds=closed_at, local_interruption_event_id=None)
        capture["local_interruptions"] = []
        capture["response_stop_measurements"] = []
        return capture, pcm

    def test_gate_closed_before_selection_cannot_validate_fresh_response_trigger(self):
        capture, pcm = self.closed_gate_fixture(101.94)
        result = analyze_capture(capture, pcm)
        occurrence = result["occurrences"][1]
        self.assertEqual(occurrence["input_verdict"], "sdk_completed_active_input")
        self.assertEqual(occurrence["response_trigger_verdict"], "unmeasured")
        reasons = result["response_control_evidence"]["trigger_diagnostics"][0]["reasons"]
        self.assertIn("response_gate_not_open_at_selection", reasons)
        self.assertIn("response_gate_not_open_at_enqueue", reasons)
        self.assertNotIn("response_pcm_stale_at_enqueue", reasons)
        self.assertIsNone(occurrence["response_control_stop"])
        self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])

    def test_gate_closes_between_selection_and_enqueue_revalidation(self):
        capture, pcm = self.closed_gate_fixture(101.955)
        result = analyze_capture(capture, pcm)
        occurrence = result["occurrences"][1]
        self.assertEqual(occurrence["input_verdict"], "sdk_completed_active_input")
        self.assertEqual(occurrence["response_trigger_verdict"], "unmeasured")
        reasons = result["response_control_evidence"]["trigger_diagnostics"][0]["reasons"]
        self.assertNotIn("response_gate_not_open_at_selection", reasons)
        self.assertIn("response_gate_not_open_at_enqueue", reasons)
        self.assertNotIn("response_pcm_stale_at_enqueue", reasons)
        self.assertIsNone(occurrence["response_control_stop"])
        self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])

    def test_deadline_is_measured_from_previous_full_sdk_completion(self):
        capture, pcm = response_fixture()
        capture["utterances"][1]["trigger"]["timeout_seconds"] = 1
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["occurrences"][1]["response_trigger_verdict"], "unmeasured")
        self.assertIn("response_trigger_deadline_exceeded", result["response_control_evidence"]["trigger_diagnostics"][0]["reasons"])

    def test_previous_sdk_failure_and_current_sdk_failure_are_not_semantic_passes(self):
        for ordinal in (0, 1):
            with self.subTest(ordinal=ordinal):
                capture, pcm = response_fixture()
                item = capture["utterances"][ordinal]
                item.update(status="failed", sdk_accepted_clip_samples=0,
                    first_successful_activity_uptime_seconds=None, last_successful_activity_uptime_seconds=None)
                item["callbacks"][0]["sdk_status"] = -50
                result = analyze_capture(capture, pcm)
                bound = result["occurrences"][1]
                self.assertEqual(bound["response_trigger_verdict"], "unmeasured" if ordinal == 0 else "validated_control_binding")
                self.assertIsNone(bound["response_control_stop"])

    def test_natural_gate_close_without_explicit_interruption_is_not_a_stop(self):
        capture, pcm = response_fixture()
        capture["local_interruptions"] = []
        capture["output_gates"][3]["local_interruption_event_id"] = None
        result = analyze_capture(capture, pcm)
        self.assertEqual(result["occurrences"][1]["response_trigger_verdict"], "validated_control_binding")
        self.assertIsNone(result["occurrences"][1]["response_control_stop"])
        self.assertIsNone(result["response_control_evidence"]["control_correlated_stop_count"])

    def test_provider_done_cannot_be_labeled_local_interruption(self):
        capture, pcm = response_fixture()
        capture["local_interruptions"][0]["cause"] = "output_audio_buffer.stopped"
        with self.assertRaisesRegex(EvidenceError, "unsupported_local_interruption_cause"):
            analyze_capture(capture, pcm)

    def test_explicit_route_close_and_nullable_tail_remain_control_only(self):
        capture, pcm = response_fixture()
        capture["local_interruptions"][0]["cause"] = "route_output_fence_close"
        capture["response_stop_measurements"][0]["last_retained_active_minus_input_seconds"] = None
        result = analyze_capture(capture, pcm)
        stop = result["occurrences"][1]["response_control_stop"]
        self.assertEqual(stop["cause"], "route_output_fence_close")
        self.assertIsNone(stop["last_retained_active_minus_input_seconds"])
        self.assertIsNone(result["actual_cuts"])

    def test_foreign_identity_and_reused_output_epoch_are_rejected(self):
        for mode in ("connection", "epoch"):
            with self.subTest(mode=mode):
                capture, pcm = response_fixture(2)
                if mode == "connection":
                    capture["utterances"][1]["response_trigger"]["identity"] = dict(
                        capture["utterances"][1]["response_trigger"]["identity"], connection_id=str(RUN))
                else:
                    capture["utterances"][2]["response_trigger"]["identity"] = deepcopy(
                        capture["utterances"][1]["response_trigger"]["identity"])
                with self.assertRaises(EvidenceError):
                    analyze_capture(capture, pcm)

    def test_bad_point_offset_time_epoch_or_silent_sample_is_rejected(self):
        for key, value in (("sample_offset", 9999), ("sample_offset", 4),
                           ("nominal_uptime_seconds", 101.5), ("local_output_epoch", True)):
            with self.subTest(key=key, value=value):
                capture, pcm = response_fixture()
                capture["utterances"][1]["response_trigger"]["latest_retained_active_sample"][key] = value
                with self.assertRaises(EvidenceError):
                    analyze_capture(capture, pcm)

    def test_reported_intervals_must_match_actual_callback_and_gate_evidence(self):
        for field in ("input_first_successful_activity_uptime_seconds", "input_to_local_gate_close_seconds",
                      "last_retained_active_minus_input_seconds"):
            with self.subTest(field=field):
                capture, pcm = response_fixture()
                capture["response_stop_measurements"][0][field] += 1
                with self.assertRaises(EvidenceError):
                    analyze_capture(capture, pcm)

    def test_preclose_callback_admission_can_have_nominal_sample_tail_after_close(self):
        capture, pcm = response_fixture()
        close = capture["output_gates"][3]["requested_uptime_seconds"]
        block = capture["decoded_sample_map"][2]
        block.update(callback_start_uptime_seconds=close - 0.00002, callback_end_uptime_seconds=close - 0.00001)
        raw = capture["response_stop_measurements"][0]
        raw["last_retained_active_minus_input_seconds"] = block["callback_start_uptime_seconds"] + 2 / RATE - raw["input_first_successful_activity_uptime_seconds"]
        result = analyze_capture(capture, pcm)
        stop = result["occurrences"][1]["response_control_stop"]
        self.assertIsNotNone(stop)
        self.assertGreater(stop["last_retained_active_minus_input_seconds"], stop["input_to_local_gate_close_seconds"])

    def test_callback_returning_after_close_cannot_qualify_stop(self):
        capture, pcm = response_fixture()
        capture["decoded_sample_map"][2]["callback_end_uptime_seconds"] = 102.05
        result = analyze_capture(capture, pcm)
        self.assertIsNone(result["occurrences"][1]["response_control_stop"])
        self.assertIn("local_epoch_closed_before_admission_or_order_ambiguous",
                      result["response_control_evidence"]["stop_diagnostics"][0]["reasons"])

    def test_null_arrays_duplicate_stops_and_unknown_semantic_claims_rejected(self):
        for mode in ("local_interruptions", "response_stop_measurements", "duplicate", "semantic"):
            with self.subTest(mode=mode):
                capture, pcm = response_fixture()
                if mode in ("local_interruptions", "response_stop_measurements"):
                    capture[mode] = None
                elif mode == "duplicate":
                    capture["response_stop_measurements"].append(deepcopy(capture["response_stop_measurements"][0]))
                else:
                    capture["response_semantic_causality"] = "proven"
                with self.assertRaises(EvidenceError):
                    analyze_capture(capture, pcm)

    def test_exact_filters_and_enqueue_failure_remain_unmeasured(self):
        for mode in ("filter", "revalidation", "failure"):
            with self.subTest(mode=mode):
                capture, pcm = response_fixture()
                item = capture["utterances"][1]
                if mode == "filter":
                    item["trigger"]["response_id"] = "resp_other"
                elif mode == "revalidation":
                    item["response_trigger"]["enqueue_revalidated"] = False
                else:
                    item["response_trigger"]["failure_code"] = "enqueue_stale"
                self.assertEqual(analyze_capture(capture, pcm)["occurrences"][1]["response_trigger_verdict"], "unmeasured")


if __name__ == "__main__":
    unittest.main()
