from copy import deepcopy
from uuid import uuid4

import pytest

from evals.responsive_voice.native_capture_evidence import EvidenceError, analyze_capture
from evals.responsive_voice.test_native_capture_evidence import fixture, response_fixture


BASIS = "authorized_response_and_epoch_at_callback_entry_and_return_not_packet_item_or_phase_attribution"


def add_associations(capture):
    connection = str(uuid4())
    capture.update(response_association_basis=BASIS, response_uncorrelated_decoded_blocks=0)
    for collection in (capture["decoded_sample_map"], capture["output_gates"]):
        for raw in collection:
            full = raw.get("response_identity")
            raw["response_association"] = (
                {key: full[key] for key in ("connection_id", "response_id", "local_output_epoch")}
                if full is not None else
                {"connection_id": connection, "response_id": f"resp_{raw['local_output_epoch']}",
                 "local_output_epoch": raw["local_output_epoch"]})


def tail(status="timeout"):
    return {"schema_version": 1, "status": status,
            "media_end_scheduled_uptime_seconds": 101.0,
            "media_end_observed_uptime_seconds": 101.001,
            "capture_deadline_uptime_seconds": 121.0,
            "finished_uptime_seconds": 121.001,
            "required_inputs_complete": True, "response_association": None,
            "generation_done_uptime_seconds": None, "natural_buffer_stop_uptime_seconds": None,
            "authorized_quiet_start_uptime_seconds": None,
            "authorized_quiet_callback_end_uptime_seconds": None,
            "authorized_quiet_blocks": 0, "required_quiet_seconds": 0.5,
            "maximum_callback_gap_seconds": 0.03,
            "decoded_drain_confirmed": False, "physical_playback_measured": False}


def test_legacy_capture_stays_unqualified_for_unrecorded_observations():
    capture, pcm, _ = fixture()
    result = analyze_capture(capture, pcm)
    assert result["capture_observations"]["response_association"]["status"] == "not_recorded"
    assert result["capture_observations"]["post_media_capture"]["status"] == "not_recorded"
    assert result["actual_cuts"] is None


def test_response_only_associations_do_not_manufacture_item_or_useful_speech():
    capture, pcm, _ = fixture()
    add_associations(capture)
    original = deepcopy(capture)
    result = analyze_capture(capture, pcm)
    observed = result["capture_observations"]["response_association"]
    assert observed["associated_blocks"] == 2
    assert observed["item_unknown_associated_blocks"] == 2
    assert observed["strong_trigger_authority_added"] is False
    assert observed["item_or_phase_attribution_established"] is False
    assert result["first_decoded_onset"]["useful_speech_established"] is False
    assert result["provider_response_attribution"] == "unavailable"
    assert capture == original


@pytest.mark.parametrize("change", ["epoch", "phase", "counter", "foreign_item", "basis"])
def test_malformed_associations_are_rejected(change):
    capture, pcm, _ = fixture()
    add_associations(capture)
    first = capture["decoded_sample_map"][0]
    if change == "epoch":
        first["response_association"]["local_output_epoch"] += 1
    elif change == "phase":
        first["response_association"]["phase"] = "final_answer"
    elif change == "counter":
        capture["response_uncorrelated_decoded_blocks"] = 1
    elif change == "foreign_item":
        first["response_identity"] = {**first["response_association"], "response_id": "resp_foreign", "item_id": "item_foreign"}
    else:
        capture["response_association_basis"] = "provider_packet_identity"
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm)


def test_weaker_association_cannot_rescue_missing_strong_interruption_identity():
    capture, pcm = response_fixture()
    add_associations(capture)
    capture["decoded_sample_map"][1]["response_identity"] = None
    capture["uncorrelated_decoded_blocks"] = 1
    result = analyze_capture(capture, pcm)
    assert result["occurrences"][1]["response_trigger_verdict"] == "unmeasured"
    assert result["occurrences"][1]["response_control_stop"] is None
    assert result["actual_cuts"] is None


def test_extra_recording_time_does_not_change_useful_answer_metrics_or_claim_playback():
    capture, pcm, labels = fixture()
    original = analyze_capture(capture, pcm, labels)
    capture["post_media_capture"] = tail()
    result = analyze_capture(capture, pcm, labels)
    assert result["occurrences"] == original["occurrences"]
    observed = result["capture_observations"]["post_media_capture"]
    assert observed["recorded_status"] == "timeout"
    assert observed["full_answer_delivery_established"] is False
    assert observed["decoded_quiet_independently_verified"] is False
    assert result["hearing_measured"] is False


@pytest.mark.parametrize("change", ["drain", "hearing", "duration", "reversed", "quiet", "status"])
def test_invalid_tail_claims_fail_closed(change):
    capture, pcm, _ = fixture()
    value = capture["post_media_capture"] = tail()
    if change == "drain":
        value["decoded_drain_confirmed"] = True
    elif change == "hearing":
        value["physical_playback_measured"] = True
    elif change == "duration":
        value["capture_deadline_uptime_seconds"] = 122
    elif change == "reversed":
        value["finished_uptime_seconds"] = 100
    elif change == "quiet":
        value["status"] = "decoded_quiet_observed"
    else:
        value["status"] = "heard"
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm)
