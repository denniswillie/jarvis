"""Offline control-file consistency, not SQL/claim/semantic/native acceptance."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import UUID

import pytest

from evals.responsive_voice.native_capture_evidence import EvidenceError, analyze_capture
from evals.responsive_voice.test_native_capture_evidence import fixture, response_fixture
from test_native_task_barrier import NOW, case as canonical_case, observer, observe


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def task_fixture(cuts=1):
    capture, pcm, _ = fixture(2 + 2 * cuts)
    scope, witness, row = canonical_case.__wrapped__()
    # Keep the canonical fake fixture's run/schema/owner/session together.
    # Rebind only the synthetic capture identity, never an isolated scope's
    # run UUID independently of its run-derived schema.
    capture["run_id"] = str(scope.run_id)
    for ordinal, item in enumerate(capture["utterances"]):
        identity = f"{scope.run_id.hex}/utterance/{ordinal}"
        item.update(occurrence_id=identity, trigger_id=identity + "/trigger")
    anchor = capture["clock_anchor"]
    anchor["utc_unix_seconds"] = NOW.timestamp()
    anchor["utc_iso8601"] = NOW.isoformat()
    utc_base = datetime.fromtimestamp(anchor["utc_unix_seconds"], timezone.utc)
    scope = scope.model_copy(update={"host_expires_at": utc_base + timedelta(seconds=100)})
    raw_scope = scope.model_dump(mode="json")
    raw_scope.pop("session_id")
    raw_scope["format"] = "jarvis.native_task_scope.v1"
    inputs = capture["utterances"]
    manifest = dict(version=1, run_id=scope.run_id.hex, owner_id=scope.owner_id, voice_route="realtime",
        task_barrier=dict(version=1, scope_file="task-scope.json", request_input_ordinal=0), utterances=[])
    receipts, stimulus_inputs = [], []
    for ordinal, item in enumerate(inputs):
        if ordinal == 0:
            trigger = dict(kind="at_time", at_seconds=7.0)
        elif ordinal == 1:
            trigger = dict(kind="after_task_claim", after_input_ordinal=0, timeout_seconds=30.0)
        elif ordinal % 2 == 0:
            trigger = dict(kind="response_output_start", after_input_ordinal=ordinal - 1,
                delay_seconds=0.3, timeout_seconds=10.0, max_pcm_age_seconds=0.05, require_task_claim=True)
        else:
            trigger = dict(kind="after_input_complete", after_input_ordinal=ordinal - 1,
                delay_seconds=1.0, timeout_seconds=10.0)
        item["trigger"] = trigger
        manifest["utterances"].append(dict(clip_id=item["clip_id"], trigger=trigger))
        sequence = None
        if ordinal == 1 or ordinal > 1 and ordinal % 2 == 0:
            enqueue = utc_base + timedelta(seconds=item["enqueue_requested_uptime_seconds"] - anchor["monotonic_reference_seconds"])
            sequence = len(receipts) + 1
            # Existing canonical fake-pool fixture generates the exact observer
            # schema. No pool connection or SQL is executed by this test.
            subject, _ = observer((scope, witness, row), initial_sequence=sequence - 1)
            receipt = observe(subject)
            receipt = receipt.model_copy(update={"observed_at": enqueue - timedelta(seconds=0.1),
                "expires_at": enqueue + timedelta(seconds=1), "scope": scope,
                "claim": receipt.claim.model_copy(update={"lease_until": enqueue + timedelta(seconds=10),
                    "worker_expires_at": enqueue + timedelta(seconds=10)})})
            receipts.append(receipt.model_dump_json().encode())
        stimulus_inputs.append(dict(input_ordinal=ordinal, status="sdk_input_complete", sample_count=item["sample_count"],
            accepted_samples=item["sdk_accepted_clip_samples"],
            first_sdk_callback_monotonic=item["first_sdk_callback_start_uptime_seconds"],
            last_sdk_callback_end_monotonic=item["last_sdk_callback_end_uptime_seconds"],
            task_claim_sequence_at_enqueue=sequence))
    manifest_bytes, scope_bytes = encoded(manifest), encoded(raw_scope)
    common = dict(run_id=str(scope.run_id), barrier_id=str(scope.barrier_id), owner_id=scope.owner_id,
        session_id=str(scope.session_id), manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        scope_file_sha256=hashlib.sha256(scope_bytes).hexdigest(), recorded_at=utc_base.isoformat())
    binding = dict(format="jarvis.native_task_session_binding.v1", **common)
    stimulus = dict(format="jarvis.native_task_stimulus_completion.v1", **common, inputs=stimulus_inputs)
    stimulus["recorded_at"] = (utc_base + timedelta(seconds=30)).isoformat()
    documents = dict(manifest_bytes=manifest_bytes, scope_bytes=scope_bytes, session_binding_bytes=encoded(binding),
        stimulus_completion_bytes=encoded(stimulus), barrier_receipts=receipts)
    return capture, pcm, documents


def change(documents, key, edit):
    value = json.loads(documents[key])
    edit(value)
    documents[key] = encoded(value)


def test_fixture_run_schema_owner_and_session_share_canonical_binding():
    capture, _, docs = task_fixture()
    scope = json.loads(docs["scope_bytes"])
    binding = json.loads(docs["session_binding_bytes"])
    run = UUID(capture["run_id"])
    assert UUID(scope["run_id"]) == run
    assert scope["database_schema"] == f"jarvis_sb_eval_{run.hex}_0"
    for raw in docs["barrier_receipts"]:
        receipt = json.loads(raw)
        assert receipt["scope"]["database_schema"] == scope["database_schema"]
        assert receipt["scope"]["owner_id"] == receipt["admission"]["owner_id"] == binding["owner_id"]
        assert UUID(receipt["scope"]["session_id"]) == UUID(receipt["admission"]["session_id"]) == UUID(binding["session_id"])


def test_foreign_schema_rejected_even_with_consistent_file_hashes():
    capture, pcm, docs = task_fixture()
    change(docs, "scope_bytes", lambda value: value.update(database_schema="jarvis_sb_eval_00000000000040008000000000000000_0"))
    new_hash = hashlib.sha256(docs["scope_bytes"]).hexdigest()
    for key in ("session_binding_bytes", "stimulus_completion_bytes"):
        change(docs, key, lambda value: value.update(scope_file_sha256=new_hash))
    with pytest.raises(EvidenceError, match="invalid_task_scope"):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("cuts", [1, 2, 3])
def test_full_control_metadata_keeps_semantic_cuts_and_continuity_unknown(cuts):
    capture, pcm, docs = task_fixture(cuts)
    result = analyze_capture(capture, pcm, task_evidence=docs)
    controls = result["task_control_evidence"]
    assert controls["verdict"] == "validated_control_metadata"
    assert controls["validated_claim_inputs"] == cuts + 1
    assert controls["task_continuity"] == "unknown" and controls["canonical_postrun_required"]
    assert result["actual_cuts"] is None and result["provider_response_attribution"] == "unavailable"
    assert not result["hearing_measured"] and not result["answer_quality_judged"]
    assert result["occurrences"][3]["task_trigger_verdict"] == "validated_input_completion"


def test_missing_scope_or_stimulus_and_old_capture_remain_honest():
    capture, pcm, docs = task_fixture()
    assert analyze_capture(capture, pcm)["task_control_evidence"]["verdict"] == "unmeasured"
    for key in ("scope_bytes", "session_binding_bytes", "stimulus_completion_bytes"):
        partial = {name: value for name, value in docs.items() if name != key}
        assert analyze_capture(capture, pcm, task_evidence=partial)["task_control_evidence"]["verdict"] == "unmeasured"
    old, pcm, _ = fixture()
    assert analyze_capture(old, pcm)["task_control_evidence"]["verdict"] == "not_recorded"


@pytest.mark.parametrize("field,value", [("session_id", "00000000-0000-4000-8000-000000000000"),
    ("owner_id", "foreign"), ("manifest_file_sha256", "0" * 64), ("scope_file_sha256", "0" * 64),
    ("run_id", "00000000-0000-4000-8000-000000000000")])
def test_foreign_session_owner_hash_or_run_is_rejected(field, value):
    capture, pcm, docs = task_fixture()
    change(docs, "stimulus_completion_bytes", lambda d: d.update({field: value}))
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "bad_count", "time_units", "sequence", "unknown_key", "failed"])
def test_completion_requires_exact_input_coverage_counts_time_and_claim(mutation):
    capture, pcm, docs = task_fixture()
    def edit(d):
        if mutation == "duplicate": d["inputs"][1] = deepcopy(d["inputs"][0])
        if mutation == "missing": d["inputs"].pop()
        if mutation == "bad_count": d["inputs"][1]["accepted_samples"] -= 1
        if mutation == "time_units": d["inputs"][1]["last_sdk_callback_end_monotonic"] *= 1000
        if mutation == "sequence": d["inputs"][1]["task_claim_sequence_at_enqueue"] = None
        if mutation == "unknown_key": d["inputs"][1]["assistant_said_task_active"] = True
        if mutation == "failed": d["inputs"][1]["status"] = "failed"
    change(docs, "stimulus_completion_bytes", edit)
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("mode", ["replay", "foreign_worker", "different_attempt", "malformed"])
def test_receipts_are_pinned_and_replay_strict(mode):
    capture, pcm, docs = task_fixture()
    if mode == "replay":
        docs["barrier_receipts"].insert(1, docs["barrier_receipts"][0])
    elif mode == "malformed":
        docs["barrier_receipts"][0] = b'{"secret":"not a receipt"}'
    else:
        value = json.loads(docs["barrier_receipts"][-1])
        if mode == "foreign_worker": value["scope"]["worker_id"] = "foreign"
        if mode == "different_attempt": value["claim"]["attempt_id"] = "00000000-0000-4000-8000-000000000000"
        docs["barrier_receipts"][-1] = encoded(value)
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm, task_evidence=docs)


def test_expired_claim_is_unmeasured_not_a_runtime_pass():
    capture, pcm, docs = task_fixture()
    value = json.loads(docs["barrier_receipts"][0])
    for key in ("observed_at", "expires_at"):
        value[key] = (datetime.fromisoformat(value[key].replace("Z", "+00:00")) - timedelta(seconds=3)).isoformat()
    docs["barrier_receipts"][0] = encoded(value)
    result = analyze_capture(capture, pcm, task_evidence=docs)
    assert result["task_control_evidence"]["verdict"] == "unmeasured"
    assert "task_claim_freshness_unconfirmed" in result["task_control_evidence"]["blockers"]


@pytest.mark.parametrize("mode", ["truncated", "zero_pcm", "silent"])
def test_missing_audio_or_incomplete_mapping_never_measures_actual_cuts(mode):
    capture, pcm, docs = task_fixture()
    if mode == "truncated": capture["input_mapping_truncated"] = True
    if mode == "zero_pcm":
        capture["decoded_sample_map"] = []
        capture["decoded_pcm"].update(sample_count=0, byte_count=0)
        pcm = b""
    if mode == "silent":
        capture["utterances"][0].update(all_samples_zero=True, source_first_active_sample=None,
            source_last_active_sample_exclusive=None, first_successful_activity_uptime_seconds=None,
            last_successful_activity_uptime_seconds=None)
    result = analyze_capture(capture, pcm, task_evidence=docs)
    assert result["actual_cuts"] is None and not result["hearing_measured"]
    if mode != "zero_pcm": assert result["task_control_evidence"]["verdict"] == "unmeasured"
    if mode == "zero_pcm": assert result["first_decoded_onset"]["sample"] is None


@pytest.mark.parametrize("field,value", [("after_input_ordinal", 3), ("delay_seconds", -1),
                                        ("delay_seconds", 11), ("timeout_seconds", 121)])
def test_input_completion_trigger_rejects_invalid_anchor_or_deadline(field, value):
    capture, pcm, _ = task_fixture()
    capture["utterances"][3]["trigger"][field] = value
    with pytest.raises(EvidenceError): analyze_capture(capture, pcm)


def test_followup_cannot_use_nominal_time_before_actual_previous_sdk_end():
    capture, pcm, _ = task_fixture()
    capture["utterances"][3]["enqueue_requested_uptime_seconds"] = capture["utterances"][2]["last_sdk_callback_end_uptime_seconds"] + 0.5
    with pytest.raises(EvidenceError, match="task_trigger_outside_sdk_completion_window"):
        analyze_capture(capture, pcm)


def test_response_control_cannot_pass_missing_required_task_claim():
    values = response_fixture()
    capture, pcm = values[:2]
    for item in capture["utterances"]:
        if item["trigger"]["kind"] == "response_output_start":
            item["trigger"]["require_task_claim"] = True
    result = analyze_capture(capture, pcm)
    for item in result["occurrences"]:
        if item["response_trigger_verdict"] != "not_applicable":
            assert item["response_trigger_verdict"] != "validated_control_binding"
            assert item["response_control_stop"] is None


def test_claim_flag_is_strict_and_only_valid_on_response_trigger():
    capture, pcm, _ = fixture()
    capture["utterances"][0]["trigger"]["require_task_claim"] = True
    with pytest.raises(EvidenceError, match="task_claim_flag_on_unrelated_trigger"):
        analyze_capture(capture, pcm)


def update_manifest(documents, edit):
    change(documents, "manifest_bytes", edit)
    digest = hashlib.sha256(documents["manifest_bytes"]).hexdigest()
    for key in ("session_binding_bytes", "stimulus_completion_bytes"):
        if key in documents:
            change(documents, key, lambda value: value.update(manifest_file_sha256=digest))


@pytest.mark.parametrize("field", ["connection_id", "response_id"])
@pytest.mark.parametrize("side", ["capture", "manifest"])
def test_only_optional_response_filter_null_and_omitted_are_equivalent(field, side):
    capture, pcm, docs = task_fixture()
    if side == "capture":
        capture["utterances"][2]["trigger"][field] = None
    else:
        update_manifest(docs, lambda value: value["utterances"][2]["trigger"].update({field: None}))
    before_capture, before_documents = deepcopy(capture), deepcopy(docs)
    result = analyze_capture(capture, pcm, task_evidence=docs)
    assert result["task_control_evidence"]["verdict"] == "validated_control_metadata"
    assert capture == before_capture and docs == before_documents
    assert result["actual_cuts"] is None


@pytest.mark.parametrize("field,first,second", [
    ("response_id", "resp_first", "resp_other"),
    ("connection_id", "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"),
])
@pytest.mark.parametrize("mode", ["absent", "null", "different"])
def test_nonnull_response_filters_are_not_normalized_away(field, first, second, mode):
    capture, pcm, docs = task_fixture()
    capture["utterances"][2]["trigger"][field] = first
    if mode != "absent":
        update_manifest(docs, lambda value: value["utterances"][2]["trigger"].update(
            {field: None if mode == "null" else second}))
    with pytest.raises(EvidenceError, match="task_manifest_input_mismatch"):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("side", ["capture", "manifest", "both"])
def test_unknown_null_trigger_field_is_rejected_even_if_both_sides_match(side):
    capture, pcm, docs = task_fixture()
    if side in ("capture", "both"):
        capture["utterances"][2]["trigger"]["unknown_filter"] = None
    if side in ("manifest", "both"):
        update_manifest(docs, lambda value: value["utterances"][2]["trigger"].update(unknown_filter=None))
    with pytest.raises(EvidenceError, match="invalid_task_trigger_fields"):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("field,value", [("response_id", "resp_exact"),
    ("connection_id", "11111111-1111-4111-8111-111111111111")])
def test_matching_nonnull_optional_filters_are_preserved(field, value):
    capture, pcm, docs = task_fixture()
    capture["utterances"][2]["trigger"][field] = value
    update_manifest(docs, lambda document: document["utterances"][2]["trigger"].update({field: value}))
    result = analyze_capture(capture, pcm, task_evidence=docs)
    assert result["task_control_evidence"]["verdict"] == "validated_control_metadata"
    assert capture["utterances"][2]["trigger"][field] == value


@pytest.mark.parametrize("field", ["connection_id", "response_id"])
def test_response_filter_null_is_not_accepted_on_another_trigger_kind(field):
    capture, pcm, docs = task_fixture()
    capture["utterances"][0]["trigger"][field] = None
    update_manifest(docs, lambda document: document["utterances"][0]["trigger"].update({field: None}))
    with pytest.raises(EvidenceError, match="invalid_task_trigger_fields"):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("field,value", [("delay_seconds", 0.4), ("timeout_seconds", 9.0),
    ("max_pcm_age_seconds", 0.04), ("after_input_ordinal", 0), ("require_task_claim", False)])
def test_other_response_trigger_differences_remain_rejected(field, value):
    capture, pcm, docs = task_fixture()
    capture["utterances"][2]["trigger"][field] = value
    with pytest.raises(EvidenceError, match="task_manifest_input_mismatch"):
        analyze_capture(capture, pcm, task_evidence=docs)


@pytest.mark.parametrize("field,value", [("response_id", 4), ("connection_id", "not-a-uuid"),
                                      ("delay_seconds", True), ("require_task_claim", 1)])
def test_optional_filter_support_keeps_typed_validation(field, value):
    capture, pcm, docs = task_fixture()
    capture["utterances"][2]["trigger"][field] = value
    update_manifest(docs, lambda document: document["utterances"][2]["trigger"].update({field: value}))
    with pytest.raises(EvidenceError):
        analyze_capture(capture, pcm, task_evidence=docs)


def test_null_response_fields_do_not_turn_missing_stimulus_or_unstarted_inputs_into_success():
    capture, pcm, docs = task_fixture()
    capture["utterances"][2]["trigger"].update(connection_id=None, response_id=None)
    for ordinal in (1, 2, 3):
        capture["utterances"][ordinal].update(status="failed" if ordinal == 1 else "not_started",
            failure_code="trigger_outputWaitExpired" if ordinal == 1 else "run_closed_before_injection",
            attempted_clip_samples=0, sdk_accepted_clip_samples=0, callbacks=[],
            enqueue_requested_uptime_seconds=None, first_sdk_callback_start_uptime_seconds=None,
            last_sdk_callback_end_uptime_seconds=None, first_successful_activity_uptime_seconds=None,
            last_successful_activity_uptime_seconds=None)
    docs.pop("stimulus_completion_bytes")
    result = analyze_capture(capture, pcm, task_evidence=docs)
    assert result["occurrences"][0]["input_verdict"] == "sdk_completed_active_input"
    assert all(item["input_verdict"] != "sdk_completed_active_input" for item in result["occurrences"][1:])
    assert result["task_control_evidence"]["verdict"] == "unmeasured"
    assert result["task_control_evidence"]["blockers"] == ["task_stimulus_completion_missing"]
    assert result["task_control_evidence"]["validated_claim_inputs"] == 0
    assert result["actual_cuts"] is None and not result["hearing_measured"]
