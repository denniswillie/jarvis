"""Frozen recovery-oracle cases; no native model, database or phone requests."""
from copy import deepcopy
from uuid import uuid4, uuid5

from evals.second_brain.native_expert_delivery import history_label
from evals.second_brain.native_expert_recovery import final_history, recovery_checks


def scenario():
    entries = []
    for ordinal, namespace in enumerate(("task_result", "task_result", "reminder_event")):
        attempts = [{"id": uuid4()} for _ in range(2 if ordinal == 0 else 1)]
        entries.append({
            "intent_id": str(uuid4()), "state": "played", "source_namespace": namespace,
            "attempt_count": len(attempts), "attempts": attempts,
            "expected_texts": [f"Exact script {ordinal}."], "actual_texts": [f"Exact script {ordinal}."],
            "chunks": [{"chunk_index": 0, "generated": True, "completed": True,
                        "forwarded_samples": 2400, "played_samples": 2400}],
        })
    first = entries[0]["intent_id"]
    result = {
        "accepted_task_ids": ["a", "b"],
        "tasks": [
            {"id": "a", "revision": 2, "state": "succeeded", "result_payload_id": "result-a"},
            {"id": "b", "revision": 1, "state": "succeeded", "result_payload_id": "result-b"}],
        "tasks_before_interrupt": [
            {"id": "a", "revision": 2, "state": "succeeded", "result_payload_id": "result-a"},
            {"id": "b", "revision": 1, "state": "running", "result_payload_id": None}],
        "interrupted_intent": first, "interrupt_state": "unknown", "client_interruption_receipt": True,
        "explicit_replays": [first], "interrupted_state_after_reconnect": "unknown",
        "before_replay": [{"intent_id": row["intent_id"], "attempt_count": 1,
                          "state": "unknown" if i == 0 else "played"} for i, row in enumerate(entries)],
        "connections": [{"session_id": "session", "audio_session_id": "audio-a"},
                        {"session_id": "session", "audio_session_id": "audio-b"}],
        "capture_flushed": [True, True], "application_errors": [], "status_response_texts": ["Current task status."],
        "native_history_items": [],
    }
    for row in entries:
        namespace = "expert" if row["source_namespace"] == "task_result" else "notification"
        result["native_history_items"].append({
            "id": uuid5(row["attempts"][-1]["id"], f"{namespace}:0").hex,
            "texts": [history_label(row["source_namespace"]) + row["expected_texts"][0]]})
    return result, entries


def test_exact_explicit_replay_and_mixed_delivery_pass():
    result, entries = scenario()
    checks = recovery_checks(result, entries)
    assert all(checks.values()), checks
    assert "one_attempt_per_result" not in checks


def test_empty_evidence_cannot_establish_recovery():
    assert not all(recovery_checks({}, []).values())


def test_automatic_replay_is_not_explicit_recovery():
    result, entries = scenario()
    result["interrupted_state_after_reconnect"] = "played"
    assert not recovery_checks(result, entries)["unknown_not_automatically_replayed"]
    result["explicit_replays"] = []
    assert not recovery_checks(result, entries)["explicit_replay_only"]


def test_unrequested_third_attempt_is_rejected():
    result, entries = scenario()
    entries[0]["attempt_count"] = 3
    assert not recovery_checks(result, entries)["only_authorized_extra_attempts"]


def test_unrequested_retry_before_replay_is_not_hidden_by_baseline_counts():
    result, entries = scenario()
    result["before_replay"][0]["attempt_count"] = 2
    entries[0]["attempt_count"] = 3
    assert not recovery_checks(result, entries)["only_authorized_extra_attempts"]


def test_duplicate_task_or_missing_revision_is_not_continuity():
    result, entries = scenario()
    result["tasks"][0]["id"] = "replacement-task"
    assert not recovery_checks(result, entries)["same_accepted_tasks"]
    result, entries = scenario()
    result["tasks"][0]["revision"] = 1
    assert not recovery_checks(result, entries)["exact_user_revision"]


def test_changed_completed_result_is_not_speech_only_interruption():
    result, entries = scenario()
    result["tasks"][0]["result_payload_id"] = "new-result"
    assert not recovery_checks(result, entries)["completed_result_unchanged"]


def test_old_history_does_not_substitute_for_new_attempt_acknowledgment():
    result, entries = scenario()
    old = {"id": uuid5(entries[0]["attempts"][0]["id"], "expert:0").hex,
           "texts": result["native_history_items"][0]["texts"]}
    result["native_history_items"].append(old)
    assert len(final_history(entries, result["native_history_items"])) == 3
    result["native_history_items"].pop(0)
    assert not recovery_checks(result, entries)["provider_accepted_final_history"]


def test_reminder_history_is_not_mislabeled_as_expert_output():
    result, entries = scenario()
    result["native_history_items"][-1]["texts"] = [
        history_label("task_result") + entries[-1]["expected_texts"][0]]
    assert not recovery_checks(result, entries)["provider_accepted_final_history"]


def test_missing_receipt_or_connection_rotation_fails():
    result, entries = scenario()
    entries[0]["chunks"][0]["completed"] = False
    assert not recovery_checks(result, entries)["complete_chunk_receipts"]
    result["connections"][1]["audio_session_id"] = "audio-a"
    assert not recovery_checks(result, entries)["same_conversation_new_audio_connection"]


def test_superseded_source_cannot_remain_available():
    result, entries = scenario()
    result["superseded_deliveries"] = [{"source_available": True}]
    assert not recovery_checks(result, entries)["superseded_results_unavailable"]


def test_every_script_and_receipt_is_required():
    result, entries = scenario()
    altered = deepcopy(entries)
    altered[-1]["actual_texts"] = ["A different reminder."]
    assert not recovery_checks(result, altered)["exact_expert_text"]
