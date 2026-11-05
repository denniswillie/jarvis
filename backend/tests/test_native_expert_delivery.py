"""Pure acceptance-oracle regressions; no model or database calls."""
from copy import deepcopy

import pytest

from evals.second_brain.native_expert_delivery import (
    delivery_checks, memory_case_for, memory_effect_checks, scenario_utterances,
)


def memory_sources():
    return {key: {"revision": 1, "state": "active", "trust": "user_input",
                  "content": {"text": key}, "original_content_retained": True}
            for key in ("cedar-v1", "cedar-v2")}


def test_memory_speech_uses_the_frozen_request_and_a_real_followup():
    case = memory_case_for("ambiguous_correction")
    utterances = scenario_utterances(case["id"])
    assert utterances[case["id"]] == case["utterance"]
    assert utterances["memory_clarification"] == "Version one only."
    assert memory_case_for("multitask") is None
    assert "memory_clarification" not in scenario_utterances("precise_correction")


def test_ambiguous_memory_cannot_mutate_before_clarification():
    before = memory_sources()
    case = memory_case_for("ambiguous_correction")
    assert all(memory_effect_checks(case, before, before, [], []).values())
    after = deepcopy(before)
    after["cedar-v1"]["revision"] = 2
    assert not memory_effect_checks(case, before, after, [], [])["untargeted_originals_unchanged"]


def test_correction_requires_one_real_accepted_worker_effect_and_keeps_other_source():
    from app.tasks.memory_corrections import MEMORY_CORRECTION_CAPABILITY
    before = memory_sources()
    after = deepcopy(before)
    after["cedar-v1"].update(revision=2, trust="derived")
    action = {"status": "accepted", "capability": MEMORY_CORRECTION_CAPABILITY}
    case = memory_case_for("precise_correction")
    assert all(memory_effect_checks(case, before, after, [action], []).values())
    for actions in ([], [action, action], [dict(action, status="submitting")]):
        assert not memory_effect_checks(case, before, after, actions, [])["exact_correction_effect_count"]
    after["cedar-v2"]["content"] = {"text": "Changed"}
    assert not memory_effect_checks(case, before, after, [action], [])["untargeted_originals_unchanged"]


def test_direct_foreground_replacement_is_not_frontier_acceptance():
    checks = memory_effect_checks(memory_case_for("ambiguous_correction"),
        memory_sources(), memory_sources(), [], [{"tool": "correct_memory_source"}])
    assert not checks["no_foreground_replacement"]


def test_clarified_memory_request_requires_the_target_correction():
    before = memory_sources()
    checks = memory_effect_checks(memory_case_for("ambiguous_correction"),
        before, before, [], [], clarified=True)
    assert not checks["exact_correction_effect_count"]
    assert not checks["one_versioned_replacement"]




def test_explicit_expert_case_is_visible_and_keeps_original_ambiguity():
    base = memory_case_for("ambiguous_correction")
    expert = memory_case_for("expert_ambiguous_correction")
    assert base["utterance"] in expert["utterance"]
    assert "delegate" in expert["utterance"]
    assert expert["sources"] == base["sources"]
    assert "memory_clarification" in scenario_utterances(expert["id"])
    assert all(memory_effect_checks(expert, memory_sources(), memory_sources(), [], []).values())


def test_source_reset_requires_exact_close_and_real_client_clear():
    from types import SimpleNamespace
    from uuid import uuid4
    from evals.second_brain.native_expert_delivery import completed_source_reset
    session, reset, audio = str(uuid4()), str(uuid4()), "fixture-audio"
    event = {"type": "session.context_invalidated", "session_id": session, "audio_session_id": audio,
        "context_reset": {"protocol_version": 1, "id": reset, "preserve_session": True}}
    phone = SimpleNamespace(events=[event, dict(event, type="session.closed")], client_events=[])
    assert completed_source_reset(phone, session_id=session, audio_session_id=audio) is None
    ack = {"event": {"type": "session.context_cleared", "reset_id": reset,
                     "audio_session_id": audio, "protocol_version": 1}}
    phone.client_events = [ack]
    assert completed_source_reset(phone, session_id=session, audio_session_id=audio) == reset
    assert completed_source_reset(phone, session_id=str(uuid4()), audio_session_id=audio) is None
    phone.client_events = [{"event": dict(ack["event"], audio_session_id="retired")}]
    assert completed_source_reset(phone, session_id=session, audio_session_id=audio) is None
    phone.client_events = [ack]
    phone.events = [dict(event, type="session.closed")]
    assert completed_source_reset(phone, session_id=session, audio_session_id=audio) is None


@pytest.mark.asyncio
async def test_conversation_never_reuses_a_resetting_transport():
    import asyncio
    from types import SimpleNamespace
    from evals.second_brain.native_expert_delivery import FixtureConversation
    conversation = FixtureConversation()
    job = asyncio.get_running_loop().create_future()
    old = SimpleNamespace(job=job, phone=SimpleNamespace(events=[{"type": "session.context_invalidated"}]))
    new = SimpleNamespace(job=job, phone=SimpleNamespace(events=[{"type": "session.created"}]))
    conversation.rows.append(old)
    conversation.current = old
    waiting = asyncio.create_task(conversation.ready())
    await asyncio.sleep(.01)
    assert not waiting.done()
    conversation.rows.append(new)
    conversation.current = new
    assert await asyncio.wait_for(waiting, 1) is new
    conversation.failure = RuntimeError("terminal connection failure")
    with pytest.raises(RuntimeError, match="terminal connection"):
        await conversation.ready()
    job.cancel()




def test_exception_diagnostics_omit_arbitrary_prose_and_local_values():
    import json
    from evals.second_brain.native_expert_delivery import exception_details
    secret = "do-not-copy-fixture-token"
    try:
        raise ValueError(secret)
    except ValueError as error:
        details = exception_details(error)
    assert details[0]["error_type"] == "ValueError"
    assert details[0]["frames"]
    assert secret not in json.dumps(details)


@pytest.mark.asyncio
async def test_failed_action_snapshot_does_not_hide_source_evidence():
    from unittest.mock import AsyncMock, patch
    from evals.second_brain.native_expert_delivery import memory_failure_snapshot
    module = "evals.second_brain.native_expert_delivery."
    sources = memory_sources()
    with patch(module + "source_snapshot", new=AsyncMock(return_value=sources)), \
         patch(module + "memory_actions", new=AsyncMock(side_effect=RuntimeError("fixture-query-failure"))):
        evidence = await memory_failure_snapshot(object(), {})
    assert evidence["after"] == sources
    assert "actions" not in evidence
    assert evidence["actions_error"][0]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_failed_source_snapshot_does_not_hide_action_receipts():
    from unittest.mock import AsyncMock, patch
    from evals.second_brain.native_expert_delivery import memory_failure_snapshot
    module = "evals.second_brain.native_expert_delivery."
    actions = [{"status": "outcome_unknown", "receipt": {"status": "failed"}}]
    with patch(module + "source_snapshot", new=AsyncMock(side_effect=RuntimeError("source-unavailable"))), \
         patch(module + "memory_actions", new=AsyncMock(return_value=actions)):
        evidence = await memory_failure_snapshot(object(), {})
    assert evidence["actions"] == actions
    assert "after" not in evidence
    assert evidence["after_error"][0]["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_memory_effect_snapshot_queries_the_actual_database_schema():
    import os
    from pathlib import Path
    from evals.second_brain.native_expert_delivery import memory_effects
    from evals.second_brain.source_fixture import source_fixture
    dsn, directory = (os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL"),
                      os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"))
    if not dsn or not directory:
        pytest.skip("Explicit manifested source fixture required")
    async with source_fixture(dsn, Path(directory), test_name="native-memory-effect-snapshot") as fixture:
        original = await fixture.publish("Synthetic retained statement.", namespace="fixture:effect-snapshot",
                                         source_key="original")
        after, actions = await memory_effects(fixture, {"original": original})
        assert after["original"]["revision"] == original.revision
        assert after["original"]["content"]["text"] == "Synthetic retained statement."
        assert actions == []




def test_expert_authorship_uses_the_actual_worker_not_missing_inbox_metadata():
    from evals.second_brain.native_expert_delivery import expert_authorship_matches
    row = dict(entry(), task_id="task-1", task_revision=2, expert_result={})
    event = {"owner_id": "fixture-owner", "task_id": "task-1", "task_revision": 2,
             "payload": {"authorship": {"kind": "expert", "model": "fixture-model",
                 "owner_id": "fixture-owner", "task_id": "task-1", "task_revision": 2},
                 "speech": [{"text": row["expected_texts"][0]}]}}
    assert expert_authorship_matches([row], [event], model="fixture-model", owner_id="fixture-owner")
    assert not expert_authorship_matches([row], [], model="fixture-model", owner_id="fixture-owner")
    for field, value in (("owner_id", "foreign"), ("task_id", "other"), ("task_revision", 1)):
        altered = deepcopy(event)
        altered["payload"]["authorship"][field] = value
        assert not expert_authorship_matches([row], [altered], model="fixture-model", owner_id="fixture-owner")
    altered = deepcopy(event)
    altered["payload"]["speech"][0]["text"] = "Unrelated expert result."
    assert not expert_authorship_matches([row], [altered], model="fixture-model", owner_id="fixture-owner")
    assert not expert_authorship_matches([row], [event], model="other-model", owner_id="fixture-owner")


def test_configured_model_and_ingestion_events_do_not_prove_frontier_reasoning():
    from evals.second_brain.native_expert_delivery import authored_worker_results
    assert list(authored_worker_results([
        {"phase": "delegated-memory-seed", "status": "searchable"},
        {"task_id": "task-1", "started_seconds": 1, "model": "fixture-model"},
    ], model="fixture-model")) == []


def entry():
    return {
        "state": "played", "attempt_count": 1,
        "expected_texts": ["Open parenthesis low plus high close parenthesis."],
        "actual_texts": ["Open parenthesis low plus high close parenthesis."],
        "chunks": [{"chunk_index": 0, "completed": True, "generated": True,
                    "forwarded_samples": 2400, "played_samples": 2400}],
    }


def test_complete_exact_expert_delivery():
    assert all(delivery_checks([entry(), entry()], expected=2).values())


def test_empty_delivery_never_passes():
    assert not any(delivery_checks([], expected=1).values())


def test_rephrased_foreground_text_is_not_expert_delivery():
    row = entry()
    row["actual_texts"] = ["Just add low and high."]
    assert not delivery_checks([row], expected=1)["exact_expert_text"]


def test_missing_receipt_and_duplicate_attempt_are_not_acceptance():
    row = entry()
    row["attempt_count"] = 2
    row["chunks"][0]["played_samples"] = 1200
    checks = delivery_checks([row], expected=1)
    assert not checks["one_attempt_per_result"]
    assert not checks["complete_chunk_receipts"]

@pytest.mark.asyncio
async def test_snapshot_uses_the_actual_hex_tts_identity():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from uuid import uuid4
    from evals.second_brain.native_expert_delivery import snapshot

    attempt, intent = uuid4(), uuid4()

    class Connection:
        async def fetch(self, query, owner, selected_attempt=None):
            if "jarvis_attention_intents" in query:
                return [{"id": intent}]
            return [{"attempt_id": attempt, **entry()["chunks"][0]}]

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Connection()

    async def tasks_list(owner, limit):
        return []

    async def detail(owner, identity):
        assert identity == intent
        return {"state": "played", "attempts": [{"id": attempt}],
                "expert_result": {"speech": [{"text": entry()["expected_texts"][0]}]}}

    phone = SimpleNamespace(events=[{"type": "transcript.done",
        "response_id": "tts_" + attempt.hex, "text": entry()["actual_texts"][0]}])
    _, entries = await snapshot(SimpleNamespace(list=tasks_list),
        SimpleNamespace(pool=Pool(), detail=detail), "fixture-owner", phone)
    assert all(delivery_checks(entries, expected=1).values())

def test_history_requires_native_acknowledgments_of_exact_text():
    from evals.second_brain.native_expert_delivery import HISTORY_LABEL, history_matches
    row = entry()
    item = {"id": "a" * 32, "texts": [HISTORY_LABEL + row["expected_texts"][0]]}
    assert history_matches([row], [item])
    assert not history_matches([row], [])
    assert not history_matches([row], [dict(item, id="a" * 41)])
    assert not history_matches([row], [dict(item, texts=[HISTORY_LABEL + "Different words."])])
    assert not history_matches([row, row], [item, item])




def test_missing_or_out_of_order_chunk_is_not_complete():
    for chunks in ([], [dict(entry()["chunks"][0], chunk_index=1)]):
        row = entry()
        row["chunks"] = chunks
        assert not delivery_checks([row], expected=1)["complete_chunk_receipts"]


def test_generation_without_playback_is_not_complete():
    row = deepcopy(entry())
    row["state"] = "awaiting_receipt"
    row["chunks"][0]["completed"] = False
    checks = delivery_checks([row], expected=1)
    assert not checks["all_played"]
    assert not checks["complete_chunk_receipts"]

@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_present,wrong_attempt", [(True, False), (False, False), (True, True)])
async def test_snapshot_completion_race_keeps_the_real_receipt_gate(receipt_present, wrong_attempt):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from uuid import uuid4
    from evals.second_brain.native_expert_delivery import snapshot

    intent, old_attempt, attempt = uuid4(), uuid4(), uuid4()
    committed = False
    reads = []

    class Connection:
        async def fetch(self, query, owner, selected_attempt=None):
            assert owner == "fixture-owner"
            if "jarvis_attention_intents" in query:
                return [{"id": intent}]
            reads.append(("chunks", selected_attempt))
            # Simulate the receipt committing between the old harness's chunk
            # query and detail query. No sleeps or scheduler luck are needed.
            complete = committed and receipt_present
            return [dict(entry()["chunks"][0],
                attempt_id=old_attempt if wrong_attempt else attempt,
                completed=complete, played_samples=2400 if complete else 1200)]

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Connection()

    async def tasks_list(owner, **kwargs):
        return []

    async def detail(owner, identity):
        nonlocal committed
        assert identity == intent
        reads.append(("detail", identity))
        committed = True
        return {"state": "played", "attempts": [{"id": old_attempt}, {"id": attempt}],
                "expert_result": {"speech": [{"text": entry()["expected_texts"][0]}]}}

    phone = SimpleNamespace(events=[{"type": "transcript.done",
        "response_id": "tts_" + attempt.hex, "text": entry()["actual_texts"][0]}])
    _, entries = await snapshot(SimpleNamespace(list=tasks_list),
        SimpleNamespace(pool=Pool(), detail=detail), "fixture-owner", phone)
    assert reads == [("detail", intent), ("chunks", attempt)]
    checks = delivery_checks(entries, expected=1)
    assert checks["complete_chunk_receipts"] is (receipt_present and not wrong_attempt)
    assert not checks["one_attempt_per_result"]  # Historical interrupted attempts stay visible.
    assert entries[0]["attempt_count"] == 2


@pytest.mark.asyncio
async def test_snapshot_without_an_attempt_does_not_read_unrelated_receipts():
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from uuid import uuid4
    from evals.second_brain.native_expert_delivery import snapshot

    intent = uuid4()

    class Connection:
        async def fetch(self, query, owner):
            assert "jarvis_attention_intents" in query
            return [{"id": intent}]

    class Pool:
        @asynccontextmanager
        async def acquire(self):
            yield Connection()

    async def tasks_list(owner, **kwargs):
        return []

    async def detail(owner, identity):
        return {"state": "pending", "attempts": [], "expert_result": {}}

    _, entries = await snapshot(SimpleNamespace(list=tasks_list),
        SimpleNamespace(pool=Pool(), detail=detail), "fixture-owner", SimpleNamespace(events=[]))
    assert entries[0]["chunks"] == []
    checks = delivery_checks(entries, expected=1)
    # No attempted delivery is rejected by the overall gate. Comparing two
    # empty chunk lists is not evidence of playback, nor a failed chunk receipt.
    assert not checks["all_played"]
    assert not checks["one_attempt_per_result"]
    assert not all(checks.values())
