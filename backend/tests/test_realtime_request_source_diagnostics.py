"""Controlled real wire/capture/dispatch composition, not native or model choice."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.realtime.request_source_diagnostics import (
    LOGGER, MAX_RECORDS, close_request_source_diagnostics, record_request_source,
)
from test_realtime_unwritten_followup_wire import (
    CURRENT_TEXT, OLD_TEXT, SELECTOR, admitted_pair, dispatch, outputs, wire_case,
)


class LogProbe:
    def __init__(self, caplog):
        self.caplog = caplog
        caplog.set_level("INFO", logger=LOGGER.name)

    @property
    def records(self):
        return [json.loads(item.getMessage().split(" ", 1)[1]) for item in self.caplog.records
                if item.name == LOGGER.name]


def stages(probe, stage):
    return [item for item in probe.records if item.get("stage") == stage]


@pytest.mark.asyncio
async def test_actual_guarded_wire_offers_unwritten_instruction_and_suppression_schema(caplog):
    async with wire_case() as case:
        session = case.h.session
        probe = LogProbe(caplog)
        state, old, current = await admitted_pair(case)
        offers = stages(probe, "payload_factory")
        assert len(offers) == 1
        event = offers[0]
        assert event["instruction_contract"] == "source_bound_user_v1"
        assert event["write_evidence"] == "factory_only_not_send_or_ack"
        assert {item["role"] for item in event["requests"]} == {"current", "retained_unwritten"}
        assert all(item["text_offered"] for item in event["requests"])
        assert event["schema"]["cancel_tool_count"] == 1
        assert event["schema"]["suppression_advertised"]
        assert event["schema"]["suppression_boolean"]
        assert event["schema"]["selector_enum_matches_offer"]
        wire = [item for item in case.socket.events if item.get("type") == "response.create"][-1]
        instructions = wire["response"]["instructions"]
        assert OLD_TEXT in instructions and CURRENT_TEXT in instructions
        assert "Perform the requested tool actions for each still-applicable retained request" in instructions
        assert "A later status check or unrelated question does not withdraw it" in instructions
        assert "Newer explicit corrections or withdrawals take precedence" in instructions
        assert {old.reference, current.reference} == {item["source_ref"] for item in event["requests"]}
        encoded = json.dumps(probe.records)
        assert OLD_TEXT not in encoded and CURRENT_TEXT not in encoded
        completed = stages(probe, "writer_finished")[-1]
        assert completed["outcome"] == "local_send_returned"
        assert completed["provider_ack"] == "unmeasured"
    assert probe.records[-1]["event"] == "request_source_diagnostics_closed"


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", ["old", "missing", "foreign", "stale"])
async def test_real_dispatch_selection_and_rejections_are_content_free(selection, caplog):
    async with wire_case() as case:
        session = case.h.session
        probe = LogProbe(caplog)
        state, old, current = await admitted_pair(case)
        args = {"task_ref": case.effects.ref, "expected_revision": 1}
        if selection != "missing":
            args[SELECTOR] = "private foreign selector must not leak" if selection == "foreign" else old.reference
        if selection == "stale":
            session._context_capture._input_versions[old.input_id] += 1
        await dispatch(case.h, "resp_current", "cancel_background_task", "call_diagnostic", args)
        await case.h.until(lambda: bool(outputs(case)))
        records = stages(probe, "selection")
        assert len(records) == 1
        record = records[0]
        if selection == "old":
            assert record["outcome"] == "selected_hot_cache"
            assert record["selected"]["source_ref"] == old.reference
            assert record["selected"]["role"] == "retained_unwritten"
            assert stages(probe, "executor_return")[-1]["outcome"] == "success"
            assert len(case.effects.effects) == 1
        else:
            assert record["outcome"] == "rejected_before_executor"
            assert record["error_code"] == {
                "missing": "tool_request_source_required", "foreign": "tool_request_source_invalid",
                "stale": "tool_request_source_unavailable",
            }[selection]
            assert not case.effects.effects
            assert not stages(probe, "executor_return")
        assert "private foreign selector" not in json.dumps(probe.records)
        assert OLD_TEXT not in json.dumps(probe.records)


@pytest.mark.asyncio
async def test_diagnostic_failure_does_not_change_real_write_or_effects():
    async with wire_case() as case:
        session = case.h.session
        with patch.object(LOGGER, "info", side_effect=RuntimeError("private diagnostic failure")):
            state, old, current = await admitted_pair(case)
            await dispatch(case.h, "resp_current", "cancel_background_task", "call_unlogged", {
                "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: old.reference,
            })
            await case.h.until(lambda: bool(outputs(case)))
        assert len(case.effects.effects) == 1
        assert session._request_source_diagnostic_counts["reporting_failed"] > 0


@pytest.mark.asyncio
async def test_canonical_withdrawal_is_not_mislabeled_as_successful_binding(caplog):
    async with wire_case() as case:
        session = case.h.session
        probe = LogProbe(caplog)
        state, old, current = await admitted_pair(case)
        case.canonical.unavailable.add(old.record.payload_id)
        await dispatch(case.h, "resp_current", "cancel_background_task", "call_withdrawn", {
            "task_ref": case.effects.ref, "expected_revision": 1, SELECTOR: old.reference,
        })
        await case.h.until(lambda: bool(outputs(case)))
        assert not case.effects.effects
        assert stages(probe, "selection")[-1]["outcome"] == "selected_hot_cache"
        assert stages(probe, "executor_return")[-1]["outcome"] == "failed"
        # The existing executor intentionally converts the inner source error.
        assert stages(probe, "executor_return")[-1]["error_code"] == "tool_execution_failed"


def diagnostic_session():
    return SimpleNamespace(context=SimpleNamespace(owner_id="owner_fixture", jarvis_session_id="session_fixture",
        audio_session_id="audio_fixture"), _visual_context=SimpleNamespace(connection=SimpleNamespace(
        connection_epoch="epoch_fixture")))


def test_cap_final_counts_and_closed_state_are_bounded(caplog):
    session = diagnostic_session()
    probe = LogProbe(caplog)
    for _ in range(MAX_RECORDS + 7):
        record_request_source(session, "selection", name="unknown private tool", code="secret code",
                              outcome="rejected_before_executor")
    close_request_source_diagnostics(session)
    close_request_source_diagnostics(session)
    record_request_source(session, "selection")
    assert len(probe.records) == MAX_RECORDS + 1
    final = probe.records[-1]
    assert final["attempted"] == MAX_RECORDS + 7
    assert final["emitted"] == MAX_RECORDS and final["dropped"] == 7
    assert final["reporting_failed"] == 0
    assert all(item["error_code"] == "redacted" and item["tool"] == "other"
               for item in probe.records[:-1])
    assert "secret" not in json.dumps(probe.records)


def test_inherited_schema_is_not_claimed_as_advertised(caplog):
    session = diagnostic_session()
    probe = LogProbe(caplog)
    record_request_source(session, "payload_factory", response={})
    assert probe.records[-1]["schema"] == {"scope": "inherited_unconfirmed"}
    assert probe.records[-1]["instruction_contract"] == "no_source_offer"


def test_nonstandard_error_is_redacted_without_changing_result(caplog):
    session = diagnostic_session()
    probe = LogProbe(caplog)
    result = {"status": "failed", "error": "private arbitrary error", "payload": "private payload"}
    record_request_source(session, "executor_return", result=result)
    assert probe.records[-1]["error_code"] == "redacted"
    assert probe.records[-1]["outcome"] == "failed"
    assert result == {"status": "failed", "error": "private arbitrary error", "payload": "private payload"}
    assert "private" not in json.dumps(probe.records)


def test_possible_write_without_return_remains_unknown_not_acknowledged(caplog):
    session = diagnostic_session()
    probe = LogProbe(caplog)
    reservation = SimpleNamespace(request_id="request_fixture", turn_generation=4, submitted=True)
    record_request_source(session, "writer_finished", reservation=reservation, outcome="no_return")
    assert probe.records[-1]["local_reservation_submitted"]
    assert probe.records[-1]["outcome"] == "no_return"
    assert probe.records[-1]["provider_ack"] == "unmeasured"
