from dataclasses import replace
import json

import pytest

from app.memory.consolidation.contracts import (
    ConsolidationValidationError, parse_consolidation_result,
)
from app.memory.consolidation.evidence import build_citation_catalog
from app.memory.consolidation.prompts import build_consolidation_prompt
from test_consolidation_contracts import _cited_output, _output, _request


def outcome_output(request, *, state="paused", basis="user_report"):
    output = _cited_output(request)
    output["task_outcomes"] = [{
        "task": "Prepare the function", "state": state, "evidence_basis": basis,
        "details": "Stopped for today; work remains unfinished.",
        "evidence": output["entity_candidates"][0]["evidence"],
    }]
    return output


@pytest.mark.parametrize("metadata", [
    {"experience_eligible": False},
    {"completion_state": "interrupted"},
    {"completion_state": "incomplete"},
    {"completion_state": "cancelled"},
    {"completion_state": "failed"},
    {"delivery_status": "incomplete"},
    {"delivery_status": "unforwarded"},
    {"delivery_status": "failed"},
    {"playback_status": "truncated"},
])
def test_diagnostic_assistant_speech_cannot_be_cited(metadata):
    request = _request(message_role="assistant")
    source = request.evidence[0]
    request = replace(request, evidence=(replace(source, payload={
        **source.payload, "metadata": metadata,
    }),))
    assert not build_citation_catalog(request)
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(_cited_output(request), request=request)
    assert caught.value.code == "uncitable_source"


def test_known_ineligible_user_transcript_cannot_support_a_fact():
    request = _request()
    source = request.evidence[0]
    request = replace(request, evidence=(replace(source, payload={
        **source.payload, "metadata": {"experience_eligible": False},
    }),))
    assert not build_citation_catalog(request)


@pytest.mark.parametrize("role", ["user", "assistant"])
@pytest.mark.parametrize("metadata", [
    {"experience_eligible": "false"},
    {"experience_eligible": []},
    {"completion_state": []},
    {"completion_state": {}},
    {"completion_state": "unfinished"},
    {"completion_state": "incomplete"},
])
def test_malformed_or_incomplete_transcript_provenance_is_uncitable_not_an_exception(role, metadata):
    request = _request(message_role=role)
    source = request.evidence[0]
    request = replace(request, evidence=(replace(source, payload={
        **source.payload, "metadata": metadata,
    }),))
    assert not build_citation_catalog(request)


@pytest.mark.parametrize("metadata", [
    {"delivery_status": []}, {"delivery_status": {}},
    {"playback_status": []}, {"playback_status": {}},
])
def test_malformed_audio_delivery_is_not_citable(metadata):
    request = _request(message_role="assistant")
    source = request.evidence[0]
    request = replace(request, evidence=(replace(source, payload={
        **source.payload, "metadata": metadata,
    }),))
    assert not build_citation_catalog(request)


def test_forwarding_provenance_is_not_hearing_confirmation_or_quotation_material():
    request = _request(message_role="assistant")
    source = request.evidence[0]
    request = replace(request, evidence=(replace(source, payload={
        **source.payload, "metadata": {
            "experience_eligible": True, "completion_state": "completed",
            "delivery_status": "forwarded", "playback_status": "unconfirmed",
            "secret_label": "PRIVATE_DIAGNOSTIC",
        },
    }),))
    payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
    entry = payload["attached_evidence"][0]
    assert entry["source_provenance"]["delivery_status"] == "forwarded"
    assert entry["source_provenance"]["delivery_confirmation"] == "unknown"
    assert "PRIVATE_DIAGNOSTIC" not in json.dumps(entry)
    assert all("forwarded" not in item["text"] for item in entry["quotation_material"])


def test_legacy_message_preserves_unknown_delivery_not_assumed_heard():
    request = _request(message_role="assistant")
    payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
    assert payload["attached_evidence"][0]["source_provenance"] == {
        "delivery_confirmation": "unknown",
    }
    parse_consolidation_result(_cited_output(request), request=request)


def test_assistant_approval_alone_cannot_become_a_semantic_fact():
    request = _request(message_role="assistant")
    output = _output(request)
    output["semantic_fact_candidates"][0]["explicitly_user_stated"] = False
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(output, request=request)
    assert caught.value.code == "unsupported_evidence_basis"


@pytest.mark.parametrize("state", ["in_progress", "paused", "abandoned", "unknown"])
def test_supported_noncompletion_outcomes_are_distinct_and_persisted(state):
    request = _request()
    output = outcome_output(request, state=state)
    result = parse_consolidation_result(output, request=request)
    assert result.model_dump()["task_outcomes"][0]["state"] == state


@pytest.mark.parametrize("basis", ["observation", "tool_result", "uncertain"])
def test_user_statement_cannot_be_relabelled_as_verified_completion(basis):
    request = _request()
    with pytest.raises(ConsolidationValidationError):
        parse_consolidation_result(
            outcome_output(request, state="completed", basis=basis), request=request,
        )


def test_reported_completion_retains_reported_not_observed_basis():
    request = _request()
    result = parse_consolidation_result(
        outcome_output(request, state="completed"), request=request,
    )
    assert result.task_outcomes[0].evidence_basis == "user_report"


def test_assistant_cannot_certify_task_completion_even_when_not_explicit():
    request = _request(message_role="assistant")
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(
            outcome_output(request, state="completed"), request=request,
        )
    assert caught.value.code == "unsupported_evidence_basis"


@pytest.mark.parametrize("execution_status,result_status,accepted", [
    ("completed", "success", True),
    ("completed", "failed", False),
    ("failed", "success", False),
    ("running", "success", False),
])
def test_tool_completion_requires_a_successful_result(execution_status, result_status, accepted):
    request = _request()
    source = replace(request.evidence[0], evidence_type="tool_execution", payload={
        "tool_name": "remember", "status": execution_status,
        "result": {"status": result_status, "description": "Saved the requested note"},
    })
    request = replace(request, evidence=(source,))
    output = _cited_output(request, text="Saved the requested note")
    output["task_outcomes"] = [{
        "task": "Save a note", "state": "completed", "evidence_basis": "tool_result",
        "details": "The tool reported a saved note; delivery to the user is unconfirmed.",
        "evidence": output["entity_candidates"][0]["evidence"],
    }]
    if accepted:
        assert parse_consolidation_result(output, request=request).task_outcomes[0].state == "completed"
    else:
        with pytest.raises(ConsolidationValidationError) as caught:
            parse_consolidation_result(output, request=request)
        assert caught.value.code == "unsupported_evidence_basis"


def test_legacy_consolidations_without_task_outcomes_remain_readable():
    request = _request()
    assert parse_consolidation_result(_output(request), request=request).task_outcomes == []


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_conversation_text_cannot_be_relabelled_as_visible_text(role):
    request = _request(message_role=role)
    output = _cited_output(request)
    output["visible_text"] = [{
        "text": "silver keys", "context": "claimed to be visible", "confidence": 0.9,
        "evidence": output["entity_candidates"][0]["evidence"],
    }]
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(output, request=request)
    assert caught.value.code == "unsupported_evidence_basis"


@pytest.mark.parametrize("text,accepted", [("HOME", True), ("HOME OFFICE", False)])
def test_visible_text_requires_literal_observation_support(text, accepted):
    request = _request()
    source = replace(request.evidence[0], evidence_type="observation", payload={
        "summary": "A label is visible", "visible_text": [{"text": "HOME"}],
    })
    request = replace(request, evidence=(source,))
    output = _cited_output(request, text="HOME")
    output["visible_text"] = [{
        "text": text, "context": "label", "confidence": 0.9,
        "evidence": output["entity_candidates"][0]["evidence"],
    }]
    if accepted:
        assert parse_consolidation_result(output, request=request).visible_text[0].text == text
    else:
        with pytest.raises(ConsolidationValidationError) as caught:
            parse_consolidation_result(output, request=request)
        assert caught.value.code == "unsupported_evidence_basis"
