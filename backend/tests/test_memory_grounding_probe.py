import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.memory.consolidation.contracts import ConsolidationResult
from app.memory.consolidation.evidence import build_citation_catalog
from evals.glasses_reliability.memory_grounding_probe import (
    CASES, FixtureRequestConsolidator, assess_structure, build_request,
)
from test_consolidation_contracts import _output, _request


def test_fixture_inputs_are_time_scoped_and_never_attach_fake_pixels():
    now = datetime(2026, 9, 13, tzinfo=UTC)
    for case in CASES:
        request = build_request(case, now=now)
        assert request.event.owner_id.startswith("memory-grounding-fixture-")
        assert not request.keyframes
        if case.scene_only:
            assert request.event.activity == ""
        assert all(request.event.started_at <= item.occurred_at <= now for item in request.evidence)
        if case.interrupted:
            assert all(
                entry.source.payload.get("role") != "assistant"
                for entry in build_citation_catalog(request).values()
            )


def test_structural_oracle_does_not_claim_a_semantic_summary_review():
    result = ConsolidationResult.model_validate(_output(_request()))
    checks = assess_structure(CASES[0], result)
    assert not checks["required_outcome_present"]
    assert set(checks) == {"required_outcome_present", "no_unsupported_completed_outcome"}


def test_forged_completed_state_fails_negative_oracle():
    output = _output(_request())
    output["task_outcomes"] = [{
        "task": "unfinished work", "state": "completed", "evidence_basis": "user_report",
        "details": "Fabricated completion", "evidence": output["entity_candidates"][0]["evidence"],
    }]
    result = ConsolidationResult.model_validate(output)
    assert not assess_structure(CASES[0], result)["no_unsupported_completed_outcome"]


def test_fixture_request_identity_is_checked_before_native_call():
    request = build_request(CASES[0])
    service = FixtureRequestConsolidator(SimpleNamespace(generate=AsyncMock()), request)
    with pytest.raises(ValueError, match="identity mismatch"):
        asyncio.run(service.consolidate(
            owner_id="wrong-owner", session_id=request.event.session_id, event_id=request.event.id,
        ))
    service._client.generate.assert_not_awaited()


def test_scene_fixture_requires_bounded_facts():
    output = _output(_request())
    result = ConsolidationResult.model_validate(output)
    scene = next(case for case in CASES if case.scene_only)
    assert not assess_structure(scene, result)["no_open_ended_scene_fact"]
    bounded = result.semantic_fact_candidates[0].model_copy(update={
        "valid_to_hint": datetime.now(UTC).isoformat(),
    })
    result = result.model_copy(update={"semantic_fact_candidates": [bounded]})
    assert assess_structure(scene, result)["no_open_ended_scene_fact"]


@pytest.mark.parametrize("end", ["not-a-time", "2026-01-01", "2099-01-01T00:00:00+00:00"])
def test_scene_oracle_rejects_unusable_or_future_end_dates(end):
    output = _output(_request())
    output["semantic_fact_candidates"][0]["valid_to_hint"] = end
    scene = next(case for case in CASES if case.scene_only)
    result = ConsolidationResult.model_validate(output)
    assert not assess_structure(scene, result)["no_open_ended_scene_fact"]
