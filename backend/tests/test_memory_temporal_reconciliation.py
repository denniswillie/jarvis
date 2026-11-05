from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from app.memory.consolidation.contracts import ConsolidationResult, ConsolidationValidationError, parse_consolidation_result
from app.memory.semantic.contracts import SemanticEvidence, SemanticReconciliationContext
from app.memory.semantic.contradictions import ActiveSemanticFact, SemanticContradictionPolicy
from app.memory.semantic.reconciler import SemanticMemoryReconciler
from test_consolidation_contracts import _output, _request
from test_semantic_contradictions import _proposal


NOW = datetime(2026, 9, 13, 12, tzinfo=UTC)


def state_proposal(**changes):
    return replace(_proposal("is", "found", explicit=True),
                   valid_from=NOW, temporal_scope="state", change_intent="transition",
                   previous_value="missing", **changes)


def known(value="missing", *, start=NOW - timedelta(minutes=1), **kwargs):
    return ActiveSemanticFact(id=uuid4(), predicate="is", value=value, valid_from=start, **kwargs)


def test_state_transition_closes_only_the_named_previous_value():
    missing, unrelated = known(), known("red")
    result = SemanticContradictionPolicy().decide(
        proposal=state_proposal(), active_facts=(missing, unrelated),
    )
    assert result.create_new
    assert result.supersede_fact_ids == (missing.id,)
    assert result.reason == "confirmed_state_transition"


@pytest.mark.parametrize("confidence,changes", [(0.86, False), (0.9, True), (0.99, True)])
def test_observed_transition_requires_confident_change_even_for_old_explicit_state(confidence, changes):
    missing = known(is_explicit=True)
    proposal = replace(state_proposal(), source="inferred_observation", confidence=confidence)
    result = SemanticContradictionPolicy().decide(proposal=proposal, active_facts=(missing,))
    assert result.create_new is changes
    assert result.supersede_fact_ids == ((missing.id,) if changes else ())


def test_unknown_transition_target_does_not_create_an_unbounded_replacement():
    result = SemanticContradictionPolicy().decide(
        proposal=state_proposal(), active_facts=(known("red"),),
    )
    assert not result.create_new
    assert result.reason == "replacement_target_not_active"


def test_direct_event_only_proposal_cannot_bypass_the_contract_boundary():
    with pytest.raises(ValueError, match="event-only"):
        SemanticContradictionPolicy().decide(
            proposal=replace(state_proposal(), temporal_scope="event"), active_facts=(known(),),
        )


def test_transition_contract_requires_a_known_non_event_prior_fact():
    request = _request()
    output = _output(request)
    fact = output["semantic_fact_candidates"][0]
    fact.update(temporal_scope="state", change_intent="transition", previous_value="bronze keys")
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(output, request=request)
    assert caught.value.code == "invalid_state_transition"
    prior = {"subject": "user", "predicate": "has", "value": "bronze keys", "temporal_scope": "state"}
    result = parse_consolidation_result(output, request=replace(request, semantic_fact_context=(prior,)))
    assert result.semantic_fact_candidates[0].change_intent == "transition"
    for alteration in ({"subject": "another person"}, {"predicate": "owns"}, {"temporal_scope": "event"}):
        with pytest.raises(ConsolidationValidationError):
            parse_consolidation_result(output, request=replace(request, semantic_fact_context=({**prior, **alteration},)))


@pytest.mark.parametrize("scope,previous", [("durable", "missing"), ("event", "missing"), ("state", None)])
def test_transition_cannot_be_disguised_as_an_event_or_omit_previous_value(scope, previous):
    request = _request()
    output = _output(request)
    output["semantic_fact_candidates"][0].update(
        temporal_scope=scope, change_intent="transition", previous_value=previous,
    )
    with pytest.raises(ConsolidationValidationError):
        parse_consolidation_result(output, request=request)


def test_late_state_evidence_stops_at_a_known_successor():
    later = known("found", start=NOW + timedelta(minutes=1), change_intent="transition",
                  previous_value="missing", is_explicit=True)
    old = replace(state_proposal(), value="missing", change_intent="assert", previous_value=None)
    bounded = SemanticContradictionPolicy().bound_validity(proposal=old, known_facts=(later,))
    assert bounded.valid_to == later.valid_from


def test_unrelated_future_state_does_not_expire_a_fact():
    later = known("blue", start=NOW + timedelta(minutes=1), change_intent="transition",
                  previous_value="red", is_explicit=True)
    old = replace(state_proposal(), value="missing", change_intent="assert", previous_value=None)
    assert SemanticContradictionPolicy().bound_validity(proposal=old, known_facts=(later,)).valid_to is None


def test_splitting_an_old_interval_preserves_its_existing_end():
    old = known(valid_to=NOW + timedelta(minutes=1))
    bounded = SemanticContradictionPolicy().bound_validity(
        proposal=state_proposal(), known_facts=(old,), replaced_ids=(old.id,),
    )
    assert bounded.valid_to == old.valid_to


@pytest.mark.parametrize("later_explicit,older_explicit,expected_bound", [
    (True, True, True), (True, False, True), (False, False, True), (False, True, False),
])
def test_exclusive_late_evidence_respects_source_strength(later_explicit, older_explicit, expected_bound):
    old = replace(_proposal("located_at", "hall", explicit=older_explicit), valid_from=NOW)
    future = ActiveSemanticFact(
        id=uuid4(), predicate="located_at", value="desk", valid_from=NOW + timedelta(minutes=1),
        is_explicit=later_explicit, confidence=0.95,
    )
    bounded = SemanticContradictionPolicy().bound_validity(proposal=old, known_facts=(future,))
    assert bounded.valid_to == (future.valid_from if expected_bound else None)


def test_weak_future_inference_does_not_silently_rewrite_a_current_fact():
    later = known("found", start=NOW + timedelta(minutes=1), change_intent="transition",
                  previous_value="missing", confidence=0.86)
    old = replace(state_proposal(), value="missing", change_intent="assert", previous_value=None)
    assert SemanticContradictionPolicy().bound_validity(proposal=old, known_facts=(later,)).valid_to is None


def test_same_fact_does_not_reinforce_a_different_validity_interval():
    prior = known("found", valid_to=NOW + timedelta(minutes=1))
    new = replace(state_proposal(), change_intent="assert", previous_value=None)
    policy = SemanticContradictionPolicy()
    result = policy.decide(proposal=new, active_facts=(prior,))
    assert result.create_new
    assert result.reinforce_fact_id is None
    assert result.supersede_fact_ids == ()
    same_window = replace(new, valid_to=prior.valid_to)
    assert policy.decide(proposal=same_window, active_facts=(prior,)).reinforce_fact_id == prior.id


def context_with_scope(scope, *, start_hint=None, end_hint=None):
    request = _request()
    output = _output(request)
    output["semantic_fact_candidates"][0].update(
        temporal_scope=scope, valid_from_hint=start_hint, valid_to_hint=end_hint,
    )
    source = request.evidence[0]
    return SemanticReconciliationContext(
        owner_id=request.event.owner_id, session_id=request.event.session_id,
        event_id=request.event.id, episode_memory_id=uuid4(), occurred_at=NOW,
        event_started_at=NOW - timedelta(minutes=1),
        result=ConsolidationResult.model_validate(output),
        evidence=(SemanticEvidence(
            evidence_type="message", evidence_ref_id=source.evidence_ref_id,
            occurred_at=NOW - timedelta(seconds=30), excerpt=source.payload["content"], message_role="user",
        ),),
    )


def test_event_snapshot_gets_a_finite_interval_without_model_guessing_an_expiry():
    context = context_with_scope("event")
    proposal = SemanticMemoryReconciler().plan(context).facts[0]
    assert proposal.valid_from == context.event_started_at
    assert proposal.valid_to == context.occurred_at
    assert proposal.temporal_scope == "event"


def test_event_snapshot_cannot_extend_beyond_its_evidence_window():
    context = context_with_scope(
        "event", start_hint=(NOW - timedelta(days=3)).isoformat(), end_hint=(NOW + timedelta(days=3)).isoformat(),
    )
    proposal = SemanticMemoryReconciler().plan(context).facts[0]
    assert (proposal.valid_from, proposal.valid_to) == (context.event_started_at, NOW)


def test_snapshot_wholly_outside_the_event_is_not_promoted():
    context = context_with_scope("event", start_hint=(NOW + timedelta(days=1)).isoformat())
    assert not SemanticMemoryReconciler().plan(context).facts


@pytest.mark.parametrize("scope", ["state", "durable"])
def test_non_snapshot_fact_can_retain_supported_open_ended_validity(scope):
    context = context_with_scope(scope)
    assert SemanticMemoryReconciler().plan(context).facts[0].valid_to is None


def test_exact_duplicate_candidates_are_deduplicated_but_distinct_intervals_are_retained():
    context = context_with_scope("event")
    fact = context.result.semantic_fact_candidates[0]
    later = fact.model_copy(update={"valid_from_hint": (NOW - timedelta(seconds=10)).isoformat()})
    result = context.result.model_copy(update={"semantic_fact_candidates": [fact, fact, later]})
    plan = SemanticMemoryReconciler().plan(replace(context, result=result))
    assert len(plan.facts) == 2
    assert {item.valid_from for item in plan.facts} == {context.event_started_at, NOW - timedelta(seconds=10)}


def test_event_relations_are_time_scoped_too():
    context = context_with_scope("event")
    citation = context.result.semantic_fact_candidates[0].evidence
    result = context.result.model_dump()
    result["relation_candidates"] = [{
        "source_entity": "user", "predicate": "with", "target_entity": "silver keys",
        "confidence": 0.95, "explicitly_user_stated": True, "temporal_scope": "event",
        "valid_from_hint": None, "valid_to_hint": None,
        "evidence": [item.model_dump() for item in citation],
    }]
    plan = SemanticMemoryReconciler().plan(replace(context, result=ConsolidationResult.model_validate(result)))
    assert len(plan.relations) == 1
    assert plan.relations[0].valid_to == NOW
