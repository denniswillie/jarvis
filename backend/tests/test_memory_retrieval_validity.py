import asyncio
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.memory.repository import _retrieval_lifecycle_predicate
from app.memory.presentation import retrieval_payload
from app.memory.retrieval.candidates import CandidateCollection
from app.memory.retrieval.models import RetrievalSourceHit
from app.memory.retrieval.query import MemoryRetrievalQuery
from app.memory.retrieval.service import HierarchicalMemoryRetrievalService
from test_hierarchical_retrieval import NOW, FakeMemoryRepository, document


def retrieve(*documents, historical=False, limit=12):
    repository = FakeMemoryRepository()
    collector = SimpleNamespace(collect=AsyncMock(return_value=CandidateCollection(
        rankings={"lexical": [
            RetrievalSourceHit(item, "lexical", 1.0, exact_match=True) for item in documents
        ]}, provider_spaces=(),
    )))
    service = HierarchicalMemoryRetrievalService(
        memory_repository=repository, perception_repository=None,
        candidate_collector=collector, result_limit=limit,
    )
    response = asyncio.run(service.search(
        owner_id="owner", request=MemoryRetrievalQuery(
            query="state", include_superseded=historical,
        ), now=NOW,
    ))
    return response, repository.audit


@pytest.mark.parametrize("valid_from,valid_to,expected", [
    (None, None, "current"),
    (NOW, None, "current"),
    (NOW - timedelta(hours=1), NOW + timedelta(seconds=1), "current"),
    (NOW - timedelta(hours=1), NOW, "historical"),
    (NOW - timedelta(hours=1), NOW - timedelta(seconds=1), "historical"),
    (NOW + timedelta(seconds=1), None, "future"),
])
def test_validity_boundaries_drive_default_filter_and_historical_labels(valid_from, valid_to, expected):
    item = replace(document(), record_type="semantic_fact", valid_from=valid_from, valid_to=valid_to)
    result, audit = retrieve(item)
    assert bool(result.results) is (expected == "current")
    assert audit["constraints"]["lifecycle_as_of"] == NOW.isoformat()
    historical, _ = retrieve(item, historical=True)
    assert historical.results[0].temporal_status == expected
    assert historical.results[0].is_historical is (expected == "historical")


@pytest.mark.parametrize("status", ["superseded", "retracted", "forgotten", "invalid"])
def test_status_cannot_be_made_current_by_an_open_ended_interval(status):
    item = replace(document(), record_status=status)
    assert not retrieve(item)[0].results
    historical = retrieve(item, historical=True)[0]
    assert bool(historical.results) is (status in {"superseded", "retracted"})
    if historical.results:
        assert historical.results[0].is_historical


def test_historical_experience_event_remains_available_without_current_fact_flag():
    event = replace(document(), record_type="experience_event", record_status="historical", valid_to=NOW)
    response, _ = retrieve(event)
    assert response.results[0].is_historical


def test_completed_episode_is_historical_but_searchable_without_superseded_flag():
    episode = replace(document(), record_type="episode", valid_from=NOW - timedelta(hours=2), valid_to=NOW - timedelta(hours=1))
    response, _ = retrieve(episode)
    assert len(response.results) == 1
    assert response.results[0].temporal_status == "historical"
    assert not retrieve(replace(episode, valid_from=NOW + timedelta(hours=1), valid_to=NOW + timedelta(hours=2)))[0].results
    assert not retrieve(replace(episode, record_status="superseded"))[0].results


def test_invalid_and_other_owner_candidates_do_not_displace_current_results():
    expired = replace(document(), id=UUID(int=20), record_type="semantic_fact", valid_to=NOW)
    foreign = replace(document(), owner_id="other-owner", id=UUID(int=21))
    current = replace(document(), id=UUID(int=22))
    response, audit = retrieve(expired, foreign, current, limit=1)
    assert [item.id for item in response.results] == [current.id]
    assert audit["candidate_ids"] == [current.id]
    historical, _ = retrieve(foreign, current, historical=True)
    assert [item.id for item in historical.results] == [current.id]


@pytest.mark.parametrize("alias", ["document", "candidate"])
def test_every_shared_repository_lifecycle_predicate_checks_both_bounds(alias):
    sql = _retrieval_lifecycle_predicate(alias)
    assert f"{alias}.valid_from <= now()" in sql
    assert f"{alias}.valid_to > now()" in sql


@pytest.mark.parametrize("start,end,expected", [
    (NOW, None, "current"),
    (NOW - timedelta(hours=1), NOW, "historical"),
    (NOW + timedelta(hours=1), None, "future"),
])
def test_temporal_classification_reaches_the_actual_tool_serializer(start, end, expected):
    item = replace(document(), valid_from=start, valid_to=end)
    response, _ = retrieve(item, historical=True)
    payload = retrieval_payload(response)["results"][0]
    assert payload["status"] == "active"
    assert payload["temporal_status"] == expected
    assert payload["is_historical"] is (expected == "historical")
