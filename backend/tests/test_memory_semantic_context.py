import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.memory.consolidation.prompts import build_consolidation_prompt
from app.memory.consolidation.service import ExperienceConsolidationService, MAX_SEMANTIC_CONTEXT_FACTS, _semantic_context_query
from app.memory.repository import MemoryRepository
from test_consolidation_contracts import _cited_output, _request
from app.memory.consolidation.contracts import ConsolidationValidationError, parse_consolidation_result


def test_prior_facts_are_context_not_new_literal_evidence():
    request = _request()
    request = replace(request, semantic_fact_context=({
        "subject": "user", "predicate": "has", "value": "unquoted prior value",
    },), semantic_fact_context_truncated=True)
    payload = json.loads(build_consolidation_prompt(request).split("\n\n", 1)[1])
    assert payload["semantic_fact_context_truncated"]
    assert payload["semantic_fact_context"][0]["value"] == "unquoted prior value"
    with pytest.raises(ConsolidationValidationError) as caught:
        parse_consolidation_result(_cited_output(request, text="unquoted prior value"), request=request)
    assert caught.value.code == "missing_literal_quotation"


def test_build_request_loads_bounded_owner_context_at_the_event_time():
    source = _request()
    experience = SimpleNamespace(
        get_event=AsyncMock(return_value=source.event), list_event_evidence=AsyncMock(return_value=[]),
        list_nonvisual_source_payloads=AsyncMock(return_value={}),
        list_context_messages=AsyncMock(return_value=[]), list_event_keyframe_sources=AsyncMock(return_value=[]),
    )
    prior = [{"subject": "a box", "predicate": "is", "value": f"state {index}"}
             for index in range(MAX_SEMANTIC_CONTEXT_FACTS + 1)]
    reader = SimpleNamespace(list_semantic_fact_context=AsyncMock(return_value=prior))
    service = ExperienceConsolidationService(
        experience_repository=experience,
        perception_repository=SimpleNamespace(list_observations_by_ids=AsyncMock(return_value=[])),
        client=SimpleNamespace(), image_loader=None, owner_working_memory_reader=None,
        active_reminder_reader=None, max_keyframes=3, max_transcripts=24, semantic_context_reader=reader,
    )
    request = asyncio.run(service.build_request(
        owner_id=source.event.owner_id, session_id=source.event.session_id, event_id=source.event.id,
    ))
    reader.list_semantic_fact_context.assert_awaited_once_with(
        owner_id=source.event.owner_id, at=source.event.ended_at, limit=MAX_SEMANTIC_CONTEXT_FACTS + 1,
        query_text="",
    )
    assert request.semantic_fact_context == tuple(prior[:MAX_SEMANTIC_CONTEXT_FACTS])
    assert request.semantic_fact_context_truncated


def test_semantic_context_query_is_owner_scoped_bounded_and_preserves_history():
    connection = SimpleNamespace(fetch=AsyncMock(return_value=[{"subject": "the box", "value": "missing"}]))
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    now = datetime.now(UTC)
    rows = asyncio.run(MemoryRepository(pool).list_semantic_fact_context(owner_id="owner", at=now, limit=41))
    sql, *args = connection.fetch.call_args.args
    assert args == ["owner", now, 41, "", []]
    assert "entity.owner_id = memory.owner_id" in sql
    assert "memory.owner_id = $1" in sql
    assert "('active', 'superseded', 'retracted')" in sql
    assert "limit $3" in sql
    assert rows[0]["value"] == "missing"


def test_context_ranking_query_uses_citable_content_not_assistant_or_metadata():
    request = _request()
    source = request.evidence[0]
    user = replace(source, evidence_type="message", payload={
        "role": "user", "modality": "text", "content": "My blue travel pouch is found.",
        "metadata": {"secret": "not relevance content"},
    })
    assistant = replace(user, payload={"role": "assistant", "modality": "text", "content": "invented identity"})
    incomplete = replace(user, payload={**user.payload, "content": "unfinished speech", "metadata": {"experience_eligible": False}})
    query = _semantic_context_query((user, assistant, incomplete))
    assert query == "My blue travel pouch is found."
    huge = replace(user, payload={**user.payload, "content": "\u00e9" * 9000})
    assert len(_semantic_context_query((huge,)).encode("utf-8")) <= 8192


def test_context_relevance_uses_owned_entity_names_before_temporal_recency():
    connection = SimpleNamespace(fetch=AsyncMock(return_value=[]))
    pool = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    asyncio.run(MemoryRepository(pool).list_semantic_fact_context(
        owner_id="owner", at=datetime.now(UTC), query_text="Travel pouch",
    ))
    sql, *args = connection.fetch.call_args.args
    assert args[-2:] == ["travel pouch", ["travel", "pouch"]]
    assert sql.index("relevance.token_hits desc") < sql.index("case when (memory.valid_from")


@pytest.mark.parametrize("limit", [0, 82, -1])
def test_context_limit_cannot_expand_into_an_unbounded_memory_dump(limit):
    with pytest.raises(ValueError, match="limit"):
        asyncio.run(MemoryRepository(None).list_semantic_fact_context(
            owner_id="owner", at=datetime.now(UTC), limit=limit,
        ))
