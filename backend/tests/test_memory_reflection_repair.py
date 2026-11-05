import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from app.memory.reflection.contracts import ReflectionValidationError
from app.memory.reflection.repository import ReflectionEpisode, ReflectionEpisodeRange
from app.memory.reflection.service import ReflectionGenerationService


EPISODE_ID = UUID("181b4495-b8ab-4908-b945-46a40c1178ee")
# Shape retained from the native pipeline failure, not a valid UUID to repair in code.
MALFORMED_ID = "181b4495-b8ab-4908-b945-46e9-a150-436f394e560f"


def episode_range():
    now = datetime.now(UTC)
    return ReflectionEpisodeRange(owner_id="fixture-owner", episodes=(ReflectionEpisode(
        id=EPISODE_ID, owner_id="fixture-owner", session_id=None,
        title="Packing is unfinished", summary="Medication still needs to be added.",
        content="The user still needs to add medication to the bag.",
        occurred_at=now, created_at=now, is_explicit=False,
    ),))


def output(reference):
    return json.dumps({
        "title": "Packing remains unfinished", "summary": "Medication remains to be packed.",
        "content": "The packing task is not complete, and no repeated routine is established.",
        "importance": 0.6, "confidence": 0.9, "supporting_episode_ids": [reference],
        "preference_candidates": [], "procedure_candidates": [],
    })


@pytest.mark.asyncio
async def test_native_malformed_reference_gets_one_model_repair_without_coercion():
    client = SimpleNamespace(generate_json=AsyncMock(side_effect=[output(MALFORMED_ID), output(str(EPISODE_ID))]))
    result = await ReflectionGenerationService(client).generate(episode_range())
    assert result.supporting_episode_ids == [str(EPISODE_ID)]
    assert client.generate_json.await_count == 2
    first, repair = client.generate_json.call_args_list
    assert "Correction required" not in first.kwargs["prompt"]
    assert "contain an invalid UUID" in repair.kwargs["prompt"]
    assert str(EPISODE_ID) in repair.kwargs["prompt"]
    assert repair.kwargs["owner_id"] == "fixture-owner"
    assert repair.kwargs["schema"] == first.kwargs["schema"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", [MALFORMED_ID, str(UUID(int=9))])
async def test_second_invalid_output_stays_rejected(reference):
    client = SimpleNamespace(generate_json=AsyncMock(return_value=output(reference)))
    with pytest.raises(ReflectionValidationError):
        await ReflectionGenerationService(client).generate(episode_range())
    assert client.generate_json.await_count == 2


@pytest.mark.asyncio
async def test_valid_first_output_does_not_add_a_model_call():
    client = SimpleNamespace(generate_json=AsyncMock(return_value=output(str(EPISODE_ID))))
    await ReflectionGenerationService(client).generate(episode_range())
    assert client.generate_json.await_count == 1


@pytest.mark.asyncio
async def test_transport_failure_is_not_relabelled_as_a_validation_repair():
    client = SimpleNamespace(generate_json=AsyncMock(side_effect=RuntimeError("transport failure")))
    with pytest.raises(RuntimeError, match="transport failure"):
        await ReflectionGenerationService(client).generate(episode_range())
    assert client.generate_json.await_count == 1
