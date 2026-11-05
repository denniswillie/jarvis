from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.knowledge.selection import MemoryAuthorityContextProvider


@pytest.mark.asyncio
@pytest.mark.parametrize("tasks_enabled", [False, True])
async def test_memory_correction_keeps_selected_delivery_and_task_authority(tasks_enabled: bool) -> None:
    services = SimpleNamespace(
        knowledge_runtime=None,
        knowledge_issue="fixture-unavailable",
        background_task_service=object() if tasks_enabled else None,
    )
    request = SimpleNamespace(requested_at_utc=datetime(2026, 10, 31, tzinfo=UTC))
    section = await MemoryAuthorityContextProvider(services).build_section(request)

    assert section.payload["legacy_fallback"] is False
    assert section.payload["configured"] is False
    assert ("correction_route" in section.payload) is tasks_enabled
    if tasks_enabled:
        policy = section.payload["correction_route"]
        assert "authorized delivery by the selected voice route" in policy
        assert "direct TTS" not in policy
        assert "memory_edit capability" in policy
        assert "preserves unrelated facts" in policy
        assert "the correction has not been made; do not bypass it" in policy
