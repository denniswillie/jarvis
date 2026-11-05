from datetime import UTC, datetime

import pytest

from app.memory.retention import MemoryRetentionRepository


pytestmark = pytest.mark.anyio


class RecordingConnection:
    def __init__(self) -> None:
        self.query = ""
        self.args = ()

    async def fetchval(self, query, *args):
        self.query = query
        self.args = args
        return 3


async def test_compaction_is_bounded_and_protects_active_memory_evidence() -> None:
    connection = RecordingConnection()
    repository = MemoryRetentionRepository()

    compacted = await repository.compact_observations(
        owner_id="owner",
        ended_before=datetime(2026, 8, 21, tzinfo=UTC),
        compacted_at=datetime(2026, 8, 28, tzinfo=UTC),
        limit=50,
        connection=connection,
    )

    assert compacted == 3
    assert "memory.status = 'active'" in connection.query
    assert "for update skip locked" in connection.query
    assert "delete from perception_observation_frames" in connection.query
    assert "primary_visual_frame_id = null" in connection.query
    assert "limit $4" in connection.query
    assert connection.args[-1] == 50
