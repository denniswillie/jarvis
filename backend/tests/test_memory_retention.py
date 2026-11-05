from datetime import UTC, datetime

import pytest

from app.memory.retention import MemoryRetentionPolicy


def test_retention_policy_produces_independent_cutoffs() -> None:
    policy = MemoryRetentionPolicy()
    now = datetime(2026, 8, 28, 12, tzinfo=UTC)

    assert (now - policy.observation_detail_cutoff(now)).days == 7
    assert (now - policy.unreferenced_observation_cutoff(now)).days == 30
    assert (now - policy.unreferenced_visual_frame_cutoff(now)).days == 30
    assert (now - policy.skipped_event_cutoff(now)).days == 90
    assert (now - policy.completed_job_cutoff(now)).days == 14
    assert (now - policy.retrieval_audit_cutoff(now)).days == 30


def test_retention_policy_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        MemoryRetentionPolicy(batch_size=0)


class RecordingConnection:
    def __init__(self) -> None:
        self.query = ""

    async def fetchrow(self, query, *args):
        self.query = query
        return {}


@pytest.mark.anyio
async def test_visual_frame_cleanup_requires_all_durable_references_to_release() -> None:
    from app.memory.retention import MemoryRetentionRepository

    connection = RecordingConnection()
    await MemoryRetentionRepository().delete_unreferenced_visual_frames(
        owner_id="owner",
        captured_before=datetime(2026, 7, 29, tzinfo=UTC),
        now=datetime(2026, 8, 28, tzinfo=UTC),
        limit=50,
        connection=connection,
    )

    assert "visual_frame_retention_reasons" in connection.query
    assert "jarvis_files" in connection.query
    assert "memory.status = 'active'" in connection.query
    assert "delete from memory_evidence evidence" in connection.query
    assert "deleted_memory_evidence" in connection.query
    assert "delete from experience_event_keyframes" in connection.query
    assert "for update skip locked" in connection.query


@pytest.mark.anyio
async def test_skipped_event_cleanup_honors_generic_retention_reasons() -> None:
    from app.memory.retention import MemoryRetentionRepository

    connection = RecordingConnection()
    await MemoryRetentionRepository().delete_expired_skipped_events(
        owner_id="owner",
        now=datetime(2026, 8, 28, tzinfo=UTC),
        limit=50,
        connection=connection,
    )

    assert "experience_event_retention_reasons" in connection.query
    assert "memory.source_experience_event_id = event.id" in connection.query
    source_guard = connection.query.split("memory.source_experience_event_id = event.id", 1)[1]
    assert "memory.status = 'active'" not in source_guard.split(")", 1)[0]
    assert "deleted_expired_event_retention_reasons" in connection.query
