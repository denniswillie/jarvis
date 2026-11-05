from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.episodic.service import EpisodePromotionService
from test_episode_promotion import _context


class EpisodeIdempotencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_decided_event_replays_without_rewriting_episode(self) -> None:
        context = _context(importance=0.9, should_promote=False)
        plan = EpisodePromotionService(
            minimum_importance=0.55,
            important_keyframe_minimum_importance=0.8,
        ).plan(context)
        episode_id = uuid4()
        connection = AsyncMock()
        connection.is_in_transaction = MagicMock(return_value=True)
        connection.fetchrow.return_value = {
            "id": context.event.id,
            "status": "consolidated",
            "consolidation_result": context.event.consolidation_result,
            "promotion_decision": "promoted",
            "promotion_reason": "already promoted",
            "promoted_episode_memory_id": episode_id,
        }
        connection.fetch.return_value = [
            {"visual_frame_id": frame.visual_frame_id}
            for frame in plan.final_keyframes
        ]
        repository = EpisodicMemoryRepository(
            object(),  # type: ignore[arg-type]
            experience_repository=object(),  # type: ignore[arg-type]
            memory_repository=object(),  # type: ignore[arg-type]
        )

        outcome = await repository.apply_promotion(
            context=context,
            plan=plan,
            decided_at=datetime.now(UTC),
            ordinary_keyframe_retention_seconds=86_400,
            skipped_event_retention_days=90,
            connection=connection,
        )

        self.assertEqual(outcome.episode_memory_id, episode_id)
        connection.execute.assert_not_awaited()
