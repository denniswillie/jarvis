from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

from app.memory.episodic.repository import EpisodicMemoryRepository
from app.memory.episodic.service import (
    EpisodePromotionPlan,
    FinalKeyframe,
    ReminderDeliveryEvidence,
)
from test_episode_promotion import _context


class EpisodeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reminder_and_frame_keep_their_actual_occurrence_times(self) -> None:
        context = _context(importance=0.9, should_promote=False)
        reminder_time = datetime.now(UTC) - timedelta(seconds=2)
        frame_time = datetime.now(UTC) - timedelta(seconds=1)
        reminder = ReminderDeliveryEvidence(
            reminder_event_id=uuid4(),
            reminder_id=uuid4(),
            occurred_at=reminder_time,
            message="Bring an umbrella.",
            metadata={"source": "scheduler"},
        )
        context = replace(
            context,
            reminder_deliveries=(reminder,),
        )
        keyframe = context.provisional_keyframes[0]
        plan = EpisodePromotionPlan(
            event_id=context.event.id,
            owner_id=context.event.owner_id,
            session_id=context.event.session_id,
            promote=True,
            reason="important",
            is_explicit=False,
            retain_keyframes_permanently=True,
            final_keyframes=(
                FinalKeyframe(
                    visual_frame_id=keyframe.visual_frame_id,
                    captured_at=frame_time,
                    rank=1,
                    reason=keyframe.reason,
                    quality_score=keyframe.quality_score,
                    selected_by="deterministic",
                ),
            ),
        )
        connection = AsyncMock()
        repository = EpisodicMemoryRepository(
            object(),  # type: ignore[arg-type]
            experience_repository=object(),  # type: ignore[arg-type]
            memory_repository=object(),  # type: ignore[arg-type]
        )

        await repository._attach_episode_evidence(
            connection,
            context=context,
            plan=plan,
            episode_id=uuid4(),
        )

        calls = connection.execute.await_args_list
        reminder_call = next(call for call in calls if call.args[3] == "reminder")
        frame_call = next(call for call in calls if call.args[3] == "frame")
        self.assertEqual(reminder_call.args[5], reminder_time)
        self.assertEqual(frame_call.args[5], frame_time)
        self.assertEqual(reminder_call.args[4], str(reminder.reminder_event_id))
