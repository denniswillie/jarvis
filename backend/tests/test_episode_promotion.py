from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from app.experience.models import ExperienceEventKeyframe
from app.memory.consolidation.contracts import parse_consolidation_result
from app.memory.episodic.service import (
    EpisodePromotionContext,
    EpisodePromotionService,
    ReminderDeliveryEvidence,
)
from test_consolidation_contracts import _output, _request


class EpisodePromotionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = EpisodePromotionService(
            minimum_importance=0.55,
            important_keyframe_minimum_importance=0.8,
        )

    def test_validated_model_recommendation_promotes_episode(self) -> None:
        context = _context(importance=0.6, should_promote=True)

        plan = self.service.plan(context)

        self.assertTrue(plan.promote)
        self.assertEqual(plan.reason, "validated consolidation recommended promotion")
        self.assertFalse(plan.retain_keyframes_permanently)

    def test_low_value_event_is_skipped_even_when_model_recommends_it(self) -> None:
        context = _context(importance=0.4, should_promote=True)

        plan = self.service.plan(context)

        self.assertFalse(plan.promote)
        self.assertEqual(plan.reason, "routine event below promotion policy")

    def test_explicit_memory_request_bypasses_importance_gate(self) -> None:
        context = replace(
            _context(importance=0.1, should_promote=False),
            has_explicit_memory_evidence=True,
        )

        plan = self.service.plan(context)

        self.assertTrue(plan.promote)
        self.assertTrue(plan.is_explicit)
        self.assertTrue(plan.retain_keyframes_permanently)

    def test_important_event_keeps_at_most_three_ordered_keyframes(self) -> None:
        context = _context(importance=0.9, should_promote=False, keyframe_count=5)

        plan = self.service.plan(context)

        self.assertTrue(plan.promote)
        self.assertTrue(plan.retain_keyframes_permanently)
        self.assertEqual(len(plan.final_keyframes), 3)
        self.assertEqual(
            str(plan.final_keyframes[0].visual_frame_id),
            context.result.selected_keyframe_ids[0],
        )
        self.assertEqual(plan.final_keyframes[0].selected_by, "gpt")
        self.assertEqual(
            [keyframe.rank for keyframe in plan.final_keyframes],
            [1, 2, 3],
        )
        self.assertEqual(
            plan.final_keyframes[0].captured_at,
            context.keyframe_capture_times[plan.final_keyframes[0].visual_frame_id],
        )

    def test_completed_mutating_tool_uses_durable_tool_metadata(self) -> None:
        context = _context(importance=0.1, should_promote=False)
        execution_id = uuid4()
        evidence = _tool_evidence(context, execution_id)
        context = replace(
            context,
            evidence=(evidence,),
            source_payloads={
                ("tool_execution", execution_id): {
                    "tool_name": "future_mutation_tool",
                    "status": "completed",
                    "mutates_state": True,
                }
            },
        )

        plan = self.service.plan(context)

        self.assertTrue(plan.promote)
        self.assertIn("future_mutation_tool", plan.reason)

    def test_reminder_delivery_promotes_without_losing_trigger_identity(self) -> None:
        now = datetime.now(UTC)
        reminder = ReminderDeliveryEvidence(
            reminder_event_id=uuid4(),
            reminder_id=uuid4(),
            occurred_at=now,
            message="Take your medication.",
            metadata={"source": "scheduler"},
        )
        context = replace(
            _context(importance=0.1, should_promote=False),
            reminder_deliveries=(reminder,),
        )

        plan = self.service.plan(context)

        self.assertTrue(plan.promote)
        self.assertTrue(context.has_reminder_delivery)

    def test_non_finite_policy_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            EpisodePromotionService(
                minimum_importance=float("nan"),
                important_keyframe_minimum_importance=0.8,
            )


def _context(
    *,
    importance: float,
    should_promote: bool,
    keyframe_count: int = 1,
) -> EpisodePromotionContext:
    request = _request()
    payload = _output(request)
    frame_ids = [uuid4() for _ in range(keyframe_count)]
    payload["episode_importance"] = importance
    payload["should_promote_to_episode"] = should_promote
    payload["selected_keyframe_ids"] = [str(frame_ids[-1])]
    result = parse_consolidation_result(
        payload,
        request=replace(
            request,
            keyframes=tuple(
                replace(request.keyframes[0], visual_frame_id=frame_id)
                for frame_id in frame_ids
            ),
        ),
    )
    now = datetime.now(UTC)
    event = replace(
        request.event,
        status="consolidated",
        title=result.event_title,
        summary=result.event_summary,
        importance=result.episode_importance,
        confidence=result.confidence,
        consolidation_result=result.model_dump(mode="json"),
        consolidation_provider="openai",
        consolidation_model="test-gpt",
        consolidated_at=now,
    )
    keyframes = tuple(
        ExperienceEventKeyframe(
            event_id=event.id,
            visual_frame_id=frame_id,
            rank=index,
            reason=f"candidate {index}",
            quality_score=0.9 - index / 100,
            selected_by="deterministic",
            created_at=now,
        )
        for index, frame_id in enumerate(frame_ids, start=1)
    )
    return EpisodePromotionContext(
        event=event,
        result=result,
        evidence=(),
        source_payloads={},
        context_messages=(),
        provisional_keyframes=keyframes,
        keyframe_capture_times={
            keyframe.visual_frame_id: now for keyframe in keyframes
        },
        has_explicit_memory_evidence=False,
        reminder_deliveries=(),
    )


def _tool_evidence(context: EpisodePromotionContext, execution_id):  # type: ignore[no-untyped-def]
    from app.experience.models import ExperienceEventEvidence

    return ExperienceEventEvidence(
        id=uuid4(),
        event_id=context.event.id,
        evidence_type="tool_execution",
        evidence_ref_id=execution_id,
        occurred_at=context.event.started_at,
        source_sequence=None,
        ingested_at=context.event.started_at,
        metadata={},
    )
