from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from app.memory.reflection.contracts import InferredPreferenceCandidate
from app.memory.reflection.job_handler import ReflectionRecoveryRule
from app.memory.reflection.repository import (
    ReflectionEpisode,
    ReflectionEpisodeRange,
    _preference_rejection_reason,
)
from app.memory.reflection.service import ReflectionTriggerService


class ReflectionPreferenceThresholdTests(unittest.TestCase):
    def test_inferred_preference_requires_three_distinct_episodes(self) -> None:
        episodes = (uuid4(), uuid4())
        candidate = InferredPreferenceCandidate(
            key="preferred_drink",
            value="tea",
            confidence=0.95,
            supporting_episode_ids=[str(value) for value in episodes],
        )

        reason = _preference_rejection_reason(
            candidate,
            episode_ids=episodes,
            minimum_episodes=3,
            minimum_confidence=0.85,
        )

        self.assertEqual(reason, "insufficient_distinct_episodes")

    def test_inferred_preference_requires_confidence_threshold(self) -> None:
        episodes = (uuid4(), uuid4(), uuid4())
        candidate = InferredPreferenceCandidate(
            key="preferred_drink",
            value="tea",
            confidence=0.84,
            supporting_episode_ids=[str(value) for value in episodes],
        )

        reason = _preference_rejection_reason(
            candidate,
            episode_ids=episodes,
            minimum_episodes=3,
            minimum_confidence=0.85,
        )

        self.assertEqual(reason, "confidence_below_threshold")

    def test_inferred_preference_accepts_sufficient_evidence(self) -> None:
        episodes = (uuid4(), uuid4(), uuid4())
        candidate = InferredPreferenceCandidate(
            key="preferred_drink",
            value="tea",
            confidence=0.85,
            supporting_episode_ids=[str(value) for value in episodes],
        )

        reason = _preference_rejection_reason(
            candidate,
            episode_ids=episodes,
            minimum_episodes=3,
            minimum_confidence=0.85,
        )

        self.assertIsNone(reason)


class _TriggerConnection:
    async def execute(self, query, *arguments):
        return "SELECT 1"


class _TriggerRepository:
    def __init__(self, episode_range: ReflectionEpisodeRange) -> None:
        self.episode_range = episode_range

    async def find_eligible_range(self, **kwargs):
        return self.episode_range

    async def list_episode_owners(self, **kwargs):
        return (self.episode_range.owner_id,)


class _TriggerJobRepository:
    def __init__(self) -> None:
        self.calls = []

    async def enqueue(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(created=True)


class ReflectionTriggerTests(unittest.IsolatedAsyncioTestCase):
    async def test_promotion_trigger_enqueues_stable_episode_range(self) -> None:
        episode_range = _episode_range()
        jobs = _TriggerJobRepository()
        trigger = ReflectionTriggerService(
            pool=None,
            repository=_TriggerRepository(episode_range),
            job_repository=jobs,
            minimum_episodes=3,
            maximum_episodes=30,
        )

        result = await trigger.enqueue_if_eligible(
            owner_id=episode_range.owner_id,
            now=datetime.now(UTC),
            connection=_TriggerConnection(),
        )

        self.assertTrue(result.created)
        self.assertEqual(len(jobs.calls), 1)
        self.assertEqual(
            jobs.calls[0]["payload"]["episode_ids"],
            [str(value) for value in episode_range.episode_ids],
        )

    async def test_startup_recovery_reuses_the_trigger_service(self) -> None:
        episode_range = _episode_range()
        jobs = _TriggerJobRepository()
        trigger = ReflectionTriggerService(
            pool=None,
            repository=_TriggerRepository(episode_range),
            job_repository=jobs,
            minimum_episodes=3,
            maximum_episodes=30,
        )
        recovery = ReflectionRecoveryRule(trigger)

        created = await recovery.reconcile(
            repository=jobs,
            connection=_TriggerConnection(),
            now=datetime.now(UTC),
        )

        self.assertEqual(created, 1)
        self.assertEqual(len(jobs.calls), 1)


def _episode_range() -> ReflectionEpisodeRange:
    now = datetime.now(UTC)
    episodes = tuple(
        ReflectionEpisode(
            id=uuid4(),
            owner_id="owner",
            session_id=uuid4(),
            title=f"Episode {index}",
            summary="Summary",
            content="Content",
            occurred_at=now,
            created_at=now,
            is_explicit=False,
        )
        for index in range(3)
    )
    return ReflectionEpisodeRange(owner_id="owner", episodes=episodes)


if __name__ == "__main__":
    unittest.main()
