from __future__ import annotations

import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.memory.reflection.contracts import ProcedureMemoryCandidate
from app.memory.reflection.repository import (
    _procedure_rejection_reason,
    normalize_procedure_key,
)


class ProcedureCandidateTests(unittest.TestCase):
    def test_procedure_key_is_stable_across_case_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_procedure_key("  Morning   Coffee  "),
            normalize_procedure_key("morning coffee"),
        )

    def test_inferred_procedure_requires_three_distinct_episodes(self) -> None:
        episode_ids = (uuid4(), uuid4())
        candidate = _candidate(episode_ids=episode_ids, is_explicit=False)
        episodes = tuple(
            SimpleNamespace(id=episode_id, is_explicit=False)
            for episode_id in episode_ids
        )

        reason = _procedure_rejection_reason(
            candidate,
            episode_ids=episode_ids,
            episodes=episodes,
            minimum_episodes=3,
        )

        self.assertEqual(reason, "insufficient_distinct_episodes")

    def test_explicit_procedure_can_use_one_explicit_episode(self) -> None:
        episode_ids = (uuid4(),)
        candidate = _candidate(episode_ids=episode_ids, is_explicit=True)
        episodes = (SimpleNamespace(id=episode_ids[0], is_explicit=True),)

        reason = _procedure_rejection_reason(
            candidate,
            episode_ids=episode_ids,
            episodes=episodes,
            minimum_episodes=3,
        )

        self.assertIsNone(reason)

    def test_explicit_claim_without_explicit_evidence_is_rejected(self) -> None:
        episode_ids = (uuid4(), uuid4(), uuid4())
        candidate = _candidate(episode_ids=episode_ids, is_explicit=True)
        episodes = tuple(
            SimpleNamespace(id=episode_id, is_explicit=False)
            for episode_id in episode_ids
        )

        reason = _procedure_rejection_reason(
            candidate,
            episode_ids=episode_ids,
            episodes=episodes,
            minimum_episodes=3,
        )

        self.assertEqual(reason, "explicit_claim_without_explicit_evidence")


def _candidate(
    *,
    episode_ids: tuple,
    is_explicit: bool,
) -> ProcedureMemoryCandidate:
    return ProcedureMemoryCandidate(
        procedure_key="morning coffee",
        title="Make morning coffee",
        summary="The user's repeated coffee routine.",
        steps=["Grind beans", "Brew coffee"],
        confidence=0.92,
        is_explicit=is_explicit,
        supporting_episode_ids=[str(value) for value in episode_ids],
    )


if __name__ == "__main__":
    unittest.main()
