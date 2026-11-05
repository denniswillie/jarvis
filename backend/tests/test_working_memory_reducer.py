from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.perception.models import (
    ObservationEntity,
    ObservationStateChange,
    VisibleTextObservation,
    VisualObservation,
)
from app.perception.working_memory import (
    WorkingMemoryPolicy,
    prune_session_working_memory,
    reduce_session_working_memory,
)


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
POLICY = WorkingMemoryPolicy(120, 30, 60, 3, 2)


class WorkingMemoryReducerTests(unittest.TestCase):
    def test_new_entities_are_stable_and_same_names_do_not_collapse(self) -> None:
        observation = _observation(
            entities=(_entity("mug", "cup_a"), _entity("mug", "cup_b"))
        )
        first = reduce_session_working_memory({}, observation, policy=POLICY)
        replay = reduce_session_working_memory({}, observation, policy=POLICY)
        self.assertEqual(first["entities"], replay["entities"])
        self.assertEqual(len(first["entities"]), 2)

    def test_existing_entity_continues_and_change_resolves(self) -> None:
        first = reduce_session_working_memory(
            {}, _observation(entities=(_entity("mug", "cup"),)), policy=POLICY
        )
        entity_id = next(iter(first["entities"]))
        second = reduce_session_working_memory(
            first,
            _observation(
                at=NOW + timedelta(seconds=1),
                entities=(
                    ObservationEntity(
                        name="blue mug", category="object", confidence=0.95,
                        attributes={"entity_id": entity_id, "state": "in hand"},
                    ),
                ),
                changes=(
                    ObservationStateChange(
                        subject=entity_id, attribute="state", before="on counter",
                        after="in hand", confidence=0.9,
                    ),
                ),
            ),
            policy=POLICY,
        )
        self.assertEqual(list(second["entities"]), [entity_id])
        self.assertEqual(second["entities"][entity_id]["state"], "in hand")
        self.assertEqual(second["recent_changes"][0]["entity_id"], entity_id)

    def test_collections_expire_and_remain_bounded(self) -> None:
        state: dict[str, object] = {}
        for index in range(5):
            state = reduce_session_working_memory(
                state,
                _observation(
                    at=NOW + timedelta(seconds=index),
                    entities=(_entity(f"object {index}", f"object_{index}"),),
                    visible_text=(VisibleTextObservation(
                        text=f"TEXT {index}", context=None, confidence=0.9
                    ),),
                ),
                policy=POLICY,
            )
        self.assertEqual(len(state["entities"]), 3)
        pruned = prune_session_working_memory(
            state, now=NOW + timedelta(seconds=70), policy=POLICY
        )
        self.assertEqual(pruned["entities"], {})
        self.assertEqual(pruned["visible_text"], [])

    def test_low_confidence_scene_change_is_rejected(self) -> None:
        state = reduce_session_working_memory({}, _observation(), policy=POLICY)
        changed = reduce_session_working_memory(
            state,
            _observation(at=NOW + timedelta(seconds=1), scene="street", confidence=0.2),
            policy=POLICY,
        )
        self.assertEqual(changed["scene"]["label"], "kitchen")
        self.assertEqual(changed["activity"]["started_at"], state["activity"]["started_at"])


def _entity(name: str, local_key: str) -> ObservationEntity:
    return ObservationEntity(
        name=name, category="object", confidence=0.9,
        attributes={"entity_id": None, "local_key": local_key, "state": "on counter"},
    )


def _observation(
    *, at: datetime = NOW, scene: str = "kitchen", confidence: float = 0.95,
    entities: tuple[ObservationEntity, ...] = (),
    visible_text: tuple[VisibleTextObservation, ...] = (),
    changes: tuple[ObservationStateChange, ...] = (),
) -> VisualObservation:
    return VisualObservation(
        id=uuid4(), owner_id="owner", session_id=uuid4(), source="glasses",
        started_at=at, ended_at=at, primary_visual_frame_id=uuid4(),
        summary="Making coffee in the kitchen.", scene=scene, activity="making coffee",
        entities=entities, visible_text=visible_text, state_changes=changes,
        memory_tags=(), keyframe_candidates=(), interrupt_candidate=None,
        starts_new_event=False, boundary_reason=None, confidence=confidence,
        analyzer_provider="test", analyzer_model="test", analyzer_contract_version=1,
        compacted_at=None, created_at=at,
    )


if __name__ == "__main__":
    unittest.main()
