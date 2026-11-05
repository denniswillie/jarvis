from __future__ import annotations

import asyncio
import json
import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.perception.contracts import FastPerceptionFrame, FastPerceptionRequest
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.perception.session_analysis_coordinator import SessionAnalysisCoordinator


class _Model:
    provider = "test"
    model = "test-model"

    async def generate_temporal_json(self, *, prompt, images):
        frame_id = prompt.split('"primary_frame_id": "', 1)[1].split('"', 1)[0]
        return "```json\n" + json.dumps(
            {
                "summary": "A static desk.",
                "scene": {"label": "desk", "confidence": 0.9},
                "activity": None,
                "entities": [],
                "visible_text": [],
                "state_changes": [],
                "memory_tags": ["desk"],
                "keyframe_candidates": [
                    {"frame_id": frame_id, "reason": "clear", "score": 0.8}
                ],
                "starts_new_event": False,
                "boundary_reason": None,
                "interrupt_candidate": None,
                "confidence": 0.9,
            }
        ) + "\n```"


class FastPerceptionParserTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyzer_accepts_fenced_json_and_preserves_image_order(self) -> None:
        analyzer = FastPerceptionAnalyzer(_Model())
        result = await analyzer.analyze(_request())
        self.assertEqual(result.summary, "A static desk.")

    async def test_coordinator_replaces_pending_with_latest(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        processed: list[int] = []

        async def process(value: int) -> None:
            processed.append(value)
            if value == 1:
                started.set()
                await release.wait()

        coordinator = SessionAnalysisCoordinator[int](process)
        session_id = uuid4()
        await coordinator.submit(session_id, 1)
        await started.wait()
        self.assertIsNone(await coordinator.submit(session_id, 2))
        self.assertEqual(await coordinator.submit(session_id, 3), 2)
        release.set()
        await coordinator.drain()

        self.assertEqual(processed, [1, 3])

    async def test_processor_failure_does_not_block_latest_pending_item(self) -> None:
        first_started = asyncio.Event()
        release = asyncio.Event()
        processed: list[int] = []

        async def process(value: int) -> None:
            processed.append(value)
            if value == 1:
                first_started.set()
                await release.wait()
                raise RuntimeError("expected test failure")

        coordinator = SessionAnalysisCoordinator[int](process)
        session_id = uuid4()
        await coordinator.submit(session_id, 1)
        await first_started.wait()
        await coordinator.submit(session_id, 2)
        release.set()
        await coordinator.drain()

        self.assertEqual(processed, [1, 2])


def _request() -> FastPerceptionRequest:
    now = datetime.now(UTC)
    return FastPerceptionRequest(
        owner_id="owner",
        session_id=uuid4(),
        window_id=uuid4(),
        frames=(
            FastPerceptionFrame(
                uuid4(), 1, now, "glasses", "image/jpeg", b"image"
            ),
        ),
        previous_working_state={},
        existing_entity_ids=frozenset(),
        recent_audio_transcripts=(),
        behavior_instructions="",
        active_reminders=(),
        active_focus=None,
        recent_interrupts=(),
        requested_at=now,
    )
