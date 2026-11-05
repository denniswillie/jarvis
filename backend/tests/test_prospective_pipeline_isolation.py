from __future__ import annotations

import unittest
from uuid import uuid4

from app.visual_memory.pipeline import VisualAnalysisPipeline


class _FailingProspectiveService:
    async def process_matches(self, **kwargs):
        raise RuntimeError("delivery unavailable")


class ProspectivePipelineIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_failure_is_best_effort_after_frame_completion(self) -> None:
        pipeline = object.__new__(VisualAnalysisPipeline)
        pipeline._prospective_service = _FailingProspectiveService()

        await pipeline._process_prospective_matches_best_effort(
            request=object(),
            matches=[object()],
            frame_id=uuid4(),
        )
