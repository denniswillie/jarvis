from __future__ import annotations

import logging

from app.model_providers.protocols import TemporalVisualInspectionModel, VisualModelImage
from app.perception.contracts import (
    FastPerceptionOutput,
    FastPerceptionRequest,
)
from app.perception.prompts import build_fast_perception_prompt, frame_attachment_label
from app.perception.references import PerceptionReferences, REFERENCE_CONTRACT_VERSION
from app.perception.result_validation import parse_referenced_perception_output


LOGGER = logging.getLogger(__name__)


class FastPerceptionAnalyzer:
    def __init__(self, model: TemporalVisualInspectionModel) -> None:
        self._model = model

    @property
    def provider(self) -> str:
        return self._model.provider

    @property
    def model(self) -> str:
        return self._model.model

    @property
    def temporal_model(self) -> TemporalVisualInspectionModel:
        """Borrow the existing visual client for bounded predicate evaluation."""
        return self._model

    async def analyze(self, request: FastPerceptionRequest) -> FastPerceptionOutput:
        references = PerceptionReferences(request)
        raw = await self._model.generate_temporal_json(
            prompt=build_fast_perception_prompt(request, references=references),
            images=tuple(
                VisualModelImage(
                    label=frame_attachment_label(index, references.frame_label(frame.frame_id)),
                    image_bytes=frame.image_bytes,
                    content_type=frame.content_type,
                )
                for index, frame in enumerate(request.frames)
            ),
        )
        result = parse_referenced_perception_output(raw, request=request, references=references)
        for candidate in result.rejected:
            LOGGER.info("visual candidate rejected contract=%s window_id=%s field=%s index=%s reason=%s",
                        REFERENCE_CONTRACT_VERSION, request.window_id,
                        candidate.field, candidate.index, candidate.reason)
        return result.output
