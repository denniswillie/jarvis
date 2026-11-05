from __future__ import annotations

import inspect
import unittest

from app.model_providers.protocols import (
    MultimodalEmbeddingModel,
    StructuredOutputModel,
    TextEmbeddingModel,
    VisualModelImage,
    VisualInspectionModel,
)


class _StructuredOutputAdapter:
    async def generate_json(self, *, prompt: str) -> str:
        return "{}"

    async def close(self) -> None:
        return None


class _TextEmbeddingAdapter:
    model = "embedding-model"

    async def embed(self, text: str) -> list[float]:
        return [0.0]

    async def close(self) -> None:
        return None


class _MultimodalEmbeddingAdapter:
    provider = "provider"
    model = "multimodal-model"

    async def embed_multimodal(
        self,
        *,
        text: str | None,
        images: tuple[VisualModelImage, ...],
    ) -> list[float]:
        return [0.0]

    async def close(self) -> None:
        return None


class _VisualInspectionAdapter:
    async def generate_json(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        content_type: str,
    ) -> str:
        return "{}"

    async def close(self) -> None:
        return None


class ModelProviderProtocolTests(unittest.TestCase):
    def test_plain_adapters_satisfy_provider_protocols(self) -> None:
        self.assertIsInstance(_StructuredOutputAdapter(), StructuredOutputModel)
        self.assertIsInstance(_TextEmbeddingAdapter(), TextEmbeddingModel)
        self.assertIsInstance(_MultimodalEmbeddingAdapter(), MultimodalEmbeddingModel)
        self.assertIsInstance(_VisualInspectionAdapter(), VisualInspectionModel)

    def test_protocol_annotations_do_not_expose_provider_sdk_types(self) -> None:
        protocol_types = (
            StructuredOutputModel,
            TextEmbeddingModel,
            MultimodalEmbeddingModel,
            VisualInspectionModel,
        )
        annotations = " ".join(
            str(signature.annotation)
            for protocol_type in protocol_types
            for _, method in inspect.getmembers(protocol_type, inspect.isfunction)
            for signature in inspect.signature(method).parameters.values()
        )
        self.assertNotIn("openai", annotations.lower())
        self.assertNotIn("google", annotations.lower())
        self.assertNotIn("httpx", annotations.lower())
