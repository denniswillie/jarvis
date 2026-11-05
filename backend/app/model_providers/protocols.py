from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class VisualModelImage:
    label: str
    image_bytes: bytes
    content_type: str


@runtime_checkable
class StructuredOutputModel(Protocol):
    async def generate_json(self, *, prompt: str) -> str: ...

    async def close(self) -> None: ...


@runtime_checkable
class TextEmbeddingModel(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, text: str) -> list[float]: ...

    async def close(self) -> None: ...


@runtime_checkable
class MultimodalEmbeddingModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def embed_multimodal(
        self,
        *,
        text: str | None,
        images: Sequence[VisualModelImage],
    ) -> list[float]: ...

    async def close(self) -> None: ...


@runtime_checkable
class VisualInspectionModel(Protocol):
    async def generate_json(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        content_type: str,
    ) -> str: ...

    async def close(self) -> None: ...


@runtime_checkable
class TemporalVisualInspectionModel(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def generate_temporal_json(
        self,
        *,
        prompt: str,
        images: Sequence[VisualModelImage],
        response_json_schema: dict[str, object] | None = None,
    ) -> str: ...
