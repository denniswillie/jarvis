from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.memory.models import EMBEDDING_DIMENSIONS, normalize_embedding
from app.model_providers.protocols import VisualModelImage
from app.settings import JarvisSettings, is_configured_secret


GEMINI_EMBEDDING_PROVIDER = "gemini"
GEMINI_EMBEDDING_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_EMBEDDING_MAX_IMAGES = 6
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})


class GeminiMultimodalEmbeddingError(Exception):
    pass


class GeminiMultimodalEmbeddingNotConfiguredError(GeminiMultimodalEmbeddingError):
    pass


@dataclass(frozen=True, slots=True)
class GeminiMultimodalEmbeddingConfig:
    api_key: str = field(repr=False)
    model: str = "gemini-embedding-2"
    dimensions: int = EMBEDDING_DIMENSIONS
    timeout_seconds: float = 20.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Gemini embedding API key cannot be blank")
        if not self.model.strip():
            raise ValueError("Gemini embedding model cannot be blank")
        if self.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Gemini embedding dimensions must be {EMBEDDING_DIMENSIONS}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError("Gemini embedding timeout must be positive")
        if self.max_attempts < 1:
            raise ValueError("Gemini embedding attempts must be positive")


def gemini_multimodal_embedding_config_from_settings(
    settings: JarvisSettings,
) -> GeminiMultimodalEmbeddingConfig:
    if not is_configured_secret(settings.GEMINI_API_KEY):
        raise GeminiMultimodalEmbeddingNotConfiguredError(
            "GEMINI_API_KEY is not configured"
        )
    model = settings.GEMINI_MEMORY_EMBEDDING_MODEL.strip()
    if not model:
        raise GeminiMultimodalEmbeddingNotConfiguredError(
            "GEMINI_MEMORY_EMBEDDING_MODEL is not configured"
        )
    return GeminiMultimodalEmbeddingConfig(
        api_key=settings.GEMINI_API_KEY.strip(),  # type: ignore[union-attr]
        model=model,
        dimensions=settings.MEMORY_EMBEDDING_DIMENSIONS,
        timeout_seconds=settings.MEMORY_EMBEDDING_TIMEOUT_SECONDS,
    )


class GeminiMultimodalEmbeddingModel:
    """Direct Gemini Embedding 2 adapter for text and bounded JPEG sequences."""

    def __init__(
        self,
        config: GeminiMultimodalEmbeddingConfig,
        *,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._config = config
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_client = client is None
        self._sleep = sleep

    @property
    def provider(self) -> str:
        return GEMINI_EMBEDDING_PROVIDER

    @property
    def model(self) -> str:
        return self._config.model

    async def embed_multimodal(
        self,
        *,
        text: str | None,
        images: Sequence[VisualModelImage],
    ) -> list[float]:
        normalized_text = text.strip() if text is not None else ""
        normalized_images = tuple(images)
        if not normalized_text and not normalized_images:
            raise ValueError("Gemini embedding input cannot be empty")
        if len(normalized_images) > GEMINI_EMBEDDING_MAX_IMAGES:
            raise ValueError(
                f"Gemini embedding accepts at most {GEMINI_EMBEDDING_MAX_IMAGES} images"
            )

        parts: list[dict[str, Any]] = []
        if normalized_text:
            parts.append({"text": normalized_text})
        for index, image in enumerate(normalized_images, start=1):
            label = image.label.strip()
            if not label:
                raise ValueError("Gemini embedding image label cannot be blank")
            if image.content_type != "image/jpeg":
                raise ValueError("Gemini embedding images must be JPEG")
            if not _is_jpeg(image.image_bytes):
                raise ValueError("Gemini embedding image bytes are not a JPEG")
            parts.extend(
                (
                    {"text": f"Image {index}: {label}"},
                    {
                        "inlineData": {
                            "mimeType": image.content_type,
                            "data": base64.b64encode(image.image_bytes).decode("ascii"),
                        }
                    },
                )
            )

        payload = {
            "content": {"parts": parts},
            "outputDimensionality": self._config.dimensions,
        }
        response = await self._post_with_retry(payload)
        values = _embedding_values(response.json())
        return list(normalize_embedding(values))

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post_with_retry(self, payload: Mapping[str, Any]) -> httpx.Response:
        url = (
            f"{GEMINI_EMBEDDING_API_BASE_URL}/models/"
            f"{self._config.model}:embedContent"
        )
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                response = await self._client.post(
                    url,
                    headers={"x-goog-api-key": self._config.api_key},
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
            except (httpx.TimeoutException, httpx.TransportError) as error:
                if attempt == self._config.max_attempts:
                    raise GeminiMultimodalEmbeddingError(
                        "Gemini embedding request failed"
                    ) from error
            else:
                if response.status_code < 400:
                    return response
                if (
                    response.status_code not in _RETRYABLE_STATUS_CODES
                    or attempt == self._config.max_attempts
                ):
                    detail = response.text.strip()[:500]
                    raise GeminiMultimodalEmbeddingError(
                        f"Gemini embedding request failed with HTTP "
                        f"{response.status_code}: {detail or 'no response body'}"
                    )
            await self._sleep(0.25 * attempt)
        raise AssertionError("Gemini embedding retry loop exited unexpectedly")


def _embedding_values(payload: Mapping[str, Any]) -> Sequence[float]:
    embedding = payload.get("embedding")
    if not isinstance(embedding, Mapping):
        raise GeminiMultimodalEmbeddingError(
            "Gemini embedding response is missing embedding"
        )
    values = embedding.get("values")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise GeminiMultimodalEmbeddingError(
            "Gemini embedding response is missing numeric values"
        )
    return values


def _is_jpeg(value: bytes) -> bool:
    return len(value) >= 4 and value[:3] == b"\xff\xd8\xff" and value[-2:] == b"\xff\xd9"
