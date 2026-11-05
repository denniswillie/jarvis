from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.settings import JarvisSettings, is_configured_secret
from app.model_providers.protocols import VisualModelImage


GEMINI_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiVisualAnalyzerNotConfigured(RuntimeError):
    pass


class GeminiVisualAnalyzerError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiVisualAnalyzerConfig:
    api_key: str = field(repr=False)
    model: str


def gemini_visual_config_from_settings(settings: JarvisSettings) -> GeminiVisualAnalyzerConfig | None:
    if not is_configured_secret(settings.GEMINI_API_KEY):
        return None
    if not is_configured_secret(settings.GEMINI_VISUAL_MODEL):
        return None
    return GeminiVisualAnalyzerConfig(
        api_key=settings.GEMINI_API_KEY or "",
        model=settings.GEMINI_VISUAL_MODEL or "",
    )


class GeminiVisualAnalyzerClient:
    def __init__(self, config: GeminiVisualAnalyzerConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0))

    async def close(self) -> None:
        await self._client.aclose()

    @property
    def model(self) -> str:
        return self._config.model

    async def generate_json(self, *, prompt: str, image_bytes: bytes, content_type: str) -> str:
        return await self._generate_json(
            [
                {"text": prompt},
                _inline_image_part(image_bytes=image_bytes, content_type=content_type),
            ]
        )

    async def generate_multiframe_json(
        self,
        *,
        prompt: str,
        images: Sequence[VisualModelImage],
        response_json_schema: dict[str, object] | None = None,
    ) -> str:
        if not prompt.strip() and not images:
            raise ValueError("Gemini analysis requires text or at least one image.")
        if any(not image.image_bytes or not image.content_type.strip() for image in images):
            raise ValueError("Gemini image attachments require bytes and a content type.")
        parts: list[dict[str, object]] = [{"text": prompt}]
        for image in images:
            parts.append({"text": image.label})
            parts.append(
                _inline_image_part(
                    image_bytes=image.image_bytes,
                    content_type=image.content_type,
                )
            )
        return await self._generate_json(parts, response_json_schema=response_json_schema)

    async def _generate_json(
        self, parts: list[dict[str, object]], *, response_json_schema: dict[str, object] | None = None,
    ) -> str:
        generation_config: dict[str, object] = {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        }
        if response_json_schema is not None:
            generation_config["responseJsonSchema"] = response_json_schema
        try:
            response = await self._client.post(
                GEMINI_GENERATE_CONTENT_URL.format(model=self._config.model),
                headers={"x-goog-api-key": self._config.api_key},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": parts,
                        }
                    ],
                    "generationConfig": generation_config,
                },
            )
        except httpx.TimeoutException:
            raise httpx.TimeoutException("Gemini visual analyzer request timed out.") from None
        except httpx.HTTPError:
            raise httpx.RequestError("Gemini visual analyzer transport failed.") from None
        if response.status_code >= 400:
            # Provider bodies may echo credentials, prompts, or image payloads.
            raise GeminiVisualAnalyzerError(
                f"Gemini visual analyzer failed with status {response.status_code}."
            )
        try:
            payload = response.json()
        except ValueError:
            raise GeminiVisualAnalyzerError("Gemini visual analyzer returned invalid JSON.") from None
        if not isinstance(payload, dict):
            raise GeminiVisualAnalyzerError("Gemini visual analyzer returned an invalid response.")
        text = _extract_text(payload)
        if not text.strip():
            raise GeminiVisualAnalyzerError("Gemini visual analyzer returned an empty response.")
        return text


def _inline_image_part(*, image_bytes: bytes, content_type: str) -> dict[str, object]:
    return {
        "inlineData": {
            "mimeType": content_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        }
    }


def _extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""
    first = candidates[0]
    if not isinstance(first, dict):
        return ""
    content = first.get("content")
    if not isinstance(content, dict):
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "\n".join(chunks)
