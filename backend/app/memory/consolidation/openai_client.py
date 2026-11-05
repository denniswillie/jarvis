from __future__ import annotations

import asyncio
import base64
import copy
import inspect
from collections.abc import Mapping
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAIError

from app.memory.consolidation.contracts import ConsolidationRequest, ConsolidationResult
from app.memory.consolidation.prompts import (
    CONSOLIDATION_INSTRUCTIONS,
    build_consolidation_prompt,
    keyframe_attachment_label,
)
from app.realtime.openai_client import safety_identifier_for_owner
from app.runtime.concurrency import ModelCallPriority, PriorityAsyncLimiter


class ConsolidationProviderTransientError(RuntimeError):
    pass


class ConsolidationProviderContractError(ValueError):
    pass


class OpenAIStructuredOutputClient:
    """Isolated Responses API client for bounded, schema-constrained consolidation."""

    provider = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        limiter: PriorityAsyncLimiter,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key cannot be blank")
        if not model.strip():
            raise ValueError("consolidation model cannot be blank")
        if timeout_seconds <= 0:
            raise ValueError("consolidation timeout must be positive")
        self.model = model.strip()
        self._timeout_seconds = timeout_seconds
        self._limiter = limiter
        self._client = client or AsyncOpenAI(api_key=api_key)

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def generate(
        self,
        request: ConsolidationRequest,
        *,
        owner_id: str,
        validation_feedback: str | None = None,
    ) -> str:
        prompt = build_consolidation_prompt(request)
        if validation_feedback:
            prompt += (
                "\n\nCorrection required after contract validation:\n"
                + validation_feedback.strip()
            )
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt}
        ]
        for keyframe in request.keyframes:
            if keyframe.image_bytes is None:
                continue
            encoded = base64.b64encode(keyframe.image_bytes).decode("ascii")
            content.extend(
                (
                    {
                        "type": "input_text",
                        "text": keyframe_attachment_label(keyframe.visual_frame_id),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{keyframe.content_type};base64,{encoded}",
                    },
                )
            )
        return await self._generate_structured_output(
            content=content,
            instructions=CONSOLIDATION_INSTRUCTIONS,
            schema_name="jarvis_experience_consolidation",
            schema=ConsolidationResult.model_json_schema(),
            owner_id=owner_id,
            max_output_tokens=4_000,
        )

    async def generate_json(
        self,
        *,
        prompt: str,
        instructions: str,
        schema_name: str,
        schema: Mapping[str, Any],
        owner_id: str,
        max_output_tokens: int,
    ) -> str:
        """Generate schema-constrained JSON for a bounded memory task."""

        if not prompt.strip():
            raise ValueError("structured-output prompt cannot be blank")
        if not instructions.strip():
            raise ValueError("structured-output instructions cannot be blank")
        if not schema_name.strip():
            raise ValueError("structured-output schema name cannot be blank")
        if max_output_tokens < 1:
            raise ValueError("structured-output token limit must be positive")
        return await self._generate_structured_output(
            content=[{"type": "input_text", "text": prompt}],
            instructions=instructions,
            schema_name=schema_name,
            schema=schema,
            owner_id=owner_id,
            max_output_tokens=max_output_tokens,
        )

    async def _generate_structured_output(
        self,
        *,
        content: list[dict[str, Any]],
        instructions: str,
        schema_name: str,
        schema: Mapping[str, Any],
        owner_id: str,
        max_output_tokens: int,
    ) -> str:
        try:
            async with self._limiter.limit(ModelCallPriority.NORMAL):
                async with asyncio.timeout(self._timeout_seconds):
                    response = await self._client.responses.create(
                        model=self.model,
                        instructions=instructions,
                        input=[{"role": "user", "content": content}],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": schema_name,
                                "strict": True,
                                "schema": strict_json_schema(schema),
                            }
                        },
                        max_output_tokens=max_output_tokens,
                        store=False,
                        safety_identifier=safety_identifier_for_owner(owner_id),
                    )
        except (TimeoutError, APIConnectionError) as error:
            raise ConsolidationProviderTransientError(
                "OpenAI consolidation request failed"
            ) from error
        except APIStatusError as error:
            if error.status_code in {408, 409, 429} or error.status_code >= 500:
                raise ConsolidationProviderTransientError(
                    "OpenAI consolidation request failed"
                ) from error
            raise ConsolidationProviderContractError(
                "OpenAI rejected the consolidation request"
            ) from error
        except OpenAIError as error:
            raise ConsolidationProviderContractError(
                "OpenAI consolidation request is invalid"
            ) from error
        output_text = getattr(response, "output_text", "")
        if not isinstance(output_text, str) or not output_text.strip():
            raise ConsolidationProviderContractError(
                "OpenAI consolidation response was empty"
            )
        return output_text


def strict_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return an OpenAI strict-compatible copy of a Pydantic JSON schema."""

    normalized = copy.deepcopy(dict(schema))
    _normalize_schema_node(normalized)
    return normalized


def _normalize_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_schema_node(item)
        return
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
    if node.get("type") == "object":
        node.setdefault("additionalProperties", False)
    if node.get("default") is None:
        node.pop("default", None)

    for value in tuple(node.values()):
        _normalize_schema_node(value)
