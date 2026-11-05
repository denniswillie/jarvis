"""Shared Jarvis context and exact-source image preparation for Gemini only."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import time
from typing import Any

from app.assistant.context import AssistantContextBuilder
from app.images import ImageUnavailableError, ResolvedImage
from app.realtime.gemini.client import GeminiTransportError


@dataclass(frozen=True, slots=True)
class PreparedImage:
    selection: dict[str, Any]
    resolved: ResolvedImage
    tool_name: str


class GeminiContext:
    def __init__(self, *, context, settings, repository, providers, registry) -> None:
        self.context, self.settings, self.repository = context, settings, repository
        self.providers, self.registry = tuple(providers), registry
        self.builder = AssistantContextBuilder(providers=self.providers)
        self.delivered_image: PreparedImage | None = None
        self.delivered_namespace: int | None = None

    async def shared_payload(self, *, include_history: bool = False) -> dict[str, Any] | None:
        if not self.providers:
            return None
        messages = (await self.repository.get_recent_messages(
            self.context.owner_id, self.context.jarvis_session_id,
            limit=self.settings.JARVIS_REALTIME_RECENT_MESSAGES_LIMIT,
        )) if include_history else []
        # Unknown/partial Gemini assistant output is diagnostic, not conversation
        # history. Keep the shared builder and GPT eligibility rules unchanged.
        messages = [message for message in messages
                    if message.metadata.get("audio_session_id") != self.context.audio_session_id
                    and not (message.role == "assistant" and message.metadata.get("audio_provider") == "gemini"
                             and message.metadata.get("experience_eligible") is not True)]
        # Native history already contains this run's speech. Re-injecting an
        # earlier command as text beside the next audio activity can make it look
        # like a renewed request, especially after a cancelled native tool call.
        assistant = await self.builder.build(
            owner_id=self.context.owner_id, session=self.context.session,
            recent_messages=messages,
        )
        return assistant.as_prompt_payload()

    async def select_current(self, *, namespace: int, not_after: datetime | None = None) -> tuple[PreparedImage | None, str]:
        if not self.settings.VISUAL_REALTIME_ENABLED:
            return None, "automatic_delivery_disabled"
        if self.delivered_image is not None and self.delivered_namespace == namespace:
            try:
                await self.revalidate(self.delivered_image)
                if self._within_cutoff(self.delivered_image, not_after):
                    return self.delivered_image, "reuse"
            except ImageUnavailableError:
                pass
        definition = self.registry.get("get_current_camera_view")
        if definition is None or definition.image_resolver is None:
            return None, "no_camera_capability"
        # This is read-only frame selection, not a manufactured model tool call.
        selection = await definition.handler({})
        try:
            image = await self.resolve_tool_image(definition.name, selection)
            if image is not None and not self._within_cutoff(image, not_after):
                return None, "image_after_cutoff"
            return image, "selected" if image is not None else "no_current_image"
        except ImageUnavailableError as error:
            return None, error.code

    @staticmethod
    def _within_cutoff(image: PreparedImage, cutoff: datetime | None) -> bool:
        if cutoff is None:
            return True
        reference = image.resolved.reference
        # Use server receipt/sample clocks for the response boundary. Capture
        # clock uncertainty and age are independently checked by the resolver.
        return all(value is not None and value <= cutoff for value in (
            reference.server_received_at, reference.sampled_at,
        ))

    async def resolve_tool_image(self, tool_name: str, result: dict) -> PreparedImage | None:
        definition = self.registry.get(tool_name)
        if definition is None or definition.image_resolver is None:
            return None
        if result.get("status") != "success" or result.get("image_status") != "selected":
            return None
        resolved = await definition.image_resolver(result)
        prepared = PreparedImage(dict(result), resolved, tool_name)
        self._validate_scope_and_size(prepared)
        return prepared

    def _validate_scope_and_size(self, image: PreparedImage) -> None:
        reference = image.resolved.reference
        if reference.owner_id != self.context.owner_id or (
            reference.source == "current" and reference.session_id != self.context.jarvis_session_id
        ):
            raise ImageUnavailableError("frame_unavailable", "The owned image is unavailable.")
        if not image.resolved.image_bytes or len(image.resolved.image_bytes) > self.settings.VISUAL_FRAME_MAX_BYTES:
            raise ImageUnavailableError("invalid_image", "The image exceeds the supported size.")

    async def revalidate(self, image: PreparedImage) -> None:
        self._validate_scope_and_size(image)
        if image.resolved.revalidate is not None:
            await image.resolved.revalidate()
        else:
            # Current images have a cheap exact-frame source resolver. Historical
            # resolvers normally provide their own prewrite policy closure.
            definition = self.registry.get(image.tool_name)
            if definition is None or definition.image_resolver is None:
                raise ImageUnavailableError("frame_unavailable", "The image resolver is unavailable.")
            current = await definition.image_resolver(image.selection)
            if (current.reference != image.resolved.reference
                    or current.image_bytes != image.resolved.image_bytes):
                raise ImageUnavailableError("source_changed", "The selected image changed.")
        reference = image.resolved.reference
        if reference.source == "current":
            freshness = reference.freshness(
                now=datetime.now(UTC), monotonic_now=time.monotonic(),
                max_age_seconds=self.settings.VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS,
            )
            if freshness.status != "fresh":
                reason = "stale_frame" if freshness.status == "stale" else "clock_uncertain"
                raise ImageUnavailableError(reason, "The image cannot establish a current view.")

    @staticmethod
    def inline_part(image: PreparedImage) -> dict[str, Any]:
        return {"inlineData": {"mimeType": image.resolved.reference.content_type,
                               "data": base64.b64encode(image.resolved.image_bytes).decode("ascii")}}

    def mark_delivered(self, image: PreparedImage, *, namespace: int) -> None:
        # A successful socket write is delivery evidence, not a model acknowledgement.
        if image.resolved.reference.source == "current":
            self.delivered_image, self.delivered_namespace = image, namespace


def bounded_context(payload: dict, *, max_characters: int) -> str:
    """Drop optional context whole, never truncate required receipt JSON mid-field."""
    result = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(result) <= max_characters:
        return result
    payload = {**payload, "memory_context": "omitted_due_to_context_budget"}
    result = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(result) > max_characters:
        raise GeminiTransportError("gemini_required_context_too_large")
    return result
