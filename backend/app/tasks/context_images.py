"""Original-image access bound to canonical evidence and a live task claim."""
from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from uuid import UUID

from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.memory.retrieval.image_evidence import HistoricalImageService
from app.perception.current_image import CurrentImageService
from app.tasks.context_repository import ContextRepository


class TaskContextImages:
    def __init__(self, repository: ContextRepository, *,
                 current: CurrentImageService | None = None,
                 historical: HistoricalImageService | None = None,
                 stored=None, source_historical=None,
                 timeout_seconds: float = 10):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("invalid task image timeout")
        self.repository = repository
        self.current = current
        self.historical = historical
        self.stored = stored
        self.source_historical = source_historical
        self.timeout_seconds = timeout_seconds

    async def select(self, claim, event_id: UUID, *, image_index=0) -> dict:
        if isinstance(image_index, bool) or not isinstance(image_index, int) or not 0 <= image_index < 16:
            raise ValueError("invalid context image index")
        record = await self.repository.source_for_claim(claim, event_id)
        if record.availability != "available" or record.content is None:
            raise ImageUnavailableError("frame_unavailable", "The image's canonical source is unavailable.")
        content = record.content
        metadata = None
        if record.kind == "tool_outcome" and record.trust == "tool_receipt":
            result = content.get("result")
            if (image_index == 0 and isinstance(result, dict)
                    and result.get("status") == "success" and result.get("image_status") == "selected"):
                metadata = result.get("image")
        elif record.kind == "observation" and record.trust in {"sensor", "derived"}:
            images = content.get("images")
            if isinstance(images, list) and image_index < len(images):
                metadata = images[image_index]
        if not isinstance(metadata, dict):
            raise ImageUnavailableError("frame_unavailable", "This source has no selected original image at that index.")
        try:
            reference = ImageReference.from_metadata(metadata, owner_id=claim.task.owner_id)
        except (KeyError, TypeError, ValueError) as error:
            raise ImageUnavailableError("frame_unavailable", "The original image reference is invalid.") from error
        if reference.source == "current" and reference.session_id != record.session_id:
            raise ImageUnavailableError("frame_unavailable", "The camera image belongs to another session.")
        snapshot = self.stored is not None and self.current is None
        original_source = reference.source
        if snapshot and reference.source == "current":
            # A different process cannot validate the producer's stream lease or
            # monotonic clock. Preserve the captured frame, never claim a live view.
            reference = replace(reference,source="historical",stream_connection_id=None,
                                sampled_at=None,sampled_monotonic=None)
        return {
            **({"image_view":"captured_snapshot","current_view_verified":False,
                "original_source":original_source,
                "meaning":"Exact previously captured pixels, not a fresh capture or proof the camera is still live."}
               if snapshot else {}),
            "status": "success", "image_status": "selected",
            "source_event_id": str(record.id), "source_revision": record.revision,
            "image_index": image_index, "image": reference.to_metadata(),
        }

    async def resolve(self, claim, selection: dict) -> ResolvedImage:
        try:
            async with asyncio.timeout(self.timeout_seconds):
                return await self._resolve(claim, selection)
        except TimeoutError as error:
            raise ImageUnavailableError("image_resolution_timeout", "The original image could not be resolved in time.") from error

    async def _resolve(self, claim, selection):
        try:
            event_id = UUID(str(selection["source_event_id"]))
            image_index = selection["image_index"]
        except (KeyError, TypeError, ValueError) as error:
            raise ImageUnavailableError("frame_unavailable", "There is no valid selected context image.") from error

        async def checked_reference():
            fresh = await self.select(claim, event_id, image_index=image_index)
            if fresh != selection:
                raise ImageUnavailableError("source_changed", "The selected image source has changed.")
            return ImageReference.from_metadata(fresh["image"], owner_id=claim.task.owner_id)

        reference = await checked_reference()
        if reference.source == "current" and self.current is not None:
            resolved = self.current.resolve_current_image(
                owner_id=claim.task.owner_id, session_id=reference.session_id, reference=reference,
            )
        elif reference.source_payload_ids:
            if self.source_historical is None:
                raise ImageUnavailableError("frame_unavailable", "Source-backed historical image access is unavailable.")
            resolved = await self.source_historical.resolve_image(owner_id=claim.task.owner_id, reference=reference)
        elif reference.source == "historical" and reference.record_id is None and self.stored is not None:
            resolved = await self.stored.resolve_image(owner_id=claim.task.owner_id,reference=reference)
        elif reference.source == "historical" and self.historical is not None:
            resolved = await self.historical.resolve_image(owner_id=claim.task.owner_id, reference=reference)
        else:
            raise ImageUnavailableError("frame_unavailable", "This original-image service is unavailable.")
        if resolved.reference != reference or not resolved.image_bytes:
            raise ImageUnavailableError("source_changed", "Image resolution did not return the exact selected evidence.")

        async def revalidate():
            # A queued transport must call this immediately before sending bytes.
            # Selection itself is not evidence that the model received the image.
            async with asyncio.timeout(self.timeout_seconds):
                await checked_reference()
                if reference.source == "current":
                    current = self.current.resolve_current_image(
                        owner_id=claim.task.owner_id, session_id=reference.session_id, reference=reference,
                    )
                    if current.reference != reference or current.image_bytes != resolved.image_bytes:
                        raise ImageUnavailableError("source_changed", "The selected original pixels have changed.")
                elif resolved.revalidate is not None:
                    await resolved.revalidate()
                else:
                    # HistoricalImageService normally supplies a metadata policy
                    # recheck. Do not silently skip policy for another adapter.
                    resolver = self.source_historical if reference.source_payload_ids else self.historical
                    if resolver is None:
                        raise ImageUnavailableError("frame_unavailable", "Historical source validation is unavailable.")
                    refreshed = await resolver.resolve_image(
                        owner_id=claim.task.owner_id, reference=reference,
                    )
                    if refreshed.reference != reference or refreshed.image_bytes != resolved.image_bytes:
                        raise ImageUnavailableError("source_changed", "The historical source has changed.")

        await revalidate()
        return ResolvedImage(reference=reference, image_bytes=resolved.image_bytes, revalidate=revalidate)
