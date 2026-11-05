"""Retained file pixels with canonical-source admission, independent of raw expiry."""
import asyncio
from hashlib import sha256

from app.files.repository import _file_from_record
from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.knowledge.models import SourceError
from app.knowledge.product_files import IMAGE_ORIGINAL_FORMAT, read_file_original
from app.visual_memory.validation import jpeg_dimensions, VisualFrameValidationError


class SavedFileImageService:
    def __init__(self, sources, stored):
        self.sources, self.stored = sources, stored

    @staticmethod
    def unavailable():
        return ImageUnavailableError("frame_unavailable", "The exact saved image is unavailable.")

    async def _read(self, owner_id, reference):
        if (reference.owner_id != owner_id or reference.source != "historical" or reference.retained_file_id is None
                or reference.record_id is not None or reference.raw_image_expires_at is not None):
            raise self.unavailable()
        try:
            async with self.sources.transaction(owner_id) as connection:
                row = await connection.fetchrow(
                    """select f.*,p.area_id from jarvis_files f join jarvis_file_publications p
                        on p.owner_id=f.owner_id and p.file_id=f.id where f.owner_id=$1 and f.id=$2""",
                    owner_id, reference.retained_file_id)
                if row is None:
                    raise self.unavailable()
                source, publication = await read_file_original(
                    connection, self.sources, owner_id, row["area_id"], _file_from_record(row))
                if (source.content.get("format") != IMAGE_ORIGINAL_FORMAT
                        or ImageReference.from_metadata(source.content["image"], owner_id=owner_id) != reference
                        or source.trust.value != "tool_receipt" or source.source_type != "saved_visual_frame"):
                    raise self.unavailable()
                if (reference.width > self.stored.max_width or reference.height > self.stored.max_height
                        or publication["size_bytes"] > self.stored.max_bytes):
                    raise ImageUnavailableError("invalid_image", "Saved image exceeds the configured limits.")
                return {key: publication[key] for key in ("storage_bucket", "storage_path", "size_bytes", "content_sha256")}
        except (SourceError, KeyError, TypeError, ValueError) as error:
            raise self.unavailable() from error

    async def resolve_image(self, *, owner_id, reference):
        try:
            async with asyncio.timeout(self.stored.timeout_seconds):
                selected = await self._read(owner_id, reference)
                try:
                    pixels = await self.stored.storage.download_object(
                        bucket=selected["storage_bucket"], path=selected["storage_path"], max_bytes=self.stored.max_bytes)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    raise ImageUnavailableError("image_storage_failed", "Saved image storage is unavailable.") from error
                try:
                    if (len(pixels) != selected["size_bytes"] or sha256(pixels).hexdigest() != selected["content_sha256"]
                            or jpeg_dimensions(pixels) != (reference.width, reference.height)):
                        raise ValueError("saved pixels changed")
                except (VisualFrameValidationError, ValueError) as error:
                    raise ImageUnavailableError("invalid_image", "Saved bytes do not match the original image.") from error
                if await self._read(owner_id, reference) != selected:
                    raise self.unavailable()
        except TimeoutError as error:
            raise ImageUnavailableError("image_resolution_timeout", "Saved image retrieval timed out.") from error

        async def revalidate():
            async with asyncio.timeout(self.stored.timeout_seconds):
                if await self._read(owner_id, reference) != selected:
                    raise self.unavailable()
        return ResolvedImage(reference=reference, image_bytes=pixels, revalidate=revalidate)
