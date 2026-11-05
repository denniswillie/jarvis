"""Read exact retained snapshots for a worker without a live-camera cache."""
from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime

from app.errors import JarvisHTTPError
from app.images import ImageUnavailableError, ResolvedImage
from app.memory.retrieval.evidence_repository import HistoricalEvidenceRepository
from app.memory.retrieval.image_evidence import HistoricalImageService
from app.tasks.context_images import TaskContextImages
from app.visual_memory.validation import jpeg_dimensions, VisualFrameValidationError


class StoredFrameImageService:
    def __init__(self, pool, storage, *, max_bytes, max_width, max_height, timeout_seconds,
                 now=lambda:datetime.now(UTC)):
        if (any(type(value) is not int or value<1 for value in (max_bytes,max_width,max_height))
                or not math.isfinite(timeout_seconds) or timeout_seconds<=0):
            raise ValueError("Stored-image bounds must be positive")
        self.pool,self.storage,self.now = pool,storage,now
        self.max_bytes,self.max_width,self.max_height = max_bytes,max_width,max_height
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def unavailable():
        return ImageUnavailableError("frame_unavailable","The exact retained snapshot is unavailable.")

    async def _read(self, owner_id, reference):
        if (owner_id!=reference.owner_id or reference.source!="historical" or reference.record_id is not None
                or reference.retained_file_id is not None
                or reference.raw_image_expires_at is None or reference.raw_image_expires_at<=self.now()):
            raise self.unavailable()
        async with self.pool.acquire() as connection:
            row = await connection.fetchrow(
                """select id,owner_id,session_id,visual_session_id,frame_sequence,captured_at,server_received_at,
                    content_type,width,height,storage_bucket,storage_path,raw_image_status,raw_image_expires_at
                    from visual_frames where owner_id=$1 and session_id=$2 and id=$3
                    and raw_image_expires_at>clock_timestamp() and $4::timestamptz>clock_timestamp()""",
                owner_id,reference.session_id,reference.frame_id,reference.raw_image_expires_at,
            )
        if row is None or row["raw_image_status"]!="available" or row["raw_image_expires_at"]<=self.now():
            raise self.unavailable()
        for key in ("captured_at","content_type","width","height"):
            if row[key]!=getattr(reference,key):
                raise self.unavailable()
        for key in ("visual_session_id","frame_sequence","server_received_at"):
            expected = getattr(reference,key)
            if expected is not None and row[key]!=expected:
                raise self.unavailable()
        path = self.storage.build_storage_path(owner_id=owner_id,session_id=row["session_id"],
                    visual_session_id=row["visual_session_id"],frame_sequence=row["frame_sequence"])
        if row["storage_bucket"]!=self.storage.bucket or row["storage_path"]!=path:
            raise self.unavailable()
        if reference.width>self.max_width or reference.height>self.max_height:
            raise ImageUnavailableError("invalid_image","The retained image exceeds dimension limits.")
        return dict(row)

    async def resolve_image(self, *, owner_id, reference):
        try:
            async with asyncio.timeout(self.timeout_seconds):
                selected = await self._read(owner_id,reference)
                try:
                    pixels = await self.storage.download_object(
                        bucket=selected["storage_bucket"],path=selected["storage_path"],max_bytes=self.max_bytes,
                    )
                except JarvisHTTPError as error:
                    if error.details.get("status_code")==404:
                        raise self.unavailable() from error
                    code = "invalid_image" if error.code=="visual_frame_storage_download_too_large" else "image_storage_failed"
                    raise ImageUnavailableError(code,"The retained image could not be loaded safely.") from error
                except TimeoutError:
                    raise
                except Exception as error:
                    raise ImageUnavailableError("image_storage_failed","Retained image storage is unavailable.") from error
                try:
                    if not pixels or len(pixels)>self.max_bytes or jpeg_dimensions(pixels)!=(reference.width,reference.height):
                        raise ImageUnavailableError("invalid_image","The retained pixels do not match the selected JPEG.")
                except VisualFrameValidationError as error:
                    raise ImageUnavailableError("invalid_image","The retained image is not a valid JPEG.") from error
                if await self._read(owner_id,reference)!=selected:
                    raise self.unavailable()
        except TimeoutError as error:
            raise ImageUnavailableError("image_resolution_timeout","Retained image lookup timed out.") from error

        async def revalidate():
            async with asyncio.timeout(self.timeout_seconds):
                if await self._read(owner_id,reference)!=selected:
                    raise self.unavailable()

        return ResolvedImage(reference=reference,image_bytes=pixels,revalidate=revalidate)


class StoredTaskImageFactory:
    """Shared storage client, but each tool remains bound to its task's canonical sources."""
    def __init__(self, pool, storage, settings):
        timeout = settings.VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS
        self.stored = StoredFrameImageService(
            pool,storage,max_bytes=settings.VISUAL_FRAME_MAX_BYTES,
            max_width=settings.VISUAL_FRAME_MAX_WIDTH,max_height=settings.VISUAL_FRAME_MAX_HEIGHT,
            timeout_seconds=timeout,
        )
        # Existing legacy record references retain their stricter record/evidence
        # relation checks. Canonical observations need no legacy memory record.
        self.historical = HistoricalImageService(
            repository=HistoricalEvidenceRepository(pool),image_loader=storage,
            max_image_bytes=settings.VISUAL_FRAME_MAX_BYTES,max_width=settings.VISUAL_FRAME_MAX_WIDTH,
            max_height=settings.VISUAL_FRAME_MAX_HEIGHT,request_timeout_seconds=timeout,
        )
        from app.knowledge.images import SourceImageService
        from app.knowledge.sources import SourceRepository
        self.source_historical = SourceImageService(SourceRepository(pool), self.stored)
        self.timeout = timeout

    async def __call__(self, context, repository):
        return TaskContextImages(repository,stored=self.stored,historical=self.historical,
                                 source_historical=self.source_historical, timeout_seconds=self.timeout)
