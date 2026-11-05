from __future__ import annotations

import asyncio
import logging

from app.visual_memory.repository import VisualFrameRepository
from app.visual_memory.storage import VisualFrameStorageClient


logger = logging.getLogger(__name__)


class VisualFrameRetentionService:
    def __init__(
        self,
        *,
        repository: VisualFrameRepository,
        storage: VisualFrameStorageClient,
        batch_size: int = 200,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._batch_size = batch_size

    async def expire_old_raw_frames(self) -> int:
        expired_reason_count = await self._repository.delete_expired_retention_reasons(
            limit=self._batch_size
        )
        if expired_reason_count:
            logger.info("removed %s expired visual frame retention reasons", expired_reason_count)
        frames = await self._repository.claim_expired_raw_frames(limit=self._batch_size)
        expired_count = 0
        for frame in frames:
            try:
                await self._storage.delete_objects(
                    bucket=frame.storage_bucket,
                    paths=[frame.storage_path],
                )
            except Exception:
                logger.exception("failed to delete expired visual frame %s", frame.id)
                await self._repository.release_frame_expiry_claim(
                    owner_id=frame.owner_id,
                    frame_id=frame.id,
                )
                continue
            await self._repository.mark_frame_expired(owner_id=frame.owner_id, frame_id=frame.id)
            expired_count += 1
        if expired_count:
            logger.info("expired %s visual frame raw images", expired_count)
        return expired_count


async def run_visual_frame_retention_loop(
    service: VisualFrameRetentionService,
    *,
    interval_seconds: int = 60,
) -> None:
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            await service.expire_old_raw_frames()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("visual frame retention pass failed")
