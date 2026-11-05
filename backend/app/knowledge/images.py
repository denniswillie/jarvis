"""Historical pixels through canonical source lineage, never legacy memory authority."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from datetime import datetime
from uuid import UUID

from app.files.source_images import SavedFileImageService
from app.knowledge.product_files import FILE_NAMESPACE, IMAGE_ORIGINAL_FORMAT
from app.images import ImageReference, ImageUnavailableError, ResolvedImage
from app.knowledge.models import MemorySourceReference, SourceError
from app.knowledge.read_provenance import SourceReadProvenance


class SourceImageService:
    """Reuse retained-pixel transport; this adapter owns only source authorization."""
    def __init__(self, sources, stored, *, owner_id=None, area_id=None, access_check=None,
                 selections=None, session_id=None):
        self.sources, self.stored = sources, stored
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check
        self.provenance = SourceReadProvenance(sources)
        self.saved = SavedFileImageService(sources, stored) if stored is not None else None
        self.selections, self.session_id = selections, session_id
        if selections is not None and (owner_id is None or selections.sources.pool is not sources.pool
                or selections.default_area_id != area_id):
            raise ValueError("Session images must share their owner, database and configured default")
        if session_id is not None and not isinstance(session_id, UUID):
            raise ValueError("A trusted image session must be an exact UUID")

    def for_session(self, session_id):
        if self.selections is None:
            return self
        return SourceImageService(self.sources, self.stored, owner_id=self.owner_id, area_id=self.area_id,
            access_check=self.access_check, selections=self.selections, session_id=session_id)

    @staticmethod
    def unavailable():
        return ImageUnavailableError("frame_unavailable", "No authorized retained image is available for this source.")

    def _authorize(self, owner_id, area_id=None):
        if self.owner_id is not None and owner_id != self.owner_id:
            raise self.unavailable()
        if self.selections is None and self.area_id is not None and area_id is not None and area_id != self.area_id:
            raise self.unavailable()
        if self.access_check is not None:
            self.access_check(owner_id)
        if self.stored is None:
            raise self.unavailable()

    @staticmethod
    def _snapshot(reference):
        return replace(reference, source="historical", record_type=None, record_id=None,
                       stream_connection_id=None, sampled_at=None, sampled_monotonic=None)

    async def _direct(self, connection, source):
        content = source.content
        images = []
        if (source.source_type == "saved_visual_frame" and source.trust.value == "tool_receipt"
                and content.get("format") == IMAGE_ORIGINAL_FORMAT
                and await connection.fetchval(
                    """select exists(select 1 from jarvis_knowledge_sources where owner_id=$1
                        and id=$2 and namespace=$3 and source_key=$4)""",
                    source.owner_id, source.source_id, FILE_NAMESPACE, str(content.get("record", {}).get("id")))):
            images = [content.get("image")]
        elif source.context_event_id is not None:
            if source.source_type == "observation" and source.trust in {"sensor", "derived"}:
                images = content.get("images", [])
            elif source.source_type == "tool_outcome" and source.trust == "tool_receipt":
                result = content.get("result", {})
                if isinstance(result, dict) and result.get("status") == "success" and result.get("image_status") == "selected":
                    images = [result.get("image")]
        elif source.source_type.startswith("legacy:") and await connection.fetchval(
            """select exists(select 1 from jarvis_legacy_import_records
                where owner_id=$1 and source_id=$2 and source_revision=$3
                  and disposition in ('imported','already_present'))""",
            source.owner_id, source.source_id, source.revision,
        ):
            # The importer freezes these original frame rows. A frame ID merely
            # mentioned in a user note or generated answer is not image evidence.
            links = content.get("references", [])
            if not isinstance(links, list) or len(links) > 256:
                raise self.unavailable()
            for item in links:
                if not isinstance(item, dict) or item.get("table") != "visual_frames":
                    continue
                frame = item.get("external_metadata")
                if not isinstance(frame, dict) or str(frame.get("id")) != item.get("key"):
                    continue
                try:
                    identity = UUID(str(frame["id"]))
                    captured = datetime.fromisoformat(frame["captured_at"])
                    if captured.tzinfo is None:
                        continue
                except (KeyError, TypeError, ValueError):
                    continue
                # M03 deliberately retains only ID/status/capture time, not
                # storage locators or pixels. Resolve the owned original frame,
                # preserving that exact clock and the live retention policy.
                current = await connection.fetchrow(
                    """select id,session_id,captured_at,content_type,width,height,raw_image_expires_at,
                              visual_session_id,frame_sequence,server_received_at from visual_frames
                        where owner_id=$1 and id=$2 and captured_at=$3 and raw_image_status='available'
                          and raw_image_expires_at>clock_timestamp()""",
                    source.owner_id, identity, captured,
                )
                if current is None:
                    continue
                images.append(ImageReference(frame_id=current["id"], owner_id=source.owner_id, source="historical",
                    **{key: current[key] for key in (
                        "session_id", "captured_at", "content_type", "width", "height",
                        "raw_image_expires_at", "visual_session_id", "frame_sequence", "server_received_at",
                    )}).to_metadata())
        if not isinstance(images, list) or len(images) > 64:
            raise self.unavailable()
        references = []
        for metadata in images:
            if not isinstance(metadata, dict):
                continue
            try:
                reference = ImageReference.from_metadata(metadata, owner_id=source.owner_id)
                if reference.source == "current" and reference.session_id != source.context_session_id:
                    continue
                references.append(self._snapshot(reference))
            except (KeyError, TypeError, ValueError):
                continue
        return references

    def _reader(self, reference):
        return self.saved if reference.retained_file_id is not None else self.stored

    async def _parents(self, connection, owner_id, area_id, payload_id):
        rows = await connection.fetch(
            """select area_id,source_id,source_revision as revision,source_generation as generation,
                      content_digest,source_usage from jarvis_context_memory_sources
                where owner_id=$1 and dependent_payload_id=$2 order by source_id,source_revision""",
            owner_id, payload_id,
        )
        if len(rows) > 32 or any(row["area_id"] != area_id for row in rows):
            raise self.unavailable()
        return [MemorySourceReference.model_validate({key: row[key] for key in MemorySourceReference.model_fields})
                for row in rows]

    async def select_image(self, *, owner_id, area_id, source_id, revision=None, frame_id=None):
        self._authorize(owner_id, area_id)
        if revision is not None and (type(revision) is not int or revision < 1):
            raise ValueError("revision must be a positive integer")
        try:
            async with asyncio.timeout(self.stored.timeout_seconds):
                if self.selections is not None:
                    area_id = await self.selections.source_area(owner_id, self.session_id, source_id)
                first = await self.sources.read(owner_id, area_id, source_id, revision=revision)
                pending, seen = deque([(first, [])]), set()
                selected = None
                while pending:
                    source, path = pending.popleft()
                    identity = (source.source_id, source.revision)
                    if identity in seen:
                        continue
                    if len(seen) >= 32:
                        raise self.unavailable()
                    seen.add(identity)
                    metadata = source.original_evidence()
                    metadata.pop("content")
                    path = [*path, metadata]
                    async with self.sources.transaction(owner_id) as connection:
                        candidates = await self._direct(connection, source)
                    # Do not wait for another pool connection while holding the
                    # source-owner transaction; one-slot worker pools must work.
                    for reference in candidates:
                        if frame_id is not None and reference.frame_id != frame_id:
                            continue
                        try:
                            await self._reader(reference)._read(owner_id, reference)
                        except ImageUnavailableError:
                            continue
                        selected = reference, path
                        break
                    if selected is not None:
                        break
                    async with self.sources.transaction(owner_id) as connection:
                        parents = await self._parents(connection, owner_id, area_id, source.payload_id)
                        resolved = await self.provenance._resolve_many(connection, owner_id, area_id, parents)
                        pending.extend((parent, path) for parent in resolved)
                if selected is None:
                    raise self.unavailable()
                reference, path = selected
                roots = await self.provenance.bind_context(owner_id, area_id, {"sources": path})
                bound = replace(reference, record_type="source_evidence", record_id=roots[0])
                await self._validate(owner_id, bound)
                return bound
        except SourceError as error:
            raise self.unavailable() from error
        except TimeoutError as error:
            raise ImageUnavailableError("image_lookup_timeout", "Source image lookup timed out.") from error

    async def _validate(self, owner_id, reference):
        self._authorize(owner_id)
        if (reference.owner_id != owner_id or reference.source != "historical"
                or not reference.source_payload_ids):
            raise self.unavailable()
        async with self.sources.transaction(owner_id) as connection:
            if self.selections is not None:
                await self.selections._session(connection, owner_id, self.session_id)
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, reference.record_id) != "available":
                raise self.unavailable()
            rows = await connection.fetch(
                """select area_id from jarvis_context_memory_sources
                    where owner_id=$1 and dependent_payload_id=$2 group by area_id""",
                owner_id, reference.record_id,
            )
            if len(rows) != 1:
                raise self.unavailable()
            area = rows[0]["area_id"]
            self._authorize(owner_id, area)
            parents = await self._parents(connection, owner_id, area, reference.record_id)
            if not parents:
                raise self.unavailable()
            originals = await self.provenance._resolve_many(connection, owner_id, area, parents)
            snapshot = self._snapshot(reference)
            for original in originals:
                if snapshot in await self._direct(connection, original):
                    return
        raise self.unavailable()

    async def resolve_image(self, *, owner_id, reference):
        self._authorize(owner_id)
        async def check():
            try:
                await self._validate(owner_id, reference)
            except SourceError as error:
                raise self.unavailable() from error
        try:
            async with asyncio.timeout(self.stored.timeout_seconds):
                await check()
                resolved = await self._reader(reference).resolve_image(owner_id=owner_id, reference=self._snapshot(reference))
                await check()
        except TimeoutError as error:
            raise ImageUnavailableError("image_resolution_timeout", "Source image resolution timed out.") from error

        async def revalidate():
            async with asyncio.timeout(self.stored.timeout_seconds):
                await check()
                await resolved.revalidate()

        return ResolvedImage(reference=reference, image_bytes=resolved.image_bytes, revalidate=revalidate)
