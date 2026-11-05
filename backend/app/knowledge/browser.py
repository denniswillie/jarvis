"""Phone browsing of canonical originals, not a competing memory inference engine."""
from __future__ import annotations

import base64
import binascii
from datetime import datetime
import hashlib
import json
from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.sources import VERSION_SQL


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _metadata(source):
    return {key: source[key] for key in (
        "source_id", "revision", "generation", "content_digest", "source_type",
        "trust", "source_time", "time_basis", "recorded_at", "is_current",
        "ingestion_status",
    )}


def _item(source, text, *, truncated):
    return {
        "id": source["source_id"], "session_id": source.get("session_id") or source.get("context_session_id"),
        "memory_type": source["source_type"],
        "status": "active" if source["is_current"] else "superseded",
        "title": "Original " + source["source_type"].replace("_", " "),
        "content": text, "summary": text[:600],
        "source_experience_event_id": None, "occurred_at": source["source_time"],
        "valid_from": None, "valid_to": None, "importance": None, "confidence": None,
        "source": "canonical_source", "is_explicit": False,
        "superseded_by_memory_id": None, "created_at": source["recorded_at"],
        "updated_at": source["recorded_at"],
        "source_evidence": _metadata(source),
        "content_truncated": truncated,
    }


def _evidence(source, text):
    return {
        "id": source["source_id"] + ":" + str(source["revision"]),
        "memory_item_id": source["source_id"], "evidence_type": source["trust"],
        "evidence_time": source["source_time"], "excerpt": text[:600],
        "confidence": None, "source_available": True,
        "retained_image_available": None, "created_at": source["recorded_at"],
        "meaning": "The retained original itself, not independent verification. No image availability was checked.",
    }


class SourceMemoryBrowser:
    """Uses the existing repository's owner lock, active-area and version fences."""
    def __init__(self, sources, reader, *, owner_id, area_id, access_check=None):
        if reader.sources is not sources:
            raise ValueError("Source browser and reader must use the same repository")
        self.sources, self.reader = sources, reader
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Source browser owner mismatch.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    async def recent(self, *, owner_id, limit=30, source_types=(), session_id=None,
                     start_time=None, end_time=None, cursor=None):
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Source page limit must be between 1 and 100")
        types = tuple(sorted(set(source_types)))
        if len(types) > 32 or any(not isinstance(t, str) or not t.strip() or len(t) > 80 or "\x00" in t for t in types):
            raise ValueError("Invalid source type filters")
        if any(t is not None and (t.tzinfo is None or t.utcoffset() is None) for t in (start_time, end_time)):
            raise ValueError("Source times require a timezone")
        if start_time is not None and end_time is not None and start_time >= end_time:
            raise ValueError("start_time must be earlier than end_time")
        scope = hashlib.sha256(_json({
            "owner": owner_id, "area": str(self.area_id), "types": types,
            "session": str(session_id) if session_id else None,
            "start": start_time.isoformat() if start_time else None,
            "end": end_time.isoformat() if end_time else None,
        }).encode()).hexdigest()
        position = self._decode_cursor(cursor, scope) if cursor else (None, None)
        items, scanned, unavailable, has_more = [], 0, 0, False
        batch_size = limit + 1
        async with self.sources.transaction(owner_id) as connection:
            await self.sources._area(connection, owner_id, self.area_id)
            while scanned < 1000:
                rows = await connection.fetch(
                    VERSION_SQL + """
                    and v.revision=s.current_revision and s.status='active' and p.availability='available'
                    and (cardinality($3::text[])=0 or v.source_type=any($3))
                    and ($4::uuid is null or coalesce(ss.session_id,ce.session_id)=$4)
                    and ($5::timestamptz is null or v.source_time >= $5)
                    and ($6::timestamptz is null or v.source_time < $6)
                    and ($7::timestamptz is null or (coalesce(v.source_time,v.recorded_at),v.source_id)<($7,$8::uuid))
                    order by coalesce(v.source_time,v.recorded_at) desc,v.source_id desc limit $9
                    """,
                    owner_id, self.area_id, list(types), session_id, start_time, end_time,
                    *position, min(batch_size, 1001 - scanned),
                )
                if not rows:
                    break
                for row in rows:
                    if len(items) == limit or scanned == 1000:
                        has_more = True
                        break
                    position = (row["source_time"] or row["recorded_at"], row["source_id"])
                    scanned += 1
                    try:
                        version = self.sources._version(row)
                    except SourceError:
                        unavailable += 1
                        continue
                    source = version.original_evidence()
                    original = _json(version.content)
                    items.append(_item(source, original[:800], truncated=len(original) > 800))
                if has_more or len(rows) < batch_size:
                    break
            self._authorize(owner_id)
        return {
            "authority": "hindsight", "view": "original_sources", "provider_contacted": False,
            "items": items, "next_cursor": self._encode_cursor(position, scope) if has_more else None,
            "unavailable_sources_skipped": unavailable, "scan_limit_reached": scanned == 1000,
            "meaning": (
                "A paginated index of readable current originals, not an exhaustive answer or generated profile. "
                "Sorted by original source_time, falling back to recorded_at only when unknown. "
                "Time filters use source_time only; unknown times do not match. Pages reflect evolving current state. "
                "Trust describes origin, not confidence or present truth. Detail exposes paginated original JSON."
            ),
        }

    async def detail(self, *, owner_id, memory_id, revision=None, offset=0, expected_digest=None):
        self._authorize(owner_id)
        result = await self.reader.explain(owner_id=owner_id, memory_id=memory_id,
            revision=revision, offset=offset, expected_digest=expected_digest)
        self._authorize(owner_id)
        source, page = result["source"], result["content_page"]
        return {
            "authority": "hindsight", "memory": _item(source, page["text"], truncated=not page["complete"]),
            "evidence": [_evidence(source, page["text"])], "links": [], "related_memories": [],
            "content_page": {key: value for key, value in page.items() if key != "text"},
            "provider_contacted": False,
            "uncertainty": (
                "This is retained original JSON, not independent verification. Source trust identifies who or what "
                "supplied it; no confidence score or current truth is inferred. Unknown event times remain unknown. "
                "Assistant/tool text does not prove physical completion. Image availability has not been checked."
            ),
        }

    @staticmethod
    def _encode_cursor(position, scope):
        value = {"view": "source-timeline-v1", "scope": scope,
                 "time": position[0].isoformat(), "id": str(position[1])}
        return base64.urlsafe_b64encode(_json(value).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_cursor(value, scope):
        try:
            if len(value) > 2048:
                raise ValueError
            raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
            payload = json.loads(raw)
            if payload["view"] != "source-timeline-v1" or payload["scope"] != scope:
                raise ValueError
            when = datetime.fromisoformat(payload["time"])
            if when.tzinfo is None or when.utcoffset() is None:
                raise ValueError
            return when, UUID(str(payload["id"]))
        except (ValueError, TypeError, KeyError, UnicodeDecodeError, binascii.Error) as error:
            raise SourceError("invalid_memory_cursor", "Cursor does not match this source timeline and its filters.", 400) from error
