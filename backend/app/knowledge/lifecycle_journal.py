"""Committed, bounded source-journal checkpoints and private operator exports.

This is recovery metadata, not model context or an authorization tool. Exported
identities remain sensitive. A checksum detects corruption, not malicious
replacement: preserve exports in trusted storage outside the restored backup.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
from pydantic import Field

from app.evidence.models import FrozenRecord, OwnedReference
from app.knowledge.models import SourceError
from app.migrations.runner import quote_schema
from app.settings import JarvisSettings, is_configured_secret


class JournalCheckpoint(FrozenRecord):
    owner_id: str
    epoch: UUID
    through_sequence: int = Field(ge=0, le=9223372036854775807, strict=True)


class SourceLifecycleJournal:
    def __init__(self, pool):
        self.pool = pool

    @staticmethod
    def _bounds(after_sequence, limit):
        if type(after_sequence) is not int or not 0 <= after_sequence <= 9223372036854775807:
            raise ValueError("Invalid source journal cursor")
        if type(limit) is not int or not 1 <= limit <= 512:
            raise ValueError("Invalid source journal page limit")

    async def checkpoint(self, owner_id):
        OwnedReference.validate_owner(owner_id)
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """select epoch,last_sequence from jarvis_source_journal_streams
                        where owner_id=$1 for share""", owner_id,
                )
                if row is None:
                    raise SourceError("source_journal_not_initialized", "No source journal exists for this owner.", 404)
        return JournalCheckpoint(owner_id=owner_id, epoch=row["epoch"], through_sequence=row["last_sequence"])

    async def page(self, owner_id, checkpoint: JournalCheckpoint, *, after_sequence=0, limit=256):
        OwnedReference.validate_owner(owner_id)
        self._bounds(after_sequence, limit)
        if checkpoint.owner_id != owner_id:
            raise SourceError("source_journal_owner_mismatch", "The journal checkpoint belongs to another scope.", 403)
        if after_sequence > checkpoint.through_sequence:
            raise ValueError("Journal cursor exceeds its checkpoint")
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                current = await connection.fetchrow(
                    """select epoch,last_sequence from jarvis_source_journal_streams
                        where owner_id=$1 for share""", owner_id,
                )
                if (current is None or current["epoch"] != checkpoint.epoch
                        or current["last_sequence"] < checkpoint.through_sequence):
                    raise SourceError("source_journal_checkpoint_unavailable", "The pinned journal history is unavailable.")
                rows = await connection.fetch(
                    """select sequence,id,kind,object_id,detail,recorded_at
                        from jarvis_source_journal_events
                        where owner_id=$1 and sequence>$2 and sequence<=$3
                        order by sequence limit $4""",
                    owner_id, after_sequence, checkpoint.through_sequence, limit,
                )
        expected = min(limit, checkpoint.through_sequence - after_sequence)
        if len(rows) != expected or any(row["sequence"] != after_sequence + index + 1
                                        for index, row in enumerate(rows)):
            raise SourceError("source_journal_gap", "The checkpoint contains a missing or reordered event.")
        events = [{
            "sequence": row["sequence"], "id": str(row["id"]), "kind": row["kind"],
            "object_id": str(row["object_id"]),
            "detail": json.loads(row["detail"]) if isinstance(row["detail"], str) else row["detail"],
            "recorded_at": row["recorded_at"].isoformat(),
        } for row in rows]
        cursor = events[-1]["sequence"] if events else after_sequence
        return {
            "checkpoint": checkpoint.model_dump(mode="json"), "after_sequence": after_sequence,
            "events": events, "next_after_sequence": cursor,
            "complete": cursor == checkpoint.through_sequence,
        }

    async def export(self, owner_id, root: Path, *, checkpoint=None, after_sequence=0, max_events=10000):
        """Write one immutable, bounded segment; continuation uses its checkpoint."""
        self._bounds(after_sequence, 1)
        if type(max_events) is not int or not 1 <= max_events <= 100000:
            raise ValueError("Invalid source journal export limit")
        checkpoint = checkpoint if checkpoint is not None else await self.checkpoint(owner_id)
        if checkpoint.owner_id != owner_id or after_sequence > checkpoint.through_sequence:
            raise ValueError("Source journal export scope or cursor mismatch")
        root = Path(root)
        if not root.is_absolute():
            raise ValueError("A private absolute export directory is required")
        # Exclusive allocation never overwrites an existing user file/export.
        root.mkdir(mode=0o700)
        cursor, count = after_sequence, 0
        digest = hashlib.sha256()
        manifest = {
            "format": "jarvis.source_journal_export.v1", "export_id": str(uuid4()),
            "checkpoint": checkpoint.model_dump(mode="json"), "after_sequence": after_sequence,
            "status": "writing", "restore_authorized": False,
        }
        _write_json(root / "manifest.json", manifest)
        try:
            descriptor = os.open(root / "events.ndjson", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                # Even an empty segment validates the checkpoint against the DB.
                while True:
                    page = await self.page(
                        owner_id, checkpoint, after_sequence=cursor,
                        limit=min(512, max_events-count) if count < max_events else 1,
                    )
                    for event in page["events"]:
                        raw = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                        handle.write(raw)
                        digest.update(raw)
                    count += len(page["events"])
                    cursor = page["next_after_sequence"]
                    if page["complete"] or count >= max_events:
                        break
                handle.flush()
                os.fsync(handle.fileno())
            manifest.update(
                status="sealed", event_count=count, next_after_sequence=cursor,
                complete=cursor == checkpoint.through_sequence, sha256=digest.hexdigest(),
            )
            _write_json(root / "manifest.json", manifest)
            return manifest
        except BaseException:
            # A failed/cancelled export is never mistaken for a sealed segment.
            manifest.update(status="incomplete", event_count=count, next_after_sequence=cursor)
            _write_json(root / "manifest.json", manifest)
            raise


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


async def _run(args):
    settings = JarvisSettings()
    if not is_configured_secret(settings.SUPABASE_DB_URL):
        raise SourceError("source_journal_database_unconfigured", "The recovery database is not configured.", 503)
    schema = quote_schema(args.schema)

    async def setup(connection):
        await connection.execute(f"set search_path to {schema},pg_catalog,public,extensions")
    pool = await asyncpg.create_pool(settings.SUPABASE_DB_URL, min_size=1, max_size=1,
                                     timeout=10, command_timeout=30, setup=setup)
    try:
        journal = SourceLifecycleJournal(pool)
        checkpoint = None
        if args.checkpoint:
            # Metadata-only input from a trusted earlier export, not model args.
            with args.checkpoint.open("rb") as handle:
                raw = handle.read(16385)
            if len(raw) > 16384:
                raise ValueError("Checkpoint manifest exceeds its bound")
            checkpoint = JournalCheckpoint.model_validate(json.loads(raw)["checkpoint"])
        result = await journal.export(settings.JARVIS_OWNER_ID, args.root, checkpoint=checkpoint,
                                      after_sequence=args.after_sequence, max_events=args.max_events)
        print(json.dumps(result))
    finally:
        await pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--after-sequence", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=10000)
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except Exception as error:
        print(json.dumps({"ok": False, "error_type": type(error).__name__,
                          "code": getattr(error, "code", None)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
