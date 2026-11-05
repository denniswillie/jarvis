"""Single-owner Live caption intake from canonical SQL, not another transcript store.

SQLite contains only owner/bank binding, cursor, one pending batch receipt and
document IDs, plus forget tombstones. Pending batch text is reconstructed from
canonical revisions. No automatic backfill, broker, or process manager.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from .client import ForgetResult, ForgetScope, MemoryDocument, MemoryRequestError
from .ingestion import CanonicalCaption, DialogueBatcher


_ACTIVE_CHECKPOINTS: set[Path] = set()


LATEST_CAPTIONS_SQL = """
WITH latest AS (
    SELECT DISTINCT ON (e.source_namespace, e.source_event_id)
        e.sequence, e.session_id, e.source_namespace, e.source_event_id,
        e.revision, e.kind, e.source_time, e.recorded_at, p.availability, p.content,
        CASE WHEN s.status='closed' THEN s.closed_at END AS session_closed_at
    FROM jarvis_context_events e
    JOIN jarvis_context_payloads p ON p.owner_id=e.owner_id AND p.id=e.payload_id
    LEFT JOIN jarvis_sessions s ON s.owner_id=e.owner_id AND s.id=e.session_id
    WHERE e.owner_id=$1 AND e.sequence>$2 AND e.sequence<=$3
      AND e.kind IN ('user_transcript','assistant_output')
      AND e.source_namespace LIKE 'voice:%'
      AND e.source_event_id LIKE 'live-caption:%'
    ORDER BY e.source_namespace,e.source_event_id,e.revision DESC
)
SELECT * FROM latest ORDER BY sequence LIMIT $4
"""


@dataclass(frozen=True)
class CaptionPage:
    captions: tuple[CanonicalCaption, ...]
    watermark: int


class AsyncpgCaptionSource:
    def __init__(self, pool, owner_id: str):
        self.pool, self.owner_id = pool, owner_id

    async def high_watermark(self) -> int:
        return int(await self.pool.fetchval(
            "SELECT COALESCE(MAX(sequence),0) FROM jarvis_context_events WHERE owner_id=$1",
            self.owner_id,
        ))

    async def read_after(self, cursor: int, *, through=None, limit=128) -> CaptionPage:
        async with self.pool.acquire() as connection:
            async with connection.transaction(isolation="repeatable_read", readonly=True):
                ceiling = through
                if ceiling is None:
                    ceiling = await connection.fetchval(
                        "SELECT COALESCE(MAX(sequence),0) FROM jarvis_context_events WHERE owner_id=$1",
                        self.owner_id,
                    )
                rows = await connection.fetch(LATEST_CAPTIONS_SQL, self.owner_id, cursor, ceiling, limit)
        captions = []
        for row in rows:
            fields = dict(row)
            fields["session_id"] = str(fields["session_id"])
            if isinstance(fields["content"], str):
                fields["content"] = json.loads(fields["content"])
            captions.append(CanonicalCaption(**fields))
        # If the page is full, do not skip unseen later captions. Non-caption
        # events may safely be crossed once every caption in the window is read.
        watermark = captions[-1].sequence if len(captions) == limit else int(ceiling)
        return CaptionPage(tuple(captions), watermark)


class LiveDialogueIntake:
    def __init__(
        self, memory, source, checkpoint_path, *, invalidate=None,
        settle_seconds=18, poll_seconds=1, operation_timeout=90, batch_size=128,
        now=lambda: datetime.now(timezone.utc),
    ):
        if source.owner_id != memory.scope.owner_id:
            raise ValueError("Caption source and memory owner must match")
        if not 15 <= settle_seconds <= 20 or poll_seconds <= 0 or operation_timeout <= 0 or batch_size < 1:
            raise ValueError("Invalid bounded intake configuration")
        self.memory, self.source = memory, source
        self.path = Path(checkpoint_path)
        self.invalidate = invalidate
        self.settle_seconds, self.poll_seconds = settle_seconds, poll_seconds
        self.operation_timeout, self.batch_size, self.now = operation_timeout, batch_size, now
        self._lock = asyncio.Lock()
        self._db = None
        self._fd = None
        self._claimed_path = None
        self._closed = False
        self._closed_sessions = set()
        self.startup_forget_result = None
        self.issue = None

    async def initialize(self):
        async with self._lock:
            if self._closed:
                raise RuntimeError("Intake is closed")
            if self._db is not None:
                return self.cursor
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            identity = self.path.resolve()
            if identity in _ACTIVE_CHECKPOINTS:
                raise BlockingIOError("An intake already owns this checkpoint in this process")
            # Single-process async ownership, as configured by the serving host.
            # Do not flock the SQLite file: on macOS that conflicts with SQLite's
            # own locks. This deliberately is not a multi-process lease service.
            _ACTIVE_CHECKPOINTS.add(identity)
            self._claimed_path = identity
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                os.fchmod(self._fd, 0o600)
                self._db = sqlite3.connect(self.path)
                self._db.row_factory = sqlite3.Row
                self._db.executescript("""
                    CREATE TABLE IF NOT EXISTS checkpoint (
                        id INTEGER PRIMARY KEY CHECK(id=1), owner_id TEXT NOT NULL,
                        bank_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                        pending_operation TEXT, pending_cursor INTEGER,
                        pending_documents TEXT NOT NULL DEFAULT '[]');
                    CREATE TABLE IF NOT EXISTS forgotten (
                        document_id TEXT PRIMARY KEY, pending_delete INTEGER NOT NULL);
                """)
                row = self._db.execute("SELECT * FROM checkpoint WHERE id=1").fetchone()
                if row is None:
                    # Initial activation starts NOW. Do not recreate this file on
                    # restart or use a new generation for every audio session.
                    cursor = await self.source.high_watermark()
                    with self._db:
                        self._db.execute("INSERT INTO checkpoint(id,owner_id,bank_id,sequence) VALUES(1,?,?,?)",
                                         (self.memory.scope.owner_id, self.memory.scope.bank_id, cursor))
                elif (row["owner_id"], row["bank_id"]) != (self.memory.scope.owner_id, self.memory.scope.bank_id):
                    raise ValueError("Checkpoint belongs to a different owner/bank")
                pending = self._pending_forgets()
                if pending:
                    # Exactly one startup retry. Failures remain durable and are
                    # exposed; no invisible recurring deletion loop.
                    self.startup_forget_result = await self._forget_locked(pending, self.invalidate)
                return self.cursor
            except BaseException:
                self._release()
                raise

    @property
    def cursor(self):
        return self._db.execute("SELECT sequence FROM checkpoint WHERE id=1").fetchone()[0]

    def _pending_forgets(self):
        return tuple(row[0] for row in self._db.execute(
            "SELECT document_id FROM forgotten WHERE pending_delete=1 ORDER BY document_id"))

    def _tombstones(self):
        return {row[0] for row in self._db.execute("SELECT document_id FROM forgotten")}

    def _advance(self, sequence):
        with self._db:
            self._db.execute("UPDATE checkpoint SET sequence=?,pending_operation=NULL,pending_cursor=NULL,"
                             "pending_documents='[]' WHERE id=1", (sequence,))

    async def _settle_pending(self, *, allow_submit=True):
        row = self._db.execute("SELECT * FROM checkpoint WHERE id=1").fetchone()
        operation = row["pending_operation"]
        if operation is None:
            return None
        path = "/operations/" + operation
        status = await self.memory._request("GET", path, missing_ok=True)
        if status is None:
            # A side-band observation never advances the caption cursor. Its
            # exact text is deliberately NOT copied into this checkpoint or
            # reconstructed from an unrelated caption. Preserve the receipt
            # and fail closed if the native operation cannot be identified.
            if row["pending_cursor"] == row["sequence"]:
                raise MemoryRequestError("observation_retention_outcome_unknown")
            if not allow_submit:
                raise MemoryRequestError("retention_outcome_unknown")
            page = await self.source.read_after(row["sequence"], through=row["pending_cursor"], limit=self.batch_size)
            identities = set(json.loads(row["pending_documents"]))
            documents = [caption.document() for caption in page.captions
                         if caption.usable and caption.document_id in identities]
            if {document.document_id for document in documents} != identities:
                raise MemoryRequestError("pending_canonical_caption_unavailable")
            # Reconstruct exactly the bounded canonical revision window. Reusing
            # the saved native operation ID cannot enqueue duplicate extraction.
            await DialogueBatcher.publish(self.memory, documents, operation_id=operation)
        deadline = time.monotonic() + self.operation_timeout
        while True:
            if status is not None and status.get("status") in {"completed", "failed", "cancelled"}:
                outcome = status["status"]
                self._advance(row["pending_cursor"] if outcome == "completed" else row["sequence"])
                return outcome
            if time.monotonic() >= deadline:
                raise MemoryRequestError("retention_completion_timeout")
            await asyncio.sleep(self.poll_seconds)
            status = await self.memory._request("GET", path, missing_ok=True)

    async def tick(self):
        await self.initialize()
        async with self._lock:
            return await self._tick_locked()

    async def retain_observation(self, document: MemoryDocument, *, is_current=None):
        """Submit one visual delta through the existing retention/forget owner.

        Returns native acceptance, NOT completed extraction. The shared pending
        slot is durable before POST; tick, the next writer, or forget settles it
        before another operation can replace it. No second queue/store exists.
        is_current is an optional synchronous generation check under this lock.
        """
        if not isinstance(document, MemoryDocument) or document.metadata.get("stream") != "observation":
            raise ValueError("Use a canonical observation document, not a dialogue copy")
        document.payload()
        await self.initialize()
        async with self._lock:
            outcome = await self._settle_pending(allow_submit=not self._pending_forgets())
            if outcome in {"failed", "cancelled"}:
                raise MemoryRequestError("retention_operation_" + outcome)
            if document.document_id in self._tombstones():
                return {"submitted": False, "status": "suppressed_forgotten",
                        "document_id": document.document_id}
            if is_current is not None and not is_current():
                return {"submitted": False, "status": "suppressed_stale_generation",
                        "document_id": document.document_id}
            operation = str(uuid4())
            with self._db:
                # pending_cursor == sequence identifies a non-caption write;
                # caption batches always advance beyond their starting cursor.
                self._db.execute("UPDATE checkpoint SET pending_operation=?,pending_cursor=sequence,"
                                 "pending_documents=? WHERE id=1",
                                 (operation, json.dumps([document.document_id])))
            await DialogueBatcher.publish(self.memory, [document], operation_id=operation)
            return {"submitted": True, "status": "accepted", "operation_id": operation,
                    "document_id": document.document_id}

    async def _tick_locked(self, *, force=False, through=None):
        outcome = await self._settle_pending(allow_submit=not self._pending_forgets())
        if outcome in {"failed", "cancelled"}:
            raise MemoryRequestError("retention_operation_" + outcome)
        page = await self.source.read_after(self.cursor, through=through, limit=self.batch_size)
        tombstones = self._tombstones()
        documents = []
        end = self.cursor
        for caption in page.captions:
            if caption.usable and caption.document_id not in tombstones:
                if (not force and caption.session_closed_at is None
                        and caption.session_id not in self._closed_sessions
                        and caption.recorded_at + timedelta(seconds=self.settle_seconds) > self.now()):
                    break
                documents.append(caption.document())
            end = caption.sequence
        else:
            end = page.watermark
        if not documents:
            self._advance(end)
            return {"cursor": end, "retained": 0}
        operation = str(uuid4())
        with self._db:
            self._db.execute("UPDATE checkpoint SET pending_operation=?,pending_cursor=?,pending_documents=? WHERE id=1",
                             (operation, end, json.dumps([document.document_id for document in documents])))
        await DialogueBatcher.publish(self.memory, documents, operation_id=operation)
        outcome = await self._settle_pending()
        if outcome != "completed":
            raise MemoryRequestError("retention_operation_" + str(outcome))
        return {"cursor": self.cursor, "retained": len(documents)}

    async def run(self, stop: asyncio.Event):
        await self.initialize()
        try:
            while not stop.is_set() and not self._closed:
                await self.tick()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.poll_seconds)
                except TimeoutError:
                    pass
        except Exception as exc:
            self.issue = exc.code if isinstance(exc, MemoryRequestError) else "dialogue_intake_failed"
            raise

    async def forget(self, ids, *, invalidate=None):
        await self.initialize()
        async with self._lock:
            return await self._forget_locked(tuple(dict.fromkeys(ids)), invalidate or self.invalidate)

    async def _forget_locked(self, ids, invalidate):
        if not ids or any(not isinstance(identity, str) or not identity for identity in ids):
            raise ValueError("Explicit document IDs are required")
        with self._db:
            self._db.executemany("INSERT INTO forgotten VALUES(?,1) ON CONFLICT(document_id) "
                                 "DO UPDATE SET pending_delete=1", [(identity,) for identity in ids])

        async def invalidate_local(scope):
            if invalidate is None:
                raise RuntimeError("Local invalidator is not configured")
            await invalidate(scope)

        try:
            # Hold the SAME lock through native extraction completion and delete.
            # Never call a missing queued document 'forgotten' before its job ends.
            await self._settle_pending(allow_submit=False)
        except Exception as exc:
            scope = ForgetScope(self.memory.scope.owner_id, self.memory.scope.bank_id, tuple(ids))
            local_error = None
            try:
                await invalidate_local(scope)
            except Exception:
                local_error = "local_invalidation_failed"
            code = exc.code if isinstance(exc, MemoryRequestError) else "retention_outcome_unknown"
            return ForgetResult(scope, (), {identity: code for identity in ids}, local_error is None, local_error)
        result = await self.memory.forget(ids, invalidate=invalidate_local)
        if result.local_invalidated:
            with self._db:
                self._db.executemany("UPDATE forgotten SET pending_delete=0 WHERE document_id=?",
                                     [(identity,) for identity in result.deleted_document_ids])
        return result

    async def close(self, session_id=None):
        """Flush a closed session, or flush a bounded final window and release.

        First drain the canonical capture writer. session_id keeps the owner-wide
        intake running; no argument shuts it down after flushing its current head.
        Other unsettled sessions can temporarily hold back a later closed session.
        """
        await self.initialize()
        async with self._lock:
            if session_id is not None:
                self._closed_sessions.add(str(session_id))
            ceiling = await self.source.high_watermark()
            try:
                while self.cursor < ceiling:
                    before = self.cursor
                    await self._tick_locked(force=session_id is None, through=ceiling)
                    if self.cursor == before:
                        break
            finally:
                if session_id is None:
                    self._closed = True
                    self._release()

    def _release(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._claimed_path is not None:
            _ACTIVE_CHECKPOINTS.discard(self._claimed_path)
            self._claimed_path = None

    def abort(self):
        """Release after the owning runner is stopped; pending receipts survive.

        This does not claim that a remote operation was cancelled or completed.
        The next initialize/run reconciles the saved receipt against Hindsight.
        """
        self._closed = True
        self._release()
