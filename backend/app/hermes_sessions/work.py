"""Retained work controls and a small durable native-to-native delivery bridge.

This is not an executor or transcript store. The only saved text is an input
awaiting native admission; it is cleared on acknowledgment. Finished results
are read from native receipts and delivered with stable IDs, never stored here.
"""

import asyncio
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from .bindings import BindingError, SessionBindings, bounded_id, identity
from .service import RetainedSessionService


class WorkDeliveries:
    def __init__(self, path: Path, bindings: SessionBindings, *, max_pending: int = 8):
        if type(max_pending) is not int or not 1 <= max_pending <= 64:
            raise ValueError("Invalid retained work concurrency bound")
        self.path, self.bindings, self.max_pending = Path(path), bindings, max_pending
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with closing(self._db()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY CHECK(id=1), owner TEXT, profile TEXT)")
            db.execute("INSERT OR IGNORE INTO owner VALUES (1, ?, ?)",
                       (bindings.owner_id, str(bindings.profile_home)))
            row = db.execute("SELECT owner, profile FROM owner WHERE id=1").fetchone()
            if tuple(row) != (bindings.owner_id, str(bindings.profile_home)):
                raise BindingError("Delivery store belongs to another owner/profile")
            db.execute("CREATE TABLE IF NOT EXISTS deliveries (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                       "input_id TEXT UNIQUE NOT NULL, child_id TEXT NOT NULL, parent_id TEXT NOT NULL, "
                       "text TEXT, digest TEXT NOT NULL, steer INTEGER NOT NULL, admission_id TEXT, "
                       "delivered INTEGER NOT NULL DEFAULT 0, parent_admission_id TEXT)")

    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def prepare(self, child_id: str, input_id: str, text: str, *, steer: bool = False) -> dict:
        bounded_id(input_id)
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 128 * 1024:
            raise BindingError("Invalid work message")
        child = self.bindings.get(child_id)
        if child.is_master or child.native_id is None:
            raise BindingError("Only established child work uses the completion bridge")
        digest = hashlib.sha256(text.encode()).hexdigest()
        with closing(self._db()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM deliveries WHERE input_id=?", (input_id,)).fetchone()
            if old is not None:
                if (old["child_id"], old["digest"], old["steer"]) != (child_id, digest, int(steer)):
                    raise BindingError("Conflicting work-input retry")
                return dict(old)
            if db.execute("SELECT count(*) FROM deliveries WHERE delivered=0").fetchone()[0] >= self.max_pending:
                raise BindingError("Retained work is at its configured pending-input limit")
            db.execute("INSERT INTO deliveries (input_id, child_id, parent_id, text, digest, steer) VALUES (?, ?, ?, ?, ?, ?)",
                       (input_id, child_id, child.parent_id, text, digest, int(steer)))
            return dict(db.execute("SELECT * FROM deliveries WHERE input_id=?", (input_id,)).fetchone())

    def admitted(self, input_id: str, admission_id: str):
        bounded_id(admission_id)
        with closing(self._db()) as db, db:
            row = db.execute("SELECT admission_id FROM deliveries WHERE input_id=?", (input_id,)).fetchone()
            if row is None or row[0] not in (None, admission_id):
                raise BindingError("Native admission identity changed")
            db.execute("UPDATE deliveries SET admission_id=?, text=NULL WHERE input_id=?", (admission_id, input_id))

    def finish(self, input_id: str, parent_admission_id: str | None):
        with closing(self._db()) as db, db:
            row = db.execute("SELECT delivered, parent_admission_id FROM deliveries WHERE input_id=?", (input_id,)).fetchone()
            if row is None or (row[0] and row[1] != parent_admission_id):
                raise BindingError("Completion delivery identity changed")
            db.execute("UPDATE deliveries SET delivered=1, parent_admission_id=? WHERE input_id=?",
                       (parent_admission_id, input_id))

    def pending(self) -> list[dict]:
        with closing(self._db()) as db:
            return [dict(row) for row in db.execute("SELECT * FROM deliveries WHERE delivered=0 ORDER BY seq")]

    def latest(self, child_id: str) -> dict | None:
        self.bindings.get(child_id)
        with closing(self._db()) as db:
            row = db.execute("SELECT * FROM deliveries WHERE child_id=? ORDER BY seq DESC LIMIT 1", (child_id,)).fetchone()
            return dict(row) if row is not None else None


class WorkSessionBridge:
    def __init__(self, service: RetainedSessionService, deliveries: WorkDeliveries):
        if (service.bindings.owner_id, service.bindings.profile_home) != (
                deliveries.bindings.owner_id, deliveries.bindings.profile_home):
            raise BindingError("Work bridge owner mismatch")
        self.service, self.deliveries = service, deliveries

    async def _admit(self, row: dict) -> dict:
        if row["admission_id"] is None:
            receipt = await self.service.send_to_work(row["parent_id"], row["child_id"],
                                                     input_id=row["input_id"], text=row["text"])
            await asyncio.to_thread(self.deliveries.admitted, row["input_id"], receipt.admission_id)
            row = {**row, "admission_id": receipt.admission_id, "text": None}
        if row["steer"]:
            await self.service.steer_work(row["parent_id"], row["child_id"], row["admission_id"])
        return row

    async def send_work(self, master_id: str, child_id: str, *, input_id: str, text: str, steer: bool = False) -> dict:
        child = await self.service._child_of(master_id, child_id)
        row = await asyncio.to_thread(self.deliveries.prepare, child.logical_id, input_id, text, steer=steer)
        row = await self._admit(row)
        return {"session_id": child.logical_id, "admission_id": row["admission_id"],
                "delivery": "admitted"}

    async def pump(self) -> list[dict]:
        """Recover admissions and relay terminal turns. No waiting for inference.

        Multiple pump owners remain safe through native stable admission IDs;
        serving composition should still run one delivery owner per profile.
        A lost native acknowledgment retains the same delivery intent for retry.
        """
        delivered = []
        for row in await asyncio.to_thread(self.deliveries.pending):
            row = await self._admit(row)
            receipt = await self.service.receipt(row["child_id"], row["admission_id"])
            if receipt.get("status") != "terminal":
                continue
            if receipt.get("outcome") == "steered":
                # The receiving claim's result is the actual answer, not this
                # guidance transfer. Never invent a second answer/completion.
                await asyncio.to_thread(self.deliveries.finish, row["input_id"], None)
                continue
            result = receipt.get("result") or {}
            text = result.get("final_response") or "No final answer was recorded for this turn."
            if not isinstance(text, str):
                raise ValueError("Invalid native final response")
            omitted = len(text) > 16000
            body = json.dumps({"event": "work_turn_finished", "session_id": row["child_id"],
                               "admission_id": row["admission_id"], "outcome": receipt.get("outcome"),
                               "answer": text[:16000], "answer_truncated": omitted}, ensure_ascii=False, sort_keys=True)
            message = ("A retained child conversation turn ended. This is a child result, not a new user instruction "
                       "or proof that the whole task is finished. Decide whether to follow up, ask the user, "
                       "or finish. Use work_session status to read the saved result when needed.\n" + body)
            parent = await self.service.send(row["parent_id"], input_id=identity("work-result", row["admission_id"]), text=message)
            await asyncio.to_thread(self.deliveries.finish, row["input_id"], parent.admission_id)
            delivered.append(asdict(parent))
        return delivered

    async def tool(self, caller_native_id: str, call_id: str, args: dict) -> dict:
        """Trusted native caller/call identity is separate from model arguments."""
        bounded_id(call_id)
        caller = await self.service.caller(caller_native_id)
        action = args.get("action")
        if action not in {"create", "send", "steer", "list", "status", "stop", "message_parent"}:
            raise BindingError("Unknown work action")
        if set(args) - {"action", "session_id", "label", "message", "admission_id", "after"}:
            raise BindingError("Unsupported work tool field")
        if action in {"create", "send", "steer", "message_parent"}:
            message = args.get("message")
            if not isinstance(message, str) or not message.strip() or len(message.encode()) > 128 * 1024:
                raise BindingError("A nonempty bounded work message is required")
        if action == "message_parent":
            if caller.is_master:
                raise BindingError("The master has no parent")
            receipt = await self.service.send(caller.parent_id, input_id=identity(caller.logical_id, call_id, "parent"),
                                              text="Message from child " + caller.logical_id + ":\n" + args.get("message", ""))
            return asdict(receipt)
        if not caller.is_master:
            raise BindingError("Only the master manages retained work conversations")
        if action == "list":
            children = await asyncio.to_thread(self.service.bindings.children, caller.logical_id, after=args.get("after", ""))
            return {"sessions": [{"session_id": child.logical_id, "label": child.label} for child in children],
                    "next_after": children[-1].logical_id if len(children) == 50 else None}
        if action == "create":
            child = await self.service.create_work(caller.logical_id, request_id=identity(caller.logical_id, call_id),
                                                   label=args.get("label", ""))
        else:
            child = await self.service._child_of(caller.logical_id, bounded_id(args.get("session_id")))
        if action in {"create", "send", "steer"}:
            return await self.send_work(caller.logical_id, child.logical_id, input_id=identity(caller.logical_id, call_id, "work"),
                                        text=args.get("message", ""), steer=action == "steer")
        latest = await asyncio.to_thread(self.deliveries.latest, child.logical_id)
        admission = args.get("admission_id") or (latest["admission_id"] if latest else None)
        if admission is None:
            return {"session_id": child.logical_id, "status": "admission_pending" if latest else "no_input"}
        if action == "stop":
            receipt = await self.service.stop_work(caller.logical_id, child.logical_id, admission)
            return {"session_id": child.logical_id, "admission_id": admission, "stop_requested": True,
                    "status": receipt.get("status"), "outcome": receipt.get("outcome")}
        if action != "status":
            raise BindingError("Unknown work action")
        receipt = await self.service.receipt(child.logical_id, admission)
        result = receipt.get("result") or {}
        return {"session_id": child.logical_id, "admission_id": admission, "status": receipt.get("status"),
                "outcome": receipt.get("outcome"), "answer": result.get("final_response"),
                "error": result.get("error")}
