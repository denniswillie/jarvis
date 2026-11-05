"""Master-selected resource maintenance, not a general scheduling system.

The backend timer pauses/resumes existing sandboxes. It never runs a user script
or calls a model to decide what to do. Successful renewals stay in local status
and logs. Only failed or unconfirmed cycles need the retained master's attention.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from app.sandbox.models import SandboxError


logger = logging.getLogger(__name__)


class KeepaliveJournal:
    """Private policy and notification receipts for the existing resource owner."""

    def __init__(self, path: Path, *, bindings):
        self.owner_id, self.profile_id = bindings.owner_id, bindings.profile_id
        path = Path(path).absolute()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA synchronous=FULL")
        try:
            with self._db:
                self._db.execute("CREATE TABLE IF NOT EXISTS identity (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                for key, value in (("owner", self.owner_id), ("profile", self.profile_id)):
                    digest = hashlib.sha256(value.encode()).hexdigest()
                    self._db.execute("INSERT OR IGNORE INTO identity VALUES (?, ?)", (key, digest))
                    if self._db.execute("SELECT value FROM identity WHERE key=?", (key,)).fetchone()[0] != digest:
                        raise SandboxError("sandbox_keepalive_identity_mismatch")
                self._db.execute("""CREATE TABLE IF NOT EXISTS policies (
                    target TEXT PRIMARY KEY, master TEXT NOT NULL, sandbox_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL, due_at REAL NOT NULL)""")
                self._db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                    event_id TEXT PRIMARY KEY, target TEXT NOT NULL, master TEXT NOT NULL,
                    sandbox_id TEXT NOT NULL, started_at REAL NOT NULL,
                    result TEXT, delivered INTEGER NOT NULL DEFAULT 0)""")
                self._db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS one_pending_attempt
                    ON attempts(target) WHERE result IS NULL""")
        except BaseException:
            self.close()
            raise

    def close(self):
        self._db.close()

    def enable(self, *, target, master, sandbox_id, due_at):
        with self._db:
            self._db.execute("""INSERT INTO policies VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(target) DO UPDATE SET master=excluded.master, sandbox_id=excluded.sandbox_id,
                enabled=1, due_at=excluded.due_at""", (target, master, sandbox_id, due_at))

    def disable(self, target):
        with self._db:
            self._db.execute("UPDATE policies SET enabled=0 WHERE target=?", (target,))

    def schedule(self, target, due_at):
        with self._db:
            self._db.execute("UPDATE policies SET due_at=? WHERE target=?", (due_at, target))

    def policy(self, target):
        row = self._db.execute("SELECT * FROM policies WHERE target=?", (target,)).fetchone()
        return dict(row) if row else None

    def due(self, now):
        return [dict(row) for row in self._db.execute("""SELECT * FROM policies WHERE enabled=1 AND due_at<=?
            AND NOT EXISTS (SELECT 1 FROM attempts WHERE attempts.target=policies.target AND result IS NULL)
            ORDER BY due_at""", (now,))]

    def begin(self, policy, now):
        event = dict(event_id="sandbox-maintenance-" + uuid4().hex, target=policy["target"],
                     master=policy["master"], sandbox_id=policy["sandbox_id"], started_at=now)
        with self._db:
            self._db.execute("""INSERT INTO attempts(event_id,target,master,sandbox_id,started_at)
                VALUES (:event_id,:target,:master,:sandbox_id,:started_at)""", event)
        return event

    def unfinished(self):
        return [dict(row) for row in self._db.execute("SELECT * FROM attempts WHERE result IS NULL ORDER BY started_at")]

    def finish(self, event, *, outcome, observed_state=None, error=None):
        payload = {key: event[key] for key in ("event_id", "target", "master", "sandbox_id", "started_at")}
        payload.update(kind="sandbox_maintenance", outcome=outcome, observed_state=observed_state)
        if error is not None:
            payload["error"] = error
        silent = outcome == "micro_pause_resumed" and error is None
        with self._db:
            self._db.execute("UPDATE attempts SET result=?, delivered=? WHERE event_id=? AND result IS NULL",
                             (json.dumps(payload), int(silent), event["event_id"]))
        if silent:
            logger.info("Hermes sandbox renewed session=%s sandbox=%s event=%s",
                        event["target"], event["sandbox_id"], event["event_id"])

    def pending_notices(self):
        return [json.loads(row[0]) for row in self._db.execute("""SELECT result FROM attempts
            WHERE result IS NOT NULL AND delivered=0 ORDER BY started_at LIMIT 4""")]

    def acknowledge(self, event_id):
        with self._db:
            self._db.execute("UPDATE attempts SET delivered=1 WHERE event_id=? AND result IS NOT NULL", (event_id,))


class SandboxKeepalive:
    def __init__(self, *, master_session_id: str, sandboxes, journal: KeepaliveJournal,
                 deliver, clock=time.time, notification_timeout=3):
        if (journal.owner_id, journal.profile_id) != (sandboxes.bindings.owner_id, sandboxes.bindings.profile_id):
            raise ValueError("Keepalive and sandbox profiles differ")
        if not isinstance(master_session_id, str) or not master_session_id:
            raise ValueError("A trusted retained master binding is required")
        if not 0 < notification_timeout <= 30:
            raise ValueError("Invalid notification timeout")
        self.master, self.sandboxes, self.journal = master_session_id, sandboxes, journal
        self.deliver, self.clock, self.notification_timeout = deliver, clock, notification_timeout
        self._lock = asyncio.Lock()

    def _master_only(self, actor_session_id):
        # actor_session_id must come from native caller resolution, not model
        # arguments. The future tool adapter owns that boundary.
        if actor_session_id != self.master:
            raise SandboxError("sandbox_keepalive_master_required")

    @staticmethod
    def _due_at(info):
        if info.expires_at is None:
            raise SandboxError("sandbox_keepalive_deadline_unavailable")
        margin = min(300, max(10, info.policy.lifetime_seconds // 10))
        return info.expires_at.timestamp() - margin

    @staticmethod
    def _code(error):
        return error.code if isinstance(error, SandboxError) else type(error).__name__

    async def enable(self, *, actor_session_id, target_session_id):
        self._master_only(actor_session_id)
        async with self._lock:
            info = await self.sandboxes.inspect(target_session_id)
            if info.state != "running":
                raise SandboxError("session_sandbox_paused")
            self.journal.enable(target=target_session_id, master=self.master,
                                sandbox_id=info.sandbox_id, due_at=self._due_at(info))
            return self.journal.policy(target_session_id)

    async def disable(self, *, actor_session_id, target_session_id):
        self._master_only(actor_session_id)
        async with self._lock:
            self.journal.disable(target_session_id)
            # Disabling maintenance neither stops programs nor pauses the VM.
            return self.journal.policy(target_session_id)

    async def _reconcile(self, event, *, error=None):
        try:
            info = await self.sandboxes.recover(event["target"])
            if info.sandbox_id != event["sandbox_id"]:
                raise SandboxError("session_sandbox_identity_mismatch")
            if info.state == "running":
                self.journal.schedule(event["target"], max(self.clock() + 5, self._due_at(info)))
                self.journal.finish(event, outcome="recovered_running_cycle_unconfirmed",
                                    observed_state="running", error=error)
            else:
                self.journal.disable(event["target"])
                self.journal.finish(event, outcome="paused_requires_master", observed_state=info.state, error=error)
        except asyncio.CancelledError:
            raise
        except Exception as failure:
            self.journal.disable(event["target"])
            self.journal.finish(event, outcome="unconfirmed", error=error or self._code(failure))

    async def tick(self):
        async with self._lock:
            # A crash may leave an accepted resume unfinished. Reconcile the
            # exact resource; do not start another cycle or replay user code.
            for event in self.journal.unfinished():
                await self._reconcile(event)
            for policy in self.journal.due(self.clock()):
                event = None
                try:
                    info = await self.sandboxes.inspect(policy["target"])
                    if info.sandbox_id != policy["sandbox_id"]:
                        raise SandboxError("session_sandbox_identity_mismatch")
                    if info.state != "running":
                        event = self.journal.begin(policy, self.clock())
                        self.journal.disable(policy["target"])
                        self.journal.finish(event, outcome="paused_requires_master", observed_state=info.state)
                        continue
                    due_at = self._due_at(info)
                    if due_at > self.clock():
                        self.journal.schedule(policy["target"], due_at)
                        continue  # Ordinary use extended the real deadline.
                    event = self.journal.begin(policy, self.clock())
                    resumed = await self.sandboxes.micro_pause(policy["target"])
                    self.journal.schedule(policy["target"], self._due_at(resumed))
                    self.journal.finish(event, outcome="micro_pause_resumed", observed_state="running")
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if event is None:
                        event = self.journal.begin(policy, self.clock())
                    await self._reconcile(event, error=self._code(error))

        # Success receipts are local status, including receipts queued by older
        # versions. Only failures/uncertainty enter native model admission.
        # A timeout keeps the same failure event for retry.
        for notice in self.journal.pending_notices():
            if notice["outcome"] == "micro_pause_resumed" and not notice.get("error"):
                self.journal.acknowledge(notice["event_id"])
                continue
            try:
                async with asyncio.timeout(self.notification_timeout):
                    acknowledged = await self.deliver(notice)
                if acknowledged is True:
                    self.journal.acknowledge(notice["event_id"])
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def run(self, stop: asyncio.Event, *, poll_seconds=5):
        if not 0.05 <= poll_seconds <= 30:
            raise ValueError("Invalid resource-maintenance polling interval")
        while not stop.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(stop.wait(), poll_seconds)
            except TimeoutError:
                pass
