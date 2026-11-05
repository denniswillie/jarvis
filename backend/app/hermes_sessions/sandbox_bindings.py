"""Private resource references, not agent history or a background-job scheduler."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

from app.hermes_sessions.sandbox_provider import RetainedSandboxScope
from app.sandbox.models import SandboxError, SandboxPolicy


@dataclass(frozen=True)
class SessionSandboxBinding:
    scope: RetainedSandboxScope
    policy: SandboxPolicy
    sandbox_id: str | None
    creation_started: bool
    prepared: bool
    generation: int
    pending_action: str | None


class SessionSandboxBindings:
    """One writer per retained profile; store only exact resource associations.

    The process lease covers the service lifetime. SQLite transactions finish
    before provider calls. A crash releases the lease, but never the binding.
    """

    def __init__(self, path: Path, *, owner_id: str, profile_id: str):
        # Validate trusted identity before creating even the private directory.
        RetainedSandboxScope(owner_id=owner_id, profile_id=profile_id,
                             logical_session_id="identity-check", allocation_id=uuid4())
        self.owner_id, self.profile_id = owner_id, profile_id
        path = Path(path).absolute()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lease = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self._db = None
        try:
            try:
                fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SandboxError("session_sandbox_owner_busy") from exc
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            os.close(fd)
            self._db = sqlite3.connect(path, timeout=1)
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA synchronous=FULL")
            with self._db:
                self._db.execute("CREATE TABLE IF NOT EXISTS identity (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                self._db.execute("""CREATE TABLE IF NOT EXISTS sandboxes (
                    logical_session_id TEXT PRIMARY KEY,
                    allocation_id TEXT NOT NULL UNIQUE,
                    policy_json TEXT NOT NULL,
                    sandbox_id TEXT UNIQUE,
                    creation_started INTEGER NOT NULL DEFAULT 0,
                    prepared INTEGER NOT NULL DEFAULT 0,
                    generation INTEGER NOT NULL DEFAULT 0,
                    pending_action TEXT CHECK (pending_action IN ('pause', 'resume'))
                )""")
                for key, value in (("owner", owner_id), ("profile", profile_id)):
                    digest = hashlib.sha256(value.encode()).hexdigest()
                    self._db.execute("INSERT OR IGNORE INTO identity VALUES (?, ?)", (key, digest))
                    if self._db.execute("SELECT value FROM identity WHERE key=?", (key,)).fetchone()[0] != digest:
                        raise SandboxError("session_sandbox_identity_mismatch")
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lease is not None:
            os.close(self._lease)
            self._lease = None

    def get(self, logical_session_id: str) -> SessionSandboxBinding | None:
        row = self._db.execute("SELECT * FROM sandboxes WHERE logical_session_id=?", (logical_session_id,)).fetchone()
        if row is None:
            return None
        return SessionSandboxBinding(
            scope=RetainedSandboxScope(owner_id=self.owner_id, profile_id=self.profile_id,
                logical_session_id=row["logical_session_id"], allocation_id=UUID(row["allocation_id"])),
            policy=SandboxPolicy.model_validate_json(row["policy_json"]),
            sandbox_id=row["sandbox_id"], creation_started=bool(row["creation_started"]),
            prepared=bool(row["prepared"]), generation=row["generation"], pending_action=row["pending_action"],
        )

    def bound_session_ids(self) -> list[str]:
        """Existing resource owners for startup recovery, not another job list."""
        return [row[0] for row in self._db.execute(
            "SELECT logical_session_id FROM sandboxes WHERE sandbox_id IS NOT NULL ORDER BY logical_session_id")]

    def get_by_allocation(self, allocation_id: UUID) -> SessionSandboxBinding | None:
        """Read an exact existing resource; never reserve or change its owner."""
        if not isinstance(allocation_id, UUID):
            raise ValueError("An exact allocation UUID is required")
        row = self._db.execute("SELECT logical_session_id FROM sandboxes WHERE allocation_id=?",
                               (str(allocation_id),)).fetchone()
        return self.get(row[0]) if row is not None else None

    def reserve(self, logical_session_id: str, policy: SandboxPolicy) -> SessionSandboxBinding:
        scope = RetainedSandboxScope(owner_id=self.owner_id, profile_id=self.profile_id,
                                    logical_session_id=logical_session_id, allocation_id=uuid4())
        with self._db:
            self._db.execute("""INSERT OR IGNORE INTO sandboxes
                (logical_session_id, allocation_id, policy_json) VALUES (?, ?, ?)""",
                (scope.logical_session_id, str(scope.allocation_id), policy.model_dump_json()))
        result = self.get(logical_session_id)
        if result.policy != policy:
            raise SandboxError("session_sandbox_policy_changed")
        return result

    def start_creation(self, binding: SessionSandboxBinding):
        with self._db:
            updated = self._db.execute("""UPDATE sandboxes SET creation_started=1
                WHERE logical_session_id=? AND allocation_id=? AND creation_started=0 AND sandbox_id IS NULL""",
                (binding.scope.logical_session_id, str(binding.scope.allocation_id)))
            if updated.rowcount != 1:
                raise SandboxError("session_sandbox_creation_already_started")

    def bind(self, binding: SessionSandboxBinding, sandbox_id: str):
        if not isinstance(sandbox_id, str) or not sandbox_id or len(sandbox_id) > 200:
            raise SandboxError("session_sandbox_identity_mismatch")
        with self._db:
            updated = self._db.execute("""UPDATE sandboxes SET sandbox_id=?
                WHERE logical_session_id=? AND allocation_id=? AND creation_started=1
                AND (sandbox_id IS NULL OR sandbox_id=?)""",
                (sandbox_id, binding.scope.logical_session_id, str(binding.scope.allocation_id), sandbox_id))
            if updated.rowcount != 1:
                raise SandboxError("session_sandbox_identity_mismatch")

    def mark_prepared(self, binding: SessionSandboxBinding):
        with self._db:
            updated = self._db.execute("""UPDATE sandboxes SET prepared=1
                WHERE logical_session_id=? AND allocation_id=? AND sandbox_id=?""",
                (binding.scope.logical_session_id, str(binding.scope.allocation_id), binding.sandbox_id))
            if updated.rowcount != 1:
                raise SandboxError("session_sandbox_identity_mismatch")

    def intend(self, binding: SessionSandboxBinding, action: str) -> int:
        if action not in {"pause", "resume"}:
            raise SandboxError("session_sandbox_action_invalid")
        with self._db:
            updated = self._db.execute("""UPDATE sandboxes SET pending_action=?, generation=generation+1
                WHERE logical_session_id=? AND allocation_id=? AND sandbox_id=? AND generation=?""",
                (action, binding.scope.logical_session_id, str(binding.scope.allocation_id),
                 binding.sandbox_id, binding.generation))
            if updated.rowcount != 1:
                raise SandboxError("session_sandbox_generation_changed")
        return binding.generation + 1

    def settle(self, logical_session_id: str, generation: int):
        with self._db:
            updated = self._db.execute("""UPDATE sandboxes SET pending_action=NULL
                WHERE logical_session_id=? AND generation=?""", (logical_session_id, generation))
            if updated.rowcount != 1:
                raise SandboxError("session_sandbox_generation_changed")
