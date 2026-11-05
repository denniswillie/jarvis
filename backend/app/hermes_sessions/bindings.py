"""Private owner/profile bindings, not another Hermes conversation or job store.

Only the relationship and frozen native creation request are stored here.
Transcripts, admissions, results, scheduling and execution remain native-owned.
The eventual sandbox service binds to logical_id, not a physical transcript.
"""

from contextlib import closing, contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3


class BindingError(ValueError):
    pass


def identity(*parts: str) -> str:
    return "jarvis-" + hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def bounded_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or "\x00" in value:
        raise BindingError("A nonempty, bounded identity is required")
    return value


@dataclass(frozen=True)
class SessionBinding:
    logical_id: str
    parent_id: str | None
    label: str
    native_id: str | None
    launch_json: str

    @property
    def is_master(self) -> bool:
        return self.parent_id is None


class SessionBindings:
    """One private database per trusted user profile; no arbitrary model paths.

    Short SQLite transactions serialize competing callers, including processes.
    No transaction or connection is held across a native RPC or model call.
    Native creation's stable request ID closes the reserve/create/bind crash gap.
    """

    def __init__(self, path: Path, *, owner_id: str, profile_home: Path):
        self.owner_id = bounded_id(owner_id)
        self.profile_home = Path(profile_home).resolve()
        self.path = Path(path).absolute()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS profile (singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                       "owner_id TEXT NOT NULL, profile_home TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO profile VALUES (1, ?, ?)",
                       (self.owner_id, str(self.profile_home)))
            row = db.execute("SELECT owner_id, profile_home FROM profile WHERE singleton=1").fetchone()
            if tuple(row) != (self.owner_id, str(self.profile_home)):
                raise BindingError("Profile belongs to a different owner or home")
            db.execute("CREATE TABLE IF NOT EXISTS conversations (logical_id TEXT PRIMARY KEY, "
                       "parent_id TEXT REFERENCES conversations(logical_id), label TEXT NOT NULL, "
                       "native_id TEXT UNIQUE, launch_json TEXT NOT NULL)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_master ON conversations ((1)) WHERE parent_id IS NULL")

    @contextmanager
    def _transaction(self):
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            with db:
                yield db

    @staticmethod
    def _binding(row) -> SessionBinding:
        if row is None:
            raise BindingError("Unknown conversation in this owner profile")
        return SessionBinding(**dict(row))

    def get(self, logical_id: str) -> SessionBinding:
        bounded_id(logical_id)
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            return self._binding(db.execute("SELECT * FROM conversations WHERE logical_id=?", (logical_id,)).fetchone())

    def by_native(self, native_id: str) -> SessionBinding:
        bounded_id(native_id)
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            return self._binding(db.execute("SELECT * FROM conversations WHERE native_id=?", (native_id,)).fetchone())

    def get_or_reserve_master(self, *, launch: dict) -> SessionBinding:
        """Reuse the master's original creation request; defaults apply only if absent."""
        logical_id = identity(self.owner_id, "master", "", "master-v1")
        encoded = json.dumps(launch, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM conversations WHERE parent_id IS NULL").fetchone()
            if row is None:
                db.execute("INSERT INTO conversations VALUES (?, NULL, 'Master', NULL, ?)",
                           (logical_id, encoded))
                row = db.execute("SELECT * FROM conversations WHERE logical_id=?", (logical_id,)).fetchone()
            binding = self._binding(row)
            if binding.logical_id != logical_id or binding.label != "Master":
                raise BindingError("Master reservation does not match this owner profile")
            return binding

    def reserve(self, *, request_id: str, parent_id: str | None, label: str, launch: dict) -> SessionBinding:
        bounded_id(request_id)
        if not isinstance(label, str) or not label.strip() or len(label) > 160:
            raise BindingError("A short conversation label is required")
        logical_id = identity(self.owner_id, "work" if parent_id else "master", parent_id or "", request_id)
        encoded = json.dumps(launch, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        with self._transaction() as db:
            if parent_id is not None:
                parent = self._binding(db.execute("SELECT * FROM conversations WHERE logical_id=?", (parent_id,)).fetchone())
                if not parent.is_master or parent.native_id is None:
                    raise BindingError("Work conversations require their established master")
            db.execute("INSERT OR IGNORE INTO conversations VALUES (?, ?, ?, NULL, ?)",
                       (logical_id, parent_id, label, encoded))
            binding = self._binding(db.execute("SELECT * FROM conversations WHERE logical_id=?", (logical_id,)).fetchone())
            if (binding.parent_id, binding.label, binding.launch_json) != (parent_id, label, encoded):
                raise BindingError("Conflicting conversation creation retry")
            return binding

    def bind(self, logical_id: str, native_id: str) -> SessionBinding:
        bounded_id(native_id)
        with self._transaction() as db:
            binding = self._binding(db.execute("SELECT * FROM conversations WHERE logical_id=?", (logical_id,)).fetchone())
            if binding.native_id not in (None, native_id):
                raise BindingError("A retained conversation cannot be replaced")
            db.execute("UPDATE conversations SET native_id=? WHERE logical_id=?", (native_id, logical_id))
            return self._binding(db.execute("SELECT * FROM conversations WHERE logical_id=?", (logical_id,)).fetchone())

    def children(self, master_id: str, *, after: str = "", limit: int = 50) -> list[SessionBinding]:
        if not self.get(master_id).is_master:
            raise BindingError("Only a master owns work conversations")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise BindingError("Invalid conversation page size")
        with closing(sqlite3.connect(self.path, timeout=5)) as db:
            db.row_factory = sqlite3.Row
            return [self._binding(row) for row in db.execute(
                "SELECT * FROM conversations WHERE parent_id=? AND logical_id>? ORDER BY logical_id LIMIT ?",
                (master_id, after, limit))]
