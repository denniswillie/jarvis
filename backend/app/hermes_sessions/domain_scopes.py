"""Immutable domain references for exact native inputs, never an admission queue."""

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sqlite3
from uuid import UUID

from .bindings import BindingError, SessionBinding, SessionBindings, bounded_id
from .domain_scope import DomainInputScope


class DomainScopeAssociations:
    def __init__(self, path: Path, bindings: SessionBindings):
        self.path, self.bindings = Path(path), bindings
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS profile "
                       "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT, home TEXT)")
            expected = (bindings.owner_id, str(bindings.profile_home))
            row = db.execute("SELECT owner, home FROM profile WHERE singleton=1").fetchone()
            if row is None:
                db.execute("INSERT INTO profile VALUES (1, ?, ?)", expected)
            elif tuple(row) != expected:
                raise BindingError("Domain scope owner/profile mismatch")
            db.execute("""CREATE TABLE IF NOT EXISTS scopes (
                logical_id TEXT NOT NULL, input_id TEXT NOT NULL, scope_json TEXT NOT NULL,
                kind TEXT NOT NULL, source_logical_id TEXT, source_input_id TEXT,
                PRIMARY KEY (logical_id, input_id),
                FOREIGN KEY (source_logical_id, source_input_id)
                    REFERENCES scopes(logical_id, input_id))""")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path)
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _encode(scope: DomainInputScope) -> str:
        data = asdict(scope)
        data["jarvis_session_id"] = str(scope.jarvis_session_id)
        data["request_event_id"] = str(scope.request_event_id) if scope.request_event_id else None
        data["allowed_tools"] = sorted(scope.allowed_tools)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _decode(raw: str) -> DomainInputScope:
        data = json.loads(raw)
        data["jarvis_session_id"] = UUID(data["jarvis_session_id"])
        data["request_event_id"] = UUID(data["request_event_id"]) if data["request_event_id"] else None
        data["allowed_tools"] = frozenset(data["allowed_tools"])
        return DomainInputScope(**data)

    def _owned(self, scope: DomainInputScope) -> SessionBinding:
        if scope.owner_id != self.bindings.owner_id:
            raise BindingError("Domain scope owner mismatch")
        return self.bindings.get(scope.logical_session_id)

    def _put(self, scope: DomainInputScope, kind: str,
             source: DomainInputScope | None = None) -> DomainInputScope:
        self._owned(scope)
        key = (scope.logical_session_id, scope.input_id)
        values = (self._encode(scope), kind,
                  source.logical_session_id if source else None,
                  source.input_id if source else None)
        with self._transaction() as db:
            old = db.execute("SELECT scope_json, kind, source_logical_id, source_input_id "
                             "FROM scopes WHERE logical_id=? AND input_id=?", key).fetchone()
            if old is not None and tuple(old) != values:
                raise BindingError("Native input already has a different domain scope")
            if old is None:
                db.execute("INSERT INTO scopes VALUES (?, ?, ?, ?, ?, ?)", key + values)
        return scope

    def bind_initial(self, scope: DomainInputScope) -> DomainInputScope:
        """Trusted host operation only; catalog/model arguments cannot bind authority."""
        return self._put(scope, "initial")

    def get(self, logical_session_id: str, native_input_id: str) -> DomainInputScope:
        self.bindings.get(logical_session_id)
        bounded_id(native_input_id)
        with self._transaction() as db:
            row = db.execute("SELECT scope_json FROM scopes WHERE logical_id=? AND input_id=?",
                             (logical_session_id, native_input_id)).fetchone()
        if row is None:
            raise BindingError("No domain scope for this exact native input")
        scope = self._decode(row[0])
        self._owned(scope)
        if (scope.logical_session_id, scope.input_id) != (logical_session_id, native_input_id):
            raise BindingError("Domain scope identity mismatch")
        return scope

    async def scope_for_input(self, binding: SessionBinding,
                              native_input_id: str) -> DomainInputScope:
        known = await asyncio.to_thread(self.bindings.get, binding.logical_id)
        if known != binding:
            raise BindingError("Domain scope binding mismatch")
        return await asyncio.to_thread(self.get, binding.logical_id, native_input_id)

    def latest_for_session(self, logical_session_id: str) -> DomainInputScope:
        """Recover the session association for a system wake, never a request ID."""
        self.bindings.get(logical_session_id)
        with self._transaction() as db:
            row = db.execute("SELECT scope_json FROM scopes WHERE logical_id=? "
                             "ORDER BY rowid DESC LIMIT 1", (logical_session_id,)).fetchone()
        if row is None:
            raise BindingError("No retained domain association for this session")
        scope = self._decode(row[0])
        self._owned(scope)
        return scope

    def derive(self, source: DomainInputScope, logical_session_id: str,
               native_input_id: str, *, kind: str) -> DomainInputScope:
        """Only parent/child inheritance; upward agent messages lose request authority."""
        origin = self._owned(source)
        if self.get(source.logical_session_id, source.input_id) != source:
            raise BindingError("Unbound source domain scope")
        target = self.bindings.get(logical_session_id)
        if kind == "work":
            valid = origin.is_master and target.parent_id == origin.logical_id
        elif kind in {"completion", "parent_message"}:
            valid = not origin.is_master and target.is_master and origin.parent_id == target.logical_id
        else:
            valid = False
        if not valid:
            raise BindingError("Invalid domain scope inheritance")
        scope = replace(source, logical_session_id=logical_session_id, input_id=native_input_id,
                        request_event_id=source.request_event_id if kind == "work" else None,
                        request_revision=source.request_revision if kind == "work" else None)
        return self._put(scope, kind, source)
