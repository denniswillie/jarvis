#!/usr/bin/env python3
"""One-time, offline migration of the existing native master's memory policy.

Default: read-only plan. --apply additionally requires Main's explicit offline
lane release. Never stops/starts a process, edits a launch request, or creates a
session. The only native write is upstream commit_refreshed_policy, hosted in
one SQLite transaction with upstream transcript guards and pre-commit checks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".jarvis-dev/hermes-session-h0.Ld4B6Z/source"
PROFILE = ROOT / ".jarvis-dev/integration-demo/profile"
STATE = PROFILE / "state.db"
BINDINGS = PROFILE / "jarvis/bindings.sqlite"
MASTER = "jarvis-8c0a9d419068115a5b2060aafa35acbe7ba481c90ee5bf4b798638489d2a0bf8"
NATIVE = "local-e4f2b997e9853186d3da83d14b0fef6ee8dda8aa0d84e8b4419c123e12da9c78"
POLICY_KEY = "gateway.local_policy.v1:" + NATIVE
BANK = "jarvis-native-d9098e00decdd8e3d75810bd89ebcbdf"
ORIGINAL_TOOLS = {"terminal", "jarvis_work", "jarvis_sandbox", "jarvis_domain"}
MEMORY = {"provider": "hindsight", "memory_enabled": False, "user_profile_enabled": False}


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


@contextmanager
def database(path, mode="ro"):
    conn = sqlite3.connect(path.as_uri() + "?mode=" + mode, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded(value) + "\n")


def binding_snapshot():
    adapter = load_json(PROFILE / "jarvis-adapter.json")
    with database(BINDINGS) as conn:
        profile = dict(conn.execute("SELECT * FROM profile WHERE singleton=1").fetchone())
        row = dict(conn.execute("SELECT * FROM conversations WHERE logical_id=?", (MASTER,)).fetchone())
    require(profile["profile_home"] == str(PROFILE) and profile["owner_id"] == adapter["owner_id"],
            "binding_profile_mismatch")
    require(row["native_id"] == NATIVE and row["parent_id"] is None and row["label"] == "Master",
            "master_binding_mismatch")
    require(row["launch_json"] == encoded(adapter["launch"]), "reservation_fingerprint_mismatch")
    require(set(adapter["launch"]["toolsets"]) == ORIGINAL_TOOLS, "launch_changed")
    return {"profile": profile, "binding": row, "adapter": adapter}


def history_snapshot(conn):
    checksum = hashlib.sha256()
    count = 0
    for row in conn.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (NATIVE,)):
        # Hash private rows in memory only. Never put messages/reasoning in reports.
        values = {key: ({"blob_hex": value.hex()} if isinstance(value, bytes) else value)
                  for key, value in dict(row).items()}
        checksum.update(encoded(values).encode() + b"\n")
        count += 1
    return {"message_rows": count, "sha256": checksum.hexdigest()}


def native_snapshot(conn):
    raw = conn.execute("SELECT value FROM state_meta WHERE key=?", (POLICY_KEY,)).fetchone()
    require(raw is not None, "native_policy_missing")
    receipt = json.loads(raw[0])
    require(receipt["session_id"] == NATIVE and receipt["entry"]["session_id"] == NATIVE,
            "native_or_transcript_identity_changed")
    require(receipt["profile_id"] == str(PROFILE) and receipt["request_id"] == MASTER,
            "native_creation_binding_changed")
    require(receipt.get("lineage", [NATIVE]) == [NATIVE], "unexpected_native_lineage")
    session = dict(conn.execute("SELECT * FROM sessions WHERE id=?", (NATIVE,)).fetchone())
    route = dict(conn.execute("SELECT * FROM gateway_routing WHERE scope='' AND session_key=?",
                              (receipt["route"],)).fetchone())
    epoch = dict(conn.execute("SELECT * FROM runtime_epoch WHERE singleton=1").fetchone())
    return {"receipt": receipt, "session": session, "route": route, "epoch": epoch,
            "history": history_snapshot(conn)}


def candidate(old):
    require(old.get("credential_ref") is None and old.get("config_secret_ref") is None,
            "credential_reference_requires_separate_rebinding")
    require(not old.get("safe_mode") and not old.get("ignore_user_config"), "unexpected_bypass_policy")
    config = json.loads(old["config_json"])
    require("memory" not in ((config.get("agent") or {}).get("disabled_toolsets") or []),
            "memory_explicitly_disabled")
    tools = set(old["toolsets"])
    require(tools in (ORIGINAL_TOOLS, ORIGINAL_TOOLS | {"memory"}), "unexpected_native_toolsets")
    require(config.get("memory") in (None, MEMORY), "unexpected_frozen_memory_settings")
    require(set(json.loads(old["request_json"])["toolsets"]) == ORIGINAL_TOOLS,
            "native_creation_request_changed")
    if tools == ORIGINAL_TOOLS | {"memory"} and config.get("memory") == MEMORY:
        return copy.deepcopy(old)
    new = copy.deepcopy(old)
    new["toolsets"] = sorted(ORIGINAL_TOOLS | {"memory"})
    config["memory"] = dict(MEMORY)
    new["config_json"] = encoded(config)
    return new


def check_provider_settings():
    from ruamel.yaml import YAML

    with (PROFILE / "config.yaml").open(encoding="utf-8") as stream:
        config = YAML(typ="safe").load(stream)
    require(all((config.get("memory") or {}).get(key) == value for key, value in MEMORY.items()),
            "profile_memory_settings_changed")
    hindsight = load_json(PROFILE / "hindsight/config.json")
    require(hindsight.get("bank_id") == BANK and hindsight.get("mode") == "local_external"
            and hindsight.get("memory_mode") == "tools"
            and hindsight.get("auto_retain") is False and hindsight.get("auto_recall") is False,
            "hindsight_settings_changed")


def dead_pid(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    raise RuntimeError("gateway_pid_still_present")


@contextmanager
def offline_lock(pid):
    dead_pid(pid)
    path = PROFILE / "gateway.lock"
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = os.fstat(fd)

        def check():
            dead_pid(pid)
            current = path.stat()
            require((current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino),
                    "gateway_lock_inode_changed")

        check()
        yield check
    finally:
        os.close(fd)  # Release the same lock inode; never unlink/overwrite gateway metadata.


def backup_database(source, target):
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    deadline = time.monotonic() + 20

    def progress(_status, _remaining, _total):
        require(time.monotonic() < deadline, "backup_deadline")

    with database(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst, pages=128, progress=progress, sleep=0.05)
        require(dst.execute("PRAGMA quick_check").fetchall() == [("ok",)], "backup_integrity_failed")


def backup(output, before, after, binding):
    backup_database(STATE, output / "state.before.sqlite")
    backup_database(BINDINGS, output / "bindings.before.sqlite")
    for source, name in ((PROFILE / "config.yaml", "config.before.yaml"),
                         (PROFILE / "jarvis-adapter.json", "jarvis-adapter.before.json"),
                         (PROFILE / "hindsight/config.json", "hindsight.before.json")):
        shutil.copyfile(source, output / name)
        os.chmod(output / name, 0o600)
    write_json(output / "native.before.json", before)
    write_json(output / "policy.proposed.json", after)
    write_json(output / "binding.before.json", binding)


class OfflineTransaction:
    """Transaction host only; all native mutation/lease logic remains upstream."""

    def __init__(self, conn, native_guards, before, proposed, binding, lock_check):
        self.conn, self.native_guards = conn, native_guards
        self.before, self.proposed, self.binding = before, proposed, binding
        self.lock_check = lock_check
        self.committed = False

    def _check_transcript_write_guards(self, *args, **kwargs):
        return self.native_guards._check_transcript_write_guards(*args, **kwargs)

    def _execute_write(self, callback):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.lock_check()
            require(native_snapshot(self.conn) == self.before, "native_state_changed_since_backup")
            require(binding_snapshot() == self.binding, "binding_changed_since_backup")
            check_provider_settings()
            lease_key = self.native_guards._session_turn_lease_key_on_conn(self.conn, NATIVE)
            # Fail even on stale rows: do not let native guards clean them as a side effect.
            require(self.conn.execute("SELECT 1 FROM session_turn_leases WHERE conversation_id=?",
                                      (lease_key,)).fetchone() is None, "turn_lease_present")
            require(self.conn.execute("SELECT 1 FROM compression_locks WHERE session_id=?",
                                      (NATIVE,)).fetchone() is None, "compression_lock_present")
            result = callback(self.conn)
            expected = copy.deepcopy(self.before)
            expected["receipt"]["policy"] = self.proposed
            expected["session"]["runtime_generation"] += 1
            require(native_snapshot(self.conn) == expected, "native_preservation_invariant_failed")
            require(binding_snapshot() == self.binding, "binding_preservation_invariant_failed")
            self.lock_check()
            self.conn.commit()
            self.committed = True
            return result
        except BaseException:
            self.conn.rollback()
            raise


def run(args):
    os.environ["HERMES_HOME"] = str(PROFILE)
    sys.path.insert(0, str(SOURCE))
    check_provider_settings()
    binding = binding_snapshot()
    with database(STATE) as conn:
        conn.execute("BEGIN")
        before = native_snapshot(conn)
    old = before["receipt"]["policy"]
    proposed = candidate(old)
    summary = {"master": MASTER, "native": NATIVE, "status": "planned",
               "generation_before": before["session"]["runtime_generation"],
               "toolsets_before": old["toolsets"], "toolsets_after": proposed["toolsets"],
               "memory_after": MEMORY, "history": before["history"],
               "creation_request_sha256": digest(old["request_json"]),
               "binding_sha256": digest(binding), "changed": proposed != old}
    if not args.apply or proposed == old:
        if proposed == old:
            summary["status"] = "already_applied"
        return summary
    require(args.offline_lane_released, "main_offline_lane_release_required")
    require(args.output is not None, "private_backup_directory_required")
    from hermes_state import SessionDB
    from hermes_state_mutation_prepared import commit_refreshed_policy

    with offline_lock(args.gateway_pid) as lock_check:
        args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(args.output, 0o700)
        backup(args.output, before, proposed, binding)
        native_guards = SessionDB(STATE, read_only=True)
        try:
            with database(STATE, "rw") as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                transaction = OfflineTransaction(conn, native_guards, before, proposed, binding, lock_check)
                generation = commit_refreshed_policy(
                    transaction, epoch=before["epoch"]["epoch"], session_id=NATIVE,
                    old=old, new=proposed, generation=before["session"]["runtime_generation"])
                summary.update(status="committed", generation_after=generation,
                               backup_directory=str(args.output), preservation_checked_before_commit=True)
        finally:
            native_guards.close()
        write_json(args.output / "report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline-lane-released", action="store_true")
    parser.add_argument("--gateway-pid", type=int, default=92121)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gateway_pid <= 0:
        parser.error("gateway-pid must be positive")
    logging.disable(logging.CRITICAL)
    try:
        result = run(args)
    except Exception as exc:
        # Reports never print arbitrary exception strings, native history or credentials.
        print(json.dumps({"status": "refused_or_failed", "error_type": type(exc).__name__,
                          "instruction": "Keep gateway offline; inspect private evidence before retry/restore."}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
