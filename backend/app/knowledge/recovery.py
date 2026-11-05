"""Operator-only evidence for quarantined physical PostgreSQL recovery.

This is NOT a restore engine, authorization tool, or application release gate.
Keep clients/workers stopped externally. A trusted checkpoint must survive outside
the backup being restored; matching it does not establish latestness, native-bank
erasure, object-storage recovery, or deployment compatibility.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.evidence.models import FrozenRecord, OwnedReference
from app.knowledge.lifecycle_journal import JournalCheckpoint
from app.migrations.runner import quote_schema


class RecoveryCheckpoint(FrozenRecord):
    format: Literal["jarvis.postgres_recovery.v1"] = "jarvis.postgres_recovery.v1"
    schema_name: str
    system_identifier: str = Field(strict=True, pattern=r"^[0-9]{1,20}$")
    timeline: int = Field(strict=True, ge=1, le=4294967295)
    target_lsn: str = Field(strict=True, pattern=r"^[0-9A-F]{1,8}/[0-9A-F]{1,8}$")
    migration_digest: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    journals: tuple[JournalCheckpoint, ...] = Field(min_length=1, max_length=256)

    @field_validator("schema_name")
    @classmethod
    def valid_schema(cls, value):
        quote_schema(value)
        return value

    @model_validator(mode="after")
    def unique_owners(self):
        owners = [pin.owner_id for pin in self.journals]
        for owner in owners:
            OwnedReference.validate_owner(owner)
        if len(set(owners)) != len(owners):
            raise ValueError("Duplicate recovery owner")
        return self


class RecoveryNotReady(RuntimeError):
    """A safe code, without original contents or database credentials."""


def _ledger_digest(rows):
    if not rows:
        raise RecoveryNotReady("recovery_migration_ledger_missing")
    value = [dict(row) for row in rows]
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_journals(checkpoint, rows):
    actual = {row["owner_id"]: row for row in rows}
    if set(actual) != {pin.owner_id for pin in checkpoint.journals}:
        raise RecoveryNotReady("recovery_journal_scope_missing")
    for pin in checkpoint.journals:
        row = actual[pin.owner_id]
        if row["epoch"] != pin.epoch or row["last_sequence"] != pin.through_sequence:
            raise RecoveryNotReady("recovery_journal_checkpoint_mismatch")
        # Unique(owner,sequence), count and endpoints establish contiguity.
        expected_edges = (1, pin.through_sequence) if pin.through_sequence else (None, None)
        if row["event_count"] != pin.through_sequence or (row["first"], row["last"]) != expected_edges:
            raise RecoveryNotReady("recovery_journal_gap")


def _check_state(checkpoint, state):
    if not state["in_recovery"] or state["pause_state"] != "paused":
        raise RecoveryNotReady("recovery_not_paused")
    if state["system_identifier"] != checkpoint.system_identifier or state["timeline"] != checkpoint.timeline:
        raise RecoveryNotReady("recovery_cluster_or_timeline_mismatch")
    if not state["target_matches"] or not state["target_reached"]:
        raise RecoveryNotReady("recovery_wal_target_mismatch")


async def capture_checkpoint(connection, *, schema, owners):
    """Capture a scoped WAL boundary after external intake/worker quiescence.

    Brief SHARE table locks hold source-journal writers and migration updates.
    This does not stop non-source writers, archive WAL, or prove all owners were
    supplied. The caller owns the complete deployment inventory and write barrier.
    """
    quoted = quote_schema(schema)
    if not 1 <= len(owners) <= 256 or len(set(owners)) != len(owners):
        raise ValueError("Invalid recovery owner inventory")
    for owner in owners:
        OwnedReference.validate_owner(owner)
    if connection.is_in_transaction():
        raise ValueError("Recovery capture requires its own transaction")
    async with connection.transaction():
        # READ COMMITTED: a lock wait cannot pin a pre-wait snapshot.
        await connection.execute(
            f"lock table {quoted}.jarvis_source_journal_streams, "
            f"{quoted}.jarvis_schema_migrations in share mode"
        )
        if await connection.fetchval("select pg_is_in_recovery()"):
            raise RecoveryNotReady("recovery_capture_requires_primary")
        rows = await connection.fetch(
            f"select owner_id,epoch,last_sequence from {quoted}.jarvis_source_journal_streams "
            "where owner_id=any($1::text[]) order by owner_id", list(owners),
        )
        if {row["owner_id"] for row in rows} != set(owners):
            raise RecoveryNotReady("recovery_journal_scope_missing")
        ledger = await connection.fetch(
            f"select version,name,checksum from {quoted}.jarvis_schema_migrations order by version"
        )
        state = await connection.fetchrow(
            """select system_identifier::text, timeline_id as timeline,
                pg_current_wal_insert_lsn()::text as target_lsn
                from pg_control_system(),pg_control_checkpoint()"""
        )
        return RecoveryCheckpoint(
            schema_name=schema, **dict(state), migration_digest=_ledger_digest(ledger),
            journals=tuple(JournalCheckpoint(owner_id=row["owner_id"], epoch=row["epoch"],
                                             through_sequence=row["last_sequence"]) for row in rows),
        )


async def verify_paused_recovery(connection, checkpoint: RecoveryCheckpoint):
    """Read-only evidence, valid only for this quarantined inspection instant."""
    quoted = quote_schema(checkpoint.schema_name)
    if connection.is_in_transaction():
        raise ValueError("Recovery verification requires its own transaction")
    async with connection.transaction(isolation="repeatable_read", readonly=True):
        # The pause-state function raises on a primary. Reject before calling it.
        if not await connection.fetchval("select pg_is_in_recovery()"):
            raise RecoveryNotReady("recovery_not_paused")
        state = await connection.fetchrow(
            """select pg_is_in_recovery() as in_recovery,
                pg_get_wal_replay_pause_state() as pause_state,
                system_identifier::text, timeline_id as timeline,
                (current_setting('recovery_target_lsn')::pg_lsn=$1::text::pg_lsn
                 and current_setting('recovery_target_action')='pause'
                 and current_setting('recovery_target_inclusive')='on'
                 and current_setting('recovery_target_timeline')=$2) as target_matches,
                coalesce(pg_last_wal_replay_lsn()>=$1::text::pg_lsn,false) as target_reached
                from pg_control_system(),pg_control_checkpoint()""",
            checkpoint.target_lsn, str(checkpoint.timeline),
        )
        _check_state(checkpoint, state)
        ledger = await connection.fetch(
            f"select version,name,checksum from {quoted}.jarvis_schema_migrations order by version"
        )
        if _ledger_digest(ledger) != checkpoint.migration_digest:
            raise RecoveryNotReady("recovery_migration_ledger_mismatch")
        rows = await connection.fetch(
            f"""select s.owner_id,s.epoch,s.last_sequence,count(e.sequence) as event_count,
                min(e.sequence) as first,max(e.sequence) as last
                from {quoted}.jarvis_source_journal_streams s
                left join {quoted}.jarvis_source_journal_events e on e.owner_id=s.owner_id
                where s.owner_id=any($1::text[]) group by s.owner_id,s.epoch,s.last_sequence""",
            [pin.owner_id for pin in checkpoint.journals],
        )
        _check_journals(checkpoint, rows)
        if (not await connection.fetchval("select pg_is_in_recovery()")
                or await connection.fetchval("select pg_get_wal_replay_pause_state()") != "paused"):
            raise RecoveryNotReady("recovery_changed_during_verification")
    return {
        "format": "jarvis.postgres_recovery_evidence.v1",
        "database_checkpoint_matches": True,
        "checked_owners": len(checkpoint.journals),
        "application_release_authorized": False,
    }
