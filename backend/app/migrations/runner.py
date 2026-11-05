from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import re

import asyncpg


GENERATED_MARKER = "-- BEGIN GENERATED VERSIONED MIGRATIONS (do not edit below this line)"
LEDGER_SQL = """create table if not exists jarvis_schema_migrations (
    version integer primary key check (version > 0),
    name text not null unique,
    checksum text not null check (checksum ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz not null default clock_timestamp()
);"""
FOUNDATION_TABLES = (
    "jarvis_knowledge_areas", "jarvis_context_streams", "jarvis_context_payloads",
    "jarvis_context_events", "jarvis_capability_preferences", "jarvis_operation_grants",
)
TASK_TABLES = (
    "jarvis_background_tasks", "jarvis_task_attempts", "jarvis_task_events", "jarvis_task_requests",
    "jarvis_task_grants", "jarvis_task_owner_scheduling", "jarvis_task_reservations", "jarvis_task_actions", "jarvis_task_scheduler_limits",
)
CONTEXT_TABLES = (
    "jarvis_context_ingress_receipts", "jarvis_context_pins", "jarvis_task_context_manifests", "jarvis_task_context_sources",
)

MEMORY_SOURCE_TABLES = (
    "jarvis_memory_banks", "jarvis_knowledge_sources", "jarvis_knowledge_source_versions", "jarvis_memory_outbox",
    "jarvis_knowledge_context_sources",
)


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return sha256(self.sql.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SchemaReadiness:
    ready: bool
    applied: tuple[int, ...]
    pending: tuple[int, ...]
    issues: tuple[str, ...]


class MigrationError(RuntimeError):
    pass


def quote_schema(schema: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
        raise ValueError("schema must be a lowercase SQL identifier")
    return '"' + schema + '"'


def load_migrations() -> tuple[Migration, ...]:
    result = []
    for item in sorted(files("app.migrations").joinpath("sql").iterdir(), key=lambda item: item.name):
        match = re.fullmatch(r"([0-9]{4})_([a-z0-9_]+)\.sql", item.name)
        if match:
            result.append(Migration(int(match[1]), match[2], item.read_text(encoding="utf-8")))
        elif item.name.endswith(".sql"):
            raise MigrationError("invalid migration filename")
    if not result:
        raise MigrationError("packaged migrations are missing")
    _validate_sequence(tuple(result))
    return tuple(result)


def _validate_sequence(migrations: tuple[Migration, ...]) -> None:
    if tuple(item.version for item in migrations) != tuple(range(1, len(migrations) + 1)):
        raise MigrationError("migration versions must be contiguous and ordered from one")
    if len({item.name for item in migrations}) != len(migrations):
        raise MigrationError("migration names must be unique")
    if any(not re.fullmatch(r"[a-z0-9_]+", item.name) for item in migrations):
        raise MigrationError("invalid migration name")


def baseline_sql() -> str:
    # The generated target must never double as renderer input: shell output
    # redirection opens/truncates it before this process starts.
    return files("app.migrations").joinpath("baseline.sql").read_text(encoding="utf-8").rstrip() + "\n"


def render_fresh_schema(migrations: tuple[Migration, ...] | None = None) -> str:
    migrations = load_migrations() if migrations is None else migrations
    _validate_sequence(migrations)
    pieces = [baseline_sql().rstrip(), "", GENERATED_MARKER, LEDGER_SQL]
    for item in migrations:
        pieces.extend((
            "", item.sql.rstrip(),
            "insert into jarvis_schema_migrations (version, name, checksum) values "
            f"({item.version}, '{item.name}', '{item.checksum}');",
        ))
    return "\n".join(pieces) + "\n"


async def _set_path(connection: asyncpg.Connection, schema: str) -> None:
    # Extension schemas are lookup-only. All tables created by migrations must
    # resolve in the explicitly selected, existing target schema.
    await connection.execute(f"set local search_path to {quote_schema(schema)}, pg_catalog, public, extensions")


async def _ledger(connection: asyncpg.Connection, schema: str):
    exists = await connection.fetchval("select to_regclass($1) is not null", f"{schema}.jarvis_schema_migrations")
    return await connection.fetch(f"select version, name, checksum from {quote_schema(schema)}.jarvis_schema_migrations order by version") if exists else []


def _ledger_issues(rows, migrations: tuple[Migration, ...]) -> list[str]:
    expected = {item.version: item for item in migrations}
    issues = []
    if tuple(row["version"] for row in rows) != tuple(range(1, len(rows) + 1)):
        issues.append("noncontiguous_ledger")
    for row in rows:
        item = expected.get(row["version"])
        if item is None:
            issues.append(f"unknown_version:{row['version']}")
        elif row["checksum"] != item.checksum or row["name"] != item.name:
            issues.append(f"checksum_or_name_mismatch:{item.version}")
    return issues


async def read_schema_readiness(
    connection: asyncpg.Connection, *, schema: str = "public", migrations: tuple[Migration, ...] | None = None,
) -> SchemaReadiness:
    quote_schema(schema)
    migrations = load_migrations() if migrations is None else migrations
    _validate_sequence(migrations)
    async with connection.transaction(readonly=True):
        rows = await _ledger(connection, schema)
        issues = _ledger_issues(rows, migrations)
        applied = tuple(row["version"] for row in rows)
        pending = tuple(item.version for item in migrations if item.version not in applied)
        if not await connection.fetchval("select to_regclass($1) is not null", f"{schema}.jarvis_sessions"):
            issues.append("baseline_missing")
        for version, tables in ((1, FOUNDATION_TABLES), (2, TASK_TABLES), (3, CONTEXT_TABLES), (4, MEMORY_SOURCE_TABLES),
                                (5, ("jarvis_task_workers",)), (6, ("jarvis_task_memory_sources",)),
                                (7, ("jarvis_task_session_context",)),
                                (8, ("jarvis_attention_intents", "jarvis_attention_attempts", "jarvis_attention_requests")),
                                (9, ("jarvis_attention_speech_chunks",)),
                                (12, ("jarvis_context_memory_sources",)),
                                (13, ("jarvis_source_journal_streams", "jarvis_source_journal_events")),
                                (16, ("jarvis_legacy_import_runs", "jarvis_legacy_import_records")),
                                (20, ("jarvis_observation_sources",)),
                                (21, ("jarvis_context_event_dependencies",)),
                                (22, ("jarvis_visual_delivery_sources",)),
                                (23, ("jarvis_reminder_attention_sources",)),
                                (24, ("jarvis_attention_visual_subjects",)),
                                (25, ("jarvis_push_devices", "jarvis_push_device_requests", "jarvis_push_submissions")),
                                (26, ("jarvis_sandbox_allocations",)),
                                (32, ("jarvis_session_knowledge_selections", "jarvis_knowledge_context_scopes")),
                                (33, ("jarvis_knowledge_area_versions",)),
                                (34, ("jarvis_project_connectors", "jarvis_project_roots", "jarvis_project_requests",
                                      "jarvis_project_file_watches")),
                                (35, ("jarvis_source_retention", "jarvis_source_retention_requests")),
                                (38, ("jarvis_responsibilities", "jarvis_responsibility_occurrences",
                                      "jarvis_responsibility_requests", "jarvis_responsibility_attention_delays")),
                                (39, ("jarvis_attention_route_evidence",)),
                                (43, ("jarvis_note_operations", "jarvis_workspace_requests"))):
            if version not in applied:
                continue
            for table in tables:
                if not await connection.fetchval("select to_regclass($1) is not null", f"{schema}.{table}"):
                    issues.append(f"missing_table:{table}")
        if 17 in applied and not await connection.fetchval(
            "select to_regclass($1) is not null", f"{schema}.jarvis_knowledge_source_sessions",
        ):
            issues.append("missing_table:jarvis_knowledge_source_sessions")
        if 42 in applied and not await attention_quiet_schema_ready(connection, schema=schema):
            issues.append("missing_attention_quiet_episode_schema")
        return SchemaReadiness(not pending and not issues, applied, pending, tuple(issues))


async def attention_quiet_schema_ready(connection, *, schema=None):
    """Local-schema capability check; never migrate or fall through to public.

    Older pinned fixtures can still exercise unrelated attention paths. Ended
    quiet episodes fail closed until this capability is actually installed.
    Full startup readiness above also checks migration sequence and checksums.
    """
    schema = schema or await connection.fetchval("select current_schema()")
    quoted = quote_schema(schema)
    if not await connection.fetchval("select to_regclass($1) is not null",
                                     f"{schema}.jarvis_schema_migrations"):
        return False
    return bool(await connection.fetchval(f"""select
        exists(select 1 from {quoted}.jarvis_schema_migrations
               where version=42 and name='attention_quiet_episodes')
        and (select count(*)=2 from pg_attribute
             where attrelid=to_regclass($1) and not attisdropped
               and attname in ('quiet_episode_id','quiet_summary_intent_id'))
        and (select count(*)=3 from pg_constraint where conrelid=to_regclass($1)
             and conname in ('jarvis_attention_quiet_pair',
                 'jarvis_attention_quiet_summary_owner','jarvis_attention_quiet_not_self'))
        """, f"{schema}.jarvis_attention_intents"))


async def apply_migrations(
    connection: asyncpg.Connection, *, schema: str = "public", fresh: bool = False,
    migrations: tuple[Migration, ...] | None = None,
) -> tuple[int, ...]:
    """Explicit administration on a dedicated connection, not hot migration.

    The deployment owner must quiesce/drain old runtime pools before schema
    changes and recreate them afterward (see this package's README). This
    function owns only its migration transaction: it cannot release another
    process's borrowers or safely retry their arbitrary in-flight work.
    """
    quote_schema(schema)
    migrations = load_migrations() if migrations is None else migrations
    _validate_sequence(migrations)
    async with connection.transaction():
        await connection.execute("set local lock_timeout = '10s'; set local statement_timeout = '60s'")
        await connection.execute(
            "select pg_advisory_xact_lock(hashtextextended(current_database() || ':jarvis-migrations:' || $1, 0))",
            schema,
        )
        if not await connection.fetchval("select exists(select 1 from pg_namespace where nspname=$1)", schema):
            raise MigrationError("target schema does not exist")
        await _set_path(connection, schema)
        if fresh:
            occupied = await connection.fetchval("""select exists (
                select 1 from pg_class c join pg_namespace n on n.oid=c.relnamespace
                where n.nspname=$1 and c.relkind in ('r','p','v','m','S','f')
                and not exists(select 1 from pg_depend d where d.classid='pg_class'::regclass
                               and d.objid=c.oid and d.deptype='e'))""", schema)
            if occupied:
                raise MigrationError("fresh installation requires an empty schema; use apply for upgrades")
            await connection.execute(baseline_sql())
        if not await connection.fetchval("select to_regclass($1) is not null", f"{schema}.jarvis_sessions"):
            raise MigrationError("baseline missing; initialize an empty database explicitly")
        await connection.execute(LEDGER_SQL)
        rows = await _ledger(connection, schema)
        issues = _ledger_issues(rows, migrations)
        if issues:
            raise MigrationError("; ".join(issues))
        applied = {row["version"] for row in rows}
        added = []
        for item in migrations:
            if item.version in applied:
                continue
            await connection.execute(item.sql)
            await connection.execute(
                "insert into jarvis_schema_migrations (version,name,checksum) values ($1,$2,$3)",
                item.version, item.name, item.checksum,
            )
            added.append(item.version)
        return tuple(added)
