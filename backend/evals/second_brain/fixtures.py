"""Disposable-schema isolation on the real database, with independent cleanup.

No production owner data or schema is rewritten. Manifests are written before
resources are created and contain no database credentials. A cleanup failure is
an error, not an ignored test teardown warning.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg

from app.migrations.runner import quote_schema


class DatabaseFixture:
    def __init__(self, dsn: str, directory: Path, *, test_name: str):
        self.dsn = dsn
        self.run_id = uuid4().hex
        self.owners = tuple(f"direct-image-eval-second-brain-{self.run_id}-{suffix}" for suffix in ("a", "b"))
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"second-brain-{self.run_id}.json"
        self.manifest = {
            "format": "jarvis.second_brain.fixture.v1", "run_id": self.run_id, "test": test_name,
            "owners": self.owners, "schemas": [], "status": "prepared", "created_at": datetime.now(UTC).isoformat(),
        }
        self.persist()

    @classmethod
    def load(cls, dsn: str, path: Path):
        """Recover an existing validated fixture without allocating a new run."""
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        fixture = cls.__new__(cls)
        fixture.dsn, fixture.path, fixture.manifest = dsn, path, manifest
        fixture.run_id, fixture.owners = manifest["run_id"], tuple(manifest["owners"])
        return fixture

    def persist(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(self.path)

    async def create_schema(self, connection: asyncpg.Connection) -> str:
        if self.manifest["status"] == "cleaned":
            raise RuntimeError("A cleaned fixture cannot allocate new resources")
        schema = f"jarvis_sb_eval_{self.run_id}_{len(self.manifest['schemas'])}"
        self.manifest["schemas"].append(schema)
        self.persist()
        await connection.execute(f"create schema {quote_schema(schema)}")
        return schema

    async def cleanup(self) -> dict:
        self.manifest["status"] = "cleaning"
        self.persist()
        try:
            result = await cleanup_manifest(self.dsn, self.manifest)
        except Exception as error:
            self.manifest.update(status="cleanup_failed", cleanup_error_type=type(error).__name__)
            self.persist()
            raise
        self.manifest.update(status="cleaned", cleanup=result)
        self.persist()
        return result


def validate_manifest(manifest: dict) -> None:
    if manifest.get("format") != "jarvis.second_brain.fixture.v1":
        raise ValueError("unsupported fixture manifest")
    run_id = manifest["run_id"]
    if UUID(run_id).hex != run_id:
        raise ValueError("invalid fixture run identity")
    expected_owners = {f"direct-image-eval-second-brain-{run_id}-{suffix}" for suffix in ("a", "b")}
    if set(manifest["owners"]) != expected_owners:
        raise ValueError("fixture owner scope does not match its run")
    for index, schema in enumerate(manifest["schemas"]):
        if schema != f"jarvis_sb_eval_{run_id}_{index}":
            raise ValueError("fixture schema scope does not match its run")
        quote_schema(schema)


async def _owner_tables(connection, schema):
    return await connection.fetch("""select c.relname from pg_class c
        join pg_namespace n on n.oid=c.relnamespace join pg_attribute a on a.attrelid=c.oid
        where n.nspname=$1 and c.relkind in ('r','p') and a.attname='owner_id' and not a.attisdropped""", schema)


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


async def cleanup_manifest(dsn: str, manifest: dict) -> dict:
    validate_manifest(manifest)
    owners = list(manifest["owners"])
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=30)
    try:
        # Each owned schema is an independently retryable cleanup unit. Holding
        # every schema's DDL locks until one final commit exceeds small hosted
        # Postgres lock tables. Keep ownership/dependency checks and DROP in the
        # same transaction; retain the complete manifest for partial-failure retry.
        for schema in manifest["schemas"]:
            async with connection.transaction():
                for table in await _owner_tables(connection, schema):
                    await connection.execute(
                        f"lock table {quote_schema(schema)}.{_quote(table['relname'])} in access exclusive mode"
                    )
                    foreign = await connection.fetchval(
                        f"select exists(select 1 from {quote_schema(schema)}.{_quote(table['relname'])} "
                        "where owner_id is null or not owner_id::text=any($1::text[]))", owners,
                    )
                    if foreign:
                        raise RuntimeError("Non-fixture owner exists in disposable schema; cleanup refused")
                external_reference = await connection.fetchval("""select exists (
                    select 1 from pg_constraint f join pg_class parent on parent.oid=f.confrelid
                    join pg_namespace pn on pn.oid=parent.relnamespace
                    join pg_namespace childn on childn.oid=f.connamespace
                    where f.contype='f' and pn.nspname=$1 and childn.nspname<>$1)""", schema)
                if external_reference:
                    raise RuntimeError("External schema references fixture data; cleanup refused")
                external_dependency = await connection.fetchval("""with objects as (
                    select 'pg_class'::regclass as classid,c.oid from pg_class c
                        join pg_namespace n on n.oid=c.relnamespace where n.nspname=$1
                    union all select 'pg_proc'::regclass,p.oid from pg_proc p
                        join pg_namespace n on n.oid=p.pronamespace where n.nspname=$1
                    union all select 'pg_type'::regclass,t.oid from pg_type t
                        join pg_namespace n on n.oid=t.typnamespace where n.nspname=$1
                ) select exists(select 1 from objects o join pg_depend d
                    on d.refclassid=o.classid and d.refobjid=o.oid
                    cross join lateral pg_identify_object(d.classid,d.objid,d.objsubid) identified
                    left join pg_constraint constraint_owner on d.classid='pg_constraint'::regclass and constraint_owner.oid=d.objid
                    left join pg_namespace constraint_namespace on constraint_namespace.oid=constraint_owner.connamespace
                    left join pg_rewrite rewrite_owner on d.classid='pg_rewrite'::regclass and rewrite_owner.oid=d.objid
                    left join pg_attrdef default_owner on d.classid='pg_attrdef'::regclass and default_owner.oid=d.objid
                    left join pg_trigger trigger_owner on d.classid='pg_trigger'::regclass and trigger_owner.oid=d.objid
                    left join pg_class owning_relation on owning_relation.oid=
                        coalesce(rewrite_owner.ev_class,default_owner.adrelid,trigger_owner.tgrelid)
                    left join pg_namespace relation_namespace on relation_namespace.oid=owning_relation.relnamespace
                    where d.deptype='n' and coalesce(identified.schema,constraint_namespace.nspname,
                        relation_namespace.nspname) is distinct from $1)""", schema)
                # Table-owned rules/constraints/defaults have no schema from
                # pg_identify_object. Resolve that ownership explicitly. Any
                # other unidentified/global normal dependency fails closed.
                if external_dependency:
                    raise RuntimeError("External schema depends on fixture objects; cleanup refused")
                # DDL teardown has a separate bounded budget; ownership and
                # external-dependency checks retain the ordinary query deadline.
                await connection.execute(f"drop schema if exists {quote_schema(schema)} cascade", timeout=120)
    finally:
        await connection.close(timeout=5)
    # A new connection proves committed removal, not just transaction-local state.
    audit = await asyncpg.connect(dsn, timeout=10, command_timeout=30)
    try:
        remaining = await audit.fetchval("select count(*) from pg_namespace where nspname=any($1::text[])", manifest["schemas"])
        public_rows = 0
        for table in await _owner_tables(audit, "public"):
            public_rows += await audit.fetchval(
                f"select count(*) from public.{_quote(table['relname'])} where owner_id::text=any($1::text[])", owners,
            )
        if remaining or public_rows:
            raise RuntimeError("Independent fixture cleanup audit found residue")
        return {"remaining_schemas": remaining, "remaining_public_owner_rows": public_rows, "owners": owners}
    finally:
        await audit.close(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry cleanup of one explicit second-brain fixture manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    from app.settings import JarvisSettings
    fixture = DatabaseFixture.load(JarvisSettings().SUPABASE_DB_URL, args.manifest)
    result = asyncio.run(fixture.cleanup())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
