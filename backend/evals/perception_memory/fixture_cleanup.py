"""Narrow cleanup for explicitly tracked, disposable evaluation owners."""
from __future__ import annotations

import asyncpg


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _reference_join(edge) -> str:
    return " and ".join(
        "child." + _identifier(child) + " = parent." + _identifier(parent)
        for child, parent in zip(edge["child_columns"], edge["parent_columns"], strict=True)
    )


def _reference_scope(edge) -> str:
    return (
        "from public." + _identifier(edge["child_table"]) + " as child "
        "join public." + _identifier(edge["parent_table"]) + " as parent on "
        + _reference_join(edge)
        + " where parent.owner_id::text = any($1::text[])"
    )


async def cleanup_fixture_owners(dsn: str, owners: set[str]) -> dict:
    if not owners:
        return {"owners": [], "removed_rows": {}, "remaining_rows": 0}
    if any(not owner.startswith(("direct-image-eval-", "evidence-reinspection-")) for owner in owners):
        raise ValueError("Cleanup accepts only explicitly tracked evaluation owners.")
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=20)
    try:
        tables = await connection.fetch("""
            select c.relname as table_name
            from pg_class c join pg_namespace n on n.oid = c.relnamespace
            join pg_attribute a on a.attrelid = c.oid
            where n.nspname = 'public' and c.relkind in ('r', 'p')
              and a.attname = 'owner_id' and not a.attisdropped
            order by c.relname
        """)
        dependencies = await connection.fetch("""
            select child.relname as child_table, parent.relname as parent_table,
                array(
                    select a.attname
                    from unnest(fk.conkey) with ordinality k(attnum, position)
                    join pg_attribute a on a.attrelid = child.oid and a.attnum = k.attnum
                    order by k.position
                ) as child_columns,
                array(
                    select a.attname
                    from unnest(fk.confkey) with ordinality k(attnum, position)
                    join pg_attribute a on a.attrelid = parent.oid and a.attnum = k.attnum
                    order by k.position
                ) as parent_columns,
                not exists (
                    select 1 from unnest(fk.conkey) k(attnum)
                    join pg_attribute a on a.attrelid = child.oid and a.attnum = k.attnum
                    where a.attnotnull
                ) as nullable
            from pg_constraint fk
            join pg_class child on child.oid = fk.conrelid
            join pg_class parent on parent.oid = fk.confrelid
            join pg_namespace cn on cn.oid = child.relnamespace
            join pg_namespace pn on pn.oid = parent.relnamespace
            where fk.contype = 'f' and cn.nspname = 'public' and pn.nspname = 'public'
        """)
        counts_sql = " union all ".join(
            "select '" + row["table_name"].replace("'", "''") + "' as table_name, count(*) as row_count "
            + "from public." + _identifier(row["table_name"])
            + " where owner_id::text = any($1::text[])"
            for row in tables
        )
        if not counts_sql:
            raise RuntimeError("Owner-scoped cleanup tables are unavailable.")
        owner_tables = {row["table_name"] for row in tables}
        owner_ids = sorted(owners)
        removed_rows: dict[str, int] = {}
        async with connection.transaction():
            # Deletions can enqueue new owner-scoped jobs. Recount between sweeps,
            # with a finite bound so a recreating trigger cannot loop forever.
            for _ in range(len(owner_tables) + 1):
                rows = await connection.fetch(counts_sql, owner_ids)
                remaining = {row["table_name"] for row in rows if row["row_count"]}
                if not remaining:
                    break
                active_edges = []
                for edge in dependencies:
                    if edge["parent_table"] not in remaining or edge["child_table"] not in owner_tables:
                        # Ownerless junctions use their declared FK actions. Never
                        # issue a broad deletion against an ownerless table.
                        continue
                    scope = _reference_scope(edge)
                    if await connection.fetchval(
                        "select exists (select 1 " + scope
                        + " and not coalesce(child.owner_id::text = any($1::text[]), false))",
                        owner_ids,
                    ):
                        raise RuntimeError("Non-fixture rows reference fixture data; cleanup refused.")
                    if edge["child_table"] in remaining and await connection.fetchval(
                        "select exists (select 1 " + scope
                        + " and child.owner_id::text = any($1::text[]))", owner_ids,
                    ):
                        active_edges.append(edge)
                while remaining:
                    leaves = sorted(table for table in remaining if not any(
                        edge["parent_table"] == table and edge["child_table"] in remaining
                        and edge["child_table"] != table for edge in active_edges
                    ))
                    if not leaves:
                        edge = next((edge for edge in active_edges if edge["nullable"]
                                     and edge["child_table"] in remaining
                                     and edge["parent_table"] in remaining
                                     and edge["child_table"] != edge["parent_table"]), None)
                        if edge is None:
                            raise RuntimeError("Non-nullable fixture cycle requires targeted cleanup.")
                        assignments = [_identifier(column) + " = null" for column in edge["child_columns"]]
                        if (
                            edge["child_table"] == "experience_events"
                            and edge["parent_table"] == "memory_items"
                            and list(edge["child_columns"]) == ["promoted_episode_memory_id"]
                        ):
                            # The promotion marker is one checked state. Keep its
                            # fields consistent while detaching only fixture pairs.
                            assignments.extend((
                                '"promotion_decision" = \'pending\'',
                                '"promotion_reason" = null',
                                '"promotion_decided_at" = null',
                            ))
                        await connection.execute(
                            "update public." + _identifier(edge["child_table"]) + " as child set "
                            + ", ".join(assignments)
                            + " from public." + _identifier(edge["parent_table"]) + " as parent where "
                            + _reference_join(edge)
                            + " and child.owner_id::text = any($1::text[])"
                            + " and parent.owner_id::text = any($1::text[])", owner_ids,
                        )
                        active_edges.remove(edge)
                        continue
                    for table in leaves:
                        status = await connection.execute(
                            "delete from public." + _identifier(table)
                            + " where owner_id::text = any($1::text[])", owner_ids,
                        )
                        removed_rows[table] = removed_rows.get(table, 0) + int(status.split()[-1])
                        remaining.remove(table)
            after = await connection.fetch(counts_sql, owner_ids)
            if any(row["row_count"] for row in after):
                raise RuntimeError("Fixture-owned rows remain after cleanup.")
        return {"owners": owner_ids, "removed_rows": removed_rows, "remaining_rows": 0}
    finally:
        await connection.close()
