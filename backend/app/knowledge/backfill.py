"""Operator-only frozen legacy import through the existing source/outbox lifecycle.

No provider I/O or alternative extraction loop. The caller keeps the legacy
application stopped between batches; database locks freeze each observed batch.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
import heapq
import json
from uuid import UUID, uuid4

from app.knowledge.legacy_identity import MAPPING_VERSION, NAMESPACE, encoded, fingerprint, quoted
from app.knowledge.legacy_populations import BY_TABLE, EVIDENCE_TABLES, POPULATIONS, SOURCE_TABLES
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.read_provenance import SourceReadProvenance
from app.migrations.runner import quote_schema





def key_for(spec, value):
    values = [value[key] for key in spec.keys]
    return str(values[0]) if len(values) == 1 else encoded(values)


@dataclass(frozen=True)
class LegacyRow:
    table: str
    key: str
    value: dict
    input_digest: str

    @property
    def identity(self):
        return self.table, self.key

    @property
    def source_key(self):
        return self.table + "/" + self.key


class LegacySnapshot:
    def __init__(self, rows, counts):
        self.rows, self.counts = rows, counts
        self.children = defaultdict(list)
        for row in rows.values():
            spec = BY_TABLE[row.table]
            if spec.mode == "attached":
                self.children[(spec.parent, str(row.value[spec.parent_key]))].append(row)
        self.input_digest = fingerprint({
            "mapping_version": MAPPING_VERSION, "counts": counts,
            "rows": [(row.table, row.key, row.input_digest) for row in sorted(rows.values(), key=lambda r: r.identity)],
        })
        self.suppressed_memories = {
            str(row.value["source_memory_id"]) for row in rows.values()
            if row.table == "memory_forgetting_tombstones" and row.value["overridden_at"] is None
        }

    def references(self, row):
        references = set()
        data = row.value
        if row.table == "perception_observations" and data.get("primary_visual_frame_id"):
            references.add(("visual_frames", str(data["primary_visual_frame_id"])))
        if row.table == "memory_items" and data.get("source_experience_event_id"):
            references.add(("experience_events", str(data["source_experience_event_id"])))
        for child in self.children[row.identity]:
            item = child.value
            if child.table in {"memory_evidence", "experience_event_evidence"}:
                table = EVIDENCE_TABLES.get(item["evidence_type"], "unresolved:" + item["evidence_type"])
                references.add((table, str(item["evidence_ref_id"])))
            elif child.table in {"experience_event_keyframes", "perception_observation_frames"}:
                references.add(("visual_frames", str(item["visual_frame_id"])))
            elif child.table == "memory_entity_mentions":
                references.add(("memory_items", str(item["memory_item_id"])))
            elif child.table in {"memory_entity_relations", "memory_links"} and item.get("supporting_memory_id"):
                references.add(("memory_items", str(item["supporting_memory_id"])))
        return sorted(references)

    def ordered_sources(self):
        sources = {key: row for key, row in self.rows.items() if row.table in SOURCE_TABLES}
        remaining, consumers = {}, defaultdict(set)
        for key, row in sources.items():
            parents = {ref for ref in self.references(row) if ref in sources and ref != key}
            remaining[key] = len(parents)
            for parent in parents:
                consumers[parent].add(key)
        ready = [key for key, count in remaining.items() if count == 0]
        heapq.heapify(ready)
        ordered = []
        while ready:
            key = heapq.heappop(ready)
            ordered.append(sources[key])
            for child in sorted(consumers[key]):
                remaining[child] -= 1
                if remaining[child] == 0:
                    heapq.heappush(ready, child)
        seen = {row.identity for row in ordered}
        return ordered, [row for key, row in sorted(sources.items()) if key not in seen]

    def original(self, row):
        references = []
        for table, key in self.references(row):
            if (table, key) == row.identity:
                state = "self_reference_not_independent_evidence"
            elif (table, key) not in self.rows:
                state = "missing_or_outside_owner_scope"
            elif table in SOURCE_TABLES:
                state = "source_backed"
            else:
                state = "external_original_not_loaded_or_verified"
            reference = {"table": table, "key": key, "state": state}
            external = self.rows.get((table, key))
            if external is not None and BY_TABLE[table].mode == "external_only":
                reference["external_metadata"] = external.value
                reference["bytes_verified"] = False
            references.append(reference)
        # Decimal tokens from PostgreSQL JSON are strings, not rounded binary floats.
        return {
            "origin": {"system": "jarvis_legacy", "table": row.table, "key": row.key,
                       "mapping_version": MAPPING_VERSION},
            "record": {k: v for k, v in row.value.items() if k != "owner_id"},
            "attachments": [
                {"table": child.table, "record": {k: v for k, v in child.value.items() if k != "owner_id"}}
                for child in self.children[row.identity]
            ],
            "references": references,
            "meaning": (
                "Literal legacy record with its existing attribution, not a new observation or authority. "
                "Saved/generated claims may be wrong; is_explicit does not establish truth. "
                "Superseded claims and validity intervals are historical, not current confirmation. "
                "Image metadata does not establish available pixels. Missing references are evidence gaps. "
                "Decimal JSON number tokens are preserved as text."
            ),
        }


class LegacyBackfill:
    def __init__(self, sources):
        self.sources = sources
        self.provenance = SourceReadProvenance(sources)

    @asynccontextmanager
    async def snapshot(self, owner_id, *, max_rows=50000, max_bytes=64 * 1024 * 1024):
        if type(max_rows) is not int or not 1 <= max_rows <= 100000:
            raise ValueError("Invalid legacy snapshot row bound")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 128 * 1024 * 1024:
            raise ValueError("Invalid legacy snapshot byte bound")
        async with self.sources.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("set local lock_timeout='5s'; set local statement_timeout='30s'; set local timezone='UTC'")
                await connection.execute(
                    "select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-legacy-import:' || $1,0))",
                    owner_id)
                schema = quote_schema(await connection.fetchval("select current_schema()"))
                available = {row["table_name"] for row in await connection.fetch(
                    "select table_name from information_schema.tables where table_schema=current_schema()")}
                selected = []
                for spec in POPULATIONS:
                    if spec.table not in available:
                        if spec.optional:
                            continue
                        raise SourceError("legacy_population_missing", "An expected legacy population is missing.")
                    selected.append(spec)
                # Hold the input fixed until all local writes in this batch settle.
                await connection.execute("lock table " + ",".join(
                    schema + "." + quoted(spec.table) for spec in sorted(selected, key=lambda x: x.table)
                ) + " in share mode")
                # Validate after lock acquisition, including DDL committed while waiting.
                columns = defaultdict(set)
                for column in await connection.fetch(
                    "select table_name,column_name from information_schema.columns where table_schema=current_schema()"):
                    columns[column["table_name"]].add(column["column_name"])
                for spec in selected:
                    if spec.mode in {"source", "attached"} and columns[spec.table] != set(spec.fields):
                        raise SourceError("legacy_source_schema_changed",
                            "A source population has unmapped or missing columns; update the explicit mapping first.")
                rows, counts, byte_count = {}, {spec.table: 0 for spec in POPULATIONS}, 0
                for spec in selected:
                    projection = ",".join("'" + field + "',r." + quoted(field) for field in spec.fields)
                    joins, scope = "", "r.owner_id=$1"
                    if not spec.owned:
                        joins = (" join " + schema + "." + quoted(spec.parent) + " p on p.id=r." + quoted(spec.parent_key))
                        scope = "p.owner_id=$1"
                    order = ",".join("r." + quoted(key) for key in spec.keys)
                    query = ("select jsonb_build_object(" + projection + ")::text as original from "
                             + schema + "." + quoted(spec.table) + " r" + joins + " where " + scope
                             + " order by " + order + " limit $2")
                    records = await connection.fetch(query, owner_id, max_rows - len(rows) + 1)
                    for record in records:
                        raw = record["original"]
                        byte_count += len(raw.encode())
                        if len(rows) >= max_rows or byte_count > max_bytes:
                            raise SourceError("legacy_snapshot_too_large", "The explicit frozen snapshot bound was exceeded.")
                        value = json.loads(raw, parse_float=str)
                        key = key_for(spec, value)
                        if (spec.table, key) in rows:
                            raise SourceError("legacy_identity_collision", "The population key is not unique.")
                        rows[(spec.table, key)] = LegacyRow(spec.table, key, value, fingerprint(value))
                        counts[spec.table] += 1
                yield LegacySnapshot(rows, counts)

    async def _begin(self, owner_id, area_id, snapshot, run_id):
        async with self.sources.transaction(owner_id) as connection:
            await self.sources._area(connection, owner_id, area_id)
            row = await connection.fetchrow(
                "select * from jarvis_legacy_import_runs where owner_id=$1 and id=$2 for update", owner_id, run_id)
            if row is None:
                await connection.execute("""insert into jarvis_legacy_import_runs
                    (id,owner_id,area_id,mapping_version,input_digest,population_counts)
                    values($1,$2,$3,$4,$5,$6::jsonb)""",
                    run_id, owner_id, area_id, MAPPING_VERSION, snapshot.input_digest, encoded(snapshot.counts))
            elif (row["area_id"] != area_id or row["mapping_version"] != MAPPING_VERSION
                  or row["input_digest"] != snapshot.input_digest):
                raise SourceError("legacy_snapshot_changed",
                    "This run belongs to a different scope or frozen input. Do not promote its old cursor.")
            return {(r["population"], r["origin_key"]): dict(r) for r in await connection.fetch(
                "select * from jarvis_legacy_import_records where owner_id=$1 and run_id=$2", owner_id, run_id)}

    @staticmethod
    def _receipt(owner_id, run_id, row, disposition, *, reason=None, source=None):
        return {
            "owner_id": owner_id, "run_id": run_id, "population": row.table, "origin_key": row.key,
            "input_digest": row.input_digest, "disposition": disposition, "reason": reason,
            "source_id": source["id"] if source else None,
            "source_revision": source["current_revision"] if source else None,
            "source_generation": source["generation"] if source else None,
        }

    @staticmethod
    async def _save(connection, receipts):
        if not receipts:
            return
        keys = tuple(receipts[0])
        await connection.executemany(
            "insert into jarvis_legacy_import_records(" + ",".join(keys) + ") values("
            + ",".join("$" + str(i) for i in range(1, len(keys) + 1)) + ")",
            [tuple(row[key] for key in keys) for row in receipts])

    async def _source_row(self, connection, owner_id, area_id, key):
        return await connection.fetchrow("""select * from jarvis_knowledge_sources
            where owner_id=$1 and area_id=$2 and namespace=$3 and source_key=$4 for update""",
            owner_id, area_id, NAMESPACE, key)

    async def _import(self, owner_id, area_id, run_id, row, snapshot, receipts):
        raw_session = row.value.get("session_id")
        session_id = None
        if raw_session is not None:
            try:
                session_id = UUID(str(raw_session))
            except (ValueError, TypeError):
                return self._receipt(owner_id, run_id, row, "quarantined", reason="invalid_original_session")
            if ("jarvis_sessions", str(session_id)) not in snapshot.rows:
                return self._receipt(owner_id, run_id, row, "quarantined", reason="original_session_outside_owner_snapshot")
        content = snapshot.original(row)
        parent_metadata = []
        linked_payload = row.value.get("source_payload_id") if row.table == "jarvis_messages" else None
        if linked_payload:
            linked_payload = UUID(linked_payload)
            async with self.sources.transaction(owner_id) as connection:
                availability = await connection.fetchval("select lock_jarvis_copy_source($1,$2)", owner_id, linked_payload)
                if availability != "available":
                    return self._receipt(owner_id, run_id, row, "suppressed", reason="linked_original_unavailable")
        for reference in snapshot.references(row):
            if reference == row.identity or reference not in snapshot.rows or reference[0] not in SOURCE_TABLES:
                continue
            receipt = receipts.get(reference)
            if receipt is None or receipt["disposition"] == "quarantined":
                return self._receipt(owner_id, run_id, row, "quarantined", reason="supporting_source_not_imported")
            if receipt["disposition"] == "suppressed":
                return self._receipt(owner_id, run_id, row, "suppressed", reason="supporting_source_suppressed")
            try:
                parent = await self.sources.read(owner_id, area_id, receipt["source_id"])
            except SourceError:
                return self._receipt(owner_id, run_id, row, "quarantined", reason="supporting_source_unavailable")
            parent_metadata.append(parent.original_evidence())
        if len(encoded(content).encode()) > 131072:
            return self._receipt(owner_id, run_id, row, "quarantined", reason="original_exceeds_payload_bound")

        async def existing(connection):
            source = await self._source_row(connection, owner_id, area_id, row.source_key)
            if source is None:
                return None
            if source["status"] != "active":
                return self._receipt(owner_id, run_id, row, "suppressed", reason="existing_source_not_active", source=source)
            current = await self.sources._read(connection, owner_id, area_id, source["id"])
            if current.content != content:
                return self._receipt(owner_id, run_id, row, "quarantined", reason="existing_source_differs", source=source)
            if current.session_id != session_id:
                return self._receipt(owner_id, run_id, row, "quarantined", reason="existing_session_association_differs", source=source)
            return self._receipt(owner_id, run_id, row, "already_present", source=source)

        async with self.sources.transaction(owner_id) as connection:
            prior = await existing(connection)
            if prior is not None:
                await self._save(connection, [prior])
                return prior

        async def link_dependencies(connection, payload, originals):
            await connection.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
                [(owner_id, payload, original) for original in sorted(set(originals))])

        async def materialize(connection, originals=()):
            # Recheck at the mutation boundary rather than relying on the earlier read.
            prior = await existing(connection)
            if prior is not None:
                raise SourceError("legacy_source_changed", "A source changed during import admission.")
            if linked_payload and await connection.fetchval(
                "select lock_jarvis_copy_source($1,$2)", owner_id, linked_payload
            ) != "available":
                raise SourceError("legacy_source_changed", "The linked original changed during import admission.")
            payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", owner_id, encoded(content))
            dependencies = set(originals)
            if linked_payload:
                dependencies.add(linked_payload)
            await link_dependencies(connection, payload, dependencies)
            field = {"jarvis_messages": "created_at", "jarvis_notes": "updated_at",
                     "perception_observations": "started_at", "experience_events": "started_at",
                     "memory_items": "occurred_at", "memory_entities": "updated_at"}[row.table]
            stamp = row.value.get(field)
            source_time = datetime.fromisoformat(stamp) if stamp else None
            trust = "derived"
            if row.table == "jarvis_messages":
                trust = {"user": "user_input", "assistant": "assistant_output"}.get(row.value["role"], "external_content")
            elif row.table == "jarvis_notes":
                trust = "external_content"
            published = await self.sources.publish_in_transaction(connection, owner_id, SourcePublication(
                area_id=area_id, namespace=NAMESPACE, source_key=row.source_key, payload_id=payload,
                source_type="legacy:" + row.table, trust=trust, source_time=source_time, session_id=session_id,
                time_basis="Legacy " + row.table + "." + field + "; not independent proof of physical event time.",
                expected_revision=0))
            source = {"id": published.source_id, "current_revision": published.revision, "generation": published.generation}
            disposition, reason = "imported", None
            if row.table == "memory_items" and (
                row.key in snapshot.suppressed_memories or row.value.get("status") in {"forgotten", "deleted"}
            ):
                source = await connection.fetchrow("""update jarvis_knowledge_sources
                    set status='forgotten',generation=generation+1 where owner_id=$1 and id=$2 returning *""",
                    owner_id, published.source_id)
                await self.sources._retire_documents(connection, owner_id, source["id"], source["generation"],
                                                     reschedule_existing=False)
                disposition, reason = "suppressed", "legacy_forgetting_barrier"
            receipt = self._receipt(owner_id, run_id, row, disposition, reason=reason, source=source)
            await self._save(connection, [receipt])
            return receipt, payload

        async with self.sources.transaction(owner_id) as connection:
            if not parent_metadata:
                result, _ = await materialize(connection)
                return result
            # A frozen legacy event can have more parents than one interactive
            # read. Keep the normal bound, but compose all groups atomically.
            # No source, receipt or outbox intent is visible before every group
            # has passed the same revision/access/copy checks.
            result = dependent = None

            async def materialize_or_attach(connection, originals):
                nonlocal result, dependent
                if dependent is None:
                    result, dependent = await materialize(connection, originals)
                else:
                    await link_dependencies(connection, dependent, originals)
                return result, dependent

            size = self.provenance.reference_limit
            for offset in range(0, len(parent_metadata), size):
                await self.provenance.bind_in_transaction(
                    connection, owner_id, area_id, parent_metadata[offset:offset + size],
                    materialize_or_attach)
            return result

    async def run(self, owner_id, area_id, *, run_id=None, batch_size=64, max_rows=50000):
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValueError("Invalid import batch size")
        if not isinstance(area_id, UUID) or (run_id is not None and not isinstance(run_id, UUID)):
            raise ValueError("Exact area/run UUIDs are required")
        run_id = run_id or uuid4()
        async with self.snapshot(owner_id, max_rows=max_rows) as snapshot:
            receipts = await self._begin(owner_id, area_id, snapshot, run_id)
            ordered, cyclic = snapshot.ordered_sources()
            cyclic_keys = {row.identity for row in cyclic}
            completed = 0
            for row in ordered + cyclic:
                if row.identity in receipts:
                    continue
                if completed >= batch_size:
                    break
                if row.identity in cyclic_keys:
                    receipt = self._receipt(owner_id, run_id, row, "quarantined", reason="cyclic_legacy_evidence")
                    async with self.sources.transaction(owner_id) as connection:
                        await self._save(connection, [receipt])
                else:
                    receipt = await self._import(owner_id, area_id, run_id, row, snapshot, receipts)
                    # Unmaterialized outcomes still need an immutable disposition.
                    if receipt["source_id"] is None:
                        async with self.sources.transaction(owner_id) as connection:
                            await self._save(connection, [receipt])
                receipts[row.identity] = receipt
                completed += 1
            metadata = []
            for row in sorted(snapshot.rows.values(), key=lambda r: r.identity):
                if row.identity in receipts or row.table in SOURCE_TABLES:
                    continue
                spec = BY_TABLE[row.table]
                parent = receipts.get((spec.parent, str(row.value.get(spec.parent_key))))
                if spec.mode == "attached" and parent is None:
                    parent_key = (spec.parent, str(row.value.get(spec.parent_key)))
                    if parent_key in snapshot.rows and spec.parent in SOURCE_TABLES:
                        continue
                    metadata.append(self._receipt(owner_id, run_id, row, "quarantined",
                                                  reason="attachment_parent_missing_or_outside_scope"))
                    continue
                disposition = spec.mode
                source = None
                reason = None
                if disposition == "attached":
                    if parent["disposition"] == "suppressed" and parent["source_id"] is None:
                        disposition, reason = "suppressed", "parent_source_suppressed"
                    elif parent["source_id"] is None or parent["disposition"] == "quarantined":
                        disposition, reason = "quarantined", "parent_source_not_imported"
                    else:
                        source = {"id": parent["source_id"], "current_revision": parent["source_revision"],
                                  "generation": parent["source_generation"]}
                metadata.append(self._receipt(owner_id, run_id, row, disposition, reason=reason, source=source))
            async with self.sources.transaction(owner_id) as connection:
                await self._save(connection, metadata)
                source_keys = {row.source_key for row in snapshot.rows.values() if row.table in SOURCE_TABLES}
                missing = await connection.fetch("""select * from jarvis_knowledge_sources
                    where owner_id=$1 and area_id=$2 and namespace=$3""", owner_id, area_id, NAMESPACE)
                missing_count = sum(source["source_key"] not in source_keys for source in missing)
                rows = await connection.fetch(
                    "select * from jarvis_legacy_import_records where owner_id=$1 and run_id=$2", owner_id, run_id)
                dispositions = dict(Counter(r["disposition"] for r in rows))
                unresolved = dispositions.get("quarantined", 0) + missing_count
                complete = len(rows) == len(snapshot.rows) and not unresolved
                status = "complete" if complete else "requires_attention" if unresolved else "pending"
                await connection.execute("update jarvis_legacy_import_runs set status=$3 where owner_id=$1 and id=$2",
                                         owner_id, run_id, status)
                changed = await connection.fetchval("""select count(*) from jarvis_legacy_import_records r
                    join jarvis_knowledge_sources s on s.owner_id=r.owner_id and s.id=r.source_id
                    where r.owner_id=$1 and r.run_id=$2 and (s.current_revision<>r.source_revision
                        or s.generation<>r.source_generation)""", owner_id, run_id)
                session_issues = await connection.fetchval("""select count(*) from jarvis_legacy_import_records r
                    join jarvis_knowledge_source_versions v on v.owner_id=r.owner_id
                        and v.source_id=r.source_id and v.revision=r.source_revision
                    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
                    left join jarvis_knowledge_source_sessions ss on ss.owner_id=v.owner_id
                        and ss.source_id=v.source_id and ss.source_revision=v.revision
                    where r.owner_id=$1 and r.run_id=$2 and r.disposition in ('imported','already_present')
                        and p.content#>>'{record,session_id}' is not null
                        and ss.session_id::text is distinct from p.content#>>'{record,session_id}'""", owner_id, run_id)
                if changed or session_issues:
                    complete = False
                    status = "requires_attention"
                    await connection.execute("update jarvis_legacy_import_runs set status=$3 where owner_id=$1 and id=$2",
                                             owner_id, run_id, status)
            return {
                "run_id": str(run_id), "area_id": str(area_id), "mapping_version": MAPPING_VERSION,
                "input_digest": snapshot.input_digest, "status": status, "local_manifest_complete": complete,
                "population_counts": snapshot.counts, "expected_records": len(snapshot.rows), "recorded_records": len(rows),
                "dispositions": dispositions, "sources_processed_in_batch": completed,
                "previous_origins_missing": missing_count, "changed_source_receipts": changed,
                "session_association_issues": session_issues,
                "provider_contacted": False, "ready_for_cutover": False,
                "meaning": "Local import accounting only. Native ingestion, writer freeze and selected-route acceptance "
                           "are separate gates. External-only records and retained operational/rebuildable data were not "
                           "ingested or deleted. No image bytes or credentials were sent to a model.",
            }
