"""Remove an exact imported message original, not independently edited history.

Existing immutable import receipts supply owner, origin ID, revision and digest.
No prose matching, new importer, original-content store or cleanup queue.
"""
from __future__ import annotations

import json
from uuid import UUID

import asyncpg

from app.knowledge.legacy_identity import NAMESPACE, fingerprint, quoted
from app.knowledge.legacy_populations import BY_TABLE
from app.knowledge.models import SourceError


BUSY = "legacy_original_cleanup_busy"
REMOVED = ("deleted", "expired", "unavailable")
UNRESOLVED = frozenset(("unmapped_identity", "current_origin_changed", "matching_unbound_original",
                        "shared_original_retained"))


def _message_id(source):
    if source["namespace"] != NAMESPACE or not source["source_key"].startswith("jarvis_messages/"):
        return None
    try:
        return UUID(source["source_key"].split("/", 1)[1])
    except ValueError:
        raise SourceError("legacy_origin_identity_invalid", "The imported message identity needs repair.") from None


async def inspect_imported_message_original(connection, owner_id, source, *, lock=False):
    message_id = _message_id(source)
    if message_id is None:
        return None
    if lock:
        # The importer holds its advisory lock and SHARE table locks while
        # writing sources through other connections. Never wait for those
        # locks while holding the source-owner lock in the reverse order.
        admitted = await connection.fetchval(
            "select pg_try_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-legacy-import:' || $1,0))",
            owner_id,
        )
        if not admitted:
            raise SourceError(BUSY, "A legacy import is active. Retry cleanup after its snapshot is released.")
        try:
            await connection.execute("lock table jarvis_messages in row exclusive mode nowait")
        except asyncpg.LockNotAvailableError:
            raise SourceError(BUSY, "A frozen legacy snapshot is active. Retry cleanup after it is released.") from None

    spec = BY_TABLE["jarvis_messages"]
    columns = {r["column_name"] for r in await connection.fetch(
        "select column_name from information_schema.columns "
        "where table_schema=current_schema() and table_name='jarvis_messages'"
    )}
    if columns != set(spec.fields):
        raise SourceError("legacy_source_schema_changed", "The original message mapping needs updating before cleanup.")
    # Match the import's exact timestamp and decimal-token representation.
    # SET LOCAL ends with the enclosing source transaction, never the pool.
    await connection.execute("set local timezone='UTC'")
    projection = ",".join("'" + field + "',m." + quoted(field) for field in spec.fields)
    raw = await connection.fetchval(
        "select jsonb_build_object(" + projection + ")::text from jarvis_messages m "
        "where m.owner_id=$1 and m.id=$2" + (" for update" if lock else ""),
        owner_id, message_id,
    )
    result = {"message_id": str(message_id), "state": "absent"}
    if raw is None:
        return result
    value = json.loads(raw, parse_float=str)
    if value["source_payload_id"] is not None:
        bound = await connection.fetchval(
            """select exists(select 1 from jarvis_knowledge_source_versions
                where owner_id=$1 and source_id=$2 and payload_id=$3)""",
            owner_id, source["id"], UUID(value["source_payload_id"]),
        )
        return result | {
            "state": ("source_bound_redacted" if value["content"] == "" else "source_bound")
                     if bound else "separate_source_retained",
            "source_payload_id": value["source_payload_id"],
        }
    input_digest = fingerprint(value)
    receipt = await connection.fetchrow(
        """select v.payload_id,p.availability from jarvis_legacy_import_records r
            join jarvis_knowledge_source_versions v on v.owner_id=r.owner_id
                and v.source_id=r.source_id and v.revision=r.source_revision
            join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
            where r.owner_id=$1 and r.source_id=$2 and r.population='jarvis_messages'
                and r.origin_key=$3 and r.input_digest=$4
                and r.disposition in ('imported','already_present')
            order by r.recorded_at,r.run_id limit 1""",
        owner_id, source["id"], str(message_id), input_digest,
    )
    if receipt is None:
        return result | {"state": "current_origin_changed"}
    # Different areas may have imported the same unchanged raw original into
    # different payloads. Payload-sharing guards alone cannot see that ownership.
    # The existing owner transaction serializes source changes. Preserve the
    # original until no retained import owns this exact recorded revision.
    shared = await connection.fetch(
        """select distinct s.id,s.area_id,s.current_revision,s.status
            from jarvis_legacy_import_records r
            join jarvis_knowledge_sources s on s.owner_id=r.owner_id and s.id=r.source_id
            where r.owner_id=$1 and r.population='jarvis_messages' and r.origin_key=$2
                and r.input_digest=$3 and r.disposition in ('imported','already_present')
                and s.id<>$4 and s.status<>'deleted'
            order by s.id limit 33""",
        owner_id, str(message_id), input_digest, source["id"],
    )
    if shared:
        return result | {
            "state": "shared_original_retained",
            "shared_original_sources": {
                "items": [{"source_id": str(r["id"]), "area_id": str(r["area_id"]),
                           "revision": r["current_revision"], "state": r["status"]}
                          for r in shared[:32]],
                "has_more": len(shared) > 32,
            },
        }
    return result | {"state": "matching_unbound_original",
                     "source_payload_id": str(receipt["payload_id"]),
                     "source_availability": receipt["availability"]}


async def delete_imported_message_original(connection, owner_id, source):
    """Trusted composition seam inside the already-authorized source deletion."""
    original = await inspect_imported_message_original(connection, owner_id, source, lock=True)
    if original is None or original["state"] != "matching_unbound_original":
        return original
    availability = original["source_availability"]
    state = availability if availability in REMOVED else "deleted"
    await connection.execute(
        """update jarvis_messages set content='',source_payload_id=$3,metadata=jsonb_build_object(
                'response_kind','source_bound_unavailable','source_status',$4::text,'source_payload_id',$3::uuid)
            where owner_id=$1 and id=$2 and source_payload_id is null""",
        owner_id, UUID(original["message_id"]), UUID(original["source_payload_id"]), state,
    )
    return original | {"state": "source_bound_redacted"}
