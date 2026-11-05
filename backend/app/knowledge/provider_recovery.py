"""Operator-only preparation after an independently restored Hindsight database.

Uses existing source/outbox identities, not a new replay engine or model tool.
Stop application intake and workers externally; absence of a live lease is not
proof they are stopped. Finish every preparation page before restarting the
ingestion worker. Reads/admission remain quarantined until separate acceptance.
"""
from __future__ import annotations

from app.knowledge.lifecycle_journal import JournalCheckpoint
from app.knowledge.models import SourceError
from app.knowledge.sources import VERSION_SQL


class ProviderRecoveryService:
    def __init__(self, sources):
        self.sources = sources

    @staticmethod
    def validate_page(owner_id, checkpoint, after_document_id, limit):
        if checkpoint.owner_id != owner_id:
            raise SourceError("memory_recovery_owner_mismatch", "Recovery checkpoint scope differs.", 403)
        if type(limit) is not int or not 1 <= limit <= 128:
            raise ValueError("Invalid memory recovery page limit")
        if after_document_id is not None and (
            not isinstance(after_document_id, str) or not 1 <= len(after_document_id) <= 200
        ):
            raise ValueError("Invalid memory recovery cursor")

    async def prepare_page(self, owner_id, area_id, *, checkpoint: JournalCheckpoint,
                           after_document_id=None, limit=64):
        self.validate_page(owner_id, checkpoint, after_document_id, limit)
        async with self.sources.transaction(owner_id) as connection:
            # This seam intentionally refuses inactive areas. Their archive/
            # reactivation policy is not silently converted into physical deletion.
            await self.sources._area(connection, owner_id, area_id)
            head = await connection.fetchrow(
                "select epoch,last_sequence from jarvis_source_journal_streams where owner_id=$1 for update",
                owner_id,
            )
            if head is None or head["epoch"] != checkpoint.epoch or head["last_sequence"] != checkpoint.through_sequence:
                raise SourceError("memory_recovery_checkpoint_changed", "Re-establish the trusted quiescent checkpoint.")
            if await connection.fetchval(
                """select exists(select 1 from jarvis_memory_outbox o
                    join jarvis_knowledge_source_versions v on v.owner_id=o.owner_id
                    and v.document_id=o.document_id where v.owner_id=$1 and v.area_id=$2
                    and o.status='leased' and o.lease_until>clock_timestamp())""", owner_id, area_id,
            ):
                raise SourceError("memory_recovery_live_lease", "Stop active memory work before recovery preparation.")
            if after_document_id is not None and not await connection.fetchval(
                """select exists(select 1 from jarvis_knowledge_source_versions
                    where owner_id=$1 and area_id=$2 and document_id=$3)""",
                owner_id, area_id, after_document_id,
            ):
                raise SourceError("memory_recovery_cursor_scope", "The cursor is not in this source inventory.")
            versions = await connection.fetch(
                """select v.document_id,s.generation from jarvis_knowledge_source_versions v
                    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
                    where v.owner_id=$1 and v.area_id=$2 and ($3::text is null or v.document_id>$3)
                    order by v.document_id limit $4""",
                owner_id, area_id, after_document_id, limit + 1,
            )
            page = versions[:limit]
            document_ids = [row["document_id"] for row in page]
            # Reuse the canonical context/source join, projecting no original body.
            current = set(await connection.fetch(
                "select document_id from (" + VERSION_SQL + " and v.document_id=any($3::text[])) candidate "
                "where source_status='active' and availability='available' "
                "and revision=current_revision and context_is_current",
                owner_id, area_id, document_ids,
            ))
            current_ids = {row["document_id"] for row in current}
            rows = await connection.fetch(
                """select * from jarvis_memory_outbox where owner_id=$1
                    and document_id=any($2::text[]) for update""", owner_id, document_ids,
            )
            intents = {(row["document_id"], row["operation"]): row for row in rows}
            changed = {"retains": 0, "deletions": 0, "obsolete_retains_cancelled": 0}
            for version in page:
                document = version["document_id"]
                retain = intents.get((document, "retain"))
                cleanup = intents.get((document, "delete"))
                if retain is None or (document not in current_ids and cleanup is None):
                    # Inventing a replacement UUID could hide an unknown old send.
                    raise SourceError("memory_recovery_receipt_missing", "An authoritative operation receipt is missing.")
                if document in current_ids:
                    if cleanup is not None and cleanup["status"] != "cancelled":
                        raise SourceError("memory_recovery_conflicting_cleanup", "Current evidence has a cleanup conflict.")
                    status = "outcome_unknown" if retain["dispatch_count"] else "pending"
                    changed["retains"] += await self._requeue(connection, retain, version["generation"], status)
                else:
                    if retain["status"] not in {"searchable", "cancelled"}:
                        await connection.execute(
                            """update jarvis_memory_outbox set status='cancelled',fence=fence+1,
                                worker_id=null,lease_until=null,last_error='memory_restore_obsolete',
                                updated_at=clock_timestamp() where id=$1""", retain["id"],
                        )
                        changed["obsolete_retains_cancelled"] += 1
                    changed["deletions"] += await self._requeue(connection, cleanup, version["generation"], "pending")
            final = await connection.fetchrow(
                "select epoch,last_sequence from jarvis_source_journal_streams where owner_id=$1", owner_id,
            )
        return {
            "documents_considered": len(page), "changed": changed,
            "next_after_document_id": document_ids[-1] if len(versions) > limit else None,
            "complete": len(versions) <= limit,
            "checkpoint": JournalCheckpoint(owner_id=owner_id, epoch=final["epoch"],
                                            through_sequence=final["last_sequence"]).model_dump(mode="json"),
            "provider_contacted": False, "restoration_accepted": False,
            "application_release_authorized": False,
        }

    @staticmethod
    async def _requeue(connection, row, generation, status):
        # Repeating preparation before the worker starts does not keep increasing
        # fences or resetting deadlines. Dispatch and cleanup inventories survive.
        if (row["status"] == status and row["generation"] == generation
                and row["failure_count"] == 0 and row["worker_id"] is None and row["lease_until"] is None):
            return 0
        await connection.execute(
            """update jarvis_memory_outbox set status=$2,generation=$3,fence=fence+1,
                failure_count=0,worker_id=null,lease_until=null,
                next_attempt_at=clock_timestamp(),last_error='memory_restore_unverified',
                updated_at=clock_timestamp() where id=$1""", row["id"], status, generation,
        )
        return 1
