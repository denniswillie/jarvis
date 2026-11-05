"""User-authorized source correction and logical suppression, not another memory engine."""
from __future__ import annotations

from app.knowledge.deletion import SourceDeletionService
from app.knowledge.models import SourceError, SourcePublication
from app.knowledge.product_notes import note_identity
from app.knowledge.product_files import file_identity
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest


class SourceLifecycleService:
    def __init__(self, sources, requests):
        if sources.pool is not requests.pool:
            raise ValueError("Source lifecycle and canonical requests must share a database")
        self.sources, self.requests = sources, requests
        # Correction capture and target publication are composed explicitly in one
        # transaction; do not independently publish the generated correction twice.
        self.history = ContextRepository(requests)

    @staticmethod
    def _revision(value):
        if type(value) is not int or value < 1:
            raise ValueError("An exact positive source revision is required")

    @staticmethod
    def _metadata(source):
        return {"source_id": str(source["id"]), "revision": source["current_revision"],
                "generation": source["generation"], "state": source["status"]}

    async def inspect(self, owner_id, area_id, source_id):
        recorded = await SourceDeletionService(self.sources).inspect(owner_id, area_id, source_id)
        return {
            "source_id": recorded["source_id"], "revision": recorded["revision"],
            "generation": recorded["generation"], "state": recorded["read_barrier"],
            "observed_at": recorded["observed_at"], "local_originals": recorded["local_originals"],
            "provider_cleanup": recorded["provider"], "provider_contacted": False,
            "full_erasure_verified": False,
        }

    async def _admit(self, connection, owner_id, area_id, source_id, revision, request, access_check,
                     *, allow_archived=False):
        await self.requests.lock_owner(connection, owner_id)
        await self.requests._voice_source(connection, owner_id, request)
        if access_check is not None:
            access_check(owner_id)
        await self.sources._area(connection, owner_id, area_id, allow_archived=allow_archived)
        # The source-owner lock serializes repository writes. Publication itself
        # takes original-payload locks before the source row and rechecks revision.
        source = await connection.fetchrow("""select * from jarvis_knowledge_sources
            where owner_id=$1 and area_id=$2 and id=$3""", owner_id, area_id, source_id)
        if source is None:
            raise SourceError("knowledge_source_not_found", "No source exists in the authorized area.", 404)
        if source["current_revision"] != revision:
            raise SourceError("knowledge_source_revision_conflict",
                              "Read the actual current source revision; do not guess a revision after interruption.")
        if source["status"] == "deleted":
            raise SourceError("knowledge_source_deleted", "The original source has been deleted.")
        return source

    async def correct(self, owner_id, area_id, source_id, *, expected_revision, replacement, request,
                      access_check=None):
        self._revision(expected_revision)
        if not isinstance(replacement, str) or not replacement.strip() or "\x00" in replacement or len(replacement) > 8000:
            raise ValueError("A nonblank correction of at most 8000 characters is required")
        async with self.sources.transaction(owner_id) as connection:
            return await self.correct_in_transaction(connection, owner_id, area_id, source_id,
                expected_revision=expected_revision, replacement=replacement, request=request,
                access_check=access_check)

    async def correct_in_transaction(self, connection, owner_id, area_id, source_id, *,
                                     expected_revision, replacement, request, access_check=None):
        """Compose an authorized correction with its local receipt in one source transaction."""
        if not connection.is_in_transaction():
            raise RuntimeError("Source correction requires an existing source transaction")
        self._revision(expected_revision)
        if not isinstance(replacement, str) or not replacement.strip() or "\x00" in replacement or len(replacement) > 8000:
            raise ValueError("A nonblank correction of at most 8000 characters is required")
        source = await self._admit(connection, owner_id, area_id, source_id, expected_revision, request, access_check)
        if source["status"] == "forgotten":
            raise SourceError("knowledge_source_forgotten", "A correction cannot silently unforget this source.")
        note_id = note_identity(source["namespace"], source["source_key"])
        if note_id is not None:
            raise SourceError("note_requires_original_edit",
                f"Use update_note with note_id={note_id} to edit this original, not an independent memory correction.")
        file_id = file_identity(source["namespace"], source["source_key"])
        if file_id is not None:
            raise SourceError("file_requires_new_original",
                f"File {file_id} is an immutable original. Create an explicitly revised file instead of changing its memory independently.")
        original = await self.sources._read(connection, owner_id, area_id, source_id)
        replacement_content = {"text": replacement}
        from app.knowledge.measurements import NAMESPACE as MEASUREMENTS, measurement_correction
        if source["namespace"] == MEASUREMENTS:
            replacement_content = measurement_correction(replacement)
        authority = await self.history._record(connection, owner_id, request.event_id)
        identity = digest({"source_id": str(source_id), "source_revision": expected_revision,
                           "request": request.parameters()})
        correction = await self.history.append_in_transaction(connection, owner_id, ContextIngress(
            session_id=request.session_id, source_namespace="knowledge:source-correction",
            source_event_id=identity, receipt_id=identity, kind="correction", trust="derived",
            source_time=authority.source_time, expected_revision=0,
            source_payload_ids=tuple(dict.fromkeys((original.payload_id, authority.payload_id))),
            payload={**replacement_content, "origin": "user_requested_correction",
                     "replaces": {"source_id": str(source_id), "revision": original.revision,
                                  "document_id": original.document_id},
                     "request": {"event_id": str(authority.id), "revision": authority.revision},
                     "meaning": "Model-formulated correction of a source, not independently observed truth."},
        ))
        revised = await self.sources.publish_in_transaction(connection, owner_id, SourcePublication(
            area_id=area_id, namespace=source["namespace"], source_key=source["source_key"],
            payload_id=correction.payload_id, source_type=correction.kind, trust=correction.trust,
            source_time=correction.source_time, session_id=correction.session_id,
            time_basis="Correction request source_time; unknown stays unknown. Prior versions retain their original clocks.",
            expected_revision=expected_revision,
        ))
        await connection.execute("""insert into jarvis_knowledge_context_sources
            (owner_id,source_id,source_revision,event_id,event_revision) values($1,$2,$3,$4,$5)""",
            owner_id, source_id, revised.revision, correction.id, correction.revision)
        if access_check is not None:
            access_check(owner_id)
        return {"source_id": str(source_id), "revision": revised.revision,
                "generation": revised.generation, "state": "active", "document_id": revised.document_id,
                "correction_event_id": str(correction.id), "prior_revision": expected_revision,
                "source_time": revised.source_time.isoformat() if revised.source_time else None,
                "trust": revised.trust.value, "ingestion": revised.ingestion_status,
                "originals_deleted": False, "provider_contacted": False}

    async def retry_cleanup(self, owner_id, area_id, source_id, *, expected_generation, request,
                            limit=32, access_check=None):
        async with self.sources.transaction(owner_id) as connection:
            await self.requests.lock_owner(connection, owner_id)
            await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            # Exact source scope remains usable after archival or deletion.
            state = await SourceDeletionService(self.sources).retry_failed_cleanup_in_transaction(
                connection, owner_id, area_id, source_id, expected_generation=expected_generation, limit=limit)
            if access_check is not None:
                access_check(owner_id)
            return state

    async def delete_group(self, owner_id, targets, *, request, access_check=None):
        """One finalized user request authorizes exactly these source revisions."""
        async with self.sources.transaction(owner_id) as connection:
            await self.requests.lock_owner(connection, owner_id)
            await self.requests._voice_source(connection, owner_id, request)
            if access_check is not None:
                access_check(owner_id)
            result = await self.sources.delete_sources_in_transaction(connection, owner_id, targets)
            if access_check is not None:
                access_check(owner_id)
            return result

    async def forget(self, owner_id, area_id, source_id, *, expected_revision, request, access_check=None):
        self._revision(expected_revision)
        async with self.sources.transaction(owner_id) as connection:
            source = await self._admit(connection, owner_id, area_id, source_id, expected_revision, request,
                                       access_check, allow_archived=True)
            changed = source["status"] != "forgotten"
            if changed:
                source = await connection.fetchrow("""update jarvis_knowledge_sources
                    set status='forgotten',generation=generation+1
                    where owner_id=$1 and id=$2 and current_revision=$3 returning *""",
                    owner_id, source_id, expected_revision)
                if source is None:
                    raise SourceError("knowledge_source_revision_conflict", "The source changed before suppression.")
            await self.sources._retire_documents(
                connection, owner_id, source_id, source["generation"], reschedule_existing=False)
            if access_check is not None:
                access_check(owner_id)
            return self._metadata(source) | {
                "changed": changed, "originals_deleted": False, "provider_cleanup": "inspect_recorded_status",
                "provider_contacted": False, "automatic_reactivation": False,
            }
