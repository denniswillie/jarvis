"""Stable storage effects and atomic file adoption, independent of an agent loop."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import Field, JsonValue, field_validator

from app.evidence.models import FrozenRecord
from app.files.models import JarvisFile, JarvisFileSource
from app.files.repository import JarvisFileRepository
from app.files.request_authority import FileRequestAuthority, REQUEST_AUTHORITY_KEY
from app.files.previews import build_preview, store_preview
from app.files.transfer import DOCUMENT_POLICY, FileManifest, FileTransferPolicy, MP4Inspection
from app.knowledge.models import MemorySourceReference, SourceError
from app.knowledge.read_provenance import SourceReadProvenance
from app.tasks.models import digest

MAX_FILE_BYTES = 8 * 1024 * 1024


def file_id_for_operation(owner_id: str, operation_key: str) -> UUID:
    """The shared deterministic identity, including before bytes exist."""
    if not isinstance(operation_key, str) or not 1 <= len(operation_key) <= 256 or "\x00" in operation_key:
        raise ValueError("A stable bounded operation identity is required")
    return uuid5(NAMESPACE_URL, "jarvis:file:" + digest({"owner": owner_id, "operation": operation_key}))


class FilePublicationSpec(FrozenRecord):
    session_id: UUID | None
    filename: str = Field(min_length=1, max_length=240)
    content_type: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    source: JarvisFileSource
    source_visual_frame_id: UUID | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("filename", "content_type", "description")
    @classmethod
    def valid_text(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("File metadata must be nonblank and contain no NUL")
        return value

    @field_validator("metadata")
    @classmethod
    def bounded_metadata(cls, value):
        if len(json.dumps(value, ensure_ascii=True, allow_nan=False)) > 16000:
            raise ValueError("File metadata exceeds its bound")
        return value


@dataclass(frozen=True)
class FilePublicationResult:
    operation_id: UUID
    file_id: UUID
    phase: str
    file: JarvisFile | None
    upload_settled: bool
    delete_requested: bool


class _ConnectionPool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


class FilePublicationService:
    """Trusted scope plus an optional in-transaction source-adoption callback.

    The callback must not perform provider I/O. Catalog activation waits for the
    complete source-backed producer adapter, not merely this storage foundation.
    """
    def __init__(self, sources, storage, *, owner_id, area_id, access_check=None, adopt=None,
                 selections=None, session_id=None, transfer_policy=None, inspect_file=None, adopt_file=None,
                 request_authority=None):
        self.sources, self.storage = sources, storage
        if request_authority is not None and (not isinstance(request_authority, FileRequestAuthority)
                or request_authority.tasks.pool is not sources.pool):
            raise ValueError("File request authority must share the source pool")
        self.request_authority = request_authority
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check
        self.adopt = adopt
        self.transfer_policy = transfer_policy or FileTransferPolicy()
        if not isinstance(self.transfer_policy, FileTransferPolicy):
            raise ValueError("File policy must be selected by trusted composition")
        self.inspect_file, self.adopt_file = inspect_file, adopt_file
        if self.transfer_policy.is_media and (inspect_file is None or adopt_file is None):
            raise ValueError("Media publication requires trusted inspection and source adoption")
        self.provenance = SourceReadProvenance(sources)
        self.selections, self.session_id = selections, session_id
        if selections is not None and (selections.sources.pool is not sources.pool
                or selections.default_area_id != area_id):
            raise ValueError("File selections must match the publication database and default area")
        if session_id is not None and not isinstance(session_id, UUID):
            raise ValueError("A trusted file session must be an exact UUID")

    @property
    def area_filter(self):
        # Session defaults choose new operations. Existing owned operations and
        # background recovery retain their persisted area, including tombstones.
        return None if self.selections is not None else self.area_id

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("file_owner_mismatch", "File publication is outside this owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    async def _lock_request_owner(self, connection, owner_id):
        if self.request_authority is not None:
            await self.request_authority.tasks.lock_owner(connection, owner_id)

    async def _request_dependencies(self, connection, owner_id, spec):
        if self.request_authority is None:
            if REQUEST_AUTHORITY_KEY in spec.metadata:
                raise SourceError("file_request_authority_required",
                                  "This publication must recover through its request-authorized service.", 403)
            return ()
        payload = await self.request_authority.validate_locked(connection, owner_id=owner_id, spec=spec)
        return (payload,)

    async def _row(self, connection, owner_id, operation_id):
        if self.selections is not None:
            await self.selections._session(connection, owner_id, self.session_id)
        row = await connection.fetchrow(
            """select f.*,p.availability,p.content as specification
                from jarvis_file_publications f join jarvis_context_payloads p
                on p.owner_id=f.owner_id and p.id=f.metadata_payload_id
                where f.owner_id=$1 and ($2::uuid is null or f.area_id=$2) and f.id=$3 for update of f""",
            owner_id, self.area_filter, operation_id)
        if row is None:
            raise SourceError("file_publication_not_found", "No file publication exists in the authorized area.", 404)
        return row

    async def _active(self, connection, row):
        await self.sources._area(connection, self.owner_id, row["area_id"])
        if row["delete_requested"] or row["availability"] != "available":
            raise SourceError("file_publication_revoked", "File publication evidence is no longer available.")
        if await connection.fetchval(
                "select lock_jarvis_copy_source($1,$2)", self.owner_id, row["metadata_payload_id"]) != "available":
            raise SourceError("file_publication_revoked", "File publication evidence changed.")
        self._authorize(self.owner_id)

    @staticmethod
    def _specification(row):
        raw = row["specification"]
        return json.loads(raw) if isinstance(raw, str) else raw

    async def _result(self, connection, row):
        file = None
        if row["phase"] == "available":
            await self._active(connection, row)
            file = await JarvisFileRepository(_ConnectionPool(connection)).get_file(
                owner_id=self.owner_id, file_id=row["file_id"])
            if file is None:
                raise SourceError("file_adoption_missing", "The adopted file record is missing.")
        return FilePublicationResult(row["id"], row["file_id"], row["phase"], file,
                                     row["upload_settled"], row["delete_requested"])

    def _selected_policy(self, selected, *, file_backed):
        baseline = self.transfer_policy if file_backed else FileTransferPolicy()
        if selected is None:
            return baseline
        if (not isinstance(selected, FileTransferPolicy) or selected.policy_id != baseline.policy_id
                or selected.max_bytes > baseline.max_bytes):
            raise ValueError("A per-operation policy may only narrow its trusted concrete transfer policy")
        return selected

    async def prepare(self, *, owner_id, operation_key, spec, content, source_payload_ids=(), memory_sources=(),
                      transfer_policy=None):
        self._authorize(owner_id)
        if not isinstance(spec, FilePublicationSpec) or type(content) is not bytes or len(content) > MAX_FILE_BYTES:
            raise ValueError("A validated specification and at most 8 MiB of exact bytes are required")
        return await self._prepare_manifest(owner_id=owner_id, operation_key=operation_key, spec=spec,
            manifest=FileManifest(size_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest()),
            policy=self._selected_policy(transfer_policy, file_backed=False),
            source_payload_ids=source_payload_ids, memory_sources=memory_sources)

    async def prepare_manifest(self, *, owner_id, operation_key, spec, manifest,
                               source_payload_ids=(), memory_sources=(), transfer_policy=None):
        """Reserve exact measured bytes; this is not proof of upload or source adoption."""
        return await self._prepare_manifest(owner_id=owner_id, operation_key=operation_key, spec=spec,
            manifest=manifest, policy=self._selected_policy(transfer_policy, file_backed=True),
            source_payload_ids=source_payload_ids, memory_sources=memory_sources)

    async def _prepare_manifest(self, *, owner_id, operation_key, spec, manifest, policy,
                                source_payload_ids=(), memory_sources=()):
        self._authorize(owner_id)
        if not isinstance(spec, FilePublicationSpec):
            raise ValueError("A validated file specification is required")
        policy.validate(manifest, spec)
        if not isinstance(operation_key, str) or not 1 <= len(operation_key) <= 256 or "\x00" in operation_key:
            raise ValueError("A stable bounded operation identity is required")
        parents = tuple(dict.fromkeys(source_payload_ids))
        if len(parents) > 32 or any(not isinstance(parent, UUID) for parent in parents):
            raise ValueError("Invalid file source dependencies")
        sha256, size_bytes = manifest.content_sha256, manifest.size_bytes
        memory_sources = tuple(memory_sources)
        file_id = file_id_for_operation(owner_id, operation_key)
        path = self.storage.build_file_path(owner_id=owner_id, session_id=spec.session_id,
                                           file_id=file_id, filename=spec.filename)
        # A corrected recording is the same single gallery item, but must not
        # overwrite its temporary phone upload before durable verified adoption.
        binding = spec.metadata.get("recording_original", {})
        finalization = binding.get("finalization_manifest", {}) if isinstance(binding, dict) else {}
        if spec.source == "upload" and isinstance(finalization, dict) and "timing_correction" in finalization:
            path += "/synced"
        async with self.sources.transaction(owner_id) as connection:
            await self.provenance.tasks.lock_owner(connection, owner_id)
            request_parents = await self._request_dependencies(connection, owner_id, spec)
            parents = tuple(dict.fromkeys((*parents, *request_parents)))
            if len(parents) > 32:
                raise ValueError("Invalid file source dependencies")
            if self.selections is not None:
                if spec.session_id != self.session_id:
                    raise SourceError("knowledge_session_unavailable", "The file belongs to another conversation.", 403)
                await self.selections._session(connection, owner_id, self.session_id)
            prior = await connection.fetchrow(
                "select id,area_id,parameters_digest from jarvis_file_publications where owner_id=$1 and operation_key=$2",
                owner_id, operation_key)
            area_id = self.area_id
            if self.selections is not None:
                area_id = (prior["area_id"] if prior is not None else UUID((await self.selections.read_in_transaction(
                    connection, owner_id, self.session_id))["area_id"]))
            policy_fields = ({"transfer_policy": policy.policy_id, "transfer_limit_bytes": policy.max_bytes}
                             if policy.policy_id != DOCUMENT_POLICY or policy.max_bytes != MAX_FILE_BYTES else {})
            parameters = digest({**policy_fields, "spec": spec.model_dump(mode="json"), "sha256": sha256, "size": size_bytes,
                                 "parents": sorted(map(str, parents)),
                                 "memory_sources": sorted((MemorySourceReference.from_metadata(item).model_dump(mode="json")
                                     for item in memory_sources), key=lambda item: (item["source_id"], item["revision"])),
                                 "area_id": str(area_id)})
            if prior is not None:
                if prior["parameters_digest"] != parameters:
                    raise SourceError("file_publication_conflict", "This operation already names different bytes or evidence.")
                return await self._result(connection, await self._row(connection, owner_id, prior["id"]))
            await self.sources._area(connection, owner_id, area_id)
            if spec.session_id is not None and not await connection.fetchval(
                    "select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2)",
                    owner_id, spec.session_id):
                raise SourceError("file_session_unavailable", "The file session is outside its owner scope.", 404)
            if spec.source_visual_frame_id is not None and not await connection.fetchval(
                    """select exists(select 1 from visual_frames where owner_id=$1 and id=$2
                        and session_id is not distinct from $3)""", owner_id, spec.source_visual_frame_id, spec.session_id):
                raise SourceError("file_frame_unavailable", "The frame is outside its owner/session scope.", 404)
            async def materialize(conn, originals):
                dependencies = tuple(dict.fromkeys((*parents, *originals)))
                if dependencies and not await conn.fetchval(
                        "select lock_jarvis_copy_sources($1,$2::uuid[])", owner_id, list(dependencies)):
                    raise SourceError("file_sources_unavailable", "File source evidence is unavailable.")
                payload = await conn.fetchval(
                    """insert into jarvis_context_payloads(owner_id,availability,content)
                        values($1,'available',$2::jsonb) returning id""", owner_id,
                    json.dumps({**policy_fields, "file_id": str(file_id), "spec": spec.model_dump(mode="json"),
                                "content_sha256": sha256, "size_bytes": size_bytes,
                                "meaning": "Publication manifest, not proof of completed storage or delivery."}, allow_nan=False))
                if dependencies:
                    await conn.executemany("""insert into jarvis_context_payload_dependencies
                        (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3)""",
                        [(owner_id, payload, parent) for parent in dependencies])
                return payload, payload
            payload = await self.provenance.bind_in_transaction(
                connection, owner_id, area_id, list(memory_sources), materialize)
            arguments = (owner_id, area_id, operation_key, parameters, file_id, payload, sha256, size_bytes,
                         self.storage.bucket, path)
            # Ordinary documents retain their original parameter digest and pre-migration SQL.
            columns, values = "", ""
            if policy_fields:
                columns, values = ",transfer_policy,transfer_limit_bytes", ",$11,$12"
                arguments += (policy.policy_id, policy.max_bytes)
            identity = await connection.fetchval(
                "insert into jarvis_file_publications(owner_id,area_id,operation_key,parameters_digest,"
                "file_id,metadata_payload_id,content_sha256,size_bytes,storage_bucket,storage_path" + columns + ")"
                " values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10" + values + ") returning id", *arguments)
            self._authorize(owner_id)
            return await self._result(connection, await self._row(connection, owner_id, identity))

    async def publish(self, *, owner_id, operation_key, spec, content, source_payload_ids=(), memory_sources=(),
                      transfer_policy=None):
        prepared = await self.prepare(owner_id=owner_id, operation_key=operation_key, spec=spec, content=content,
            source_payload_ids=source_payload_ids, memory_sources=memory_sources, transfer_policy=transfer_policy)
        return await self._publish_prepared(owner_id, prepared, spec, content=content)

    async def publish_file(self, *, owner_id, operation_key, spec, local_file,
                           source_payload_ids=(), memory_sources=(), transfer_policy=None):
        from app.files.local_file import VerifiedLocalFile
        self._authorize(owner_id)
        if self.request_authority is not None:
            raise SourceError("file_request_mode_unsupported", "Request-scoped file-backed publication is not enabled.", 403)
        if not isinstance(local_file, VerifiedLocalFile):
            raise ValueError("An owned verified local file is required")
        # Inspect before crossing the upload boundary, then re-inspect stored bytes before adoption.
        manifest = FileManifest.from_file(local_file)
        selected = self._selected_policy(transfer_policy, file_backed=True)
        selected.validate(manifest, spec)
        if selected.is_media:
            await self._inspect(local_file, selected)
        prepared = await self.prepare_manifest(owner_id=owner_id, operation_key=operation_key, spec=spec,
            manifest=manifest, source_payload_ids=source_payload_ids, memory_sources=memory_sources,
            transfer_policy=selected)
        return await self._publish_prepared(owner_id, prepared, spec, local_file=local_file)

    async def adopt_recording_upload(self, *, owner_id, recording_id, spec, bucket, path,
                                     local_file, inspection, source_payload_ids):
        """Adopt one authorized recording target already uploaded by the phone.

        This trusted producer entrypoint does not relax ordinary reconcile's
        unsubmitted-object guard. It accepts a measured storage snapshot and
        freezes only the server-derived recording target; reconcile independently
        reads/inspects storage again and performs the existing atomic adoption.
        No byte upload or overwrite is performed here.
        """
        from app.files.local_file import VerifiedLocalFile
        from app.files.transfer import MP4_CAPTURE_POLICY
        self._authorize(owner_id)
        if (not isinstance(recording_id, UUID) or not isinstance(spec, FilePublicationSpec)
                or not isinstance(local_file, VerifiedLocalFile) or not isinstance(inspection, MP4Inspection)
                or self.transfer_policy.policy_id != MP4_CAPTURE_POLICY or self.adopt_file is None
                or spec.source != "upload" or not isinstance(spec.session_id, UUID)):
            raise ValueError("An authorized inspected recording original is required")
        operation_key = "recording:" + str(recording_id)
        file_id = file_id_for_operation(owner_id, operation_key)
        expected_path = self.storage.build_file_path(owner_id=owner_id, session_id=spec.session_id,
            file_id=file_id, filename=spec.filename)
        manifest = FileManifest.from_file(local_file)
        self.transfer_policy.validate(manifest, spec)
        inspection.validate_file(manifest, self.transfer_policy)
        binding = spec.metadata.get("recording_original", {})
        frozen = binding.get("finalization_manifest", {})
        if (bucket != self.storage.bucket or path != expected_path
                or binding.get("recording_id") != str(recording_id)
                or frozen.get("size_bytes") != manifest.size_bytes
                or frozen.get("content_sha256") != manifest.content_sha256
                or frozen.get("media_inspection") != inspection.model_dump(mode="json")
                or spec.metadata.get("media_inspection") != inspection.model_dump(mode="json")
                or len(source_payload_ids) != 1):
            raise SourceError("recording_adoption_mismatch", "The external object differs from its recording target.", 403)
        # Detect a closed or changed snapshot before committing external evidence.
        async for _ in local_file.iter_chunks():
            pass
        prepared = await self.prepare_manifest(owner_id=owner_id, operation_key=operation_key, spec=spec,
            manifest=manifest, source_payload_ids=source_payload_ids)
        async with self.sources.transaction(owner_id) as connection:
            await self.provenance.tasks.lock_owner(connection, owner_id)
            row = await self._row(connection, owner_id, prepared.operation_id)
            await self._active(connection, row)
            if (row["file_id"] != file_id or row["storage_bucket"] != bucket or row["storage_path"] != path):
                raise SourceError("recording_adoption_mismatch", "The publication target changed.", 403)
            if row["phase"] == "prepared":
                if row["upload_token"] is not None:
                    raise SourceError("recording_adoption_mismatch", "The prepared object already has a submitted upload.")
                await connection.execute("""update jarvis_file_publications
                    set upload_token=$3,upload_settled=true,phase='uploaded' where owner_id=$1 and id=$2""",
                    owner_id, row["id"], file_id)
        return await self.reconcile(owner_id=owner_id, operation_id=prepared.operation_id)

    async def _publish_prepared(self, owner_id, prepared, spec, *, content=None, local_file=None):
        if prepared.phase != "prepared":
            return await self.reconcile(owner_id=owner_id, operation_id=prepared.operation_id)
        async with self.sources.transaction(owner_id) as connection:
            await self._lock_request_owner(connection, owner_id)
            row = await self._row(connection, owner_id, prepared.operation_id)
            submit = row["phase"] == "prepared"
            if submit:
                spec = FilePublicationSpec.model_validate(self._specification(row)["spec"])
                await self._request_dependencies(connection, owner_id, spec)
                await self._active(connection, row)
                token = uuid4()
                await connection.execute(
                    "update jarvis_file_publications set phase='uploading',upload_token=$3 where owner_id=$1 and id=$2",
                    owner_id, row["id"], token)
        if submit:
            try:
                if local_file is None:
                    receipt = await self.storage.upload_bytes(path=row["storage_path"], content=content,
                                                               content_type=spec.content_type)
                else:
                    receipt = await self.storage.upload_file(path=row["storage_path"], local_file=local_file,
                                                              content_type=spec.content_type)
                if (receipt.bucket, receipt.path, receipt.size_bytes) != (
                        row["storage_bucket"], row["storage_path"], row["size_bytes"]):
                    raise SourceError("file_upload_receipt_mismatch", "The receipt names a different object.")
            except BaseException:
                await asyncio.shield(self._upload_outcome(owner_id, row["id"], token, settled=False))
                raise
            await self._upload_outcome(owner_id, row["id"], token, settled=True)
        return await self.reconcile(owner_id=owner_id, operation_id=prepared.operation_id)

    async def _upload_outcome(self, owner_id, operation_id, token, *, settled):
        async with self.sources.transaction(owner_id) as connection:
            row = await self._row(connection, owner_id, operation_id)
            if row["upload_token"] != token:
                raise SourceError("file_upload_token_mismatch", "Receipt does not match the submitted upload.")
            if row["phase"] == "deleted" or row["upload_settled"]:
                return
            phase = ("delete_pending" if row["delete_requested"] else "quarantined"
                     if row["phase"] == "quarantined" else "uploaded" if settled else "outcome_unknown")
            await connection.execute(
                "update jarvis_file_publications set upload_settled=$3,phase=$4 where owner_id=$1 and id=$2",
                owner_id, operation_id, settled, phase)

    async def _inspect(self, local_file, policy):
        if self.inspect_file is None or self.adopt_file is None:
            raise SourceError("file_media_adapter_unavailable", "Trusted media inspection/adoption is not configured.", 503)
        inspection = await self.inspect_file(local_file, policy)
        if not isinstance(inspection, MP4Inspection):
            raise ValueError("The inspector must return a measured MP4 inspection")
        inspection.validate_file(FileManifest.from_file(local_file), policy)
        # An inspector cannot replace the measured snapshot while checking it.
        async for _ in local_file.iter_chunks():
            pass
        return inspection

    @asynccontextmanager
    async def _stored_content(self, row, *, inspect_media=False, verify_manifest=True, max_bytes=None):
        policy = FileTransferPolicy.from_row(row)
        bound = row["size_bytes"] if max_bytes is None else max_bytes
        content = None
        try:
            if policy.is_media:
                if inspect_media and (self.inspect_file is None or self.adopt_file is None):
                    raise SourceError("file_media_adapter_unavailable", "Trusted media adoption is not configured.", 503)
                content = await self.storage.read_object_file(
                    bucket=row["storage_bucket"], path=row["storage_path"], max_bytes=bound)
                manifest = FileManifest.from_file(content) if content is not None else None
            else:
                content = await self.storage.read_object(
                    bucket=row["storage_bucket"], path=row["storage_path"], max_bytes=bound)
                manifest = (FileManifest(size_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest())
                            if content is not None else None)
            if verify_manifest and manifest is not None and (manifest.size_bytes != row["size_bytes"]
                                        or manifest.content_sha256 != row["content_sha256"]):
                raise SourceError("file_content_mismatch", "Stored bytes do not match the publication manifest.")
            inspection = (await self._inspect(content, policy)
                          if content is not None and policy.is_media and inspect_media else None)
            yield content, inspection
        finally:
            if content is not None and policy.is_media:
                content.close()

    async def _object_exists(self, row, *, verify_manifest=True):
        async with self._stored_content(row, verify_manifest=verify_manifest,
                                        max_bytes=FileTransferPolicy.from_row(row).max_bytes) as (content, _):
            return content is not None

    async def reconcile(self, *, owner_id, operation_id):
        self._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            await self._lock_request_owner(connection, owner_id)
            row = await self._row(connection, owner_id, operation_id)
            if row["phase"] in {"deleted", "quarantined"}:
                return await self._result(connection, row)
            cleanup = row["delete_requested"]
            if not cleanup:
                if row["phase"] != "available":
                    await self._request_dependencies(connection, owner_id,
                        FilePublicationSpec.model_validate(self._specification(row)["spec"]))
                await self._active(connection, row)
                if row["phase"] == "prepared":
                    return await self._result(connection, row)
        if cleanup:
            return await self._cleanup(owner_id, operation_id)
        async with self._stored_content(row, inspect_media=row["phase"] != "available") as (content, inspection):
            if content is None:
                async with self.sources.transaction(owner_id) as connection:
                    current = await self._row(connection, owner_id, operation_id)
                    if not current["delete_requested"] and current["phase"] != "deleted":
                        await connection.execute(
                            "update jarvis_file_publications set phase='outcome_unknown' where owner_id=$1 and id=$2",
                            owner_id, operation_id)
                    return await self._result(connection, await self._row(connection, owner_id, operation_id))
            preview = (await build_preview(content, self._specification(row)["spec"]["content_type"])
                       if row["phase"] != "available" else None)
            async with self.sources.transaction(owner_id) as connection:
                await self._lock_request_owner(connection, owner_id)
                current = await self._row(connection, owner_id, operation_id)
                if current["phase"] in {"deleted", "quarantined"}:
                    return await self._result(connection, current)
                if current["upload_token"] is None:
                    raise SourceError("file_upload_not_submitted", "Unsubmitted work cannot adopt an external object.")
                if current["phase"] != "available":
                    if not current["delete_requested"]:
                        await self._request_dependencies(connection, owner_id,
                            FilePublicationSpec.model_validate(self._specification(current)["spec"]))
                    await connection.execute(
                        "update jarvis_file_publications set upload_settled=true,phase=$3 where owner_id=$1 and id=$2",
                        owner_id, operation_id, "delete_pending" if current["delete_requested"] else "uploaded")
                if not current["delete_requested"]:
                    await self._active(connection, current)
                    spec = FilePublicationSpec.model_validate(self._specification(current)["spec"])
                    repository = JarvisFileRepository(_ConnectionPool(connection))
                    file = await repository.get_file(owner_id=owner_id, file_id=current["file_id"])
                    if current["phase"] != "available":
                        if file is not None:
                            raise SourceError("file_adoption_conflict", "A file already occupies this unadopted identity.")
                        file = await repository.create_file(
                            owner_id=owner_id, file_id=current["file_id"], storage_bucket=current["storage_bucket"],
                            storage_path=current["storage_path"], size_bytes=current["size_bytes"], **spec.model_dump())
                        if FileTransferPolicy.from_row(current).is_media:
                            await self.adopt_file(connection, file, spec, content, inspection,
                                                  current["metadata_payload_id"])
                        elif self.adopt is not None:
                            await self.adopt(connection, file, spec, content, current["metadata_payload_id"])
                        await connection.execute(
                            "update jarvis_file_publications set phase='available' where owner_id=$1 and id=$2",
                            owner_id, operation_id)
                        await store_preview(connection, owner_id, current["file_id"], current["content_sha256"], preview)
                    self._authorize(owner_id)
                    return await self._result(connection, await self._row(connection, owner_id, operation_id))
        return await self._cleanup(owner_id, operation_id)

    async def delete(self, *, owner_id, operation_id):
        self._authorize(owner_id)
        if self.request_authority is not None:
            raise SourceError("file_request_mode_unsupported", "Request-scoped file deletion is not enabled.", 403)
        async with self.sources.transaction(owner_id) as connection:
            row = await self._row(connection, owner_id, operation_id)
            if row["phase"] != "deleted":
                await connection.execute(
                    "update jarvis_file_publications set delete_requested=true,phase='delete_pending' where owner_id=$1 and id=$2",
                    owner_id, operation_id)
                await connection.execute(
                    "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                    owner_id, row["metadata_payload_id"])
        return await self._cleanup(owner_id, operation_id)

    async def _cleanup(self, owner_id, operation_id):
        async with self.sources.transaction(owner_id) as connection:
            row = await self._row(connection, owner_id, operation_id)
            if row["phase"] == "deleted":
                return await self._result(connection, row)
            if not row["delete_requested"]:
                raise SourceError("file_delete_not_requested", "Cleanup requires a recorded deletion.")
        if row["upload_token"] is not None:
            exists = await self._object_exists(row, verify_manifest=not row["upload_settled"])
            if not row["upload_settled"]:
                # Absence now does not prove that an unknown upload cannot arrive later.
                if not exists:
                    async with self.sources.transaction(owner_id) as connection:
                        return await self._result(connection, await self._row(connection, owner_id, operation_id))
                await self._upload_outcome(owner_id, operation_id, row["upload_token"], settled=True)
            if exists:
                await self.storage.delete_object(bucket=row["storage_bucket"], path=row["storage_path"])
            if await self._object_exists(row, verify_manifest=False):
                async with self.sources.transaction(owner_id) as connection:
                    return await self._result(connection, await self._row(connection, owner_id, operation_id))
        async with self.sources.transaction(owner_id) as connection:
            current = await self._row(connection, owner_id, operation_id)
            if current["upload_token"] is not None and not current["upload_settled"]:
                return await self._result(connection, current)
            await connection.execute(
                """update jarvis_file_publications set phase='deleted',storage_bucket=null,storage_path=null
                    where owner_id=$1 and id=$2""", owner_id, operation_id)
            return await self._result(connection, await self._row(connection, owner_id, operation_id))

    async def reconcile_pending(self, *, owner_id, limit=16):
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("File reconciliation limit must be between 1 and 32")
        async with self.sources.pool.acquire() as connection:
            if self.selections is not None:
                await self.selections._session(connection, owner_id, self.session_id)
            identities = await connection.fetch(
                """select id from jarvis_file_publications where owner_id=$1 and ($2::uuid is null or area_id=$2)
                    and phase not in ('prepared','available','quarantined','deleted') order by updated_at,id limit $3""",
                owner_id, self.area_filter, limit)
        outcomes = []
        for row in identities:
            try:
                result = await self.reconcile(owner_id=owner_id, operation_id=row["id"])
                outcomes.append({"operation_id": str(row["id"]), "phase": result.phase})
            except Exception as error:
                outcomes.append({"operation_id": str(row["id"]), "phase": "unresolved",
                                 "error_type": type(error).__name__})
        if identities:
            async with self.sources.pool.acquire() as connection:
                await connection.execute(
                    """update jarvis_file_publications set updated_at=clock_timestamp()
                        where owner_id=$1 and ($2::uuid is null or area_id=$2) and id=any($3::uuid[])
                        and phase not in ('available','deleted')""",
                    owner_id, self.area_filter, [row["id"] for row in identities])
        return outcomes

    async def _read_available_row(self, *, owner_id, file_id):
        self._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            identity = await connection.fetchval(
                "select id from jarvis_file_publications where owner_id=$1 and ($2::uuid is null or area_id=$2) and file_id=$3",
                owner_id, self.area_filter, file_id)
            if identity is None:
                raise SourceError("file_publication_not_found", "No file exists in the authorized area.", 404)
            row = await self._row(connection, owner_id, identity)
            await self._active(connection, row)
            if row["phase"] != "available":
                raise SourceError("file_not_available", "The file has not been durably adopted.")
        FileTransferPolicy.from_row(row)
        return identity, row

    async def _revalidate_read(self, owner_id, identity):
        async with self.sources.transaction(owner_id) as connection:
            current = await self._row(connection, owner_id, identity)
            await self._active(connection, current)
            if current["phase"] != "available":
                raise SourceError("file_not_available", "The file changed while being read.")

    async def read_content(self, *, owner_id, file_id):
        identity, row = await self._read_available_row(owner_id=owner_id, file_id=file_id)
        if FileTransferPolicy.from_row(row).is_media:
            raise SourceError("file_requires_stream", "Media originals require the file-backed reader.")
        async with self._stored_content(row) as (content, _):
            if content is None:
                raise SourceError("file_content_unavailable", "Verified retained bytes are unavailable.")
            await self._revalidate_read(owner_id, identity)
            return content

    async def read_manifest(self, *, owner_id, file_id):
        _, row = await self._read_available_row(owner_id=owner_id, file_id=file_id)
        return (FileManifest(size_bytes=row["size_bytes"], content_sha256=row["content_sha256"]),
                FileTransferPolicy.from_row(row))

    async def read_file(self, *, owner_id, file_id):
        """Caller owns the verified private file and must close it on every path."""
        identity, row = await self._read_available_row(owner_id=owner_id, file_id=file_id)
        local_file = await self.storage.read_object_file(
            bucket=row["storage_bucket"], path=row["storage_path"], max_bytes=row["size_bytes"])
        try:
            if (local_file is None or local_file.size_bytes != row["size_bytes"]
                    or local_file.content_sha256 != row["content_sha256"]):
                raise SourceError("file_content_unavailable", "Verified retained bytes are unavailable.")
            await self._revalidate_read(owner_id, identity)
            return local_file
        except BaseException:
            if local_file is not None:
                local_file.close()
            raise
