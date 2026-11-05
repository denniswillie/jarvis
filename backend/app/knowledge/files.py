"""Complete text-file originals using the shared storage and source/outbox paths."""
from dataclasses import replace
from hashlib import sha256
import json
from uuid import UUID, uuid4

from app.images import ImageReference, ImageUnavailableError
from app.files.publication import FilePublicationService, FilePublicationSpec
from app.files.request_authority import FileRequestAuthority
from app.files.generated import ArtifactSnapshot
from app.files.local_file import VerifiedLocalFile
from app.files.transfer import MP4Inspection
from app.files.repository import JarvisFileRepository, _file_from_record
from app.files.tools import _file_payload, text_file_parameters
from app.knowledge.models import MemorySourceReference, SourceError, SourcePublication
from app.knowledge.product_files import FILE_NAMESPACE, FILE_ORIGINAL_FORMAT, IMAGE_ORIGINAL_FORMAT, GENERATED_ORIGINAL_FORMAT, file_sources, read_file_original
from app.knowledge.read_provenance import SourceReadProvenance
from app.tasks.models import TaskError, digest, json_object
from app.tools.inputs import success
from app.tools.invocation import current_tool_invocation
from app.tools.provenance import SourceBoundToolResult
from app.visual_memory.validation import jpeg_dimensions, VisualFrameValidationError


class SourceFileSearchResult(list):
    """File results retain their read area even when no files match."""
    def __init__(self, files, area_id):
        super().__init__(files)
        self.knowledge_scope = {"area_id": str(area_id), "origin": "session_selection"}


class SourceFileRepository(JarvisFileRepository):
    """Verified file originals, source-backed reads and explicit retention."""
    list_limit = 8

    def __init__(self, sources, storage, *, owner_id, area_id, access_check=None, current_images=None,
                 task_evidence=None, task_recovery=None, selections=None, session_id=None,
                 media_policy=None, media_inspector=None, request_authority=None):
        super().__init__(sources.pool)
        if request_authority is not None and (not isinstance(request_authority, FileRequestAuthority)
                or request_authority.tasks.pool is not sources.pool
                or task_evidence is not None or task_recovery is not None):
            raise ValueError("Conversational file authority needs the source pool and cannot replace worker grants")
        self.request_authority = request_authority
        self.sources, self.storage = sources, storage
        self.current_images = current_images
        self.task_evidence = task_evidence
        self.selections, self.session_id = selections, session_id
        self.media_policy, self.media_inspector = media_policy, media_inspector
        if (media_policy is None) != (media_inspector is None):
            raise ValueError("Media policy and trusted inspector must be configured together")
        if selections is not None and (selections.sources.pool is not sources.pool
                or selections.default_area_id != area_id or task_evidence is not None):
            raise ValueError("Session files must share the default scope and cannot replace worker grants")
        if session_id is not None and not isinstance(session_id, UUID):
            raise ValueError("A trusted file session must be an exact UUID")
        if task_recovery is not None and (task_evidence is not None or task_recovery.sources is not sources
                or task_recovery.owner_id != owner_id):
            raise ValueError("Task file recovery and active execution are separate matching scopes")
        self.task_recovery = task_recovery
        if task_evidence is not None and (task_evidence.sources is not sources
                or task_evidence.area_id != area_id or task_evidence.context.task.owner_id != owner_id):
            raise ValueError("File task evidence must match its source scope")
        self.owner_id, self.area_id, self.access_check = owner_id, area_id, access_check
        self.legacy_repository = JarvisFileRepository(sources.pool)
        self.provenance = SourceReadProvenance(sources)
        self.publications = FilePublicationService(sources, storage, owner_id=owner_id, area_id=area_id,
            access_check=access_check, adopt=self._adopt, selections=selections, session_id=session_id,
            transfer_policy=media_policy, inspect_file=media_inspector,
            adopt_file=self._adopt_media if media_policy is not None else None, request_authority=request_authority)

    def for_session(self, session_id):
        if self.selections is None:
            return self
        return SourceFileRepository(self.sources, self.storage, owner_id=self.owner_id, area_id=self.area_id,
            access_check=self.access_check, current_images=self.current_images,
            selections=self.selections, session_id=session_id,
            media_policy=self.media_policy, media_inspector=self.media_inspector, request_authority=self.request_authority)

    async def _area_for(self, connection, file_id=None):
        if self.selections is None:
            return self.area_id
        await self.selections._session(connection, self.owner_id, self.session_id)
        if file_id is not None:
            rows = await file_sources(connection, self.owner_id, file_id)
            areas = {row["area_id"] for row in rows}
            publication_area = await connection.fetchval(
                "select area_id from jarvis_file_publications where owner_id=$1 and file_id=$2",
                self.owner_id, file_id)
            if publication_area is not None:
                areas.add(publication_area)
            if len(areas) > 1:
                raise SourceError("file_source_conflict", "The file has competing area mappings; no area was guessed.")
            if areas:
                return areas.pop()
        return UUID((await self.selections.read_in_transaction(connection, self.owner_id, self.session_id))["area_id"])

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("file_owner_mismatch", "The file is outside this owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)
        if current_tool_invocation().task is not None and self.task_evidence is None:
            raise SourceError("file_task_grant_required",
                "Worker file access requires the artifact grant/evidence adapter, not the foreground file path.", 403)

    async def create_file(self, **kwargs):
        raise SourceError("file_requires_publication",
            "Use verified file publication; raw metadata insertion would bypass the original source.")

    async def delete_file(self, **kwargs):
        raise SourceError("file_requires_source_deletion",
            "Use delete_source_file so source revocation and storage cleanup remain linked.")

    async def _mapping(self, connection, file_id, *, required=True, area_id=None):
        area_id = self.area_id if area_id is None else area_id
        rows = await file_sources(connection, self.owner_id, file_id)
        if len([row for row in rows if row["status"] != "deleted"]) > 1:
            raise SourceError("file_shared_original", "This file has competing retained source mappings.")
        selected = [row for row in rows if row["area_id"] == area_id]
        if len(selected) != 1:
            if not rows and not required:
                return None
            raise SourceError("file_source_unavailable",
                "This file has not been source-published into the selected area; no legacy fallback was used.")
        return selected[0]

    async def _expose(self, connection, file, *, area_id=None):
        area_id = self.area_id if area_id is None else area_id
        source, _ = await read_file_original(connection, self.sources, self.owner_id, area_id, file)
        metadata = source.original_evidence()
        metadata.pop("content")
        citation = None
        if self.task_evidence is not None:
            citation = (await self.task_evidence.bind_in_transaction(connection, [metadata]))[0]
        self._authorize(self.owner_id)
        return file.model_copy(update={"memory_source": metadata, "task_evidence": citation})

    async def _adopt_media(self, connection, file, spec, local_file, inspection, metadata_payload_id):
        if not isinstance(local_file, VerifiedLocalFile) or not isinstance(inspection, MP4Inspection):
            raise SourceError("file_media_original_invalid", "Media adoption requires exact inspected file bytes.")
        # The existing generated snapshot/action owner supplies authority. This
        # does not admit caller uploads or infer camera provenance from an MP4.
        expected = spec.metadata.get("generated_artifact", {}).get("validation", {}).get("media_inspection")
        if expected != inspection.model_dump(mode="json"):
            raise SourceError("file_media_inspection_changed", "Stored media differs from its authorized inspection.")
        await self._adopt(connection, file, spec, local_file, metadata_payload_id)

    async def _adopt(self, connection, file, spec, content, metadata_payload_id):
        self._authorize(self.owner_id)
        if self.selections is not None:
            area_id = await self._area_for(connection, file.id)
            fixed = SourceFileRepository(self.sources, self.storage, owner_id=self.owner_id, area_id=area_id,
                access_check=self.access_check, current_images=self.current_images, task_recovery=self.task_recovery,
                media_policy=self.media_policy, media_inspector=self.media_inspector,
                request_authority=self.request_authority)
            return await fixed._adopt(connection, file, spec, content, metadata_payload_id)
        action, snapshot = None, None
        if "task_artifact" in spec.metadata and self.task_evidence is None:
            if self.task_recovery is None:
                raise TaskError("file_task_adoption_required", "Unfinished worker files require their task authorization.", 403)
            action, snapshot = await self.task_recovery.authorize_adoption(
                connection, self.area_id, file, spec, content, metadata_payload_id)
        if self.task_evidence is not None:
            if "generated_artifact" in spec.metadata:
                invocation = current_tool_invocation().task
                raw = await connection.fetchval("""select p.content from jarvis_task_actions a
                    join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
                    join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
                    where a.owner_id=$1 and a.id=$2""",
                    self.owner_id, invocation.action_id if invocation is not None else None)
                snapshot = ArtifactSnapshot.model_validate(json_object(raw)) if raw is not None else None
                if snapshot is None:
                    raise TaskError("file_generated_scope_missing", "Generated-file authority is unavailable.", 403)
                action = await self._generated_action(connection, snapshot)
                if (not snapshot.matches(content) or spec.session_id != self.task_evidence.context.task.session_id
                        or spec.filename != snapshot.request.filename
                        or spec.description != snapshot.request.description
                        or spec.content_type != snapshot.inspection.content_type or spec.source != "assistant"
                        or spec.source_visual_frame_id is not None
                        or spec.metadata != self._generated_metadata(action, snapshot)):
                    raise TaskError("file_generated_snapshot_changed", "The original differs from its exact submitted snapshot.", 403)
            else:
                action = await self._task_action(connection, spec.session_id,
                    (spec.filename, content.decode("utf-8", errors="strict"), spec.description))
                if spec.metadata != self._task_metadata(action):
                    raise TaskError("file_task_lineage_mismatch", "File lineage does not match its submitting task.", 403)
        original = {"record": file.model_dump(mode="json"),
                    "content_sha256": (content.content_sha256 if isinstance(content, VerifiedLocalFile)
                                       else sha256(content).hexdigest())}
        if snapshot is not None:
            original.update(format=GENERATED_ORIGINAL_FORMAT, validation=snapshot.inspection.summary(),
                text_excerpt=snapshot.inspection.text_excerpt, text_complete=snapshot.inspection.text_complete,
                evidence_payload_id=str(snapshot.evidence_payload_id),
                meaning="Exact stored generated bytes; an excerpt or metadata is not a substitute for the original.")
            source_type, trust, source_time = "generated_file", "assistant_output", file.created_at
            time_basis = "Verified file-adoption time; generated claims are not observed events."
        elif "workspace_note_export" in spec.metadata:
            from app.files.workspace_exports import validate_note_export_original
            text = await validate_note_export_original(connection, self.sources,
                owner_id=self.owner_id, area_id=self.area_id, spec=spec, content=content,
                metadata_payload_id=metadata_payload_id)
            original.update(format=FILE_ORIGINAL_FORMAT, text=text)
            source_type, trust, source_time = "generated_text_file", "assistant_output", file.created_at
            time_basis = "Exact export of a retained note version; stored claims are not physical observations."
        elif spec.source == "assistant" and spec.content_type == "text/plain; charset=utf-8":
            text = content.decode("utf-8", errors="strict")
            self._validate_text(text)
            original.update(format=FILE_ORIGINAL_FORMAT, text=text)
            source_type, trust, source_time = "generated_text_file", "assistant_output", file.created_at
            time_basis = "Verified file-adoption time; generated text is not proof its claims occurred."
        elif spec.source == "visual_frame" and spec.content_type == "image/jpeg":
            reference = ImageReference.from_metadata(spec.metadata["capture_reference"], owner_id=self.owner_id)
            if (reference.source != "current" or reference.retained_file_id is not None or reference.record_id is not None
                    or reference.frame_id != spec.source_visual_frame_id or reference.session_id != spec.session_id):
                raise SourceError("file_capture_reference_invalid", "The file does not name its admitted camera frame.")
            row = await connection.fetchrow(
                """select captured_at,content_type,width,height,visual_session_id,frame_sequence,server_received_at
                    from visual_frames where owner_id=$1 and session_id=$2 and id=$3""",
                self.owner_id, reference.session_id, reference.frame_id)
            if row is None or any(row[key] != getattr(reference, key) for key in row.keys()):
                raise SourceError("file_capture_frame_changed", "The original frame identity or capture clock changed.")
            try:
                if jpeg_dimensions(content) != (reference.width, reference.height):
                    raise ValueError("dimensions")
            except (VisualFrameValidationError, ValueError) as error:
                raise SourceError("file_capture_pixels_invalid", "The saved JPEG does not match its capture receipt.") from error
            retained = replace(reference, source="historical", retained_file_id=file.id, raw_image_expires_at=None,
                               stream_connection_id=None, sampled_at=None, sampled_monotonic=None)
            original.update(format=IMAGE_ORIGINAL_FORMAT, image=retained.to_metadata(),
                            meaning="Actual saved camera pixels; the description is not verified visual truth.",
                            pixel_origin="sensor", description_origin="assistant_output")
            source_type, trust, source_time = "saved_visual_frame", "tool_receipt", reference.captured_at
            time_basis = "Original camera captured_at; file adoption does not refresh the scene or its clock."
        else:
            raise SourceError("file_source_type_unsupported", "The producer has no supported original representation.")
        payload = await connection.fetchval(
            """insert into jarvis_context_payloads(owner_id,availability,content)
                values($1,'available',$2::jsonb) returning id""", self.owner_id,
            json.dumps(original, allow_nan=False))
        await connection.execute(
            """insert into jarvis_context_payload_dependencies
                (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3)""",
            self.owner_id, payload, metadata_payload_id)
        source = await self.sources.publish_in_transaction(connection, self.owner_id, SourcePublication(
            area_id=self.area_id, namespace=FILE_NAMESPACE, source_key=str(file.id), payload_id=payload,
            source_type=source_type, trust=trust, source_time=source_time,
            session_id=file.session_id, expected_revision=0, time_basis=time_basis))
        if action is not None:
            await self._task_receipt(connection, action, file, source,
                                     tool_name="publish_generated_file" if snapshot is not None else "create_text_file")

    def _task_metadata(self, action):
        ctx = self.task_evidence.context
        return {"task_artifact": {"task_id": str(ctx.task.id), "area_id": str(self.area_id),
            "task_revision": ctx.claim.task_revision,
            "attempt_id": str(ctx.claim.attempt_id), "action_id": str(action["id"])}}

    async def _task_action(self, connection, session_id, parameters):
        ctx, invocation = self.task_evidence.context, current_tool_invocation()
        task = invocation.task
        if (session_id != ctx.task.session_id or task is None or task.task_id != ctx.task.id
                or task.task_revision != ctx.claim.task_revision or task.attempt_id != ctx.claim.attempt_id
                or task.action_id is None):
            raise TaskError("file_write_scope_mismatch", "A current authorized task action is required.", 403)
        await ctx.tasks.lock_owner(connection, self.owner_id)
        _, action = await ctx.actions._locked_action(connection, ctx.claim, task.action_id)
        grant = await ctx.tasks.validate_grant(connection, self.owner_id, action["grant_id"])
        scope = json_object(grant["scope"])
        if (action["status"] != "submitting" or action["submission_token"] is None
                or action["operation_key"] != task.operation_key or grant["capability"] != "files.create_text"
                or action["parameters_digest"] != digest(scope)
                or set(scope) - {"filename", "content", "description"}
                or text_file_parameters(scope) != parameters
                or invocation.request_event_id != grant["request_event_id"]
                or invocation.request_revision != grant["request_revision"]):
            raise TaskError("file_write_not_authorized", "The file does not match its exact submitted grant.", 403)
        return action

    async def _task_receipt(self, connection, action, file, source, *, tool_name="create_text_file"):
        # Original, lineage and effect receipt commit together. Uploads happen
        # earlier; interruption cannot turn an adopted file into a second save.
        if self.task_evidence is None:
            return await self.task_recovery.record_receipt(connection, action, file, source, tool_name=tool_name)
        metadata = source.original_evidence()
        metadata.pop("content")
        citation = (await self.task_evidence.bind_in_transaction(connection, [metadata]))[0]
        file = file.model_copy(update={"memory_source": metadata, "task_evidence": citation})
        ctx = self.task_evidence.context
        await ctx.actions.record_receipt_in_transaction(connection, self.owner_id, ctx.task.id,
            action["id"], action["submission_token"], "accepted", {"tool": tool_name,
                "parameters_digest": action["parameters_digest"], "result": success({"file": _file_payload(file)})})
        receipt = await connection.fetchval(
            "select receipt_payload_id from jarvis_task_actions where owner_id=$1 and id=$2",
            self.owner_id, action["id"])
        await connection.execute("""insert into jarvis_context_payload_dependencies
            (owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3) on conflict do nothing""",
            self.owner_id, receipt, source.payload_id)

    @staticmethod
    def _validate_text(content):
        if (not isinstance(content, str) or not content.strip() or len(content) > 200000
                or "\x00" in content):
            raise ValueError("A nonblank UTF-8 text file of at most 200000 characters without NUL is required")
        content.encode("utf-8", errors="strict")

    @staticmethod
    def _operation_key(session_id, parameters, *, kind="text-file"):
        invocation = current_tool_invocation()
        if invocation.request_event_id is not None:
            identity = {"event": str(invocation.request_event_id), "revision": invocation.request_revision,
                        "parameters": parameters}
        elif invocation.provider_call_id is not None:
            identity = {"session": str(session_id), "call": invocation.provider_call_id}
        else:
            identity = {"direct": str(uuid4())}
        return kind + ":" + digest(identity)

    async def create_text_file(self, *, owner_id, session_id, filename, content, description,
                               operation_key=None, source_payload_ids=(), memory_sources=()):
        self._authorize(owner_id)
        if self.selections is not None and session_id != self.session_id:
            raise SourceError("knowledge_session_unavailable", "The file belongs to another conversation.", 403)
        self._validate_text(content)
        spec = FilePublicationSpec(session_id=session_id, filename=filename, description=description,
                                   content_type="text/plain; charset=utf-8", source="assistant")
        if self.request_authority is not None:
            if operation_key is not None or source_payload_ids or memory_sources:
                raise SourceError("file_request_scope_override", "Request-scoped file identity and dependencies are server-owned.", 403)
            spec = self.request_authority.bind(owner_id=owner_id, spec=spec)
        parameters = {"filename": filename, "content": content, "description": description}
        action = None
        if self.task_evidence is not None:
            if operation_key is not None or source_payload_ids or memory_sources:
                raise TaskError("file_task_scope_override", "Task file identity and evidence are server-owned.", 403)
            async with self.sources.transaction(owner_id) as connection:
                action = await self._task_action(connection, session_id, (filename, content, description))
            operation_key = "task-text-file:" + str(action["id"])
            source_payload_ids = (action["parameters_payload_id"],)
            spec = spec.model_copy(update={"metadata": self._task_metadata(action)})
        result = await self.publications.publish(owner_id=owner_id,
            operation_key=operation_key or self._operation_key(session_id, parameters), spec=spec,
            content=content.encode("utf-8"), source_payload_ids=source_payload_ids, memory_sources=memory_sources)
        if result.phase != "available":
            raise SourceError("file_publication_unresolved",
                f"File operation {result.operation_id} is {result.phase}; do not claim it was saved or retry with a new identity.")
        file = await self.get_file(owner_id=owner_id, file_id=result.file_id)
        if action is not None:
            # Return the exact atomic receipt, even if ingestion progress changed
            # between adoption and this read. Source identity was rechecked above.
            async with self.sources.transaction(owner_id) as connection:
                raw = await connection.fetchval("""select p.content from jarvis_task_actions a
                    join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                    where a.owner_id=$1 and a.id=$2 and a.status='accepted' and p.availability='available'""",
                    owner_id, action["id"])
            if raw is None:
                raise TaskError("file_task_receipt_unavailable", "The adopted file receipt is unavailable.")
            saved = json_object(raw)["result"]["file"]
            file = file.model_copy(update={"memory_source": saved["memory_source"],
                                         "task_evidence": saved.get("task_evidence")})
        return file

    def _generated_metadata(self, action, snapshot):
        return self._task_metadata(action) | {"generated_artifact": {
            "operation_key": snapshot.request.operation_key, "format": snapshot.request.format,
            "validation": snapshot.inspection.summary(), "evidence_payload_id": str(snapshot.evidence_payload_id)}}

    async def _generated_action(self, connection, snapshot):
        if self.task_evidence is None:
            raise TaskError("file_generated_task_required", "Generated-file publication requires a worker grant.", 403)
        ctx, invocation = self.task_evidence.context, current_tool_invocation()
        selected = invocation.task
        if (selected is None or selected.task_id != ctx.task.id or selected.task_revision != ctx.claim.task_revision
                or selected.attempt_id != ctx.claim.attempt_id or selected.action_id is None):
            raise TaskError("file_write_scope_mismatch", "A current authorized task action is required.", 403)
        await ctx.tasks.lock_owner(connection, self.owner_id)
        _, action = await ctx.actions._locked_action(connection, ctx.claim, selected.action_id)
        grant = await ctx.tasks.validate_grant(connection, self.owner_id, action["grant_id"])
        parent = await ctx.tasks.validate_grant(connection, self.owner_id, snapshot.parent_grant_id)
        attached = await connection.fetchval("""select exists(select 1 from jarvis_task_grants
            where owner_id=$1 and task_id=$2 and grant_id=$3)""",
            self.owner_id, ctx.task.id, snapshot.parent_grant_id)
        if (action["status"] not in {"submitting", "outcome_unknown"} or action["submission_token"] is None
                or action["operation_key"] != selected.operation_key or grant["capability"] != "files.publish_snapshot"
                or parent["capability"] != "files.publish_generated" or not attached
                or snapshot.task_id != ctx.task.id
                or ArtifactSnapshot.model_validate(json_object(grant["scope"])) != snapshot
                or snapshot.inspection.format != snapshot.request.format
                or action["parameters_digest"] != digest(snapshot.model_dump(mode="json"))
                or (grant["request_event_id"], grant["request_revision"]) !=
                   (parent["request_event_id"], parent["request_revision"])
                or invocation.request_event_id != grant["request_event_id"]
                or invocation.request_revision != grant["request_revision"]):
            raise TaskError("file_write_not_authorized", "The generated file does not match its submitted grant.", 403)
        return action

    async def publish_generated_snapshot(self, *, owner_id, snapshot, content=None):
        self._authorize(owner_id)
        if self.request_authority is not None:
            raise SourceError("file_request_mode_unsupported", "Request-scoped generated-file publication is not enabled.", 403)
        snapshot = ArtifactSnapshot.model_validate(snapshot)
        async with self.sources.transaction(owner_id) as connection:
            action = await self._generated_action(connection, snapshot)
            key = "task-generated-file:" + str(action["id"])
            prior = await connection.fetchrow("""select id,phase from jarvis_file_publications
                where owner_id=$1 and operation_key=$2""", owner_id, key)
        if content is not None and not snapshot.matches(content):
            raise TaskError("file_generated_snapshot_changed", "The exact generated bytes changed.", 403)
        if prior is not None:
            # Even an unknown upload is reconciled from storage. No new VM read
            # or upload is permitted across the previous effect boundary.
            result = await self.publications.reconcile(owner_id=owner_id, operation_id=prior["id"])
        else:
            if action["status"] != "submitting" or content is None:
                raise TaskError("file_generated_original_required", "An unresolved effect cannot invent or resubmit original bytes.")
            spec = FilePublicationSpec(session_id=self.task_evidence.context.task.session_id,
                filename=snapshot.request.filename, content_type=snapshot.inspection.content_type,
                description=snapshot.request.description, source="assistant",
                metadata=self._generated_metadata(action, snapshot))
            if snapshot.inspection.format == "mp4":
                result = await self.publications.publish_file(owner_id=owner_id, operation_key=key, spec=spec,
                    local_file=content, source_payload_ids=(action["parameters_payload_id"],),
                    transfer_policy=snapshot.publication_policy)
            else:
                result = await self.publications.publish(owner_id=owner_id, operation_key=key, spec=spec,
                    content=content, source_payload_ids=(action["parameters_payload_id"],),
                    transfer_policy=snapshot.publication_policy)
        if result.phase != "available":
            raise SourceError("file_publication_unresolved",
                f"File operation {result.operation_id} is {result.phase}; reuse its status, not a new upload.")
        file = await self.get_file(owner_id=owner_id, file_id=result.file_id)
        async with self.sources.transaction(owner_id) as connection:
            raw = await connection.fetchval("""select p.content from jarvis_task_actions a
                join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
                where a.owner_id=$1 and a.id=$2 and a.status='accepted' and p.availability='available'""",
                owner_id, action["id"])
        if raw is None:
            raise TaskError("file_task_receipt_unavailable", "The adopted file receipt is unavailable.")
        saved = json_object(raw)["result"]["file"]
        return file.model_copy(update={"memory_source": saved["memory_source"], "task_evidence": saved.get("task_evidence")})

    async def capture_current_frame(self, *, owner_id, session_id, filename, description, operation_key=None):
        self._authorize(owner_id)
        if self.request_authority is not None:
            raise SourceError("file_request_mode_unsupported", "Request-scoped camera capture is not enabled.", 403)
        if self.selections is not None and session_id != self.session_id:
            raise SourceError("knowledge_session_unavailable", "The capture belongs to another conversation.", 403)
        if self.task_evidence is not None:
            raise TaskError("file_capture_not_granted", "Worker camera capture is not an authorized file capability.", 403)
        key = operation_key or self._operation_key(session_id, {"filename": filename, "description": description},
                                                   kind="image-file")
        async with self.sources.transaction(owner_id) as connection:
            prior = await connection.fetchrow(
                "select id from jarvis_file_publications where owner_id=$1 and operation_key=$2", owner_id, key)
            row = await self.publications._row(connection, owner_id, prior["id"]) if prior else None
            area_id = row["area_id"] if row is not None else await self._area_for(connection)
            await self.sources._area(connection, owner_id, area_id)
            if row is not None:
                if row["phase"] == "deleted" or row["availability"] != "available":
                    raise SourceError("file_publication_revoked", "The earlier saved-view operation is no longer available.")
                spec = FilePublicationSpec.model_validate(self.publications._specification(row)["spec"])
                if (spec.session_id, spec.filename, spec.description, spec.source) != (
                        session_id, filename, description, "visual_frame"):
                    raise SourceError("file_capture_replay_conflict", "This operation already names a different saved view.")
        if row is not None and row["phase"] != "prepared":
            result = await self.publications.reconcile(owner_id=owner_id, operation_id=row["id"])
        else:
            if self.current_images is None:
                raise ImageUnavailableError("no_active_stream", "Current camera capture is unavailable.")
            reference = (ImageReference.from_metadata(spec.metadata["capture_reference"], owner_id=owner_id)
                         if row is not None else self.current_images.select_current_image(owner_id=owner_id, session_id=session_id))
            resolved = self.current_images.resolve_current_image(owner_id=owner_id, session_id=session_id, reference=reference)
            spec = FilePublicationSpec(session_id=session_id, filename=filename, description=description,
                content_type="image/jpeg", source="visual_frame", source_visual_frame_id=reference.frame_id,
                metadata={"capture_reference": reference.to_metadata()}) if row is None else spec
            result = await self.publications.publish(owner_id=owner_id, operation_key=key, spec=spec, content=resolved.image_bytes)
        if result.phase != "available":
            raise SourceError("file_publication_unresolved",
                f"Image operation {result.operation_id} is {result.phase}; no new camera frame was substituted.")
        return await self.get_file(owner_id=owner_id, file_id=result.file_id)

    async def get_file(self, *, owner_id, file_id):
        self._authorize(owner_id)
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection, file_id)
            await self.sources._area(connection, owner_id, area_id)
            record = await connection.fetchrow(
                "select * from jarvis_files where owner_id=$1 and id=$2", owner_id, file_id)
            if record is None:
                return None
            return await self._expose(connection, _file_from_record(record), area_id=area_id)

    async def list_files(self, *, owner_id, query, limit):
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= self.list_limit:
            raise ValueError("Source file list limit must be between 1 and 8")
        pattern = f"%{query}%" if query else None
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection)
            await self.sources._area(connection, owner_id, area_id)
            unmatched = await connection.fetchval(
                """select exists(select 1 from jarvis_files f where f.owner_id=$1
                    and ($2::text is null or f.filename ilike $2 or f.description ilike $2)
                    and not exists(select 1 from jarvis_knowledge_sources s where s.owner_id=f.owner_id
                        and s.namespace=$3 and s.source_key=f.id::text))""", owner_id, pattern, FILE_NAMESPACE)
            if unmatched:
                raise SourceError("file_import_incomplete", "Matching legacy files still require source publication.")
            records = await connection.fetch(
                """select f.* from jarvis_files f join jarvis_file_publications p
                    on p.owner_id=f.owner_id and p.file_id=f.id
                    join jarvis_knowledge_sources s on s.owner_id=f.owner_id
                        and s.namespace=$4 and s.source_key=f.id::text and s.area_id=p.area_id
                    where f.owner_id=$1 and p.area_id=$3 and p.phase='available' and s.status='active'
                        and ($2::text is null or f.filename ilike $2 or f.description ilike $2)
                    order by f.created_at desc,f.id limit $5""", owner_id, pattern, area_id, FILE_NAMESPACE, limit)
            files = [await self._expose(connection, _file_from_record(row), area_id=area_id) for row in records]
            self._authorize(owner_id)
            return SourceFileSearchResult(files, area_id) if self.selections is not None else files

    async def list_file_page(self, *, owner_id, query, limit=8, media_only=False, cursor=None):
        """Browse canonical Files with filtering before the bounded page limit."""
        from app.files.pagination import FilePage, decode_cursor, encode_cursor
        self._authorize(owner_id)
        if type(limit) is not int or not 1 <= limit <= self.list_limit or type(media_only) is not bool:
            raise ValueError("A file page needs a limit from1to8 and an explicit media filter")
        before_time, before_id = decode_cursor(cursor)
        pattern = f"%{query}%" if query else None
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection)
            await self.sources._area(connection, owner_id, area_id)
            # Keep the established legacy-import fence, scoped to this filter.
            unmatched = await connection.fetchval("""select exists(select 1 from jarvis_files f
                where f.owner_id=$1 and ($2::text is null or f.filename ilike $2 or f.description ilike $2)
                    and (not $4::boolean or f.content_type like 'image/%' or f.content_type='video/mp4')
                    and not exists(select 1 from jarvis_knowledge_sources s where s.owner_id=f.owner_id
                        and s.namespace=$3 and s.source_key=f.id::text))""",
                owner_id, pattern, FILE_NAMESPACE, media_only)
            if unmatched:
                raise SourceError("file_import_incomplete", "Matching legacy files still require source publication.")
            records = await connection.fetch("""select f.* from jarvis_files f
                join jarvis_file_publications p on p.owner_id=f.owner_id and p.file_id=f.id
                join jarvis_knowledge_sources s on s.owner_id=f.owner_id
                    and s.namespace=$4 and s.source_key=f.id::text and s.area_id=p.area_id
                where f.owner_id=$1 and p.area_id=$3 and p.phase='available' and s.status='active'
                    and ($2::text is null or f.filename ilike $2 or f.description ilike $2)
                    and (not $6::boolean or f.content_type like 'image/%' or f.content_type='video/mp4')
                    and ($7::timestamptz is null or f.created_at<$7
                        or (f.created_at=$7 and f.id>$8::uuid))
                order by f.created_at desc,f.id limit $5""",
                owner_id, pattern, area_id, FILE_NAMESPACE, limit + 1, media_only, before_time, before_id)
            selected = records[:limit]
            files = [await self._expose(connection, _file_from_record(row), area_id=area_id) for row in selected]
            self._authorize(owner_id)
            next_cursor = (encode_cursor(selected[-1]["created_at"], selected[-1]["id"])
                           if len(records) > limit else None)
            return FilePage(files, next_cursor)

    async def read_content(self, *, owner_id, file_id):
        first = await self.get_file(owner_id=owner_id, file_id=file_id)
        if first is None:
            raise SourceError("file_not_found", "No retained file exists in the authorized area.", 404)
        content = await self.publications.read_content(owner_id=owner_id, file_id=file_id)
        current = await self.get_file(owner_id=owner_id, file_id=file_id)
        if current is None or MemorySourceReference.from_metadata(first.memory_source) != MemorySourceReference.from_metadata(
                current.memory_source):
            raise SourceError("file_source_changed", "File evidence changed while reading the stored bytes.")
        return content

    async def read_file(self, *, owner_id, file_id):
        """Exact source-checked disk read; caller owns closing the private file."""
        first = await self.get_file(owner_id=owner_id, file_id=file_id)
        if first is None:
            raise SourceError("file_not_found", "No retained file exists in the authorized area.", 404)
        local_file = await self.publications.read_file(owner_id=owner_id, file_id=file_id)
        try:
            current = await self.get_file(owner_id=owner_id, file_id=file_id)
            if (current is None or MemorySourceReference.from_metadata(first.memory_source)
                    != MemorySourceReference.from_metadata(current.memory_source)):
                raise SourceError("file_source_changed", "File evidence changed while reading the stored bytes.")
            return local_file
        except BaseException:
            local_file.close()
            raise

    async def delete_source_file(self, *, owner_id, file_id):
        self._authorize(owner_id)
        if self.request_authority is not None:
            raise SourceError("file_request_mode_unsupported", "Request-scoped file deletion is not enabled.", 403)
        if self.task_evidence is not None:
            raise TaskError("file_delete_not_granted", "Worker file deletion is not an authorized capability.", 403)
        async with self.sources.transaction(owner_id) as connection:
            area_id = await self._area_for(connection, file_id)
            await self.sources._area(connection, owner_id, area_id, allow_archived=True)
            publication_id = await connection.fetchval(
                "select id from jarvis_file_publications where owner_id=$1 and area_id=$2 and file_id=$3",
                owner_id, area_id, file_id)
            if publication_id is None:
                raise SourceError("file_publication_missing", "No tracked file publication exists in the authorized area.", 404)
            mapping = await self._mapping(connection, file_id, required=False, area_id=area_id)
            if mapping is not None:
                await self.sources.delete_source_in_transaction(
                    connection, owner_id, area_id, mapping["id"], expected_revision=mapping["current_revision"])
            self._authorize(owner_id)
        result = await self.publications.delete(owner_id=owner_id, operation_id=publication_id)
        return {"file_id": str(file_id), "local_access": "deleted", "storage_cleanup": result.phase,
                "deleted": result.phase == "deleted", "provider_memory_cleanup": "inspect_recorded_status",
                "full_erasure_verified": False}

    async def capture_tool_result(self, result, context):
        self._authorize(context.capture.owner_id)
        if self.sources.pool is not context.capture.repository.pool:
            raise SourceError("knowledge_context_pool_mismatch", "File capture requires the same source database.")
        files = result["files"] if context.tool_name == "list_files" else [result["file"]]
        area_id, scope = self.area_id, {}
        if self.selections is not None:
            if context.capture.session_id != self.session_id:
                raise SourceError("knowledge_session_unavailable", "The file result belongs to another conversation.", 403)
            selected = result["knowledge_scope"] if context.tool_name == "list_files" else result["file"]["memory_source"]
            area_id = UUID(selected["area_id"])
            if any(UUID(file["memory_source"]["area_id"]) != area_id for file in files):
                raise SourceError("knowledge_result_scope_mismatch", "A file result contains conflicting area references.")
            scope = {"publication_area_id": area_id}
        async def materialize(connection, originals):
            record = await context.capture.bound_tool_outcome(
                connection, context.call_id, context.tool_name, result, originals, **scope)
            self._authorize(self.owner_id)
            return SourceBoundToolResult(result, record), record.payload_id
        return await self.provenance.bind(self.owner_id, area_id,
            [file["memory_source"] for file in files], materialize)

    async def revalidate_receipt(self, action, result):
        if action.capability not in {"files.create_text", "files.publish_snapshot"} or result.get("status") != "success":
            return result
        file = await self.get_file(owner_id=self.owner_id, file_id=UUID(result["file"]["id"]))
        if file is None or MemorySourceReference.from_metadata(file.memory_source) != MemorySourceReference.from_metadata(
                result["file"]["memory_source"]):
            raise SourceError("file_receipt_source_changed", "The saved file changed after this receipt.")
        return result | {"file": result["file"] | {"task_evidence": file.task_evidence}}
