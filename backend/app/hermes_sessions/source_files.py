"""Retained-session files through existing storage, source and media services.

No task grants or foreground requests are invented. Exact native input scopes
supply owner/session authority; FilePublicationService still owns stable uploads,
byte rechecks, previews and atomic file adoption.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import logging

from app.files.generated import ArtifactInspection, ArtifactRequest, EXTENSIONS
from app.files.local_file import VerifiedLocalFile
from app.files.publication import FilePublicationService, FilePublicationSpec
from app.files.tools import _file_payload, create_file_tools, text_file_parameters
from app.files.transfer import FileManifest, FileTransferPolicy, MP4_OUTPUT_MAX_BYTES, MP4_OUTPUT_MAX_DURATION_MS
from app.hermes_sessions.bindings import BindingError, identity
from app.knowledge.models import MemorySourceReference, SourceError, SourcePublication
from app.knowledge.product_files import FILE_NAMESPACE, GENERATED_ORIGINAL_FORMAT
from app.sandbox.file_transfer import INPUT_ROOT, workspace_file
from app.tools.inputs import required_uuid
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition

LOG = logging.getLogger(__name__)
_AUTH = "hermes_session_authority"


class RetainedFileTools:
    def __init__(self, *, service, scope_for_input, files, sandbox_service=None):
        self.service, self.scope_for_input, self.files = service, scope_for_input, files
        self.sandbox_service = sandbox_service
        self.profile = hashlib.sha256(str(service.bindings.profile_home).encode()).hexdigest()
        self.allowed_tools = frozenset({"list_files", "get_file_metadata", "create_text_file"} |
            ({"sandbox_stage_file", "sandbox_publish_file"} if sandbox_service is not None else set()))
        self.write_tools = self.allowed_tools - {"list_files", "get_file_metadata"}
        self.media_policy = self.media_inspector = None
        if files.media_policy is not None and files.media_inspector is not None:
            self.media_policy = FileTransferPolicy("mp4_output_v1", files.media_policy.max_bytes)
            self.media_inspector = getattr(files.media_inspector, "inspect_retained_input", files.media_inspector)

    async def check(self, scope, tool):
        if scope.owner_id != self.service.bindings.owner_id or scope.owner_id != self.files.owner_id:
            raise BindingError("retained_file_owner_mismatch")
        binding = self.service.bindings.get(scope.logical_session_id)
        if binding.native_id is None:
            raise BindingError("retained_file_session_unavailable")
        current = await self.scope_for_input(binding, scope.input_id)
        if current != scope or tool not in current.allowed_tools:
            raise BindingError("retained_file_input_unavailable")
        if self.files.access_check is not None:
            self.files.access_check(scope.owner_id)

    def publication(self, session_id):
        files = self.files
        return FilePublicationService(files.sources, files.storage,
            owner_id=files.owner_id, area_id=files.area_id, access_check=files.access_check,
            selections=files.selections, session_id=session_id, adopt=self.adopt,
            transfer_policy=self.media_policy, inspect_file=self.media_inspector,
            adopt_file=self.adopt_media if self.media_policy is not None else None)

    async def authorize_publication(self, connection, spec):
        raw = spec.metadata.get(_AUTH)
        if (not isinstance(raw, dict) or set(raw) != {
                "owner_id", "profile", "logical_session_id", "input_id", "tool"}
                or raw["owner_id"] != self.files.owner_id or raw["profile"] != self.profile
                or raw["tool"] not in {"create_text_file", "sandbox_publish_file"}):
            raise SourceError("retained_file_authority_invalid", "File authority is not this retained session.", 403)
        binding = self.service.bindings.get(raw["logical_session_id"])
        scope = await self.scope_for_input(binding, raw["input_id"])
        await self.check(scope, raw["tool"])
        if spec.session_id != scope.jarvis_session_id or not await connection.fetchval(
                "select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2)",
                scope.owner_id, scope.jarvis_session_id):
            raise SourceError("retained_file_session_invalid", "File session no longer belongs to this owner.", 403)
        return scope

    async def adopt(self, connection, file, spec, content, metadata_payload_id):
        await self.authorize_publication(connection, spec)
        inspection = ArtifactInspection.model_validate(spec.metadata.get("inspection"))
        manifest = (FileManifest.from_file(content) if isinstance(content, VerifiedLocalFile) else
                    FileManifest(size_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest()))
        if (inspection.size_bytes != manifest.size_bytes or inspection.content_sha256 != manifest.content_sha256
                or inspection.content_type != spec.content_type or spec.source != "assistant"
                or spec.source_visual_frame_id is not None):
            raise SourceError("retained_file_bytes_changed", "The published bytes differ from their inspection.", 403)
        files = self.files.for_session(spec.session_id)
        area_id = await files._area_for(connection, file.id)
        original = {"format": GENERATED_ORIGINAL_FORMAT, "record": file.model_dump(mode="json"),
            "content_sha256": manifest.content_sha256, "validation": inspection.summary(),
            "text_excerpt": inspection.text_excerpt, "text_complete": inspection.text_complete,
            "evidence_payload_id": str(metadata_payload_id),
            "meaning": "Verified generated file bytes, not independent evidence of the author's claims."}
        payload = await connection.fetchval(
            "insert into jarvis_context_payloads(owner_id,availability,content) "
            "values($1,'available',$2::jsonb) returning id", files.owner_id, json.dumps(original, allow_nan=False))
        await connection.execute("insert into jarvis_context_payload_dependencies "
            "(owner_id,dependent_payload_id,source_payload_id) values($1,$2,$3)",
            files.owner_id, payload, metadata_payload_id)
        await files.sources.publish_in_transaction(connection, files.owner_id, SourcePublication(
            area_id=area_id, namespace=FILE_NAMESPACE, source_key=str(file.id), payload_id=payload,
            source_type="generated_file", trust="assistant_output", source_time=file.created_at,
            time_basis="Verified file adoption; assistant output is not a new observed event.",
            session_id=spec.session_id, expected_revision=0))

    async def adopt_media(self, connection, file, spec, local, inspection, metadata_payload_id):
        expected = spec.metadata.get("inspection", {}).get("media_inspection")
        if inspection.model_dump(mode="json") != expected:
            raise SourceError("retained_media_changed", "Stored media differs from the inspected output.")
        await self.adopt(connection, file, spec, local, metadata_payload_id)

    def _operation(self, scope, tool):
        call_id = current_tool_invocation().provider_call_id
        if not call_id:
            raise BindingError("retained_file_call_required")
        return identity(scope.logical_session_id, scope.input_id, tool, call_id)

    async def publish(self, scope, *, tool, request, content, policy):
        await self.check(scope, tool)
        if isinstance(content, VerifiedLocalFile):
            content._check()
            if self.media_inspector is None:
                raise SourceError("retained_media_unavailable", "Trusted media inspection is not configured.")
            measured = await self.media_inspector(content, policy)
            measured.validate_file(FileManifest.from_file(content), policy)
            inspection = ArtifactInspection(format=request.format, size_bytes=content.size_bytes,
                content_sha256=content.content_sha256, media_inspection=measured)
        else:
            inspection = ArtifactInspection(format=request.format, size_bytes=len(content),
                content_sha256=hashlib.sha256(content).hexdigest())
        spec = FilePublicationSpec(session_id=scope.jarvis_session_id, filename=request.filename,
            description=request.description, content_type=inspection.content_type, source="assistant",
            metadata={_AUTH: {"owner_id": scope.owner_id, "profile": self.profile,
                "logical_session_id": scope.logical_session_id, "input_id": scope.input_id, "tool": tool},
                "inspection": inspection.model_dump(mode="json"),
                **({"media_inspection": inspection.media_inspection.model_dump(mode="json")}
                   if inspection.media_inspection is not None else {})})
        publications = self.publication(scope.jarvis_session_id)
        arguments = dict(owner_id=scope.owner_id, operation_key=request.operation_key, spec=spec, transfer_policy=policy)
        await self.check(scope, tool)
        if isinstance(content, VerifiedLocalFile):
            result = await publications.publish_file(**arguments, local_file=content)
        else:
            result = await publications.publish(**arguments, content=content)
        if result.phase != "available":
            raise SourceError("file_publication_unresolved",
                f"File operation {result.operation_id} is {result.phase}; do not claim it was saved.")
        file = await self.files.for_session(scope.jarvis_session_id).get_file(owner_id=scope.owner_id, file_id=result.file_id)
        if file is None:
            raise SourceError("file_original_unavailable", "The saved original could not be read back.")
        return {"status": "success", "file": _file_payload(file)}

    async def stage(self, scope, file_id):
        await self.check(scope, "sandbox_stage_file")
        files = self.files.for_session(scope.jarvis_session_id)
        original = await files.get_file(owner_id=scope.owner_id, file_id=file_id)
        if original is None:
            raise SourceError("file_not_found", "No owned source file was found.", 404)
        reference = MemorySourceReference.from_metadata(original.memory_source)
        manifest, policy = await files.publications.read_manifest(owner_id=scope.owner_id, file_id=file_id)
        local = None
        started = False
        sandboxes = self.sandbox_service()
        try:
            content = (await files.read_file(owner_id=scope.owner_id, file_id=file_id) if policy.is_media else
                       await files.read_content(owner_id=scope.owner_id, file_id=file_id))
            local = content if isinstance(content, VerifiedLocalFile) else None
            actual = (FileManifest.from_file(local) if local is not None else
                      FileManifest(size_bytes=len(content), content_sha256=hashlib.sha256(content).hexdigest()))
            if actual != manifest or original.size_bytes != manifest.size_bytes:
                raise SourceError("file_source_changed", "The source file changed before staging.")
            key = hashlib.sha256((scope.logical_session_id + str(file_id) + reference.model_dump_json()
                                  + manifest.content_sha256).encode()).hexdigest()

            async def revalidate():
                await self.check(scope, "sandbox_stage_file")
                current = await files.get_file(owner_id=scope.owner_id, file_id=file_id)
                if current is None or MemorySourceReference.from_metadata(current.memory_source) != reference:
                    raise SourceError("file_source_changed", "The staged source is no longer current.")

            async with sandboxes.use(scope.logical_session_id) as handle:
                await revalidate()
                started = True
                receipt = (await sandboxes.provider.stage_input_file(handle, key, local, policy=policy)
                           if local is not None else await sandboxes.provider.stage_input(handle, key, content))
                if receipt != {"path": INPUT_ROOT + "/" + key, "sha256": manifest.content_sha256,
                               "size_bytes": manifest.size_bytes}:
                    raise SourceError("file_stage_receipt_invalid", "The sandbox did not confirm the exact input.")
                await revalidate()
            return {"status": "success", "file": _file_payload(original),
                    "sandbox_input": dict(receipt, sandbox_id=handle.sandbox_id, read_only=True)}
        except BaseException:
            if started:
                try:
                    await asyncio.shield(sandboxes.pause(scope.logical_session_id))
                except Exception:
                    LOG.warning("Could not pause retained sandbox after failed file staging")
            raise
        finally:
            if local is not None:
                local.close()

    def module(self, scope):
        owner = self
        files = self.files.for_session(scope.jarvis_session_id)

        class Module:
            name = "retained_files"
            repository = files

            def build_tools(self, context):
                if context.owner_id != scope.owner_id or context.session_id != scope.jarvis_session_id:
                    raise BindingError("retained_file_build_scope_mismatch")
                tools = [tool for tool in create_file_tools(repository=files, storage=files.storage,
                    owner_id=scope.owner_id, session_id=scope.jarvis_session_id)
                    if tool.name in {"list_files", "get_file_metadata"}]

                async def text(arguments):
                    filename, content, description = text_file_parameters(arguments)
                    request = ArtifactRequest(operation_key=owner._operation(scope, "create_text_file"),
                        filename=filename, format="text", description=description)
                    return await owner.publish(scope, tool="create_text_file", request=request,
                        content=content.encode("utf-8"), policy=FileTransferPolicy())

                async def stage(arguments):
                    if set(arguments) != {"file_id"}:
                        raise ValueError("Use only an owned file_id")
                    return await owner.stage(scope, required_uuid(arguments, "file_id"))

                async def publish(arguments):
                    if set(arguments) - {"path", "filename", "format", "description"}:
                        raise ValueError("Unsupported file output fields")
                    path = workspace_file(arguments["path"])
                    request = ArtifactRequest(operation_key=owner._operation(scope, "sandbox_publish_file"),
                        filename=arguments["filename"], format=arguments["format"],
                        description=arguments.get("description") or "Jarvis-generated file.")
                    if not any(request.filename.lower().endswith(ext) for ext in EXTENSIONS[request.format]):
                        raise ValueError("Filename extension does not match the output format")
                    policy = owner.media_policy if request.format == "mp4" else FileTransferPolicy()
                    if policy is None:
                        raise SourceError("retained_media_unavailable", "MP4 publication is not configured.")
                    await owner.check(scope, "sandbox_publish_file")
                    sandboxes = owner.sandbox_service()
                    await sandboxes.inspect(scope.logical_session_id)
                    local = None
                    try:
                        async with sandboxes.use(scope.logical_session_id) as handle:
                            if request.format == "mp4":
                                local = await sandboxes.provider.read_workspace_file_snapshot(
                                    handle, path, policy=policy, max_bytes=policy.max_bytes)
                                content = local
                            else:
                                content = await sandboxes.provider.read_workspace_file(handle, path, max_bytes=policy.max_bytes)
                        return await owner.publish(scope, tool="sandbox_publish_file", request=request,
                                                   content=content, policy=policy)
                    finally:
                        if local is not None:
                            local.close()

                tools.append(ToolDefinition("create_text_file", "Create and retain a text file in Jarvis Files.",
                    {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"},
                     "description": {"type": ["string", "null"]}}, "required": ["filename", "content"],
                     "additionalProperties": False}, text, mutates_state=True))
                if owner.sandbox_service is not None:
                    tools.extend([
                        ToolDefinition("sandbox_stage_file", "Copy an owned Jarvis file into this session's sandbox. "
                            "Use the returned read-only path; no URL or host path is accepted.",
                            {"type": "object", "properties": {"file_id": {"type": "string"}},
                             "required": ["file_id"], "additionalProperties": False}, stage, mutates_state=True),
                        ToolDefinition("sandbox_publish_file", "Publish an existing workspace-relative sandbox file "
                            "to Jarvis Files. Documents: 8 MiB. "
                            f"MP4: at most {MP4_OUTPUT_MAX_BYTES // 1_000_000} MB and "
                            f"{MP4_OUTPUT_MAX_DURATION_MS // 60_000} minutes, H.264/AAC. "
                            "Only a returned saved-file receipt means publication completed.",
                            {"type": "object", "properties": {"path": {"type": "string"}, "filename": {"type": "string"},
                             "format": {"type": "string", "enum": list(EXTENSIONS)}, "description": {"type": "string"}},
                             "required": ["path", "filename", "format"], "additionalProperties": False},
                            publish, mutates_state=True)])
                return [replace(tool, result_capture=files.capture_tool_result,
                                capture_after_commit=tool.mutates_state) for tool in tools]

        return Module()
