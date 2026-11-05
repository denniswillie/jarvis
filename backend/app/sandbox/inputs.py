"""Stage only originals admitted by the task's existing source readers."""
import asyncio
from dataclasses import dataclass
import hashlib
import json
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.files.local_file import LocalFileError, VerifiedLocalFile
from app.files.transfer import FileTransferPolicy
from app.images import ImageUnavailableError
from app.knowledge.models import MemorySourceReference, SourceError
from app.sandbox.file_transfer import MAX_TRANSFER_BYTES
from app.tasks.models import ClaimLost, TaskError, json_text
from app.tools.inputs import failed
from app.tools.registry import ToolDefinition


class ContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID


class ImageInput(ContextInput):
    image_index: int = Field(default=0, ge=0, lt=16, strict=True)


class SavedFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file_id: UUID


@dataclass(frozen=True)
class _MediaInput:
    """An owned exact snapshot plus its retained server-side transfer policy."""
    local_file: VerifiedLocalFile
    policy: FileTransferPolicy


class SandboxInputService:
    def __init__(self, bridge, broker, *, files=None):
        self.bridge, self.broker, self.files = bridge, broker, files
        self._staged_revalidators = []

    async def revalidate_staged(self):
        """Derived workspace output cannot outlive an admitted input's authority."""
        for revalidate in self._staged_revalidators:
            await revalidate()

    def definitions(self):
        result = []
        def add(name, description, model, operation):
            async def handler(arguments):
                try:
                    request = model.model_validate(arguments)
                except ValidationError:
                    return failed("invalid_tool_input", "Use this input tool's declared source fields; "
                                  "host paths, URLs and replacement bytes are not accepted.")
                return await self.stage(operation, request)
            result.append(ToolDefinition(name, description, model.model_json_schema(), handler, mutates_state=True))
        if "read_task_context_source" in self.bridge.read_tools:
            add("sandbox_stage_context", "Copy the complete exact admitted session source into a read-only sandbox "
                "input. Use its event_id. Returns the source revision/trust, byte hash and native input path. "
                "Unavailable or superseded evidence is not replaced by a summary. The input is not a durable artifact.",
                ContextInput, self.context_source)
        if "read_task_context_image" in self.bridge.read_tools:
            add("sandbox_stage_image", "Copy an admitted source's actual original image bytes into a read-only "
                "sandbox input, preserving capture metadata. This is not a new camera capture or proof of a live view. "
                "Select event_id and image_index as for read_task_context_image.", ImageInput, self.context_image)
        if self.files is not None and "get_file_metadata" in self.bridge.read_tools:
            add("sandbox_stage_file", "Copy a currently authorized saved-file original into a read-only sandbox "
                "input by file_id. Preserves source identity and exact bytes. Retained MP4 files use their "
                "server-admitted transfer limit; other inputs retain the document limit. "
                "No host path or external URL is accepted.",
                SavedFileInput, self.saved_file)
        return tuple(result)

    async def read_tool(self, name, arguments):
        if name not in self.bridge.read_tools:
            raise TaskError("sandbox_input_not_allowed", "This source reader is not granted to the task.", 403)
        await self.bridge._check()
        value = await self.bridge._handler(self.bridge.registry.get(name), arguments, "sandbox-input-"+uuid4().hex)
        await self.bridge._check()
        if value.get("status")!="success":
            raise TaskError("sandbox_input_unavailable", "The source reader did not return an available original.")
        return value

    async def context_source(self, request):
        offset, chunks, source, total = 0, [], None, None
        while True:
            page = await self.read_tool("read_task_context_source",
                {"event_id":str(request.event_id),"offset":offset,"max_characters":64000})
            if (page.get("encoding")!="json_ascii" or page.get("offset")!=offset
                    or not isinstance(page.get("content_json"),str)
                    or type(page.get("total_characters")) is not int
                    or not 0<=page["total_characters"]<=MAX_TRANSFER_BYTES):
                raise TaskError("sandbox_input_invalid", "The original source exceeds its bound or paging contract.")
            if source is None:
                source, total = page["source"], page["total_characters"]
                if source.get("availability")!="available":
                    raise TaskError("sandbox_input_unavailable", "The original source is unavailable.")
            if page["source"]!=source or page["total_characters"]!=total:
                raise TaskError("sandbox_input_changed", "The source changed during input assembly.")
            chunks.append(page["content_json"])
            end = offset+len(page["content_json"])
            if end>total or (page["next_offset"] is None and end!=total):
                raise TaskError("sandbox_input_invalid", "Source pagination did not match its declared length.")
            if page["next_offset"] is None:
                break
            if page["next_offset"]!=end or end<=offset:
                raise TaskError("sandbox_input_invalid", "Source pagination did not advance.")
            offset=end
        content="".join(chunks).encode("ascii")
        json.loads(content)  # Reject partial/corrupt serialization, not interpret its claims.
        async def revalidate():
            current=await self.read_tool("read_task_context_source",
                {"event_id":str(request.event_id),"max_characters":64000})
            if current["source"]!=source or current["total_characters"]!=total:
                raise TaskError("sandbox_input_changed", "The staged source changed.")
        return content, {"kind":"context","filename":"context.json","content_type":"application/json","source":source}, revalidate

    async def context_image(self, request):
        selection=await self.read_tool("read_task_context_image", request.model_dump(mode="json"))
        resolved=await self.bridge.resolve_image("read_task_context_image",selection)
        if resolved.revalidate is None:
            raise TaskError("sandbox_input_unverifiable", "Original pixels require a source-policy recheck.")
        return resolved.image_bytes, {"kind":"image","filename":"captured-image",
            "content_type":resolved.reference.content_type,"selection":selection}, resolved.revalidate

    async def saved_file(self, request, *, max_bytes=None):
        owner=self.bridge.context.task.owner_id
        file=await self.files.get_file(owner_id=owner,file_id=request.file_id)
        if file is None:
            raise SourceError("file_not_found", "No authorized saved-file original is available.", 404)
        reference=MemorySourceReference.from_metadata(file.memory_source)
        async def revalidate():
            current=await self.files.get_file(owner_id=owner,file_id=request.file_id)
            if current is None or MemorySourceReference.from_metadata(current.memory_source)!=reference:
                raise SourceError("file_source_changed", "The input's original source changed.")
        if file.content_type == "video/mp4":
            # The persisted publication supplies the concrete policy. Neither
            # the model's arguments nor a filename can widen an input limit.
            manifest, policy=await self.files.publications.read_manifest(owner_id=owner,file_id=request.file_id)
            if (not isinstance(policy,FileTransferPolicy) or not policy.is_media
                    or not 0 < manifest.size_bytes <= policy.max_bytes
                    or manifest.size_bytes != file.size_bytes):
                raise TaskError("sandbox_media_input_invalid", "The saved media has no matching admitted transfer policy.")
            if max_bytes is not None and manifest.size_bytes > max_bytes:
                raise TaskError("media_source_limit", "The exact original exceeds this task's input byte limit.")
            local=await self.files.read_file(owner_id=owner,file_id=request.file_id)
            try:
                if (not isinstance(local,VerifiedLocalFile) or local.size_bytes != manifest.size_bytes
                        or local.content_sha256 != manifest.content_sha256):
                    raise SourceError("file_source_changed", "The stored media differs from its exact publication.")
                local._check()
                await revalidate()
                content=_MediaInput(local,policy)
            except BaseException:
                if isinstance(local,VerifiedLocalFile):
                    local.close()
                raise
        else:
            content=await self.files.read_content(owner_id=owner,file_id=request.file_id)
        return content, {"kind":"saved_file","file_id":str(file.id),"filename":file.filename,
            "content_type":file.content_type,"source":file.memory_source,"task_evidence":file.task_evidence}, revalidate

    async def stage(self, operation, request, *, allocation_id=None):
        if allocation_id is not None and not isinstance(allocation_id, UUID):
            raise TaskError("sandbox_original_allocation_required", "Use the existing trusted allocation identity.", 403)
        context=self.bridge.context
        reservation=await context.reserve("sandbox-input:"+uuid4().hex,tool_steps=1)
        await context.budgets.submit(context.claim,reservation["id"])
        allocation=None
        content=None
        try:
            content, metadata, revalidate=await operation(request)
            if isinstance(content,_MediaInput):
                content.local_file._check()
                if not 0 < content.local_file.size_bytes <= content.policy.max_bytes:
                    raise TaskError("sandbox_input_too_large", "The exact media exceeds its retained transfer bound.")
                content_hash=content.local_file.content_sha256
            elif type(content) is not bytes or len(content)>MAX_TRANSFER_BYTES:
                raise TaskError("sandbox_input_too_large", "The exact source exceeds the input byte bound.")
            else:
                content_hash=hashlib.sha256(content).hexdigest()
            if allocation_id is None:
                allocation=await self.broker._ready(await self.broker.allocation())
            else:
                # Recovery may restage exact original input only in the same
                # still-ready allocation. Never reserve, prepare, discover or
                # replace a VM just because the original evidence is missing.
                current=await self.broker.allocations.current(context.claim,allocation_id)
                if current["phase"]!="ready":
                    raise TaskError("sandbox_original_allocation_unavailable", "The original input allocation is not ready.")
                allocation=current
            await revalidate()
            await self.bridge._check()
            identity=hashlib.sha256((json_text(metadata)+":"+content_hash).encode()).hexdigest()
            if isinstance(content,_MediaInput):
                receipt=await self.broker.provider.stage_input_file(self.broker.allocations.handle(allocation),
                    identity,content.local_file,policy=content.policy)
            else:
                receipt=await self.broker.provider.stage_input(self.broker.allocations.handle(allocation),identity,content)
            # Recheck after transport too. Lost source authority revokes the
            # entire attempt VM, not just the returned path to its copied bytes.
            await revalidate()
            await self.bridge._check()
            self._staged_revalidators.append(revalidate)
            return {"status":"success","sandbox_input":dict(receipt,source=metadata,read_only=True,
                meaning="Exact source snapshot for computation; not a durable user download or verified truth.")}
        except BaseException as error:
            # Transport errors and cancellation can occur after bytes reached
            # the VM too. Revoke that same allocation on every unsuccessful
            # staging attempt; never leave copied input usable after a lost
            # source check or silently allocate a replacement.
            if allocation is not None:
                await asyncio.shield(self.broker.allocations.request_cleanup(context.task.owner_id,allocation["id"]))
            if isinstance(error,LocalFileError):
                raise TaskError("sandbox_input_changed", "The exact local input changed or became unavailable.") from None
            raise
        finally:
            if isinstance(content,_MediaInput):
                content.local_file.close()
            await asyncio.shield(context.settle(reservation["id"],input_tokens=0,output_tokens=0,tool_steps=1))
