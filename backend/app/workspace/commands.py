"""C0 command shapes and dispatch over existing Workspace domain services.

Core supplies authenticated/admitted context and owns tool registration, command
identity, deadlines and source-aware audit capture. This adapter has no journal,
provider, admission or execution lifecycle of its own.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.files.tools import _file_payload
from app.files.workspace_access import file_version
from app.workspace.models import StoredText, WorkspaceContext
from app.tasks.media_editing import MEDIA_TOOL, MediaInstruction


FileVersion = Annotated[StoredText, Field(min_length=1, max_length=256)]
Query = Annotated[StoredText, Field(max_length=500)]
Limit = Annotated[int, Field(strict=True, ge=1, le=8)]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FileRead(_Input):
    domain: Literal["artifact"]
    artifact_kind: Literal["file"]
    artifact_id: UUID
    version_id: FileVersion | None = None


class _FileSearch(_Input):
    domain: Literal["artifact"]
    artifact_kind: Literal["file"]
    query: Query | None = None
    limit: Limit = 8


class _OperationRead(_Input):
    domain: Literal["operation"]
    operation_id: UUID


class _FileOpen(_Input):
    action: Literal["open"]
    artifact_kind: Literal["file"]
    artifact_id: UUID
    version_id: FileVersion | None = None


class _FileDownload(_Input):
    action: Literal["download"]
    artifact_kind: Literal["file"]
    artifact_id: UUID
    version_id: FileVersion | None = None


def _adapter(models):
    return TypeAdapter(models[0] if len(models) == 1 else Union[tuple(models)])


def _wire_schema(adapter):
    """Flatten only our root command union for Realtime function parameters.

    Branch-dependent requirements remain enforced by normalize, not weakened in
    the handler. Keep nested document/patch schemas and their references intact.
    """
    schema = adapter.json_schema()
    definitions = schema.get("$defs", {})
    branches = schema.get("oneOf", schema.get("anyOf", [schema]))
    branches = [definitions[item["$ref"].removeprefix("#/$defs/")] if "$ref" in item else item
                for item in branches]
    candidates = {}
    for branch in branches:
        for name, field in branch["properties"].items():
            choices = candidates.setdefault(name, [])
            if field not in choices:
                choices.append(field)
    properties = {}
    for name, choices in candidates.items():
        if name in {"action", "domain", "artifact_kind"}:
            properties[name] = {"type": "string", "enum": list(dict.fromkeys(item["const"] for item in choices))}
        else:
            properties[name] = choices[0] if len(choices) == 1 else {"anyOf": choices}
    required = [name for name in branches[0].get("required", [])
                if all(name in item.get("required", []) for item in branches)]
    result = {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
    if definitions:
        result["$defs"] = definitions
    return result


def _json_input(value):
    """Accept JSON data only; reject coercible Python values and unsafe text."""
    if type(value) is not dict:
        raise ValueError("Command arguments must be a JSON object")
    pending = [value]
    # JSON serialization detects cycles/depth and non-finite numbers before
    # walking the tree. No argument content is retained in an audit here.
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
        encoded.encode("utf-8")
        while pending:
            item = pending.pop()
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("JSON object keys must be strings")
                pending.extend(item.keys())
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
            elif type(item) is str:
                if "\x00" in item:
                    raise ValueError("Command text cannot contain NUL")
            elif item is not None and type(item) not in (int, float, bool):
                raise ValueError("Command arguments must contain JSON values only")
    except (TypeError, RecursionError, UnicodeError) as error:
        raise ValueError("Command arguments must contain valid JSON data") from error
    return value


def _available(summary, **payload):
    return {"status": "success", "outcome": "available", "effect_committed": False,
            "summary": summary, **payload}


def _file_view(file):
    return {"artifact": {"artifact_id": str(file.id), "version_id": file_version(file),
                         "kind": "file", "title": file.filename},
            "file": _file_payload(file)}


class WorkspaceCommands:
    def __init__(self, presentations=None, files=None, *, media_editor=None):
        self.presentations, self.files = presentations, files
        self.media_editor = media_editor
        reads = [_OperationRead]
        if files is not None:
            reads += [_FileRead, _FileSearch]
        self._adapters = {"live_read": _adapter(reads)}
        if media_editor is not None and files is not None:
            self._adapters[MEDIA_TOOL] = TypeAdapter(MediaInstruction)
        if presentations is not None and files is not None:
            present = [_FileOpen]
            if presentations.exporter is not None:
                present.append(_FileDownload)
            self._adapters["live_present"] = _adapter(present)

    def schemas(self) -> dict[str, dict]:
        """Only the supplied services' backed commands/branches are advertised."""
        return {name: _wire_schema(adapter) for name, adapter in self._adapters.items()}

    def normalize(self, name, arguments) -> dict:
        adapter = self._adapters.get(name)
        if adapter is None:
            raise ValueError("Unsupported Workspace command")
        value = _json_input(arguments)
        parsed = adapter.validate_python(value)
        return parsed.model_dump(mode="json", exclude_none=True)

    async def execute(self, ctx: WorkspaceContext, name, normalized, *, command_admission=None) -> dict:
        if not isinstance(ctx, WorkspaceContext):
            raise TypeError("Workspace commands require trusted WorkspaceContext")
        arguments = self.normalize(name, normalized)
        if name == MEDIA_TOOL:
            return await self.media_editor.submit(ctx, arguments, command_admission=command_admission)
        if name == "live_present":
            arguments["artifact_id"] = UUID(arguments["artifact_id"])
            return await self.presentations.present(ctx, **arguments)
        if arguments["domain"] == "operation":
            operation_id = UUID(arguments["operation_id"])
            result = None
            if result is None and self.media_editor is not None:
                result = await self.receipt(ctx, MEDIA_TOOL, operation_id)
            if result is None and self.presentations is not None:
                result = await self.receipt(ctx, "live_present", operation_id)
            if result is not None:
                return _available("The saved Workspace operation outcome is available.",
                                  operation_id=str(operation_id), operation=result)
            return {"status": "success", "outcome": "outcome_unknown", "effect_committed": False,
                    "operation_id": str(operation_id), "summary": "No authoritative Workspace operation outcome is available.",
                    "error": {"code": "workspace_operation_unknown", "retryable": False}}
        artifact_id = arguments.get("artifact_id")
        if artifact_id is not None:
            artifact_id = UUID(artifact_id)
            file = await self.files.get(ctx.owner_id, ctx.session_id, artifact_id, arguments.get("version_id"))
            return _available("Retained file metadata is available.", **_file_view(file))
        repository = await self.files.scoped(ctx.owner_id, ctx.session_id)
        files = await repository.list_files(owner_id=ctx.owner_id, query=arguments.get("query"), limit=arguments["limit"])
        return _available("Retained file search results are available.",
                          items=[_file_view(file) for file in files], page_full=len(files) == arguments["limit"])

    async def receipt(self, ctx: WorkspaceContext, name, operation_id) -> dict | None:
        """Read an owning receipt only; never retry an effect or cached content."""
        if not isinstance(ctx, WorkspaceContext):
            raise TypeError("Workspace commands require trusted WorkspaceContext")
        if name not in {"live_present", MEDIA_TOOL}:
            return None
        operation_id = TypeAdapter(UUID).validate_python(operation_id)
        if name == MEDIA_TOOL:
            return await self.media_editor.receipt(ctx, operation_id) if self.media_editor is not None else None
        if self.presentations is not None:
            return await self.presentations.operation_receipt(ctx, operation_id)
        return None
