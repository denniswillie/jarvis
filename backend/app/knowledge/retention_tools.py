"""Conversational explicit source retention, separate from correction or forgetting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from app.knowledge.models import SourceError
from app.knowledge.retention import SourceRetentionService
from app.tasks.models import TaskError, TaskVoiceSource, digest
from app.tools.inputs import failed, success
from app.tools.invocation import current_tool_invocation
from app.tools.registry import ToolDefinition


class RetentionScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    area_id: UUID
    source_id: UUID


class RetentionChange(RetentionScope):
    expected_source_revision: int = Field(ge=1, strict=True)
    expected_source_generation: int = Field(ge=1, strict=True)
    expected_policy_revision: int = Field(ge=0, strict=True)
    expires_at: AwareDatetime | None = Field(
        description="Explicit timezone-aware expiry requested by the user; null cancels automatic expiry.")


def audit_retention_arguments(arguments):
    """Rejected/extra fields must not become an unlinked private-content copy."""
    result = {"input_digest": digest(arguments)}
    for key in ("area_id", "source_id"):
        try:
            result[key] = str(UUID(str(arguments.get(key, ""))))
        except ValueError:
            pass
    for key in ("expected_source_revision", "expected_source_generation", "expected_policy_revision"):
        if type(arguments.get(key)) is int:
            result[key] = arguments[key]
    # No raw date/prose or arbitrary values, even before schema validation.
    result["expiry_supplied"] = "expires_at" in arguments
    result["cancel_requested"] = "expires_at" in arguments and arguments["expires_at"] is None
    return result


def audit_retention_result(result):
    receipt = {key: result[key] for key in
        ("status", "source_id", "area_id", "source_state", "source_revision", "source_generation",
         "applied_policy_revision", "replayed", "full_erasure_verified") if key in result}
    if isinstance(result.get("error"), dict):
        receipt["error"] = {"code": result["error"].get("code", "retention_operation_failed")}
    return receipt


@dataclass(frozen=True)
class SourceRetentionToolModule:
    sources: object
    requests: object
    owner_id: str
    access_check: Callable | None = None
    name: ClassVar[str] = "source_retention"

    def authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Retention is outside this owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    def build_tools(self, context):
        self.authorize(context.owner_id)
        service = SourceRetentionService(self.sources, self.requests)

        async def inspect(args):
            result = await service.inspect(context.owner_id, args.area_id, args.source_id)
            self.authorize(context.owner_id)
            return success(result)

        async def change(args):
            invocation = current_tool_invocation()
            if invocation.task is not None or invocation.request_event_id is None:
                raise SourceError("knowledge_request_not_finalized",
                    "A current finalized user request is required to change retention.", 403)
            result = await service.set_policy(context.owner_id, args.area_id, args.source_id,
                expected_source_revision=args.expected_source_revision,
                expected_source_generation=args.expected_source_generation,
                expected_policy_revision=args.expected_policy_revision, expires_at=args.expires_at,
                request=TaskVoiceSource(invocation.request_event_id, invocation.request_revision, context.session_id),
                access_check=self.authorize)
            return success(result)

        def definition(name, description, schema, handler, *, mutates=False):
            async def invoke(arguments):
                try:
                    args = schema.model_validate(arguments)
                    self.authorize(context.owner_id)
                    return await handler(args)
                except ValidationError:
                    return failed("invalid_tool_input", "Use exact source IDs/revisions and a timezone-aware expiry or null.")
                except (SourceError, TaskError) as error:
                    return failed(error.code, error.message)
            return ToolDefinition(name=name, description=description,
                input_schema=schema.model_json_schema(), handler=invoke, mutates_state=mutates,
                audit_arguments=audit_retention_arguments, audit_result=audit_retention_result)

        result = [definition("inspect_memory_retention",
            "Read the exact source's expiry policy and suppression-tombstone retention, even after archive/deletion. "
            "Returns metadata, not original text or proof of completed provider erasure.",
            RetentionScope, inspect)]
        if context.session_id is not None:
            result.append(definition("set_memory_retention",
                "Set or cancel automatic expiry only for the exact source and date the current user requests. "
                "Inspect retention first and use all returned current revisions. Clarify ambiguous targets or dates; "
                "do not invent an expiry from age, inactivity, a stored document or an inferred obligation. "
                "Null cancels expiry without restoring deleted data. Referenced evidence is held, changed sources "
                "require renewed authorization, and shared originals are never silently included. "
                "This schedules cleanup; it does not mean the source has been deleted or an obligation cancelled. "
                "An exact retry recovers the same operation without overwriting a newer policy.",
                RetentionChange, change, mutates=True))
        return result
