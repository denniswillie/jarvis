from __future__ import annotations

import logging
import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from app.assistant.tool_limits import ToolCallBudget, ToolCallLimitError
from app.experience.evidence import durable_tool_payload
from app.experience.repository import ToolExecutionRepository
from app.experience.service import ExperienceEventService
from app.tools.invocation import CanonicalRequestIdentity, bind_tool_invocation
from app.tools.module import ToolBuildContext, ToolModule
from app.tools.registry import JarvisToolRegistry, create_tool_registry
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextRecord
from app.tools.provenance import SourceBoundToolResult, ToolResultCapture, restore_source_result


logger = logging.getLogger(__name__)


class JarvisToolExecutor:
    def __init__(
        self,
        registry: JarvisToolRegistry,
        *,
        execution_repository: ToolExecutionRepository | None = None,
        experience_event_service: ExperienceEventService | None = None,
        owner_id: str | None = None,
        session_id: UUID | None = None,
        context_capture: SessionContextCapture | None = None,
        require_source_tracking: bool = False,
        defer_experience_attachment: bool = False,
        on_request_evidence: Callable[[str | None, ContextRecord | None], None] | None = None,
    ) -> None:
        self._registry = registry
        self._execution_repository = execution_repository
        self._experience_event_service = experience_event_service
        self._owner_id = owner_id
        self._session_id = session_id
        self._context_capture = context_capture
        self._require_source_tracking = require_source_tracking
        self._defer_experience_attachment = defer_experience_attachment
        self._on_request_evidence = on_request_evidence

    @classmethod
    def from_modules(
        cls,
        *,
        modules: Sequence[ToolModule],
        context: ToolBuildContext,
    ) -> JarvisToolExecutor:
        return cls(create_tool_registry(modules=modules, context=context))

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        budget: ToolCallBudget | None = None,
        provider_call_id: str | None = None,
    ) -> dict[str, Any]:
        result = await self._execute(name, arguments, budget=budget, provider_call_id=provider_call_id)
        definition = self._registry.get(name)
        if (self._context_capture is not None and not isinstance(result, SourceBoundToolResult)
                and not (definition is not None and getattr(definition, "local_snapshot_read", False) is True)):
            audit = definition.audit_result(result) if definition is not None and getattr(definition, "audit_result", None) else result
            self._context_capture.tool_outcome(provider_call_id, name, audit,
                mutates_state=bool(definition is not None and definition.mutates_state))
        return result

    async def _execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        budget: ToolCallBudget | None = None,
        provider_call_id: str | None = None,
    ) -> dict[str, Any]:
        tool = self._registry.get(name)
        if tool is None:
            return {
                "status": "failed",
                "error": {
                    "code": "tool_not_found",
                    "message": f"Tool is not registered: {name}",
                },
            }
        if budget is not None:
            try:
                budget.record(tool_name=name, arguments=arguments)
            except ToolCallLimitError as error:
                logger.warning("Jarvis tool blocked name=%s code=%s", name, error.code)
                return {
                    "status": "failed",
                    "error": {"code": error.code, "message": str(error)},
                }
        if (
            getattr(tool, "local_snapshot_read", False) is not True
            and provider_call_id is not None
            and self._execution_repository is not None
            and self._owner_id is not None
            and self._session_id is not None
        ):
            return await self._execute_durable(
                tool=tool,
                name=name,
                arguments=arguments,
                provider_call_id=provider_call_id,
            )
        return await self._invoke(
            tool=tool,
            name=name,
            arguments=arguments,
            provider_call_id=provider_call_id,
        )

    async def _execute_durable(
        self,
        *,
        tool: Any,
        name: str,
        arguments: dict[str, Any],
        provider_call_id: str,
    ) -> dict[str, Any]:
        assert self._execution_repository is not None
        assert self._owner_id is not None
        assert self._session_id is not None
        capture_result = getattr(tool, "result_capture", None)

        async def execute_and_persist() -> dict[str, Any]:
            try:
                execution, created = (
                    await self._execution_repository.start_execution_with_status(
                        owner_id=self._owner_id,
                        session_id=self._session_id,
                        provider_call_id=provider_call_id,
                        tool_name=name,
                        arguments=tool.audit_arguments(arguments) if getattr(tool, "audit_arguments", None) else arguments,
                        mutates_state=tool.mutates_state,
                        started_at=datetime.now(UTC),
                    )
                )
            except Exception:
                logger.exception("Jarvis tool execution could not be persisted name=%s", name)
                return {
                    "status": "failed",
                    "error": {
                        "code": "tool_execution_persistence_failed",
                        "message": "Tool execution could not be recorded safely.",
                    },
                }

            if execution.status == "completed":
                if capture_result is not None:
                    try:
                        async with asyncio.timeout(self._context_capture.write_timeout if self._context_capture else 5):
                            return await restore_source_result(self._context_capture, owner_id=self._owner_id,
                                session_id=self._session_id, call_id=provider_call_id, name=name, saved=execution.result or {})
                    except Exception:
                        return {"status":"failed", "error":{"code":"tool_result_source_unavailable",
                            "message":"The saved result's original evidence is unavailable. Do not reconstruct it."}}
                return dict(execution.result or {})
            if execution.status == "failed":
                return {"status": "failed", "error": dict(execution.error or {})}
            if not created:
                return {
                    "status": "failed",
                    "error": {
                        "code": "tool_execution_in_progress",
                        "message": "This tool call is already being processed.",
                    },
                }

            result = await self._invoke(
                tool=tool,
                name=name,
                arguments=arguments,
                provider_call_id=provider_call_id,
            )
            completed_at = datetime.now(UTC)
            safe_outcome = durable_tool_payload(
                result.audit_receipt() if isinstance(result, SourceBoundToolResult)
                else tool.audit_result(result) if getattr(tool, "audit_result", None) else result,
                status=str(result.get("status", "failed")),
            )
            try:
                if result.get("status") == "success":
                    execution = await self._execution_repository.finalize_execution(
                        owner_id=self._owner_id,
                        session_id=self._session_id,
                        provider_call_id=provider_call_id,
                        status="completed",
                        completed_at=completed_at,
                        result=safe_outcome,
                    )
                else:
                    raw_error = safe_outcome.get("error")
                    error = (
                        dict(raw_error)
                        if isinstance(raw_error, dict)
                        else {
                            "code": "tool_execution_failed",
                            "message": f"Tool execution failed: {name}",
                        }
                    )
                    execution = await self._execution_repository.finalize_execution(
                        owner_id=self._owner_id,
                        session_id=self._session_id,
                        provider_call_id=provider_call_id,
                        status="failed",
                        completed_at=completed_at,
                        error=error,
                    )
            except Exception:
                logger.exception("Jarvis terminal tool execution could not be recorded name=%s", name)
                return {
                    "status": "failed",
                    "error": {
                        "code": "tool_execution_persistence_failed",
                        "message": "The tool ran, but its outcome could not be recorded safely.",
                    },
                }
            if self._experience_event_service is not None and capture_result is None:
                try:
                    if self._defer_experience_attachment:
                        # The terminal execution is already durable. The voice
                        # caller need not wait for optional experience grouping;
                        # the service owns and drains that bounded work.
                        await self._experience_event_service.defer_tool_execution(execution)
                    else:
                        await self._experience_event_service.attach_tool_execution(execution)
                except Exception:
                    logger.exception(
                        "Jarvis tool execution could not be attached to experience event name=%s",
                        name,
                    )
            return result

        if self._experience_event_service is None or capture_result is not None:
            return await execute_and_persist()
        async with self._experience_event_service.activity(
            owner_id=self._owner_id,
            session_id=self._session_id,
        ):
            return await execute_and_persist()

    async def _invoke(
        self,
        *,
        tool: Any,
        name: str,
        arguments: dict[str, Any],
        provider_call_id: str | None,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        capture_result = getattr(tool, "result_capture", None)
        local_snapshot_read = getattr(tool, "local_snapshot_read", False) is True
        result = None
        try:
            if (capture_result is not None and self._context_capture is None
                    and (self._require_source_tracking or self._execution_repository is not None)):
                raise ValueError("Source-backed reads require canonical result capture")
            request = (await self._context_capture.request_for_call(provider_call_id)
                       if self._context_capture is not None and not local_snapshot_read else None)
            if self._on_request_evidence is not None and not local_snapshot_read:
                # Reuse this invocation's resolved evidence. The observer is
                # synchronous: no second lookup, background work or tool replay.
                self._on_request_evidence(provider_call_id, request)
            with bind_tool_invocation(provider_call_id=provider_call_id,
                    request_event_id=request.id if request is not None else None,
                    request_revision=request.revision if request is not None else None,
                    request_identity=CanonicalRequestIdentity.from_record(request)):
                result = await tool.handler(arguments)
                if capture_result is not None and result.get("status") == "success":
                    if self._context_capture is not None:
                        async with asyncio.timeout(self._context_capture.write_timeout):
                            result = await capture_result(result, ToolResultCapture(
                                self._context_capture, provider_call_id, name))
                        if not isinstance(result, SourceBoundToolResult):
                            raise TypeError("Source reader did not return bound evidence")
        except Exception:
            local_success = (getattr(tool, "capture_after_commit", False)
                             and isinstance(result, dict) and result.get("status") == "success")
            logger.exception(
                "Jarvis tool failed name=%s duration_ms=%.1f",
                name,
                (time.perf_counter() - started_at) * 1000,
            )
            result = {
                "status": "failed",
                "error": {
                    "code": "tool_receipt_unavailable_after_commit" if local_success else "tool_execution_failed",
                    "message": (
                        "The operation reported local success, but its source-backed receipt is unavailable. "
                        "It may already be stored; inspect source state rather than assuming the write failed."
                        if local_success else f"Tool execution failed: {name}"
                    ),
                },
            }
            if local_success:
                result["error"]["local_effect_reported"] = True
            if local_snapshot_read:
                result.update(outcome="failed", effect_committed=False)
        else:
            logger.info(
                "Jarvis tool completed name=%s success=%s duration_ms=%.1f",
                name,
                result.get("status") == "success",
                (time.perf_counter() - started_at) * 1000,
            )
        return result
