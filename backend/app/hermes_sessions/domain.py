"""Retained Hermes access to existing Jarvis tools, without a temporary task.

The caller and input scope come from trusted native/admission adapters, never
model arguments. Native Hermes owns conversation history. Jarvis's existing
executor and domain services still own tool receipts and source policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.hermes_sessions.bindings import BindingError, bounded_id, identity
from app.hermes_sessions.domain_capture import NativeToolCapture
from app.hermes_sessions.domain_scope import DomainInputScope
from app.images import ResolvedImage
from app.tasks.models import json_object
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


@dataclass(frozen=True)
class DomainToolOutcome:
    payload: dict[str, Any]
    # Transient bytes and the original revalidation callback stay out of JSON
    # receipts. A native transport must revalidate again before sending pixels.
    image: ResolvedImage | None = None


def _failure(code, message):
    return DomainToolOutcome({"status": "failed", "error": {"code": code, "message": message}})


class RetainedDomainAdapter:
    """Existing domain tools with explicitly qualified, request-bound writes.

    scope_for_input(binding, input_id) must resolve an admission's stored scope.
    It must not use the latest voice turn or accept owner/session IDs from JSON.
    modules_for_scope(scope) constructs existing independently owned modules.
    Neither callback starts a model, performs eager memory recall, or makes a task.
    """

    def __init__(self, service, *, sessions, contexts, executions,
                 scope_for_input, modules_for_scope, settings=None,
                 qualified_write_tools: frozenset[str] = frozenset(),
                 retained_write_tools: frozenset[str] = frozenset()):
        pool = contexts.pool
        if (pool is None or getattr(sessions, "_pool", None) is not pool
                or getattr(executions, "_pool", None) is not pool):
            raise ValueError("Session, request and tool receipts must share their database")
        self.service, self.sessions = service, sessions
        self.contexts, self.executions = contexts, executions
        self.scope_for_input, self.modules_for_scope = scope_for_input, modules_for_scope
        self.settings = settings
        if not isinstance(qualified_write_tools, frozenset) or len(qualified_write_tools) > 64:
            raise ValueError("Qualified writes must be a bounded immutable tool set")
        for name in qualified_write_tools:
            bounded_id(name)
        # Trusted composition qualifies effect-time request checks in the domain
        # service. This is not model-supplied permission or a blanket write switch.
        self.qualified_write_tools = qualified_write_tools
        if (not isinstance(retained_write_tools, frozenset)
                or not retained_write_tools.issubset(qualified_write_tools)):
            raise ValueError("Retained writes must be explicitly qualified tools")
        # These services authorize from the retained owner/session at effect
        # time. A callback/completion is not fabricated into a user request.
        self.retained_write_tools = retained_write_tools

    async def _prepare(self, native_session_id, input_id):
        bounded_id(native_session_id)
        bounded_id(input_id)
        binding = await self.service.caller(native_session_id)
        scope = await self.scope_for_input(binding, input_id)
        if (not isinstance(scope, DomainInputScope)
                or scope.owner_id != self.service.bindings.owner_id
                or scope.logical_session_id != binding.logical_id
                or scope.input_id != input_id):
            raise BindingError("domain_input_scope_mismatch")
        session = await self.sessions.get_session(scope.owner_id, scope.jarvis_session_id)
        # The repository scopes this lookup by owner. JarvisSession deliberately
        # has no owner_id field; do not depend on fields invented by test doubles.
        if session is None or session.id != scope.jarvis_session_id:
            raise BindingError("domain_session_unavailable")
        modules = await self.modules_for_scope(scope)
        registry = create_tool_registry(modules=modules, context=ToolBuildContext(
            settings=self.settings, owner_id=scope.owner_id, session_id=scope.jarvis_session_id))
        if not scope.allowed_tools.issubset({tool.name for tool in registry.list()}):
            raise BindingError("domain_capability_unavailable")
        return scope, registry

    async def catalog(self, *, native_session_id, input_id):
        scope, registry = await self._prepare(native_session_id, input_id)
        return [{"name": tool.name, "description": tool.description, "parameters": tool.input_schema}
                for tool in registry.list()
                if tool.name in scope.allowed_tools and (not tool.mutates_state or (
                    tool.name in self.qualified_write_tools and (
                        scope.request_event_id is not None or tool.name in self.retained_write_tools)))]

    async def invoke(self, *, native_session_id, input_id, call_id, name, arguments):
        """Only name/arguments are model-supplied. No automatic execution retry."""
        bounded_id(call_id)
        bounded_id(name)
        if not isinstance(arguments, dict):
            return _failure("invalid_tool_input", "Tool arguments must be an object.")
        arguments = json_object(arguments)
        scope, registry = await self._prepare(native_session_id, input_id)
        tool = registry.get(name)
        if tool is None or name not in scope.allowed_tools:
            return _failure("domain_tool_not_allowed", "This input cannot use that tool.")
        if tool.mutates_state and name not in self.qualified_write_tools:
            return _failure("domain_write_unavailable",
                            "Retained-session write authorization is not connected; no operation ran.")
        if (tool.mutates_state and scope.request_event_id is None
                and name not in self.retained_write_tools):
            return _failure("domain_write_request_required",
                            "This input has no original user-request authority; no operation ran.")
        provider_call_id = identity(scope.logical_session_id, input_id, call_id)
        capture = NativeToolCapture(self.contexts, scope=scope, call_id=provider_call_id)
        # Check a supplied request even when the executor can replay a receipt.
        # This makes corrections/deletion visible without fetching whole history.
        await capture.prepare()
        executor = JarvisToolExecutor(registry, execution_repository=self.executions,
            owner_id=scope.owner_id, session_id=scope.jarvis_session_id,
            context_capture=capture, require_source_tracking=True)
        payload = await executor.execute(name, arguments, provider_call_id=provider_call_id)
        try:
            await capture.flush()
        except Exception:
            return _failure("domain_result_capture_unavailable",
                            "The tool result could not be recorded with its evidence; do not invent it.")
        image = None
        if payload.get("status") == "success" and tool.image_resolver is not None:
            image = await tool.image_resolver(payload)
            if not isinstance(image, ResolvedImage):
                raise TypeError("Image tool did not resolve original pixels")
            if image.revalidate is not None:
                await image.revalidate()
        return DomainToolOutcome(payload, image)
