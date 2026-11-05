"""Task-scoped shared tools. Hermes owns reasoning, Jarvis owns effect authority."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.tasks.models import ClaimLost, TaskError, digest, json_object, json_text
from app.tasks.runner import WorkerContext
from app.tools.inputs import failed
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.tools.registry import JarvisToolRegistry


@dataclass(frozen=True)
class GrantedToolAction:
    grant_id: UUID
    capability: str
    tool_name: str
    parameters: dict
    request_event_id: UUID
    request_revision: int

    def __post_init__(self):
        if not isinstance(self.grant_id, UUID) or not isinstance(self.request_event_id, UUID):
            raise ValueError("Grant and source request identities must be UUIDs")
        if type(self.request_revision) is not int or self.request_revision < 1:
            raise ValueError("Grant request revision must be a positive integer")

    @property
    def reference(self):
        return "op_" + self.grant_id.hex


class TaskToolBridge:
    """Created by the trusted worker using its exact owner/session registry.

    Mutating tools accept only an opaque reference to an exact task-attached
    grant. Arbitrary proposed mutations require new authorization upstream.
    Read tools keep their production schemas. Returned evidence is not copied
    into another transcript store; original-image transport remains explicit.
    """

    def __init__(self, context: WorkerContext, registry: JarvisToolRegistry, *,
                 read_tools: tuple[str, ...], actions: tuple[GrantedToolAction, ...],
                 max_result_characters=128000, memory_evidence=None, receipt_revalidator=None):
        if type(max_result_characters) is not int or not 2000 <= max_result_characters <= 1000000:
            raise ValueError("Invalid task tool result bound")
        if len(read_tools) != len(set(read_tools)):
            raise ValueError("Duplicate task read tool")
        self.context, self.registry = context, registry
        self.memory_evidence = memory_evidence
        self.receipt_revalidator = receipt_revalidator
        self.managed_plugins = None
        self.read_tools = frozenset(read_tools)
        self.actions = {action.reference: deepcopy(action) for action in actions}
        if len(self.actions) != len(actions):
            raise ValueError("Duplicate task action reference")
        for name in self.read_tools:
            tool = registry.get(name)
            if tool is None or tool.mutates_state:
                raise ValueError("A read-only allowlist cannot include mutating or missing tools")
        for action in actions:
            tool = registry.get(action.tool_name)
            if tool is None or not tool.mutates_state:
                raise ValueError("An action requires a registered mutating tool")
            json_object(action.parameters)
        self.max_result_characters = max_result_characters

    def bind_managed_plugins(self, runtime):
        """Trusted worker composition only; request JSON never supplies a pin."""
        from app.plugins.managed import ManagedFilesDataRuntime
        if not isinstance(runtime, ManagedFilesDataRuntime):
            raise TypeError("Managed publication requires its trusted runtime")
        if self.managed_plugins is not None and self.managed_plugins is not runtime:
            raise ValueError("A task bridge cannot replace its managed runtime")
        self.managed_plugins = runtime

    async def _managed_publication_pin(self, action):
        if self.managed_plugins is None:
            return None
        tasks, claim = self.context.tasks, self.context.claim
        async with tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await tasks.lock_claim(connection, claim)
            return await self.managed_plugins.require_publication(
                connection, tasks, task, action.grant_id, action.parameters)

    @classmethod
    async def from_modules(cls, context, modules, *, settings=None, **options):
        registry = create_tool_registry(
            modules=modules, context=ToolBuildContext(
                settings=settings, owner_id=context.task.owner_id, session_id=context.task.session_id,
            ),
        )
        return await cls.create(context, registry, **options)

    @classmethod
    async def create(cls, context, registry, *, read_tools=(), capability_tools=None, **bounds):
        """Resolve grants from the ledger, not model-supplied IDs/parameters."""
        actions = []
        bindings = dict(capability_tools or {})
        tasks, claim = context.tasks, context.claim
        async with tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await tasks.lock_claim(connection, claim)
            if await tasks.context_problem(connection, task):
                raise ClaimLost()
            grants = await connection.fetch(
                "select grant_id from jarvis_task_grants where owner_id=$1 and task_id=$2 order by grant_id",
                task.owner_id, task.id,
            )
            for row in grants:
                grant = await tasks.validate_grant(connection, task.owner_id, row["grant_id"])
                name = bindings.get(grant["capability"])
                if name is not None:
                    actions.append(GrantedToolAction(
                        grant["id"], grant["capability"], name, json_object(grant["scope"]),
                        grant["request_event_id"], grant["request_revision"],
                    ))
        return cls(context, registry, read_tools=tuple(read_tools), actions=tuple(actions), **bounds)

    def catalog(self):
        schemas = []
        for tool in self.registry.list():
            refs = sorted(ref for ref, action in self.actions.items() if action.tool_name == tool.name)
            if tool.name in self.read_tools:
                parameters, description = deepcopy(tool.input_schema), tool.description
            elif refs:
                parameters = {
                    "type": "object", "additionalProperties": False, "required": ["operation_ref"],
                    "properties": {"operation_ref": {"type": "string", "enum": refs}},
                }
                description = (
                    tool.description + " Execute only the exact previously authorized operation selected "
                    "by operation_ref. No new parameters are accepted. A repeated reference reuses its "
                    "durable receipt; a new provider call ID never authorizes another write."
                )
            else:
                continue
            schemas.append({"name": tool.name, "description": description, "parameters": parameters})
        return schemas

    def authorized_operations(self):
        return [{
            "operation_ref": action.reference, "tool": action.tool_name,
            "parameters": deepcopy(action.parameters),
        } for action in self.actions.values()]

    async def _check(self):
        tasks, claim = self.context.tasks, self.context.claim
        async with tasks.transaction(claim.task.owner_id) as connection:
            task, _ = await tasks.lock_claim(connection, claim)
            if await tasks.context_problem(connection, task):
                raise ClaimLost()
            for action in self.actions.values():
                if not await connection.fetchval(
                    "select exists(select 1 from jarvis_task_grants where owner_id=$1 and task_id=$2 and grant_id=$3)",
                    task.owner_id, task.id, action.grant_id,
                ):
                    raise TaskError("task_tool_grant_not_attached", "This operation is not granted to this task.", 403)

    def _scope(self, *, action_id=None, operation_key=None):
        claim = self.context.claim
        return TaskToolInvocation(claim.task.id, claim.task_revision, claim.attempt_id, action_id, operation_key)

    async def _handler(self, tool, arguments, provider_call_id, *, action=None, action_id=None, operation_key=None):
        if action is not None:
            # Also covers the publisher's existing uncertain-outcome recovery
            # path, which deliberately does not submit a second action.
            await self._managed_publication_pin(action)
        task = self.context.task
        with bind_tool_invocation(
            provider_call_id=provider_call_id,
            request_event_id=action.request_event_id if action is not None else task.request_event_id,
            request_revision=action.request_revision if action is not None else task.request_revision,
            task=self._scope(action_id=action_id, operation_key=operation_key),
        ):
            result = json_object(await tool.handler(deepcopy(arguments)))
        if len(json_text(result)) > self.max_result_characters:
            raise TaskError("task_tool_result_too_large", "The tool result exceeds the configured bound.")
        return result

    async def invoke(self, name, arguments, *, provider_call_id):
        if (not isinstance(provider_call_id, str) or not 1 <= len(provider_call_id) <= 200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded native tool-call identity is required")
        arguments = json_object(arguments)
        if len(json_text(arguments)) > 64000:
            return failed("invalid_tool_input", "Tool arguments exceed the task transport bound.")
        await self._check()
        tool = self.registry.get(name)
        if tool is None or (name not in self.read_tools and not any(
            action.tool_name == name for action in self.actions.values()
        )):
            return failed("task_tool_not_allowed", "This tool is not granted to the current task.")
        if not tool.mutates_state:
            reservation = await self.context.reserve("tool-read:" + uuid4().hex, tool_steps=1)
            await self.context.budgets.submit(self.context.claim, reservation["id"])
            try:
                result = await self._handler(tool, arguments, provider_call_id)
            except ClaimLost:
                raise
            except Exception:
                result = failed("task_tool_failed", "The read tool failed; do not infer that its data is absent.")
            finally:
                await asyncio.shield(self.context.settle(
                    reservation["id"], input_tokens=0, output_tokens=0, tool_steps=1,
                ))
            await self._check()
            return result

        reference = arguments.get("operation_ref")
        action = self.actions.get(reference) if isinstance(reference, str) else None
        if set(arguments) != {"operation_ref"} or action is None or action.tool_name != name:
            return failed("task_operation_not_allowed", "Use this tool's exact supplied operation_ref without new parameters.")
        return await self.execute_granted(action, tool, provider_call_id=provider_call_id)

    async def execute_granted(self, action, tool, *, provider_call_id):
        """Trusted producers share the exact same effect boundary as ordinary tools."""
        if (self.actions.get(action.reference) != action or action.tool_name != tool.name
                or not tool.mutates_state):
            raise ValueError("Execute only an attached exact action with its trusted mutating handler")
        if (not isinstance(provider_call_id, str) or not 1 <= len(provider_call_id) <= 200
                or not provider_call_id.strip() or "\x00" in provider_call_id):
            raise ValueError("A bounded native tool-call identity is required")
        await self._check()
        operation_key = "tool-grant:" + action.grant_id.hex
        managed_pin = await self._managed_publication_pin(action)
        record = await self.context.actions.prepare(
            self.context.claim, operation_key, action.grant_id, action.capability, action.parameters,
        )
        if record["status"] not in {"prepared", "authorized"}:
            return await self._replay(action, record)
        try:
            await self.context.actions.authorize(self.context.claim, record["id"])
            options = ({"managed_plugins": self.managed_plugins, "managed_pin": managed_pin}
                       if managed_pin is not None else {})
            token = await self.context.actions.begin_submission(self.context.claim, record["id"], **options)
        except ClaimLost:
            raise
        except TaskError:
            # Another admitted invocation may have crossed the boundary.
            current = await self.context.actions.get(self.context.task.owner_id, self.context.task.id, record["id"])
            if current["status"] in {"prepared", "authorized"}:
                raise
            return await self._replay(action, current)
        try:
            result = await self._handler(
                tool, action.parameters, provider_call_id, action=action,
                action_id=record["id"], operation_key=operation_key,
            )
        except BaseException:
            await asyncio.shield(self._record(action, record["id"], token, "outcome_unknown", {
                "status": "failed", "error": {"code": "task_effect_outcome_unknown",
                                             "message": "The action may have run; it must not be repeated automatically."},
            }))
            raise
        status = "accepted" if result.get("status") == "success" else "outcome_unknown"
        await self._record(action, record["id"], token, status, result)
        await self._check()
        return result | {"task_operation": self._operation(action, record["id"], status, replayed=False)}

    async def _record(self, action, action_id, token, status, result):
        await self.context.actions.record_receipt(
            self.context.task.owner_id, self.context.task.id, action_id, token, status,
            {"tool": action.tool_name, "parameters_digest": digest(action.parameters), "result": result},
        )

    @staticmethod
    def _operation(action, action_id, status, *, replayed):
        return {"operation_ref": action.reference, "action_id": str(action_id), "state": status, "replayed": replayed}

    async def _replay(self, action, record):
        await self._check()
        await self._managed_publication_pin(action)
        status = record["status"]
        metadata = self._operation(action, record["id"], status, replayed=True)
        if status in {"submitting", "outcome_unknown"}:
            return {"status": "outcome_unknown", "task_operation": metadata,
                    "meaning": "This operation may already have run. No new submission was made."}
        tasks, task = self.context.tasks, self.context.task
        async with tasks.transaction(task.owner_id) as connection:
            current, _ = await tasks.lock_claim(connection, self.context.claim)
            if await tasks.context_problem(connection, current):
                raise ClaimLost()
            row = await connection.fetchrow(
                "select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2 for share",
                task.owner_id, record["receipt_payload_id"],
            )
            if row is None or row["availability"] != "available":
                raise TaskError("task_operation_receipt_unavailable", "The recorded action receipt is unavailable.")
            receipt = json_object(row["content"])
        await self._check()
        if status in {"accepted", "rejected"}:
            if receipt.get("tool") != action.tool_name or receipt.get("parameters_digest") != digest(action.parameters):
                raise TaskError("task_operation_receipt_mismatch", "The receipt does not match the authorized operation.")
            result = json_object(receipt["result"])
            if self.receipt_revalidator is not None:
                result = await self.receipt_revalidator(action,result)
            return result | {"task_operation": metadata}
        if status == "reconciled":
            outcome = record.get("reconciled_outcome")
            if outcome not in {"accepted", "rejected", "not_submitted"}:
                raise TaskError("task_operation_reconciliation_invalid", "The action has no valid reconciliation outcome.")
            return {
                "status": "success" if outcome == "accepted" else "failed",
                "task_operation": metadata | {"reconciled_outcome": outcome},
                "reconciliation": receipt,
                "meaning": "Trusted reconciliation, not a replay of missing original tool output. No new submission was made.",
            }
        raise TaskError("task_operation_not_executable", "This action's recorded state does not permit submission.")

    async def verified_operation(self, reference):
        """Read an accepted exact operation without preparing or repeating a write."""
        await self._check()
        action = self.actions.get(reference)
        if action is None:
            raise TaskError("task_result_operation_not_allowed", "The result claims an ungranted operation.")
        async with self.context.tasks.transaction(self.context.task.owner_id) as connection:
            record = await connection.fetchrow(
                "select * from jarvis_task_actions where owner_id=$1 and task_id=$2 and operation_key=$3",
                self.context.task.owner_id, self.context.task.id, "tool-grant:" + action.grant_id.hex,
            )
        if record is None or not (record["status"] == "accepted" or (
            record["status"] == "reconciled" and record["reconciled_outcome"] == "accepted"
        )):
            raise TaskError("task_result_operation_unconfirmed", "The claimed operation has no accepted receipt.")
        return await self._replay(action, dict(record))

    async def verified_memory(self, citation):
        await self._check()
        if self.memory_evidence is None:
            raise TaskError("task_memory_not_configured", "This task has no selected memory evidence reader.")
        await self.memory_evidence.verify(citation)
        await self._check()

    async def resolve_image(self, name, result):
        await self._check()
        tool = self.registry.get(name)
        if name not in self.read_tools or tool is None or tool.image_resolver is None:
            raise TaskError("task_image_not_allowed", "This task tool has no authorized image resolver.", 403)
        image = await tool.image_resolver(result)
        await self._check()
        return image
