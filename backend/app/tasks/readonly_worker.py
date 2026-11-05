"""Restricted context/tools for the existing Hermes provider and task runner."""
from app.memory.tools import MemoryToolModule
from app.tasks.context import TaskContextBuilder
from app.tasks.context_repository import ContextRepository
from app.tasks.context_tools import TaskContextToolModule
from app.tasks.hermes import PreparedHermesTask
from app.tasks.memory_evidence import TaskMemoryEvidence, TaskSourceMemoryReadBackend
from app.tasks.models import TaskError
from app.tasks.readonly import CAPABILITY, readonly_source
from app.tasks.tool_bridge import TaskToolBridge
from app.tools.module import FunctionToolModule
from app.tools.registry import ToolDefinition


class ReadonlyContextRepository(ContextRepository):
    async def snapshot(self, *args, **kwargs):
        result = await super().snapshot(*args, **kwargs)
        # Raw observations retain provisional speech semantics. Do not also
        # copy evolving display captions into this immutable task snapshot.
        return result | {key: [r for r in result[key] if r.availability == "available"]
                         for key in ("recent", "pinned")}


async def prepare_readonly(context, factory):
    repository = ReadonlyContextRepository(context.tasks)
    builder = TaskContextBuilder(repository)
    images = await factory.images_factory(context, repository) if factory.images_factory else None
    modules = [TaskContextToolModule(builder, context.claim, images=images)]
    readers = ["read_task_context_archive", "read_task_context_source"]
    if images is not None:
        readers.append("read_task_context_image")
    evidence = None
    if factory.knowledge is not None:
        knowledge = factory.knowledge
        knowledge.context_for(context.task.owner_id)
        area = context.task.knowledge_area_id
        if area is not None:
            evidence = TaskMemoryEvidence(context, knowledge.sources, area_id=area, access_check=knowledge.context_for)
            backend = TaskSourceMemoryReadBackend(knowledge.sources, knowledge.retrieval,
                owner_id=context.task.owner_id, area_id=area, access_check=knowledge.context_for, task_evidence=evidence)
            modules.append(MemoryToolModule(read_backend=backend))
            readers.extend(("search_memory", "explain_memory", "list_memory_versions"))

    async def status(arguments):
        if arguments:
            return {"status": "failed", "error": {"code": "invalid_tool_input"}}
        rows = await context.tasks.list(context.task.owner_id, limit=20, active_only=True)
        return {"status": "success", "tasks": [{"task_ref": context.tasks.reference(t.id),
            "status": t.status.value, "revision": t.revision} for t in rows], "limit": 20}

    modules.append(FunctionToolModule("readonly_status", lambda _: [ToolDefinition(
        "read_background_task_status", "Read bounded owner task status, not mutation or result text.",
        {"type": "object", "properties": {}, "additionalProperties": False}, status)]))
    readers.append("read_background_task_status")
    bridge = await TaskToolBridge.from_modules(context, modules, read_tools=tuple(readers),
                                               capability_tools={}, memory_evidence=evidence)
    if bridge.actions:
        raise TaskError("readonly_actions_forbidden", "A read-only worker cannot contain action handlers.")
    manifest = await builder.build(context.claim, shared_context={"admission_profile": CAPABILITY,
        "action_authority": False, "authorized_operations": [],
        "speech_evidence": "Received provisional fragments are observations, not finalized requests. "
                           "Do not infer missing words, confirmation, completed actions or hearing.",
        "memory_available": evidence is not None, "image_resolver_configured": images is not None})
    return PreparedHermesTask(manifest, bridge)


async def is_readonly(context):
    async with context.tasks.transaction(context.task.owner_id) as connection:
        task, _ = await context.tasks.lock_claim(connection, context.claim)
        return await readonly_source(context.tasks, connection, task) is not None
