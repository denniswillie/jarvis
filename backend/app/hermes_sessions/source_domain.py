"""Compose retained-session source tools without a memory/provider startup dependency."""
from dataclasses import dataclass
from uuid import UUID

from app.experience.repository import ToolExecutionRepository
from app.hermes_sessions.bindings import BindingError
from app.hermes_sessions.domain import RetainedDomainAdapter
from app.hermes_sessions.source_files import RetainedFileTools
from app.hermes_sessions.source_images import RetainedSandboxImages
from app.knowledge.files import SourceFileRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.sessions.repository import SessionRepository
from app.tasks.context_repository import ContextRepository
from app.tools.module import FunctionToolModule
from app.tools.module import ToolBuildContext
from app.hermes_sessions.domain_scope import DomainInputScope
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class RetainedSourceDomain:
    adapter: RetainedDomainAdapter
    files: SourceFileRepository
    file_tools: RetainedFileTools
    allowed_tools: frozenset[str]
    catalog: list[dict]


def compose_source_domain(service, *, scope_for_input, sources, requests,
                          storage, area_id, selections=None, access_check=None,
                          settings=None, files=None, sandbox_service=None,
                          read_live_state=None, memory_reader=None, enable_legacy_memory=False,
                          additional_modules=(), additional_tools=frozenset(),
                          additional_retained_writes=frozenset()) -> RetainedSourceDomain:
    """Reuse initialized source services; no eager recall, task or provider call.

    Additional domain modules and writes are explicitly qualified by their owner.
    Legacy memory is opt-in and is not needed by Slice 1.
    read_live_state(scope, arguments) is a supplied async, owner-scoped callback.
    """
    owner = service.bindings.owner_id
    if sources.pool is not requests.pool or not isinstance(area_id, UUID):
        raise ValueError("Retained source tools require one database and an exact area")
    if settings is not None and settings.JARVIS_OWNER_ID != owner:
        raise ValueError("Retained tool settings belong to another owner")
    if selections is not None and getattr(selections, "sources", None) is not sources:
        raise ValueError("Retained selections must share the owned source repository")
    if not additional_retained_writes.issubset(additional_tools):
        raise ValueError("Additional writes must be explicit domain tools")
    files = files if files is not None else SourceFileRepository(sources, storage,
        owner_id=owner, area_id=area_id, selections=selections, access_check=access_check)
    if (files.sources is not sources or files.owner_id != owner or files.area_id != area_id
            or files.selections is not selections or files.storage is not storage
            or files.task_evidence is not None or files.task_recovery is not None
            or files.request_authority is not None):
        raise ValueError("Retained files must share the owned source scope, without foreground/task authority")
    file_tools = RetainedFileTools(service=service, scope_for_input=scope_for_input,
                                  files=files, sandbox_service=sandbox_service)
    legacy = ()
    allowed = file_tools.allowed_tools | additional_tools
    if sandbox_service is not None:
        allowed |= {"sandbox_view_image"}
    qualified = file_tools.write_tools | additional_retained_writes
    if enable_legacy_memory:
        from app.knowledge.conversation_writes import ConversationMemoryWriteBackend
        from app.knowledge.memory_backend import SourceMemoryReadBackend
        from app.memory.tools import MemoryToolModule
        if (not isinstance(memory_reader, SourceMemoryReadBackend) or memory_reader.sources is not sources
                or memory_reader.owner_id != owner or memory_reader.area_id != area_id
                or memory_reader.selections is not selections or memory_reader.session_id is not None):
            raise ValueError("Retained memory must use the same owner/source/area selection, without a fixed session")
        writer = ConversationMemoryWriteBackend(sources, requests, owner_id=owner,
            area_id=area_id, selections=selections, access_check=access_check)
        legacy = (MemoryToolModule(read_backend=memory_reader, write_backend=writer),)
        qualified |= {"remember"}
    if read_live_state is not None:
        if not callable(read_live_state):
            raise ValueError("Live State requires an async scoped reader")
        allowed |= {"read_live_state"}

    def scoped_modules(scope):
        if scope.owner_id != owner:
            raise BindingError("domain_input_scope_mismatch")
        modules = legacy + (file_tools.module(scope),) + tuple(additional_modules)
        if sandbox_service is not None:
            images = RetainedSandboxImages(scope, sandbox_service, file_tools.check)
            modules += (FunctionToolModule("retained_images", lambda context: [images.definition()]),)
        if read_live_state is not None:
            async def read(arguments):
                return await read_live_state(scope, arguments)
            modules += (FunctionToolModule("retained_live_state", lambda context: [ToolDefinition(
                "read_live_state", "Read relevant current Jarvis state (human surroundings, live device location, reminders, sandboxes, plugins) when needed; no automatic context loading.",
                {"type": "object", "properties": {"query": {"type": "string"}},
                 "additionalProperties": False}, read, local_snapshot_read=True)]),)
        return modules

    async def modules_for_scope(scope):
        if access_check is not None:
            access_check(scope.owner_id)
        return scoped_modules(scope)

    adapter = RetainedDomainAdapter(service, sessions=SessionRepository(sources.pool),
        contexts=ContextRepository(requests, publication=CanonicalSourcePublication(sources, area_id, selections)),
        executions=ToolExecutionRepository(sources.pool),
        scope_for_input=scope_for_input, modules_for_scope=modules_for_scope, settings=settings,
        qualified_write_tools=frozenset(qualified),
        retained_write_tools=frozenset(file_tools.write_tools | additional_retained_writes))
    # Discovery only: no binding, session, grant, or tool invocation is created.
    # Native Hermes needs schemas before its first real input can be admitted.
    schema_scope = DomainInputScope(owner, "schema-only", "schema-only",
        UUID(int=0), frozenset(allowed))
    context = ToolBuildContext(settings, owner, schema_scope.jarvis_session_id)
    catalog = [{"name": tool.name, "description": tool.description,
                "parameters": tool.input_schema}
               for module in scoped_modules(schema_scope) for tool in module.build_tools(context)
               if tool.name in allowed]
    return RetainedSourceDomain(adapter, files, file_tools, frozenset(allowed), catalog)
