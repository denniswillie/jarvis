"""Bind existing Workspace receipts to source-backed content; no new worker."""
from app.errors import JarvisHTTPError
from app.files.tools import FileToolModule
from app.files.workspace_access import WorkspaceFileAccess, file_version
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import MemoryAuthority
from app.workspace.presentation import PreparedContent, PresentationRepository, PresentationService


class WorkspaceContentResolver:
    def __init__(self, files=None):
        self.files = files

    async def __call__(self, ctx, kind, artifact_id, version_id):
        if kind == "file" and self.files is not None:
            return file_version(await self.files.get(ctx.owner_id, ctx.session_id, artifact_id, version_id))
        raise JarvisHTTPError(503, "workspace_content_unavailable", "This retained content producer is unavailable.")


class WorkspaceContentExporter:
    def __init__(self, files):
        self.files = files

    async def __call__(self, ctx, request):
        if request.artifact_kind != "file" or request.action != "download":
            raise JarvisHTTPError(422, "workspace_export_invalid", "Only retained file downloads are supported.")
        selected = await self.files.get(ctx.owner_id, ctx.session_id, request.artifact_id, request.version_id)
        if selected.content_type == "video/mp4":
            file, local_file, metadata, policy = await self.files.read_file_with_policy(
                ctx.owner_id, ctx.session_id, request.artifact_id, request.version_id)
            try:
                return PreparedContent(file_id=file.id, byte_count=local_file.size_bytes,
                    content_sha256=local_file.content_sha256, transfer_policy=policy.policy_id,
                    transfer_limit_bytes=policy.max_bytes)
            finally:
                # Preparation establishes exact bytes, not a Phone save. The
                # authenticated client download owns a separate staged copy.
                local_file.close()
        file, content, checksum = await self.files.read(
            ctx.owner_id, ctx.session_id, request.artifact_id, request.version_id)
        if len(content) > 8 * 1024 * 1024:
            raise JarvisHTTPError(422, "workspace_export_too_large", "This phone delivery supports files up to 8 MiB.",
                                 details={"effect_committed": False})
        return PreparedContent(file_id=file.id, byte_count=len(content), content_sha256=checksum)


def initialize_workspace_presentations(services):
    services.workspace_presentations = None
    runtime = getattr(services, "knowledge_runtime", None)
    pool = getattr(services, "db_pool", None)
    if services.memory_authority != MemoryAuthority.HINDSIGHT:
        recording = getattr(services, "recording_source_access", None)
        if recording is None or pool is None or recording.sources.pool is not pool:
            return
        recording.check(services.settings.JARVIS_OWNER_ID)
        access = WorkspaceFileAccess(recording.files)
        services.workspace_presentations = PresentationService(PresentationRepository(pool),
            resolver=WorkspaceContentResolver(access),
            exporter=WorkspaceContentExporter(access))
        return
    if (services.memory_authority != MemoryAuthority.HINDSIGHT
            or runtime is None or pool is None):
        return
    matches = [module.repository for module in services.tool_modules
               if isinstance(module, FileToolModule) and isinstance(module.repository, SourceFileRepository)]
    files = matches[0] if len(matches) == 1 else None
    if files is not None and (files.owner_id != services.settings.JARVIS_OWNER_ID
            or files.sources is not runtime.sources or files.sources.pool is not pool
            or files.task_evidence is not None):
        files = None
    access = WorkspaceFileAccess(files) if files is not None else None
    exporter = WorkspaceContentExporter(access) if access is not None else None
    services.workspace_presentations = PresentationService(PresentationRepository(pool),
        resolver=WorkspaceContentResolver(access), exporter=exporter)
