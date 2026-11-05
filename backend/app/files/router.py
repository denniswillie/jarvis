"""Authenticated original-file access; no bearer URLs or legacy-memory fallback."""
from typing import Annotated
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.files.tools import FileToolModule, _file_payload
from app.files.previews import preview_references, read_preview
from app.files.workspace_access import WorkspaceFileAccess, file_version
from app.knowledge.files import SourceFileRepository
from app.runtime.feature_config import MemoryAuthority

router = APIRouter(prefix="/files", tags=["files"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
PRIVATE_HEADERS = {"Cache-Control": "private, no-store", "Vary": "Authorization",
                   "X-Content-Type-Options": "nosniff"}


class _PrivateFileResponse(StreamingResponse):
    def __init__(self, local_file, *, media_type, headers):
        self.local_file = local_file
        super().__init__(local_file.iter_chunks(), media_type=media_type, headers=headers)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Also covers disconnect before the first chunk and header-send failure.
            self.local_file.close()


def _repository(request, owner_id):
    services = getattr(request.app.state, "runtime_services", None)
    if services is None or services.settings.JARVIS_OWNER_ID != owner_id:
        raise JarvisHTTPError(503, "source_files_unavailable", "Authenticated source-backed file access is unavailable.")
    if services.memory_authority != MemoryAuthority.HINDSIGHT:
        access = getattr(services, "recording_source_access", None)
        if access is None or access.sources.pool is not services.db_pool:
            raise JarvisHTTPError(503, "source_files_unavailable", "Authenticated source-backed file access is unavailable.")
        access.check(owner_id)
        return access.files
    matches = [module.repository for module in services.tool_modules
               if isinstance(module, FileToolModule) and isinstance(module.repository, SourceFileRepository)]
    if len(matches) != 1:
        raise JarvisHTTPError(503, "source_files_unavailable", "The selected file source is not ready.")
    return matches[0]


@router.get("")
async def list_files(request: Request, principal: Principal,
                     query: Annotated[str | None, Query(max_length=500)] = None,
                     limit: Annotated[int, Query(ge=1, le=8)] = 8,
                     session_id: UUID | None = None,
                     media_only: bool = False,
                     cursor: Annotated[str | None, Query(min_length=1, max_length=200)] = None):
    repository = await WorkspaceFileAccess(_repository(request, principal.owner_id)).scoped(principal.owner_id, session_id)
    if media_only or cursor is not None:
        page = await repository.list_file_page(owner_id=principal.owner_id, query=query,
            limit=limit, media_only=media_only, cursor=cursor)
        previews = await preview_references(repository, principal.owner_id, page.files)
        return JSONResponse({"files": [_file_payload(file) | {"thumbnail": previews.get(file.id)} for file in page.files],
            "page_full": page.next_cursor is not None, "next_cursor": page.next_cursor}, headers=PRIVATE_HEADERS)
    files = await repository.list_files(
        owner_id=principal.owner_id, query=query, limit=limit)
    previews = await preview_references(repository, principal.owner_id, files)
    return JSONResponse({"files": [_file_payload(file) | {"thumbnail": previews.get(file.id)} for file in files],
                         "page_full": len(files) == limit}, headers=PRIVATE_HEADERS)


@router.get("/{file_id}")
async def file_metadata(file_id: UUID, request: Request, principal: Principal,
                        session_id: UUID | None = None,
                        version_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None):
    access = WorkspaceFileAccess(_repository(request, principal.owner_id))
    file = await access.get(
        principal.owner_id, session_id, file_id, version_id)
    extra = {}
    if file.content_type == "video/mp4":
        file, extra = await access.media_metadata(principal.owner_id, session_id, file_id, version_id)
    previews = await preview_references(access.repository, principal.owner_id, [file])
    return JSONResponse({"file": _file_payload(file) | extra | {"thumbnail": previews.get(file.id)}}, headers=PRIVATE_HEADERS)


@router.get("/{file_id}/thumbnail")
async def file_thumbnail(file_id: UUID, request: Request, principal: Principal,
                         version_id: Annotated[str, Query(min_length=1, max_length=256)],
                         session_id: UUID | None = None):
    access = WorkspaceFileAccess(_repository(request, principal.owner_id))
    await access.get(principal.owner_id, session_id, file_id, version_id)
    preview = await read_preview(access.repository, principal.owner_id, file_id)
    # Recheck authorization/version after reading derivative bytes as for originals.
    await access.get(principal.owner_id, session_id, file_id, version_id)
    if preview is None:
        raise JarvisHTTPError(404, "thumbnail_unavailable", "A preview is not available. The original can still be opened.")
    data, checksum = preview
    return Response(data, media_type="image/jpeg", headers=PRIVATE_HEADERS | {
        "X-Jarvis-Version-Id": version_id, "X-Jarvis-Content-Sha256": checksum})


@router.get("/{file_id}/content")
async def file_content(file_id: UUID, request: Request, principal: Principal,
                       session_id: UUID | None = None,
                       version_id: Annotated[str | None, Query(min_length=1, max_length=256)] = None):
    access = WorkspaceFileAccess(_repository(request, principal.owner_id))
    selected = await access.get(principal.owner_id, session_id, file_id, version_id)
    if selected.content_type == "video/mp4":
        file, local_file, metadata = await access.read_file(principal.owner_id, session_id, file_id, version_id)
        try:
            return _PrivateFileResponse(local_file, media_type=file.content_type, headers=PRIVATE_HEADERS | {
                "Content-Length": str(local_file.size_bytes),
                "Content-Disposition": "attachment; filename*=UTF-8''" + quote(file.filename, safe=""),
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "ETag": '"' + metadata["content_sha256"] + '"',
                "X-Jarvis-Version-Id": file_version(file),
                "X-Content-SHA256": metadata["content_sha256"],
            })
        except BaseException:
            local_file.close()
            raise
    file, content, checksum = await access.read(
        principal.owner_id, session_id, file_id, version_id)
    return Response(content, media_type=file.content_type, headers=PRIVATE_HEADERS | {
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote(file.filename, safe=""),
        "Content-Security-Policy": "default-src 'none'; sandbox",
        "X-Jarvis-Version-Id": file_version(file),
        "X-Content-SHA256": checksum,
    })


@router.delete("/{file_id}")
async def delete_file(file_id: UUID, request: Request, principal: Principal, session_id: UUID | None = None):
    repository = await WorkspaceFileAccess(_repository(request, principal.owner_id)).scoped(principal.owner_id, session_id)
    result = await repository.delete_source_file(owner_id=principal.owner_id, file_id=file_id)
    return JSONResponse(result, status_code=200 if result["deleted"] else 202, headers=PRIVATE_HEADERS)
