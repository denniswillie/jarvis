"""Owned-session access to immutable retained files, without version fallback."""
import hashlib
from uuid import UUID

from app.errors import JarvisHTTPError
from app.files.transfer import MP4Inspection
from app.files.previews import build_preview, store_preview


def file_version(file):
    value = (file.memory_source or {}).get("document_id")
    if not isinstance(value, str) or not value or len(value) > 256:
        raise JarvisHTTPError(409, "file_version_unavailable", "The file has no exact retained source version.")
    return value


class WorkspaceFileAccess:
    def __init__(self, repository):
        self.repository = repository

    async def scoped(self, owner_id, session_id):
        self.repository._authorize(owner_id)
        if session_id is None:
            if self.repository.selections is not None:
                raise JarvisHTTPError(403, "workspace_session_required", "An owned session is required for selected-area files.")
            return self.repository
        if not isinstance(session_id, UUID):
            raise JarvisHTTPError(403, "workspace_session_required", "An owned session is required.")
        async with self.repository.sources.pool.acquire() as connection:
            if not await connection.fetchval(
                "select true from jarvis_sessions where owner_id=$1 and id=$2", owner_id, session_id
            ):
                raise JarvisHTTPError(404, "workspace_session_unavailable", "The session is outside this owner scope.")
        return self.repository.for_session(session_id)

    @staticmethod
    async def _get(repository, owner_id, file_id, version_id):
        file = await repository.get_file(owner_id=owner_id, file_id=file_id)
        if file is None:
            raise JarvisHTTPError(404, "file_not_found", "No retained file exists in the authorized area.")
        exact = file_version(file)
        if version_id is not None and exact != version_id:
            raise JarvisHTTPError(404, "file_version_unavailable", "The requested file version is unavailable; no newer version was substituted.")
        return file

    async def get(self, owner_id, session_id, file_id, version_id=None):
        repository = await self.scoped(owner_id, session_id)
        return await self._get(repository, owner_id, file_id, version_id)

    async def read(self, owner_id, session_id, file_id, version_id=None):
        repository = await self.scoped(owner_id, session_id)
        first = await self._get(repository, owner_id, file_id, version_id)
        content = await repository.read_content(owner_id=owner_id, file_id=file_id)
        current = await self._get(repository, owner_id, file_id, file_version(first))
        if current.size_bytes is not None and len(content) != current.size_bytes:
            raise JarvisHTTPError(409, "file_content_mismatch", "The retained bytes do not match the saved file size.")
        checksum = hashlib.sha256(content).hexdigest()
        await self._cache_preview(repository, owner_id, current, content, checksum)
        current = await self._get(repository, owner_id, file_id, file_version(first))
        return current, content, checksum

    @staticmethod
    async def _cache_preview(repository, owner_id, file, content, checksum):
        # Legacy items gain a preview only when the user has already fetched
        # their original. Never fetch an original for gallery population.
        if not (file.content_type.startswith("image/") or file.content_type == "video/mp4"):
            return
        async with repository.sources.pool.acquire() as connection:
            exists = await connection.fetchval("""select preview_jpeg is not null
                from jarvis_file_publications where owner_id=$1 and file_id=$2""", owner_id, file.id)
        if exists:
            return
        preview = await build_preview(content, file.content_type)
        if preview is not None:
            async with repository.sources.pool.acquire() as connection:
                await store_preview(connection, owner_id, file.id, checksum, preview)

    async def _media_metadata(self, owner_id, session_id, file_id, version_id):
        repository = await self.scoped(owner_id, session_id)
        first = await self._get(repository, owner_id, file_id, version_id)
        manifest, policy = await repository.publications.read_manifest(owner_id=owner_id, file_id=file_id)
        current = await self._get(repository, owner_id, file_id, file_version(first))
        if current.content_type != "video/mp4" or not policy.is_media or current.size_bytes != manifest.size_bytes:
            raise JarvisHTTPError(409, "file_media_unavailable", "The file has no verified media transfer policy.")
        try:
            measured = current.metadata.get("media_inspection") or current.metadata.get(
                "generated_artifact", {}).get("validation", {}).get("media_inspection")
            # Early retained-Hermes publications nested the verified report.
            # Preserve access to those immutable files; the same manifest and
            # media-policy validation below still applies before any byte read.
            if measured is None and isinstance(current.metadata.get("hermes_session_authority"), dict):
                report = current.metadata.get("inspection")
                if isinstance(report, dict):
                    measured = report.get("media_inspection")
            inspection = MP4Inspection.model_validate(measured)
            inspection.validate_file(manifest, policy)
        except (ValueError, TypeError):
            raise JarvisHTTPError(409, "file_media_unavailable", "The exact media inspection is unavailable.") from None
        return current, {
            "version_id": file_version(current), "content_sha256": manifest.content_sha256,
            "transfer_policy": policy.policy_id, "transfer_limit_bytes": policy.max_bytes,
            "media_inspection": inspection.model_dump(mode="json"),
        }, policy

    async def media_metadata(self, owner_id, session_id, file_id, version_id):
        file, metadata, _ = await self._media_metadata(owner_id, session_id, file_id, version_id)
        return file, metadata

    async def read_file(self, owner_id, session_id, file_id, version_id):
        file, local_file, metadata, _ = await self.read_file_with_policy(owner_id, session_id, file_id, version_id)
        return file, local_file, metadata

    async def read_file_with_policy(self, owner_id, session_id, file_id, version_id):
        """Internal delivery preparation retains server policy; no client policy input."""
        if version_id is None:
            raise JarvisHTTPError(422, "file_version_required", "An exact media version is required.")
        first, metadata, policy = await self._media_metadata(owner_id, session_id, file_id, version_id)
        repository = await self.scoped(owner_id, session_id)
        local_file = await repository.read_file(owner_id=owner_id, file_id=file_id)
        try:
            current = await self._get(repository, owner_id, file_id, file_version(first))
            if (local_file.size_bytes != current.size_bytes
                    or local_file.content_sha256 != metadata["content_sha256"]):
                raise JarvisHTTPError(409, "file_content_mismatch", "The retained bytes differ from the exact manifest.")
            await self._cache_preview(repository, owner_id, current, local_file, metadata["content_sha256"])
            current = await self._get(repository, owner_id, file_id, file_version(first))
            return current, local_file, metadata, policy
        except BaseException:
            local_file.close()
            raise
