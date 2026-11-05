from __future__ import annotations

from urllib.parse import quote
from uuid import UUID

import httpx

from app.errors import JarvisHTTPError
from app.storage.supabase_config import SupabaseStorageConfig
from app.visual_memory.models import VisualFrameStorageObject


class VisualFrameStorageClient:
    def __init__(self, config: SupabaseStorageConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=20.0)

    @property
    def bucket(self) -> str:
        return self._config.visual_frame_bucket

    async def close(self) -> None:
        await self._client.aclose()

    def build_storage_path(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        visual_session_id: UUID,
        frame_sequence: int,
    ) -> str:
        safe_owner = owner_id.replace("/", "_")
        return (
            f"owners/{safe_owner}/sessions/{session_id}/frames/"
            f"{visual_session_id}/{frame_sequence:012d}.jpg"
        )

    async def upload_frame(
        self,
        *,
        path: str,
        image_bytes: bytes,
        content_type: str,
    ) -> VisualFrameStorageObject:
        response = await self._client.post(
            f"{self._base_url}/storage/v1/object/{self.bucket}/{path}",
            content=image_bytes,
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
                "Content-Type": content_type,
                "x-upsert": "false",
            },
        )
        if response.status_code >= 400:
            raise JarvisHTTPError(
                502,
                "visual_frame_storage_upload_failed",
                "Visual frame upload to Supabase storage failed.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )
        return VisualFrameStorageObject(
            bucket=self.bucket,
            path=path,
            size_bytes=len(image_bytes),
        )

    async def delete_objects(self, *, bucket: str, paths: list[str]) -> None:
        if not paths:
            return
        response = await self._client.request(
            "DELETE",
            f"{self._base_url}/storage/v1/object/{bucket}",
            json={"prefixes": paths},
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise JarvisHTTPError(
                502,
                "visual_frame_storage_delete_failed",
                "Visual frame cleanup from Supabase storage failed.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )

    async def download_object(self, *, bucket: str, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or max_bytes < 1:
                raise ValueError("max_bytes must be positive")
            return await self._download_bounded(bucket=bucket, path=path, max_bytes=max_bytes)
        response = await self._client.get(
            f"{self._base_url}/storage/v1/object/authenticated/"
            f"{quote(bucket, safe='')}/{quote(path, safe='/')}",
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
            },
        )
        if response.status_code >= 400:
            raise JarvisHTTPError(
                502,
                "visual_frame_storage_download_failed",
                "Visual frame download from Supabase storage failed.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )
        return response.content

    async def _download_bounded(self, *, bucket: str, path: str, max_bytes: int) -> bytes:
        def too_large() -> JarvisHTTPError:
            return JarvisHTTPError(
                502, "visual_frame_storage_download_too_large",
                "Visual frame exceeds the accepted byte limit.",
            )

        async with self._client.stream(
            "GET",
            f"{self._base_url}/storage/v1/object/authenticated/"
            f"{quote(bucket, safe='')}/{quote(path, safe='/')}",
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
            },
        ) as response:
            if response.status_code >= 400:
                # Do not read an unbounded error body or expose private storage locators.
                raise JarvisHTTPError(
                    502, "visual_frame_storage_download_failed",
                    "Visual frame download from Supabase storage failed.",
                    details={"status_code": response.status_code},
                )
            try:
                declared_bytes = int(response.headers.get("Content-Length", "0"))
            except ValueError:
                declared_bytes = 0
            if declared_bytes > max_bytes:
                raise too_large()
            content = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=min(65_536, max_bytes + 1)):
                if len(content) + len(chunk) > max_bytes:
                    raise too_large()
                content.extend(chunk)
            return bytes(content)

    @property
    def _base_url(self) -> str:
        return self._config.supabase_url.rstrip("/")
