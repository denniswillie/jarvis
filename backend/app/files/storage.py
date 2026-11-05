from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
from uuid import UUID, uuid4
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from app.errors import JarvisHTTPError
from app.files.local_file import CHUNK_BYTES, LocalFileError, VerifiedLocalFile, _snapshot_chunks
from app.files.models import StoredFileObject
from app.storage.supabase_config import SupabaseStorageConfig


@dataclass(frozen=True)
class SignedRecordingUpload:
    """Storage transport only; recording authority remains with the caller.

    The credential is not a saved-file receipt or a per-recording byte-limit
    guarantee. Its expiry must be retained by the upload/deletion owner.
    """
    bucket: str
    path: str
    resumable_url: str
    expires_at: datetime
    token: str = field(repr=False)


class JarvisFileStorageClient:
    def __init__(self, config: SupabaseStorageConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def bucket(self) -> str:
        return self._config.files_bucket

    async def close(self) -> None:
        await self._client.aclose()

    def build_file_path(self, *, owner_id: str, session_id: UUID | None, file_id: UUID, filename: str) -> str:
        safe_owner = _safe_path_part(owner_id)
        # The requested filename is display metadata, never part of the object key.
        session_part = str(session_id) if session_id is not None else "no-session"
        return f"owners/{safe_owner}/sessions/{session_part}/files/{file_id}/content"

    async def sign_recording_upload(self, *, path: str) -> SignedRecordingUpload:
        """Sign exactly one non-upserting target, without sending media bytes.

        Only the authenticated recording owner may call this with its own
        derived path. The returned token comes from the configured TLS storage
        service, never a caller-supplied JWT. Parsing it validates the response
        binding and reports expiry; Storage still verifies its signature.
        """
        base = urlsplit(self._base_url)
        if (base.scheme != "https" or not base.hostname or base.username or base.password
                or base.query or base.fragment or base.path not in {"", "/"}
                or not isinstance(path, str) or not 1 <= len(path) <= 1024
                or path.startswith("/") or "\\" in path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or any(ord(char) < 32 for char in path)):
            raise ValueError("An HTTPS storage origin and exact relative object path are required")
        signed_path = f"/object/upload/sign/{quote(self.bucket, safe='')}/{quote(path, safe='/')}"
        try:
            response = await self._client.post(self._base_url + "/storage/v1" + signed_path,
                json={}, headers={"Authorization": f"Bearer {self._config.service_role_key}",
                         "apikey": self._config.service_role_key,
                         "Content-Type": "application/json", "x-upsert": "false"},
                follow_redirects=False)
            if response.status_code != 200:
                raise ValueError("Signing was not confirmed")
            value = response.json()
            returned = urlsplit(value["url"])
            query = parse_qs(returned.query, strict_parsing=True)
            if (returned.scheme or returned.netloc or returned.fragment or returned.path != signed_path
                    or set(query) != {"token"} or len(query["token"]) != 1):
                raise ValueError("The signed target changed")
            token = query["token"][0]
            if (not 1 <= len(token) <= 16384 or len(token.split(".")) != 3
                    or value.get("token", token) != token):
                raise ValueError("Invalid signed capability")
            encoded = token.split(".")[1]
            claims = json.loads(base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True))
            now = datetime.now(timezone.utc).timestamp()
            if (not isinstance(claims, dict) or claims.get("url") != self.bucket + "/" + path
                    or claims.get("upsert") is not False or type(claims.get("exp")) is not int
                    or not now < claims["exp"] <= now + 2 * 3600 + 90):
                raise ValueError("Unexpected target, overwrite permission or expiry")
            expiry = datetime.fromtimestamp(claims["exp"], timezone.utc)
        except (httpx.HTTPError, ValueError, KeyError, TypeError, UnicodeError):
            # Provider text, URL and token must not enter error payloads/logs.
            raise JarvisHTTPError(502, "recording_upload_signing_unconfirmed",
                "A signed upload target could not be confirmed; no recording was saved.") from None
        # Hosted projects have a dedicated storage origin; custom HTTPS origins
        # retain their configured host. Never accept a response-selected host.
        origin = self._base_url
        if base.port is None and re.fullmatch(r"[a-z0-9-]+\.supabase\.co", base.hostname):
            origin = "https://" + base.hostname.removesuffix(".supabase.co") + ".storage.supabase.co"
        return SignedRecordingUpload(self.bucket, path, origin + "/storage/v1/upload/resumable/sign", expiry, token)

    async def upload_bytes(self, *, path: str, content: bytes, content_type: str) -> StoredFileObject:
        response = await self._client.post(
            f"{self._base_url}/storage/v1/object/{quote(self.bucket, safe='')}/{quote(path, safe='/')}",
            content=content,
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
                "Content-Type": content_type,
                "x-upsert": "false",
            },
        )
        if response.status_code not in {200, 201}:
            raise JarvisHTTPError(
                502,
                "jarvis_file_upload_failed",
                "Jarvis file upload to Supabase storage failed.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=len(content))

    async def upload_file(self, *, path: str, local_file: VerifiedLocalFile,
                          content_type: str) -> StoredFileObject:
        """Single non-upserting request; the publication owner handles unknown effects."""
        if not isinstance(local_file, VerifiedLocalFile):
            raise TypeError("A verified local file is required")
        local_file._check()
        complete = False

        async def content():
            nonlocal complete
            async for block in local_file.iter_chunks():
                yield block
            complete = True

        try:
            async with self._client.stream("POST",
                    f"{self._base_url}/storage/v1/object/{quote(self.bucket, safe='')}/{quote(path, safe='/')}",
                    content=content(), headers={
                        "Authorization": f"Bearer {self._config.service_role_key}",
                        "apikey": self._config.service_role_key, "Content-Type": content_type,
                        "Content-Length": str(local_file.size_bytes), "x-upsert": "false",
                    }) as response:
                if response.status_code not in {200, 201}:
                    raise JarvisHTTPError(502, "jarvis_file_upload_failed",
                        "Jarvis file upload to Supabase storage failed.",
                        details={"status_code": response.status_code})
                if not complete:
                    raise JarvisHTTPError(502, "jarvis_file_upload_incomplete",
                        "Storage did not consume the complete verified upload.")
        except httpx.HTTPError:
            raise JarvisHTTPError(502, "jarvis_file_upload_failed",
                "Jarvis file upload could not be confirmed.") from None
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=local_file.size_bytes)

    async def copy_object(
        self,
        *,
        source_bucket: str,
        source_path: str,
        destination_path: str,
        content_type: str,
    ) -> StoredFileObject:
        response = await self._client.get(
            f"{self._base_url}/storage/v1/object/{quote(source_bucket, safe='')}/{quote(source_path, safe='/')}",
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
            },
        )
        if response.status_code != 200:
            raise JarvisHTTPError(
                502,
                "jarvis_file_source_download_failed",
                "Source object could not be downloaded for Jarvis file promotion.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )
        return await self.upload_bytes(
            path=destination_path,
            content=response.content,
            content_type=content_type,
        )

    async def delete_object(self, *, bucket: str, path: str) -> None:
        response = await self._client.request(
            "DELETE",
            f"{self._base_url}/storage/v1/object/{quote(bucket, safe='')}",
            json={"prefixes": [path]},
            headers={
                "Authorization": f"Bearer {self._config.service_role_key}",
                "apikey": self._config.service_role_key,
                "Content-Type": "application/json",
            },
        )
        if response.status_code >= 400:
            raise JarvisHTTPError(
                502,
                "jarvis_file_delete_failed",
                "Jarvis file deletion from Supabase storage failed.",
                details={"status_code": response.status_code, "body": response.text[:500]},
            )

    async def read_object(self, *, bucket: str, path: str, max_bytes: int) -> bytes | None:
        """Bounded authenticated read; only an exact missing-object receipt means absent."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("A nonnegative object byte bound is required")
        headers = {"Authorization": f"Bearer {self._config.service_role_key}",
                   "apikey": self._config.service_role_key, "Cache-Control": "no-cache"}
        url = f"{self._base_url}/storage/v1/object/authenticated/{quote(bucket, safe='')}/{quote(path, safe='/')}"
        async with self._client.stream("GET", url, headers=headers,
                                       params={"jarvis_file_read": uuid4().hex}) as response:
            if response.status_code != 200:
                body = bytearray()
                async for block in response.aiter_bytes():
                    body.extend(block)
                    if len(body) > 2048:
                        break
                if len(body) <= 2048 and response.status_code in {400, 404}:
                    import json
                    try:
                        error = json.loads(body)
                    except (ValueError, UnicodeDecodeError):
                        error = None
                    if isinstance(error, dict) and error.get("code") == "NoSuchKey" and str(error.get("statusCode")) == "404":
                        return None
                raise JarvisHTTPError(502, "jarvis_file_read_failed",
                    "Retained file availability could not be established.",
                    details={"status_code": response.status_code})
            content = bytearray()
            async for block in response.aiter_bytes():
                if len(content) + len(block) > max_bytes:
                    raise JarvisHTTPError(502, "jarvis_file_size_exceeded", "Retained file exceeds its byte bound.")
                content.extend(block)
            return bytes(content)

    async def read_object_file(self, *, bucket: str, path: str, max_bytes: int) -> VerifiedLocalFile | None:
        """Bounded disk-backed read. Caller owns the returned handle and close()."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("A nonnegative object byte bound is required")
        headers = {"Authorization": f"Bearer {self._config.service_role_key}",
                   "apikey": self._config.service_role_key, "Cache-Control": "no-cache",
                   "Accept-Encoding": "identity"}
        url = f"{self._base_url}/storage/v1/object/authenticated/{quote(bucket, safe='')}/{quote(path, safe='/')}"
        result = None
        try:
            async with self._client.stream("GET", url, headers=headers,
                    params={"jarvis_file_read": uuid4().hex}) as response:
                if response.status_code != 200:
                    # Match read_object exactly: Supabase may transport a
                    # NoSuchKey/404 receipt as HTTP400. Other errors are unknown.
                    body = bytearray()
                    async for block in response.aiter_raw(chunk_size=2049):
                        body.extend(block)
                        if len(body) > 2048:
                            break
                    if len(body) <= 2048 and response.status_code in {400, 404}:
                        import json
                        try:
                            error = json.loads(body)
                        except (ValueError, UnicodeDecodeError):
                            error = None
                        if (isinstance(error, dict) and error.get("code") == "NoSuchKey"
                                and str(error.get("statusCode")) == "404"):
                            return None
                    raise JarvisHTTPError(502, "jarvis_file_read_failed",
                        "Retained file availability could not be established.",
                        details={"status_code": response.status_code})
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise JarvisHTTPError(502, "jarvis_file_read_failed",
                        "Retained file transfer used an unsupported content encoding.")
                length = response.headers.get("Content-Length")
                if length is not None:
                    if not length.isascii() or not length.isdecimal():
                        raise JarvisHTTPError(502, "jarvis_file_length_mismatch",
                            "Retained file has an invalid declared byte count.")
                    length = int(length)
                    if length > max_bytes:
                        raise JarvisHTTPError(502, "jarvis_file_size_exceeded",
                            "Retained file exceeds its byte bound.")
                result = await _snapshot_chunks(response.aiter_raw(chunk_size=CHUNK_BYTES),
                    max_bytes=max_bytes, expected_size=length)
            return result
        except BaseException as error:
            # This also covers cancellation or failure while closing a response
            # after all bytes were written: ownership has not transferred yet.
            if result is not None:
                result.close()
            if isinstance(error, LocalFileError):
                code = {"local_file_size_exceeded": "jarvis_file_size_exceeded",
                        "local_file_length_mismatch": "jarvis_file_length_mismatch"}.get(
                            error.code, "jarvis_file_read_failed")
                raise JarvisHTTPError(502, code, "Retained file transfer could not be verified.") from None
            if isinstance(error, httpx.HTTPError):
                raise JarvisHTTPError(502, "jarvis_file_read_failed",
                    "Retained file availability could not be established.") from None
            raise

    @property
    def _base_url(self) -> str:
        return self._config.supabase_url.rstrip("/")


def _safe_path_part(value: str) -> str:
    return value.replace("/", "_").replace("..", "_")
