from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import asyncpg
import httpx
from bs4 import BeautifulSoup

from app.settings import JarvisSettings


MAX_RETURNED_TEXT_CHARACTERS = 12_000
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
SUPPORTED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
LOCAL_HOSTNAME_SUFFIXES = (".localhost", ".local", ".internal")


class WebFetchError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ValidatedURL:
    url: str
    hostname: str
    port: int


AddressResolver = Callable[[str, int], Awaitable[list[ipaddress.IPv4Address | ipaddress.IPv6Address]]]


class URLSafetyValidator:
    def __init__(self, resolver: AddressResolver | None = None) -> None:
        self._resolver = resolver or _resolve_addresses

    async def validate(self, raw_url: str) -> ValidatedURL:
        try:
            parsed = urlsplit(raw_url.strip())
            port = parsed.port
        except ValueError as error:
            raise _url_not_allowed() from error

        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if scheme not in {"http", "https"} or not hostname:
            raise _url_not_allowed()
        if parsed.username is not None or parsed.password is not None:
            raise _url_not_allowed()

        expected_port = 443 if scheme == "https" else 80
        if port is not None and port != expected_port:
            raise _url_not_allowed()
        port = port or expected_port

        if _is_local_hostname(hostname):
            raise _url_not_allowed()

        try:
            addresses = await self._resolver(hostname, port)
        except (OSError, socket.gaierror) as error:
            raise WebFetchError("web_fetch_failed", "The URL could not be resolved.") from error
        if not addresses or any(not address.is_global for address in addresses):
            raise _url_not_allowed()

        return ValidatedURL(
            url=parsed._replace(fragment="").geturl(),
            hostname=hostname,
            port=port,
        )


class WebFetchCacheRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_fresh(
        self,
        *,
        owner_id: str,
        url: str,
        max_age_seconds: int,
    ) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                select url, title, text_content, fetched_at, metadata
                from web_fetch_cache
                where owner_id = $1
                  and url = $2
                  and fetched_at >= now() - make_interval(secs => $3)
                """,
                owner_id,
                url,
                max_age_seconds,
            )
        return dict(record) if record is not None else None

    async def upsert(
        self,
        *,
        owner_id: str,
        url: str,
        title: str,
        text_content: str,
        metadata: dict[str, Any],
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                insert into web_fetch_cache (
                    owner_id,
                    url,
                    title,
                    text_content,
                    fetched_at,
                    metadata
                )
                values ($1, $2, $3, $4, now(), $5::jsonb)
                on conflict (owner_id, url) do update
                set title = excluded.title,
                    text_content = excluded.text_content,
                    fetched_at = excluded.fetched_at,
                    metadata = excluded.metadata
                """,
                owner_id,
                url,
                title,
                text_content,
                json.dumps(metadata),
            )


class WebFetcher:
    def __init__(
        self,
        *,
        settings: JarvisSettings,
        repository: WebFetchCacheRepository,
        validator: URLSafetyValidator | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._validator = validator or URLSafetyValidator()
        self._transport = transport

    async def fetch(self, *, owner_id: str, url: str) -> dict[str, Any]:
        requested = await self._validator.validate(url)
        cached = await self._repository.get_fresh(
            owner_id=owner_id,
            url=requested.url,
            max_age_seconds=self._settings.WEB_FETCH_CACHE_TTL_SECONDS,
        )
        if cached is not None:
            metadata = _metadata(cached.get("metadata"))
            return {
                "url": metadata.get("final_url") or cached["url"],
                "title": cached.get("title") or "",
                "text": str(cached["text_content"])[:MAX_RETURNED_TEXT_CHARACTERS],
                "fetched_at": cached["fetched_at"].isoformat(),
                "cached": True,
            }

        current_url = requested.url
        redirect_count = 0
        timeout = httpx.Timeout(
            self._settings.WEB_FETCH_TIMEOUT_SECONDS,
            connect=self._settings.WEB_FETCH_TIMEOUT_SECONDS,
        )
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            ) as client:
                while True:
                    validated = await self._validator.validate(current_url)
                    async with client.stream(
                        "GET",
                        validated.url,
                        headers={
                            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
                            "Accept-Encoding": "identity",
                            "User-Agent": "Jarvis/0.1 (+https://your-jarvis.example.invalid)",
                        },
                    ) as response:
                        if response.status_code in REDIRECT_STATUS_CODES:
                            location = response.headers.get("location")
                            if not location:
                                raise WebFetchError("web_fetch_failed", "The URL returned an invalid redirect.")
                            if redirect_count >= self._settings.WEB_FETCH_MAX_REDIRECTS:
                                raise WebFetchError(
                                    "web_fetch_too_many_redirects",
                                    "The URL redirected too many times.",
                                )
                            current_url = urljoin(validated.url, location)
                            redirect_count += 1
                            continue

                        if response.status_code >= 400:
                            raise WebFetchError(
                                "web_fetch_failed",
                                f"The page returned HTTP {response.status_code}.",
                            )
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if content_type not in SUPPORTED_CONTENT_TYPES:
                            raise WebFetchError(
                                "web_fetch_unsupported_content_type",
                                "The URL did not return a supported HTML or text page.",
                            )
                        content_encoding = response.headers.get("content-encoding", "identity").lower()
                        if content_encoding not in {"", "identity"}:
                            raise WebFetchError(
                                "web_fetch_unsupported_content_encoding",
                                "The page returned an unsupported content encoding.",
                            )
                        body = await _read_bounded_body(
                            response,
                            max_bytes=self._settings.WEB_FETCH_MAX_BYTES,
                        )
                        final_url = validated.url
                        break
        except WebFetchError:
            raise
        except httpx.TimeoutException as error:
            raise WebFetchError("web_fetch_timeout", "The page took too long to respond.") from error
        except httpx.HTTPError as error:
            raise WebFetchError("web_fetch_failed", "The page could not be fetched.") from error

        text_body = body.decode(response.encoding or "utf-8", errors="replace")
        title, text_content = _extract_readable_text(text_body, content_type=content_type)
        if not text_content:
            raise WebFetchError("web_fetch_empty", "The page did not contain readable text.")
        text_content = text_content[:MAX_RETURNED_TEXT_CHARACTERS]
        metadata = {
            "final_url": final_url,
            "content_type": content_type,
            "redirect_count": redirect_count,
        }
        await self._repository.upsert(
            owner_id=owner_id,
            url=requested.url,
            title=title,
            text_content=text_content,
            metadata=metadata,
        )
        return {
            "url": final_url,
            "title": title,
            "text": text_content,
            "cached": False,
        }


async def _resolve_addresses(
    hostname: str,
    port: int,
) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    def resolve() -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        return list({ipaddress.ip_address(record[4][0]) for record in records})

    return await asyncio.to_thread(resolve)


def _is_local_hostname(hostname: str) -> bool:
    if hostname == "localhost" or hostname == "metadata.google.internal":
        return True
    return any(hostname.endswith(suffix) for suffix in LOCAL_HOSTNAME_SUFFIXES)


async def _read_bounded_body(response: httpx.Response, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_raw():
        total += len(chunk)
        if total > max_bytes:
            raise WebFetchError("web_fetch_too_large", "The page exceeded the fetch size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_readable_text(body: str, *, content_type: str) -> tuple[str, str]:
    if content_type == "text/plain":
        return "", _normalize_text(body)

    soup = BeautifulSoup(body, "lxml")
    title = _normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    for element in soup(["script", "style", "noscript", "template", "svg"]):
        element.decompose()
    content = soup.find("article") or soup.find("main") or soup.body or soup
    return title[:500], _normalize_text(content.get_text("\n", strip=True))


def _normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _url_not_allowed() -> WebFetchError:
    return WebFetchError(
        "web_fetch_url_not_allowed",
        "That URL is not allowed. Use a public HTTP or HTTPS URL on a standard port.",
    )
