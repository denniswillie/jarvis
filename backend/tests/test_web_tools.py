from __future__ import annotations

import ipaddress
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx

from app.settings import JarvisSettings
from app.web.fetch import URLSafetyValidator, WebFetchError, WebFetcher
from app.web.search import OpenAIWebSearch, WebSearchError


class _CacheRepository:
    def __init__(self, cached: dict[str, Any] | None = None) -> None:
        self.cached = cached
        self.upserts: list[dict[str, Any]] = []

    async def get_fresh(self, **_: Any) -> dict[str, Any] | None:
        return self.cached

    async def upsert(self, **values: Any) -> None:
        self.upserts.append(values)


async def _public_resolver(_: str, __: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    return [ipaddress.ip_address("93.184.216.34")]


class URLSafetyValidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_private_and_credentialed_urls(self) -> None:
        async def private_resolver(_: str, __: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
            return [ipaddress.ip_address("169.254.169.254")]

        for url in (
            "file:///etc/passwd",
            "http://user:secret@example.com/",
            "https://example.com:8443/",
            "http://localhost/",
        ):
            with self.subTest(url=url), self.assertRaises(WebFetchError) as context:
                await URLSafetyValidator(_public_resolver).validate(url)
            self.assertEqual(context.exception.code, "web_fetch_url_not_allowed")

        with self.assertRaises(WebFetchError) as context:
            await URLSafetyValidator(private_resolver).validate("http://metadata.example/")
        self.assertEqual(context.exception.code, "web_fetch_url_not_allowed")

    async def test_accepts_public_standard_port_url_and_removes_fragment(self) -> None:
        result = await URLSafetyValidator(_public_resolver).validate("https://example.com/path#section")
        self.assertEqual(result.url, "https://example.com/path")
        self.assertEqual(result.port, 443)


class WebFetcherTests(unittest.IsolatedAsyncioTestCase):
    def _settings(self, *, max_bytes: int = 2_000_000, max_redirects: int = 3) -> JarvisSettings:
        return JarvisSettings(
            _env_file=None,
            WEB_FETCH_MAX_BYTES=max_bytes,
            WEB_FETCH_MAX_REDIRECTS=max_redirects,
        )

    async def test_extracts_and_caches_html(self) -> None:
        repository = _CacheRepository()
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/html"},
                stream=httpx.ByteStream(
                    b"<html><head><title>Example</title></head><body><main>Hello world</main></body></html>"
                ),
                request=request,
            )
        )
        fetcher = WebFetcher(
            settings=self._settings(),
            repository=repository,  # type: ignore[arg-type]
            validator=URLSafetyValidator(_public_resolver),
            transport=transport,
        )

        result = await fetcher.fetch(owner_id="owner", url="https://example.com/article")

        self.assertEqual(result["title"], "Example")
        self.assertEqual(result["text"], "Hello world")
        self.assertFalse(result["cached"])
        self.assertEqual(len(repository.upserts), 1)

    async def test_revalidates_redirect_target(self) -> None:
        async def resolver(hostname: str, _: int) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
            address = "127.0.0.1" if hostname == "private.example" else "93.184.216.34"
            return [ipaddress.ip_address(address)]

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                302,
                headers={"location": "http://private.example/metadata"},
                request=request,
            )
        )
        fetcher = WebFetcher(
            settings=self._settings(),
            repository=_CacheRepository(),  # type: ignore[arg-type]
            validator=URLSafetyValidator(resolver),
            transport=transport,
        )

        with self.assertRaises(WebFetchError) as context:
            await fetcher.fetch(owner_id="owner", url="https://example.com/")
        self.assertEqual(context.exception.code, "web_fetch_url_not_allowed")

    async def test_rejects_oversized_response(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                stream=httpx.ByteStream(b"12345"),
                request=request,
            )
        )
        fetcher = WebFetcher(
            settings=self._settings(max_bytes=4),
            repository=_CacheRepository(),  # type: ignore[arg-type]
            validator=URLSafetyValidator(_public_resolver),
            transport=transport,
        )

        with self.assertRaises(WebFetchError) as context:
            await fetcher.fetch(owner_id="owner", url="https://example.com/")
        self.assertEqual(context.exception.code, "web_fetch_too_large")

    async def test_returns_fresh_cached_page_without_network(self) -> None:
        repository = _CacheRepository(
            {
                "url": "https://example.com/",
                "title": "Cached",
                "text_content": "Cached text",
                "fetched_at": datetime.now(UTC),
                "metadata": {"final_url": "https://example.com/final"},
            }
        )
        transport = httpx.MockTransport(lambda request: self.fail(f"unexpected request: {request.url}"))
        fetcher = WebFetcher(
            settings=self._settings(),
            repository=repository,  # type: ignore[arg-type]
            validator=URLSafetyValidator(_public_resolver),
            transport=transport,
        )

        result = await fetcher.fetch(owner_id="owner", url="https://example.com/")

        self.assertTrue(result["cached"])
        self.assertEqual(result["url"], "https://example.com/final")


class _FakeResponses:
    def __init__(self, response: Any) -> None:
        self._response = response

    async def create(self, **_: Any) -> Any:
        return self._response


class _FakeOpenAIClient:
    def __init__(self, response: Any) -> None:
        self.responses = _FakeResponses(response)


class OpenAIWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_configuration(self) -> None:
        search = OpenAIWebSearch(
            settings=JarvisSettings(
                _env_file=None,
                OPENAI_API_KEY="replace_me",
                OPENAI_WEB_SEARCH_MODEL="gpt-5.6",
            )
        )
        with self.assertRaises(WebSearchError) as context:
            await search.search(owner_id="owner", query="query", limit=5)
        self.assertEqual(context.exception.code, "web_search_not_configured")

    async def test_returns_bounded_cited_results(self) -> None:
        annotation = SimpleNamespace(
            type="url_citation",
            url="https://example.com/source",
            title="Source",
            start_index=0,
            end_index=6,
        )
        output_text = SimpleNamespace(type="output_text", text="Answer with citation", annotations=[annotation])
        message = SimpleNamespace(type="message", content=[output_text])
        response = SimpleNamespace(output=[message], output_text="Answer with citation")
        search = OpenAIWebSearch(
            settings=JarvisSettings(
                _env_file=None,
                OPENAI_API_KEY="test-key",
                OPENAI_WEB_SEARCH_MODEL="gpt-5.6",
            ),
            client=_FakeOpenAIClient(response),  # type: ignore[arg-type]
        )

        result = await search.search(owner_id="owner", query="query", limit=5)

        self.assertEqual(result["answer"], "Answer with citation")
        self.assertEqual(result["results"][0]["url"], "https://example.com/source")
        self.assertEqual(result["results"][0]["snippet"], "Answer")


if __name__ == "__main__":
    unittest.main()
