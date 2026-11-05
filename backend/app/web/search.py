from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

from openai import AsyncOpenAI, OpenAIError

from app.realtime.openai_client import safety_identifier_for_owner
from app.settings import JarvisSettings, is_configured_secret


MAX_ANSWER_CHARACTERS = 5_000
MAX_SNIPPET_CHARACTERS = 500
MAX_SOURCE_TITLE_CHARACTERS = 300


class WebSearchError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class OpenAIWebSearch:
    def __init__(self, *, settings: JarvisSettings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client

    async def search(self, *, owner_id: str, query: str, limit: int) -> dict[str, Any]:
        if not is_configured_secret(self._settings.OPENAI_API_KEY) or not is_configured_secret(
            self._settings.OPENAI_WEB_SEARCH_MODEL
        ):
            raise WebSearchError(
                "web_search_not_configured",
                "Web search is not configured.",
            )

        client = self._client or AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)
        owns_client = self._client is None
        try:
            async with asyncio.timeout(self._settings.WEB_SEARCH_TIMEOUT_SECONDS):
                response = await client.responses.create(
                    model=self._settings.OPENAI_WEB_SEARCH_MODEL,
                    instructions=(
                        "Search the live web for the user's query. Return a concise factual answer "
                        "with inline citations. Prefer authoritative primary sources."
                    ),
                    input=query,
                    tools=[{"type": "web_search", "search_context_size": "low"}],
                    tool_choice="required",
                    include=["web_search_call.action.sources"],
                    max_output_tokens=1_000,
                    store=False,
                    safety_identifier=safety_identifier_for_owner(owner_id),
                )
        except TimeoutError as error:
            raise WebSearchError("web_search_timeout", "Web search took too long.") from error
        except OpenAIError as error:
            raise WebSearchError("web_search_failed", "Web search could not be completed.") from error
        finally:
            if owns_client:
                await client.close()

        return _bounded_search_result(response, query=query, limit=limit)


def _bounded_search_result(response: Any, *, query: str, limit: int) -> dict[str, Any]:
    answer_parts: list[str] = []
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for item in _as_list(_field(response, "output")):
        item_type = _field(item, "type")
        if item_type == "message":
            for content in _as_list(_field(item, "content")):
                if _field(content, "type") != "output_text":
                    continue
                text = str(_field(content, "text") or "")
                if text:
                    answer_parts.append(text)
                for annotation in _as_list(_field(content, "annotations")):
                    if _field(annotation, "type") != "url_citation":
                        continue
                    _append_source(
                        sources,
                        seen_urls,
                        url=_field(annotation, "url"),
                        title=_field(annotation, "title"),
                        snippet=_annotation_snippet(text, annotation),
                        limit=limit,
                    )
        elif item_type == "web_search_call":
            action = _field(item, "action")
            for source in _as_list(_field(action, "sources")):
                _append_source(
                    sources,
                    seen_urls,
                    url=_field(source, "url"),
                    title=_field(source, "title"),
                    snippet=_field(source, "snippet"),
                    limit=limit,
                )

    answer = "\n".join(answer_parts).strip()
    if not answer:
        answer = str(_field(response, "output_text") or "").strip()
    return {
        "query": query,
        "answer": answer[:MAX_ANSWER_CHARACTERS],
        "results": sources[:limit],
    }


def _append_source(
    sources: list[dict[str, str]],
    seen_urls: set[str],
    *,
    url: Any,
    title: Any,
    snippet: Any,
    limit: int,
) -> None:
    if len(sources) >= limit or not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return
    if url in seen_urls:
        return
    seen_urls.add(url)
    hostname = urlsplit(url).hostname or url
    sources.append(
        {
            "title": (str(title).strip() if title else hostname)[:MAX_SOURCE_TITLE_CHARACTERS],
            "url": url,
            "snippet": (str(snippet).strip() if snippet else "")[:MAX_SNIPPET_CHARACTERS],
        }
    )


def _annotation_snippet(text: str, annotation: Any) -> str:
    start = _field(annotation, "start_index")
    end = _field(annotation, "end_index")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text):
        return text[start:end]
    return ""


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []
