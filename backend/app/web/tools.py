from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from app.tools.inputs import (
    ToolInputError,
    failed,
    input_error_payload,
    optional_positive_int,
    required_string,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition
from app.web.fetch import WebFetchError, WebFetcher
from app.web.search import OpenAIWebSearch, WebSearchError


@dataclass(frozen=True)
class WebToolModule:
    fetcher: WebFetcher | None
    search: OpenAIWebSearch | None
    name: ClassVar[str] = "web"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        return create_web_tools(
            owner_id=context.owner_id,
            fetcher=self.fetcher,
            search=self.search,
        )


def create_web_tools(
    *,
    owner_id: str,
    fetcher: WebFetcher | None,
    search: OpenAIWebSearch | None,
) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []

    if fetcher is not None:
        async def fetch_url(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                url = required_string(arguments, "url", max_length=4_096)
            except ToolInputError as error:
                return input_error_payload(error)
            try:
                result = await fetcher.fetch(owner_id=owner_id, url=url)
            except WebFetchError as error:
                return failed(error.code, error.message)
            except Exception:
                return failed("web_fetch_failed", "The page could not be fetched.")
            return success(result)

        tools.append(
            ToolDefinition(
                name="fetch_url",
                description=(
                    "Fetch readable text from one known public HTTP or HTTPS page. Use this after "
                    "web_search when a specific source needs closer reading. Private networks, login "
                    "pages, non-standard ports, large responses, and non-text content are rejected."
                ),
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["url"],
                    "properties": {
                        "url": {"type": "string", "description": "A public HTTP or HTTPS page URL."},
                    },
                },
                handler=fetch_url,
            )
        )

    if search is not None:
        async def web_search(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                query = required_string(arguments, "query", max_length=1_000)
                limit = optional_positive_int(arguments, "limit", default=5, max_value=10)
            except ToolInputError as error:
                return input_error_payload(error)
            try:
                result = await search.search(owner_id=owner_id, query=query, limit=limit)
            except WebSearchError as error:
                return failed(error.code, error.message)
            except Exception:
                return failed("web_search_failed", "Web search could not be completed.")
            return success(result)

        tools.append(
            ToolDefinition(
                name="web_search",
                description=(
                    "Search the live public web for current or outside knowledge. Returns a concise "
                    "answer and bounded source URLs. Call fetch_url separately to inspect one result."
                ),
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string", "description": "A focused web search query."},
                        "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
                    },
                },
                handler=web_search,
            )
        )

    return tools
