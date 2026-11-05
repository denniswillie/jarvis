from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.model_providers.protocols import VisualModelImage
from app.visual_memory.gemini_client import (
    GeminiVisualAnalyzerClient,
    GeminiVisualAnalyzerError,
)


class GeminiFastVisualModelError(RuntimeError):
    pass


class _RequestStoreHTTPClient:
    """Opt-in request field, reusing the existing HTTP and response paths."""

    def __init__(self, delegate, store: bool) -> None:
        self.delegate, self.store = delegate, store

    async def post(self, url, **kwargs):
        body = kwargs.get("json")
        if not isinstance(body, dict):
            raise ValueError("Gemini store option requires a JSON request")
        if "store" in body and body["store"] is not self.store:
            raise ValueError("Conflicting Gemini request store option")
        kwargs["json"] = {**body, "store": self.store}
        return await self.delegate.post(url, **kwargs)

    def __getattr__(self, name):
        return getattr(self.delegate, name)


class GeminiFastVisualModel:
    """Temporal visual-model adapter sharing the existing Gemini HTTP client."""

    provider = "google_gemini"

    def __init__(
        self, client: GeminiVisualAnalyzerClient, *, store: bool | None = None,
    ) -> None:
        if store is not None:
            if type(store) is not bool:
                raise ValueError("Gemini store option must be bool or None")
            # This adapter already owns a package-internal Gemini client. Opt-in
            # affects that client, so evaluation callers supply a dedicated one.
            # None leaves legacy HTTP requests and shared production clients intact.
            current = client._client
            if isinstance(current, _RequestStoreHTTPClient):
                if current.store is not store:
                    raise ValueError("Conflicting Gemini client store options")
            else:
                client._client = _RequestStoreHTTPClient(current, store)
        self._client = client

    @property
    def model(self) -> str:
        return self._client.model

    async def generate_temporal_json(
        self,
        *,
        prompt: str,
        images: Sequence[VisualModelImage],
        response_json_schema: dict[str, object] | None = None,
    ) -> str:
        return await self._client.generate_multiframe_json(
            prompt=prompt,
            images=images,
            **({"response_json_schema": response_json_schema} if response_json_schema is not None else {}),
        )

    async def generate_historical_evidence_json(
        self,
        *,
        prompt: str,
        images: Sequence[VisualModelImage],
    ) -> str:
        try:
            return await self._client.generate_multiframe_json(
                prompt=prompt,
                images=images,
            )
        except (GeminiVisualAnalyzerError, httpx.HTTPError, ValueError) as error:
            raise GeminiFastVisualModelError(
                "Gemini historical evidence inspection failed"
            ) from error
