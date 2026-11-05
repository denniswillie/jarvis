from __future__ import annotations

import json
import math
import unittest

import httpx

from app.memory.models import EMBEDDING_DIMENSIONS
from app.model_providers.gemini_multimodal_embeddings import (
    GeminiMultimodalEmbeddingConfig,
    GeminiMultimodalEmbeddingError,
    GeminiMultimodalEmbeddingModel,
)
from app.model_providers.protocols import VisualModelImage


_JPEG = b"\xff\xd8\xff\xe0image\xff\xd9"


class GeminiMultimodalEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_request_uses_configured_dimension_and_model(self) -> None:
        requests: list[httpx.Request] = []
        config = GeminiMultimodalEmbeddingConfig(api_key="test-key")
        model, client = _model_with_responses(
            requests,
            [_success_response()],
            config=config,
        )

        embedding = await model.embed_multimodal(text="A quiet cafe", images=())

        self.assertEqual(len(embedding), EMBEDDING_DIMENSIONS)
        request = requests[0]
        self.assertTrue(request.url.path.endswith("/models/gemini-embedding-2:embedContent"))
        self.assertEqual(request.headers["x-goog-api-key"], "test-key")
        payload = json.loads(request.content)
        self.assertEqual(payload["outputDimensionality"], EMBEDDING_DIMENSIONS)
        self.assertEqual(payload["content"]["parts"], [{"text": "A quiet cafe"}])
        self.assertNotIn("test-key", repr(config))
        await client.aclose()

    async def test_text_and_images_are_aggregated_in_one_request(self) -> None:
        requests: list[httpx.Request] = []
        model, client = _model_with_responses(requests, [_success_response()])

        await model.embed_multimodal(
            text="Event summary",
            images=(
                VisualModelImage("first", _JPEG, "image/jpeg"),
                VisualModelImage("second", _JPEG, "image/jpeg"),
            ),
        )

        payload = json.loads(requests[0].content)
        parts = payload["content"]["parts"]
        self.assertEqual(parts[0], {"text": "Event summary"})
        self.assertEqual(parts[1], {"text": "Image 1: first"})
        self.assertEqual(parts[3], {"text": "Image 2: second"})
        self.assertEqual(parts[2]["inlineData"]["mimeType"], "image/jpeg")
        await client.aclose()

    async def test_transient_http_failure_is_retried(self) -> None:
        requests: list[httpx.Request] = []
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)

        model, client = _model_with_responses(
            requests,
            [httpx.Response(503, text="busy"), _success_response()],
            sleep=sleep,
        )

        await model.embed_multimodal(text="retry", images=())

        self.assertEqual(len(requests), 2)
        self.assertEqual(sleeps, [0.25])
        await client.aclose()

    async def test_permanent_http_failure_is_not_retried(self) -> None:
        requests: list[httpx.Request] = []
        model, client = _model_with_responses(
            requests,
            [httpx.Response(400, text="invalid")],
        )

        with self.assertRaisesRegex(GeminiMultimodalEmbeddingError, "HTTP 400"):
            await model.embed_multimodal(text="invalid", images=())

        self.assertEqual(len(requests), 1)
        await client.aclose()

    async def test_invalid_vectors_are_rejected(self) -> None:
        requests: list[httpx.Request] = []
        model, client = _model_with_responses(
            requests,
            [_success_response([math.nan] * EMBEDDING_DIMENSIONS)],
        )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            await model.embed_multimodal(text="invalid vector", images=())

        await client.aclose()

    async def test_input_is_bounded_and_requires_real_jpegs(self) -> None:
        requests: list[httpx.Request] = []
        model, client = _model_with_responses(requests, [])

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            await model.embed_multimodal(text=None, images=())
        with self.assertRaisesRegex(ValueError, "at most 6"):
            await model.embed_multimodal(
                text="too many",
                images=tuple(
                    VisualModelImage(str(index), _JPEG, "image/jpeg")
                    for index in range(7)
                ),
            )
        with self.assertRaisesRegex(ValueError, "not a JPEG"):
            await model.embed_multimodal(
                text=None,
                images=(VisualModelImage("bad", b"not-jpeg", "image/jpeg"),),
            )

        self.assertEqual(requests, [])
        await client.aclose()


def _success_response(values: list[float] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        content=json.dumps(
            {
                "embedding": {
                    "values": values or [0.01] * EMBEDDING_DIMENSIONS,
                }
            },
            allow_nan=True,
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _model_with_responses(
    requests: list[httpx.Request],
    responses: list[httpx.Response],
    *,
    sleep=None,
    config: GeminiMultimodalEmbeddingConfig | None = None,
) -> tuple[GeminiMultimodalEmbeddingModel, httpx.AsyncClient]:
    queued = list(responses)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return queued.pop(0)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    kwargs = {"client": client}
    if sleep is not None:
        kwargs["sleep"] = sleep
    model = GeminiMultimodalEmbeddingModel(
        config or GeminiMultimodalEmbeddingConfig(api_key="test-key"),
        **kwargs,
    )
    return model, client
