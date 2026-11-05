from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import httpx
from openai import APIConnectionError, APIStatusError

from app.memory.consolidation.openai_client import (
    ConsolidationProviderContractError,
    ConsolidationProviderTransientError,
    OpenAIStructuredOutputClient,
    strict_json_schema,
)
from app.memory.consolidation.contracts import ConsolidationResult
from app.runtime.concurrency import PriorityAsyncLimiter
from test_consolidation_contracts import _request


class _FailingResponses:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def create(self, **_: object) -> object:
        raise self._error


class _FailingClient:
    def __init__(self, error: Exception) -> None:
        self.responses = _FailingResponses(error)

    async def close(self) -> None:
        return None


class ConsolidationOpenAIClientTests(unittest.IsolatedAsyncioTestCase):
    def test_strict_schema_requires_every_property_recursively(self) -> None:
        schema = strict_json_schema(ConsolidationResult.model_json_schema())

        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        fact_schema = schema["$defs"]["SemanticFactCandidate"]
        self.assertEqual(set(fact_schema["required"]), set(fact_schema["properties"]))
        self.assertIn("change_intent", fact_schema["required"])

    async def test_generate_sends_strict_schema_to_responses_api(self) -> None:
        responses = _RecordingResponses()
        client = OpenAIStructuredOutputClient(
            api_key="test-key",
            model="test-model",
            timeout_seconds=1,
            limiter=PriorityAsyncLimiter(1),
            client=SimpleNamespace(responses=responses, close=lambda: None),  # type: ignore[arg-type]
        )

        await client.generate(_request(), owner_id="owner")

        schema = responses.arguments["text"]["format"]["schema"]
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    async def test_connection_and_server_failures_are_retryable(self) -> None:
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        for error in (
            APIConnectionError(request=request),
            _status_error(500),
        ):
            with self.subTest(error=type(error).__name__):
                client = _client(error)
                with self.assertRaises(ConsolidationProviderTransientError):
                    await client.generate(_request(), owner_id="owner")

    async def test_bad_request_is_terminal(self) -> None:
        client = _client(_status_error(400))

        with self.assertRaises(ConsolidationProviderContractError):
            await client.generate(_request(), owner_id="owner")


    async def test_repair_uses_the_same_catalog_schema_and_limiter_path(self):
        request = _request()
        responses = _RecordingResponses()
        client = OpenAIStructuredOutputClient(
            api_key="test-key", model="test-model", timeout_seconds=1,
            limiter=PriorityAsyncLimiter(1),
            client=SimpleNamespace(responses=responses, close=lambda: None),
        )
        await client.generate(request, owner_id="owner", validation_feedback="safe correction")
        arguments = responses.arguments
        prompt = arguments["input"][0]["content"][0]["text"]
        raw_catalog = prompt.split("\n\n", 1)[1].split(
            "\n\nCorrection required after contract validation:", 1,
        )[0]
        payload = json.loads(raw_catalog)
        source = payload["attached_evidence"][0]
        self.assertEqual(source["evidence_ref_id"], str(request.evidence[0].evidence_ref_id))
        self.assertNotIn("evidence_id", source)
        self.assertIn("safe correction", prompt)
        self.assertTrue(arguments["text"]["format"]["strict"])
        self.assertFalse(arguments["store"])


def _client(error: Exception) -> OpenAIStructuredOutputClient:
    return OpenAIStructuredOutputClient(
        api_key="test-key",
        model="test-model",
        timeout_seconds=1,
        limiter=PriorityAsyncLimiter(1),
        client=_FailingClient(error),  # type: ignore[arg-type]
    )


class _RecordingResponses:
    def __init__(self) -> None:
        self.arguments: dict[str, object] = {}

    async def create(self, **arguments: object) -> object:
        self.arguments = arguments
        return SimpleNamespace(output_text="{}")


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code, request=request)
    return APIStatusError(
        f"OpenAI returned {status_code}",
        response=response,
        body=None,
    )


if __name__ == "__main__":
    unittest.main()
