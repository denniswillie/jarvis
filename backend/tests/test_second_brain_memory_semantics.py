"""The native memory harness must not repair evidence with its frozen oracle."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.knowledge.models import SourceError, SourceVersion
from evals.second_brain.hindsight_semantic_trace import (
    EXPECTED_SOURCES, PausedNativeRetainClient, exercise, load_corpus, publish_case, run_case,
)
from evals.second_brain.memory_answers import AnswerControlError, NativeAnswerControl, answer_request, parse_answer


def source(alias="coding-01"):
    return SourceVersion(
        owner_id="fixture-owner", area_id=uuid4(), source_id=uuid4(), revision=1, generation=1,
        payload_id=uuid4(), document_id="fixture-document", bank_name="fixture-bank",
        content_digest="a" * 64, source_type="user_report", trust="user_input",
        source_time=datetime(2026, 9, 10, 18, tzinfo=UTC), time_basis="frozen fixture",
        recorded_at=datetime(2026, 9, 14, tzinfo=UTC),
        content={"fixture_source_id": alias, "text": "The function is incomplete.", "synthetic": True},
        is_current=True, ingestion_status="searchable",
    )


def native_response(answer=None, **changes):
    result = answer or {"answer": "The evidence is incomplete.", "source_ids": ["source-one"], "uncertainties": []}
    return {
        "id": "resp-fixture", "model": "gpt-5.4-mini-fixture", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": json.dumps(result)}]}],
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        **changes,
    }


class NativeAnswerControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_request_has_only_question_and_actual_evidence(self):
        requests = []
        async def receive(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=native_response())
        control = NativeAnswerControl("fixture-secret", transport=httpx.MockTransport(receive))
        self.addAsyncCleanup(control.close)
        result = await control.answer("What happened?", {"source": "actual fixture evidence"})
        sent = requests[0]
        self.assertFalse(sent["store"])
        self.assertEqual(sent["model"], "gpt-5.4-mini")
        self.assertTrue(sent["text"]["format"]["strict"])
        self.assertEqual(json.loads(sent["input"]), {
            "question": "What happened?", "retrieved_original_evidence": {"source": "actual fixture evidence"},
        })
        self.assertEqual(result["usage"]["total_tokens"], 30)
        self.assertNotIn("semantic_grade", result)
        self.assertNotIn("oracle", sent["input"])
        self.assertNotIn("fixture-secret", str(result))

    async def test_http_error_does_not_expose_response_body_or_credentials(self):
        control = NativeAnswerControl(
            "private-key", transport=httpx.MockTransport(
                lambda request: httpx.Response(400, text="private-body-and-key")),
        )
        self.addAsyncCleanup(control.close)
        with self.assertRaises(AnswerControlError) as caught:
            await control.answer("fixture", {})
        self.assertEqual(str(caught.exception), "answer_control_http_400")

    async def test_provider_response_size_is_bounded(self):
        control = NativeAnswerControl(
            "fixture", transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"x" * 1_000_001)),
        )
        self.addAsyncCleanup(control.close)
        with self.assertRaises(AnswerControlError) as caught:
            await control.answer("fixture", {})
        self.assertEqual(caught.exception.code, "answer_control_response_too_large")

    def test_incomplete_refused_missing_and_invalid_answers_are_not_success(self):
        refusal = native_response(output=[{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "refusal": "No"}],
        }])
        cases = [
            native_response(status="incomplete"), native_response(output=[]), refusal,
            native_response({"answer": "", "source_ids": [], "uncertainties": []}),
            native_response({"answer": "fixture", "source_ids": [1], "uncertainties": []}),
            native_response({"answer": "fixture", "source_ids": [], "uncertainties": [], "passed": True}),
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises((AnswerControlError, ValueError)):
                parse_answer(value)


class FrozenMemoryHarnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.source = source()
        self.search = {"status": "success", "results": [{
            "memory_id": str(self.source.source_id), "source_id": str(self.source.source_id),
            "revision": 1, "content_digest": self.source.content_digest, "content": self.source.content,
        }]}
        self.original = {
            "status": "success", "memory_id": str(self.source.source_id),
            "source": self.source.original_evidence() | {"content": None},
            "content_page": {"complete": True, "text": json.dumps(self.source.content)},
        }
        self.tools = {
            "search_memory": SimpleNamespace(handler=AsyncMock(return_value=self.search)),
            "explain_memory": SimpleNamespace(handler=AsyncMock(return_value=self.original)),
        }
        self.answers = SimpleNamespace(answer=AsyncMock(return_value={
            "answer": "It remains incomplete.", "source_ids": [str(self.source.source_id)], "uncertainties": [],
        }))
        self.question = {"id": "M01", "query": "Was the function completed?", "oracle": "GOLD_ORACLE_NOT_MODEL_INPUT"}

    async def run_case(self, **changes):
        return await run_case(self.question | changes, self.tools, {"coding-01": self.source}, self.answers)

    def test_corpus_preserves_all_six_frozen_cases_and_nine_originals(self):
        corpus, digest = load_corpus()
        self.assertTrue(corpus["synthetic"])
        self.assertEqual(len(corpus["sources"]), 9)
        self.assertEqual({question["id"] for question in corpus["questions"]}, set(EXPECTED_SOURCES))
        self.assertEqual(len(digest), 64)
        self.assertIn("no observation or user confirmation", next(
            item["text"] for item in corpus["sources"] if item["id"] == "dose-01"))

    async def test_integrity_success_never_self_grades_or_sends_the_oracle(self):
        result = await self.run_case()
        self.assertTrue(all(result["checks"].values()))
        self.assertIsNone(result["semantic_grade"])
        self.assertNotIn(self.question["oracle"], json.dumps(self.answers.answer.await_args.args))
        self.assertEqual(self.answers.answer.await_args.args[0], self.question["query"])
        self.tools["search_memory"].handler.assert_awaited_once_with({"query": self.question["query"]})

    async def test_missing_required_evidence_is_reported_not_filled_from_bindings(self):
        result = await self.run_case(id="M04", query="What are the cafe and robotics commitments?")
        self.assertFalse(result["checks"]["required_source_coverage"])
        sent = self.answers.answer.await_args.args[1]
        self.assertNotIn("business-01", json.dumps(sent))
        self.assertEqual(len(sent["explain_memory"]), 1)

    async def test_unknown_or_changed_original_prevents_answer_generation(self):
        self.original["source"]["content_digest"] = "b" * 64
        result = await self.run_case()
        self.assertEqual(result["error"]["code"], "untrusted_original_expansion")
        self.answers.answer.assert_not_awaited()

    async def test_partial_expansion_and_failed_search_are_not_replaced_by_fixture_text(self):
        self.original["content_page"]["complete"] = False
        result = await self.run_case()
        self.assertEqual(result["error"]["code"], "frozen_original_expansion_incomplete")
        self.answers.answer.assert_not_awaited()
        self.tools["search_memory"].handler.return_value = {"status": "failed"}
        result = await self.run_case()
        self.assertEqual(result["error"]["code"], "shared_search_failed")
        self.answers.answer.assert_not_awaited()

    async def test_fabricated_answer_reference_is_a_failed_structural_check(self):
        self.answers.answer.return_value["source_ids"] = ["made-up"]
        result = await self.run_case()
        self.assertFalse(result["checks"]["answer_reference_integrity"])
        self.assertIsNone(result["semantic_grade"])

    async def test_correction_uses_one_logical_source_and_preserves_origin(self):
        corpus, _ = load_corpus()
        fixture = SimpleNamespace(publish_payload=AsyncMock())
        correction = next(item for item in corpus["sources"] if item["id"] == "gym-02")
        await publish_case(fixture, correction, expected_revision=1)
        args = fixture.publish_payload.await_args
        self.assertEqual(args.kwargs["source_key"], "gym-01")
        self.assertEqual(args.kwargs["expected_revision"], 1)
        self.assertEqual(args.kwargs["trust"], "user_input")
        self.assertEqual(args.args[0]["text"], correction["text"])
        self.assertEqual(args.args[0]["fixture_source_id"], "gym-02")
        self.assertIsNotNone(args.kwargs["source_time"].tzinfo)

    async def test_fault_injection_pauses_before_the_real_native_send(self):
        client = PausedNativeRetainClient("http://127.0.0.1:19000", "fixture")
        self.addAsyncCleanup(client.close)
        client.paused_source = self.source.source_id
        operation = uuid4()
        with patch("app.knowledge.hindsight.HindsightClient.retain", new_callable=AsyncMock) as native:
            native.return_value = {"operation_id": str(operation), "async": True}
            pending = asyncio.create_task(client.retain(self.source, operation))
            try:
                await asyncio.wait_for(client.entered.wait(), timeout=1)
                native.assert_not_awaited()
                self.assertEqual(client.retains[self.source.document_id], 0)
                client.release.set()
                await pending
                native.assert_awaited_once_with(self.source, operation)
                self.assertEqual(client.retains[self.source.document_id], 1)
            finally:
                client.release.set()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_early_fixture_failure_closes_both_clients_and_keeps_failure_visible(self):
        corpus, digest = load_corpus()
        fixture = SimpleNamespace(
            sources=object(), owner="fixture-owner", area=uuid4(),
            publish_payload=AsyncMock(side_effect=SourceError("fixture_publication_stopped", "Synthetic stop")),
        )
        cleanup = []
        @asynccontextmanager
        async def owned_fixture(*args, **kwargs):
            try:
                yield fixture
            finally:
                cleanup.append(True)
        client = SimpleNamespace(close=AsyncMock())
        answers = SimpleNamespace(close=AsyncMock())
        with patch("evals.second_brain.hindsight_semantic_trace.source_fixture", owned_fixture), \
             patch("evals.second_brain.hindsight_semantic_trace.PausedNativeRetainClient", return_value=client), \
             patch("evals.second_brain.hindsight_semantic_trace.NativeAnswerControl", return_value=answers), \
             patch("evals.second_brain.hindsight_semantic_trace.save"):
            result = await exercise(Path("/synthetic"), {"url": "fixture", "token": "fixture"},
                                    SimpleNamespace(SUPABASE_DB_URL="fixture", OPENAI_API_KEY="fixture"),
                                    corpus, digest)
        self.assertEqual(result["error"]["code"], "fixture_publication_stopped")
        self.assertFalse(result["stage_accepted"])
        self.assertIsNone(result["semantic_grade"])
        self.assertEqual(cleanup, [True])
        client.close.assert_awaited_once()
        answers.close.assert_awaited_once()


class ExpectedRejectionDrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_the_exact_expected_failed_record_can_be_terminal(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from uuid import uuid4
        from evals.second_brain.hindsight_semantic_trace import drain
        rejected, unexpected = uuid4(), uuid4()
        fixture = SimpleNamespace(owner="synthetic-owner")
        worker = SimpleNamespace(step=AsyncMock(return_value="done"))
        for failed, allowed in ((rejected, True), (unexpected, False)):
            rows = [{"id": failed, "status": "failed"}, {"id": uuid4(), "status": "done"}]
            with self.subTest(failed=failed), patch("evals.second_brain.hindsight_semantic_trace.queue_rows",
                                                  new=AsyncMock(return_value=rows)):
                if allowed:
                    self.assertEqual(await drain(fixture, worker, {"worker_events": []}, label="test",
                                                 allowed_failed_ids={rejected}), rows)
                else:
                    from evals.second_brain.hindsight_trace import NativeTraceFailure
                    with self.assertRaises(NativeTraceFailure) as caught:
                        await drain(fixture, worker, {"worker_events": []}, label="test", allowed_failed_ids={rejected})
                    self.assertEqual(caught.exception.code, "native_outbox_failed")
