"""Offline contract tests: MockTransport, no providers, database, or Docker."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import unittest
from unittest.mock import AsyncMock

import httpx

from app.shared_memory import (
    BankScope, DialogueBatcher, DialogueTurn, MemoryDocument, MemoryRequestError,
    NativeMemory, ObservationDelta, retain_observation,
)
from app.shared_memory.client import MENTAL_MODELS


NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)


class MemoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.handler = lambda request: httpx.Response(200, json={"success": True, "async": True, "operation_id": "op"})

        def dispatch(request):
            self.requests.append(request)
            return self.handler(request)

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(dispatch))
        self.scope = BankScope.for_owner("owner", generation="new-demo")
        self.memory = NativeMemory(scope=self.scope, base_url="http://memory.test", api_key="test-secret", http_client=self.http)

    async def asyncTearDown(self):
        await self.http.aclose()

    def payload(self, index=-1):
        return json.loads(self.requests[index].content)

    def turn(self, source="t1", speaker="user", when=NOW):
        return DialogueTurn(source, speaker, "I put my keys in the drawer.", when, when + timedelta(seconds=2))

    def batcher(self, **kwargs):
        self.clock = [0.0]
        self.checkpoint = AsyncMock()
        return DialogueBatcher(self.memory, "session", checkpoint=self.checkpoint,
                               clock=lambda: self.clock[0], **kwargs)

    def test_bank_generation_and_owner_partition(self):
        self.assertEqual(self.scope, BankScope.for_owner("owner", generation="new-demo"))
        self.assertNotEqual(self.scope.bank_id, BankScope.for_owner("other", generation="new-demo").bank_id)
        self.assertNotEqual(self.scope.bank_id, BankScope.for_owner("owner", generation="later-cutover").bank_id)
        with self.assertRaises(ValueError):
            BankScope.for_owner("owner", generation="")

    async def test_fresh_bank_refuses_existing(self):
        with self.assertRaisesRegex(MemoryRequestError, "bank_already_exists"):
            await self.memory.create_fresh_bank()
        self.assertEqual([r.method for r in self.requests], ["GET"])

    async def test_fresh_bank_creates_no_backfill(self):
        self.handler = lambda r: httpx.Response(404) if r.method == "GET" else httpx.Response(200, json={"bank_id": self.scope.bank_id})
        await self.memory.create_fresh_bank()
        self.assertEqual([r.method for r in self.requests], ["GET", "PUT"])
        self.assertTrue(self.payload()["enable_observations"])

    async def test_native_recall_and_reflect(self):
        await self.memory.recall("Where were my keys?", at=NOW)
        self.assertTrue(self.requests[-1].url.path.endswith("/memories/recall"))
        self.assertEqual(self.payload()["query_timestamp"], NOW.isoformat())
        self.assertIn("world", self.payload()["types"])
        await self.memory.reflect("What routines have I described?", context="A direct user question")
        self.assertTrue(self.requests[-1].url.path.endswith("/reflect"))
        self.assertEqual(self.payload()["context"], "A direct user question")
        self.assertTrue(all(self.scope.bank_id in r.url.path for r in self.requests))

    async def test_redirect_not_followed_or_credential_leaked(self):
        self.handler = lambda r: httpx.Response(307, headers={"Location": "https://other.test"})
        with self.assertRaises(MemoryRequestError) as error:
            await self.memory.recall("keys")
        self.assertEqual(error.exception.status, 307)
        self.assertEqual(len(self.requests), 1)
        self.assertNotIn("test-secret", str(error.exception))

    async def test_false_success_not_accepted(self):
        self.handler = lambda r: httpx.Response(200, json={"success": False, "error": "private source"})
        with self.assertRaisesRegex(MemoryRequestError, "rejected_request"):
            await self.memory.recall("keys")

    async def test_seed_three_native_models_once(self):
        existing = set()

        def handle(request):
            if request.method == "GET":
                model = request.url.path.rsplit("/", 1)[-1]
                return httpx.Response(200, json={"id": model}) if model in existing else httpx.Response(404)
            body = json.loads(request.content)
            existing.add(body["id"])
            self.assertTrue(body["trigger"]["refresh_after_consolidation"])
            self.assertTrue(body["trigger"]["exclude_mental_models"])
            return httpx.Response(200, json={"mental_model_id": body["id"], "operation_id": "pending"})

        self.handler = handle
        first = await self.memory.seed_mental_models()
        self.assertEqual(set(first), set(MENTAL_MODELS))
        self.assertEqual(first["core_values_and_tradeoffs"]["operation_id"], "pending")
        second = await self.memory.seed_mental_models()
        self.assertTrue(all(value["existing"] for value in second.values()))
        self.assertEqual(sum(r.method == "POST" for r in self.requests), 3)

    async def test_native_document_correction_uses_same_id_replace(self):
        await self.memory.retain([MemoryDocument("doc", "Stopped work, not completed.", NOW)])
        await self.memory.retain([MemoryDocument("doc", "Completed the work the next day.", NOW + timedelta(days=1))])
        self.assertEqual(self.payload(0)["items"][0]["document_id"], self.payload(1)["items"][0]["document_id"])
        self.assertEqual(self.payload(1)["items"][0]["update_mode"], "replace")

    async def test_forget_partial_remote_then_retry(self):
        invalidation = AsyncMock()
        self.handler = lambda r: httpx.Response(503) if r.url.path.endswith("/b") else httpx.Response(200, json={"success": True})
        result = await self.memory.forget(["a", "b", "a"], invalidate=invalidation)
        self.assertEqual(result.deleted_document_ids, ("a",))
        self.assertEqual(list(result.failed_documents), ["b"])
        self.assertFalse(result.requested_deletions_applied)
        self.assertTrue(result.local_invalidated)
        self.assertEqual(invalidation.call_args.args[0].owner_id, "owner")
        self.handler = lambda r: httpx.Response(404) if r.url.path.endswith("/a") else httpx.Response(200, json={"success": True})
        result = await self.memory.forget(["a", "b"], invalidate=invalidation)
        self.assertTrue(result.requested_deletions_applied)
        self.assertEqual(result.derived_cleanup, "native_async_unverified")

    async def test_forget_local_failure_is_not_success(self):
        invalidation = AsyncMock(side_effect=RuntimeError("private state"))
        result = await self.memory.forget(["a"], invalidate=invalidation)
        self.assertFalse(result.requested_deletions_applied)
        self.assertEqual(result.local_error, "local_invalidation_failed")
        self.assertNotIn("private state", repr(result))

    async def test_document_target_is_url_encoded(self):
        await self.memory.get_document("a/b?x")
        self.assertIn(b"a%2Fb%3Fx", self.requests[-1].url.raw_path)

    async def test_batch_waits_for_settle_and_preserves_speakers_times(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())
        self.clock[0] = 17
        self.assertIsNone(await batcher.flush_due())
        batcher.enqueue(self.turn("t2", "assistant", NOW + timedelta(seconds=4)))
        self.clock[0] = 34
        self.assertIsNone(await batcher.flush_due())
        self.clock[0] = 35
        await batcher.flush_due()
        items = self.payload()["items"]
        self.assertEqual(len(items), 2)
        self.assertIn("User:", items[0]["content"])
        self.assertIn("GPT-Live:", items[1]["content"])
        self.assertEqual(items[0]["timestamp"], NOW.isoformat())
        self.checkpoint.assert_awaited_once_with(("t1", "t2"))
        self.assertEqual(batcher.pending_count, 0)

    async def test_close_flushes_without_waiting(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())
        await batcher.close()
        self.assertEqual(len(self.requests), 1)
        with self.assertRaises(RuntimeError):
            batcher.enqueue(self.turn("new"))

    async def test_only_active_finalized_dialogue(self):
        batcher = self.batcher()
        self.assertFalse(batcher.enqueue(self.turn(), active=False))
        self.assertFalse(batcher.enqueue(DialogueTurn("t1", "user", "partial", NOW, NOW, finalized=False)))
        self.assertTrue(batcher.enqueue(self.turn()))
        self.assertFalse(batcher.enqueue(self.turn()))
        await batcher.close()
        self.assertEqual(len(self.payload()["items"]), 1)

    async def test_failed_http_preserves_pending(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())
        self.handler = lambda r: httpx.Response(503)
        with self.assertRaises(MemoryRequestError):
            await batcher.close()
        self.assertEqual(batcher.pending_count, 1)
        self.checkpoint.assert_not_awaited()
        self.handler = lambda r: httpx.Response(200, json={"success": True})
        await batcher.close()
        self.assertEqual(self.payload(0), self.payload(1))

    async def test_checkpoint_failure_retry_keeps_same_documents(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())
        self.checkpoint.side_effect = RuntimeError("checkpoint unavailable")
        with self.assertRaises(RuntimeError):
            await batcher.close()
        self.checkpoint.side_effect = None
        await batcher.close()
        self.assertEqual(self.payload(0), self.payload(1))

    async def test_restored_checkpoint_skips_replayed_source(self):
        batcher = self.batcher(accepted_source_ids=("t1",))
        self.assertFalse(batcher.enqueue(self.turn()))
        self.assertIsNone(await batcher.close())
        self.assertFalse(self.requests)

    async def test_ids_independent_of_batch_boundaries_and_months(self):
        original = self.turn().document("session")
        batcher = self.batcher()
        batcher.enqueue(self.turn("older", when=NOW - timedelta(days=90)))
        batcher.enqueue(self.turn())
        await batcher.close()
        self.assertEqual(self.payload()["items"][1]["document_id"], original.document_id)
        self.assertEqual(self.payload()["items"][0]["timestamp"], (NOW - timedelta(days=90)).isoformat())

    async def test_concurrent_flushes_do_not_send_twice(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())
        await asyncio.gather(batcher.flush_due(force=True), batcher.flush_due(force=True))
        self.assertEqual(len(self.requests), 1)

    async def test_new_turn_during_retain_not_lost(self):
        batcher = self.batcher()
        batcher.enqueue(self.turn())

        def handle(request):
            batcher.enqueue(self.turn("t2"))
            return httpx.Response(200, json={"success": True})

        self.handler = handle
        await batcher.flush_due(force=True)
        self.assertEqual(batcher.pending_count, 1)

    async def test_observation_null_skips_and_nonnull_keeps_original_evidence(self):
        self.assertIsNone(await retain_observation(self.memory, None))
        delta = ObservationDelta("t1", "session", "Keys placed in a drawer; drawer label unclear.",
                                 NOW - timedelta(days=1), NOW - timedelta(days=1), NOW, ("frame1",), ("file1",))
        await retain_observation(self.memory, delta)
        item = self.payload()["items"][0]
        self.assertEqual(item["timestamp"], (NOW - timedelta(days=1)).isoformat())
        self.assertEqual(json.loads(item["metadata"]["frame_ids"]), ["frame1"])
        self.assertNotEqual(item["document_id"], self.turn().document("session").document_id)
        self.assertNotIn("GPT-Live:", item["content"])

    async def test_invalid_original_timestamp_rejected_before_network(self):
        with self.assertRaises(ValueError):
            await self.memory.retain([MemoryDocument("doc", "text", NOW.replace(tzinfo=None))])
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
