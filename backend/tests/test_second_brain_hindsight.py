import json
from datetime import UTC, datetime
import unittest
from uuid import uuid4

import httpx

from app.knowledge.hindsight import HindsightClient, HindsightError, original_document, document_metadata
from app.knowledge.models import SourceVersion


def source(**changes):
    return SourceVersion(**(dict(
        owner_id="fixture", area_id=uuid4(), source_id=uuid4(), revision=1, generation=1,
        payload_id=uuid4(), document_id="source-fixture", bank_name="jarvis-fixture",
        content_digest="a"*64, source_type="report", trust="user_input", source_time=None,
        time_basis="not supplied", recorded_at=datetime.now(UTC),
        content={"text":"Fixture original, not a completed action."}, is_current=True, ingestion_status="pending",
    ) | changes))


class HindsightAdapterTests(unittest.IsolatedAsyncioTestCase):
    def client(self, handler, **kwargs):
        client = HindsightClient("https://memory.invalid", "private-fixture-token",
                                 transport=httpx.MockTransport(handler), **kwargs)
        self.addAsyncCleanup(client.close)
        return client

    async def test_async_retain_has_stable_operation_identity_unknown_time_and_original_provenance(self):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            self.assertEqual(request.headers["authorization"], "Bearer private-fixture-token")
            return httpx.Response(200, json={"async":True, "operation_id":body["operation_id"]})
        client = self.client(handler)
        evidence, operation = source(), uuid4()
        await client.retain(evidence, operation)
        await client.retain(evidence, operation)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["items"][0]["timestamp"], "unset")
        self.assertEqual(calls[0]["items"][0]["metadata"], document_metadata(evidence))
        self.assertEqual(calls[0]["items"][0]["content"], original_document(evidence))
        self.assertNotIn("private-fixture-token", repr(client.diagnostics()))
        self.assertIsNone(client.diagnostics()["provider_token_usage"])

    async def test_operation_absence_accepts_native_receipt_and_http_404(self):
        operation = uuid4()
        for response in (
            httpx.Response(200, json={"operation_id": str(operation), "status": "not_found"}),
            httpx.Response(404),
        ):
            with self.subTest(status=response.status_code):
                client = self.client(lambda request: response)
                self.assertIsNone(await client.operation("fixture", operation))

    async def test_operation_status_and_identity_are_validated_before_normalizing_absence(self):
        operation = uuid4()
        for response in (
            {"operation_id": str(uuid4()), "status": "not_found"},
            {"status": "not_found"},
            {"operation_id": str(operation), "status": "unknown"},
            {"operation_id": str(uuid4()), "status": "completed"},
        ):
            with self.subTest(response=response):
                client = self.client(lambda request: httpx.Response(200, json=response))
                with self.assertRaises(HindsightError) as raised:
                    await client.operation("fixture", operation)
                self.assertEqual(raised.exception.code, "hindsight_operation_receipt_mismatch")
        for status in ("pending", "processing", "completed", "failed", "cancelled"):
            with self.subTest(status=status):
                client = self.client(lambda request: httpx.Response(
                    200, json={"operation_id": str(operation), "status": status},
                ))
                self.assertEqual((await client.operation("fixture", operation))["status"], status)

    async def test_foreign_receipts_and_saved_models_are_rejected(self):
        client = self.client(lambda r: httpx.Response(200, json={"async":True,"operation_id":str(uuid4())}))
        with self.assertRaisesRegex(HindsightError, "configured memory service"):
            await client.retain(source(), uuid4())
        client = self.client(lambda r: httpx.Response(200, json={"items":[{"id":"unexpected"}],"total":1}))
        with self.assertRaises(HindsightError) as raised:
            await client.check_bank_policy("fixture")
        self.assertEqual(raised.exception.code, "hindsight_saved_models_not_supported")

    async def test_no_redirects_unbounded_payloads_or_raw_provider_error_disclosure(self):
        client = self.client(lambda r: httpx.Response(302, headers={"location":"https://untrusted.invalid"}))
        with self.assertRaises(HindsightError):
            await client.document("fixture","doc")
        client = self.client(lambda r: httpx.Response(200, content=b"x"*1100), max_response_bytes=1024)
        with self.assertRaises(HindsightError) as raised:
            await client.document("fixture","doc")
        self.assertEqual(raised.exception.code,"hindsight_response_too_large")
        client = self.client(lambda r: httpx.Response(500, text="PRIVATE ORIGINAL"))
        with self.assertRaises(HindsightError) as raised:
            await client.document("fixture","doc")
        self.assertNotIn("PRIVATE ORIGINAL",str(raised.exception))
        self.assertTrue(raised.exception.retryable)

    async def test_original_expansion_rejects_changed_content_and_metadata(self):
        evidence = source()
        document = dict(id=evidence.document_id,bank_id=evidence.bank_name,
                        original_text=original_document(evidence),document_metadata=document_metadata(evidence))
        HindsightClient.verify_document(evidence,document)
        for change in ({"bank_id":"foreign"},{"original_text":"Fabricated success"},{"document_metadata":{}}):
            with self.subTest(change=change), self.assertRaises(HindsightError):
                HindsightClient.verify_document(evidence,document|change)

    def operation_client(self, operations, deletes):
        def handler(request):
            segments = request.url.path.split("/")
            identity = segments[-2] if request.method == "DELETE" else segments[-1]
            if request.method == "DELETE":
                self.assertTrue(request.url.path.endswith("/delete"))
                if identity not in operations:
                    return httpx.Response(404)
                operations.pop(identity)
                deletes.append(identity)
                return httpx.Response(200, json={"success": True, "operation_id": identity, "message": "deleted"})
            return httpx.Response(200, json=operations.get(identity, {"operation_id": identity, "status": "not_found"}))
        return self.client(handler)

    async def test_terminal_inventory_and_cleanup_include_children_and_handle_replay(self):
        root, child, unrelated = (str(uuid4()) for _ in range(3))
        operations = {
            root: {"operation_id": root, "status": "completed",
                   "child_operations": [{"operation_id": child, "status": "completed"}]},
            child: {"operation_id": child, "status": "completed"},
            unrelated: {"operation_id": unrelated, "status": "completed"},
        }
        deleted = []
        client = self.operation_client(operations, deleted)
        inventory = await client.terminal_operations("fixture", root)
        self.assertEqual([str(item) for item in inventory], [root, child])
        self.assertTrue(await client.erase_operations("fixture", inventory))
        self.assertEqual(deleted, [child, root])
        self.assertIn(unrelated, operations)
        self.assertTrue(await client.erase_operations("fixture", inventory))
        self.assertFalse(await client.delete_terminal_operation("fixture", root))

    async def test_unknown_unprepared_operation_is_not_deletion_proof(self):
        client = self.operation_client({}, [])
        with self.assertRaises(HindsightError) as raised:
            await client.terminal_operations("fixture", uuid4())
        self.assertEqual(raised.exception.code, "hindsight_retention_outcome_unknown")

    async def test_active_child_or_scheduled_retry_prevents_erasure(self):
        root, child = str(uuid4()), str(uuid4())
        operations = {
            root: {"operation_id": root, "status": "completed",
                   "child_operations": [{"operation_id": child, "status": "processing"}]},
            child: {"operation_id": child, "status": "processing"},
        }
        deleted = []
        client = self.operation_client(operations, deleted)
        self.assertIsNone(await client.terminal_operations("fixture", root))
        self.assertFalse(await client.erase_operations("fixture", (root, child)))
        operations[child].update(status="failed", next_retry_at="future")
        self.assertIsNone(await client.terminal_operations("fixture", root))
        self.assertFalse(await client.erase_operations("fixture", (root, child)))
        self.assertFalse(deleted)

    async def test_new_children_are_not_silently_deleted_without_durable_inventory(self):
        root, child = str(uuid4()), str(uuid4())
        operations = {root: {"operation_id": root, "status": "completed",
                            "child_operations": [{"operation_id": child, "status": "completed"}]}}
        deleted = []
        client = self.operation_client(operations, deleted)
        with self.assertRaises(HindsightError) as raised:
            await client.erase_operations("fixture", (root,))
        self.assertEqual(raised.exception.code, "hindsight_cleanup_inventory_changed")
        self.assertFalse(deleted)

    async def test_operation_deletion_requires_an_exact_positive_receipt(self):
        identity = uuid4()
        for response in (
            {"operation_id": str(uuid4()), "success": True},
            {"operation_id": str(identity), "success": False},
            {"operation_id": str(identity), "success": 1},
        ):
            with self.subTest(response=response):
                client = self.client(lambda request: httpx.Response(200, json=response))
                with self.assertRaises(HindsightError):
                    await client.delete_terminal_operation("fixture", identity)

    async def test_operation_inventory_is_bounded_and_child_references_are_validated(self):
        for children in (
            "not-a-list", [{}], [{"operation_id": "../foreign", "status": "completed"}],
            [{"operation_id": str(uuid4()), "status": "unknown"}],
            [{"operation_id": str(uuid4()), "status": "completed"} for _ in range(65)],
        ):
            with self.subTest(children=children), self.assertRaises(HindsightError):
                HindsightClient.child_operation_ids({"child_operations": children})
        identities = [str(uuid4()) for _ in range(65)]
        operations = {
            identity: {"operation_id": identity, "status": "completed",
                       "child_operations": ([{"operation_id": identities[i+1], "status": "completed"}]
                                            if i+1 < len(identities) else [])}
            for i, identity in enumerate(identities)
        }
        with self.assertRaises(HindsightError) as raised:
            await self.operation_client(operations, []).terminal_operations("fixture", identities[0])
        self.assertEqual(raised.exception.code, "hindsight_operation_inventory_too_large")

    async def test_plaintext_remote_credentials_and_url_injected_authority_are_refused(self):
        for url in ("http://remote.invalid","https://user:secret@memory.invalid","https://memory.invalid/path"):
            with self.assertRaises(ValueError):
                HindsightClient(url,"secret")

    async def test_effective_bank_privacy_and_original_storage_are_required(self):
        bank = "fixture"
        safe = {"bank_id": bank, "config": {"audit_log_enabled": False, "store_document_text": True}, "overrides": {}}
        for configuration, accepted in (
            (safe, True), (safe | {"bank_id": "foreign"}, False),
            (safe | {"config": {"audit_log_enabled": True, "store_document_text": True}}, False),
            (safe | {"config": {"audit_log_enabled": False, "store_document_text": False}}, False),
            (safe | {"config": {"audit_log_enabled": 0, "store_document_text": True}}, False),
            (safe | {"config": {}}, False),
        ):
            with self.subTest(configuration=configuration):
                calls = []
                def handler(request):
                    calls.append((request.method, request.url.path))
                    if request.url.path.endswith("/mental-models"):
                        return httpx.Response(200, json={"items": [], "total": 0})
                    if request.url.path.endswith("/config"):
                        return httpx.Response(200, json=configuration)
                    return httpx.Response(200, json={"results": []})
                client = self.client(handler)
                if accepted:
                    self.assertEqual((await client.recall(bank, "fixture query"))["results"], [])
                else:
                    with self.assertRaises(HindsightError) as caught:
                        await client.recall(bank, "fixture query")
                    self.assertEqual(caught.exception.code, "hindsight_bank_policy_not_satisfied")
                    self.assertFalse(any(method == "POST" for method, path in calls))

    def recall_client(self, response):
        calls = []
        def handler(request):
            calls.append(request)
            if request.url.path.endswith("/mental-models"):
                return httpx.Response(200, json={"items": [], "total": 0})
            if request.url.path.endswith("/config"):
                return httpx.Response(200, json={
                    "bank_id": "fixture", "config": {"audit_log_enabled": False, "store_document_text": True},
                    "overrides": {},
                })
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/v1/default/banks/fixture/memories/recall")
            return response
        return self.client(handler), calls

    async def test_recall_uses_bounded_native_ranking_without_uncalibrated_positive_floor(self):
        result = {"results": [{"document_id": "source-fixture", "scores": {"semantic": 0.13}}]}
        client, calls = self.recall_client(httpx.Response(200, json=result))
        self.assertEqual(await client.recall("fixture", "coding"), result)
        self.assertEqual(json.loads(calls[-1].content), {
            "query": "coding", "budget": "low", "max_tokens": 2400, "min_scores": {"semantic": 0},
            "include": {"source_facts": {"max_tokens": 2400, "max_tokens_per_observation": 1200}},
        })
        self.assertEqual([request.method for request in calls], ["GET", "GET", "POST"])
        await client.recall("fixture", "coding", max_tokens=128)
        self.assertEqual(json.loads(calls[-1].content)["max_tokens"], 128)
        count = len(calls)
        for budget in (127, 8001, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                await client.recall("fixture", "coding", max_tokens=budget)
        self.assertEqual(len(calls), count)

    async def test_broader_recall_does_not_normalize_failure_or_malformed_results_to_empty_memory(self):
        client, _ = self.recall_client(httpx.Response(200, json={"results": []}))
        self.assertEqual(await client.recall("fixture", "query"), {"results": []})
        for response, code in (
            (httpx.Response(503, text="private provider failure"), "hindsight_http_error"),
            (httpx.Response(200, json={"results": {}}), "hindsight_invalid_recall"),
            (httpx.Response(200, json={}), "hindsight_invalid_recall"),
        ):
            with self.subTest(code=code):
                client, _ = self.recall_client(response)
                with self.assertRaises(HindsightError) as caught:
                    await client.recall("fixture", "query")
                self.assertEqual(caught.exception.code, code)
