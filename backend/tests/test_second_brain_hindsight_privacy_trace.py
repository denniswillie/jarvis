"""The native privacy probe cannot count an unproven negative as deletion."""
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.hindsight import HindsightClient, HindsightError, document_metadata, original_document
from app.knowledge.models import SourceVersion
from evals.second_brain.hindsight_privacy_trace import (
    PrivacyProbeFailure, inspect_deletion, operation_tree_with_payload, operation_with_payload, require_positive,
)


class NativePrivacyProbeTests(unittest.IsolatedAsyncioTestCase):
    def source(self, marker):
        now = datetime(2026, 9, 14, 12, tzinfo=UTC)
        return SourceVersion(
            owner_id="fixture-owner", area_id=uuid4(), source_id=uuid4(), revision=1, generation=1,
            payload_id=uuid4(), document_id="fixture-" + uuid4().hex, bank_name="fixture-bank",
            content_digest="a" * 64, source_type="experience", trust="user_input", source_time=now,
            time_basis="synthetic report time", recorded_at=now, content={"text": marker},
            is_current=True, ingestion_status="pending",
        )

    def setup_client(self):
        source, control = self.source("ORCHID-FIXTURE"), self.source("LANTERN-FIXTURE")
        docs = {
            item.document_id: dict(
                id=item.document_id, bank_id=item.bank_name, original_text=original_document(item),
                document_metadata=document_metadata(item),
            ) for item in (source, control)
        }
        operation_id = uuid4()
        operation = {
            "operation_id": str(operation_id), "status": "completed",
            "task_payload": {"original": original_document(source)},
        }
        async def document(bank, identity):
            return docs.get(identity)
        async def delete_document(bank, identity):
            docs.pop(identity, None)
        client = SimpleNamespace(
            document=AsyncMock(side_effect=document), delete_document=AsyncMock(side_effect=delete_document),
            verify_document=HindsightClient.verify_document, _bank=HindsightClient._bank,
            _call=AsyncMock(return_value=operation),
        )
        local = AsyncMock(return_value={
            "canonical_read_barrier_deleted": True, "canonical_original_unavailable": True,
            "vendor_cleanup_not_prematurely_complete": True,
        })
        return client, source, control, operation_id, operation, docs, local

    async def test_document_disappearance_does_not_hide_remaining_operation_original(self):
        client, source, control, op, _, _, local = self.setup_client()
        result = await inspect_deletion(
            client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local,
        )
        self.assertTrue(result["checks"]["target_document_removed"])
        self.assertTrue(result["checks"]["unrelated_control_preserved"])
        self.assertFalse(result["checks"]["operation_payload_removed"])
        self.assertFalse(result["document_operation_scope_passed"])
        self.assertFalse(result["full_privacy_acceptance"])
        local.assert_awaited_once()

    async def test_missing_operation_positive_control_prevents_deletion(self):
        client, source, control, op, operation, _, local = self.setup_client()
        operation["task_payload"] = None
        with self.assertRaises(PrivacyProbeFailure):
            await inspect_deletion(client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local)
        local.assert_not_awaited()
        client.delete_document.assert_not_awaited()

    async def test_missing_document_positive_control_prevents_deletion(self):
        client, source, control, op, _, docs, local = self.setup_client()
        docs.pop(source.document_id)
        with self.assertRaises(HindsightError):
            await inspect_deletion(client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local)
        local.assert_not_awaited()
        client.delete_document.assert_not_awaited()

    async def test_unproven_local_read_barrier_prevents_remote_deletion_claim(self):
        client, source, control, op, _, _, local = self.setup_client()
        local.return_value["canonical_original_unavailable"] = False
        with self.assertRaises(PrivacyProbeFailure):
            await inspect_deletion(client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local)
        client.delete_document.assert_not_awaited()

    async def test_unknown_operation_shape_is_not_counted_as_erasure(self):
        client, _, _, op, _, _, _ = self.setup_client()
        client._call.return_value = {"status": "completed"}
        with self.assertRaises(PrivacyProbeFailure):
            await operation_with_payload(client, "fixture-bank", op)

    async def test_child_payload_is_checked_even_when_parent_disappears_after_deletion(self):
        client, source, control, op, operation, docs, local = self.setup_client()
        child_id = str(uuid4())
        child = dict(operation, operation_id=child_id)
        operation["task_payload"] = None
        operation["child_operations"] = [{"operation_id": child_id, "status": "completed"}]
        async def call(method, path, **kwargs):
            if child_id in path:
                return child
            return operation if source.document_id in docs else None
        client._call.side_effect = call
        result = await inspect_deletion(
            client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local,
        )
        self.assertFalse(result["checks"]["operation_payload_removed"])
        self.assertEqual(result["operation_payload_evidence"]["positive_payload_owner_ids"], [child_id])
        self.assertIn(child_id, result["operation_payload_evidence"]["checked_after"])

    async def test_confirmed_child_payload_removal_can_pass_only_bounded_scope(self):
        client, source, control, op, operation, docs, local = self.setup_client()
        child_id = str(uuid4())
        child = dict(operation, operation_id=child_id)
        operation["task_payload"] = None
        operation["child_operations"] = [{"operation_id": child_id, "status": "completed"}]
        async def call(method, path, **kwargs):
            identity = child_id if child_id in path else str(op)
            if source.document_id not in docs:
                return {"operation_id": identity, "status": "not_found"}
            return child if identity == child_id else operation
        client._call.side_effect = call
        result = await inspect_deletion(
            client, source, control, op, "ORCHID-FIXTURE", "LANTERN-FIXTURE", local,
        )
        self.assertTrue(result["document_operation_scope_passed"])
        self.assertFalse(result["full_privacy_acceptance"])

    async def test_child_inventory_is_bounded_and_invalid_references_fail_closed(self):
        client, source, _, op, operation, _, _ = self.setup_client()
        operation["child_operations"] = [{"operation_id": "../foreign", "status": "completed"}]
        with self.assertRaises(PrivacyProbeFailure):
            await operation_tree_with_payload(client, source.bank_name, op)
        operation["child_operations"] = [{"operation_id": str(uuid4()), "status": "completed"}]
        with self.assertRaises(PrivacyProbeFailure):
            await operation_tree_with_payload(client, source.bank_name, op, max_operations=1)

    def test_empty_truthy_or_missing_checks_are_not_positive_controls(self):
        for checks in ({}, {"control": False}, {"control": None}, {"control": 1}):
            with self.subTest(checks=checks), self.assertRaises(PrivacyProbeFailure):
                require_positive(checks)
        require_positive({"control": True})
