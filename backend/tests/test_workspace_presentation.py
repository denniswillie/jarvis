"""Isolated delivery-contract checks; no database/provider/device operations."""
from datetime import UTC, datetime, timedelta
from contextlib import asynccontextmanager
import unittest
from uuid import uuid4

from pydantic import ValidationError

from app.errors import JarvisHTTPError
from app.knowledge.models import CanonicalOwnerBusy, SourceError
from app.workspace.presentation import (
    LocalReceipt, PresentationAck, PresentationRequest, PresentationRepository, PresentationService, acknowledge, expire,
)
from app.workspace.models import WorkspaceContext


class PresentationTransitionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.session = uuid4()
        self.request = PresentationRequest(request_id=uuid4(), action="download", artifact_kind="note",
            artifact_id=uuid4(), version_id="source_v1", format="text", content_ready=False,
            created_at=self.now, expires_at=self.now + timedelta(minutes=5))
        self.receipt = LocalReceipt(file_id=uuid4(), version_id="source_v1", byte_count=12,
            content_sha256="a" * 64, destination="app_storage")

    def ack(self, state, **extra):
        return PresentationAck(session_id=self.session, state=state, **extra)

    def ready(self):
        return PresentationRequest.model_validate(self.request.model_dump() | {
            "content_ready": True, "file_id": self.receipt.file_id, "byte_count": 12, "content_sha256": "a" * 64})

    def test_claim_before_export_is_rejected(self):
        with self.assertRaises(JarvisHTTPError) as error:
            acknowledge(self.request, self.ack("claimed"), self.now)
        self.assertEqual(error.exception.code, "content_not_ready")

    def test_saved_exact_bytes_and_duplicate_ack_do_not_repeat_effect(self):
        claimed = acknowledge(self.ready(), self.ack("claimed"), self.now)
        saved = acknowledge(claimed, self.ack("saved_local", local_receipt=self.receipt), self.now)
        self.assertEqual(saved.state, "saved_local")
        self.assertEqual(acknowledge(saved, self.ack("saved_local", local_receipt=self.receipt), self.now), saved)
        self.assertEqual(acknowledge(saved, self.ack("claimed"), self.now), saved)
        changed = self.receipt.model_copy(update={"destination": "files"})
        with self.assertRaises(JarvisHTTPError):
            acknowledge(saved, self.ack("saved_local", local_receipt=changed), self.now)

    def test_no_receipt_or_wrong_version_cannot_claim_saved(self):
        with self.assertRaises(ValidationError):
            self.ack("saved_local")
        with self.assertRaises(ValidationError):
            self.ack("displayed", local_receipt=self.receipt)
        claimed = acknowledge(self.ready(), self.ack("claimed"), self.now)
        for changes in ({"version_id": "source_v2"}, {"file_id": uuid4()}, {"byte_count": 11}, {"content_sha256": "b" * 64}):
            with self.subTest(changes=changes), self.assertRaises(JarvisHTTPError) as error:
                acknowledge(claimed, self.ack("saved_local", local_receipt=self.receipt.model_copy(update=changes)), self.now)
            self.assertEqual(error.exception.code, "presentation_receipt_mismatch")

    def test_completion_requires_claim_and_matching_action(self):
        with self.assertRaises(JarvisHTTPError):
            acknowledge(self.ready(), self.ack("saved_local", local_receipt=self.receipt), self.now)
        claimed = acknowledge(self.ready(), self.ack("claimed"), self.now)
        with self.assertRaises(JarvisHTTPError):
            acknowledge(claimed, self.ack("displayed"), self.now)

    def test_expiry_cannot_reactivate_but_claimed_transfer_can_finish(self):
        later = self.now + timedelta(minutes=6)
        self.assertEqual(acknowledge(self.ready(), self.ack("claimed"), later).state, "expired")
        claimed = acknowledge(self.ready(), self.ack("claimed"), self.now)
        self.assertEqual(expire(claimed, later).state, "claimed")
        self.assertEqual(acknowledge(claimed, self.ack("saved_local", local_receipt=self.receipt), later).state, "saved_local")

    def test_open_display_and_failure_are_distinct(self):
        request = PresentationRequest.model_validate(self.request.model_dump() | {
            "action": "open", "format": None, "content_ready": True})
        claimed = acknowledge(request, self.ack("claimed"), self.now)
        self.assertEqual(PresentationService.outcome(claimed)["outcome"], "outcome_unknown")
        shown = acknowledge(claimed, self.ack("displayed"), self.now)
        self.assertEqual(PresentationService.outcome(shown)["outcome"], "completed")
        with self.assertRaises(JarvisHTTPError):
            acknowledge(shown, self.ack("failed", error_code="transport_error"), self.now)
        failed = acknowledge(request, self.ack("failed", error_code="source_unavailable"), self.now)
        self.assertEqual(acknowledge(failed, self.ack("failed", error_code="source_unavailable"), self.now), failed)

    def test_request_and_receipt_validation_reject_unsafe_states(self):
        with self.assertRaises(ValidationError):
            PresentationRequest.model_validate(self.request.model_dump() | {"content_ready": True})
        with self.assertRaises(ValidationError):
            LocalReceipt.model_validate(self.receipt.model_dump() | {"byte_count": True})
        with self.assertRaises(ValidationError):
            self.ack("claimed", error_code="not_an_outcome")


class ReadOnlyConnection:
    """Reject lock/write attempts; this does not emulate PostgreSQL ordering."""
    def __init__(self, row, now):
        self.row, self.now = row, now
        self.transactions = []

    @asynccontextmanager
    async def transaction(self, *, readonly):
        self.transactions.append(readonly)
        if not readonly:
            raise AssertionError("A delivery read must not open a write transaction")
        yield self

    def check_read(self, query):
        if any(token in query.lower() for token in ("for update", "for share", "advisory", "update ", "insert ")):
            raise AssertionError("A delivery read attempted a mutation/lock")

    async def execute(self, *args):
        raise AssertionError("A delivery read attempted to execute a write")

    async def fetchval(self, query, *args):
        self.check_read(query)
        return self.now if "clock_timestamp" in query else True

    async def fetchrow(self, query, *args):
        self.check_read(query)
        return self.row

    async def fetch(self, query, *args):
        self.check_read(query)
        return [self.row]


class ReadOnlyPool:
    def __init__(self, conn):
        self.conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self.conn


class PresentationReadTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.ctx = WorkspaceContext("owner", uuid4(), uuid4())
        self.request = PresentationRequest(request_id=self.ctx.operation_id, action="open", artifact_kind="note",
            artifact_id=uuid4(), version_id="v1", content_ready=True, created_at=self.now,
            expires_at=self.now + timedelta(minutes=5))
        self.row = {"session_id": self.ctx.session_id, "parameters_digest": "digest", "payload": self.request.wire()}

    async def test_expired_read_is_visible_without_persisting_or_locking(self):
        conn = ReadOnlyConnection(self.row, self.now + timedelta(minutes=6))
        repo = PresentationRepository(ReadOnlyPool(conn))
        result = await repo.get(self.ctx, self.ctx.operation_id, digest="digest")
        items = await repo.list(self.ctx)
        self.assertEqual(result.state, "expired")
        self.assertEqual(items[0].state, "expired")
        self.assertEqual(self.row["payload"]["state"], "pending")
        self.assertEqual(conn.transactions, [True, True])

    async def test_read_rejects_reused_operation_and_different_session(self):
        repo = PresentationRepository(ReadOnlyPool(ReadOnlyConnection(self.row, self.now)))
        with self.assertRaises(JarvisHTTPError) as mismatch:
            await repo.get(self.ctx, self.ctx.operation_id, digest="other")
        self.assertEqual(mismatch.exception.code, "operation_conflict")
        self.row["session_id"] = uuid4()
        with self.assertRaises(JarvisHTTPError) as mismatch:
            await repo.get(self.ctx, self.ctx.operation_id)
        self.assertEqual(mismatch.exception.code, "workspace_operation_scope")


class MemoryRequests:
    """Service orchestration fixture, not persistence/concurrency qualification."""
    def __init__(self, now):
        self.now, self.value, self.digest = now, None, None
        self.reservations = 0

    async def get(self, ctx, request_id, *, digest=None):
        if self.value is not None and digest is not None and digest != self.digest:
            raise JarvisHTTPError(409, "operation_conflict", "Different operation parameters")
        return self.value

    async def reserve(self, ctx, *, digest, target):
        self.reservations += 1
        self.digest = digest
        self.value = PresentationRequest(request_id=ctx.operation_id, created_at=self.now,
            expires_at=self.now + timedelta(minutes=5), **target)
        return self.value

    async def mark_ready(self, ctx, request_id, prepared):
        self.value = PresentationRequest.model_validate(self.value.model_dump() | prepared.model_dump() | {"content_ready": True})
        return self.value

    async def ack(self, ctx, request_id, ack):
        self.value = acknowledge(self.value, ack, self.now)
        return self.value

    async def list(self, ctx):
        return [] if self.value is None else [self.value]


class PresentationServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ctx = WorkspaceContext("owner", uuid4(), uuid4())
        self.repo = MemoryRequests(datetime(2026, 11, 1, tzinfo=UTC))
        self.current = "v1"
        self.available = True
        self.targets = []
        self.artifact = uuid4()
        self.prepared = {"file_id": uuid4(), "byte_count": 12, "content_sha256": "a" * 64}

    async def resolve(self, ctx, kind, artifact, version):
        self.targets.append(version)
        if not self.available:
            raise JarvisHTTPError(404, "source_unavailable", "Withdrawn")
        return version or self.current

    async def test_retry_keeps_reserved_version_after_current_changes(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        first = await service.present(self.ctx, action="open", artifact_id=self.artifact)
        self.current = "v2"
        second = await service.present(self.ctx, action="open", artifact_id=self.artifact)
        self.assertEqual(second, first)
        self.assertEqual(second["artifact"]["version_id"], "v1")
        self.assertEqual(self.repo.reservations, 1)
        self.assertEqual(self.targets, [None, "v1", "v1"])
        with self.assertRaises(JarvisHTTPError) as error:
            await service.present(self.ctx, action="open", artifact_id=uuid4())
        self.assertEqual(error.exception.code, "operation_conflict")

    async def test_unknown_export_retries_same_request_and_only_ready_after_receipt(self):
        attempts = []
        async def exporter(ctx, request):
            attempts.append((request.request_id, request.version_id))
            if len(attempts) == 1:
                raise TimeoutError("Upload may have committed")
            return self.prepared
        service = PresentationService(self.repo, resolver=self.resolve, exporter=exporter)
        result = await service.present(self.ctx, action="download", artifact_id=self.artifact)
        self.assertEqual(result["outcome"], "outcome_unknown")
        self.assertFalse(self.repo.value.content_ready)
        self.current = "v2"
        result = await service.present(self.ctx, action="download", artifact_id=self.artifact)
        self.assertEqual(result["outcome"], "accepted")
        self.assertTrue(self.repo.value.content_ready)
        self.assertEqual(attempts, [(self.ctx.operation_id, "v1")] * 2)
        self.assertEqual(self.repo.reservations, 1)

    async def test_known_no_effect_export_failure_is_terminal(self):
        async def exporter(ctx, request):
            raise JarvisHTTPError(422, "unsupported_export", "Known no effect", details={"effect_committed": False})
        service = PresentationService(self.repo, resolver=self.resolve, exporter=exporter)
        result = await service.present(self.ctx, action="download", artifact_id=self.artifact)
        self.assertEqual(result["outcome"], "failed")
        self.assertEqual(self.repo.value.error_code, "unsupported_export")

    async def test_unavailable_producer_does_not_reserve_or_resolve(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        with self.assertRaises(JarvisHTTPError) as error:
            await service.present(self.ctx, action="download", artifact_id=self.artifact)
        self.assertEqual(error.exception.code, "workspace_export_unavailable")
        self.assertEqual(self.repo.reservations, 0)
        self.assertEqual(self.targets, [])

    async def test_withdrawn_offer_is_hidden_and_cannot_be_claimed(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        await service.present(self.ctx, action="open", artifact_id=self.artifact)
        self.available = False
        self.assertEqual(await service.list_requests(self.ctx), {"items": []})
        with self.assertRaises(JarvisHTTPError):
            await service.acknowledge(self.ctx, self.ctx.operation_id,
                PresentationAck(session_id=self.ctx.session_id, state="claimed"))
        self.assertEqual(self.repo.value.state, "pending")

    async def test_withdrawn_409_offer_does_not_hide_unrelated_valid_offer(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        await service.present(self.ctx, action="open", artifact_id=self.artifact)
        withdrawn = self.repo.value
        valid = withdrawn.model_copy(update={"request_id": uuid4(), "artifact_id": uuid4()})
        async def items(ctx):
            return [withdrawn, valid]
        async def resolve(ctx, kind, artifact, version):
            if artifact == self.artifact:
                raise SourceError("knowledge_source_unavailable", "Withdrawn original")
            return version
        self.repo.list = items
        service.resolver = resolve
        self.assertEqual(await service.list_requests(self.ctx), {"items": [valid.wire()]})
        async def busy(*args):
            raise CanonicalOwnerBusy()
        service.resolver = busy
        with self.assertRaises(CanonicalOwnerBusy):
            await service.list_requests(self.ctx)

    async def test_terminal_ack_after_withdrawal_records_only_frozen_claimed_outcome(self):
        async def exporter(ctx, request):
            return self.prepared
        service = PresentationService(self.repo, resolver=self.resolve, exporter=exporter)
        await service.present(self.ctx, action="download", artifact_id=self.artifact)
        await service.acknowledge(self.ctx, self.ctx.operation_id,
            PresentationAck(session_id=self.ctx.session_id, state="claimed"))
        self.available = False
        receipt = LocalReceipt(version_id="v1", destination="app_storage", **self.prepared)
        ack = PresentationAck(session_id=self.ctx.session_id, state="saved_local", local_receipt=receipt)
        saved = await service.acknowledge(self.ctx, self.ctx.operation_id, ack)
        self.assertEqual(saved["request"]["state"], "saved_local")
        self.assertEqual(await service.acknowledge(self.ctx, self.ctx.operation_id, ack), saved)
        self.assertEqual(await service.list_requests(self.ctx), {"items": []})
        replay = await service.present(self.ctx, action="download", artifact_id=self.artifact)
        self.assertEqual(replay["outcome"], "completed")
        self.assertTrue(replay["effect_committed"])
        wrong = ack.model_copy(update={"local_receipt": receipt.model_copy(update={"content_sha256": "b" * 64})})
        with self.assertRaises(JarvisHTTPError):
            await service.acknowledge(self.ctx, self.ctx.operation_id, wrong)

    async def test_withdrawal_cannot_skip_claim_or_prevent_owner_dismissal(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        await service.present(self.ctx, action="open", artifact_id=self.artifact)
        self.available = False
        with self.assertRaises(JarvisHTTPError) as error:
            await service.acknowledge(self.ctx, self.ctx.operation_id,
                PresentationAck(session_id=self.ctx.session_id, state="displayed"))
        self.assertEqual(error.exception.code, "presentation_not_claimed")
        result = await service.acknowledge(self.ctx, self.ctx.operation_id,
            PresentationAck(session_id=self.ctx.session_id, state="dismissed"))
        self.assertEqual(result["request"]["state"], "dismissed")

    async def test_executor_reconciliation_keeps_committed_reservation_after_withdrawal(self):
        service = PresentationService(self.repo, resolver=self.resolve)
        self.assertIsNone(await service.operation_receipt(self.ctx, self.ctx.operation_id))
        accepted = await service.present(self.ctx, action="open", artifact_id=self.artifact)
        self.available = False
        receipt = await service.operation_receipt(self.ctx, self.ctx.operation_id)
        self.assertEqual(receipt, accepted)
        self.assertTrue(receipt["effect_committed"])
        self.assertNotIn("content", receipt)


if __name__ == "__main__":
    unittest.main()
