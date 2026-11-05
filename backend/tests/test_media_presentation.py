"""Large media uses trusted bounds without widening note/document delivery."""
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.files.transfer import FileTransferPolicy
from app.runtime.workspace_delivery import WorkspaceContentExporter
from app.workspace.presentation import (LocalReceipt, PreparedContent, PresentationAck, PresentationRequest, PresentationRepository,
                                        acknowledge, expire)
from test_file_backed_publication import Local


class MediaPresentationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(timezone.utc)
        self.file_id = uuid4()
        self.prepared = dict(file_id=self.file_id, byte_count=15_431_453, content_sha256='a' * 64,
                             transfer_policy='mp4_capture_v1', transfer_limit_bytes=40_000_000)
        self.base = dict(request_id=uuid4(), action='download', artifact_kind='file', artifact_id=self.file_id,
                         version_id='exact-v1', created_at=self.now, expires_at=self.now + timedelta(minutes=5))

    def test_document_shape_and_eight_megabyte_ceiling_remain(self):
        old = dict(file_id=self.file_id, byte_count=10, content_sha256='a' * 64)
        self.assertEqual(PreparedContent(**old).model_dump(), old)
        self.assertNotIn('transfer_policy', PresentationRequest(**self.base).wire())
        with self.assertRaises(ValueError):
            PreparedContent(**(old | {'byte_count': 8 * 1024**2 + 1}))
        with self.assertRaises(ValueError):
            PresentationRequest(**(self.base | old | {'byte_count': 8 * 1024**2 + 1, 'content_ready': True}))

    def test_trusted_media_policy_binds_ready_file_download_only(self):
        prepared = PreparedContent(**self.prepared)
        request = PresentationRequest(**(self.base | prepared.model_dump() | {'content_ready': True}))
        self.assertEqual(request.wire()['transfer_limit_bytes'], 40_000_000)
        self.assertEqual(request.wire()['transfer_policy'], 'mp4_capture_v1')
        for changes in ({'artifact_kind': 'note'}, {'action': 'open'}, {'content_ready': False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PresentationRequest.model_validate(request.model_dump() | changes)

    def test_unpaired_or_overstated_media_bounds_rejected(self):
        for changes in ({'transfer_policy': None}, {'transfer_limit_bytes': None},
                        {'transfer_limit_bytes': 10_000_000}, {'transfer_limit_bytes': True},
                        {'transfer_policy': 'mp4_output_v1', 'transfer_limit_bytes': 40_000_000},
                        {'byte_count': 0}, {'byte_count': 41_000_000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PreparedContent(**(self.prepared | changes))

    def test_large_saved_receipt_must_match_exact_claimed_content(self):
        request = PresentationRequest(**(self.base | self.prepared | {'content_ready': True}))
        session = uuid4()
        claimed = acknowledge(request, PresentationAck(session_id=session, state='claimed'), self.now)
        receipt = LocalReceipt(file_id=self.file_id, version_id='exact-v1', byte_count=self.prepared['byte_count'],
                               content_sha256='a' * 64, destination='app_storage')
        for changes in ({'version_id': 'different'}, {'byte_count': self.prepared['byte_count'] - 1},
                        {'content_sha256': 'b' * 64}, {'file_id': uuid4()}):
            with self.subTest(changes=changes), self.assertRaises(JarvisHTTPError):
                acknowledge(claimed, PresentationAck(session_id=session, state='saved_local',
                    local_receipt=receipt.model_copy(update=changes)), self.now)
        saved = acknowledge(claimed, PresentationAck(session_id=session, state='saved_local', local_receipt=receipt), self.now)
        self.assertEqual(saved.state, 'saved_local')
        self.assertEqual(saved.transfer_limit_bytes, request.transfer_limit_bytes)
        self.assertEqual(saved.local_receipt, receipt)
        self.assertEqual(acknowledge(saved, PresentationAck(session_id=session, state='saved_local',
                                                       local_receipt=receipt), self.now), saved)

    def test_expiry_preserves_frozen_policy_without_new_effect(self):
        request = PresentationRequest(**(self.base | self.prepared | {'content_ready': True}))
        expired = expire(request, self.now + timedelta(minutes=6))
        self.assertEqual(expired.state, 'expired')
        self.assertEqual(expired.transfer_policy, request.transfer_policy)
        self.assertEqual(expired.byte_count, request.byte_count)


class MediaExporterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.file = SimpleNamespace(id=uuid4(), content_type='video/mp4')
        self.local = Local(b'synthetic')
        # Simulate a checked large file without allocating it in this orchestration test.
        self.local.size_bytes = 15_431_453
        self.policy = FileTransferPolicy('mp4_capture_v1', 40_000_000)
        self.access = SimpleNamespace(get=AsyncMock(return_value=self.file), read=AsyncMock(),
            read_file_with_policy=AsyncMock(return_value=(self.file, self.local, {}, self.policy)))
        self.exporter = WorkspaceContentExporter(self.access)
        self.ctx = SimpleNamespace(owner_id='owner', session_id=uuid4())
        self.request = SimpleNamespace(artifact_kind='file', action='download', artifact_id=self.file.id,
                                       version_id='exact-v1')

    async def test_media_preparation_reuses_exact_file_then_closes_staging_copy(self):
        result = await self.exporter(self.ctx, self.request)
        self.assertEqual(result.file_id, self.file.id)
        self.assertEqual(result.byte_count, 15_431_453)
        self.assertEqual(result.transfer_limit_bytes, 40_000_000)
        self.assertTrue(self.local.closed)
        self.access.read.assert_not_awaited()

        self.access.read_file_with_policy.assert_awaited_once_with('owner', self.ctx.session_id,
                                                                  self.file.id, 'exact-v1')

    async def test_invalid_prepared_result_still_closes_staging_copy(self):
        self.local.size_bytes = 41_000_000
        with self.assertRaises(ValueError):
            await self.exporter(self.ctx, self.request)
        self.assertTrue(self.local.closed)
        self.access.read.assert_not_awaited()


class PreparationReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_media_replay_cannot_drop_or_change_its_policy(self):
        now = datetime.now(timezone.utc)
        ctx = SimpleNamespace(owner_id='owner', session_id=uuid4(), operation_id=uuid4())
        prepared = PreparedContent(file_id=uuid4(), byte_count=1024, content_sha256='a' * 64,
                                   transfer_policy='mp4_output_v1', transfer_limit_bytes=20_000_000)
        request = PresentationRequest(request_id=ctx.operation_id, action='download', artifact_kind='file',
            artifact_id=prepared.file_id, version_id='v1', content_ready=True, created_at=now,
            expires_at=now + timedelta(minutes=5), **prepared.model_dump())
        connection = SimpleNamespace(fetchval=AsyncMock(return_value=now))
        @asynccontextmanager
        async def transaction(_):
            yield connection
        repository = PresentationRepository(None)
        repository._transaction = transaction
        repository._row = AsyncMock(return_value={'payload': request.wire()})
        repository._save = AsyncMock()
        self.assertEqual(await repository.mark_ready(ctx, ctx.operation_id, prepared), request)
        for other in (PreparedContent(file_id=prepared.file_id, byte_count=prepared.byte_count,
                                      content_sha256=prepared.content_sha256),
                      prepared.model_copy(update={'transfer_limit_bytes': 19_000_000})):
            with self.subTest(other=other), self.assertRaises(JarvisHTTPError) as error:
                await repository.mark_ready(ctx, ctx.operation_id, other)
            self.assertEqual(error.exception.code, 'presentation_content_conflict')
        repository._save.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
