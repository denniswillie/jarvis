"""File-backed publication boundaries; offline effects are deliberately simulated."""
import asyncio
from contextlib import asynccontextmanager
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.files.publication import FilePublicationResult, FilePublicationService, FilePublicationSpec
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection
from app.knowledge.models import SourceError


class PolicyTests(unittest.TestCase):
    def test_document_bound_stays_small_and_media_requires_deployment_evidence(self):
        self.assertEqual(FileTransferPolicy().max_bytes, 8 * 1024**2)
        for value in (None, 0, -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FileTransferPolicy('mp4_capture_v1', value)
        self.assertEqual(FileTransferPolicy('mp4_capture_v1', 40_000_000).max_bytes, 40_000_000)
        self.assertEqual(FileTransferPolicy('mp4_output_v1', 100_000_000).max_bytes, 32 * 1024**2)

    def test_media_mime_cannot_widen_default_policy_and_oversize_fails(self):
        spec = FilePublicationSpec(session_id=None, filename='sample.mp4', content_type='video/mp4',
                                   description='Authorized test fixture', source='assistant')
        manifest = FileManifest(size_bytes=9 * 1024**2, content_sha256='0' * 64)
        with self.assertRaises(ValueError):
            FileTransferPolicy().validate(manifest, spec)
        FileTransferPolicy('mp4_capture_v1', 10 * 1024**2).validate(manifest, spec)
        with self.assertRaises(ValueError):
            FileTransferPolicy('mp4_capture_v1', 8 * 1024**2).validate(manifest, spec)
        with self.assertRaises(ValueError):
            FileTransferPolicy('mp4_capture_v1', 10 * 1024**2).validate(
                manifest, spec.model_copy(update={'content_type': 'application/octet-stream'}))

    def test_duration_and_exact_hash_belong_to_measured_file(self):
        inspection = MP4Inspection(size_bytes=100, content_sha256='0' * 64, duration_ms=31_000,
                                   width=1080, height=1920, video_codec='h264', audio_codec='aac')
        manifest = FileManifest(size_bytes=100, content_sha256='0' * 64)
        inspection.validate_file(manifest, FileTransferPolicy('mp4_capture_v1', 50_000_000))
        with self.assertRaises(ValueError):
            inspection.validate_file(manifest, FileTransferPolicy('mp4_output_v1', 50_000_000))
        with self.assertRaises(ValueError):
            inspection.validate_file(manifest.model_copy(update={'content_sha256': '1' * 64}),
                                     FileTransferPolicy('mp4_capture_v1', 50_000_000))

    def test_persisted_policy_cannot_infer_wider_limits_or_accept_unknown_kind(self):
        with self.assertRaises(ValueError):
            FileTransferPolicy.from_row({'size_bytes': 9 * 1024**2})
        for policy in ('caller_unlimited', 'mp4_capture_v1'):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                FileTransferPolicy.from_row({'transfer_policy': policy, 'size_bytes': 20})


class Local:
    def __init__(self, content):
        self.content = content
        self.size_bytes = len(content)
        self.content_sha256 = hashlib.sha256(content).hexdigest()
        self.closed = False

    async def iter_chunks(self):
        if self.closed:
            raise ValueError('closed')
        yield self.content

    def close(self):
        self.closed = True


class PublicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.in_transaction = False
        self.revoked = False
        self.content = b'fixture bytes; not inspected real video'
        self.local = Local(self.content)
        self.spec = FilePublicationSpec(session_id=None, filename='sample.mp4', content_type='video/mp4',
                                        description='Explicit synthetic media fixture', source='assistant')
        self.row = dict(id=uuid4(), file_id=uuid4(), phase='prepared', upload_token=None,
                        upload_settled=False, delete_requested=False, storage_bucket='private', storage_path='fixed',
                        size_bytes=len(self.content), content_sha256=self.local.content_sha256,
                        transfer_policy='mp4_output_v1', transfer_limit_bytes=20_000_000,
                        metadata_payload_id=uuid4(), specification={'spec': self.spec.model_dump(mode='json')})
        connection = SimpleNamespace(execute=self.execute, fetchval=AsyncMock(return_value=self.row['id']))
        @asynccontextmanager
        async def transaction(owner):
            self.assertFalse(self.in_transaction)
            self.in_transaction = True
            try:
                yield connection
            finally:
                self.in_transaction = False
        sources = SimpleNamespace(pool=None, transaction=transaction)
        self.storage = SimpleNamespace(upload_file=AsyncMock(side_effect=self.upload),
                                       read_object_file=AsyncMock(side_effect=self.download),
                                       read_object=AsyncMock(), upload_bytes=AsyncMock())
        self.inspection = MP4Inspection(size_bytes=len(self.content), content_sha256=self.local.content_sha256,
                                       duration_ms=1000, width=1080, height=1920, video_codec='h264', audio_codec='aac')
        self.adopt = AsyncMock(side_effect=self.adoption)
        self.service = FilePublicationService(sources, self.storage, owner_id='owner', area_id=uuid4(),
            transfer_policy=FileTransferPolicy('mp4_output_v1', 20_000_000),
            inspect_file=self.inspect, adopt_file=self.adopt)
        self.service._row = AsyncMock(side_effect=lambda *args: dict(self.row))
        self.service._active = AsyncMock(side_effect=self.active)
        self.service._result = AsyncMock(side_effect=lambda connection, row: self.result(row))

    def result(self, row=None):
        row = row or self.row
        return FilePublicationResult(row['id'], row['file_id'], row['phase'], None,
                                     row['upload_settled'], row['delete_requested'])

    async def active(self, *args):
        if self.revoked:
            raise SourceError('revoked', 'withdrawn')

    async def execute(self, query, *args):
        if "phase='uploading'" in query:
            self.row.update(phase='uploading', upload_token=args[2])
        elif 'upload_settled=$3,phase=$4' in query:
            self.row.update(upload_settled=args[2], phase=args[3])
        elif "phase='outcome_unknown'" in query:
            self.row['phase'] = 'outcome_unknown'
        elif 'upload_settled=true,phase=$3' in query:
            self.row.update(upload_settled=True, phase=args[2])
        elif "phase='available'" in query:
            self.row['phase'] = 'available'
        elif "phase='deleted'" in query:
            self.row.update(phase='deleted', storage_bucket=None, storage_path=None)
        else:
            raise AssertionError(query)

    async def upload(self, **kwargs):
        self.assertFalse(self.in_transaction)
        return SimpleNamespace(bucket='private', path='fixed', size_bytes=len(self.content))

    async def download(self, **kwargs):
        self.assertFalse(self.in_transaction)
        self.assertEqual(kwargs['max_bytes'], len(self.content))
        return self.local

    async def inspect(self, local, policy):
        self.assertFalse(self.in_transaction)
        return self.inspection

    async def adoption(self, *args):
        self.assertTrue(self.in_transaction)
        self.assertIs(args[3], self.local)
        self.assertIs(args[4], self.inspection)

    async def test_lost_ack_sets_unknown_and_replay_does_not_resubmit(self):
        self.storage.upload_file.side_effect = TimeoutError('lost acknowledgement')
        with self.assertRaises(TimeoutError):
            await self.service._publish_prepared('owner', self.result(), self.spec, local_file=self.local)
        token = self.row['upload_token']
        self.assertIsNotNone(token)
        self.assertEqual(self.row['phase'], 'outcome_unknown')
        self.storage.read_object_file.side_effect = None
        self.storage.read_object_file.return_value = None
        replay = await self.service._publish_prepared('owner', self.result(), self.spec, local_file=self.local)
        self.assertEqual(replay.phase, 'outcome_unknown')
        self.assertEqual(self.row['upload_token'], token)
        self.assertEqual(self.storage.upload_file.await_count, 1)
        self.adopt.assert_not_awaited()

    async def test_cancelled_upload_keeps_same_unknown_token(self):
        self.storage.upload_file.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.service._publish_prepared('owner', self.result(), self.spec, local_file=self.local)
        self.assertEqual(self.row['phase'], 'outcome_unknown')
        self.assertIsNotNone(self.row['upload_token'])
        self.assertFalse(self.row['upload_settled'])

    async def test_reconcile_verified_file_inspects_outside_transaction_and_adopts_atomically(self):
        self.row.update(phase='outcome_unknown', upload_token=uuid4())
        with patch('app.files.publication.JarvisFileRepository') as repository:
            repository.return_value.get_file = AsyncMock(return_value=None)
            repository.return_value.create_file = AsyncMock(return_value=SimpleNamespace(id=self.row['file_id']))
            result = await self.service.reconcile(owner_id='owner', operation_id=self.row['id'])
        self.assertEqual(result.phase, 'available')
        self.adopt.assert_awaited_once()
        self.storage.upload_file.assert_not_awaited()
        self.storage.read_object.assert_not_awaited()
        self.assertTrue(self.local.closed)

    async def test_corrupt_stored_file_is_closed_and_never_adopted(self):
        self.row.update(phase='uploaded', upload_token=uuid4())
        self.local.content_sha256 = 'f' * 64
        with self.assertRaises(SourceError):
            await self.service.reconcile(owner_id='owner', operation_id=self.row['id'])
        self.assertTrue(self.local.closed)
        self.adopt.assert_not_awaited()

    async def test_post_download_revocation_closes_file_and_blocks_delivery(self):
        self.row.update(phase='available', upload_token=uuid4(), upload_settled=True)
        async def revoked_download(**kwargs):
            self.assertFalse(self.in_transaction)
            self.revoked = True
            return self.local
        self.storage.read_object_file.side_effect = revoked_download
        with self.assertRaises(SourceError):
            await self.service.read_file(owner_id='owner', file_id=self.row['file_id'])
        self.assertTrue(self.local.closed)

    async def test_exact_read_transfers_cleanup_to_caller_and_never_buffers_media(self):
        self.row.update(phase='available', upload_token=uuid4(), upload_settled=True)
        result = await self.service.read_file(owner_id='owner', file_id=self.row['file_id'])
        self.assertIs(result, self.local)
        self.assertFalse(result.closed)
        result.close()
        with self.assertRaises(SourceError):
            await self.service.read_content(owner_id='owner', file_id=self.row['file_id'])
        self.storage.read_object.assert_not_awaited()

    async def test_bad_inspection_cannot_reach_adoption_and_releases_file(self):
        self.row.update(phase='uploaded', upload_token=uuid4())
        self.inspection = self.inspection.model_copy(update={'duration_ms': 31_000})
        with self.assertRaises(ValueError):
            await self.service.reconcile(owner_id='owner', operation_id=self.row['id'])
        self.assertTrue(self.local.closed)
        self.adopt.assert_not_awaited()

    async def test_settled_corrupt_object_can_be_deleted_but_unknown_cannot(self):
        self.row.update(phase='delete_pending', delete_requested=True, upload_token=uuid4())
        self.local.content_sha256 = 'f' * 64
        self.storage.read_object_file.side_effect = None
        self.storage.read_object_file.return_value = self.local
        self.storage.delete_object = AsyncMock()
        with self.assertRaises(SourceError):
            await self.service._cleanup('owner', self.row['id'])
        self.storage.delete_object.assert_not_awaited()
        self.row['upload_settled'] = True
        self.storage.read_object_file.side_effect = [Local(b'corrupt and different length'), None]
        result = await self.service._cleanup('owner', self.row['id'])
        self.assertEqual(result.phase, 'deleted')
        self.storage.delete_object.assert_awaited_once_with(bucket='private', path='fixed')


if __name__ == '__main__':
    unittest.main()
