"""Exact file-backed HTTP access and private-file cleanup with simulated sources."""
import asyncio
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.files.router import _PrivateFileResponse
from app.files.transfer import FileManifest, FileTransferPolicy, MP4Inspection
from app.files.workspace_access import WorkspaceFileAccess
from test_file_backed_publication import Local


class MediaAccessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.local = Local(b'exact bytes')
        self.manifest = FileManifest.from_file(self.local)
        self.policy = FileTransferPolicy('mp4_output_v1', 20_000_000)
        self.inspection = MP4Inspection(size_bytes=self.local.size_bytes,
            content_sha256=self.local.content_sha256, duration_ms=1000, width=1080, height=1920,
            video_codec='h264', audio_codec='aac')
        self.file = SimpleNamespace(id=uuid4(), content_type='video/mp4', size_bytes=self.local.size_bytes,
            metadata={'media_inspection': self.inspection.model_dump(mode='json')},
            memory_source={'document_id': 'exact-v1'})
        self.repository = SimpleNamespace(_authorize=MagicMock(), selections=None,
            get_file=AsyncMock(return_value=self.file), read_file=AsyncMock(return_value=self.local),
            publications=SimpleNamespace(read_manifest=AsyncMock(return_value=(self.manifest, self.policy))))
        self.access = WorkspaceFileAccess(self.repository)
        # These byte-only fixtures test access and response ownership. Preview
        # generation is covered separately with decodable image/video inputs.
        self.access._cache_preview = AsyncMock()

    async def test_metadata_is_exact_and_reports_effective_not_global_bound(self):
        file, fields = await self.access.media_metadata('owner', None, self.file.id, 'exact-v1')
        self.assertIs(file, self.file)
        self.assertEqual(fields['transfer_limit_bytes'], 20_000_000)
        self.assertEqual(fields['version_id'], 'exact-v1')
        self.assertEqual(fields['content_sha256'], self.local.content_sha256)
        self.repository.read_file.assert_not_awaited()

    async def test_missing_version_never_fetches_bytes(self):
        with self.assertRaises(JarvisHTTPError):
            await self.access.read_file('owner', None, self.file.id, None)
        self.repository.get_file.assert_not_awaited()
        self.repository.read_file.assert_not_awaited()

    async def test_retained_nested_inspection_preserves_existing_media_access(self):
        self.file.metadata = {'hermes_session_authority': {'owner_id': 'owner'},
            'inspection': {'media_inspection': self.inspection.model_dump(mode='json')}}
        _, fields = await self.access.media_metadata('owner', None, self.file.id, 'exact-v1')
        self.assertEqual(fields['content_sha256'], self.local.content_sha256)
        self.repository.read_file.assert_not_awaited()

    async def test_retained_nested_inspection_still_requires_matching_manifest(self):
        self.file.metadata = {'hermes_session_authority': {'owner_id': 'owner'},
            'inspection': {'media_inspection': self.inspection.model_dump(mode='json') |
                {'content_sha256': 'a' * 64}}}
        with self.assertRaises(JarvisHTTPError):
            await self.access.media_metadata('owner', None, self.file.id, 'exact-v1')
        self.repository.read_file.assert_not_awaited()

    async def test_stale_version_and_missing_inspection_cannot_fetch_bytes(self):
        with self.assertRaises(JarvisHTTPError):
            await self.access.read_file('owner', None, self.file.id, 'old')
        self.file.metadata = {}
        with self.assertRaises(JarvisHTTPError):
            await self.access.read_file('owner', None, self.file.id, 'exact-v1')
        self.repository.read_file.assert_not_awaited()

    async def test_withdrawal_after_fetch_closes_private_file(self):
        self.repository.get_file.side_effect = [self.file, self.file, None]
        with self.assertRaises(JarvisHTTPError):
            await self.access.read_file('owner', None, self.file.id, 'exact-v1')
        self.assertTrue(self.local.closed)

    async def test_checksum_change_closes_private_file(self):
        self.repository.read_file.return_value = changed = Local(b'other bytes')
        with self.assertRaises(JarvisHTTPError):
            await self.access.read_file('owner', None, self.file.id, 'exact-v1')
        self.assertTrue(changed.closed)

    async def test_success_transfers_ownership_to_response(self):
        file, local, fields = await self.access.read_file('owner', None, self.file.id, 'exact-v1')
        self.assertIs(local, self.local)
        self.assertFalse(local.closed)
        sent = []
        async def send(message):
            sent.append(message)
        response = _PrivateFileResponse(local, media_type='video/mp4', headers={
            'Content-Length': str(local.size_bytes), 'Cache-Control': 'private, no-store'})
        await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, AsyncMock(), send)
        self.assertEqual(b''.join(m.get('body', b'') for m in sent), b'exact bytes')
        self.assertTrue(local.closed)

    async def test_header_failure_and_cancel_before_body_both_close_private_file(self):
        for error in (RuntimeError('header failure'), asyncio.CancelledError()):
            local = Local(b'private')
            response = _PrivateFileResponse(local, media_type='video/mp4', headers={})
            with self.assertRaises(type(error)):
                await response({'type': 'http', 'asgi': {'spec_version': '2.4'}}, AsyncMock(),
                               AsyncMock(side_effect=error))
            self.assertTrue(local.closed)


if __name__ == '__main__':
    unittest.main()
