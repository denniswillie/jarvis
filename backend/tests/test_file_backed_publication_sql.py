"""Core-only manifested SQL qualification of the file-backed publication foundation.

Defaults to synthetic bytes/inspector. Core may explicitly supply an already
qualified real MP4 and pinned ffprobe; neither mode proves task admission,
provider storage, Phone behavior or live capture.
"""
import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import asyncpg

from app.files.local_file import snapshot_file
from app.files.media_inspection import inspect_mp4_file
from app.files.models import StoredFileObject
from app.files.publication import FilePublicationService, FilePublicationSpec
from app.files.storage import JarvisFileStorageClient
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.knowledge.models import SourceError
from app.errors import JarvisHTTPError
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import LocalReceipt, PreparedContent, PresentationAck, PresentationRepository
from evals.second_brain.source_fixture import source_fixture


class DiskStorage:
    bucket = 'synthetic-media'
    build_file_path = JarvisFileStorageClient.build_file_path

    def __init__(self, root):
        self.root = root
        self.uploads = 0
        self.lose_ack = True

    async def upload_file(self, *, path, local_file, content_type):
        self.uploads += 1
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as target:
            async for chunk in local_file.iter_chunks():
                target.write(chunk)
        if self.lose_ack:
            raise TimeoutError('Synthetic lost upload acknowledgement')
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=local_file.size_bytes)

    async def read_object_file(self, *, bucket, path, max_bytes):
        if bucket != self.bucket:
            raise AssertionError('wrong fixture bucket')
        target = self.root / path
        return await snapshot_file(target, max_bytes=max_bytes) if target.exists() else None

    async def delete_object(self, *, bucket, path):
        if bucket != self.bucket:
            raise AssertionError('wrong fixture bucket')
        (self.root / path).unlink(missing_ok=True)


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested Core database lane required')
class FileBackedPublicationSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_file_unknown_recovery_policy_and_delete(self):
        fixture = await self.enterAsyncContext(source_fixture(
            os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
            Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']), test_name=self.id()))
        directory = tempfile.TemporaryDirectory(prefix='jarvis-media-sql-')
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        selected = os.getenv('JARVIS_MEDIA_FIXTURE_FILE')
        probe = None
        expected_sha = None
        if selected is not None:
            original = Path(selected)
            self.assertTrue(original.is_absolute(), 'Explicit qualified media path required')
            expected_sha = os.environ['JARVIS_MEDIA_FIXTURE_SHA256']
            probe = Path(os.environ['JARVIS_MEDIA_FFPROBE_PATH'])
            self.assertTrue(probe.is_absolute(), 'Explicit qualified ffprobe path required')
            self.assertEqual(hashlib.sha256(probe.read_bytes()).hexdigest(),
                             os.environ['JARVIS_MEDIA_FFPROBE_SHA256'])
        else:
            original = root / 'fixture.mp4'
            # Explicitly not a real movie in the deterministic foundation mode.
            with original.open('wb') as stream:
                for _ in range(144):
                    stream.write(b'x' * 65536)
        local_file = await snapshot_file(original, max_bytes=20_000_000, expected_sha256=expected_sha)
        self.addCleanup(local_file.close)
        self.assertGreater(local_file.size_bytes, 8 * 1024**2)
        print('media_fixture_mode=' + ('qualified_local_mp4' if probe is not None else 'synthetic_bytes')
              + ' size_bytes=' + str(local_file.size_bytes) + ' sha256=' + local_file.content_sha256)
        storage = DiskStorage(root / 'objects')
        adopted = []

        async def inspector(local, policy):
            if probe is not None:
                return await inspect_mp4_file(local, policy, ffprobe_path=probe, timeout=10)
            return MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                                 duration_ms=10_000, width=1080, height=1920, video_codec='h264', audio_codec='aac')

        async def adopt(connection, file, spec, local, measured, payload):
            self.assertTrue(connection.is_in_transaction())
            self.assertEqual(local.content_sha256, local_file.content_sha256)
            adopted.append(file.id)
            # Actual source adapter/admission belongs to the subsequent integrated producer case.

        service = FilePublicationService(fixture.sources, storage, owner_id=fixture.owner,
            area_id=fixture.area, transfer_policy=FileTransferPolicy('mp4_output_v1', 20_000_000),
            inspect_file=inspector, adopt_file=adopt)
        spec = FilePublicationSpec(session_id=None, filename='fixture.mp4', content_type='video/mp4',
                                   description='Synthetic file-backed SQL qualification', source='assistant')
        with self.assertRaises(TimeoutError):
            await service.publish_file(owner_id=fixture.owner, operation_key='one-stable-render',
                                       spec=spec, local_file=local_file)
        async with fixture.pool.acquire() as connection:
            before = await connection.fetchrow('select * from jarvis_file_publications where owner_id=$1', fixture.owner)
        self.assertEqual(before['phase'], 'outcome_unknown')
        self.assertEqual(before['transfer_policy'], 'mp4_output_v1')
        self.assertEqual(before['transfer_limit_bytes'], 20_000_000)
        self.assertIsNotNone(before['upload_token'])
        self.assertEqual(adopted, [])
        result = await service.reconcile(owner_id=fixture.owner, operation_id=before['id'])
        self.assertEqual(result.phase, 'available')
        replay = await service.publish_file(owner_id=fixture.owner, operation_key='one-stable-render',
                                           spec=spec, local_file=local_file)
        self.assertEqual(replay.file_id, result.file_id)
        self.assertEqual(storage.uploads, 1)
        self.assertEqual(adopted, [result.file_id])
        returned = await service.read_file(owner_id=fixture.owner, file_id=result.file_id)
        try:
            self.assertEqual((returned.size_bytes, returned.content_sha256),
                             (local_file.size_bytes, local_file.content_sha256))
        finally:
            returned.close()
        with self.assertRaises(SourceError):
            await service.read_content(owner_id=fixture.owner, file_id=result.file_id)
        # Exercise the real C0 journal with synthetic admitted context/version
        # and Phone acknowledgement. This is not source-admission or device proof.
        async with fixture.pool.acquire() as connection:
            session_id = await connection.fetchval(
                'insert into jarvis_sessions(owner_id) values($1) returning id', fixture.owner)
        ctx = WorkspaceContext(fixture.owner, session_id, uuid4())
        journal = PresentationRepository(fixture.pool)
        offer = await journal.reserve(ctx, digest=ctx.operation_id.hex * 2, target={
            'action': 'download', 'artifact_kind': 'file', 'artifact_id': result.file_id,
            'version_id': 'synthetic-media-version', 'content_ready': False})
        prepared = PreparedContent(file_id=result.file_id, byte_count=local_file.size_bytes,
            content_sha256=local_file.content_sha256, transfer_policy='mp4_output_v1',
            transfer_limit_bytes=20_000_000)
        ready = await journal.mark_ready(ctx, offer.request_id, prepared)
        self.assertEqual(ready.transfer_policy, 'mp4_output_v1')
        self.assertEqual(await journal.mark_ready(ctx, offer.request_id, prepared), ready)
        with self.assertRaises(JarvisHTTPError) as changed_policy:
            await journal.mark_ready(ctx, offer.request_id,
                                     prepared.model_copy(update={'transfer_limit_bytes': 19_000_000}))
        self.assertEqual(changed_policy.exception.code, 'presentation_content_conflict')
        await journal.ack(ctx, offer.request_id, PresentationAck(session_id=session_id, state='claimed'))
        receipt = LocalReceipt(file_id=result.file_id, version_id=offer.version_id,
            byte_count=local_file.size_bytes, content_sha256=local_file.content_sha256, destination='app_storage')
        with self.assertRaises(JarvisHTTPError):
            await journal.ack(ctx, offer.request_id, PresentationAck(session_id=session_id, state='saved_local',
                local_receipt=receipt.model_copy(update={'content_sha256': '0' * 64})))
        saved = await journal.ack(ctx, offer.request_id, PresentationAck(session_id=session_id, state='saved_local',
                                                                       local_receipt=receipt))
        self.assertEqual((await journal.get(ctx, offer.request_id)).local_receipt, receipt)
        self.assertEqual(saved.transfer_limit_bytes, ready.transfer_limit_bytes)
        async with fixture.pool.acquire() as connection:
            for query, value in (
                ('update jarvis_file_publications set transfer_limit_bytes=$2 where id=$1', 21_000_000),
                ('update jarvis_file_publications set transfer_policy=$2 where id=$1', 'mp4_capture_v1'),
                ('update jarvis_file_publications set upload_token=$2 where id=$1', uuid4()),
            ):
                with self.assertRaises(asyncpg.CheckViolationError):
                    await connection.execute(query, before['id'], value)
        # Settled corruption must not prevent privacy cleanup of the owned object.
        (storage.root / result.file.storage_path).write_bytes(b'corrupt')
        deleted = await service.delete(owner_id=fixture.owner, operation_id=before['id'])
        self.assertEqual(deleted.phase, 'deleted')
        self.assertFalse((storage.root / result.file.storage_path).exists())
        self.assertEqual(storage.uploads, 1)
