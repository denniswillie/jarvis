"""Gallery traffic and derivative boundaries, including real SQL when opted in."""
import asyncio
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.errors import JarvisHTTPError
from app.files.local_file import snapshot_file
from app.files.previews import build_preview
from app.files.publication import FilePublicationService, FilePublicationSpec
from app.files.router import router
from app.knowledge.files import SourceFileRepository
from app.knowledge.hindsight import original_document
from app.knowledge.models import SourcePublication
from app.settings import JarvisSettings, get_settings
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture
from test_second_brain_hindsight import source


def picture():
    output = BytesIO()
    Image.new('RGB', (1200, 800), '#3d7468').save(output, 'PNG')
    return output.getvalue()


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_derivative_is_small_decodable_and_metadata_free(self):
        preview = await build_preview(picture(), 'image/png')
        self.assertIsNotNone(preview)
        self.assertLessEqual(len(preview.jpeg), 65536)
        with Image.open(BytesIO(preview.jpeg)) as image:
            self.assertEqual(image.size, (480, 320))
            self.assertFalse(image.getexif())
        self.assertIsNone(await build_preview(b'not an image', 'image/png'))
        self.assertIsNone(await build_preview(picture(), 'text/plain'))

    @unittest.skipUnless(shutil.which('ffmpeg'), 'Existing ffmpeg is required')
    async def test_video_frame_uses_verified_file_and_keeps_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'synthetic.mp4'
            process = await asyncio.create_subprocess_exec(shutil.which('ffmpeg'), '-v', 'error',
                '-f', 'lavfi', '-i', 'color=c=teal:s=640x360:r=1', '-t', '1', '-c:v', 'libx264',
                '-threads', '1', '-pix_fmt', 'yuv420p', str(path))
            self.assertEqual(await process.wait(), 0)
            original = await snapshot_file(path, max_bytes=1024 * 1024)
            try:
                preview = await build_preview(original, 'video/mp4')
                self.assertIsNotNone(preview)
                self.assertEqual((preview.width, preview.height), (480, 270))
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original.content_sha256)
            finally:
                original.close()

    def test_hindsight_receives_reference_not_manifest_or_analysis(self):
        file_id = str(uuid4())
        evidence = source(source_type='recording_file', content={
            'format': 'jarvis.recording_file_original.v1',
            'record': {'id': file_id, 'content_type': 'video/mp4', 'metadata': {'technical': 'private'}},
            'content_sha256': 'a' * 64, 'finalization_manifest': {'huge': 'technical envelope'}})
        document = json.loads(original_document(evidence))
        self.assertEqual(document, {'event': 'recording_saved', 'saved_at': evidence.recorded_at.isoformat(),
            'resource': {'file_id': file_id, 'version_id': evidence.document_id, 'media_type': 'video/mp4'}})
        self.assertIn('finalization_manifest', evidence.content, 'Original evidence is preserved')
        self.assertNotIn('technical', original_document(evidence))


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
    os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'), 'Explicit manifested database required')
class GallerySQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_metadata_thumbnail_original_and_access_boundaries(self):
        async with source_fixture(os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
                Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']), test_name=self.id()) as fixture:
            storage = MemoryStorage()
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval('insert into jarvis_sessions(owner_id) values($1) returning id', fixture.owner)
                foreign = await connection.fetchval('insert into jarvis_sessions(owner_id) values($1) returning id', fixture.other)
            async def adopt(connection, file, spec, content, metadata_payload_id):
                payload = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                    values($1,'available',$2::jsonb) returning id""", fixture.owner, json.dumps({
                    'format': 'jarvis.generated_file_original.v1', 'record': file.model_dump(mode='json'),
                    'content_sha256': hashlib.sha256(content).hexdigest()}))
                await fixture.sources.publish_in_transaction(connection, fixture.owner, SourcePublication(
                    area_id=fixture.area, namespace='product:files', source_key=str(file.id), payload_id=payload,
                    source_type='generated_file', trust='assistant_output', source_time=datetime.now(UTC),
                    expected_revision=0, time_basis='Synthetic fixture', session_id=session))
            service = FilePublicationService(fixture.sources, storage, owner_id=fixture.owner, area_id=fixture.area, adopt=adopt)
            result = await service.publish(owner_id=fixture.owner, operation_key='gallery-preview',
                spec=FilePublicationSpec(session_id=session, filename='image.png', description='Synthetic test image',
                    content_type='image/png', source='assistant'), content=picture())
            self.assertEqual(result.phase, 'available')
            files = SourceFileRepository(fixture.sources, storage, owner_id=fixture.owner, area_id=fixture.area)
            original_read = storage.read_object
            storage.read_object = AsyncMock(side_effect=AssertionError('Grid fetched an original'))
            app = FastAPI(); app.include_router(router)
            app.dependency_overrides[get_settings] = lambda: JarvisSettings(_env_file=None,
                JARVIS_STATIC_BEARER_TOKEN='gallery-fixture-token', JARVIS_OWNER_ID=fixture.owner)
            @app.exception_handler(JarvisHTTPError)
            async def error(request, exc):
                return JSONResponse({'code': exc.code}, status_code=exc.status_code)
            headers = {'Authorization': 'Bearer gallery-fixture-token'}
            with patch('app.files.router._repository', return_value=files):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://gallery.invalid') as client:
                    page = await client.get('/files', params={'session_id':str(session), 'media_only':'true'}, headers=headers)
                    self.assertEqual(page.status_code, 200, page.text)
                    item = page.json()['files'][0]; thumb = item['thumbnail']
                    self.assertLess(len(page.content), 5000)
                    self.assertNotIn('preview_jpeg', page.text)
                    query = {'session_id':str(session), 'version_id':thumb['version_id']}
                    preview = await client.get(thumb['path'], params=query, headers=headers)
                    self.assertEqual(preview.status_code, 200, preview.text[:100])
                    self.assertEqual(hashlib.sha256(preview.content).hexdigest(), thumb['content_sha256'])
                    self.assertEqual(len(preview.content), thumb['size_bytes'])
                    self.assertEqual(preview.headers['cache-control'], 'private, no-store')
                    for changed, auth, status in ((query | {'version_id':'stale'}, headers, 404),
                            (query | {'session_id':str(foreign)}, headers, 404), (query, {}, 401)):
                        denied = await client.get(thumb['path'], params=changed, headers=auth)
                        self.assertEqual(denied.status_code, status)
                    storage.read_object.assert_not_awaited()
                    storage.read_object = AsyncMock(side_effect=original_read)
                    content = await client.get(f'/files/{result.file_id}/content', params=query, headers=headers)
                    self.assertEqual(content.content, picture())
                    storage.read_object.assert_awaited_once()
                    # Simulate a legacy item: absence never prevents opening;
                    # that already-requested original supplies its first preview.
                    async with fixture.pool.acquire() as connection:
                        await connection.execute('update jarvis_file_publications set preview_jpeg=null,preview_sha256=null,preview_width=null,preview_height=null where owner_id=$1', fixture.owner)
                    missing = await client.get(thumb['path'], params=query, headers=headers)
                    self.assertEqual(missing.status_code, 404)
                    reopened = await client.get(f'/files/{result.file_id}/content', params=query, headers=headers)
                    self.assertEqual(reopened.status_code, 200)
                    regenerated = await client.get(thumb['path'], params=query, headers=headers)
                    self.assertEqual(regenerated.status_code, 200)
                    self.assertEqual(storage.read_object.await_count, 2)
                    async with fixture.pool.acquire() as connection:
                        await connection.execute("update jarvis_file_publications set delete_requested=true,phase='delete_pending' where owner_id=$1", fixture.owner)
                        self.assertIsNone(await connection.fetchval('select preview_jpeg from jarvis_file_publications where owner_id=$1', fixture.owner))
