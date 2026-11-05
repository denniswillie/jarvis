"""Media snapshots preserve document digests and require explicit measured-file authority."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from app.files.generated import (ArtifactInspection, ArtifactPolicy, ArtifactRequest, ArtifactSnapshot,
                                 inspect_artifact, inspect_file_artifact)
from app.files.local_file import snapshot_file
from app.files.transfer import MP4Inspection


class GeneratedMediaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name).resolve() / 'synthetic.mp4'
        # Deliberately not a real MP4: this tests exact binding, not codec inspection.
        path.write_bytes(b'x' * (9 * 1024**2))
        self.local = await snapshot_file(path, max_bytes=32 * 1024**2)
        self.addCleanup(self.local.close)
        self.request = ArtifactRequest(operation_key='render-one', filename='render.mp4', format='mp4',
                                       description='Explicit synthetic fixture')
        self.policy = ArtifactPolicy(transfer_policy='mp4_output_v1', formats=('mp4',),
                                     max_file_bytes=20_000_000)
        self.measured = MP4Inspection(size_bytes=self.local.size_bytes,
            content_sha256=self.local.content_sha256, duration_ms=10_000, width=1080, height=1920,
            video_codec='h264', audio_codec='aac')

    async def test_default_document_serialization_preserves_existing_grant_digest(self):
        self.assertEqual(ArtifactPolicy().model_dump(mode='json'), {
            'max_files': 8, 'max_file_bytes': 8 * 1024**2, 'max_total_bytes': 32 * 1024**2,
            'formats': ['text', 'csv', 'json', 'pdf', 'xlsx']})
        request = ArtifactRequest(operation_key='doc', filename='doc.txt', format='text', description='doc')
        inspection = inspect_artifact(request, b'hello', ArtifactPolicy())
        expected = {'format': 'text', 'content_sha256': hashlib.sha256(b'hello').hexdigest(), 'size_bytes': 5,
                    'text_excerpt': 'hello', 'text_complete': True, 'item_count': None,
                    'meaning': 'Publication metadata only; format is declared, not parsed. '
                               'Not verification of document validity, factual claims, formula results or malware safety.'}
        self.assertEqual(inspection.model_dump(mode='json'), expected)
        old = {'request': request.model_dump(mode='json'), 'inspection': expected,
               'parent_grant_id': str(uuid4()), 'task_id': str(uuid4()), 'evidence_payload_id': str(uuid4())}
        self.assertEqual(ArtifactSnapshot.model_validate(old).model_dump(mode='json'), old)

    async def test_media_policy_cannot_silently_expand_document_allowance(self):
        for kwargs in ({'formats': ('mp4',)}, {'max_file_bytes': 9 * 1024**2},
                       {'transfer_policy': 'mp4_output_v1'},
                       {'transfer_policy': 'mp4_output_v1', 'formats': ('mp4', 'text')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ArtifactPolicy(**kwargs)
        self.assertEqual(self.policy.model_dump(mode='json')['transfer_policy'], 'mp4_output_v1')

    async def test_exact_large_file_snapshot_and_retained_measurement(self):
        inspection = inspect_file_artifact(self.request, self.local, self.policy, self.measured)
        snapshot = ArtifactSnapshot(request=self.request, inspection=inspection,
                                    parent_grant_id=uuid4(), task_id=uuid4(), evidence_payload_id=uuid4())
        self.assertTrue(snapshot.matches(self.local))
        self.assertFalse(snapshot.matches(b'x'))
        self.assertEqual(snapshot.inspection.summary()['media_inspection'], self.measured.model_dump(mode='json'))
        self.assertEqual(ArtifactSnapshot.model_validate(snapshot.model_dump(mode='json')), snapshot)

    async def test_bytes_and_claimed_hash_do_not_establish_inspected_media(self):
        with self.assertRaises(ValueError):
            inspect_artifact(self.request, b'fake movie', self.policy)
        with self.assertRaises(ValueError):
            inspect_file_artifact(self.request, self.local, self.policy,
                                 self.measured.model_copy(update={'content_sha256': '0' * 64}))
        with self.assertRaises(ValueError):
            ArtifactInspection(format='mp4', size_bytes=10, content_sha256='0' * 64)

    async def test_smaller_grant_and_duration_remain_enforced(self):
        with self.assertRaises(ValueError):
            inspect_file_artifact(self.request, self.local,
                self.policy.model_copy(update={'max_file_bytes': 1024}), self.measured)
        exact = self.measured.model_copy(update={'duration_ms': 1_800_000})
        self.assertEqual(inspect_file_artifact(self.request, self.local, self.policy,
            exact).media_inspection.duration_ms, 1_800_000)
        with self.assertRaises(ValueError):
            inspect_file_artifact(self.request, self.local, self.policy,
                                 self.measured.model_copy(update={'duration_ms': 1_800_001}))


if __name__ == '__main__':
    unittest.main()
