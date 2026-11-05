"""Core-scheduled SQL/source/action composition for retained media edits.

Uses one existing manifested disposable database fixture and real production
publication/adoption/receipt/loader code. Storage is local disk; media bytes and
inspection are explicitly synthetic. No provider, codec, renderer, cloud
storage, model, native UI or physical-device qualification is performed here.
"""
import asyncio
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from uuid import UUID, uuid4

from app.files.generated import ArtifactPolicy, ArtifactRequest
from app.files.local_file import snapshot_file
from app.files.models import StoredFileObject
from app.files.tools import FileToolModule
from app.files.transfer import FileTransferPolicy, MP4Inspection
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.sandbox.file_transfer import INPUT_ROOT
from app.tasks.generated_artifacts import GeneratedArtifactPublisher
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import TaskStatus, WorkerOutcome
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tasks.worker import WorkerContextFactory
from app.workspace.media_publication import MediaPublicationStore, publication_requests
from app.workspace.media_recipe import (
    MediaRecipeError, MediaRecipePolicy, RetainedRef, SourceInspection, TemplateInventoryEntry, validate_recipe,
)
from app.workspace.media_render import AuthorizedRenderInput, RendererBinding, prepare_render
from evals.second_brain.sandbox_fixture import sandbox_task_fixture
from tests.test_file_backed_publication_sql import DiskStorage
from tests.test_workspace_media_recipe import recipe


LIMIT = 128 * 1024


class MediaPlanDiskStorage(DiskStorage):
    """Both existing publication byte paths, with upload effects counted."""

    async def upload_bytes(self, *, path, content, content_type):
        self.uploads += 1
        destination = self.root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(content)
        return StoredFileObject(bucket=self.bucket, path=path, size_bytes=len(content))

    async def read_object(self, *, bucket, path, max_bytes):
        if bucket != self.bucket:
            raise AssertionError('wrong fixture bucket')
        target = self.root / path
        if not target.exists():
            return None
        with target.open('rb') as stream:
            content = stream.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError('fixture read exceeds bound')
        return content


@unittest.skipUnless(os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') and
                     os.getenv('JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'),
                     'Explicit manifested Core SQL lane required')
class MediaPublicationSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_plan_correction_unknown_recovery_and_foreground_reopen(self):
        async with asyncio.timeout(180):
            async with sandbox_task_fixture(
                os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
                Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']),
                test_name=self.id(), with_session=True,
                requested_outcome='Exercise durable media records with synthetic bytes; no actual render.',
            ) as fixture:
                await self.exercise(fixture)

    async def exercise(self, fixture):
        self.f = fixture
        self.owner, self.task_id = fixture.source.owner, fixture.task.id
        context = TaskRunner(fixture.tasks, {}).context(fixture.claim)
        await WorkerContextFactory(fixture.source.pool)(context)
        directory = tempfile.TemporaryDirectory(prefix='jarvis-media-plan-sql-')
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.storage = MediaPlanDiskStorage(self.root / 'objects')
        self.storage.lose_ack = False
        self.transfer = FileTransferPolicy('mp4_output_v1', LIMIT)

        async def synthetic_inspector(local, policy):
            # Deliberately declared fixture metadata, not a codec/render result.
            self.assertLessEqual(local.size_bytes, policy.max_bytes)
            return MP4Inspection(size_bytes=local.size_bytes, content_sha256=local.content_sha256,
                                 duration_ms=10000, width=720, height=1280, video_codec='h264', audio_codec='aac')

        evidence = TaskMemoryEvidence(context, fixture.source.sources, area_id=fixture.source.area)
        self.worker_files = SourceFileRepository(fixture.source.sources, self.storage, owner_id=self.owner,
            area_id=fixture.source.area, task_evidence=evidence, media_policy=self.transfer,
            media_inspector=synthetic_inspector)
        self.foreground = SourceFileRepository(fixture.source.sources, self.storage, owner_id=self.owner,
            area_id=fixture.source.area, media_policy=self.transfer, media_inspector=synthetic_inspector)
        self.writer = MediaPublicationStore(fixture.tasks, self.worker_files)
        self.reader = MediaPublicationStore(fixture.tasks, self.foreground)
        bridge = await TaskToolBridge.from_modules(context, (FileToolModule(self.worker_files, self.storage),),
            read_tools=('list_files', 'get_file_metadata'), memory_evidence=evidence,
            receipt_revalidator=self.worker_files.revalidate_receipt)
        allowance = ArtifactPolicy(max_files=8, max_file_bytes=LIMIT, max_total_bytes=2 * 1024 * 1024,
                                   formats=('json', 'mp4'), mp4_max_file_bytes=LIMIT)
        async with fixture.tasks.transaction(self.owner) as connection:
            scope = await fixture.tasks.payload(connection, self.owner, allowance.model_dump(mode='json'))
            parent = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.publish_generated',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.owner, str(uuid4()), fixture.event, scope)
            await connection.execute('insert into jarvis_task_grants(owner_id,task_id,grant_id) values($1,$2,$3)',
                                     self.owner, self.task_id, parent)
        self.publisher = GeneratedArtifactPublisher(bridge, self.worker_files, parent)

        # This control predates reading the source/recipe and must not disappear
        # when that later recipe is withdrawn. All records use the actual source owner.
        control = await self.publisher.publish(ArtifactRequest(operation_key='independent-control',
            filename='control.json', format='json', description='Independent synthetic control.'),
            b'{"control":true}', provider_call_id='control')
        self.assertEqual(control['status'], 'success', control)
        source = await self.local('source', b'opaque synthetic source; not an MP4')
        original = await self.publisher.publish(ArtifactRequest(operation_key='synthetic-source',
            filename='original.mp4', format='mp4', description='Opaque synthetic input; not a captured video.'),
            source, provider_call_id='source')
        self.assertEqual(original['status'], 'success', original)
        original_file = await self.foreground.get_file(owner_id=self.owner, file_id=UUID(original['file']['id']))
        source_ref = self.reference(original_file, source.content_sha256)
        plan = self.plan(source_ref, source.size_bytes, 'first-edit')
        recipe_receipt = await self.writer.publish_recipe(plan, self.publisher, provider_call_id='recipe-one')
        self.assertEqual(recipe_receipt['status'], 'success', recipe_receipt)
        saved = await self.reader.load_plan(self.owner, self.task_id, plan.digest)
        self.assertEqual(saved.plan, plan)
        self.assertEqual(str(saved.file.file_id), recipe_receipt['file']['id'])
        self.assertEqual(saved.action_id, UUID(recipe_receipt['task_operation']['action_id']))
        print('media_publication_sql: exact recipe saved', flush=True)

        # The storage effect occurs, but no accepted publication is reported.
        # Reconcile under the same identity without the original local output.
        output = await self.local('output-one', b'opaque synthetic render one')
        self.storage.lose_ack = True
        with self.assertRaises(TimeoutError):
            await self.writer.publish_output(plan, self.publisher, output, provider_call_id='output-one')
        before = await self.action_rows()
        unknown = [row for row in before if row['status'] == 'outcome_unknown']
        self.assertEqual(len(unknown), 1, before)
        self.assertIsNone(await self.reader.load_accepted_output(self.owner, self.task_id, plan.digest))
        output.close()
        self.storage.lose_ack = False
        uploads = self.storage.uploads
        recovered = await self.writer.publish_output(plan, self.publisher, provider_call_id='recover-output-one')
        self.assertEqual(recovered['status'], 'success', recovered)
        self.assertEqual(recovered['task_operation']['state'], 'accepted')
        self.assertEqual(recovered['task_operation']['action_id'], str(unknown[0]['id']))
        self.assertEqual(self.storage.uploads, uploads)
        first = await self.reader.load_accepted_output(self.owner, self.task_id, plan.digest)
        self.assertEqual(str(first.file.file_id), recovered['file']['id'])
        self.assertEqual(first.action_id, unknown[0]['id'])
        self.assertEqual(first.file.content_sha256, hashlib.sha256(b'opaque synthetic render one').hexdigest())
        replay = await self.writer.publish_output(plan, self.publisher, provider_call_id='repeat-output-one')
        self.assertEqual(replay['file'], recovered['file'])
        self.assertTrue(replay['task_operation']['replayed'])
        self.assertEqual(self.storage.uploads, uploads)
        print('media_publication_sql: lost acknowledgement reconciled without another upload', flush=True)

        previous = await self.reader.previous_recipe(self.owner, saved.file)
        correction = self.plan(source_ref, source.size_bytes, 'corrected-edit', previous=previous)
        corrected_recipe = await self.writer.publish_recipe(correction, self.publisher, provider_call_id='recipe-two')
        self.assertEqual(corrected_recipe['status'], 'success', corrected_recipe)
        second_bytes = await self.local('output-two', b'opaque synthetic render two, different bytes')
        corrected = await self.writer.publish_output(correction, self.publisher, second_bytes, provider_call_id='output-two')
        self.assertEqual(corrected['status'], 'success', corrected)
        second = await self.reader.load_accepted_output(self.owner, self.task_id, correction.digest)
        self.assertNotEqual(first.file, second.file)
        self.assertEqual(second.file.content_sha256, second_bytes.content_sha256)
        self.assertEqual(await self.reader.load_accepted_output(self.owner, self.task_id, plan.digest), first)
        self.assertEqual((await self.reader.load_plan(self.owner, self.task_id, plan.digest)).file, saved.file)
        self.assertEqual(await self.foreground.read_content(owner_id=self.owner, file_id=saved.file.file_id), plan.provenance_bytes)
        print('media_publication_sql: correction preserves original recipe and output', flush=True)

        self.assertIsNone(await self.reader.load_plan(self.owner, uuid4(), plan.digest))
        with self.assertRaises(SourceError):
            await self.reader.load_plan(fixture.source.other, self.task_id, plan.digest)
        old = saved.file.model_copy(update={'version_id': 'wrong-opaque-version'})
        with self.assertRaises(MediaRecipeError):
            await self.reader.previous_recipe(self.owner, old)

        self.assertTrue(await fixture.tasks.finish(fixture.claim, WorkerOutcome(
            TaskStatus.SUCCEEDED, {'fixture': 'media publication state checks, not actual rendering'})))
        self.assertEqual((await fixture.tasks.get(self.owner, self.task_id)).status, TaskStatus.SUCCEEDED)
        actions = await self.action_rows()
        uploads = self.storage.uploads
        self.assertEqual(len(actions), 6)  # control, source, two recipes and two outputs
        self.assertTrue(all(row['status'] == 'accepted' for row in actions), actions)
        self.assertEqual(await self.reader.load_accepted_output(self.owner, self.task_id, plan.digest), first)
        self.assertEqual(await self.reader.load_accepted_output(self.owner, self.task_id, correction.digest), second)
        self.assertEqual(await self.action_rows(), actions)
        self.assertEqual(self.storage.uploads, uploads)
        print('media_publication_sql: foreground reopen survives completed worker without writes', flush=True)

        # Removing the retained recipe must invalidate its dependent outputs and
        # the later correction, without erasing earlier independent originals.
        removed = await self.foreground.delete_source_file(owner_id=self.owner, file_id=saved.file.file_id)
        self.assertTrue(removed['deleted'], removed)
        for selected in (plan, correction):
            await self.assert_unavailable(selected)
        self.assertIsNotNone(await self.foreground.get_file(owner_id=self.owner, file_id=original_file.id))
        self.assertEqual(await self.foreground.read_content(owner_id=self.owner, file_id=UUID(control['file']['id'])),
                         b'{"control":true}')
        self.assertEqual(self.storage.uploads, uploads)
        print('media_publication_sql: recipe withdrawal denies descendants and preserves independent records', flush=True)

    async def local(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        local = await snapshot_file(path, max_bytes=LIMIT)
        self.addCleanup(local.close)
        return local

    @staticmethod
    def reference(file, checksum):
        return RetainedRef(file_id=file.id, version_id=file.memory_source['document_id'], content_sha256=checksum)

    def plan(self, source_ref, source_size, key, *, previous=None):
        value = copy.deepcopy(recipe())
        value['sources'][0]['file'] = source_ref.model_dump(mode='json')
        value['segments'][0].update(in_ms=0, out_ms=5000)
        value['segments'][1].update(in_ms=5000, out_ms=10000)
        value['previous_recipe'] = None if previous is None else previous.file.model_dump(mode='json')
        value['captions'][0]['text'] = 'Initial fixture' if previous is None else 'Corrected fixture'
        inspection = SourceInspection(file=source_ref, duration_ms=10000, size_bytes=source_size,
            video_decodable=True, audio_streams=1, audio_decodable=True)
        binding = RendererBinding(template=value['template'], physical_template_id='synthetic-no-provider',
            build_id='synthetic-no-build', software_sha256='c' * 64, build_receipt_sha256='d' * 64,
            policy=MediaRecipePolicy(max_source_bytes=LIMIT, max_source_duration_ms=10000, max_output_frames=240),
            max_output_bytes=LIMIT)
        checked = validate_recipe(value, inspections=[inspection], policy=binding.policy, previous=previous,
            inventory=[TemplateInventoryEntry(template=value['template'], output_profile=value['output_profile'], ready=True)])
        return prepare_render(owner_id=self.owner, task_id=self.task_id, operation_key=key, recipe=checked,
            source=AuthorizedRenderInput(self.owner, self.task_id, 'synthetic-no-vm', INPUT_ROOT + '/' + source_ref.content_sha256,
                                         inspection), binding=binding, previous=previous)

    async def action_rows(self):
        async with self.f.source.pool.acquire() as connection:
            rows = await connection.fetch('select id,status,operation_key from jarvis_task_actions '
                                          'where owner_id=$1 and task_id=$2 order by id', self.owner, self.task_id)
        return [dict(row) for row in rows]

    async def assert_unavailable(self, plan):
        try:
            output = await self.reader.load_accepted_output(self.owner, self.task_id, plan.digest)
        except (MediaRecipeError, SourceError):
            return
        self.assertIsNone(output)


if __name__ == '__main__':
    unittest.main()
