"""Offline publication-boundary tests; no SQL, storage service, codec or provider.

Real immutable file/snapshot models exercise the adapter. The database and
source authority are explicit fakes, so these do not qualify the SQL join or
real source-dependency withdrawal. Core owns that integration lane.
"""
import copy
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from app.files.generated import ArtifactInspection, ArtifactSnapshot
from app.files.local_file import snapshot_file
from app.files.models import JarvisFile
from app.files.tools import _file_payload
from app.files.transfer import MP4Inspection
from app.tasks.models import digest
from app.workspace.media_publication import (
    MAX_PLAN_BYTES, MediaPublicationStore, publication_requests, restore_plan,
)
from app.workspace.media_recipe import (
    MediaRecipeError, PreviousRecipe, RetainedRef, TemplateInventoryEntry, validate_recipe,
)
from app.workspace.media_render import prepare_render, present_accepted_output
from app.workspace.models import WorkspaceContext
from tests.test_workspace_media_render import inputs


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class StoredPlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = prepare_render(**inputs())

    def restore(self, raw=None, **changes):
        return restore_plan(self.plan.provenance_bytes if raw is None else raw, **({
            'owner_id': self.plan.owner_id, 'task_id': self.plan.task_id,
            'plan_digest': self.plan.digest} | changes))

    def test_exact_plan_round_trip_and_separate_stable_publications(self):
        self.assertEqual(self.restore(), self.plan)
        first = publication_requests(self.plan)
        self.assertEqual(first, publication_requests(self.restore()))
        self.assertEqual([r.format for r in first], ['json', 'mp4'])
        self.assertEqual(len({r.operation_key for r in first}), 2)
        self.assertTrue(all(self.plan.digest in r.operation_key for r in first))

    def test_wrong_owner_task_and_digest_rejected(self):
        for change in ({'owner_id': 'other'}, {'task_id': UUID(int=999)}, {'plan_digest': '0' * 64}):
            with self.subTest(change=change), self.assertRaises(MediaRecipeError):
                self.restore(**change)

    def test_noncanonical_duplicate_unknown_and_changed_layout_rejected_even_with_new_hash(self):
        raw = self.plan.provenance_bytes
        invalid = [json.dumps(json.loads(raw), indent=2).encode(),
                   raw.replace(b'{', b'{"owner_id":"test-owner",', 1)]
        for field, value in [('layout', 'unknown-layout'), ('recipe_sha256', '0' * 64),
                             ('extra', True), ('input_path', '/tmp/arbitrary')]:
            data = json.loads(raw); data[field] = value
            invalid.append(json.dumps(data, sort_keys=True, separators=(',', ':')).encode())
        for value in invalid:
            with self.subTest(value=value[:70]), self.assertRaises(MediaRecipeError):
                self.restore(value, plan_digest=sha(value))

    def test_byte_type_bounds_and_mutated_plan_rejected(self):
        for value in (b'', b'x' * (MAX_PLAN_BYTES + 1), bytearray(self.plan.provenance_bytes)):
            with self.subTest(size=len(value)), self.assertRaises(MediaRecipeError):
                self.restore(value, plan_digest=sha(value))
        with self.assertRaises(MediaRecipeError):
            publication_requests(replace(self.plan, digest='0' * 64))


class PublicationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan = prepare_render(**inputs())
        self.rows, self.records, self.originals, self.bytes = {}, {}, {}, {}
        self.parent_available = True
        self.connection = SimpleNamespace(fetch=AsyncMock(side_effect=self.fetch),
                                          fetchval=AsyncMock(side_effect=lambda *args: self.parent_available))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.pool = SimpleNamespace(acquire=acquire)
        self.tasks = SimpleNamespace(pool=self.pool)
        self.files = SimpleNamespace(sources=SimpleNamespace(pool=self.pool),
            _authorize=Mock(side_effect=self.authorize), get_file=AsyncMock(side_effect=self.get_file),
            read_content=AsyncMock(side_effect=self.read_content), _area_for=AsyncMock(return_value=UUID(int=30)))
        self.store = MediaPublicationStore(self.tasks, self.files)
        self.source_read = self.enterContext(patch('app.workspace.media_publication.read_file_original',
                                                  new=AsyncMock(side_effect=self.read_original)))
        self.recipe_file = self.add_publication(self.plan, 'json', 40)
        self.output_file = self.add_publication(self.plan, 'mp4', 50)
        original_ref = self.plan.recipe.recipe.sources[0].file
        self.add_file(original_ref.file_id, 'original.mp4', 'Original recording', 'video/mp4',
                      self.plan.staging()['size_bytes'], original_ref.content_sha256, {}, original_ref.version_id)
        self.publisher = SimpleNamespace(files=self.files,
            context=SimpleNamespace(tasks=self.tasks, task=SimpleNamespace(owner_id=self.plan.owner_id, id=self.plan.task_id)),
            publish=AsyncMock(return_value={'status': 'success', 'task_operation': {'state': 'accepted'}}))

    def authorize(self, owner_id):
        if owner_id != self.plan.owner_id:
            raise PermissionError('owner mismatch')

    async def fetch(self, query, owner, task, operation_key):
        # Records are preselected by the query parameters. SQL is not executed.
        return copy.deepcopy(self.rows.get((owner, task, operation_key), []))

    async def get_file(self, *, owner_id, file_id):
        self.authorize(owner_id)
        file = self.records.get(file_id)
        return file.model_copy(deep=True) if file else None

    async def read_content(self, *, owner_id, file_id):
        self.authorize(owner_id)
        return self.bytes[file_id]

    async def read_original(self, connection, sources, owner, area, file, *, lock):
        self.assertIs(connection, self.connection)
        self.assertIs(sources, self.files.sources)
        self.assertEqual(owner, self.plan.owner_id)
        self.assertEqual(area, UUID(int=30))
        self.assertFalse(lock)
        if file.id not in self.originals:
            raise PermissionError('source withdrawn')
        content_hash, metadata = self.originals[file.id]
        return SimpleNamespace(content={'content_sha256': content_hash},
                               original_evidence=lambda: copy.deepcopy(metadata)), object()

    def add_file(self, identity, filename, description, mime, size, content_hash, metadata, version=None):
        evidence = {'source_id': str(UUID(int=identity.int + 1000)), 'revision': 1, 'generation': 1,
                    'content_digest': 'b' * 64, 'is_current': True,
                    'document_id': version or 'source_version_' + str(identity)}
        file = JarvisFile(id=identity, session_id=UUID(int=31), filename=filename, content_type=mime,
            storage_bucket='fixture', storage_path=str(identity), size_bytes=size, description=description,
            source='assistant', source_visual_frame_id=None, created_at=datetime(2026, 11, 2, tzinfo=timezone.utc),
            metadata=metadata, memory_source=evidence)
        self.records[identity] = file
        self.originals[identity] = (content_hash, copy.deepcopy(evidence))
        return file

    def add_publication(self, plan, format, number):
        request = publication_requests(plan)[0 if format == 'json' else 1]
        raw = plan.provenance_bytes if format == 'json' else b'fixture only, not real media'
        measured = None if format == 'json' else MP4Inspection(size_bytes=len(raw), content_sha256=sha(raw),
            duration_ms=10000, width=720, height=1280, video_codec='h264', audio_codec='aac')
        inspection = ArtifactInspection(format=format, content_sha256=sha(raw), size_bytes=len(raw), media_inspection=measured)
        snapshot = ArtifactSnapshot(request=request, inspection=inspection, parent_grant_id=UUID(int=35),
                                    task_id=plan.task_id, evidence_payload_id=UUID(int=number + 1))
        action_id, grant_id, payload_id = UUID(int=number + 2), UUID(int=number + 3), UUID(int=number + 4)
        file = self.add_file(UUID(int=number), request.filename, request.description, inspection.content_type,
            len(raw), sha(raw), {'task_artifact': {'task_id': str(plan.task_id), 'action_id': str(action_id),
                                                'area_id': str(UUID(int=30))},
                'generated_artifact': {'operation_key': request.operation_key, 'format': format,
                    'validation': inspection.summary(), 'evidence_payload_id': str(snapshot.evidence_payload_id)}})
        self.bytes[file.id] = raw
        scope = snapshot.model_dump(mode='json')
        row = {'id': action_id, 'grant_id': grant_id, 'operation_key': 'tool-grant:' + grant_id.hex,
            'parameters_digest': digest(scope), 'parameters_payload_id': payload_id, 'scope_payload_id': payload_id,
            'grant_operation_key': 'artifact-snapshot:' + plan.task_id.hex + ':' + digest({'operation_key': request.operation_key}),
            'scope': scope, 'receipt': {'tool': 'publish_generated_file', 'parameters_digest': digest(scope),
                                     'result': {'status': 'success', 'file': _file_payload(file)}},
            'knowledge_area_id': UUID(int=30)}
        self.rows[(plan.owner_id, plan.task_id, request.operation_key)] = [row]
        return file

    def row(self, format='mp4'):
        request = publication_requests(self.plan)[0 if format == 'json' else 1]
        return self.rows[(self.plan.owner_id, self.plan.task_id, request.operation_key)][0]

    async def load(self):
        return await self.store.load_accepted_output(self.plan.owner_id, self.plan.task_id, self.plan.digest)

    async def test_reopen_uses_same_accepted_action_and_file_without_publish(self):
        first = await self.load(); second = await self.load()
        self.assertEqual(first, second)
        self.assertEqual(first.action_id, self.row()['id'])
        self.assertEqual(first.file.file_id, self.output_file.id)
        self.assertEqual(first.file.version_id, self.output_file.memory_source['document_id'])
        self.assertEqual(first.render_plan_sha256, self.plan.digest)
        self.publisher.publish.assert_not_awaited()
        self.assertTrue(all(call.kwargs['file_id'] == self.recipe_file.id for call in self.files.read_content.await_args_list))

    async def test_read_side_cannot_cross_owner_or_database(self):
        with self.assertRaises(PermissionError):
            await self.store.load_plan('foreign', self.plan.task_id, self.plan.digest)
        self.connection.fetch.assert_not_awaited()
        with self.assertRaises(ValueError):
            MediaPublicationStore(SimpleNamespace(pool=object()), self.files)

    async def test_missing_recipe_or_output_has_no_accepted_result(self):
        for format in ('json', 'mp4'):
            request = publication_requests(self.plan)[0 if format == 'json' else 1]
            key = (self.plan.owner_id, self.plan.task_id, request.operation_key)
            saved = self.rows.pop(key)
            self.assertIsNone(await self.load())
            self.rows[key] = saved

    async def test_duplicate_acceptance_and_missing_parent_deny(self):
        request = publication_requests(self.plan)[1]
        rows = self.rows[(self.plan.owner_id, self.plan.task_id, request.operation_key)]
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(MediaRecipeError): await self.load()
        rows.pop(); self.parent_available = False
        with self.assertRaises(MediaRecipeError): await self.load()

    async def test_receipt_scope_and_action_mismatches_deny(self):
        row = self.row()
        changes = [('parameters_digest', '0' * 64), ('parameters_payload_id', UUID(int=999)),
                   ('operation_key', 'different-action'), ('grant_operation_key', 'different-snapshot'),
                   ('knowledge_area_id', None), ('receipt', {'result': []}),
                   ('receipt', {'result': {'file': {'id': 'bad'}}})]
        for key, value in changes:
            original = row[key]; row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(MediaRecipeError): await self.load()
            row[key] = original
        row['receipt']['parameters_digest'] = '0' * 64
        with self.assertRaises(MediaRecipeError): await self.load()

    async def test_altered_file_lineage_opaque_version_and_generation_deny(self):
        file = self.records[self.output_file.id]
        changes = [dict(filename='other.mp4'), dict(metadata={}),
                   dict(memory_source=file.memory_source | {'document_id': 'substitute-version'}),
                   dict(memory_source=file.memory_source | {'generation': 2})]
        for change in changes:
            self.records[file.id] = file.model_copy(update=change)
            with self.subTest(change=change), self.assertRaises(MediaRecipeError): await self.load()
        self.records[file.id] = file

    async def test_missing_or_withdrawn_current_output_denies_without_republishing(self):
        file = self.records.pop(self.output_file.id)
        with self.assertRaises(MediaRecipeError): await self.load()
        self.records[file.id] = file
        del self.originals[file.id]
        with self.assertRaises(PermissionError): await self.load()
        self.publisher.publish.assert_not_awaited()

    async def test_current_hash_change_and_generation_change_deny(self):
        saved = self.originals[self.output_file.id]
        for changed in (('0' * 64, saved[1]), (saved[0], saved[1] | {'generation': 2})):
            self.originals[self.output_file.id] = changed
            with self.assertRaises(MediaRecipeError): await self.load()

    async def test_recipe_bytes_and_post_read_version_must_still_match(self):
        self.bytes[self.recipe_file.id] += b' '
        with self.assertRaises(MediaRecipeError): await self.load()
        self.bytes[self.recipe_file.id] = self.plan.provenance_bytes

        async def changed_during_read(**kwargs):
            file = self.records[self.recipe_file.id]
            self.records[file.id] = file.model_copy(update={'memory_source': file.memory_source | {'document_id': 'changed'}})
            return self.plan.provenance_bytes

        self.files.read_content.side_effect = changed_during_read
        with self.assertRaises(MediaRecipeError): await self.load()

    async def test_operational_failure_is_not_relabelled_absent_or_successful(self):
        self.connection.fetch.side_effect = TimeoutError('database unavailable')
        with self.assertRaises(TimeoutError): await self.load()
        self.publisher.publish.assert_not_awaited()

    async def test_actual_loader_connects_to_exact_version_presentation(self):
        presentations = SimpleNamespace(present=AsyncMock(return_value={'outcome': 'pending', 'displayed': False}))
        ctx = WorkspaceContext(self.plan.owner_id, UUID(int=300), UUID(int=301))
        result = await present_accepted_output(self.plan, ctx=ctx, action='open',
            load_accepted_output=self.store.load_accepted_output, presentations=presentations)
        self.assertEqual(result, {'outcome': 'pending', 'displayed': False})
        presentations.present.assert_awaited_once_with(ctx, action='open', artifact_kind='file',
            artifact_id=self.output_file.id, version_id=self.output_file.memory_source['document_id'])

    async def test_recipe_publication_uses_exact_original_and_provenance(self):
        result = await self.store.publish_recipe(self.plan, self.publisher, provider_call_id='call-one')
        self.assertIs(result, self.publisher.publish.return_value)
        self.publisher.publish.assert_awaited_once_with(publication_requests(self.plan)[0], self.plan.provenance_bytes,
                                                       provider_call_id='call-one')

    async def test_no_publish_after_source_change_or_foreign_worker(self):
        source_id = self.plan.recipe.recipe.sources[0].file.file_id
        self.originals[source_id] = ('0' * 64, self.originals[source_id][1])
        with self.assertRaises(MediaRecipeError):
            await self.store.publish_recipe(self.plan, self.publisher, provider_call_id='call-one')
        self.publisher.context.task.id = UUID(int=999)
        with self.assertRaises(MediaRecipeError):
            await self.store.publish_output(self.plan, self.publisher, provider_call_id='call-one')
        self.publisher.publish.assert_not_awaited()

    async def test_output_requires_accepted_recipe_and_verified_bytes(self):
        request = publication_requests(self.plan)[0]
        key = (self.plan.owner_id, self.plan.task_id, request.operation_key)
        row = self.rows.pop(key)
        with self.assertRaises(MediaRecipeError):
            await self.store.publish_output(self.plan, self.publisher, provider_call_id='call-one')
        self.rows[key] = row
        with self.assertRaises(MediaRecipeError):
            await self.store.publish_output(self.plan, self.publisher, b'raw bytes', provider_call_id='call-one')
        self.publisher.publish.assert_not_awaited()

    async def test_fresh_output_and_unknown_result_delegate_once_to_existing_publisher(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / 'fixture'
            path.write_bytes(b'fixture only; publisher owns MP4 inspection')
            local = await snapshot_file(path, max_bytes=100)
            try:
                self.publisher.publish.return_value = {'status': 'outcome_unknown'}
                result = await self.store.publish_output(self.plan, self.publisher, local, provider_call_id='call-one')
                self.assertEqual(result, {'status': 'outcome_unknown'})
                self.publisher.publish.assert_awaited_once_with(publication_requests(self.plan)[1], local,
                                                               provider_call_id='call-one')
            finally:
                local.close()

    async def test_reconciliation_passes_no_bytes_and_preserves_original_key(self):
        await self.store.publish_output(self.plan, self.publisher, provider_call_id='recovery')
        self.publisher.publish.assert_awaited_once_with(publication_requests(self.plan)[1], None,
                                                       provider_call_id='recovery')

    async def test_correction_keeps_previous_saved_recipe_and_original_output(self):
        previous_ref = RetainedRef(file_id=self.recipe_file.id, version_id=self.recipe_file.memory_source['document_id'],
                                  content_sha256=self.plan.digest)
        previous = await self.store.previous_recipe(self.plan.owner_id, previous_ref)
        self.assertEqual(previous, PreviousRecipe(file=previous_ref, edit_id=self.plan.recipe.recipe.edit_id))
        args = inputs(); value = args['recipe'].recipe.model_dump(mode='json')
        value['previous_recipe'] = previous_ref.model_dump(mode='json')
        value['captions'][0]['text'] = 'Corrected caption'
        args['recipe'] = validate_recipe(value, inspections=[args['source'].inspection], policy=args['binding'].policy,
            inventory=[TemplateInventoryEntry(template=value['template'], output_profile=value['output_profile'], ready=True)],
            previous=previous)
        correction = prepare_render(**(args | {'operation_key': 'edit-two'}), previous=previous)
        before = copy.deepcopy(self.records)
        await self.store.publish_recipe(correction, self.publisher, provider_call_id='correct')
        self.assertNotEqual(correction.digest, self.plan.digest)
        self.assertEqual(self.records, before)
        self.assertEqual((await self.load()).file.file_id, self.output_file.id)

    async def test_predecessor_cannot_be_an_output_or_unaccepted_recipe(self):
        ref = RetainedRef(file_id=self.output_file.id, version_id=self.output_file.memory_source['document_id'],
                          content_sha256=self.row()['scope']['inspection']['content_sha256'])
        with self.assertRaises(MediaRecipeError):
            await self.store.previous_recipe(self.plan.owner_id, ref)


if __name__ == '__main__':
    unittest.main()
