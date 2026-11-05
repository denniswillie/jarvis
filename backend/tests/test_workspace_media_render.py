"""Adapter boundaries only: no SQL, image allocation, render or phone effects."""
import copy
from dataclasses import replace
import json
import shlex
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS
from app.sandbox.file_transfer import INPUT_ROOT
from app.workspace.media_recipe import (
    MediaRecipeError, MediaRecipePolicy, PreviousRecipe, SourceInspection,
    TemplateInventoryEntry, validate_recipe,
)
from app.workspace.media_render import (
    AcceptedMediaOutput, AuthorizedRenderInput, LAUNCHER, RendererBinding,
    prepare_render, present_accepted_output,
)
from app.workspace.models import WorkspaceContext
from tests.test_workspace_media_recipe import recipe, reference


def inputs():
    value = recipe()
    inspection = SourceInspection(file=reference(), duration_ms=30000, size_bytes=12000000,
        video_decodable=True, audio_streams=1, audio_decodable=True)
    binding = RendererBinding(template=value['template'], physical_template_id='test-image',
        build_id='test-build', software_sha256='c'*64, build_receipt_sha256='d'*64,
        policy=MediaRecipePolicy(max_source_bytes=20000000, max_source_duration_ms=30000,
                                max_output_frames=240), max_output_bytes=20000000)
    checked = validate_recipe(value, inspections=[inspection], policy=binding.policy,
        inventory=[TemplateInventoryEntry(template=value['template'], output_profile=value['output_profile'], ready=True)])
    return dict(owner_id='test-owner', task_id=UUID(int=200), operation_key='edit-one', recipe=checked,
        source=AuthorizedRenderInput('test-owner', UUID(int=200), 'test-vm', INPUT_ROOT+'/'+('e'*64), inspection),
        binding=binding)


class MediaRenderPlanTests(unittest.TestCase):
    def test_exact_envelope_command_and_provenance_are_stable(self):
        args = inputs(); plan = prepare_render(**args)
        self.assertEqual(plan, prepare_render(**args))
        self.assertEqual(shlex.split(plan.command().command), [LAUNCHER, plan.input_directory+'/input.json',
            plan.input_directory+'/staging', plan.directory+'/render', plan.output_path])
        self.assertEqual(plan.command().operation_key, 'edit-one')
        self.assertEqual(json.loads(plan.envelope_bytes)['recipe'], args['recipe'].template_props())
        self.assertEqual(json.loads(plan.provenance_bytes)['recipe'], json.loads(args['recipe'].canonical_bytes))
        self.assertEqual(plan.staging()['size_bytes'], 12000000)
        self.assertEqual(plan.staging()['source_from'], args['source'].path)
        self.assertEqual(plan.staging()['source_to'], plan.input_directory+'/staging/inputs/'+('a'*64)+'.mp4')
        # Returned containers are copies, not mutable retained plan state.
        plan.staging()['source_from'] = '/changed'
        self.assertEqual(plan.staging()['source_from'], args['source'].path)

    def test_shell_and_markup_looking_caption_remains_literal_data(self):
        args = inputs(); value = args['recipe'].recipe.model_dump(mode='json')
        text = '$(touch /tmp/no) `whoami` <img src=x> "quote"\nnext'
        value['captions'][0]['text'] = text
        args['recipe'] = validate_recipe(value, inspections=[args['source'].inspection], policy=args['binding'].policy,
            inventory=[TemplateInventoryEntry(template=value['template'], output_profile=value['output_profile'], ready=True)])
        plan = prepare_render(**args)
        self.assertEqual(json.loads(plan.envelope_bytes)['recipe']['captions'][0]['text'], text)
        self.assertNotIn('touch', plan.command().command)
        self.assertNotIn('<img', plan.command().command)

    def test_wrong_owner_task_path_and_allocation_rejected(self):
        args = inputs()
        for change in ({'owner_id':'foreign'}, {'task_id':UUID(int=201)}, {'path':'/tmp/source'},
                       {'path':INPUT_ROOT+'/../source'}, {'path':INPUT_ROOT+'/'+('e'*64)+'; echo x'}):
            with self.subTest(change=change), self.assertRaises(MediaRecipeError):
                prepare_render(**(args | {'source':replace(args['source'], **change)}))
        plan = prepare_render(**args)
        plan.require_allocation(SimpleNamespace(sandbox_id='test-vm', template_id='test-image'))
        for handle in [SimpleNamespace(sandbox_id='other', template_id='test-image'),
                       SimpleNamespace(sandbox_id='test-vm', template_id='other')]:
            with self.assertRaises(MediaRecipeError):plan.require_allocation(handle)

    def test_tampered_validated_result_is_revalidated(self):
        args = inputs()
        for changes in ({'digest':'f'*64}, {'canonical_bytes':b'{}'}, {'duration_in_frames':241}):
            with self.subTest(changes=changes), self.assertRaises(MediaRecipeError):
                prepare_render(**(args | {'recipe':replace(args['recipe'], **changes)}))

    def test_stale_source_or_template_never_substitutes_current(self):
        args = inputs()
        for field, value in [('version_id','newer'), ('content_sha256','f'*64), ('file_id',str(UUID(int=9)))]:
            record = args['source'].inspection.model_dump(mode='json');record['file'][field] = value
            with self.subTest(field=field), self.assertRaises(MediaRecipeError):
                prepare_render(**(args | {'source':replace(args['source'], inspection=SourceInspection.model_validate(record))}))
        binding = args['binding'].model_dump(mode='json');binding['template']['bundle_sha256'] = 'f'*64
        with self.assertRaises(MediaRecipeError):prepare_render(**(args | {'binding':RendererBinding.model_validate(binding)}))

    def test_policy_and_audio_must_fit_actual_template(self):
        args = inputs()
        for field, limit in [('max_source_bytes',CAPTURE_MAX_BYTES + 1),
                             ('max_source_duration_ms',CAPTURE_MAX_DURATION_MS + 1), ('max_output_frames',721)]:
            binding = args['binding'].model_dump(mode='json');binding['policy'][field] = limit
            with self.subTest(field=field), self.assertRaises(MediaRecipeError):
                prepare_render(**(args | {'binding':RendererBinding.model_validate(binding)}))
        record = args['source'].inspection.model_dump(mode='json');record['audio_streams'] = 2
        with self.assertRaises(MediaRecipeError):
            prepare_render(**(args | {'source':replace(args['source'], inspection=SourceInspection.model_validate(record))}))

    def test_correction_keeps_predecessor_and_original_plan(self):
        args = inputs(); first = prepare_render(**args); value = copy.deepcopy(recipe())
        value['previous_recipe'] = reference(2,'f');value['captions'][0]['text'] = 'Correction'
        previous = PreviousRecipe(file=value['previous_recipe'], edit_id=value['edit_id'])
        checked = validate_recipe(value, inspections=[args['source'].inspection], policy=args['binding'].policy,
            inventory=[TemplateInventoryEntry(template=value['template'], output_profile=value['output_profile'], ready=True)], previous=previous)
        revised = prepare_render(**(args | {'operation_key':'edit-two','recipe':checked}), previous=previous)
        self.assertNotEqual(first.digest, revised.digest)
        self.assertNotEqual(first.output_path, revised.output_path)
        self.assertEqual(json.loads(revised.provenance_bytes)['recipe']['previous_recipe'], value['previous_recipe'])
        self.assertIsNone(json.loads(first.provenance_bytes)['recipe']['previous_recipe'])
        with self.assertRaises(MediaRecipeError):prepare_render(**(args | {'recipe':checked}))

    def test_same_key_and_canonical_command_required(self):
        plan=prepare_render(**inputs()); command=plan.command();plan.require_command(command)
        for changes in ({'operation_key':'retry-new'}, {'command':command.command+'; true'}, {'cwd':'somewhere'}):
            with self.subTest(changes=changes), self.assertRaises(MediaRecipeError):
                plan.require_command(command.model_copy(update=changes))


class MediaPresentationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan=prepare_render(**inputs());self.ctx=WorkspaceContext('test-owner',UUID(int=300),UUID(int=301))
        self.output=AcceptedMediaOutput(owner_id='test-owner',task_id=self.plan.task_id,action_id=UUID(int=302),
            render_plan_sha256=self.plan.digest,file=reference(4,'f'),content_type='video/mp4',size_bytes=2000000)
        self.loader=AsyncMock(return_value=self.output)
        self.service=SimpleNamespace(present=AsyncMock(return_value={'outcome':'pending','displayed':False}))

    async def present(self, **changes):
        return await present_accepted_output(self.plan, **({'ctx':self.ctx,'action':'open',
            'load_accepted_output':self.loader,'presentations':self.service} | changes))

    async def test_unrequested_presentation_does_nothing(self):
        self.assertIsNone(await self.present(action=None));self.loader.assert_not_awaited();self.service.present.assert_not_awaited()

    async def test_actual_owner_outcome_is_forwarded_without_display_claim(self):
        for action in ('open','download'):
            result=await self.present(action=action)
            self.assertEqual(result,{'outcome':'pending','displayed':False})
            self.service.present.assert_awaited_with(self.ctx,action=action,artifact_kind='file',
                artifact_id=self.output.file.file_id,version_id=self.output.file.version_id)
        self.loader.assert_awaited_with(self.plan.owner_id,self.plan.task_id,self.plan.digest)

    async def test_missing_stale_foreign_or_oversize_acceptance_cannot_present(self):
        for value in [None,self.output.model_dump(),self.output.model_copy(update={'task_id':UUID(int=999)}),
                      self.output.model_copy(update={'render_plan_sha256':'0'*64}),
                      self.output.model_copy(update={'owner_id':'foreign'}),
                      self.output.model_copy(update={'size_bytes':20000001}),
                      self.output.model_copy(update={'size_bytes':0}),
                      self.output.model_copy(update={'content_type':'text/html'})]:
            self.loader.return_value=value
            with self.subTest(value=value),self.assertRaises(MediaRecipeError):await self.present()
        self.service.present.assert_not_awaited()

    async def test_no_foreground_admission_or_foreign_owner_denied_before_read(self):
        for ctx in [WorkspaceContext('test-owner',UUID(int=300)),WorkspaceContext('foreign',UUID(int=300),UUID(int=301))]:
            with self.assertRaises(MediaRecipeError):await self.present(ctx=ctx)
        self.loader.assert_not_awaited();self.service.present.assert_not_awaited()

    async def test_access_withdrawal_and_unknown_outcome_propagate_without_retry(self):
        self.service.present.side_effect=RuntimeError('source withdrawn')
        with self.assertRaisesRegex(RuntimeError,'withdrawn'):await self.present()
        self.service.present.assert_awaited_once();self.loader.assert_awaited_once()


if __name__=='__main__':unittest.main()
