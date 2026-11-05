"""First-phase audio-only retained support through actual runner/stager checks.

Existing compiler fixtures provide structural WAV bytes, not new scenario speech.
Host preparation is deliberately stopped after real staging; no native/model/SQL
or memory/response/index acceptance is implied. Full lifecycle coverage remains
in the existing retained-sequence suite.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from evals.responsive_voice import native_replay_run as runner
from tests.test_native_replay_run import Fixture


def retained_audio_only(root):
    """Two separately bound admissions using existing compiler/input fixtures."""
    f = Fixture(root)
    f.prepared('baseline', 0, duration=45)
    first = f.options.input_manifest
    payload = runner.read_json(first)
    payload['frames'] = []
    payload['post_media_capture_seconds'] = 20
    runner.save(first, payload)
    second = first.with_name('second-phase.json')
    runner.save(second, payload)
    sequence_path = root/'sequence.json'
    sequence = {'format':'jarvis.native_replay.retained_sequence.v1',
                'sequence_id':str(uuid4()), 'phases':[]}
    grants = [f.options.risk_authorization, root/'risk-second.json']
    authorizations = [str(uuid4()), str(uuid4())]
    for index, manifest in enumerate((first, second)):
        sequence['phases'].append({'invocation_id':str(uuid4()), 'phase_start_seconds':45*index,
            'input_manifest':str(manifest), 'media_root':str(manifest.parent),
            'speech_manifest':str(f.options.speech_manifest), 'risk_authorization':str(grants[index])})
    runner.save(sequence_path, sequence)
    sequence_sha = runner.scenarios.digest(sequence_path.read_bytes())
    original_grant = runner.read_json(grants[0])
    for index, phase in enumerate(sequence['phases']):
        document = {**original_grant, 'authorization_id':authorizations[index],
            'input_mode':runner.PREPARED, 'scenario':'baseline', 'cuts':0,
            'input_manifest_sha256':runner.scenarios.digest(Path(phase['input_manifest']).read_bytes()),
            'voice_route':'realtime', 'workload':'none', 'task_aware':False,
            'retained_sequence':{
                'sequence_id':sequence['sequence_id'], 'sequence_plan_sha256':sequence_sha,
                'phase_index':index, 'replay_invocation_id':phase['invocation_id'],
                'phase_start_seconds':phase['phase_start_seconds'], 'post_media_capture_seconds':20,
                'previous_authorization_id':authorizations[index-1] if index else None,
                'accept_unconfirmed_prior_provider_closure':bool(index)}}
        runner.save(grants[index], document)
    f.options = replace(f.options, retained_sequence=sequence_path, lease_seconds=790)
    return f, sequence, grants, authorizations


class RetainedFirstZeroCameraTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_sequence_admits_first_audio_only_stage_before_existing_host_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            f, sequence, grants, identities = retained_audio_only(Path(directory))
            seen = []

            async def stop_after_staging(options):
                f.prepares += 1
                seen.append(options)
                raise RuntimeError('OFFLINE_STOP_AFTER_REAL_STAGE_BEFORE_ALLOCATION')

            f.prepare = stop_after_staging
            with patch.object(runner, 'stage_prepared_media', wraps=runner.stage_prepared_media) as stage:
                result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(stage.call_count, 1)
            self.assertIs(stage.call_args.kwargs['allow_empty_frames'], True)
            self.assertEqual((f.prepares, f.launches, f.cleanups), (1, 0, 0))
            self.assertEqual(len(result['phases']), 1)
            phase = result['phases'][0]
            self.assertEqual(phase['fatal']['error_type'], 'RuntimeError')
            self.assertEqual(phase['launch_attempts'], 0)
            self.assertEqual(result['status'], 'failed')  # Deliberate host boundary stop.
            self.assertTrue(result['local_cleanup_complete'])
            self.assertEqual((result['minimum_lease_seconds'], result['lease_seconds']), (790, 790))
            staged = runner.read_json(seen[0].artifact_dir/'inputs/media-plan.json')
            self.assertEqual(staged['frames'], [])
            self.assertEqual(staged['voice_route'], 'realtime')
            self.assertEqual((staged['duration_seconds'], staged['post_media_capture_seconds']), (45, 20))
            self.assertTrue(staged['utterances'])
            self.assertTrue(all(x['trigger']['kind']=='at_time' for x in staged['utterances']))
            speech = runner.read_json(seen[0].artifact_dir/'inputs/speech/speech.json')
            self.assertEqual(len(speech['clips']), 1)
            self.assertGreater(speech['clips'][0]['sample_count'], 0)
            self.assertEqual(len(phase['media']), 1)
            self.assertTrue(phase['media'][0]['path'].endswith('.wav'))
            self.assertTrue((f.options.ledger_dir/f'authorization-{identities[0]}.spent').is_file())
            self.assertFalse((f.options.ledger_dir/f'authorization-{identities[1]}.spent').exists())
            self.assertFalse(list(f.options.ledger_dir.glob('*.active.json')))

    def test_both_exactly_bound_phases_stage_real_audio_without_dummy_images(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = retained_audio_only(Path(directory))
            retained, phases = runner.load_retained_sequence(f.options)
            self.assertEqual(retained['minimum_lease_seconds'], 790)
            for index, options in enumerate(phases):
                with self.subTest(index=index):
                    options.artifact_dir.mkdir(parents=True)
                    authorization = runner.read_json(options.risk_authorization)
                    plan, records = runner.stage_media(options,
                        expected_manifest_sha256=authorization['input_manifest_sha256'],
                        allow_empty_frames=True)
                    self.assertEqual(plan['frames'], [])
                    self.assertEqual(len(records), 1)
                    self.assertTrue(records[0]['path'].endswith('.wav'))
                    self.assertTrue(plan['utterances'])

    async def test_ordinary_one_shot_still_rejects_empty_camera_with_valid_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, grants, _ = retained_audio_only(Path(directory))
            document = runner.read_json(grants[0])
            document.pop('retained_sequence')
            runner.save(grants[0], document)
            options = replace(f.options, retained_sequence=None)
            result = await runner.run_once(options, host=f, process=f)
            self.assertEqual(result['fatal']['code'], 'invalid_frames')
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_sequence_grant_cannot_be_used_as_an_ordinary_one_shot(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = retained_audio_only(Path(directory))
            with self.assertRaisesRegex(Exception, 'sequence_authorization_requires_retained_runner'):
                await runner.run_once(replace(f.options, retained_sequence=None), host=f, process=f)
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertFalse(f.options.ledger_dir.exists())

    async def test_manifest_hash_drift_still_blocks_before_host_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = retained_audio_only(Path(directory))
            manifest = runner.read_json(f.options.input_manifest)
            manifest['utterances'][0]['trigger']['at_seconds'] += 1
            runner.save(f.options.input_manifest, manifest)
            result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(result['phases'][0]['fatal']['code'], 'authorized_manifest_hash_mismatch')
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_each_phase_binding_is_required_before_any_stage_or_admission(self):
        corruptions = {
            'sequence_id':str(uuid4()), 'sequence_plan_sha256':'0'*64,
            'phase_index':9, 'replay_invocation_id':str(uuid4()),
            'phase_start_seconds':99, 'post_media_capture_seconds':0,
            'previous_authorization_id':'not-the-previous-admission'}
        for index in (0, 1):
            for key, value in corruptions.items():
                with self.subTest(index=index, key=key), tempfile.TemporaryDirectory() as directory:
                    f, _, grants, _ = retained_audio_only(Path(directory))
                    document = runner.read_json(grants[index])
                    document['retained_sequence'][key] = value
                    runner.save(grants[index], document)
                    with self.assertRaisesRegex(Exception, 'retained_sequence_authorization_mismatch'):
                        await runner.run_once(f.options, host=f, process=f)
                    self.assertEqual((f.prepares, f.launches), (0, 0))
                    self.assertFalse(f.options.ledger_dir.exists())

    async def test_second_phase_cannot_reuse_first_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, grants, identities = retained_audio_only(Path(directory))
            document = runner.read_json(grants[1])
            document['authorization_id'] = identities[0]
            runner.save(grants[1], document)
            with self.assertRaisesRegex(Exception, 'retained_authorization_reused'):
                await runner.run_once(f.options, host=f, process=f)
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertFalse(f.options.ledger_dir.exists())

    async def test_first_camera_free_phase_requires_declared_timed_audio(self):
        for inputs in ([], [{'clip_id':'unused','trigger':{'kind':'output_start',
                         'after_cursor':0,'delay_seconds':.1,'timeout_seconds':1}}]):
            with self.subTest(inputs=inputs), tempfile.TemporaryDirectory() as directory:
                f, _, _, _ = retained_audio_only(Path(directory))
                document = runner.read_json(f.options.input_manifest)
                document['utterances'] = inputs
                runner.save(f.options.input_manifest, document)
                with self.assertRaisesRegex(Exception, 'retained_phase_requires_timed_speech'):
                    await runner.run_once(f.options, host=f, process=f)
                self.assertEqual((f.prepares, f.launches), (0, 0))
                self.assertFalse(f.options.ledger_dir.exists())

    async def test_declared_input_cannot_substitute_missing_selected_speech(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = retained_audio_only(Path(directory))
            runner.save(f.options.speech_manifest, {'format':'jarvis.speech_assets.v1','clips':[]})
            result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(result['phases'][0]['fatal']['code'], 'empty_prepared_input_mismatch')
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_declared_audio_still_requires_exact_existing_wav_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = retained_audio_only(Path(directory))
            speech = runner.read_json(f.options.speech_manifest)
            wav = f.options.speech_manifest.parent/speech['clips'][0]['relative_path']
            wav.write_bytes(wav.read_bytes()+b'UNAUTHORIZED_TEST_TAIL')
            authorization = runner.read_json(f.options.risk_authorization)
            with self.assertRaisesRegex(runner.scenarios.ScenarioError, 'speech_hash_mismatch'):
                runner.stage_media(f.options,
                    expected_manifest_sha256=authorization['input_manifest_sha256'],
                    allow_empty_frames=True)
            result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(result['phases'][0]['fatal']['error_type'], 'ScenarioError')
            self.assertIsNone(result['phases'][0]['fatal']['code'])
            self.assertEqual((f.prepares, f.launches), (0, 0))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_audio_only_does_not_expand_retained_route_or_mode_scope(self):
        changes = ({'voice_route':'live'}, {'task_aware':True}, {'workload':'reasoning-reminder'},
                   {'cuts':1}, {'scenario':'quiet'}, {'input_mode':'baseline'})
        for values in changes:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                f, _, _, _ = retained_audio_only(Path(directory))
                with self.assertRaisesRegex(Exception, 'retained_sequence_requires_prepared_realtime_baseline'):
                    await runner.run_once(replace(f.options, **values), host=f, process=f)
                self.assertEqual((f.prepares, f.launches), (0, 0))
                self.assertFalse(f.options.ledger_dir.exists())
