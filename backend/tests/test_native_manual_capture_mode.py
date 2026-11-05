"""Offline manual-mode adoption through the existing runner and stager.

Temporary grants and media are test fixtures, never actual authorizations or
canonical/native acceptance. No SQL, services, providers or simulator are used.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4
import wave

from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.frontend_host_qualification import QualificationFailure


ABSENT = object()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


class ManualCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.serial = 0

    def phase(self, *, mode=ABSENT, frames=1, duration=67, tail=0, at=5):
        self.serial += 1
        root = self.root/('phase-'+str(self.serial))
        speech = root/'speech/speech.json'
        wav = speech.parent/'u01.wav'
        wav.parent.mkdir(parents=True)
        count = 48000
        with wave.open(str(wav), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(struct.pack('<h', 1000)*count)
        write_json(speech, {'format':'jarvis.speech_assets.v1', 'clips':[{
            'id':'u01', 'relative_path':'u01.wav', 'sha256':sha(wav),
            'sample_rate_hz':48000, 'channels':1, 'sample_width_bytes':2,
            'sample_count':count, 'first_active_sample':0,
            'last_active_sample_exclusive':count, 'energy_threshold_abs_pcm16':64,
        }]})
        frame_rows = []
        if frames:
            frame = root/'frame.jpg'
            frame.write_bytes(b'\xff\xd8test-only-signature-fixture\xff\xd9')
            frame_rows.append({'relative_path':'frame.jpg','sha256':sha(frame),'at_seconds':0})
        template = {
            'version':1, 'run_id':str(uuid4()), 'backend_base_url':'http://127.0.0.1:1',
            'owner_id':'compiler-placeholder', 'area_id':str(uuid4()),
            'memory_bank_id':'compiler-placeholder', 'voice_route':'realtime',
            'bearer_token_file':'compiler-placeholder', 'duration_seconds':duration,
            'post_media_capture_seconds':tail, 'speech_manifest':'speech/speech.json',
            'frames':frame_rows, 'utterances':[{'clip_id':'u01','trigger':{'kind':'at_time','at_seconds':at}}],
        }
        if mode is not ABSENT:
            template['capture_mode'] = mode
        manifest = root/'manifest.json'
        write_json(manifest, template)
        artifact = self.root/('out-'+str(self.serial))
        artifact.mkdir()
        exporter = self.root/'exporter.json'
        write_json(exporter, [sys.executable, '{resource_manifest}', '{output}'])
        options = runner.RunOptions(
            input_manifest=manifest, media_root=root, speech_manifest=speech,
            artifact_dir=artifact, ledger_dir=self.root/'ledger',
            risk_authorization=root/'test-authorization.json',
            canonical_export_command=exporter, hindsight_url='http://127.0.0.1:1',
            hindsight_api_key_file=self.root/'unused.key',
            input_mode=runner.PREPARED, scenario='baseline', cuts=0,
        )
        return options, template

    def stage(self, options, *, expected='auto', retained=False):
        return runner.stage_media(options, expected_manifest_sha256=sha(options.input_manifest),
            expected_capture_mode=expected, allow_empty_frames=retained)

    def authorization(self, options, *, mode=ABSENT, binding=None):
        value = {
            'schema_version':1, 'authorization_id':str(uuid4()),
            'expires_at':(datetime.now(UTC)+timedelta(minutes=10)).isoformat(),
            'allow_one_launch':True, 'exclusive_simulator_app':True,
            'scenario':options.scenario, 'simulator_id':options.simulator_id,
            'bundle_id':options.bundle_id, 'input_mode':runner.PREPARED, 'cuts':0,
            'input_manifest_sha256':sha(options.input_manifest),
            'workload':options.workload, 'voice_route':options.voice_route,
            'historical_attempts':[{'audio_session_id':runner.HISTORICAL_UNKNOWN,
                'decision':'risk_accepted', 'evidence_or_rationale':'OFFLINE SYNTHETIC TEST ONLY; not launch authorization'}],
        }
        if mode is not ABSENT:
            value['capture_mode'] = mode
        if binding is not None:
            value['retained_sequence'] = binding
        write_json(options.risk_authorization, value)
        return value

    def sequence(self, modes=('manual','manual'), *, workload=runner.NO_WORKLOAD):
        first, _ = self.phase(mode=modes[0], frames=0)
        second, _ = self.phase(mode=modes[1], frames=0, duration=65, tail=20, at=45)
        first, second = replace(first, workload=workload), replace(second, workload=workload)
        sequence_id = str(uuid4())
        phases = []
        for index, options in enumerate((first, second)):
            phases.append({'invocation_id':str(uuid4()), 'phase_start_seconds':67*index,
                **{key:str(getattr(options,key)) for key in (
                    'input_manifest','media_root','speech_manifest','risk_authorization')}})
        sequence_path = self.root/'sequence.json'
        write_json(sequence_path, {'format':'jarvis.native_replay.retained_sequence.v1',
                                  'sequence_id':sequence_id,'phases':phases})
        previous = None
        for index, options in enumerate((first, second)):
            binding = {'sequence_id':sequence_id, 'sequence_plan_sha256':sha(sequence_path),
                'phase_index':index,'replay_invocation_id':phases[index]['invocation_id'],
                'phase_start_seconds':67*index,'post_media_capture_seconds':20*index,
                'previous_authorization_id':previous,
                'accept_unconfirmed_prior_provider_closure':bool(index)}
            authorization = self.authorization(options, mode=modes[index], binding=binding)
            previous = authorization['authorization_id']
        options = replace(first, artifact_dir=self.root/'sequence-run',
                          retained_sequence=sequence_path, lease_seconds=812)
        return options, phases

    def test_absent_null_and_auto_preserve_automatic_default(self):
        for mode in (ABSENT, None, 'auto'):
            with self.subTest(mode='absent' if mode is ABSENT else mode):
                options, template = self.phase(mode=mode)
                plan, _ = self.stage(options)
                self.assertEqual(runner.declared_capture_mode(plan), 'auto')
                self.assertEqual('capture_mode' in plan, 'capture_mode' in template)
                if mode is not ABSENT:
                    self.assertEqual(plan['capture_mode'], mode)
                self.assertEqual(plan['utterances'], template['utterances'])
                self.assertEqual(plan['duration_seconds'], 67)
                self.assertEqual(plan['post_media_capture_seconds'], 0)

    def test_manual_retained_stages_exact_pcm_and_does_not_copy_fixture_authority(self):
        options, template = self.phase(mode='manual', frames=0)
        plan, records = self.stage(options, expected='manual', retained=True)
        self.assertEqual(plan['capture_mode'], 'manual')
        self.assertEqual(plan['frames'], [])
        self.assertEqual(plan['utterances'], template['utterances'])
        for key in ('run_id','owner_id','area_id','memory_bank_id','backend_base_url',
                    'bearer_token_file','replay_invocation_id','replay_source_clock'):
            self.assertNotIn(key, plan)
        self.assertEqual(sha(options.media_root/'speech/u01.wav'),
                         sha(options.artifact_dir/'inputs/speech/u01.wav'))
        self.assertEqual(len(records), 1)

    def test_unknown_manifest_modes_fail_closed(self):
        for mode in ('automatic','MANUAL','',True,0,[],{}):
            with self.subTest(mode=mode):
                options, _ = self.phase(mode=mode)
                with self.assertRaises(QualificationFailure):
                    self.stage(options, retained=True)
                self.assertFalse((options.artifact_dir/'inputs').exists())

    def test_manifest_mode_must_match_authorization_mode(self):
        for actual, expected in (('manual','auto'),('auto','manual'),(None,'manual')):
            with self.subTest(actual=actual, expected=expected):
                options, _ = self.phase(mode=actual)
                with self.assertRaisesRegex(QualificationFailure, 'capture_mode_authorization_mismatch'):
                    self.stage(options, expected=expected, retained=True)

    def test_invalid_authorization_modes_are_not_defaults(self):
        for mode in (None, True, [], 'automatic', 'MANUAL'):
            with self.subTest(mode=mode):
                options, _ = self.phase()
                self.authorization(options, mode=mode)
                with self.assertRaisesRegex(QualificationFailure, 'invalid_capture_mode_authorization'):
                    runner.validate_risk(options)

    def test_ordinary_one_shot_still_rejects_manual_and_empty_camera(self):
        for mode, expected, frames in (('manual','manual',1),(ABSENT,'auto',0)):
            with self.subTest(mode='absent' if mode is ABSENT else mode):
                options, _ = self.phase(mode=mode, frames=frames)
                with self.assertRaises(QualificationFailure):
                    self.stage(options, expected=expected)

    def test_manual_authorization_requires_retained_binding(self):
        options, _ = self.phase(mode='manual')
        self.authorization(options, mode='manual')
        with self.assertRaisesRegex(QualificationFailure, 'manual_capture_requires_retained_prepared_baseline'):
            runner.validate_risk(options)

    def test_fixed45_legacy_mode_cannot_silently_drop_manual_declaration(self):
        options, template = self.phase(mode='manual', duration=45)
        template.pop('post_media_capture_seconds')
        write_json(options.input_manifest, template)
        options = replace(options, input_mode='baseline', scenario=runner.SCENARIO, cuts=None)
        with self.assertRaises(QualificationFailure):
            self.stage(options, expected='manual')

    def test_manual_rejects_route_task_worker_and_profile_escalation(self):
        changes = [
            {'voice_route':'live'}, {'task_aware':True}, {'scenario':'pause_continue','cuts':1},
            {'workload':runner.REASONING_REMINDER}, {'workload':'unknown'},
            {'worker_model':'unapproved'}, {'worker_api_key_file':Path('/unused')},
            {'worker_python':Path('/unused')}, {'result_hold_seconds':1},
        ]
        for change in changes:
            with self.subTest(change=change):
                options, template = self.phase(mode='manual')
                if 'voice_route' in change:
                    template['voice_route'] = change['voice_route']
                    write_json(options.input_manifest, template)
                with self.assertRaises(QualificationFailure):
                    self.stage(replace(options, **change), expected='manual', retained=True)

    def test_manual_rejects_task_barrier_and_non_time_trigger(self):
        for trigger in ('output_start','response_output_start','after_task_claim','after_input_complete'):
            with self.subTest(trigger=trigger):
                options, template = self.phase(mode='manual')
                template['utterances'][0]['trigger'] = {'kind':trigger}
                write_json(options.input_manifest, template)
                with self.assertRaises(QualificationFailure):
                    self.stage(options, expected='manual', retained=True)
        options, template = self.phase(mode='manual')
        template['task_barrier'] = {'version':1}
        write_json(options.input_manifest, template)
        with self.assertRaises(QualificationFailure):
            self.stage(options, expected='manual', retained=True)

    def test_manual_still_requires_real_timed_speech(self):
        options, template = self.phase(mode='manual', frames=0)
        template['utterances'] = []
        write_json(options.input_manifest, template)
        write_json(options.speech_manifest, {'format':'jarvis.speech_assets.v1','clips':[]})
        with self.assertRaises(QualificationFailure):
            self.stage(options, expected='manual', retained=True)

    def test_manual_does_not_relax_scene_time_hash_or_extra_field_gates(self):
        cases = ('late','hash','authority')
        for case in cases:
            with self.subTest(case=case):
                options, template = self.phase(mode='manual')
                if case == 'late':
                    template['utterances'][0]['trigger']['at_seconds'] = 67
                elif case == 'authority':
                    template['replay_invocation_id'] = str(uuid4())
                write_json(options.input_manifest, template)
                expected_hash = sha(options.input_manifest)
                if case == 'hash':
                    template['duration_seconds'] = 68
                    write_json(options.input_manifest, template)
                with self.assertRaises(QualificationFailure):
                    runner.stage_media(options, expected_manifest_sha256=expected_hash,
                        expected_capture_mode='manual', allow_empty_frames=True)

    def test_two_manual_phases_preserve_source_timing_lease_and_exact_grants(self):
        options, phases = self.sequence()
        retained, selected = runner.load_retained_sequence(options)
        self.assertEqual(retained['minimum_lease_seconds'], 812)
        self.assertEqual(retained['lease_seconds'], 812)
        self.assertEqual([p['phase_start_seconds'] for p in retained['phases']], [0,67])
        self.assertEqual([p['duration_seconds'] for p in retained['phases']], [67,65])
        self.assertEqual([p['post_media_capture_seconds'] for p in retained['phases']], [0,20])
        self.assertEqual([p['invocation_id'] for p in retained['phases']],
                         [p['invocation_id'] for p in phases])
        for choice in selected:
            choice.artifact_dir.mkdir(parents=True)
            plan, _ = self.stage(choice, expected='manual', retained=True)
            self.assertEqual(plan['capture_mode'], 'manual')
        self.assertFalse(options.ledger_dir.exists())

    def test_each_phase_has_its_own_mode_and_visual_profile_needs_no_worker(self):
        options, _ = self.sequence(modes=('auto','manual'), workload=runner.VISUAL_MEMORY)
        retained, selected = runner.load_retained_sequence(options)
        self.assertEqual(len(retained['authorization_ids']), 2)
        self.assertNotEqual(*retained['authorization_ids'])
        self.assertTrue(all(choice.workload == runner.VISUAL_MEMORY for choice in selected))

    def test_retained_phase_mode_mismatch_fails_before_admission(self):
        options, phases = self.sequence()
        path = Path(phases[1]['risk_authorization'])
        document = json.loads(path.read_text())
        document['capture_mode'] = 'auto'
        write_json(path, document)
        with self.assertRaisesRegex(QualificationFailure, 'capture_mode_authorization_mismatch'):
            runner.load_retained_sequence(options)
        self.assertFalse(options.ledger_dir.exists())

    def test_existing_sequence_uuid_and_invocation_binding_remain_exact(self):
        for field in ('sequence_id','replay_invocation_id'):
            with self.subTest(field=field):
                options, phases = self.sequence()
                path = Path(phases[1]['risk_authorization'])
                document = json.loads(path.read_text())
                document['retained_sequence'][field] = str(uuid4())
                write_json(path, document)
                with self.assertRaisesRegex(QualificationFailure, 'retained_sequence_authorization_mismatch'):
                    runner.load_retained_sequence(options)
                self.assertFalse(options.ledger_dir.exists())

    async def test_actual_session_path_carries_manual_grant_to_stager_before_fixture(self):
        options, _ = self.sequence()
        retained, selected = runner.load_retained_sequence(options)
        options = selected[0]
        options.ledger_dir.mkdir()
        retained['journal'] = options.artifact_dir.parent.parent/'native-sequence.json'
        retained['index'] = 0
        lock = os.open(options.ledger_dir/'test.lock', os.O_CREAT|os.O_RDWR, 0o600)
        self.addCleanup(os.close, lock)
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        retained['lock'] = lock
        runner.save(options.ledger_dir/('simulator-'+options.simulator_id.upper()+'.active.json'),
                    {'journal':str(retained['journal'])})
        seen = []
        async def stop_before_fixture(choice):
            seen.append(json.loads((choice.artifact_dir/'inputs/media-plan.json').read_text()))
            raise ValueError('offline fixture boundary; no allocation')
        host = type('Host', (), {})()
        host.prepare = AsyncMock(side_effect=stop_before_fixture)
        process = type('Process', (), {})()
        process.command = AsyncMock(side_effect=AssertionError('native call forbidden'))
        report = await runner._run_session(options, host=host, process=process, retained=retained)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['capture_mode'], 'manual')
        self.assertEqual(report['launch_attempts'], 0)
        self.assertEqual(report['status'], 'failed')
        host.prepare.assert_awaited_once()
        process.command.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
