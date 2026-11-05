"""Offline retained-phase host metadata confinement, without child or SQL startup."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import test_native_manual_capture_mode as manual
import test_native_memory_readiness_runner as memory_runner
from evals.responsive_voice import frontend_host_qualification as host
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.frontend_host import ReplayHostScope
from evals.responsive_voice.native_memory_readiness_control import HOST_FORMAT, config_from_metadata


def scoped_resources(metadata):
    run = uuid4().hex
    scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
        database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(), memory_bank_id='jarvis_'+uuid4().hex,
        expires_at=datetime.now(UTC)+timedelta(seconds=60))
    resources = {key:str(getattr(scope,key)) for key in
        ('run_id','owner_id','database_schema','area_id','memory_bank_id')}
    resources.update(capability_profile=host.NATIVE_MEDIA,
        native_options={'workload':host.NO_WORKLOAD,'voice_route':'realtime'},
        native_memory_readiness=metadata)
    return scope, resources


class MemoryReadinessHostPathTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.origin, self.invocation = uuid4(), uuid4()
        self.sessions = self.root/'sessions'
        self.sessions.mkdir()
        self.fixture = self.sessions/f'00-{self.origin}'
        self.current = self.sessions/f'01-{self.invocation}'
        for path in (self.fixture,self.current):
            path.mkdir(mode=0o700)
        self.metadata = {'format':HOST_FORMAT,'invocation_id':str(self.invocation),
            'origin_invocation_id':str(self.origin),'manifest_sha256':'a'*64,
            'phase_start_seconds':115,'source_anchor_sha256':'b'*64,
            'configuration_path':str(self.current/'memory-readiness-configuration.json'),
            'status_path':str(self.current/'memory-readiness-status.json'),
            'approved_gates':[{'target_input_ordinal':1,'trigger':{
                'kind':'after_manual_response','after_input_ordinal':0,'at_seconds':37,
                'timeout_seconds':18,'canonical_requirement':'finalized_input',
                'playback_requirement':'authenticated_buffer_stop'}}]}
        self.scope,self.resources = scoped_resources(self.metadata)

    def test_phase_one_sibling_preserves_exact_metadata_and_normal_child_composition(self):
        env = host.memory_readiness_child_environment(self.resources,self.scope,self.fixture)
        self.assertEqual(json.loads(env['JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL']),self.metadata)
        with patch.dict('os.environ',{**env,'JARVIS_FRONTEND_REPLAY_PROFILE':host.NATIVE_MEDIA,
                'JARVIS_FRONTEND_REPLAY_WORKLOAD':host.NO_WORKLOAD,
                'JARVIS_FRONTEND_REPLAY_VOICE_ROUTE':'realtime'},clear=True), \
                patch.object(host,'create_replay_application') as create:
            host.child_replay_application(self.scope)
        create.assert_called_once_with(self.scope,native_memory_readiness=config_from_metadata(self.metadata))
        self.assertEqual(list(self.current.iterdir()),[])

    def test_original_same_directory_path_stays_supported(self):
        self.metadata.update(invocation_id=str(self.origin),phase_start_seconds=0,source_anchor_sha256=None,
            configuration_path=str(self.fixture/'memory-readiness-configuration.json'),
            status_path=str(self.fixture/'memory-readiness-status.json'))
        env = host.memory_readiness_child_environment(self.resources,self.scope,self.fixture)
        self.assertEqual(json.loads(env['JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL']),self.metadata)
        self.assertEqual(host.memory_readiness_child_environment({},self.scope,self.fixture),{})

    def test_foreign_or_incomplete_retained_bindings_are_rejected(self):
        for key,value in (('origin_invocation_id',str(uuid4())),('invocation_id',str(uuid4())),
                ('invocation_id',str(self.origin)),('phase_start_seconds',0),('source_anchor_sha256',None)):
            with self.subTest(key=key,value=value):
                resources = deepcopy(self.resources)
                resources['native_memory_readiness'][key] = value
                with self.assertRaisesRegex(host.QualificationFailure,'^memory_readiness_metadata_path$'):
                    host.memory_readiness_child_environment(resources,self.scope,self.fixture)

    def test_only_exact_same_sequence_phase_one_directory_is_accepted(self):
        candidates = (self.root/'other'/'sessions'/self.current.name,
            self.sessions/f'02-{self.invocation}',self.sessions/'arbitrary',self.current/'nested')
        for directory in candidates:
            with self.subTest(directory=directory.name):
                resources = deepcopy(self.resources)
                resources['native_memory_readiness'].update(
                    configuration_path=str(directory/'memory-readiness-configuration.json'),
                    status_path=str(directory/'memory-readiness-status.json'))
                with self.assertRaisesRegex(host.QualificationFailure,'^memory_readiness_metadata_path$'):
                    host.memory_readiness_child_environment(resources,self.scope,self.fixture)
        with self.assertRaisesRegex(host.QualificationFailure,'^memory_readiness_metadata_path$'):
            host.memory_readiness_child_environment(self.resources,self.scope,self.sessions/f'00-{uuid4()}')

    def test_fixed_control_names_and_sibling_status_are_preserved(self):
        for key,path,error in (
                ('configuration_path',self.current/'unrelated.json',host.QualificationFailure),
                ('status_path',self.current/'unrelated.json',host.QualificationFailure),
                ('status_path',self.fixture/'memory-readiness-status.json',ValueError)):
            with self.subTest(key=key,path=path.name):
                resources = deepcopy(self.resources)
                resources['native_memory_readiness'][key] = str(path)
                with self.assertRaises(error):
                    host.memory_readiness_child_environment(resources,self.scope,self.fixture)

    def test_symlinked_phase_or_ancestor_cannot_escape_confinement(self):
        self.current.rmdir()
        outside = self.root/'outside'; outside.mkdir(mode=0o700)
        self.current.symlink_to(outside,target_is_directory=True)
        with self.assertRaises(OSError):
            host.memory_readiness_child_environment(self.resources,self.scope,self.fixture)
        self.current.unlink(); self.current.mkdir(mode=0o700)
        renamed = self.root/'actual-sessions'; self.sessions.rename(renamed)
        self.sessions.symlink_to(renamed,target_is_directory=True)
        with self.assertRaises(OSError):
            host.memory_readiness_child_environment(self.resources,self.scope,self.fixture)

    def test_both_origin_and_current_directory_must_be_private(self):
        from evals.responsive_voice.native_task_barrier import BarrierError
        for directory in (self.fixture,self.current):
            with self.subTest(directory=directory.name):
                directory.chmod(0o755)
                try:
                    with self.assertRaisesRegex(BarrierError,'^control_root_not_private$'):
                        host.memory_readiness_child_environment(self.resources,self.scope,self.fixture)
                finally:
                    directory.chmod(0o700)

    def test_fixture_scope_and_profile_guards_remain_strict(self):
        for key in ('run_id','owner_id','database_schema','area_id','memory_bank_id'):
            with self.subTest(scope=key), self.assertRaisesRegex(host.QualificationFailure,
                    '^memory_readiness_resource_scope_mismatch$'):
                host.memory_readiness_child_environment({**self.resources,key:'foreign'},self.scope,self.fixture)
        mutations = ({'capability_profile':host.NO_MEDIA},
            {'native_options':{'workload':host.REASONING_REMINDER,'voice_route':'realtime'}},
            {'native_options':{'workload':host.NO_WORKLOAD,'voice_route':'live'}},
            {'native_task_replay':{}})
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(host.QualificationFailure,
                    '^memory_readiness_profile_mismatch$'):
                host.memory_readiness_child_environment({**self.resources,**mutation},self.scope,self.fixture)
        visual = {**self.resources,'native_options':{'workload':host.VISUAL_MEMORY,'voice_route':'realtime'}}
        self.assertIn('JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL',
            host.memory_readiness_child_environment(visual,self.scope,self.fixture))


class RetainedMemoryMetadataCompositionTests(unittest.TestCase):
    # Reuse fixture builders without inheriting or recollecting their test cases.
    phase = manual.ManualCaptureTests.phase
    authorization = manual.ManualCaptureTests.authorization
    sequence = manual.ManualCaptureTests.sequence
    gate = memory_runner.MemoryRunnerTests.gate

    def setUp(self):
        manual.ManualCaptureTests.setUp(self)
        self.root = self.root.resolve()

    def test_real_two_phase_loader_and_metadata_reuse_first_fixture_directory(self):
        options,phases = self.sequence()
        for phase in phases:
            manifest = Path(phase['input_manifest'])
            plan = json.loads(manifest.read_bytes())
            plan.update(memory_readiness={'version':1,'scope_file':'memory-readiness-scope.json'})
            plan['utterances'].append({'clip_id':'u01','trigger':{'kind':'after_manual_response',
                'after_input_ordinal':0,'at_seconds':plan['utterances'][0]['trigger']['at_seconds']+5,
                'timeout_seconds':5,
                'canonical_requirement':'finalized_input','playback_requirement':'authenticated_buffer_stop'}})
            manual.write_json(manifest,plan)
            grant_path = Path(phase['risk_authorization'])
            grant = json.loads(grant_path.read_bytes())
            grant.update(memory_readiness=True,input_manifest_sha256=manual.sha(manifest))
            manual.write_json(grant_path,grant)
        retained,selected = runner.load_retained_sequence(options)
        retained['source_anchor'] = {'sha256':'b'*64}
        for index,choice in enumerate(selected):
            choice.artifact_dir.mkdir(mode=0o700,parents=True)
            plan,_ = runner.stage_media(choice,expected_manifest_sha256=manual.sha(choice.input_manifest),
                expected_capture_mode='manual',expected_memory_readiness=True,allow_empty_frames=True)
            retained['index'] = index
            metadata = runner.memory_replay_metadata(choice,json.loads(choice.risk_authorization.read_bytes()),
                plan,retained)
            if index == 0:
                scope,resources = scoped_resources(metadata)
            else:
                resources['native_memory_readiness'] = metadata
            env = host.memory_readiness_child_environment(resources,scope,selected[0].artifact_dir)
            self.assertEqual(json.loads(env['JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL']),metadata)
            self.assertEqual(Path(metadata['configuration_path']).parent,choice.artifact_dir)
        self.assertFalse(options.ledger_dir.exists())


if __name__ == '__main__':
    unittest.main()
