"""Actual prepared stager/retained-loader/child-config paths, entirely offline."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

import test_native_manual_capture_mode as manual
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.frontend_host import ReplayHostScope
from evals.responsive_voice import frontend_host_qualification as host
from evals.responsive_voice.native_memory_readiness_control import SCOPE_NAME, config_from_metadata


class MemoryRunnerTests(unittest.TestCase):
    def setUp(self):
        manual.ManualCaptureTests.setUp(self)
        self.root = self.root.resolve()
    phase=manual.ManualCaptureTests.phase
    authorization=manual.ManualCaptureTests.authorization
    sequence=manual.ManualCaptureTests.sequence

    def gate(self,options):
        value=json.loads(options.input_manifest.read_bytes())
        value['memory_readiness']={'version':1,'scope_file':SCOPE_NAME}
        value['utterances'].append({'clip_id':'u01','trigger':{'kind':'after_manual_response',
            'after_input_ordinal':0,'at_seconds':10,'timeout_seconds':15,
            'canonical_requirement':'retain_searchable','playback_requirement':'authenticated_buffer_stop'}})
        manual.write_json(options.input_manifest,value)
        return value

    def stage(self,options,enabled=True):
        return runner.stage_media(options,expected_manifest_sha256=manual.sha(options.input_manifest),
            expected_capture_mode='manual',expected_memory_readiness=enabled,allow_empty_frames=True)

    def test_actual_stager_camera_free_first_phase_exact_gates_and_pcm(self):
        options,_=self.phase(mode='manual',frames=0)
        template=self.gate(options)
        plan,records=self.stage(options)
        self.assertEqual(plan['utterances'],template['utterances'])
        self.assertEqual(plan['memory_readiness'],template['memory_readiness'])
        self.assertEqual(plan['frames'],[])
        self.assertEqual(len(records),1)
        self.assertEqual(manual.sha(options.media_root/'speech/u01.wav'),
                         manual.sha(options.artifact_dir/'inputs/speech/u01.wav'))

    def test_explicit_authorization_mode_route_prior_and_bounds_fail_closed(self):
        for mutation in ('grant','auto','live','prior','timeout','late','scope'):
            with self.subTest(mutation=mutation):
                options,_=self.phase(mode='manual',frames=0)
                value=self.gate(options)
                if mutation=='auto':value['capture_mode']='auto'
                if mutation=='live':value['voice_route']='live'
                if mutation=='prior':value['utterances'][1]['trigger']['after_input_ordinal']=1
                if mutation=='timeout':value['utterances'][1]['trigger']['timeout_seconds']=121
                if mutation=='late':value['utterances'][1]['trigger']['at_seconds']=60
                if mutation=='scope':value['memory_readiness']['scope_file']='../other.json'
                manual.write_json(options.input_manifest,value)
                with self.assertRaises(Exception):self.stage(options,enabled=mutation!='grant')
                self.assertFalse((options.artifact_dir/'inputs').exists())

    def test_fresh_per_phase_grants_exact_manifest_and_default_other_phase(self):
        options,phases=self.sequence()
        self.gate(options)
        grant=json.loads(options.risk_authorization.read_bytes())
        grant.update(memory_readiness=True,input_manifest_sha256=manual.sha(options.input_manifest))
        manual.write_json(options.risk_authorization,grant)
        retained,selected=runner.load_retained_sequence(options)
        self.assertEqual(len(retained['authorization_ids']),2)
        self.assertEqual(retained['minimum_lease_seconds'],812)
        for index,choice in enumerate(selected):
            choice.artifact_dir.mkdir(parents=True)
            plan,_=self.stage(choice,enabled=index==0)
            self.assertEqual('memory_readiness' in plan,index==0)
        self.assertFalse(options.ledger_dir.exists())

    def test_missing_or_escalated_grant_fails_before_admission(self):
        for flag in (None,False,'true',True):
            with self.subTest(flag=flag):
                options,_=self.sequence()
                self.gate(options)
                grant=json.loads(options.risk_authorization.read_bytes())
                grant['input_manifest_sha256']=manual.sha(options.input_manifest)
                if flag is not None:grant['memory_readiness']=flag
                if flag is True:grant['capture_mode']='auto'
                manual.write_json(options.risk_authorization,grant)
                with self.assertRaises(Exception):runner.load_retained_sequence(options)
                self.assertFalse(options.ledger_dir.exists())

    def test_owned_fixture_scope_and_authorized_input_sha_not_runtime_hash(self):
        options,_=self.sequence();self.gate(options)
        grant=json.loads(options.risk_authorization.read_bytes())
        grant.update(memory_readiness=True,input_manifest_sha256=manual.sha(options.input_manifest))
        manual.write_json(options.risk_authorization,grant)
        retained,selected=runner.load_retained_sequence(options)
        choice=selected[0];choice.artifact_dir.mkdir(mode=0o700,parents=True)
        plan,_=self.stage(choice)
        retained['index']=0
        metadata=runner.memory_replay_metadata(choice,grant,plan,retained)
        run=uuid4().hex
        resources={'run_id':run,'owner_id':f'direct-image-eval-second-brain-{run}-a',
            'database_schema':f'jarvis_sb_eval_{run}_0','area_id':str(uuid4()),
            'memory_bank_id':'jarvis_'+uuid4().hex}
        staged=self.root/'runtime'/'inputs';staged.mkdir(parents=True,mode=0o700)
        controls=staged.parent/'controls';controls.mkdir(mode=0o700)
        plan.update({k:resources[k] for k in ('run_id','owner_id','area_id','memory_bank_id')})
        plan['memory_readiness']['manifest_sha256']=grant['input_manifest_sha256']
        runner.save(staged/'replay.json',plan)
        configured=runner.configure_memory_replay(metadata,resources=resources,staged=staged,
            controls=controls,plan=plan,artifact_dir=choice.artifact_dir)
        self.assertEqual(json.loads((staged/SCOPE_NAME).read_bytes()),resources)
        self.assertEqual(configured['manifest_sha256'],grant['input_manifest_sha256'])
        self.assertEqual(configured['runtime_manifest_sha256'],manual.sha(staged/'replay.json'))
        self.assertNotEqual(configured['runtime_manifest_sha256'],configured['manifest_sha256'])
        scope=ReplayHostScope(**resources,expires_at=datetime.now(UTC)+timedelta(seconds=60))
        resources.update(capability_profile=host.NATIVE_MEDIA,native_options={
            'workload':host.NO_WORKLOAD,'voice_route':'realtime'},native_memory_readiness=metadata)
        env=host.memory_readiness_child_environment(resources,scope,choice.artifact_dir)
        with patch.dict('os.environ',{**env,'JARVIS_FRONTEND_REPLAY_PROFILE':host.NATIVE_MEDIA,
            'JARVIS_FRONTEND_REPLAY_WORKLOAD':host.NO_WORKLOAD,'JARVIS_FRONTEND_REPLAY_VOICE_ROUTE':'realtime'},clear=True), \
                patch.object(host,'create_replay_application') as create:
            host.child_replay_application(scope)
        self.assertEqual(create.call_args.kwargs['native_memory_readiness'],config_from_metadata(metadata))
        for key,value in (('database_schema','wrong'),('memory_bank_id','jarvis_'+uuid4().hex)):
            with self.subTest(key=key),self.assertRaises(Exception):
                host.memory_readiness_child_environment({**resources,key:value},scope,choice.artifact_dir)


if __name__=='__main__':unittest.main()
