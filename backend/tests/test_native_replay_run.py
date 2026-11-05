"""Offline lifecycle boundaries only: no simulator, HTTP, SQL or provider calls."""
import asyncio
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch
from uuid import UUID, uuid4

from evals.responsive_voice.frontend_host_qualification import (
    AdmissionTrace, NATIVE_MEDIA, NO_MEDIA, configure_native_diagnostic_logging,
    sanitized_webrtc_diagnostic, native_host_options, child_environment, ReplayHostScope, native_route_observation,
    NO_WORKLOAD, REASONING_REMINDER,
)
from evals.responsive_voice.native_replay_run import (
    HISTORICAL_UNKNOWN, SCENARIO, PREPARED, MAX_CAPTURE_PCM_BYTES, ProcessBoundary, RunOptions, provider_evidence,
    capture_budget, read_json, run_once, save, stage_media, host_options, HostBoundary,
)
from evals.responsive_voice import native_interruption_scenarios as compiler


class Fixture:
    def __init__(self, root):
        self.root, self.events = root, []
        self.container = root/'container'
        self.container.mkdir()
        media = root/'media'
        media.mkdir()
        frames = []
        for index, at in enumerate((20,22,24)):
            content = b'offline-image-fixture-'+str(index).encode()
            path = media/f'{index}.jpg'
            path.write_bytes(content)
            frames.append({'at_seconds':at,'relative_path':path.name,'sha256':hashlib.sha256(content).hexdigest()})
        speech = media/'question.wav'
        speech.write_bytes(b'offline-pcm-boundary-fixture')
        save(media/'speech.json',{'format':'jarvis.speech_assets.v1','clips':[{
            'id':'current_activity','relative_path':speech.name,'sha256':hashlib.sha256(speech.read_bytes()).hexdigest(),
            'sample_rate_hz':48000,'channels':1,'sample_width_bytes':2,'sample_count':4,
            'first_active_sample':0,'last_active_sample_exclusive':4,'energy_threshold_abs_pcm16':128}]})
        template = {'version':1,'run_id':'old-run-do-not-reuse','owner_id':'old-owner',
            'bearer_token_file':'missing-old-token','backend_base_url':'http://old.invalid',
            'voice_route':'realtime','duration_seconds':45,'speech_manifest':'speech.json','frames':frames,
            'utterances':[{'clip_id':'current_activity','trigger':{'kind':'at_time','at_seconds':25}}]}
        save(media/'replay.json',template)
        self.options = RunOptions(input_manifest=media/'replay.json',media_root=media,speech_manifest=media/'speech.json',
            artifact_dir=root/'run',ledger_dir=root/'ledger',risk_authorization=root/'risk.json',
            canonical_export_command=root/'export-command.json',hindsight_url='http://127.0.0.1:9',
            hindsight_api_key_file=root/'unused-key')
        save(self.options.risk_authorization, {'schema_version':1,'authorization_id':str(uuid4()),
            'allow_one_launch':True,'exclusive_simulator_app':True,'scenario':SCENARIO,
            'simulator_id':self.options.simulator_id,'bundle_id':self.options.bundle_id,
            'expires_at':(datetime.now(UTC)+timedelta(minutes=5)).isoformat(),
            'historical_attempts':[{'audio_session_id':HISTORICAL_UNKNOWN,'decision':'risk_accepted',
                'evidence_or_rationale':'OFFLINE TEST ONLY; no native launch or provider authorization.'}]})
        save(self.options.canonical_export_command,[sys.executable,'offline-export-boundary','{resource_manifest}','{output}'])
        self.run = uuid4().hex
        self.launches = self.prepares = self.cleanups = 0
        self.fail_launch = self.fail_export = self.fail_terminate = self.block_capture = False
        self.launched = asyncio.Event()
        self.child_stopped = False
        self.plan = None
        self.capture_wait = None
        self.remaining_lease = None
        self.worker_starts = self.worker_closes = 0
        self.worker_bad_readiness = self.worker_bad_close = self.worker_fail_start = False
        self.worker_start_clean = True
        self.block_worker_ready = False
        self.worker_started = asyncio.Event()

    def live_baseline(self):
        self.options=replace(self.options,voice_route='live',live_model='offline-live-model',live_voice='marin')
        manifest=read_json(self.options.input_manifest)
        manifest['voice_route']='live'
        save(self.options.input_manifest,manifest)
        if self.options.input_mode==PREPARED:
            self.authorize_prepared()
        document=read_json(self.options.risk_authorization)
        document.update(voice_route='live',live_model=self.options.live_model,live_voice=self.options.live_voice)
        save(self.options.risk_authorization,document)

    def integrated(self):
        source = self.root/'caller-owned-worker-key'
        source.write_text('offline-key-not-a-credential')
        source.chmod(0o600)
        self.options=replace(self.options,workload=REASONING_REMINDER,worker_model='offline-approved-model',
            worker_api_key_file=source,worker_python=Path(sys.executable),worker_reasoning_effort='low')
        document=read_json(self.options.risk_authorization)
        document.update(workload=REASONING_REMINDER,frontier_model=self.options.worker_model,
                        frontier_reasoning_effort=self.options.worker_reasoning_effort)
        save(self.options.risk_authorization,document)

    async def start_worker(self, options, resources):
        self.worker_starts += 1
        self.worker_started.set()
        self.events.append('worker_start')
        self.readiness_receipt={key:resources[key] for key in ('run_id','owner_id','database_schema','area_id','memory_bank_id')}
        self.readiness_receipt.update(status='ready',registry_ready=True,model=options.worker_model,
            reasoning_effort=options.worker_reasoning_effort,worker_id='offline-worker',host_id='offline-host',pid=123)
        if self.worker_bad_readiness:
            self.readiness_receipt['model']='wrong-model'
        if self.worker_fail_start:
            error=RuntimeError('offline_worker_start_failed')
            error.receipt=await self.close() if self.worker_start_clean else {'status':'failed'}
            error.resource_manifest=self.root/'offline-worker-resource.json'
            raise error
        if self.block_worker_ready:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as error:
                error.receipt={'close_receipt':await self.close()}
                raise
        self.events.append('worker_registry_ready')
        return self

    async def close(self):
        self.worker_closes += 1
        self.events.append('worker_closed')
        return {'status':'failed' if self.worker_bad_close else 'closed',
                'worker_joined':not self.worker_bad_close,'readiness_inactive':not self.worker_bad_close,
                'pipes_complete':True,'secrets_removed':True,'registry_state':'withdrawn',
                'evidence_kind':'OFFLINE_WORKER_BOUNDARY_FAKE_NOT_REGISTRY_ACCEPTANCE'}

    def prepared(self, case='correction_chain', cuts=1, duration=90):
        """Actual compiler, synthetic WAV/JPEG fixtures; no media quality claim."""
        buffer = io.BytesIO()
        with wave.open(buffer,'wb') as wav:
            wav.setparams((1,2,48000,0,'NONE','not compressed'))
            wav.writeframes(b'\xe8\x03'*960)
        data = buffer.getvalue()
        keys = ['baseline/current_view','setup/background_task_setup','setup/reminder_setup','setup/memory_setup',
            *('conversation/'+name for name in ('speech_only_pause','continue_answer','ambiguous_stop',
              'parallel_question','keep_working','correction_left','correction_right','correction_label',
              'explicit_cancel','repeat_cancel')),
            *('state/'+name for name in ('operation_status','quiet_until_resume','quiet_direct_question',
              'resume_summary','correct_notebook','recall_notebook','forget_notebook','cancel_if_pending'))]
        clips = {key:compiler.Clip(*key.split('/'),data,compiler.digest(data),compiler.digest(b'\xe8\x03'*960),
                                  960,0,960,128) for key in keys}
        frame = self.root/'synthetic.jpg'
        frame.write_bytes(b'\xff\xd8OFFLINE SIGNATURE ONLY NOT IMAGE QUALITY\xff\xd9')
        authority = self.root/'offline-expectations.txt'
        authority.write_text('\n'.join(identity for values in compiler.CASES.values() for identity in values))
        compiled = compiler.compile_scenario(case,cuts=cuts,duration=duration,clips=clips,authority=authority,
            binding={'run_id':uuid4().hex,'backend_base_url':'http://127.0.0.1:1','owner_id':'old-owner',
                     'area_id':'old-area','memory_bank_id':'old-bank','voice_route':'realtime'},
            frame_plan=[{'at_seconds':at,'relative_path':frame.name,'sha256':compiler.digest(frame.read_bytes())}
                        for at in (1,2,3)],frame_root=self.root)
        output = self.root/'compiled'
        compiler.write_inputs(compiled,output)
        # Neither this token nor the compiler's expectations may enter app inputs.
        (output/'token.txt').write_text('old-token-do-not-copy')
        self.options = replace(self.options,input_manifest=output/'manifest.json',media_root=output,
            speech_manifest=output/'speech.json',input_mode=PREPARED,scenario=case,cuts=cuts)
        self.authorize_prepared()
        return compiled

    def authorize_prepared(self):
        document = read_json(self.options.risk_authorization)
        document.update(input_mode=PREPARED,scenario=self.options.scenario,cuts=self.options.cuts,
            input_manifest_sha256=compiler.digest(self.options.input_manifest.read_bytes()))
        save(self.options.risk_authorization,document)

    def response_prepared(self, cuts=1):
        """Frozen trigger input fixtures, not native response/cut evidence."""
        self.prepared('correction_chain',cuts)
        manifest = read_json(self.options.input_manifest)
        for ordinal,item in enumerate(manifest['utterances']):
            trigger = item['trigger']
            if trigger['kind']=='output_start':
                item['trigger'] = {'kind':'response_output_start','after_input_ordinal':ordinal-1,
                    'delay_seconds':.3,'timeout_seconds':trigger['timeout_seconds'],'max_pcm_age_seconds':.05}
        save(self.options.input_manifest,manifest)
        self.authorize_prepared()
        return manifest

    async def prepare(self, options):
        self.prepares += 1
        self.lease_seconds = options.lease_seconds
        self.native_options = host_options(options)
        self.events.append('prepare')
        self.manifest = options.artifact_dir/f'frontend-host-resources-{self.run}.json'
        self.token = options.artifact_dir/f'frontend-host-bearer-{self.run}.token'
        fd = os.open(self.token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as stream:
            stream.write('offline-new-token-not-a-credential\n')
        self.resource = {'run_id':self.run,'owner_id':f'direct-image-eval-second-brain-{self.run}-a',
            'area_id':str(uuid4()),'memory_bank_id':'jarvis_'+uuid4().hex,
            'database_schema':f'jarvis_sb_eval_{self.run}_0','bearer_token_file':str(self.token),
            'cleanup_command':['offline-exact-fixture-cleanup',str(self.manifest)]}
        save(self.manifest,self.resource)
        return {'status':'prepared','resource_manifest':str(self.manifest)}

    async def serve(self, manifest, stop, ready):
        self.events.append('serve')
        info = {**self.resource,'backend_base_url':'http://127.0.0.1:9','capability_profile':NATIVE_MEDIA,
                'native_options':self.native_options,
                'database_pool':{'min_size':1,'max_size':self.native_options['native_pool_max_size']},
                'provider_configuration':{'voice_route':self.native_options['voice_route'],
                    'selected_model':self.native_options['live_model'],'selected_voice':self.native_options['live_voice'],
                    'live_enabled':self.native_options['voice_route']=='live'},
                'audio_attempt_limit':1,'expires_at':(datetime.now(UTC)+timedelta(
                    seconds=self.lease_seconds if self.remaining_lease is None else self.remaining_lease)).isoformat()}
        try:
            await ready(info)
            await stop.wait()
        finally:
            self.child_stopped = True
            self.events.append('host_stopped')
        events=([{'type':'live.media.prepare','provider_session_id':'live_offline_session'}]
                if self.native_options['voice_route']=='live' else [])
        return {'status':'passed','child':{'websocket_trace':[{'path':'/realtime/audio/ws','events':events}]},
            'child_output_drain':{'provider_diagnostics':[
            {'stage':'hangup_http','call_id':'rtc_offline','creation_outcome':'created',
             'closure_outcome':'accepted','http_status':200,'remote_closure_confirmed':False}]},
            'evidence_kind':'OFFLINE_HOST_BOUNDARY_FAKE_NOT_ACCEPTANCE'}

    async def cleanup(self, manifest):
        assert self.child_stopped
        assert Path(manifest)==self.manifest
        self.events.append('fixture_cleanup')
        self.cleanups += 1
        self.token.unlink(missing_ok=True)
        return {'status':'passed','evidence_kind':'OFFLINE_CLEANUP_BOUNDARY_FAKE_NOT_SQL_AUDIT'}

    async def command(self, argv, log_path, timeout=20):
        if argv[1]=='offline-export-boundary':
            self.events.append('canonical_export')
            assert not self.child_stopped
            if self.fail_export:
                raise RuntimeError('offline_export_failure')
            resource = read_json(argv[2])
            save(Path(argv[3]),{'format':'jarvis.native_replay.read_only_evidence.v1','status':'captured',
                'run_id':resource['run_id'],'owner_id':resource['owner_id'],'schema':resource['database_schema'],
                'transaction_read_only':'on','evidence_kind':'OFFLINE_EXPORT_BOUNDARY_FAKE_NOT_SQL'})
        elif argv[2]=='get_app_container':
            return {'returncode':0,'stdout':str(self.container)}
        elif argv[2]=='launch':
            self.launches += 1
            self.events.append('launch')
            self.plan = read_json(argv[argv.index('--jarvis-replay-manifest')+1])
            self.launched.set()
            if self.fail_launch:
                return {'returncode':1,'stdout':''}
            capture = self.container/'Documents/Replay'/str(UUID(self.run))
            capture.mkdir(parents=True)
            save(capture/'capture.json',{'schema_version':2,'run_id':self.run,
                'evidence_kind':'OFFLINE_PROCESS_BOUNDARY_FAKE_NOT_CAPTURE_ACCEPTANCE'})
            (capture/'decoded.pcm').write_bytes(b'')
        elif argv[2]=='terminate':
            self.events.append('terminate')
            if self.fail_terminate:
                return {'returncode':1,'stdout':''}
        else:
            raise AssertionError('unplanned process command')
        return {'returncode':0,'stdout':''}

    async def wait_capture(self, path, seconds):
        self.capture_wait = seconds
        if self.block_capture:
            await asyncio.Event().wait()
        assert path.is_file()


class NativeReplayRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_trigger_single_double_triple_preserves_runtime_fields_and_one_launch(self):
        for cuts in (1,2,3):
            for constrained in (False,True):
                with self.subTest(cuts=cuts,constrained=constrained),tempfile.TemporaryDirectory() as directory:
                    f = Fixture(Path(directory))
                    manifest = f.response_prepared(cuts)
                    for ordinal,item in enumerate(manifest['utterances']):
                        trigger = item['trigger']
                        if trigger['kind']!='response_output_start':
                            continue
                        if ordinal==3:
                            trigger['after_input_ordinal']=0
                        if constrained:
                            trigger.update(response_id=f'resp_expected-{ordinal}:part.1',
                                connection_id='01234567-89AB-CDEF-0123-456789ABCDEF')
                    save(f.options.input_manifest,manifest)
                    f.authorize_prepared()
                    result = await run_once(f.options,host=f,process=f)
                    self.assertEqual(result['status'],'lifecycle_complete',result.get('fatal'))
                    self.assertEqual((f.prepares,f.launches,f.cleanups),(1,1,1))
                    self.assertEqual(f.plan['utterances'],manifest['utterances'])
                    self.assertEqual(f.plan['voice_route'],'realtime')
                    self.assertEqual(f.plan['version'],1)
                    self.assertNotIn('response_trigger_schema_version',f.plan)
                    self.assertNotIn('expectations',f.plan)
                    self.assertFalse(result['voice_pass'])
                    self.assertTrue(result['local_cleanup_complete'])

    async def test_response_trigger_requires_structurally_earlier_integer_input(self):
        cases = [(1,value) for value in (-1,True,0.0,'0',None,1,2)]+[(0,0)]
        for target,value in cases:
            with self.subTest(target=target,value=value),tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                manifest = f.response_prepared()
                trigger = dict(manifest['utterances'][1]['trigger'])
                trigger['after_input_ordinal']=value
                manifest['utterances'][target]['trigger']=trigger
                save(f.options.input_manifest,manifest)
                f.authorize_prepared()
                result = await run_once(f.options,host=f,process=f)
                self.assertEqual(result['status'],'failed')
                self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_response_trigger_rejects_missing_unknown_unsafe_and_nonfinite_fields(self):
        required = ('kind','after_input_ordinal','delay_seconds','timeout_seconds','max_pcm_age_seconds')
        faults = [('remove',key,None) for key in required]
        faults += [('set','after_cursor',0),('set','after_input',0)]
        faults += [('set',key,value) for key in ('delay_seconds','timeout_seconds','max_pcm_age_seconds')
                   for value in (True,'0.1',None,-.1,float('nan'),float('inf'),float('-inf'))]
        faults += [('set','delay_seconds',90),('set','timeout_seconds',0),('set','timeout_seconds',121),
                   ('set','max_pcm_age_seconds',0),('set','max_pcm_age_seconds',121)]
        faults += [('set','response_id',value) for value in
                   (None,'','x'*161,'resp\nunsafe','resp/unsafe','r\u00e9ponse',123)]
        faults += [('set','connection_id',value) for value in (None,'','not-a-uuid','0'*32,True,{})]
        for operation,key,value in faults:
            with self.subTest(operation=operation,key=key,value=value),tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                manifest = f.response_prepared()
                trigger = manifest['utterances'][1]['trigger']
                if operation=='remove':
                    trigger.pop(key)
                else:
                    trigger[key]=value
                save(f.options.input_manifest,manifest)
                f.authorize_prepared()
                result = await run_once(f.options,host=f,process=f)
                self.assertEqual(result['status'],'failed')
                self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_response_trigger_accepts_zero_delay_small_positive_bounds_and_maximum_safe_id(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            manifest = f.response_prepared()
            manifest['utterances'][1]['trigger'].update(delay_seconds=0,timeout_seconds=.001,
                max_pcm_age_seconds=.001,response_id='r'*160)
            save(f.options.input_manifest,manifest)
            f.authorize_prepared()
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'lifecycle_complete',result.get('fatal'))
            self.assertEqual(f.plan['utterances'],manifest['utterances'])

    async def test_response_trigger_deadline_includes_referenced_full_clip_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            manifest = f.response_prepared()
            first = manifest['utterances'][0]['trigger']['at_seconds']
            speech = read_json(f.options.speech_manifest)
            clips = {item['id']:item for item in speech['clips']}
            final_seconds = clips[manifest['utterances'][1]['clip_id']]['sample_count']/compiler.RATE
            manifest['utterances'][1]['trigger']['timeout_seconds']=manifest['duration_seconds']-first-final_seconds-5
            save(f.options.input_manifest,manifest)
            f.authorize_prepared()
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'failed')
            self.assertIn('scenario_cannot_fit_deadline',str(result['fatal']))
            self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    def test_response_trigger_rejects_live_at_the_input_staging_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.response_prepared()
            f.live_baseline()
            authorization = read_json(f.options.risk_authorization)
            with self.assertRaisesRegex(Exception,'response_trigger_requires_realtime'):
                stage_media(f.options,expected_manifest_sha256=authorization['input_manifest_sha256'])
            self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_response_fields_do_not_expand_legacy_triggers_or_runtime_manifest(self):
        for location in ('at_time','output_start','manifest'):
            with self.subTest(location=location),tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                f.prepared()
                manifest = read_json(f.options.input_manifest)
                if location=='manifest':
                    manifest['response_trigger_schema_version']=1
                else:
                    target=next(item['trigger'] for item in manifest['utterances']
                                if item['trigger']['kind']==location)
                    target['after_input_ordinal']=0
                save(f.options.input_manifest,manifest)
                f.authorize_prepared()
                result = await run_once(f.options,host=f,process=f)
                self.assertEqual(result['status'],'failed')
                self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_one_baseline_rebinds_scope_exports_before_stop_and_cleans(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'lifecycle_complete')
            self.assertEqual((f.prepares,f.launches,f.cleanups),(1,1,1))
            self.assertEqual(f.plan['run_id'],f.run)
            self.assertEqual(f.plan['owner_id'],f.resource['owner_id'])
            self.assertNotIn('old-',json.dumps(f.plan))
            self.assertLess(f.events.index('canonical_export'),f.events.index('host_stopped'))
            self.assertLess(f.events.index('host_stopped'),f.events.index('fixture_cleanup'))
            self.assertTrue(result['local_cleanup_complete'])
            self.assertTrue(all(not Path(path).exists() for path in result['token_files']))
            self.assertFalse(Path(result['active_ledger']).exists())
            self.assertFalse(result['voice_pass'])
            self.assertFalse(result['provider_evidence']['remote_closure_confirmed'])
            self.assertEqual(result['provider_evidence']['closure_by_call'],{'rtc_offline':'accepted'})
            self.assertEqual(result['provider_evidence']['historical_attempts'][0]['outcome'],'unknown')
            self.assertEqual((f.capture_wait,f.lease_seconds),(95,305))
            self.assertEqual(f.native_options['native_pool_max_size'],5)
            self.assertEqual((f.worker_starts,f.worker_closes),(0,0))

    async def test_actual_compiler_all_25_variants_preserve_runtime_only_and_fresh_scope(self):
        variants = 0
        for case in compiler.CASES:
            for cuts in ((0,) if case=='baseline' else (1,2,3)):
                with self.subTest(case=case,cuts=cuts), tempfile.TemporaryDirectory() as directory:
                    f = Fixture(Path(directory))
                    compiled = f.prepared(case,cuts)
                    result = await run_once(f.options,host=f,process=f)
                    self.assertEqual(result['status'],'lifecycle_complete',result.get('fatal'))
                    self.assertEqual((f.prepares,f.launches,f.cleanups),(1,1,1))
                    self.assertEqual(f.plan['utterances'],compiled.manifest['utterances'])
                    self.assertEqual([v['at_seconds'] for v in f.plan['frames']],[1,2,3])
                    self.assertEqual([v['sha256'] for v in f.plan['frames']],
                                     [v['sha256'] for v in compiled.manifest['frames']])
                    self.assertEqual((f.capture_wait,f.lease_seconds),(140,350))
                    self.assertEqual(f.plan['run_id'],f.run)
                    for key in ('owner_id','area_id','memory_bank_id'):
                        self.assertEqual(f.plan[key],f.resource[key])
                    self.assertNotIn('old-',json.dumps(f.plan))
                    staged = Path(result['staged_inputs'])
                    self.assertFalse((staged/'expectations.json').exists())
                    self.assertFalse((staged/'token.txt').exists())
                    self.assertTrue((f.options.media_root/'expectations.json').exists())
                    speech = read_json(staged/'speech/speech.json')
                    self.assertEqual([v['id'] for v in speech['clips']],[v['id'] for v in compiled.speech['clips']])
                    for clip in speech['clips']:
                        self.assertEqual(set(clip),{'id','relative_path','sha256','sample_rate_hz','channels',
                            'sample_width_bytes','sample_count','first_active_sample','last_active_sample_exclusive',
                            'energy_threshold_abs_pcm16'})
                    self.assertLess(f.events.index('canonical_export'),f.events.index('host_stopped'))
                    self.assertTrue(result['local_cleanup_complete'])
                    self.assertFalse(result['voice_pass'])
                    self.assertTrue(all(not Path(path).exists() for path in result['token_files']))
                    variants += 1
        self.assertEqual(variants,25)

    async def test_120_second_media_plus_capture_tail_has_170_second_wait_and_380_second_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.prepared(cuts=3,duration=120)
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'lifecycle_complete')
            self.assertEqual((f.capture_wait,f.lease_seconds),(170,380))
            self.assertEqual(result['capture_budget']['decoded_pcm_byte_limit'],170*48000*2)
            self.assertLessEqual(result['capture_budget']['decoded_pcm_byte_limit'],MAX_CAPTURE_PCM_BYTES)

    async def test_prepared_authorization_binds_exact_input_case_and_cuts(self):
        for fault in ('hash','case','cuts','mode'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                f.prepared()
                document = read_json(f.options.risk_authorization)
                field,value = {'hash':('input_manifest_sha256','0'*64),'case':('scenario','pause_continue'),
                               'cuts':('cuts',2),'mode':('input_mode','baseline')}[fault]
                document[field] = value
                save(f.options.risk_authorization,document)
                if fault=='hash':
                    result = await run_once(f.options,host=f,process=f)
                    self.assertEqual(result['status'],'failed')
                else:
                    with self.assertRaises(Exception):
                        await run_once(f.options,host=f,process=f)
                self.assertEqual((f.prepares,f.launches),(0,0))

    async def test_prepared_invalid_runtime_media_fail_before_fixture_allocation(self):
        faults = ('duration','wire_version','route','oracle','trigger','frame_order','frame_hash','path',
                  'duplicate_clip','wav','cut_count','lease')
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                f.prepared()
                manifest = read_json(f.options.input_manifest)
                if fault=='duration': manifest['duration_seconds']=181
                elif fault=='wire_version': manifest['version']=2
                elif fault=='route': manifest['voice_route']='live'
                elif fault=='oracle': manifest['expectations']={'expected_answer':'DO NOT INJECT'}
                elif fault=='trigger': manifest['utterances'][1]['trigger']['kind']='after_state'
                elif fault=='frame_order': manifest['frames'][1]['at_seconds']=1
                elif fault=='frame_hash': manifest['frames'][0]['sha256']='0'*64
                elif fault=='path': manifest['frames'][0]['relative_path']='../synthetic.jpg'
                elif fault in ('duplicate_clip','wav'):
                    speech = read_json(f.options.speech_manifest)
                    if fault=='duplicate_clip': speech['clips'].append(speech['clips'][0])
                    else:
                        clip = speech['clips'][0]
                        (f.options.media_root/clip['relative_path']).write_bytes(b'not-a-wav')
                        clip['sha256']=compiler.digest(b'not-a-wav')
                    save(f.options.speech_manifest,speech)
                elif fault=='cut_count':
                    manifest['utterances'][1]['trigger']={'kind':'at_time','at_seconds':20}
                elif fault=='lease': f.options=replace(f.options,lease_seconds=300)
                save(f.options.input_manifest,manifest)
                f.authorize_prepared()
                result = await run_once(f.options,host=f,process=f)
                self.assertEqual(result['status'],'failed')
                self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_remaining_lease_fences_launch_after_host_is_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.prepared(duration=120)
            f.remaining_lease=200
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'failed')
            self.assertEqual((f.prepares,f.launches,f.cleanups),(1,0,1))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_prepared_failure_and_cancel_keep_existing_cleanup_and_no_retry(self):
        for fault in ('launch','export','cancel'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                f = Fixture(Path(directory))
                f.prepared('pause_continue',3)
                f.fail_launch=fault=='launch'
                f.fail_export=fault=='export'
                f.block_capture=fault=='cancel'
                task = asyncio.create_task(run_once(f.options,host=f,process=f))
                if fault=='cancel':
                    await asyncio.wait_for(f.launched.wait(),2)
                    task.cancel()
                result = await asyncio.wait_for(task,3)
                self.assertEqual(f.launches,1)
                self.assertTrue(f.child_stopped)
                self.assertTrue(all(not Path(path).exists() for path in result['token_files']))
                self.assertEqual(result['status'],{'launch':'failed','export':'cleanup_pending','cancel':'interrupted'}[fault])

    async def test_simulator_lock_blocks_concurrent_prepared_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'one').mkdir()
            (root/'two').mkdir()
            one,two = Fixture(root/'one'),Fixture(root/'two')
            one.prepared()
            two.prepared()
            two.options=replace(two.options,ledger_dir=one.options.ledger_dir)
            one.block_capture=True
            task=asyncio.create_task(run_once(one.options,host=one,process=one))
            try:
                await asyncio.wait_for(one.launched.wait(),2)
                blocked=await run_once(two.options,host=two,process=two)
                self.assertEqual(blocked['status'],'failed')
                self.assertEqual((two.prepares,two.launches),(0,0))
            finally:
                task.cancel()
                result=await asyncio.wait_for(task,3)
            self.assertTrue(result['local_cleanup_complete'])

    def test_capture_budgets_reject_nonfinite_or_oversize_and_preserve_baseline_default(self):
        self.assertEqual(capture_budget(45,None)['lease_seconds'],300)
        for duration in (0,181,float('nan'),float('inf'),True):
            with self.subTest(duration=duration), self.assertRaises(Exception):
                capture_budget(duration,None)

    async def test_integrated_worker_ready_before_launch_and_closed_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared('pause_continue',1)
            f.integrated()
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'lifecycle_complete',result.get('fatal'))
            self.assertEqual((f.worker_starts,f.worker_closes,f.launches),(1,1,1))
            self.assertEqual(f.lease_seconds,470)
            self.assertLess(f.events.index('worker_registry_ready'),f.events.index('launch'))
            self.assertLess(f.events.index('terminate'),f.events.index('worker_closed'))
            self.assertLess(f.events.index('worker_closed'),f.events.index('canonical_export'))
            self.assertLess(f.events.index('canonical_export'),f.events.index('fixture_cleanup'))
            self.assertTrue(f.options.worker_api_key_file.exists())
            self.assertEqual(result['cleanup']['worker']['registry_state'],'withdrawn')
            self.assertFalse(result['voice_pass'])

    async def test_workload_requires_prepared_input_and_matching_explicit_model_authorization(self):
        for fault in ('baseline','missing_opt_in','model','reasoning'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory))
                if fault!='baseline': f.prepared()
                f.integrated()
                document=read_json(f.options.risk_authorization)
                if fault=='missing_opt_in': document.pop('workload')
                elif fault=='model': document['frontier_model']='different-model'
                elif fault=='reasoning': document['frontier_reasoning_effort']='high'
                save(f.options.risk_authorization,document)
                with self.assertRaises(Exception):
                    await run_once(f.options,host=f,process=f)
                self.assertEqual((f.prepares,f.worker_starts,f.launches),(0,0,0))

    async def test_registry_mismatch_fences_launch_but_joins_worker_and_cleans(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared()
            f.integrated()
            f.worker_bad_readiness=True
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'failed')
            self.assertEqual((f.launches,f.worker_closes,f.cleanups),(0,1,1))
            self.assertTrue(result['local_cleanup_complete'])

    async def test_unjoined_worker_blocks_export_and_fixture_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared()
            f.integrated()
            f.worker_bad_close=True
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'cleanup_pending')
            self.assertNotIn('canonical_export',f.events)
            self.assertEqual(f.cleanups,0)
            self.assertTrue(f.child_stopped)
            self.assertTrue(Path(result['active_ledger']).exists())
            self.assertTrue(all(not Path(path).exists() for path in result['token_files']))

    async def test_worker_start_failure_requires_positive_cleanup_receipt(self):
        for joined in (True,False):
            with self.subTest(joined=joined), tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory))
                f.prepared()
                f.integrated()
                f.worker_fail_start=True
                f.worker_start_clean=joined
                result=await run_once(f.options,host=f,process=f)
                self.assertEqual(f.launches,0)
                self.assertEqual(result['status'],'failed' if joined else 'cleanup_pending')
                self.assertEqual('canonical_export' in f.events,joined)
                self.assertEqual(f.cleanups,int(joined))

    async def test_cancel_during_worker_readiness_joins_before_export_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared()
            f.integrated()
            f.block_worker_ready=True
            task=asyncio.create_task(run_once(f.options,host=f,process=f))
            await asyncio.wait_for(f.worker_started.wait(),2)
            task.cancel()
            result=await asyncio.wait_for(task,3)
            self.assertEqual(result['status'],'interrupted')
            self.assertEqual((f.launches,f.worker_closes,f.cleanups),(0,1,1))
            self.assertLess(f.events.index('worker_closed'),f.events.index('canonical_export'))

    async def test_workload_voice_launch_failure_still_joins_before_export_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared()
            f.integrated()
            f.fail_launch=True
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'failed')
            self.assertEqual((f.worker_closes,f.launches,f.cleanups),(1,1,1))
            self.assertLess(f.events.index('worker_closed'),f.events.index('canonical_export'))

    async def test_host_boundary_delegates_to_existing_worker_helper(self):
        from evals.responsive_voice.native_task_worker import NativeTaskWorker, NativeTaskWorkerError
        class PrivateReadBoundary(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared()
            f.integrated()
            f.options.artifact_dir.mkdir(mode=0o700)
            resources={'fixture_manifest':str(f.options.artifact_dir/'fixture.json'),
                       'scope_file':str(f.options.artifact_dir/'scope.json')}
            recorded={}
            async def start(*,options,resources,dsn):
                recorded.update(options=options,resources=resources,dsn=dsn)
                # Execute the actual helper's path guards, stopping at its first
                # private-file read. No owner validation, SQL or process startup
                # is replaced by an acceptance claim.
                with patch('evals.responsive_voice.native_task_worker._read_private',
                           side_effect=PrivateReadBoundary) as private_read:
                    with self.assertRaises(PrivateReadBoundary):
                        NativeTaskWorker(options,resources,dsn)._validate()
                    private_read.assert_called_once_with(resources['fixture_manifest'])
                old_parent=options.artifact_dir/'task-worker'
                old_parent.mkdir(mode=0o700)
                with patch('evals.responsive_voice.native_task_worker._read_private') as private_read:
                    with self.assertRaisesRegex(NativeTaskWorkerError,'fixture_manifest_directory_mismatch'):
                        NativeTaskWorker(options.model_copy(update={'artifact_dir':old_parent}),resources,dsn)._validate()
                    private_read.assert_not_called()
                return 'offline-helper-boundary'
            with patch('evals.responsive_voice.native_task_worker.start',new=start):
                result=await HostBoundary('offline-dsn-no-connection').start_worker(f.options,resources)
            self.assertEqual(result,'offline-helper-boundary')
            self.assertEqual(recorded['options'].hermes_python,f.options.worker_python)
            self.assertEqual(recorded['options'].model,f.options.worker_model)
            self.assertEqual(recorded['options'].artifact_dir,f.options.artifact_dir)
            self.assertEqual(recorded['options'].readiness_timeout_seconds,45)

    def test_host_task_limits_match_worker_environment_without_enabling_baseline(self):
        from evals.responsive_voice.native_task_worker import NativeTaskWorkerOptions, expected_host_environment
        class Configured:
            JARVIS_REALTIME_MAX_SESSION_SECONDS=600
            def model_dump(self,**kwargs):
                return {'OPENAI_API_KEY':'offline-key','GEMINI_API_KEY':'offline-key',
                    'OPENAI_LIVE_ENABLED':False,'SECOND_BRAIN_ACTIVE_SECONDS':999999,
                    'SECOND_BRAIN_OWNER_CONCURRENCY':999999}
        worker_options=NativeTaskWorkerOptions(model='offline-approved-model',reasoning_effort='low',
            api_key_file=Path('/offline/provider-key'),hermes_python=Path(sys.executable),
            artifact_dir=Path('/offline/fixture'))
        expected=expected_host_environment(worker_options)
        limit_keys={'SECOND_BRAIN_'+key.upper() for key in worker_options.task_limits.model_dump()}
        run=uuid4().hex
        scope=ReplayHostScope(run_id=run,owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0',area_id=uuid4(),memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        cases=((NO_MEDIA,NO_WORKLOAD,'realtime',1),(NATIVE_MEDIA,NO_WORKLOAD,'realtime',5),
               (NATIVE_MEDIA,NO_WORKLOAD,'live',5),(NATIVE_MEDIA,REASONING_REMINDER,'realtime',5),
               (NATIVE_MEDIA,REASONING_REMINDER,'realtime',1))
        with patch.dict(os.environ,{key:'999999' for key in limit_keys}):
            for profile,workload,route,pool in cases:
                with self.subTest(profile=profile,workload=workload,route=route,pool=pool):
                    integrated=workload==REASONING_REMINDER
                    env=child_environment(dsn='offline-dsn',scope=scope,token='offline-token',
                        origin='http://127.0.0.1:1',hindsight_url='http://127.0.0.1:9',
                        api_key_file=Path('/offline/memory-key'),profile=profile,configured=Configured(),
                        native_pool_max_size=pool,workload=workload,voice_route=route,
                        frontier_model=worker_options.model if integrated else None,
                        frontier_reasoning_effort=worker_options.reasoning_effort if integrated else None,
                        live_model='offline-live-model' if route=='live' else None,
                        live_voice='marin' if route=='live' else None)
                    if integrated:
                        self.assertEqual({key:env[key] for key in expected},expected)
                    else:
                        self.assertTrue(limit_keys.isdisjoint(env))
                        for key in ('SECOND_BRAIN_FRONTIER_ENABLED','SECOND_BRAIN_ATTENTION_VOICE_ENABLED',
                                    'REMINDER_SCHEDULER_ENABLED'):
                            self.assertEqual(env[key],'false')
                    self.assertEqual(env['SUPABASE_DB_POOL_MAX_SIZE'],str(pool))

    def test_native_pool_and_opt_in_workload_effective_environment(self):
        class Configured:
            JARVIS_REALTIME_MAX_SESSION_SECONDS=600
            def model_dump(self,**kwargs):
                return {'OPENAI_API_KEY':'offline-key','GEMINI_API_KEY':'offline-key',
                        'SUPABASE_DB_POOL_MAX_SIZE':99,'SECOND_BRAIN_FRONTIER_ENABLED':True}
        run=uuid4().hex
        scope=ReplayHostScope(run_id=run,owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0',area_id=uuid4(),memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        common=dict(dsn='offline-dsn-no-connection',scope=scope,token='offline-scoped-token',
            origin='http://127.0.0.1:1',hindsight_url='http://127.0.0.1:9',api_key_file=Path('/offline/key'))
        for profile,maximum,workload in ((NO_MEDIA,None,NO_WORKLOAD),(NATIVE_MEDIA,None,NO_WORKLOAD),
                (NATIVE_MEDIA,1,NO_WORKLOAD),(NATIVE_MEDIA,5,REASONING_REMINDER)):
            with self.subTest(profile=profile,pool=maximum,workload=workload):
                integrated=workload==REASONING_REMINDER
                env=child_environment(**common,profile=profile,configured=Configured(),
                    native_pool_max_size=maximum,workload=workload,
                    frontier_model='offline-approved-model' if integrated else None,
                    frontier_reasoning_effort='low' if integrated else None)
                self.assertEqual(env['SUPABASE_DB_POOL_MIN_SIZE'],'1')
                self.assertEqual(env['SUPABASE_DB_POOL_MAX_SIZE'],str(maximum or (5 if profile==NATIVE_MEDIA else 1)))
                for key in ('SECOND_BRAIN_FRONTIER_ENABLED','SECOND_BRAIN_ATTENTION_VOICE_ENABLED','REMINDER_SCHEDULER_ENABLED'):
                    self.assertEqual(env[key],'true' if integrated else 'false')
                for key in ('SECOND_BRAIN_SANDBOX_ENABLED','SECOND_BRAIN_ARTIFACTS_ENABLED',
                            'SECOND_BRAIN_PUSH_ENABLED','SECOND_BRAIN_MAIL_ENABLED'):
                    self.assertEqual(env[key],'false')
                self.assertEqual(env['SECOND_BRAIN_KNOWLEDGE_AREA_ID'],str(scope.area_id))
                if integrated:
                    self.assertEqual(env['SECOND_BRAIN_FRONTIER_MODEL'],'offline-approved-model')
                    self.assertEqual(env['SECOND_BRAIN_FRONTIER_REASONING_EFFORT'],'low')
                if profile==NO_MEDIA:
                    self.assertEqual(env['OPENAI_API_KEY'],'')
        with self.assertRaises(Exception): native_host_options(NO_MEDIA,native_pool_max_size=5)
        with self.assertRaises(Exception): native_host_options(NO_MEDIA,workload=REASONING_REMINDER,
            frontier_model='offline-model',frontier_reasoning_effort='low')

    async def test_explicit_native_pool_one_is_recorded_without_enabling_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.options=replace(f.options,native_pool_max_size=1)
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'lifecycle_complete')
            self.assertEqual(result['native_options']['native_pool_max_size'],1)
            self.assertEqual(result['native_options']['workload'],NO_WORKLOAD)
            self.assertEqual(f.worker_starts,0)

    async def test_live_baseline_rebinds_once_without_realtime_closure_claims(self):
        for prepared in (False,True):
            with self.subTest(prepared=prepared),tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory))
                if prepared: f.prepared('baseline',0,duration=45)
                f.live_baseline()
                result=await run_once(f.options,host=f,process=f)
                self.assertEqual(result['status'],'lifecycle_complete',result.get('fatal'))
                self.assertEqual((f.launches,f.worker_starts,f.cleanups),(1,0,1))
                self.assertEqual(f.plan['voice_route'],'live')
                self.assertEqual(f.plan['run_id'],f.run)
                evidence=result['provider_evidence']
                self.assertEqual(evidence['provider_session_ids'],['live_offline_session'])
                self.assertEqual(evidence['finalization_outcome'],'unavailable')
                self.assertFalse(evidence['remote_closure_confirmed'])
                self.assertFalse(evidence['realtime_hangup_applicable'])
                self.assertNotIn('created_call_ids',evidence)
                self.assertEqual(evidence['ignored_realtime_diagnostic_count'],1)
                self.assertTrue(result['local_cleanup_complete'])

    async def test_live_route_model_and_voice_require_fresh_explicit_authorization(self):
        for key in ('voice_route','live_model','live_voice'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory))
                f.live_baseline()
                document=read_json(f.options.risk_authorization)
                document.pop(key)
                save(f.options.risk_authorization,document)
                with self.assertRaises(Exception):
                    await run_once(f.options,host=f,process=f)
                self.assertEqual((f.prepares,f.launches),(0,0))

    async def test_live_prepared_manifest_route_must_match_authorized_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            f=Fixture(Path(directory))
            f.prepared('baseline',0)
            f.live_baseline()
            manifest=read_json(f.options.input_manifest)
            manifest['voice_route']='realtime'
            save(f.options.input_manifest,manifest)
            f.authorize_prepared()
            result=await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'failed')
            self.assertEqual((f.prepares,f.launches),(0,0))

    async def test_live_rejects_interruption_or_integrated_workload_scope(self):
        for integrated in (False,True):
            with self.subTest(integrated=integrated),tempfile.TemporaryDirectory() as directory:
                f=Fixture(Path(directory))
                f.prepared('baseline',0) if integrated else f.prepared('correction_chain',1)
                f.live_baseline()
                if integrated: f.integrated()
                with self.assertRaises(Exception):
                    await run_once(f.options,host=f,process=f)
                self.assertEqual((f.prepares,f.worker_starts,f.launches),(0,0,0))

    def test_live_environment_passes_existing_route_gate_without_provider_creation(self):
        from app.settings import JarvisSettings
        from app.realtime.live_selection import select_live_voice_route,require_live_client_offer
        from app.realtime.provider_selection import AudioProvider
        class Configured:
            JARVIS_REALTIME_MAX_SESSION_SECONDS=600
            def model_dump(self,**kwargs):
                return {'OPENAI_API_KEY':'offline-key','GEMINI_API_KEY':'offline-key','OPENAI_LIVE_ENABLED':False,
                    'OPENAI_LIVE_MODEL':'old-configured-model','OPENAI_LIVE_VOICE':'old-voice',
                    'OPENAI_REALTIME_MODEL':'unchanged-realtime-model','OPENAI_REALTIME_VOICE':'cedar'}
        run=uuid4().hex
        scope=ReplayHostScope(run_id=run,owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0',area_id=uuid4(),memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        common=dict(dsn='offline-dsn',scope=scope,token='offline-token',origin='http://127.0.0.1:1',
            hindsight_url='http://127.0.0.1:9',api_key_file=Path('/offline/key'),profile=NATIVE_MEDIA,configured=Configured())
        realtime=child_environment(**common)
        self.assertEqual(realtime['JARVIS_FRONTEND_REPLAY_VOICE_ROUTE'],'realtime')
        self.assertEqual(realtime['OPENAI_LIVE_ENABLED'],'false')
        live=child_environment(**common,voice_route='live',live_model='offline-live-model',live_voice='marin')
        settings=JarvisSettings(_env_file=None,**live)
        selection=select_live_voice_route(settings=settings,requested_values=['live'],
            provider=AudioProvider.OPENAI,transport='webrtc')
        self.assertEqual((selection.model,selection.voice),('offline-live-model','marin'))
        self.assertEqual(settings.OPENAI_REALTIME_MODEL,'unchanged-realtime-model')
        self.assertEqual(settings.OPENAI_REALTIME_VOICE,'cedar')
        self.assertIsNone(select_live_voice_route(settings=settings,requested_values=['realtime'],
            provider=AudioProvider.OPENAI,transport='webrtc'))
        with self.assertRaises(Exception):
            require_live_client_offer(capture_mode='auto',client_capabilities={})
        for name in ('SECOND_BRAIN_FRONTIER_ENABLED','SECOND_BRAIN_ATTENTION_VOICE_ENABLED',
                     'REMINDER_SCHEDULER_ENABLED','SECOND_BRAIN_SANDBOX_ENABLED','SECOND_BRAIN_ARTIFACTS_ENABLED',
                     'SECOND_BRAIN_PUSH_ENABLED','SECOND_BRAIN_MAIL_ENABLED'):
            self.assertEqual(live[name],'false')

    async def test_live_host_route_fence_burns_one_admission_without_fallback(self):
        calls,sent=[],[]
        async def host(scope,receive,send): calls.append(scope)
        async def receive(): return {'type':'websocket.connect'}
        async def send(message): sent.append(message)
        trace=AdmissionTrace(host,audio_attempt_limit=1,expected_voice_route='live')
        scope={'type':'websocket','path':'/realtime/audio/ws','query_string':b'voice_route=realtime'}
        await trace(scope,receive,send)
        self.assertEqual(sent[-1]['reason'],'evaluation_voice_route_mismatch')
        await trace({**scope,'query_string':b'voice_route=live'},receive,send)
        self.assertEqual(sent[-1]['reason'],'evaluation_audio_attempt_limit')
        self.assertEqual(calls,[])

    def test_live_route_trace_retains_actual_capabilities_but_no_sdp_or_caption(self):
        from app.voice_routes.live_adapter import LiveVoiceRoute
        class Bindings:
            audio_session_id='offline-audio-session'
        capabilities=LiveVoiceRoute(Bindings()).capabilities.model_dump(mode='json')
        record=native_route_observation({'type':'voice_route.selected','audio_session_id':'offline-audio-session',
            'capabilities':capabilities,'sdp':'private-sdp','text':'private-caption','authorization':'private-key'})
        self.assertEqual(record['capabilities'],capabilities)
        self.assertEqual(record['capabilities']['image_input'],'unsupported')
        self.assertEqual(record['capabilities']['client_playback_evidence'],'unsupported')
        self.assertNotIn('private-',json.dumps(record))
        prepare=native_route_observation({'type':'live.media.prepare','provider_session_id':'live_offline_session',
            'sdp':'private-sdp','model':'sk-secret'})
        self.assertIsNone(prepare['model'])
        self.assertEqual(prepare['provider_session_id'],'live_offline_session')

    def test_live_unknown_finalization_never_uses_realtime_hangup_or_local_exit(self):
        evidence=provider_evidence([{'stage':'hangup_http','call_id':'rtc_old','closure_outcome':'accepted',
            'http_status':200}],True,voice_route='live',native_events=[{'type':'voice_route.event',
                'state':'closed','reason':'provider_finalized'},{'type':'local.exit','status':'passed'}])
        self.assertEqual(evidence['creation_outcome'],'unknown')
        self.assertEqual(evidence['finalization_outcome'],'unavailable')
        self.assertFalse(evidence['remote_closure_confirmed'])
        self.assertNotIn('rtc_old',json.dumps(evidence))

    def test_live_configuration_is_never_enabled_for_no_media_or_without_explicit_identity(self):
        with self.assertRaises(Exception):
            native_host_options(NO_MEDIA,voice_route='live',live_model='offline-live-model',live_voice='marin')
        with self.assertRaises(Exception):
            native_host_options(NATIVE_MEDIA,voice_route='live')
        with self.assertRaises(Exception):
            native_host_options(NATIVE_MEDIA,live_model='offline-live-model',live_voice='marin')

    async def test_risk_gate_blocks_before_any_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            document = read_json(f.options.risk_authorization)
            document['historical_attempts'] = []
            save(f.options.risk_authorization,document)
            with self.assertRaisesRegex(Exception,'historical_unknown_risk_gate_unresolved'):
                await run_once(f.options,host=f,process=f)
            self.assertEqual((f.prepares,f.launches,f.cleanups),(0,0,0))

    async def test_missing_exporter_blocks_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            save(f.options.canonical_export_command,[])
            with self.assertRaisesRegex(Exception,'canonical_exporter_command_required'):
                await run_once(f.options,host=f,process=f)
            self.assertEqual(f.prepares,0)

    async def test_authorization_is_spent_even_when_launch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.fail_launch = True
            first = await run_once(f.options,host=f,process=f)
            self.assertEqual(first['status'],'failed')
            self.assertEqual(f.launches,1)
            second = await run_once(replace(f.options,artifact_dir=f.root/'second'),host=f,process=f)
            self.assertEqual(second['status'],'failed')
            self.assertEqual((f.prepares,f.launches,f.cleanups),(1,1,1))

    async def test_export_failure_stops_local_owners_preserves_manifested_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.fail_export = True
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'cleanup_pending')
            self.assertTrue(f.child_stopped)
            self.assertEqual(f.cleanups,0)
            self.assertTrue(Path(result['active_ledger']).exists())
            self.assertTrue(all(not Path(path).exists() for path in result['token_files']))
            self.assertEqual(result['cleanup_command'],f.resource['cleanup_command'])

    async def test_cancellation_joins_host_exports_and_removes_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.block_capture = True
            task = asyncio.create_task(run_once(f.options,host=f,process=f))
            await asyncio.wait_for(f.launched.wait(),2)
            task.cancel()
            result = await asyncio.wait_for(task,3)
            self.assertEqual(result['status'],'interrupted')
            self.assertTrue(result['local_cleanup_complete'])
            self.assertEqual((f.launches,f.cleanups),(1,1))

    async def test_app_termination_failure_is_not_cleanup_success(self):
        with tempfile.TemporaryDirectory() as directory:
            f = Fixture(Path(directory))
            f.fail_terminate = True
            result = await run_once(f.options,host=f,process=f)
            self.assertEqual(result['status'],'cleanup_pending')
            self.assertFalse(result['local_cleanup_complete'])

    async def test_one_audio_attempt_gate_never_delegates_second_connection(self):
        calls, sent = [], []
        async def host(scope,receive,send):
            calls.append(scope)
            await send({'type':'websocket.close','code':1008,'reason':'normal_auth_rejected'})
        async def send(message):
            sent.append(message)
        async def receive():
            return {'type':'websocket.connect'}
        app = AdmissionTrace(host,audio_attempt_limit=1)
        scope = {'type':'websocket','path':'/realtime/audio/ws','query_string':b''}
        await app(scope,receive,send)
        await app(scope,receive,send)
        self.assertEqual(len(calls),1)
        self.assertEqual(sent[-1]['reason'],'evaluation_audio_attempt_limit')
        self.assertEqual(app.audio_attempts,1)

    async def test_native_only_logger_survives_cap_and_exit_with_accepted_metadata(self):
        root, logger = logging.getLogger(), logging.getLogger('app.realtime.webrtc')
        parents = (logging.getLogger('app'), logging.getLogger('app.realtime'),
                   logging.getLogger('app.realtime.live_session'), logging.getLogger('app.attention'))
        diagnostic_loggers = (logger, logging.getLogger('app.realtime.telemetry'),
                              logging.getLogger('app.realtime.live_session.finalization'),
                              logging.getLogger('app.realtime.request_source_diagnostics'),
                              logging.getLogger('app.attention.native_voice'))
        with ExitStack() as cleanup:
            for target in (root, *parents, *diagnostic_loggers):
                cleanup.callback(target.setLevel, target.level)
            # Use the public setters so inherited levels and cached decisions
            # from earlier tests cannot determine this fixture's outcome.
            logger.setLevel(logging.INFO)
            self.assertTrue(logger.isEnabledFor(logging.INFO))
            root.setLevel(logging.WARNING)
            for target in (*parents, *diagnostic_loggers):
                target.setLevel(logging.NOTSET)
            configure_native_diagnostic_logging(NO_MEDIA)
            for target in diagnostic_loggers:
                self.assertFalse(target.isEnabledFor(logging.INFO))
            configure_native_diagnostic_logging(NATIVE_MEDIA)
            self.assertEqual(root.level,logging.WARNING)
            for target in diagnostic_loggers:
                self.assertTrue(target.isEnabledFor(logging.INFO))
                self.assertFalse(target.isEnabledFor(logging.DEBUG))
        program = """
import json,logging,sys
from app.logging_config import configure_logging
from evals.responsive_voice.frontend_host_qualification import configure_native_diagnostic_logging,NATIVE_MEDIA
configure_logging('WARNING')
configure_native_diagnostic_logging(NATIVE_MEDIA)
sys.stdout.write('x'*(2*1024*1024)+'\\n'); sys.stdout.flush()
logger=logging.getLogger('app.realtime.webrtc')
for outcome,status in [('requested',None),('accepted',200)]:
    logger.info('WebRTC remote cleanup diagnostic=%s',json.dumps(dict(stage='hangup_http',call_id='rtc_offline',creation_outcome='created',closure_outcome=outcome,http_status=status,remote_closure_confirmed=False)))
"""
        with tempfile.TemporaryDirectory() as directory:
            result = await ProcessBoundary().command([sys.executable,'-c',program],Path(directory)/'child.log')
            self.assertEqual(result['returncode'],0)
            drain = result['output_drain']
            self.assertTrue(drain['truncated'])
            self.assertTrue(drain['complete'])
            self.assertEqual(len(drain['provider_diagnostics']),2)
            evidence = provider_evidence(drain['provider_diagnostics'],True)
            self.assertEqual(evidence['created_call_ids'],['rtc_offline'])
            self.assertEqual(evidence['closure_by_call'],{'rtc_offline':'accepted'})
            self.assertFalse(evidence['remote_closure_confirmed'])

    async def test_unknown_hangup_and_unsafe_identity_never_become_confirmed(self):
        line = 'app.realtime.webrtc WebRTC remote cleanup diagnostic='+json.dumps({
            'stage':'hangup_http','creation_outcome':'unknown','closure_outcome':'unknown',
            'call_id':'sk-do-not-retain','http_status':None,'authorization':'secret','remote_closure_confirmed':True})
        record = sanitized_webrtc_diagnostic(line)
        self.assertNotIn('secret',json.dumps(record))
        self.assertIsNone(record['call_id'])
        evidence = provider_evidence([record],True)
        self.assertEqual(evidence['creation_outcome'],'unknown')
        self.assertFalse(evidence['remote_closure_confirmed'])


if __name__=='__main__':
    unittest.main()
