"""No-provider checks for the isolated experiment and privacy/budget boundaries."""
import asyncio
from datetime import datetime, timezone
from dataclasses import replace
import json
import os
from pathlib import Path
import ssl
import stat
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID
from synthetic_webrtc import (Browser, INSTRUCTIONS, QUESTION, child_environment, encoded_bytes,
    fixture_guard, make_selection, projection_report, wire_text, HarnessError, offline_failure, safe_code, CDP,
    CreationAudit, AuditedHTTPClient, Provider, verified_tls_context, qualification_exit_code, main,
    TrialPermit, PRIOR_UNKNOWN_TRIAL, run_route, FixtureContextUpdate, make_context_correction,
    validate_context_correction, correction_fact_check, assess_correction_turn, make_speech,
    CANONICAL_RESULT_TEXT, CANONICAL_RESULT_FACTS, CORRECTION_UPDATE_ID, CORRECTION_QUESTION,
    context_correction_provider_behavior)
import httpx
from app.voice_routes.live_protocol import (LiveAppendKind, LiveControlProtocol,
    LiveSessionConfiguration, webrtc_session_request)


class SyntheticFixtureTests(unittest.TestCase):
    def test_full_useful_observation_fits_and_keeps_provenance(self):
        selection=make_selection();report=projection_report(selection)
        self.assertIn('The display reads 42.',selection.text)
        self.assertEqual(selection.selected_entry_ids,('display',))
        self.assertEqual(selection.omitted_entry_ids,())
        self.assertEqual(len(selection.dependencies),1)
        self.assertEqual(len(selection.parent_payload_root_ids),1)
        self.assertGreater(selection.expires_at,datetime.now(timezone.utc))
        self.assertLessEqual(report['encoded_document_bytes'],500)
        self.assertEqual(json.loads(wire_text(selection))['text'],selection.text)
    def test_counter_is_encoded_bytes_not_character_count(self):
        self.assertEqual(encoded_bytes('\u6e2c'),3)
        self.assertEqual(encoded_bytes(json.dumps('\u6e2c',ensure_ascii=True)),8)
    def test_credentials_not_in_child_process_environment(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'fake-key','OTHER_SECRET':'private'}):
            child=child_environment()
        self.assertNotIn('OPENAI_API_KEY',child);self.assertNotIn('OTHER_SECRET',child)
    def test_actual_live_shape_has_no_pcm_or_start_or_responses_backend(self):
        value=webrtc_session_request(LiveSessionConfiguration(INSTRUCTIONS),
            'v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n',count_tokens=encoded_bytes)
        self.assertFalse(value['session']['store'])
        self.assertEqual(value['session']['delegation'],{'type':'client'})
        self.assertEqual(value['transport']['type'],'webrtc')
        self.assertNotIn('format',value['session']['audio'])
        self.assertNotIn('api_key',json.dumps(value))
    def test_actual_live_protocol_accepts_compact_context_budget(self):
        protocol=LiveControlProtocol(route_id=UUID(int=5),route_generation=1,
            provider_session_id='fixture-live',count_tokens=encoded_bytes)
        protocol.mark_attached(authorized_session_id='fixture-live')
        command=protocol.prepare_append(LiveAppendKind.THINKING,wire_text(make_selection()))
        event=protocol.take_for_write(command.command_id)
        self.assertEqual(event['type'],'session.thinking.append')
        self.assertIsNone(event['delegation_id'])
        self.assertLessEqual(encoded_bytes(event['content']),500)
    def test_question_does_not_contain_answer(self):
        self.assertNotIn('42',QUESTION);self.assertNotIn('forty',QUESTION)


class SourceGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_guard_accepts_only_current_complete_fixture(self):
        selection=make_selection()
        async with fixture_guard(selection): pass
        from dataclasses import replace
        with self.assertRaises(RuntimeError):
            async with fixture_guard(replace(selection,selected_entry_ids=())): pass


class MeasurementGateTests(unittest.TestCase):
    def check(self):
        return dict(physical_gain=0,overflow=False,latency_ms=400,threshold=.002,
            meters={kind:dict(processed_blocks=100,input_blocks=90,non_silent_blocks=40,peak=.1)
                    for kind in ('input','return_source','output')})
    def test_non_silent_complete_path_passes(self):
        self.assertIsNone(offline_failure(self.check()))
    def test_processed_silent_output_is_not_mislabeled_missing_decoded_samples(self):
        check=self.check();check['meters']['output'].update(non_silent_blocks=0,peak=0)
        self.assertEqual(offline_failure(check),'offline_output_audio_not_observed')
    def test_unprocessed_output_and_empty_input_have_distinct_codes(self):
        for field,code in (('processed_blocks','offline_output_meter_not_running'),
                           ('input_blocks','offline_output_meter_no_input')):
            with self.subTest(field=field):
                check=self.check();check['meters']['output'][field]=0
                self.assertEqual(offline_failure(check),code)
    def test_return_source_failure_is_separate_from_receiver(self):
        check=self.check();check['meters']['return_source']['non_silent_blocks']=0
        self.assertEqual(offline_failure(check),'offline_return_source_audio_not_observed')
    def test_physical_gain_never_passes(self):
        check=self.check();check['physical_gain']=.01
        self.assertEqual(offline_failure(check),'offline_output_safety_or_capacity_failed')
    def test_specific_harness_code_does_not_expose_arbitrary_error_text(self):
        self.assertEqual(safe_code(HarnessError('offline_output_meter_no_input')),'offline_output_meter_no_input')
        self.assertEqual(safe_code(RuntimeError('secret provider body')),'RuntimeError')
        with self.assertRaises(ValueError): HarnessError('secret provider body')


class BrowserFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_named_local_browser_failures_are_exposed(self):
        from unittest.mock import AsyncMock
        cdp=CDP(None)
        for description,expected in (('Error: media_connection_timeout\n at local','media_connection_timeout'),
                                     ('Error: secret arbitrary text','browser_evaluation_failed')):
            cdp.command=AsyncMock(return_value={'exceptionDetails':{'exception':{'description':description}}})
            with self.assertRaises(HarnessError) as caught: await cdp.evaluate('offline fixture')
            self.assertEqual(safe_code(caught.exception),expected)
    async def test_browser_cleanup_can_be_repeated(self):
        from pathlib import Path
        browser=Browser(Path('/unused'),Path('/unused'))
        first=await browser.close();second=await browser.close()
        self.assertIs(first,second)
        self.assertTrue(first['loopback_server_stopped'])
    async def test_browser_shutdown_bounds_stalled_media_and_socket(self):
        from unittest.mock import AsyncMock
        async def stall(*args): await asyncio.Event().wait()
        browser=Browser(Path('/unused'),Path('/unused'),close_timeout=.02)
        browser.cdp=SimpleNamespace(evaluate=AsyncMock(side_effect=stall))
        browser.socket=SimpleNamespace(close=AsyncMock(side_effect=stall))
        started=time.monotonic();result=await browser.close()
        self.assertLess(time.monotonic()-started,.5)
        self.assertEqual(result['media_error'],'TimeoutError')
        self.assertEqual(result['socket_close_error'],'TimeoutError')
        self.assertTrue(result['loopback_server_stopped'])


class CreationCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_receipt_precedes_sdp_schema_validation_and_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'identity.json';audit=CreationAudit('live',path)
            async def serve(request):
                return httpx.Response(201,json={'session':{'id':'fixture_live_123'},
                    'transport':{'sdp':'DO_NOT_PERSIST'},'extra':'BODY_SECRET'},
                    headers={'x-request-id':'req_fixture'})
            async with AuditedHTTPClient(creation_audit=audit,transport=httpx.MockTransport(serve)) as client:
                response=await client.post('https://api.openai.com/v1/live/sessions',
                                           headers={'Authorization':'Bearer FAKE_KEY'})
                retained=json.loads(path.read_text())
                self.assertEqual(retained['provider_session_id'],'fixture_live_123')
                self.assertEqual(retained['request_id'],'req_fixture')
                self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
                self.assertNotIn('BODY_SECRET',path.read_text())
                self.assertNotIn('FAKE_KEY',path.read_text())
                self.assertNotIn('DO_NOT_PERSIST',path.read_text())
                from app.voice_routes.live_protocol import parse_webrtc_session_answer
                with self.assertRaises(ValueError):parse_webrtc_session_answer(response.json())
    async def test_malformed_truncated_duplicate_and_missing_ids_remain_unknown(self):
        cases=(b'{"session":{"id":"fixture_live_123"},',
               b'{"session":{"id":"first","id":"second"}}',b'{"session":{}}')
        for body in cases:
            with self.subTest(body=body):
                audit=CreationAudit('live')
                async with AuditedHTTPClient(creation_audit=audit,
                    transport=httpx.MockTransport(lambda r:httpx.Response(201,content=body))) as client:
                    with self.assertRaises(HarnessError):await client.post('https://api.openai.com/v1/live/sessions')
                self.assertIsNone(audit.value['provider_session_id'])
                self.assertEqual(audit.value['http_status'],201)
                self.assertEqual(audit.value['create_attempts'],1)
    async def test_oversized_body_is_not_regex_mined_for_identity(self):
        audit=CreationAudit('live')
        async with AuditedHTTPClient(creation_audit=audit,
            transport=httpx.MockTransport(lambda r:httpx.Response(201,content=b'x'*524289))) as client:
            with self.assertRaises(HarnessError) as error:
                await client.post('https://api.openai.com/v1/live/sessions')
        self.assertEqual(safe_code(error.exception),'creation_response_too_large')
        self.assertIsNone(audit.value['provider_session_id'])
    async def test_ambiguous_creation_cannot_be_retried(self):
        audit=CreationAudit('live');calls=[]
        async def serve(request):
            calls.append(request);raise httpx.ReadTimeout('unconfirmed creation')
        async with AuditedHTTPClient(creation_audit=audit,transport=httpx.MockTransport(serve)) as client:
            with self.assertRaises(httpx.ReadTimeout):await client.post('https://api.openai.com/v1/live/sessions')
            with self.assertRaisesRegex(HarnessError,'duplicate_creation_forbidden'):
                await client.post('https://api.openai.com/v1/live/sessions')
        self.assertEqual(len(calls),1)
    async def test_realtime_identity_survives_failed_body_read(self):
        class BrokenBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'v=0'
                raise httpx.ReadError('truncated SDP')
        audit=CreationAudit('realtime')
        async with AuditedHTTPClient(creation_audit=audit,transport=httpx.MockTransport(lambda r:
            httpx.Response(201,headers={'location':'/v1/realtime/calls/rtc_fixture'},stream=BrokenBody()))) as client:
            with self.assertRaises(httpx.ReadError):await client.post('https://api.openai.com/v1/realtime/calls')
        self.assertEqual(audit.value['realtime_call_id'],'rtc_fixture')


class FakeSocket:
    def __init__(self,identity='fixture_live_123',*,finalize=True,close_hangs=False):
        self.identity=identity;self.finalize=finalize;self.close_hangs=close_hangs
        self.sent=[];self.queue=asyncio.Queue();self.close_calls=0
    async def send(self,payload):
        value=json.loads(payload);self.sent.append(value)
        if value['type']=='session.close' and self.finalize:
            self.queue.put_nowait(json.dumps({'type':'session.closed','event_id':'closed_fixture',
                'session':{'id':self.identity},'reason':'close_requested','usage':{'seconds':0.0}}))
    async def recv(self):return await self.queue.get()
    async def close(self):
        self.close_calls+=1
        if self.close_hangs:await asyncio.Event().wait()


class PartialStartCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'live-creation.json'
        self.requests=[];self.connections=[];self.sockets=[];self.report={}
        self.settings=SimpleNamespace(OPENAI_API_KEY='FAKE_SERVER_ONLY_KEY',OPENAI_LIVE_MODEL='gpt-live-1',
            OPENAI_LIVE_VOICE='marin',OPENAI_REALTIME_MODEL='gpt-realtime-2.1',OPENAI_REALTIME_VOICE='cedar',
            OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=1,OPENAI_REALTIME_TRANSCRIPTION_MODEL=None,
            JARVIS_AUDIO_FORMAT='pcm16',JARVIS_AUDIO_SAMPLE_RATE=24000)
    def provider(self,*,invalid_sdp=False,attach_fails=False,finalize=True,
                 wrong_identity=False,close_hangs=False,route='live',malformed=False):
        async def serve(request):
            self.requests.append(request.url.path)
            if request.url.path.endswith('/hangup'):return httpx.Response(200)
            if route=='realtime':
                return httpx.Response(201,text='v=0\r\nm=audio 9 RTP/SAVPF 111\r\n',
                                      headers={'location':'/v1/realtime/calls/rtc_fixture'})
            if malformed:return httpx.Response(201,content=b'{"session":')
            return httpx.Response(201,json={'session':{'id':'fixture_live_123'},
                'transport':{'type':'invalid' if invalid_sdp else 'webrtc','sdp':'v=0\r\nm=audio 9 RTP/SAVPF 111\r\n'}})
        async def connector(uri,**kwargs):
            self.connections.append(uri)
            self.assertEqual(kwargs['ssl'].verify_mode,ssl.CERT_REQUIRED)
            self.assertTrue(kwargs['ssl'].check_hostname)
            self.assertGreater(kwargs['ssl'].cert_store_stats()['x509_ca'],0)
            self.assertTrue(self.path.exists(),'Identity must be durable before any attach.')
            key='realtime_call_id' if route=='realtime' else 'provider_session_id'
            expected='rtc_fixture' if route=='realtime' else 'fixture_live_123'
            self.assertEqual(json.loads(self.path.read_text())[key],expected)
            if attach_fails and len(self.connections)==1:raise ssl.SSLCertVerificationError('fixture TLS failure')
            socket=FakeSocket('other_session' if wrong_identity else 'fixture_live_123',
                              finalize=finalize,close_hangs=close_hangs)
            self.sockets.append(socket);return socket
        return Provider(route,self.settings,make_selection(),self.report,audit_path=self.path,
            http_transport=httpx.MockTransport(serve),connector=connector,
            shutdown_timeout=.04,local_close_timeout=.03)
    async def test_invalid_sdp_still_closes_exact_retained_session(self):
        provider=self.provider(invalid_sdp=True)
        with self.assertRaises(Exception):await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        cleanup=await provider.close()
        self.assertTrue(cleanup['provider_close_confirmed'])
        self.assertEqual(self.requests,['/v1/live/sessions'])
        self.assertEqual(self.connections,['wss://api.openai.com/v1/live/sessions/fixture_live_123/attach'])
        self.assertEqual([v['type'] for v in self.sockets[0].sent],['session.close'])
        self.assertTrue(json.loads(self.path.read_text())['provider_close_confirmed'])
    async def test_attach_failure_retains_identity_and_cleanup_attaches_without_create_retry(self):
        provider=self.provider(attach_fails=True)
        with self.assertRaises(Exception):await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        self.assertEqual(self.report['failure_stage'],'live_sideband_attachment')
        self.assertEqual(self.report['failure'],'attachment_unconfirmed')
        cleanup=await provider.close()
        self.assertTrue(cleanup['provider_close_confirmed'])
        self.assertEqual(len(self.requests),1);self.assertEqual(len(self.connections),2)
        self.assertEqual(self.connections[0],self.connections[1])
    async def test_successful_attachment_uses_existing_socket_and_close_is_single_use(self):
        provider=self.provider()
        await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        first,second=await asyncio.gather(provider.close(),provider.close())
        self.assertIs(first,second);self.assertTrue(first['provider_close_confirmed'])
        self.assertEqual(len(self.connections),1)
        self.assertEqual(len([v for v in self.sockets[0].sent if v['type']=='session.close']),1)
    async def test_unreadable_response_never_dials_a_guessed_identity(self):
        provider=self.provider(malformed=True)
        with self.assertRaises(Exception):await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        cleanup=await provider.close()
        self.assertFalse(cleanup['provider_close_confirmed'])
        self.assertEqual(cleanup['finalization_error'],'live_cleanup_identity_unavailable')
        self.assertEqual(self.connections,[])
    async def test_wrong_session_closed_event_is_not_closure_evidence(self):
        provider=self.provider(invalid_sdp=True,wrong_identity=True)
        with self.assertRaises(Exception):await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        cleanup=await provider.close()
        self.assertFalse(cleanup['provider_close_confirmed'])
        self.assertEqual(cleanup['finalization_error'],'session_binding_mismatch')
        self.assertEqual(self.sockets[0].close_calls,1)
    async def test_stalled_finalization_and_socket_close_are_bounded_not_success(self):
        provider=self.provider(invalid_sdp=True,finalize=False,close_hangs=True)
        with self.assertRaises(Exception):await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        started=time.monotonic();cleanup=await provider.close()
        self.assertLess(time.monotonic()-started,.5)
        self.assertFalse(cleanup['provider_close_confirmed'])
        self.assertEqual(cleanup['finalization_error'],'TimeoutError')
        self.assertEqual(cleanup['recovery_sideband_close_error'],'TimeoutError')
        self.assertEqual(len(self.requests),1)
    async def test_realtime_tls_failure_preserves_call_for_hangup(self):
        provider=self.provider(route='realtime',attach_fails=True)
        with self.assertRaises(ssl.SSLCertVerificationError):
            await provider.start('v=0\r\nm=audio 9 RTP/SAVPF 111\r\n')
        cleanup=await provider.close()
        self.assertTrue(cleanup['provider_close_confirmed'])
        self.assertEqual(self.requests,['/v1/realtime/calls','/v1/realtime/calls/rtc_fixture/hangup'])
    async def test_real_boundary_is_disabled_before_http_creation(self):
        provider=Provider('live',self.settings,make_selection(),{})
        with self.assertRaisesRegex(HarnessError,'prior_live_session_closure_unconfirmed'):
            await provider.start('unused')
        self.assertIsNone(provider.http);self.assertEqual(provider.audit.value['create_attempts'],0)


class QualificationHonestyTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_provider_gate_precedes_artifacts_browser_and_credentials(self):
        with self.assertRaisesRegex(HarnessError,'prior_live_session_closure_unconfirmed'):
            await main(SimpleNamespace(run_providers=True))
    def test_ca_context_is_verified_and_loaded_without_network(self):
        context=verified_tls_context()
        self.assertEqual(context.verify_mode,ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertGreater(context.cert_store_stats()['x509_ca'],0)
    def test_failed_provider_and_unconfirmed_cleanup_cannot_exit_success(self):
        for report in ({'failure':'fixture_failure'}, {'stopped_for_unresolved_provider_cleanup':True},
            {'routes':[{'failure':'TLS','cleanup':{'provider_close_confirmed':True}}]},
            {'routes':[{'cleanup':{'provider_close_confirmed':False}}]}):
            with self.subTest(report=report):self.assertEqual(qualification_exit_code(report),1)
        self.assertEqual(qualification_exit_code({'routes':[],'offline_checks_passed':True}),0)
    def test_javascript_meter_muted_consumer_and_cleanup_without_browser_or_devices(self):
        html=Path(__file__).with_name('synthetic_webrtc.html').read_text()
        program=r'''
const vm=require('node:vm'),assert=require('node:assert/strict'),fs=require('node:fs');
const html=fs.readFileSync(0,'utf8'),source=html.split('<script>')[1].split('</script>')[0];
let pauses=0,closes=0,played=false;
const box={window:{},setTimeout,performance:{now:()=>0},MediaStream:class{constructor(tracks){this.tracks=tracks;}},
  document:{createElement:()=>({play(){assert.equal(this.muted,true);assert.equal(this.volume,0);played=true;return Promise.resolve();},
    pause(){pauses++;}})}};
vm.createContext(box);vm.runInContext(source,box);
const messages=[];let Processor;
const worklet={currentFrame:48000,sampleRate:48000,AudioWorkletProcessor:class{constructor(){this.port={postMessage:x=>messages.push(x)};}},
  registerProcessor:(name,value)=>{Processor=value;}};
vm.createContext(worklet);vm.runInContext(vm.runInContext('meterSource',box),worklet);
const meter=new Processor(),zero=new Float32Array(128),signal=new Float32Array(128);signal[10]=.1;
meter.process([[]],[[new Float32Array(128)]]);
meter.process([[zero]],[[new Float32Array(128)]]);
meter.process([[signal]],[[new Float32Array(128)]]);
assert.equal(messages.length,3);assert.equal(messages[0].has_input,false);
assert.equal(messages[1].has_input,true);assert.equal(messages[1].first,null);assert.equal(messages[1].peak,0);
assert.equal(messages[2].first,(48000+10)/48000);assert.ok(messages[2].peak>.09);
box.fakeContext={currentTime:3,sampleRate:48000,state:'running',close:async()=>{closes++;box.fakeContext.state='closed';}};
vm.runInContext('ctx=fakeContext; sink={gain:{value:0}}; active={input:[{last:2}],output:[],return_source:[]};',box);
let result=box.window.result();assert.equal(result.latency_ms,null);assert.equal(result.first_remote_sample,null);
vm.runInContext('active.output=[{first:1.9,last:2.1}];',box);assert.ok(box.window.result().latency_ms<0);
vm.runInContext('activateRemotePlayback({kind:"audio"});',box);assert.equal(played,true);
(async()=>{const first=box.window.cleanup(),second=box.window.cleanup();assert.equal(first,second);
 await first;assert.equal(pauses,1);assert.equal(closes,1);console.log('meter, muted renderer and idempotent cleanup passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',program],input=html,text=True,capture_output=True,
            timeout=10,env=child_environment())
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('idempotent cleanup passed',result.stdout)


class TrialPermitTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.now=1000.0
    def permit(self,**kwargs):
        values=dict(acknowledgment=PRIOR_UNKNOWN_TRIAL,run_id=str(UUID(int=999)),
            output=self.root/'output',permit_root=self.root/'permits',clock=lambda:self.now)
        values.update(kwargs);return TrialPermit(**values)
    def test_missing_and_wrong_acknowledgments_are_rejected_before_consumption(self):
        for ack,code in ((None,'prior_live_session_closure_unconfirmed'),
                         ('another-trial','wrong_prior_unknown_acknowledgment')):
            with self.subTest(ack=ack), self.assertRaisesRegex(HarnessError,code):
                self.permit(acknowledgment=ack)
        self.assertFalse((self.root/'permits').exists())
    def test_run_identity_and_limits_are_validated_before_consumption(self):
        for changes in ({'run_id':None},{'run_id':'not-a-uuid'}, {'max_work_seconds':61},
                        {'max_work_seconds':0},{'max_trial_seconds':241},{'max_trial_seconds':80}):
            with self.subTest(changes=changes),self.assertRaises(HarnessError):self.permit(**changes)
        self.assertFalse((self.root/'permits').exists())
    def test_unique_run_consumption_survives_new_output_path(self):
        permit=self.permit();original=permit.path.read_text()
        self.assertEqual(stat.S_IMODE(permit.path.stat().st_mode),0o600)
        with self.assertRaisesRegex(HarnessError,'run_permit_already_consumed'):
            self.permit(output=self.root/'different-output')
        self.assertEqual(permit.path.read_text(),original)
    def test_realtime_first_and_one_attempt_each_two_total(self):
        permit=self.permit()
        with self.assertRaisesRegex(HarnessError,'realtime_cleanup_required_before_live'):
            permit.consume_creation('live')
        permit.consume_creation('realtime')
        with self.assertRaisesRegex(HarnessError,'trial_creation_budget_exhausted'):
            permit.consume_creation('realtime')
        with self.assertRaisesRegex(HarnessError,'realtime_cleanup_required_before_live'):
            permit.consume_creation('live')
        permit.record_outcome('realtime',provider_close_confirmed=True)
        permit.consume_creation('live')
        with self.assertRaisesRegex(HarnessError,'trial_creation_budget_exhausted'):
            permit.consume_creation('live')
        self.assertEqual(permit.value['attempts'],{'realtime':1,'live':1})
    def test_unconfirmed_new_cleanup_stops_permit_and_preserves_old_unknown(self):
        permit=self.permit();permit.consume_creation('realtime')
        permit.record_outcome('realtime',provider_close_confirmed=False)
        with self.assertRaisesRegex(HarnessError,'trial_stopped_after_unconfirmed_cleanup'):
            permit.consume_creation('live')
        self.assertEqual(permit.value['attempts'],{'realtime':1,'live':0})
        self.assertEqual(json.loads(permit.path.read_text())['prior_unknown']['status'],'UNKNOWN')
    def test_success_of_new_trial_never_changes_old_unknown(self):
        permit=self.permit()
        for route in ('realtime','live'):
            permit.consume_creation(route);permit.record_outcome(route,provider_close_confirmed=True)
        permit.finish({'routes':[{'cleanup':{'provider_close_confirmed':True}}]})
        self.assertEqual(permit.value['prior_unknown'],
            {'trial':PRIOR_UNKNOWN_TRIAL,'status':'UNKNOWN','acknowledged':True})
    def test_overall_deadline_reserves_cleanup_before_any_new_creation(self):
        permit=self.permit();self.assertEqual(permit.work_budget('realtime'),60)
        self.now+=185
        self.assertEqual(permit.work_budget('realtime'),15)
        self.now+=15
        with self.assertRaisesRegex(HarnessError,'trial_cleanup_reserve_reached'):
            permit.consume_creation('realtime')
        self.assertEqual(permit.value['attempts']['realtime'],0)
        self.assertEqual(permit.remaining(),40)


class PermitExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_create_without_response_is_durable_and_cannot_retry_with_new_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);calls=[]
            permit=TrialPermit(acknowledgment=PRIOR_UNKNOWN_TRIAL,run_id=str(UUID(int=1000)),
                output=root/'output',permit_root=root/'permits')
            async def fail(request):calls.append(request);raise httpx.ReadTimeout('ambiguous response')
            for attempt in range(2):
                audit=CreationAudit('realtime',root/f'creation-{attempt}.json',permit=permit)
                async with AuditedHTTPClient(creation_audit=audit,transport=httpx.MockTransport(fail)) as http:
                    expected=httpx.ReadTimeout if attempt==0 else HarnessError
                    with self.assertRaises(expected):await http.post('https://api.openai.com/v1/realtime/calls')
            self.assertEqual(len(calls),1)
            retained=json.loads(permit.path.read_text())
            self.assertEqual(retained['attempts'],{'realtime':1,'live':0})
            permit.record_outcome('realtime',provider_close_confirmed=False)
            with self.assertRaises(HarnessError):permit.consume_creation('live')
    async def test_work_timeout_still_joins_provider_and_browser_inside_reserved_time(self):
        from unittest.mock import AsyncMock
        import synthetic_webrtc as experiment
        with tempfile.TemporaryDirectory() as directory,patch.object(TrialPermit,'CLEANUP_RESERVE_SECONDS',.2):
            root=Path(directory);order=[]
            permit=TrialPermit(acknowledgment=PRIOR_UNKNOWN_TRIAL,run_id=str(UUID(int=1001)),
                output=root/'output',permit_root=root/'permits')
            permit.deadline=time.monotonic()+.25
            class FakeBrowser:
                def __init__(self,*args,**kwargs):pass
                async def close(self):
                    await asyncio.sleep(.01);order.append('browser_closed');return {'closed':True}
            class FakeProvider:
                def __init__(self,route,settings,selection,report,**kwargs):self.report=report
                async def close(self):
                    await asyncio.sleep(.01);order.append('provider_closed')
                    self.report['cleanup']={'provider_close_confirmed':False,'create_response_statuses':[]}
                    return self.report['cleanup']
            async def blocked(*args):
                order.append('work_started')
                try:await asyncio.Event().wait()
                finally:order.append('work_cancelled')
            with patch.object(experiment,'Browser',FakeBrowser),patch.object(experiment,'Provider',FakeProvider),\
                 patch.object(experiment,'drive_route',blocked):
                result=await run_route('realtime',SimpleNamespace(OPENAI_REALTIME_MODEL='fixture',
                    OPENAI_REALTIME_VOICE='fixture'),make_selection(),root/'unused.wav',root/'route',permit=permit)
            self.assertEqual(result['failure'],'provider_work_timeout')
            self.assertEqual(order,['work_started','work_cancelled','provider_closed','browser_closed'])
            self.assertLessEqual(result['work_budget_seconds'],.05)
            self.assertGreater(permit.remaining(),0)
            self.assertEqual(permit.value['attempts'],{'realtime':0,'live':0})
    async def test_wrong_cli_ack_never_opens_browser_or_creates_provider(self):
        from unittest.mock import AsyncMock
        import synthetic_webrtc as experiment
        with tempfile.TemporaryDirectory() as directory,patch.object(experiment.Browser,'start',new_callable=AsyncMock) as start:
            args=SimpleNamespace(run_providers=True,acknowledge_prior_unknown='wrong',
                one_run_id=str(UUID(int=1002)),output=Path(directory)/'output')
            with self.assertRaisesRegex(HarnessError,'wrong_prior_unknown_acknowledgment'):await main(args)
            start.assert_not_awaited();self.assertFalse(args.output.exists())


class RealtimeContextAcknowledgmentTests(unittest.IsolatedAsyncioTestCase):
    def provider(self):
        provider=Provider('realtime',SimpleNamespace(),make_selection(),{},initialization_timeout=.01)
        provider.realtime_context_item={'id':'fixture_context_item','type':'message','role':'system',
            'content':[{'type':'input_text','text':wire_text(provider.selection)}]}
        return provider
    def event(self,provider,kind='conversation.item.done',**item_changes):
        item=json.loads(json.dumps(provider.realtime_context_item))
        item.update(status='completed');item.update(item_changes)
        return {'type':kind,'event_id':'server_ack_fixture','item':item}
    def test_done_matches_exact_submitted_item_and_records_receipt(self):
        provider=self.provider();event=self.event(provider)
        provider.observe_realtime_context(event)
        self.assertTrue(provider.context_ack.is_set())
        receipt=provider.report['realtime_context_receipt']
        self.assertEqual(receipt['item'],provider.realtime_context_item)
        self.assertEqual(receipt['event_type'],'conversation.item.done')
        self.assertIn('not proof the model used',receipt['finality'])
    def test_added_is_not_final_even_with_completed_status(self):
        provider=self.provider()
        provider.observe_realtime_context(self.event(provider,'conversation.item.added'))
        self.assertTrue(provider.report['realtime_context_added_observed'])
        self.assertFalse(provider.context_ack.is_set())
        self.assertNotIn('realtime_context_receipt',provider.report)
        provider.observe_realtime_context(self.event(provider))
        self.assertTrue(provider.context_ack.is_set())
    def test_foreign_mutated_nonfinal_and_legacy_events_cannot_ack(self):
        provider=self.provider()
        cases=[self.event(provider,id='foreign_item'),self.event(provider,role='user'),
            self.event(provider,type='function_call'),self.event(provider,status='incomplete'),
            self.event(provider,status='in_progress'),self.event(provider,'conversation.item.created'),
            self.event(provider,content=[{'type':'input_text','text':'The display does not read 42.'}]),
            self.event(provider,content=provider.realtime_context_item['content']+[{'type':'input_text','text':'extra'}]),
            self.event(provider,content=[{'type':'output_text','text':wire_text(provider.selection)}]),
            {'type':'conversation.item.done','item':provider.realtime_context_item},
            {'type':'conversation.item.done','event_id':'fixture','item':None}]
        for event in cases:
            with self.subTest(event=event):
                provider.observe_realtime_context(event)
                self.assertFalse(provider.context_ack.is_set())
                self.assertNotIn('realtime_context_receipt',provider.report)
    def test_done_without_optional_status_is_final_but_before_submission_is_not(self):
        provider=self.provider();event=self.event(provider);event['item'].pop('status')
        expected=provider.realtime_context_item;provider.realtime_context_item=None
        provider.observe_realtime_context(event)
        self.assertFalse(provider.context_ack.is_set())
        provider.realtime_context_item=expected;provider.observe_realtime_context(event)
        self.assertTrue(provider.context_ack.is_set())
    def test_duplicate_does_not_replace_first_matching_receipt(self):
        provider=self.provider();event=self.event(provider)
        provider.observe_realtime_context(event)
        receipt=json.loads(json.dumps(provider.report['realtime_context_receipt']))
        event['event_id']='later_duplicate';provider.observe_realtime_context(event)
        self.assertEqual(provider.report['realtime_context_receipt'],receipt)
    async def test_initialization_deadlines_are_named_not_provider_work_timeouts(self):
        for stage in ('realtime_session_update_ack','realtime_context_finalized_ack','live_context_ack'):
            with self.subTest(stage=stage):
                provider=self.provider()
                with self.assertRaisesRegex(HarnessError,'^'+stage+'_timeout$'):
                    await provider.wait_initialization(asyncio.Event(),stage)
                self.assertEqual(provider.stage,stage)
                self.assertFalse(provider.context_ack.is_set())
    async def test_real_start_reader_consumes_matching_added_then_done(self):
        import synthetic_webrtc as experiment
        sent=[]
        class FixtureConnection:
            def __init__(self,**kwargs):
                self.answer='v=0\r\n';self.queue=asyncio.Queue();self.call_id=None
            async def connect(self):pass
            async def send_event(self,event):
                sent.append(event)
                if event['type']=='session.update':
                    self.queue.put_nowait({'type':'session.updated'})
                if event['type']=='conversation.item.create':
                    for kind in ('conversation.item.added','conversation.item.done'):
                        self.queue.put_nowait({'type':kind,'event_id':'server_'+kind,
                            'item':dict(event['item'],status='completed')})
            async def receive_event(self):return await self.queue.get()
            async def close(self):pass
        async def connector(*args,**kwargs):raise AssertionError('No network permitted')
        provider=Provider('realtime',SimpleNamespace(),make_selection(),{},
            http_transport=httpx.MockTransport(lambda request:httpx.Response(500)),connector=connector,
            initialization_timeout=.2)
        with patch.object(experiment,'VerifiedRealtimeConnection',FixtureConnection),\
             patch.object(experiment,'build_session_update_event',return_value={'type':'session.update'}):
            try:
                answer=await provider.start('v=0\r\n')
                self.assertEqual(answer,'v=0\r\n')
                self.assertEqual([event['type'] for event in sent],['session.update','conversation.item.create'])
                self.assertTrue(provider.context_ack.is_set())
                self.assertTrue(provider.report['realtime_context_added_observed'])
                self.assertEqual(provider.report['realtime_context_receipt']['item'],sent[1]['item'])
                self.assertEqual(provider.audit.value['create_attempts'],0)
                self.assertNotIn('reader_error',provider.report)
            finally:await provider.close()


class TimeoutClassificationTests(unittest.IsolatedAsyncioTestCase):
    async def route_failure(self,error):
        import synthetic_webrtc as experiment
        class FixtureBrowser:
            def __init__(self,*args,**kwargs):pass
            async def close(self):return {'closed':True}
        class FixtureProvider:
            def __init__(self,route,settings,selection,report,**kwargs):
                self.report=report;self.stage='realtime_context_finalized_ack'
            async def close(self):self.report['cleanup']={'provider_close_confirmed':False}
        async def fail(*args):raise error
        with tempfile.TemporaryDirectory() as directory,patch.object(experiment,'Browser',FixtureBrowser),\
             patch.object(experiment,'Provider',FixtureProvider),patch.object(experiment,'drive_route',fail):
            return await run_route('realtime',SimpleNamespace(OPENAI_REALTIME_MODEL='fixture',
                OPENAI_REALTIME_VOICE='fixture'),make_selection(),Path(directory)/'unused.wav',Path(directory)/'route')
    async def test_named_initialization_timeout_survives_route_reporting(self):
        result=await self.route_failure(HarnessError('realtime_context_finalized_ack_timeout'))
        self.assertEqual(result['failure'],'realtime_context_finalized_ack_timeout')
        self.assertEqual(result['failure_stage'],'realtime_context_finalized_ack')
        self.assertFalse(result['work_deadline_expired'])
    async def test_inner_operation_timeout_is_not_expired_outer_deadline(self):
        result=await self.route_failure(TimeoutError())
        self.assertEqual(result['failure'],'provider_operation_timeout')
        self.assertFalse(result['work_deadline_expired'])


class ContextCorrectionFixtureTests(unittest.TestCase):
    def test_whole_corrected_fact_set_fits_original_budget_with_exact_source_revision(self):
        initial=make_selection();update=make_context_correction(initial)
        validate_context_correction(initial,update)
        self.assertLessEqual(encoded_bytes(wire_text(update.selection)),500)
        self.assertEqual(update.selection.selected_entry_ids,('display',))
        self.assertEqual(update.selection.omitted_entry_ids,())
        self.assertIn(CANONICAL_RESULT_TEXT,update.selection.text)
        self.assertEqual(update.selection.dependencies[0].source_id,initial.dependencies[0].source_id)
        self.assertEqual(update.selection.dependencies[0].revision,2)
        self.assertEqual(update.base_cursor,initial.cursor)
        self.assertEqual(update.selection.cursor.revision,2)
        print(json.dumps({'correction_encoded_bytes':encoded_bytes(wire_text(update.selection)),
                          'whole_fact_set':CANONICAL_RESULT_FACTS}))
    def test_stale_duplicate_and_foreign_source_updates_fail(self):
        initial=make_selection();update=make_context_correction(initial)
        candidates=[replace(update,update_id='foreign'),replace(update,selection=initial),
            replace(update,selection=make_selection(revision=2,summary=CANONICAL_RESULT_TEXT,source_id=UUID(int=99))),
            replace(update,selection=make_selection(revision=2,summary=CANONICAL_RESULT_TEXT,owner_id='foreign')),
            replace(update,selection=make_selection(revision=2,summary=CANONICAL_RESULT_TEXT,epoch=UUID(int=99)))]
        for candidate in candidates:
            with self.subTest(update=candidate.update_id),self.assertRaises(HarnessError):
                validate_context_correction(initial,candidate)
        with self.assertRaises(HarnessError):validate_context_correction(update.selection,update)
    def test_omitted_or_altered_whole_fact_entry_is_not_silently_published(self):
        initial=make_selection();update=make_context_correction(initial)
        for summary in ('x'*1000,CANONICAL_RESULT_TEXT.replace('0.25 kg','2.5 kg')):
            with self.subTest(summary=summary[:40]),self.assertRaises(HarnessError):
                validate_context_correction(initial,replace(update,selection=make_selection(revision=2,summary=summary)))
    def test_exact_and_explicit_spoken_expression_preserve_facts(self):
        for text in (CANONICAL_RESULT_TEXT,
            'Display number is seventeen. Mass: zero point two five kilograms. Task not completed. '
            'Confidence uncertain. Code: open parenthesis low plus high close parenthesis floor divided by two.'):
            with self.subTest(text=text):self.assertTrue(correction_fact_check(text)['passed'])
    def test_each_omitted_fact_and_ack_only_output_fails(self):
        pieces=CANONICAL_RESULT_TEXT.split('. ')
        for piece in pieces:
            with self.subTest(omitted=piece):
                self.assertFalse(correction_fact_check(CANONICAL_RESULT_TEXT.replace(piece,''))['passed'])
        self.assertFalse(correction_fact_check('Acknowledged.')['passed'])
    def test_contradictions_cannot_hide_behind_correct_substrings(self):
        for extra in (' Display reads 42.',' Mass: 2.5 kg.',' Task completed.',
                      ' Confidence certain.',' Code: low + high // 2.'):
            with self.subTest(extra=extra):
                result=correction_fact_check(CANONICAL_RESULT_TEXT+extra)
                self.assertFalse(result['passed']);self.assertTrue(result['contradictions'])
    def test_actual_output_and_samples_required_not_just_receipt_or_canonical_input(self):
        receipt={'update_id':CORRECTION_UPDATE_ID,'source_validated':True,'provider_acknowledged':True}
        turn={'transcript':CANONICAL_RESULT_TEXT,'last_input_sample':1,'first_remote_sample':2,
              'first_decoded_audio_ms':1000,'physical_gain':0}
        self.assertTrue(assess_correction_turn(turn,receipt)['passed'])
        self.assertIsNone(assess_correction_turn(turn,receipt)['first_useful_audio_ms'])
        for changes in ({'transcript':''},{'transcript':'The display reads 42.'},
                        {'first_remote_sample':None},{'overflow':True},{'physical_gain':1}):
            with self.subTest(changes=changes):
                self.assertFalse(assess_correction_turn(dict(turn,**changes),receipt)['passed'])
        for changes in ({'update_id':'foreign'},{'source_validated':False},{'provider_acknowledged':False}):
            with self.subTest(changes=changes):
                self.assertFalse(assess_correction_turn(turn,dict(receipt,**changes))['passed'])
    def test_synthetic_clips_fit_two_provider_budget_without_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            _,initial=make_speech(Path(directory))
            _,corrected=make_speech(Path(directory),question=CORRECTION_QUESTION,stem='correction')
        self.assertLess(2*(initial+corrected),60)
        print(json.dumps({'initial_input_seconds':initial,'correction_input_seconds':corrected,
                          'maximum_two_provider_input_seconds':2*(initial+corrected)}))
    def test_second_browser_fixture_is_fixed_local_audio_not_microphone(self):
        program=r'''
const vm=require('node:vm'),assert=require('node:assert/strict'),fs=require('node:fs');
const source=fs.readFileSync(0,'utf8').split('<script>')[1].split('</script>')[0];
let requests=0;
const box={window:{},setTimeout,performance:{now:()=>0},fetch:async url=>{
 assert.equal(url,'/correction.wav');requests++;return {ok:true,arrayBuffer:async()=>new ArrayBuffer(0)};}};
vm.createContext(box);vm.runInContext(source,box);
vm.runInContext('ctx={decodeAudioData:async()=>({duration:8})}; fixture={duration:3};',box);
(async()=>{assert.equal((await box.window.useCorrectionFixture()).duration,8);
 assert.equal(vm.runInContext('fixture.duration',box),8);
 vm.runInContext('ctx.decodeAudioData=async()=>({duration:16});',box);
 await assert.rejects(box.window.useCorrectionFixture(),/synthetic_input_budget/);
 assert.equal(vm.runInContext('fixture.duration',box),8);assert.equal(requests,2);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
        result=subprocess.run(['node','-e',program],input=Path(__file__).with_name('synthetic_webrtc.html').read_text(),
            text=True,capture_output=True,timeout=10,env=child_environment())
        self.assertEqual(result.returncode,0,result.stderr)


class ContextCorrectionWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_realtime_foreign_or_old_ack_does_not_release_correction(self):
        provider=Provider('realtime',SimpleNamespace(),make_selection(),{},initialization_timeout=.3)
        sent=asyncio.Event();writes=[]
        class FixtureTransport:
            async def send_event(self,event):writes.append(event);sent.set()
        provider.transport=FixtureTransport();update=make_context_correction(provider.selection)
        task=asyncio.create_task(provider.publish_context_correction(update))
        try:
            await asyncio.wait_for(sent.wait(),1)
            for item in (dict(provider.realtime_context_item,id='fixture_context_item'),
                         dict(provider.realtime_context_item,content=[{'type':'input_text','text':'foreign'}])):
                provider.observe_realtime_context({'type':'conversation.item.done','event_id':'foreign','item':item})
                self.assertFalse(provider.context_ack.is_set());self.assertFalse(task.done())
            provider.observe_realtime_context({'type':'conversation.item.done','event_id':'exact',
                'item':dict(provider.realtime_context_item,status='completed')})
            record=await task
            self.assertTrue(record['provider_acknowledged']);self.assertTrue(record['written'])
            self.assertEqual(writes[0]['item']['id'],CORRECTION_UPDATE_ID)
            self.assertEqual(writes[0]['item']['content'][0]['text'],wire_text(update.selection))
            with self.assertRaises(HarnessError):await provider.publish_context_correction(update)
            self.assertEqual(len(writes),1)
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    async def test_live_harness_correlates_current_update_not_old_append_ack(self):
        from app.voice_routes.live_protocol import LiveCommandStatus
        provider=Provider('live',SimpleNamespace(),make_selection(),{})
        provider.context_command_id=CORRECTION_UPDATE_ID
        queue=asyncio.Queue();foreign_seen=asyncio.Event()
        class FixtureTransport:
            async def receive(self):
                event=await queue.get()
                if event['command_id']=='fixture_context':foreign_seen.set()
                return event
        provider.transport=FixtureTransport()
        # Adapter-correlation fixture only. Actual protocol shapes retain their
        # own existing tests; this is not a provider ACK or network observation.
        provider.protocol=SimpleNamespace(provider_session_id='fixture',accept=lambda event,**kwargs:
            SimpleNamespace(command_id=event['command_id'],status=LiveCommandStatus.ACCEPTED))
        task=asyncio.create_task(provider.read())
        try:
            queue.put_nowait({'type':'session.thinking.appended','command_id':'fixture_context'})
            await asyncio.wait_for(foreign_seen.wait(),1)
            self.assertFalse(provider.context_ack.is_set())
            queue.put_nowait({'type':'session.thinking.appended','command_id':CORRECTION_UPDATE_ID})
            await asyncio.wait_for(provider.context_ack.wait(),1)
        finally:
            task.cancel();await asyncio.gather(task,return_exceptions=True)


class ContextCorrectionReportLabelTests(unittest.TestCase):
    def route(self,name='realtime'):
        return {'route':name,'creation':{'create_attempts':1,'http_status':201},
            'turns':[{'label':'initial','contains_expected_number':True},
                {'label':'corrected','transcript':CANONICAL_RESULT_TEXT,'last_input_sample':1,
                 'first_remote_sample':2,'first_decoded_audio_ms':1000,'physical_gain':0}],
            'context_updates':[{'update_id':CORRECTION_UPDATE_ID,'source_validated':True,
                                'provider_acknowledged':True}],
            'cleanup':{'provider_close_confirmed':True}}
    def test_no_routes_and_failed_creation_do_not_claim_provider_behavior(self):
        self.assertEqual(context_correction_provider_behavior({}),'not_exercised')
        route={'route':'realtime','creation':{'create_attempts':1},'failure':'creation_unconfirmed'}
        self.assertEqual(context_correction_provider_behavior({'routes':[route]}),'attempted_no_correction_output')
    def test_only_one_observed_route_is_partial_and_two_complete_routes_pass(self):
        self.assertEqual(context_correction_provider_behavior({'routes':[self.route()]}),'exercised_partial')
        self.assertEqual(context_correction_provider_behavior({'routes':[self.route(),self.route('live')]}),
                         'exercised_both_passed')
    def test_ack_or_stored_pass_flag_cannot_replace_actual_output_facts(self):
        for text in ('','Acknowledged.','Display reads 42.'):
            with self.subTest(text=text):
                route=self.route();route['turns'][1]['transcript']=text
                route['correction_qualification']={'passed':True}
                self.assertNotEqual(context_correction_provider_behavior({'routes':[route,self.route('live')]}),
                                    'exercised_both_passed')
    def test_failure_missing_samples_and_unknown_cleanup_remain_incomplete(self):
        variants=[]
        route=self.route();route['failure']='fixture_failed';variants.append(route)
        route=self.route();route['cleanup']['provider_close_confirmed']=False;variants.append(route)
        route=self.route();route['turns'][1]['first_remote_sample']=None;variants.append(route)
        route=self.route();route['context_updates'][0]['source_validated']=False;variants.append(route)
        route=self.route();route['turns'][0]['contains_expected_number']=False;variants.append(route)
        for route in variants:
            with self.subTest(route=route):
                self.assertEqual(context_correction_provider_behavior({'routes':[route,self.route('live')]}),
                                 'exercised_failed_or_incomplete')
        self.assertEqual(context_correction_provider_behavior({'routes':[self.route(),self.route('live')],
            'failure':'trial_overall_deadline_exceeded'}),'exercised_failed_or_incomplete')
    def test_duplicate_or_unknown_route_is_not_two_provider_coverage(self):
        for routes in ([self.route(),self.route()],[self.route('foreign')]):
            with self.subTest(routes=routes):
                self.assertEqual(context_correction_provider_behavior({'routes':routes}),'invalid_route_results')

    def test_permit_final_deadline_overrides_previously_passing_label(self):
        with tempfile.TemporaryDirectory() as directory:
            clock=[10.0]
            permit=TrialPermit(acknowledgment=PRIOR_UNKNOWN_TRIAL,
                run_id=str(UUID(int=81)),output=Path(directory)/'report',
                permit_root=Path(directory)/'permits',clock=lambda:clock[0])
            report={'scenario':'context-correction','routes':[self.route(),self.route('live')],
                    'correction_provider_behavior':'exercised_both_passed'}
            clock[0]=250.001
            permit.finish(report)
            self.assertEqual(report['failure'],'trial_overall_deadline_exceeded')
            self.assertEqual(report['correction_provider_behavior'],'exercised_failed_or_incomplete')
            self.assertEqual(qualification_exit_code(report),1)
            self.assertEqual(permit.value['status'],'failed')
            self.assertEqual(permit.value['elapsed_seconds'],report['elapsed_seconds'])
            self.assertEqual(report['creation_attempts'],permit.value['attempts'])

    def test_permit_finalization_preserves_in_budget_and_existing_failure_results(self):
        for elapsed,prior_failure in ((239.999,None),(240.0,None),(100.0,'fixture_failed')):
            with self.subTest(elapsed=elapsed,prior_failure=prior_failure), tempfile.TemporaryDirectory() as directory:
                clock=[10.0]
                permit=TrialPermit(acknowledgment=PRIOR_UNKNOWN_TRIAL,
                    run_id=str(UUID(int=82)),output=Path(directory)/'report',
                    permit_root=Path(directory)/'permits',clock=lambda:clock[0])
                report={'scenario':'context-correction','routes':[self.route(),self.route('live')]}
                if prior_failure:
                    report['failure']=prior_failure
                clock[0]+=elapsed
                permit.finish(report)
                self.assertEqual(report.get('failure'),prior_failure)
                self.assertEqual(report['correction_provider_behavior'],
                    'exercised_failed_or_incomplete' if prior_failure else 'exercised_both_passed')
                self.assertEqual(permit.value['status'],'failed' if prior_failure else 'completed')
                self.assertEqual(permit.value['elapsed_seconds'],report['elapsed_seconds'])


if __name__=='__main__': unittest.main()
