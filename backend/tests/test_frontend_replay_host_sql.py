"""Actual isolated child startup and admission, never remote voice acceptance."""
import asyncio
import json
import logging
import os
import sys
import ssl
from pathlib import Path
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import patch

import asyncpg
import certifi
import httpx

from app.knowledge.hindsight import HindsightClient
from app.knowledge.service_entrypoint import secret_file
from evals.responsive_voice.frontend_host_qualification import (
    QualificationFailure, cleanup_owned_native_bank, cleanup_prepared_host,
    native_bank_inventory, qualify_frontend_host, serve_prepared_host, NO_MEDIA, NATIVE_MEDIA,
    child_environment, cleanup_owned_storage, configured_storage, count_owned_rows, ChildOutputDrain,
    list_owned_storage, save_json, storage_http, configure_native_diagnostic_logging,
)
from evals.responsive_voice.frontend_host import ReplayHostScope, scoped_database_url
from app.settings import JarvisSettings
from app.storage.supabase_config import SupabaseStorageConfig
from app.visual_memory.storage import VisualFrameStorageClient
from app.errors import JarvisHTTPError
from app.realtime.live_state_bridge import install_responsive_voice
from app.realtime.telemetry import RealtimeTrace


class FrontendReplayHostLoggingTests(unittest.TestCase):
    def setUp(self):
        self.loggers = {name: logging.getLogger(name) for name in (
            '', 'app', 'app.realtime', 'app.realtime.webrtc', 'app.realtime.telemetry',
            'app.realtime.direct_session', 'app.voice_routes.bridge',
            'uvicorn', 'uvicorn.error', 'uvicorn.access', 'httpx', 'httpcore', 'websockets.client',
        )}
        original = {name: logger.level for name, logger in self.loggers.items()}
        for name, logger in self.loggers.items():
            self.addCleanup(logger.setLevel, original[name])
            logger.setLevel(logging.WARNING)
        self.loggers['httpcore'].setLevel(logging.ERROR)

    def levels(self):
        return {name: logger.level for name, logger in self.loggers.items()}

    def test_native_profile_enables_only_existing_sanitized_loggers(self):
        expected = self.levels() | {
            'app.realtime.webrtc': logging.INFO,
            'app.realtime.telemetry': logging.INFO,
        }
        configure_native_diagnostic_logging(NATIVE_MEDIA)
        self.assertEqual(self.levels(), expected)
        configure_native_diagnostic_logging(NATIVE_MEDIA)
        self.assertEqual(self.levels(), expected)
        self.assertTrue(self.loggers['app.realtime.telemetry'].isEnabledFor(logging.INFO))

    def test_non_native_profiles_preserve_existing_levels(self):
        expected = self.levels()
        for profile in (NO_MEDIA, None, 'unsupported_profile'):
            with self.subTest(profile=profile):
                configure_native_diagnostic_logging(profile)
                self.assertEqual(self.levels(), expected)
        # A nonnative configuration also must not reset explicit caller levels.
        configure_native_diagnostic_logging(NATIVE_MEDIA)
        expected = self.levels()
        configure_native_diagnostic_logging(NO_MEDIA)
        self.assertEqual(self.levels(), expected)

    def test_enabled_trace_keeps_allowlisted_controls_and_excludes_private_bodies(self):
        configure_native_diagnostic_logging(NATIVE_MEDIA)
        logger = self.loggers['app.realtime.telemetry']
        trace = RealtimeTrace(provider='openai', audio_session_id='offline_audio', jarvis_session_id=uuid4())
        private = 'PRIVATE_SECRET Bearer sk-secret SDP v=0\nprivate transcript'
        with patch.object(logger, 'handle') as handle:
            trace.event('control_handler_finished', event_type='input_audio_buffer.committed',
                origin='provider', response_id='resp_fixture', turn_generation=2, duration_ms=1.25,
                response_requested=False, reason=private, item_id='x'*161,
                receive_wait_ms=float('inf'), user_audio_pending=private,
                text=private, transcript=private, api_key=private, authorization=private,
                headers={'authorization':private}, sdp=private, prompt=private,
                payload={'text':private}, body=private, content=private, audio_base64=private)
            trace.event(private, response_id='resp_fixture')
        handle.assert_called_once()
        log_record = handle.call_args.args[0]
        self.assertEqual(log_record.levelno, logging.INFO)
        message = log_record.getMessage()
        self.assertTrue(message.startswith('realtime_trace '))
        record = json.loads(message.removeprefix('realtime_trace '))
        self.assertEqual(set(record), {
            'event_type', 'origin', 'response_id', 'turn_generation', 'duration_ms', 'response_requested',
            'provider', 'audio_session_id', 'jarvis_session_id',
            'schema', 'event', 'process_id', 'at', 'monotonic_s',
        })
        self.assertEqual(record['event_type'], 'input_audio_buffer.committed')
        self.assertEqual(record['response_id'], 'resp_fixture')
        self.assertEqual(record['turn_generation'], 2)
        self.assertIs(record['response_requested'], False)
        for forbidden in ('PRIVATE_SECRET', 'Bearer', 'sk-secret', 'SDP', 'private transcript'):
            self.assertNotIn(forbidden, message)


class FrontendReplayHostOfflineTests(unittest.IsolatedAsyncioTestCase):
    async def test_supported_inventory_and_independent_cleanup_never_get_bank_root(self):
        bank = 'jarvis_'+'a'*32
        banks, calls, clients = {bank, 'unrelated-shared-bank'}, [], []
        def handle(request):
            calls.append((request.method, request.url.path))
            if request.method=='GET' and request.url.path=='/v1/default/banks':
                return httpx.Response(200, json={'banks':[{'bank_id':b} for b in sorted(banks)]})
            if request.method=='DELETE' and request.url.path=='/v1/default/banks/'+bank:
                banks.discard(bank)
                return httpx.Response(200, json={})
            return httpx.Response(405, json={'detail':'Method Not Allowed'})
        def factory(*args, **kwargs):
            client = HindsightClient(*args, **kwargs, transport=httpx.MockTransport(handle))
            clients.append(client)
            return client
        with tempfile.TemporaryDirectory() as directory:
            key = Path(directory)/'key'
            fd = os.open(key, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w') as stream:
                stream.write('offline-qualification-key-only\n')
            result = await cleanup_owned_native_bank(bank, hindsight_url='http://127.0.0.1:9',
                api_key_file=key, client_factory=factory)
        self.assertEqual(calls, [('DELETE','/v1/default/banks/'+bank), ('GET','/v1/default/banks')])
        self.assertEqual(banks, {'unrelated-shared-bank'})
        self.assertEqual(len(clients), 2)
        self.assertIsNot(clients[0], clients[1])
        self.assertTrue(result['native_bank_absent'] and result['independent_client_audit'])

    async def test_inventory_rejects_malformed_or_ambiguous_records(self):
        for payload in ({'banks':None}, {'banks':[{}]}, {'banks':[{'bank_id':'same'},{'bank_id':'same'}]}):
            client = HindsightClient('http://127.0.0.1:9', 'offline-key', private_http_origin='http://127.0.0.1:9',
                transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
            try:
                with self.assertRaises(QualificationFailure):
                    await native_bank_inventory(client)
            finally:
                await client.close()

    async def test_native_profile_explicit_credentials_and_isolation(self):
        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(), memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(seconds=120))
        settings = JarvisSettings(_env_file=None, OPENAI_API_KEY='offline-openai', GEMINI_API_KEY='offline-gemini',
            OPENAI_REALTIME_MODEL='offline-realtime', SUPABASE_URL='https://storage.invalid',
            SUPABASE_SERVICE_ROLE_KEY='offline-storage', SUPABASE_STORAGE_BUCKET='fixture',
            RESPONSIVE_VOICE_ENABLED=False, OPENAI_LIVE_ENABLED=False, GEMINI_LIVE_ENABLED=False)
        arguments = dict(dsn='postgresql://fixture.invalid/db?search_path=isolated', scope=scope,
            token='offline-bearer-not-real', origin='http://127.0.0.1:9', hindsight_url='http://127.0.0.1:8',
            api_key_file='/offline-key')
        default = child_environment(**arguments)
        actual = child_environment(**arguments, profile=NATIVE_MEDIA, configured=settings)
        self.assertEqual(default['JARVIS_FRONTEND_REPLAY_PROFILE'], NO_MEDIA)
        self.assertEqual(default['OPENAI_API_KEY'], '')
        self.assertEqual(default['SUPABASE_SERVICE_ROLE_KEY'], '')
        self.assertEqual(actual['OPENAI_API_KEY'], 'offline-openai')
        self.assertEqual(actual['GEMINI_API_KEY'], 'offline-gemini')
        self.assertEqual(actual['SUPABASE_SERVICE_ROLE_KEY'], 'offline-storage')
        effective = JarvisSettings(_env_file=None, **actual)
        disabled = JarvisSettings(_env_file=None, **default)
        self.assertIs(effective.RESPONSIVE_VOICE_ENABLED, True)
        self.assertIs(disabled.RESPONSIVE_VOICE_ENABLED, False)
        self.assertEqual(effective.OPENAI_REALTIME_MODEL, 'offline-realtime')
        self.assertEqual(effective.AUDIO_PROVIDER_DEFAULT, 'openai')
        self.assertEqual(str(effective.OPENAI_LIVE_ENABLED).lower(), 'false')
        self.assertEqual(str(effective.GEMINI_LIVE_ENABLED).lower(), 'false')
        self.assertEqual(actual['JARVIS_OWNER_ID'], scope.owner_id)
        self.assertEqual(actual['SUPABASE_DB_URL'], arguments['dsn'])
        self.assertEqual(actual['SECOND_BRAIN_PUSH_ENABLED'], 'false')
        self.assertEqual(actual['SECOND_BRAIN_MAIL_ENABLED'], 'false')
        self.assertEqual(actual['SECOND_BRAIN_SANDBOX_ENABLED'], 'false')
        self.assertEqual(actual['SECOND_BRAIN_ARTIFACTS_ENABLED'], 'false')
        self.assertLessEqual(int(actual['JARVIS_REALTIME_MAX_SESSION_SECONDS']), 120)

    async def test_effective_native_profile_passes_real_enable_gate_without_provider_execution(self):
        """Gate qualification only, not a fabricated live-runtime acceptance."""
        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(), memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(seconds=120))
        for configured_flag in (False, True):
            configured = JarvisSettings(_env_file=None, RESPONSIVE_VOICE_ENABLED=configured_flag,
                OPENAI_API_KEY='offline-openai', GEMINI_API_KEY='offline-gemini',
                OPENAI_REALTIME_MODEL='offline-realtime', OPENAI_LIVE_ENABLED=False, GEMINI_LIVE_ENABLED=False,
                SUPABASE_URL='https://storage.invalid', SUPABASE_SERVICE_ROLE_KEY='offline-storage')
            for profile, expected_code in ((NO_MEDIA, 'responsive_voice_disabled'),
                                           (NATIVE_MEDIA, 'live_state_unavailable')):
                with self.subTest(configured_flag=configured_flag, profile=profile):
                    env = child_environment(dsn='postgresql://fixture.invalid/db?search_path=isolated',
                        scope=scope, token='offline-bearer-not-real', origin='http://127.0.0.1:9',
                        hindsight_url='http://127.0.0.1:8', api_key_file='/offline-key',
                        profile=profile, configured=configured)
                    effective = JarvisSettings(_env_file=None, **env)
                    # The actual installer rejects at the next prerequisite when
                    # enabled. None supplies no runtime/session/provider objects:
                    # no DB, provider connection, Live State or API call is made.
                    with self.assertRaises(JarvisHTTPError) as failure:
                        await install_responsive_voice(None, None, effective, {'protocol_version':1})
                    self.assertEqual(failure.exception.code, expected_code)
                    self.assertIs(effective.RESPONSIVE_VOICE_ENABLED, profile==NATIVE_MEDIA)

    async def test_native_child_default_tls_trust_and_explicit_overrides(self):
        """Fresh Python SSL contexts only; no socket, handshake or API request."""
        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(), memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(seconds=120))
        configured = JarvisSettings(_env_file=None, OPENAI_API_KEY='offline-openai', GEMINI_API_KEY='offline-gemini',
            OPENAI_REALTIME_MODEL='offline-realtime', SUPABASE_URL='https://storage.invalid',
            SUPABASE_SERVICE_ROLE_KEY='offline-storage', OPENAI_LIVE_ENABLED=False, GEMINI_LIVE_ENABLED=False)
        arguments = dict(dsn='postgresql://fixture.invalid/db?search_path=isolated', scope=scope,
            token='offline-bearer-not-real', origin='http://127.0.0.1:9', hindsight_url='http://127.0.0.1:8',
            api_key_file='/offline-key', configured=configured)
        program = (
            "import json,os,ssl; context=ssl.create_default_context(); "
            "print(json.dumps({'ca_count':len(context.get_ca_certs()),"
            "'verify_mode':int(context.verify_mode),'check_hostname':context.check_hostname,"
            "'SSL_CERT_FILE':os.environ.get('SSL_CERT_FILE'),'SSL_CERT_DIR':os.environ.get('SSL_CERT_DIR')}))"
        )
        with tempfile.TemporaryDirectory() as directory:
            explicit_file = Path(directory)/'caller-ca.pem'
            explicit_file.symlink_to(Path(certifi.where()).resolve())
            cases = (
                ('certifi_fallback', {}, True),
                ('explicit_file', {'SSL_CERT_FILE':str(explicit_file)}, True),
                ('explicit_directory', {'SSL_CERT_DIR':directory}, False),
                ('explicit_file_and_directory', {'SSL_CERT_FILE':str(explicit_file), 'SSL_CERT_DIR':directory}, True),
            )
            for name, supplied, expect_roots in cases:
                with self.subTest(case=name), patch.dict(os.environ, {}, clear=False):
                    os.environ.pop('SSL_CERT_FILE', None)
                    os.environ.pop('SSL_CERT_DIR', None)
                    os.environ.update(supplied)
                    effective = child_environment(**arguments, profile=NATIVE_MEDIA)
                    expected_file = supplied.get('SSL_CERT_FILE') if supplied else str(Path(certifi.where()).resolve())
                    self.assertEqual(effective.get('SSL_CERT_FILE'), expected_file)
                    self.assertEqual(effective.get('SSL_CERT_DIR'), supplied.get('SSL_CERT_DIR'))
                    if not supplied:
                        default = child_environment(**arguments, profile=NO_MEDIA)
                        self.assertNotIn('SSL_CERT_FILE', default)
                        self.assertNotIn('SSL_CERT_DIR', default)
                    child = await asyncio.create_subprocess_exec(sys.executable, '-c', program,
                        cwd=directory, env=effective, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                    try:
                        stdout, stderr = await asyncio.wait_for(child.communicate(), 5)
                        self.assertEqual(child.returncode, 0, stderr.decode(errors='replace'))
                        observed = json.loads(stdout)
                        self.assertEqual(observed['SSL_CERT_FILE'], expected_file)
                        self.assertEqual(observed['SSL_CERT_DIR'], supplied.get('SSL_CERT_DIR'))
                        self.assertEqual(observed['verify_mode'], ssl.CERT_REQUIRED)
                        self.assertIs(observed['check_hostname'], True)
                        if expect_roots:
                            self.assertGreater(observed['ca_count'], 0)
                        print(json.dumps({'native_child_tls_case':name, 'ca_count':observed['ca_count'],
                            'certificate_verification_required':True, 'hostname_verification':True,
                            'explicit_settings_preserved':bool(supplied), 'network_calls':0}))
                    finally:
                        if child.returncode is None:
                            child.kill()
                            await child.wait()

    async def test_child_log_drains_beyond_pipe_and_retention_capacity_during_control(self):
        """Real local process/pipes only; no app, DB, provider or service startup."""
        program = r'''
import json, os, sys
for stream in (sys.stdout.buffer, sys.stderr.buffer):
    stream.write(b'x'*(512*1024)+b'\n')
    stream.flush()
print('FRONTEND_HOST_READY '+json.dumps({'pid':os.getpid()}), flush=True)
if sys.stdin.readline()!='PING\n':
    raise SystemExit(3)
for stream in (sys.stdout.buffer, sys.stderr.buffer):
    stream.write(b'y'*(2*1024*1024)+b'\n')
    stream.flush()
ack_staging=sys.argv[1]+'.staging'
fd=os.open(ack_staging,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as ack:
    json.dump({'pid':os.getpid(),'control_request':'PING','handled':True},ack)
os.replace(ack_staging,sys.argv[1])
if sys.stdin.readline()!='STOP\n':
    raise SystemExit(4)
print('FRONTEND_HOST_EXIT '+json.dumps({'status':'passed'}),flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            log_path, ack_path = Path(directory)/'private-child.log', Path(directory)/'control-ack.json'
            log = os.fdopen(os.open(log_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb')
            child = drain = None
            try:
                child = await asyncio.create_subprocess_exec(sys.executable, '-u', '-c', program, str(ack_path),
                    cwd=directory, env={'PYTHONUNBUFFERED':'1'}, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                drain = ChildOutputDrain(child, log, byte_limit=32768)
                notice = await drain.wait_ready(timeout=5)
                self.assertEqual(notice['pid'], child.pid)
                child.stdin.write(b'PING\n')
                await child.stdin.drain()
                async with asyncio.timeout(5):
                    while not ack_path.exists():
                        await asyncio.sleep(.01)
                # The ACK follows multi-megabyte writes on BOTH pipes. It cannot
                # be reached if retention truncation stops either pipe reader.
                ack = json.loads(ack_path.read_text())
                self.assertEqual(ack, {'pid':child.pid, 'control_request':'PING', 'handled':True})
                self.assertIsNone(child.returncode)
                child.stdin.write(b'STOP\n')
                await child.stdin.drain()
                child.stdin.close()
                await asyncio.wait_for(child.wait(), 5)
                result = await drain.wait_closed(timeout=5)
                self.assertEqual(child.returncode, 0)
                self.assertTrue(result['complete'])
                self.assertEqual(result['errors'], [])
                self.assertTrue(result['truncated'])
                self.assertEqual(result['exit_notice_status'], 'passed')
                self.assertGreater(sum(result['bytes_read'].values()), 5*1024*1024)
                self.assertGreater(result['discarded_bytes'], 4*1024*1024)
                self.assertLessEqual(log_path.stat().st_size, 32768)
                self.assertEqual(log_path.stat().st_size, result['retained_log_bytes'])
                self.assertEqual(log_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(log_path.read_bytes().count(ChildOutputDrain.TRUNCATION_MARKER), 1)
                print(json.dumps({'local_pipe_regression':result, 'control_request_handled':True,
                                  'child_returncode':child.returncode}, default=str))
            finally:
                if child is not None and child.returncode is None:
                    child.kill()
                    await child.wait()
                if drain is not None:
                    await drain.wait_closed(timeout=5)
                log.close()

    async def test_child_log_reports_eof_before_readiness_without_hanging(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory)/'private-child.log'
            log = os.fdopen(os.open(log_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb')
            child = drain = None
            try:
                child = await asyncio.create_subprocess_exec(sys.executable, '-u', '-c', "print('no readiness notice')",
                    cwd=directory, env={'PYTHONUNBUFFERED':'1'}, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE)
                drain = ChildOutputDrain(child, log)
                with self.assertRaisesRegex(QualificationFailure, 'child_stdout_ended_before_ready'):
                    await drain.wait_ready(timeout=5)
                await asyncio.wait_for(child.wait(), 5)
                result = await drain.wait_closed(timeout=5)
                self.assertTrue(result['complete'])
                self.assertFalse(result['truncated'])
                self.assertEqual(result['errors'], [])
            finally:
                if child is not None and child.returncode is None:
                    child.kill()
                    await child.wait()
                if drain is not None:
                    await drain.wait_closed(timeout=5)
                log.close()

    async def test_storage_exact_manifest_orphan_audit_and_failure_preservation(self):
        for retain_on_delete in (False, True):
            with self.subTest(retain_on_delete=retain_on_delete):
                run = uuid4().hex
                owner = f'direct-image-eval-second-brain-{run}-a'
                prefix = f'owners/{owner}/sessions/'
                stem = prefix+str(uuid4())+'/frames/'+str(uuid4())+'/'
                keys = [stem+'000000000001.jpg', stem+'000000000002.jpg']
                outside = 'owners/unrelated/sessions/personal.jpg'
                objects, clients, deletions, persisted = {*keys, outside}, [], [], []
                resources = {'run_id':run, 'owner_id':owner,
                    'storage':{'prefix':prefix, 'bucket':'fixture', 'origin':'https://storage.invalid'}}
                def handle(request):
                    payload = json.loads(request.content)
                    if request.method=='POST':
                        folder = payload['prefix']+'/'
                        children = {}
                        for key in objects:
                            if key.startswith(folder):
                                tail = key[len(folder):]
                                name = tail.split('/')[0]
                                children[name] = {'name':name, 'id':None if '/' in tail else 'object-id'}
                        return httpx.Response(200, json=[children[name] for name in sorted(children)])
                    self.assertEqual(request.method, 'DELETE')
                    self.assertTrue(persisted and persisted[-1]['storage_cleanup']['status']=='manifested_before_delete')
                    self.assertEqual(set(payload['prefixes']), set(keys))
                    deletions.extend(payload['prefixes'])
                    if not retain_on_delete:
                        objects.difference_update(payload['prefixes'])
                    return httpx.Response(200, json=[])
                def factory(config):
                    client = httpx.AsyncClient(base_url='https://storage.invalid', transport=httpx.MockTransport(handle))
                    clients.append(client)
                    return client
                def persist():
                    persisted.append(json.loads(json.dumps(resources)))
                config = SupabaseStorageConfig('https://storage.invalid', 'offline', 'fixture', 'unused')
                call = cleanup_owned_storage(resources, [{'bucket':'fixture','key':keys[0]}], persist,
                    config=config, client_factory=factory)
                if retain_on_delete:
                    with self.assertRaisesRegex(QualificationFailure, 'independent_storage_absence_audit_failed'):
                        await call
                    self.assertEqual(resources['storage_cleanup']['status'], 'incomplete')
                    self.assertEqual(set(resources['storage_cleanup']['remaining_keys']), set(keys))
                else:
                    result = await call
                    self.assertEqual(result['status'], 'passed')
                    self.assertEqual(result['prefix_only_key_count'], 1)
                self.assertEqual(set(deletions), set(keys))
                self.assertIn(outside, objects)
                self.assertEqual(len(clients), 2)
                self.assertIsNot(clients[0], clients[1])

    async def test_storage_refuses_foreign_sql_identity_before_network(self):
        run = uuid4().hex
        owner = f'direct-image-eval-second-brain-{run}-a'
        resources = {'run_id':run, 'owner_id':owner,
            'storage':{'prefix':f'owners/{owner}/sessions/', 'bucket':'fixture'}}
        with self.assertRaisesRegex(QualificationFailure, 'storage_key_outside_owned_prefix'):
            await cleanup_owned_storage(resources, [{'bucket':'fixture','key':'owners/personal/sessions/x'}],
                lambda: None, config=object(), client_factory=lambda _: self.fail('network must not start'))


REQUIRED = ('JARVIS_SECOND_BRAIN_TEST_DB_URL', 'JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR',
            'JARVIS_FRONTEND_HOST_HINDSIGHT_URL', 'JARVIS_FRONTEND_HOST_HINDSIGHT_API_KEY_FILE')


@unittest.skipUnless(all(os.getenv(name) for name in REQUIRED), 'Explicit SQL/native-bank qualification scope required')
class FrontendReplayHostSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_child_http_ws_and_canonical_sql_binding(self):
        report = await qualify_frontend_host(dsn=os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL'],
            directory=Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']),
            hindsight_url=os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_URL'],
            api_key_file=Path(os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_API_KEY_FILE']))
        print(json.dumps({'status':report['status'], 'report':report['report_path'],
            'checks':report['checks'], 'cleanup':report['cleanup'], 'fatal':report.get('fatal')}, default=str))
        self.assertEqual(report['status'], 'passed', json.dumps(report.get('fatal')))
        self.assertFalse(report['voice_pass'])
        self.assertFalse(report['provider_calls'])
        self.assertFalse(report['media_calls'])
        self.assertFalse(report['public_migration'])
        self.assertEqual(report['cleanup']['sql']['remaining_schemas'], 0)
        self.assertEqual(report['cleanup']['sql']['remaining_public_owner_rows'], 0)
        self.assertTrue(report['cleanup']['native_bank_absent'])
        self.assertTrue(report['cleanup']['child_exited'])
        self.assertEqual(report['cleanup']['child_returncode'], 0)
        self.assertTrue(report['child']['normal_create_app'])
        self.assertTrue(report['child']['normal_lifespan'])
        self.assertTrue(report['child']['normal_shutdown'])
        self.assertEqual(report['child']['schema_relations']['jarvis_background_tasks'], 'jarvis_background_tasks')
        self.assertIsNone(report['child']['schema_relations']['jarvis_tasks'])
        self.assertIn('jarvis_background_tasks', report['child']['empty_data_counts'])
        self.assertTrue(report['child']['runtime_closed'])
        self.assertFalse(report['child']['host_ready_after_shutdown'])
        self.assertEqual(report['child']['knowledge_http_requests_after_probes'], 0)
        self.assertTrue(all(check['status']=='passed' for check in report['checks']))

    async def test_prepared_scope_serve_and_independent_cleanup(self):
        dsn = os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL']
        prepared = await qualify_frontend_host(dsn=dsn,
            directory=Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR']),
            hindsight_url=os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_URL'],
            api_key_file=Path(os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_API_KEY_FILE']),
            prepare_only=True, lifetime_seconds=120)
        self.assertEqual(prepared['status'], 'prepared', json.dumps(prepared.get('fatal')))
        stop = asyncio.Event()
        served = cleanup = None
        async def probe(info):
            token = secret_file(info['bearer_token_file'])
            self.assertNotIn(token, json.dumps(info))
            headers = {'Authorization':'Bearer '+token, 'X-Jarvis-Evaluation-Run':info['run_id']}
            async with httpx.AsyncClient(base_url=info['backend_base_url'], timeout=10, trust_env=False) as http:
                scope = await http.get('/eval/replay/'+info['run_id']+'/scope', headers=headers)
                self.assertEqual(scope.status_code, 200)
                self.assertEqual(scope.json()['owner_id'], info['owner_id'])
                response = await http.post('/sessions/start', headers=headers, json={'resume_active':False})
                self.assertEqual(response.status_code, 200)
            save_json(Path(info['control_handle']['path']), {'run_id':info['run_id'], 'action':'stop'})
        try:
            served = await serve_prepared_host(manifest=prepared['resource_manifest'], dsn=dsn,
                stop_event=stop, ready_callback=probe, auto_cleanup=False)
            self.assertEqual(served['status'], 'passed', json.dumps(served.get('fatal')))
            self.assertTrue(served['child']['normal_shutdown'])
            self.assertFalse(served['child']['storage_configured'])
            self.assertEqual(served['stop_reason'], 'owned_control_file')
        finally:
            cleanup = await cleanup_prepared_host(manifest=prepared['resource_manifest'], dsn=dsn)
            print(json.dumps({'prepared_report':prepared['report_path'],
                'serve_report':served['report_path'] if served else None, 'cleanup':cleanup}, default=str))
        self.assertEqual(cleanup['status'], 'passed', json.dumps(cleanup.get('fatal')))
        self.assertEqual(cleanup['cleanup']['sql']['remaining_schemas'], 0)
        self.assertEqual(cleanup['cleanup']['sql']['remaining_public_owner_rows'], 0)
        self.assertTrue(cleanup['cleanup']['native_bank_absent'])
        self.assertTrue(cleanup['cleanup']['independent_client_audit'])
        self.assertTrue(cleanup['cleanup']['bearer_token_removed'])

    async def test_native_profile_lease_storage_sql_and_orphan_cleanup(self):
        """Actual normal child and storage; no WS, model call, or media inference."""
        dsn = os.environ['JARVIS_SECOND_BRAIN_TEST_DB_URL']
        directory = Path(os.environ['JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR'])
        prepared = await qualify_frontend_host(dsn=dsn, directory=directory,
            hindsight_url=os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_URL'],
            api_key_file=Path(os.environ['JARVIS_FRONTEND_HOST_HINDSIGHT_API_KEY_FILE']),
            prepare_only=True, lifetime_seconds=60, profile=NATIVE_MEDIA)
        self.assertEqual(prepared['status'], 'prepared', json.dumps(prepared.get('fatal')))
        manifest = Path(prepared['resource_manifest'])
        resources = json.loads(manifest.read_text())
        config = configured_storage(resources)
        sentinel = json.loads(json.dumps(resources))
        sentinel['owner_id'] = resources['owner_id'][:-1]+'b'
        sentinel['storage']['prefix'] = f"owners/{sentinel['owner_id']}/sessions/"
        sentinel_path = directory/f"frontend-host-storage-sentinel-{resources['run_id']}.json"
        sentinel_key = sentinel['storage']['prefix']+str(uuid4())+'/frames/'+str(uuid4())+'/000000000001.jpg'
        # Persist ownership before any upload, including a partially failed POST.
        sentinel['storage_cleanup'] = {'objects':[{'bucket':config.visual_frame_bucket, 'key':sentinel_key}]}
        save_json(sentinel_path, sentinel)
        served = None
        diagnostic_path = directory/f"frontend-host-relation-diagnostic-{resources['run_id']}.json"
        async def probe(info):
            token = secret_file(info['bearer_token_file'])
            self.assertNotIn(token, json.dumps(info))
            self.assertEqual(info['capability_profile'], NATIVE_MEDIA)
            headers = {'authorization':'Bearer '+token, 'x-jarvis-evaluation-run':info['run_id']}
            async with httpx.AsyncClient(base_url=info['backend_base_url'], timeout=10, trust_env=False) as http:
                self.assertEqual((await http.get('/eval/replay/'+info['run_id']+'/scope', headers=headers)).status_code, 200)
                self.assertEqual((await http.get('/eval/replay/'+info['run_id']+'/scope')).status_code, 403)
                response = await http.post('/sessions/start', headers=headers, json={'resume_active':False})
                self.assertEqual(response.status_code, 200)
                session = UUID(response.json()['session']['id'])
            visual_session = uuid4()
            uploader = VisualFrameStorageClient(config)
            keys = [uploader.build_storage_path(owner_id=info['owner_id'], session_id=session,
                visual_session_id=visual_session, frame_sequence=index) for index in (1, 2)]
            upload_manifest = directory/f"frontend-host-synthetic-uploads-{info['run_id']}.json"
            save_json(upload_manifest, {'run_id':info['run_id'], 'synthetic_not_image_quality':True,
                'bucket':config.visual_frame_bucket, 'keys':keys, 'sentinel_key':sentinel_key})
            try:
                for key in [*keys, sentinel_key]:
                    await uploader.upload_frame(path=key, image_bytes=b'bounded synthetic cleanup fixture; not image content',
                                                content_type='image/jpeg')
            finally:
                await uploader.close()
            connection = await asyncpg.connect(scoped_database_url(dsn, resources['database_schema']), timeout=10)
            try:
                self.assertEqual(await connection.fetchval('select count(*) from jarvis_sessions where owner_id=$1',
                    info['owner_id']), 1)
                diagnostic = {}
                with self.assertRaises(asyncpg.UndefinedTableError):
                    await count_owned_rows(connection, info['owner_id'], diagnostic, ('jarvis_tasks',))
                save_json(diagnostic_path, diagnostic)
                self.assertEqual(diagnostic['private_sql_error']['sqlstate'], '42P01')
                self.assertEqual(diagnostic['private_sql_error']['relation'], 'jarvis_tasks')
                await connection.execute('insert into visual_frames(owner_id,session_id,visual_session_id,frame_sequence,'
                    'source,captured_at,content_type,width,height,storage_bucket,storage_path,raw_image_expires_at) '
                    "values($1,$2,$3,1,'iphone_camera',clock_timestamp(),'image/jpeg',1,1,$4,$5,clock_timestamp()+interval '1 hour')",
                    info['owner_id'], session, visual_session, config.visual_frame_bucket, keys[0])
            finally:
                await connection.close()
        try:
            served = await serve_prepared_host(manifest=manifest, dsn=dsn, ready_callback=probe, auto_cleanup=True)
            self.assertEqual(served['status'], 'passed', json.dumps(served.get('fatal')))
            self.assertEqual(served['stop_reason'], 'lease_expired')
            child = served['child']
            self.assertTrue(child['normal_create_app'] and child['normal_lifespan'] and child['normal_shutdown'])
            self.assertTrue(child['storage_configured'])
            self.assertTrue(child['provider_configuration']['openai_key_present'])
            self.assertTrue(child['provider_configuration']['gemini_key_present'])
            self.assertTrue(child['provider_configuration']['responsive_voice_enabled'])
            self.assertFalse(child['provider_configuration']['provider_execution_qualified'])
            self.assertEqual(child['websocket_trace'], [])
            self.assertEqual(child['knowledge_http_requests_after_probes'], 0)
            self.assertEqual(child['empty_data_counts']['visual_frames'], 1)
            cleanup = served['cleanup']['cleanup']
            self.assertEqual(cleanup['storage']['sql_key_count'], 1)
            self.assertEqual(cleanup['storage']['prefix_only_key_count'], 1)
            self.assertEqual(len(cleanup['storage']['objects']), 2)
            self.assertEqual(cleanup['storage']['remaining_keys'], [])
            self.assertTrue(cleanup['storage']['independent_client_audit'])
            self.assertTrue(cleanup['native_bank_absent'])
            self.assertEqual(cleanup['sql']['remaining_schemas'], 0)
            self.assertEqual(cleanup['sql']['remaining_public_owner_rows'], 0)
            self.assertTrue(cleanup['bearer_token_removed'])
            async with storage_http(config) as audit:
                self.assertEqual(await list_owned_storage(audit, sentinel), [sentinel_key])
        finally:
            if served is None or served.get('cleanup', {}).get('status')!='passed':
                recovery = await cleanup_prepared_host(manifest=manifest, dsn=dsn)
                print(json.dumps({'native_cleanup_recovery':recovery}, default=str))
            sentinel_cleanup = await cleanup_owned_storage(sentinel, [], lambda: save_json(sentinel_path, sentinel), config=config)
            print(json.dumps({'prepared_report':prepared['report_path'],
                'serve_report':served['report_path'] if served else None,
                'sentinel_cleanup_manifest':str(sentinel_path), 'sentinel_cleanup':sentinel_cleanup,
                'private_relation_diagnostic':str(diagnostic_path)}, default=str))


if __name__=='__main__':
    unittest.main()
