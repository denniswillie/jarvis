"""Real prepared staging and runner binding, with offline host/process ports.

No native, provider or SQL acceptance. Reproduces e9af's string-vs-bool host
receipt mismatch without reading credentials or reusing its spent authorization.
"""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.frontend_host_qualification import (
    NATIVE_MEDIA, ReplayHostScope, child_environment, native_provider_configuration,
)
from test_native_replay_run import Fixture


class NativeLivePreflightTests(unittest.IsolatedAsyncioTestCase):
    def settings_from_child_environment(self, voice_route='live'):
        class Configured:
            JARVIS_REALTIME_MAX_SESSION_SECONDS = 600

            def model_dump(self, **kwargs):
                return {'OPENAI_API_KEY':'offline-key', 'GEMINI_API_KEY':'offline-key',
                    'OPENAI_LIVE_ENABLED':False, 'OPENAI_LIVE_MODEL':'unselected-live-model',
                    'OPENAI_LIVE_VOICE':'unselected-live-voice',
                    'OPENAI_REALTIME_MODEL':'unchanged-realtime-model', 'OPENAI_REALTIME_VOICE':'cedar'}

        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(), memory_bank_id='jarvis_'+uuid4().hex,
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        # Explicit fake configuration; neither .env nor the process's provider
        # secrets enter this regression.
        with patch.dict(os.environ, {}, clear=True):
            environment = child_environment(dsn='offline-dsn', scope=scope,
                token='offline-token', origin='http://127.0.0.1:1',
                hindsight_url='http://127.0.0.1:9', api_key_file=Path('/offline/key'),
                profile=NATIVE_MEDIA, configured=Configured(), voice_route=voice_route,
                live_model='gpt-live-1' if voice_route=='live' else None,
                live_voice='marin' if voice_route=='live' else None)
            return JarvisSettings(_env_file=None, **environment)

    def prepared_live_fixture(self, root):
        f = Fixture(root)
        f.prepared('baseline', 0, duration=45)
        original = f.options.input_manifest
        original_bytes = original.read_bytes()
        manifest = runner.read_json(original)
        manifest.update(voice_route='live', post_media_capture_seconds=20)
        sibling = original.with_name('manifest-live-offline.json')
        runner.save(sibling, manifest)
        f.options = replace(f.options, input_manifest=sibling, voice_route='live',
                            live_model='gpt-live-1', live_voice='marin', lease_seconds=330)
        f.authorize_prepared()
        authorization = runner.read_json(f.options.risk_authorization)
        authorization.update(voice_route='live', live_model='gpt-live-1', live_voice='marin')
        runner.save(f.options.risk_authorization, authorization)
        return f, original, original_bytes, manifest

    def bind_real_host_receipt(self, f, *, override=None):
        settings = self.settings_from_child_environment()
        # This exact shape was missed by the older boolean-only fake host.
        self.assertEqual(settings.OPENAI_LIVE_ENABLED, 'true')
        self.assertIs(type(settings.OPENAI_LIVE_ENABLED), str)
        original_serve = f.serve

        async def serve(manifest, stop, ready):
            async def bound_ready(info):
                configuration = native_provider_configuration(settings, voice_route='live')
                if override:
                    configuration.update(override)
                f.host_configuration = configuration
                await ready({**info, 'provider_configuration':configuration})

            result = await original_serve(manifest, stop, bound_ready)
            if not f.launches:
                # The existing fixture's synthetic route event is only relevant
                # after its fake process launch, never a real provider request.
                result['child']['websocket_trace'] = []
            return result

        f.serve = serve

    async def test_env_string_enablement_reaches_actual_live_stage_and_bound_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            f, original, original_bytes, source = self.prepared_live_fixture(Path(directory))
            self.bind_real_host_receipt(f)
            with patch.object(runner, 'stage_prepared_media', wraps=runner.stage_prepared_media) as stage:
                result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(result['status'], 'lifecycle_complete', result.get('fatal'))
            self.assertEqual((f.prepares, f.launches, f.cleanups, f.worker_starts), (1, 1, 1, 0))
            self.assertIs(f.host_configuration['live_enabled'], True)
            self.assertFalse(f.host_configuration['provider_execution_qualified'])
            self.assertEqual(original.read_bytes(), original_bytes)
            self.assertEqual(f.options.input_manifest.parent.resolve(), f.options.media_root.resolve())
            self.assertEqual(f.plan['voice_route'], 'live')
            self.assertEqual(f.plan['run_id'], f.run)
            self.assertEqual(f.plan['owner_id'], f.resource['owner_id'])
            self.assertEqual(f.plan['area_id'], f.resource['area_id'])
            self.assertEqual(f.plan['memory_bank_id'], f.resource['memory_bank_id'])
            self.assertEqual(f.plan['utterances'], source['utterances'])
            self.assertEqual([(x['at_seconds'], x['sha256']) for x in f.plan['frames']],
                             [(x['at_seconds'], x['sha256']) for x in source['frames']])
            staged = runner.read_json(f.options.artifact_dir/'inputs/media-plan.json')
            self.assertEqual(staged['voice_route'], 'live')
            self.assertEqual((staged['duration_seconds'], staged['post_media_capture_seconds']), (45, 20))
            self.assertNotIn('live_model', f.plan)
            self.assertNotIn('live_voice', f.plan)
            self.assertNotIn('expectations', f.plan)
            self.assertEqual(result['capture_budget']['capture_wait_seconds'], 95)
            self.assertEqual(result['capture_budget']['lease_seconds'], 330)
            self.assertIn('native_host_ready', [row['event'] for row in result['events']])
            self.assertTrue(result['local_cleanup_complete'])

    async def test_raw_string_host_receipt_reproduces_prelaunch_failure_not_voice_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            f, _, _, _ = self.prepared_live_fixture(Path(directory))
            self.bind_real_host_receipt(f, override={'live_enabled':'true'})
            result = await runner.run_once(f.options, host=f, process=f)
            self.assertEqual(result['fatal']['code'], 'native_live_configuration_receipt_mismatch')
            self.assertEqual((f.prepares, f.launches, f.cleanups), (1, 0, 1))
            self.assertEqual(result['launch_attempts'], 0)
            self.assertEqual(len(result['media']), 4)  # Three actual staged fixtures plus selected WAV.
            self.assertNotIn('native_host_ready', [row['event'] for row in result['events']])
            self.assertEqual(result['provider_evidence']['creation_outcome'], 'not_requested')
            self.assertTrue(result['local_cleanup_complete'])
            identity = result['authorization_id']
            self.assertTrue((f.options.ledger_dir/f'authorization-{identity}.spent').is_file())
            self.assertFalse(f.token.exists())

    async def test_mismatched_host_bindings_still_fail_closed_before_launch(self):
        for override in ({'live_enabled':False}, {'live_enabled':None}, {'live_enabled':'false'},
                         {'selected_model':'wrong-model'}, {'selected_voice':'wrong-voice'},
                         {'voice_route':'realtime'}):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                f, _, _, _ = self.prepared_live_fixture(Path(directory))
                self.bind_real_host_receipt(f, override=override)
                result = await runner.run_once(f.options, host=f, process=f)
                self.assertEqual(result['fatal']['code'], 'native_live_configuration_receipt_mismatch')
                self.assertEqual((f.launches, f.cleanups), (0, 1))
                self.assertTrue(result['local_cleanup_complete'])

    def test_host_receipt_uses_existing_selector_for_enabled_forms(self):
        settings = self.settings_from_child_environment()
        for value in (True, 'true', 'TRUE', '1', 'on', 'yes'):
            with self.subTest(value=value):
                receipt = native_provider_configuration(
                    settings.model_copy(update={'OPENAI_LIVE_ENABLED':value}), voice_route='live')
                self.assertIs(receipt['live_enabled'], True)
                self.assertEqual((receipt['selected_model'], receipt['selected_voice']), ('gpt-live-1', 'marin'))
                self.assertFalse(receipt['provider_execution_qualified'])

    def test_disabled_or_malformed_live_flag_never_becomes_true_by_truthiness(self):
        settings = self.settings_from_child_environment()
        for value in (False, 'false', '0', 'off', 'no', 'not-a-boolean', '', None):
            with self.subTest(value=value), self.assertRaises(JarvisHTTPError):
                native_provider_configuration(
                    settings.model_copy(update={'OPENAI_LIVE_ENABLED':value}), voice_route='live')

    def test_realtime_receipt_and_unselected_live_setting_remain_unchanged(self):
        settings = self.settings_from_child_environment('realtime').model_copy(
            update={'OPENAI_LIVE_ENABLED':'unused-invalid-live-setting'})
        receipt = native_provider_configuration(settings, voice_route='realtime')
        self.assertEqual(receipt['voice_route'], 'realtime')
        self.assertEqual((receipt['selected_model'], receipt['selected_voice']),
                         ('unchanged-realtime-model', 'cedar'))
        self.assertEqual(receipt['live_enabled'], 'unused-invalid-live-setting')
        self.assertFalse(receipt['provider_execution_qualified'])
