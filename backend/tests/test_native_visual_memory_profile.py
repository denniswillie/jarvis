"""Offline normal configuration/registration, not native session advertisement."""
from datetime import UTC, datetime, timedelta
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.reminders.duty_runtime import initialize_responsibilities
from app.runtime.feature_config import Capability, SecondBrainSettings
from app.settings import JarvisSettings
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry
from evals.responsive_voice.frontend_host import ReplayHostScope
from evals.responsive_voice.frontend_host_qualification import (
    NATIVE_MEDIA, NO_MEDIA, NO_WORKLOAD, REASONING_REMINDER, VISUAL_MEMORY,
    VISUAL_MEMORY_REQUIRED_TOOLS, QualificationFailure, child_environment,
    native_host_options, visual_memory_runtime_qualification,
)
from test_episode_check_tools import CaptureStore


def environment(workload=VISUAL_MEMORY, *, scope=None, profile=NATIVE_MEDIA, **options):
    if scope is None:
        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
            database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(),
            memory_bank_id='jarvis_'+uuid4().hex, expires_at=datetime.now(UTC)+timedelta(minutes=5))
    with patch.dict(os.environ, {}, clear=True):
        configured = JarvisSettings(_env_file=None, OPENAI_API_KEY='offline-openai',
            GEMINI_API_KEY='offline-gemini', OPENAI_REALTIME_MODEL='offline-realtime',
            SUPABASE_URL='https://storage.invalid', SUPABASE_SERVICE_ROLE_KEY='offline-storage',
            RESPONSIVE_VOICE_ENABLED=False, JARVIS_REALTIME_MAX_SESSION_SECONDS=30,
            OPENAI_LIVE_ENABLED=False, GEMINI_LIVE_ENABLED=False)
        env = child_environment(dsn='postgresql://fixture.invalid/offline', scope=scope,
            token='offline-bearer', origin='http://127.0.0.1:9',
            hindsight_url='http://127.0.0.1:8', api_key_file='/offline-key',
            profile=profile, workload=workload, configured=configured, **options)
        settings = JarvisSettings(_env_file=None, **env)
        flags = SecondBrainSettings(_env_file=None, **{
            key.removeprefix('SECOND_BRAIN_').lower():value for key,value in env.items()
            if key.startswith('SECOND_BRAIN_')})
    return env, scope, settings, flags


class VisualMemoryConfigurationTests(unittest.TestCase):
    def test_explicit_profile_only_changes_workload_and_responsibilities_from_native_baseline(self):
        baseline, scope, _, baseline_flags = environment(NO_WORKLOAD)
        actual, _, settings, flags = environment(scope=scope)
        self.assertEqual({key for key in actual if actual[key]!=baseline[key]},
            {'JARVIS_FRONTEND_REPLAY_WORKLOAD', 'SECOND_BRAIN_RESPONSIBILITIES_ENABLED'})
        self.assertEqual(actual['SECOND_BRAIN_ENABLED'], 'true')
        self.assertEqual(actual['SECOND_BRAIN_RESPONSIBILITIES_ENABLED'], 'true')
        self.assertEqual(actual['RESPONSIVE_VOICE_ENABLED'], 'true')
        self.assertIs(settings.REMINDER_SCHEDULER_ENABLED, False)
        enabled = flags.resolve(Capability.RESPONSIBILITIES, database_url=settings.SUPABASE_DB_URL)
        self.assertTrue(enabled.enabled and enabled.configured)
        self.assertFalse(baseline_flags.resolve(Capability.RESPONSIBILITIES,
            database_url=settings.SUPABASE_DB_URL).enabled)
        for name in ('FRONTIER', 'ATTENTION_VOICE', 'PUSH', 'MAIL', 'SANDBOX', 'ARTIFACTS'):
            with self.subTest(capability=name):
                self.assertEqual(actual['SECOND_BRAIN_'+name+'_ENABLED'], 'false')
        self.assertNotIn('SECOND_BRAIN_FRONTIER_MODEL', actual)
        self.assertNotIn('JARVIS_FRONTEND_REPLAY_TASK_CONTROL', actual)

    def test_none_nonnative_and_reasoning_reminder_defaults_are_unchanged(self):
        for profile, workload, options in (
            (NO_MEDIA, NO_WORKLOAD, {}), (NATIVE_MEDIA, NO_WORKLOAD, {}),
            (NATIVE_MEDIA, REASONING_REMINDER,
                {'frontier_model':'offline-model', 'frontier_reasoning_effort':'medium'}),
        ):
            with self.subTest(profile=profile, workload=workload):
                env, _, settings, _ = environment(workload, profile=profile, **options)
                self.assertEqual(env['SECOND_BRAIN_RESPONSIBILITIES_ENABLED'], 'false')
                self.assertIs(settings.RESPONSIVE_VOICE_ENABLED, profile==NATIVE_MEDIA)
                integrated = workload==REASONING_REMINDER
                for key in ('SECOND_BRAIN_FRONTIER_ENABLED','SECOND_BRAIN_ATTENTION_VOICE_ENABLED',
                            'REMINDER_SCHEDULER_ENABLED'):
                    self.assertEqual(env[key], 'true' if integrated else 'false')
        self.assertEqual(native_host_options(NATIVE_MEDIA)['workload'], NO_WORKLOAD)

    def test_visual_memory_cannot_select_live_nonnative_or_frontier_configuration(self):
        for changes, code in (
            ({'profile':NO_MEDIA}, 'visual_memory_requires_native_realtime'),
            ({'voice_route':'live', 'live_model':'offline-live', 'live_voice':'marin'},
                'visual_memory_requires_native_realtime'),
            ({'frontier_model':'offline-model', 'frontier_reasoning_effort':'medium'},
                'frontier_configuration_without_workload'),
            ({'live_model':'offline-live'}, 'live_configuration_without_live_route'),
        ):
            with self.subTest(changes=changes), self.assertRaises(QualificationFailure) as failure:
                native_host_options(**{'profile':NATIVE_MEDIA, 'workload':VISUAL_MEMORY, **changes})
            self.assertEqual(failure.exception.code, code)

    def test_parent_flags_and_worker_configuration_are_not_inherited(self):
        with patch.dict(os.environ, {
            'SECOND_BRAIN_FRONTIER_ENABLED':'true', 'SECOND_BRAIN_ATTENTION_VOICE_ENABLED':'true',
            'SECOND_BRAIN_FRONTIER_MODEL':'private-model', 'SECOND_BRAIN_PUSH_ENABLED':'true',
            'SECOND_BRAIN_MAIL_ENABLED':'true', 'SECOND_BRAIN_SANDBOX_ENABLED':'true',
            'SECOND_BRAIN_ARTIFACTS_ENABLED':'true', 'REMINDER_SCHEDULER_ENABLED':'true',
            'JARVIS_FRONTEND_REPLAY_TASK_CONTROL':'private-control',
        }):
            # Build explicitly under the caller environment to exercise the
            # production whitelist, not environment()'s hermetic setup.
            _, scope, settings, _ = environment()
            env = child_environment(dsn='postgresql://fixture.invalid/offline', scope=scope,
                token='offline-bearer', origin='http://127.0.0.1:9',
                hindsight_url='http://127.0.0.1:8', api_key_file='/offline-key',
                profile=NATIVE_MEDIA, workload=VISUAL_MEMORY, configured=settings)
        self.assertNotIn('private-model', repr(env))
        self.assertNotIn('private-control', repr(env))
        for name in ('FRONTIER', 'ATTENTION_VOICE', 'PUSH', 'MAIL', 'SANDBOX', 'ARTIFACTS'):
            self.assertEqual(env['SECOND_BRAIN_'+name+'_ENABLED'], 'false')
        self.assertEqual(env['REMINDER_SCHEDULER_ENABLED'], 'false')


class VisualMemoryRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def normal_runtime(self, *, schema=True, workload=VISUAL_MEMORY):
        _, scope, settings, flags = environment(workload)
        pool = CaptureStore()
        pool.schema = schema
        services = SimpleNamespace(settings=settings, db_pool=pool, tool_modules=())
        # Real feature resolution, initializer, capability probe, registration
        # and lifecycle. Only the SQL port is the existing offline fixture.
        responsibility = await initialize_responsibilities(services, flags)
        if responsibility is not None:
            self.addAsyncCleanup(responsibility.close)
        return services, flags, scope, responsibility

    async def test_effective_environment_initializes_normal_owner_and_registry_without_advertisement_claim(self):
        runtime, flags, scope, responsibility = await self.normal_runtime()
        self.assertIsNotNone(responsibility)
        self.assertIsNone(runtime.responsibility_issue)
        self.assertEqual(responsibility.episode_checks.capability.status, 'available')
        receipt = visual_memory_runtime_qualification(runtime, flags=flags, owner_id=scope.owner_id)
        self.assertEqual(receipt['registered_module'], 'episode_checks')
        self.assertTrue(receipt['owner_and_pool_match'])
        self.assertEqual(receipt['required_session_tool_names'], list(VISUAL_MEMORY_REQUIRED_TOOLS))
        self.assertEqual(receipt['session_advertisement_status'], 'unobserved')
        self.assertIsNone(receipt['advertised_session_tool_names'])
        self.assertIs(receipt['session_advertisement_qualified'], False)
        self.assertIs(receipt['frontier_worker_required'], False)
        registry = JarvisToolRegistry()
        context = ToolBuildContext(settings=runtime.settings, owner_id=scope.owner_id, session_id=uuid4())
        for module in runtime.tool_modules:
            registry.register_many(module.build_tools(context))
        self.assertTrue(set(VISUAL_MEMORY_REQUIRED_TOOLS).issubset({tool.name for tool in registry.list()}))
        # This is local registration for an offline scope, not a model request.
        self.assertFalse(runtime.db_pool.rules)
        self.assertFalse(runtime.db_pool.reminders)
        self.assertFalse(receipt['session_advertisement_qualified'])
        await responsibility.close()
        self.assertTrue(responsibility.task.done())

    async def test_default_profile_does_not_initialize_owner_or_qualify(self):
        runtime, flags, scope, responsibility = await self.normal_runtime(workload=NO_WORKLOAD)
        self.assertIsNone(responsibility)
        self.assertEqual(runtime.tool_modules, ())
        with self.assertRaises(QualificationFailure) as failure:
            visual_memory_runtime_qualification(runtime, flags=flags, owner_id=scope.owner_id)
        self.assertEqual(failure.exception.code, 'native_visual_memory_configuration_unavailable')

    async def test_unavailable_episode_schema_does_not_fallback_to_registered_scheduled_duties(self):
        runtime, flags, scope, responsibility = await self.normal_runtime(schema=False)
        self.assertIsNotNone(responsibility)
        self.assertEqual([module.name for module in runtime.tool_modules], ['responsibilities','episode_checks'])
        with self.assertRaises(QualificationFailure) as failure:
            visual_memory_runtime_qualification(runtime, flags=flags, owner_id=scope.owner_id)
        self.assertEqual(failure.exception.code, 'native_visual_memory_episode_owner_unavailable')

    async def test_missing_duplicate_module_and_foreign_pool_fail_closed(self):
        runtime, flags, scope, _ = await self.normal_runtime()
        for updates, code in (
            ({'tool_modules':()}, 'native_visual_memory_episode_module_missing_or_ambiguous'),
            ({'tool_modules':runtime.tool_modules*2}, 'native_visual_memory_episode_module_missing_or_ambiguous'),
            ({'db_pool':CaptureStore()}, 'native_visual_memory_owner_pool_mismatch'),
        ):
            altered = SimpleNamespace(**{**vars(runtime), **updates})
            with self.subTest(code=code), self.assertRaises(QualificationFailure) as failure:
                visual_memory_runtime_qualification(altered, flags=flags, owner_id=scope.owner_id)
            self.assertEqual(failure.exception.code, code)

    async def test_wrong_owner_disabled_master_or_delivery_configuration_rejected_without_secret_details(self):
        runtime, flags, scope, _ = await self.normal_runtime()
        with self.assertRaises(QualificationFailure) as failure:
            visual_memory_runtime_qualification(runtime, flags=flags, owner_id='not-the-fixture-owner')
        self.assertEqual(failure.exception.code, 'native_visual_memory_owner_scope_mismatch')
        for changes, code in (
            ({'enabled':'false'}, 'native_visual_memory_configuration_unavailable'),
            ({'responsibilities_enabled':'false'}, 'native_visual_memory_configuration_unavailable'),
            ({'frontier_enabled':'true'}, 'native_visual_memory_unexpected_delivery_or_worker'),
            ({'push_enabled':'true', 'push_private_key':'PRIVATE-DO-NOT-LOG'},
                'native_visual_memory_unexpected_delivery_or_worker'),
        ):
            with self.subTest(changes=tuple(changes)), self.assertRaises(QualificationFailure) as failure:
                visual_memory_runtime_qualification(runtime, flags=flags.model_copy(update=changes),
                    owner_id=scope.owner_id)
            self.assertEqual(failure.exception.code, code)
            self.assertNotIn('PRIVATE-DO-NOT-LOG', str(failure.exception))
        runtime.settings = runtime.settings.model_copy(update={'REMINDER_SCHEDULER_ENABLED':True})
        with self.assertRaises(QualificationFailure) as failure:
            visual_memory_runtime_qualification(runtime, flags=flags, owner_id=scope.owner_id)
        self.assertEqual(failure.exception.code, 'native_visual_memory_unexpected_delivery_or_worker')
