"""Configuration, ownership and shutdown contracts; no external services."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import SecretStr, ValidationError

from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime, initialize_knowledge_runtime
from app.knowledge.sources import SourceRepository
from app.realtime.router import _create_context_capture
from app.runtime.feature_config import Capability, KnowledgeConfiguration, SecondBrainSettings, TaskLimits
from app.runtime.services import JarvisRuntimeServices
from app.tasks.repository import TaskRepository


def configuration(**changes):
    return KnowledgeConfiguration(**(dict(
        base_url="http://127.0.0.1:19999", api_key=SecretStr("fixture-private-key"),
        area_id=uuid4(), poll_seconds=0.01, shutdown_seconds=0.01,
    ) | changes))


class KnowledgeConfigurationTests(unittest.TestCase):
    def settings(self, **changes):
        return SecondBrainSettings(_env_file=None, **(dict(
            enabled="true", knowledge_enabled="true", hindsight_url="http://127.0.0.1:19999",
            hindsight_api_key="fixture-private-key", knowledge_area_id=str(uuid4()),
        ) | changes))

    def test_disabled_invalid_optional_configuration_is_inert(self):
        settings = self.settings(enabled="false", hindsight_url="private malformed URL",
                                 knowledge_area_id="bad", knowledge_poll_seconds="bad")
        result = settings.resolve(Capability.KNOWLEDGE)
        self.assertFalse(result.enabled)
        self.assertIsNone(result.issue)

    def test_selected_configuration_is_redacted_and_not_native_readiness(self):
        result = self.settings().resolve(Capability.KNOWLEDGE, database_url="configured")
        self.assertTrue(result.configured)
        self.assertIsInstance(result.configuration, KnowledgeConfiguration)
        self.assertNotIn("fixture-private-key", repr(result))
        self.assertNotIn("fixture-private-key", result.configuration.model_dump_json())
        self.assertFalse(hasattr(result, "native_ready"))

    def test_missing_key_and_invalid_fields_are_sanitized(self):
        missing = self.settings(hindsight_api_key="replace_me").resolve(
            Capability.KNOWLEDGE, database_url="configured",
        )
        self.assertEqual(missing.issue, "memory_api_key_not_configured")
        bad = self.settings(knowledge_area_id="private-invalid-area").resolve(
            Capability.KNOWLEDGE, database_url="configured",
        )
        self.assertEqual(bad.invalid_fields, ("area_id",))
        self.assertNotIn("private-invalid-area", repr(bad))

    def test_service_url_cannot_carry_credentials_or_use_remote_plaintext(self):
        for url in (
            "http://remote.example", "https://user:secret@example.com", "https://example.com?key=secret",
            "https://example.com/path", "https://example.com#secret", "http://localhost:0",
        ):
            with self.subTest(url=url), self.assertRaises(ValidationError):
                configuration(base_url=url)

    def test_native_admission_limit_is_bounded_and_resolved_lazily(self):
        result = self.settings(knowledge_native_inflight_limit="3").resolve(
            Capability.KNOWLEDGE, database_url="configured",
        )
        self.assertEqual(result.configuration.native_inflight_limit, 3)
        for value in ("0", "65", "not-a-number"):
            with self.subTest(value=value):
                result = self.settings(knowledge_native_inflight_limit=value).resolve(
                    Capability.KNOWLEDGE, database_url="configured",
                )
                self.assertFalse(result.configured)
                self.assertEqual(result.invalid_fields, ("native_inflight_limit",))
        self.assertFalse(self.settings(enabled="false", knowledge_native_inflight_limit="invalid").resolve(
            Capability.KNOWLEDGE,
        ).enabled)

    def test_request_step_and_lease_bounds_are_consistent(self):
        for changes in (
            {"http_timeout_seconds": 30}, {"worker_deadline_seconds": 60},
            {"poll_seconds": float("nan")}, {"shutdown_seconds": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                configuration(**changes)


class KnowledgeRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def runtime(self):
        pool = object()
        client = SimpleNamespace(close=AsyncMock(), diagnostics=lambda: {"provider_token_usage": None})
        runtime = KnowledgeRuntime(
            owner_id="fixture-owner", sources=SourceRepository(pool),
            tasks=TaskRepository(pool, TaskLimits()), client=client, configuration=configuration(),
        )
        self.addAsyncCleanup(runtime.close)
        self.assertEqual(runtime.outbox.max_in_flight, runtime.configuration.native_inflight_limit)
        return runtime

    async def test_explicit_memory_backend_is_owner_bound_and_cannot_outlive_runtime(self):
        from app.memory.retrieval.query import MemoryRetrievalQuery
        runtime = self.runtime()
        backend = runtime.memory_read_backend_for(runtime.owner_id)
        self.assertIs(backend.sources, runtime.sources)
        with self.assertRaises(SourceError):
            runtime.memory_read_backend_for("foreign-owner")
        await runtime.close()
        with self.assertRaises(SourceError) as caught:
            await backend.search(owner_id=runtime.owner_id, request=MemoryRetrievalQuery(query="fixture"))
        self.assertEqual(caught.exception.code, "knowledge_intake_closed")

    async def test_cooperative_shutdown_is_idempotent_and_closes_intake(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        async def run(owner, worker, stop, **kwargs):
            entered.set()
            await stop.wait()
        runtime.worker = SimpleNamespace(run=run)
        runtime.start()
        await entered.wait()
        self.assertIs(runtime.context_for("fixture-owner"), runtime.context_repository)
        await runtime.close()
        await runtime.close()
        self.assertTrue(runtime.task.done())
        self.assertTrue(runtime.retention_task.done())
        self.assertIs(runtime.retention.sources, runtime.sources)
        runtime.client.close.assert_awaited_once()
        with self.assertRaises(SourceError):
            runtime.context_for("fixture-owner")
        with self.assertRaises(RuntimeError):
            runtime.start()

    async def test_retention_timer_runs_one_bounded_owned_pass_and_stops(self):
        runtime = self.runtime()
        waits = []
        async def timer(awaitable, *, timeout):
            # Consume the event coroutine without a real one-minute unit-test wait.
            awaitable.close()
            waits.append(timeout)
            raise TimeoutError
        async def process(owner, *, limit):
            self.assertEqual(owner, runtime.owner_id)
            self.assertEqual(limit, 16)
            runtime.stop.set()
            return [{"status": "scheduled", "reason": "retained_dependents"}]
        runtime.retention = SimpleNamespace(process_due=AsyncMock(side_effect=process))
        with patch("app.knowledge.runtime.asyncio.wait_for", side_effect=timer):
            await runtime._expire_sources()
        self.assertEqual(waits, [60])
        runtime.retention.process_due.assert_awaited_once()
        self.assertEqual(runtime.retention_issue, "source_retention_needs_attention")

    async def test_retention_failure_is_visible_without_logging_private_prose(self):
        runtime = self.runtime()
        async def timer(awaitable, *, timeout):
            awaitable.close()
            raise TimeoutError
        async def process(*args, **kwargs):
            runtime.stop.set()
            raise RuntimeError("synthetic-private-retention-error")
        runtime.retention = SimpleNamespace(process_due=AsyncMock(side_effect=process))
        with (
            patch("app.knowledge.runtime.asyncio.wait_for", side_effect=timer),
            self.assertLogs("app.knowledge.runtime", level="WARNING") as captured,
        ):
            await runtime._expire_sources()
        self.assertEqual(runtime.retention_issue, "source_retention_failed")
        self.assertNotIn("synthetic-private-retention-error", "\n".join(captured.output))

    async def test_shutdown_cancels_an_in_progress_retention_pass(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        async def timer(awaitable, *, timeout):
            awaitable.close()
            raise TimeoutError
        async def process(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        runtime.retention = SimpleNamespace(process_due=AsyncMock(side_effect=process))
        with patch("app.knowledge.runtime.asyncio.wait_for", side_effect=timer):
            runtime.retention_task = asyncio.create_task(runtime._expire_sources())
            await entered.wait()
        await runtime.close()
        self.assertTrue(runtime.retention_task.cancelled())
        runtime.client.close.assert_awaited_once()

    async def test_stuck_step_is_cancelled_after_shutdown_grace(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        async def run(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        runtime.worker = SimpleNamespace(run=run)
        runtime.start()
        await entered.wait()
        await runtime.close()
        self.assertTrue(runtime.task.cancelled())
        runtime.client.close.assert_awaited_once()

    async def test_failed_worker_is_observable_without_logging_raw_exception(self):
        runtime = self.runtime()
        async def run(*args, **kwargs):
            raise RuntimeError("synthetic-private-exception-detail")
        runtime.worker = SimpleNamespace(run=run)
        with self.assertLogs("app.knowledge.runtime", level="WARNING") as messages:
            runtime.start()
            await asyncio.gather(runtime.task, return_exceptions=True)
        self.assertEqual(runtime.issue, "memory_worker_failed")
        self.assertNotIn("synthetic-private-exception-detail", "\n".join(messages.output))
        await runtime.close()
        runtime.client.close.assert_awaited_once()

    async def test_already_cancelled_worker_does_not_abort_runtime_shutdown(self):
        runtime = self.runtime()
        entered = asyncio.Event()
        async def run(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        runtime.worker = SimpleNamespace(run=run)
        runtime.start()
        await entered.wait()
        with self.assertLogs("app.knowledge.runtime", level="WARNING"):
            runtime.task.cancel()
            await asyncio.gather(runtime.task, return_exceptions=True)
        # A worker cancellation is not a cancellation of the cleanup caller.
        await runtime.close()
        runtime.client.close.assert_awaited_once()

    async def test_cancelling_shutdown_caller_still_propagates_after_cleanup(self):
        runtime = self.runtime()
        runtime.configuration = configuration(shutdown_seconds=1)
        entered, closing = asyncio.Event(), asyncio.Event()
        async def run(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        runtime.worker = SimpleNamespace(run=run)
        runtime.start()
        await entered.wait()
        native_wait = asyncio.wait
        async def observed_wait(*args, **kwargs):
            closing.set()
            return await native_wait(*args, **kwargs)
        with patch("app.knowledge.runtime.asyncio.wait", side_effect=observed_wait):
            cleanup = asyncio.create_task(runtime.close())
            await closing.wait()
            cleanup.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cleanup
        self.assertTrue(runtime.task.cancelled())
        runtime.client.close.assert_awaited_once()

    async def test_cancelled_memory_worker_does_not_skip_application_pool_cleanup(self):
        runtime = self.runtime()
        runtime.task = asyncio.create_task(asyncio.sleep(60))
        runtime.task.cancel()
        await asyncio.gather(runtime.task, return_exceptions=True)
        events = []
        async def close(label):
            events.append(label)
        async def close_memory():
            await close("memory")
        runtime.client.close.side_effect = close_memory
        services = JarvisRuntimeServices(
            settings=object(),
            realtime_session_manager=SimpleNamespace(close_all=lambda: close("audio")),
            sensory_frame_window=object(), latest_visual_frame_buffer=object(),
            domain_event_bus=object(), model_limiter=object(),
            knowledge_runtime=runtime, db_pool=SimpleNamespace(close=lambda: close("pool")),
        )
        await services.close()
        self.assertEqual(events, ["audio", "memory", "pool"])

    async def test_capture_helper_binds_both_providers_and_rejects_foreign_owner(self):
        runtime = self.runtime()
        services = SimpleNamespace(knowledge_runtime=runtime, background_task_service=None)
        for provider in ("openai", "gemini"):
            capture = _create_context_capture(
                services, owner_id="fixture-owner", session_id=uuid4(),
                provider=provider, audio_session_id="fixture-" + provider,
            )
            self.assertIs(capture.repository, runtime.context_repository)
            self.assertEqual(capture.provider, provider)
            await capture.close()
        with self.assertRaises(SourceError):
            _create_context_capture(
                services, owner_id="other-owner", session_id=uuid4(),
                provider="openai", audio_session_id="foreign",
            )
        self.assertIsNone(_create_context_capture(
            SimpleNamespace(knowledge_runtime=None, background_task_service=None),
            owner_id="fixture-owner", session_id=uuid4(), provider="openai", audio_session_id="off",
        ))

    async def test_disabled_initializer_never_opens_database_or_client(self):
        services = SimpleNamespace(
            knowledge_runtime=None, knowledge_issue=None,
            settings=SimpleNamespace(SUPABASE_DB_URL=None),
        )
        with patch("app.knowledge.runtime.get_second_brain_settings",
                   return_value=SecondBrainSettings(_env_file=None, enabled="false")), \
             patch.object(KnowledgeRuntime, "create", new_callable=AsyncMock) as create:
            await initialize_knowledge_runtime(services)
        create.assert_not_awaited()
        self.assertEqual(services.knowledge_issue, "knowledge_disabled")

    async def test_enabled_shutdown_drains_audio_then_memory_before_pool(self):
        events = []
        async def close(label):
            events.append(label)
        services = JarvisRuntimeServices(
            settings=object(),
            realtime_session_manager=SimpleNamespace(close_all=lambda: close("audio")),
            sensory_frame_window=object(), latest_visual_frame_buffer=object(),
            domain_event_bus=object(), model_limiter=object(),
            knowledge_runtime=SimpleNamespace(close=lambda: close("memory")),
            db_pool=SimpleNamespace(close=lambda: close("pool")),
        )
        await services.close()
        await services.close()
        self.assertEqual(events, ["audio", "memory", "pool"])
