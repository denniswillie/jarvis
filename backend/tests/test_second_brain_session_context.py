import asyncio
import json
from datetime import UTC, datetime
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.assistant.context import AssistantContextBuilder
from app.assistant.context_providers import ContextBuildRequest, CurrentTimeContextProvider, SessionContextProvider
from app.runtime.feature_config import SecondBrainSettings
from app.sessions.models import JarvisSession
from app.tasks.context import encoded_size
from app.tasks.service import initialize_task_runtime
from app.tasks.session_context import SessionTaskContextProvider, SessionTaskContextService
from test_second_brain_context_window import source


def session():
    now = datetime(2026, 9, 14, tzinfo=UTC)
    return JarvisSession(
        id=uuid4(), title="Synthetic session", status="active", created_at=now,
        updated_at=now, closed_at=None, last_activity_at=now, behavior_instructions="Fixture policy",
    )


class SessionTaskContextTests(unittest.IsolatedAsyncioTestCase):
    def service(self, *, budget=12000):
        repository = SimpleNamespace(tasks=SimpleNamespace(context_summaries=AsyncMock()), snapshot=AsyncMock())
        repository.tasks.context_summaries.return_value = {"total": 1, "terminal_total": 0, "recent_terminal_tasks": [], "tasks": [{
            "id": str(uuid4()), "title": "Synthetic background work", "status": "running",
            "revision": 2, "session_id": str(uuid4()), "updated_at": datetime.now(UTC).isoformat(),
        }]}
        self.record = source(1, text="Do not repeat this old private command", kind="assistant_output", trust="assistant_output")
        repository.snapshot.return_value = {
            "watermark": 3, "total": 3, "recent": [self.record],
            "pinned": [], "pinned_total": 0,
        }
        return SessionTaskContextService(repository, max_characters=budget)

    async def test_both_paths_can_receive_the_optional_section_without_replaying_source_text(self):
        service = self.service()
        providers = (CurrentTimeContextProvider(), SessionContextProvider(), SessionTaskContextProvider(service))
        request_session = session()
        assistant = await AssistantContextBuilder(providers=providers).build(
            owner_id="fixture-owner", session=request_session, recent_messages=[],
        )
        payload = assistant.as_prompt_payload()["second_brain"]
        self.assertEqual(payload["active_task_count"], 1)
        self.assertEqual(payload["active_tasks"][0]["status"], "running")
        reference = payload["canonical_history"]["source_references"][0]
        self.assertEqual(reference["id"], str(self.record.id))
        self.assertEqual(reference["heard_state"], "unknown")
        self.assertFalse(payload["canonical_history"]["content_included"])
        self.assertIn("Task tools expose status, not expert prose. Results need authorized delivery by the selected voice route.",
                      payload["meaning"])
        self.assertNotIn("direct TTS", json.dumps(payload["meaning"]))
        self.assertNotIn("Do not repeat", json.dumps(payload))
        self.assertNotIn("grants", json.dumps(payload))
        service.repository.snapshot.assert_awaited_once_with(
            "fixture-owner", request_session.id, max_messages=16, max_pins=8,
        )

    async def test_foreground_exposes_question_availability_not_expert_wording(self):
        service = self.service()
        summaries = service.repository.tasks.context_summaries.return_value
        summaries["tasks"][0].update(pending_question="PRIVATE-EXPERT-QUESTION", pending_question_available=True)
        summaries.update(terminal_total=1, recent_terminal_tasks=[{
            "task_ref": "task_123456789012", "status": "paused", "revision": 2,
            "pending_question": "PRIVATE-OLDER-QUESTION"}])
        original = deepcopy(summaries)
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        for task in (*result["active_tasks"], *result["recent_terminal_tasks"]):
            self.assertNotIn("pending_question", task)
            self.assertTrue(task["pending_question_available"])
        self.assertNotIn("PRIVATE-", json.dumps(result))
        self.assertEqual(summaries, original)
        summaries["tasks"][0].update(pending_question=None, pending_question_available=False)
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        self.assertFalse(result["active_tasks"][0]["pending_question_available"])

    async def test_legacy_prompt_has_no_new_field_when_provider_is_disabled(self):
        assistant = await AssistantContextBuilder(providers=(
            CurrentTimeContextProvider(), SessionContextProvider(),
        )).build(owner_id="fixture-owner", session=session(), recent_messages=[])
        self.assertNotIn("second_brain", assistant.as_prompt_payload())

    async def test_lookup_failure_is_unknown_not_an_empty_task_list(self):
        service = self.service()
        service.repository.tasks.context_summaries.side_effect = RuntimeError("fixture failure")
        assistant = await AssistantContextBuilder(providers=(
            CurrentTimeContextProvider(), SessionContextProvider(), SessionTaskContextProvider(service),
        )).build(owner_id="fixture-owner", session=session(), recent_messages=[])
        payload = assistant.as_prompt_payload()["second_brain"]
        self.assertEqual(payload["status"], "unavailable")
        self.assertNotIn("active_task_count", payload)

    async def test_source_indexes_and_task_summaries_share_one_envelope_budget(self):
        service = self.service(budget=2048)
        service.repository.tasks.context_summaries.return_value = {
            "total": 100, "tasks": [{"id": str(uuid4()), "title": "x" * 200, "status": "running"} for _ in range(20)],
            "terminal_total": 80, "recent_terminal_tasks": [
                {"id": str(uuid4()), "title": "y" * 200, "status": "cancelled"} for _ in range(20)],
        }
        service.repository.snapshot.return_value = {
            "watermark": 80, "total": 80, "pinned": [],
            "recent": [source(i) for i in range(16)], "pinned_total": 0,
        }
        result = await service.describe(ContextBuildRequest("fixture-owner", session(), (), datetime.now(UTC)))
        self.assertLessEqual(encoded_size(result), 2048)
        self.assertGreater(result["omitted_active_task_count"], 0)
        self.assertGreater(result["omitted_terminal_task_count"], 0)
        self.assertGreater(result["canonical_history"]["unindexed_source_count"], 0)

    async def test_terminal_states_stay_visible_without_claiming_answer_delivery(self):
        service = self.service()
        terminals = [{"task_ref": "task_" + str(i) * 12, "revision": 2, "status": state}
                     for i, state in enumerate(("cancelled", "succeeded", "failed"), 1)]
        service.repository.tasks.context_summaries.return_value.update(
            terminal_total=5, recent_terminal_tasks=terminals)
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        self.assertEqual(result["recent_terminal_tasks"], terminals)
        self.assertEqual(result["terminal_task_count"], 5)
        self.assertEqual(result["omitted_terminal_task_count"], 2)
        self.assertEqual(result["active_task_count"], 1)
        self.assertNotIn("expert_result", json.dumps(result))
        self.assertNotIn("delivered", json.dumps(terminals))

    async def test_active_tasks_have_priority_within_combined_count_bound(self):
        service = self.service()
        service.max_tasks = 1
        service.repository.tasks.context_summaries.return_value.update(
            terminal_total=4, recent_terminal_tasks=[{"status": "cancelled"}])
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        self.assertEqual(len(result["active_tasks"]), 1)
        self.assertEqual(result["recent_terminal_tasks"], [])
        self.assertEqual(result["omitted_terminal_task_count"], 4)
        service.repository.tasks.context_summaries.assert_awaited_once_with("fixture-owner", limit=1)

    async def test_terminal_only_snapshot_does_not_claim_no_task_history(self):
        service = self.service()
        terminal = {"task_ref": "task_123456789012", "status": "cancelled", "revision": 2}
        service.repository.tasks.context_summaries.return_value = {
            "total": 0, "tasks": [], "terminal_total": 1, "recent_terminal_tasks": [terminal]}
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        self.assertEqual(result["active_task_count"], 0)
        self.assertEqual(result["recent_terminal_tasks"], [terminal])
        self.assertEqual(result["omitted_terminal_task_count"], 0)

    async def test_no_session_does_not_claim_a_complete_empty_history(self):
        service = self.service()
        result = await service.describe(ContextBuildRequest("fixture-owner", None, (), datetime.now(UTC)))
        self.assertEqual(result["canonical_history"]["status"], "no_session")
        service.repository.snapshot.assert_not_awaited()

    async def test_optional_context_timeout_is_bounded(self):
        service = self.service()
        service.timeout_seconds = 0.01
        async def blocked(*args, **kwargs):
            await asyncio.Event().wait()
        service.repository.tasks.context_summaries.side_effect = blocked
        with self.assertRaises(TimeoutError):
            await service.describe(ContextBuildRequest("fixture-owner", session(), (), datetime.now(UTC)))


class SessionContextRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_happens_after_schema_readiness_and_preserves_other_providers(self):
        connection = AsyncMock()
        connection.fetchval.return_value = "fixture_schema"
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        original = (CurrentTimeContextProvider(),)
        services = SimpleNamespace(
            settings=SimpleNamespace(SUPABASE_DB_URL="configured"), db_pool=pool,
            background_task_providers={}, assistant_context_providers=original,
        )
        flags = SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true")
        with patch("app.tasks.service.get_second_brain_settings", return_value=flags), patch(
            "app.tasks.service.read_schema_readiness", AsyncMock(return_value=SimpleNamespace(ready=True)),
        ):
            await initialize_task_runtime(services)
        self.assertFalse(hasattr(services,"background_task_runner_task"))
        self.assertIsNone(services.background_task_issue)
        self.assertEqual(services.assistant_context_providers[:-1], original)
        self.assertIsInstance(services.assistant_context_providers[-1], SessionTaskContextProvider)
        self.assertIs(services.assistant_context_providers[-1].service, services.background_task_context_service)
