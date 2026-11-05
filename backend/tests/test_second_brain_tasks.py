import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.runtime.feature_config import SecondBrainSettings, TaskLimits
from app.tasks.budgets import validate_amounts
from app.tasks.models import Reconciliation, TaskCreate, TaskError, TaskMutation, TaskStatus, WorkerOutcome, digest, require_transition
from app.tasks.service import TaskService, initialize_task_runtime


class TaskContractTests(unittest.TestCase):
    def test_complete_transition_matrix(self):
        # This independently specifies the contract rather than importing the
        # implementation's transition table as the expected answer.
        expected = {
            "queued": {"running", "paused", "cancelled"},
            "running": {"queued", "waiting_for_input", "paused", "succeeded", "failed", "cancel_requested", "cancelled", "outcome_unknown"},
            "waiting_for_input": {"queued", "paused", "cancelled"},
            "paused": {"queued", "cancelled", "cancel_requested", "outcome_unknown"},
            "succeeded": {"queued"}, "failed": {"queued", "cancelled"},
            "cancel_requested": {"cancelled", "outcome_unknown"}, "cancelled": {"queued"},
            "outcome_unknown": {"queued", "running", "waiting_for_input", "cancel_requested", "cancelled", "succeeded", "failed", "paused"},
        }
        for old in TaskStatus:
            for new in TaskStatus:
                with self.subTest(old=old, new=new):
                    if old == new or new.value in expected[old.value]:
                        require_transition(old, new)
                    else:
                        with self.assertRaises(TaskError):
                            require_transition(old, new)

    def test_request_identity_cannot_include_an_arbitrary_owner_or_unknown_backend(self):
        base = {"request_id": uuid4(), "title": "Fixture", "requested_outcome": "Test the contract."}
        for change in ({"owner_id": "other"}, {"backend_kind": "arbitrary-url"}, {"title": " "}, {"requested_outcome": "x" * 32_001}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                TaskCreate(**(base | change))
        with self.assertRaises(ValidationError):
            TaskMutation(request_id=uuid4(), expected_revision=0)

    def test_budget_values_and_json_are_bounded(self):
        for values in ((-1, 0, 0), (True, 0, 0), (1.5, 0, 0), (1_000_000_001, 0, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_amounts(*values)
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            digest({"bad": float("nan")})
        for changes in ({"max_attempts": 0}, {"lease_seconds": 0}, {"poll_seconds": float("inf")}, {"shutdown_seconds": 0}):
            with self.assertRaises(ValidationError):
                TaskLimits(**changes)

    def test_only_proved_terminal_reconciliation_carries_a_result(self):
        outcome = WorkerOutcome(TaskStatus.SUCCEEDED, {"answer": "Fixture"})
        Reconciliation("terminal", outcome)
        for state, payload in (("running", outcome), ("unknown", outcome), ("terminal", None), ("invented", None)):
            with self.assertRaises(ValueError):
                Reconciliation(state, payload)
        with self.assertRaises(ValueError):
            WorkerOutcome(TaskStatus.RUNNING, {"answer": "not finished"})


class TaskRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_provider_is_a_refusal_not_a_fake_queued_task(self):
        repository = AsyncMock()
        repository.replay_create.return_value = None
        service = TaskService(repository, {})
        with self.assertRaises(TaskError) as caught:
            await service.create("owner", TaskCreate(request_id=uuid4(), title="Fixture", requested_outcome="Fixture"))
        self.assertEqual(caught.exception.status_code, 503)
        repository.create.assert_not_awaited()

    async def test_feature_off_does_not_touch_the_pool(self):
        pool = AsyncMock()
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="configured"), db_pool=pool)
        with patch("app.tasks.service.get_second_brain_settings", return_value=SecondBrainSettings(_env_file=None, enabled="false", lease_seconds="bad")):
            await initialize_task_runtime(services)
        self.assertEqual(services.background_task_issue, "tasks_disabled")
        pool.acquire.assert_not_called()

    async def test_invalid_enabled_settings_fail_only_this_capability(self):
        services = SimpleNamespace(settings=SimpleNamespace(SUPABASE_DB_URL="configured"), db_pool=AsyncMock())
        with patch("app.tasks.service.get_second_brain_settings", return_value=SecondBrainSettings(_env_file=None, enabled="true", tasks_enabled="true", lease_seconds="bad")):
            await initialize_task_runtime(services)
        self.assertEqual(services.background_task_issue, "invalid_configuration")
