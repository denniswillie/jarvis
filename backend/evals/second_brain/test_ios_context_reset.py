"""Offline fixture lock-order coverage; no SQL, provider, or simulator calls."""
import ast
from contextlib import asynccontextmanager
import inspect
import json
from types import SimpleNamespace
import unittest
from uuid import UUID

from asyncpg.exceptions import DeadlockDetectedError

from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain import ios_context_reset as fixture_module


class RecordingConnection:
    def __init__(self, events, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.executions = 0

    @asynccontextmanager
    async def transaction(self):
        self.events.append(("begin",))
        try:
            yield
        except BaseException:
            self.events.append(("rollback",))
            raise
        else:
            self.events.append(("commit",))

    async def execute(self, sql, *args):
        self.executions += 1
        self.events.append(("execute", sql, args))
        if self.executions == self.fail_at:
            raise DeadlockDetectedError("PRIVATE_QUERY_CONTENT_MUST_NOT_BE_RETAINED")


class RecordingPool:
    def __init__(self, events, fail_at=None):
        self.events = events
        self.connection = RecordingConnection(events, fail_at)

    @asynccontextmanager
    async def acquire(self):
        self.events.append(("acquire",))
        try:
            yield self.connection
        finally:
            self.events.append(("release",))


class ContextResetFixtureDeletionTests(unittest.IsolatedAsyncioTestCase):
    owner = "isolated-offline-owner"
    payload = UUID("11111111-1111-4111-8111-111111111111")

    async def exercise(self, events, fail_at=None):
        pool = RecordingPool(events, fail_at)
        repository = ContextRepository(TaskRepository(pool, TaskLimits()))
        await fixture_module._delete_fixture_source(
            repository, SimpleNamespace(owner=self.owner), self.payload)

    async def test_actual_task_transaction_locks_owner_before_unchanged_deletion(self):
        events = []
        await self.exercise(events)
        self.assertEqual([event[0] for event in events],
                         ["acquire", "begin", "execute", "execute", "commit", "release"])
        self.assertEqual(events[2][1],
            "select pg_advisory_xact_lock(hashtextextended(current_schema() || ':jarvis-task-owner:' || $1,0))")
        self.assertEqual(events[2][2], (self.owner,))
        self.assertEqual(" ".join(events[3][1].split()),
            "update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2")
        self.assertEqual(events[3][2], (self.owner, self.payload))

    async def test_owner_lock_failure_does_not_attempt_deletion_or_retry(self):
        events = []
        with self.assertRaises(DeadlockDetectedError):
            await self.exercise(events, fail_at=1)
        self.assertEqual([event[0] for event in events],
                         ["acquire", "begin", "execute", "rollback", "release"])
        self.assertIn("pg_advisory_xact_lock", events[2][1])

    async def test_deletion_failure_rolls_back_and_propagates_without_retry(self):
        events = []
        with self.assertRaises(DeadlockDetectedError):
            await self.exercise(events, fail_at=2)
        self.assertEqual([event[0] for event in events],
                         ["acquire", "begin", "execute", "execute", "rollback", "release"])


class ContextResetFixtureDiagnosticTests(unittest.TestCase):
    def test_sqlstate_is_retained_without_message_query_or_detail(self):
        error = DeadlockDetectedError("PRIVATE_QUERY_CONTENT_MUST_NOT_BE_RETAINED")
        error.detail = "PRIVATE_QUERY_CONTENT_MUST_NOT_BE_RETAINED"
        result = fixture_module._safe_failure(error)
        self.assertEqual(result, {"error_type": "DeadlockDetectedError", "sqlstate": "40P01"})
        self.assertNotIn("PRIVATE_QUERY", json.dumps(result))

    def test_invalid_sqlstate_cannot_export_arbitrary_content(self):
        for value in (None, 12345, "40p01", "SECRET material", "40P01\n", "40P01" * 100):
            with self.subTest(value_type=type(value).__name__):
                error = RuntimeError("PRIVATE_CONTENT")
                error.sqlstate = value
                self.assertEqual(fixture_module._safe_failure(error),
                                 {"error_type": "RuntimeError", "sqlstate": None})

    def test_non_database_failure_has_no_invented_sqlstate(self):
        self.assertEqual(fixture_module._safe_failure(TimeoutError("PRIVATE_CONTENT")),
                         {"error_type": "TimeoutError", "sqlstate": None})

    def test_stage_order_and_all_existing_reset_oracles_remain(self):
        # Structural guard complements actual transaction execution above; it
        # does not claim the native reset/reconnect integration passed.
        tree = ast.parse(inspect.getsource(fixture_module.run))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        stages = {node.value.value: node.lineno for node in assignments
                  if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                  and any(isinstance(target, ast.Name) and target.id == "stage" for target in node.targets)}
        names = ["source_delete_transaction", "reset_await", "reset_ack_checks", "reset_session_preservation"]
        self.assertEqual([stages[name] for name in names], sorted(stages[name] for name in names))
        deletion = next(node for node in ast.walk(tree) if isinstance(node, ast.Await)
                        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                        and node.value.func.id == "_delete_fixture_source")
        self.assertLess(stages["source_delete_transaction"], deletion.lineno)
        self.assertLess(deletion.lineno, stages["reset_await"])
        wait = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "wait_for"
                    and len(node.args) == 2 and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "running" and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value == 12)
        self.assertLess(stages["reset_await"], wait.lineno)
        self.assertLess(wait.lineno, stages["reset_ack_checks"])
        checks = {target.slice.value for node in assignments for target in node.targets
                  if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                  and target.value.id == "checks" and isinstance(target.slice, ast.Constant)}
        self.assertEqual(checks, {
            "initial_context_contains_source", "reset_acknowledged", "old_provider_closed",
            "session_preserved_after_reset", "removed_text_absent", "removed_source_not_rebound",
            "independent_interruption_preserved", "missing_input_is_explicit",
            "clear_ack_retained", "conversation_preserved",
        })
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "all" and ast.unparse(node.args[0]) == "checks.values()"
                            for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
