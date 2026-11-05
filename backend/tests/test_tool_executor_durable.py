import unittest
from types import SimpleNamespace
from uuid import uuid4

from app.tools.executor import JarvisToolExecutor
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from unittest.mock import AsyncMock


class _Registry:
    def __init__(self, handler) -> None:
        self._tool = SimpleNamespace(handler=handler, mutates_state=True)

    def get(self, name: str):
        return self._tool if name == "mutate" else None


class _ExecutionRepository:
    def __init__(self, execution, *, created: bool, fail_finalize: bool = False) -> None:
        self.execution = execution
        self.created = created
        self.fail_finalize = fail_finalize

    async def start_execution_with_status(self, **kwargs):
        self.start_kwargs = kwargs
        return self.execution, self.created

    async def finalize_execution(self, **kwargs):
        if self.fail_finalize:
            raise RuntimeError("database unavailable")
        return SimpleNamespace(
            **{
                **self.execution.__dict__,
                "status": kwargs["status"],
                "result": kwargs.get("result"),
                "error": kwargs.get("error"),
            }
        )


def _started_execution():
    return SimpleNamespace(status="started", result=None, error=None)


class DurableToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_selection_is_persisted_and_replayed_without_resolver_or_attachment_claim(self):
        selection = {
            "status": "success", "image_status": "selected",
            "image": {"frame_id": str(uuid4()), "source": "current", "captured_at": "2026-09-09T12:00:00+00:00"},
        }
        handler = AsyncMock(return_value=selection)
        resolver = AsyncMock()
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition(
            name="image_selection", description="Select evidence.", input_schema={"type": "object"},
            handler=handler, image_resolver=resolver,
        ))
        repository = _ExecutionRepository(_started_execution(), created=True)
        finalize = AsyncMock(wraps=repository.finalize_execution)
        repository.finalize_execution = finalize
        executor = JarvisToolExecutor(
            registry, execution_repository=repository, owner_id="owner", session_id=uuid4(),
        )
        result = await executor.execute("image_selection", {}, provider_call_id="image-call")
        self.assertEqual(result, selection)
        self.assertEqual(finalize.await_args.kwargs["result"], selection)
        self.assertFalse(repository.start_kwargs["mutates_state"])
        repository.execution = SimpleNamespace(status="completed", result=selection, error=None)
        repository.created = False
        replay = await executor.execute("image_selection", {}, provider_call_id="image-call")
        self.assertEqual(replay, selection)
        self.assertEqual(replay["image_status"], "selected")
        self.assertNotIn("provider_item_id", replay)
        handler.assert_awaited_once()
        resolver.assert_not_awaited()
        finalize.assert_awaited_once()

    async def test_existing_started_call_is_not_executed_again(self) -> None:
        calls = 0

        async def handler(arguments):
            nonlocal calls
            calls += 1
            return {"status": "success"}

        executor = JarvisToolExecutor(
            _Registry(handler),  # type: ignore[arg-type]
            execution_repository=_ExecutionRepository(
                _started_execution(),
                created=False,
            ),  # type: ignore[arg-type]
            owner_id="owner",
            session_id=uuid4(),
        )

        result = await executor.execute(
            "mutate",
            {},
            provider_call_id="call-1",
        )

        self.assertEqual(calls, 0)
        self.assertEqual(result["error"]["code"], "tool_execution_in_progress")
        self.assertTrue(executor._execution_repository.start_kwargs["mutates_state"])

    async def test_finalize_failure_does_not_report_success(self) -> None:
        calls = 0

        async def handler(arguments):
            nonlocal calls
            calls += 1
            return {"status": "success", "value": "changed"}

        executor = JarvisToolExecutor(
            _Registry(handler),  # type: ignore[arg-type]
            execution_repository=_ExecutionRepository(
                _started_execution(),
                created=True,
                fail_finalize=True,
            ),  # type: ignore[arg-type]
            owner_id="owner",
            session_id=uuid4(),
        )

        result = await executor.execute(
            "mutate",
            {},
            provider_call_id="call-2",
        )

        self.assertEqual(calls, 1)
        self.assertEqual(
            result["error"]["code"],
            "tool_execution_persistence_failed",
        )
