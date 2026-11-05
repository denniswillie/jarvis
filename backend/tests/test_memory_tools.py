from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from app.memory.tools import MemoryToolModule
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.registry import JarvisToolRegistry


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


class _MemoryService:
    def __init__(self) -> None:
        self.search_call = None
        self.remember_call = None

    async def search(self, **kwargs):
        self.search_call = kwargs
        return SimpleNamespace(
            retrieval_run_id=uuid4(),
            candidate_count=0,
            provider_spaces=(),
            results=(),
            evidence_reinspection=None,
        )

    async def remember(self, **kwargs):
        self.remember_call = kwargs
        memory = SimpleNamespace(
            id=uuid4(),
            session_id=kwargs["session_id"],
            memory_type="semantic_fact",
            status="active",
            title="Remembered fact",
            content=kwargs["content"],
            summary=kwargs["content"],
            source_experience_event_id=None,
            occurred_at=NOW,
            valid_from=NOW,
            valid_to=None,
            importance=0.85,
            confidence=1.0,
            source="explicit_agent_memory",
            is_explicit=True,
            superseded_by_memory_id=None,
            created_at=NOW,
            updated_at=NOW,
        )
        return SimpleNamespace(
            memory=memory,
            evidence=(),
            indexing_status="queued",
            replayed=False,
            warnings=(),
        )


class _ForgettingService:
    def __init__(self) -> None:
        self.call = None

    async def forget(self, **kwargs):
        self.call = kwargs
        return SimpleNamespace(
            memory=SimpleNamespace(id=kwargs["memory_id"]),
            replayed=False,
        )


class MemoryToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.service = _MemoryService()
        self.forgetting_service = _ForgettingService()
        self.session_id = uuid4()
        module = MemoryToolModule(
            repository=SimpleNamespace(),  # type: ignore[arg-type]
            entity_repository=SimpleNamespace(),  # type: ignore[arg-type]
            service=self.service,  # type: ignore[arg-type]
            forgetting_service=self.forgetting_service,  # type: ignore[arg-type]
        )
        self.tools = {
            tool.name: tool
            for tool in module.build_tools(
                ToolBuildContext(
                    settings=None,
                    owner_id="owner",
                    session_id=self.session_id,
                )
            )
        }

    async def test_search_uses_context_owner_and_current_session(self) -> None:
        result = await self.tools["search_memory"].handler(
            {
                "query": "station poster",
                "seconds_before_now": 300,
                "current_session_only": True,
                "include_evidence_metadata": True,
            }
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.service.search_call["owner_id"], "owner")
        self.assertEqual(
            self.service.search_call["request"].session_id,
            self.session_id,
        )
        self.assertFalse(self.service.search_call["reinspect_evidence"])
        self.assertTrue(self.service.search_call["request"].include_evidence_metadata)

    async def test_search_does_not_expose_or_accept_secondary_model_reinspection(self) -> None:
        tool = self.tools["search_memory"]
        self.assertNotIn("reinspect_evidence", tool.input_schema["properties"])
        for extra in ({"reinspect_evidence": True}, {"owner_id": "other"}, {"url": "untrusted"}):
            with self.subTest(extra=extra):
                result = await tool.handler({"query": "station poster", **extra})
                self.assertEqual(result["error"]["code"], "invalid_tool_input")
        self.assertIsNone(self.service.search_call)

    async def test_remember_is_distinct_and_mutating(self) -> None:
        tool = self.tools["remember"]
        result = await tool.handler(
            {"memory_type": "fact", "content": "The blue key opens the shed."}
        )
        self.assertTrue(tool.mutates_state)
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.service.remember_call["owner_id"], "owner")
        self.assertEqual(self.service.remember_call["session_id"], self.session_id)
        self.assertIn("dedicated", tool.description)

    async def test_remember_receives_provider_call_identity(self) -> None:
        registry = JarvisToolRegistry()
        registry.register(self.tools["remember"])
        executor = JarvisToolExecutor(registry)

        result = await executor.execute(
            "remember",
            {"memory_type": "fact", "content": "The blue key opens the shed."},
            provider_call_id="call-remember-123",
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            self.service.remember_call["operation_id"],
            "call-remember-123",
        )

    async def test_module_keeps_compatibility_history_tool(self) -> None:
        self.assertEqual(
            set(self.tools),
            {
                "get_semantic_fact_history",
                "search_memory",
                "remember",
                "explain_memory",
                "forget_memory",
            },
        )


if __name__ == "__main__":
    unittest.main()
