from __future__ import annotations

import unittest
from uuid import uuid4

from app.memory.repository import MemoryRepository, MemoryScopeError


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *_: object) -> None:
        return None


class _Connection:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    async def fetchval(self, query: str, *args: object) -> object:
        self.calls.append((query, args))
        return self.result


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def acquire(self) -> _AsyncContext:
        return _AsyncContext(self.connection)


class RetrievalAuditRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_audit_records_bounded_ids_provider_spaces_and_session_scope(self) -> None:
        run_id = uuid4()
        candidate_ids = [uuid4(), uuid4()]
        connection = _Connection(run_id)
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        result = await repository.record_retrieval_run(
            owner_id="owner",
            session_id=uuid4(),
            query="Where did I leave my keys?",
            constraints={"memory_types": ["episode"]},
            candidate_ids=candidate_ids,
            result_ids=[candidate_ids[1]],
            reinspection_performed=True,
            provider_spaces=["openai:text-embedding-3-small", "event:multimodal"],
            latency_ms=42,
        )

        self.assertEqual(result, run_id)
        query, args = connection.calls[0]
        self.assertIn("memory_retrieval_runs", query)
        self.assertIn("jarvis_sessions", query)
        self.assertEqual(args[4], candidate_ids)
        self.assertEqual(args[5], [candidate_ids[1]])

    async def test_audit_rejects_secret_bearing_constraints_before_database_write(self) -> None:
        connection = _Connection(uuid4())
        repository = MemoryRepository(_Pool(connection))  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "sensitive field"):
            await repository.record_retrieval_run(
                owner_id="owner",
                session_id=None,
                query="keys",
                constraints={"provider": {"access_token": "secret"}},
                candidate_ids=[],
                result_ids=[],
                reinspection_performed=False,
                provider_spaces=[],
                latency_ms=1,
            )

        self.assertEqual(connection.calls, [])

    async def test_audit_rejects_signed_url_values(self) -> None:
        repository = MemoryRepository(_Pool(_Connection(uuid4())))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "signed credentials"):
            await repository.record_retrieval_run(
                owner_id="owner",
                session_id=None,
                query="keys",
                constraints={"source": "https://example.test/file?X-Amz-Signature=abc"},
                candidate_ids=[],
                result_ids=[],
                reinspection_performed=False,
                provider_spaces=[],
                latency_ms=1,
            )

    async def test_results_must_be_selected_from_candidates(self) -> None:
        repository = MemoryRepository(_Pool(_Connection(uuid4())))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "selected from candidates"):
            await repository.record_retrieval_run(
                owner_id="owner",
                session_id=None,
                query="keys",
                constraints={},
                candidate_ids=[uuid4()],
                result_ids=[uuid4()],
                reinspection_performed=False,
                provider_spaces=[],
                latency_ms=1,
            )

    async def test_candidate_audit_is_bounded(self) -> None:
        repository = MemoryRepository(_Pool(_Connection(uuid4())))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "more than 200"):
            await repository.record_retrieval_run(
                owner_id="owner",
                session_id=None,
                query="keys",
                constraints={},
                candidate_ids=[uuid4() for _ in range(201)],
                result_ids=[],
                reinspection_performed=False,
                provider_spaces=[],
                latency_ms=1,
            )

    async def test_session_outside_owner_scope_is_rejected(self) -> None:
        repository = MemoryRepository(_Pool(_Connection(None)))  # type: ignore[arg-type]
        with self.assertRaisesRegex(MemoryScopeError, "retrieval session"):
            await repository.record_retrieval_run(
                owner_id="owner",
                session_id=uuid4(),
                query="keys",
                constraints={},
                candidate_ids=[],
                result_ids=[],
                reinspection_performed=False,
                provider_spaces=[],
                latency_ms=1,
            )

    async def test_provider_spaces_reject_urls_and_secret_shaped_values(self) -> None:
        repository = MemoryRepository(_Pool(_Connection(uuid4())))  # type: ignore[arg-type]
        for provider_space in (
            "https://provider.test/model",
            "sk-secret-value",
            "provider?access_token=secret",
        ):
            with self.subTest(provider_space=provider_space):
                with self.assertRaisesRegex(ValueError, "non-secret identifier"):
                    await repository.record_retrieval_run(
                        owner_id="owner",
                        session_id=None,
                        query="keys",
                        constraints={},
                        candidate_ids=[],
                        result_ids=[],
                        reinspection_performed=False,
                        provider_spaces=[provider_space],
                        latency_ms=1,
                    )
