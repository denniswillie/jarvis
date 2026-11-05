"""Shared voice/router identity mapping; canonical service remains the authority."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.attention.notification_ledger.admission import TaskAdmissionAdapter, canonical_request_id
from app.live_state.models import SessionScope
from app.tasks.models import TaskError, json_text
from app.tasks.voice_tools import StartArguments, MutationArguments
from app.tools.invocation import CanonicalRequestIdentity


class CanonicalService:
    def __init__(self):
        self.repository = SimpleNamespace(resolve_reference=AsyncMock(return_value=uuid4()))
        self.receipts = {}
        self.calls = []
        self.mutations = []
        self.lock = asyncio.Lock()

    async def create(self, owner, request, *, source, mode):
        async with self.lock:
            signature = json_text(request.model_dump(mode="json") | {"source": source.parameters(), "mode": mode})
            self.calls.append((owner, request, source, mode))
            previous = self.receipts.get(request.request_id)
            if previous:
                if previous[0] != signature:
                    raise TaskError("task_request_conflict", "The same request changed")
                return previous[1]
            result = SimpleNamespace(id=uuid4(), status="queued")
            self.receipts[request.request_id] = (signature, result)
            return result

    async def mutate(self, owner, task_id, operation, request, *, source):
        self.mutations.append((owner, task_id, operation, request, source))
        return SimpleNamespace(id=task_id, status="cancel_requested")


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.scope = SessionScope(owner_id="An unchanged owner: name", session_id=uuid4(), area_id=uuid4())
        self.event_id = uuid4()
        self.record = SimpleNamespace(authorizes_request=True, revision=1,
                                      source_namespace="voice:session", source_event_id="canonical-user-item")
        self.sources = SimpleNamespace(get=AsyncMock(return_value=self.record))
        self.service = CanonicalService()
        self.adapter = TaskAdmissionAdapter(self.service, sources=self.sources)
        self.args = StartArguments(title="Inspect the code", requested_outcome="Explain the visible error.")

    async def test_key_matches_existing_voice_tool_identity_exactly(self):
        expected = uuid5(NAMESPACE_URL, json_text({"namespace": "jarvis.voice_task.v1",
            "owner": self.scope.owner_id, "session": str(self.scope.session_id),
            "source_namespace": self.record.source_namespace, "source_event_id": self.record.source_event_id,
            "operation": "create", "task_ref": None, "slot": 1}))
        await self.adapter.create(self.scope, self.event_id, 1, self.args)
        self.assertEqual(self.service.calls[0][1].request_id, expected)
        self.sources.get.assert_awaited_once_with(self.scope.owner_id, self.event_id, session_id=self.scope.session_id)

    async def test_router_and_voice_concurrent_proposals_share_canonical_acceptance(self):
        first, second = await asyncio.gather(
            self.adapter.create(self.scope, self.event_id, 1, self.args),
            self.adapter.create(self.scope, self.event_id, 1, self.args))
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.service.receipts), 1)
        self.assertEqual(len(self.service.calls), 2)  # No competing local dedupe authority.

    async def test_changed_arguments_same_slot_conflict_not_second_task(self):
        await self.adapter.create(self.scope, self.event_id, 1, self.args)
        with self.assertRaises(TaskError):
            await self.adapter.create(self.scope, self.event_id, 1,
                self.args.model_copy(update={"requested_outcome": "A different job"}))
        self.assertEqual(len(self.service.receipts), 1)

    async def test_asr_revision_preserves_request_key_but_conflicts_if_already_admitted(self):
        await self.adapter.create(self.scope, self.event_id, 1, self.args)
        first_key = self.service.calls[0][1].request_id
        self.record.revision = 2
        with self.assertRaises(TaskError):
            await self.adapter.create(self.scope, self.event_id, 2, self.args)
        self.assertEqual(first_key, self.service.calls[-1][1].request_id)
        self.assertEqual(len(self.service.receipts), 1)

    async def test_independent_slot_changes_identity(self):
        first = await self.adapter.create(self.scope, self.event_id, 1, self.args)
        second = await self.adapter.create(self.scope, self.event_id, 1, self.args.model_copy(update={"request_slot": 2}))
        self.assertNotEqual(first.id, second.id)

    async def test_nonfinal_or_stale_request_never_reaches_service(self):
        self.record.authorizes_request = False
        with self.assertRaises(TaskError):
            await self.adapter.create(self.scope, self.event_id, 1, self.args)
        self.record.authorizes_request = True
        self.record.revision = 2
        with self.assertRaises(TaskError):
            await self.adapter.create(self.scope, self.event_id, 1, self.args)
        self.assertFalse(self.service.calls)

    async def test_missing_canonical_names_are_not_replaced_with_provider_call_ids(self):
        self.record.source_event_id = None
        with self.assertRaises(TaskError):
            await self.adapter.create(self.scope, self.event_id, 1, self.args)
        self.assertFalse(self.service.calls)

    async def test_capabilities_map_to_existing_service_not_new_grants(self):
        combinations = [
            (["reasoning"], "none", False, False),
            (["publish_files"], "offline_sandbox", True, False),
            (["sandbox_network"], "networked_sandbox", False, False),
            (["sandbox_desktop"], "desktop_sandbox", False, False),
            (["sandbox_desktop", "sandbox_network", "publish_files", "memory_edit"], "networked_desktop_sandbox", True, True),
        ]
        for slot, (requested, environment, publish, editing) in enumerate(combinations, 1):
            with self.subTest(environment=environment):
                await self.adapter.create(self.scope, self.event_id, 1,
                    self.args.model_copy(update={"requested_capabilities": requested, "request_slot": slot}))
                request = self.service.calls[-1][1]
                self.assertEqual(request.execution_environment, environment)
                self.assertEqual(request.publish_artifacts, publish)
                self.assertEqual(request.edit_memory, editing)
                self.assertEqual(request.grant_ids, ())

    async def test_cancel_is_targeted_canonical_mutation_not_speech_interruption(self):
        args = MutationArguments(task_ref="task_123456abcdef", expected_revision=3, message="Stop that job")
        await self.adapter.mutate(self.scope, self.event_id, 1, "cancel", args)
        owner, task_id, operation, request, source = self.service.mutations[0]
        self.assertEqual(owner, self.scope.owner_id)
        self.assertEqual(operation, "cancel")
        self.assertEqual(request.expected_revision, 3)
        self.assertEqual(source.event_id, self.event_id)
        self.service.repository.resolve_reference.assert_awaited_once_with(owner, args.task_ref)
        self.assertEqual(task_id, self.service.repository.resolve_reference.return_value)

    async def test_scope_and_operation_remain_part_of_identity(self):
        names = CanonicalRequestIdentity("canonical", "same-item")
        identity = canonical_request_id(self.scope, names, "create")
        self.assertNotEqual(identity, canonical_request_id(self.scope.model_copy(update={"session_id": uuid4()}), names, "create"))
        self.assertNotEqual(identity, canonical_request_id(self.scope, names, "cancel", task_ref="task_123456abcdef"))
        with self.assertRaises(ValueError):
            canonical_request_id(self.scope, names, "create", slot=True)

