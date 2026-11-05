"""Real shared executor with isolated repositories; no provider or database calls."""
import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.hermes_sessions.bindings import BindingError
from app.hermes_sessions.domain import DomainInputScope, RetainedDomainAdapter
from app.images import ImageReference, ResolvedImage
from app.sessions.models import JarvisSession
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskError
from app.tools.invocation import current_tool_invocation
from app.tools.module import FunctionToolModule
from app.tools.provenance import SourceBoundToolResult, source_read_arguments
from app.tools.registry import ToolDefinition


class Contexts:
    def __init__(self, pool):
        self.pool, self.records, self.events, self.by_receipt = pool, {}, [], {}
        self.fail_append = False

    async def get(self, owner, identity, *, session_id=None):
        record = self.records[identity]
        if record.owner_id != owner or record.session_id != session_id:
            raise TaskError("scope_mismatch", "Wrong original scope")
        return record

    async def append(self, owner, event):
        if self.fail_append:
            raise OSError("fixture unavailable")
        key = (owner, event.source_namespace, event.source_event_id, event.receipt_id)
        if key in self.by_receipt:
            return self.by_receipt[key]
        self.events.append(event)
        record = ContextRecord(id=uuid4(), owner_id=owner, session_id=event.session_id,
            sequence=len(self.events), origin_sequence=len(self.events), source_namespace=event.source_namespace,
            source_event_id=event.source_event_id, revision=1, supersedes_event_id=None,
            kind=event.kind, trust=event.trust, finality=event.finality, source_time=event.source_time,
            recorded_at=datetime.now(UTC), payload_id=uuid4(), availability="available",
            content_digest=None, content=event.payload, is_current=True)
        self.records[record.id] = self.by_receipt[key] = record
        return record

    async def append_in_transaction(self, connection, owner, event, **kwargs):
        return await self.append(owner, event)


class Executions:
    def __init__(self, pool):
        self._pool, self.records = pool, {}

    async def start_execution_with_status(self, **kwargs):
        key = (kwargs["owner_id"], kwargs["session_id"], kwargs["provider_call_id"])
        existing = self.records.get(key)
        if existing is not None:
            if existing.tool_name != kwargs["tool_name"] or existing.arguments != kwargs["arguments"]:
                raise ValueError("Conflicting call identity")
            return existing, False
        record = SimpleNamespace(**deepcopy(kwargs), status="started", result=None, error=None)
        self.records[key] = record
        return record, True

    async def finalize_execution(self, **kwargs):
        key = (kwargs["owner_id"], kwargs["session_id"], kwargs["provider_call_id"])
        record = self.records[key]
        record.status = kwargs["status"]
        record.result, record.error = kwargs.get("result"), kwargs.get("error")
        return record


class DomainTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool = object()
        self.contexts, self.executions = Contexts(self.pool), Executions(self.pool)
        self.session_id = uuid4()
        now = datetime.now(UTC)
        self.session = JarvisSession(id=self.session_id, title="Fixture", status="active",
            created_at=now, updated_at=now, closed_at=None, last_activity_at=now,
            behavior_instructions="")
        self.sessions = SimpleNamespace(_pool=self.pool, get_session=AsyncMock(return_value=self.session))
        self.binding = SimpleNamespace(logical_id="master", native_id="native-master")
        self.service = SimpleNamespace(bindings=SimpleNamespace(owner_id="owner"),
                                       caller=AsyncMock(return_value=self.binding))
        self.scope = DomainInputScope("owner", "master", "input-1", self.session_id,
                                     frozenset({"read", "write"}))
        self.handler = AsyncMock(return_value={"status": "success", "value": "fixture"})
        self.writer = AsyncMock(return_value={"status": "success"})
        self.definitions = [ToolDefinition("read", "Read an original.", {"type": "object"}, self.handler),
                            ToolDefinition("write", "Write.", {"type": "object"}, self.writer, mutates_state=True)]
        self.built = []

        def build(context):
            self.built.append(context)
            return self.definitions

        self.modules = [FunctionToolModule("fixture", build)]
        self.scope_resolver = AsyncMock(side_effect=lambda binding, input_id: self.scope)
        self.module_factory = AsyncMock(return_value=self.modules)
        self.adapter = RetainedDomainAdapter(self.service, sessions=self.sessions,
            contexts=self.contexts, executions=self.executions, scope_for_input=self.scope_resolver,
            modules_for_scope=self.module_factory)

    async def invoke(self, **kwargs):
        return await self.adapter.invoke(**dict(native_session_id="native-master", input_id="input-1",
            call_id="call-1", name="read", arguments={}) | kwargs)

    async def original(self):
        record = ContextRecord(id=uuid4(), owner_id="owner", session_id=self.session_id,
            sequence=1, origin_sequence=1, source_namespace="voice:fixture", source_event_id="input-1",
            revision=1, supersedes_event_id=None, kind="user_transcript", trust="user_input",
            finality="final", source_time=datetime.now(UTC), recorded_at=datetime.now(UTC),
            payload_id=uuid4(), availability="available", content_digest=None,
            content={"text": "Read my history."}, is_current=True)
        self.contexts.records[record.id] = record
        self.scope = replace(self.scope, request_event_id=record.id, request_revision=1)
        return record

    async def test_catalog_hides_writes_and_binds_actual_domain_session(self):
        catalog = await self.adapter.catalog(native_session_id="native-master", input_id="input-1")
        self.assertEqual([tool["name"] for tool in catalog], ["read"])
        self.assertEqual(self.built[0].owner_id, "owner")
        self.assertEqual(self.built[0].session_id, self.session_id)
        self.sessions.get_session.assert_awaited_once_with("owner", self.session_id)
        self.assertFalse(hasattr(self.session, "owner_id"))
        self.handler.assert_not_called()

    async def test_write_and_unlisted_tool_are_not_invoked(self):
        self.assertEqual((await self.invoke(name="write")).payload["error"]["code"], "domain_write_unavailable")
        self.assertEqual((await self.invoke(name="unknown")).payload["error"]["code"], "domain_tool_not_allowed")
        self.writer.assert_not_called()
        self.assertEqual(self.executions.records, {})

    async def test_foreign_or_unbound_native_session_is_rejected(self):
        self.service.caller.side_effect = BindingError("unknown_native_session")
        with self.assertRaises(BindingError):
            await self.invoke()
        self.scope_resolver.assert_not_called()

    def enable_fixture_write(self):
        self.adapter = RetainedDomainAdapter(self.service, sessions=self.sessions,
            contexts=self.contexts, executions=self.executions, scope_for_input=self.scope_resolver,
            modules_for_scope=self.module_factory, qualified_write_tools=frozenset({"write"}))

    async def test_qualified_write_requires_original_user_request(self):
        self.enable_fixture_write()
        catalog = await self.adapter.catalog(native_session_id="native-master", input_id="input-1")
        self.assertEqual([tool["name"] for tool in catalog], ["read"])
        outcome = await self.invoke(name="write")
        self.assertEqual(outcome.payload["error"]["code"], "domain_write_request_required")
        self.writer.assert_not_called()
        self.assertEqual(self.executions.records, {})

    async def test_retained_write_uses_owned_input_without_fabricating_a_request(self):
        self.adapter = RetainedDomainAdapter(self.service, sessions=self.sessions,
            contexts=self.contexts, executions=self.executions, scope_for_input=self.scope_resolver,
            modules_for_scope=self.module_factory, qualified_write_tools=frozenset({"write"}),
            retained_write_tools=frozenset({"write"}))
        observed = []
        async def writer(arguments):
            observed.append(current_tool_invocation())
            return {"status": "success"}
        self.writer.side_effect = writer
        self.assertEqual((await self.invoke(name="write")).payload["status"], "success")
        self.assertIsNone(observed[0].request_event_id)
        self.assertIsNone(observed[0].request_revision)
        self.assertIsNone(observed[0].task)
        self.assertEqual(self.contexts.events[0].kind, "tool_outcome")
        self.scope = replace(self.scope, allowed_tools=frozenset({"read"}))
        self.assertEqual((await self.invoke(name="write")).payload["error"]["code"], "domain_tool_not_allowed")

    def test_retained_write_must_also_be_qualified(self):
        with self.assertRaises(ValueError):
            RetainedDomainAdapter(self.service, sessions=self.sessions,
                contexts=self.contexts, executions=self.executions, scope_for_input=self.scope_resolver,
                modules_for_scope=self.module_factory, retained_write_tools=frozenset({"write"}))

    async def test_qualified_write_uses_exact_request_and_replays_once(self):
        request = await self.original()
        self.enable_fixture_write()
        seen = []

        async def writer(arguments):
            seen.append(current_tool_invocation())
            return {"status": "success", "saved": True}

        self.writer.side_effect = writer
        catalog = await self.adapter.catalog(native_session_id="native-master", input_id="input-1")
        self.assertEqual([tool["name"] for tool in catalog], ["read", "write"])
        first, replay = await self.invoke(name="write"), await self.invoke(name="write")
        self.assertEqual(first.payload, replay.payload)
        self.writer.assert_awaited_once()
        self.assertEqual(seen[0].request_event_id, request.id)
        self.assertEqual(seen[0].request_revision, request.revision)
        self.assertIsNone(seen[0].task)

    async def test_qualified_write_does_not_bypass_changed_request(self):
        request = await self.original()
        self.enable_fixture_write()
        self.contexts.records[request.id] = request.model_copy(update={"is_current": False})
        with self.assertRaises(TaskError):
            await self.invoke(name="write")
        self.writer.assert_not_called()

    async def test_qualified_write_does_not_override_input_capabilities(self):
        await self.original()
        self.enable_fixture_write()
        self.scope = replace(self.scope, allowed_tools=frozenset({"read"}))
        self.assertEqual((await self.invoke(name="write")).payload["error"]["code"], "domain_tool_not_allowed")
        self.writer.assert_not_called()

    async def test_write_qualification_is_trusted_immutable_configuration(self):
        with self.assertRaises(ValueError):
            RetainedDomainAdapter(self.service, sessions=self.sessions, contexts=self.contexts,
                executions=self.executions, scope_for_input=self.scope_resolver,
                modules_for_scope=self.module_factory, qualified_write_tools={"write"})

    async def test_scope_cannot_cross_owner_conversation_or_input(self):
        original = self.scope
        for changes in ({"owner_id": "other"}, {"logical_session_id": "child"}, {"input_id": "newer"}):
            self.scope = replace(original, **changes)
            with self.assertRaises(BindingError):
                await self.invoke()
        self.handler.assert_not_called()

    async def test_missing_owned_domain_session_is_not_recreated(self):
        self.sessions.get_session.return_value = None
        with self.assertRaises(BindingError):
            await self.invoke()
        self.handler.assert_not_called()

    async def test_foreign_session_lookup_remains_owner_scoped(self):
        foreign_id = uuid4()
        self.scope = replace(self.scope, jarvis_session_id=foreign_id)
        self.sessions.get_session.return_value = None
        with self.assertRaises(BindingError):
            await self.invoke()
        self.sessions.get_session.assert_awaited_once_with("owner", foreign_id)
        self.module_factory.assert_not_called()
        self.handler.assert_not_called()

    async def test_repository_result_must_match_exact_session(self):
        self.sessions.get_session.return_value = self.session.model_copy(update={"id": uuid4()})
        with self.assertRaises(BindingError):
            await self.invoke()
        self.module_factory.assert_not_called()
        self.handler.assert_not_called()

    async def test_canonical_original_and_tool_context_are_preserved(self):
        request = await self.original()
        seen = []

        async def handler(arguments):
            seen.append(current_tool_invocation())
            return {"status": "success", "value": "read"}

        self.handler.side_effect = handler
        await self.invoke(arguments={"owner_id": "model-cannot-choose-owner"})
        self.assertEqual(seen[0].request_event_id, request.id)
        self.assertEqual(seen[0].request_revision, 1)
        self.assertEqual(seen[0].request_identity.namespace, request.source_namespace)
        self.assertIsNone(seen[0].task)
        self.assertIsNone(current_tool_invocation().request_event_id)
        self.assertIn(request.payload_id, self.contexts.events[0].source_payload_ids)
        self.assertEqual(self.contexts.events[0].kind, "tool_outcome")
        self.assertEqual(self.built[0].owner_id, "owner")

    async def test_changed_or_deleted_original_blocks_execution(self):
        original = await self.original()
        for changes in ({"is_current": False}, {"availability": "deleted"}, {"revision": 2},
                        {"kind": "tool_outcome", "trust": "tool_receipt"}):
            self.contexts.records[original.id] = original.model_copy(update=changes)
            with self.assertRaises(TaskError):
                await self.invoke()
        self.handler.assert_not_called()

    async def test_duplicate_call_reuses_existing_execution(self):
        first, second = await self.invoke(), await self.invoke()
        self.assertEqual(first.payload, second.payload)
        self.handler.assert_awaited_once()
        self.assertEqual(len(self.executions.records), 1)
        self.assertEqual(len(self.contexts.events), 1)

    async def test_conflicting_call_is_not_reexecuted(self):
        await self.invoke(arguments={"query": "one"})
        with self.assertLogs("app.tools.executor", level="ERROR"):
            outcome = await self.invoke(arguments={"query": "two"})
        self.assertEqual(outcome.payload["status"], "failed")
        self.handler.assert_awaited_once()

    async def test_result_capture_audit_and_revocation_are_not_discarded(self):
        evidence_id = uuid4()

        async def capture(result, context):
            record = await context.capture.bound_tool_outcome(None, context.call_id,
                context.tool_name, result, [evidence_id])
            return SourceBoundToolResult(result, record)

        self.definitions[0] = replace(self.definitions[0], result_capture=capture,
                                      audit_arguments=source_read_arguments)
        first = await self.invoke(arguments={"query": "private query"})
        execution = next(iter(self.executions.records.values()))
        self.assertNotIn("query", execution.arguments)
        self.assertIn("source_bound_tool_result", execution.result)
        self.assertNotIn("value", execution.result)
        self.assertIn(evidence_id, self.contexts.events[0].source_payload_ids)
        replay = await self.invoke(arguments={"query": "private query"})
        self.assertEqual(replay.payload, first.payload)
        record = self.contexts.records[first.payload.event_id]
        self.contexts.records[record.id] = record.model_copy(update={"availability": "deleted", "content": None})
        denied = await self.invoke(arguments={"query": "private query"})
        self.assertEqual(denied.payload["error"]["code"], "tool_result_source_unavailable")
        self.handler.assert_awaited_once()

    async def test_image_resolver_returns_bytes_separately_and_keeps_revalidation(self):
        reference = ImageReference(owner_id="owner", source="historical", content_type="image/jpeg",
                                   frame_id=uuid4(), session_id=self.session_id,
                                   width=10, height=10, captured_at=datetime.now(UTC))
        validate = AsyncMock()
        resolved = ResolvedImage(reference, b"fixture-jpeg", validate)
        resolver = AsyncMock(return_value=resolved)
        self.definitions[0] = replace(self.definitions[0], image_resolver=resolver)
        outcome = await self.invoke()
        self.assertIs(outcome.image, resolved)
        validate.assert_awaited_once()
        self.assertNotIn("image_bytes", outcome.payload)
        self.assertNotIn("fixture-jpeg", str(next(iter(self.executions.records.values())).result))

    async def test_capture_failure_is_not_reported_as_a_usable_result(self):
        self.contexts.fail_append = True
        first = await self.invoke()
        self.assertEqual(first.payload["error"]["code"], "domain_result_capture_unavailable")
        self.contexts.fail_append = False
        self.assertEqual((await self.invoke()).payload["status"], "success")
        self.handler.assert_awaited_once()

    async def test_new_input_uses_new_domain_session_without_replacing_master(self):
        await self.invoke()
        new_session = uuid4()
        self.scope = replace(self.scope, input_id="input-2", jarvis_session_id=new_session)
        self.sessions.get_session.return_value = self.session.model_copy(update={"id": new_session})
        await self.invoke(input_id="input-2")
        self.assertEqual([context.session_id for context in self.built], [self.session_id, new_session])
        self.assertEqual(self.handler.await_count, 2)
        self.assertEqual(len(self.executions.records), 2)

    async def test_cancellation_does_not_blindly_restart_tool(self):
        entered = asyncio.Event()

        async def slow(arguments):
            entered.set()
            await asyncio.Event().wait()

        self.handler.side_effect = slow
        pending = asyncio.create_task(self.invoke())
        await entered.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        replay = await self.invoke()
        self.assertEqual(replay.payload["error"]["code"], "tool_execution_in_progress")
        self.handler.assert_awaited_once()

    async def test_invalid_scope_and_mismatched_database_are_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.scope, request_event_id=uuid4())
        with self.assertRaises(ValueError):
            replace(self.scope, allowed_tools={"read"})
        with self.assertRaises(ValueError):
            RetainedDomainAdapter(self.service, sessions=self.sessions, contexts=self.contexts,
                executions=Executions(object()), scope_for_input=self.scope_resolver,
                modules_for_scope=self.module_factory)


if __name__ == "__main__":
    unittest.main()
