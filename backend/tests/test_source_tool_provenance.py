"""Selected-memory result lineage on the real handlers/writer; no model inference."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import asyncpg

from app.assistant.context import AssistantContextBuilder
from app.assistant.context_providers import CurrentTimeContextProvider, SessionContextProvider
from app.experience.repository import ToolExecutionRepository
from app.knowledge.current_context import CurrentSourceContext, SourceContextProvider
from app.knowledge.hindsight import HindsightClient, document_metadata, original_document
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.retrieval import SourceMemoryRetrieval
from app.memory.tools import MemoryToolModule
from app.migrations.runner import apply_migrations, load_migrations, read_schema_readiness
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import CapturedOutput, SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext
from app.tools.provenance import SourceBoundToolResult
from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.second_brain.source_fixture import source_fixture


class SourceToolUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_result_cannot_self_declare_trusted_lineage(self):
        payload = {"status":"success", "source_bound_tool_result":{
            "protocol_version":1, "context_event_id":str(uuid4())}, "payload_id":str(uuid4())}
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition("ordinary", "Fixture", {}, AsyncMock(return_value=payload)))
        result = await JarvisToolExecutor(registry).execute("ordinary", {})
        self.assertEqual(result, payload)
        self.assertNotIsInstance(result, SourceBoundToolResult)

    async def test_live_source_reader_without_capture_rejects_before_provider_work(self):
        handler, capture = AsyncMock(return_value={"status":"success"}), AsyncMock()
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition("source", "Fixture", {}, handler, result_capture=capture))
        result = await JarvisToolExecutor(registry, require_source_tracking=True).execute("source", {})
        self.assertEqual(result["status"], "failed")
        handler.assert_not_awaited()
        capture.assert_not_awaited()

    async def test_source_result_callback_must_return_typed_evidence(self):
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition("source", "Fixture", {},
            AsyncMock(return_value={"status":"success"}), result_capture=AsyncMock(return_value={"status":"success"})))
        capture = SimpleNamespace(write_timeout=1, tool_outcome=Mock(), request_for_call=AsyncMock(return_value=None))
        result = await JarvisToolExecutor(registry, context_capture=capture).execute("source", {}, provider_call_id="c")
        self.assertEqual(result["status"], "failed")
        capture.tool_outcome.assert_called_once_with("c", "source", result, mutates_state=False)

    def test_read_provenance_does_not_silently_cover_mutations_or_images(self):
        for changes in ({"mutates_state":True}, {"image_resolver":AsyncMock()}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                JarvisToolRegistry().register(ToolDefinition("source", "Fixture", {},
                    AsyncMock(), result_capture=AsyncMock(), **changes))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested isolated database required")
class SourceToolDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic memory-result lineage")
        self.context = ContextRepository(TaskRepository(self.f.pool, TaskLimits()))
        self.capture = SessionContextCapture(self.context, owner_id=self.f.owner, session_id=self.session.id,
            provider="openai", audio_session_id="source-tool-fixture")
        self.addAsyncCleanup(self.capture.close)
        self.documents, self.candidates = {}, []
        self.provider = SimpleNamespace(
            recall=AsyncMock(side_effect=lambda bank, query: {"results":list(self.candidates)}),
            document=AsyncMock(side_effect=lambda bank, key: self.documents[key]),
            verify_document=HindsightClient.verify_document)
        self.backend = SourceMemoryReadBackend(self.f.sources,
            SourceMemoryRetrieval(self.f.sources, self.provider, timeout_seconds=20),
            owner_id=self.f.owner, area_id=self.f.area)
        tools = MemoryToolModule(read_backend=self.backend).build_tools(ToolBuildContext(None, self.f.owner, self.session.id))
        self.registry = JarvisToolRegistry()
        self.handlers = {}
        for definition in tools:
            self.handlers[definition.name] = AsyncMock(wraps=definition.handler)
            self.registry.register(replace(definition, handler=self.handlers[definition.name]))
        self.executions = ToolExecutionRepository(self.f.pool)
        self.experience = SimpleNamespace(attach_tool_execution=AsyncMock(),
            activity=lambda **kwargs: self.fail("Selected reads must not open legacy experience intake"))
        self.executor = JarvisToolExecutor(self.registry, execution_repository=self.executions,
            experience_event_service=self.experience, owner_id=self.f.owner, session_id=self.session.id,
            context_capture=self.capture, require_source_tracking=True)

    async def explain(self, source, call_id="read-one", **arguments):
        return await self.executor.execute("explain_memory", {"memory_id":str(source.source_id), **arguments},
                                           provider_call_id=call_id)

    async def payload(self, identity):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2",
                                             self.f.owner, identity)

    def voice(self):
        writes, phone = [], []
        async def send(raw):
            writes.append(json.loads(raw))
        async def send_text(raw):
            phone.append(json.loads(raw))
        async def close(**kwargs):
            pass
        voice = RealtimeAudioSession(websocket=SimpleNamespace(send_text=send_text, close=close),
            context=RealtimeSessionContext(audio_session_id="source-tool-fixture", owner_id=self.f.owner,
                jarvis_session_id=self.session.id, client_kind="phone_audio", session=self.session),
            settings=JarvisSettings(), repository=self.messages, tool_modules=(), context_providers=(),
            context_capture=self.capture)
        voice.openai._websocket = SimpleNamespace(send=send, close=close)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        return voice, writes, phone

    async def test_stored_result_is_source_bound_but_execution_log_keeps_only_reference(self):
        source = await self.f.publish("Synthetic private recall content.")
        result = await self.explain(source)
        self.assertIsInstance(result, SourceBoundToolResult)
        record = await self.context.get(self.f.owner, result.event_id, session_id=self.session.id)
        self.assertEqual(record.content["result"], result)
        self.assertIn("Synthetic private recall content.", json.dumps(record.content))
        execution = (await self.executions.get_executions_by_call_ids(
            owner_id=self.f.owner, session_id=self.session.id, provider_call_ids=["read-one"]))[0]
        self.assertEqual(set(execution.arguments), {"arguments_digest"})
        self.assertEqual(execution.result, result.audit_receipt())
        self.assertNotIn("Synthetic private recall content.", json.dumps(execution.result))
        self.experience.attach_tool_execution.assert_not_awaited()
        replay = await self.explain(source)
        self.assertIsInstance(replay, SourceBoundToolResult)
        self.assertEqual(replay.event_id, result.event_id)
        self.handlers["explain_memory"].assert_awaited_once()

    async def test_correction_invalidates_current_read_and_replay_but_keeps_original(self):
        source = await self.f.publish("Synthetic progress was incomplete.")
        result = await self.explain(source)
        await self.f.publish("Synthetic progress was checked later.", expected_revision=1)
        self.assertEqual((await self.payload(result.payload_id))["availability"], "unavailable")
        historical = await self.f.sources.read(self.f.owner, self.f.area, source.source_id, revision=1)
        self.assertEqual(historical.content, source.content)
        replay = await self.explain(source)
        self.assertEqual(replay["error"]["code"], "tool_result_source_unavailable")
        self.assertNotIn("Synthetic progress was incomplete.", json.dumps(replay))
        self.handlers["explain_memory"].assert_awaited_once()

    async def test_explicit_historical_read_survives_correction_but_not_deletion(self):
        source = await self.f.publish("Synthetic first version.")
        await self.f.publish("Synthetic second version.", expected_revision=1)
        result = await self.explain(source, revision=1)
        self.assertIsInstance(result, SourceBoundToolResult)
        self.assertFalse(result["source"]["is_current"])
        await self.f.publish("Synthetic third version.", expected_revision=2)
        self.assertEqual((await self.payload(result.payload_id))["availability"], "available")
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=3)
        self.assertEqual((await self.payload(result.payload_id))["availability"], "deleted")
        replay = await self.explain(source, revision=1)
        self.assertEqual(replay["status"], "failed")

    async def test_deleted_result_cannot_cross_queued_real_provider_writer(self):
        source = await self.f.publish("Synthetic queued private evidence.")
        result = await self.explain(source)
        voice, writes, _ = self.voice()
        await voice.openai._send_lock.acquire()
        sending = asyncio.create_task(voice._write_tool_output("read-one", result))
        try:
            await asyncio.sleep(0)
            await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        finally:
            voice.openai._send_lock.release()
        self.assertFalse(await sending)
        self.assertTrue(voice._context_revoked)
        self.assertEqual(writes, [])

    async def test_accepted_tool_output_and_native_followup_share_original_removal(self):
        source = await self.f.publish("Synthetic evidence used in a reply.")
        result = await self.explain(source)
        voice, writes, _ = self.voice()
        self.assertTrue(await voice._write_tool_output("read-one", result))
        self.assertEqual(json.loads(writes[0]["item"]["output"]), result)
        self.assertNotIn("source_bound_tool_result", json.loads(writes[0]["item"]["output"]))
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        # Use the actual policy/capture lifecycle. A generated transcript alone
        # is not a terminal, policy-admitted source for a conversation message.
        await voice._handle_openai_event({"type":"response.created",
            "response":{"id":"source-reply", "metadata":writes[-1]["response"]["metadata"]}})
        await voice._handle_openai_event({"type":"response.output_item.added", "response_id":"source-reply",
            "item":{"id":"source-reply-item", "type":"message", "role":"assistant", "phase":"final_answer"}})
        text = "Synthetic answer based on recalled evidence."
        await voice._handle_openai_event({"type":"response.output_audio_transcript.done",
            "response_id":"source-reply", "item_id":"source-reply-item", "transcript":text})
        await voice._handle_openai_event({"type":"response.done",
            "response":{"id":"source-reply", "status":"completed", "output":[]}})
        self.assertTrue(await self.capture.flush())
        self.assertIsNone(self.capture.issue)
        output = self.capture._outputs["source-reply-item"]
        self.assertIsInstance(output, CapturedOutput)
        self.assertTrue(output.terminal)
        self.assertTrue(output.completed)
        self.assertFalse(output.interrupted)
        self.assertIsNone(output.provenance)
        snapshot = voice._source_context.source_for(reservation.request_id)
        self.assertIsNotNone(snapshot)
        self.assertEqual(output.source_payload_ids, (snapshot,))
        self.assertEqual(output.playback_state, "unknown")
        messages = await self.messages.get_recent_messages(self.f.owner, self.session.id)
        self.assertEqual(len(messages), 1)
        message = messages[0]
        self.assertEqual(message.content, text)
        self.assertEqual(message.provider_response_id, "source-reply")
        self.assertEqual(message.provider_item_id, "source-reply-item")
        self.assertEqual(message.metadata["realtime_output"]["phase"], "final_answer")
        self.assertEqual(message.metadata["realtime_output"]["delivery_disposition"], "forwarded")
        self.assertIsNotNone(message.source_payload_id)
        self.assertNotEqual(message.source_payload_id, snapshot)
        async with self.f.pool.acquire() as connection:
            self.assertTrue(await connection.fetchval("""select true from jarvis_context_payload_dependencies
                where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3""",
                self.f.owner, snapshot, message.source_payload_id))
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        self.assertEqual((await self.messages.get_recent_messages(self.f.owner, self.session.id))[0].content, "")
        self.assertEqual((await self.payload(result.payload_id))["availability"], "deleted")
        self.assertEqual((await self.payload(message.source_payload_id))["availability"], "deleted")

    async def test_current_source_index_has_nonserialized_provenance_and_retires_on_correction(self):
        source = await self.f.publish("Synthetic original omitted from index.")
        builder = AssistantContextBuilder(providers=(CurrentTimeContextProvider(), SessionContextProvider(),
            SourceContextProvider(CurrentSourceContext(self.f.sources, timeout_seconds=10), self.f.area)))
        context = await builder.build(owner_id=self.f.owner, session=self.session, recent_messages=[])
        self.assertEqual(len(context.provider_source_payload_ids), 1)
        self.assertEqual(context.source_payload_ids, context.provider_source_payload_ids)
        refreshed = await builder.build(owner_id=self.f.owner, session=self.session, recent_messages=[])
        self.assertEqual(refreshed.source_payload_ids, context.source_payload_ids)
        self.assertNotEqual(refreshed.knowledge_sources["observed_at"], context.knowledge_sources["observed_at"])
        self.assertNotIn("Synthetic original omitted from index.", json.dumps(context.as_prompt_payload()))
        self.assertNotIn("provider_source_payload_ids", context.as_prompt_payload())
        await self.f.publish("Synthetic corrected original.", expected_revision=1)
        self.assertEqual((await self.payload(context.source_payload_ids[0]))["availability"], "unavailable")
        self.assertEqual((await self.payload(source.payload_id))["availability"], "available")

    async def test_unavailable_cached_index_is_replaced_not_reused(self):
        await self.f.publish()
        builder = AssistantContextBuilder(providers=(CurrentTimeContextProvider(), SessionContextProvider(),
            SourceContextProvider(CurrentSourceContext(self.f.sources, timeout_seconds=10), self.f.area)))
        first = await builder.build(owner_id=self.f.owner, session=self.session, recent_messages=[])
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='unavailable',content=null where owner_id=$1 and id=$2",
                                     self.f.owner, first.source_payload_ids[0])
        second = await builder.build(owner_id=self.f.owner, session=self.session, recent_messages=[])
        self.assertNotEqual(first.source_payload_ids, second.source_payload_ids)
        self.assertEqual((await self.payload(second.source_payload_ids[0]))["availability"], "available")

    async def test_search_binds_all_returned_originals_and_ignores_vendor_prose(self):
        originals = [await self.f.publish("Synthetic fact "+str(i), source_key="search-"+str(i)) for i in range(2)]
        for source in originals:
            self.documents[source.document_id] = {"id":source.document_id, "bank_id":source.bank_name,
                "original_text":original_document(source), "document_metadata":document_metadata(source)}
            self.candidates.append({"document_id":source.document_id, "text":"Invented vendor answer."})
        result = await self.executor.execute("search_memory", {"query":"Synthetic facts"}, provider_call_id="search")
        self.assertIsInstance(result, SourceBoundToolResult)
        self.assertNotIn("Invented vendor answer.", json.dumps(result))
        async with self.f.pool.acquire() as connection:
            bound = await connection.fetch("select source_id from jarvis_context_memory_sources where owner_id=$1 and dependent_payload_id=$2",
                                           self.f.owner, result.payload_id)
        self.assertEqual({row["source_id"] for row in bound}, {source.source_id for source in originals})
        await self.f.sources.delete_source(self.f.owner, self.f.area, originals[1].source_id, expected_revision=1)
        self.assertEqual((await self.payload(result.payload_id))["availability"], "deleted")

    async def test_empty_search_has_durable_empty_result_not_a_provider_availability_claim(self):
        result = await self.executor.execute("search_memory", {"query":"Synthetic absent result"}, provider_call_id="empty")
        self.assertIsInstance(result, SourceBoundToolResult)
        self.assertEqual(result["results"], [])
        self.assertIn("not exhaustive", result["meaning"])
        replay = await self.executor.execute("search_memory", {"query":"Synthetic absent result"}, provider_call_id="empty")
        self.assertEqual(replay.event_id, result.event_id)
        self.handlers["search_memory"].assert_awaited_once()

    async def test_canonical_correction_retires_current_read_before_source_publication(self):
        publication = ContextRepository(self.context.tasks, publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        event = ContextIngress(session_id=self.session.id, source_namespace="voice:canonical-fixture",
            source_event_id="utterance", receipt_id="one", kind="user_transcript", trust="user_input",
            payload={"text":"Synthetic first user statement."})
        original = await publication.append(self.f.owner, event)
        async with self.f.pool.acquire() as connection:
            source_id = await connection.fetchval("select source_id from jarvis_knowledge_context_sources where owner_id=$1 and event_id=$2",
                                                 self.f.owner, original.id)
        source = await self.f.sources.read(self.f.owner, self.f.area, source_id)
        result = await self.explain(source)
        revised = await self.context.append(self.f.owner, event.model_copy(update={
            "receipt_id":"two", "expected_revision":1, "payload":{"text":"Synthetic corrected user statement."}}))
        self.assertEqual((await self.payload(result.payload_id))["availability"], "unavailable")
        gap = await self.explain(source, call_id="publication-gap", revision=1)
        self.assertEqual(gap["error"]["code"], "knowledge_source_unavailable")
        async with publication.publication.transaction(self.context.tasks, self.f.owner) as connection:
            await publication.publication.publish(connection, self.f.owner, revised)
        historical = await self.explain(source, call_id="historical", revision=1)
        self.assertIsInstance(historical, SourceBoundToolResult)
        self.assertFalse(historical["source"]["is_current"])

    async def test_area_revocation_retires_its_read_materializations(self):
        source = await self.f.publish("Synthetic area-scoped evidence.")
        result = await self.explain(source)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_knowledge_areas set status='archived',revision=revision+1 where owner_id=$1 and id=$2",
                                     self.f.owner, self.f.area)
        self.assertEqual((await self.payload(result.payload_id))["availability"], "unavailable")

    async def test_changed_arguments_and_foreign_replay_reference_cannot_reuse_result(self):
        source = await self.f.publish()
        result = await self.explain(source)
        changed = await self.explain(source, revision=1)
        self.assertEqual(changed["status"], "failed")
        from app.tools.provenance import restore_source_result
        with self.assertRaises(Exception):
            await restore_source_result(self.capture, owner_id=self.f.other, session_id=self.session.id,
                                        call_id="read-one", name="explain_memory", saved=result.audit_receipt())


    async def test_real_function_call_path_captures_and_fences_memory_output(self):
        source = await self.f.publish("Synthetic full-path recall.")
        voice, writes, _ = self.voice()
        voice.tool_executor = self.executor
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        await voice._send_reserved_response_locked(reservation)
        await voice._register_response({"id":"calling-response", "metadata":writes[-1]["response"]["metadata"]})
        state = voice._active_responses["calling-response"]
        await voice._execute_function_call(name="explain_memory",
            arguments_text=json.dumps({"memory_id":str(source.source_id)}), call_id="read-one",
            generation=voice._turn_generation, source_state=state, tool_chain_id="fixture-chain", budget=None)
        outputs = [e for e in writes if e.get("item", {}).get("type") == "function_call_output"]
        self.assertEqual(len(outputs), 1)
        self.assertIn("Synthetic full-path recall.", outputs[0]["item"]["output"])
        replay = await self.explain(source)
        self.assertIsInstance(replay, SourceBoundToolResult)
        self.assertTrue(voice._source_context.has_sources)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)
        with self.assertRaises(LiveContextError):
            async with voice._source_context.context_guard():
                self.fail("A deleted tool result must invalidate the connected context")

    async def test_legacy_cached_source_result_is_not_replayed_as_unbound_text(self):
        from datetime import UTC, datetime
        from app.tools.provenance import source_read_arguments
        source = await self.f.publish()
        args = {"memory_id":str(source.source_id)}
        await self.executions.start_execution_with_status(owner_id=self.f.owner, session_id=self.session.id,
            provider_call_id="read-one", tool_name="explain_memory", arguments=source_read_arguments(args),
            mutates_state=False, started_at=datetime.now(UTC))
        await self.executions.finalize_execution(owner_id=self.f.owner, session_id=self.session.id,
            provider_call_id="read-one", status="completed", completed_at=datetime.now(UTC),
            result={"status":"success", "content":"Synthetic old unlinked cached text."})
        result = await self.explain(source)
        self.assertEqual(result["error"]["code"], "tool_result_source_unavailable")
        self.assertNotIn("Synthetic old unlinked cached text.", json.dumps(result))
        self.handlers["explain_memory"].assert_not_awaited()

    async def test_database_rejects_unlinked_wrong_digest_and_false_historical_references(self):
        source = await self.f.publish()
        async with self.f.pool.acquire() as connection:
            for linked, digest, usage in ((False, source.content_digest, "current"),
                                          (True, "0"*64, "current"),
                                          (True, source.content_digest, "historical")):
                with self.subTest(linked=linked, valid_digest=digest==source.content_digest, usage=usage):
                    with self.assertRaises(asyncpg.CheckViolationError):
                        async with connection.transaction():
                            root = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
                                values($1,'available','{}') returning id""", self.f.owner)
                            if linked:
                                await connection.execute("""insert into jarvis_context_payload_dependencies
                                    (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                                    self.f.owner, source.payload_id, root)
                            await connection.execute("""insert into jarvis_context_memory_sources
                                (owner_id,dependent_payload_id,area_id,source_id,source_revision,source_generation,
                                 source_payload_id,content_digest,source_usage) values($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                                self.f.owner, root, self.f.area, source.source_id, source.revision,
                                source.generation, source.payload_id, digest, usage)


    async def test_upgrade_from_previous_schema_and_readiness(self):
        connection = await asyncpg.connect(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"], timeout=10)
        try:
            schema = await self.f.fixture.create_schema(connection)
            migrations = load_migrations()
            self.assertGreater(len(migrations), 1)
            latest = (migrations[-1].version,)
            await apply_migrations(connection, schema=schema, fresh=True, migrations=migrations[:-1])
            before = await read_schema_readiness(connection, schema=schema, migrations=migrations)
            self.assertFalse(before.ready)
            self.assertEqual(before.pending, latest)
            self.assertEqual(await apply_migrations(connection, schema=schema, migrations=migrations), latest)
            after = await read_schema_readiness(connection, schema=schema, migrations=migrations)
            self.assertTrue(after.ready)
            self.assertEqual(after.pending, ())
            self.assertEqual(after.applied, tuple(item.version for item in migrations))
        finally:
            await connection.close()
