import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.context import TaskContextBuilder, encoded_size
from app.tasks.context_events import ContextRecord, ContextWindow
from app.tasks.context_repository import ContextRepository
from app.tasks.context_tools import TaskContextToolModule
from app.tasks.session_context import SessionTaskContextService
from app.tasks.models import ClaimLost, TaskError, json_text
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


def source(sequence, *, text="Synthetic context", **changes):
    now = datetime(2026, 9, 14, tzinfo=UTC)
    return ContextRecord(**({
        "id": uuid4(), "owner_id": "fixture-owner", "session_id": None,
        "sequence": sequence, "origin_sequence": sequence,
        "source_namespace": "fixture", "source_event_id": str(sequence), "revision": 1,
        "supersedes_event_id": None, "kind": "user_transcript", "trust": "user_input",
        "finality": "final", "source_time": now, "recorded_at": now, "payload_id": uuid4(),
        "availability": "available", "content_digest": "a" * 64,
        "content": {"text": text}, "is_current": True,
    } | changes))


def claim():
    return SimpleNamespace(task=SimpleNamespace(
        id=uuid4(), owner_id="fixture-owner", session_id=None, revision=1,
        title="Fixture task", requested_outcome="Preserve the original instructions",
        request_event_id=uuid4(), request_revision=1,
    ))


class BoundedContextTests(unittest.IsolatedAsyncioTestCase):
    def builder(self, records, *, pinned=(), total=None, pinned_total=None, budget=6000):
        repository = AsyncMock()
        repository.snapshot.return_value = {
            "watermark": 100, "total": len(records) if total is None else total,
            "recent": records, "pinned": list(pinned),
            "selected_pinned_ids": {record.id for record in pinned},
            "pinned_total": len(pinned) if pinned_total is None else pinned_total,
        }
        async def save(task, **values):
            return SimpleNamespace(content=values["content"], records=values["records"])
        repository.store_manifest.side_effect = save
        return TaskContextBuilder(repository, window=ContextWindow(max_characters=budget))

    async def test_exact_request_pointer_and_priority_do_not_replace_original_with_summary(self):
        original = source(1, text="Do not make changes.", revision=2)
        unrelated = source(2, text="A pinned unrelated observation.")
        task = claim()
        task.task.request_event_id, task.task.request_revision = original.id, original.revision
        task.task.requested_outcome = "A summary mistakenly says do not propose changes."
        builder = self.builder([unrelated, original], pinned=[unrelated], total=2)
        manifest = await builder.build(task)
        self.assertEqual(manifest.records[0].id, original.id)
        self.assertEqual(manifest.content["task"]["requested_outcome_role"], "assignment_summary")
        self.assertEqual(manifest.content["task"]["request_source"],
            {"event_id": str(original.id), "revision": 2, "reader": "read_task_context_source"})
        self.assertIn("Do not make changes.", json.dumps(manifest.content["messages"]))
        self.assertEqual([message["id"] for message in manifest.content["messages"]],
                         [str(original.id), str(unrelated.id)])

    async def test_oversized_request_keeps_exact_retrieval_pointer(self):
        original, task = source(1, text="Original restriction. " * 1000), claim()
        task.task.request_event_id = original.id
        builder = self.builder([original], budget=3000)
        manifest = await builder.build(task)
        self.assertEqual(manifest.content["task"]["request_source"]["event_id"], str(original.id))
        self.assertEqual(manifest.content["messages"], [])
        self.assertLessEqual(encoded_size(manifest.content), 3000)
        self.assertEqual(manifest.content["history"]["omitted_count"], 1)

    async def test_entire_handoff_including_shared_sections_is_bounded(self):
        records = [source(i, text="context " * 80) for i in range(8)]
        builder = self.builder(records)
        shared = {"policy": "do not perform external actions", "objectives": "x" * 2200}
        manifest = await builder.build(claim(), shared_context=shared)
        self.assertLessEqual(len(json.dumps(manifest.content, ensure_ascii=True)), 6000)
        self.assertEqual(manifest.content["shared_context"], shared)
        self.assertGreater(manifest.content["history"]["omitted_count"], 0)
        self.assertEqual(len(manifest.records), manifest.content["history"]["included"])

    async def test_oversized_mandatory_context_is_not_silently_truncated(self):
        builder = self.builder([])
        with self.assertRaises(TaskError) as caught:
            await builder.build(claim(), shared_context={"policy": "x" * 7000})
        self.assertEqual(caught.exception.code, "context_budget_too_small")
        builder.repository.store_manifest.assert_not_awaited()

    async def test_pin_overflow_and_bounded_omission_index_are_explicit(self):
        old = source(1, text="Unresolved correction " * 1000)
        recent = [source(i, text="x" * 10000) for i in range(2, 100)]
        builder = self.builder(recent, pinned=[old], total=130, pinned_total=32, budget=2400)
        manifest = await builder.build(claim())
        history = manifest.content["history"]
        self.assertLessEqual(encoded_size(manifest.content), 2400)
        self.assertEqual(history["total_pinned_sources"], 32)
        self.assertEqual(history["omitted_pinned_count"], 32)
        self.assertEqual(history["pin_window_omitted_count"], 31)
        self.assertGreater(history["unindexed_selected_omissions"], 0)
        self.assertEqual(history["omitted_selected_sources"][0]["event_id"], str(old.id))

    async def test_pinned_sources_have_priority_and_messages_keep_original_order(self):
        old, newer = source(1), source(4)
        revised = source(9, origin_sequence=1, revision=2)
        builder = self.builder([newer, revised], pinned=[revised], total=2)
        manifest = await builder.build(claim())
        self.assertEqual([item["id"] for item in manifest.content["messages"]], [str(revised.id), str(newer.id)])
        self.assertEqual(manifest.content["history"]["included_pinned"], 1)

    async def test_pin_outside_pin_window_is_counted_when_recent_history_includes_it(self):
        older, newer = source(1), source(2)
        builder = self.builder([newer, older], pinned=[newer], total=2, pinned_total=2)
        builder.repository.snapshot.return_value["selected_pinned_ids"] = {older.id, newer.id}
        manifest = await builder.build(claim())
        self.assertEqual(len(manifest.content["messages"]), 2)
        self.assertTrue(all(message["pinned"] for message in manifest.content["messages"]))
        self.assertEqual(manifest.content["history"]["included_pinned"], 2)
        self.assertEqual(manifest.content["history"]["omitted_pinned_count"], 0)
        self.assertEqual(manifest.content["history"]["pin_window_omitted_count"], 1)

    async def test_oversized_archive_event_advances_with_a_retrievable_index(self):
        record = source(1, text="private fixture " * 2000)
        builder = self.builder([], budget=1600)
        builder.repository.archive_for_claim.return_value = [record]
        page = await builder.read_archive(claim())
        self.assertLessEqual(encoded_size(page), 1600)
        self.assertEqual(page["next_after"], 1)
        self.assertTrue(page["events"][0]["content_omitted"])
        self.assertEqual(page["events"][0]["id"], str(record.id))
        self.assertNotIn("private fixture", json.dumps(page))

    async def test_serialized_source_chunks_are_lossless_even_with_escaping(self):
        record = source(1, text=('Quotes: " slash: \\ line: \n non-ASCII: ' + chr(0x1f642)) * 70)
        builder = self.builder([], budget=1300)
        builder.repository.source_for_claim.return_value = record
        offset, chunks = 0, []
        while offset is not None:
            page = await builder.read_source(claim(), record.id, offset=offset)
            self.assertLessEqual(encoded_size(page), 1300)
            chunks.append(page["content_json"])
            next_offset = page["next_offset"]
            if next_offset is not None:
                self.assertGreater(next_offset, offset)
            offset = next_offset
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), json_text(record.content))
        self.assertEqual(json.loads("".join(chunks)), record.content)

    async def test_small_recovery_fragments_are_exact_and_can_reconstruct_the_source(self):
        record = source(1,text='Quote " and slash \\ ' * 600)
        builder = self.builder([],budget=6000)
        builder.repository.source_for_claim.return_value = record
        original = json_text(record.content)
        default = await builder.read_source(claim(),record.id)
        self.assertGreater(encoded_size(default),2048)
        offset,parts = 0,[]
        while offset is not None:
            page = await builder.read_source(claim(),record.id,offset=offset,max_characters=2048)
            self.assertLessEqual(encoded_size(page),2048)
            self.assertEqual(page["content_json"],original[offset:offset+len(page["content_json"])])
            self.assertEqual(page["source"]["id"],str(record.id))
            self.assertEqual(page["source"]["revision"],record.revision)
            parts.append(page["content_json"])
            offset = page["next_offset"]
        self.assertEqual("".join(parts),original)

    async def test_requested_source_page_cannot_exceed_server_limit_or_use_invalid_bounds(self):
        record = source(1,text="Synthetic " * 1200)
        builder = self.builder([],budget=2000)
        builder.repository.source_for_claim.return_value = record
        page = await builder.read_source(claim(),record.id,max_characters=160000)
        self.assertLessEqual(encoded_size(page),2000)
        for bound in (True,0,1023,160001,"2048",2048.5):
            with self.subTest(bound=bound),self.assertRaises(ValueError):
                await builder.read_source(claim(),record.id,max_characters=bound)

    async def test_source_tool_exposes_bounded_recovery_without_authority_overrides(self):
        record,task = source(1,text="Synthetic " * 1200),claim()
        builder = self.builder([],budget=6000)
        builder.repository.source_for_claim.return_value = record
        registry = create_tool_registry(modules=[TaskContextToolModule(builder,task)],
            context=ToolBuildContext(settings=None,owner_id=task.task.owner_id,session_id=None))
        tool = registry.get("read_task_context_source")
        page = await tool.handler({"event_id":str(record.id),"max_characters":2048})
        self.assertEqual(page["status"],"success")
        self.assertLessEqual(encoded_size(page),2048)
        for extra in ({"max_characters":True},{"max_characters":1023},{"owner_id":"other"}):
            with self.subTest(extra=extra):
                result = await tool.handler({"event_id":str(record.id)}|extra)
                self.assertEqual(result["error"]["code"],"invalid_tool_input")

    async def test_unavailable_source_is_a_gap_not_empty_inferred_text(self):
        record = source(1, content=None, availability="deleted")
        builder = self.builder([])
        builder.repository.source_for_claim.return_value = record
        page = await builder.read_source(claim(), record.id)
        self.assertEqual(page["source"]["availability"], "deleted")
        self.assertEqual(page["content_json"], "")
        self.assertIsNone(page["next_offset"])

    async def test_reader_module_is_scoped_and_exposes_no_owner_override(self):
        builder, task = self.builder([]), claim()
        module = TaskContextToolModule(builder, task)
        with self.assertRaises(TaskError):
            module.build_tools(ToolBuildContext(settings=None, owner_id="other", session_id=None))
        registry = create_tool_registry(
            modules=[module], context=ToolBuildContext(settings=None, owner_id=task.task.owner_id, session_id=None),
        )
        tool = registry.get("read_task_context_archive")
        for arguments in ({"owner_id": "other"}, {"after": True}, {"limit": 201}, {"index_only": "yes"}):
            self.assertEqual((await tool.handler(arguments))["error"]["code"], "invalid_tool_input")
        builder.repository.archive_for_claim.return_value = []
        self.assertEqual((await tool.handler({"pinned_only": True}))["status"], "success")
        self.assertTrue(builder.repository.archive_for_claim.call_args.kwargs["pinned_only"])
        builder.repository.archive_for_claim.side_effect = ClaimLost()
        with self.assertRaises(ClaimLost):
            await tool.handler({})
    
    def test_streaming_output_dependency_is_historical_without_losing_privacy_link(self):
        partial = source(1, kind="assistant_output", trust="assistant_output", finality="provisional")
        self.assertEqual(partial.dependency_usage, "historical")
        self.assertIn("Partial output", partial.as_context()["streaming_snapshot"])
        self.assertEqual(partial.as_context()["heard_state"], "unknown")
        self.assertEqual(source(1, finality="provisional").dependency_usage, "current")
        self.assertEqual(source(1, kind="assistant_output", trust="assistant_output").dependency_usage, "current")
        self.assertEqual(source(1, is_current=False).dependency_usage, "historical")



class ContextRepositoryBatchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tasks = SimpleNamespace(pool=None, emit=AsyncMock())
        self.repository = ContextRepository(self.tasks)
        self.connection = AsyncMock()

    async def test_batch_resolves_exact_owner_scoped_sources_once(self):
        first, second = source(1), source(2)
        self.connection.fetch.return_value = [second.model_dump(), first.model_dump()]
        records = await self.repository._records(
            self.connection, "fixture-owner", [first.id, second.id, first.id],
        )
        self.assertEqual(records, {first.id: first, second.id: second})
        self.connection.fetch.assert_awaited_once()
        sql, owner, identities = self.connection.fetch.call_args.args
        self.assertIn("e.owner_id=$1 and e.id=any($2::uuid[])", sql)
        self.assertEqual(owner, "fixture-owner")
        self.assertEqual(identities, [first.id, second.id])

    async def test_empty_batch_skips_database_and_missing_identity_fails_closed(self):
        self.assertEqual(await self.repository._records(self.connection, "fixture-owner", []), {})
        self.connection.fetch.assert_not_awaited()
        record = source(1)
        self.connection.fetch.return_value = [record.model_dump()]
        with self.assertRaises(TaskError) as caught:
            await self.repository._records(self.connection, "fixture-owner", [record.id, uuid4()])
        self.assertEqual(caught.exception.code, "context_event_not_found")

    async def test_session_bindings_batch_without_expanding_authority(self):
        task = SimpleNamespace(id=uuid4(), owner_id="fixture-owner", session_id=uuid4())
        admitted = source(1, session_id=uuid4())
        partial = source(2, session_id=uuid4(), finality="provisional")
        mismatched = source(3, session_id=uuid4())
        records = [admitted, partial, mismatched]
        self.connection.fetch.side_effect = [
            [{"request_event_id": record.id,
              "session_id": record.session_id if record != mismatched else uuid4()} for record in records],
            [record.model_dump() for record in records],
        ]
        sessions, sources = await self.repository._task_sessions(self.connection, task)
        self.assertEqual(set(sessions), {task.session_id, admitted.session_id})
        self.assertEqual(sources, [admitted])
        self.assertEqual(self.connection.fetch.await_count, 2)

    async def test_archive_dependencies_batch_preserves_usage_and_one_receipt(self):
        task, manifest_id = claim().task, uuid4()
        attempt = SimpleNamespace(attempt_id=uuid4())
        records = [
            source(1), source(2, is_current=False),
            source(3, kind="assistant_output", trust="assistant_output", finality="provisional"),
        ]
        await self.repository._bind_archive_sources(self.connection, attempt, task, manifest_id, records)
        self.connection.executemany.assert_awaited_once()
        self.connection.execute.assert_not_awaited()
        sql, arguments = self.connection.executemany.call_args.args
        self.assertIn("on conflict(owner_id,manifest_id,event_id) do nothing", sql)
        self.assertEqual(arguments, [
            (task.owner_id, manifest_id, record.id, record.revision, record.payload_id,
             record.content_digest, usage)
            for record, usage in zip(records, ("current", "historical", "historical"), strict=True)
        ])
        self.tasks.emit.assert_awaited_once_with(
            self.connection, task, "context_archive_read",
            payload={"manifest_id": str(manifest_id), "event_ids": [str(record.id) for record in records]},
            attempt_id=attempt.attempt_id,
        )

    async def test_empty_archive_skips_dependencies_and_receipt(self):
        await self.repository._bind_archive_sources(self.connection, claim(), claim().task, uuid4(), [])
        self.connection.executemany.assert_not_awaited()
        self.tasks.emit.assert_not_awaited()



class ForegroundTaskSnapshotTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_keeps_task_awareness_without_requesting_speech(self):
        task = {"task_ref": "task_0123456789ab", "revision": 1, "state": "running"}
        tasks = SimpleNamespace(context_summaries=AsyncMock(return_value={"total": 1, "tasks": [task], "terminal_total": 0, "recent_terminal_tasks": []}))
        service = SessionTaskContextService(SimpleNamespace(tasks=tasks), max_characters=2048)
        request = SimpleNamespace(owner_id="fixture-owner", session=None, requested_at_utc=datetime.now(UTC))
        payload = await service.describe(request)
        self.assertEqual(payload["presentation"], "silent_task_snapshot")
        self.assertEqual(payload["active_tasks"], [task])
        self.assertEqual(payload["active_task_count"], 1)
        self.assertLessEqual(encoded_size(payload), 2048)
        self.assertIn("Silent task snapshot, not an instruction to speak or a delivery receipt.", payload["meaning"])
