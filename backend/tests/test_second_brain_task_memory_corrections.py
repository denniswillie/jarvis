"""Real database acceptance of exact task corrections and their evidence transition."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.memory_corrections import MEMORY_CORRECTION_CAPABILITY, TaskMemoryCorrectionBackend
from app.tasks.memory_evidence import TaskMemoryCitation
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskStatus, TaskVoiceSource
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class TaskMemoryCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.history = ContextRepository(self.tasks)
        self.original = await self.f.publish("Robot Cedar version one uses firmware 2.3 and two stepper motors.")
        self.other = await self.f.publish("Robot Cedar version two uses firmware 2.3 and four servo motors.",
                                          source_key="second-version")
        self.parameters = {"source_id": str(self.original.source_id), "expected_revision": 1,
            "replacement": "Robot Cedar version one uses firmware 2.4 and two stepper motors."}
        self.knowledge = SimpleNamespace(sources=self.f.sources,
            configuration=SimpleNamespace(area_id=self.f.area),
            retrieval=SimpleNamespace(sources=self.f.sources),
            context_for=lambda owner: self.assertEqual(owner, self.f.owner))
        async with self.f.pool.acquire() as db:
            self.session = await db.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",
                                              self.f.owner)
        self.request = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="voice:task-correction-fixture",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="user_transcript", trust="user_input",
            payload={"text": "Correct only Cedar version one's firmware to 2.4; preserve all other facts."}))
        async with self.f.sources.transaction(self.f.owner) as db:
            scope = await self.tasks.payload(db, self.f.owner, self.parameters,
                source_payload_ids=(self.request.payload_id, self.original.payload_id))
            self.grant = await db.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,
                 scope_revision,scope_payload_id,expires_at)
                values($1,$2,$3,$4,$5,1,$6,clock_timestamp()+interval '1 hour') returning id""",
                self.f.owner, MEMORY_CORRECTION_CAPABILITY, uuid4().hex, self.request.id, self.request.revision, scope)

    async def prepared(self, *, correction=True, model="fixture-model"):
        task = await self.tasks.create(self.f.owner, TaskCreate(
            request_id=uuid4(), title="Synthetic memory correction" if correction else "Independent reader",
            requested_outcome="Apply the exact authorized correction." if correction else "Read the source only.",
            session_id=self.session, knowledge_area_id=self.f.area,
            grant_ids=(self.grant,) if correction else ()), model=model,
            source=TaskVoiceSource(self.request.id, self.request.revision, self.session), mode="background")
        claim = await self.tasks.claim_next("task-correction-fixture", ("frontier",))
        self.assertEqual(claim.task.id, task.id)
        context = TaskRunner(self.tasks, {}).context(claim)
        prepared = await WorkerContextFactory(self.f.pool, knowledge=self.knowledge)(context)
        return context, prepared

    async def read(self, prepared, source=None):
        result = await prepared.tools.invoke("explain_memory",
            {"memory_id": str((source or self.original).source_id)}, provider_call_id=uuid4().hex)
        self.assertEqual(result["status"], "success")
        return result

    async def correct(self, prepared):
        return await prepared.tools.invoke("correct_memory_source",
            {"operation_ref": "op_" + self.grant.hex}, provider_call_id=uuid4().hex)

    async def current(self):
        return await self.f.sources.read(self.f.owner, self.f.area, self.original.source_id)

    async def bind_foreground_read(self, *sources):
        event = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="voice:fixture-memory-read",
            source_event_id=uuid4().hex, receipt_id=uuid4().hex, kind="tool_outcome", trust="tool_receipt",
            source_payload_ids=tuple(source.payload_id for source in sources),
            payload={"tool": "search_memory", "marker": "obsolete-source-copy-only",
                     "results": [source.original_evidence() for source in sources]}))
        async with self.f.pool.acquire() as db:
            for source in sources:
                await db.execute("""insert into jarvis_context_memory_sources
                    (owner_id,dependent_payload_id,area_id,source_id,source_revision,source_generation,
                     source_payload_id,content_digest,source_usage)
                    values($1,$2,$3,$4,$5,$6,$7,$8,'current')""",
                    self.f.owner,event.payload_id,self.f.area,source.source_id,source.revision,
                    source.generation,source.payload_id,source.content_digest)
        return event

    async def test_self_correction_rebuilds_retired_foreground_reads_without_resurrecting_them(self):
        import json
        event = await self.bind_foreground_read(self.original, self.other)
        reader, rp = await self.prepared(correction=False)
        ctx, prepared = await self.prepared()
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual((await self.current()).revision, 2)
        self.assertEqual((await self.current()).content["text"], self.parameters["replacement"])
        self.assertEqual((await self.correct(prepared))["task_operation"]["replayed"], True)
        async with self.f.pool.acquire() as db:
            old = await db.fetchrow("""select m.status,p.availability,p.content
                from jarvis_task_context_manifests m join jarvis_context_payloads p
                on p.owner_id=m.owner_id and p.id=m.payload_id where m.id=$1""", prepared.manifest.id)
            self.assertEqual(old["status"], "revoked")
            self.assertEqual(old["availability"], "deleted")
            self.assertIsNone(old["content"])
            self.assertEqual(await db.fetchval("select availability from jarvis_context_payloads where id=$1",
                                              event.payload_id), "unavailable")
            self.assertEqual(await db.fetchval("select status from jarvis_task_context_manifests where id=$1",
                                              rp.manifest.id), "revoked")
            manifest_id = await db.fetchval("select context_manifest_id from jarvis_task_attempts where id=$1",
                                            ctx.claim.attempt_id)
        successor = await self.history.read_manifest(self.f.owner, ctx.task.id, manifest_id)
        message = next(row for row in successor.content["messages"] if row["id"] == str(event.id))
        self.assertIsNone(message["content"])
        self.assertEqual(message["availability"], "unavailable")
        self.assertNotIn("obsolete-source-copy-only", json.dumps(successor.content))
        self.assertIn(str(event.id), successor.content["local_memory_correction"]["retired_context_event_ids"])
        for citation in [result["task_evidence"], *result["historical_evidence"]]:
            await prepared.tools.verified_memory(TaskMemoryCitation(**citation))
        with self.assertRaises(ClaimLost):
            await rp.tools._check()
        # Version two was seen only through the foreground read, not an extra
        # worker read. Its original remains a dependency after retiring the copy.
        await self.f.sources.delete_source(self.f.owner, self.f.area, self.other.source_id, expected_revision=1)
        with self.assertRaises(ClaimLost):
            await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def test_self_correction_cannot_hide_an_unrelated_context_removal(self):
        from app.knowledge.lifecycle import SourceLifecycleService
        await self.bind_foreground_read(self.original)
        unrelated = await self.bind_foreground_read(self.other)
        _, prepared = await self.prepared()
        actual = SourceLifecycleService.correct_in_transaction
        async def changed(service, connection, *args, **kwargs):
            result = await actual(service, connection, *args, **kwargs)
            await connection.execute("""update jarvis_context_payloads
                set availability='deleted',content=null where owner_id=$1 and id=$2""",
                self.f.owner, unrelated.payload_id)
            return result
        with patch.object(SourceLifecycleService, "correct_in_transaction", changed):
            with self.assertRaises(TaskError) as caught:
                await self.correct(prepared)
        self.assertEqual(caught.exception.code, "memory_correction_context_unavailable")
        self.assertEqual((await self.current()).revision, 1)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions where status='accepted'"), 0)

    async def prior_expert_question(self, *, unrelated=False):
        from app.attention.repository import AttentionRepository
        from app.tasks.models import json_object
        ctx, reader = await self.prepared(correction=False)
        citation = (await self.read(reader, self.other if unrelated else self.original))["task_evidence"]
        if not unrelated:
            await self.read(reader, self.other)
        outcome = await HermesTaskResult(status="waiting_for_input",
            answer="Which Cedar version should be corrected?",
            speech=[{"text": "Which Cedar version should be corrected?"}],
            evidence=[citation], completed_operation_refs=[],
            uncertainties=["Version unspecified."]).verified(ctx, reader.tools)
        self.assertTrue(await self.tasks.finish(ctx.claim, outcome))
        task = await self.tasks.get(self.f.owner, ctx.task.id)
        delivery = await AttentionRepository(self.tasks).claim_next(self.f.owner, self.session, "fixture-question")
        self.assertEqual(delivery.source_payload_id, task.result_payload_id)
        # The live task publisher admits this exact original to conversation
        # history; omitting that link would miss the production failure.
        async with self.f.pool.acquire() as db:
            async with db.transaction():
                payload = json_object(await db.fetchval(
                    "select content from jarvis_context_payloads where id=$1", task.result_payload_id))
                event = await self.history.append_in_transaction(db, self.f.owner, ContextIngress(
                    session_id=self.session, source_namespace="fixture:prior-expert",
                    source_event_id=uuid4().hex, receipt_id=uuid4().hex,
                    kind="tool_outcome", trust="tool_receipt", payload=payload),
                    payload_id=task.result_payload_id)
        return task, delivery, event

    async def assert_correction_preserves_revoked_question_gap(self, *, revoked_before):
        import asyncpg
        import json
        task, delivery, event = await self.prior_expert_question()
        if revoked_before:
            async with self.f.pool.acquire() as db:
                await db.execute("update jarvis_attention_intents set state='revoked' where id=$1",
                                 delivery.intent_id)
        ctx, prepared = await self.prepared()
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual((await self.current()).revision, 2)
        self.assertTrue((await self.correct(prepared))["task_operation"]["replayed"])
        async with self.f.pool.acquire() as db:
            manifest_id = await db.fetchval(
                "select context_manifest_id from jarvis_task_attempts where id=$1", ctx.claim.attempt_id)
            self.assertEqual(await db.fetchval(
                "select availability from jarvis_context_payloads where id=$1", task.result_payload_id), "available")
            self.assertEqual(await db.fetchval(
                "select state from jarvis_attention_intents where id=$1", delivery.intent_id), "revoked")
            self.assertEqual(await db.fetchval(
                "select lock_jarvis_copy_source($1,$2)", self.f.owner, task.result_payload_id), "revoked")
            with self.assertRaises(asyncpg.CheckViolationError):
                async with db.transaction():
                    await self.tasks.payload(db, self.f.owner, {"text": "Forbidden replay"},
                                             source_payload_ids=(task.result_payload_id,))
        successor = await self.history.read_manifest(self.f.owner, ctx.task.id, manifest_id)
        question = next(row for row in successor.content["messages"] if row["id"] == str(event.id))
        self.assertEqual(question["availability"], "unavailable")
        self.assertIsNone(question["content"])
        self.assertNotIn("Which Cedar version should be corrected?", json.dumps(successor.content))
        self.assertIn(str(event.id), successor.content["local_memory_correction"]["retired_context_event_ids"])
        for citation in [result["task_evidence"], *result["historical_evidence"]]:
            await prepared.tools.verified_memory(TaskMemoryCitation(**citation))
        if not revoked_before:
            # This dependency was learned only through the prior task's
            # manifest, not through a direct read by the correcting task.
            await self.f.sources.delete_source(self.f.owner, self.f.area, self.other.source_id, expected_revision=1)
            with self.assertRaises(ClaimLost):
                await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def test_correction_retires_its_prior_question_without_copying_the_revoked_answer(self):
        await self.assert_correction_preserves_revoked_question_gap(revoked_before=False)

    async def test_correction_keeps_an_already_revoked_question_as_an_explicit_gap(self):
        await self.assert_correction_preserves_revoked_question_gap(revoked_before=True)

    async def test_correction_cannot_hide_unrelated_answer_revocation(self):
        from app.knowledge.lifecycle import SourceLifecycleService
        _, delivery, _ = await self.prior_expert_question(unrelated=True)
        _, prepared = await self.prepared()
        actual = SourceLifecycleService.correct_in_transaction
        async def changed(service, connection, *args, **kwargs):
            result = await actual(service, connection, *args, **kwargs)
            await connection.execute("update jarvis_attention_intents set state='revoked' where id=$1",
                                     delivery.intent_id)
            return result
        with patch.object(SourceLifecycleService, "correct_in_transaction", changed):
            with self.assertRaises(TaskError) as caught:
                await self.correct(prepared)
        self.assertEqual(caught.exception.code, "memory_correction_context_unavailable")
        self.assertEqual((await self.current()).revision, 1)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions where status='accepted'"), 0)

    async def test_already_redacted_question_can_become_deleted_during_correction(self):
        await self.bind_foreground_read(self.original)
        task, delivery, event = await self.prior_expert_question()
        copied = await self.history.append(self.f.owner, ContextIngress(
            session_id=self.session, source_namespace="voice:fixture-redacted-question",
            source_event_id="question", receipt_id="question", kind="assistant_output", trust="assistant_output",
            source_payload_ids=(task.result_payload_id,),
            payload={"text": "Which Cedar version?", "authored_by": "frontier_expert",
                     "attempt_id": str(delivery.attempt_id)}))
        async with self.f.pool.acquire() as db:
            await db.execute("update jarvis_attention_intents set state='revoked' where id=$1", delivery.intent_id)
        self.assertEqual((await self.history.get(self.f.owner, copied.id)).availability, "unavailable")
        ctx, prepared = await self.prepared()
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            manifest_id = await db.fetchval(
                "select context_manifest_id from jarvis_task_attempts where id=$1", ctx.claim.attempt_id)
        successor = await self.history.read_manifest(self.f.owner, ctx.task.id, manifest_id)
        for identity in (event.id, copied.id):
            message = next(row for row in successor.content["messages"] if row["id"] == str(identity))
            self.assertEqual(message["availability"], "deleted")
            self.assertIsNone(message["content"])
        self.assertIsNone((await self.history.get(self.f.owner, copied.id)).content)
        await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def derived_playback_memory(self, *, unrelated=False):
        from app.knowledge.publication import CanonicalSourcePublication
        task, _, _ = await self.prior_expert_question(unrelated=unrelated)
        publisher = ContextRepository(self.tasks,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        # Deliberately outside the selected conversation window: a memory read
        # must carry its own lineage, including older telemetry revisions.
        derived = await publisher.append(self.f.owner, ContextIngress(
            session_id=None, source_namespace="fixture:playback-memory", source_event_id=uuid4().hex,
            receipt_id=uuid4().hex, kind="played_output_boundary", trust="sensor",
            source_payload_ids=(task.result_payload_id,),
            payload={"played_samples": 12000, "boundary": "played_sample_lower_bound"}))
        async with self.f.pool.acquire() as db:
            source_id = await db.fetchval("""select source_id from jarvis_knowledge_source_versions
                where owner_id=$1 and payload_id=$2""", self.f.owner, derived.payload_id)
        return await self.f.sources.read(self.f.owner, self.f.area, source_id)

    async def test_correction_retires_derived_memory_but_keeps_its_original_evidence(self):
        from uuid import UUID
        memory = await self.derived_playback_memory()
        ctx, prepared = await self.prepared()
        stale = (await self.read(prepared, memory))["task_evidence"]
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual((await self.current()).revision, 2)
        self.assertIn(str(memory.source_id), {row["source_id"] for row in result["retired_memory_evidence"]})
        with self.assertRaises(SourceError):
            await prepared.tools.verified_memory(TaskMemoryCitation(**stale))
        await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))
        self.assertTrue((await self.correct(prepared))["task_operation"]["replayed"])
        async with self.f.pool.acquire() as db:
            manifest_id = UUID(result["context_manifest_id"])
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_task_memory_sources
                where manifest_id=$1 and source_id=$2""", manifest_id, memory.source_id), 0)
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_task_memory_sources
                where manifest_id=$1 and source_id=$2""", manifest_id, self.other.source_id), 1)
            self.assertEqual(await db.fetchval(
                "select availability from jarvis_context_payloads where id=$1", memory.payload_id), "unavailable")
        await self.f.sources.delete_source(self.f.owner, self.f.area, self.other.source_id, expected_revision=1)
        with self.assertRaises(ClaimLost):
            await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def test_correction_cannot_drop_unrelated_derived_memory(self):
        from app.knowledge.lifecycle import SourceLifecycleService
        memory = await self.derived_playback_memory(unrelated=True)
        _, prepared = await self.prepared()
        await self.read(prepared, memory)
        actual = SourceLifecycleService.correct_in_transaction
        async def changed(service, connection, *args, **kwargs):
            result = await actual(service, connection, *args, **kwargs)
            await connection.execute("""update jarvis_context_payloads
                set availability='deleted',content=null where owner_id=$1 and id=$2""",
                self.f.owner, memory.payload_id)
            return result
        with patch.object(SourceLifecycleService, "correct_in_transaction", changed):
            with self.assertRaises(TaskError) as caught:
                await self.correct(prepared)
        self.assertEqual(caught.exception.code, "memory_correction_context_unavailable")
        self.assertEqual((await self.current()).revision, 1)

    async def test_local_correction_receipt_and_successor_preserve_other_reader_fence(self):
        reader, rp = await self.prepared(correction=False)
        reader_evidence = (await self.read(rp))["task_evidence"]
        ctx, prepared = await self.prepared()
        prior = (await self.read(prepared))["task_evidence"]
        await self.read(prepared, self.other)
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        current = await self.current()
        self.assertEqual(current.content["text"], self.parameters["replacement"])
        self.assertEqual(current.revision, 2)
        self.assertEqual(result["task_evidence"]["source_usage"], "current")
        self.assertEqual(result["historical_evidence"][0]["source_usage"], "historical")
        self.assertEqual(result["historical_evidence"][0]["generation"], 2)
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, self.other.source_id)).content,
                         self.other.content)
        with self.assertRaises(ClaimLost):
            await rp.tools.verified_memory(TaskMemoryCitation(**reader_evidence))
        with self.assertRaises(SourceError):
            await prepared.tools.verified_memory(TaskMemoryCitation(**prior))
        for citation in [result["task_evidence"], *result["historical_evidence"]]:
            await prepared.tools.verified_memory(TaskMemoryCitation(**citation))
        outcome = await HermesTaskResult.model_validate({
            "status": "succeeded", "answer": "Only version one's firmware was corrected.",
            "speech": [{"text": "Only version one's firmware was corrected."}],
            "evidence": [result["task_evidence"], *result["historical_evidence"]],
            "completed_operation_refs": [result["task_operation"]["operation_ref"]], "uncertainties": [],
        }).verified(ctx, prepared.tools)
        self.assertTrue(await self.tasks.finish(ctx.claim, outcome))
        task = await self.tasks.get(self.f.owner, ctx.task.id)
        self.assertEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertIsNotNone(task.result_payload_id)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval(
                "select status from jarvis_task_context_manifests where id=$1", prepared.manifest.id), "superseded")
            self.assertEqual(await db.fetchval(
                "select status from jarvis_task_context_manifests where id=$1", rp.manifest.id), "superseded")
            self.assertEqual(await db.fetchval(
                "select count(*) from jarvis_task_actions where status='accepted'"), 1)
        await self.f.sources.delete_source(self.f.owner, self.f.area, self.original.source_id, expected_revision=2)
        self.assertIsNone((await self.tasks.get(self.f.owner, ctx.task.id)).result_payload_id)

    async def test_retries_and_concurrent_call_ids_do_not_repeat_correction(self):
        ctx, prepared = await self.prepared()
        await self.read(prepared)
        results = await asyncio.gather(self.correct(prepared), self.correct(prepared))
        self.assertTrue(all(row["status"] in {"success", "outcome_unknown"} for row in results))
        replay = await self.correct(prepared)
        self.assertEqual(replay["status"], "success")
        self.assertTrue(replay["task_operation"]["replayed"])
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 1)
            self.assertEqual(await db.fetchval("""select count(*) from jarvis_context_events
                where source_namespace='knowledge:source-correction'"""), 1)
        self.assertEqual((await ctx.budgets.usage(self.f.owner, ctx.task.id))["steps"], 2)

    async def test_receipt_failure_rolls_back_correction_and_manifest_transition(self):
        ctx, prepared = await self.prepared()
        await self.read(prepared)
        record = ctx.actions.record_receipt_in_transaction
        async def fail(connection, owner, task, action, token, status, receipt):
            if status == "accepted":
                raise RuntimeError("Synthetic receipt failure")
            return await record(connection, owner, task, action, token, status, receipt)
        with patch.object(ctx.actions, "record_receipt_in_transaction", fail):
            with self.assertRaisesRegex(RuntimeError, "Synthetic receipt failure"):
                await self.correct(prepared)
        self.assertEqual((await self.current()).revision, 1)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_context_manifests"), 1)
            self.assertEqual(await db.fetchval(
                "select status from jarvis_task_context_manifests where id=$1", prepared.manifest.id), "active")
            self.assertEqual(await db.fetchval(
                "select count(*) from jarvis_context_events where source_namespace='knowledge:source-correction'"), 0)
        self.assertEqual((await self.correct(prepared))["status"], "outcome_unknown")

    async def test_cancelled_acknowledgment_does_not_erase_committed_correction(self):
        ctx, prepared = await self.prepared()
        await self.read(prepared)
        committed, release = asyncio.Event(), asyncio.Event()
        real = TaskMemoryCorrectionBackend.correct
        async def paused(backend, **arguments):
            result = await real(backend, **arguments)
            committed.set()
            await release.wait()
            return result
        with patch.object(TaskMemoryCorrectionBackend, "correct", paused):
            running = asyncio.create_task(self.correct(prepared))
            try:
                await asyncio.wait_for(committed.wait(), 20)
                self.assertEqual((await self.correct(prepared))["status"], "success")
                running.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await running
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
        self.assertEqual((await self.current()).revision, 2)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select status from jarvis_task_actions"), "accepted")

    async def test_ungranted_correction_cannot_change_source(self):
        _, readonly = await self.prepared(correction=False)
        self.assertNotIn("correct_memory_source", {tool["name"] for tool in readonly.tools.catalog()})
        self.assertEqual((await self.correct(readonly))["status"], "failed")
        self.assertEqual((await self.current()).revision, 1)

    async def test_exact_grant_binds_original_atomically_without_a_prior_model_read(self):
        _, prepared = await self.prepared()
        result = await self.correct(prepared)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task_operation"]["state"], "accepted")
        self.assertEqual((await self.current()).content["text"], self.parameters["replacement"])
        self.assertEqual(result["task_evidence"]["revision"], 2)
        self.assertEqual(result["historical_evidence"][0]["revision"], 1)
        for citation in [result["task_evidence"], *result["historical_evidence"]]:
            await prepared.tools.verified_memory(TaskMemoryCitation(**citation))

    async def test_direct_invocation_cannot_expand_the_exact_grant(self):
        _, prepared = await self.prepared()
        result = await prepared.tools.invoke("correct_memory_source",
            {"operation_ref": "op_" + self.grant.hex, "replacement": "Unapproved changed text"},
            provider_call_id=uuid4().hex)
        self.assertEqual(result["status"], "failed")
        self.assertEqual((await self.current()).revision, 1)
        async with self.f.pool.acquire() as db:
            self.assertEqual(await db.fetchval("select count(*) from jarvis_task_actions"), 0)

    async def test_atomic_original_read_still_rejects_a_stale_granted_revision(self):
        _, prepared = await self.prepared()
        await self.f.publish("An independently corrected source.", expected_revision=1)
        with self.assertRaises(SourceError) as caught:
            await self.correct(prepared)
        self.assertEqual(caught.exception.code, "knowledge_source_revision_conflict")
        self.assertEqual((await self.current()).content["text"], "An independently corrected source.")
        self.assertEqual((await self.current()).revision, 2)

    async def test_independent_source_change_cannot_be_hidden_by_local_correction(self):
        _, prepared = await self.prepared()
        await self.read(prepared)
        await self.read(prepared, self.other)
        await self.f.publish("Version two was independently updated.", source_key="second-version", expected_revision=1)
        with self.assertRaises(ClaimLost):
            await self.correct(prepared)
        self.assertEqual((await self.current()).revision, 1)

    async def test_successor_still_revokes_on_unrelated_evidence_deletion(self):
        _, prepared = await self.prepared()
        await self.read(prepared)
        await self.read(prepared, self.other)
        result = await self.correct(prepared)
        await self.f.sources.delete_source(self.f.owner, self.f.area, self.other.source_id, expected_revision=1)
        with self.assertRaises(ClaimLost):
            await prepared.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    @unittest.skipUnless(os.getenv("JARVIS_TEST_NATIVE_CORRECTIONS") == "1",
                         "Explicit real Hermes/Hindsight correction acceptance required")
    async def test_native_worker_corrects_and_finishes_with_current_evidence(self):
        import time
        from app.knowledge.runtime import KnowledgeRuntime
        from app.runtime.feature_config import DEFAULT_FRONTIER_MODEL, FrontierConfiguration, KnowledgeConfiguration
        from app.settings import JarvisSettings
        from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
        from evals.second_brain.hindsight_semantic_trace import drain
        from evals.second_brain.hindsight_service import save
        root = Path(os.environ["JARVIS_NATIVE_CORRECTION_ROOT"])
        report = {"model": DEFAULT_FRONTIER_MODEL, "worker_events": [], "tools": [],
                  "scope": "exact_preauthorized_correction_not_foreground_admission"}
        self.knowledge = await KnowledgeRuntime.create(pool=self.f.pool, owner_id=self.f.owner, tasks=self.tasks,
            configuration=KnowledgeConfiguration(base_url=os.environ["JARVIS_S04_MEMORY_URL"],
                api_key=os.environ["JARVIS_S04_MEMORY_TOKEN"], area_id=self.f.area))
        try:
            await drain(self.f, self.knowledge.worker, report, label="native-exact-correction-seed")
            ctx, _ = await self.prepared(model=DEFAULT_FRONTIER_MODEL)
            factory = WorkerContextFactory(self.f.pool, knowledge=self.knowledge)
            async def prepare(context):
                prepared = await factory(context)
                actual_invoke = prepared.tools.invoke
                async def observe(name, arguments, *, provider_call_id):
                    event = {"name": name, "arguments": arguments}
                    report["tools"].append(event)
                    try:
                        result = await actual_invoke(name, arguments, provider_call_id=provider_call_id)
                        event["result"] = result
                        return result
                    except Exception as error:
                        event["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
                        raise
                prepared.tools.invoke = observe
                return prepared
            provider = HermesTaskProvider(FrontierConfiguration(model=DEFAULT_FRONTIER_MODEL,
                reasoning_effort="medium", api_key=JarvisSettings().OPENAI_API_KEY),
                HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]), root / "hermes-state"),
                prepare=prepare)
            started = time.monotonic()
            outcome = await asyncio.wait_for(provider.execute(ctx), 240)
            report.update(worker_seconds=time.monotonic() - started,
                outcome={"status": outcome.status.value, "payload": outcome.payload},
                usage=await ctx.budgets.usage(self.f.owner, ctx.task.id))
            accepted = await self.tasks.finish(ctx.claim, outcome)
            current = await self.current()
            report["current_original"] = current.original_evidence()
            report["task_result_available"] = (await self.tasks.get(self.f.owner, ctx.task.id)).result_payload_id is not None
            self.assertTrue(accepted)
            self.assertEqual(outcome.status, TaskStatus.SUCCEEDED)
            self.assertEqual(current.content["text"], self.parameters["replacement"])
            self.assertEqual(current.revision, 2)
            self.assertTrue(report["task_result_available"])
            self.assertEqual(outcome.payload["completed_operation_refs"], ["op_" + self.grant.hex])
            self.assertIn("correct_memory_source", [row["name"] for row in outcome.payload["tool_receipts"]])
            self.assertTrue(any(row.get("source_id") == str(current.source_id) and row["revision"] == 2
                                for row in outcome.payload["evidence"]))
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "code": getattr(error, "code", None)}
            raise
        finally:
            await self.knowledge.close()
            save(root / "native-correction-result.json", report)
