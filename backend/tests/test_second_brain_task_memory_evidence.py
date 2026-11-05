"""Exact selected-memory evidence through production task/tool handlers."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from pydantic import ValidationError

from app.knowledge.models import SourceError
from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import KnowledgeConfiguration, TaskLimits, FrontierConfiguration
from app.settings import JarvisSettings
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.memory_evidence import TaskMemoryCitation
from app.tasks.models import ClaimLost, TaskCreate, TaskError, TaskStatus, WorkerOutcome
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.source_fixture import source_fixture


class MemoryCitationTests(unittest.IsolatedAsyncioTestCase):
    def result(self, evidence):
        return HermesTaskResult.model_validate({
            "status":"succeeded","answer":"Synthetic source-backed answer.",
            "speech":[{"text":"Synthetic source-backed answer."}],
            "evidence":evidence,"completed_operation_refs":[],"uncertainties":[],
        })

    def citation(self):
        return dict(source_id=str(uuid4()),revision=1,generation=1,
                    content_digest="a"*64,source_usage="current")

    async def test_memory_citations_use_selected_evidence_verifier_not_event_lookup(self):
        bridge = SimpleNamespace(verified_memory=AsyncMock())
        ctx = SimpleNamespace(tasks=SimpleNamespace(pool=None),
            task=SimpleNamespace(id=uuid4(),owner_id="fixture-owner",model="fixture-model"),
            claim=SimpleNamespace(task_revision=1,attempt_id=uuid4()))
        citation = self.citation()
        outcome = await self.result([citation]).verified(ctx,bridge)
        bridge.verified_memory.assert_awaited_once_with(TaskMemoryCitation(**citation))
        self.assertEqual(outcome.payload["evidence"],[citation])

    async def test_unread_memory_citation_cannot_claim_success(self):
        bridge = SimpleNamespace(verified_memory=AsyncMock(side_effect=TaskError("not_read","Not read.")))
        with self.assertRaises(TaskError):
            await self.result([self.citation()]).verified(SimpleNamespace(tasks=SimpleNamespace(pool=None)),bridge)

    def test_memory_ids_cannot_be_mixed_with_event_ids_or_omit_provenance(self):
        citation = self.citation()
        for value in (citation|{"event_id":str(uuid4())},
                      {key:value for key,value in citation.items() if key!="generation"},
                      citation|{"content_digest":"not-a-digest"}):
            with self.subTest(value=value),self.assertRaises(ValidationError):
                self.result([value])

    def test_all_citation_schema_fields_are_required(self):
        schema = HermesTaskResult.model_json_schema()
        for name in ("TaskMemoryCitation","TaskEvidenceCitation"):
            definition = schema["$defs"][name]
            self.assertEqual(set(definition["required"]),set(definition["properties"]))
            self.assertFalse(definition["additionalProperties"])


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested second-brain database required")
class TaskMemoryEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],self.root,test_name=self.id(),
        ))
        self.tasks = TaskRepository(self.f.pool,TaskLimits(
            lease_seconds=300,request_output_tokens=4096,task_output_tokens=20000,task_input_tokens=100000,
        ))
        self.source = await self.f.publish("I paused binary search; it is incomplete. Synthetic marker "+uuid4().hex)
        self.knowledge = SimpleNamespace(
            sources=self.f.sources,configuration=SimpleNamespace(area_id=self.f.area),
            retrieval=SimpleNamespace(sources=self.f.sources),
            context_for=lambda owner: self.assertEqual(owner,self.f.owner),
        )

    async def prepared(self, *, knowledge=None, use_default_area=False, outcome="Recall my reported coding progress."):
        task = await self.tasks.create(self.f.owner,TaskCreate(
            request_id=uuid4(),title="Synthetic memory task",requested_outcome=outcome,
            knowledge_area_id=None if use_default_area else self.f.area,
        ),model=os.getenv("JARVIS_S04_MODEL","fixture-model"))
        claim = await self.tasks.claim_next("memory-evidence-fixture",("frontier",))
        self.assertEqual(task.id,claim.task.id)
        ctx = TaskRunner(self.tasks,{}).context(claim)
        prepared = await WorkerContextFactory(self.f.pool,knowledge=knowledge or self.knowledge)(ctx)
        return ctx,prepared

    async def read(self, prepared, **extra):
        return await prepared.tools.invoke("explain_memory",
            {"memory_id":str(self.source.source_id),**extra},provider_call_id="fixture-"+uuid4().hex)

    async def test_actual_shared_explain_binds_exact_original_and_verifies_result(self):
        ctx,p = await self.prepared()
        result = await self.read(p)
        self.assertEqual(result["status"],"success")
        self.assertIn(self.source.content["text"],result["content_page"]["text"])
        citation = result["task_evidence"]
        self.assertEqual(citation["source_id"],str(self.source.source_id))
        self.assertNotIn("event_id",citation)
        outcome = await HermesTaskResult.model_validate(dict(
            status="succeeded",answer="You reported pausing, not completing the code.",
            speech=[{"text":"You reported pausing, not completing the code."}],
            evidence=[citation],completed_operation_refs=[],uncertainties=[],
        )).verified(ctx,p.tools)
        self.assertTrue(await self.tasks.finish(ctx.claim,outcome))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_memory_sources"),1)

    async def test_final_citation_cannot_bind_a_source_that_the_task_never_read(self):
        _,p = await self.prepared()
        citation = TaskMemoryCitation.from_metadata(self.source.original_evidence())
        with self.assertRaises(TaskError) as error:
            await p.tools.verified_memory(citation)
        self.assertEqual(error.exception.code,"task_memory_evidence_not_read")

    async def test_corrected_source_fences_running_task_and_old_final_answer(self):
        ctx,p = await self.prepared()
        result = await self.read(p)
        await self.f.publish("Corrected: I restarted the exercise.",expected_revision=1)
        with self.assertRaises(ClaimLost):
            await p.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))
        await self.tasks.finish(ctx.claim,WorkerOutcome(TaskStatus.SUCCEEDED,{"answer":"obsolete"}))
        task = await self.tasks.get(self.f.owner,ctx.task.id)
        self.assertEqual(task.status,TaskStatus.PAUSED)
        self.assertIsNone(task.result_payload_id)

    async def test_correction_removes_completed_result_but_retains_historical_audit(self):
        ctx,p = await self.prepared()
        await self.read(p)
        await self.tasks.finish(ctx.claim,WorkerOutcome(TaskStatus.SUCCEEDED,{"answer":self.source.content["text"]}))
        task = await self.tasks.get(self.f.owner,ctx.task.id)
        old_payload = task.result_payload_id
        self.assertIsNotNone(old_payload)
        await self.f.publish("I corrected the earlier report.",expected_revision=1)
        self.assertIsNone((await self.tasks.get(self.f.owner,ctx.task.id)).result_payload_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select availability from jarvis_context_payloads where id=$1",old_payload),"available")

    async def test_deletion_removes_original_manifest_checkpoints_and_completed_answer(self):
        ctx,p = await self.prepared()
        await self.read(p)
        await self.tasks.checkpoint(ctx.claim,{"answer":self.source.content["text"]})
        await self.tasks.finish(ctx.claim,WorkerOutcome(TaskStatus.SUCCEEDED,{"answer":self.source.content["text"]}))
        await self.f.sources.delete_source(self.f.owner,self.f.area,self.source.source_id,expected_revision=1)
        self.assertIsNone((await self.tasks.get(self.f.owner,ctx.task.id)).result_payload_id)
        async with self.f.pool.acquire() as connection:
            count = await connection.fetchval(
                "select count(*) from jarvis_context_payloads where owner_id=$1 and content::text like $2",
                self.f.owner,"%"+self.source.content["text"]+"%")
            self.assertEqual(count,0)
            self.assertEqual(await connection.fetchval(
                "select status from jarvis_task_context_manifests where id=$1",p.manifest.id),"revoked")

    async def test_explicit_historical_read_is_labeled_and_deletion_still_wins(self):
        await self.f.publish("Current corrected statement.",expected_revision=1)
        ctx,p = await self.prepared()
        result = await self.read(p,revision=1)
        self.assertEqual(result["task_evidence"]["source_usage"],"historical")
        await p.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))
        # Direct removal of an old original must revoke its readers even when
        # the source's current version uses another payload.
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                                     self.source.payload_id)
        with self.assertRaises(ClaimLost):
            await p.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def test_stale_metadata_foreign_area_and_forged_digest_cannot_be_bound(self):
        _,p = await self.prepared()
        evidence = p.tools.memory_evidence
        with self.assertRaises(SourceError):
            await evidence.bind([self.source.original_evidence()|{"content_digest":"0"*64}])
        other = await self.f.publish("Another source.",source_key="another")
        foreign = await self.f.publish("Foreign area original.",source_key="foreign")
        async with self.f.pool.acquire() as connection:
            area = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) values($1,'Other','test') returning id",self.f.owner)
        from app.tasks.memory_evidence import TaskMemoryEvidence
        outside = TaskMemoryEvidence(evidence.context,self.f.sources,area_id=area)
        with self.assertRaises(SourceError):
            await outside.bind([foreign.original_evidence()])
        await self.f.publish("Corrected original.",expected_revision=1)
        with self.assertRaises(SourceError):
            await evidence.bind([self.source.original_evidence()])
        self.assertNotEqual(other.source_id,self.source.source_id)

    async def test_withdrawing_default_memory_area_revokes_task_and_saved_result(self):
        ctx,p = await self.prepared(use_default_area=True)
        self.assertIsNone(ctx.task.knowledge_area_id)
        await self.read(p)
        await self.tasks.finish(ctx.claim,WorkerOutcome(TaskStatus.SUCCEEDED,{"answer":self.source.content["text"]}))
        async with self.f.pool.acquire() as connection:
            await connection.execute(
                "update jarvis_knowledge_areas set status='deleted',revision=revision+1 where owner_id=$1 and id=$2",
                self.f.owner,self.f.area)
            self.assertEqual(await connection.fetchval(
                "select status from jarvis_task_context_manifests where id=$1",p.manifest.id),"revoked")
        self.assertIsNone((await self.tasks.get(self.f.owner,ctx.task.id)).result_payload_id)

    async def test_closed_selected_runtime_cannot_release_previously_read_evidence(self):
        _,p = await self.prepared()
        result = await self.read(p)
        def closed(_):
            raise SourceError("knowledge_intake_closed","Closed.",503)
        p.tools.memory_evidence.access_check = closed
        with self.assertRaises(SourceError):
            await p.tools.verified_memory(TaskMemoryCitation(**result["task_evidence"]))

    async def native_knowledge(self):
        configuration = KnowledgeConfiguration(
            base_url=os.environ["JARVIS_S04_MEMORY_URL"],api_key=os.environ["JARVIS_S04_MEMORY_TOKEN"],
            area_id=self.f.area,poll_seconds=.2,
        )
        knowledge = await KnowledgeRuntime.create(
            pool=self.f.pool,owner_id=self.f.owner,configuration=configuration,tasks=self.tasks,
        )
        self.addAsyncCleanup(knowledge.close)
        knowledge.start()
        deadline = asyncio.get_running_loop().time()+240
        while asyncio.get_running_loop().time()<deadline:
            source = await self.f.sources.read(self.f.owner,self.f.area,self.source.source_id)
            if source.ingestion_status=="searchable":
                break
            if source.ingestion_status in {"failed","cancelled"}:
                self.fail("Native source ingestion did not succeed: "+source.ingestion_status)
            await asyncio.sleep(.5)
        self.assertEqual(source.ingestion_status,"searchable")
        return knowledge, source

    async def native_read(self, knowledge, source, request):
        ctx,_ = await self.prepared(knowledge=knowledge,outcome=request)
        provider = HermesTaskProvider(
            FrontierConfiguration(model=os.environ["JARVIS_S04_MODEL"],
                                  reasoning_effort=os.getenv("JARVIS_S04_REASONING_EFFORT","medium"),
                                  api_key=JarvisSettings().OPENAI_API_KEY),
            HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]),self.root/"native-state"),
            prepare=WorkerContextFactory(self.f.pool,knowledge=knowledge),
        )
        recalls = []
        native_recall, verified_recall = knowledge.client.recall, knowledge.retrieval.recall
        async def trace_native(bank, query, **arguments):
            result = await native_recall(bank, query, **arguments)
            recalls.append({"layer": "native", "query": query, "result": result})
            return result
        async def trace_verified(owner_id, area_id, query, **arguments):
            result = await verified_recall(owner_id, area_id, query, **arguments)
            recalls.append({"layer": "verified", "query": query,
                            "scope": arguments.get("scope"), "result": result})
            return result
        try:
            # Test-only traces of actual native calls, not substituted responses.
            # All originals and reports in this manifested fixture are synthetic.
            with patch.object(knowledge.client, "recall", side_effect=trace_native), \
                 patch.object(knowledge.retrieval, "recall", side_effect=trace_verified):
                outcome = await asyncio.wait_for(provider.execute(ctx),150)
        finally:
            (self.root/("memory-recall-trace-"+str(ctx.task.id)+".json")).write_text(json.dumps({
                "source_id": str(source.source_id), "document_id": source.document_id,
                "ingestion_status": source.ingestion_status, "calls": recalls,
            }, indent=2, default=str))
        await self.tasks.finish(ctx.claim,outcome)
        (self.root/("memory-native-"+str(ctx.task.id)+".json")).write_text(json.dumps({
            "status":outcome.status,"payload":outcome.payload,
            "usage":await ctx.budgets.usage(self.f.owner,ctx.task.id),
            "source_id":str(source.source_id),"source_revision":source.revision,
        },indent=2))
        return ctx, outcome

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_MEMORY_URL"),
                         "Actual native Hermes and isolated Hindsight service required")
    async def test_native_hermes_reads_hindsight_and_cites_actual_original(self):
        knowledge, source = await self.native_knowledge()
        ctx, outcome = await self.native_read(knowledge, source,
            "Use search_memory to find my reported coding progress, then explain_memory to read its original. "
            "Tell me whether I completed or paused the exercise and include the exact synthetic marker. "
            "Say whether this is my report or a direct observation. Cite the tool's task_evidence.")
        self.assertEqual(outcome.status,TaskStatus.SUCCEEDED)
        marker = self.source.content["text"].rsplit(" ",1)[1]
        self.assertIn(marker,outcome.payload["answer"])
        self.assertIn("report",outcome.payload["answer"].lower())
        self.assertTrue(any(word in outcome.payload["answer"].lower() for word in ("paused","incomplete")))
        self.assertIn(str(source.source_id),[item.get("source_id") for item in outcome.payload["evidence"]])
        names = [item["name"] for item in outcome.payload["tool_receipts"]]
        self.assertIn("search_memory",names)
        self.assertIn("explain_memory",names)
        self.assertFalse(outcome.payload["completed_operation_refs"])
        await knowledge.close()
        await self.f.sources.delete_source(self.f.owner,self.f.area,source.source_id,expected_revision=1)
        self.assertIsNone((await self.tasks.get(self.f.owner,ctx.task.id)).result_payload_id)

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_MEMORY_URL"),
                         "Actual native Hermes and isolated Hindsight service required")
    async def test_native_unrelated_candidate_does_not_establish_user_preference(self):
        knowledge, source = await self.native_knowledge()
        request = "What is my favorite programming language? Search my memories before answering."
        candidates = await knowledge.retrieval.recall(self.f.owner, self.f.area, request)
        self.assertTrue(any(item["source_id"] == str(source.source_id) for item in candidates["sources"]))
        _, outcome = await self.native_read(knowledge, source, request)
        self.assertEqual(outcome.status, TaskStatus.WAITING)
        self.assertTrue(outcome.payload["uncertainties"])
        self.assertFalse(outcome.payload["completed_operation_refs"])
        self.assertIn("search_memory", [item["name"] for item in outcome.payload["tool_receipts"]])

    async def test_version_discovery_binds_historical_originals_and_removal_revokes_task(self):
        await self.f.publish("Current synthetic report", expected_revision=1)
        _, prepared = await self.prepared()
        result = await prepared.tools.invoke("list_memory_versions",
            {"memory_id": str(self.source.source_id)}, provider_call_id="history-" + uuid4().hex)
        self.assertEqual(result["status"], "success")
        self.assertEqual([row["revision"] for row in result["versions"]], [2, 1])
        self.assertEqual([row["task_evidence"]["source_usage"] for row in result["versions"]],
                         ["current", "historical"])
        self.assertTrue(all("content" not in row for row in result["versions"]))
        for row in result["versions"]:
            await prepared.tools.verified_memory(TaskMemoryCitation(**row["task_evidence"]))
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_task_memory_sources"), 2)
            await connection.execute(
                "update jarvis_context_payloads set availability='deleted',content=null where id=$1",
                self.source.payload_id)
        with self.assertRaises(ClaimLost):
            await prepared.tools.verified_memory(TaskMemoryCitation(**result["versions"][-1]["task_evidence"]))

    @unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_MEMORY_URL"),
                         "Actual native Hermes and isolated Hindsight service required")
    async def test_native_old_robot_constraint_and_corrected_history(self):
        corpus = json.loads((Path(__file__).parents[1] / "evals/second_brain/s09_retrieval_cases.json").read_text())
        self.assertEqual(corpus["version"], 1)
        self.assertTrue(corpus["synthetic"])
        bindings = {}
        for item in corpus["sources"]:
            source = await self.f.publish(item["text"], source_key=item["key"],
                expected_revision=item["revision"] - 1, source_time=datetime.fromisoformat(item["time"]))
            bindings[item["key"] + ":" + str(item["revision"])] = source
        knowledge, marker = await self.native_knowledge()
        current = {value.source_id: value for value in bindings.values()}
        async def wait_for_sources():
            while True:
                originals = [await self.f.sources.read(self.f.owner, self.f.area, identity) for identity in current]
                if all(source.ingestion_status == "searchable" for source in originals):
                    return
                self.assertFalse(any(source.ingestion_status in {"failed", "cancelled"} for source in originals))
                await asyncio.sleep(.5)
        await asyncio.wait_for(wait_for_sources(), 240)
        case = corpus["cases"][0]
        candidates = await knowledge.retrieval.recall(self.f.owner, self.f.area, case["query"])
        candidate_ids = {row["source_id"] for row in candidates["sources"]}
        candidate_expected = {str(bindings[key].source_id) for key in case["required_current"]}
        candidate_pass = candidate_expected <= candidate_ids
        trace = {
            "case_id": case["id"], "query": case["query"],
            "bindings": {key: {"source_id": str(value.source_id), "revision": value.revision}
                         for key, value in bindings.items()},
            "candidate_recall_pass": candidate_pass, "candidates": candidates,
            "answer_semantic_grade": None,
        }
        trace_path = self.root / "s09-native-oracle.json"
        trace_path.write_text(json.dumps(trace, indent=2, default=str))
        self.assertTrue(candidate_pass, "Native candidate recall missed a required current original")
        _, outcome = await self.native_read(knowledge, marker, case["query"])
        trace["answer"] = outcome.payload
        trace["answer_status"] = str(outcome.status)
        trace_path.write_text(json.dumps(trace, indent=2, default=str))
        self.assertEqual(outcome.status, TaskStatus.SUCCEEDED)
        actual = {(item.get("source_id"), item.get("revision"), item.get("source_usage"))
                  for item in outcome.payload["evidence"]}
        for usage, keys in (("current", case["required_current"]), ("historical", case["required_historical"])):
            for key in keys:
                source = bindings[key]
                self.assertIn((str(source.source_id), source.revision, usage), actual)
        names = {item["name"] for item in outcome.payload["tool_receipts"]}
        self.assertTrue({"search_memory", "explain_memory", "list_memory_versions"} <= names)
        self.assertFalse(outcome.payload["completed_operation_refs"])
