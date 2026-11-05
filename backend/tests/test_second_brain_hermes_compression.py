"""Real automatic Hermes compaction through the production task/model boundary."""
import asyncio
import json
import os
from pathlib import Path
import secrets
import unittest
from uuid import uuid4

from app.runtime.feature_config import FrontierConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.context_events import ContextIngress, ContextWindow
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes import HermesContextFactory, HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.model_gateway import OpenAIResponsesHTTP
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from evals.second_brain.source_fixture import source_fixture


class ObservedUpstream(OpenAIResponsesHTTP):
    """Observe synthetic requests; optionally fault only the summary transport."""
    def __init__(self, key, *, fail_compression=False):
        super().__init__(key)
        self.fail_compression = fail_compression
        self.requests = []
        self.compression_attempts = 0
        self.primary_after_compression = False
        self.source_pages = {}

    async def events(self, payload):
        auxiliary = "text" not in payload
        for item in payload.get("input",[]):
            if not isinstance(item,dict) or item.get("type")!="function_call_output":
                continue
            try:
                result = json.loads(item.get("output",""))
            except (ValueError,TypeError):
                continue
            if not isinstance(result,dict) or not isinstance(result.get("source"),dict):
                continue
            content = result.get("content_json")
            if not isinstance(content,str):
                continue
            self.source_pages.setdefault(item["call_id"],{
                "source_id":result["source"]["id"],"offset":result["offset"],
                "next_offset":result["next_offset"],"total_characters":result["total_characters"],
                "returned_characters":len(content),"after_compression":bool(self.compression_attempts),
            })
        self.requests.append({"purpose":"compression" if auxiliary else "task",
                              "max_output_tokens":payload["max_output_tokens"]})
        if auxiliary:
            self.compression_attempts += 1
            if self.fail_compression:
                raise RuntimeError("Synthetic compression transport failure")
        elif self.compression_attempts:
            self.primary_after_compression = True
        async for event in super().events(payload):
            yield event


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR")
    and os.getenv("JARVIS_HERMES_PYTHON") and os.getenv("JARVIS_S04_COMPRESSION_LIVE")=="1",
    "Explicit isolated database and native compression opt-in required",
)
class NativeCompressionTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, *, fail_compression=False):
        root = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        async with source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],root,test_name=self.id()) as fixture:
            tasks = TaskRepository(fixture.pool,TaskLimits(
                lease_seconds=600,request_output_tokens=4096,task_output_tokens=40000,
                task_input_tokens=1200000,tool_steps=30,
            ))
            async with fixture.pool.acquire() as connection:
                session = await connection.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id",fixture.owner)
            contexts = ContextRepository(tasks)
            marker = "PROJECT-"+uuid4().hex[:12]
            facts = [
                "The project identifier is "+marker+".",
                "The binary-search midpoint must be mid = (low + high) // 2.",
                "The user reported pausing this work for today, not completing it.",
                "The next coding step is to add the search-bound updates.",
                "No tests have been run; do not describe the function as tested.",
                "The sample list is sorted in ascending order.",
                "No camera images are supplied; do not claim visual inspection.",
                "No write or email action has been authorized or performed.",
            ]
            events = []
            for index,fact in enumerate(facts):
                # Unrelated high-entropy lab output creates real context pressure.
                # The semantic facts are original source content, not expected answers in the task.
                events.append(await contexts.append(fixture.owner,ContextIngress(
                    session_id=session,source_namespace="fixture-compaction",source_event_id=str(uuid4()),
                    receipt_id=str(uuid4()),kind="observation",trust="derived",
                    payload={"note_index":index+1,"fact":fact,"synthetic_noise_not_evidence":secrets.token_hex(14000),
                             "end_of_note":index+1},
                )))
            await contexts.append(fixture.owner,ContextIngress(
                session_id=session,source_namespace="fixture-compaction",source_event_id=str(uuid4()),
                receipt_id=str(uuid4()),kind="user_request",trust="user_input",
                payload={"text":"Review my eight original lab notes, preserving stated uncertainty and permissions."},
            ))
            task = await tasks.create(fixture.owner,TaskCreate(
                request_id=uuid4(),title="Read original notes across compaction",session_id=session,
                requested_outcome="First read the eight original source payloads in this exact order, one source tool call "
                    "per model turn for this initial pass. Read complete payloads in the initial pass with the default "
                    "page size (omit max_characters), even if an initial excerpt is present: "+", ".join(str(e.id) for e in events)+". "
                    "Each source has a fact and unrelated synthetic noise; the noise has no semantic significance. "
                    "After reading all eight complete payloads, report the project ID, exact midpoint code, actual "
                    "progress and next step, testing status, input ordering, visual evidence and authorized actions. "
                    "Cite all eight original sources. Do not write anything or invent missing evidence. "
                    "Continue this assignment after context compaction. If a summary omits needed details, "
                    "reread original sources using the available tools rather than asking me for them or guessing. "
                    "For recovery, use smaller exact fragments (for example max_characters=2048) to recover "
                    "the needed complete fact without reloading unrelated large output. "
                    "Make the answer a JSON object encoded as text, without markdown, with these keys: "
                    "project_id, midpoint_assignment, progress_status (paused, completed, or unknown), "
                    "next_step (short text), tests_run (boolean or null if unknown), "
                    "input_order (ascending, descending, or unknown), images_supplied (boolean or null), "
                    "and authorized_actions (array of action names).",
            ),model=os.environ["JARVIS_S04_MODEL"])
            claim = await tasks.claim_next("native-compression-fixture",("frontier",))
            self.assertEqual(claim.task.id,task.id)
            context = TaskRunner(tasks,{}).context(claim)
            upstream = ObservedUpstream(JarvisSettings().OPENAI_API_KEY,fail_compression=fail_compression)
            provider = HermesTaskProvider(
                FrontierConfiguration(model=task.model,reasoning_effort="low",api_key=JarvisSettings().OPENAI_API_KEY),
                HermesRuntimeConfiguration(Path(os.environ["JARVIS_HERMES_PYTHON"]),root/"native-state",
                    context_tokens=65536,compression_tail_messages=3),
                prepare=HermesContextFactory(context_window=ContextWindow(max_messages=1)),
                upstream_factory=lambda:upstream,
            )
            result = None
            try:
                result = await asyncio.wait_for(provider.execute(context),360)
                self.assertTrue(await tasks.finish(claim,result))
            finally:
                rows = []
                async with fixture.pool.acquire() as connection:
                    rows = await connection.fetch(
                        "select content from jarvis_context_payloads where owner_id=$1 "
                        "and content->>'format'='jarvis.hermes.model_request.v1'",fixture.owner)
                checkpoints = [json.loads(row["content"]) if isinstance(row["content"],str) else row["content"] for row in rows]
                usage = await context.budgets.usage(fixture.owner,task.id)
                proof = {"format":"jarvis.s04.compression_fixture.v1","task_id":str(task.id),
                    "mode":"failure" if fail_compression else "success","requests":upstream.requests,
                    "compression_attempts":upstream.compression_attempts,
                    "primary_after_compression":upstream.primary_after_compression,
                    "source_pages":list(upstream.source_pages.values()),"checkpoints":checkpoints,"usage":usage,
                    "outcome":None if result is None else {"status":result.status,"reason":result.reason,"payload":result.payload}}
                (root/("compression-"+str(task.id)+".json")).write_text(json.dumps(proof,indent=2)+"\n")
            self.assertGreaterEqual(upstream.compression_attempts,1,"The native compressor must actually call its model")
            self.assertTrue(all(request["max_output_tokens"]<=4096 for request in upstream.requests))
            self.assertGreater(usage["input"],0)
            self.assertGreater(usage["output"],0)
            if fail_compression:
                self.assertNotEqual(result.status,TaskStatus.SUCCEEDED)
                self.assertFalse(upstream.primary_after_compression,
                                 "Unknown summary usage must fence later primary inference")
            else:
                self.assertEqual(result.status,TaskStatus.SUCCEEDED,result.reason)
                self.assertTrue(upstream.primary_after_compression)
                self.assertTrue(any(page["after_compression"] and page["next_offset"] is not None
                                    for page in upstream.source_pages.values()),
                                "Compaction recovery must exercise an exact smaller source fragment")
                self.assertTrue(any(item["purpose"]=="compression" for item in checkpoints))
                answer = json.loads(result.payload["answer"])
                self.assertEqual(answer["project_id"],marker)
                self.assertEqual(answer["midpoint_assignment"],"mid = (low + high) // 2")
                self.assertEqual(answer["progress_status"],"paused")
                self.assertIn("bound",answer["next_step"].lower())
                self.assertIs(answer["tests_run"],False)
                self.assertEqual(answer["input_order"],"ascending")
                self.assertIs(answer["images_supplied"],False)
                self.assertEqual(answer["authorized_actions"],[])
                self.assertEqual({item["event_id"] for item in result.payload["evidence"]},
                                 {str(event.id) for event in events})
                self.assertFalse(result.payload["completed_operation_refs"])
                self.assertGreaterEqual(sum(receipt["name"]=="read_task_context_source"
                                            for receipt in result.payload["tool_receipts"]),8)
                native = [json.loads(path.read_text()) for path in (root/"native-state").glob("*/run.json")]
                run = next(value for value in native if value["task_id"]==str(task.id))
                self.assertGreaterEqual(run["native_result"]["context_compressions"],1)

    async def test_native_automatic_compaction_preserves_original_task_evidence(self):
        await self.run_case()

    async def test_native_failed_summary_does_not_claim_completion_or_continue_inference(self):
        await self.run_case(fail_compression=True)
