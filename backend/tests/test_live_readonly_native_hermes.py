"""Bounded real pinned child, with synthetic SQL rows and model Responses.

Run only with an explicitly scoped JARVIS_HERMES_PYTHON. The installed native
Hermes agent, production child/loopback adapter, task runner, tools, budgets,
capture, result validation and Live adapter all execute. No paid model, DB,
long-running service or device. These are protocol tests, not reasoning evals.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from app.realtime.live_delegation import current_result, result_guard
from app.realtime.live_session import LiveAudioSession
from app.runtime.feature_config import FrontierConfiguration
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.models import ClaimLost, TaskError, TaskStatus
from app.tasks.readonly import CAPABILITY
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from app.voice_routes.contracts import DeliveryEvidence, TranscriptRevision
from native_readonly_fixture import ControlledReadonlyUpstream, NativeReadonlyStore, now
import test_live_readonly_delegation as owner_fixture
import test_live_readonly_flow as flow_fixture


@unittest.skipUnless(os.getenv("JARVIS_HERMES_PYTHON"), "Explicit pinned native interpreter required")
class NativeReadonlyQualificationTests(unittest.IsolatedAsyncioTestCase):
    admit = owner_fixture.ReadonlyOwnerTests.admit
    wire_start = flow_fixture.ReadonlyFlowTests.wire_start

    async def asyncSetUp(self):
        self.python = Path(os.environ["JARVIS_HERMES_PYTHON"])
        self.assertTrue(self.python.is_absolute() and self.python.is_file())
        self.store_patch = patch.object(owner_fixture, "ReadonlyStore", NativeReadonlyStore)
        self.store_patch.start()
        self.addCleanup(self.store_patch.stop)
        await owner_fixture.ReadonlyOwnerTests.asyncSetUp(self)
        self.root = tempfile.TemporaryDirectory(prefix="native-readonly-", dir=Path(__file__).resolve().parents[2]/".jarvis-dev")
        self.addCleanup(self.root.cleanup)
        self.jobs, self.runs = [], []
        self.provider = self.runner = None
        self.no_paid = patch("app.tasks.hermes.OpenAIResponsesHTTP", side_effect=AssertionError("paid upstream forbidden"))
        self.no_paid.start()
        self.addCleanup(self.no_paid.stop)
        self.service.providers["frontier"].model = "gpt-5.6-terra"
        await self.wire_start()
        await self.wire.inject(type="session.input_transcript.delta", event_id="native-user-fragment",
                               delta="What number is visible?", start_ms=1, end_ms=90)
        await self.wire.until(lambda: any(isinstance(e, TranscriptRevision) for e in self.wire.events))
        caption = next(e for e in self.wire.events if isinstance(e, TranscriptRevision))
        self.assertTrue(self.captions.transcript(caption, self.wire.route.transcript_parts))
        self.observations = await self.captions.observations(self.wire.route.transcript_parts)
        await self.wire.inject(type="session.delegation.created", event_id="native-delegation", offset_ms=90,
            delegation={"id":"d1", "type":"delegation", "target":"client"})
        stream = self.wire.route.delegation_requests()
        async with asyncio.timeout(1):
            self.request = await anext(stream)
        await stream.aclose()

    async def asyncTearDown(self):
        if hasattr(self, "upstream"):
            self.upstream.count_release.set()
            self.upstream.stream_release.set()
        for job in self.jobs:
            if not job.done():
                job.cancel()
        if self.jobs:
            await asyncio.wait_for(asyncio.gather(*self.jobs, return_exceptions=True), 10)
        if self.provider is not None:
            for run in tuple(self.provider._runs.values()):
                await asyncio.wait_for(run.stop(), 5)
        for run in self.runs:
            self.assertTrue(run.finished.is_set(), "native adapter did not finish joined cleanup")
            self.assertIsNotNone(run.process.returncode, "native child still running")
            self.assertTrue(run.server_task.done(), "temporary gateway still running")
            self.assertFalse(run.home.exists(), "native private copies remain")
        await self.wire.close()
        await owner_fixture.ReadonlyOwnerTests.asyncTearDown(self)

    async def start_native(self, *, hold=False, budget_reject=False):
        self.receipt = await self.admit()
        self.host = flow_fixture.GuardHost(self, self.wire, self.receipt)
        self.assertIsNone(await self.tasks.claim_next("unqualified", ("frontier",), worker_capabilities=()))
        self.claim = await self.tasks.claim_next("native-readonly", ("frontier",), owner_id=self.scope.owner_id,
            provider_models={"frontier":"gpt-5.6-terra"}, worker_capabilities=(CAPABILITY,))
        self.assertIsNotNone(self.claim)
        answer = {"status":"waiting_for_input", "answer":"The observed number is not confirmed.",
            "speech":[{"text":"The observed number is not confirmed."}],
            "evidence":[{"event_id":str(self.observations[0].id), "revision":self.observations[0].revision}],
            "completed_operation_refs":[], "uncertainties":["No legible current number is established."]}
        self.upstream = ControlledReadonlyUpstream(answer, hold=hold, budget_reject=budget_reject)
        factory = WorkerContextFactory(self.db)

        async def prepare(context):
            self.prepared = await factory(context)
            return self.prepared

        self.provider = HermesTaskProvider(FrontierConfiguration(model="gpt-5.6-terra", reasoning_effort="low",
            api_key="synthetic-unused"), HermesRuntimeConfiguration(self.python, Path(self.root.name),
            host_id="native-readonly-fixture", max_iterations=4, tool_timeout_seconds=5, stop_seconds=.3),
            prepare=prepare, upstream_factory=lambda:self.upstream)
        self.runner = TaskRunner(self.tasks, {"frontier":self.provider}, worker_id="native-readonly",
            owner_id=self.scope.owner_id, host_id="native-readonly-fixture", capabilities=(CAPABILITY,))
        self.job = asyncio.create_task(self.runner._execute(self.claim))
        self.jobs.append(self.job)
        entered = asyncio.create_task(self.upstream.count_entered.wait())
        try:
            done, _ = await asyncio.wait((entered, self.job), timeout=25, return_when=asyncio.FIRST_COMPLETED)
            self.assertIn(entered, done, "native child ended before controlled model-count boundary")
            self.run = self.provider._runs[self.claim.attempt_id]
            self.runs.append(self.run)
            self.assertIsNotNone(self.run.process.pid)
            self.assertIsNone(self.run.process.returncode)
        finally:
            entered.cancel()
            await asyncio.gather(entered, return_exceptions=True)
            self.upstream.count_release.set()
        if hold:
            await asyncio.wait_for(self.upstream.stream_entered.wait(), 5)
        return self.run

    async def finish_native(self):
        await asyncio.wait_for(self.job, 25)
        self.assertTrue(self.run.finished.is_set())
        self.assertIsNotNone(self.run.process.returncode)
        self.assertTrue(self.run.server_task.done())
        self.assertFalse(self.run.home.exists())
        self.assertFalse(self.provider._runs)
        task = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        print("NATIVE case=" + self._testMethodName + " child_started=1 child_joined=1 gateway_joined=1 private_home_removed=1")
        return task

    async def result(self):
        async with self.tasks.transaction(self.scope.owner_id) as connection:
            return await current_result(self.tasks, connection, self.receipt, self.route)

    async def recover(self, *, new_provider=False):
        recovery = await self.tasks.claim_recovery("native-recovery", owner_id=self.scope.owner_id,
            host_id="native-readonly-fixture", backend_kinds=("frontier",))
        self.assertIsNotNone(recovery)
        provider = self.provider
        if new_provider:
            provider = HermesTaskProvider(self.provider.frontier, self.provider.configuration,
                upstream_factory=lambda: (_ for _ in ()).throw(AssertionError("recovery must not dispatch")))
        context = self.runner.context(recovery)
        proof = await provider.reconcile(context, cancel=recovery.task.pending_status is not None)
        self.assertEqual(proof.state, "terminal")
        await self.tasks.resolve_recovery(recovery, proof)
        self.assertEqual((await self.tasks.get(self.scope.owner_id, self.receipt.task.id)).attempt_count, 1)

    async def test_native_readonly_owner_claim_child_tool_budget_and_guarded_live_reply(self):
        await self.start_native()
        task = await self.finish_native()
        self.assertEqual(task.status, TaskStatus.WAITING)
        self.assertEqual(len(self.upstream.requests), 2)
        self.assertEqual({tool["name"] for tool in self.prepared.tools.catalog()},
            {"read_task_context_archive", "read_task_context_source", "read_background_task_status"})
        self.assertFalse(self.prepared.tools.actions or self.db.grants)
        result = await self.result()
        payload = self.db.payloads[result.result_payload_id]["content"]
        self.assertEqual(payload["tool_receipts"][0]["name"], "read_background_task_status")
        self.assertEqual(payload["tool_receipts"][0]["status"], "success")
        self.assertEqual(await self.runner.context(self.claim).budgets.usage(self.scope.owner_id, task.id),
                         {"input":22, "output":14, "steps":1})
        response = await self.wire.route.submit_delegation_result(result,
            guard=lambda expected: result_guard(self.host, self.receipt, expected))
        self.assertTrue(response.provider_received)
        self.assertFalse(any(isinstance(event, DeliveryEvidence) for event in self.wire.events))
        self.assertFalse(any(source.authorizes_request for source in self.observations))

    async def test_actual_child_gateway_rejects_forbidden_tools_without_action_authority(self):
        run = await self.start_native(hold=True)
        async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
            for name in ("remember", "create_reminder", "terminal", "browser", "create_note"):
                response = await client.post(run.marker["gateway_url"]+"/task/tools/"+name,
                    headers={"Authorization":"Bearer "+run.gateway.token},
                    json={"arguments":{}, "call_id":"probe-"+name})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["error"]["code"], "task_tool_not_allowed")
        self.assertFalse(self.db.grants)
        self.assertEqual(sum(r["tool_steps"] for r in self.db.reservations.values()), 0)
        self.upstream.stream_release.set()
        self.assertEqual((await self.finish_native()).status, TaskStatus.WAITING)

    async def test_native_budget_rejection_prevents_controlled_inference_dispatch(self):
        await self.start_native(budget_reject=True)
        task = await self.finish_native()
        self.assertFalse(self.upstream.requests)
        self.assertNotEqual(task.status, TaskStatus.SUCCEEDED)
        self.assertFalse(self.db.reservations)

    async def test_native_cancellation_fences_and_recovers_original_attempt(self):
        await self.start_native(hold=True)
        before = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        original_deadline = self.db.attempts[self.claim.attempt_id]["lease_until"]
        self.assertGreater(original_deadline, now())
        self.assertIsNone(await self.tasks.claim_recovery("too-early", owner_id=self.scope.owner_id,
            host_id="native-readonly-fixture", backend_kinds=("frontier",)))
        cancellation_started = now()
        self.assertTrue(await self.admission.cancel(self.receipt, reason="cancelled"))
        pending = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        attempt = self.db.attempts[self.claim.attempt_id]
        self.assertEqual((pending.status, pending.pending_status), (TaskStatus.CANCEL_REQUESTED, "cancelled"))
        self.assertEqual((pending.revision, pending.fence), (before.revision+1, before.fence+1))
        self.assertEqual(pending.current_attempt_id, self.claim.attempt_id)
        self.assertIsNone(pending.result_payload_id)
        self.assertEqual((attempt["status"], attempt["provider_state"]), ("running", "live"))
        self.assertGreaterEqual(attempt["lease_until"], cancellation_started)
        self.assertLessEqual(attempt["lease_until"], now())
        self.assertLess(attempt["lease_until"], original_deadline)
        self.assertIsNone(self.run.process.returncode, "Cancellation is not proof of child exit")
        with self.assertRaises(ClaimLost):
            await self.tasks.renew(self.claim)

        # No fake clock advance or sleep until the old lease expires. Recovery
        # takes the original attempt immediately, not a second work submission.
        recovery = await self.tasks.claim_recovery("native-recovery", owner_id=self.scope.owner_id,
            host_id="native-readonly-fixture", backend_kinds=("frontier",))
        self.assertIsNotNone(recovery)
        self.assertEqual(recovery.attempt_id, self.claim.attempt_id)
        self.assertTrue(recovery.recovering)
        self.assertEqual(recovery.task.pending_status, "cancelled")
        self.assertEqual(recovery.task.status, TaskStatus.CANCEL_REQUESTED)
        self.assertEqual(recovery.fence, pending.fence+1)
        self.assertEqual(recovery.task.fence, recovery.fence)
        owned = self.db.attempts[recovery.attempt_id]
        self.assertEqual((owned["status"], owned["worker_id"], owned["fence"]),
                         ("recovering", "native-recovery", recovery.fence))
        self.assertGreater(owned["lease_until"], now())
        self.assertIsNone(await self.tasks.claim_recovery("duplicate-controller", owner_id=self.scope.owner_id,
            host_id="native-readonly-fixture", backend_kinds=("frontier",)))
        self.assertIsNone(await self.tasks.claim_next("duplicate-worker", ("frontier",),
                                                     worker_capabilities=(CAPABILITY,)))

        proof = await self.provider.reconcile(self.runner.context(recovery), cancel=True)
        self.assertEqual(proof.state, "terminal")
        self.assertTrue(self.run.finished.is_set())
        self.assertIsNotNone(self.run.process.returncode)
        self.assertTrue(self.run.server_task.done())
        self.assertFalse(self.run.home.exists())
        # Even a proven child exit does not bypass the canonical recovery commit.
        self.assertEqual((await self.tasks.get(self.scope.owner_id, self.receipt.task.id)).status,
                         TaskStatus.CANCEL_REQUESTED)
        await self.tasks.resolve_recovery(recovery, proof)
        await self.finish_native()
        task = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertEqual(task.reason, "old_revision_reconciled")
        self.assertIsNone(task.pending_status)
        self.assertIsNone(task.current_attempt_id)
        self.assertIsNone(task.result_payload_id)
        self.assertEqual((task.revision, task.fence, task.attempt_count), (before.revision+1, recovery.fence, 1))
        terminal = self.db.attempts[self.claim.attempt_id]
        self.assertEqual((terminal["status"], terminal["provider_state"]), ("interrupted", "terminal"))
        with self.assertRaises(ClaimLost):
            await self.tasks.renew(recovery)
        self.assertFalse(await self.admission.cancel(self.receipt, reason="cancelled"))
        self.assertIsNone(await self.tasks.claim_recovery("after-terminal", owner_id=self.scope.owner_id,
            host_id="native-readonly-fixture", backend_kinds=("frontier",)))
        self.assertIsNone(await self.tasks.claim_next("after-terminal", ("frontier",),
                                                     worker_capabilities=(CAPABILITY,)))
        self.assertEqual(len(self.db.attempts), 1)
        self.assertEqual(len(self.upstream.requests), 1)
        self.assertFalse(self.db.grants or self.prepared.tools.actions)
        self.assertFalse(any(row["type"] == "session.commentary.append" for row in self.wire.wire.socket.sent))
        with self.assertRaises(TaskError):
            await self.result()

    async def test_native_source_withdrawal_rejects_output_and_recovery_keeps_it_paused(self):
        await self.start_native(hold=True)
        async with self.tasks.transaction(self.scope.owner_id):
            self.db.payloads[self.observations[0].payload_id].update(availability="deleted", content=None)
        self.upstream.stream_release.set()
        await self.finish_native()
        # Time passage is an external storage fixture, not an invented valid lease.
        self.db.attempts[self.claim.attempt_id]["lease_until"] = now()-timedelta(seconds=1)
        await self.recover()
        task = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertIsNone(task.result_payload_id)
        with self.assertRaises(TaskError):
            await self.result()

    async def test_native_completed_but_unsent_result_supersession_blocks_actual_writer(self):
        await self.start_native()
        self.assertEqual((await self.finish_native()).status, TaskStatus.WAITING)
        result = await self.result()
        entered, release = asyncio.Event(), asyncio.Event()

        @asynccontextmanager
        async def delayed(expected):
            entered.set()
            await release.wait()
            async with result_guard(self.host, self.receipt, expected):
                yield

        sending = asyncio.create_task(self.wire.route.submit_delegation_result(result, guard=delayed))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertTrue(await LiveAudioSession.cancel_readonly_delegation(self.host, "d1",
                expected_revision=1, reason="superseded"))
            release.set()
            receipt = await asyncio.wait_for(sending, 2)
            self.assertFalse(receipt.provider_received)
            self.assertFalse(any(row["type"] == "session.commentary.append" for row in self.wire.wire.socket.sent))
        finally:
            release.set()
            await asyncio.gather(sending, return_exceptions=True)

    async def test_native_owned_child_crash_recovery_preserves_unknown_usage_without_retry(self):
        run = await self.start_native(hold=True)
        run.process.kill()  # Exact owned child handle, never a guessed PID.
        task = await self.finish_native()
        self.assertEqual(task.status, TaskStatus.UNKNOWN)
        count = len(self.upstream.requests)
        await self.recover(new_provider=True)
        task = await self.tasks.get(self.scope.owner_id, self.receipt.task.id)
        self.assertEqual(task.status, TaskStatus.PAUSED)
        self.assertEqual(len(self.upstream.requests), count)
        reservations = list(self.db.reservations.values())
        self.assertEqual(len(reservations), 1)
        self.assertEqual(reservations[0]["status"], "unknown")
        usage = await self.runner.context(self.claim).budgets.usage(self.scope.owner_id, task.id)
        self.assertGreater(usage["input"]+usage["output"], 0)


if __name__ == "__main__":
    unittest.main()
