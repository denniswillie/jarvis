"""Direct Voice guidance uses native transfer, never a new permission grant."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from app.hermes_sessions.bindings import BindingError
from app.hermes_sessions.client import HermesGatewayError, HermesOutcomeUnknown
from app.hermes_sessions.domain_service import compose_domain_work
from test_work import ResultFixture


class GuidanceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        (self.home / "jarvis-adapter.json").write_text(json.dumps({
            "owner_id": "owner", "domain_tools": True, "launch": {"source": "cli"}}))
        self.client = ResultFixture(self.home)
        self.bridge = compose_domain_work(self.home, self.client)
        self.service = self.bridge.service
        self.master = await self.service.master()
        self.session_id = uuid4()
        self.scope = await self.service.bind_input_scope(self.master.logical_id, input_id="guidance",
            jarvis_session_id=self.session_id, allowed_tools=frozenset({"sandbox_publish_file"}))
        self.receipt = await self.service.send(self.master.logical_id, input_id="guidance", text="Original guidance")

    async def steer(self):
        return await self.service.steer_live_input(self.receipt, jarvis_session_id=self.session_id)

    async def test_native_transfer_preserves_input_and_scope(self):
        await self.steer()
        self.assertEqual(self.client.steers, [{"session_id": self.master.native_id,
            "admission_id": self.receipt.admission_id}])
        self.assertEqual(len(self.client.inputs), 1)
        self.assertEqual(self.bridge.scopes.get(self.master.logical_id, self.receipt.native_input_id), self.scope)

    async def test_wrong_voice_session_and_unbound_identity_rejected(self):
        with self.assertRaises(BindingError):
            await self.service.steer_live_input(self.receipt, jarvis_session_id=uuid4())
        with self.assertRaises(BindingError):
            await self.service.steer_live_input(replace(self.receipt, input_id="different"),
                                                 jarvis_session_id=self.session_id)
        self.assertEqual(self.client.steers, [])

    async def test_request_authority_is_not_reinterpreted_as_voice_guidance(self):
        await self.service.bind_input_scope(self.master.logical_id, input_id="versioned",
            jarvis_session_id=self.session_id, allowed_tools=frozenset(),
            request_event_id=uuid4(), request_revision=1)
        receipt = await self.service.send(self.master.logical_id, input_id="versioned", text="Versioned request")
        with self.assertRaises(BindingError):
            await self.service.steer_live_input(receipt, jarvis_session_id=self.session_id)
        self.assertEqual(self.client.steers, [])

    async def test_native_busy_race_keeps_original_admission(self):
        original = self.client.request
        async def request(method, **params):
            if method == "prompt.steer_queued":
                raise HermesGatewayError("admission_started")
            return await original(method, **params)
        self.client.request = request
        result = await self.steer()
        self.assertEqual(result["admission_id"], self.receipt.admission_id)
        self.assertEqual(len(self.client.inputs), 1)

    async def test_unknown_control_is_retried_with_same_admission_not_new_input(self):
        original = self.client.request
        fail = True
        async def request(method, **params):
            nonlocal fail
            if method == "prompt.steer_queued" and fail:
                fail = False
                raise HermesOutcomeUnknown(method, params["admission_id"])
            return await original(method, **params)
        self.client.request = request
        with self.assertRaises(HermesOutcomeUnknown):
            await self.steer()
        await self.steer()
        self.assertEqual(self.client.steers[-1]["admission_id"], self.receipt.admission_id)
        self.assertEqual(len(self.client.inputs), 1)

    async def redirect(self, response=None, error=None, running=True):
        self.redirect_calls = []
        original = self.client.request
        async def request(method, **params):
            if method == "session.resume" and params.get("omit_messages"):
                return {"running": running, "execution_generation": 7, "pending": [
                    {"admission_id": "active-admission", "status": "started", "execution_generation": 7}]}
            if method == "session.redirect":
                self.redirect_calls.append(params)
                if error:
                    raise error
                return response or {"status": "redirected"}
            return await original(method, **params)
        self.client.request = request
        return await self.service.control_live_input(self.master.logical_id, input_id="guidance",
            text="No, the first result", jarvis_session_id=self.session_id, delivery="redirect")

    async def test_redirect_targets_current_generation_and_reuses_result(self):
        receipt, status = await self.redirect()
        self.assertEqual((receipt.admission_id, status), ("active-admission", "redirected"))
        self.assertEqual(self.redirect_calls[0]["execution_generation"], 7)
        self.assertEqual(self.redirect_calls[0]["text"], "No, the first result")
        self.assertEqual(len(self.client.inputs), 1)  # No second queued copy.

    async def test_idle_and_definite_refusal_allow_queue_fallback(self):
        self.assertIsNone(await self.redirect(running=False))
        self.assertEqual(self.redirect_calls, [])
        self.assertIsNone(await self.redirect(error=HermesGatewayError("stale_generation")))
        self.assertIsNone(await self.redirect(response={"status": "rejected", "delivered": False}))

    async def test_uncertain_redirect_never_authorizes_resubmission(self):
        for response, error in (({"status": "rejected"}, None),
                                (None, HermesOutcomeUnknown("session.redirect", "guidance"))):
            receipt, status = await self.redirect(response=response, error=error)
            self.assertEqual(status, "unconfirmed")
            self.assertEqual(receipt.admission_id, "active-admission")

    async def test_redirect_rejects_wrong_voice_scope(self):
        with self.assertRaises(BindingError):
            await self.service.control_live_input(self.master.logical_id, input_id="guidance",
                text="No, the first result", jarvis_session_id=uuid4(), delivery="redirect")

    async def test_direct_steer_joins_current_turn_without_another_admission(self):
        original = self.client.request
        calls = []
        async def request(method, **params):
            if method == "session.resume" and params.get("omit_messages"):
                return {"running": True, "execution_generation": 8, "pending": [
                    {"admission_id": "running-turn", "status": "started", "execution_generation": 8}]}
            if method == "session.steer":
                calls.append(params)
                return {"status": "queued"}  # Native: buffered in this turn, not a new admission.
            return await original(method, **params)
        self.client.request = request
        receipt, status = await self.service.control_live_input(self.master.logical_id,
            input_id="guidance", text="Also check its date", jarvis_session_id=self.session_id, delivery="steer")
        self.assertEqual((receipt.admission_id, status), ("running-turn", "steering"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.client.inputs), 1)
