"""Controlled HTTP-client handoff tests, never APNs or physical wire evidence."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx

from app.attention.push_models import PushAuthorization, PushClaim, PushOutcome, PushRequest
from app.attention.push_provider import APNsProvider
from app.attention.push_service import PushService
from test_phone_push import config


class ControlledManager:
    def __init__(self):
        self.connected = False
        self.lock = asyncio.Lock()

    async def has_connected_voice(self, owner):
        async with self.lock:
            return self.connected


class ControlledHTTP:
    """Client entry is observable; mock partial progress is not real wire proof."""
    def __init__(self):
        self.requests = []
        self.callback = None
        self.mode = "accepted"
        self.simulated_partial = 0

    async def handle(self, request):
        self.requests.append(request)
        if self.callback:
            await self.callback(request)
        if self.mode == "partial":
            self.simulated_partial += 1
            raise httpx.ReadError("Controlled partial/uncertain transport", request=request)
        if self.mode == "unknown":
            raise httpx.ReadTimeout("Controlled uncertain transport", request=request)
        return httpx.Response(200, headers={"apns-id": request.headers["apns-id"]})


class PushHandoffTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.configuration = config(fallback_seconds=0)
        self.claim = PushClaim("handoff-offline", uuid4(), uuid4(), 1, uuid4(), 1, "fixture")
        self.request = PushRequest(self.claim, "ab" * 32, "sandbox", self.configuration.topic, None, "normal")
        self.control = ControlledHTTP()
        self.provider = APNsProvider(self.configuration,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.control.handle)))
        self.addAsyncCleanup(self.provider.close)
        self.manager = ControlledManager()
        self.outcomes = []

        async def finish(claim, outcome):
            self.outcomes.append(outcome)
            return outcome

        self.repository = SimpleNamespace(recover=AsyncMock(), claim_next=AsyncMock(return_value=self.claim),
            prepare_submission=AsyncMock(return_value=self.request),
            authorize_send=AsyncMock(return_value=PushAuthorization(self.claim, True, "fixture_authorized")),
            finish=AsyncMock(side_effect=finish))
        self.service = PushService(self.repository, self.provider, self.manager, self.claim.owner_id, self.configuration)

    async def test_missing_or_mismatched_authorization_never_enters_http(self):
        missing = await self.provider.send(self.request)
        self.assertEqual(missing.reason, "push_handoff_authorization_required")
        for value in (None, True, PushAuthorization(replace(self.claim, attempt_id=uuid4()), True, "wrong_claim")):
            with self.subTest(receipt=type(value).__name__):
                result = await self.provider.send(self.request, authorize=AsyncMock(return_value=value))
                self.assertEqual(result.state, "revoked")
        self.assertEqual(self.control.requests, [])

    async def test_repeated_quiet_denials_preserve_reason_after_prepare(self):
        for count in (1, 2, 3):
            with self.subTest(repeats=count):
                self.repository.authorize_send.return_value = PushAuthorization(self.claim, False, "owner_quiet_quiet")
                for _ in range(count):
                    self.assertTrue(await self.service.run_once())
                    self.assertEqual(self.outcomes[-1].reason, "owner_quiet_quiet")
                    self.assertEqual(self.outcomes[-1].state, "revoked")
        self.assertEqual(self.control.requests, [])
        self.assertEqual(self.service.status()["provider_access"], "unverified")

    async def test_voice_reconnect_after_prepare_stops_before_canonical_handoff(self):
        async def prepare(claim):
            self.manager.connected = True
            return self.request
        self.repository.prepare_submission.side_effect = prepare
        self.assertTrue(await self.service.run_once())
        self.repository.authorize_send.assert_not_awaited()
        self.assertEqual(self.outcomes[-1].reason, "voice_reconnected")
        self.assertEqual(self.control.requests, [])

    async def test_handoff_precedes_client_entry_without_voice_lock(self):
        async def entered(request):
            self.repository.authorize_send.assert_awaited_once_with(self.request)
            self.assertFalse(self.manager.lock.locked())
        self.control.callback = entered
        self.assertTrue(await self.service.run_once())
        self.assertEqual(len(self.control.requests), 1)
        self.assertEqual(self.outcomes[-1].state, "accepted")
        self.assertEqual(self.outcomes[-1].provider_status, 200)
        self.assertEqual(self.service.status()["delivery"], "not_proven_by_provider_acceptance")
        self.assertEqual(self.service.status()["submission"], "durable_authorization_not_wire_evidence")

    async def test_controlled_partial_and_unknown_transport_never_claim_acceptance(self):
        for mode in ("partial", "unknown"):
            with self.subTest(mode=mode):
                self.control.mode = mode
                self.assertTrue(await self.service.run_once())
                self.assertEqual(self.outcomes[-1].state, "unknown")
                self.assertIsNone(self.outcomes[-1].provider_status)
                self.assertIsNone(self.outcomes[-1].retry_seconds)
        self.assertEqual(self.service.status()["provider_access"], "unverified")

    async def test_cancellation_before_authorization_is_not_a_submission(self):
        entered = asyncio.Event()
        async def authorize(request):
            entered.set()
            await asyncio.Future()
        self.repository.authorize_send.side_effect = authorize
        pending = asyncio.create_task(self.service.run_once())
        try:
            await asyncio.wait_for(entered.wait(), 2)
        finally:
            pending.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await pending
        self.assertEqual(self.outcomes[-1].state, "failed")
        self.assertEqual(self.outcomes[-1].reason, "push_service_stopped")
        self.assertEqual(self.control.requests, [])
