"""Real owner/source/lease gates in disposable schemas; no calls to Apple."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr

from app.attention.models import AttentionError, InboxMutation
from app.attention.push_models import DeviceRegistration, PushOutcome
from app.attention.push_repository import PushRepository
from app.attention.push_service import PushService
from app.runtime.feature_config import PushConfiguration, SecondBrainSettings
from app.attention.service import initialize_attention_runtime
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.work_phone_fixture import create_app


class Manager:
    connected = False
    async def has_connected_voice(self, owner): return self.connected


class Provider:
    def __init__(self):
        self.requests = []
        self.outcome = PushOutcome("accepted", provider_status=200)
        self.callback = None
        self.closed = False
    async def send(self, request, *, authorize):
        decision = await authorize(request)
        if not decision.allowed:
            return PushOutcome("revoked", decision.reason)
        self.requests.append(request)
        if self.callback: await self.callback(request)
        return self.outcome
    async def close(self): self.closed = True


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database required")
class PhonePushRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.app, self.ids = await create_app(self.fixture, "synthetic-work-token")
        self.attention = self.app.state.runtime_services.attention_service.repository
        self.configuration = PushConfiguration(private_key=SecretStr("fixture-not-used-by-provider"),
            key_id="ABCDEFGHIJ", team_id="0123456789", fallback_seconds=0)
        self.repository = PushRepository(self.attention, self.configuration)
        self.provider, self.manager = Provider(), Manager()
        self.service = PushService(self.repository, self.provider, self.manager, self.fixture.owner, self.configuration)
        self.app.state.runtime_services.attention_service.push = self.service
        self.addAsyncCleanup(self.service.close)
        self.client = await self.enterAsyncContext(httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app),
            base_url="http://fixture.invalid", headers={"Authorization": "Bearer synthetic-work-token"}))
        self.intent = UUID(self.ids["question_intent_id"])
        self.task = await self.attention.tasks.get(self.fixture.owner, UUID(self.ids["waiting_task_id"]))
        # Keep this admission test focused on a single current unanswered question.
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("""update jarvis_attention_intents set state='dismissed',generation=generation+1
                where owner_id=$1 and id<>$2""", self.fixture.owner, self.intent)
        self.device = uuid4()
        self.registration = dict(request_id=str(uuid4()), expected_revision=0, token="ab" * 64,
            environment="sandbox", topic="com.symbiote.jarvis", permission="authorized", enabled=True)
        response = await self.client.put("/attention/devices/" + str(self.device), json=self.registration)
        self.assertEqual(response.status_code, 200, response.text)

    async def detail(self):
        return (await self.client.get("/attention/" + str(self.intent))).json()

    async def expire(self, claim):
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()-interval '1 second' where owner_id=$1 and id=$2",
                                     self.fixture.owner, claim.attempt_id)

    async def test_device_registration_auth_idempotency_rotation_and_revocation(self):
        path = "/attention/devices/" + str(self.device)
        self.assertEqual((await self.client.get("/attention/devices", headers={"Authorization": "Bearer wrong"})).status_code, 403)
        repeated = (await self.client.put(path, json=self.registration)).json()
        self.assertTrue(repeated["replayed_request"])
        self.assertEqual(repeated["accepted_revision"], 1)
        changed = self.registration | {"request_id": str(uuid4()), "token": "cd" * 64, "expected_revision": 1}
        self.assertEqual((await self.client.put(path, json=changed)).json()["device"]["revision"], 2)
        repeated = (await self.client.put(path, json=self.registration)).json()
        self.assertEqual(repeated["device"]["revision"], 2)
        self.assertEqual(repeated["accepted_revision"], 1)
        self.assertEqual((await self.client.put(path, json=changed | {"request_id": str(uuid4())})).status_code, 409)
        foreign = uuid4()
        await self.repository.change_device(self.fixture.other, foreign,
            DeviceRegistration(**(self.registration | {"request_id": str(uuid4()), "token": "ef" * 64})))
        self.assertEqual((await self.client.put("/attention/devices/" + str(foreign), json=self.registration)).status_code, 404)
        visible = (await self.client.get("/attention/devices")).json()
        self.assertEqual([row["id"] for row in visible], [str(self.device)])
        self.assertNotIn("token", json.dumps(visible))
        self.assertNotIn(self.registration["token"], json.dumps(visible))
        body = {"request_id": str(uuid4()), "expected_revision": 2}
        result = (await self.client.post(path + "/revoke", json=body)).json()
        self.assertFalse(result["device"]["enabled"])
        self.assertEqual((await self.client.post(path + "/revoke", json=body)).json()["accepted_revision"], 3)
        async with self.fixture.pool.acquire() as connection:
            self.assertIsNone(await connection.fetchval("select token from jarvis_push_devices where owner_id=$1 and id=$2", self.fixture.owner, self.device))
        self.assertFalse(await self.service.run_once())

    async def test_accepted_push_blocks_automatic_voice_but_explicit_replay_does_not_rerun_work(self):
        self.manager.connected = True
        self.assertFalse(await self.service.run_once())
        self.manager.connected = False
        before = await self.attention.tasks.get(self.fixture.owner, self.task.id)
        self.assertTrue(await self.service.run_once())
        detail = await self.detail()
        self.assertEqual(detail["phone_notice_state"], "accepted")
        self.assertEqual(detail["state"], "pending")
        self.assertIn("not_established", detail["hearing"])
        self.assertEqual(detail["attempts"][-1]["channel"], "phone")
        self.assertEqual(detail["attempts"][-1]["phase"], "accepted")
        self.assertFalse(await self.service.run_once())
        voice = await self.attention.claim_next(self.fixture.owner, self.task.session_id, "fixture-voice")
        self.assertIsNone(voice)
        await self.attention.mutate(self.fixture.owner, self.intent, "replay",
            InboxMutation(request_id=uuid4(), expected_generation=detail["generation"]))
        claim = await self.attention.claim_next(self.fixture.owner, self.task.session_id, "fixture-voice")
        self.assertIsNotNone(claim)
        self.assertEqual((await self.attention.tasks.get(self.fixture.owner, self.task.id)), before)
        self.assertEqual(len(self.provider.requests), 1)

    async def test_submitted_crash_recovers_while_connected_without_repeating_push(self):
        claim = await self.repository.claim_next(self.fixture.owner, "crashed-phone")
        request = await self.repository.prepare_submission(claim)
        self.assertIsNotNone(request)
        self.assertTrue((await self.repository.authorize_send(request)).allowed)
        await self.expire(claim)
        self.manager.connected = True
        self.assertFalse(await self.service.run_once())
        self.assertEqual((await self.detail())["phone_notice_state"], "unknown")
        self.assertIsNone(await self.attention.claim_next(self.fixture.owner, self.task.session_id, "voice"))
        self.manager.connected = False
        self.assertFalse(await self.service.run_once())
        self.assertEqual(self.provider.requests, [])

    async def test_unsubmitted_crash_releases_voice_and_device_revocation_fences_submission(self):
        claim = await self.repository.claim_next(self.fixture.owner, "crashed-phone")
        await self.expire(claim)
        self.manager.connected = True
        self.assertFalse(await self.service.run_once())
        self.assertIsNone((await self.detail())["phone_notice_state"])
        self.manager.connected = False
        claim = await self.repository.claim_next(self.fixture.owner, "new-phone")
        response = await self.client.post("/attention/devices/" + str(self.device) + "/revoke",
            json={"request_id": str(uuid4()), "expected_revision": 1})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(await self.repository.prepare_submission(claim))
        await self.repository.finish(claim, PushOutcome("revoked", "device_revoked"))
        self.assertIsNone((await self.detail())["phone_notice_state"])
        self.assertIsNotNone(await self.attention.claim_next(self.fixture.owner, self.task.session_id, "voice"))

    async def test_unknown_transport_is_not_retried(self):
        self.provider.outcome = PushOutcome("unknown", "fixture_network_uncertain")
        self.assertTrue(await self.service.run_once())
        self.assertEqual((await self.detail())["phone_notice_state"], "unknown")
        self.assertFalse(await self.service.run_once())
        self.assertEqual(len(self.provider.requests), 1)

    async def test_retryable_rejection_waits_then_uses_a_new_attempt_without_mutating_work(self):
        self.provider.outcome = PushOutcome("failed", "ServiceUnavailable", 503, retry_seconds=900)
        self.assertTrue(await self.service.run_once())
        self.assertIsNone((await self.detail())["phone_notice_state"])
        self.assertFalse(await self.service.run_once())
        self.service._provider_backoff = 0
        self.assertFalse(await self.service.run_once())
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("update jarvis_push_submissions set retry_at=clock_timestamp()-interval '1 second' where owner_id=$1", self.fixture.owner)
        self.provider.outcome = PushOutcome("accepted", provider_status=200)
        self.assertTrue(await self.service.run_once())
        self.assertEqual(len(self.provider.requests), 2)
        self.assertNotEqual(self.provider.requests[0].claim.attempt_id, self.provider.requests[1].claim.attempt_id)

    async def test_stale_token_response_does_not_invalidate_new_registration(self):
        async def rotate(request):
            await self.repository.change_device(self.fixture.owner, self.device, DeviceRegistration(**(
                self.registration | {"request_id": str(uuid4()), "expected_revision": 1, "token": "ff" * 64})))
        self.provider.callback = rotate
        self.provider.outcome = PushOutcome("failed", "Unregistered", 410, invalidate_token=True,
                                           invalidated_at=datetime.now(UTC) + timedelta(seconds=1))
        self.assertTrue(await self.service.run_once())
        device = (await self.repository.devices(self.fixture.owner))[0]
        self.assertTrue(device["enabled"])
        self.assertEqual(device["revision"], 2)
        self.assertIsNone(device["invalid_reason"])

    async def test_source_revocation_during_submission_never_restores_the_answer(self):
        async def revoke(request):
            result = await self.client.post("/__fixture/revoke/" + str(self.intent))
            self.assertEqual(result.status_code, 200)
        self.provider.callback = revoke
        self.assertTrue(await self.service.run_once())
        detail = await self.detail()
        self.assertFalse(detail["source_available"])
        self.assertIsNone(detail["expert_result"])
        self.assertIsNone(detail["phone_notice_state"])
        self.assertEqual(detail["attempts"][-1]["phase"], "revoked")
        self.assertFalse(await self.service.run_once())
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select provider_status from jarvis_push_submissions where owner_id=$1", self.fixture.owner), 200)

    async def test_quiet_until_and_permission_denial_prevent_delivery(self):
        path = "/attention/devices/" + str(self.device)
        quiet = datetime.now(UTC) + timedelta(hours=1)
        response = await self.client.put(path, json=self.registration | {
            "request_id": str(uuid4()), "expected_revision": 1, "quiet_until": quiet.isoformat()})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(await self.service.run_once())
        response = await self.client.put(path, json=self.registration | {
            "request_id": str(uuid4()), "expected_revision": 2, "enabled": False, "permission": "denied", "token": None})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(await self.service.run_once())
        self.assertEqual(self.provider.requests, [])

    async def test_source_revocation_before_submission_never_calls_provider(self):
        claim = await self.repository.claim_next(self.fixture.owner, "fixture-admission")
        await self.client.post("/__fixture/revoke/" + str(self.intent))
        self.assertIsNone(await self.repository.prepare_submission(claim))
        await self.repository.finish(claim, PushOutcome("revoked", "source_removed"))
        self.assertFalse(await self.service.run_once())
        self.assertEqual(self.provider.requests, [])

    async def test_old_notified_generations_do_not_starve_new_notifications(self):
        async with self.fixture.pool.acquire() as connection:
            await connection.execute("""with old as (
                insert into jarvis_attention_intents
                    (owner_id,source_namespace,source_id,source_revision,source_payload_id,task_id,task_revision,
                     preferred_session_id,state,generation,created_at)
                select owner_id,source_namespace,gen_random_uuid(),source_revision,source_payload_id,task_id,
                       task_revision,preferred_session_id,'pending',2,clock_timestamp()-interval '1 hour'
                from jarvis_attention_intents cross join generate_series(1,55) where owner_id=$1 and id=$2
                returning owner_id,id
            ), receipts as (
                insert into jarvis_attention_attempts
                    (owner_id,intent_id,generation,channel,worker_id,phase,lease_until,ended_at)
                select owner_id,id,1,'phone','synthetic-old-notice','accepted',clock_timestamp(),clock_timestamp()
                from old returning owner_id,id
            )
            insert into jarvis_push_submissions(owner_id,attempt_id,device_id,device_revision,provider_status)
            select owner_id,id,$3,1,200 from receipts""", self.fixture.owner, self.intent, self.device)
        self.assertTrue(await self.service.run_once())
        self.assertEqual(self.provider.requests[0].claim.intent_id, self.intent)
        self.assertFalse(await self.service.run_once())
        self.assertEqual(len(self.provider.requests), 1)

    async def test_invalid_native_key_leaves_existing_inbox_available(self):
        runtime = self.app.state.runtime_services
        runtime.db_pool = self.fixture.pool
        runtime.settings = runtime.settings.model_copy(update={
            "SUPABASE_DB_URL": os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]})
        runtime.realtime_session_manager = self.manager
        flags = SecondBrainSettings(_env_file=None, enabled="true", attention_enabled="true",
            push_enabled="true", push_private_key="fixture-invalid-pem",
            push_key_id="ABCDEFGHIJ", push_team_id="0123456789")
        with patch("app.attention.service.get_second_brain_settings", return_value=flags):
            await initialize_attention_runtime(runtime)
        self.addAsyncCleanup(runtime.attention_service.close)
        self.assertIsNone(runtime.attention_issue)
        self.assertEqual(runtime.attention_service.push_issue, "push_initialization_failed")
        self.assertIsNone(runtime.attention_service.push)
        self.assertTrue((await self.detail())["source_available"])
        self.assertEqual((await self.client.get("/attention/devices")).status_code, 503)

    async def test_shutdown_settles_a_submitted_request_before_database_close(self):
        entered = asyncio.Event()
        async def pending(request):
            entered.set()
            await asyncio.Future()
        self.provider.callback = pending
        self.service.start()
        await asyncio.wait_for(entered.wait(), timeout=20)
        await self.service.close()
        self.assertTrue(self.provider.closed)
        detail = await self.detail()
        self.assertEqual(detail["phone_notice_state"], "unknown")
        self.assertEqual(detail["attempts"][-1]["phase"], "unknown")

    async def test_reminder_push_is_only_a_notice_not_physical_completion(self):
        rows = await self.attention.list(self.fixture.owner, include_dismissed=True)
        reminder = next(row for row in rows if row["source_namespace"] == "reminder_event")
        await self.attention.mutate(self.fixture.owner, self.intent, "dismiss",
            InboxMutation(request_id=uuid4(), expected_generation=1))
        await self.attention.mutate(self.fixture.owner, reminder["id"], "replay",
            InboxMutation(request_id=uuid4(), expected_generation=reminder["generation"]))
        async with self.fixture.pool.acquire() as connection:
            before = await connection.fetchval("select status from jarvis_reminders where owner_id=$1", self.fixture.owner)
        self.assertTrue(await self.service.run_once())
        self.assertEqual(self.provider.requests[0].claim.intent_id, reminder["id"])
        detail = await self.attention.detail(self.fixture.owner, reminder["id"])
        self.assertEqual(detail["state"], "pending")
        self.assertEqual(detail["phone_notice_state"], "accepted")
        self.assertIn("not_established", detail["hearing"])
        async with self.fixture.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select status from jarvis_reminders where owner_id=$1", self.fixture.owner), before)

    async def test_token_refresh_keeps_quiet_until_unless_explicitly_cleared(self):
        path = "/attention/devices/" + str(self.device)
        quiet = datetime.now(UTC) + timedelta(hours=1)
        changed = self.registration | {"request_id": str(uuid4()), "expected_revision": 1,
                                       "quiet_until": quiet.isoformat()}
        self.assertEqual((await self.client.put(path, json=changed)).status_code, 200)
        refresh = self.registration | {"request_id": str(uuid4()), "expected_revision": 2, "token": "cd" * 64}
        result = await self.client.put(path, json=refresh)
        self.assertEqual(result.status_code, 200)
        self.assertIsNotNone(result.json()["device"]["quiet_until"])
        self.assertFalse(await self.service.run_once())
        # Omission and explicitly clearing a quiet preference are different effects.
        self.assertEqual((await self.client.put(path, json=refresh | {"quiet_until": None})).status_code, 409)
        cleared = refresh | {"request_id": str(uuid4()), "expected_revision": 3, "quiet_until": None}
        self.assertEqual((await self.client.put(path, json=cleared)).status_code, 200)
        self.assertTrue(await self.service.run_once())
