"""Protocol/config tests use generated signing keys and MockTransport, never APNs."""
import asyncio
from datetime import UTC, datetime
import json
import logging
import unittest
from unittest.mock import patch
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from hpack import Decoder, Encoder
import httpx
import jwt
from pydantic import SecretStr, ValidationError

from app.attention.push_models import DeviceRegistration, PushAuthorization, PushClaim, PushRequest
from app.attention.push_provider import APNsProvider
from app.runtime.feature_config import Capability, PushConfiguration, SecondBrainSettings
from app.realtime.session_manager import RealtimeSessionManager


def generated_key():
    return ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()


def config(**changes):
    return PushConfiguration(private_key=SecretStr(generated_key()), key_id="ABCDEFGHIJ",
        team_id="0123456789", **changes)


async def protocol_authorization(request):
    """Protocol-only approval; durable one-use admission is qualified in SQL."""
    return PushAuthorization(request.claim, True, "fixture_protocol_authorization")


class PushConfigurationTests(unittest.TestCase):
    def test_lazy_disabled_key_and_selected_invalid_configuration_are_sanitized(self):
        flags = SecondBrainSettings(_env_file=None, enabled="false", push_enabled="bad",
                                    push_private_key="private-broken-fixture", push_http_timeout_seconds="bad")
        self.assertFalse(flags.resolve(Capability.PUSH).enabled)
        flags = SecondBrainSettings(_env_file=None, enabled="true", push_enabled="true")
        self.assertEqual(flags.resolve(Capability.PUSH, database_url="configured").issue, "push_key_not_configured")
        flags = SecondBrainSettings(_env_file=None, enabled="true", push_enabled="true",
            push_private_key="private-broken-fixture", push_key_id="ABCDEFGHIJ", push_team_id="0123456789",
            push_http_timeout_seconds="bad")
        resolution = flags.resolve(Capability.PUSH, database_url="configured")
        self.assertEqual(resolution.issue, "invalid_configuration")
        self.assertNotIn("private-broken-fixture", repr(resolution))
        self.assertNotIn("private-broken-fixture", repr(flags))

    def test_native_key_is_required_but_not_exposed_in_config(self):
        value = config()
        self.assertNotIn("BEGIN PRIVATE KEY", repr(value))
        self.assertNotIn("BEGIN PRIVATE KEY", value.model_dump_json())
        with self.assertRaises(ValueError):
            APNsProvider(value.model_copy(update={"private_key": SecretStr("not-a-key")}))
        with self.assertRaises(ValidationError):
            config(http_timeout_seconds=29, lease_seconds=30)

    def test_device_permissions_token_shape_and_quiet_time_are_validated(self):
        values = dict(request_id=uuid4(), expected_revision=0, environment="sandbox",
            topic="com.symbiote.jarvis", token="AB" * 80, permission="authorized", enabled=True)
        record = DeviceRegistration(**values)
        self.assertEqual(record.token.get_secret_value(), "ab" * 80)
        self.assertNotIn("ab" * 80, repr(record))
        for updates in ({"token": "../token"}, {"token": "abc"}, {"environment": "http://external.invalid"},
                        {"topic": "invalid\nheader"}, {"permission": "denied"},
                        {"quiet_until": datetime(2026, 1, 1)}, {"owner_id": "other"}):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                DeviceRegistration(**(values | updates))
        DeviceRegistration(**(values | {"enabled": False, "permission": "denied", "token": None}))


class APNsProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.configuration = config()
        self.claim = PushClaim("fixture-owner", uuid4(), uuid4(), 1, uuid4(), 1, "fixture-worker")
        self.request = PushRequest(self.claim, "ab" * 64, "sandbox", self.configuration.topic, None, "normal")

    async def provider(self, handler, **kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = APNsProvider(self.configuration, client=client, **kwargs)
        self.addAsyncCleanup(provider.close)
        return provider

    async def test_signed_private_payload_identity_expiry_and_connection_reuse(self):
        requests = []
        def handle(request):
            requests.append(request)
            return httpx.Response(200, headers={"apns-id": str(self.claim.attempt_id)})
        clock = [1789551000]
        provider = await self.provider(handle, clock=lambda: clock[0])
        original_client = provider.client
        self.assertEqual((await provider.send(self.request, authorize=protocol_authorization)).state, "accepted")
        clock[0] += 60
        await provider.send(self.request, authorize=protocol_authorization)
        self.assertIs(provider.client, original_client)
        first = requests[0]
        self.assertEqual(first.url.host, "api.sandbox.push.apple.com")
        self.assertEqual(first.headers["apns-expiration"], "0")
        self.assertEqual(first.headers["apns-collapse-id"], str(self.claim.intent_id))
        self.assertEqual(first.headers["apns-priority"], "5")
        payload = json.loads(first.content)
        self.assertEqual(payload["jarvis"], {"version": 1, "intent_id": str(self.claim.intent_id)})
        self.assertEqual(set(payload), {"aps", "jarvis"})
        self.assertEqual(payload["aps"]["alert"]["body"], "An update is available in your inbox.")
        token = first.headers["authorization"].split(" ", 1)[1]
        decoded = jwt.decode(token, provider._key.public_key(), algorithms=["ES256"],
                             options={"verify_iat": False})
        self.assertEqual(decoded, {"iss": "0123456789", "iat": clock[0] - 60})
        self.assertEqual(jwt.get_unverified_header(token)["kid"], "ABCDEFGHIJ")
        self.assertEqual(requests[1].headers["authorization"], first.headers["authorization"])
        clock[0] += 3000
        await provider.send(self.request, authorize=protocol_authorization)
        self.assertNotEqual(requests[2].headers["authorization"], first.headers["authorization"])

    async def test_transport_logs_hide_push_secrets_without_hiding_concurrent_requests(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def handle(request):
            if request.url.host.endswith("push.apple.com"):
                encoded = Encoder().encode([
                    (":path", request.url.raw_path),
                    ("authorization", request.headers["authorization"]),
                ])
                Decoder().decode(encoded)
                started.set()
                await release.wait()
            return httpx.Response(200)
        provider = await self.provider(handle)
        with self.assertLogs(level="DEBUG") as captured:
            pending = asyncio.create_task(provider.send(self.request, authorize=protocol_authorization))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                await provider.client.get("https://fixture.invalid/ordinary-concurrent")
            finally:
                release.set()
                outcome = await asyncio.wait_for(pending, timeout=5)
            await provider.client.get("https://fixture.invalid/ordinary-after")
        self.assertEqual(outcome.state, "accepted")
        output = "\n".join(captured.output)
        self.assertNotIn(self.request.token, output)
        self.assertNotIn(provider._token, output)
        self.assertNotIn("push.apple.com", output)
        self.assertIn("ordinary-concurrent", output)
        self.assertIn("ordinary-after", output)

    async def test_transport_privacy_context_resets_after_error_and_cancellation(self):
        for failure in (httpx.ReadTimeout, asyncio.CancelledError):
            with self.subTest(failure=failure.__name__):
                def handle(request):
                    if request.url.host.endswith("push.apple.com"):
                        raise failure("fixture")
                    return httpx.Response(200)
                provider = await self.provider(handle)
                with self.assertLogs("httpx", level=logging.INFO) as captured:
                    if failure is asyncio.CancelledError:
                        with self.assertRaises(asyncio.CancelledError):
                            await provider.send(self.request, authorize=protocol_authorization)
                    else:
                        self.assertEqual((await provider.send(self.request, authorize=protocol_authorization)).state, "unknown")
                    await provider.client.get("https://fixture.invalid/ordinary-after-failure")
                self.assertIn("ordinary-after-failure", "\n".join(captured.output))

    async def test_timeout_is_unknown_not_retryable_rejection(self):
        def handle(request): raise httpx.ReadTimeout("fixture", request=request)
        provider = await self.provider(handle)
        outcome = await provider.send(self.request, authorize=protocol_authorization)
        self.assertEqual(outcome.state, "unknown")
        self.assertIsNone(outcome.retry_seconds)
        self.assertNotIn(self.request.token, repr(outcome))

    async def test_status_mapping_invalidation_and_bounded_retry(self):
        cases = [
            (410, {"reason": "Unregistered", "timestamp": 1789551000000}, True, None),
            (410, {"reason": "Unregistered"}, False, None),
            (400, {"reason": "BadDeviceToken"}, True, None),
            (403, {"reason": "InvalidProviderToken"}, False, None),
            (429, {"reason": "TooManyRequests"}, False, 60),
            (503, {"reason": "ServiceUnavailable"}, False, 900),
            (400, {"reason": {"private": "unexpected"}}, False, None),
        ]
        for status, body, invalid, retry in cases:
            with self.subTest(status=status, body=body):
                provider = await self.provider(lambda request, s=status, b=body: httpx.Response(s, json=b))
                result = await provider.send(self.request, authorize=protocol_authorization)
                self.assertEqual(result.state, "failed")
                self.assertEqual(result.invalidate_token, invalid)
                self.assertEqual(result.retry_seconds, retry)
                self.assertNotIn("unexpected", repr(result))
        provider = await self.provider(lambda request: httpx.Response(410, text="not json"))
        self.assertEqual((await provider.send(self.request, authorize=protocol_authorization)).reason, "apns_rejected")

    async def test_topic_and_environment_are_not_model_selected_network_destinations(self):
        calls = []
        provider = await self.provider(lambda request: calls.append(request) or httpx.Response(200))
        for changes in ({"topic": "another.app"}, {"environment": "production"}):
            request = PushRequest(self.claim, self.request.token,
                changes.get("environment", "sandbox"), changes.get("topic", self.configuration.topic), None, "normal")
            self.assertEqual((await provider.send(request, authorize=protocol_authorization)).reason, "push_registration_mismatch")
        self.assertEqual(calls, [])

    async def test_voice_presence_is_scoped_and_never_location_evidence(self):
        manager = RealtimeSessionManager()
        class Session:
            async def try_attention(self): pass
        identity = uuid4()
        self.assertFalse(await manager.has_connected_voice("fixture-owner"))
        await manager.register(audio_session_id="fixture", owner_id="fixture-owner",
                               jarvis_session_id=identity, session=Session())
        self.assertTrue(await manager.has_connected_voice("fixture-owner"))
        self.assertFalse(await manager.has_connected_voice("another"))
        await manager.unregister(audio_session_id="fixture", owner_id="fixture-owner", jarvis_session_id=identity)
        self.assertFalse(await manager.has_connected_voice("fixture-owner"))
