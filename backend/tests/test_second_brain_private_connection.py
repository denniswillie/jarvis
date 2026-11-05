"""Scoped private transport and mounted-key contracts, without external services."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
from pydantic import ValidationError

from app.knowledge.connection import hindsight_origin
from app.knowledge.hindsight import HindsightClient, HindsightError
from app.knowledge.runtime import KnowledgeRuntime
from app.runtime.feature_config import Capability, KnowledgeConfiguration, SecondBrainSettings


ORIGIN = "http://hindsight-api:8888"


def settings(**changes):
    return SecondBrainSettings(_env_file=None, **(dict(
        enabled="true", knowledge_enabled="true", hindsight_url=ORIGIN,
        hindsight_private_http_origin=ORIGIN, hindsight_api_key="fixture-memory-key",
        knowledge_area_id=str(uuid4()),
    ) | changes))


class PrivateConnectionTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults_stay_https_or_loopback(self):
        for origin in ("https://memory.invalid", "http://127.0.0.1:8888", "http://localhost:8888", "http://[::1]:8888"):
            with self.subTest(origin=origin):
                self.assertEqual(hindsight_origin(origin), origin)
        for origin in (ORIGIN, "http://10.20.0.5:8888", "http://remote.example"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                hindsight_origin(origin)

    def test_opt_in_is_exact_and_private_not_a_general_http_bypass(self):
        for origin in (ORIGIN, "http://10.1.2.3:8888", "http://172.21.0.2:8888",
                       "http://192.168.1.2:8888", "http://[fd00::2]:8888"):
            with self.subTest(origin=origin):
                self.assertEqual(hindsight_origin(origin, approved_http_origin=origin), origin)
        for origin in ("http://remote.example", "http://8.8.8.8", "http://169.254.169.254",
                       "http://2130706433", "http://0x08080808", "http://[fe80::1]",
                       "http://[2001:4860:4860::8888]"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                hindsight_origin(origin, approved_http_origin=origin)
        for origin in ("http://hindsight-api:8889", "http://different-api:8888"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                hindsight_origin(origin, approved_http_origin=ORIGIN)

    def test_origins_never_carry_credentials_path_query_or_control_characters(self):
        for origin in ("http://user:private@hindsight-api:8888", "http://@hindsight-api:8888",
                       ORIGIN+"/documents", ORIGIN+"?key=private", ORIGIN+"#private",
                       "http://hindsight-api:0", "http://hindsight-api:70000",
                       "http://hindsight-api:", "http://hindsight-api:bad",
                       " http://hindsight-api:8888", "http://hindsight-api:\n8888"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                hindsight_origin(origin, approved_http_origin=origin)
        self.assertEqual(hindsight_origin("HTTP://HINDSIGHT-API:8888/", approved_http_origin=ORIGIN), ORIGIN)

    def test_configuration_and_client_share_the_opt_in(self):
        resolved = settings().resolve(Capability.KNOWLEDGE, database_url="fixture-only")
        self.assertTrue(resolved.configured)
        self.assertEqual(resolved.configuration.private_http_origin, ORIGIN)
        self.assertEqual(resolved.configuration.base_url, ORIGIN)
        rejected = settings(hindsight_private_http_origin="").resolve(
            Capability.KNOWLEDGE, database_url="fixture-only")
        self.assertFalse(rejected.configured)
        self.assertNotIn("fixture-memory-key", repr(resolved))

    def test_mounted_key_is_resolved_only_for_enabled_knowledge(self):
        with TemporaryDirectory() as directory:
            secret = Path(directory)/"key"
            secret.write_text("fixture-mounted-private-key\n")
            resolved = settings(hindsight_api_key="", hindsight_api_key_file=str(secret)).resolve(
                Capability.KNOWLEDGE, database_url="fixture-only")
            self.assertTrue(resolved.configured)
            self.assertEqual(resolved.configuration.api_key.get_secret_value(), "fixture-mounted-private-key")
            self.assertNotIn("fixture-mounted-private-key", repr(resolved))
            self.assertNotIn("fixture-mounted-private-key", resolved.configuration.model_dump_json())
        for disabled in ({"enabled":"false"}, {"knowledge_enabled":"false"}):
            with self.subTest(disabled=disabled), patch("app.knowledge.service_entrypoint.secret_file") as read:
                resolved = settings(hindsight_api_key="", hindsight_api_key_file="/unavailable", **disabled).resolve(
                    Capability.KNOWLEDGE, database_url="fixture-only")
                self.assertFalse(resolved.enabled)
                read.assert_not_called()

    def test_ambiguous_and_unavailable_keys_are_sanitized(self):
        with patch("app.knowledge.service_entrypoint.secret_file") as read:
            ambiguous = settings(hindsight_api_key_file="/private-secret-location").resolve(
                Capability.KNOWLEDGE, database_url="fixture-only")
            self.assertEqual(ambiguous.issue, "memory_api_key_configuration_ambiguous")
            read.assert_not_called()
        with TemporaryDirectory() as directory:
            missing = settings(hindsight_api_key="", hindsight_api_key_file=str(Path(directory)/"missing")).resolve(
                Capability.KNOWLEDGE, database_url="fixture-only")
            self.assertEqual(missing.issue, "memory_api_key_file_unavailable")
            self.assertNotIn(directory, repr(missing))
        malformed = settings(hindsight_api_key="", hindsight_api_key_file="relative-key").resolve(
            Capability.KNOWLEDGE, database_url="fixture-only")
        self.assertEqual(malformed.issue, "memory_api_key_file_unavailable")

    async def test_client_keeps_authentication_and_rejects_redirects(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(302, headers={"location":"http://different-api:8888"})
        client = HindsightClient(ORIGIN, "fixture-memory-key", private_http_origin=ORIGIN,
                                 transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.close)
        with self.assertRaises(HindsightError):
            await client.document("fixture-bank", "fixture-document")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].headers["authorization"], "Bearer fixture-memory-key")
        self.assertEqual(calls[0].url.host, "hindsight-api")
        with self.assertRaises(ValueError):
            HindsightClient(ORIGIN, "fixture-memory-key")

    async def test_runtime_passes_the_validated_origin_to_the_client(self):
        config = settings().resolve(Capability.KNOWLEDGE, database_url="fixture-only").configuration
        connection = SimpleNamespace(fetchval=AsyncMock(return_value="public"))
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        sources = MagicMock()
        sources.pool = pool
        sources.transaction.return_value.__aenter__ = AsyncMock(return_value=connection)
        sources._area = AsyncMock()
        client = SimpleNamespace(close=AsyncMock())
        with patch("app.knowledge.runtime.read_schema_readiness", AsyncMock(return_value=SimpleNamespace(ready=True))), \
             patch("app.knowledge.runtime.SourceRepository", return_value=sources), \
             patch("app.knowledge.runtime.HindsightClient", return_value=client) as factory:
            runtime = await KnowledgeRuntime.create(
                pool=pool, owner_id="fixture-owner", configuration=config, tasks=SimpleNamespace(pool=pool))
            factory.assert_called_once_with(ORIGIN, "fixture-memory-key", timeout=20,
                                             private_http_origin=ORIGIN)
            await runtime.close()
        client.close.assert_awaited_once()
