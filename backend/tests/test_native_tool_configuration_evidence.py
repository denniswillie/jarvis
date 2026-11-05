"""Actual native writer + real registry; external sockets/SQL are controlled offline.

An echoed configuration is not model adoption or a per-update causal ACK.
"""
import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.assistant.base_prompt import build_base_prompt
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.telemetry import safe_fields, tool_configuration_fields
from app.settings import JarvisSettings
import test_episode_check_tools as owner_fixture


REQUIRED = {"track_visual_memory", "list_episode_checks", "change_episode_check"}
OFFER = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"


def records(logs):
    return [row for record in logs.records if "realtime_trace " in record.getMessage()
            for row in [json.loads(record.getMessage().split("realtime_trace ", 1)[1])]
            if row["event"].startswith("provider_tool_configuration")]


class NativeToolConfigurationEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await owner_fixture.EpisodeToolTests.asyncSetUp(self)
        self.native = DirectRealtimeAudioSession(
            offer=OFFER, websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="tool-config-fixture", owner_id=self.scope.owner_id,
                jarvis_session_id=self.scope.session_id, client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None, RESPONSIVE_VOICE_ENABLED=True,
                OPENAI_API_KEY="fixture-only-not-a-real-api-key"),
            repository=SimpleNamespace(get_recent_messages=AsyncMock(return_value=[]), create_message=AsyncMock()),
            tool_modules=(self.module,), context_providers=())
        self.provider = self.native.openai
        self.wire = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(), close=AsyncMock())
        self.provider._websocket = self.wire
        self.context = SimpleNamespace(source_payload_ids=(), as_prompt_payload=lambda: {})
        self.addAsyncCleanup(self.native.close)

    async def update(self):
        await self.native._send_complete_session_update(self.context)
        return json.loads(self.wire.send.call_args.args[0])

    async def echo(self, configuration):
        event = {"type": "session.updated", "event_id": "provider-echo", "session": configuration}
        self.wire.recv.return_value = json.dumps(event)
        self.assertEqual(await self.provider.receive_event(), event)

    async def test_real_registry_native_configuration_reaches_actual_writer_before_any_frame(self):
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            sent = await self.update()
            await self.echo(sent["session"])
        names = {t["name"] for t in sent["session"]["tools"]}
        self.assertTrue(REQUIRED <= names)
        self.assertEqual(sent["session"]["tool_choice"], "auto")
        self.assertNotIn("format", sent["session"]["audio"]["input"])
        rows = records(logs)
        self.assertEqual([r["status"] for r in rows], ["write_possible", "write_returned", "provider_observed"])
        self.assertTrue(all(REQUIRED <= set(r["tool_names"]) for r in rows))
        self.assertEqual(len({r["tool_configuration_sha256"] for r in rows}), 1)
        self.assertEqual(rows[-1]["configuration_match"], "configuration_only")
        self.assertEqual(rows[-1]["configuration_dispatch_id"], rows[0]["dispatch_id"])
        self.assertFalse(self.store.rules)  # Advertisement does not fabricate admission.
        self.assertNotIn("instructions", rows[-1])

    async def test_http_creation_is_transport_only_followed_by_complete_native_update(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith("/hangup"):
                return httpx.Response(200)
            return httpx.Response(201, text=OFFER, headers={"location": "/v1/realtime/calls/rtc_fixture"})
        self.provider._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=self.wire)):
            await self.provider.connect()
        self.assertNotIn(b'"tools"', requests[0].content)
        sent = await self.update()
        self.assertTrue(REQUIRED <= {t["name"] for t in sent["session"]["tools"]})

    async def test_startup_enrichment_keeps_same_registry_tools(self):
        self.native._direct_warming_context = self.context
        self.native._assistant_context = self.context
        self.assertTrue(await self.native._publish_startup_context(self.context, 0))
        sent = json.loads(self.wire.send.call_args.args[0])
        self.assertTrue(REQUIRED <= {t["name"] for t in sent["session"]["tools"]})

    async def test_echo_before_send_return_is_observation_not_false_write_completion(self):
        async def send(encoded):
            await self.echo(json.loads(encoded)["session"])
        self.wire.send.side_effect = send
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await self.update()
        self.assertEqual([r["status"] for r in records(logs)],
                         ["write_possible", "provider_observed", "write_returned"])

    async def test_mismatched_duplicate_and_other_connection_echoes_remain_distinct(self):
        sent = await self.update()
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await self.echo({**sent["session"], "tools": []})
            await self.echo(sent["session"])
            await self.echo(sent["session"])
            self.provider._websocket = SimpleNamespace(recv=AsyncMock(return_value=json.dumps({
                "type": "session.updated", "session": sent["session"]})), close=AsyncMock())
            await self.provider.receive_event()
        rows = records(logs)
        self.assertEqual([r["configuration_match"] for r in rows],
                         ["different", "configuration_only", "configuration_only", "unavailable"])
        self.wire.send.assert_awaited_once()
        self.assertTrue(all(r["status"] == "provider_observed" for r in rows))

    async def test_failed_or_cancelled_write_remains_unknown_without_invented_ack(self):
        for error in (RuntimeError("PRIVATE_PROVIDER_ERROR"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                self.wire.send.side_effect = error
                with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                    with self.assertRaises(type(error)):
                        await self.update()
                rows = records(logs)
                self.assertEqual([r["status"] for r in rows], ["write_possible", "write_outcome_unknown"])
                self.assertNotIn("PRIVATE_PROVIDER_ERROR", str(rows))
                self.assertFalse(any("configuration_match" in r for r in rows))

    async def test_guard_failure_and_factory_skip_produce_no_tool_write_evidence(self):
        @asynccontextmanager
        async def rejected():
            raise RuntimeError("PRIVATE_GUARD")
            yield
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(RuntimeError):
                await self.provider.send_event({"type": "session.update", "session": {"tools": []}},
                                               write_guard=rejected)
            self.assertFalse(await self.provider.send_event(lambda: None))
        self.assertEqual(records(logs), [])
        self.wire.send.assert_not_awaited()

    async def test_ordinary_response_inheritance_and_explicit_empty_override_are_not_confused(self):
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await self.provider.send_event({"type": "response.create", "response": {}})
            await self.provider.send_event({"type": "response.create", "response": {"tools": [], "tool_choice": "none"}})
        rows = [r for r in records(logs) if r["status"] == "write_returned"]
        self.assertEqual([r["configuration_state"] for r in rows], ["inherited", "explicit"])
        self.assertNotIn("tool_count", rows[0])
        self.assertEqual(rows[1]["tool_count"], 0)
        self.assertEqual(rows[1]["tool_choice"], "none")

    async def test_diagnostic_failure_and_record_cap_do_not_change_writes_or_reads(self):
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=RuntimeError("PRIVATE_LOG_FAILURE")):
            sent = await self.update()
            await self.echo(sent["session"])
        self.provider._tool_configuration_records = 0
        with patch("app.realtime.openai_client.TOOL_CONFIGURATION_RECORD_LIMIT", 3):
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                for _ in range(4):
                    await self.update()
        rows = records(logs)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["event"], "provider_tool_configuration_truncated")
        self.assertEqual(self.wire.send.await_count, 5)

    async def test_disabled_gate_advertises_no_episode_tools_without_forcing_admission(self):
        disabled = DirectRealtimeAudioSession(offer=OFFER,
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()), context=self.native.context,
            settings=JarvisSettings(_env_file=None, RESPONSIVE_VOICE_ENABLED=False),
            repository=SimpleNamespace(get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(self.module,), context_providers=())
        self.addAsyncCleanup(disabled.close)
        disabled.openai._websocket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
        await disabled._send_complete_session_update(self.context)
        sent = json.loads(disabled.openai._websocket.send.call_args.args[0])
        self.assertEqual(sent["session"]["tools"], [])
        self.assertEqual(sent["session"]["tool_choice"], "none")
        self.assertFalse(self.store.rules)


class ToolConfigurationPrivacyAndPolicyTests(unittest.TestCase):
    def test_fingerprint_is_deterministic_schema_sensitive_and_content_free(self):
        config = {"instructions": "PRIVATE_INSTRUCTIONS", "tool_choice": "auto", "tools": [
            {"type": "function", "name": "track_visual_memory", "description": "PRIVATE_DESCRIPTION",
             "parameters": {"type": "object"}},
            {"type": "function", "name": "PRIVATE_TOOL_NAME", "parameters": {}}]}
        result = tool_configuration_fields(config)
        self.assertEqual(result["tool_names"], ["track_visual_memory"])
        self.assertEqual(result["other_tool_count"], 1)
        self.assertNotIn("PRIVATE", str(result))
        self.assertEqual(result, tool_configuration_fields(json.loads(json.dumps(config, sort_keys=True))))
        altered = json.loads(json.dumps(config))
        altered["tools"][0]["parameters"]["additionalProperties"] = False
        self.assertNotEqual(result["tool_configuration_sha256"],
                            tool_configuration_fields(altered)["tool_configuration_sha256"])

    def test_missing_null_malformed_and_oversized_shapes_never_claim_empty_tools(self):
        self.assertEqual(tool_configuration_fields({})["configuration_state"], "inherited")
        for config in (None, {"tools": None}, {"tools": [None]}, {"tools": [{}] * 65},
                       {"tools": [{"description": "x" * 262145}]}):
            with self.subTest(kind=type(config).__name__):
                self.assertEqual(tool_configuration_fields(config), {"configuration_state": "unavailable"})
        result = safe_fields({"tool_names": ["track_visual_memory", "PRIVATE"],
                              "tool_configuration_sha256": "PRIVATE", "tool_choice": "PRIVATE",
                              "configuration_state": "PRIVATE", "instructions": "PRIVATE"})
        self.assertEqual(result, {"tool_names": ["track_visual_memory"]})

    def test_future_tracking_precedes_camera_prerequisite_without_location_or_admission_claim(self):
        for enabled in (False, True):
            with self.subTest(tasks_enabled=enabled):
                prompt = build_base_prompt(tasks_enabled=enabled)
                for required in ("even before frames arrive", "not a prerequisite", "no_frame",
                    "Only tracking.status=accepted", "not an observed or saved", "later findings need visual",
                    "If the tool is absent or admission fails", "not remember with an unobserved location",
                    "not a substitute for requested placement-memory tracking"):
                    self.assertIn(required, prompt)
                self.assertIn("Use search_memory for historical recall", prompt)
                self.assertIn("use get_current_camera_view", prompt)
                self.assertNotIn("pepper", prompt.lower())

    def test_tool_description_accepts_future_observation_without_seeded_evidence(self):
        from app.reminders.episode_tools import EpisodeCheckToolModule
        from app.tools.module import ToolBuildContext
        module = EpisodeCheckToolModule(SimpleNamespace())
        definitions = {t.name: t for t in module.build_tools(ToolBuildContext(
            settings=SimpleNamespace(RESPONSIVE_VOICE_ENABLED=True), owner_id="owner", session_id=uuid4()))}
        tool = definitions["track_visual_memory"]
        self.assertIn("before frames arrive", tool.description)
        self.assertIn("not a prerequisite", tool.description)
        self.assertIn("never that a location fact was saved", tool.description)
        self.assertNotIn("location", tool.input_schema["properties"])
        self.assertNotIn("pepper", tool.description.lower())
