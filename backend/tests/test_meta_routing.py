"""Provider wire compatibility and unchanged non-executing failure behavior."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from app.model_providers.routing import (
    META_BASE_URL, RouterModelConfiguration, create_router_client, meta_api_key,
)
from test_live_router import RouterTests, response


class RouterConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_keeps_luna_and_never_inherits_meta_endpoint(self):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": META_BASE_URL}):
            client, config = create_router_client(NS(OPENAI_API_KEY="fixture-openai"), environ={})
        try:
            self.assertEqual(str(client.base_url), "https://api.openai.com/v1/")
            self.assertFalse(config.structured_selection)
            self.assertEqual(config.parameters(), {
                "model": "gpt-6-luna", "reasoning": {"effort": "none"},
                "tool_choice": "required", "max_output_tokens": 256})
        finally:
            await client.close()

    async def test_meta_uses_separate_key_and_supported_parameters(self):
        with TemporaryDirectory() as root:
            key = Path(root) / "key"
            key.write_text("fixture-meta\n")
            client, config = create_router_client(NS(OPENAI_API_KEY="fixture-openai"), environ={
                "JARVIS_ROUTER_PROVIDER": "meta", "JARVIS_META_API_KEY_FILE": str(key.resolve())})
            try:
                self.assertEqual(client.api_key, "fixture-meta")
                self.assertEqual(str(client.base_url), META_BASE_URL + "/")
                self.assertEqual(config.model, "muse-spark-1.3")
                self.assertEqual(config.reasoning_effort, "minimal")
                self.assertEqual(config.tool_choice, "auto")
                self.assertGreater(config.max_output_tokens, 256)
                self.assertTrue(config.structured_selection)
            finally:
                await client.close()

    async def test_invalid_provider_or_missing_meta_key_fails_before_any_request(self):
        with self.assertRaises(ValueError):
            create_router_client(NS(OPENAI_API_KEY="fixture"), environ={"JARVIS_ROUTER_PROVIDER": "typo"})
        with self.assertRaises(ValueError):
            meta_api_key({})
        with TemporaryDirectory() as root:
            key = Path(root) / "key"
            key.write_text("REPLACE_WITH_META_MODEL_API_KEY")
            with self.assertRaises(ValueError):
                meta_api_key({"JARVIS_META_API_KEY_FILE": str(key.resolve())})


class MetaRouterTests(RouterTests):
    # Inherit the existing behavioral cases, exercising exactly the same routing
    # pipeline with Meta-compatible request settings. No alternate action executor.
    def setUp(self):
        super().setUp()
        self.router.model_configuration = RouterModelConfiguration(
            model="muse-spark-1.3", reasoning_effort="minimal",
            tool_choice="auto", max_output_tokens=2048)

    async def test_meta_plain_text_never_becomes_an_action_or_extra_model_call(self):
        self.client.responses.create.return_value = NS(status="completed",
            output=[NS(type="message", content="Recording has started")], model_dump=lambda **kw: {})
        await self.run_route()
        self.work.assert_not_awaited()
        self.session.read_live_state.assert_not_awaited()
        self.client.responses.create.assert_awaited_once()
        self.assertIn("No action was started", self.publish.await_args.args[0].text)

    async def test_meta_handoff_preserves_packet_with_supported_wire_settings(self):
        self.client.responses.create.return_value = response("send_to_hermes")
        await self.run_route()
        self.work.assert_awaited_once()
        self.assertIs(self.work.await_args.args[0], self.packet)
        request = self.client.responses.create.call_args.kwargs
        self.assertEqual(request["model"], "muse-spark-1.3")
        self.assertEqual(request["reasoning"], {"effort": "minimal"})
        self.assertEqual(request["tool_choice"], "auto")
        self.assertFalse(request["parallel_tool_calls"])
        self.assertFalse(request["store"])
        self.assertNotIn("api_key", json.dumps(request))

    def test_quoted_null_is_not_silently_converted(self):
        from app.live_state.companion_tools import SCHEMAS
        from app.realtime.live_router import selected_call
        from pydantic import ValidationError

        model, description = SCHEMAS["control_app"]
        arguments = model.model_validate({
            "action": "recording.start", "include_jarvis_audio": True,
        }).model_dump(mode="json")
        arguments["volume"] = "null"
        schema = model.model_json_schema()
        schema["required"] = list(schema["properties"])
        tool = {"type": "function", "name": "control_app", "description": description,
                "strict": True, "parameters": schema}
        _, selected = selected_call(response("control_app", json.dumps(arguments)), [tool])
        self.assertEqual(selected["volume"], "null")
        with self.assertRaises(ValidationError):
            model.model_validate(selected)
