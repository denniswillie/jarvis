"""Offline canaries against the exact pinned vendor source, without importing a gateway.

Run from repo root:
PYTHONPATH=.:backend:backend/tests python -m unittest hermes_sessions.test_static_prompt_compat
Vendor fixtures live in .jarvis-dev; absent fixtures explicitly skip this suite.
"""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from scripts import stage_hermes_static_prompts as compat


def functions(source: str, names: set[str], namespace: dict) -> dict:
    tree = ast.parse(source)
    selected = [node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    if {node.name for node in selected} != names:
        raise AssertionError("Missing pinned function")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-compat-canary>", "exec"), namespace)
    return namespace


class StaticPromptCompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((compat.REPO / path).exists() for path, _ in compat.TARGETS.values()):
            raise unittest.SkipTest("Exact private vendor fixtures are not installed")
        cls.originals = {name: (compat.REPO / path).read_bytes() for name, (path, _) in compat.TARGETS.items()}
        cls.changed = {name: compat.patched(name, data).decode() for name, data in cls.originals.items()}

    def setUp(self):
        self.env = patch.dict(os.environ, {compat.FLAG: "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def plugin(self):
        ns = functions(self.changed["hindsight"], {"system_prompt_block", "_recall", "_tool_recall"}, {
            "os": os, "datetime": datetime, "timezone": timezone, "asyncio": asyncio,
            "logger": logging.getLogger("compat-canary"),
            "_SYSTEM_PROMPT_TAILS": {"tools": "STATIC_TOOL_GUIDANCE", "hybrid": "STATIC_HYBRID_GUIDANCE"},
            "_TOOL_GUIDANCE": "STATIC_TOOL_GUIDANCE", "_DEFAULT_TIMEOUT": 30,
            "_EXTRA_BANK_TIMEOUT_SHARE": 0.8, "_bank_not_written_yet": lambda exc: False,
        })
        obj = types.SimpleNamespace(
            _memory_mode="tools", _auto_recall=False, _bank_id="BANK_CANARY",
            _budget="BUDGET_CANARY", _recall_max_tokens=1500, _recall_tags=["TAG_CANARY"],
            _recall_tags_match="any", _recall_types=["world"], _recall_min_scores=None,
            _timeout=30, _build_recall_bank_ids=lambda: ["BANK_CANARY", "MIRROR_CANARY"],
            _extra_bank_available=lambda *args: True, _mark_bank_ok=lambda *args: None,
        )
        for name in ("system_prompt_block", "_recall", "_tool_recall"):
            setattr(obj, name, types.MethodType(ns[name], obj))
        return obj

    def native(self):
        never = Mock(side_effect=AssertionError("Dynamic fetch must not run"))
        pb = types.SimpleNamespace(build_environment_hints=never, build_context_files_prompt=never)
        ns = functions(self.changed["hermes"], {
            "_jarvis_static_instructions_only", "_active_profile_line", "_timestamp_line",
            "_alibaba_identity_part", "_post_workspace_parts", "_bot_mode_parts", "_coding_parts",
            "_context_files_part", "_memory_parts", "_join_tier", "build_system_prompt_parts",
        }, {
            "os": os, "_pb": pb,
            "_identity_parts": lambda *args: (["STATIC_SOUL"], True),
            "_guidance_parts": lambda *args: ["STATIC_TOOL_SAFETY"],
            "_skills_prompt": lambda *args: "STATIC_SKILL_CAPABILITIES",
            "_auto_load_parts": lambda *args: ["STATIC_PINNED_SKILL"],
            "_frozen_plugin_prompt_sections": lambda *args: (),
            "_plugin_section_blocks": lambda *args: ["STATIC_PLUGIN_GUIDANCE"],
            "_agent_home": lambda agent: agent.home,
            "platform_hint": lambda agent: "STATIC_PLATFORM_GUIDANCE",
            "resolve_context_cwd": lambda: "/WORKSPACE_CANARY",
            "_workspace_pin_key": never, "_seed_workspace_pin": never,
            "HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS": "STATIC_HELP",
            "HERMES_AGENT_HELP_GUIDANCE": "STATIC_SKILL_HELP",
        })
        provider = self.plugin()
        agent = types.SimpleNamespace(
            valid_tool_names={"terminal", "hindsight_recall"}, platform="PLATFORM_CANARY",
            model="MODEL_CANARY", provider="alibaba", session_id="SESSION_CANARY", pass_session_id=True,
            home="/PROFILE_CANARY", skip_context_files=False, _environment_probe=True,
            _bot_mode_protocol=True, _session_title_hint="BOT_CHAT",
            _memory_store=types.SimpleNamespace(format_for_system_prompt=never),
            _memory_enabled=True, _user_profile_enabled=True,
            _memory_manager=types.SimpleNamespace(build_system_prompt=provider.system_prompt_block),
            _frozen_workspace_snapshot=("OLD_PIN", "OLD_WORKSPACE_CANARY"),
        )
        coding = Mock(side_effect=lambda **kw: (["STATIC_CODING_SAFETY"], [kw["workspace_block"]] if kw["workspace_block"] else [], ["STATIC_OPERATOR_POLICY"]))
        modules = {
            "hermes_time": types.SimpleNamespace(get_timezone=lambda: "Etc/UTC",
                now=lambda: datetime(2026, 11, 4, 6, 7, 8, tzinfo=timezone.utc),
                safe_strftime=lambda value, fmt: value.strftime(fmt)),
            "agent.coding_context": types.SimpleNamespace(coding_system_prompt_parts=coding),
            "agent.memory_manager_tools": types.SimpleNamespace(memory_provider_tools_exposed=lambda _: True),
            "tools.env_probe": types.SimpleNamespace(get_environment_probe_line=never),
            "tools.bot_mode_probe": types.SimpleNamespace(BOT_CHAT_TITLE="BOT_CHAT", epoch_line=never,
                get_bot_mode_protocol_section=lambda _: "STATIC_BOT_SAFETY"),
        }
        return ns, agent, modules, coding, never

    def test_static_clock_keeps_time_and_timezone_without_runtime_identity(self):
        ns, agent, modules, _, _ = self.native()
        with patch.dict("sys.modules", modules):
            clock = ns["_timestamp_line"](agent)
        self.assertEqual(clock, "Clock (as of prompt build): 2026-11-04T06:07:08+00:00 (Etc/UTC)")
        for value in (agent.session_id, agent.model, agent.provider, agent.platform, agent.home):
            self.assertNotIn(value, clock)

    def test_constructed_prompt_has_static_guidance_without_runtime_canaries(self):
        ns, agent, modules, coding, never = self.native()
        with patch.dict("sys.modules", modules):
            parts = ns["build_system_prompt_parts"](agent, "EXPLICIT_CALLER_POLICY")
        prompt = "\n".join(parts.values())
        self.assertNotIn("CANARY", prompt)
        for text in ("STATIC_SOUL", "STATIC_TOOL_SAFETY", "STATIC_PLATFORM_GUIDANCE", "STATIC_BOT_SAFETY",
                     "STATIC_CODING_SAFETY", "STATIC_OPERATOR_POLICY", "STATIC_PLUGIN_GUIDANCE",
                     "EXPLICIT_CALLER_POLICY", "STATIC_TOOL_GUIDANCE", "Do not modify another profile"):
            self.assertIn(text, prompt)
        self.assertEqual(coding.call_args.kwargs["workspace_block"], "")
        self.assertEqual(agent._frozen_workspace_snapshot, ("OLD_PIN", "OLD_WORKSPACE_CANARY"))
        never.assert_not_called()

    def test_runtime_metadata_changes_do_not_change_constructed_prompt(self):
        ns, agent, modules, _, _ = self.native()
        with patch.dict("sys.modules", modules):
            before = ns["build_system_prompt_parts"](agent, "EXPLICIT_CALLER_POLICY")
            agent.session_id, agent.model, agent.provider = "OTHER_SESSION", "OTHER_MODEL", "OTHER_PROVIDER"
            agent.home, agent.platform = "/OTHER_PROFILE", "OTHER_PLATFORM"
            after = ns["build_system_prompt_parts"](agent, "EXPLICIT_CALLER_POLICY")
        self.assertEqual(before, after)

    def test_hindsight_header_is_bank_and_budget_independent(self):
        obj = self.plugin()
        before = obj.system_prompt_block()
        obj._bank_id, obj._budget = "OTHER_BANK", "OTHER_BUDGET"
        self.assertEqual(before, obj.system_prompt_block())
        self.assertEqual(before, "# Hindsight Memory\nSTATIC_TOOL_GUIDANCE")

    def test_flag_off_keeps_legacy_header_and_model_identity(self):
        obj = self.plugin()
        ns, agent, _, _, _ = self.native()
        with patch.dict(os.environ, {compat.FLAG: "0"}):
            self.assertIn("Bank: BANK_CANARY, budget: BUDGET_CANARY", obj.system_prompt_block())
            self.assertIn("MODEL_CANARY", "\n".join(ns["_alibaba_identity_part"](agent)))

    def test_hybrid_without_auto_recall_does_not_claim_injected_memories(self):
        obj = self.plugin()
        obj._memory_mode = "hybrid"
        self.assertEqual(obj.system_prompt_block(), "# Hindsight Memory\nSTATIC_TOOL_GUIDANCE")

    def test_actual_tool_handler_passes_current_clock_only_as_api_data(self):
        obj, calls = self.plugin(), []

        async def arecall(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(results=[types.SimpleNamespace(text="EXACT_SOURCE_ANSWER")])

        obj._run_hindsight_operation = lambda fn: asyncio.run(fn(types.SimpleNamespace(arecall=arecall)))
        start = datetime.now(timezone.utc)
        answer = obj._tool_recall({"query": "USER_QUERY_CANARY yesterday"})
        end = datetime.now(timezone.utc)
        self.assertEqual(answer, "1. EXACT_SOURCE_ANSWER")
        self.assertEqual([call["bank_id"] for call in calls], ["BANK_CANARY", "MIRROR_CANARY"])
        self.assertEqual(len({call["query_timestamp"] for call in calls}), 1)
        for call in calls:
            timestamp = datetime.fromisoformat(call["query_timestamp"])
            self.assertLessEqual(start, timestamp)
            self.assertLessEqual(timestamp, end)
            self.assertEqual(call["budget"], "BUDGET_CANARY")
            self.assertEqual(call["max_tokens"], 1500)
            self.assertEqual(call["query"], "USER_QUERY_CANARY yesterday")
            self.assertEqual(call["tags"], ["TAG_CANARY"])
            self.assertNotIn(call["query_timestamp"], obj.system_prompt_block())

    def test_pinned_async_sdk_accepts_query_timestamp(self):
        sdk = compat.REPO / ".jarvis-dev/shared-memory-live.vboQT9/deps314-hermes/hindsight_client/hindsight_client.py"
        if not sdk.exists():
            self.skipTest("Pinned SDK source is not installed")
        method = next(node for node in ast.walk(ast.parse(sdk.read_text()))
                      if isinstance(node, ast.AsyncFunctionDef) and node.name == "arecall")
        self.assertIn("query_timestamp", [arg.arg for arg in method.args.args])
        self.assertTrue(any(isinstance(node, ast.keyword) and node.arg == "query_timestamp"
                            for node in ast.walk(method)))

    def test_only_named_instruction_and_recall_functions_change(self):
        allowed = {
            "hermes": {"_active_profile_line", "_timestamp_line", "_alibaba_identity_part",
                       "_post_workspace_parts", "_bot_mode_parts", "_coding_parts", "_context_files_part",
                       "_memory_parts", "build_system_prompt_parts"},
            "hindsight": {"system_prompt_block", "_recall"},
        }
        for name in compat.TARGETS:
            original = ast.parse(self.originals[name])
            changed = ast.parse(self.changed[name])
            before = {node.name: ast.dump(node) for node in ast.walk(original) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            after = {node.name: ast.dump(node) for node in ast.walk(changed) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            actual = {key for key in before if before[key] != after[key]}
            # Nested _recall_all is byte-identical; user/tool/event handlers stay intact.
            self.assertEqual(actual, allowed[name])
            compile(self.changed[name], f"<{name}-patched>", "exec")

    def fixture(self, root: Path):
        repo = root / "repo"
        for name, (relative, _) in compat.TARGETS.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.originals[name])
        return repo, root / "stage"

    def test_staging_backs_up_both_without_mutating_originals(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, output = self.fixture(Path(tmp))
            manifest = compat.stage(repo, output)
            self.assertEqual(manifest["status"], "staged_not_applied")
            for name, (relative, _) in compat.TARGETS.items():
                self.assertEqual((repo / relative).read_bytes(), self.originals[name])
                self.assertEqual((output / "originals" / f"{name}.py").read_bytes(), self.originals[name])
                self.assertEqual((output / "patched" / f"{name}.py").read_text(), self.changed[name])

    def test_drift_rejected_before_staging_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, output = self.fixture(Path(tmp))
            (repo / compat.TARGETS["hindsight"][0]).write_text("drift")
            with self.assertRaises(ValueError):
                compat.stage(repo, output)
            self.assertFalse(output.exists())

    def test_offline_apply_requires_acknowledgment_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, output = self.fixture(Path(tmp))
            compat.stage(repo, output)
            with self.assertRaises(ValueError):
                compat.apply_offline(repo, output)
            receipt = compat.apply_offline(repo, output, offline=True)
            self.assertEqual(receipt["replaced"], ["hermes", "hindsight"])
            for name, (relative, _) in compat.TARGETS.items():
                self.assertEqual((repo / relative).read_text(), self.changed[name])
            again = compat.apply_offline(repo, output, offline=True)
            self.assertEqual(again["replaced"], [])
            self.assertEqual(again["already_applied"], ["hermes", "hindsight"])

    def test_second_target_drift_blocks_first_target_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, output = self.fixture(Path(tmp))
            compat.stage(repo, output)
            (repo / compat.TARGETS["hindsight"][0]).write_text("drift")
            with self.assertRaises(ValueError):
                compat.apply_offline(repo, output, offline=True)
            self.assertEqual((repo / compat.TARGETS["hermes"][0]).read_bytes(), self.originals["hermes"])


if __name__ == "__main__":
    unittest.main()
