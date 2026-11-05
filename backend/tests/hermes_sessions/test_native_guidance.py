"""The pinned native source-label workaround does not widen other sessions."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from app.hermes_sessions.bindings import SessionBindings
from app.hermes_sessions.native_guidance import enable_live_guidance


class NativeGuidanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        (self.home / "jarvis-adapter.json").write_text(json.dumps({
            "owner_id": "owner", "native_live_guidance": True}))
        bindings = SessionBindings(self.home / "jarvis" / "bindings.sqlite",
            owner_id="owner", profile_home=self.home)
        master = bindings.get_or_reserve_master(launch={"source": "cli"})
        bindings.bind(master.logical_id, "native-master")
        child = bindings.reserve(request_id="child", parent_id=master.logical_id,
                                 label="Child", launch={"source": "cli"})
        bindings.bind(child.logical_id, "native-child")
        self.original = lambda authority, row: row.get("native_controls", [])
        self.native = ModuleType("gateway.session_queued_controls")
        self.native.queued_controls = self.original
        self.native._TEXT_ONLY = frozenset({"text", "surface_v1", "display_v1", "local_operator_v1"})
        policy = ModuleType("gateway.session_policy")
        policy.policy_for_source = lambda runner, source: NS(source=source)
        gateway = ModuleType("gateway")
        gateway.session_queued_controls = self.native
        modules = patch.dict(sys.modules, {"gateway": gateway,
            "gateway.session_queued_controls": self.native, "gateway.session_policy": policy})
        modules.start()
        self.addCleanup(modules.stop)
        self.authority = NS(runner=NS(), sessions={
            "native-master": NS(source="cli"), "native-child": NS(source="cli"),
            "unbound": NS(source="cli")})
        self.row = {"status": "queued", "target_session_id": "native-master",
                    "payload": {"text": "Change the edit"}}

    def test_only_owned_cli_master_gets_steer_not_replace(self):
        enable_live_guidance(self.home)
        self.assertEqual(self.native.queued_controls(self.authority, self.row), ["steer"])
        for sid in ("native-child", "unbound"):
            with self.subTest(sid=sid):
                self.assertEqual(self.native.queued_controls(self.authority,
                    {**self.row, "target_session_id": sid}), [])
        self.authority.sessions["native-master"].source = "cron"
        self.assertEqual(self.native.queued_controls(self.authority, self.row), [])

    def test_native_payload_and_status_exclusions_preserved(self):
        enable_live_guidance(self.home)
        for payload in ({"text": ""}, {"text": " ", "surface_v1": {}},
                        {"text": "guide", "attachments": []},
                        {"text": "guide", "display_v1": {"kind": "hidden"}}):
            with self.subTest(payload=payload):
                self.assertEqual(self.native.queued_controls(self.authority,
                    {**self.row, "payload": payload}), [])
        self.assertEqual(self.native.queued_controls(self.authority,
            {**self.row, "status": "started"}), [])

    def test_native_result_and_repeated_registration_unchanged(self):
        enable_live_guidance(self.home)
        installed = self.native.queued_controls
        enable_live_guidance(self.home)
        self.assertIs(installed, self.native.queued_controls)
        self.assertEqual(installed(self.authority,
            {**self.row, "native_controls": ["steer", "replace"]}), ["steer", "replace"])

    def test_disabled_profile_does_not_patch_native(self):
        (self.home / "jarvis-adapter.json").write_text(json.dumps({"owner_id": "owner"}))
        enable_live_guidance(self.home)
        self.assertIs(self.native.queued_controls, self.original)
