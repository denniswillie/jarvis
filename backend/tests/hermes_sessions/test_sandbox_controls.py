"""Offline control RPC and native admission composition; no cloud/model/service launch."""
import asyncio
from contextvars import ContextVar
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from app.hermes_sessions.bindings import SessionBindings, identity
from app.hermes_sessions.domain_plugin import NativeInputContext
from app.hermes_sessions.domain_scope import DomainInputScope
from app.hermes_sessions.domain_scopes import DomainScopeAssociations
from app.hermes_sessions.domain_service import ScopedRetainedSessionService
from app.hermes_sessions.domain_transport import DomainToolServer
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandbox_controls import NativeMasterNoticeDelivery, SandboxControls, register_sandbox_controls
from app.hermes_sessions.sandbox_keepalive import KeepaliveJournal, SandboxKeepalive
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.hermes_sessions.service import RetainedSessionService
from app.hermes_sessions.terminal_transport import invoke_sandbox_control
from app.sandbox.models import SandboxError, SandboxPolicy
from test_sandbox_keepalive import ClockProvider
from test_service import NativeFixture


class SandboxControlsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-sandbox-controls-")
        self.home = Path(self.temp.name).resolve()
        self.bindings = SessionBindings(self.home / "sessions.sqlite", owner_id="owner", profile_home=self.home)
        self.native = NativeFixture(self.home)
        self.launch = {"source": "cli", "cwd": str(self.home), "toolsets": []}
        self.sessions = RetainedSessionService(self.native, self.bindings, launch=self.launch)
        # The socket exists before the native master; no fake domain adapter.
        self.server = await DomainToolServer(self.home, None).start()
        self.preinstall = await asyncio.to_thread(invoke_sandbox_control, self.home,
            context=NativeInputContext("not-created", "input", "admission", "call", "turn"),
            arguments={"action": "status"})
        self.assertEqual(self.native.creations, {})
        self.master = await self.sessions.master()
        self.child = await self.sessions.create_work(self.master.logical_id, request_id="a", label="Child")
        self.sibling = await self.sessions.create_work(self.master.logical_id, request_id="b", label="Sibling")
        self.store = SessionSandboxBindings(self.home / "resources.sqlite", owner_id="owner", profile_id="profile")
        self.now = 1_000_000.0
        self.provider = ClockProvider(lambda: self.now)
        self.sandboxes = RetainedSandboxService(self.store, self.provider, policy=SandboxPolicy())
        self.journal = KeepaliveJournal(self.home / "keepalive.sqlite", bindings=self.store)
        self.delivery = NativeMasterNoticeDelivery(sessions=self.sessions, master_session_id=self.master.logical_id)
        self.keepalive = SandboxKeepalive(master_session_id=self.master.logical_id, sandboxes=self.sandboxes,
            journal=self.journal, deliver=self.delivery.deliver, clock=lambda: self.now)
        self.controls = SandboxControls(sessions=self.sessions, sandboxes=self.sandboxes, keepalive=self.keepalive)
        self.server.install_sandbox_controls(self.controls)
        self.context = NativeInputContext(self.master.native_id, "input", "admission", "call", "TURN-not-input")

    async def asyncTearDown(self):
        await self.server.close()
        self.journal.close()
        self.store.close()
        self.temp.cleanup()

    async def call(self, action, *, actor=None, target=None, **extra):
        args = {"action": action, **extra}
        if target is not None:
            args["target_session_id"] = target
        context = replace(self.context, native_session_id=(actor or self.master).native_id)
        return await asyncio.to_thread(invoke_sandbox_control, self.home, context=context, arguments=args)

    def notice(self, **changes):
        return dict(event_id="sandbox-maintenance-" + "a" * 32, target=self.child.logical_id,
            master=self.master.logical_id, sandbox_id="retained-1", started_at=self.now,
            kind="sandbox_maintenance", outcome="micro_pause_resumed", observed_state="running") | changes

    async def test_staged_install_no_domain_repository_and_default_disabled(self):
        self.assertEqual(self.preinstall["error"]["code"], "sandbox_controls_disabled")
        self.assertEqual(self.provider.creates, 0)
        self.server.install_sandbox_controls(self.controls)
        with self.assertRaisesRegex(ValueError, "another profile"):
            self.server.install_sandbox_controls(SimpleNamespace(profile_home=self.home / "other"))

    async def test_status_resume_and_keepalive_do_not_create_resources(self):
        for action in ("status", "pause", "resume", "keepalive_enable", "keepalive_disable"):
            result = await self.call(action, target=self.child.logical_id)
            self.assertEqual(result["error"]["code"], "session_sandbox_not_allocated")
        self.assertEqual((self.provider.creates, self.provider.resumes, self.provider.pauses), (0, 0, 0))

    async def test_child_can_status_pause_resume_own_existing_resource(self):
        first = await self.sandboxes.ensure(self.child.logical_id)
        for action, state in (("status", "running"), ("pause", "paused"), ("resume", "running")):
            result = await self.call(action, actor=self.child)
            self.assertEqual(result["sandbox"]["state"], state)
            self.assertEqual(result["sandbox"]["sandbox_id"], first.sandbox_id)
        self.assertEqual(self.provider.creates, 1)

    async def test_child_cannot_control_other_session_or_any_keepalive(self):
        await self.sandboxes.ensure(self.child.logical_id)
        await self.sandboxes.ensure(self.sibling.logical_id)
        for action in ("status", "pause", "resume"):
            result = await self.call(action, actor=self.child, target=self.sibling.logical_id)
            self.assertEqual(result["error"]["code"], "sandbox_control_own_session_required")
        for action in ("keepalive_enable", "keepalive_disable"):
            self.assertEqual((await self.call(action, actor=self.child))["error"]["code"], "sandbox_keepalive_master_required")
        self.assertEqual((self.provider.pauses, self.provider.resumes), (0, 0))

    async def test_master_can_control_child_and_select_keepalive(self):
        await self.sandboxes.ensure(self.child.logical_id)
        self.assertEqual((await self.call("pause", target=self.child.logical_id))["sandbox"]["state"], "paused")
        self.assertEqual((await self.call("resume", target=self.child.logical_id))["sandbox"]["state"], "running")
        self.assertTrue((await self.call("keepalive_enable", target=self.child.logical_id))["keepalive"]["enabled"])
        self.assertFalse((await self.call("keepalive_disable", target=self.child.logical_id))["keepalive"]["enabled"])
        self.assertEqual((self.provider.creates, self.provider.pauses, self.provider.resumes), (1, 1, 1))

    async def test_foreign_selector_model_actor_and_extra_fields_fail_closed(self):
        await self.sandboxes.ensure(self.child.logical_id)
        for args in ({"actor_session_id": self.master.logical_id}, {"native_session_id": self.master.native_id},
                     {"input_id": "invented"}, {"sandbox_id": "retained-1"}):
            self.assertEqual((await self.call("pause", actor=self.child, **args))["status"], "failed")
        self.assertEqual((await self.call("resume", target="other-profile-session"))["status"], "failed")
        self.assertEqual(self.provider.resumes, 0)

    async def test_missing_provider_resource_is_not_recreated(self):
        info = await self.sandboxes.ensure(self.child.logical_id)
        del self.provider.items[info.sandbox_id]
        result = await self.call("resume", target=self.child.logical_id)
        self.assertEqual(result["error"]["code"], "sandbox_not_found")
        self.assertEqual(self.provider.creates, 1)

    async def test_current_native_caller_compaction_is_resolved(self):
        await self.sandboxes.ensure(self.child.logical_id)
        snapshot = self.native.snapshot
        async def compacted(native_id):
            return await snapshot(self.child.native_id if native_id == "child-tip" else native_id)
        self.native.snapshot = compacted
        result = await self.call("status", actor=SimpleNamespace(native_id="child-tip"))
        self.assertEqual(result["target_session_id"], self.child.logical_id)

    async def test_profile_owner_composition_is_checked(self):
        wrong = SimpleNamespace(bindings=SimpleNamespace(owner_id="foreign"))
        with self.assertRaisesRegex(ValueError, "owners differ"):
            SandboxControls(sessions=wrong, sandboxes=self.sandboxes, keepalive=self.keepalive)
        with self.assertRaisesRegex(ValueError, "same sandbox"):
            SandboxControls(sessions=self.sessions, sandboxes=self.sandboxes,
                            keepalive=SimpleNamespace(sandboxes=object()))

    async def test_real_keepalive_success_does_not_admit_master_input(self):
        await self.sandboxes.ensure(self.child.logical_id)
        await self.call("keepalive_enable", target=self.child.logical_id)
        self.now += 270
        await self.keepalive.tick()
        self.assertEqual(len(self.native.inputs), 0)
        self.assertEqual(self.journal.pending_notices(), [])
        self.assertEqual((self.provider.pauses, self.provider.resumes), (1, 1))

    async def test_lost_native_ack_retries_same_id_and_text_without_new_cycle(self):
        await self.sandboxes.ensure(self.child.logical_id)
        await self.call("keepalive_enable", target=self.child.logical_id)
        self.native.lose_submit_reply = True
        self.provider.resume_error = RuntimeError("fixture resume failed")
        self.now += 270
        await self.keepalive.tick()
        event = self.journal.pending_notices()[0]
        await self.keepalive.tick()
        submits = [params for method, params in self.native.calls if method == "prompt.submit"]
        self.assertEqual(len(submits), 2)
        self.assertEqual(submits[0], submits[1])
        self.assertEqual(submits[0]["input_id"], identity(self.master.logical_id, "input", event["event_id"]))
        self.assertEqual(len(self.native.inputs), 1)
        self.assertEqual((self.provider.pauses, self.provider.resumes), (1, 2))
        self.assertEqual(self.journal.pending_notices(), [])

    async def test_notice_rejects_foreign_master_unknown_target_and_oversized_data(self):
        for notice in (self.notice(master=self.child.logical_id), self.notice(target="foreign"),
                       self.notice(error="x" * 201), self.notice(outcome="invented")):
            self.assertFalse(await self.delivery.deliver(notice))
        self.assertEqual(self.native.inputs, {})

    async def test_delivery_does_not_create_missing_master(self):
        delivery = NativeMasterNoticeDelivery(sessions=self.sessions, master_session_id="absent")
        before = len(self.native.creations)
        self.assertFalse(await delivery.deliver(self.notice(master="absent")))
        self.assertEqual(len(self.native.creations), before)

    def scoped(self):
        scopes = DomainScopeAssociations(self.home / "scopes.sqlite", self.bindings)
        return ScopedRetainedSessionService(self.native, self.bindings, launch=self.launch, scopes=scopes)

    async def test_domain_notice_requires_explicit_scope_callback(self):
        with self.assertRaisesRegex(ValueError, "authority-stripped"):
            NativeMasterNoticeDelivery(sessions=self.scoped(), master_session_id=self.master.logical_id)
        self.assertEqual(self.native.inputs, {})

    async def test_domain_read_scope_is_bound_before_native_admission(self):
        service = self.scoped()
        jarvis_id = uuid4()
        async def prepare_scope(*, master_session_id, native_input_id, notice):
            return DomainInputScope("owner", master_session_id, native_input_id, jarvis_id,
                                    frozenset({"read_fixture"}))
        delivery = NativeMasterNoticeDelivery(sessions=service, master_session_id=self.master.logical_id,
                                              prepare_scope=prepare_scope)
        submit = self.native.submit
        async def guarded(native_id, input_id, text):
            scope = service.scopes.get(self.master.logical_id, input_id)
            self.assertEqual(scope.jarvis_session_id, jarvis_id)
            self.assertIsNone(scope.request_event_id)
            self.assertIsNone(scope.request_revision)
            return await submit(native_id, input_id, text)
        self.native.submit = guarded
        self.assertTrue(await delivery.deliver(self.notice()))
        self.assertTrue(await delivery.deliver(self.notice()))
        self.assertEqual(len(self.native.inputs), 1)

    async def test_domain_callback_cannot_reintroduce_request_or_foreign_scope(self):
        service = self.scoped()
        good = DomainInputScope("owner", self.master.logical_id,
            identity(self.master.logical_id, "input", self.notice()["event_id"]), uuid4(), frozenset({"read_fixture"}))
        for bad in (replace(good, request_event_id=uuid4(), request_revision=1),
                    replace(good, owner_id="foreign"), replace(good, logical_session_id=self.child.logical_id),
                    replace(good, input_id="wrong")):
            async def prepare_scope(**kwargs):
                return bad
            delivery = NativeMasterNoticeDelivery(sessions=service, master_session_id=self.master.logical_id,
                                                  prepare_scope=prepare_scope)
            self.assertFalse(await delivery.deliver(self.notice()))
        self.assertEqual(self.native.inputs, {})


class SandboxControlPluginTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.submission, self.call, self.turn = [ContextVar(name, default=None) for name in ("submission", "call", "turn")]
        self.worker = Mock(return_value=False)
        self.rpc = Mock(return_value={"status": "success"})
        modules = {}
        for name, fields in {
            "hermes_constants": dict(get_hermes_home=lambda: self.home),
            "tools.approval_context": dict(_approval_tool_call_id=self.call, _approval_turn_id=self.turn),
            "gateway.session_display": dict(_submission_turn=self.submission),
            "agent.runtime_session_store": dict(is_worker_process=self.worker),
        }.items():
            module = ModuleType(name)
            module.__dict__.update(fields)
            modules[name] = module
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("app.hermes_sessions.sandbox_controls.invoke_sandbox_control", self.rpc)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ctx = Mock()

    def handler(self):
        (self.home / "jarvis-adapter.json").write_text('{"sandbox_controls": true}')
        register_sandbox_controls(self.ctx)
        entry = self.ctx.register_tool.call_args.kwargs
        self.assertEqual((entry["name"], entry["toolset"], entry["is_async"]), ("session_sandbox", "jarvis_sandbox", False))
        self.assertFalse(entry["schema"]["parameters"]["additionalProperties"])
        return entry["handler"]

    def test_absent_and_false_opt_in_register_nothing(self):
        register_sandbox_controls(self.ctx)
        (self.home / "jarvis-adapter.json").write_text('{"sandbox_controls": false}')
        register_sandbox_controls(self.ctx)
        self.ctx.register_tool.assert_not_called()

    def test_private_context_is_fresh_and_not_model_identity(self):
        handler = self.handler()
        self.call.set("call")
        self.turn.set("not-input")
        for admission, input_id in (("a", "i-a"), ("b", "i-b")):
            self.submission.set((admission, input_id))
            args = {"action": "status", "target_session_id": "selector"}
            self.assertEqual(json.loads(handler(args, session_id="trusted-native"))["status"], "success")
            context = self.rpc.call_args.kwargs["context"]
            self.assertEqual((context.native_session_id, context.admission_id, context.input_id),
                             ("trusted-native", admission, input_id))

    def test_missing_and_managed_context_never_reach_rpc(self):
        handler = self.handler()
        self.assertEqual(json.loads(handler({"action": "status"}, session_id="native"))["status"], "failed")
        self.submission.set(("admission", "input"))
        self.call.set("call")
        self.turn.set("turn")
        self.worker.return_value = True
        self.assertEqual(json.loads(handler({"action": "status"}, session_id="native"))["status"], "failed")
        self.rpc.assert_not_called()

    def test_native_import_never_loads_resource_owner_or_backend_packages(self):
        script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'asyncpg', 'fastapi', 'supabase', 'e2b', 'pydantic'} or fullname in {
            'app.hermes_sessions.sandbox_keepalive', 'app.hermes_sessions.sandbox_bindings',
            'app.hermes_sessions.sandboxes', 'app.hermes_sessions.domain'}:
            raise AssertionError('Forbidden native import: ' + fullname)
sys.meta_path.insert(0, Block())
import app.hermes_sessions.sandbox_controls
import app.hermes_sessions.plugin
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
