"""Offline qualification of reference ownership and scope-before-admission ordering."""

from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from uuid import uuid4

from app.hermes_sessions.bindings import BindingError, SessionBindings, identity
from app.hermes_sessions.client import HermesOutcomeUnknown
from app.hermes_sessions.domain_scopes import DomainScopeAssociations
from app.hermes_sessions.domain_service import compose_domain_work
from app.hermes_sessions.plugin import compose
from test_work import ResultFixture


class DomainScopesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-domain-scopes-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.config = {"owner_id": "owner", "domain_tools": True,
                       "launch": {"source": "cli", "cwd": str(self.home), "toolsets": []}}
        (self.home / "jarvis-adapter.json").write_text(json.dumps(self.config))
        self.client = ResultFixture(self.home)
        self.bridge = compose_domain_work(self.home, self.client)
        self.master = await self.bridge.service.master()
        self.scope = await self.bind("first")
        self.receipt = await self.bridge.service.send(self.master.logical_id, input_id="first", text="FIRST")

    async def bind(self, input_id):
        return await self.bridge.service.bind_input_scope(
            self.master.logical_id, input_id=input_id, jarvis_session_id=uuid4(),
            allowed_tools=frozenset({"read_fixture"}), request_event_id=uuid4(), request_revision=1,
        )

    async def tool(self, args, *, call="call-1", scope=None, admission=None):
        scope = scope or self.scope
        return await self.bridge.tool(
            self.master.native_id, call, args, input_id=scope.input_id,
            admission_id=admission or self.receipt.admission_id,
        )

    async def create(self):
        return await self.tool({"action": "create", "label": "Child", "message": "CHILD_SECRET"})

    async def test_initial_binding_is_explicit_immutable_and_not_latest(self):
        second = await self.bind("second")
        restored = compose(self.home, self.client)
        self.assertTrue(restored.domain_scoped)
        self.assertEqual(await restored.scopes.scope_for_input(self.master, self.scope.input_id), self.scope)
        self.assertNotEqual(second.jarvis_session_id, self.scope.jarvis_session_id)
        with self.assertRaises(BindingError):
            await restored.service.send(self.master.logical_id, input_id="unbound", text="NO")
        with self.assertRaises(BindingError):
            restored.scopes.bind_initial(replace(self.scope, jarvis_session_id=uuid4()))
        with self.assertRaises(BindingError):
            restored.scopes.get(self.master.logical_id, "missing")
        self.assertEqual(restored.scopes.bind_initial(self.scope), self.scope)
        self.assertEqual(len(self.client.inputs), 1)

    async def test_child_scope_exists_before_native_create_and_submit(self):
        original = self.client.request
        observed = []
        async def guarded(method, **params):
            if method == "session.create":
                child = self.bridge.service.bindings.get(params["request_id"])
                service_input = identity(self.master.logical_id,
                                         identity(self.receipt.admission_id, "call-1"), "work")
                scope = self.bridge.scopes.get(child.logical_id, identity(child.logical_id, "input", service_input))
                observed.append(("create", scope.jarvis_session_id))
            if method == "prompt.submit":
                child = self.bridge.service.bindings.by_native(params["session_id"])
                scope = self.bridge.scopes.get(child.logical_id, params["input_id"])
                observed.append(("submit", scope.jarvis_session_id))
            return await original(method, **params)
        self.client.request = guarded
        await self.create()
        self.assertEqual(observed, [("create", self.scope.jarvis_session_id),
                                    ("submit", self.scope.jarvis_session_id)])

    async def test_two_original_sessions_propagate_by_exact_input_and_scoped_steer_is_fifo(self):
        child = await self.create()
        second = await self.bind("second")
        receipt = await self.bridge.service.send(self.master.logical_id, input_id="second", text="SECOND")
        steered = await self.tool({"action": "steer", "session_id": child["session_id"], "message": "NEXT"},
                                  call="call-2", scope=second, admission=receipt.admission_id)
        self.assertEqual(steered["delivery"], "queued")
        self.assertFalse(steered["immediate_steering"])
        self.assertEqual(self.client.steers, [])
        rows = self.bridge.deliveries.pending()
        self.assertEqual(len(rows), 2)
        origins = [self.bridge.scopes.get(row["child_id"], identity(row["child_id"], "input", row["input_id"]))
                   for row in rows]
        self.assertEqual([scope.jarvis_session_id for scope in origins],
                         [self.scope.jarvis_session_id, second.jarvis_session_id])
        self.assertTrue(all(not row["steer"] for row in rows))
        await self.bridge.service.steer_work(self.master.logical_id, child["session_id"], steered["admission_id"])
        self.assertFalse(any(method == "prompt.steer_queued" for method, _ in self.client.calls))

    async def test_completion_restart_and_parent_message_drop_request_authority(self):
        child = await self.create()
        row = self.bridge.deliveries.pending()[0]
        child_scope = self.bridge.scopes.get(row["child_id"], identity(row["child_id"], "input", row["input_id"]))
        self.assertEqual(child_scope.request_event_id, self.scope.request_event_id)
        restored = compose_domain_work(self.home, self.client)
        wakes = await restored.pump()
        self.assertEqual(len(wakes), 1)
        wake_scope = restored.scopes.get(self.master.logical_id, wakes[0]["native_input_id"])
        self.assertEqual(wake_scope.jarvis_session_id, self.scope.jarvis_session_id)
        self.assertEqual(wake_scope.allowed_tools, self.scope.allowed_tools)
        self.assertIsNone(wake_scope.request_event_id)
        self.assertIsNone(wake_scope.request_revision)
        self.assertEqual(await restored.pump(), [])
        with self.assertRaises(BindingError):
            restored.scopes.bind_initial(replace(wake_scope, request_event_id=uuid4(), request_revision=1))
        binding = restored.service.bindings.get(child["session_id"])
        message = await restored.tool(binding.native_id, "parent-call",
                                      {"action": "message_parent", "message": "Information"},
                                      input_id=child_scope.input_id, admission_id=child["admission_id"])
        parent_scope = restored.scopes.get(self.master.logical_id, message["native_input_id"])
        self.assertIsNone(parent_scope.request_event_id)
        self.assertEqual(parent_scope.jarvis_session_id, self.scope.jarvis_session_id)

    async def live_child(self):
        self.scope = await self.bridge.service.bind_input_scope(
            self.master.logical_id, input_id="live-parent", jarvis_session_id=uuid4(),
            allowed_tools=frozenset({"read_fixture"}))
        self.receipt = await self.bridge.service.send(
            self.master.logical_id, input_id="live-parent", text="Live request")
        child = await self.create()
        binding = self.bridge.service.bindings.get(child["session_id"])
        original = self.client.snapshot

        async def running(session_id):
            state = await original(session_id)
            if session_id == binding.native_id:
                inputs = [(input_id, value[1]) for (sid, input_id), value in self.client.inputs.items()
                          if sid == session_id]
                state.update(running=True, execution_generation=1, pending=[
                    {"input_id": input_id, "admission_id": receipt["admission_id"],
                     "status": "started" if index == 0 else "queued",
                     "execution_generation": 1 if index == 0 else None}
                    for index, (input_id, receipt) in enumerate(inputs)])
            return state

        self.client.snapshot = running
        return child

    async def test_unversioned_same_scope_child_guidance_uses_native_steer(self):
        child = await self.live_child()
        result = await self.tool({"action": "steer", "session_id": child["session_id"],
                                  "message": "Open the visible browser"}, call="live-guidance")
        self.assertEqual(len(self.client.steers), 1)
        self.assertEqual(self.client.steers[0]["admission_id"], result["admission_id"])
        self.assertTrue(self.bridge.deliveries.pending()[-1]["steer"])
        self.assertEqual(len(self.client.inputs), 4)  # Two parent and two child inputs.

    async def test_unversioned_guidance_from_different_phone_session_stays_queued(self):
        child = await self.live_child()
        other = await self.bridge.service.bind_input_scope(
            self.master.logical_id, input_id="other-phone", jarvis_session_id=uuid4(),
            allowed_tools=self.scope.allowed_tools)
        receipt = await self.bridge.service.send(self.master.logical_id, input_id="other-phone", text="Other")
        await self.tool({"action": "steer", "session_id": child["session_id"], "message": "Different scope"},
                        call="other-guidance", scope=other, admission=receipt.admission_id)
        self.assertEqual(self.client.steers, [])

    async def test_unversioned_guidance_cannot_change_active_child_tool_scope(self):
        child = await self.live_child()
        other = await self.bridge.service.bind_input_scope(
            self.master.logical_id, input_id="other-tools", jarvis_session_id=self.scope.jarvis_session_id,
            allowed_tools=frozenset({"write_fixture"}))
        receipt = await self.bridge.service.send(self.master.logical_id, input_id="other-tools", text="Other")
        await self.tool({"action": "steer", "session_id": child["session_id"], "message": "Different tools"},
                        call="other-guidance", scope=other, admission=receipt.admission_id)
        self.assertEqual(self.client.steers, [])

    async def test_lost_submit_reply_recovers_from_scope_refs_without_transcript(self):
        self.client.lose_submit_reply = True
        with self.assertRaises(HermesOutcomeUnknown):
            await self.create()
        restored = compose_domain_work(self.home, self.client)
        self.assertEqual(len(await restored.pump()), 1)
        self.assertEqual(len(self.client.inputs), 3)
        self.assertNotIn(b"CHILD_SECRET", restored.scopes.path.read_bytes())
        with closing(sqlite3.connect(restored.scopes.path)) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(scopes)")}
        self.assertEqual(columns, {"logical_id", "input_id", "scope_json", "kind",
                                   "source_logical_id", "source_input_id"})

    async def test_scope_owner_and_parent_relationship_fail_closed(self):
        with self.assertRaises(BindingError):
            self.bridge.scopes.bind_initial(replace(self.scope, owner_id="foreign"))
        other = SessionBindings(self.home / "foreign.sqlite", owner_id="foreign", profile_home=self.home)
        with self.assertRaises(BindingError):
            DomainScopeAssociations(self.bridge.scopes.path, other)
        with self.assertRaises(BindingError):
            self.bridge.scopes.derive(self.scope, self.master.logical_id, "new", kind="work")
        with self.assertRaises(BindingError):
            await self.bridge.service.create_work(self.master.logical_id, request_id="unscoped", label="No")

    async def test_canonical_binding_resolves_compacted_native_tip(self):
        original = self.client.snapshot
        async def compacted(session_id):
            return await original(self.master.native_id if session_id == "compacted-tip" else session_id)
        self.client.snapshot = compacted
        result = await self.bridge.tool("compacted-tip", "compacted-call", {"action": "list"},
                                        input_id=self.scope.input_id, admission_id=self.receipt.admission_id)
        self.assertIsInstance(result, dict)

    async def test_unscoped_composition_remains_opt_out(self):
        config = {**self.config, "domain_tools": False}
        (self.home / "jarvis-adapter.json").write_text(json.dumps(config))
        legacy = compose(self.home, self.client)
        self.assertFalse(getattr(legacy, "domain_scoped", False))
        await legacy.service.send(self.master.logical_id, input_id="legacy", text="Unscoped fixture")
        with self.assertRaises(BindingError):
            compose_domain_work(self.home, self.client)
