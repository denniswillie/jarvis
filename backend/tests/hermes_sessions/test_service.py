"""Binding/service tests: native execution and tool qualification are separate."""

import asyncio
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.hermes_sessions.bindings import BindingError, SessionBindings
from app.hermes_sessions.client import HermesGatewayError, HermesOutcomeUnknown
from app.hermes_sessions.service import RetainedSessionService


class NativeFixture:
    def __init__(self, home):
        self.profile_home = home
        self.connection_generation = 1
        self.creations, self.inputs, self.calls = {}, {}, []
        self.lose_create_reply = self.lose_submit_reply = False

    async def request(self, method, **params):
        self.calls.append((method, params))
        if method == "session.create":
            key = params["request_id"]
            native = self.creations.setdefault(key, "native-" + key)
            if self.lose_create_reply:
                self.lose_create_reply = False
                raise HermesOutcomeUnknown(method, key)
            return {"session_id": native}
        if method == "prompt.submit":
            key = params["session_id"], params["input_id"]
            previous = self.inputs.get(key)
            if previous and previous[0] != params["text"]:
                raise HermesGatewayError("admission_conflict")
            reply = {"admission_id": "admission-" + params["input_id"], "status": "queued"}
            self.inputs.setdefault(key, (params["text"], reply))
            if self.lose_submit_reply:
                self.lose_submit_reply = False
                raise HermesOutcomeUnknown(method, params["input_id"])
            return reply
        if method in {"prompt.receipt", "prompt.cancel"}:
            matches = [row for (sid, _), row in self.inputs.items() if sid == params["session_id"]
                       and row[1]["admission_id"] == params["admission_id"]]
            if not matches:
                raise HermesGatewayError("not_found")
            return {"admission_id": params["admission_id"], "status": "terminal",
                    "outcome": "cancelled" if method == "prompt.cancel" else "completed"}
        raise AssertionError(method)

    async def snapshot(self, session_id):
        self.calls.append(("session.resume", {"session_id": session_id}))
        if session_id not in self.creations.values():
            raise HermesGatewayError("not_found")
        return {"session_id": session_id}

    async def submit(self, session_id, input_id, text):
        return await self.request("prompt.submit", session_id=session_id, input_id=input_id, text=text)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-retained-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.path = self.home / "bindings.sqlite"
        self.bindings = SessionBindings(self.path, owner_id="fixture-owner", profile_home=self.home)
        self.native = NativeFixture(self.home)
        self.launch = {"source": "cli", "cwd": str(self.home), "toolsets": []}
        self.service = RetainedSessionService(self.native, self.bindings, launch=self.launch)

    async def test_concurrent_master_creation_is_one_native_identity(self):
        other = RetainedSessionService(self.native, SessionBindings(
            self.path, owner_id="fixture-owner", profile_home=self.home), launch=self.launch)
        first, second = await asyncio.gather(self.service.master(), other.master())
        self.assertEqual(first, second)
        self.assertEqual(len(self.native.creations), 1)

    async def test_lost_creation_reply_reuses_reserved_request(self):
        self.native.lose_create_reply = True
        with self.assertRaises(HermesOutcomeUnknown):
            await self.service.master()
        result = await self.service.master()
        self.assertEqual(len(self.native.creations), 1)
        self.assertEqual(result.native_id, next(iter(self.native.creations.values())))

    async def test_restart_keeps_master_children_and_no_transcript_copy(self):
        master = await self.service.master()
        child = await self.service.create_work(master.logical_id, request_id="work-1", label="Research")
        await self.service.send_to_work(master.logical_id, child.logical_id, input_id="first", text="ORIGINAL_SECRET_TEXT")
        restored = RetainedSessionService(self.native, SessionBindings(
            self.path, owner_id="fixture-owner", profile_home=self.home), launch=self.launch)
        self.assertEqual(await restored.master(), master)
        self.assertEqual(await restored.create_work(master.logical_id, request_id="work-1", label="Research"), child)
        self.assertNotIn(b"ORIGINAL_SECRET_TEXT", self.path.read_bytes())
        with closing(sqlite3.connect(self.path)) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {"profile", "conversations"})

    async def test_lost_admission_reply_no_automatic_retry_and_explicit_retry_deduplicates(self):
        master = await self.service.master()
        self.native.lose_submit_reply = True
        with self.assertRaises(HermesOutcomeUnknown):
            await self.service.send(master.logical_id, input_id="source-message-7", text="Original words")
        self.assertEqual(sum(name == "prompt.submit" for name, _ in self.native.calls), 1)
        receipt = await self.service.send(master.logical_id, input_id="source-message-7", text="Original words")
        self.assertEqual(receipt.input_id, "source-message-7")
        self.assertEqual(len(self.native.inputs), 1)
        with self.assertRaises(HermesGatewayError):
            await self.service.send(master.logical_id, input_id="source-message-7", text="Changed words")

    async def test_foreign_owner_home_and_client_rejected(self):
        for owner, home in (("another-owner", self.home), ("fixture-owner", self.home / "other")):
            with self.assertRaises(BindingError):
                SessionBindings(self.path, owner_id=owner, profile_home=home)
        with self.assertRaises(BindingError):
            RetainedSessionService(NativeFixture(self.home / "other"), self.bindings, launch=self.launch)

    async def test_child_cannot_create_siblings_or_control_master(self):
        master = await self.service.master()
        child = await self.service.create_work(master.logical_id, request_id="one", label="Work")
        with self.assertRaises(BindingError):
            await self.service.create_work(child.logical_id, request_id="nested", label="Nested")
        with self.assertRaises(BindingError):
            await self.service.send_to_work(child.logical_id, master.logical_id, input_id="bad", text="Bad")
        with self.assertRaises(BindingError):
            await self.service.send("unknown-session", input_id="bad", text="Bad")

    async def test_exact_child_cancel_retains_all_bindings(self):
        master = await self.service.master()
        a = await self.service.create_work(master.logical_id, request_id="a", label="A")
        b = await self.service.create_work(master.logical_id, request_id="b", label="B")
        first = await self.service.send_to_work(master.logical_id, a.logical_id, input_id="one", text="A")
        await self.service.send_to_work(master.logical_id, b.logical_id, input_id="one", text="B")
        with self.assertRaises(HermesGatewayError):
            await self.service.stop_work(master.logical_id, b.logical_id, first.admission_id)
        result = await self.service.stop_work(master.logical_id, a.logical_id, first.admission_id)
        self.assertEqual(result["outcome"], "cancelled")
        self.assertEqual(len(self.bindings.children(master.logical_id)), 2)
        self.assertEqual(self.bindings.get(a.logical_id), a)

    async def test_missing_native_conversation_is_not_recreated(self):
        master = await self.service.master()
        self.native.creations.clear()
        self.native.connection_generation += 1
        with self.assertRaises(HermesGatewayError):
            await self.service.master()
        self.assertEqual(self.native.creations, {})
        self.assertEqual(self.bindings.get(master.logical_id), master)

    async def test_create_conflicts_do_not_silently_reconfigure(self):
        master = await self.service.master()
        child = await self.service.create_work(master.logical_id, request_id="one", label="A")
        creation_count = sum(method == "session.create" for method, _ in self.native.calls)
        with self.assertRaises(BindingError):
            await self.service.create_work(master.logical_id, request_id="one", label="B")
        updated = RetainedSessionService(self.native, self.bindings, launch={**self.launch, "model": "other"})
        reused = await updated.master()
        self.assertEqual(reused.logical_id, master.logical_id)
        self.assertEqual(reused.native_id, master.native_id)
        self.assertEqual(reused.launch_json, master.launch_json)
        self.assertEqual(json.loads(reused.launch_json), self.launch)
        self.assertEqual(self.bindings.get(master.logical_id), master)
        with self.assertRaises(BindingError):
            await updated.create_work(master.logical_id, request_id="one", label="A")
        self.assertEqual(self.bindings.get(child.logical_id), child)
        self.assertEqual(sum(method == "session.create" for method, _ in self.native.calls), creation_count)

    async def test_followups_do_not_reload_history_every_time(self):
        master = await self.service.master()
        for i in range(3):
            await self.service.send(master.logical_id, input_id=str(i), text="Original " + str(i))
        self.assertFalse(any(method == "session.resume" for method, _ in self.native.calls))
        self.native.connection_generation += 1
        await self.service.send(master.logical_id, input_id="four", text="After reconnect")
        await self.service.send(master.logical_id, input_id="five", text="Still attached")
        self.assertEqual(sum(method == "session.resume" for method, _ in self.native.calls), 1)

    async def test_binding_replacement_and_unbounded_inputs_refused(self):
        master = await self.service.master()
        with self.assertRaises(BindingError):
            self.bindings.bind(master.logical_id, "replacement")
        with self.assertRaises(BindingError):
            await self.service.send(master.logical_id, input_id="huge", text="x" * (128 * 1024 + 1))
        with self.assertRaises(BindingError):
            self.bindings.children(master.logical_id, limit=1000)


if __name__ == "__main__":
    unittest.main()
