"""Work tools and delivery recovery, with a non-executing native peer fixture."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from test_service import NativeFixture
from app.hermes_sessions.bindings import BindingError, SessionBindings
from app.hermes_sessions.client import HermesOutcomeUnknown
from app.hermes_sessions.service import RetainedSessionService
from app.hermes_sessions.work import WorkDeliveries, WorkSessionBridge


class ResultFixture(NativeFixture):
    def __init__(self, home):
        super().__init__(home)
        self.unfinished = set()
        self.steers = []

    async def request(self, method, **params):
        if method == "prompt.steer_queued":
            self.steers.append(params)
            return {"admission_id": params["admission_id"], "status": "started"}
        result = await super().request(method, **params)
        if method == "prompt.cancel":
            result["ref"] = {"profile_id": "/private/native/profile"}
            result["authority_epoch"] = 123
        if method == "prompt.receipt":
            if params["admission_id"] in self.unfinished:
                result["status"] = "started"
            else:
                result["result"] = {"final_response": "Verified child result", "messages": [{"reasoning": "PRIVATE"}]}
        return result


class WorkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-work-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.bindings = SessionBindings(self.home / "bindings.sqlite", owner_id="owner", profile_home=self.home)
        self.client = ResultFixture(self.home)
        self.service = RetainedSessionService(self.client, self.bindings, launch={"source": "cli", "toolsets": []})
        self.deliveries = WorkDeliveries(self.home / "deliveries.sqlite", self.bindings)
        self.bridge = WorkSessionBridge(self.service, self.deliveries)
        self.master = await self.service.master()

    async def create(self, call_id="create-a", label="A"):
        return await self.bridge.tool(self.master.native_id, call_id,
                                      {"action": "create", "label": label, "message": "Original work instruction"})

    async def test_tool_creation_and_retry_use_one_work_and_one_input(self):
        first = await self.create()
        second = await self.create()
        self.assertEqual(first, second)
        self.assertEqual(len(self.bindings.children(self.master.logical_id)), 1)
        self.assertEqual(len(self.client.inputs), 1)
        self.assertEqual(len(self.deliveries.pending()), 1)

    async def test_completion_wakes_master_once_with_native_result_only(self):
        await self.create()
        wakes = await self.bridge.pump()
        self.assertEqual(len(wakes), 1)
        self.assertEqual(wakes[0]["session_id"], self.master.logical_id)
        parent_inputs = [row[0] for (sid, _), row in self.client.inputs.items() if sid == self.master.native_id]
        self.assertEqual(len(parent_inputs), 1)
        self.assertIn("Verified child result", parent_inputs[0])
        self.assertNotIn("PRIVATE", parent_inputs[0])
        self.assertEqual(await self.bridge.pump(), [])

    async def test_running_child_does_not_wake_master_or_block_pump(self):
        created = await self.create()
        self.client.unfinished.add(created["admission_id"])
        self.assertEqual(await self.bridge.pump(), [])
        self.assertEqual(len(self.client.inputs), 1)
        self.client.unfinished.clear()
        self.assertEqual(len(await self.bridge.pump()), 1)

    async def test_lost_child_admission_reply_recovers_same_input(self):
        self.client.lose_submit_reply = True
        with self.assertRaises(HermesOutcomeUnknown):
            await self.create()
        self.assertEqual(len(self.client.inputs), 1)
        self.assertIsNone(self.deliveries.pending()[0]["admission_id"])
        await self.bridge.pump()
        self.assertEqual(len(self.client.inputs), 2)  # one child input and one master wake
        self.assertEqual(self.deliveries.pending(), [])

    async def test_lost_master_wake_reply_does_not_wake_twice(self):
        await self.create()
        self.client.lose_submit_reply = True
        with self.assertRaises(HermesOutcomeUnknown):
            await self.bridge.pump()
        self.assertEqual(len(self.client.inputs), 2)
        await self.bridge.pump()
        self.assertEqual(len(self.client.inputs), 2)

    async def test_crash_after_native_wake_before_local_ack_recovers(self):
        await self.create()
        original = self.deliveries.finish
        def crash(*_args):
            raise OSError("fixture crash before delivery acknowledgment")
        self.deliveries.finish = crash
        with self.assertRaises(OSError):
            await self.bridge.pump()
        self.deliveries.finish = original
        restored = WorkSessionBridge(self.service, WorkDeliveries(self.home / "deliveries.sqlite", self.bindings))
        await restored.pump()
        self.assertEqual(len(self.client.inputs), 2)
        self.assertEqual(restored.deliveries.pending(), [])

    async def test_finished_child_accepts_followup_with_same_identity(self):
        first = await self.create()
        await self.bridge.pump()
        next_input = await self.bridge.tool(self.master.native_id, "followup",
                                          {"action": "send", "session_id": first["session_id"], "message": "Continue this work"})
        self.assertEqual(next_input["session_id"], first["session_id"])
        self.assertNotEqual(next_input["admission_id"], first["admission_id"])
        self.assertEqual(len(self.bindings.children(self.master.logical_id)), 1)

    async def test_child_cannot_manage_sibling_but_can_message_parent(self):
        first, second = await self.create(), await self.create("create-b", "B")
        child = self.bindings.get(first["session_id"])
        with self.assertRaises(BindingError):
            await self.bridge.tool(child.native_id, "forged", {"action": "send", "session_id": second["session_id"], "message": "Stop"})
        receipt = await self.bridge.tool(child.native_id, "ask", {"action": "message_parent", "message": "Please enable keepalive for my work"})
        self.assertEqual(receipt["session_id"], self.master.logical_id)

    async def test_steer_targets_native_admission_without_new_session(self):
        child = await self.create()
        result = await self.bridge.tool(self.master.native_id, "steer", {"action": "steer", "session_id": child["session_id"], "message": "Use the new requirements"})
        self.assertEqual(self.client.steers[-1]["admission_id"], result["admission_id"])
        self.assertEqual(len(self.bindings.children(self.master.logical_id)), 1)

    async def test_pending_text_is_cleared_and_result_is_not_mirrored(self):
        await self.create()
        row = self.deliveries.pending()[0]
        self.assertIsNone(row["text"])
        await self.bridge.pump()
        with closing(sqlite3.connect(self.deliveries.path)) as db:
            self.assertEqual(db.execute("SELECT text, delivered FROM deliveries").fetchall(), [(None, 1)])
            columns = {row[1] for row in db.execute("PRAGMA table_info(deliveries)")}
        self.assertNotIn("result", columns)

    async def test_limits_and_retry_payload_conflicts(self):
        self.deliveries.max_pending = 1
        first = await self.create()
        with self.assertRaises(BindingError):
            await self.bridge.tool(self.master.native_id, "next", {"action": "send", "session_id": first["session_id"], "message": "Next"})
        with self.assertRaises(BindingError):
            await self.bridge.tool(self.master.native_id, "create-a", {"action": "create", "label": "A", "message": "Conflicting retry"})

    async def test_status_excludes_internal_messages(self):
        first = await self.create()
        status = await self.bridge.tool(self.master.native_id, "status", {"action": "status", "session_id": first["session_id"]})
        self.assertEqual(status["answer"], "Verified child result")
        self.assertNotIn("PRIVATE", json.dumps(status))

    async def test_stop_reply_excludes_native_profile_and_authority_metadata(self):
        first = await self.create()
        reply = await self.bridge.tool(self.master.native_id, "stop", {"action": "stop", "session_id": first["session_id"]})
        self.assertTrue(reply["stop_requested"])
        self.assertEqual(reply["outcome"], "cancelled")
        self.assertNotIn("ref", reply)
        self.assertNotIn("authority_epoch", reply)
        self.assertNotIn("/private", json.dumps(reply))

    async def test_invalid_action_and_blank_message_create_no_work(self):
        for args in ({"action": "delete"}, {"action": "create", "label": "Empty", "message": "  "}):
            with self.assertRaises(BindingError):
                await self.bridge.tool(self.master.native_id, "invalid", args)
        self.assertEqual(self.bindings.children(self.master.logical_id), [])

    async def test_replayed_admission_does_not_assert_work_is_unfinished(self):
        await self.create()
        await self.bridge.pump()
        replay = await self.create()
        self.assertEqual(replay["delivery"], "admitted")
        self.assertNotIn("finished", replay)


if __name__ == "__main__":
    unittest.main()
