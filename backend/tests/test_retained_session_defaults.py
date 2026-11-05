"""New launch defaults must not rewrite retained native creation identities."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from app.hermes_sessions.bindings import BindingError, SessionBindings
from app.hermes_sessions.service import RetainedSessionService


class NativeClient:
    def __init__(self, profile):
        self.profile_home = profile
        self.connection_generation = 1
        self.creates = []
        self.sessions = {}
        self.snapshots = []

    async def request(self, method, **params):
        assert method == "session.create"
        self.creates.append(copy.deepcopy(params))
        request_id = params["request_id"]
        if request_id in self.sessions:
            assert self.sessions[request_id][1] == params, "Native creation retry changed"
        else:
            self.sessions[request_id] = ("native-" + request_id, copy.deepcopy(params))
        return {"session_id": self.sessions[request_id][0]}

    async def snapshot(self, session_id):
        assert session_id in {value[0] for value in self.sessions.values()}
        self.snapshots.append(session_id)
        return {"session_id": session_id}


class RetainedSessionDefaultsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.profile = Path(self.directory.name).resolve()
        self.bindings = SessionBindings(self.profile / "bindings.sqlite", owner_id="owner-1",
                                        profile_home=self.profile)
        self.client = NativeClient(self.profile)
        self.original_launch = {"source": "cli", "toolsets": ["terminal", "jarvis_work"], "max_turns": 20}
        self.new_launch = {**self.original_launch, "toolsets": ["terminal", "jarvis_work", "memory"]}
        self.original = RetainedSessionService(self.client, self.bindings, launch=self.original_launch)
        self.updated = RetainedSessionService(self.client, self.bindings, launch=self.new_launch)

    async def test_changed_defaults_reuse_bound_master(self):
        master = await self.original.master()
        reused = await self.updated.master()
        self.assertEqual(reused, master)
        self.assertEqual(self.bindings.get(master.logical_id), master)
        self.assertEqual(json.loads(reused.launch_json), self.original_launch)
        self.assertEqual(len(self.client.creates), 1)
        self.assertEqual(self.client.snapshots, [master.native_id])

    async def test_unbound_master_recovers_original_launch(self):
        reserved = self.bindings.reserve(request_id="master-v1", parent_id=None,
                                         label="Master", launch=self.original_launch)
        # Native create committed, but the reply/bind was lost before defaults changed.
        native = await self.client.request("session.create", request_id=reserved.logical_id,
                                           **json.loads(reserved.launch_json))
        recovered = await self.updated.master()
        self.assertEqual(recovered.logical_id, reserved.logical_id)
        self.assertEqual(recovered.native_id, native["session_id"])
        self.assertEqual(recovered.launch_json, reserved.launch_json)
        self.assertEqual(self.client.creates[0], self.client.creates[1])
        self.assertEqual(len(self.client.sessions), 1)

    async def test_new_children_use_current_defaults(self):
        master = await self.original.master()
        old_child = await self.original.create_work(master.logical_id, request_id="old-work", label="Old")
        reused = await self.updated.master()
        new_child = await self.updated.create_work(reused.logical_id, request_id="new-work", label="New")
        self.assertEqual(reused, master)
        self.assertEqual(new_child.parent_id, master.logical_id)
        self.assertEqual(json.loads(new_child.launch_json), self.new_launch)
        self.assertEqual(self.client.creates[-1]["toolsets"], self.new_launch["toolsets"])
        self.assertEqual(self.bindings.get(old_child.logical_id), old_child)
        self.assertEqual(len(self.client.sessions), 3)

    async def test_child_retries_remain_strict(self):
        master = await self.original.master()
        child = await self.original.create_work(master.logical_id, request_id="work-1", label="Work")
        retry = await self.original.create_work(master.logical_id, request_id="work-1", label="Work")
        self.assertEqual(retry, child)
        with self.assertRaisesRegex(BindingError, "Conflicting conversation creation retry"):
            await self.updated.create_work(master.logical_id, request_id="work-1", label="Work")
        with self.assertRaisesRegex(BindingError, "Conflicting conversation creation retry"):
            await self.original.create_work(master.logical_id, request_id="work-1", label="Changed")
        self.assertEqual(self.bindings.get(child.logical_id), child)
        self.assertEqual(len(self.client.creates), 2)


if __name__ == "__main__":
    unittest.main()
