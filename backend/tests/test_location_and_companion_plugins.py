from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
import unittest
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.live_state.companion import CompanionState
from app.live_state.location_router import router as location_router
from app.plugins.companion_marketplace import CompanionPluginService
from app.plugins.router import router as plugins_router
from app.realtime.live_state_reply import compact_live_state_reply


class FakeWorkingMemory:
    def __init__(self):
        self.row = None

    async def get_owner_working_memory(self, *, owner_id):
        return deepcopy(self.row)

    async def compare_and_swap_owner_working_memory(self, *, owner_id, expected_revision, state, expires_at=None):
        current_rev = 0 if self.row is None else self.row.revision
        if current_rev != expected_revision:
            return NS(applied=False, state=deepcopy(self.row))
        self.row = NS(owner_id=owner_id, revision=current_rev + 1, state=deepcopy(state))
        return NS(applied=True, state=deepcopy(self.row))


class FakeReminders:
    async def list_reminders(self, *, owner_id, status="active", limit=128):
        return []


class FakeBox:
    pass


class FakeProvider:
    def __init__(self):
        self.scripts: list[str] = []

    async def _connect(self, handle):
        return FakeBox()

    async def _run_privileged(self, box, script, *, timeout):
        self.scripts.append(script)
        if "apt-get remove" in script:
            return NS(stdout="JARVIS_PLUGIN_REMOVED\n", stderr="", exit_code=0)
        return NS(stdout="JARVIS_PLUGIN_OK:2.10.36-2\n", stderr="", exit_code=0)


class FakeSandboxes:
    def __init__(self, initial_states: dict[str, str]):
        self.states = dict(initial_states)
        self.events: list[tuple[str, str]] = []
        self.provider = FakeProvider()
        self.bindings = NS(
            bound_session_ids=lambda: list(self.states.keys()),
        )
        self._locks = {sid: asyncio.Lock() for sid in initial_states}

    @asynccontextmanager
    async def _lock(self, sid: str):
        lock = self._locks.setdefault(sid, asyncio.Lock())
        async with lock:
            self.events.append((sid, "lock_acquired"))
            try:
                yield
            finally:
                self.events.append((sid, "lock_released"))

    def _required(self, sid: str):
        return NS(scope=NS(logical_session_id=sid, owner_id="owner-1"), sandbox_id=f"sbx-{sid}")

    async def _existing(self, binding):
        sid = binding.scope.logical_session_id
        return binding, NS(state=self.states[sid], sandbox_id=binding.sandbox_id)

    async def _lifecycle(self, sid: str, action: str):
        self.events.append((sid, action))
        self.states[sid] = "paused" if action == "pause" else "running"
        return NS(state=self.states[sid], sandbox_id=f"sbx-{sid}")


def _build_test_app(runtime_services, owner_id="owner-1") -> FastAPI:
    app = FastAPI()
    app.state.runtime_services = runtime_services

    @app.exception_handler(JarvisHTTPError)
    async def _err(_req, exc: JarvisHTTPError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id=owner_id)
    app.include_router(location_router)
    app.include_router(plugins_router)
    return app


class LocationAndCompanionPluginsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 11, 4, 16, 0, tzinfo=timezone.utc)
        self.memory = FakeWorkingMemory()
        self.state = CompanionState(
            owner_id="owner-1",
            working_memory=self.memory,
            reminders=FakeReminders(),
            clock=lambda: self.now,
        )
        await self.state.initialize()

    async def test_location_update_and_pull_only_live_state_reply(self):
        runtime = NS(companion_state=self.state, companion_plugin_service=None, plugin_policy_service=None)
        client = TestClient(_build_test_app(runtime))

        # Initially unset
        self.assertNotIn("location", self.state.read())
        res0 = client.get("/live-state/location")
        self.assertEqual(res0.status_code, 200)
        self.assertIsNone(res0.json()["location"])

        # Push Rathmines fix
        res1 = client.put("/live-state/location", json={
            "latitude": 53.3219,
            "longitude": -6.2654,
            "horizontal_accuracy_m": 18.4,
            "place_name": "Rathmines, Dublin",
            "locality": "Dublin",
            "country": "Ireland",
            "timezone": "Europe/Dublin",
            "captured_at": self.now.isoformat(),
        })
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["status"], "applied")
        self.assertEqual(res1.json()["location"]["place_name"], "Rathmines, Dublin")
        self.assertFalse(res1.json()["location"]["stale"])

        # Out-of-order older fix is ignored
        older = (self.now - timedelta(minutes=5)).isoformat()
        res_old = client.put("/live-state/location", json={
            "latitude": 51.5,
            "longitude": -0.1,
            "place_name": "London",
            "captured_at": older,
        })
        self.assertEqual(res_old.json()["status"], "superseded")
        self.assertEqual(self.state.read()["location"]["place_name"], "Rathmines, Dublin")

        # Pull-only Live State reply includes location for human & all, omits for system
        snapshot = self.state.read()
        human_reply = compact_live_state_reply(snapshot, "human")
        system_reply = compact_live_state_reply(snapshot, "system")
        all_reply = compact_live_state_reply(snapshot, "all")
        self.assertEqual(human_reply["location"]["place_name"], "Rathmines, Dublin")
        self.assertEqual(all_reply["location"]["locality"], "Dublin")
        self.assertNotIn("location", system_reply)

        # Staleness flips after 30 minutes
        self.now += timedelta(minutes=31)
        self.assertTrue(self.state.read()["location"]["stale"])

    async def test_companion_plugins_pause_stream_cutoff_root_install_and_resume(self):
        sandboxes = FakeSandboxes({"sess-running": "running", "sess-idle": "paused"})
        lease_id = uuid4()
        session_uuid = uuid4()
        closed_leases: list = []
        sent_events: list = []

        class FakeViews:
            def __init__(self):
                self.leases = {
                    lease_id: NS(
                        id=lease_id,
                        target=NS(owner_id="owner-1", session_id=session_uuid),
                    )
                }
                self.remembered: list[tuple[str, str]] = []
                self.access = NS(
                    _remember=lambda binding, info=None, state="unknown": self.remembered.append(
                        (binding.scope.logical_session_id, state)
                    )
                )

            def stop(self, owner_id, lid):
                closed_leases.append((owner_id, lid))
                self.leases.pop(lid, None)
                return True

        views = FakeViews()
        controls = NS(_notify=lambda sid, payload: sent_events.append((sid, payload)) or True)
        runtime = NS(
            companion_state=self.state,
            plugin_policy_service=None,
            sandbox_views=views,
            app_controls=controls,
            hermes_profile_runtime=NS(sandboxes=sandboxes),
        )
        plugins = CompanionPluginService(
            owner_id="owner-1",
            working_memory=self.memory,
            runtime_services=runtime,
            clock=lambda: self.now,
        )
        await plugins.initialize()
        runtime.companion_plugin_service = plugins
        self.state.plugin_projection = plugins.live_state_projection

        # Instant catalog read
        cat = plugins.get_extensions_catalog(owner_id="owner-1")
        self.assertEqual(len(cat["installed"]), 3)
        self.assertGreaterEqual(len(cat["available"]), 6)

        # Legacy GET /plugins compatibility in companion mode
        legacy = plugins.get_legacy_catalog(owner_id="owner-1")
        self.assertEqual(legacy.schema_version, 1)
        self.assertEqual(len(legacy.items), 3)

        # Start GIMP install -> cuts stream, pauses running VM, runs root apt-get, resumes running VM & re-pauses idle VM
        started = await plugins.start_install(owner_id="owner-1", plugin_id="gimp")
        self.assertTrue(started["sandbox_maintenance"]["active"])
        await plugins._active_task

        self.assertEqual(closed_leases, [("owner-1", lease_id)])
        self.assertEqual(sent_events, [(session_uuid, {"type": "workspace.sandbox.closed", "view_id": str(lease_id)})])
        self.assertEqual(sandboxes.states["sess-running"], "running")
        self.assertEqual(sandboxes.states["sess-idle"], "paused")
        self.assertEqual(len(sandboxes.provider.scripts), 2)
        self.assertIn("apt-get install -y --no-install-recommends gimp", sandboxes.provider.scripts[0])
        self.assertIn("! /usr/bin/sudo -n true", sandboxes.provider.scripts[0])

        after = plugins.get_extensions_catalog(owner_id="owner-1")
        self.assertFalse(after["sandbox_maintenance"]["active"])
        gimp_card = next(item for item in after["installed"] if item["id"] == "gimp")
        self.assertEqual(gimp_card["version"], "2.10.36-2")
        self.assertIn("gimp", self.state.read()["system"]["plugins"]["installed"])
