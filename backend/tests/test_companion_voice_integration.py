"""Real state/tools connected to the voice seams; no paid provider calls."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.quiet_delivery import QuietDeliverySuppressed
from app.live_state.companion import CompanionState
from app.live_state.companion_tools import CompanionTools
from app.live_state.companion_runtime import initialize_companion, connect_companion_tools
from app.realtime.live_retained import RetainedLiveWork
from app.realtime.live_session import LiveAudioSession, _DELEGATING_INSTRUCTIONS
from app.tasks.models import TaskError
from test_companion_state import Memory, Reminders
from test_live_router import RouterTests, response


class CompanionVoiceTests(unittest.IsolatedAsyncioTestCase):
    def test_voice_delegates_actions_without_a_feature_catalogue(self):
        self.assertIn("Use backend delegation", _DELEGATING_INSTRUCTIONS)
        self.assertIn("an action outside this conversation", _DELEGATING_INSTRUCTIONS)
        self.assertIn("Handle ordinary conversation directly", _DELEGATING_INSTRUCTIONS)
        self.assertNotIn("control music playback", _DELEGATING_INSTRUCTIONS)

    def test_voice_owns_actions_without_claiming_unconfirmed_completion(self):
        self.assertIn("Speak in first person", _DELEGATING_INSTRUCTIONS)
        self.assertIn("tools and background agents are your capabilities", _DELEGATING_INSTRUCTIONS)
        self.assertIn("Describe pending work as what you are starting", _DELEGATING_INSTRUCTIONS)
        self.assertIn("State success only when confirmed by backend results", _DELEGATING_INSTRUCTIONS)

    async def asyncSetUp(self):
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(), reminders=Reminders())
        await self.state.initialize()
        self.tools = CompanionTools(self.state)
        self.fixture = RouterTests()
        self.fixture.setUp()
        self.fixture.session.services = NS(companion_state=self.state, companion_tools=self.tools)
        self.fixture.session.context = NS(owner_id="fixture", jarvis_session_id=uuid4())

    async def test_luna_neither_advertises_nor_executes_removed_quiet_tool(self):
        self.fixture.client.responses.create.return_value = response("set_quiet_mode",
            '{"active":true,"duration_seconds":null}')
        await self.fixture.run_route()
        self.assertNotIn("set_quiet_mode", {
            tool["name"] for tool in self.fixture.client.responses.create.call_args.kwargs["tools"]})
        self.assertNotIn("quiet_mode", self.state.read()["system"])
        self.fixture.client.responses.create.assert_awaited_once()
        self.fixture.work.assert_not_awaited()
        self.fixture.publish.assert_awaited_once()
        self.assertIn("No action was started", self.fixture.publish.await_args.args[0].text)

    async def test_luna_creates_operational_reminder_without_recall(self):
        self.fixture.client.responses.create.return_value = response("manage_reminder",
            '{"action":"create","reminder_id":null,"title":"Umbrella",'
            '"condition":"When leaving","repeat_until_confirmed":false}')
        await self.fixture.run_route()
        self.assertEqual(self.state.read()["system"]["reminders"][0]["title"], "Umbrella")
        self.fixture.session.read_live_state.assert_not_awaited()
        self.fixture.work.assert_not_awaited()

    async def test_rejected_mutation_is_not_claimed_successful(self):
        self.fixture.client.responses.create.return_value = response("manage_reminder",
            '{"action":"create","reminder_id":null,"title":"Umbrella",'
            '"condition":"When leaving","repeat_until_confirmed":"false"}')
        await self.fixture.run_route()
        self.assertEqual(self.state.read()["system"]["reminders"], [])
        self.assertIn("unconfirmed", self.fixture.publish.await_args.args[0].text)

    async def test_live_read_is_pull_only_and_retains_work_status(self):
        session = object.__new__(LiveAudioSession)
        session.services = self.fixture.session.services
        session.context = self.fixture.session.context
        session.work = NS(snapshot=lambda: {"last_admission": "known-work"})
        value, roots = await session.read_live_state()
        self.assertEqual(roots, ())
        self.assertEqual(value["system"]["work"]["last_admission"], "known-work")
        session.context = NS(owner_id="other")
        with self.assertRaises(Exception):
            await session.read_live_state()

    async def test_completion_ignores_old_quiet_but_preserves_readiness_and_owner(self):
        work = object.__new__(RetainedLiveWork)
        route = self.fixture.route
        work._detached = False
        work.session = NS(ready=True, request=NS(route=route), services=self.fixture.session.services)
        result = NS(route=route)
        async with work._guard(result):
            pass
        self.state._section["quiet"] = {"active": True, "expires_at": None}
        async with work._guard(result):
            pass
        self.state._ready = False
        with self.assertRaises(QuietDeliverySuppressed):
            async with work._guard(result):
                self.fail("unready completion reached provider")
        self.state._ready = True
        async with work._guard(result):
            pass
        self.state.owner_id = "other"
        with self.assertRaises(TaskError):
            async with work._guard(result):
                self.fail("foreign state admitted completion")

    async def test_composition_installs_one_invalidator_before_intake_tools(self):
        services = NS(db_pool=object(), settings=NS(JARVIS_OWNER_ID="fixture"),
                      shared_memory_intake=object(), realtime_session_manager=NS())
        with patch("app.live_state.companion_runtime.PerceptionRepository", return_value=Memory()), \
                patch("app.live_state.companion_runtime.ReminderRepository", return_value=Reminders()):
            await initialize_companion(services, environ={"JARVIS_COMPANION_STATE_ENABLED": "true"})
        connect_companion_tools(services)
        self.assertIs(services.shared_memory_invalidator.__self__, services.companion_state)
        self.assertIs(services.companion_tools.memory_intake, services.shared_memory_intake)
