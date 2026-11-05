"""Offline retained viewer integration; no gateway, E2B or database network."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.app_controls import AppControlService
from app.live_state.companion import CompanionState
from app.live_state.companion_runtime import connect_companion_views
from app.hermes_sessions.bindings import SessionBindings
from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.hermes_sessions.sandbox_provider import RetainedSandboxInfo
from app.hermes_sessions.sandboxes import RetainedSandboxService
from app.runtime.sandbox_viewing import initialize_sandbox_views
from app.sandbox.models import SandboxError, SandboxPolicy
from app.sandbox.retained_view_access import RetainedSandboxViewAccess
from app.sandbox.view_provider import DesktopConnection
from app.sandbox.view_router import ViewRequest
from app.sandbox.view_service import SandboxViewService


class RetainedViewTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name).resolve()
        self.owner, self.phone_session = "retained-view-owner", uuid4()
        self.sessions = SessionBindings(self.home / "bindings.sqlite", owner_id=self.owner, profile_home=self.home)
        master = self.sessions.reserve(request_id="master", parent_id=None, label="Master", launch={})
        self.master = self.sessions.bind(master.logical_id, "native-master")
        self.resources = SessionSandboxBindings(self.home / "sandboxes.sqlite", owner_id=self.owner, profile_id=str(self.home))
        self.addCleanup(self.resources.close)
        self.handles = {}

        async def info(sandbox_id, scope, policy):
            return self.handles[sandbox_id]

        async def pause(handle):
            value = handle.model_copy(update={"state": "paused", "expires_at": None})
            self.handles[handle.sandbox_id] = value
            return value

        async def resume(handle, *, timeout_seconds):
            value = handle.model_copy(update={"state": "running", "expires_at": datetime.now(UTC) + timedelta(seconds=timeout_seconds)})
            self.handles[handle.sandbox_id] = value
            return value

        self.provider = SimpleNamespace(info=AsyncMock(side_effect=info), pause=AsyncMock(side_effect=pause),
            resume=AsyncMock(side_effect=resume), create=AsyncMock(side_effect=AssertionError("must not create")),
            prepare=AsyncMock())
        self.sandboxes = RetainedSandboxService(self.resources, self.provider, policy=SandboxPolicy(template="desktop"))
        self.connection = SimpleNamespace(fetchval=AsyncMock(
            side_effect=lambda query, owner, session: owner == self.owner and session == self.phone_session))

        @asynccontextmanager
        async def acquire():
            yield self.connection

        self.pool = SimpleNamespace(acquire=acquire)
        self.runtime = SimpleNamespace(home=self.home, resources=self.resources, sandboxes=self.sandboxes,
            bridge=SimpleNamespace(service=SimpleNamespace(bindings=self.sessions)))
        self.now = 0
        self.access = RetainedSandboxViewAccess(self.runtime, pool=self.pool, owner_id=self.owner, clock=lambda: self.now)
        self.viewer = SimpleNamespace(open=AsyncMock(side_effect=self.desktop_connection),
                                      inspect=AsyncMock(side_effect=self.desktop_connection))
        self.views = SandboxViewService(self.access, self.viewer)

    @staticmethod
    async def desktop_connection(handle):
        return DesktopConnection("wss://protected.invalid", "private-token", handle.expires_at)

    def resource(self, label="Computer", *, state="running", template="desktop", prepared=True):
        child = self.sessions.reserve(request_id=label, parent_id=self.master.logical_id, label=label, launch={})
        child = self.sessions.bind(child.logical_id, "native-" + label)
        policy = self.sandboxes.policy.model_copy(update={"template": template})
        binding = self.resources.reserve(child.logical_id, policy)
        self.resources.start_creation(binding)
        sandbox_id = "box-" + label.lower()
        self.resources.bind(binding, sandbox_id)
        binding = self.resources.get(child.logical_id)
        if prepared:
            self.resources.mark_prepared(binding)
        binding = self.resources.get(child.logical_id)
        self.handles[sandbox_id] = RetainedSandboxInfo(sandbox_id=sandbox_id, scope=binding.scope,
            policy=policy, state=state, expires_at=datetime.now(UTC) + timedelta(minutes=10) if state == "running" else None,
            template_id=template)
        return binding

    async def explicit_state(self):
        state = CompanionState(owner_id=self.owner,
            working_memory=SimpleNamespace(get_owner_working_memory=AsyncMock(return_value=None)),
            reminders=SimpleNamespace(list_reminders=AsyncMock(return_value=[])))
        await state.initialize()
        presentations = SimpleNamespace(present=AsyncMock(return_value={
            "outcome": "accepted", "summary": "Pending display",
            "presentation": {"state": "pending", "request_id": str(uuid4())}}))
        controls = AppControlService(owner_id=self.owner, pool=self.pool,
            sandbox_views=self.views, presentations=presentations)
        connect_companion_views(SimpleNamespace(companion_state=state,
            sandbox_views=self.views, app_controls=controls))
        return state, controls, presentations

    async def test_explicit_read_discovers_new_binding_and_ids_start_exact_view(self):
        state, controls, presentations = await self.explicit_state()
        self.assertEqual((await state.read_current())["system"]["sandboxes"]["items"], [])
        binding = self.resource("Fresh")
        self.assertEqual(state.read()["system"]["sandboxes"]["items"], [])
        value = await state.read_current()
        card = value["system"]["sandboxes"]["items"][0]
        self.assertEqual(card["allocation_id"], str(binding.scope.allocation_id))
        self.assertTrue(card["can_view"])
        self.assertEqual(value["system"]["app"]["sandboxes"]["items"], [card])
        self.connection.fetchval.assert_not_awaited()
        result = await controls.invoke(self.owner, self.phone_session, "sandbox.start",
            allocation_id=card["allocation_id"], generation=card["generation"])
        self.assertEqual(result["outcome"], "accepted")
        self.assertEqual(presentations.present.call_args.kwargs["artifact_id"], binding.scope.allocation_id)
        self.provider.create.assert_not_awaited()
        self.provider.resume.assert_not_awaited()
        self.provider.prepare.assert_not_awaited()
        self.assertEqual(self.now, 0)

    async def test_explicit_read_observes_resume_without_waiting_for_cache_expiry(self):
        binding = self.resource("Resumed", state="paused")
        state, _, _ = await self.explicit_state()
        before = (await state.read_current())["system"]["sandboxes"]["items"][0]
        self.assertTrue(before["can_resume_and_view"])
        await self.sandboxes.resume(binding.scope.logical_session_id)
        self.assertEqual(state.read()["system"]["sandboxes"]["items"][0]["state"], "paused")
        after = (await state.read_current())["system"]["sandboxes"]["items"][0]
        self.assertEqual(after["state"], "running")
        self.assertTrue(after["can_view"])
        self.assertEqual(after["generation"], before["generation"])
        self.assertEqual(after["allocation_id"], before["allocation_id"])
        self.provider.resume.assert_awaited_once()
        self.provider.create.assert_not_awaited()
        self.assertEqual(self.now, 0)

    async def test_explicit_refresh_failure_projects_unknown_without_mutation(self):
        self.resource()
        state, _, _ = await self.explicit_state()
        await state.read_current()
        self.provider.info.side_effect = SandboxError("sandbox_provider_outcome_uncertain")
        card = (await state.read_current())["system"]["sandboxes"]["items"][0]
        self.assertEqual(card["state"], "unknown")
        self.assertFalse(card["can_view"])
        self.provider.resume.assert_not_awaited()
        self.provider.create.assert_not_awaited()

    async def test_explicit_inventory_validates_owner_and_page_before_io(self):
        self.resource()
        for owner, limit in (("other", 8), (self.owner, 0), (self.owner, 33)):
            with self.subTest(owner=owner, limit=limit), self.assertRaises(JarvisHTTPError):
                await self.views.refresh_inventory(owner, limit=limit)
        self.provider.info.assert_not_awaited()
        self.connection.fetchval.assert_not_awaited()

    async def test_inventory_includes_paused_and_terminal_without_resume(self):
        self.resource("Desktop")
        self.resource("Paused", state="paused")
        self.resource("Terminal", template="base")
        result = await self.views.choices(self.owner, self.phone_session)
        cards = {card["title"]: card for card in result["items"]}
        self.assertTrue(cards["Desktop"]["can_view"])
        self.assertTrue(cards["Paused"]["can_resume_and_view"])
        self.assertEqual(cards["Terminal"]["capability"], "terminal")
        self.assertFalse(cards["Terminal"]["can_view"])
        self.assertIsNone(cards["Paused"]["expires_at"])
        self.assertNotIn("task_id", cards["Desktop"])
        self.provider.resume.assert_not_awaited()
        self.provider.create.assert_not_awaited()
        self.provider.prepare.assert_not_awaited()

    async def test_cached_reads_are_ram_only_and_explicitly_stale(self):
        self.assertFalse(self.views.cached_inventory(self.owner)["ready"])
        self.resource()
        await self.views.choices(self.owner, self.phone_session)
        self.provider.info.reset_mock()
        self.connection.fetchval.reset_mock()
        with patch.object(self.resources, "bound_session_ids", side_effect=AssertionError("cache read hit SQLite")):
            card = self.views.cached_inventory(self.owner)["items"][0]
            self.assertEqual(card["state"], "running")
            self.assertIsNone(self.views.current(self.owner))
            self.now = 31
            card = self.views.cached_inventory(self.owner)["items"][0]
            self.assertEqual(card["state"], "unknown")
            self.assertEqual(card["last_known_state"], "running")
            self.assertTrue(card["stale"])
            self.assertFalse(card["can_view"])
        self.provider.info.assert_not_awaited()
        self.connection.fetchval.assert_not_awaited()

    async def test_exact_owner_phone_session_allocation_and_generation(self):
        binding = self.resource()
        generation = self.access._generation(binding)
        for owner, phone, allocation, version in (
            ("other", self.phone_session, binding.scope.allocation_id, generation),
            (self.owner, uuid4(), binding.scope.allocation_id, generation),
            (self.owner, self.phone_session, uuid4(), generation),
            (self.owner, self.phone_session, binding.scope.allocation_id, "0" * 64),
        ):
            with self.subTest(owner=owner, allocation=allocation), self.assertRaises(JarvisHTTPError):
                await self.views.open(owner, phone, allocation, version, resume=True)
        self.provider.info.assert_not_awaited()
        self.provider.resume.assert_not_awaited()
        self.viewer.open.assert_not_awaited()

    async def test_paused_requires_explicit_resume_and_preserves_exact_resource(self):
        binding = self.resource(state="paused")
        generation = self.access._generation(binding)
        with self.assertRaises(JarvisHTTPError) as error:
            await self.views.open(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        self.assertEqual(error.exception.code, "sandbox_not_running")
        self.provider.resume.assert_not_awaited()
        lease = await self.views.open(self.owner, self.phone_session, binding.scope.allocation_id, generation, resume=True)
        self.assertEqual(lease.target.handle.sandbox_id, binding.sandbox_id)
        self.assertEqual(lease.target.allocation_id, binding.scope.allocation_id)
        self.assertEqual(lease.target.generation, generation)
        self.assertEqual(lease.target.hermes_session_id, binding.scope.logical_session_id)
        self.provider.resume.assert_awaited_once()
        self.provider.create.assert_not_awaited()
        self.views.revoke(self.owner, self.phone_session, lease.id)
        self.assertIsNone(self.views.current(self.owner))
        self.provider.pause.assert_not_awaited()

    async def test_micro_pause_does_not_replace_view_identity(self):
        binding = self.resource()
        generation = self.access._generation(binding)
        lease = await self.views.open(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        await self.sandboxes.micro_pause(binding.scope.logical_session_id)
        self.assertGreater(self.resources.get_by_allocation(binding.scope.allocation_id).generation, binding.generation)
        await self.views.validate(lease)
        self.assertEqual(self.views.current(self.owner)["target"]["generation"], generation)

    async def test_terminal_only_never_resumes_or_starts_a_view(self):
        binding = self.resource(template="base", state="paused")
        with self.assertRaises(JarvisHTTPError) as error:
            await self.views.open(self.owner, self.phone_session, binding.scope.allocation_id,
                                  self.access._generation(binding), resume=True)
        self.assertEqual(error.exception.code, "sandbox_desktop_unavailable")
        self.provider.info.assert_not_awaited()
        self.provider.resume.assert_not_awaited()

    async def test_inspection_failure_is_unknown_not_deletion(self):
        binding = self.resource()
        self.provider.info.side_effect = SandboxError("sandbox_provider_outcome_uncertain")
        result = await self.views.choices(self.owner, self.phone_session)
        self.assertEqual(result["items"][0]["state"], "unknown")
        self.assertIsNotNone(self.resources.get_by_allocation(binding.scope.allocation_id))

    async def test_pending_lifecycle_is_not_observed_running(self):
        binding = self.resource()
        self.resources.intend(binding, "resume")
        result = await self.views.choices(self.owner, self.phone_session)
        self.assertEqual(result["items"][0]["state"], "resuming")
        self.assertFalse(result["items"][0]["can_view"])
        self.provider.info.assert_not_awaited()

    async def test_binding_lookup_is_exact_and_close_preserves_runtime(self):
        binding = self.resource()
        self.assertEqual(self.resources.get_by_allocation(binding.scope.allocation_id), binding)
        self.assertIsNone(self.resources.get_by_allocation(uuid4()))
        with self.assertRaises(ValueError):
            self.resources.get_by_allocation(str(binding.scope.allocation_id))
        await self.views.close()
        self.assertEqual(self.resources.get_by_allocation(binding.scope.allocation_id), binding)
        self.provider.pause.assert_not_awaited()

    async def test_composition_borrows_provider_without_native_readiness_or_presentations(self):
        services = SimpleNamespace(settings=SimpleNamespace(JARVIS_OWNER_ID=self.owner), db_pool=self.pool,
                                   workspace_presentations=None)
        secret_store = object()
        with patch("app.runtime.sandbox_viewing.TaskRepository", side_effect=AssertionError("legacy registry")):
            initialize_sandbox_views(services, retained_runtime=self.runtime, secrets=secret_store)
        self.assertIs(services.sandbox_views.access.resources, self.resources)
        self.assertIs(services.sandbox_views.provider.provider, self.provider)
        self.assertIs(services.sandbox_views.provider.secrets, secret_store)
        self.provider.info.assert_not_awaited()
        await services.sandbox_views.close()
        self.resource("StillOwned")

    async def test_exact_retained_presentation_uses_existing_journal(self):
        binding = self.resource()
        presentations = SimpleNamespace(present=AsyncMock(return_value={"status": "pending"}))
        ctx = SimpleNamespace(owner_id=self.owner, session_id=self.phone_session, operation_id=uuid4())
        result = await self.views.present(ctx, presentations, allocation_id=binding.scope.allocation_id,
                                          generation=self.access._generation(binding))
        self.assertEqual(result["status"], "pending")
        presentations.present.assert_awaited_once_with(ctx, action="open", artifact_kind="sandbox",
            artifact_id=binding.scope.allocation_id, version_id=self.access._generation(binding))
        self.assertIsNone(self.views.current(self.owner))

    async def test_inventory_pagination_and_wrong_owner_cache(self):
        self.resource("One"); self.resource("Two")
        first = await self.views.choices(self.owner, self.phone_session, limit=1)
        self.assertTrue(first["has_more"])
        second = await self.views.choices(self.owner, self.phone_session, after=first["next_cursor"], limit=1)
        self.assertFalse(second["has_more"])
        self.assertNotEqual(first["items"][0]["allocation_id"], second["items"][0]["allocation_id"])
        with self.assertRaises(JarvisHTTPError):
            self.views.cached_inventory("other")
        with self.assertRaises(JarvisHTTPError):
            await self.views.choices(self.owner, self.phone_session, limit=0)

    def test_resume_is_explicit_strict_boolean(self):
        from pydantic import ValidationError
        fields = dict(session_id=self.phone_session, allocation_id=uuid4(), generation="a" * 64)
        self.assertFalse(ViewRequest(**fields).resume)
        self.assertTrue(ViewRequest(**fields, resume=True).resume)
        with self.assertRaises(ValidationError):
            ViewRequest(**fields, resume="yes")

    async def test_thumbnail_is_lazy_cached_and_never_starts_view(self):
        binding = self.resource()
        generation = self.access._generation(binding)
        self.viewer.thumbnail = AsyncMock(return_value=b"jpeg-preview")
        await self.views.choices(self.owner, self.phone_session)
        self.viewer.thumbnail.assert_not_awaited()
        first = await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        second = await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        self.assertEqual(first, second)
        self.assertEqual(first["content_type"], "image/jpeg")
        self.assertEqual(self.viewer.thumbnail.await_count, 1)
        self.assertIsNone(self.views.cached_inventory(self.owner)["items"][0]["thumbnail"])
        self.assertIsNone(self.views.current(self.owner))
        self.viewer.open.assert_not_awaited()
        self.provider.resume.assert_not_awaited()

    async def test_thumbnail_keeps_cached_paused_image_without_waking(self):
        binding = self.resource()
        generation = self.access._generation(binding)
        self.viewer.thumbnail = AsyncMock(return_value=b"jpeg-preview")
        first = await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        await self.sandboxes.pause(binding.scope.logical_session_id)
        with patch("app.sandbox.view_service.time.monotonic", return_value=float("inf")):
            paused = await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        self.assertEqual(first["image"], paused["image"])
        self.assertTrue(paused["stale"])
        self.assertEqual(self.viewer.thumbnail.await_count, 1)
        self.provider.resume.assert_not_awaited()

    async def test_thumbnail_rejects_wrong_owner_session_generation_and_terminal(self):
        binding = self.resource()
        generation = self.access._generation(binding)
        self.viewer.thumbnail = AsyncMock(return_value=b"jpeg-preview")
        for owner, session, selected in [("other", self.phone_session, generation),
                (self.owner, uuid4(), generation), (self.owner, self.phone_session, "0" * 64)]:
            with self.assertRaises(JarvisHTTPError):
                await self.views.thumbnail(owner, session, binding.scope.allocation_id, selected)
        terminal = self.resource("TerminalPreview", template="base")
        with self.assertRaises(JarvisHTTPError):
            await self.views.thumbnail(self.owner, self.phone_session, terminal.scope.allocation_id, self.access._generation(terminal))
        self.viewer.thumbnail.assert_not_awaited()

    async def test_thumbnail_paused_without_cache_and_oversize_are_honest_failures(self):
        binding = self.resource(state="paused")
        generation = self.access._generation(binding)
        self.viewer.thumbnail = AsyncMock(return_value=b"x" * 32769)
        with self.assertRaises(JarvisHTTPError):
            await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        self.viewer.thumbnail.assert_not_awaited()
        await self.sandboxes.resume(binding.scope.logical_session_id)
        with self.assertRaises(JarvisHTTPError):
            await self.views.thumbnail(self.owner, self.phone_session, binding.scope.allocation_id, generation)
        self.assertFalse(self.views._thumbnails)


if __name__ == "__main__":
    unittest.main()
