"""View access over existing retained bindings, never another resource owner."""
import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import time
from uuid import UUID

from app.hermes_sessions.bindings import BindingError
from app.sandbox.models import SandboxError
from app.sandbox.view_access import unavailable


@dataclass(frozen=True)
class RetainedDesktopTarget:
    owner_id: str = field(repr=False)
    session_id: UUID
    allocation_id: UUID
    generation: str
    title: str
    deadline: datetime
    handle: object = field(repr=False)
    hermes_session_id: str

    def wire(self):
        return {"allocation_id": str(self.allocation_id), "generation": self.generation,
                "hermes_session_id": self.hermes_session_id, "title": self.title,
                "expires_at": self.deadline.isoformat()}


class RetainedSandboxViewAccess:
    """Borrow the running profile's bindings, locks and provider.

    refresh_choices is an explicit bounded provider read. cached_inventory is
    RAM-only and marks aged observations unknown. Neither starts a conversation.
    """

    def __init__(self, runtime, *, pool, owner_id, cache_seconds=30, clock=time.monotonic):
        self.resources = runtime.resources
        self.sessions = runtime.bridge.service.bindings
        self.sandboxes = runtime.sandboxes
        self.profile_id = str(runtime.home)
        if (self.resources.owner_id != owner_id or self.sessions.owner_id != owner_id
                or self.resources.profile_id != self.profile_id
                or str(self.sessions.profile_home) != self.profile_id
                or self.sandboxes.bindings is not self.resources):
            raise ValueError("Viewer must borrow the same owner/profile resource services")
        self.pool, self.owner_id = pool, owner_id
        self.cache_seconds, self.clock = cache_seconds, clock
        self._cards = {}
        self._refreshed = False
        self._refresh_slots = asyncio.Semaphore(4)
        self.closed = False

    def check(self, owner_id):
        if self.closed:
            raise unavailable("sandbox_view_closed", "Sandbox viewing is unavailable.", 503)
        if owner_id != self.owner_id:
            raise unavailable("sandbox_view_owner_mismatch", "This sandbox belongs to another account.", 403)

    async def _session(self, owner_id, session_id):
        self.check(owner_id)
        if not isinstance(session_id, UUID):
            raise unavailable("sandbox_view_session_unavailable", "The current session is unavailable.", 404)
        async with self.pool.acquire() as connection:
            exists = await connection.fetchval(
                "select true from jarvis_sessions where owner_id=$1 and id=$2", owner_id, session_id)
        if not exists:
            raise unavailable("sandbox_view_session_unavailable", "The current session is unavailable.", 404)

    @staticmethod
    def _generation(binding):
        scope = binding.scope
        values = [scope.owner_id, scope.profile_id, scope.logical_session_id,
                  str(scope.allocation_id), binding.sandbox_id, binding.policy.fingerprint]
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

    def _owned(self, binding, generation=None):
        if binding is None or binding.sandbox_id is None:
            raise unavailable("sandbox_not_running", "This retained sandbox is unavailable.", 404)
        scope = binding.scope
        if scope.owner_id != self.owner_id or scope.profile_id != self.profile_id:
            raise unavailable("sandbox_view_owner_mismatch", "This sandbox belongs to another account.", 403)
        try:
            session = self.sessions.get(scope.logical_session_id)
        except BindingError:
            raise unavailable("sandbox_view_session_unavailable", "The retained conversation is unavailable.", 404) from None
        if session.native_id is None:
            raise unavailable("sandbox_not_ready", "The retained conversation is not ready.")
        if generation is not None and generation != self._generation(binding):
            raise unavailable("sandbox_view_replaced", "This sandbox selection is no longer current.")
        return session

    def _binding(self, allocation_id, generation=None):
        if not isinstance(allocation_id, UUID):
            raise unavailable("sandbox_selection_required", "Choose the sandbox to view.", 422)
        binding = self.resources.get_by_allocation(allocation_id)
        self._owned(binding, generation)
        return binding

    def _remember(self, binding, info=None, *, state=None):
        session = self._owned(binding)
        state = state or (info.state if info is not None else "unknown")
        desktop = binding.policy.template == "desktop"
        card = {"allocation_id": str(binding.scope.allocation_id),
                "hermes_session_id": binding.scope.logical_session_id,
                "generation": self._generation(binding), "title": session.label[:160],
                "state": state, "capability": "desktop" if desktop else "terminal",
                "can_view": desktop and binding.prepared and state == "running",
                "can_resume_and_view": desktop and state == "paused",
                "observed_at": datetime.now(UTC).isoformat(),
                "expires_at": info.expires_at.isoformat() if info is not None and info.expires_at else None,
                "thumbnail": None, "stale": False}
        key = card["hermes_session_id"]
        self._cards.pop(key, None)
        self._cards[key] = (card, self.clock())
        while len(self._cards) > 128:
            self._cards.pop(next(iter(self._cards)))
        return dict(card)

    def cached_inventory(self, owner_id, *, after="", limit=8):
        self.check(owner_id)
        if type(limit) is not int or not 1 <= limit <= 32 or not isinstance(after, str) or len(after) > 256:
            raise unavailable("sandbox_selection_required", "Invalid sandbox page.", 422)
        keys = sorted(key for key in self._cards if key > after)
        items = []
        for key in keys[:limit]:
            value, checked = self._cards[key]
            card = dict(value)
            deadline_passed = (card["expires_at"] is not None
                               and datetime.fromisoformat(card["expires_at"]) <= datetime.now(UTC))
            if self.clock() - checked >= self.cache_seconds or deadline_passed:
                card.update(last_known_state=card["state"], state="unknown", stale=True,
                            can_view=False, can_resume_and_view=False)
            items.append(card)
        return {"items": items, "has_more": len(keys) > limit,
                "next_cursor": keys[limit - 1] if len(keys) > limit else None,
                "ready": self._refreshed}

    async def _inspect(self, binding):
        async with asyncio.timeout(5):
            info = await self.sandboxes.inspect(binding.scope.logical_session_id)
        current = self._binding(binding.scope.allocation_id, self._generation(binding))
        if info.scope != current.scope or info.sandbox_id != current.sandbox_id or info.policy != current.policy:
            raise unavailable("sandbox_view_replaced", "This sandbox selection is no longer current.")
        return current, info

    async def refresh_choices(self, owner_id, session_id, *, after="", limit=8):
        # Validate page arguments before SQL or provider I/O.
        self.cached_inventory(owner_id, after=after, limit=limit)
        await self._session(owner_id, session_id)
        return await self.refresh_inventory(owner_id, after=after, limit=limit)

    async def refresh_inventory(self, owner_id, *, after="", limit=8):
        """Explicit owner-scoped read, independent of a connected phone.

        Inspect existing retained bindings only. Never create, prepare or resume
        a sandbox; ordinary snapshots continue to use the RAM-only cache.
        """
        self.cached_inventory(owner_id, after=after, limit=limit)
        ids = [key for key in self.resources.bound_session_ids() if key > after]

        async def refresh(key):
            async with self._refresh_slots:
                binding = self.resources.get(key)
                self._owned(binding)
                if binding.pending_action is not None:
                    return self._remember(binding, state="resuming" if binding.pending_action == "resume" else "pausing")
                try:
                    current, info = await self._inspect(binding)
                except (SandboxError, TimeoutError, OSError):
                    # An inspection outage is not evidence of shutdown/deletion.
                    return self._remember(binding, state="unknown")
                return self._remember(current, info)

        items = await asyncio.gather(*(refresh(key) for key in ids[:limit]))
        self._refreshed = True
        return {"items": items, "has_more": len(ids) > limit,
                "next_cursor": ids[limit - 1] if len(ids) > limit else None, "ready": True}

    def _desktop(self, binding):
        if binding.policy.template != "desktop":
            raise unavailable("sandbox_desktop_unavailable", "This retained sandbox is terminal-only.")
        if binding.pending_action is not None:
            raise unavailable("sandbox_not_ready", "A sandbox lifecycle operation is still pending.")

    def _target(self, owner_id, session_id, binding, info):
        self._remember(binding, info)
        self._desktop(binding)
        if info.state != "running":
            raise unavailable("sandbox_not_running", "This sandbox is paused. Choose Resume and view.")
        if not binding.prepared:
            raise unavailable("sandbox_not_ready", "This sandbox's desktop is not ready.")
        if info.expires_at is None or info.expires_at <= datetime.now(UTC):
            raise unavailable("sandbox_view_access_ended", "This sandbox's running time has ended.", 410)
        session = self._owned(binding)
        return RetainedDesktopTarget(owner_id, session_id, binding.scope.allocation_id,
            self._generation(binding), session.label[:160], info.expires_at, info, session.logical_id)

    async def allocation(self, owner_id, session_id, allocation_id, generation=None):
        await self._session(owner_id, session_id)
        binding = self._binding(allocation_id, generation)
        self._desktop(binding)
        try:
            binding, info = await self._inspect(binding)
        except (SandboxError, TimeoutError, OSError):
            self._remember(binding, state="unknown")
            raise unavailable("sandbox_view_connection_failed", "The sandbox could not be inspected. Try again.", 503) from None
        return self._target(owner_id, session_id, binding, info)

    async def authorize_thumbnail(self, owner_id, session_id, allocation_id, generation):
        """Authorize an exact cached image without waking or inspecting a VM."""
        await self._session(owner_id, session_id)
        binding = self._binding(allocation_id, generation)
        self._desktop(binding)

    async def resume(self, owner_id, session_id, allocation_id, generation):
        await self._session(owner_id, session_id)
        binding = self._binding(allocation_id, generation)
        self._desktop(binding)
        try:
            binding, info = await self._inspect(binding)
            if info.state == "paused":
                self._remember(binding, state="resuming")
                async with asyncio.timeout(25):
                    info = await self.sandboxes.resume(binding.scope.logical_session_id)
                binding = self._binding(allocation_id, generation)
                if info.scope != binding.scope or info.sandbox_id != binding.sandbox_id or info.policy != binding.policy:
                    raise unavailable("sandbox_view_replaced", "This sandbox selection is no longer current.")
        except (SandboxError, TimeoutError, OSError):
            self._remember(binding, state="unknown")
            raise unavailable("sandbox_view_connection_failed", "Resume could not be confirmed. Refresh this sandbox.", 503) from None
        return self._target(owner_id, session_id, binding, info)

    async def close(self):
        self.closed = True
        self._cards.clear()
        # Runtime retains ownership of resource stores, provider and native work.
