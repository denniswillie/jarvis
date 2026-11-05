"""Short-lived authenticated desktop views, separate from sandbox lifetime."""
import asyncio
import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import secrets
import time
from uuid import UUID, uuid4

from app.errors import JarvisHTTPError
from app.sandbox.models import SandboxError
from app.sandbox.view_access import DesktopTarget, unavailable
from app.sandbox.retained_view_access import RetainedDesktopTarget


@dataclass
class DesktopLease:
    id: UUID
    target: DesktopTarget | RetainedDesktopTarget
    expires_at: datetime
    ticket: str = field(repr=False)
    closed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    connected: bool = False

    def wire(self):
        return {"id": str(self.id), "target": self.target.wire(), "expires_at": self.expires_at.isoformat()}


class SandboxViewService:
    def __init__(self, access, provider):
        self.access, self.provider = access, provider
        self.leases = {}
        self._opening = asyncio.Lock()
        self._thumbnail_lock = asyncio.Lock()
        self._thumbnails = {}
        self.closed = False

    async def choices(self, owner, session, *, after="", limit=8):
        refresh = getattr(self.access, "refresh_choices", None)
        if refresh is not None:
            return await refresh(owner, session, after=after, limit=limit)
        targets, more = await self.access.choices(owner, session)
        return {"items": [target.wire() for target in targets], "has_more": more}

    def cached_inventory(self, owner, *, after="", limit=8):
        self.access.check(owner)
        cached = getattr(self.access, "cached_inventory", None)
        return cached(owner, after=after, limit=limit) if cached is not None else {
            "items": [], "has_more": False, "next_cursor": None, "ready": False}

    async def refresh_inventory(self, owner, *, after="", limit=8):
        """Refresh an explicit state read without requiring phone connectivity."""
        self.access.check(owner)
        refresh = getattr(self.access, "refresh_inventory", None)
        if refresh is not None:
            return await refresh(owner, after=after, limit=limit)
        return self.cached_inventory(owner, after=after, limit=limit)

    async def thumbnail(self, owner, session, allocation, generation):
        # Images live only in this bounded disposable viewer cache, never in
        # the compact Voice/Live State inventory or a second resource registry.
        authorize = getattr(self.access, "authorize_thumbnail", None)
        if authorize is None:
            raise unavailable("sandbox_thumbnail_unavailable", "No screen preview is available.", 404)
        await authorize(owner, session, allocation, generation)
        key = (owner, allocation, generation)
        async with self._thumbnail_lock:
            if self.closed:
                raise unavailable("sandbox_view_closed", "Sandbox viewing is unavailable.", 503)
            cached = self._thumbnails.get(key)
            if cached is not None and time.monotonic() - cached[0] < 30:
                return dict(cached[1])
            try:
                target = await self.access.allocation(owner, session, allocation, generation)
            except JarvisHTTPError as error:
                if cached is not None and error.code == "sandbox_not_running":
                    return dict(cached[1], stale=True)
                raise
            data = await self._provider("thumbnail", target)
            if not isinstance(data, bytes) or not 0 < len(data) <= 32768:
                raise unavailable("sandbox_thumbnail_unavailable", "No screen preview is available.", 503)
            await authorize(owner, session, allocation, generation)
            if self.closed:
                raise unavailable("sandbox_view_closed", "Sandbox viewing is unavailable.", 503)
            result = {"allocation_id": str(allocation), "generation": generation,
                      "content_type": "image/jpeg", "image": base64.b64encode(data).decode("ascii"),
                      "captured_at": datetime.now(UTC).isoformat(), "stale": False}
            self._thumbnails[key] = (time.monotonic(), result)
            while len(self._thumbnails) > 16:
                self._thumbnails.pop(next(iter(self._thumbnails)))
            return dict(result)

    def current(self, owner):
        """Current local lease only, not proof the phone displayed its screen."""
        self.access.check(owner)
        self._prune()
        return next((lease.wire() for lease in self.leases.values()
                     if lease.target.owner_id == owner), None)

    async def resolve_presentation(self, ctx, allocation_id, generation):
        target = await self.access.allocation(ctx.owner_id, ctx.session_id, allocation_id, generation)
        await self._provider("inspect", target)
        return target.generation

    async def present(self, ctx, presentations, *, task_id=None, allocation_id=None, generation=None, resume=False):
        """Trusted Hermes seam: exact existing work, existing open/claim/ACK journal."""
        if task_id is not None:
            if resume:
                raise unavailable("sandbox_resume_unavailable", "Explicit resume requires a retained allocation.", 422)
            target = await self.access.resolve(ctx.owner_id, ctx.session_id, task_id=task_id, allocation_id=allocation_id)
        else:
            target = await self._target(ctx.owner_id, ctx.session_id, allocation_id, generation, resume=resume)
        await self._provider("inspect", target)
        return await presentations.present(ctx, action="open", artifact_kind="sandbox",
                                          artifact_id=target.allocation_id, version_id=target.generation)

    async def _provider(self, method, target):
        try:
            async with asyncio.timeout(25):
                return await getattr(self.provider, method)(target.handle)
        except SandboxError as error:
            if error.code in {"sandbox_not_found", "sandbox_not_running", "sandbox_paused"}:
                raise unavailable() from None
            if error.code == "sandbox_view_access_ended":
                raise unavailable(error.code, "This sandbox's viewing time has ended.", 410) from None
            if error.code == "sandbox_view_credential_unavailable":
                raise unavailable(error.code, "This sandbox isn't ready to share its screen. Ask Jarvis to prepare it.") from None
            raise unavailable("sandbox_view_connection_failed", "The screen couldn't connect. Try again.", 503) from None
        except (TimeoutError, OSError):
            raise unavailable("sandbox_view_connection_failed", "The screen couldn't connect. Try again.", 503) from None

    def _prune(self):
        now = datetime.now(UTC)
        for identity, lease in list(self.leases.items()):
            if lease.expires_at <= now or lease.closed.is_set():
                lease.closed.set()
                self.leases.pop(identity, None)

    async def _target(self, owner, session, allocation, generation, *, resume=False):
        if resume:
            operation = getattr(self.access, "resume", None)
            if operation is None:
                raise unavailable("sandbox_resume_unavailable", "This viewer cannot resume that resource.", 422)
            return await operation(owner, session, allocation, generation)
        return await self.access.allocation(owner, session, allocation, generation)

    async def open(self, owner, session, allocation, generation, *, resume=False):
        # Bound setup, even if a client submits concurrent open requests.
        if self.closed or self._opening.locked():
            raise unavailable("sandbox_view_busy", "The screen is connecting. Try again shortly.", 503)
        async with self._opening:
            self._prune()
            target = await self._target(owner, session, allocation, generation, resume=resume)
            connection = await self._provider("open", target)
            target = await self.access.allocation(owner, session, allocation, generation)
            if self.closed:
                raise unavailable("sandbox_view_closed", "Sandbox viewing is unavailable.", 503)
            # One current screen per owner, including across phone reconnects;
            # replacement actively
            # disconnects the old view without stopping its sandbox/work.
            for lease in self.leases.values():
                if lease.target.owner_id == owner:
                    lease.closed.set()
            self._prune()
            if len(self.leases) >= 16:
                raise unavailable("sandbox_view_busy", "Too many screens are open. Try again shortly.", 503)
            expiry = min(target.deadline, connection.expires_at, datetime.now(UTC) + timedelta(minutes=10))
            if expiry <= datetime.now(UTC):
                raise unavailable("sandbox_view_access_ended", "This sandbox's viewing time has ended.", 410)
            lease = DesktopLease(uuid4(), target, expiry, secrets.token_urlsafe(32))
            self.leases[lease.id] = lease
            return lease

    def get(self, owner, session, identity):
        self._prune()
        lease = self.leases.get(identity)
        if (self.closed or lease is None or lease.target.owner_id != owner
                or lease.target.session_id != session):
            raise unavailable("sandbox_view_expired", "This screen view has ended. Open it again.", 410)
        self.access.check(owner)
        return lease

    async def validate(self, lease, *, native=True):
        current = self.get(lease.target.owner_id, lease.target.session_id, lease.id)
        target = await self.access.allocation(current.target.owner_id, current.target.session_id,
                                             current.target.allocation_id, current.target.generation)
        return await self._provider("inspect", target) if native else None

    def revoke(self, owner, session, identity):
        lease = self.leases.get(identity)
        if lease is not None and lease.target.owner_id == owner and lease.target.session_id == session:
            lease.closed.set()
            self.leases.pop(identity, None)

    def stop(self, owner, identity):
        """Owner-selected lease only; a stale stop cannot close its replacement."""
        self.access.check(owner)
        if not isinstance(identity, UUID):
            raise unavailable("sandbox_selection_required", "Choose the exact view to close.", 422)
        lease = self.leases.get(identity)
        if lease is None or lease.target.owner_id != owner:
            return False
        self.revoke(owner, lease.target.session_id, identity)
        return True

    async def close(self):
        self.closed = True
        self._thumbnails.clear()
        for lease in self.leases.values():
            lease.closed.set()
        self.leases.clear()
        await self.access.close()
