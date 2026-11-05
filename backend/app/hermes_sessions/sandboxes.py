"""Lazy retained sandboxes. Native Hermes owns commands and conversations.

All session IDs here are trusted logical bindings resolved by the native session
adapter, never a model-selected owner/profile or a native terminal cache key.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.hermes_sessions.sandbox_bindings import SessionSandboxBindings
from app.sandbox.models import SandboxError, SandboxPolicy


class RetainedSandboxService:
    def __init__(self, bindings: SessionSandboxBindings, provider, *, policy: SandboxPolicy | None = None):
        # Retained agent computers are viewable by default. Keep the global
        # task policy unchanged and honor an explicitly selected template.
        if policy is None:
            policy = SandboxPolicy(template="desktop")
        elif "template" not in policy.model_fields_set:
            policy = policy.model_copy(update={"template": "desktop"})
        self.bindings, self.provider, self.policy = bindings, provider, policy
        self._locks: dict[str, asyncio.Lock] = {}
        self.on_ready = None

    def _lock(self, session_id: str):
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _required(self, session_id):
        binding = self.bindings.get(session_id)
        if binding is None:
            raise SandboxError("session_sandbox_not_allocated")
        # A desktop default applies only to new resources. Existing terminal
        # VMs retain their exact policy, metadata and identity. Other policy
        # changes still require explicit reconciliation, not silent widening.
        legacy_terminal = (self.policy.template == "desktop" and
                           binding.policy == self.policy.model_copy(update={"template": "base"}))
        if binding.policy != self.policy and not legacy_terminal:
            raise SandboxError("session_sandbox_policy_changed")
        return binding

    @staticmethod
    def _accept(binding, info):
        if (info.scope != binding.scope or info.policy != binding.policy
                or (binding.sandbox_id is not None and info.sandbox_id != binding.sandbox_id)):
            raise SandboxError("session_sandbox_identity_mismatch")
        return info

    async def _existing(self, binding):
        if binding.sandbox_id is None:
            if not binding.creation_started:
                raise SandboxError("session_sandbox_not_allocated")
            matches = await self.provider.discover(binding.scope, binding.policy)
            if len(matches) != 1:
                # Even an empty list is not permission to repeat an uncertain
                # create. A delayed provider acknowledgement may still arrive.
                raise SandboxError("session_sandbox_creation_unknown" if not matches
                                   else "session_sandbox_creation_ambiguous")
            info = self._accept(binding, await self.provider.info(matches[0], binding.scope, binding.policy))
            self.bindings.bind(binding, info.sandbox_id)
            return self._required(binding.scope.logical_session_id), info
        info = await self.provider.info(binding.sandbox_id, binding.scope, binding.policy)
        return binding, self._accept(binding, info)

    async def _ready(self, binding, info):
        if info.state != "running":
            raise SandboxError("session_sandbox_paused")
        if not binding.prepared:
            # Existing SDK bootstrap is marker-idempotent. A failed prepare
            # must never discard the allocation or allocate a replacement VM.
            await self.provider.prepare(info)
            self.bindings.mark_prepared(binding)
        if self.on_ready is not None:
            await self.on_ready(info)
        return info

    async def _ensure(self, session_id):
        binding = (self._required(session_id) if self.bindings.get(session_id) is not None
                   else self.bindings.reserve(session_id, self.policy))
        if binding.pending_action is not None:
            raise SandboxError("session_sandbox_lifecycle_pending")
        if not binding.creation_started:
            self.bindings.start_creation(binding)
            info = self._accept(binding, await self.provider.create(binding.scope, binding.policy))
            self.bindings.bind(binding, info.sandbox_id)
            binding = self._required(session_id)
        else:
            binding, info = await self._existing(binding)
        return await self._ready(binding, info)

    async def ensure(self, session_id: str):
        """Allocate once on first use; do not silently resume an idle sandbox."""
        async with self._lock(session_id):
            return await self._ensure(session_id)

    async def inspect(self, session_id: str):
        """Observe the provider without allocating, reconnecting or resuming."""
        async with self._lock(session_id):
            _, info = await self._existing(self._required(session_id))
            return info

    @asynccontextmanager
    async def use(self, session_id: str):
        """Fence a short command launch/poll against lifecycle operations.

        Release this context after launch, not after a background program ends.
        Both native process polling and future maintenance use this same seam.
        """
        async with self._lock(session_id):
            yield await self._ensure(session_id)

    async def _lifecycle(self, session_id, action, *, recover=False):
        binding, info = await self._existing(self._required(session_id))
        if recover:
            action = binding.pending_action
            if action is None:
                return info
            generation = binding.generation
        else:
            generation = self.bindings.intend(binding, action)
        if action == "pause":
            if info.state != "paused":
                info = self._accept(binding, await self.provider.pause(info))
            if info.state != "paused":
                raise SandboxError("session_sandbox_pause_unconfirmed")
        elif action == "resume":
            if info.state != "running":
                info = self._accept(binding, await self.provider.resume(info, timeout_seconds=binding.policy.lifetime_seconds))
            if info.state != "running":
                raise SandboxError("session_sandbox_resume_unconfirmed")
            await self._ready(binding, info)
        else:
            raise SandboxError("session_sandbox_action_invalid")
        self.bindings.settle(session_id, generation)
        return info

    async def pause(self, session_id: str):
        async with self._lock(session_id):
            return await self._lifecycle(session_id, "pause")

    async def resume(self, session_id: str):
        async with self._lock(session_id):
            return await self._lifecycle(session_id, "resume")

    async def recover(self, session_id: str):
        """Finish only an already requested lifecycle action, never rerun work."""
        async with self._lock(session_id):
            return await self._lifecycle(session_id, None, recover=True)

    async def micro_pause(self, session_id: str):
        """Pause/resume one running VM under a single execution fence.

        A caller must already have master-authorized keepalive policy. Persist
        the intended resume before pausing, so recovery never leaves a sandbox
        paused merely because the backend stopped between the provider calls.
        This method neither installs a timer nor notifies a model.
        """
        async with self._lock(session_id):
            binding, info = await self._existing(self._required(session_id))
            if binding.pending_action is not None:
                raise SandboxError("session_sandbox_lifecycle_pending")
            if info.state != "running":
                raise SandboxError("session_sandbox_paused")
            generation = self.bindings.intend(binding, "resume")
            paused = self._accept(binding, await self.provider.pause(info))
            if paused.state != "paused":
                raise SandboxError("session_sandbox_pause_unconfirmed")
            resumed = self._accept(binding, await self.provider.resume(
                paused, timeout_seconds=binding.policy.lifetime_seconds))
            if resumed.state != "running":
                raise SandboxError("session_sandbox_resume_unconfirmed")
            await self._ready(binding, resumed)
            self.bindings.settle(session_id, generation)
            return resumed
