"""One profile's composition of existing native, resource and delivery services.

The native gateway process is provisioned/started separately. This owner opens
the private tool socket before waiting for it, and reuses the existing reconnecting
host. It never owns model execution, a job registry or sandbox deletion.
"""
import asyncio
from pathlib import Path
from uuid import uuid4

from .background_events import BackgroundEvents, background_wake_router
from .client import HermesGatewayClient, HermesGatewayError
from .domain_transport import DomainToolServer
from .host import RetainedProfileHost
from .plugin import compose
from .sandbox_bindings import SessionSandboxBindings
from .sandbox_callbacks import SandboxCallbacks
from .sandbox_controls import NativeMasterNoticeDelivery, SandboxControls
from .sandbox_keepalive import KeepaliveJournal, SandboxKeepalive
from .sandboxes import RetainedSandboxService
from .terminal_adapter import RetainedTerminalAdapter


class RetainedProfileRuntime:
    def __init__(self, profile_home, *, provider, sandbox_policy, callback_endpoint,
                 signing_key, domain_adapter=None, prepare_event_scope=None,
                 client_factory=HermesGatewayClient, bridge_factory=compose):
        """Trusted composition only; signing_key must persist across restarts.

        For domain-enabled profiles, supply the existing domain adapter and an
        authority-stripped scope callback(session_id, native_input_id, event).
        The caller applies BACKGROUND_INSTRUCTIONS to the provisioned profile;
        importing/constructing this object does not modify native prompts.
        """
        self.home = Path(profile_home).resolve()
        self.client = client_factory(self.home)
        self.bridge = bridge_factory(self.home, self.client)
        if hasattr(self.bridge.service, "scopes") and (domain_adapter is None or prepare_event_scope is None):
            raise ValueError("Domain-enabled runtime needs its adapter and event scope preparation")
        self.resources = SessionSandboxBindings(self.home / "jarvis" / "sandboxes.sqlite",
            owner_id=self.bridge.service.bindings.owner_id, profile_id=str(self.home))
        try:
            self.events = BackgroundEvents(sessions=self.bridge.service, sandbox_bindings=self.resources,
                signing_key=signing_key, prepare_scope=prepare_event_scope)
            self.callbacks = SandboxCallbacks(events=self.events, endpoint=callback_endpoint)
            self.sandboxes = RetainedSandboxService(self.resources, provider, policy=sandbox_policy)
            self.terminal = RetainedTerminalAdapter(sessions=self.bridge.service,
                sandboxes=self.sandboxes, callbacks=self.callbacks)
            self.transport = DomainToolServer(self.home, domain_adapter, terminal=self.terminal, timeout=120)
            self.journal = KeepaliveJournal(self.home / "jarvis" / "keepalive.sqlite", bindings=self.resources)
        except BaseException:
            self.resources.close()
            raise
        self.host = RetainedProfileHost(self.home, client_factory=lambda home: self.client,
                                        bridge_factory=lambda home, client: self.bridge)
        self.prepare_event_scope = prepare_event_scope
        # Snapshot only preexisting work. A new empty profile receives no model input.
        self._recovery_targets = set(self.resources.bound_session_ids())
        for row in self.bridge.deliveries.pending():
            self._recovery_targets.update((row["child_id"], row["parent_id"]))
        self.recovery_id = "backend-" + uuid4().hex
        self.recovery_errors = {}
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._used = False
        self.master = self.keepalive = None

    def current_events(self):
        return self.events if self._ready.is_set() and self.host.ready else None

    def router(self):
        return background_wake_router(self.current_events)

    async def wait_ready(self, *, timeout=60):
        async with asyncio.timeout(timeout):
            await self._ready.wait()

    async def send(self, *, input_id, text):
        await self.wait_ready(timeout=10)
        return await self.host.send(input_id=input_id, text=text)

    async def retry_recovery(self):
        """One bounded pass, also callable after a reported startup failure.

        Retry identity remains stable for this runtime. Failures stay visible;
        no perpetual recovery poller or command retry mechanism is introduced.
        """
        for target in sorted(self._recovery_targets):
            if self._stop.is_set():
                return
            try:
                await self.events.restart_notice(session_id=target, recovery_id=self.recovery_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.recovery_errors[target] = getattr(error, "reason", type(error).__name__)
            else:
                self._recovery_targets.discard(target)
                self.recovery_errors.pop(target, None)

    async def run(self, stop: asyncio.Event):
        if self._used:
            raise RuntimeError("A profile runtime has one lifecycle; construct a new owner after close")
        self._used = True
        tasks = []
        try:
            await self.transport.start()
            host_task = asyncio.create_task(self.host.run(self._stop))
            tasks.append(host_task)
            await self.host.wait_ready(timeout=60)
            self.master = await self.bridge.service.master()

            async def prepare_master_scope(*, master_session_id, native_input_id, notice):
                return await self.prepare_event_scope(session_id=master_session_id,
                    native_input_id=native_input_id, event=notice)

            notice = NativeMasterNoticeDelivery(sessions=self.bridge.service,
                master_session_id=self.master.logical_id,
                prepare_scope=prepare_master_scope if self.prepare_event_scope is not None else None)
            self.keepalive = SandboxKeepalive(master_session_id=self.master.logical_id,
                sandboxes=self.sandboxes, journal=self.journal, deliver=notice.deliver)
            self.transport.install_sandbox_controls(SandboxControls(sessions=self.bridge.service,
                sandboxes=self.sandboxes, keepalive=self.keepalive))
            maintenance = asyncio.create_task(self.keepalive.run(self._stop))
            tasks.extend((maintenance, asyncio.create_task(self.retry_recovery())))
            self._ready.set()
            stopped = asyncio.create_task(stop.wait())
            tasks.append(stopped)
            done, _ = await asyncio.wait((host_task, maintenance, stopped), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stopped:
                    await task
                    raise RuntimeError("A required profile service stopped unexpectedly")
        finally:
            self._ready.clear()
            self._stop.set()
            # Closing transports does not cancel native turns or kill/pause VMs.
            # Their existing preservation timeout applies if this backend is down.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.transport.close()
            finally:
                self.journal.close()
                self.resources.close()
