"""One reconnecting delivery owner for an already provisioned native profile.

This process never starts a model or sandbox just because time elapsed. Native
Hermes owns execution; stopping/restarting this host only detaches its transport.
Profile provisioning and native gateway process management are separate owners.
"""

import argparse
import asyncio
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal

from .client import HermesGatewayClient, HermesGatewayError
from .plugin import compose


class HostAlreadyRunning(RuntimeError):
    pass


class RetainedProfileHost:
    def __init__(self, profile_home: Path, *, poll_seconds: float = 0.5, retry_seconds: float = 1,
                 client_factory=HermesGatewayClient, bridge_factory=compose):
        if not 0.02 <= poll_seconds <= 30 or not 0.02 <= retry_seconds <= 30:
            raise ValueError("Invalid profile-host timing bounds")
        self.home = Path(profile_home).resolve()
        self.poll_seconds, self.retry_seconds = poll_seconds, retry_seconds
        self.client_factory, self.bridge_factory = client_factory, bridge_factory
        self._ready = asyncio.Event()
        self._bridge = None
        self._running = False
        self.last_error = None
        self.connections = 0
        self.on_pump = None

    @contextmanager
    def _owner(self):
        directory = self.home / "jarvis"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(directory / "delivery-host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise HostAlreadyRunning("This native profile already has a delivery host") from None
            yield
        finally:
            os.close(fd)

    async def wait_ready(self, *, timeout: float = 10):
        async with asyncio.timeout(timeout):
            await self._ready.wait()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    async def send(self, *, input_id: str, text: str):
        """Admit original input; never wait for a model answer or auto-resubmit."""
        await self.wait_ready()
        bridge = self._bridge
        if bridge is None:
            raise HermesGatewayError("host_reconnecting")
        master = await bridge.service.master()
        return await bridge.service.send(master.logical_id, input_id=input_id, text=text)

    async def _wait(self, client, stop, sequence):
        changed = asyncio.create_task(client.wait_for_change(sequence))
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({changed, stopping}, timeout=self.poll_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if changed in done and not stop.is_set():
                await changed
                # Coalesce token/event bursts. This is ordinary delivery I/O,
                # not another inference or a conversation turn-taking delay.
                try:
                    await asyncio.wait_for(stop.wait(), timeout=min(0.05, self.poll_seconds))
                except TimeoutError:
                    pass
        finally:
            changed.cancel()
            stopping.cancel()
            await asyncio.gather(changed, stopping, return_exceptions=True)

    async def run(self, stop: asyncio.Event):
        if self._running:
            raise HostAlreadyRunning("This host is already running")
        self._running = True
        try:
            with self._owner():
                while not stop.is_set():
                    try:
                        async with self.client_factory(self.home) as client:
                            bridge = self.bridge_factory(self.home, client)
                            self._bridge = bridge
                            self.connections += 1
                            self.last_error = None
                            self._ready.set()
                            while not stop.is_set():
                                sequence = client.change_sequence
                                delivered = await bridge.pump()
                                if self.on_pump is not None:
                                    await self.on_pump(bridge, delivered)
                                await self._wait(client, stop, sequence)
                    except HermesGatewayError as exc:
                        self.last_error = exc.reason[:120]
                    except OSError:
                        # Storage/transport recovery retries preserve the same
                        # durable intent and native identity. No raw error text.
                        self.last_error = "host_io_unavailable"
                    finally:
                        self._bridge = None
                        self._ready.clear()
                    if not stop.is_set():
                        try:
                            await asyncio.wait_for(stop.wait(), timeout=self.retry_seconds)
                        except TimeoutError:
                            pass
        finally:
            self._running = False
            self._bridge = None
            self._ready.clear()


async def _main(home):
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, stop.set)
    await RetainedProfileHost(home).run(stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-home", required=True, type=Path)
    asyncio.run(_main(parser.parse_args().profile_home))
