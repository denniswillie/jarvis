"""Retained master/work conversations over native Hermes admission and history.

Trusted composition supplies a private profile, owner binding store and launch
policy. This is not an HTTP endpoint or a model-facing arbitrary RPC interface.
Domain authorization and original-transcript attachment remain caller-owned.
"""

import asyncio
from dataclasses import dataclass
import json

from .bindings import BindingError, SessionBinding, SessionBindings, bounded_id, identity
from .client import HermesGatewayClient, HermesGatewayError


@dataclass(frozen=True)
class InputReceipt:
    session_id: str
    input_id: str
    native_input_id: str
    admission_id: str
    status: str


class RetainedSessionService:
    def __init__(self, client: HermesGatewayClient, bindings: SessionBindings, *, launch: dict):
        if client.profile_home != bindings.profile_home:
            raise BindingError("Client and owner bindings must name the same profile")
        # The model never controls provider credentials, cwd, toolsets or profile.
        if set(launch) - {"source", "cwd", "model", "toolsets", "reasoning", "max_turns", "ignore_rules"}:
            raise BindingError("Unsupported retained-session launch field")
        self.client, self.bindings = client, bindings
        self.launch = json.loads(json.dumps(launch, allow_nan=False))
        self._attached: set[tuple[int, str]] = set()

    async def _materialize(self, binding: SessionBinding) -> SessionBinding:
        if binding.native_id is None:
            created = await self.client.request("session.create", request_id=binding.logical_id,
                                                **json.loads(binding.launch_json))
            native_id = created.get("session_id")
            bounded_id(native_id)
            binding = await asyncio.to_thread(self.bindings.bind, binding.logical_id, native_id)
            self._attached.add((self.client.connection_generation, native_id))
        else:
            await self._attach(binding)
        return binding

    async def _attach(self, binding: SessionBinding):
        if binding.native_id is None:
            raise BindingError("Conversation creation has not been reconciled")
        key = (self.client.connection_generation, binding.native_id)
        if key not in self._attached:
            snapshot = await self.client.snapshot(binding.native_id)
            if snapshot.get("session_id") != binding.native_id:
                raise HermesGatewayError("retained_identity_mismatch")
            self._attached = {entry for entry in self._attached if entry[0] == key[0]}
            self._attached.add(key)

    async def master(self) -> SessionBinding:
        reserved = await asyncio.to_thread(self.bindings.get_or_reserve_master, launch=self.launch)
        return await self._materialize(reserved)

    async def create_work(self, master_id: str, *, request_id: str, label: str) -> SessionBinding:
        reserved = await asyncio.to_thread(self.bindings.reserve, request_id=request_id, parent_id=master_id,
                                          label=label, launch=self.launch)
        return await self._materialize(reserved)

    async def _owned(self, session_id: str) -> SessionBinding:
        binding = await asyncio.to_thread(self.bindings.get, session_id)
        await self._attach(binding)
        return binding

    async def caller(self, native_session_id: str) -> SessionBinding:
        """Resolve a trusted runtime caller, including its physical compaction tip."""
        bounded_id(native_session_id)
        snapshot = await self.client.snapshot(native_session_id)
        binding = await asyncio.to_thread(self.bindings.by_native, snapshot.get("session_id"))
        self._attached.add((self.client.connection_generation, binding.native_id))
        return binding

    async def steer_work(self, master_id: str, work_id: str, admission_id: str) -> dict:
        """Transfer an already-admitted follow-up using the native busy control.

        A normal race leaves the same input queued or already running, never
        interrupts the current turn or fabricates another admission.
        """
        await self._child_of(master_id, work_id)
        return await self.steer_input(work_id, admission_id)

    async def steer_input(self, session_id: str, admission_id: str) -> dict:
        """Transfer an existing admission to its session's running native turn."""
        bounded_id(admission_id)
        binding = await self._owned(session_id)
        try:
            return await self.client.request("prompt.steer_queued", session_id=binding.native_id,
                                             admission_id=admission_id)
        except HermesGatewayError as exc:
            if exc.reason not in {"admission_started", "session_idle", "steer_unavailable", "unsupported_control"}:
                raise
            return await self.receipt(session_id, admission_id)

    async def send(self, session_id: str, *, input_id: str, text: str) -> InputReceipt:
        bounded_id(input_id)
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 128 * 1024:
            raise BindingError("Input must be nonempty and within the attachment budget")
        binding = await self._owned(session_id)
        native_input_id = identity(session_id, "input", input_id)
        # Native admission performs payload conflict checks and durable deduplication.
        # A lost reply is not retried here. Caller retains input_id and original text.
        reply = await self.client.submit(binding.native_id, native_input_id, text)
        admission_id, status = reply.get("admission_id"), reply.get("status")
        bounded_id(admission_id)
        if not isinstance(status, str) or not status:
            raise HermesGatewayError("invalid_admission_receipt")
        return InputReceipt(session_id, input_id, native_input_id, admission_id, status)

    async def send_to_work(self, master_id: str, work_id: str, *, input_id: str, text: str) -> InputReceipt:
        await self._child_of(master_id, work_id)
        return await self.send(work_id, input_id=input_id, text=text)

    async def _child_of(self, master_id: str, work_id: str) -> SessionBinding:
        master = await asyncio.to_thread(self.bindings.get, master_id)
        child = await asyncio.to_thread(self.bindings.get, work_id)
        if not master.is_master or child.parent_id != master.logical_id:
            raise BindingError("Work conversation does not belong to this master")
        return child

    async def receipt(self, session_id: str, admission_id: str) -> dict:
        bounded_id(admission_id)
        binding = await self._owned(session_id)
        return await self.client.request("prompt.receipt", session_id=binding.native_id,
                                         admission_id=admission_id, include_result=True)

    async def stop_work(self, master_id: str, work_id: str, admission_id: str) -> dict:
        bounded_id(admission_id)
        binding = await self._child_of(master_id, work_id)
        await self._attach(binding)
        # Cancel this exact native admission, not a sibling, future turn or sandbox.
        return await self.client.request("prompt.cancel", session_id=binding.native_id, admission_id=admission_id)
