"""Opt-in scope association around the existing retained conversation bridge."""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID

from .bindings import BindingError, SessionBindings, bounded_id, identity
from .domain_scope import DomainInputScope
from .domain_scopes import DomainScopeAssociations
from .service import InputReceipt, RetainedSessionService
from .client import HermesGatewayError, HermesOutcomeUnknown
from .work import WorkDeliveries, WorkSessionBridge


@dataclass(frozen=True)
class _Dispatch:
    scope: DomainInputScope
    call_id: str
    action: str | None


_dispatch: ContextVar[_Dispatch | None] = ContextVar("jarvis_domain_dispatch", default=None)


class ScopedRetainedSessionService(RetainedSessionService):
    def __init__(self, client, bindings, *, launch, scopes: DomainScopeAssociations):
        super().__init__(client, bindings, launch=launch)
        if scopes.bindings is not bindings:
            raise BindingError("Scope associations must share the service bindings")
        self.scopes = scopes

    async def bind_input_scope(self, logical_session_id: str, *, input_id: str,
                               jarvis_session_id: UUID, allowed_tools: frozenset[str],
                               request_event_id: UUID | None = None,
                               request_revision: int | None = None) -> DomainInputScope:
        """Bind a trusted host request BEFORE send; input_id is the service-level ID."""
        bounded_id(input_id)
        scope = DomainInputScope(
            owner_id=self.bindings.owner_id, logical_session_id=logical_session_id,
            input_id=identity(logical_session_id, "input", input_id),
            jarvis_session_id=jarvis_session_id, allowed_tools=allowed_tools,
            request_event_id=request_event_id, request_revision=request_revision,
        )
        return await asyncio.to_thread(self.scopes.bind_initial, scope)

    async def create_work(self, master_id, *, request_id, label):
        dispatch = _dispatch.get()
        if dispatch is None or dispatch.scope.logical_session_id != master_id:
            raise BindingError("Scoped child creation requires the originating native input")
        reserved = await asyncio.to_thread(
            self.bindings.reserve, request_id=request_id, parent_id=master_id,
            label=label, launch=self.launch,
        )
        child_input = identity(master_id, dispatch.call_id, "work")
        await asyncio.to_thread(
            self.scopes.derive, dispatch.scope, reserved.logical_id,
            identity(reserved.logical_id, "input", child_input), kind="work",
        )
        # Scope is durable even before session.create can execute native hooks.
        return await self._materialize(reserved)

    async def send(self, session_id, *, input_id, text):
        bounded_id(input_id)
        native_input_id = identity(session_id, "input", input_id)
        dispatch = _dispatch.get()
        if dispatch is not None and dispatch.action == "message_parent":
            expected = identity(dispatch.scope.logical_session_id, dispatch.call_id, "parent")
            if input_id != expected:
                raise BindingError("Unexpected scoped parent message input")
            await asyncio.to_thread(self.scopes.derive, dispatch.scope, session_id,
                                    native_input_id, kind="parent_message")
        await asyncio.to_thread(self.scopes.get, session_id, native_input_id)
        return await super().send(session_id, input_id=input_id, text=text)

    async def steer_live_input(self, receipt, *, jarvis_session_id):
        """Trusted direct-owner Voice guidance, not a model tool or new grant.

        Native Hermes applies it to the master's active turn at delivery time,
        under that turn's UNCHANGED tool/request authority. A transfer neither
        rebinds _submission_turn nor grants the incoming input's permissions.
        Other producers keep their existing, separately scoped FIFO behavior.
        """
        binding = await self._owned(receipt.session_id)
        scope = await asyncio.to_thread(self.scopes.get, receipt.session_id, receipt.native_input_id)
        if (not binding.is_master or scope.jarvis_session_id != jarvis_session_id
                or scope.request_event_id is not None
                or receipt.native_input_id != identity(receipt.session_id, "input", receipt.input_id)):
            raise BindingError("Live guidance requires the direct owner's unversioned master input")
        return await super().steer_input(receipt.session_id, receipt.admission_id)

    async def steer_work(self, master_id, work_id, admission_id):
        # General child inputs retain their own scope; only direct owner Voice
        # opts into guidance under the active master's unchanged authority.
        await self._child_of(master_id, work_id)
        return {**await self.receipt(work_id, admission_id),
                "steering": "fifo", "immediate_steering": False}

    async def control_live_input(self, session_id, *, input_id, text, jarvis_session_id, delivery):
        """Use a native live control before admission; None means safe to queue instead.

        Guidance joins the existing turn, whose admission owns its result.
        Unknown delivery must never fall through to a second submission.
        """
        bounded_id(input_id)
        if delivery not in {"steer", "redirect"}:
            raise BindingError("Unknown native live control")
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 128 * 1024:
            raise BindingError("Input must be nonempty and within the attachment budget")
        binding = await self._owned(session_id)
        native_input = identity(session_id, "input", input_id)
        scope = await asyncio.to_thread(self.scopes.get, session_id, native_input)
        if (not binding.is_master or scope.jarvis_session_id != jarvis_session_id
                or scope.request_event_id is not None):
            raise BindingError("Live control requires the direct owner's unversioned master input")
        state = await self.client.request("session.resume", session_id=binding.native_id, omit_messages=True)
        generation = state.get("execution_generation")
        active = next((row for row in state.get("pending", ())
            if row.get("status") == "started" and not row.get("steering")
            and row.get("execution_generation") == generation), None)
        if state.get("running") is not True or type(generation) is not int or active is None:
            return None
        bounded_id(active["admission_id"])
        receipt = InputReceipt(session_id, input_id, native_input, active["admission_id"], "started")
        try:
            result = await self.client.request("session." + delivery, _operation_id=native_input,
                session_id=binding.native_id, execution_generation=generation, text=text)
        except HermesOutcomeUnknown:
            return receipt, "unconfirmed"
        except HermesGatewayError as exc:
            if exc.reason in {"stale_generation", "unsupported_control", "execution_not_ready"}:
                return None  # Native rejected before handing text to an agent.
            raise
        if result.get("status") == "rejected" and result.get("delivered") is False:
            return None
        expected = "redirected" if delivery == "redirect" else "queued"
        status = ("redirected" if delivery == "redirect" else "steering") if result.get("status") == expected else "unconfirmed"
        return receipt, status


class ScopedWorkSessionBridge(WorkSessionBridge):
    domain_scoped = True

    def __init__(self, service: ScopedRetainedSessionService, deliveries):
        super().__init__(service, deliveries)
        self.scopes = service.scopes

    async def tool(self, caller_native_id, call_id, args, *, input_id, admission_id):
        bounded_id(call_id)
        bounded_id(admission_id)
        caller = await self.service.caller(caller_native_id)
        scope = await self.scopes.scope_for_input(caller, input_id)
        scoped_call_id = identity(admission_id, call_id)
        token = _dispatch.set(_Dispatch(scope, scoped_call_id, args.get("action")))
        try:
            return await super().tool(caller_native_id, scoped_call_id, args)
        finally:
            _dispatch.reset(token)

    async def send_work(self, master_id, child_id, *, input_id, text, steer=False):
        dispatch = _dispatch.get()
        if dispatch is None or dispatch.scope.logical_session_id != master_id:
            raise BindingError("Scoped child send requires the originating native input")
        await asyncio.to_thread(self.scopes.derive, dispatch.scope, child_id,
                                identity(child_id, "input", input_id), kind="work")
        result = await super().send_work(master_id, child_id, input_id=input_id,
                                         text=text, steer=False)
        if steer:
            result.update(delivery="queued", steering="fifo", immediate_steering=False,
                          instruction="Submitted through the ordinary FIFO queue; "
                                      "not injected into an in-flight turn.")
        return result

    async def _admit(self, row):
        row = await super()._admit(row)
        source = await asyncio.to_thread(
            self.scopes.get, row["child_id"], identity(row["child_id"], "input", row["input_id"]),
        )
        wake_input = identity("work-result", row["admission_id"])
        await asyncio.to_thread(
            self.scopes.derive, source, row["parent_id"],
            identity(row["parent_id"], "input", wake_input), kind="completion",
        )
        return row


def compose_domain_work(profile_home: Path, client, *, config=None) -> ScopedWorkSessionBridge:
    """Compose only opt-in profiles; no gateway/server/process is launched here."""
    home = Path(profile_home).resolve()
    if config is None:
        config = json.loads((home / "jarvis-adapter.json").read_text())
    if config.get("domain_tools") is not True:
        raise BindingError("Domain tools require explicit profile opt-in")
    directory = home / "jarvis"
    bindings = SessionBindings(directory / "bindings.sqlite", owner_id=config["owner_id"],
                               profile_home=home)
    scopes = DomainScopeAssociations(directory / "domain-scopes.sqlite", bindings)
    service = ScopedRetainedSessionService(client, bindings, launch=config["launch"], scopes=scopes)
    deliveries = WorkDeliveries(directory / "deliveries.sqlite", bindings,
                                max_pending=config.get("max_pending", 8))
    return ScopedWorkSessionBridge(service, deliveries)
