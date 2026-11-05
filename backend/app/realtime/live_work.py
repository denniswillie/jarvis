"""Native Live conversation to existing Hermes work, without a routing model.

The binding selects one explicit work conversation; it never looks up 'newest
task'. Its durable task continues when audio disconnects. Only delivery and
phone-dependent commands require the live voice connection.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from app.tasks.conversation import TaskConversation
from app.tasks.live_command_exchange import LiveCommandExchange, REQUEST
from app.tasks.live_commands import LiveCommandTaskAdmission, live_command_source
from app.tasks.models import TaskError, TaskStatus
from app.voice_routes.bridge import utc_now
from app.voice_routes.live_results import LiveDelegationResult
from app.attention.quiet_delivery import assert_unsolicited_allowed


class LiveWorkRuntime:
    """API lifespan ownership for accepted work's domain RPC, not its execution."""
    def __init__(self):
        self.bindings = set()
        self.closed = False

    def bind(self, session):
        binding = LiveWorkBridge(session, self)
        self.bindings.add(binding)
        return binding

    async def close(self):
        self.closed = True
        await asyncio.gather(*(item.close() for item in tuple(self.bindings)), return_exceptions=True)


class LiveWorkBridge:
    def __init__(self, session, runtime):
        from app.realtime.live_domain_commands import LiveDomainCommands
        self.session, self.runtime = session, runtime
        self.service = session.services.background_task_service
        self.tasks = self.service.repository
        self.admission = LiveCommandTaskAdmission(self.service, work=True)
        self.mailbox = TaskConversation(self.tasks)
        self.exchange = LiveCommandExchange(self.tasks)
        self.commands = LiveDomainCommands(session.services, session.application, session.context)
        self.receipt = None
        self.route = None
        self.delegation_id = None
        self.sequence = 0
        self.seen_parts = set()
        self.pending_parts = ()
        self.forward_captions = True
        self._jobs = set()
        self._dispatching = set()
        self._outbox = asyncio.Queue(maxsize=16)
        self._started = self._detached = self._closed = False
        self.last_issue = None

    def _spawn(self, coroutine):
        job = asyncio.create_task(coroutine)
        self._jobs.add(job)
        def finished(done):
            self._jobs.discard(done)
            if not done.cancelled() and done.exception() is not None:
                self.last_issue = getattr(done.exception(), "code", "live_work_transport_failed")
                self.session.last_omission = self.last_issue
        job.add_done_callback(finished)
        return job

    async def __call__(self, packet, *, publish, is_current):
        if self._closed or self.runtime.closed or not is_current():
            raise TaskError("live_work_unavailable", "The work transport is unavailable.")
        if self.route is not None and self.route != packet.route:
            raise TaskError("live_work_route_mismatch", "Work belongs to a different audio route.")
        if self.receipt is None:
            async with self.tasks.transaction(packet.route.scope.owner_id) as connection:
                if not is_current():
                    raise TaskError("live_work_unavailable", "The audio connection changed.")
                self.receipt = await self.admission.admit(connection, packet, command_catalog=self.commands.catalog())
            self.route = packet.route
        else:
            await self.mailbox.append_live(self.receipt.task.id, packet)
        self.delegation_id = packet.window.request.delegation_id
        self.seen_parts.update(self.session.capture._key(part) for part in packet.window.parts)
        self.publish = publish
        if not self._started:
            self._started = True
            self._spawn(self._poll())
            self._spawn(self._deliver())
            if self.forward_captions:
                self._spawn(self._captions())

    def route_followups(self):
        """Use explicit routed packets, not every conversational fragment."""
        if self._started:
            raise RuntimeError("Select follow-up routing before starting work")
        self.forward_captions = False
        self.pending_parts = ()

    def offer_captions(self, parts):
        # The provider reader does no database/model work here. Coalesce the
        # current bounded window; original capture owns the complete evidence.
        if self.forward_captions:
            self.pending_parts = tuple(parts[-64:])

    def snapshot(self):
        return {"selected_work_id": str(self.receipt.task.id) if self.receipt else None,
                "last_issue": self.last_issue,
                "meaning": "This connection's explicit work conversation, not the newest account task."}

    async def _captions(self):
        while not self._closed and not self._detached:
            parts = tuple(part for part in self.pending_parts
                          if part.fragment.speaker == "user"
                          and self.session.capture._key(part) not in self.seen_parts)
            if parts and self.receipt is not None:
                observations = await self.session.capture.observations(parts)
                accepted = await self.mailbox.append_live_observations(self.receipt.task.id, self.route, observations)
                # Idle speech stays in canonical session history. Only native
                # delegation wakes idle work, not small talk or a status update.
                self.seen_parts.update(self.session.capture._key(part) for part in parts)
            await asyncio.sleep(.1)

    async def _dispatch(self, row):
        try:
            await self.exchange.dispatch(row["id"], route=self.route,
                is_current=lambda: not self._closed and not self.runtime.closed,
                execute=self.commands.execute, invocation_context=True)
        finally:
            self._dispatching.discard(row["id"])

    async def _poll(self):
        while not self._closed and not self.runtime.closed:
            rows = await self.tasks.events(self.route.scope.owner_id, self.receipt.task.id,
                                            after=self.sequence, limit=100)
            for row in rows:
                self.sequence = row["sequence"]
                kind, body = row["event_type"], row.get("content") or {}
                if kind == REQUEST:
                    if len(self._dispatching) >= 4:
                        raise TaskError("live_command_capacity", "The bounded command transport is full.")
                    self._dispatching.add(row["id"])
                    self._spawn(self._dispatch(row))
                elif (kind in {"live_voice_reply", "live_voice_notice"}
                        and row.get("availability") == "available" and row.get("payload_id") is not None):
                    # Only an explicit agent-selected message can enter Voice.
                    # Acceptance, progress and ordinary stored results stay pull-only.
                    if body.get("text") and body.get("agent_selected") is True and self.session.ready and not self._detached:
                        await self._outbox.put((row, body["text"], True, self.delegation_id))
            if self._detached:
                async with self.tasks.transaction(self.route.scope.owner_id) as connection:
                    task = await self.tasks.load(connection, self.route.scope.owner_id, self.receipt.task.id)
                if task.status not in {TaskStatus.RUNNING, TaskStatus.QUEUED} and not self._dispatching:
                    self.runtime.bindings.discard(self)
                    return
            await asyncio.sleep(.1)

    @asynccontextmanager
    async def _guard(self, result):
        if self._detached or not self.session.ready:
            raise TaskError("live_work_voice_unavailable", "The voice connection is not available.")
        async with self.session.live.scope_guard() as connection:
            if result.unsolicited:
                await assert_unsolicited_allowed(connection, self.route.scope.owner_id, origin="live_hermes_notice")
            task = await self.tasks.load(connection, self.route.scope.owner_id, result.task_id)
            source = await live_command_source(self.tasks, connection, task)
            if source is None or source.id != self.receipt.source.id:
                raise TaskError("live_work_source_changed", "The original work source changed.")
            for root in result.roots:
                if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", task.owner_id, root) != "available":
                    raise TaskError("live_work_result_withdrawn", "A result source was withdrawn.")
            self.session._source_context.observe(result.roots)
            async with self.session._source_context.context_guard(connection=connection):
                yield connection

    async def _deliver(self):
        while not self._closed and not self._detached:
            row, text, speak, delegation_id = await self._outbox.get()
            try:
                if self._detached or row is None or not self.session.ready:
                    continue
                result = LiveDelegationResult(self.route, delegation_id, self.receipt.task.id,
                    row["task_revision"], row["payload_id"], text, utc_now() + timedelta(minutes=5),
                    (self.receipt.source.payload_id, row["payload_id"]), kind="delegation_result", speak=speak,
                    unsolicited=row["event_type"] == "live_voice_notice")
                receipt = await self.publish(result, guard=self._guard)
                if not receipt.provider_received:
                    self.last_issue = receipt.reason or "live_work_delivery_unconfirmed"
            finally:
                self._outbox.task_done()

    def detach(self):
        # Audio shutdown does not cancel accepted work or an admitted effect.
        self._detached = True
        # Wake a publisher waiting on an empty queue; it must not retain the
        # disconnected session forever. The separate command poller remains.
        if not self._outbox.full():
            self._outbox.put_nowait((None, None, False, None))
        if not self._started:
            self.runtime.bindings.discard(self)

    async def close(self):
        self._closed = True
        for job in tuple(self._jobs):
            job.cancel()
        if self._jobs:
            await asyncio.gather(*tuple(self._jobs), return_exceptions=True)
        self.runtime.bindings.discard(self)
