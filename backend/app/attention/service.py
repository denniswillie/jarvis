"""Lazy durable inbox integration; provider playback is a separate capability."""
import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

from app.attention.admission_diagnostics import AdmissionDiagnostics, observe_admission, record_admission
from app.attention.repository import AttentionRepository
from app.attention.models import AttentionError
from app.attention.quiet import quiet_delivery_decision
from app.attention.tts import OpenAISpeechProvider, SpeechConfiguration
from app.migrations.runner import read_schema_readiness
from app.runtime.feature_config import AttentionConfiguration, AttentionVoiceConfiguration, PushConfiguration, Capability, get_second_brain_settings


class AttentionService:
    def __init__(self, repository):
        self.repository = repository
        self.voice_issue = "attention_voice_disabled"
        self.push = None
        self.push_issue = "push_disabled"
        self.responsibilities = None
        self._task = None
        self._maintenance_task = None
        self.maintenance_issue = "attention_maintenance_not_started"
        self._client = None
        self._deliveries = set()
        self._closed = False
        self._voice_route = "direct_tts"
        self.admission_diagnostics = AdmissionDiagnostics()

    def phone_status(self):
        if self.push is not None:
            return self.push.status()
        return {"state": "unavailable", "reason": self.push_issue,
                "inbox_available": True, "provider_access": "unverified"}

    def voice_status(self):
        available = not self._closed and self._task is not None and not self._task.done() and self.voice_issue is None
        return {"route":self._voice_route, "state":"available" if available else "unavailable",
                "reason":None if available else self.voice_issue or "attention_voice_stopped",
                "inbox_available":True, "hearing":"not_established_by_route_availability"}

    def start_voice(self, *, manager, owner, speech, config, route="direct_tts"):
        if self._closed or self._task is not None:
            raise RuntimeError("Attention voice worker already started or service closed")
        self._voice_route = route
        self.voice_issue = None
        self._task = asyncio.create_task(self._run_voice(manager, owner, speech, config))

    def start_maintenance(self, *, owner, poll_seconds=1.0):
        """Canonical recovery is independent of any speech provider or route."""
        if self._closed or self._maintenance_task is not None:
            raise RuntimeError("Attention maintenance already started or service closed")
        if not 0 < poll_seconds <= 60:
            raise ValueError("Attention maintenance polling must be bounded")
        self.maintenance_issue = None
        self._maintenance_task = asyncio.create_task(
            self._run_maintenance(owner, poll_seconds), name="attention-canonical-maintenance")

    async def _maintenance_cycle(self, owner):
        async with self.repository.tasks.transaction(owner) as connection:
            await self.repository._recover(connection, owner)
            await connection.execute("select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", owner)
        await self.enforce_owner_quiet(owner)
        if reconcile := getattr(self.repository, "reconcile_notifications", None):
            await reconcile(owner)

    async def enforce_owner_quiet(self, owner):
        """Request speech-only retirement of this owner's active notifications.

        The existing maintenance worker calls this; finalized quiet-command
        admission may also call it AFTER its preference transaction commits.
        It is not a quiet authority, event bus, new queue or native playback
        receipt. Claim/submission/writer guards remain mandatory independently.
        Direct replies are not registered notification deliveries. Already
        settled provider context and audio played/buffered elsewhere cannot be
        recalled by this method. Its return value counts stop requests only.
        """
        # Freeze identities before reading policy. A later resume/new speaker
        # must not inherit an old snapshot's interruption authority.
        candidates = tuple(delivery for delivery in self._deliveries
            if delivery.claim.owner_id == owner
            and (delivery.task is None or not delivery.task.done()))
        if not candidates:
            return 0
        unavailable = False
        try:
            async with self.repository.tasks.transaction(owner) as connection:
                decision = await quiet_delivery_decision(connection, owner, kind="unsolicited", channel="voice")
            if decision.allowed:
                return 0
            reason = decision.reason
        except asyncio.CancelledError:
            raise
        except Exception:
            # Failure to read authority is not permission to keep an autonomous
            # notification speaking. Fence known notification owners, then
            # expose the failure to maintenance/control admission without secrets.
            unavailable = True
            reason = "owner_quiet_unavailable"
        stopped = 0
        for delivery in candidates:
            if (delivery not in self._deliveries
                    or delivery.task is not None and delivery.task.done()
                    or not delivery.current()):
                continue
            # No DB transaction, owner lock or global voice/state lock spans
            # transport interruption. Each delivery owns its existing one-shot
            # clear/fence and retains the output lane until cleanup completes.
            await delivery.stop(reason)
            stopped += 1
        if unavailable:
            raise AttentionError("owner_quiet_unavailable", "Canonical quiet propagation could not be confirmed.")
        return stopped


    async def stop_dismissed_result(self, owner, receipt):
        """Post-commit stop of only the dismissed generation's current local delivery."""
        if receipt.get("durable_suppression") is not True:
            return {"status": "not_requested", "stop_requests": 0}
        intent_id, payload_id = UUID(receipt["intent_id"]), UUID(receipt["source_payload_id"])
        generation = receipt["revoked_generation"]
        candidates = tuple(delivery for delivery in self._deliveries
            if delivery.claim.owner_id == owner and delivery.claim.intent_id == intent_id
            and delivery.claim.generation == generation
            and delivery.claim.source_payload_id == payload_id)
        stopped = 0
        try:
            # No owner/task/intent lock spans transport. Existing writers remain fenced
            # independently; a local stop cannot retract already-forwarded audio.
            async with asyncio.timeout(3):
                for delivery in candidates:
                    if (delivery not in self._deliveries
                            or delivery.task is not None and delivery.task.done()
                            or not delivery.current()):
                        continue
                    await delivery.stop("task_result_dismissed")
                    stopped += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            return {"status": "unconfirmed", "stop_requests": stopped}
        return {"status": "requested" if stopped else "no_active_matching_delivery", "stop_requests": stopped}

    async def _run_maintenance(self, owner, poll_seconds):
        while not self._closed:
            try:
                await self._maintenance_cycle(owner)
                self.maintenance_issue = None
            except asyncio.CancelledError:
                raise
            except Exception:
                self.maintenance_issue = "attention_maintenance_poll_failed"
            await asyncio.sleep(poll_seconds)

    async def _run_voice(self, manager, owner, speech, config):
        worker_id = "attention:" + uuid4().hex
        while not self._closed:
            try:
                with observe_admission(self.admission_diagnostics, owner):
                    async with self.repository.pool.acquire() as connection:
                        pending = await connection.fetchval("""select exists(select 1 from jarvis_attention_intents
                            where owner_id=$1 and state='pending' and not_before<=clock_timestamp())""", owner)
                    if pending:
                        delivery = await manager.deliver_attention(owner_id=owner, repository=self.repository,
                            speech=speech, worker_id=worker_id, receipt_seconds=config.receipt_seconds)
                        if delivery is not None:
                            self._deliveries.add(delivery)
                            delivery.task.add_done_callback(lambda _task, item=delivery: self._deliveries.discard(item))
                    else:
                        record_admission("no_pending")
                self.voice_issue = None
            except asyncio.CancelledError:
                raise
            except Exception:
                self.voice_issue = "attention_voice_poll_failed"
            await asyncio.sleep(config.poll_seconds)

    async def close(self):
        self._closed = True
        if self.responsibilities is not None:
            await self.responsibilities.close()
        if self.push is not None:
            await self.push.close()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
        deliveries = tuple(self._deliveries)
        for delivery in deliveries:
            await delivery.stop("service_shutdown")
        await asyncio.gather(*(delivery.task for delivery in deliveries), return_exceptions=True)
        if self._client is not None:
            await self._client.close()


def _bind_quiet_tools(services, service):
    from app.preferences.tools import PreferenceToolModule

    for module in getattr(services, "tool_modules", ()):
        if isinstance(module, PreferenceToolModule):
            module.bind_attention_service(service)


async def initialize_attention_runtime(services):
    # Tool modules precede attention startup. Bind only the ready canonical
    # service, and leave unavailable initialization explicitly disconnected.
    _bind_quiet_tools(services, None)
    flags = get_second_brain_settings()
    resolution = flags.resolve(Capability.ATTENTION, database_url=services.settings.SUPABASE_DB_URL)
    services.attention_issue = resolution.issue or "attention_disabled"
    manager = getattr(services, "realtime_session_manager", None)
    if manager is not None:
        # Selected-but-unavailable must not silently dispatch through legacy speech.
        manager.attention_notifications_enabled = resolution.enabled
        manager.attention_repository = None
    if not resolution.enabled or not resolution.configured or services.db_pool is None:
        return
    tasks = getattr(services, "background_task_service", None)
    if tasks is None or not isinstance(resolution.configuration, AttentionConfiguration):
        services.attention_issue = "attention_tasks_unavailable"
        return
    try:
        async with services.db_pool.acquire() as connection:
            schema = await connection.fetchval("select current_schema()")
            readiness = await read_schema_readiness(connection, schema=schema)
        if not readiness.ready:
            services.attention_issue = "attention_schema_not_ready"
            return
        config = resolution.configuration
        services.attention_service = AttentionService(AttentionRepository(
            tasks.repository, lease_seconds=config.lease_seconds, max_attempts=config.max_attempts))
        _bind_quiet_tools(services, services.attention_service)
        services.attention_issue = None
        tasks.attention_service = services.attention_service
        if manager is not None:
            manager.attention_repository = services.attention_service.repository
        voice = flags.resolve(Capability.ATTENTION_VOICE, database_url=services.settings.SUPABASE_DB_URL,
                              provider_key=services.settings.OPENAI_API_KEY)
        service = services.attention_service
        service.start_maintenance(owner=services.settings.JARVIS_OWNER_ID)
        service.voice_issue = voice.issue or "attention_voice_disabled"
        lean_selected = getattr(services.settings, "LEAN_VOICE_ENABLED", False)
        if (voice.enabled and voice.configured and not lean_selected
                and isinstance(voice.configuration, AttentionVoiceConfiguration)):
            from openai import AsyncOpenAI
            config = voice.configuration
            try:
                service._client = AsyncOpenAI(api_key=config.api_key.get_secret_value(), max_retries=0)
                speech = OpenAISpeechProvider(service._client, SpeechConfiguration(model=config.model, voice=config.voice))
                service.start_voice(manager=services.realtime_session_manager,
                    owner=services.settings.JARVIS_OWNER_ID, speech=speech, config=config,
                    route="session_selected" if getattr(services.settings, "RESPONSIVE_VOICE_ENABLED", False) else "direct_tts")
            except Exception:
                service.voice_issue = "attention_voice_initialization_failed"
        if (voice.enabled and (lean_selected or getattr(services.settings, "RESPONSIVE_VOICE_ENABLED", False))
                and service._task is None):
            # Explicit voice policy still applies, but native delivery does not
            # require TTS model/configuration. The manager refuses legacy speech
            # when this selected worker has no legacy speech provider.
            service.start_voice(manager=services.realtime_session_manager,
                owner=services.settings.JARVIS_OWNER_ID, speech=None,
                config=SimpleNamespace(poll_seconds=1.0, receipt_seconds=60), route="native_voice")
        push = flags.resolve(Capability.PUSH, database_url=services.settings.SUPABASE_DB_URL)
        service.push_issue = push.issue or "push_disabled"
        if push.enabled and push.configured and isinstance(push.configuration, PushConfiguration):
            from app.attention.push_provider import APNsProvider
            from app.attention.push_repository import PushRepository
            from app.attention.push_service import PushService
            provider = None
            try:
                provider = APNsProvider(push.configuration)
                service.push = PushService(PushRepository(service.repository, push.configuration),
                    provider, manager, services.settings.JARVIS_OWNER_ID, push.configuration)
                service.push.start()
                service.push_issue = None
            except Exception:
                if provider is not None:
                    await provider.close()
                service.push = None
                service.push_issue = "push_initialization_failed"
        from app.reminders.duty_runtime import initialize_responsibilities
        service.responsibilities = await initialize_responsibilities(services, flags)
    except Exception:
        services.attention_issue = "attention_initialization_failed"
