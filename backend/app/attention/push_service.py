"""Phone fallback while voice is disconnected, owned by the API lifespan."""
import asyncio
import time
from uuid import uuid4

from app.attention.push_models import PushAuthorization, PushOutcome


class PushService:
    def __init__(self, repository, provider, manager, owner, configuration):
        self.repository, self.provider, self.manager = repository, provider, manager
        self.owner, self.configuration = owner, configuration
        self.worker = "push:" + uuid4().hex
        self.issue = None
        self._task = None
        self._closed = False
        self._provider_backoff = 0.0
        self._accepted_once = False

    def status(self):
        active = not self._closed and self._task is not None and not self._task.done()
        return {"state": ("running" if self.issue is None else "degraded") if active else "unavailable",
                "provider_access": "accepted_at_least_one_request" if self._accepted_once else "unverified",
                "reason": self.issue or (None if active else "push_delivery_stopped"),
                "environment": self.configuration.environment, "topic": self.configuration.topic,
                "notification_content": "generic_inbox_pointer", "delivery": "not_proven_by_provider_acceptance",
                "submission": "durable_authorization_not_wire_evidence"}

    def start(self):
        self._task = asyncio.create_task(self._run())

    async def run_once(self):
        if self._closed:
            return False
        # Leases must recover even after voice reconnects or a provider backoff.
        await self.repository.recover(self.owner)
        if time.monotonic() < self._provider_backoff or await self.manager.has_connected_voice(self.owner):
            return False
        claim = await self.repository.claim_next(self.owner, self.worker)
        if claim is None:
            return False
        submitted = False
        try:
            if await self.manager.has_connected_voice(self.owner):
                outcome = PushOutcome("failed", "voice_reconnected", retry_seconds=1)
            else:
                request = await self.repository.prepare_submission(claim)
                if request is None:
                    outcome = PushOutcome("revoked", "push_admission_revoked")
                else:
                    async def authorize(candidate):
                        nonlocal submitted
                        if candidate != request:
                            return PushAuthorization(candidate.claim, False, "push_request_changed")
                        if await self.manager.has_connected_voice(self.owner):
                            return PushAuthorization(candidate.claim, False, "voice_reconnected")
                        decision = await self.repository.authorize_send(candidate)
                        submitted = submitted or decision.allowed
                        return decision
                    async with asyncio.timeout(self.configuration.http_timeout_seconds):
                        outcome = await self.provider.send(request, authorize=authorize)
            outcome = await self.repository.finish(claim, outcome)
            self.issue = outcome.reason if outcome.state in {"unknown", "failed"} else None
            self._accepted_once = self._accepted_once or outcome.state == "accepted"
            if outcome.provider_status in {403, 429} or (
                outcome.provider_status is not None and outcome.provider_status >= 500):
                self._provider_backoff = time.monotonic() + (outcome.retry_seconds or 900)
            return True
        except asyncio.CancelledError:
            await self.repository.finish(claim, PushOutcome("unknown" if submitted else "failed",
                "push_service_stopped", retry_seconds=None if submitted else 1))
            raise
        except Exception:
            # Never turn a possibly transmitted request into an automatic retry.
            outcome = await self.repository.finish(claim, PushOutcome("unknown" if submitted else "failed",
                "push_submission_uncertain" if submitted else "push_preparation_failed",
                retry_seconds=None if submitted else 60))
            self.issue = outcome.reason
            return False

    async def _run(self):
        while not self._closed:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.issue = "push_poll_failed"
            await asyncio.sleep(self.configuration.poll_seconds)

    async def close(self):
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.provider.close()
