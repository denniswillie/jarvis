"""Worker-only credentials and cleanup; API admission uses public policy metadata."""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.runtime.feature_config import Capability, SandboxConfiguration
from app.sandbox.allocations import SandboxAllocationRepository
from app.sandbox.cleanup import SandboxCleaner, SandboxCleanupRepository
from app.sandbox.models import E2BConfiguration, SandboxScope
from app.tasks.models import TaskError


@dataclass(frozen=True)
class SandboxWorkerRuntime:
    provider: object
    cleaner: SandboxCleaner
    configuration: SandboxConfiguration
    enabled: bool
    owner_id: str

    @property
    def capabilities(self):
        return self.configuration.policy.required_capabilities if self.enabled else ()

    async def run(self, stop, worker_id):
        await self.cleaner.run(stop, worker_id, owner_id=self.owner_id,
                               poll_seconds=self.configuration.cleanup_poll_seconds)

    async def final_cleanup(self, worker_id):
        # Bounded shutdown; an uncertain cleanup remains durable for restart.
        async with asyncio.timeout(self.configuration.shutdown_seconds):
            while await self.cleaner.run_once(self.owner_id, worker_id) is not None:
                pass


async def load_sandbox_runtime(flags, repository, owner_id, *, database_url, key_file=None, provider_factory=None):
    resolution = flags.resolve(Capability.SANDBOX, database_url=database_url)
    if resolution.enabled and not resolution.configured:
        raise TaskError("sandbox_configuration_invalid", "Selected sandbox policy is unavailable.", 503)
    configuration = resolution.configuration if resolution.configured else SandboxConfiguration()
    async with repository.pool.acquire() as connection:
        outstanding = await connection.fetchval("""select exists(select 1 from jarvis_sandbox_allocations
            where owner_id=$1 and phase<>'cleaned' and submission_token is not null)""", owner_id)
    if not key_file:
        if resolution.enabled or outstanding:
            raise TaskError("sandbox_cleanup_credential_required",
                "The worker needs its E2B credential for selected execution or outstanding resource cleanup.", 503)
        return None
    try:
        key = Path(key_file).read_text().strip()
        credentials = E2BConfiguration(api_key=key, request_timeout_seconds=configuration.request_timeout_seconds)
    except (OSError, ValueError):
        raise TaskError("sandbox_worker_credential_unavailable", "The worker's E2B secret file is unavailable.", 503) from None
    if provider_factory is None:
        # Lazy: neither the voice/API process nor an unused worker needs the SDK.
        from app.sandbox.e2b_provider import E2BProvider
        from app.sandbox.view_secrets import SandboxViewSecrets
        view_secrets = SandboxViewSecrets.configured()
        provider_factory = lambda credentials: E2BProvider(credentials, view_secrets=view_secrets)
    provider = provider_factory(credentials)
    # Authenticated read only. No VM is created for readiness.
    await provider.discover(SandboxScope(owner_id=owner_id, task_id=uuid4(), task_revision=1,
        attempt_id=uuid4(), allocation_id=uuid4()), configuration.policy)
    allocations = SandboxAllocationRepository(repository)
    return SandboxWorkerRuntime(provider, SandboxCleaner(SandboxCleanupRepository(allocations), provider),
                                 configuration, resolution.enabled, owner_id)
