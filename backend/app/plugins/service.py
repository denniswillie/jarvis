"""Account API composition; Core activates it only with real managed-use fences.

Authority availability alone does not enable writes. The explicit runtime gate
defaults off until admission/submission checks are connected and qualified.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.errors import JarvisHTTPError
from app.plugins.authority import PluginAuthority
from app.plugins.catalog import project_catalog
from app.plugins.deployment import PluginDeploymentBaseline
from app.plugins.policy import PluginPolicyChange, PluginPolicyState
from app.plugins.repository import PluginPolicyRepository


class PluginPolicyService:
    def __init__(self, repository: PluginPolicyRepository, *,
                 authority: PluginAuthority | None = None,
                 changes_enabled: bool = False, clock=None):
        if type(changes_enabled) is not bool:
            raise TypeError("Plugin change activation must be explicit")
        if changes_enabled and authority is None:
            raise ValueError("Plugin changes require the managed-use authority")
        self.repository = repository
        self.authority = authority
        self.changes_enabled = changes_enabled
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def _deployment_baseline(self) -> PluginDeploymentBaseline | None:
        # Runtime composition supplies this once to the authority. There is no
        # second service default or request-controlled initialization path.
        baseline = getattr(self.authority, "deployment_baseline", None)
        if baseline is not None and not isinstance(baseline, PluginDeploymentBaseline):
            raise TypeError("Deployment defaults require the shared trusted baseline carrier")
        return baseline

    def _catalog(self, state: PluginPolicyState):
        baseline = self._deployment_baseline()
        initial = baseline is not None and baseline.matches(state)
        policy = state.policy if state.revision > 0 or initial else None
        if policy is not None and policy.revision != state.revision:
            policy = None
        # Defaults are known only for an absent ledger matching Core's fixed
        # baseline. This does not establish installation/readiness or activate
        # mutations without the separate managed-use runtime gate.
        return project_catalog(policy=policy, now=self.clock(),
            changes_enabled=self.changes_enabled and self.authority is not None and policy is not None)

    async def get(self, *, owner_id: str):
        baseline = self._deployment_baseline()
        options = {"baseline": baseline.policy} if baseline is not None else {}
        async with self.repository.pool.acquire() as connection:
            state = await self.repository.read(connection, owner_id, **options)
        return self._catalog(state)

    async def update(self, *, owner_id: str, command: PluginPolicyChange):
        if not self.changes_enabled or self.authority is None:
            raise JarvisHTTPError(503, "plugin_policy_changes_unavailable",
                "Plugin changes are not available right now. Refresh to check your current choices.")
        try:
            # The authority owns the source/task lock order, canonical UI
            # ingress, replay-before-CAS and atomic policy append. No provider
            # operation or second preference store is introduced here.
            receipt = await self.authority.mutate(owner_id=owner_id, command=command)
        except JarvisHTTPError as error:
            if error.code in {"context_receipt_conflict", "context_revision_conflict",
                              "context_identity_conflict", "context_finality_conflict"}:
                raise JarvisHTTPError(409, "plugin_policy_mutation_conflict",
                    "This change identifier already refers to different choices.") from error
            if error.code in {"plugin_request_unavailable", "context_source_removed",
                              "context_dependency_unavailable"}:
                # A removed policy source is a domain conflict, not revoked
                # account authentication. The phone must not sign out for it.
                raise JarvisHTTPError(409, "plugin_policy_source_unavailable",
                    "The plugin choices changed or became unavailable. Refresh first.") from error
            raise
        return {
            "mutation_id": str(command.mutation_id),
            "applied_revision": receipt.applied_revision,
            "request_replayed": receipt.replayed,
            "catalog": self._catalog(receipt.current).model_dump(mode="json"),
        }
