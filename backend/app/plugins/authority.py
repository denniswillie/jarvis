"""Trusted composition over existing context, policy and task transactions.

This is not an HTTP authority, installer, readiness probe or new ledger. Callers
must retain the server-created binding on every managed route, including the
actual submission transition. A denied managed operation must not be retried as
an unbound general-terminal operation. Independently admitted general-terminal
work is outside this policy.

Borrowed connections MUST already hold the existing task-owner transaction lock
for the supplied owner (and any source-publication locks required by the context
repository). Transaction presence is checked here; advisory-lock ownership is a
caller contract. No nested transaction or competing lock is acquired.

An explicitly supplied immutable deployment baseline may initialize absent
account policy without fabricating a user event. No baseline is implicit, and
an unavailable account head never falls back to deployment defaults.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal
from uuid import UUID

from app.errors import JarvisHTTPError
from app.plugins.deployment import PluginDeploymentBaseline
from app.plugins.policy import (
    CATEGORIES, UI_NAMESPACE, PluginPolicyChange, PluginPolicyState,
    changed_policy, validate_policy,
)
from app.tasks.context_events import ContextIngress


def _owner(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError("A nonblank authenticated owner is required")


def _transaction(connection) -> None:
    if not connection.is_in_transaction():
        raise RuntimeError("Plugin authority requires the caller's owner-locked transaction")


def _denied(code: str, message: str, status: int = 403):
    return JarvisHTTPError(status, code, message)


@dataclass(frozen=True)
class ManagedPluginBinding:
    """Internal identity pinned by an existing job owner, never model authority.

    Policy revision records selection provenance. Unrelated later policy changes
    do not invalidate an unchanged enabled configuration. Inventory, readiness,
    source/account grants and ordinary task authority remain separate checks.
    """

    owner_id: str
    plugin_id: str
    configuration_id: UUID
    policy_revision: int
    deployment_baseline: str | None = None

    def __post_init__(self):
        _owner(self.owner_id)
        if not isinstance(self.plugin_id, str) or self.plugin_id not in CATEGORIES:
            raise ValueError("A supported managed plugin is required")
        if not isinstance(self.configuration_id, UUID):
            raise TypeError("A canonical configuration UUID is required")
        if type(self.policy_revision) is not int or self.policy_revision < 0:
            raise ValueError("A confirmed policy revision is required")
        if self.policy_revision == 0:
            if not isinstance(self.deployment_baseline, str) or not (1 <= len(self.deployment_baseline) <= 130):
                raise ValueError("Revision zero requires explicit trusted deployment provenance")
        elif self.deployment_baseline is not None:
            raise ValueError("Account policy selection must not claim deployment provenance")


@dataclass(frozen=True)
class SubmittedManagedJob:
    """A trusted reader's projection of an existing durable submitted job.

    Constructing this carrier does not establish submission. The reader supplied
    to require_submitted_scope must resolve the original binding and submission
    token from the owning task/action/sandbox records, not from request JSON or
    the requested binding. Prepared/queued work must return None. The job owner
    still validates its normal scope, grants and lifecycle for the requested use.
    """

    binding: ManagedPluginBinding
    job_id: UUID
    submission_token: UUID

    def __post_init__(self):
        if not isinstance(self.binding, ManagedPluginBinding):
            raise TypeError("A trusted managed binding is required")
        if not isinstance(self.job_id, UUID) or not isinstance(self.submission_token, UUID):
            raise TypeError("Existing job and submission UUIDs are required")


SubmittedJobReader = Callable[[object, str, UUID, UUID], Awaitable[SubmittedManagedJob | None]]


@dataclass(frozen=True)
class PluginMutationReceipt:
    applied_revision: int
    replayed: bool
    current: PluginPolicyState


class PluginAuthority:
    def __init__(self, repository, context_repository, tasks, *,
                 deployment_baseline: PluginDeploymentBaseline | None = None):
        if deployment_baseline is not None and not isinstance(deployment_baseline, PluginDeploymentBaseline):
            raise TypeError("Deployment defaults require an explicit trusted baseline carrier")
        self.repository = repository
        self.contexts = context_repository
        self.tasks = tasks
        self._deployment_baseline = deployment_baseline

    @property
    def deployment_baseline(self) -> PluginDeploymentBaseline | None:
        return self._deployment_baseline

    @asynccontextmanager
    async def _connection(self, owner_id, connection):
        _owner(owner_id)
        if connection is not None:
            _transaction(connection)
            yield connection
            return
        # Match ContextRepository.append's existing source/task lock ordering.
        publication = getattr(self.contexts, "publication", None)
        transaction = (publication.transaction(self.tasks, owner_id) if publication is not None
                       else self.tasks.transaction(owner_id))
        async with transaction as owned:
            _transaction(owned)
            yield owned

    async def _read(self, connection, owner_id):
        options = {"lock": True}
        if self.deployment_baseline is not None:
            options["baseline"] = self.deployment_baseline.policy
        return await self.repository.read(connection, owner_id, **options)

    async def _current(self, connection, owner_id):
        state = await self._read(connection, owner_id)
        if (not isinstance(state, PluginPolicyState) or state.policy is None
                or type(state.revision) is not int or state.revision < 0
                or (state.revision == 0 and (self.deployment_baseline is None
                    or not self.deployment_baseline.matches(state)))):
            raise _denied("plugin_policy_unavailable", "Confirmed plugin choices are unavailable.", 409)
        policy = validate_policy(state.policy)
        if policy.revision != state.revision:
            raise _denied("plugin_policy_unavailable", "Plugin policy identity is inconsistent.", 409)
        return state

    async def mutate(self, *, owner_id: str, command: PluginPolicyChange,
                     connection=None) -> PluginMutationReceipt:
        """Atomically admit the actual UI intent and append or replay its policy.

        Replay retains the original expected_revision and returns the original
        applied revision, even when the current policy is newer or unavailable.
        The caller projects current into its catalog; this receipt does not claim
        installed software, readiness, completed preparation or enabled controls.
        """
        if not isinstance(command, PluginPolicyChange):
            raise TypeError("A validated plugin UI command is required")
        event = ContextIngress(
            session_id=None, source_namespace=UI_NAMESPACE,
            source_event_id=str(command.mutation_id), receipt_id=str(command.mutation_id),
            expected_revision=0, kind="user_request", trust="user_input", finality="final",
            payload=command.ingress_payload(),
        )
        async with self._connection(owner_id, connection) as current:
            source = await self.contexts.append_in_transaction(current, owner_id, event)
            applied = await self.repository.applied(current, owner_id, command, source)
            if applied is not None:
                state = await self._read(current, owner_id)
                return PluginMutationReceipt(applied, True, state)
            previous = await self._current(current, owner_id)
            policy = changed_policy(previous.policy, command, source.id)
            state = await self.repository.append(current, owner_id, command, source, previous, policy)
            return PluginMutationReceipt(state.revision, False, state)

    async def select_new_work(self, *, owner_id: str, plugin_id: str | None = None,
                              category: str | None = None, connection=None) -> ManagedPluginBinding:
        """Select exactly one explicit plugin or remembered default; never fall back.

        Selection is not submission. Persist the returned binding with the
        existing job scope, and recheck it at the real new-work boundary.
        """
        if (plugin_id is None) == (category is None):
            raise ValueError("Select exactly one plugin or category")
        if plugin_id is not None and (not isinstance(plugin_id, str) or plugin_id not in CATEGORIES):
            raise ValueError("Unknown managed plugin")
        if category is not None and (not isinstance(category, str) or category not in CATEGORIES.values()):
            raise ValueError("Unknown managed category")
        async with self._connection(owner_id, connection) as current:
            state = await self._current(current, owner_id)
            selected = plugin_id
            if selected is None:
                selected = next((choice.plugin_id for choice in state.policy.defaults
                                 if choice.category == category), None)
            if selected is None:
                raise _denied("plugin_default_unavailable", "No confirmed managed default is selected.", 409)
            choice = next((entry for entry in state.policy.entries if entry.plugin_id == selected), None)
            if choice is None:
                raise _denied("plugin_choice_unavailable", "This managed plugin has no confirmed choice.", 409)
            if not choice.desired_enabled:
                raise _denied("plugin_disabled", "This plugin is disabled for new managed work.")
            return ManagedPluginBinding(owner_id, selected, choice.configuration_id, state.revision,
                self.deployment_baseline.identity if state.revision == 0 else None)

    @staticmethod
    def _binding(owner_id, binding):
        _owner(owner_id)
        if not isinstance(binding, ManagedPluginBinding):
            raise _denied("plugin_binding_required", "Managed work must retain its trusted plugin binding.")
        if binding.owner_id != owner_id:
            raise _denied("plugin_binding_owner_mismatch", "The managed binding belongs to another owner.")

    async def require_new_work(self, connection, *, owner_id: str,
                               binding: ManagedPluginBinding) -> ManagedPluginBinding:
        """Fence new jobs, queued work, selection, reinstall and preparation.

        Call on the existing owner-locked connection before minting a durable
        submission token, with no provider work inside this transaction. Do not
        use this as a blanket guard on already-submitted job continuation.
        """
        _transaction(connection)
        self._binding(owner_id, binding)
        if binding.policy_revision == 0 and (self.deployment_baseline is None
                or binding.deployment_baseline != self.deployment_baseline.identity):
            raise _denied("plugin_baseline_changed", "The managed job's deployment defaults are unavailable.", 409)
        state = await self._current(connection, owner_id)
        choice = next((entry for entry in state.policy.entries if entry.plugin_id == binding.plugin_id), None)
        if choice is None:
            raise _denied("plugin_choice_unavailable", "The managed plugin choice is unavailable.", 409)
        if not choice.desired_enabled:
            raise _denied("plugin_disabled", "This plugin is disabled for new managed work.")
        if (choice.configuration_id != binding.configuration_id
                or binding.policy_revision > state.revision):
            raise _denied("plugin_configuration_changed", "The managed job's selected configuration is no longer current.", 409)
        return binding

    def new_work_check(self, *, owner_id: str, binding: ManagedPluginBinding):
        """Return the async (connection) callback shape used by task admission."""
        self._binding(owner_id, binding)

        async def check(connection):
            return await self.require_new_work(connection, owner_id=owner_id, binding=binding)

        return check

    async def require_submitted_scope(self, connection, *, owner_id: str,
                                      binding: ManagedPluginBinding, job_id: UUID,
                                      submission_token: UUID,
                                      purpose: Literal["continue", "status", "cleanup"],
                                      read_submitted: SubmittedJobReader) -> SubmittedManagedJob:
        """Check the exact original pin without applying today's enable policy.

        This cannot admit selection, preparation, reinstall or another job. The
        trusted reader is an integration callback over existing durable records;
        no new submitted-job store is introduced here. Unknown submission denies.
        """
        _transaction(connection)
        self._binding(owner_id, binding)
        if purpose not in {"continue", "status", "cleanup"}:
            raise _denied("plugin_new_work_requires_admission", "New managed work requires current policy admission.")
        if not isinstance(job_id, UUID) or not isinstance(submission_token, UUID):
            raise _denied("plugin_submitted_scope_unavailable", "An exact existing submission is required.")
        pin = await read_submitted(connection, owner_id, job_id, submission_token)
        if (not isinstance(pin, SubmittedManagedJob) or pin.binding != binding
                or pin.job_id != job_id or pin.submission_token != submission_token):
            raise _denied("plugin_submitted_scope_unavailable", "The original submitted managed scope could not be established.")
        return pin
