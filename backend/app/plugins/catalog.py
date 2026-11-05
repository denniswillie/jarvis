"""Pure owner-scoped catalog projection; performs no I/O or authority writes."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from app.plugins.models import (
    InventoryObservation, LifecycleObservation, OperationView, PluginCard,
    PluginCatalog, PolicySnapshot, ReadinessObservation,
)


DESCRIPTORS = (
    ("video_editor", "Video Editor", "Create and edit videos", "video_editing"),
    ("browser", "Browser", "Browse websites and find information", "web_browsing"),
    ("files_data", "Files & data", "Work with files, tables, and calculations", "files_data"),
)


def _index(model, observations):
    result = {}
    for value in observations:
        item = model.model_validate(value)
        if item.plugin_id in result:
            raise ValueError("Multiple observations for one plugin are ambiguous")
        result[item.plugin_id] = item
    return result


def _operation(item, now):
    if not item.current_at(now):
        return OperationView(operation_ref=item.operation_ref, configuration_id=item.configuration_id,
            phase="unknown", reason_code="operation_status_stale",
            recovery=item.recovery if item.recovery is not None and item.recovery.kind == "recheck" else None)
    return OperationView(operation_ref=item.operation_ref, configuration_id=item.configuration_id,
        phase=item.phase, reason_code=item.reason_code, recovery=item.recovery)


def project_catalog(*, policy: PolicySnapshot | None, inventory: Sequence[InventoryObservation] = (),
                    readiness: Sequence[ReadinessObservation] = (),
                    operations: Sequence[LifecycleObservation] = (), now: datetime,
                    changes_enabled: bool = False) -> PluginCatalog:
    """Project explicit scoped facts; unknown facts are never an enabled default.

    The caller must authenticate scope and obtain current authoritative inputs.
    Ready means the observed bundle can start supported work; it is never a
    source/account/task grant. Mutation activation must come explicitly from
    the runtime after actual enforcement integration; it defaults unavailable.
    """
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("An offset-aware evaluation time is required")
    if type(changes_enabled) is not bool:
        raise TypeError("Plugin change activation must be explicit")
    snapshot = None if policy is None else PolicySnapshot.model_validate(policy)
    choices = {} if snapshot is None else {item.plugin_id: item for item in snapshot.entries}
    defaults = {} if snapshot is None else {item.category: item.plugin_id for item in snapshot.defaults}
    categories = {item[0]: item[3] for item in DESCRIPTORS}
    for category, plugin in defaults.items():
        choice = choices.get(plugin)
        if choice is None or categories[plugin] != category:
            raise ValueError("A preferred default must name a known compatible plugin")
    installed = _index(InventoryObservation, inventory)
    qualified = _index(ReadinessObservation, readiness)
    lifecycle = _index(LifecycleObservation, operations)
    cards = []
    for plugin, title, description, category in DESCRIPTORS:
        choice = choices.get(plugin)
        observed = installed.get(plugin)
        ready = qualified.get(plugin)
        operation = lifecycle.get(plugin)
        config = None if choice is None else choice.configuration_id
        # Facts are meaningful only for this exact current plugin configuration.
        observed = observed if observed is not None and observed.configuration_id == config and observed.current_at(now) else None
        ready = ready if ready is not None and ready.configuration_id == config and ready.current_at(now) else None
        operation = operation if operation is not None and operation.configuration_id == config else None
        operation_view = _operation(operation, now) if operation is not None else None
        # This expires observations, not the remembered user preference. A
        # disconnected phone must not keep showing an expired ready status.
        deadlines = [item.expires_at for item in (observed, ready, operation)
                     if item is not None and item.current_at(now)]
        state = "unknown"
        reason = "plugin_policy_unavailable" if snapshot is None else "plugin_choice_unavailable"
        if choice is not None:
            if not choice.desired_enabled:
                state, reason = "unavailable", "plugin_disabled"
            elif operation_view is not None:
                if operation_view.phase in {"preparing", "installing", "removing"}:
                    state, reason = "unavailable", "plugin_change_pending"
                elif operation_view.phase == "failed":
                    state, reason = "unavailable", operation_view.reason_code or "plugin_change_failed"
                else:
                    state, reason = "unknown", operation_view.reason_code or "plugin_change_unknown"
            elif observed is None or observed.state == "unknown":
                reason = "plugin_inventory_unknown"
            elif observed.state == "absent":
                state, reason = "unavailable", "plugin_not_installed"
            elif ready is None:
                reason = "plugin_readiness_unknown"
            elif ready.state != "ready":
                state, reason = ready.state, ready.reason_code or "plugin_readiness_unknown"
            elif ready.bundle_sha256 != observed.bundle_sha256:
                reason = "plugin_bundle_mismatch"
            else:
                state, reason = "ready", None
        cards.append(PluginCard(plugin_id=plugin, title=title, description=description, category=category,
            selected_default=defaults.get(category) == plugin,
            desired_enabled=None if choice is None else choice.desired_enabled, configuration_id=config,
            installed="unknown" if observed is None else observed.state,
            installed_version=None if observed is None else observed.bundle_version,
            readiness=state, reason_code=reason,
            observations_valid_until=min(deadlines) if deadlines else None, operation=operation_view,
            can_change=changes_enabled and choice is not None))
    return PluginCatalog(evaluated_at=now, policy_revision=None if snapshot is None else snapshot.revision,
        policy_available=snapshot is not None, items=tuple(cards))
