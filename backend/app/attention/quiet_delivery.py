"""Canonical quiet gates for legacy response and Live callback writers.

Origins are supplied by trusted runtime dispatch, never a caption, model argument
or mutable prompt flag. An observation-backed Live readonly callback is unsolicited:
it is not a confirmed user-turn receipt and cannot grant mutation authority.
"""
from contextlib import asynccontextmanager

from app.attention.models import AttentionError
from app.attention.quiet import OWNER_SHARED_LOCK, quiet_delivery_decision


UNSOLICITED_ORIGINS = frozenset({
    "visual_interrupt", "reminder", "prospective", "live_readonly_result", "live_hermes_notice",
})


class QuietDeliverySuppressed(AttentionError):
    def __init__(self, reason, origin):
        super().__init__(reason, "Canonical owner quiet policy suppressed unsolicited delivery.")
        self.origin = origin


def unsolicited_response_origin(kind, metadata=None):
    """Follow trusted continuation lineage, not just the current response kind."""
    value = (metadata or {}).get("jarvis_delivery_kind", kind)
    return value if value in UNSOLICITED_ORIGINS else None


async def assert_unsolicited_allowed(connection, owner, *, origin, channel="voice"):
    if origin not in UNSOLICITED_ORIGINS:
        raise ValueError("An explicit trusted unsolicited origin is required.")
    try:
        decision = await quiet_delivery_decision(connection, owner, kind="unsolicited", channel=channel)
    except Exception:
        raise QuietDeliverySuppressed("owner_quiet_unavailable", origin) from None
    if not decision.allowed:
        raise QuietDeliverySuppressed(decision.reason, origin)
    return decision


@asynccontextmanager
async def quiet_response_guard(sources, *, origin, source_guard=None):
    """Hold shared owner exclusion through a bounded actual writer.

    Borrow the same transaction for source validation: do not acquire a second
    connection/owner lock after copied source locks. Ordinary solicited responses
    never enter this guard. Existing writer timeouts remain authoritative.
    """
    @asynccontextmanager
    async def checked(connection):
        # Validate transaction ownership before any SQL, then retain it through
        # quiet policy, source admission and the actual bounded provider writer.
        async with sources.use_transaction(connection):
            # This path only reads policy. Never upgrade a borrowed shared
            # source guard: concurrent upgrades could deadlock each other.
            await connection.execute(OWNER_SHARED_LOCK, sources.owner_id)
            await assert_unsolicited_allowed(connection, sources.owner_id, origin=origin)
            if source_guard is None:
                yield connection
            else:
                async with source_guard():
                    yield connection

    # ContextVars propagate to children; this hook accepts only the exact task
    # that borrowed the connection. Never reuse a parent's transaction in a
    # child, or acquire a second owner lock while already holding one ourselves.
    borrowed = sources._current_connection()
    if borrowed is not None:
        async with checked(borrowed) as connection:
            yield connection
        return
    if sources.pool is None:
        raise QuietDeliverySuppressed("owner_quiet_unavailable", origin)
    async with sources.pool.acquire() as connection, connection.transaction():
        async with checked(connection):
            yield connection

