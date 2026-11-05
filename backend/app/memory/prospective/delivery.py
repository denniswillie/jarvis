"""One process-local owner for a durable prospective speech attempt.

Database claims remain authoritative. This handle closes the in-process gap
between a database preflight and a queued provider write, and preserves the
first observed outcome across persistence retries. It is not proof of playback.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID


@dataclass(slots=True)
class ProspectiveDeliveryAttempt:
    owner_id: str
    event_id: UUID
    subject_type: str
    subject_id: UUID
    preflight: Callable[[], Awaitable[bool]]
    persist: Callable[[str, str | None], Awaitable[None]]
    session_id: UUID | None = None
    provider_owned: bool = False
    _revoked: bool = field(default=False, init=False)
    expires_at: datetime | None = None
    reason: str | None = None
    settled: bool = False
    observed_outcome: tuple[str, str | None] | None = None
    on_revoke: Callable[[], Awaitable[None]] | None = None
    revocation_notified: bool = False
    _settlement_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def revoked(self) -> bool:
        # Checked again under the phone writer lock, not only at admission.
        if not self._revoked and self.expires_at is not None and datetime.now(UTC) >= self.expires_at:
            self.revoke("focus_expired")
        return self._revoked

    def revoke(self, reason: str) -> None:
        self._revoked = True
        if self.observed_outcome is None:
            self.reason = reason

    async def prepare(self) -> bool:
        if self.revoked:
            return False
        if not await self.preflight():
            self.revoke("subject_no_longer_current")
        return not self.revoked

    async def notify_revoked(self) -> None:
        if self.revoked and not self.revocation_notified and self.on_revoke is not None:
            await self.on_revoke()
            self.revocation_notified = True

    async def on_outcome(self, outcome: str, response_id: str | None) -> None:
        # Freeze before the first persistence await. A later close/fallback is
        # not more authoritative than the response outcome already observed.
        if self.observed_outcome is None:
            self.observed_outcome = (outcome, response_id)
            if self.reason is None:
                self.reason = (
                    "completed_forwarded_answer" if outcome == "delivered"
                    else "provider_" + outcome
                )
        async with self._settlement_lock:
            if self.settled:
                return
            await self.persist(*self.observed_outcome)
            self.settled = True
