"""Compact current-state API; existing repositories retain durable ownership.

No LLM calls, vector search, scheduler, task manager, or automatic Voice pushes.
One instance belongs to one owner in the serving process. Optional projections
must be RAM-only functions supplied by the sandbox/app service owners.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4


STATE_KEY = "companion_live_state"


def now_utc():
    return datetime.now(timezone.utc)


def iso(value):
    if value.tzinfo is None:
        raise ValueError("An observation must have a timezone")
    return value.isoformat()


def parse_time(value):
    return datetime.fromisoformat(value) if value else None


class CompanionState:
    def __init__(self, *, owner_id, working_memory, reminders, clock=now_utc,
                 sandbox_projection=None, app_projection=None, stale_after_seconds=300,
                 sandbox_refresh=None, plugin_projection=None):
        if not owner_id or stale_after_seconds <= 0:
            raise ValueError("Owner and positive freshness interval required")
        self.owner_id = owner_id
        self.working_memory, self.reminders = working_memory, reminders
        self.clock = clock
        self.sandbox_projection, self.app_projection = sandbox_projection, app_projection
        self.sandbox_refresh = sandbox_refresh
        self.plugin_projection = plugin_projection
        self.stale_after_seconds = stale_after_seconds
        self._root, self._section, self._revision = {}, {}, 0
        self._reminders = {}
        self._lock = asyncio.Lock()
        self._ready = False

    async def initialize(self):
        async with self._lock:
            if self._ready:
                return
            row = await self.working_memory.get_owner_working_memory(owner_id=self.owner_id)
            if row is not None:
                self._root, self._revision = deepcopy(dict(row.state)), row.revision
            self._section = deepcopy(self._root.get(STATE_KEY, {}))
            await self.refresh_reminders()
            self._ready = True

    async def refresh_reminders(self):
        rows = await self.reminders.list_reminders(owner_id=self.owner_id, status="active", limit=128)
        self._reminders = {str(row.id): row.model_dump(mode="json") for row in rows}

    async def _persist(self, changes):
        """Called under the writer lock; preserve unrelated working-memory keys."""
        for _ in range(2):
            section = deepcopy(self._root.get(STATE_KEY, {}))
            section.update(deepcopy(changes))
            root = deepcopy(self._root)
            root[STATE_KEY] = section
            result = await self.working_memory.compare_and_swap_owner_working_memory(
                owner_id=self.owner_id, expected_revision=self._revision, state=root,
                expires_at=None,
            )
            self._root = deepcopy(dict(result.state.state))
            self._revision = result.state.revision
            self._section = deepcopy(self._root.get(STATE_KEY, {}))
            if result.applied:
                return
        raise RuntimeError("companion_state_write_conflict")

    async def read_current(self):
        """Explicit pull: refresh bounded sandbox facts, then project RAM state."""
        if self.sandbox_refresh is not None:
            await self.sandbox_refresh(self.owner_id)
        return self.read()

    def read(self):
        human = deepcopy(self._section.get("human"))
        if human is not None:
            observed = parse_time(human.get("last_observed_at"))
            human["stale"] = observed is None or (self.clock() - observed).total_seconds() > self.stale_after_seconds
        system = {
            "reminders": deepcopy(list(self._reminders.values())),
            "sandboxes": deepcopy(self.sandbox_projection(self.owner_id)) if self.sandbox_projection else None,
            "app": deepcopy(self.app_projection(self.owner_id)) if self.app_projection else None,
        }
        if self.plugin_projection is not None:
            system["plugins"] = deepcopy(self.plugin_projection(self.owner_id))
        result = {
            "ready": self._ready, "as_of": iso(self.clock()), "revision": self._revision,
            "human_generation": self._section.get("human_generation", 0), "human": human,
            "system": system,
        }
        location = deepcopy(self._section.get("location"))
        if location is not None:
            captured = parse_time(location.get("captured_at"))
            location["stale"] = captured is None or (self.clock() - captured).total_seconds() > 1800
            result["location"] = location
        return result

    def directory(self):
        inventory = self.sandbox_projection(self.owner_id) if self.sandbox_projection else {}
        cards = inventory.get("items", []) if isinstance(inventory, dict) else []
        return {"reminders": [[row["id"], row["title"][:20]]
                              for row in list(self._reminders.values())[:1]],
                "sandboxes": [[card["allocation_id"], card["title"][:20], card["state"]]
                              for card in cards[:1]],
                "partial": len(self._reminders) > 1 or len(cards) > 1 or inventory.get("has_more", False)}

    def notice_allowed(self, *, observed_at=None):
        # Conversational silence is managed by Voice, not persisted policy.
        return self._ready

    @asynccontextmanager
    async def delivery_guard(self, *, observed_at=None):
        # Keep readiness admission without a separate conversational quiet gate.
        if not self.notice_allowed(observed_at=observed_at):
            from app.attention.quiet_delivery import QuietDeliverySuppressed
            raise QuietDeliverySuppressed("live_state_not_ready", "live_hermes_notice")
        yield

    async def update_human(self, content, *, observed_at, source_document_ids=(), generation=None):
        async with self._lock:
            current_generation = self._section.get("human_generation", 0)
            if generation is not None and generation != current_generation:
                return False
            previous = self._section.get("human")
            if previous and parse_time(previous["last_observed_at"]) > observed_at:
                return False
            human = {"content": deepcopy(content), "last_observed_at": iso(observed_at),
                     "generated_at": iso(self.clock()), "revision": (previous or {}).get("revision", 0) + 1,
                     "source_document_ids": list(dict.fromkeys(source_document_ids))}
            await self._persist({"human": human})
            return True

    async def update_location(self, location: dict, *, captured_at: datetime) -> bool:
        """Store authenticated companion device location in Live State RAM + working memory."""
        captured_iso = iso(captured_at)
        async with self._lock:
            previous = self._section.get("location")
            if previous and previous.get("captured_at") and parse_time(previous["captured_at"]) > captured_at:
                return False
            record = {
                **deepcopy(location),
                "captured_at": captured_iso,
                "updated_at": iso(self.clock()),
                "revision": (previous or {}).get("revision", 0) + 1,
            }
            await self._persist({"location": record})
            return True

    async def reserve_reminder_notice(self, *, reminder_id, revision, evidence_at,
                                      generation, source_document_ids=(), waiting_for_assistant=(),
                                      evidence_description=""):
        """One occurrence is one active reminder; speech never completes it."""
        async with self._lock:
            row = self._reminders.get(str(reminder_id))
            now = self.clock()
            if (row is None or row.get("updated_at") != revision
                    or generation != self._section.get("human_generation", 0)
                    or not self.notice_allowed(observed_at=evidence_at)
                    or not 0 <= (now - evidence_at).total_seconds() <= 60):
                return None
            notices = deepcopy(self._section.get("notices", {}))
            previous = notices.get(str(reminder_id))
            if previous and previous["reminder_revision"] == revision:
                repeat = row.get("metadata", {}).get("repeat_until_confirmed", False)
                if previous["status"] in {"provider_received", "assistant_output_observed"} and not repeat:
                    return None
                if previous["status"] == "deferred":
                    # The original barrier and TTL never grow with later traffic.
                    if previous.get("waiting_for_assistant") or not self._notice_valid(previous):
                        return None
                    previous["status"] = "pending"
                    await self._persist({"notices": notices})
                    return deepcopy(previous)
                if (now - parse_time(previous["attempted_at"])).total_seconds() < 30:
                    return None
            notice = {"id": str(uuid4()), "reminder_id": str(reminder_id),
                "reminder_revision": revision, "generation": generation,
                "text": ("Reminder condition matched: " + row["title"]
                         + "\nObserved (" + iso(evidence_at) + "): " + evidence_description[:600]
                         if evidence_description else "Reminder: " + row["title"]),
                "evidence_description": evidence_description[:600], "evidence_at": iso(evidence_at),
                "source_document_ids": list(source_document_ids),
                "attempted_at": iso(now), "expires_at": iso(now + timedelta(seconds=30)),
                "status": "deferred" if waiting_for_assistant else "pending",
                "waiting_for_assistant": dict(waiting_for_assistant)}
            notices.pop(str(reminder_id), None)
            notices[str(reminder_id)] = notice
            notices = dict(list(notices.items())[-128:])
            await self._persist({"notices": notices})
            return deepcopy(notice)

    def _notice_valid(self, notice):
        row = self._reminders.get(notice["reminder_id"])
        saved = self._section.get("notices", {}).get(notice["reminder_id"], {})
        return bool(row and saved.get("id") == notice["id"]
            and row.get("updated_at") == notice["reminder_revision"]
            and notice["generation"] == self._section.get("human_generation", 0)
            and self.clock() < parse_time(notice["expires_at"])
            and self.notice_allowed(observed_at=parse_time(notice["evidence_at"])))

    def notice_current(self, notice):
        saved = self._section.get("notices", {}).get(notice["reminder_id"], {})
        return self._notice_valid(notice) and saved.get("status") == "pending"

    def reminder_notification_status(self, reminder_id, revision):
        notice = self._section.get("notices", {}).get(str(reminder_id), {})
        return notice.get("status") if notice.get("reminder_revision") == revision else None

    async def observe_reminder_delivery(self, delivery, *, generation, source_document_ids=()):
        """Annotate an eligible captured occurrence, never invent delivery or hearing."""
        async with self._lock:
            row = self._reminders.get(str(delivery.reminder_id))
            now = self.clock()
            if (row is None or row.get("updated_at") != delivery.reminder_revision
                    or generation != self._section.get("human_generation", 0)
                    or not 0 <= (now - delivery.evidence_at).total_seconds() <= 60):
                return False
            notices = deepcopy(self._section.get("notices", {}))
            previous = notices.get(str(delivery.reminder_id), {})
            if (previous.get("status") not in {"pending", "deferred"}
                    or previous.get("reminder_revision") != delivery.reminder_revision
                    or previous.get("generation") != generation
                    or not self._notice_valid(previous)
                    or not self.notice_allowed(observed_at=delivery.evidence_at)
                    or len(delivery.sources) != 1
                    or delivery.sources[0] not in {
                        f"dialogue:{source}:{revision}"
                        for source, revision in previous.get("waiting_for_assistant", {}).items()
                    }):
                return False
            if previous.get("notified_at") and parse_time(previous["notified_at"]) >= delivery.evidence_at:
                return False
            notice = {**previous,
                "status": "assistant_output_observed", "notified_at": iso(delivery.evidence_at),
                "evidence_at": iso(delivery.evidence_at),
                "notification_source": {"sources": list(delivery.sources), "quote": delivery.quote},
                "source_document_ids": list(source_document_ids), "waiting_for_assistant": {}}
            notices[str(delivery.reminder_id)] = notice
            await self._persist({"notices": dict(list(notices.items())[-128:])})
            return True

    async def consume_assistant_revisions(self, revisions):
        """Release only fixed captured barriers, using the existing notice records."""
        consumed, ready = dict(revisions), []
        if not consumed:
            return ready
        async with self._lock:
            notices = deepcopy(self._section.get("notices", {}))
            changed = False
            for notice in notices.values():
                if notice["status"] != "deferred":
                    continue
                waiting = notice.get("waiting_for_assistant", {})
                remaining = {key: revision for key, revision in waiting.items()
                             if consumed.get(key, 0) < revision}
                if remaining != waiting:
                    notice["waiting_for_assistant"] = remaining
                    changed = True
                    if not remaining and self._notice_valid(notice):
                        ready.append(deepcopy(notice))
            if changed:
                await self._persist({"notices": notices})
        return ready

    async def finish_reminder_notice(self, notice, status):
        if status not in {"provider_received", "unavailable", "rejected", "unknown"}:
            raise ValueError("Unknown notice delivery result")
        async with self._lock:
            notices = deepcopy(self._section.get("notices", {}))
            saved = notices.get(notice["reminder_id"], {})
            if saved.get("id") != notice["id"]:
                return
            if saved.get("status") == "assistant_output_observed":
                saved["provider_status"] = status
            else:
                saved["status"] = status
            await self._persist({"notices": notices})

    async def manage_reminder(self, *, action, session_id=None, reminder_id=None,
                              title=None, condition=None, details="", source_document_ids=(),
                              repeat_until_confirmed=False):
        if action not in {"create", "complete", "delete"}:
            raise ValueError("Use create, complete this occurrence, or delete this reminder")
        if action == "create":
            if (not isinstance(title, str) or not 1 <= len(title.strip()) <= 200
                    or not isinstance(condition, str) or not 1 <= len(condition.strip()) <= 1000
                    or not isinstance(details, str) or len(details) > 2000
                    or type(repeat_until_confirmed) is not bool):
                raise ValueError("A bounded title and situational condition are required")
            row = await self.reminders.create_reminder(owner_id=self.owner_id, session_id=session_id,
                title=title.strip(), condition=condition.strip(), details=details, remind_at=None,
                metadata={"operational": True, "source_document_ids": list(source_document_ids),
                          "repeat_until_confirmed": repeat_until_confirmed})
        else:
            identity = UUID(str(reminder_id))
            method = self.reminders.complete_reminder if action == "complete" else self.reminders.delete_reminder
            row = await method(owner_id=self.owner_id, reminder_id=identity)
        await self.refresh_reminders()
        return row.model_dump(mode="json") if row is not None else None

    async def invalidate_memory(self, scope):
        if scope.owner_id != self.owner_id:
            raise ValueError("Memory invalidation belongs to another owner")
        identities = set(scope.document_ids)
        async with self._lock:
            # The current human snapshot is disposable derived context. Clear
            # it conservatively; do not risk retaining an unindexed old claim.
            # A pending analysis from before this invalidation may not restore it.
            await self._persist({"human": None,
                "human_generation": self._section.get("human_generation", 0) + 1})
            await self.refresh_reminders()
            for row in tuple(self._reminders.values()):
                if identities.intersection(row.get("metadata", {}).get("source_document_ids", ())):
                    await self.reminders.delete_reminder(owner_id=self.owner_id, reminder_id=UUID(row["id"]))
            await self.refresh_reminders()
