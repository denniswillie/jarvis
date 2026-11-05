"""Deliver explicit situational reminders; no queue of stale announcements."""


class CompanionNotices:
    def __init__(self, state, session_manager):
        self.state, self.sessions = state, session_manager
        self.last_status = None

    async def on_reminder(self, match, source_document_ids, human_generation):
        notice = await self.state.reserve_reminder_notice(
            reminder_id=match.reminder_id, revision=match.reminder_revision,
            evidence_at=match.evidence_at, generation=human_generation,
            source_document_ids=source_document_ids, evidence_description=match.description)
        if notice is None or not self.state.notice_current(notice):
            return
        try:
            result = await self.sessions.send_companion_notice(self.state.owner_id, notice)
            status = result.get("status", "unknown")
            if status not in {"provider_received", "unavailable", "rejected", "unknown"}:
                status = "unknown"
        except Exception:
            status = "unknown"
        self.last_status = status
        await self.state.finish_reminder_notice(notice, status)
