"""Rebuild voice context from canonical records, with explicit omissions and gaps."""
from dataclasses import replace

from app.tasks.context_history import canonical_history_payload, pending_tool_acknowledgment


def recover_voice_context(context, snapshot, *, max_characters=24000):
    payload, sources, covered = canonical_history_payload(snapshot, max_characters=max_characters)
    pending = pending_tool_acknowledgment(snapshot["records"])
    if pending is not None:
        selected = {item["id"] for item in payload["records"]}
        request = next(record for record in snapshot["records"] if str(record.id) == pending["request_event_id"])
        required = {str(record.id) for record in snapshot["records"]
                    if record.origin_sequence >= request.origin_sequence}
        # A truncated result or missing later output is not proof of an
        # unanswered operation. Let a future user request clarify it instead.
        if required <= selected:
            payload["pending_tool_acknowledgment"] = pending
    # Identity matching only. Never guess duplicates from similar text.
    recent = [message for message in context.recent_messages if message.provider_item_id not in covered]
    return replace(context, recent_messages=recent, voice_continuity=payload,
                   voice_continuity_source_ids=sources)
