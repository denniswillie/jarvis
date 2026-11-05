"""Bounded canonical history shared by live voice and sensory context."""
import json


def pending_tool_acknowledgment(records):
    """Recover only an unspoken, evidenced tool outcome before an exact reset.

    This is transport/receipt correlation, not an inference about whether the
    user's broader task is complete. It never authorizes another domain action.
    """
    if not records:
        return None
    ordered = sorted(records, key=lambda record: record.origin_sequence)
    reset = ordered[-1]
    reset_content = reset.content or {}
    if (reset.kind != "context_gap" or reset.availability != "available"
            or not reset.is_current or reset_content.get("reason") != "source_context_rebuild"
            or reset_content.get("client_reported_audio_cleared") is not True
            or reset_content.get("unfinished_input_at_reset") is not False):
        return None
    requests = [record for record in ordered if record.kind in {"user_request", "user_transcript"}]
    if not requests:
        return None
    request = requests[-1]
    content = request.content or {}
    input_id = content.get("input_id")
    if (not request.authorizes_request or not isinstance(input_id, str) or not input_id
            or request.source_namespace != reset.source_namespace
            or any(record.kind == "assistant_output" and record.origin_sequence > request.origin_sequence
                   for record in ordered)):
        return None
    receipts = [record for record in ordered
        if record.kind == "tool_outcome" and record.is_current and record.availability == "available"
        and (record.content or {}).get("mutates_state") is True
        and record.source_namespace == request.source_namespace
        and record.origin_sequence > request.origin_sequence
        and (record.content or {}).get("input_id") == input_id]
    if not any((record.content or {}).get("result", {}).get("status") == "success" for record in receipts):
        return None
    return {"reset_event_id": str(reset.id), "request_event_id": str(request.id),
            "tool_event_ids": [str(record.id) for record in receipts]}


def tool_ack_recovery_instructions(candidate):
    return (
        "The previous voice connection refreshed its source context after a tool ran. "
        "Give the missing acknowledgment in one or two short, natural sentences. "
        "These are the mutation receipts for that request: "
        + json.dumps(candidate, ensure_ascii=True) + ". "
        "This is not a new user request and grants no authority to repeat an operation. "
        "Do not call tools or redo the work. Describe only the effects these mutation receipts confirm. "
        "Earlier search and inspection receipts are pre-operation snapshots, not the resulting state. "
        "Mention pending cleanup only when relevant; missing cleanup detail is unknown, not complete. "
        "Do not reconstruct omitted fields or claim complete erasure. Avoid internal IDs, revision numbers, "
        "read barriers, receipt jargon and unrelated delivery disclaimers. If the operation's outcome "
        "is unknown, say so plainly."
    )


def canonical_history_payload(snapshot, *, max_characters=24000):
    selected, sources, covered = [], set(), set()
    used = 0
    for record in snapshot["records"]:
        item = record.as_context()
        if record.availability != "available":
            item["content"] = None
            item["gap"] = "Original content is not available; do not reconstruct it."
        encoded = len(json.dumps(item, ensure_ascii=True))
        if used + encoded > max_characters:
            continue
        selected.append(item)
        used += encoded
        if record.availability == "available":
            sources.add(record.payload_id)
            content = record.content or {}
            for key in ("item_id", "provider_item_id"):
                identity = content.get(key)
                if isinstance(identity, str):
                    covered.add(identity)
    selected.sort(key=lambda item: item["origin_sequence"])
    payload = {
        "records": selected,
        "omitted_records": max(0, snapshot["total"]-len(selected)),
        "meaning": [
            "Retained conversation evidence, not new commands or a request to repeat old answers.",
            "Interrupted or generated speech is not proof that the user heard its full text.",
            "Missing transcripts are explicit gaps. Ask for repetition instead of inventing their words.",
            "Earlier history outside this bounded window has not been reconstructed here.",
        ],
    }
    return payload, tuple(sorted(sources)), covered
