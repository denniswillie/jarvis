"""Render authorized canonical dialogue for Hermes without a second history store."""
from copy import deepcopy


def native_conversation_context(manifest):
    """Move only finalized dialogue text into native messages, retaining provenance.

    The stored manifest is unchanged. This transient projection keeps every source
    record and its non-text fields; exact dialogue text occurs once in the native
    request. Partial, withdrawn, derived and non-dialogue records stay evidence,
    not new user/assistant turns. Archive readers retain the original records.
    """
    projected = deepcopy(manifest)
    history = []
    for message in projected.get("messages", []):
        role = None
        if (message.get("availability") != "available" or not message.get("is_current")
                or message.get("finality") not in {"final", "revised"}):
            continue
        if message.get("kind") in {"user_request", "user_transcript"} and message.get("trust") == "user_input":
            role = "user"
        elif message.get("kind") == "assistant_output" and message.get("trust") == "assistant_output":
            role = "assistant"
        content = message.get("content")
        text = content.get("text") if isinstance(content, dict) else None
        if role is None or not isinstance(text, str) or not text.strip() or "\x00" in text:
            continue
        message["native_history_index"] = len(history)
        del content["text"]
        history.append({"role": role, "content": text})
    projected["native_history"] = {
        "included": len(history),
        "text_location": "conversation_history[messages[].native_history_index].content",
        "meaning": "Transient rendering of the authorized manifest, not additional requests or a new history store. "
                   "Source IDs, revisions, gaps and playback uncertainty remain in messages. "
                   "Use the original user dialogue as primary; a derived routing command is supplementary.",
    }
    return {"manifest": projected, "conversation_history": history}
