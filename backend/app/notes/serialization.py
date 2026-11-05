"""Serialize retained historical note records; not a model tool catalog."""
from typing import Any
from app.notes.models import JarvisNote


def _note_payload(note: JarvisNote) -> dict[str, Any]:
    result = {
        "id": str(note.id),
        "session_id": str(note.session_id) if note.session_id else None,
        "title": note.title,
        "body": note.body,
        "source": note.source,
        "created_at": note.created_at.isoformat(),
        "updated_at": note.updated_at.isoformat(),
        "metadata": note.metadata,
    }

    if note.memory_source is not None:
        # Full originals remain available through the existing paged source reader.
        # Bound search and durable receipts without silently truncating stored text.
        result.update(body=note.body[:1000], body_complete=len(note.body) <= 1000,
                      body_characters=len(note.body), metadata=None,
                      memory_source=note.memory_source,
                      meaning="Exact stored note; body may be a preview and metadata is omitted. "
                              "Use explain_memory for the full original. Storage is not independent proof "
                              "that its statement is true, nor proof that Hindsight ingestion has completed.")
        if note.task_evidence is not None:
            result["task_evidence"] = note.task_evidence
    return result
