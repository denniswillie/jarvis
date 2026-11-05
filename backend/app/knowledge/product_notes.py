"""Trusted identity of editable notes, including the existing M03 import map."""
from uuid import UUID

from app.knowledge.models import SourceError

NOTE_NAMESPACE = "product:notes"
IMPORTED_NAMESPACE = "legacy:m03:v1"


def note_identity(namespace, source_key):
    if namespace == NOTE_NAMESPACE:
        candidate = source_key
    elif namespace == IMPORTED_NAMESPACE and source_key.startswith("jarvis_notes/"):
        candidate = source_key.removeprefix("jarvis_notes/")
    else:
        return None
    try:
        identity = UUID(candidate)
    except (ValueError, TypeError, AttributeError):
        return None
    return identity if str(identity) == candidate else None


async def note_sources(connection, owner_id, note_id):
    return await connection.fetch(
        """select * from jarvis_knowledge_sources where owner_id=$1
            and ((namespace=$2 and source_key=$3) or (namespace=$4 and source_key=$5))
            order by area_id,id""",
        owner_id, NOTE_NAMESPACE, str(note_id), IMPORTED_NAMESPACE, "jarvis_notes/" + str(note_id))


async def delete_note_original(connection, owner_id, source):
    """Single-source deletion cannot silently remove another area's original."""
    note_id = note_identity(source["namespace"], source["source_key"])
    if note_id is None:
        return
    aliases = await note_sources(connection, owner_id, note_id)
    if any(row["id"] != source["id"] and row["status"] != "deleted" for row in aliases):
        raise SourceError("note_shared_original",
            "This note has another retained source mapping. Resolve its area ownership before deleting the original.")
    await connection.execute("delete from jarvis_notes where owner_id=$1 and id=$2", owner_id, note_id)
