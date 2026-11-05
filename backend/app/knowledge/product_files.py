"""Trusted identity and physical-original deletion for source-backed files."""
from uuid import UUID

from app.knowledge.models import SourceError

FILE_NAMESPACE = "product:files"
FILE_ORIGINAL_FORMAT = "jarvis.text_file_original.v1"
IMAGE_ORIGINAL_FORMAT = "jarvis.image_file_original.v1"
GENERATED_ORIGINAL_FORMAT = "jarvis.generated_file_original.v1"
RECORDING_ORIGINAL_FORMAT = "jarvis.recording_file_original.v1"


def file_identity(namespace, source_key):
    if namespace != FILE_NAMESPACE:
        return None
    try:
        identity = UUID(source_key)
    except (ValueError, TypeError, AttributeError):
        return None
    return identity if str(identity) == source_key else None


async def file_sources(connection, owner_id, file_id):
    return await connection.fetch(
        """select * from jarvis_knowledge_sources where owner_id=$1
            and namespace=$2 and source_key=$3 order by area_id,id""",
        owner_id, FILE_NAMESPACE, str(file_id))


async def delete_file_original(connection, owner_id, source):
    """Privacy deletes originals and derived files; corrections never call this."""
    file_id = file_identity(source["namespace"], source["source_key"])
    # Older migration fixtures and the pre-file deployment retain their delete path.
    if await connection.fetchval("select to_regclass('jarvis_file_publications')") is None:
        if file_id is not None:
            raise SourceError("file_publication_missing", "File cleanup schema is unavailable.")
        return
    if file_id is not None:
        aliases = await file_sources(connection, owner_id, file_id)
        if any(row["id"] != source["id"] and row["status"] != "deleted" for row in aliases):
            raise SourceError("file_shared_original",
                "Resolve this file's competing source mappings before deleting its original.")
    publications = await connection.fetch(
        """with recursive copies(id) as (
            select payload_id from jarvis_knowledge_source_versions
                where owner_id=$1 and source_id=$2
            union
            select d.dependent_payload_id from jarvis_context_payload_dependencies d
                join copies c on c.id=d.source_payload_id where d.owner_id=$1
        )
        select p.* from jarvis_file_publications p where p.owner_id=$1
            and ((p.file_id=$3 and p.area_id=$4) or p.metadata_payload_id in (select id from copies))
            order by p.id for update""", owner_id, source["id"], file_id, source["area_id"])
    if file_id is not None and not any(row["file_id"] == file_id for row in publications):
        raise SourceError("file_publication_missing", "The original has no tracked storage-cleanup identity.")
    for publication in publications:
        if publication["phase"] != "deleted":
            await connection.execute(
                """update jarvis_file_publications set delete_requested=true,phase='delete_pending'
                    where owner_id=$1 and id=$2""", owner_id, publication["id"])
            await connection.execute(
                """update jarvis_context_payloads set availability='deleted',content=null
                    where owner_id=$1 and id=$2""", owner_id, publication["metadata_payload_id"])

async def read_file_original(connection, sources, owner_id, area_id, file, *, lock=True):
    """Exact original admission; unlocked snapshots are metadata links, not byte grants.

    Readers share the publication lock with task/grant validation. Taking an
    update lock here before binding task evidence reverses the heartbeat's
    task-owner -> source order. SHARE still excludes publication changes and
    deletion while admitting the original.
    """
    rows = await file_sources(connection, owner_id, file.id)
    if len([row for row in rows if row["status"] != "deleted"]) > 1:
        raise SourceError("file_shared_original", "This file has competing retained source mappings.")
    selected = [row for row in rows if row["area_id"] == area_id]
    if len(selected) != 1:
        raise SourceError("file_source_unavailable", "The file has no source original in the authorized area.")
    source = await sources._read(connection, owner_id, area_id, selected[0]["id"])
    if (source.content.get("format") not in {FILE_ORIGINAL_FORMAT, IMAGE_ORIGINAL_FORMAT,
                                           GENERATED_ORIGINAL_FORMAT, RECORDING_ORIGINAL_FORMAT}
            or source.content.get("record") != file.model_dump(mode="json")):
        raise SourceError("file_original_changed", "File metadata and its source original disagree.")
    publication = await connection.fetchrow(
        "select * from jarvis_file_publications where owner_id=$1 and area_id=$2 and file_id=$3"
        + (" for share" if lock else ""),
        owner_id, area_id, file.id)
    metadata_state = (await connection.fetchval(
        "select lock_jarvis_copy_source($1,$2)" if lock else
        "select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
        owner_id, publication["metadata_payload_id"]) if publication is not None else None)
    if (publication is None or publication["phase"] != "available" or publication["delete_requested"]
            or publication["content_sha256"] != source.content.get("content_sha256")
            or (publication["storage_bucket"], publication["storage_path"], publication["size_bytes"])
                != (file.storage_bucket, file.storage_path, file.size_bytes)
            or metadata_state != "available"):
        raise SourceError("file_original_unavailable", "The verified file original is no longer available.")
    return source, publication
