"""Explicit observation-to-notification lineage, shared by existing producers."""
from uuid import UUID

from app.knowledge.models import SourceError
from app.tasks.models import json_object


async def observation_delivery_source(publication, *, owner_id, session_id, observation_id):
    async with publication.transaction(owner_id) as connection:
        payload = await connection.fetchval(
            """select payload_id from jarvis_observation_sources
               where owner_id=$1 and session_id=$2 and observation_id=$3""",
            owner_id, session_id, observation_id)
        if payload is None or not await source_is_current(connection, owner_id, payload):
            raise SourceError("visual_delivery_source_unavailable", "The alert's observation is unavailable.")
        return payload


async def source_is_current(connection, owner_id, payload_id):
    if payload_id is None:
        return True
    return await connection.fetchval(
        "select lock_jarvis_copy_sources($1,$2::uuid[])", owner_id, [payload_id])


async def lock_visual_delivery_source(connection, *, owner_id, session_id, payload_id, frame_ids):
    """Enter before operational row locks and retain the fence through commit."""
    if payload_id is None:
        return
    if not await source_is_current(connection, owner_id, payload_id):
        raise SourceError("visual_delivery_source_unavailable", "The alert's observation is unavailable.")
    original = await connection.fetchval(
        """select p.content from jarvis_observation_sources b
           join jarvis_context_payloads p on p.owner_id=b.owner_id and p.id=b.payload_id
           where b.owner_id=$1 and b.session_id=$2 and b.payload_id=$3""",
        owner_id, session_id, payload_id)
    frames = tuple(frame_ids)
    if original is None or not frames or len(frames) > 64:
        raise SourceError("visual_delivery_source_mismatch", "The alert must use its owned observation.")
    content = json_object(original)
    allowed = {item["frame_id"] for item in content.get("frame_links", ())}
    try:
        supplied = {str(UUID(str(value))) for value in frames}
    except (ValueError, TypeError, AttributeError) as error:
        raise SourceError("visual_delivery_source_mismatch", "The alert has invalid frame identities.") from error
    if not supplied.issubset(allowed):
        raise SourceError("visual_delivery_source_mismatch", "The alert references frames outside its observation.")


async def bind_visual_delivery_source(connection, *, owner_id, session_id, kind, delivery_id, payload_id):
    if payload_id is None:
        return
    await connection.execute(
        """insert into jarvis_visual_delivery_sources
           (owner_id,delivery_kind,delivery_id,session_id,payload_id) values($1,$2,$3,$4,$5)""",
        owner_id, kind, delivery_id, session_id, payload_id)
