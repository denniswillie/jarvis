"""Capture removable visual-model inputs using canonical conversation and sources."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from typing import Mapping
from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.read_provenance import SourceReadProvenance
from app.knowledge.sources import SOURCE_VERSION_SQL
from app.perception.live_brief import (
    live_brief_input_fields,
    live_brief_parent_payload_ids,
    seal_live_brief_input,
)
from app.tasks.context_history import canonical_history_payload
from app.tasks.models import digest, json_object


@dataclass(frozen=True)
class ObservationInputSnapshot:
    owner_id: str
    session_id: UUID
    payload_id: UUID
    request_digest: str


def working_observation_ids(state):
    """Read the reducer's explicit provenance fields, never infer identities from text."""
    identities = set()
    if state.get("last_observation_id"):
        identities.add(UUID(str(state["last_observation_id"])))
    nodes = [state.get("scene"), state.get("activity")]
    nodes.extend(state.get("entities", {}).values())
    nodes.extend(state.get("visible_text", ()))
    nodes.extend(state.get("recent_changes", ()))
    for node in nodes:
        if isinstance(node, Mapping) and node.get("source_observation_id"):
            identities.add(UUID(str(node["source_observation_id"])))
    return tuple(sorted(identities))


def request_document(request):
    """Exact non-pixel model input; bytes are represented only by their digest."""
    value = {
        "owner_id": request.owner_id, "session_id": str(request.session_id),
        "window_id": str(request.window_id), "requested_at": request.requested_at.isoformat(),
        "frames": [{"frame_id": str(frame.frame_id), "frame_sequence": frame.frame_sequence,
                    "captured_at": frame.captured_at.isoformat(), "source": frame.source,
                    "content_type": frame.content_type, "sha256": sha256(frame.image_bytes).hexdigest()}
                   for frame in request.frames],
        "previous_working_state": dict(request.previous_working_state),
        "existing_entity_ids": sorted(request.existing_entity_ids),
        "recent_audio_transcripts": [asdict(item) for item in request.recent_audio_transcripts],
        "behavior_instructions": request.behavior_instructions,
        "active_reminders": list(request.active_reminders), "active_focus": request.active_focus,
        "recent_interrupts": list(request.recent_interrupts), "voice_context": request.voice_context,
        "input_context_gaps": list(request.input_context_gaps),
    }
    value.update(live_brief_input_fields(request))
    if request.camera_connection_id is not None:
        value["camera_stream"] = {"visual_session_id": str(request.camera_visual_session_id),
                                  "connection_id": str(request.camera_connection_id)}
    def encode(value):
        if isinstance(value, (datetime, UUID)):
            return value.isoformat() if isinstance(value, datetime) else str(value)
        raise TypeError("Unsupported visual input value")
    return json_object(json.dumps(value, default=encode, allow_nan=False))


class ObservationInputCapture:
    def __init__(self, context):
        self.context = context
        self.sources = context.publication.sources
        self.area_id = context.publication.area_id
        self.provenance = SourceReadProvenance(self.sources)

    async def _observation_payloads(self, connection, owner_id, session_id, observation_ids):
        if not observation_ids:
            return {}
        rows = await connection.fetch(
            """select observation_id,payload_id from jarvis_observation_sources
                where owner_id=$1 and session_id=$5 and observation_id=any($2::uuid[])
                union all
                select substring(s.source_key from length('perception_observations/')+1)::uuid,v.payload_id
                from jarvis_knowledge_sources s join jarvis_knowledge_source_versions v
                    on v.owner_id=s.owner_id and v.source_id=s.id and v.revision=1
                where s.owner_id=$1 and s.area_id=$3 and s.namespace='legacy:m03:v1'
                    and s.source_key=any($4::text[])
                    and exists(select 1 from jarvis_knowledge_source_sessions ss
                        where ss.owner_id=s.owner_id and ss.source_id=s.id and ss.source_revision=1
                            and ss.session_id=$5)""",
            owner_id, list(observation_ids), self.area_id,
            ["perception_observations/" + str(identity) for identity in observation_ids], session_id)
        result = {}
        for row in rows:
            result.setdefault(row["observation_id"], set()).add(row["payload_id"])
        return result

    async def _parents(self, connection, owner_id, candidates):
        """Resolve one bounded batch; indexed originals also require current source scope."""
        if not candidates:
            return {}, {}
        if len(candidates) > 128:
            raise SourceError("observation_context_limit", "Visual context exceeds the source evidence bound.")
        available = {row["id"]: row["availability"] == "available" for row in await connection.fetch(
            "select id,availability from jarvis_context_payloads where owner_id=$1 and id=any($2::uuid[])",
            owner_id, list(candidates))}
        indexed = {row["payload_id"] for row in await connection.fetch(
            "select distinct payload_id from jarvis_knowledge_source_versions where owner_id=$1 and payload_id=any($2::uuid[])",
            owner_id, list(candidates))}
        metadata = {}
        rows = await connection.fetch(SOURCE_VERSION_SQL + " and v.payload_id=any($3::uuid[])",
                                      owner_id, self.area_id, list(candidates))
        seen = set()
        for row in rows:
            payload_id = row["payload_id"]
            seen.add(payload_id)
            try:
                if row["area_status"] != "active":
                    raise SourceError("observation_context_archived", "The source area is archived.")
                source = self.sources._version(row)
            except SourceError:
                available[payload_id] = False
            else:
                metadata.setdefault(payload_id, []).append(source.original_evidence())
        for payload_id in indexed - seen:
            available[payload_id] = False
        return available, metadata

    async def prepare(self, request, context):
        owner_id, session_id = request.owner_id, request.session_id
        if (context.owner_id, context.session_id) != (owner_id, session_id):
            raise SourceError("observation_context_scope", "Visual context belongs to another session.", 403)
        async with self.context.publication.transaction(self.context.tasks, owner_id) as connection:
            history = await self.context.voice_history_in_transaction(connection, owner_id, session_id)
            # Preserve the existing frame-relative transcript policy. A late
            # transcript revision is not silently moved before this image.
            history["records"] = [record for record in history["records"]
                                  if record.recorded_at <= request.primary_frame.captured_at]
            gaps = list(request.input_context_gaps)
            # A pending transcript can have an available metadata payload without
            # any words. Do not make that absence a current semantic dependency:
            # normal ASR finalization would revoke otherwise valid visual input.
            # Omit its metadata too, before selecting either content or parents.
            missing_transcripts = {
                record.id for record in history["records"]
                if record.kind == "user_transcript"
                and record.finality in {"provisional", "unavailable"}
                and record.content is not None and record.content.get("text") is None
            }
            if missing_transcripts:
                history["records"] = [record for record in history["records"]
                                      if record.id not in missing_transcripts]
                gaps.append({"component": "recent_conversation", "reason": "transcript_not_finalized",
                             "omitted_messages": len(missing_transcripts),
                             "meaning": "Missing transcript text is unknown; do not infer words or commands."})
            try:
                observation_ids = working_observation_ids(request.previous_working_state)
            except (ValueError, TypeError, AttributeError):
                observation_ids = ()
            interrupted_frames = [item.visual_frame_id for item in context.recent_visual_interrupts
                                  if item.visual_frame_id is not None]
            observations = await connection.fetch(
                """select id,primary_visual_frame_id from perception_observations
                    where owner_id=$1 and session_id=$2 and primary_visual_frame_id=any($3::uuid[])
                    order by id limit 129""", owner_id, session_id, interrupted_frames)
            if len(observations) > 128:
                raise SourceError("observation_context_limit", "Visual interrupt evidence exceeds its bound.")
            by_frame = {}
            for row in observations:
                by_frame.setdefault(row["primary_visual_frame_id"], []).append(row["id"])
            mapped = await self._observation_payloads(
                connection, owner_id, session_id, tuple(set(observation_ids) | {row["id"] for row in observations}))
            candidates = {record.payload_id for record in history["records"] if record.availability == "available"}
            candidates.update(payload for values in mapped.values() for payload in values)
            brief_parents = live_brief_parent_payload_ids(request)
            candidates.update(brief_parents)
            available, metadata = await self._parents(connection, owner_id, candidates)
            # An assignment is inseparable from its exact canonical binding.
            # Never retain its prose while silently dropping an unavailable root.
            if any(not available.get(parent, False) for parent in brief_parents):
                raise SourceError("observation_brief_context_unavailable",
                                  "The perception assignment's canonical originals are unavailable.")
            history["records"] = [
                record if record.availability != "available" or available.get(record.payload_id, False)
                else record.model_copy(update={"availability": "unavailable", "content": None, "content_digest": None})
                for record in history["records"]]
            voice, parents, covered = canonical_history_payload(history)
            parents = set(parents)
            parents.update(brief_parents)
            # Canonical voice capture is the selected conversation authority.
            # An independently retained operational copy must not resurrect
            # deleted/corrected canonical speech, even if its own parent survives.
            missing_messages = sum(message.provider_item_id not in covered
                                   for message in context.recent_audio_transcripts)
            if missing_messages:
                gaps.append({"component": "recent_conversation", "reason": "canonical_original_unavailable",
                             "omitted_messages": missing_messages})
            working = dict(request.previous_working_state)
            working_parents = {payload for identity in observation_ids for payload in mapped.get(identity, ())}
            if working and (not observation_ids or any(identity not in mapped for identity in observation_ids)
                            or not all(available.get(payload, False) for payload in working_parents)):
                working = {}
                gaps.append({"component": "previous_working_state", "reason": "original_evidence_unavailable"})
            else:
                parents.update(working_parents)
            interrupts = []
            for item, rendered in zip(context.recent_visual_interrupts, request.recent_interrupts, strict=True):
                identities = by_frame.get(item.visual_frame_id, ())
                originals = {payload for identity in identities for payload in mapped.get(identity, ())}
                if (not originals or any(identity not in mapped for identity in identities)
                        or not all(available.get(payload, False) for payload in originals)):
                    gaps.append({"component": "recent_interrupt", "reason": "observation_original_unavailable",
                                 "interrupt_id": str(item.id)})
                    continue
                parents.update(originals)
                interrupts.append(rendered)
            prepared = replace(request, previous_working_state=working,
                existing_entity_ids=request.existing_entity_ids.intersection(working.get("entities", {})),
                recent_audio_transcripts=(), recent_interrupts=tuple(interrupts),
                voice_context=voice, input_context_gaps=tuple(gaps))
            document = request_document(prepared)
            fingerprint = digest(document)
            if parents and not await connection.fetchval(
                    "select lock_jarvis_copy_sources($1,$2::uuid[])", owner_id, sorted(parents)):
                raise SourceError("observation_context_changed", "Visual input evidence changed during capture.")
            payload_id = await self.context.tasks.payload(connection, owner_id, {
                "format": "jarvis.visual_input.v1", "request_digest": fingerprint, "request": document,
                "policy_provenance": (
                    "Behavior instructions, active focus and reminder configuration are an operational "
                    "snapshot, not proof of a physical action. Their product lifecycle remains separate."),
            })
            if parents:
                await connection.executemany(
                    """insert into jarvis_context_payload_dependencies(owner_id,source_payload_id,dependent_payload_id)
                        values($1,$2,$3)""", [(owner_id, parent, payload_id) for parent in sorted(parents)])
            selected_voice = {UUID(item["id"]) for item in voice["records"] if item.get("content") is not None}
            canonical_parents = [record for record in history["records"]
                                 if record.id in selected_voice and record.payload_id in parents]
            if canonical_parents:
                await connection.executemany(
                    """insert into jarvis_context_event_dependencies(owner_id,dependent_payload_id,
                        event_id,event_revision,source_payload_id,source_usage) values($1,$2,$3,$4,$5,$6)""",
                    [(owner_id, payload_id, record.id, record.revision, record.payload_id, record.dependency_usage)
                     for record in canonical_parents])
            references = [item for parent in sorted(parents) for item in metadata.get(parent, ())]
            async def materialize(conn, originals):
                return None, payload_id
            # Reuse the existing current-source fence in bounded batches. Each
            # batch binds the same input original; no extra memory engine or
            # provider operation is created for these input snapshots.
            for offset in range(0, len(references), self.provenance.reference_limit):
                await self.provenance.bind_in_transaction(
                    connection, owner_id, self.area_id,
                    references[offset:offset+self.provenance.reference_limit], materialize)
            snapshot = ObservationInputSnapshot(owner_id, session_id, payload_id, fingerprint)
            return seal_live_brief_input(prepared, snapshot), snapshot

    async def check(self, connection, snapshot, *, owner_id, session_id, frame_ids):
        if not isinstance(snapshot, ObservationInputSnapshot) or (
                snapshot.owner_id, snapshot.session_id) != (owner_id, session_id):
            raise SourceError("observation_input_scope", "Analysis input belongs to another owner or session.", 403)
        if await connection.fetchval(
                "select lock_jarvis_copy_source($1,$2)", owner_id, snapshot.payload_id) != "available":
            raise SourceError("observation_input_removed", "The visual analysis used removed or superseded context.")
        raw = await connection.fetchval(
            "select content from jarvis_context_payloads where owner_id=$1 and id=$2", owner_id, snapshot.payload_id)
        content = json_object(raw)
        document = content.get("request", {})
        if (content.get("format") != "jarvis.visual_input.v1"
                or content.get("request_digest") != snapshot.request_digest
                or digest(document) != snapshot.request_digest
                or document.get("owner_id") != owner_id or document.get("session_id") != str(session_id)
                or tuple(item["frame_id"] for item in document.get("frames", ())) != tuple(map(str, frame_ids))):
            raise SourceError("observation_input_mismatch", "The analysis input does not match its original frame window.")

    async def validate(self, request, snapshot):
        if digest(request_document(request)) != snapshot.request_digest:
            raise SourceError("observation_input_changed", "The prepared visual request was changed before submission.")
        async with self.context.publication.transaction(self.context.tasks, request.owner_id) as connection:
            await self.check(connection, snapshot, owner_id=request.owner_id, session_id=request.session_id,
                             frame_ids=tuple(frame.frame_id for frame in request.frames))

    async def discard(self, snapshot):
        """Remove only an unused input, including uncertain publication outcomes."""
        async with self.context.publication.transaction(self.context.tasks, snapshot.owner_id) as connection:
            await connection.execute(
                """update jarvis_context_payloads p set content=null,availability='deleted'
                    where p.owner_id=$1 and p.id=$2 and p.availability<>'deleted'
                        and not exists(select 1 from jarvis_context_payload_dependencies d
                            where d.owner_id=p.owner_id and d.source_payload_id=p.id)""",
                snapshot.owner_id, snapshot.payload_id)
