"""Publish actual analyzer observations without running a second memory engine."""
from contextlib import asynccontextmanager

from app.images import ImageReference
from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputCapture
from app.tasks.context_events import ContextIngress
from app.tasks.models import json_object


OBSERVATION_NAMESPACE = "perception:analyzer:v1"


class UnavailableObservationPublication:
    async def prepare_inputs(self, request, context):
        raise SourceError("knowledge_unavailable", "Visual source capture is unavailable.", 503)

    @asynccontextmanager
    async def transaction(self, owner_id):
        raise SourceError("knowledge_unavailable",
                          "Visual memory capture requires the configured source repository.", 503)
        yield  # Keep the same async context-manager contract; never admit a write.


class ObservationPublication:
    def __init__(self, context, *, owner_id, access_check=None):
        if context.publication is None:
            raise ValueError("Observation publication requires canonical source publication")
        self.context, self.owner_id, self.access_check = context, owner_id, access_check
        self.inputs = ObservationInputCapture(context)

    async def prepare_inputs(self, request, context):
        self._authorize(request.owner_id)
        return await self.inputs.prepare(request, context)

    async def validate_input(self, request, snapshot):
        self._authorize(request.owner_id)
        await self.inputs.validate(request, snapshot)

    async def discard_input(self, snapshot):
        self._authorize(snapshot.owner_id)
        await self.inputs.discard(snapshot)

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("knowledge_owner_mismatch", "Observation is outside the runtime owner.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    @asynccontextmanager
    async def transaction(self, owner_id):
        self._authorize(owner_id)
        async with self.context.publication.transaction(self.context.tasks, owner_id) as connection:
            yield connection
            self._authorize(owner_id)

    async def publish(self, connection, observation, links, *, analysis_input=None):
        self._authorize(observation.owner_id)
        if not connection.is_in_transaction():
            raise RuntimeError("Observation publication must share the original transaction")
        if not links or len(links) > 64:
            raise SourceError("observation_evidence_invalid", "Observation needs its bounded original frame window.")
        frame_ids = [link.visual_frame_id for link in links]
        if analysis_input is not None:
            await self.inputs.check(connection, analysis_input, owner_id=observation.owner_id,
                                    session_id=observation.session_id, frame_ids=tuple(frame_ids))
        frames = await connection.fetch(
            """select id,session_id,captured_at,content_type,width,height,raw_image_expires_at,
                      visual_session_id,frame_sequence,server_received_at
                from visual_frames where owner_id=$1 and session_id=$2 and id=any($3::uuid[])
                order by captured_at,id for share""",
            observation.owner_id, observation.session_id, frame_ids)
        if len(frames) != len(frame_ids) or any(
                frame["captured_at"] < observation.started_at or frame["captured_at"] > observation.ended_at
                for frame in frames):
            raise SourceError("observation_evidence_invalid", "Frame ownership or original capture interval disagrees.")
        original = await connection.fetchval(
            "select to_jsonb(o) from perception_observations o where owner_id=$1 and id=$2",
            observation.owner_id, observation.id)
        if original is None:
            raise SourceError("observation_original_missing", "The analyzer original must be stored first.")
        images = [ImageReference(
            frame_id=frame["id"], owner_id=observation.owner_id, source="historical",
            **{key: frame[key] for key in (
                "session_id", "captured_at", "content_type", "width", "height", "raw_image_expires_at",
                "visual_session_id", "frame_sequence", "server_received_at")},
        ).to_metadata() for frame in frames]
        record = await self.context.append_in_transaction(connection, observation.owner_id, ContextIngress(
            session_id=observation.session_id, source_namespace=OBSERVATION_NAMESPACE,
            source_event_id=str(observation.id), receipt_id=str(observation.id),
            kind="observation", trust="derived", source_time=observation.ended_at, expected_revision=0,
            source_payload_ids=(analysis_input.payload_id,) if analysis_input is not None else (),
            payload={
                "origin": "perception_observations", "record": json_object(original), "images": images,
                "analysis_input_id": str(analysis_input.payload_id) if analysis_input is not None else None,
                "input_provenance": "captured" if analysis_input is not None else "not_supplied_by_trusted_caller",
                "frame_links": [{"frame_id": str(link.visual_frame_id), "ordinal": link.ordinal,
                                 "role": link.role} for link in links],
                "meaning": (
                    "Model interpretation of the linked camera frames, informed by the analyzer's session "
                    "context; not a user report or independent proof of a physical action. Confidence, "
                    "uncertainty and capture times belong to the original record. Image metadata does not "
                    "guarantee pixels are still retained; image retrieval must check current availability."),
            }))
        await connection.execute(
            """insert into jarvis_observation_sources(owner_id,observation_id,session_id,payload_id)
                values($1,$2,$3,$4)""",
            observation.owner_id, observation.id, observation.session_id, record.payload_id)
        return record
