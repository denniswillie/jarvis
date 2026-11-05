"""Visual producer/source adapters for the existing durable attention ledger."""
from datetime import timedelta
import json
from uuid import UUID

from pydantic import ValidationError

from app.attention.models import AttentionError
from app.attention.notifications import (
    NotificationOutput, PerceptionFocusCondition, VisualNotificationContent,
)
from app.tasks.models import json_object


VISUAL_NAMESPACES = {"visual_alert", "prospective_event"}


async def _perception_focus_problem(connection, focus, subject, *, content=None):
    if subject is None or subject["generation"] != focus.generation:
        return "perception_focus_changed"
    # A prospective delivery may already have settled its own find-once event.
    # This is only the existing receipt/replay exception, never new authority to
    # complete a focus or permission for a generic alert to reuse completion.
    own_completion = (
        content is not None and content.source_namespace == "prospective_event"
        and content.subject_type == "visual_focus" and content.subject_id == focus.subject_id
        and subject["state"] == "completed"
        and subject["completed_event_id"] == str(content.source_id)
    )
    if subject["state"] != "active" and not own_completion:
        return "perception_focus_changed"
    if subject["expires_at"] is not None and await connection.fetchval(
            "select $1::timestamptz<=clock_timestamp()", subject["expires_at"]):
        return "perception_focus_expired"
    return None


async def enqueue_visual(tasks, connection, *, owner_id, kind, event, observation_payload_id,
                         max_age_seconds, perception_focus=None):
    focus = None
    if perception_focus is not None:
        try:
            focus = PerceptionFocusCondition.model_validate(perception_focus)
        except (ValidationError, ValueError, TypeError) as error:
            raise AttentionError("visual_attention_focus_invalid", "A canonical focus binding is required.") from error
    if observation_payload_id is None or type(max_age_seconds) not in (int, float) or not 1 <= max_age_seconds <= 3600:
        raise AttentionError("visual_attention_source_required", "A current observation and freshness policy are required.")
    prospective = kind == "prospective"
    evidence = json_object(event["evidence"]) if prospective else {}
    identities = tuple(UUID(str(value)) for value in evidence.get("supporting_frame_ids", ())) if prospective else (event["visual_frame_id"],)
    frames = await connection.fetch(
        """select id,captured_at,server_received_at from visual_frames
           where owner_id=$1 and session_id=$2 and id=any($3::uuid[])""",
        owner_id, event["session_id"], list(identities))
    if not identities or len(identities) > 64 or len(frames) != len(set(identities)):
        raise AttentionError("visual_attention_source_mismatch", "The notification requires its actual owned frames.")
    captured = min(frame["captured_at"] for frame in frames)
    received = max((frame["server_received_at"] for frame in frames if frame["server_received_at"]), default=None)
    expires = captured + timedelta(seconds=max_age_seconds)
    if focus is not None:
        # The producer's effect guard owns the canonical transaction/locks. We
        # retain only its exact backend reference; no inferred or latest focus.
        focus_subject = await connection.fetchrow(
            """select * from jarvis_attention_visual_subjects where owner_id=$1
               and subject_type=$2 and subject_id=$3""", owner_id, "visual_focus", focus.subject_id)
        problem = await _perception_focus_problem(connection, focus, focus_subject)
        if problem:
            raise AttentionError(problem, "The perception assignment is no longer current.")
        if focus_subject["expires_at"] is not None:
            expires = min(expires, focus_subject["expires_at"])
    subject = None
    if prospective:
        subject_type = "visual_focus" if event["visual_focus_id"] else "conditional_reminder"
        subject_id = event["visual_focus_id"] or event["reminder_id"]
        subject = await connection.fetchrow(
            """select * from jarvis_attention_visual_subjects where owner_id=$1
               and subject_type=$2 and subject_id=$3""", owner_id, subject_type, subject_id)
        if subject is None or subject["state"] != "active" or subject["claim_token"] != event["claim_token"]:
            raise AttentionError("visual_attention_subject_changed", "The tracked subject is no longer current.")
        if subject["expires_at"] is not None:
            expires = min(expires, subject["expires_at"])
    namespace = "prospective_event" if prospective else "visual_alert"
    content = VisualNotificationContent(source_namespace=namespace, source_id=event["id"],
        session_id=event["session_id"], observation_payload_id=observation_payload_id,
        message=event["message"], supporting_frame_ids=identities, source_captured_at=captured,
        source_received_at=received, expires_at=expires,
        perception_focus=focus,
        **({"subject_type":subject_type, "subject_id":subject_id,
            "subject_generation":subject["generation"], "claim_token":event["claim_token"]} if prospective else {}))
    payload = await tasks.payload(connection, owner_id, content.model_dump(mode="json"))
    await connection.execute("""insert into jarvis_context_payload_dependencies
        (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
        owner_id, observation_payload_id, payload)
    return await connection.fetchval(
        """insert into jarvis_attention_intents(owner_id,source_namespace,source_id,source_revision,
           source_payload_id,preferred_session_id,privacy,expires_at,coalesce_key)
           values($1,$2,$3,1,$4,$5,'sensitive',$6,$7) returning id""",
        owner_id, namespace, event["id"], payload, event["session_id"], expires,
        "observation:" + str(observation_payload_id) if not prospective else None)


async def visual_source(connection, row, *, lock=False):
    raw = await connection.fetchrow(
        """select availability,content from jarvis_context_payloads where owner_id=$1 and id=$2"""
        + (" for share" if lock else ""), row["owner_id"], row["source_payload_id"])
    if raw is None or raw["availability"] != "available" or raw["content"] is None:
        return None, "source_payload_unavailable"
    try:
        content = VisualNotificationContent.model_validate(json_object(raw["content"]))
    except (ValidationError, ValueError, TypeError):
        return None, "notification_source_invalid"
    if (content.source_id != row["source_id"] or content.source_namespace != row["source_namespace"]
            or row["source_revision"] != 1 or content.expires_at.tzinfo is None
            or content.source_captured_at.tzinfo is None):
        return None, "notification_source_invalid"
    prospective = content.source_namespace == "prospective_event"
    table = "jarvis_prospective_delivery_events" if prospective else "visual_interrupts"
    valid = await connection.fetchval(
        f"""select exists(select 1 from {table} e join jarvis_visual_delivery_sources b
            on b.owner_id=e.owner_id and b.delivery_id=e.id and b.delivery_kind=$4
            where e.owner_id=$1 and e.id=$2 and e.message=$3 and b.payload_id=$5 and b.session_id=$6)""",
        row["owner_id"], row["source_id"], content.message,
        "prospective" if prospective else "visual", content.observation_payload_id, content.session_id)
    if not valid:
        return None, "notification_source_changed"
    # Load only this notification's dependencies, in stable order. FOR SHARE
    # lasts through the caller's existing transaction and actual output write.
    # An in-memory cache or an available historical payload is not applicability.
    references = set()
    if prospective:
        references.add((content.subject_type, content.subject_id))
    if content.perception_focus is not None:
        references.add(("visual_focus", content.perception_focus.subject_id))
    subjects = {}
    for subject_type, subject_id in sorted(references, key=lambda value: (str(value[0]), str(value[1]))):
        subjects[subject_type, subject_id] = await connection.fetchrow(
            """select * from jarvis_attention_visual_subjects where owner_id=$1
               and subject_type=$2 and subject_id=$3""" + (" for share" if lock else ""),
            row["owner_id"], subject_type, subject_id)
    if content.perception_focus is not None:
        problem = await _perception_focus_problem(connection, content.perception_focus,
            subjects["visual_focus", content.perception_focus.subject_id], content=content)
        if problem:
            return None, problem
    if prospective:
        # Only this small operational fence is held through the actual write.
        # Raw event/subject reconciliation happens AFTER the audio transaction.
        subject = subjects[content.subject_type, content.subject_id]
        if subject is None or subject["generation"] != content.subject_generation:
            return None, "visual_subject_changed"
        owned_active = subject["state"] == "active" and subject["claim_token"] == content.claim_token
        own_completion = subject["state"] == "completed" and subject["completed_event_id"] == str(content.source_id)
        if not (owned_active or own_completion):
            return None, "visual_subject_changed"
        if subject["expires_at"] is not None and await connection.fetchval(
                "select $1::timestamptz<=clock_timestamp()", subject["expires_at"]):
            return None, "visual_subject_expired"
    if await connection.fetchval("select $1::timestamptz<=clock_timestamp()", content.expires_at):
        return None, "visual_evidence_expired"
    return NotificationOutput(content), None


async def reconcile_visual_notifications(pool, owner, *, limit=50):
    """Project durable delivery outcomes, not model claims, into search bookkeeping.

    No source/intent/audio locks are held here. Event -> subject -> subject fence
    matches source-copy removal. The attention receipt remains canonical even if
    this transaction fails; the existing coordinator retries on its next poll.
    """
    async with pool.acquire() as connection, connection.transaction():
        rows = await connection.fetch(
            """select e.id from jarvis_prospective_delivery_events e
               join jarvis_attention_intents i on i.owner_id=e.owner_id
                 and i.source_namespace='prospective_event' and i.source_id=e.id
               where e.owner_id=$1 and e.event_type in ('generated','dispatched_to_session')
                 and i.state in ('played','dismissed','revoked','expired')
               order by e.created_at,e.id limit $2 for update of e skip locked""", owner, limit)
        for candidate in rows:
            event = await connection.fetchrow("select * from jarvis_prospective_delivery_events where owner_id=$1 and id=$2",
                                               owner, candidate["id"])
            intent = await connection.fetchrow(
                """select * from jarvis_attention_intents where owner_id=$1
                   and source_namespace='prospective_event' and source_id=$2""", owner, event["id"])
            if intent is None or intent["state"] not in {"played","dismissed","revoked","expired"}:
                continue
            delivered = intent["state"] == "played"
            evidence = json_object(event["evidence"]) | {
                "attention_event_id":str(event["id"]), "attention_intent_id":str(intent["id"]),
                "playback_status":"client_confirmed" if delivered else "unconfirmed",
                "physical_action_status":"not_established"}
            if event["visual_focus_id"] is not None:
                await connection.execute(
                    """update jarvis_visual_focuses set
                       completed_at=case when $4 and mode='find_once' then coalesce(completed_at,clock_timestamp())
                                         else completed_at end,
                       completion_evidence=case when $4 and mode='find_once' then $5::jsonb else completion_evidence end,
                       trigger_claim_token=null,trigger_claimed_at=null
                       where owner_id=$1 and id=$2 and trigger_claim_token=$3
                         and completed_at is null and cleared_at is null""",
                    owner, event["visual_focus_id"], event["claim_token"], delivered, json.dumps(evidence))
            else:
                await connection.execute(
                    """update jarvis_reminders set
                       status=case when $4 then 'completed' else status end,
                       completed_at=case when $4 then coalesce(completed_at,clock_timestamp()) else completed_at end,
                       triggered_at=case when $4 then coalesce(triggered_at,clock_timestamp()) else triggered_at end,
                       metadata=case when $4 then metadata || jsonb_build_object(
                         'attention_completed_event_id',$5::text,'physical_action_status','not_established') else metadata end,
                       trigger_claim_token=null,trigger_claimed_at=null,updated_at=clock_timestamp()
                       where owner_id=$1 and id=$2 and trigger_claim_token=$3
                         and status='active' and triggered_at is null""",
                    owner, event["reminder_id"], event["claim_token"], delivered, str(event["id"]))
            await connection.execute(
                """update jarvis_prospective_delivery_events set event_type=$3,
                   metadata=metadata || $4::jsonb,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                owner, event["id"], "delivered_to_session" if delivered else "missed",
                json.dumps({"delivery_route":"attention","attention_state":intent["state"],
                            "playback_status":"client_confirmed" if delivered else "unconfirmed",
                            "physical_action_status":"not_established"}))
