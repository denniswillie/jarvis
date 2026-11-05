"""Bounded release of owner quiet through existing attention intents/attempts.

There is no scheduler, second queue, inferred resume or delivery receipt here.
Claims and writer guards call this under the canonical task-owner transaction.
An episode gets one immutable summary (including an empty, dismissed tombstone).
One automatic claim is permitted across voice/phone; a crashed or ambiguous
summary is not retried. Explicit inbox replay addresses original items instead.
"""
from pydantic import ValidationError

from app.attention.models import AttentionError
from app.attention.notifications import NotificationOutput, QuietSummaryContent, QuietSummaryMember
from app.attention.quiet import read_owner_quiet
from app.migrations.runner import attention_quiet_schema_ready
from app.tasks.models import json_object


NAMESPACE = "quiet_resume_summary"
MAX_MEMBERS = 3
SCAN_LIMIT = 50
MEMBER_FIELDS = ("generation", "source_namespace", "source_id", "source_revision",
                 "source_payload_id", "task_revision")


def summary_message(members):
    labels = (("task_result", "task update"), ("reminder_event", "reminder"),
              ("visual_alert", "visual notice"), ("prospective_event", "tracked-event notice"))
    parts = []
    for namespace, label in labels:
        count = sum(member.source_namespace == namespace for member in members)
        if count:
            parts.append(f"{count} {label}" + ("s" if count != 1 else ""))
    return "Quiet-mode summary: " + ", ".join(parts) + " available in your inbox." if parts else ""


async def _exposed(connection, row):
    # A prior submission is ambiguous even without PCM, playback or an ACK.
    return await connection.fetchval("""select exists(select 1 from jarvis_attention_attempts
        where owner_id=$1 and intent_id=$2 and
        (submitted_at is not null or forwarded_samples>0 or coalesce(played_samples,0)>0
         or phase in ('submitted','forwarding','awaiting_receipt','played','accepted','unknown')))""",
        row["owner_id"], row["id"])


async def _eligible(attention, connection, row, task, now, *, lock=False):
    if row["source_namespace"] not in {"task_result", "reminder_event", "visual_alert", "prospective_event"}:
        return False
    if row["state"] not in {"pending", "failed", "claimed"} or row["attempt_count"] >= attention.max_attempts:
        return False
    if row["not_before"] > now or row["expires_at"] is not None and row["expires_at"] <= now:
        return False
    if task is not None and task.status not in {"succeeded", "failed", "waiting_for_input", "outcome_unknown"}:
        return False
    if await _exposed(connection, row):
        return False
    output, problem = await attention._source(connection, row, task, lock=lock)
    return problem is None and output is not None and output.speech_ready


async def coalesce_ended_episode(attention, connection, owner, decision):
    """Return False on missing release authority/schema, never allow a backlog.

    Legacy inactive owners do not require schema 42 or query its columns. An
    explicit resume after source withdrawal cannot reconstruct a lost episode:
    that case remains fail closed rather than manufacturing an episode boundary.
    """
    if decision.reason not in {"owner_quiet_resumed", "owner_quiet_expired"}:
        return decision.allowed
    snapshot = await read_owner_quiet(connection, owner)
    if snapshot.suppresses_unsolicited or snapshot.policy is None:
        return False
    episode = snapshot.policy.episode
    if episode.episode_id is None or not await attention_quiet_schema_ready(connection):
        return False
    cutoff = episode.resumed_at if episode.state == "resumed" else episode.expires_at
    if episode.expires_at is not None:
        cutoff = min(cutoff, episode.expires_at)
    # Duty pause/snooze remains separate and is re-evaluated even on writer entry.
    await connection.execute("select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", owner)
    existing = await connection.fetchval("""select id from jarvis_attention_intents
        where owner_id=$1 and source_namespace=$2 and source_id=$3 and source_revision=1""",
        owner, NAMESPACE, episode.episode_id)
    if existing is not None:
        return True
    candidates = await connection.fetch("""select id from jarvis_attention_intents i
        where owner_id=$1 and quiet_episode_id is null and created_at<=$2
          and state in ('pending','failed','claimed') and source_namespace<>$3
          and not exists(select 1 from jarvis_attention_requests r
              where r.owner_id=i.owner_id and r.intent_id=i.id and r.operation='replay'
                and r.accepted_generation=i.generation and r.created_at>$2)
        order by case urgency when 'high' then 0 else 1 end,created_at,id limit $4""",
        owner, cutoff, NAMESPACE, SCAN_LIMIT)
    members, expiries = [], []
    for candidate in candidates:
        row, task = await attention._entry(connection, owner, candidate["id"], lock=True)
        if await _eligible(attention, connection, row, task, snapshot.evaluated_at, lock=True):
            members.append(QuietSummaryMember(intent_id=row["id"], **{key: row[key] for key in MEMBER_FIELDS}))
            if row["expires_at"] is not None:
                expiries.append(row["expires_at"])
            if len(members) == MAX_MEMBERS:
                break
    content = QuietSummaryContent(source_id=episode.episode_id, cutoff=cutoff,
                                  members=tuple(members), message=summary_message(members))
    payload = await attention.tasks.payload(connection, owner, content.model_dump(mode="json"),
        source_payload_ids=tuple(member.source_payload_id for member in members))
    summary = await connection.fetchval("""insert into jarvis_attention_intents
        (owner_id,source_namespace,source_id,source_revision,source_payload_id,privacy,state,expires_at,reason)
        values($1,$2,$3,1,$4,'sensitive',$5,$6,$7)
        on conflict(owner_id,source_namespace,source_id,source_revision) do nothing returning id""",
        owner, NAMESPACE, episode.episode_id, payload, "pending" if members else "dismissed",
        min(expiries) if expiries else None, None if members else "quiet_no_eligible_updates")
    if summary is None:
        raise AttentionError("quiet_summary_conflict", "The existing owner transaction must serialize release.")
    # Include items beyond the bounded scan and older summaries. Do not resurrect
    # them later when a duty unpauses, a session reconnects or a lease expires.
    await connection.execute("""update jarvis_attention_intents i
        set quiet_episode_id=$3,quiet_summary_intent_id=$4
        where owner_id=$1 and created_at<=$2 and id<>$4 and quiet_episode_id is null
          and state in ('pending','failed','unknown','claimed','awaiting_receipt')
          and not exists(select 1 from jarvis_attention_requests r
              where r.owner_id=i.owner_id and r.intent_id=i.id and r.operation='replay'
                and r.accepted_generation=i.generation and r.created_at>$2)""",
        owner, cutoff, episode.episode_id, summary)
    await connection.execute("""update jarvis_attention_attempts a
        set phase=case when submitted_at is null then 'failed' else 'unknown' end,
            reason='quiet_episode_coalesced',ended_at=clock_timestamp()
        from jarvis_attention_intents i where i.owner_id=$1 and i.quiet_summary_intent_id=$2
          and a.owner_id=i.owner_id and a.intent_id=i.id
          and a.phase in ('claimed','submitted','forwarding','awaiting_receipt')""", owner, summary)
    await connection.execute("""update jarvis_attention_intents i
        set state=case when exists(select 1 from jarvis_attention_attempts a
            where a.owner_id=i.owner_id and a.intent_id=i.id and a.submitted_at is not null)
            then 'unknown' else 'failed' end, reason='quiet_episode_coalesced'
        where owner_id=$1 and quiet_summary_intent_id=$2 and state in ('claimed','awaiting_receipt')""",
        owner, summary)
    return True


async def _content(connection, row):
    payload = await connection.fetchrow("""select content,availability from jarvis_context_payloads
        where owner_id=$1 and id=$2""", row["owner_id"], row["source_payload_id"])
    if payload is None or payload["availability"] != "available" or payload["content"] is None:
        return None
    try:
        content = QuietSummaryContent.model_validate(json_object(payload["content"]))
    except (ValidationError, ValueError, TypeError):
        return None
    if (content.source_id != row["source_id"] or row["source_revision"] != 1
            or content.message != summary_message(content.members)
            or len({member.intent_id for member in content.members}) != len(content.members)):
        return None
    return content


async def lock_summary_members(attention, connection, row):
    # Dependency removal locks originals before their derived summary payload.
    # Acquire those same source/task/intent fences before the summary's fence.
    content = await _content(connection, row)
    if content:
        for member in sorted(content.members, key=lambda item: str(item.intent_id)):
            await attention._entry(connection, row["owner_id"], member.intent_id, lock=True)


async def summary_source(attention, connection, row, *, lock=False):
    content = await _content(connection, row)
    if content is None or not content.members or row.get("quiet_episode_id") is not None:
        return None, "quiet_summary_unavailable"
    snapshot = await read_owner_quiet(connection, row["owner_id"])
    if (snapshot.suppresses_unsolicited or snapshot.policy is None
            or snapshot.policy.episode.episode_id != content.source_id):
        return None, "quiet_summary_superseded"
    for member in content.members:
        original, task = await attention._entry(connection, row["owner_id"], member.intent_id, lock=lock)
        if (original.get("quiet_summary_intent_id") != row["id"]
                or any(original[key] != getattr(member, key) for key in MEMBER_FIELDS)
                or not await _eligible(attention, connection, original, task, snapshot.evaluated_at, lock=lock)):
            return None, "quiet_summary_member_changed"
    return NotificationOutput(content), None
