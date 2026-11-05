from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID, uuid4

import asyncpg
from app.perception.applicability import effect_connection

from app.db import json_object
from app.knowledge.models import SourceError
from app.live_state.perception import CheckResolution
from app.knowledge.visual_delivery_sources import (
    bind_visual_delivery_source, lock_visual_delivery_source, source_is_current,
)
from app.memory.prospective.models import (
    ProspectiveDeliveryEvent,
    ProspectiveDeliveryOutcome,
    ProspectiveSubjectType,
    VisualFocus,
    VisualFocusMode,
)


class ProspectiveRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def set_visual_focus(
        self,
        *,
        owner_id: str,
        target: str,
        description: str,
        purpose: str,
        mode: VisualFocusMode,
        expires_at: datetime,
        created_from_message_id: UUID | None = None,
    ) -> VisualFocus:
        async with self._pool.acquire() as conn, conn.transaction():
            await _lock_owner(conn, owner_id)
            await conn.execute(
                """
                update jarvis_visual_focuses
                set cleared_at = coalesce(cleared_at, now()),
                    trigger_claim_token = null,
                    trigger_claimed_at = null
                where owner_id = $1
                  and completed_at is null
                  and cleared_at is null
                """,
                owner_id,
            )
            row = await conn.fetchrow(
                """
                insert into jarvis_visual_focuses (
                    owner_id,
                    target,
                    description,
                    purpose,
                    mode,
                    created_from_message_id,
                    expires_at
                )
                values ($1, $2, $3, $4, $5, $6, $7)
                returning *
                """,
                owner_id,
                target,
                description,
                purpose,
                mode,
                created_from_message_id,
                expires_at,
            )
        return _focus_from_record(row)

    async def get_active_visual_focus(self, *, owner_id: str) -> VisualFocus | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                select focus.*, (
                    select jsonb_build_object(
                        'status', case
                            when event.event_type in ('generated', 'dispatched_to_session')
                                 and event.metadata @> '{"dispatch_started": true}'::jsonb
                            then 'awaiting_outcome'
                            else event.event_type
                        end,
                        'reason', event.metadata->>'reason',
                        'updated_at', event.updated_at,
                        'playback_status', coalesce(event.metadata->>'playback_status', 'unconfirmed')
                    )
                    from jarvis_prospective_delivery_events event
                    where event.owner_id = focus.owner_id and event.visual_focus_id = focus.id
                    order by event.created_at desc, event.id desc
                    limit 1
                ) as delivery_status
                from jarvis_visual_focuses focus
                where focus.owner_id = $1
                  and focus.completed_at is null
                  and focus.cleared_at is null
                  and focus.expires_at > now()
                order by focus.created_at desc
                limit 1
                """,
                owner_id,
            )
        return _focus_from_record(row) if row is not None else None

    async def clear_active_visual_focus(
        self, *, owner_id: str, expected_focus_id: UUID | None = None,
    ) -> VisualFocus | None:
        async with self._pool.acquire() as conn, conn.transaction():
            await _lock_owner(conn, owner_id)
            row = await conn.fetchrow(
                """
                update jarvis_visual_focuses
                set cleared_at = coalesce(cleared_at, now()),
                    trigger_claim_token = null,
                    trigger_claimed_at = null
                where id = (
                    select id
                    from jarvis_visual_focuses
                    where owner_id = $1
                      and ($2::uuid is null or id = $2)
                      and completed_at is null
                      and cleared_at is null
                      and ($2::uuid is not null or expires_at > now())
                    order by created_at desc
                    limit 1
                )
                returning *
                """,
                owner_id,
                expected_focus_id,
            )
        return _focus_from_record(row) if row is not None else None

    async def try_claim_subject(
        self,
        *,
        owner_id: str,
        subject_type: ProspectiveSubjectType,
        subject_id: UUID,
        claim_timeout_seconds: int,
    ) -> UUID | None:
        claim_token = uuid4()
        if subject_type == "visual_focus":
            query = """
                update jarvis_visual_focuses
                set trigger_claim_token = $3,
                    trigger_claimed_at = now()
                where owner_id = $1
                  and id = $2
                  and completed_at is null
                  and cleared_at is null
                  and expires_at > now()
                  and not exists (
                      select 1 from jarvis_prospective_delivery_events
                      where owner_id = $1 and visual_focus_id = $2
                        and event_type in ('generated', 'dispatched_to_session')
                        and (metadata @> '{"dispatch_started": true}'::jsonb
                             or metadata @> '{"attention_owned": true}'::jsonb)
                  )
                  and (
                      trigger_claimed_at is null
                      or trigger_claimed_at <= now() - make_interval(secs => $4::double precision)
                  )
                returning trigger_claim_token
            """
        else:
            query = """
                update jarvis_reminders
                set trigger_claim_token = $3,
                    trigger_claimed_at = now(),
                    trigger_attempt_count = trigger_attempt_count + 1,
                    last_trigger_error = null,
                    updated_at = now()
                where owner_id = $1
                  and id = $2
                  and status = 'active'
                  and triggered_at is null
                  and nullif(btrim(condition), '') is not null
                  and (not_before is null or not_before <= now())
                  and jarvis_responsibility_notice_time(owner_id,id,now()) <= now()
                  and not exists (
                      select 1 from jarvis_prospective_delivery_events
                      where owner_id = $1 and reminder_id = $2
                        and event_type in ('generated', 'dispatched_to_session')
                        and (metadata @> '{"dispatch_started": true}'::jsonb
                             or metadata @> '{"attention_owned": true}'::jsonb)
                  )
                  and (
                      trigger_claimed_at is null
                      or trigger_claimed_at <= now() - make_interval(secs => $4::double precision)
                  )
                returning trigger_claim_token
            """
        async with self._pool.acquire() as conn, conn.transaction():
            await _lock_owner(conn, owner_id)
            claimed = await conn.fetchval(
                query,
                owner_id,
                subject_id,
                claim_token,
                claim_timeout_seconds,
            )
        return claim_token if claimed == claim_token else None

    async def reserve_delivery_event(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        subject_type: ProspectiveSubjectType,
        subject_id: UUID,
        claim_token: UUID,
        occurrence_key: str,
        message: str,
        evidence: Mapping[str, Any],
        dedupe_window_seconds: int,
        source_payload_id: UUID | None = None,
        attention=None,
        max_age_seconds: float | None = None,
        effect_guard=None,
        focus_decision=None,
    ) -> ProspectiveDeliveryEvent | None:
        visual_focus_id = subject_id if subject_type == "visual_focus" else None
        reminder_id = subject_id if subject_type == "conditional_reminder" else None
        subject_table = (
            "jarvis_visual_focuses"
            if subject_type == "visual_focus"
            else "jarvis_reminders"
        )
        event_subject_column = (
            "visual_focus_id"
            if subject_type == "visual_focus"
            else "reminder_id"
        )
        active_condition = _active_subject_condition(subject_type)
        async with effect_connection(self._pool, effect_guard) as conn, conn.transaction():
            await lock_visual_delivery_source(conn, owner_id=owner_id, session_id=session_id,
                payload_id=source_payload_id, frame_ids=evidence.get("supporting_frame_ids", ()))
            await _lock_owner(conn, owner_id)
            if focus_decision is not None:
                if subject_type != "visual_focus" or subject_id != focus_decision.focus_id:
                    raise SourceError("focus_decision_subject", "The decision belongs to another watch.")
                await self.apply_focus_decision(conn, owner_id=owner_id, decision=focus_decision)
            owns_claim = await conn.fetchval(
                f"""
                select true
                from {subject_table}
                where owner_id = $1
                  and id = $2
                  and trigger_claim_token = $3
                  and {active_condition}
                for update
                """,
                owner_id,
                subject_id,
                claim_token,
            )
            if owns_claim is not True:
                return None
            recently_reserved = await conn.fetchval(
                f"""
                select exists (
                    select 1
                    from jarvis_prospective_delivery_events
                    where owner_id = $1
                      and {event_subject_column} = $2
                      and created_at > now() - make_interval(
                          secs => $3::double precision
                      )
                )
                """,
                owner_id,
                subject_id,
                dedupe_window_seconds,
            )
            if recently_reserved:
                return None
            row = await conn.fetchrow(
                """
                insert into jarvis_prospective_delivery_events (
                    owner_id,
                    session_id,
                    visual_focus_id,
                    reminder_id,
                    claim_token,
                    occurrence_key,
                    message,
                    evidence
                )
                values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
                on conflict do nothing
                returning *
                """,
                owner_id,
                session_id,
                visual_focus_id,
                reminder_id,
                claim_token,
                occurrence_key,
                message,
                json.dumps(dict(evidence)),
            )
            if row is not None:
                await bind_visual_delivery_source(conn, owner_id=owner_id, session_id=session_id,
                    kind="prospective", delivery_id=row["id"], payload_id=source_payload_id)
                if attention is not None:
                    intent = await attention.enqueue_visual(conn, owner_id=owner_id, kind="prospective", event=row,
                        observation_payload_id=source_payload_id, max_age_seconds=max_age_seconds,
                        **({"perception_focus": effect_guard.focus_fence}
                           if effect_guard is not None and effect_guard.focus_fence is not None else {}))
                    await conn.execute("""update jarvis_prospective_delivery_events set metadata=metadata || $3::jsonb
                        where owner_id=$1 and id=$2""", owner_id, row["id"], json.dumps({
                            "delivery_route":"attention", "attention_owned":True, "attention_intent_id":str(intent),
                            "playback_status":"unconfirmed", "physical_action_status":"not_established"}))
                    if focus_decision is not None:
                        # The queued notification must also lose authority if its
                        # qualified decision is withdrawn, not just its image.
                        await conn.execute("""insert into jarvis_context_payload_dependencies
                            (owner_id,source_payload_id,dependent_payload_id)
                            select owner_id,$3,source_payload_id from jarvis_attention_intents
                            where owner_id=$1 and id=$2 on conflict do nothing""",
                            owner_id, intent, focus_decision.record.payload_id)
        return _event_from_record(row) if row is not None else None

    async def apply_focus_decision(self, connection, *, owner_id, decision):
        """Owner-side qualification on the effect transaction, never a model command.

        The migration advances semantic revision, not attention generation.
        No completed_at/completion_evidence/claim/duty field is written here.
        """
        from app.memory.prospective.focus_evidence import FocusEvidenceDecision
        if not isinstance(decision, FocusEvidenceDecision) or not connection.is_in_transaction():
            raise SourceError("focus_decision_invalid", "A canonical focus decision is required.")
        await _lock_owner(connection, owner_id)
        row = await connection.fetchrow("""select * from jarvis_visual_focuses
            where owner_id=$1 and id=$2 for update""", owner_id, decision.focus_id)
        if row is None:
            raise SourceError("focus_decision_missing", "The focus no longer exists.")
        resolution = await decision.validate(connection, _focus_from_record(row))
        changed = await connection.fetchval("""update jarvis_visual_focuses set check_resolution=$4::jsonb
            where owner_id=$1 and id=$2 and revision=$3 and completed_at is null
              and cleared_at is null and expires_at>clock_timestamp() returning revision""",
            owner_id, decision.focus_id, row["revision"], resolution.model_dump_json())
        if changed is None:
            raise SourceError("focus_decision_changed", "The focus changed before settlement.")
        return changed

    async def record_focus_decision(self, *, owner_id, decision, effect_guard):
        # A legacy match may already own delivery. Still retain qualified typed
        # evidence, without attempting a second reservation or spoken message.
        async with effect_connection(self._pool, effect_guard) as conn, conn.transaction():
            return await self.apply_focus_decision(conn, owner_id=owner_id, decision=decision)

    async def begin_delivery(
        self, *, owner_id: str, event_id: UUID, session_id: UUID,
        source_payload_id: UUID | None = None,
    ) -> bool:
        """Fence unknown native-write outcomes without disabling stale-claim recovery."""
        async with self._pool.acquire() as conn, conn.transaction():
            if not await source_is_current(conn, owner_id, source_payload_id):
                return False
            await _lock_owner(conn, owner_id)
            event = await conn.fetchrow(
                "select * from jarvis_prospective_delivery_events "
                "where owner_id = $1 and id = $2 for update",
                owner_id, event_id,
            )
            if (event is None or event["event_type"] != "generated"
                    or json_object(_decode_json(event["metadata"])).get("attention_owned") is True):
                return False
            is_focus = event["visual_focus_id"] is not None
            subject_type = "visual_focus" if is_focus else "conditional_reminder"
            table = "jarvis_visual_focuses" if is_focus else "jarvis_reminders"
            subject_id = event["visual_focus_id"] if is_focus else event["reminder_id"]
            current = await conn.fetchval(
                f"select true from {table} where owner_id = $1 and id = $2 "
                f"and trigger_claim_token = $3 and {_active_subject_condition(subject_type)} "
                "for update",
                owner_id, subject_id, event["claim_token"],
            )
            if current is not True:
                return False
            # Mark before the network write: losing its acknowledgement is not
            # proof that the user heard nothing. Only an outcome or new intent
            # can release this fence; lease expiry must not replay the speech.
            await conn.execute(
                "update jarvis_prospective_delivery_events "
                "set event_type = 'dispatched_to_session', session_id = $3, "
                "metadata = metadata || $4::jsonb, updated_at = now() "
                "where owner_id = $1 and id = $2",
                owner_id, event_id, session_id,
                json.dumps({"dispatch_started": True, "playback_status": "unconfirmed"}),
            )
        return True

    async def mark_event_dispatched(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        session_id: UUID,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                update jarvis_prospective_delivery_events
                set event_type = 'dispatched_to_session',
                    session_id = $3,
                    updated_at = now()
                where owner_id = $1
                  and id = $2
                  and event_type = 'generated'
                """,
                owner_id,
                event_id,
                session_id,
            )

    async def resolve_delivery_event(
        self,
        *,
        owner_id: str,
        event_id: UUID,
        outcome: ProspectiveDeliveryOutcome,
        provider_response_id: str | None,
        metadata: Mapping[str, Any] | None = None,
        source_payload_id: UUID | None = None,
    ) -> ProspectiveDeliveryEvent | None:
        async with self._pool.acquire() as conn, conn.transaction():
            source_current = await source_is_current(conn, owner_id, source_payload_id)
            await _lock_owner(conn, owner_id)
            event = await conn.fetchrow(
                """
                select *
                from jarvis_prospective_delivery_events
                where owner_id = $1 and id = $2
                for update
                """,
                owner_id,
                event_id,
            )
            if event is None:
                return None
            if json_object(_decode_json(event["metadata"])).get("attention_owned") is True:
                return _event_from_record(event)
            if event["event_type"] in {
                "delivered_to_session",
                "missed",
                "failed",
                "interrupted",
            }:
                return _event_from_record(event)

            if not source_current:
                outcome = "failed"
                metadata = {"reason": "source_unavailable", "playback_status": "unconfirmed"}
            delivered = outcome == "delivered"
            if event["visual_focus_id"] is not None:
                await conn.execute(
                    """
                    update jarvis_visual_focuses
                    set completed_at = case
                            when $4 and mode = 'find_once'
                            then coalesce(completed_at, now())
                            else completed_at
                        end,
                        completion_evidence = case
                            when $4 and mode = 'find_once'
                            then $5::jsonb
                            else completion_evidence
                        end,
                        trigger_claim_token = null,
                        trigger_claimed_at = null
                    where owner_id = $1
                      and id = $2
                      and trigger_claim_token = $3
                      and completed_at is null
                      and cleared_at is null
                    """,
                    owner_id,
                    event["visual_focus_id"],
                    event["claim_token"],
                    delivered,
                    json.dumps(_decode_json(event["evidence"])),
                )
            else:
                await conn.execute(
                    """
                    update jarvis_reminders
                    set status = case when $4 then 'completed' else status end,
                        completed_at = case
                            when $4 then coalesce(completed_at, now())
                            else completed_at
                        end,
                        triggered_at = case
                            when $4 then coalesce(triggered_at, now())
                            else triggered_at
                        end,
                        trigger_claim_token = null,
                        trigger_claimed_at = null,
                        updated_at = now()
                    where owner_id = $1
                      and id = $2
                      and trigger_claim_token = $3
                      and status = 'active'
                      and triggered_at is null
                    """,
                    owner_id,
                    event["reminder_id"],
                    event["claim_token"],
                    delivered,
                )

            event_type = (
                "delivered_to_session" if outcome == "delivered" else outcome
            )
            updated = await conn.fetchrow(
                """
                update jarvis_prospective_delivery_events
                set event_type = $3,
                    provider_response_id = coalesce($4, provider_response_id),
                    metadata = metadata || $5::jsonb,
                    updated_at = now()
                where owner_id = $1 and id = $2
                returning *
                """,
                owner_id,
                event_id,
                event_type,
                provider_response_id,
                json.dumps(dict(metadata or {})),
            )
        return _event_from_record(updated)

    async def release_subject_claim(
        self,
        *,
        owner_id: str,
        subject_type: ProspectiveSubjectType,
        subject_id: UUID,
        claim_token: UUID,
    ) -> None:
        table = (
            "jarvis_visual_focuses"
            if subject_type == "visual_focus"
            else "jarvis_reminders"
        )
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                update {table}
                set trigger_claim_token = null,
                    trigger_claimed_at = null
                where owner_id = $1 and id = $2 and trigger_claim_token = $3
                """,
                owner_id,
                subject_id,
                claim_token,
            )


async def _lock_owner(conn, owner_id: str) -> None:
    # Existing transaction-scoped advisory-lock pattern, including the first
    # concurrent set when no subject row exists to lock.
    await conn.execute(
        "select pg_advisory_xact_lock(hashtextextended($1, 0))",
        "prospective:" + owner_id,
    )


def _active_subject_condition(subject_type: str) -> str:
    if subject_type == "visual_focus":
        return "completed_at is null and cleared_at is null and expires_at > now()"
    return (
        "status = 'active' and triggered_at is null "
        "and nullif(btrim(condition), '') is not null"
    )


def _focus_from_record(record: Mapping[str, Any]) -> VisualFocus:
    evidence = record["completion_evidence"]
    return VisualFocus(
        id=record["id"],
        owner_id=record["owner_id"],
        target=record["target"],
        description=record["description"],
        purpose=record["purpose"],
        mode=record["mode"],
        created_from_message_id=record["created_from_message_id"],
        created_at=record["created_at"],
        expires_at=record["expires_at"],
        completed_at=record["completed_at"],
        cleared_at=record["cleared_at"],
        completion_evidence=(
            json_object(_decode_json(evidence)) if evidence is not None else None
        ),
        trigger_claim_token=record["trigger_claim_token"],
        trigger_claimed_at=record["trigger_claimed_at"],
        delivery_status=(
            json_object(_decode_json(record["delivery_status"]))
            if record.get("delivery_status") is not None else None
        ),
        revision=record.get("revision", 1),
        episode_revision=record.get("episode_revision", 1),
        check_resolution=(CheckResolution.model_validate(_decode_json(record["check_resolution"]))
                          if record.get("check_resolution") is not None else None),
    )


def _event_from_record(record: Mapping[str, Any]) -> ProspectiveDeliveryEvent:
    return ProspectiveDeliveryEvent(
        id=record["id"],
        owner_id=record["owner_id"],
        session_id=record["session_id"],
        visual_focus_id=record["visual_focus_id"],
        reminder_id=record["reminder_id"],
        claim_token=record["claim_token"],
        occurrence_key=record["occurrence_key"],
        message=record["message"],
        event_type=record["event_type"],
        evidence=json_object(_decode_json(record["evidence"])),
        provider_response_id=record["provider_response_id"],
        created_at=record["created_at"],
        updated_at=record["updated_at"],
    )


def _decode_json(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value
