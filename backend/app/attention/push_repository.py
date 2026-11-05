"""Device ownership and phone attempts using the existing attention ledger."""
from dataclasses import replace

import asyncpg

from app.attention.models import AttentionError
from app.attention.push_models import PushAuthorization, PushClaim, PushOutcome, PushRequest
from app.attention.quiet import quiet_delivery_decision
from app.attention.quiet_resume import coalesce_ended_episode
from app.tasks.models import digest


class PushRepository:
    def __init__(self, attention, configuration):
        self.attention, self.tasks = attention, attention.tasks
        self.configuration = configuration

    @staticmethod
    def public_device(row):
        return {key: row[key] for key in ("id", "revision", "environment", "topic", "permission",
            "enabled", "quiet_until", "invalid_reason", "registered_at", "updated_at")}

    async def devices(self, owner):
        async with self.tasks.read_transaction() as connection:
            rows = await connection.fetch("select * from jarvis_push_devices where owner_id=$1 order by updated_at desc,id", owner)
        return [self.public_device(row) for row in rows]

    async def change_device(self, owner, identity, request, *, revoke=False):
        if not revoke and (request.environment != self.configuration.environment or request.topic != self.configuration.topic):
            raise AttentionError("push_registration_mismatch", "The app must match this server's APNs environment and topic.")
        parameters = request.model_dump(mode="json", exclude={"request_id", "token"})
        parameters.update(device_id=str(identity), revoke=revoke,
                          token_digest=None if revoke else request.token_hash(),
                          quiet_until_supplied=not revoke and "quiet_until" in request.model_fields_set)
        signature = digest(parameters)
        try:
            async with self.tasks.transaction(owner) as connection:
                row = await connection.fetchrow("select * from jarvis_push_devices where id=$1 for update", identity)
                if row and row["owner_id"] != owner:
                    raise AttentionError("push_device_not_found", "No owned device registration exists.", 404)
                previous = await connection.fetchrow(
                    "select * from jarvis_push_device_requests where owner_id=$1 and request_id=$2", owner, request.request_id)
                if previous:
                    if previous["parameters_digest"] != signature:
                        raise AttentionError("push_request_conflict", "This request identity was already used differently.")
                    return {"device": self.public_device(row), "accepted_revision": previous["accepted_revision"],
                            "replayed_request": True}
                if (row["revision"] if row else 0) != request.expected_revision:
                    raise AttentionError("push_device_revision_conflict", "Refresh the device registration before changing it.")
                if revoke:
                    if row is None:
                        raise AttentionError("push_device_not_found", "No owned device registration exists.", 404)
                    row = await connection.fetchrow("""update jarvis_push_devices set revision=revision+1,
                        enabled=false,token=null,token_digest=null,invalid_reason='revoked_by_owner',
                        updated_at=clock_timestamp() where owner_id=$1 and id=$2 returning *""", owner, identity)
                else:
                    if request.enabled and await connection.fetchval(
                        "select count(*) from jarvis_push_devices where owner_id=$1 and enabled and id<>$2", owner, identity) >= 10:
                        raise AttentionError("push_device_limit", "Revoke an old device before enabling another one.")
                    token = request.token.get_secret_value() if request.enabled else None
                    token_digest = request.token_hash() if token else None
                    row = await connection.fetchrow("""insert into jarvis_push_devices
                        (id,owner_id,revision,environment,topic,token,token_digest,permission,enabled,quiet_until)
                        values($1,$2,1,$3,$4,$5,$6,$7,$8,$9)
                        on conflict(id) do update set revision=jarvis_push_devices.revision+1,
                        environment=excluded.environment,topic=excluded.topic,token=excluded.token,
                        token_digest=excluded.token_digest,permission=excluded.permission,enabled=excluded.enabled,
                        quiet_until=case when $10::boolean then excluded.quiet_until else jarvis_push_devices.quiet_until end,
                        invalid_reason=null,registered_at=clock_timestamp(),
                        updated_at=clock_timestamp() returning *""",
                        identity, owner, request.environment, request.topic, token, token_digest,
                        request.permission, request.enabled, request.quiet_until, "quiet_until" in request.model_fields_set)
                await connection.execute("""insert into jarvis_push_device_requests
                    (owner_id,request_id,device_id,parameters_digest,accepted_revision) values($1,$2,$3,$4,$5)""",
                    owner, request.request_id, identity, signature, row["revision"])
                return {"device": self.public_device(row), "accepted_revision": row["revision"], "replayed_request": False}
        except asyncpg.UniqueViolationError:
            raise AttentionError("push_registration_conflict", "This device address already has an active registration.") from None

    async def _project(self, connection, owner, identity, generation):
        state = await connection.fetchval("""select phase from jarvis_attention_attempts
            where owner_id=$1 and intent_id=$2 and generation=$3 and channel='phone'
              and phase in ('claimed','submitted','accepted','unknown')
            order by case phase when 'accepted' then 0 when 'unknown' then 1 when 'submitted' then 2 else 3 end
            limit 1""", owner, identity, generation)
        await connection.execute("""update jarvis_attention_intents set phone_notice_generation=$3,
            phone_notice_state=$4 where owner_id=$1 and id=$2 and generation=$5""",
            owner, identity, generation if state else None, state, generation)

    async def _recover(self, connection, owner):
        expired = await connection.fetch("""select id,intent_id,generation from jarvis_attention_attempts
            where owner_id=$1 and channel='phone' and phase in ('claimed','submitted')
              and lease_until<=clock_timestamp() order by intent_id,id""", owner)
        for item in expired:
            await self.attention._entry(connection, owner, item["intent_id"], lock=True)
            recovered = await connection.fetchrow("""update jarvis_attention_attempts set
                phase=case when submitted_at is null then 'failed' else 'unknown' end,
                reason='push_lease_expired',ended_at=clock_timestamp()
                where owner_id=$1 and id=$2 and phase in ('claimed','submitted') and lease_until<=clock_timestamp()
                returning phase""", owner, item["id"])
            if recovered:
                await connection.execute("""update jarvis_push_submissions set retry_at=
                    case when $3 then clock_timestamp() else null end where owner_id=$1 and attempt_id=$2""",
                    owner, item["id"], recovered["phase"] == "failed")
                await self._project(connection, owner, item["intent_id"], item["generation"])

    async def recover(self, owner):
        async with self.tasks.transaction(owner) as connection:
            await self.attention._recover(connection, owner)
            await self._recover(connection, owner)

    async def claim_next(self, owner, worker):
        async with self.tasks.transaction(owner) as connection:
            await self.attention._recover(connection, owner)
            await self._recover(connection, owner)
            await connection.execute("select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", owner)
            quiet = await quiet_delivery_decision(connection, owner, kind="unsolicited", channel="push")
            if not quiet.allowed:
                return None
            if not await coalesce_ended_episode(self.attention, connection, owner, quiet):
                return None
            if await connection.fetchval("""select exists(select 1 from jarvis_attention_attempts
                where owner_id=$1 and channel='phone' and phase in ('claimed','submitted'))""", owner):
                return None
            # Filter eligibility before LIMIT: old notified, expired, quiet or
            # permanently rejected items must not starve newer inbox updates.
            candidates = await connection.fetch("""select i.id,d.id as device_id,d.revision as device_revision
                from jarvis_attention_intents i join jarvis_push_devices d on d.owner_id=i.owner_id
                left join lateral (
                    select a.phase,p.retry_at from jarvis_attention_attempts a
                    join jarvis_push_submissions p on p.owner_id=a.owner_id and p.attempt_id=a.id
                    where a.owner_id=i.owner_id and a.intent_id=i.id and a.generation=i.generation
                      and p.device_id=d.id and p.device_revision=d.revision
                    order by a.created_at desc,a.id desc limit 1
                ) recent on true
                where i.owner_id=$1 and i.state in ('pending','unknown','failed')
                  and not exists(select 1 from jarvis_attention_attempts v
                      where v.owner_id=i.owner_id and v.intent_id=i.id and v.generation=i.generation
                        and v.channel='voice' and (v.submitted_at is not null or
                            v.phase in ('claimed','submitted','forwarding','awaiting_receipt','played','unknown')))
                  and to_jsonb(i)->>'quiet_episode_id' is null
                  and (i.source_namespace<>'quiet_resume_summary' or not exists(
                      select 1 from jarvis_attention_attempts a where a.owner_id=i.owner_id and a.intent_id=i.id))
                  and i.not_before<=clock_timestamp()
                  and (i.expires_at is null or i.expires_at>clock_timestamp())
                  and i.created_at<=clock_timestamp()-$2::double precision*interval '1 second'
                  and d.enabled and d.token is not null and d.permission in ('authorized','provisional','ephemeral')
                  and d.environment=$3 and d.topic=$4 and (d.quiet_until is null or d.quiet_until<=clock_timestamp())
                  and not exists (
                    select 1 from jarvis_attention_attempts a join jarvis_push_submissions p
                      on p.owner_id=a.owner_id and p.attempt_id=a.id
                    where a.owner_id=i.owner_id and a.intent_id=i.id and p.device_id=d.id
                      and a.phase in ('claimed','submitted','accepted','unknown'))
                  and (recent.phase is null or (recent.phase='failed' and recent.retry_at<=clock_timestamp()))
                  and (select count(*) from jarvis_attention_attempts a join jarvis_push_submissions p
                       on p.owner_id=a.owner_id and p.attempt_id=a.id
                       where a.owner_id=i.owner_id and a.intent_id=i.id and a.generation=i.generation
                         and p.device_id=d.id and p.device_revision=d.revision)<$5
                order by case i.urgency when 'high' then 0 else 1 end,i.created_at,i.id,d.id limit 50""",
                owner, self.configuration.fallback_seconds, self.configuration.environment,
                self.configuration.topic, self.configuration.max_attempts)
            for candidate in candidates:
                row, task = await self.attention._entry(connection, owner, candidate["id"], lock=True)
                if row["state"] not in {"pending", "unknown", "failed"}:
                    continue
                output, problem = await self.attention._source(connection, row, task, lock=True)
                if problem or output is None:
                    await connection.execute("""update jarvis_attention_intents set state='revoked',
                        generation=generation+1,reason=$3,updated_at=clock_timestamp()
                        where owner_id=$1 and id=$2""", owner, row["id"], problem or "push_source_unavailable")
                    continue
                identity = await connection.fetchval("""insert into jarvis_attention_attempts
                    (owner_id,intent_id,generation,channel,worker_id,lease_until)
                    values($1,$2,$3,'phone',$4,clock_timestamp()+$5::double precision*interval '1 second')
                    returning id""", owner, row["id"], row["generation"], worker, self.configuration.lease_seconds)
                await connection.execute("""insert into jarvis_push_submissions
                    (owner_id,attempt_id,device_id,device_revision) values($1,$2,$3,$4)""",
                    owner, identity, candidate["device_id"], candidate["device_revision"])
                await self._project(connection, owner, row["id"], row["generation"])
                return PushClaim(owner, row["id"], identity, row["generation"],
                                 candidate["device_id"], candidate["device_revision"], worker)
        return None

    async def prepare_submission(self, claim):
        """Read a current request snapshot without authorizing network I/O."""
        request, _ = await self._submission(claim)
        return request

    async def authorize_send(self, request):
        """Consume the claim immediately before client invocation, never over I/O.

        submitted_at means durable handoff authorization. It is not a wire
        timestamp or proof of APNs acceptance, display or hearing. A second
        authorization of the same attempt is denied, including after restart.
        """
        current, reason = await self._submission(request.claim, authorize=True, expected=request)
        return PushAuthorization(request.claim, current is not None, reason)

    async def _submission(self, claim, *, authorize=False, expected=None):
        async with self.tasks.transaction(claim.owner_id) as connection:
            await connection.execute("select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", claim.owner_id)
            quiet = await quiet_delivery_decision(connection, claim.owner_id, kind="unsolicited", channel="push")
            if not quiet.allowed:
                return None, quiet.reason
            if not await coalesce_ended_episode(self.attention, connection, claim.owner_id, quiet):
                return None, "owner_quiet_release_unavailable"
            row, task = await self.attention._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            if row.get("quiet_episode_id") is not None:
                return None, "owner_quiet_coalesced"
            device = await connection.fetchrow("""select * from jarvis_push_devices where owner_id=$1 and id=$2
                and revision=$3 and enabled and token is not null
                and permission in ('authorized','provisional','ephemeral')
                and (quiet_until is null or quiet_until<=clock_timestamp()) for update""",
                claim.owner_id, claim.device_id, claim.device_revision)
            attempt = await connection.fetchrow("""select a.id from jarvis_attention_attempts a
                join jarvis_push_submissions p on p.owner_id=a.owner_id and p.attempt_id=a.id
                where a.owner_id=$1 and a.id=$2 and a.intent_id=$3 and a.generation=$4 and a.worker_id=$5
                and a.channel='phone' and a.phase='claimed' and a.lease_until>clock_timestamp()
                and p.device_id=$6 and p.device_revision=$7 for update of a""",
                claim.owner_id, claim.attempt_id, claim.intent_id, claim.generation, claim.worker_id,
                claim.device_id, claim.device_revision)
            valid = (device is not None and attempt is not None and row["generation"] == claim.generation
                     and row["state"] in {"pending", "unknown", "failed"}
                     and device["topic"] == self.configuration.topic and device["environment"] == self.configuration.environment)
            if valid:
                valid = await connection.fetchval("""select $1::timestamptz<=clock_timestamp()
                    and ($2::timestamptz is null or $2>clock_timestamp())""", row["not_before"], row["expires_at"])
            if valid:
                valid = not await connection.fetchval("""select exists(select 1 from jarvis_attention_attempts
                    where owner_id=$1 and intent_id=$2 and generation=$3 and channel='voice'
                      and (submitted_at is not null or phase in
                          ('claimed','submitted','forwarding','awaiting_receipt','played','unknown')))""",
                    claim.owner_id, claim.intent_id, claim.generation)
            output, problem = await self.attention._source(connection, row, task, lock=True) if valid else (None, "push_claim_lost")
            if problem or output is None:
                return None, problem or "push_source_unavailable"
            current = PushRequest(claim, device["token"], device["environment"], device["topic"], row["expires_at"], row["urgency"])
            if expected is not None and current != expected:
                return None, "push_request_changed"
            if authorize:
                await connection.execute("""update jarvis_attention_attempts set phase='submitted',
                    submitted_at=clock_timestamp(),reason='push_handoff_authorized'
                    where owner_id=$1 and id=$2""", claim.owner_id, claim.attempt_id)
                await self._project(connection, claim.owner_id, claim.intent_id, claim.generation)
            return current, "push_handoff_authorized" if authorize else "push_prepared_not_authorized"

    async def finish(self, claim, outcome: PushOutcome):
        async with self.tasks.transaction(claim.owner_id) as connection:
            await self.attention._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            device = await connection.fetchrow("select * from jarvis_push_devices where owner_id=$1 and id=$2 for update",
                                              claim.owner_id, claim.device_id)
            attempt = await connection.fetchrow("""select * from jarvis_attention_attempts
                where owner_id=$1 and id=$2 and intent_id=$3 and generation=$4 and worker_id=$5
                and channel='phone' for update""",
                claim.owner_id, claim.attempt_id, claim.intent_id, claim.generation, claim.worker_id)
            if attempt is None:
                raise AttentionError("push_claim_lost", "The phone delivery attempt no longer exists.")
            if (attempt["submitted_at"] is not None and attempt["phase"] in {"claimed", "submitted"}
                    and outcome.state in {"failed", "revoked"} and outcome.provider_status is None):
                # Cancellation may land after COMMIT but before authorize_send
                # returns. Durable evidence, not a caller's local flag, wins.
                outcome = replace(outcome, state="unknown", retry_seconds=None,
                    reason="push_submission_uncertain" if outcome.reason == "push_preparation_failed" else outcome.reason)
            if attempt["phase"] in {"claimed", "submitted"}:
                await connection.execute("""update jarvis_attention_attempts set phase=$3,reason=$4,
                    ended_at=clock_timestamp() where owner_id=$1 and id=$2""",
                    claim.owner_id, claim.attempt_id, outcome.state, outcome.reason)
            # Preserve an actual provider response even if source revocation won
            # while the bounded HTTP request was already in flight.
            if attempt["phase"] in {"claimed", "submitted"} or outcome.provider_status is not None:
                await connection.execute("""update jarvis_push_submissions set provider_id=$3,provider_status=$4,
                    provider_reason=$5,retry_at=case when $6::integer is null then null
                    else clock_timestamp()+$6*interval '1 second' end where owner_id=$1 and attempt_id=$2""",
                    claim.owner_id, claim.attempt_id, outcome.provider_id, outcome.provider_status,
                    outcome.reason, outcome.retry_seconds if attempt["phase"] in {"claimed", "submitted"} else None)
            if outcome.invalidate_token and device and device["revision"] == claim.device_revision:
                if outcome.invalidated_at is None or device["registered_at"] <= outcome.invalidated_at:
                    await connection.execute("""update jarvis_push_devices set enabled=false,token=null,token_digest=null,
                        invalid_reason=$4,revision=revision+1,updated_at=clock_timestamp()
                        where owner_id=$1 and id=$2 and revision=$3""",
                        claim.owner_id, claim.device_id, claim.device_revision, outcome.reason)
            await self._project(connection, claim.owner_id, claim.intent_id, claim.generation)
            return outcome
