"""Owner/source-fenced inbox and delivery attempts, independent of model memory."""
from contextlib import asynccontextmanager
import asyncio
import logging
from uuid import UUID

from app.attention.admission_diagnostics import completed_claim, stage_admission
from app.attention.models import ACTIVE_PHASES, AttentionError, DeliveryClaim, InboxMutation
from app.attention.quiet import quiet_delivery_decision
from app.attention.quiet_resume import NAMESPACE as QUIET_SUMMARY, coalesce_ended_episode, lock_summary_members, summary_source
from app.attention.notifications import NotificationOutput, enqueue_reminder, reminder_source
from app.attention.visual import VISUAL_NAMESPACES, enqueue_visual, reconcile_visual_notifications, visual_source
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.models import TaskError, TaskStatus, digest, json_object


class AttentionRepository:
    def __init__(self, tasks, *, lease_seconds=30, max_attempts=3):
        if type(lease_seconds) is not int or not 5 <= lease_seconds <= 300:
            raise ValueError("Invalid attention lease")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 20:
            raise ValueError("Invalid attention attempt limit")
        self.tasks, self.pool = tasks, tasks.pool
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts

    async def _entry(self, connection, owner, identity, *, lock=False):
        row = await connection.fetchrow("select * from jarvis_attention_intents where owner_id=$1 and id=$2", owner, identity)
        if row is None:
            raise AttentionError("attention_not_found", "No owned inbox item exists.", 404)
        source_payload = row["source_payload_id"]
        if lock and row["source_namespace"] == QUIET_SUMMARY:
            await lock_summary_members(self, connection, row)
        if lock:
            # Source removal locks the payload before its AFTER trigger revokes
            # intents. Take the same order; intent -> payload can deadlock with
            # deletion even though every individual write is owner-scoped.
            await connection.fetchrow("select id from jarvis_context_payloads where owner_id=$1 and id=$2 for share",
                                      owner, source_payload)
        # Match task publication's lock order: task before intent. Never hold an
        # intent row while waiting for a task whose trigger needs that intent.
        task = await self.tasks.load(connection, owner, row["task_id"], lock=lock) if row["task_id"] else None
        if lock:
            row = await connection.fetchrow("select * from jarvis_attention_intents where owner_id=$1 and id=$2 for update", owner, identity)
            if row["source_payload_id"] != source_payload:
                raise AttentionError("attention_source_changed", "The intent source changed during admission.")
        return row, task

    async def enqueue_reminder(self, connection, *, owner_id, reminder, event):
        return await enqueue_reminder(self.tasks, connection, owner_id=owner_id, reminder=reminder, event=event)

    async def enqueue_visual(self, connection, **kwargs):
        return await enqueue_visual(self.tasks, connection, **kwargs)

    async def reconcile_notifications(self, owner):
        # Playback cannot be un-acknowledged because a projection is delayed.
        try:
            async with asyncio.timeout(5):
                await reconcile_visual_notifications(self.pool, owner)
        except Exception:
            logging.getLogger(__name__).warning("Visual attention projection pending")

    async def _source(self, connection, row, task, *, lock=False):
        if row["source_namespace"] == QUIET_SUMMARY and task is None:
            return await summary_source(self, connection, row, lock=lock)
        if row["source_namespace"] in VISUAL_NAMESPACES and task is None:
            return await visual_source(connection, row, lock=lock)
        if row["source_namespace"] == "reminder_event" and task is None:
            return await reminder_source(connection, row, lock=lock)
        if row["source_namespace"] != "task_result" or task is None:
            return None, "source_adapter_unavailable"
        if task.revision != row["task_revision"] or task.result_payload_id != row["source_payload_id"]:
            return None, "task_result_changed"
        problem = await self.tasks.context_problem(connection, task, lock=lock)
        if problem:
            return None, problem
        manifests = await connection.fetch("""select m.status from jarvis_task_events e
            join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
            join jarvis_task_context_manifests m on m.owner_id=a.owner_id and m.id=a.context_manifest_id
            where e.owner_id=$1 and e.task_id=$2 and e.payload_id=$3""" + (" for share of m" if lock else ""),
            task.owner_id, task.id, row["source_payload_id"])
        if any(manifest["status"] != "active" for manifest in manifests):
            return None, "context_manifest_invalidated"
        payload = await connection.fetchrow("""select content,availability from jarvis_context_payloads
            where owner_id=$1 and id=$2""" + (" for share" if lock else ""), row["owner_id"], row["source_payload_id"])
        if payload is None or payload["availability"] != "available" or payload["content"] is None:
            return None, "source_payload_unavailable"
        content = json_object(payload["content"])
        from app.tasks.note_result import FORMAT as NOTE_RESULT_FORMAT, note_result_source
        if content.get("format") == NOTE_RESULT_FORMAT:
            return await note_result_source(connection, row, task, content)
        try:
            output = inspect_expert_output(content, task=task)
        except TaskError as error:
            # One corrupt result is unavailable, not permission to hide the
            # owner's other valid inbox items or release its unverified prose.
            return None, error.code
        return (output, None) if output.expert_available else (None, output.unavailable_reason)

    @staticmethod
    def _public(row, problem):
        return {key: row[key] for key in ("id", "source_namespace", "task_id", "task_revision", "state",
            "generation", "urgency", "privacy", "attempt_count", "reason", "created_at", "updated_at")} | {
            "source_available":problem is None, "source_unavailable_reason":problem,
            "hearing":"not_established_by_inbox_or_generation_state",
            "phone_notice_state":row.get("phone_notice_state") if row.get("phone_notice_generation") == row["generation"] else None}

    async def detail(self, owner, identity):
        from app.tasks.artifacts import result_artifacts

        async with self.tasks.read_transaction() as connection:
            row, task = await self._entry(connection, owner, identity)
            output, problem = await self._source(connection, row, task)
            artifacts = (await result_artifacts(connection, self.tasks, task, output)
                         if task is not None and output and not isinstance(output, NotificationOutput) else [])
            attempts = await connection.fetch("""select id,generation,channel,session_id,phase,forwarded_samples,
                played_samples,submitted_at,first_audio_at,generation_finished_at,ended_at,reason
                from jarvis_attention_attempts where owner_id=$1 and intent_id=$2 order by created_at,id""", owner, identity)
            return {**self._public(row, problem),
                "expert_result":output.content.model_dump(mode="json") if output and not isinstance(output, NotificationOutput) else None,
                "expert_speech_prepared":bool(output and not isinstance(output, NotificationOutput) and output.speech_ready),
                "notification":output.content.model_dump(mode="json") if isinstance(output, NotificationOutput) else None,
                "speech_prepared":bool(output and output.speech_ready),
                "artifacts": artifacts, "attempts":[dict(value) for value in attempts]}

    async def list(self, owner, *, limit=20, offset=0, include_dismissed=False, task_id=None):
        if not 1 <= limit <= 100 or not 0 <= offset <= 100000:
            raise ValueError("Invalid inbox pagination")
        async with self.tasks.read_transaction() as connection:
            ids = await connection.fetch("""select id from jarvis_attention_intents where owner_id=$1
                and ($2 or state not in ('dismissed','revoked','expired'))
                and ($5::uuid is null or task_id=$5)
                order by created_at desc,id limit $3 offset $4""", owner, include_dismissed, limit, offset, task_id)
            result = []
            for identity in ids:
                row, task = await self._entry(connection, owner, identity["id"])
                _, problem = await self._source(connection, row, task)
                result.append(self._public(row, problem))
            return result

    async def mutate(self, owner, identity, operation, request: InboxMutation):
        if operation not in {"dismiss", "replay"}:
            raise ValueError("Unknown inbox operation")
        signature = digest({"intent_id":str(identity), "operation":operation,
                            "generation":request.expected_generation})
        async with self.tasks.transaction(owner) as connection:
            row, task = await self._entry(connection, owner, identity, lock=True)
            result = await self._mutate_locked(connection, owner, identity, operation, request,
                                               row, task, signature)
            if result["replayed_request"]:
                return result
        if operation == "dismiss" and row["source_namespace"] == "prospective_event":
            await self.reconcile_notifications(owner)
        return result

    async def _mutate_locked(self, connection, owner, identity, operation, request, row, task, signature):
        """Existing inbox mutation, shared by the source-bound completed-task adapter."""
        previous = await connection.fetchrow("select * from jarvis_attention_requests where owner_id=$1 and request_id=$2", owner, request.request_id)
        if previous:
            if previous["parameters_digest"] != signature:
                raise AttentionError("attention_request_conflict", "This request identity was already used differently.")
            return {"id":identity, "generation":previous["accepted_generation"], "replayed_request":True}
        if request.expected_generation != row["generation"]:
            raise AttentionError("attention_generation_conflict", "Read the current inbox item before changing it.")
        if operation == "replay":
            if row["source_namespace"] == QUIET_SUMMARY:
                raise AttentionError("quiet_summary_not_replayable", "Request an original inbox item rather than replaying a quiet summary.")
            if row["attempt_count"] >= self.max_attempts:
                raise AttentionError("attention_attempt_limit", "This intent has reached its configured delivery-attempt limit.")
            output, problem = await self._source(connection, row, task, lock=True)
            if problem or not output.speech_ready:
                raise AttentionError("attention_source_unavailable", "No current speech is available for replay.")
            if row["state"] in {"claimed", "awaiting_receipt"}:
                raise AttentionError("attention_delivery_active", "The existing delivery must be settled first.")
            if row["state"] in {"revoked", "expired"}:
                raise AttentionError("attention_source_unavailable", "This delivery intent is no longer eligible.")
        await connection.execute("""update jarvis_attention_attempts set phase='revoked',reason='inbox_dismissed',
            ended_at=clock_timestamp() where owner_id=$1 and intent_id=$2 and phase=any($3::text[])""",
            owner, identity, list(ACTIVE_PHASES))
        generation = row["generation"] + 1
        await connection.execute("""update jarvis_attention_intents set state=$3,generation=$4,
            reason=null,not_before=clock_timestamp(),updated_at=clock_timestamp()
            where owner_id=$1 and id=$2""", owner, identity, "dismissed" if operation == "dismiss" else "pending", generation)
        if operation == "replay" and row.get("quiet_episode_id") is not None:
            await connection.execute("""update jarvis_attention_intents set
                quiet_episode_id=null,quiet_summary_intent_id=null where owner_id=$1 and id=$2""", owner, identity)
        await connection.execute("""insert into jarvis_attention_requests(owner_id,request_id,intent_id,operation,
            parameters_digest,accepted_generation) values($1,$2,$3,$4,$5,$6)""",
            owner, request.request_id, identity, operation, signature, generation)
        return {"id":identity, "generation":generation, "replayed_request":False}

    async def dismiss_task_result(self, owner, task_id, request, *, source):
        """Dismiss one exact current result; no future mute or task-state rewrite."""
        async with self.tasks.transaction(owner) as connection:
            await self.tasks._voice_source(connection, owner, source)
            task = await self.tasks.load(connection, owner, task_id)
            if task.revision != request.expected_revision:
                raise TaskError("task_revision_conflict", "Read the current task revision before changing delivery.")
            if task.status != TaskStatus.SUCCEEDED:
                raise TaskError("task_state_conflict", "Completed-result dismissal requires completed work.")
            previous = await connection.fetchrow(
                "select * from jarvis_attention_requests where owner_id=$1 and request_id=$2",
                owner, request.request_id)
            # The publication trigger creates eligible intents in the SAME transaction
            # as result_payload_id. Do not select latest, create a substitute, or promise
            # future suppression when an exact current intent is missing.
            rows = await connection.fetch("""select id,generation from jarvis_attention_intents
                where owner_id=$1 and task_id=$2 and task_revision=$3
                and source_namespace='task_result' and source_payload_id=$4
                and source_id=$4 and source_revision=$3 limit 2""",
                owner, task_id, task.revision, task.result_payload_id)
            if len(rows) != 1:
                if previous:
                    raise AttentionError("attention_source_changed", "The recorded dismissal no longer matches this result.")
                return task, {"status": "not_established", "durable_suppression": False,
                    "reason": "exact_result_intent_missing" if not rows else "exact_result_intent_ambiguous",
                    "task_revision": task.revision, "hearing": "not_established", "prior_delivery": "not_retracted"}
            identity = rows[0]["id"]
            if previous and (previous["intent_id"] != identity or previous["operation"] != "dismiss"):
                raise AttentionError("attention_request_conflict", "This request identity was already used differently.")
            # _entry preserves payload -> task -> intent locking, as ordinary dismissal does.
            row, current = await self._entry(connection, owner, identity, lock=True)
            if (current.status != TaskStatus.SUCCEEDED or current.revision != request.expected_revision
                    or current.result_payload_id != task.result_payload_id
                    or row["task_id"] != task_id or row["task_revision"] != current.revision
                    or row["source_namespace"] != "task_result"
                    or row["source_id"] != current.result_payload_id
                    or row["source_revision"] != current.revision
                    or row["source_payload_id"] != current.result_payload_id):
                raise AttentionError("attention_source_changed", "The exact task result changed before dismissal.")
            generation = previous["accepted_generation"] - 1 if previous else rows[0]["generation"]
            signature = digest({"intent_id": str(identity), "operation": "dismiss", "generation": generation,
                "task_id": str(task_id), "result_payload_id": str(current.result_payload_id),
                "task_request": request.model_dump(mode="json"), "voice_source": source.parameters()})
            result = await self._mutate_locked(connection, owner, identity, "dismiss",
                InboxMutation(request_id=request.request_id, expected_generation=generation), row, current, signature)
            suppressed = (not result["replayed_request"]
                          or row["generation"] == result["generation"] and row["state"] == "dismissed")
            return current, {
                "status": "dismissed" if suppressed else "historical_dismissal",
                "durable_suppression": suppressed, "intent_id": str(identity),
                "task_revision": current.revision, "source_payload_id": str(current.result_payload_id),
                "generation": result["generation"], "revoked_generation": generation,
                "replayed_request": result["replayed_request"],
                "hearing": "not_established", "prior_delivery": "not_retracted",
            }

    async def _recover(self, connection, owner):
        expired = await connection.fetch("""select id,intent_id from jarvis_attention_attempts
            where owner_id=$1 and channel='voice' and phase=any($2::text[]) and lease_until<=clock_timestamp()
            order by intent_id,id""", owner, list(ACTIVE_PHASES))
        for candidate in expired:
            # Match source -> task -> intent -> attempt ordering used by delivery
            # and source-revocation triggers, rather than locking attempts first.
            await self._entry(connection, owner, candidate["intent_id"], lock=True)
            item = await connection.fetchrow("""update jarvis_attention_attempts set
                phase=case when submitted_at is null then 'failed' else 'unknown' end,
                reason='delivery_lease_expired',ended_at=clock_timestamp()
                where owner_id=$1 and id=$2 and phase=any($3::text[]) and lease_until<=clock_timestamp()
                returning intent_id,generation,phase""", owner, candidate["id"], list(ACTIVE_PHASES))
            if item is None:
                continue
            await connection.execute("""update jarvis_attention_intents set state=$4,generation=generation+1,
                reason='delivery_lease_expired',updated_at=clock_timestamp()
                where owner_id=$1 and id=$2 and generation=$3""", owner, item["intent_id"], item["generation"],
                "pending" if item["phase"] == "failed" else "unknown")

        # Admission can be impossible without a connected voice session. Sweep
        # stale unsubmitted visual candidates here as well; unknown submitted
        # delivery is never reissued just because a freshness window elapsed.
        stale = await connection.fetch("""select id from jarvis_attention_intents
            where owner_id=$1 and source_namespace=any($2::text[]) and state in ('pending','failed','unknown')
            order by created_at,id limit 50""", owner, list(VISUAL_NAMESPACES))
        for item in stale:
            row, task = await self._entry(connection, owner, item["id"], lock=True)
            _, problem = await self._source(connection, row, task, lock=False)
            if problem and not (row["state"] == "unknown" and problem == "visual_evidence_expired"):
                await connection.execute("""update jarvis_attention_intents set state=$3,reason=$4,
                    generation=generation+1,updated_at=clock_timestamp() where owner_id=$1 and id=$2""",
                    owner, row["id"], "expired" if problem == "visual_evidence_expired" else "revoked", problem)

    @completed_claim
    async def claim_next(self, owner, session_id: UUID, worker_id: str, *, admission=None):
        if not 1 <= len(worker_id) <= 200:
            raise ValueError("Invalid delivery worker identity")
        async with self.tasks.transaction(owner) as connection:
            if not await connection.fetchval("select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2 and status='active')", owner, session_id):
                raise AttentionError("attention_session_unavailable", "An active owned session is required.")
            await self._recover(connection, owner)
            quiet = await quiet_delivery_decision(connection, owner, kind="unsolicited", channel="voice")
            if not quiet.allowed:
                # Keep durable results and pending intents; maintenance still ran.
                stage_admission("quiet_denied" if quiet.reason == "owner_quiet_quiet" else "quiet_unavailable",
                    owner=owner, session_id=session_id, episode_id=quiet.episode_id, revision=quiet.revision)
                return None
            coalesced = await coalesce_ended_episode(self, connection, owner, quiet)
            if quiet.reason in {"owner_quiet_resumed", "owner_quiet_expired"}:
                stage_admission("coalescing_completed" if coalesced else "coalescing_blocked",
                    owner=owner, session_id=session_id, episode_id=quiet.episode_id, revision=quiet.revision)
            if not coalesced:
                return None
            if await connection.fetchval("select exists(select 1 from jarvis_attention_attempts where owner_id=$1 and channel='voice' and phase=any($2::text[]))", owner, list(ACTIVE_PHASES)):
                return None
            candidates = await connection.fetch("""select id from jarvis_attention_intents where owner_id=$1
                and state='pending' and not_before<=clock_timestamp()
                and to_jsonb(jarvis_attention_intents)->>'quiet_episode_id' is null
                and (source_namespace<>'quiet_resume_summary' or not exists(
                    select 1 from jarvis_attention_attempts a where a.owner_id=jarvis_attention_intents.owner_id
                    and a.intent_id=jarvis_attention_intents.id))
                and not exists(select 1 from jarvis_attention_attempts p
                    where p.owner_id=jarvis_attention_intents.owner_id and p.intent_id=jarvis_attention_intents.id
                    and p.channel='phone' and p.generation=jarvis_attention_intents.generation
                    and p.phase in ('claimed','submitted','accepted','unknown'))
                order by case urgency when 'high' then 0 else 1 end,created_at,id limit 50""", owner)
            for candidate in candidates:
                row, task = await self._entry(connection, owner, candidate["id"], lock=True)
                expired = row["expires_at"] is not None and await connection.fetchval("select $1::timestamptz<=clock_timestamp()", row["expires_at"])
                if expired or row["attempt_count"] >= self.max_attempts:
                    await connection.execute("update jarvis_attention_intents set state=$3,reason=$4 where owner_id=$1 and id=$2", owner, row["id"],
                        "expired" if expired else "failed", "intent_expired" if expired else "delivery_attempt_limit")
                    continue
                output, problem = await self._source(connection, row, task, lock=False)
                if problem or not output.speech_ready:
                    await connection.execute("update jarvis_attention_intents set state=$4,reason=$3,generation=generation+1 where owner_id=$1 and id=$2", owner, row["id"], problem or "speech_not_prepared", "revoked" if problem else "failed")
                    continue
                rejection = admission(row, output) if admission is not None else None
                if rejection:
                    if rejection in {"superseded_observation", "source_time_unavailable"}:
                        await connection.execute("update jarvis_attention_intents set state='expired',reason=$3 where owner_id=$1 and id=$2",
                                                 owner, row["id"], rejection)
                    continue
                attempt = await connection.fetchval("""insert into jarvis_attention_attempts(owner_id,intent_id,generation,
                    channel,session_id,worker_id,lease_until) values($1,$2,$3,'voice',$4,$5,
                    clock_timestamp()+$6::double precision*interval '1 second') returning id""",
                    owner, row["id"], row["generation"], session_id, worker_id, self.lease_seconds)
                await connection.execute("update jarvis_attention_intents set state='claimed',attempt_count=attempt_count+1,updated_at=clock_timestamp() where owner_id=$1 and id=$2", owner, row["id"])
                stage_admission("claimed", owner=owner, session_id=session_id)
                return DeliveryClaim(owner, row["id"], attempt, row["generation"], session_id, worker_id, row["source_payload_id"], row["source_namespace"],
                    getattr(output.content, "subject_type", None), getattr(output.content, "subject_id", None))
        return None

    @asynccontextmanager
    async def guard(self, claim):
        """Hold source/claim fences through the caller's bounded write boundary."""
        async with self.tasks.transaction(claim.owner_id) as connection:
            quiet = await quiet_delivery_decision(connection, claim.owner_id, kind="unsolicited", channel="voice")
            if not quiet.allowed:
                raise AttentionError(quiet.reason, "Owner quiet state suppresses unsolicited speech.")
            if not await coalesce_ended_episode(self, connection, claim.owner_id, quiet):
                raise AttentionError("owner_quiet_release_unavailable", "Quiet release requires the canonical episode schema.")
            row, task = await self._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            if row.get("quiet_episode_id") is not None:
                raise AttentionError("owner_quiet_coalesced", "This announcement was withheld from automatic backlog replay.")
            attempt = await connection.fetchrow("""select * from jarvis_attention_attempts where owner_id=$1
                and id=$2 and intent_id=$3 and generation=$4 and session_id=$5 and worker_id=$6
                and phase=any($7::text[]) and lease_until>clock_timestamp() for update""",
                claim.owner_id, claim.attempt_id, claim.intent_id, claim.generation,
                claim.session_id, claim.worker_id, list(ACTIVE_PHASES))
            if (attempt is None or row["generation"] != claim.generation or row["state"] not in {"claimed", "awaiting_receipt"}
                    or row["source_payload_id"] != claim.source_payload_id):
                raise AttentionError("attention_claim_lost", "This attempt no longer owns delivery.")
            if not await connection.fetchval("select exists(select 1 from jarvis_sessions where owner_id=$1 and id=$2 and status='active')", claim.owner_id, claim.session_id):
                raise AttentionError("attention_session_unavailable", "The delivery session is no longer active.")
            output, problem = await self._source(connection, row, task, lock=True)
            if problem or not output.speech_ready:
                raise AttentionError("attention_source_unavailable", "The source is no longer available for speech.")
            yield connection, row, attempt, output

    async def renew(self, claim):
        async with self.guard(claim) as (connection, _, _, _output):
            await connection.execute("update jarvis_attention_attempts set lease_until=clock_timestamp()+$3::double precision*interval '1 second' where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id, self.lease_seconds)

    async def prepare(self, claim, speech_provider):
        async with self.guard(claim) as (connection, row, attempt, output):
            if isinstance(output, NotificationOutput) and output.content.authored_by == "note_service":
                raise AttentionError("note_result_requires_native_route", "Runtime note outcomes use the native voice route.")
            texts = output.speech_texts if isinstance(output, NotificationOutput) else tuple(chunk.text for chunk in output.content.speech)
            requests = tuple(speech_provider.prepare(text) for text in texts)
            plan = {"format":"jarvis.attention.speech_plan.v1", "source_payload_id":str(row["source_payload_id"]),
                "result_digest":output.result_digest if isinstance(output, NotificationOutput) else output.authorship.result_digest,
                "chunks":[{"index":i, "input_digest":request.digest, "model":request.model,
                    "voice":request.voice, "instructions":request.instructions} for i, request in enumerate(requests)]}
            if attempt["plan_digest"] is not None:
                if attempt["plan_digest"] != digest(plan):
                    raise AttentionError("attention_plan_conflict", "An attempt cannot change its speech plan.")
            else:
                if attempt["phase"] != "claimed":
                    raise AttentionError("attention_submission_state", "Speech preparation must precede submission.")
                payload = await self.tasks.payload(connection, claim.owner_id, plan)
                await connection.execute("update jarvis_attention_attempts set plan_payload_id=$3,plan_digest=$4 where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id, payload, digest(plan))
            for index, request in enumerate(requests):
                await connection.execute("""insert into jarvis_attention_speech_chunks
                    (owner_id,attempt_id,chunk_index,input_digest,response_id,item_id)
                    values($1,$2,$3,$4,$5,$6) on conflict do nothing""",
                    claim.owner_id, claim.attempt_id, index, request.digest,
                    f"tts_{claim.attempt_id.hex}", f"tts_{claim.attempt_id.hex}_{index}")
            return requests

    async def submit_chunk(self, claim, index, request):
        """Commit the exact, one-use HTTP request before any TTS network I/O."""
        async with self.guard(claim) as (connection, _, attempt, _output):
            if attempt["phase"] not in {"claimed", "submitted", "forwarding"}:
                raise AttentionError("attention_submission_state", "No active speech plan exists.")
            chunk = await connection.fetchrow("""update jarvis_attention_speech_chunks
                set submitted_at=clock_timestamp() where owner_id=$1 and attempt_id=$2
                and chunk_index=$3 and input_digest=$4 and submitted_at is null returning item_id""",
                claim.owner_id, claim.attempt_id, index, request.digest)
            if chunk is None:
                raise AttentionError("attention_chunk_conflict", "The exact unsubmitted chunk is required.")
            await connection.execute("""update jarvis_attention_attempts set
                phase=case when phase='claimed' then 'submitted' else phase end,
                submitted_at=coalesce(submitted_at,clock_timestamp()) where owner_id=$1 and id=$2""",
                claim.owner_id, claim.attempt_id)
        return True

    @asynccontextmanager
    async def write_guard(self, claim, index, *, start_sample=None, samples=0):
        """Entered inside the common socket lock, not before waiting for it."""
        async with self.guard(claim) as (connection, _, _attempt, _output):
            chunk = await connection.fetchrow("""select * from jarvis_attention_speech_chunks
                where owner_id=$1 and attempt_id=$2 and chunk_index=$3 for update""",
                claim.owner_id, claim.attempt_id, index)
            if chunk is None or chunk["submitted_at"] is None:
                raise AttentionError("attention_submission_state", "No submitted chunk exists.")
            if samples and (type(samples) is not int or samples < 0 or
                            start_sample != chunk["forwarded_samples"] or chunk["generated_at"] is not None):
                raise AttentionError("attention_audio_sequence", "PCM forwarding must be contiguous.")
            sent = False
            def mark_sent():
                nonlocal sent
                sent = True
            yield mark_sent
            if sent and samples:
                await connection.execute("""update jarvis_attention_speech_chunks
                    set forwarded_samples=forwarded_samples+$4 where owner_id=$1 and attempt_id=$2 and chunk_index=$3""",
                    claim.owner_id, claim.attempt_id, index, samples)
                await connection.execute("""update jarvis_attention_attempts set phase='forwarding',
                    forwarded_samples=forwarded_samples+$3,first_audio_at=coalesce(first_audio_at,clock_timestamp())
                    where owner_id=$1 and id=$2""", claim.owner_id, claim.attempt_id, samples)

    async def chunk_generated(self, claim, index, *, samples, first_bytes_ms, provider_request_id):
        async with self.guard(claim) as (connection, _, _attempt, _output):
            changed = await connection.fetchval("""update jarvis_attention_speech_chunks
                set generated_at=clock_timestamp(),first_bytes_ms=$5,provider_request_id=$6
                where owner_id=$1 and attempt_id=$2 and chunk_index=$3 and forwarded_samples=$4
                and forwarded_samples>0 and submitted_at is not null and generated_at is null returning true""",
                claim.owner_id, claim.attempt_id, index, samples, first_bytes_ms, provider_request_id)
            if not changed:
                raise AttentionError("attention_audio_sequence", "Generation requires all forwarded samples.")

    async def playback_receipt(self, claim, *, item_id, response_id, samples, completed):
        """Authenticated session telemetry, not proof of human understanding."""
        if type(samples) is not int or samples < 0 or type(completed) is not bool:
            raise AttentionError("attention_playback_invalid", "Invalid playback boundary.")
        all_played = False
        async with self.tasks.transaction(claim.owner_id) as connection:
            row, source_task = await self._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            _, source_problem = await self._source(connection, row, source_task, lock=True)
            attempt = await connection.fetchrow("""select * from jarvis_attention_attempts
                where owner_id=$1 and id=$2 and intent_id=$3 and generation=$4 and session_id=$5 and worker_id=$6
                for update""", claim.owner_id, claim.attempt_id, claim.intent_id,
                claim.generation, claim.session_id, claim.worker_id)
            chunk = await connection.fetchrow("""select * from jarvis_attention_speech_chunks
                where owner_id=$1 and attempt_id=$2 and item_id=$3 and response_id=$4 for update""",
                claim.owner_id, claim.attempt_id, item_id, response_id)
            if (attempt is None or chunk is None or samples > chunk["forwarded_samples"]
                    or samples < (chunk["played_samples"] or 0)
                    or completed and (chunk["generated_at"] is None or samples != chunk["forwarded_samples"])):
                raise AttentionError("attention_playback_invalid", "Receipt does not match this delivery.")
            await connection.execute("""update jarvis_attention_speech_chunks set played_samples=$4,
                playback_completed_at=case when $5 then coalesce(playback_completed_at,clock_timestamp())
                else playback_completed_at end where owner_id=$1 and attempt_id=$2 and item_id=$3""",
                claim.owner_id, claim.attempt_id, item_id, samples, completed)
            totals = await connection.fetchrow("""select sum(coalesce(played_samples,0)) as samples,
                bool_and(playback_completed_at is not null) as complete from jarvis_attention_speech_chunks
                where owner_id=$1 and attempt_id=$2""", claim.owner_id, claim.attempt_id)
            await connection.execute("update jarvis_attention_attempts set played_samples=$3 where owner_id=$1 and id=$2",
                                     claim.owner_id, claim.attempt_id, totals["samples"])
            if (source_problem is None and totals["complete"] and attempt["generation_finished_at"] is not None
                    and attempt["phase"] in {"awaiting_receipt", "played"}
                    and row["generation"] == claim.generation and row["state"] in {"awaiting_receipt", "played"}):
                await connection.execute("update jarvis_attention_attempts set phase='played',ended_at=coalesce(ended_at,clock_timestamp()) where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id)
                await connection.execute("update jarvis_attention_intents set state='played',updated_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.intent_id)
                all_played = True
        if all_played and claim.source_namespace == "prospective_event":
            await self.reconcile_notifications(claim.owner_id)
        return all_played

    async def settle(self, claim, *, reason, interrupted=False):
        """Terminalize only this attempt; never mutate or re-execute its task."""
        async with self.tasks.transaction(claim.owner_id) as connection:
            row, _task = await self._entry(connection, claim.owner_id, claim.intent_id, lock=True)
            attempt = await connection.fetchrow("""select * from jarvis_attention_attempts
                where owner_id=$1 and id=$2 and intent_id=$3 and generation=$4 and session_id=$5 and worker_id=$6 for update""",
                claim.owner_id, claim.attempt_id, claim.intent_id, claim.generation, claim.session_id, claim.worker_id)
            if attempt is None or attempt["phase"] not in ACTIVE_PHASES:
                return
            phase = "interrupted" if interrupted else "unknown" if attempt["submitted_at"] else "failed"
            await connection.execute("update jarvis_attention_attempts set phase=$3,reason=$4,ended_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id, phase, reason)
            if row["generation"] == claim.generation and row["state"] in {"claimed", "awaiting_receipt"}:
                await connection.execute("update jarvis_attention_intents set state=$3,reason=$4,updated_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.intent_id,
                                         "unknown" if attempt["submitted_at"] else "failed", reason)

    async def submitted(self, claim):
        async with self.guard(claim) as (connection, _, attempt, _output):
            if attempt["phase"] != "claimed" or attempt["plan_payload_id"] is None:
                raise AttentionError("attention_submission_state", "A prepared, unsubmitted attempt is required.")
            await connection.execute("update jarvis_attention_attempts set phase='submitted',submitted_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id)

    async def forwarded(self, claim, *, samples):
        if type(samples) is not int or samples < 1:
            raise ValueError("Forwarded PCM must contain samples")
        async with self.guard(claim) as (connection, _, attempt, _output):
            if attempt["phase"] not in {"submitted", "forwarding"}:
                raise AttentionError("attention_submission_state", "No active speech submission exists.")
            await connection.execute("""update jarvis_attention_attempts set phase='forwarding',forwarded_samples=forwarded_samples+$3,
                first_audio_at=coalesce(first_audio_at,clock_timestamp()) where owner_id=$1 and id=$2""", claim.owner_id, claim.attempt_id, samples)

    async def generation_finished(self, claim):
        async with self.guard(claim) as (connection, _, attempt, _output):
            if attempt["phase"] != "forwarding" or attempt["forwarded_samples"] < 1:
                raise AttentionError("attention_no_forwarded_audio", "No forwarded audio can be completed.")
            await connection.execute("update jarvis_attention_attempts set phase='awaiting_receipt',generation_finished_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.attempt_id)
            await connection.execute("update jarvis_attention_intents set state='awaiting_receipt',updated_at=clock_timestamp() where owner_id=$1 and id=$2", claim.owner_id, claim.intent_id)
