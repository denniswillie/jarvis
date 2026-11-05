-- Product delivery records, not inferred memories or a second agent runtime.
create table jarvis_attention_intents (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    source_namespace text not null check(length(source_namespace) between 1 and 80),
    source_id uuid not null,
    source_revision bigint not null check(source_revision>0),
    source_payload_id uuid not null,
    task_id uuid,
    task_revision bigint check(task_revision>0),
    preferred_session_id uuid,
    state text not null default 'pending' check(state in
        ('pending','claimed','awaiting_receipt','played','unknown','failed','dismissed','revoked','expired')),
    generation bigint not null default 1 check(generation>0),
    urgency text not null default 'normal' check(urgency in ('normal','high')),
    privacy text not null default 'private' check(privacy in ('private','sensitive')),
    coalesce_key text check(length(coalesce_key) between 1 and 200),
    not_before timestamptz not null default clock_timestamp(),
    expires_at timestamptz check(isfinite(expires_at)),
    attempt_count integer not null default 0 check(attempt_count>=0),
    reason text,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,source_namespace,source_id,source_revision),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,preferred_session_id) references jarvis_sessions(owner_id,id),
    check((task_id is null)=(task_revision is null))
);
create index jarvis_attention_pending on jarvis_attention_intents(owner_id,state,not_before,created_at);

create table jarvis_attention_attempts (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    intent_id uuid not null,
    generation bigint not null check(generation>0),
    channel text not null check(channel in ('voice','phone')),
    session_id uuid,
    worker_id text not null check(length(worker_id) between 1 and 200),
    phase text not null default 'claimed' check(phase in
        ('claimed','submitted','forwarding','awaiting_receipt','played','interrupted','failed','unknown','revoked')),
    plan_payload_id uuid,
    plan_digest text check(plan_digest ~ '^[0-9a-f]{64}$'),
    forwarded_samples bigint not null default 0 check(forwarded_samples>=0),
    played_samples bigint check(played_samples>=0 and played_samples<=forwarded_samples),
    first_audio_at timestamptz,
    submitted_at timestamptz,
    generation_finished_at timestamptz,
    lease_until timestamptz not null check(isfinite(lease_until)),
    reason text,
    created_at timestamptz not null default clock_timestamp(),
    ended_at timestamptz,
    unique(owner_id,id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,plan_payload_id) references jarvis_context_payloads(owner_id,id),
    check((plan_payload_id is null)=(plan_digest is null)),
    check(channel<>'voice' or session_id is not null)
);
create unique index jarvis_attention_one_active_channel on jarvis_attention_attempts(owner_id,channel)
    where phase in ('claimed','submitted','forwarding','awaiting_receipt');
create unique index jarvis_attention_one_active_attempt on jarvis_attention_attempts(owner_id,intent_id)
    where phase in ('claimed','submitted','forwarding','awaiting_receipt');

create table jarvis_attention_requests (
    owner_id text not null,
    request_id uuid not null,
    intent_id uuid not null,
    operation text not null check(operation in ('dismiss','replay')),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_generation bigint not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id)
);
create trigger jarvis_attention_requests_append_only before update on jarvis_attention_requests
for each row execute function reject_jarvis_foundation_update();

create function jarvis_attention_revoke_task_results() returns trigger language plpgsql as $$
begin
    update jarvis_attention_intents set state='revoked',generation=generation+1,
        reason='task_result_changed',updated_at=clock_timestamp()
    where owner_id=new.owner_id and task_id=new.id and state<>'revoked'
      and (task_revision<>new.revision or source_payload_id is distinct from new.result_payload_id);
    update jarvis_attention_attempts a set phase='revoked',reason='task_result_changed',ended_at=clock_timestamp()
    from jarvis_attention_intents i where a.owner_id=i.owner_id and a.intent_id=i.id
      and i.owner_id=new.owner_id and i.task_id=new.id and i.state='revoked'
      and a.phase in ('claimed','submitted','forwarding','awaiting_receipt');
    if new.result_payload_id is not null then
        insert into jarvis_attention_intents(owner_id,source_namespace,source_id,source_revision,
            source_payload_id,task_id,task_revision,preferred_session_id)
        select new.owner_id,'task_result',p.id,new.revision,p.id,new.id,new.revision,new.session_id
        from jarvis_context_payloads p where p.owner_id=new.owner_id and p.id=new.result_payload_id
          and p.availability='available' and p.content->>'format'='jarvis.hermes.task_result.v2'
          and p.content->'authorship'->>'kind'='expert'
          and p.content->'authorship'->>'owner_id'=new.owner_id
          and p.content->'authorship'->>'task_id'=new.id::text
          and p.content->'authorship'->>'task_revision'=new.revision::text
        on conflict(owner_id,source_namespace,source_id,source_revision) do nothing;
    end if;
    return new;
end $$;
create trigger jarvis_attention_task_result after update of result_payload_id,revision on jarvis_background_tasks
for each row execute function jarvis_attention_revoke_task_results();

create function jarvis_attention_revoke_payload() returns trigger language plpgsql as $$
begin
    if new.availability<>'available' or new.content is null then
        update jarvis_attention_intents set state='revoked',generation=generation+1,
            reason='source_payload_unavailable',updated_at=clock_timestamp()
        where owner_id=new.owner_id and source_payload_id=new.id and state<>'revoked';
        update jarvis_attention_attempts a set phase='revoked',reason='source_payload_unavailable',ended_at=clock_timestamp()
        from jarvis_attention_intents i where a.owner_id=i.owner_id and a.intent_id=i.id
          and i.owner_id=new.owner_id and i.source_payload_id=new.id
          and a.phase in ('claimed','submitted','forwarding','awaiting_receipt');
    end if;
    return new;
end $$;
create trigger jarvis_attention_payload_changed after update of availability,content on jarvis_context_payloads
for each row execute function jarvis_attention_revoke_payload();

-- Existing completed results are not automatically replayed by an upgrade.
-- Intents are atomically created on subsequent result publications.
