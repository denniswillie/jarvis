-- Additive foundations. No routines, workers, providers, or grants are enabled.
alter table jarvis_sessions add constraint jarvis_sessions_owner_identity unique (owner_id, id);

create table jarvis_knowledge_areas (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    name text not null check (length(btrim(name)) between 1 and 200),
    domain text not null check (length(btrim(domain)) between 1 and 200),
    status text not null default 'active' check (status in ('active','archived','deleted')),
    revision bigint not null default 1 check (revision > 0),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique (owner_id, id)
);
create index jarvis_knowledge_areas_owner_status_idx on jarvis_knowledge_areas(owner_id,status,id);

create function guard_jarvis_knowledge_area_update() returns trigger language plpgsql as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.created_at<>old.created_at
        or new.revision<>old.revision+1 or (old.status='deleted' and new.status<>'deleted') then
        raise exception 'Knowledge area identity is immutable and updates require the next revision' using errcode='23514';
    end if;
    new.updated_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_knowledge_areas_identity before update on jarvis_knowledge_areas
for each row execute function guard_jarvis_knowledge_area_update();

create table jarvis_context_streams (
    owner_id text primary key check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    last_sequence bigint not null default 0 check (last_sequence >= 0)
);

create table jarvis_context_payloads (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    availability text not null check (availability in ('available','unavailable','expired','deleted','not_finalized')),
    content jsonb,
    content_digest text check(content_digest ~ '^[0-9a-f]{64}$'),
    created_at timestamptz not null default clock_timestamp(),
    removed_at timestamptz,
    unique(owner_id,id),
    check ((content is null)=(content_digest is null)),
    check (content is null or (jsonb_typeof(content)='object' and octet_length(content::text)<=262144)),
    check ((availability in ('available','not_finalized') and content is not null and removed_at is null)
        or (availability in ('unavailable','expired','deleted') and content is null))
);

create function guard_jarvis_context_payload_update() returns trigger language plpgsql as $$
begin
    if tg_op='UPDATE' then
        if new.id<>old.id or new.owner_id<>old.owner_id or new.created_at<>old.created_at
            or new.content is not null or new.availability not in ('unavailable','expired','deleted')
            or (old.availability='deleted' and new.availability<>'deleted') then
            raise exception 'Context payloads are immutable except for irreversible removal' using errcode='23514';
        end if;
        new.removed_at := coalesce(old.removed_at, clock_timestamp());
    else
        new.created_at := clock_timestamp();
    end if;
    -- convert_to is STABLE, so this must not be a generated-column expression.
    -- Compute in the same write and ignore any caller-supplied digest.
    new.content_digest := encode(sha256(convert_to(new.content::text,'UTF8')),'hex');
    return new;
end; $$;
create trigger jarvis_context_payload_removal_guard before insert or update on jarvis_context_payloads
for each row execute function guard_jarvis_context_payload_update();

create table jarvis_context_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null references jarvis_context_streams(owner_id),
    sequence bigint not null check(sequence>0),
    session_id uuid,
    source_namespace text not null check(length(btrim(source_namespace)) between 1 and 100),
    source_event_id text not null check(length(btrim(source_event_id)) between 1 and 512),
    revision integer not null default 1 check(revision>0),
    supersedes_event_id uuid,
    kind text not null check(kind in ('user_request','user_transcript','assistant_output','played_output_boundary',
        'tool_outcome','observation','task_update','correction','context_gap')),
    finality text not null check(finality in ('provisional','final','revised','unavailable')),
    trust text not null check(trust in ('user_input','sensor','external_content','assistant_output','tool_receipt','derived')),
    source_time timestamptz,
    recorded_at timestamptz not null default clock_timestamp(),
    payload_id uuid not null,
    unique(owner_id,id),
    unique(owner_id,id,revision),
    unique(owner_id,sequence),
    unique(owner_id,source_namespace,source_event_id,revision),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,supersedes_event_id) references jarvis_context_events(owner_id,id),
    check ((revision=1 and supersedes_event_id is null) or (revision>1 and supersedes_event_id is not null))
);
create index jarvis_context_events_session_sequence_idx on jarvis_context_events(owner_id,session_id,sequence);
create index jarvis_context_events_payload_idx on jarvis_context_events(owner_id,payload_id);
create index jarvis_context_events_supersedes_idx on jarvis_context_events(owner_id,supersedes_event_id);

create function assign_jarvis_context_sequence() returns trigger language plpgsql set search_path from current as $$
begin
    if new.sequence is not null then
        raise exception 'Context sequence is server assigned' using errcode='23514';
    end if;
    insert into jarvis_context_streams(owner_id,last_sequence) values(new.owner_id,1)
        on conflict(owner_id) do update set last_sequence=jarvis_context_streams.last_sequence+1
        returning last_sequence into new.sequence;
    if new.revision>1 and not exists (
        select 1 from jarvis_context_events e where e.owner_id=new.owner_id and e.id=new.supersedes_event_id
        and e.source_namespace=new.source_namespace and e.source_event_id=new.source_event_id
        and e.session_id is not distinct from new.session_id and e.revision=new.revision-1
    ) then
        raise exception 'Context revision must reference its preceding source revision' using errcode='23514';
    end if;
    new.recorded_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_context_sequence before insert on jarvis_context_events
for each row execute function assign_jarvis_context_sequence();

create function reject_jarvis_foundation_update() returns trigger language plpgsql as $$
begin
    raise exception 'Historical record is append only; append a revision or remove its payload' using errcode='23514';
end; $$;
create trigger jarvis_context_events_append_only before update on jarvis_context_events
for each row execute function reject_jarvis_foundation_update();

create table jarvis_capability_preferences (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    capability text not null check(length(btrim(capability)) between 1 and 100),
    knowledge_area_id uuid,
    revision bigint not null check(revision>0),
    request_event_id uuid not null,
    request_revision integer not null,
    policy_payload_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique nulls not distinct(owner_id,capability,knowledge_area_id,revision),
    foreign key(owner_id,knowledge_area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,policy_payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_capability_preferences_append_only before update on jarvis_capability_preferences
for each row execute function reject_jarvis_foundation_update();

create table jarvis_operation_grants (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    capability text not null check(length(btrim(capability)) between 1 and 100),
    operation_key text not null check(length(btrim(operation_key)) between 1 and 200),
    request_event_id uuid not null,
    request_revision integer not null,
    scope_revision integer not null check(scope_revision>0),
    scope_payload_id uuid not null,
    status text not null default 'active' check(status in ('active','revoked','expired')),
    created_at timestamptz not null default clock_timestamp(),
    expires_at timestamptz not null,
    revoked_at timestamptz,
    unique(owner_id,id),
    unique(owner_id,request_event_id,request_revision,operation_key,scope_revision),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,scope_payload_id) references jarvis_context_payloads(owner_id,id),
    check(isfinite(created_at) and isfinite(expires_at) and expires_at>created_at),
    check((status='revoked')=(revoked_at is not null))
);
create index jarvis_operation_grants_expiry_idx on jarvis_operation_grants(owner_id,expires_at) where status='active';

create function guard_jarvis_explicit_request() returns trigger language plpgsql set search_path from current as $$
declare target_payload uuid;
begin
    -- Serialize authority creation against revisions of this owner's inputs.
    perform 1 from jarvis_context_streams where owner_id=new.owner_id for update;
    if not exists (
        select 1 from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=new.request_revision
            and e.trust='user_input' and e.kind in ('user_request','user_transcript')
            and e.finality in ('final','revised') and p.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
    end if;
    if tg_table_name='jarvis_operation_grants' then target_payload:=new.scope_payload_id;
    else target_payload:=new.policy_payload_id; end if;
    if not exists(select 1 from jarvis_context_payloads where owner_id=new.owner_id
        and id=target_payload and availability='available') then
        raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
    end if;
    new.created_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_operation_grants_explicit_request before insert on jarvis_operation_grants
for each row execute function guard_jarvis_explicit_request();
create trigger jarvis_capability_preferences_explicit_request before insert on jarvis_capability_preferences
for each row execute function guard_jarvis_explicit_request();

create function guard_jarvis_grant_update() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['status','revoked_at'])<>(to_jsonb(old)-array['status','revoked_at'])
        or old.status<>'active' or new.status not in ('revoked','expired') then
        raise exception 'Operation grants can only be revoked or expired, never broadened' using errcode='23514';
    end if;
    if new.status='revoked' then new.revoked_at:=clock_timestamp(); end if;
    return new;
end; $$;
create trigger jarvis_operation_grants_immutable_scope before update on jarvis_operation_grants
for each row execute function guard_jarvis_grant_update();
