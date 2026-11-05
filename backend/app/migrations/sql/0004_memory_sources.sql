-- M02 source substrate only. No vendor calls, live route or ingestion worker.
create table jarvis_memory_banks (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    provider text not null default 'hindsight' check(provider='hindsight'),
    bank_name text not null unique default ('jarvis_' || replace(gen_random_uuid()::text,'-','')),
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,area_id),
    unique(owner_id,area_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id)
);
create trigger jarvis_memory_banks_immutable before update on jarvis_memory_banks
for each row execute function reject_jarvis_foundation_update();

create table jarvis_knowledge_sources (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    namespace text not null check(length(btrim(namespace)) between 1 and 100),
    source_key text not null check(length(btrim(source_key)) between 1 and 512),
    current_revision bigint not null check(current_revision>0),
    generation bigint not null check(generation>0),
    status text not null default 'active' check(status in ('active','unavailable','deleted')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    unique(owner_id,area_id,namespace,source_key),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id)
);

create table jarvis_knowledge_source_versions (
    owner_id text not null,
    area_id uuid not null,
    source_id uuid not null,
    revision bigint not null check(revision>0),
    generation bigint not null check(generation>0),
    payload_id uuid not null,
    bank_id uuid not null,
    document_id text not null unique default ('source_' || replace(gen_random_uuid()::text,'-','')),
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_type text not null check(length(btrim(source_type)) between 1 and 80),
    trust text not null check(trust in ('user_input','sensor','external_content','assistant_output','tool_receipt','derived')),
    source_time timestamptz check(source_time is null or isfinite(source_time)),
    time_basis text not null check(length(btrim(time_basis)) between 1 and 200),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,revision),
    unique(owner_id,document_id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,bank_id) references jarvis_memory_banks(owner_id,area_id,id)
);
alter table jarvis_knowledge_sources add constraint jarvis_source_current_version
foreign key(owner_id,id,current_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision)
deferrable initially deferred;
create index jarvis_source_versions_payload_idx on jarvis_knowledge_source_versions(owner_id,payload_id);
create trigger jarvis_source_versions_immutable before update on jarvis_knowledge_source_versions
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_source_update() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['current_revision','generation','status','updated_at'])
        <> (to_jsonb(old)-array['current_revision','generation','status','updated_at'])
        or old.status='deleted' or new.generation<>old.generation+1
        or (new.status='active' and new.current_revision<>old.current_revision+1)
        or (new.status<>'active' and new.current_revision<>old.current_revision) then
        raise exception 'Source updates require immutable identity and the next generation' using errcode='23514';
    end if;
    new.updated_at:=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_source_identity before update on jarvis_knowledge_sources
for each row execute function guard_jarvis_source_update();

create table jarvis_memory_outbox (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    document_id text not null,
    generation bigint not null check(generation>0),
    operation text not null check(operation in ('retain','delete')),
    status text not null default 'pending'
        check(status in ('pending','leased','accepted','searchable','outcome_unknown','done','failed','cancelled')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,document_id,operation),
    foreign key(owner_id,document_id) references jarvis_knowledge_source_versions(owner_id,document_id)
);
create index jarvis_memory_outbox_pending_idx on jarvis_memory_outbox(owner_id,status,created_at,id);

create function invalidate_jarvis_memory_payload() returns trigger language plpgsql set search_path from current as $$
begin
    if new.availability in ('available','not_finalized') then return new; end if;
    update jarvis_knowledge_sources s set
        status=case when new.availability='deleted' then 'deleted' else 'unavailable' end,
        generation=s.generation+1
    where s.owner_id=new.owner_id and s.status<>'deleted' and exists(
        select 1 from jarvis_knowledge_source_versions v where v.owner_id=s.owner_id
        and v.source_id=s.id and v.revision=s.current_revision and v.payload_id=new.id);
    update jarvis_memory_outbox o set status='cancelled',updated_at=clock_timestamp()
    where o.owner_id=new.owner_id and o.operation='retain' and o.status='pending'
        and exists(select 1 from jarvis_knowledge_source_versions v
            where v.owner_id=o.owner_id and v.document_id=o.document_id and v.payload_id=new.id);
    insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
    select v.owner_id,v.document_id,s.generation,'delete' from jarvis_knowledge_source_versions v
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    where v.owner_id=new.owner_id and v.payload_id=new.id
    on conflict(owner_id,document_id,operation) do update
        set generation=excluded.generation,status='pending',updated_at=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_memory_payload_removal after update on jarvis_context_payloads
for each row execute function invalidate_jarvis_memory_payload();

-- Durable execution ownership is separate from source revision/generation.
alter table jarvis_memory_outbox
    add column fence bigint not null default 0 check(fence>=0),
    add column worker_id text,
    add column lease_until timestamptz,
    add column dispatch_count integer not null default 0 check(dispatch_count>=0),
    add column failure_count integer not null default 0 check(failure_count>=0),
    add column next_attempt_at timestamptz not null default clock_timestamp(),
    add column last_error text,
    add column cleanup_operation_ids uuid[]
        check(cleanup_operation_ids is null or (
            operation='delete' and cardinality(cleanup_operation_ids) between 0 and 64
            and coalesce(array_ndims(cleanup_operation_ids),1)=1
            and array_position(cleanup_operation_ids,null) is null
        ));

-- This is a terminal-operation inventory, persisted before deleting any remote
-- records. NULL means unprepared; an empty array means no retain was dispatched.
-- Keep the IDs after cleanup so lost acknowledgements and rescheduled tombstones
-- do not turn intentional operation deletion into an unknown retention outcome.
create function guard_jarvis_memory_cleanup_inventory() returns trigger language plpgsql as $$
begin
    if old.cleanup_operation_ids is not null
       and new.cleanup_operation_ids is distinct from old.cleanup_operation_ids then
        raise exception 'A prepared memory cleanup inventory is immutable' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_memory_cleanup_inventory_immutable before update on jarvis_memory_outbox
for each row execute function guard_jarvis_memory_cleanup_inventory();
create index jarvis_memory_outbox_due_idx on jarvis_memory_outbox(owner_id,next_attempt_at,id)
    where status in ('pending','accepted','outcome_unknown','leased');

-- Exact canonical lineage. Original contents stay in jarvis_context_payloads.
create table jarvis_knowledge_context_sources (
    owner_id text not null,
    source_id uuid not null,
    source_revision bigint not null,
    event_id uuid not null,
    event_revision integer not null,
    primary key(owner_id,source_id,source_revision),
    foreign key(owner_id,source_id,source_revision)
        references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,event_id,event_revision) references jarvis_context_events(owner_id,id,revision)
);
create index jarvis_knowledge_context_event_idx on jarvis_knowledge_context_sources(owner_id,event_id);
create trigger jarvis_knowledge_context_sources_immutable before update on jarvis_knowledge_context_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_knowledge_context_source() returns trigger language plpgsql set search_path from current as $$
begin
    if not exists(select 1 from jarvis_context_events e
        join jarvis_knowledge_source_versions v on v.owner_id=e.owner_id and v.payload_id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.event_id and e.revision=new.event_revision
        and e.trust=v.trust and e.kind=v.source_type and e.source_time is not distinct from v.source_time
        and v.source_id=new.source_id and v.revision=new.source_revision
        and e.finality in ('final','revised')
        and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
            and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
            and newer.revision>e.revision)) then
        raise exception 'Knowledge lineage requires the exact current canonical original' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_knowledge_context_source_guard before insert on jarvis_knowledge_context_sources
for each row execute function guard_jarvis_knowledge_context_source();

create function invalidate_jarvis_knowledge_context_revision() returns trigger language plpgsql set search_path from current as $$
begin
    if new.revision=1 then return new; end if;
    update jarvis_knowledge_sources s set status='unavailable',generation=s.generation+1
    where s.owner_id=new.owner_id and s.status='active' and exists(
        select 1 from jarvis_knowledge_context_sources link
        join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
        where link.owner_id=s.owner_id and link.source_id=s.id and link.source_revision=s.current_revision
        and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
        and prior.revision<new.revision);
    update jarvis_memory_outbox o set status='cancelled',updated_at=clock_timestamp()
    where o.owner_id=new.owner_id and o.operation='retain' and o.status='pending' and exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_context_sources link on link.owner_id=v.owner_id
            and link.source_id=v.source_id and link.source_revision=v.revision
        join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
        where v.owner_id=o.owner_id and v.document_id=o.document_id
        and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
        and prior.revision<new.revision);
    insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
    select v.owner_id,v.document_id,s.generation,'delete'
    from jarvis_knowledge_source_versions v
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_knowledge_context_sources link on link.owner_id=v.owner_id
        and link.source_id=v.source_id and link.source_revision=v.revision
    join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
    where v.owner_id=new.owner_id and prior.source_namespace=new.source_namespace
        and prior.source_event_id=new.source_event_id and prior.revision<new.revision
    on conflict(owner_id,document_id,operation) do update set generation=excluded.generation,
        status='pending',next_attempt_at=clock_timestamp(),updated_at=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_knowledge_context_revision after insert on jarvis_context_events
for each row execute function invalidate_jarvis_knowledge_context_revision();
