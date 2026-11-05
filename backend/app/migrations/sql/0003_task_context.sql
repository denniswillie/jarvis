create table jarvis_context_ingress_receipts (
    owner_id text not null,
    source_namespace text not null,
    source_event_id text not null,
    receipt_id text not null check(length(receipt_id) between 1 and 512),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    event_id uuid not null,
    primary key(owner_id,source_namespace,source_event_id,receipt_id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);
create trigger jarvis_context_ingress_receipts_append_only before update on jarvis_context_ingress_receipts
for each row execute function reject_jarvis_foundation_update();

create table jarvis_context_pins (
    owner_id text not null,
    session_id uuid not null,
    event_id uuid not null,
    reason text not null check(length(reason) between 1 and 200),
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,session_id,event_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);

create table jarvis_task_context_manifests (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    context_watermark bigint not null check(context_watermark>=0),
    payload_id uuid not null,
    status text not null default 'active' check(status in ('active','superseded','revoked')),
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);

create table jarvis_task_context_sources (
    owner_id text not null,
    manifest_id uuid not null,
    event_id uuid not null,
    event_revision integer not null,
    source_usage text not null default 'current' check(source_usage in ('current','historical')),
    payload_id uuid not null,
    content_digest text,
    primary key(owner_id,manifest_id,event_id),
    foreign key(owner_id,manifest_id) references jarvis_task_context_manifests(owner_id,id),
    foreign key(owner_id,event_id,event_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_task_context_sources_payload_idx on jarvis_task_context_sources(owner_id,payload_id);
alter table jarvis_task_attempts add column context_manifest_id uuid;
alter table jarvis_task_attempts add constraint jarvis_task_attempt_context_scope
    foreign key(owner_id,task_id,context_manifest_id) references jarvis_task_context_manifests(owner_id,task_id,id);

create function invalidate_jarvis_context_revision() returns trigger language plpgsql set search_path from current as $$
begin
    if new.revision>1 then
        update jarvis_task_context_manifests m set status='superseded'
        where m.owner_id=new.owner_id and m.status='active' and exists (
            select 1 from jarvis_task_context_sources s join jarvis_context_events e
                on e.owner_id=s.owner_id and e.id=s.event_id
            where s.owner_id=m.owner_id and s.manifest_id=m.id
                and e.source_namespace=new.source_namespace and e.source_event_id=new.source_event_id
                and e.revision<new.revision and s.source_usage='current'
        );
    end if;
    return new;
end; $$;
create trigger jarvis_context_revision_invalidates_manifests after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_revision();

create function revoke_jarvis_context_payload_dependents() returns trigger language plpgsql set search_path from current as $$
declare derived_payloads uuid[];
begin
    if new.availability in ('deleted','expired','unavailable') and old.availability is distinct from new.availability then
        with revoked as (
            update jarvis_task_context_manifests m set status='revoked'
            where m.owner_id=new.owner_id and m.status<>'revoked' and exists (
                select 1 from jarvis_task_context_sources s where s.owner_id=m.owner_id
                    and s.manifest_id=m.id and s.payload_id=new.id
            ) returning m.payload_id
        ) select array_agg(payload_id) into derived_payloads from revoked;
        update jarvis_context_payloads set availability='deleted',content=null
        where owner_id=new.owner_id and id=any(derived_payloads) and id<>new.id
            and availability in ('available','not_finalized');
    end if;
    return new;
end; $$;
create trigger jarvis_context_removal_revokes_manifests after update on jarvis_context_payloads
for each row execute function revoke_jarvis_context_payload_dependents();
