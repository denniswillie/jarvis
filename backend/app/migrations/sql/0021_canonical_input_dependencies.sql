-- Removable canonical-input copies also work while memory publication is paused.
-- This is a provenance fence, not an additional memory engine or model loop.
create table jarvis_context_event_dependencies (
    owner_id text not null,
    dependent_payload_id uuid not null,
    event_id uuid not null,
    event_revision bigint not null check(event_revision>0),
    source_payload_id uuid not null,
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,dependent_payload_id,event_id),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);
create index jarvis_context_event_dependency_event_idx on jarvis_context_event_dependencies(owner_id,event_id);
create trigger jarvis_context_event_dependencies_immutable before update on jarvis_context_event_dependencies
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_context_event_dependency() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_context_events e
        join jarvis_context_payloads original on original.owner_id=e.owner_id and original.id=e.payload_id
        join jarvis_context_payload_dependencies d on d.owner_id=e.owner_id and d.source_payload_id=e.payload_id
        join jarvis_context_payloads copy on copy.owner_id=d.owner_id and copy.id=d.dependent_payload_id
        where e.owner_id=new.owner_id and e.id=new.event_id and e.revision=new.event_revision
            and e.payload_id=new.source_payload_id and copy.id=new.dependent_payload_id
            and original.availability='available' and copy.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        raise exception 'Canonical input dependency must match the available current original' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_event_dependency_guard before insert on jarvis_context_event_dependencies
for each row execute function guard_jarvis_context_event_dependency();

create function invalidate_jarvis_context_event_dependencies() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.revision>1 then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_event_dependencies dependency
            join jarvis_context_events original on original.owner_id=dependency.owner_id
                and original.id=dependency.event_id
            where dependency.owner_id=p.owner_id and dependency.dependent_payload_id=p.id
                and dependency.source_usage='current'
                and original.source_namespace=new.source_namespace
                and original.source_event_id=new.source_event_id and original.revision<new.revision
        );
    end if;
    return new;
end $$;
create trigger jarvis_context_event_dependency_revision after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_event_dependencies();
