-- Selected-memory read materializations. Originals are not replaced or duplicated
-- in generic execution logs; current claims and explicit historical reads differ.
create table jarvis_context_memory_sources (
    owner_id text not null,
    dependent_payload_id uuid not null,
    area_id uuid not null,
    source_id uuid not null,
    source_revision bigint not null,
    source_generation bigint not null check(source_generation>0),
    source_payload_id uuid not null,
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,dependent_payload_id,source_id,source_revision),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_context_memory_source_idx on jarvis_context_memory_sources(owner_id,source_id);
create index jarvis_context_memory_area_idx on jarvis_context_memory_sources(owner_id,area_id);
create trigger jarvis_context_memory_sources_immutable before update on jarvis_context_memory_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_context_memory_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
        join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
        join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
        join jarvis_context_payload_dependencies d on d.owner_id=p.owner_id and d.source_payload_id=p.id
        join jarvis_context_payloads dependent on dependent.owner_id=d.owner_id and dependent.id=d.dependent_payload_id
        left join jarvis_knowledge_context_sources cs on cs.owner_id=v.owner_id
            and cs.source_id=v.source_id and cs.source_revision=v.revision
        left join jarvis_context_events ce on ce.owner_id=cs.owner_id and ce.id=cs.event_id
        where v.owner_id=new.owner_id and v.area_id=new.area_id and v.source_id=new.source_id
            and v.revision=new.source_revision and s.generation=new.source_generation
            and v.payload_id=new.source_payload_id and v.content_digest=new.content_digest
            and p.availability='available' and s.status='active' and a.status='active'
            and d.dependent_payload_id=new.dependent_payload_id and dependent.availability='available'
            and ((new.source_usage='historical' and (s.current_revision>v.revision or exists(
                select 1 from jarvis_context_events newer where ce.id is not null
                    and newer.owner_id=ce.owner_id and newer.source_namespace=ce.source_namespace
                    and newer.source_event_id=ce.source_event_id and newer.revision>ce.revision))) or (
                new.source_usage='current' and
                s.current_revision=v.revision and not exists(
                    select 1 from jarvis_context_events newer where ce.id is not null
                        and newer.owner_id=ce.owner_id and newer.source_namespace=ce.source_namespace
                        and newer.source_event_id=ce.source_event_id and newer.revision>ce.revision)))
    ) then
        raise exception 'Context memory evidence must match its owned original and dependency' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_source_guard before insert on jarvis_context_memory_sources
for each row execute function guard_jarvis_context_memory_source();

create function invalidate_jarvis_context_memory_revision() returns trigger
language plpgsql set search_path from current as $$
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status
       or old.current_revision is distinct from new.current_revision then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r where r.owner_id=p.owner_id
                and r.dependent_payload_id=p.id and r.source_id=new.id
                and (new.status<>'active' or (r.source_usage='current'
                    and (r.source_generation<>new.generation or r.source_revision<>new.current_revision))));
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_revision after update on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_context_memory_revision();

create function invalidate_jarvis_context_memory_area() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.status<>'active' and old.status is distinct from new.status then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r where r.owner_id=p.owner_id
                and r.dependent_payload_id=p.id and r.area_id=new.id);
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_area after update on jarvis_knowledge_areas
for each row execute function invalidate_jarvis_context_memory_area();

create function invalidate_jarvis_context_memory_event() returns trigger
language plpgsql set search_path from current as $$
begin
    -- Canonical correction wins even before asynchronous publication catches up.
    if new.revision>1 then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r
            join jarvis_knowledge_context_sources cs on cs.owner_id=r.owner_id
                and cs.source_id=r.source_id and cs.source_revision=r.source_revision
            join jarvis_context_events prior on prior.owner_id=cs.owner_id and prior.id=cs.event_id
            where r.owner_id=p.owner_id and r.dependent_payload_id=p.id and r.source_usage='current'
                and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
                and prior.revision<new.revision);
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_event after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_memory_event();
