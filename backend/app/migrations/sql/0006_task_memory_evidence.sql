-- Task dependencies on M02 originals. No provider call or live memory cutover.
create table jarvis_task_memory_sources (
    owner_id text not null,
    manifest_id uuid not null,
    area_id uuid not null,
    source_id uuid not null,
    source_revision bigint not null,
    source_generation bigint not null check(source_generation>0),
    payload_id uuid not null,
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,manifest_id,source_id,source_revision),
    foreign key(owner_id,manifest_id) references jarvis_task_context_manifests(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_task_memory_payload_idx on jarvis_task_memory_sources(owner_id,payload_id);
create index jarvis_task_memory_source_idx on jarvis_task_memory_sources(owner_id,source_id);
create trigger jarvis_task_memory_sources_immutable before update on jarvis_task_memory_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_task_memory_source() returns trigger language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
        join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
        join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
        where v.owner_id=new.owner_id and v.area_id=new.area_id and v.source_id=new.source_id
            and v.revision=new.source_revision and s.generation=new.source_generation
            and v.payload_id=new.payload_id and v.content_digest=new.content_digest
            and p.availability='available' and s.status='active' and a.status='active'
            and ((new.source_usage='current' and s.current_revision=v.revision)
                 or (new.source_usage='historical' and s.current_revision>v.revision))
    ) then
        raise exception 'Task memory dependency must match available original evidence' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_source_guard before insert on jarvis_task_memory_sources
for each row execute function guard_jarvis_task_memory_source();

create function invalidate_jarvis_task_memory_manifests(p_owner text,p_manifests uuid[],p_removed boolean)
returns void language plpgsql set search_path from current as $$
declare derived_payloads uuid[];
begin
    if p_manifests is null then return; end if;
    update jarvis_task_context_manifests set status=case when p_removed then 'revoked' else 'superseded' end
    where owner_id=p_owner and id=any(p_manifests)
        and (status='active' or (p_removed and status<>'revoked'));

    -- Completed results have no current_attempt_id. Use their exact event/attempt
    -- lineage, so an older invalidated attempt cannot hide a newer answer.
    update jarvis_background_tasks t set result_payload_id=null
    where t.owner_id=p_owner and t.result_payload_id is not null and exists(
        select 1 from jarvis_task_events e
        join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
        where e.owner_id=t.owner_id and e.task_id=t.id and e.payload_id=t.result_payload_id
            and a.context_manifest_id=any(p_manifests)
    );
    if p_removed then
        select array_agg(distinct payload_id) into derived_payloads from (
            select m.payload_id from jarvis_task_context_manifests m
                where m.owner_id=p_owner and m.id=any(p_manifests)
            union
            select a.checkpoint_payload_id from jarvis_task_attempts a
                where a.owner_id=p_owner and a.context_manifest_id=any(p_manifests)
            union
            select e.payload_id from jarvis_task_events e
                join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
                where a.owner_id=p_owner and a.context_manifest_id=any(p_manifests)
        ) derived where payload_id is not null;
        update jarvis_context_payloads set availability='deleted',content=null
        where owner_id=p_owner and id=any(derived_payloads) and availability in ('available','not_finalized');
    end if;
end; $$;

create function invalidate_jarvis_task_memory_revision() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
        where owner_id=new.owner_id and source_id=new.id
            and (source_generation<>new.generation or new.status<>'active');
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,new.status<>'active');
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_revision after update on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_task_memory_revision();

create function revoke_jarvis_task_memory_payload() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if new.availability in ('deleted','expired','unavailable') and old.availability is distinct from new.availability then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
            where owner_id=new.owner_id and payload_id=new.id;
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,true);
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_payload after update on jarvis_context_payloads
for each row execute function revoke_jarvis_task_memory_payload();

create function revoke_jarvis_task_memory_area() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if new.status<>'active' and old.status is distinct from new.status then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
            where owner_id=new.owner_id and area_id=new.id;
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,true);
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_area after update on jarvis_knowledge_areas
for each row execute function revoke_jarvis_task_memory_area();
