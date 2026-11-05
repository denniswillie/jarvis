-- Area metadata history uses removable originals, not permanent private audit prose.
create table jarvis_knowledge_area_versions (
    owner_id text not null,
    area_id uuid not null,
    revision bigint not null check (revision > 0),
    status text not null check (status in ('active','archived','deleted')),
    origin text not null check (origin in ('migration_baseline','created','updated')),
    metadata_payload_id uuid,
    area_updated_at timestamptz not null,
    recorded_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,area_id,revision),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,metadata_payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_knowledge_area_versions_immutable before update or delete
on jarvis_knowledge_area_versions for each row execute function guard_jarvis_knowledge_scope_identity();

create function retain_jarvis_knowledge_area_revision(
    area_row jarvis_knowledge_areas, revision_origin text
) returns void language plpgsql set search_path from current as $$
declare original_id uuid;
begin
    if area_row.status <> 'deleted' then
        insert into jarvis_context_payloads(owner_id,availability,content)
        values(area_row.owner_id,'available',jsonb_build_object(
            'name',area_row.name,'domain',area_row.domain,'description',area_row.description))
        returning id into original_id;
    else
        -- Deletion must not leave earlier private names/descriptions in this new copy.
        update jarvis_context_payloads p set availability='deleted',content=null
        where p.owner_id=area_row.owner_id and p.availability<>'deleted' and exists(
            select 1 from jarvis_knowledge_area_versions h
            where h.owner_id=p.owner_id and h.area_id=area_row.id and h.metadata_payload_id=p.id);
    end if;
    insert into jarvis_knowledge_area_versions(
        owner_id,area_id,revision,status,origin,metadata_payload_id,area_updated_at)
    values(area_row.owner_id,area_row.id,area_row.revision,area_row.status,
        revision_origin,original_id,area_row.updated_at);
end; $$;

create function capture_jarvis_knowledge_area_revision() returns trigger
language plpgsql set search_path from current as $$
begin
    perform retain_jarvis_knowledge_area_revision(
        new,case when tg_op='INSERT' then 'created' else 'updated' end);
    return new;
end; $$;
create trigger jarvis_knowledge_areas_capture_revision after insert or update
on jarvis_knowledge_areas for each row execute function capture_jarvis_knowledge_area_revision();

-- Do not invent metadata from revisions that predate this history.
do $$
declare area_row jarvis_knowledge_areas;
begin
    for area_row in select * from jarvis_knowledge_areas loop
        perform retain_jarvis_knowledge_area_revision(area_row,'migration_baseline');
    end loop;
end; $$;
