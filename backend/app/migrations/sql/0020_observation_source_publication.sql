-- Bind ongoing analyzer originals to canonical sources without changing the
-- frozen M03 legacy row shapes. Existing observations are not implicitly imported.
create table jarvis_observation_sources (
    owner_id text not null,
    observation_id uuid not null,
    session_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,observation_id),
    unique(owner_id,payload_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
-- Keep the identity after removing the disposable operational original, so a
-- delayed analyzer replay cannot recreate it under the same observation ID.
create trigger jarvis_observation_sources_immutable before update on jarvis_observation_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_observation_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from perception_observations o
        join jarvis_context_events e on e.owner_id=o.owner_id and e.session_id=o.session_id
            and e.source_namespace='perception:analyzer:v1' and e.source_event_id=o.id::text
        join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
        where o.owner_id=new.owner_id and o.id=new.observation_id and o.session_id=new.session_id
            and p.id=new.payload_id and p.availability='available'
            and e.kind='observation' and e.trust='derived' and e.finality='final'
            and p.content->>'origin'='perception_observations' and p.content->'record'=to_jsonb(o)
    ) then
        raise exception 'Observation binding must match the owned canonical original' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_observation_source_guard before insert on jarvis_observation_sources
for each row execute function guard_jarvis_observation_source();

create function guard_jarvis_observation_replay() returns trigger
language plpgsql set search_path from current as $$
begin
    if exists(select 1 from jarvis_observation_sources
              where owner_id=new.owner_id and observation_id=new.id) then
        raise exception 'A published observation identity cannot be recreated' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_observation_replay_guard before insert on perception_observations
for each row execute function guard_jarvis_observation_replay();

create function invalidate_jarvis_observation_original(p_owner text,p_payload uuid) returns void
language plpgsql set search_path from current as $$
declare binding record;
begin
    for binding in select * from jarvis_observation_sources
        where owner_id=p_owner and payload_id=p_payload
    loop
        -- A reducer may have combined older observations into the latest cache.
        -- Invalidate the session cache, not just its last-observation pointer.
        -- Advance its revision so an in-flight stale CAS cannot restore it.
        update jarvis_session_working_memory set state='{}'::jsonb,revision=revision+1,
            last_observation_id=null,last_observation_ended_at=null,
            expires_at=clock_timestamp(),updated_at=clock_timestamp()
        where owner_id=p_owner and session_id=binding.session_id;
        update visual_frames f set frame_description=null,memory_tags='{}'::text[],
            updated_at=clock_timestamp()
        where f.owner_id=p_owner and exists(select 1 from perception_observation_frames l
            where l.observation_id=binding.observation_id and l.visual_frame_id=f.id);
        -- These are operational aggregates, not independent original evidence.
        -- Keep other evidence rows and timing; force subsequent aggregation to
        -- start from empty state using the existing updated_at conflict check.
        update experience_events e set title='',summary='',scene=null,activity=null,state='{}'::jsonb,
            consolidation_result=case when consolidation_result is null then null else '{}'::jsonb end,
            updated_at=clock_timestamp()
        where e.owner_id=p_owner and exists(select 1 from experience_event_evidence evidence
            where evidence.event_id=e.id and evidence.evidence_type='observation'
                and evidence.evidence_ref_id=binding.observation_id);
        delete from experience_event_evidence evidence using experience_events e
        where e.id=evidence.event_id and e.owner_id=p_owner
            and evidence.evidence_type='observation' and evidence.evidence_ref_id=binding.observation_id;
        delete from perception_observations where owner_id=p_owner and id=binding.observation_id;
    end loop;
end $$;

create function invalidate_jarvis_observation_payload() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
        and old.availability is distinct from new.availability then
        perform invalidate_jarvis_observation_original(new.owner_id,new.id);
    end if;
    return new;
end $$;
create trigger jarvis_observation_payload_invalidation after update of availability on jarvis_context_payloads
for each row execute function invalidate_jarvis_observation_payload();

create function invalidate_jarvis_observation_source_revision() returns trigger
language plpgsql set search_path from current as $$
declare original record;
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        for original in select v.payload_id from jarvis_knowledge_source_versions v
            join jarvis_observation_sources b on b.owner_id=v.owner_id and b.payload_id=v.payload_id
            where v.owner_id=new.owner_id and v.source_id=new.id
        loop
            perform invalidate_jarvis_observation_original(new.owner_id,original.payload_id);
        end loop;
    end if;
    return new;
end $$;
create trigger jarvis_observation_source_invalidation after update of generation,status on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_observation_source_revision();
