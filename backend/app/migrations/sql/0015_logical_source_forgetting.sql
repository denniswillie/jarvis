-- Logical forgetting suppresses source-backed recall without erasing originals.
alter table jarvis_knowledge_sources drop constraint jarvis_knowledge_sources_status_check;
alter table jarvis_knowledge_sources add constraint jarvis_knowledge_sources_status_check
    check(status in ('active','unavailable','forgotten','deleted'));

create or replace function guard_jarvis_source_update() returns trigger language plpgsql as $$
begin
    -- Upstream invalidation must not remove a user's suppression marker.
    -- Physical deletion remains allowed; no automatic unforget is supported.
    if old.status='forgotten' then
        if new.status='unavailable' then new.status:='forgotten'; end if;
        if new.status not in ('forgotten','deleted') then
            raise exception 'Forgotten sources cannot be automatically republished' using errcode='23514';
        end if;
    end if;
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
