-- Additive owner metadata only. No backfilled visual proof or queued delivery.
alter table jarvis_visual_focuses
    add column revision bigint not null default 1 check (revision > 0),
    add column episode_revision bigint not null default 1 check (episode_revision > 0),
    add column check_resolution jsonb check (check_resolution is null or coalesce((
        jsonb_typeof(check_resolution) = 'object'
        and check_resolution ?& array['basis','source_refs','resolved_at','applicable_episode_or_object_ref']
        and jsonb_typeof(check_resolution->'basis') = 'string'
        and check_resolution->>'basis' in ('user_report','visual_observation','trusted_source')
        and case when jsonb_typeof(check_resolution->'source_refs') = 'array'
            then jsonb_array_length(check_resolution->'source_refs') between 1 and 8
            else false end
        and jsonb_typeof(check_resolution->'applicable_episode_or_object_ref') = 'object'
        and jsonb_typeof(check_resolution->'resolved_at') = 'string'), false));

create function version_jarvis_visual_focus_check() returns trigger
language plpgsql set search_path from current as $$
declare rebound boolean; changed boolean;
begin
    -- A corrected descriptor starts a new episode version; old evidence must
    -- never be silently repurposed. Ordinary delivery/claim churn is excluded.
    rebound := row(new.target,new.description,new.purpose,new.mode,new.created_at)
        is distinct from row(old.target,old.description,old.purpose,old.mode,old.created_at);
    new.episode_revision := old.episode_revision + case when rebound then 1 else 0 end;
    if rebound then new.check_resolution := null; end if;
    changed := rebound or row(new.expires_at,new.cleared_at,new.completed_at,new.check_resolution)
        is distinct from row(old.expires_at,old.cleared_at,old.completed_at,old.check_resolution);
    new.revision := old.revision + case when changed then 1 else 0 end;
    return new;
end $$;
create trigger jarvis_visual_focus_check_version before update on jarvis_visual_focuses
for each row execute function version_jarvis_visual_focus_check();
