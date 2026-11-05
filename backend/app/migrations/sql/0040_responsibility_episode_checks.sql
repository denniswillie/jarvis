-- Extend the existing owner. No alternate check table or placeholder reminders.
alter table jarvis_responsibilities add constraint jarvis_responsibility_spec_kind check (
    jsonb_typeof(spec)='object' and coalesce(spec->>'kind','scheduled') in ('scheduled','episode_check')
    and (coalesce(spec->>'kind','scheduled')<>'episode_check' or next_due_at is null)
);

alter table jarvis_responsibility_occurrences
    alter column reminder_id drop not null,
    add column check_revision bigint not null default 1 check(check_revision>0),
    add column episode_id uuid,
    add column episode_revision bigint,
    add column check_payload jsonb,
    add column check_lifecycle text,
    add column check_resolution jsonb;

-- PostgreSQL CHECK accepts SQL NULL. JSON-null fields must not bypass the shape.
alter table jarvis_responsibility_occurrences add constraint jarvis_responsibility_episode_shape check ((
    (episode_id is null and reminder_id is not null and episode_revision is null
        and check_payload is null and check_lifecycle is null and check_resolution is null)
    or (episode_id is not null and episode_revision is not null and episode_revision>0
        and check_payload is not null and jsonb_typeof(check_payload)='object'
        and check_payload ?& array['question','origin','dependencies','admission_request','last_request','episode_started_at']
        and check_payload->>'origin'='explicit_request'
        and check_lifecycle is not null and check_lifecycle in ('tracking','resolved','dismissed','expired')
        and ((check_lifecycle='resolved' and check_resolution is not null and jsonb_typeof(check_resolution)='object'
            and check_resolution ?& array['basis','source_refs','resolved_at','applicable_episode_or_object_ref']
            and check_resolution->>'basis' in ('user_report','visual_observation','trusted_source'))
            or (check_lifecycle<>'resolved' and check_resolution is null)))
) is true);
create unique index jarvis_responsibility_episode_identity
    on jarvis_responsibility_occurrences(owner_id,episode_id) where episode_id is not null;

create function guard_jarvis_responsibility_kind() returns trigger
language plpgsql set search_path from current as $$
begin
    if coalesce(new.spec->>'kind','scheduled')<>coalesce(old.spec->>'kind','scheduled') then
        raise exception 'Responsibility kind cannot be repurposed' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_responsibility_kind_immutable before update on jarvis_responsibilities
for each row execute function guard_jarvis_responsibility_kind();

create function guard_jarvis_responsibility_episode() returns trigger
language plpgsql set search_path from current as $$
declare kind text;
begin
    select coalesce(spec->>'kind','scheduled') into kind from jarvis_responsibilities
        where owner_id=new.owner_id and id=new.responsibility_id for share;
    if kind is null or (kind='episode_check')<>(new.episode_id is not null) then
        raise exception 'Episode shape must match its canonical rule kind' using errcode='23514';
    end if;
    if tg_op='UPDATE' and old.episode_id is not null then
        if (new.id,new.owner_id,new.responsibility_id,new.episode_id,new.created_at)
            is distinct from (old.id,old.owner_id,old.responsibility_id,old.episode_id,old.created_at) then
            raise exception 'A new episode requires a new occurrence identity' using errcode='23514';
        end if;
        if new.episode_revision<>old.episode_revision and new.episode_revision<>old.episode_revision+1 then
            raise exception 'Episode correction must advance its revision once' using errcode='23514';
        end if;
        if (new.check_payload,new.check_lifecycle,new.check_resolution,new.episode_revision,
            new.status,new.report_state,new.report_event_id,new.report_revision)
            is distinct from (old.check_payload,old.check_lifecycle,old.check_resolution,old.episode_revision,
                old.status,old.report_state,old.report_event_id,old.report_revision)
            and new.check_revision<>old.check_revision+1 then
            raise exception 'Semantic check changes require a new check revision' using errcode='23514';
        end if;
        if new.check_revision<old.check_revision then
            raise exception 'Check revisions cannot go backwards' using errcode='23514';
        end if;
    end if;
    return new;
end $$;
create trigger jarvis_responsibility_episode_guard before insert or update on jarvis_responsibility_occurrences
for each row execute function guard_jarvis_responsibility_episode();
