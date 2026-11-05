-- Ready workers advertise optional computation support; unknown old resources fence replacements.
alter table jarvis_task_workers add column capabilities text[] not null default '{}'
    check(cardinality(capabilities)<=16);
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_allocations_owner_id_grant_id_key;
create unique index jarvis_sandbox_one_unfinished_grant on jarvis_sandbox_allocations(owner_id,grant_id)
    where phase<>'cleaned';
create function guard_jarvis_sandbox_grant_task() returns trigger
language plpgsql set search_path from current as $$
begin
    if exists(select 1 from jarvis_sandbox_allocations
              where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'A sandbox grant cannot move between tasks' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_grant_task_guard before insert on jarvis_sandbox_allocations
for each row execute function guard_jarvis_sandbox_grant_task();

-- Bind scope at attachment, not only after the first VM allocation.
do $$
begin
    if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants g
              on g.owner_id=tg.owner_id and g.id=tg.grant_id where g.capability='sandbox.execute'
              group by tg.owner_id,tg.grant_id having count(*)>1) then
        raise exception 'Existing sandbox grants require unambiguous task ownership' using errcode='23514';
    end if;
end $$;
create function guard_jarvis_sandbox_grant_attachment() returns trigger
language plpgsql set search_path from current as $$
declare selected_capability text;
begin
    select capability into selected_capability from jarvis_operation_grants
        where owner_id=new.owner_id and id=new.grant_id for update;
    if selected_capability='sandbox.execute' and exists(select 1 from jarvis_task_grants
        where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'A sandbox grant is bound to one task' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_grant_attachment before insert or update on jarvis_task_grants
for each row execute function guard_jarvis_sandbox_grant_attachment();
