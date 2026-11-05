-- Generated-output allowances and exact derived effects are bound to one task.
create or replace function guard_jarvis_sandbox_grant_attachment() returns trigger
language plpgsql set search_path from current as $$
declare selected_capability text;
begin
    select capability into selected_capability from jarvis_operation_grants
        where owner_id=new.owner_id and id=new.grant_id for update;
    if selected_capability in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
       and exists(select 1 from jarvis_task_grants
                  where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'An execution or generated-output grant is bound to one task' using errcode='23514';
    end if;
    return new;
end $$;
