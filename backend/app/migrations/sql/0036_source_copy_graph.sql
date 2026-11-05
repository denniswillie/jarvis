-- Remove per-descendant trigger recursion without weakening atomic read fences.
-- This is still one transaction, not bounded/resumable graph cleanup.
create or replace function revoke_jarvis_source_bound_copies(
    p_owner text,p_source uuid,p_availability text) returns void
language plpgsql set search_path from current as $$
begin
    if p_availability not in ('deleted','expired','unavailable') then
        raise exception 'Invalid source-copy removal state' using errcode='23514';
    end if;
    update jarvis_messages set content='',metadata=jsonb_build_object(
        'response_kind','source_bound_unavailable','source_status',p_availability,'source_payload_id',p_source)
    where owner_id=p_owner and source_payload_id=p_source;

    -- One statement updates the reachable mutable descendants. Its AFTER-row
    -- triggers still run, including message, task, source and outbox fences,
    -- but ordinarily find no mutable children left to recursively update.
    -- Do not disable those triggers: a writer which held a descendant's share
    -- lock may commit a new child after this statement's snapshot was taken.
    -- The descendant's ordinary trigger then catches that late committed copy.
    with recursive affected(id) as (
        select d.dependent_payload_id
        from jarvis_context_payload_dependencies d
        join jarvis_context_payloads p on p.owner_id=d.owner_id and p.id=d.dependent_payload_id
        where d.owner_id=p_owner and d.source_payload_id=p_source
            and (p.availability in ('available','not_finalized')
                 or (p_availability='deleted' and p.availability<>'deleted'))
        union
        select d.dependent_payload_id
        from affected parent
        join jarvis_context_payload_dependencies d on d.owner_id=p_owner and d.source_payload_id=parent.id
        join jarvis_context_payloads p on p.owner_id=d.owner_id and p.id=d.dependent_payload_id
        where p.availability in ('available','not_finalized')
            or (p_availability='deleted' and p.availability<>'deleted')
    )
    update jarvis_context_payloads p set availability=p_availability,content=null
    where p.owner_id=p_owner and p.id in (select id from affected)
        and (p.availability in ('available','not_finalized')
             or (p_availability='deleted' and p.availability<>'deleted'));
end $$;
