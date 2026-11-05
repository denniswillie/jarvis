-- Original-backed copies, not another memory engine. Only explicit migration
-- execution applies this to a live schema; all acceptance uses manifested data.
create index jarvis_attention_source_payload_idx on jarvis_attention_intents(owner_id,source_payload_id);

create table jarvis_context_payload_dependencies (
    owner_id text not null,
    source_payload_id uuid not null,
    dependent_payload_id uuid not null,
    primary key(owner_id,source_payload_id,dependent_payload_id),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    check(source_payload_id<>dependent_payload_id)
);
create index jarvis_context_dependency_child_idx on jarvis_context_payload_dependencies(owner_id,dependent_payload_id);
create trigger jarvis_context_dependencies_immutable before update on jarvis_context_payload_dependencies
for each row execute function reject_jarvis_foundation_update();

alter table jarvis_messages add column source_payload_id uuid;
alter table jarvis_messages add constraint jarvis_message_source_scope
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id);
create index jarvis_messages_source_payload_idx on jarvis_messages(owner_id,source_payload_id)
    where source_payload_id is not null;

-- Bind only exact, owned historical references. No text matching or inference.
update jarvis_messages m set source_payload_id=i.source_payload_id
from jarvis_attention_intents i
where m.owner_id=i.owner_id and m.metadata->>'response_kind'='expert_tts'
    and m.metadata->>'intent_id'=i.id::text;

insert into jarvis_context_payload_dependencies(owner_id,source_payload_id,dependent_payload_id)
select distinct p.owner_id,i.source_payload_id,p.id
from jarvis_context_payloads p
join jarvis_context_events e on e.owner_id=p.owner_id and e.payload_id=p.id
join jarvis_attention_attempts a on a.owner_id=p.owner_id and a.id::text=p.content->>'attempt_id'
join jarvis_attention_intents i on i.owner_id=a.owner_id and i.id=a.intent_id
join jarvis_context_payloads original on original.owner_id=i.owner_id and original.id=i.source_payload_id
where e.source_namespace like 'voice:%' and e.kind='assistant_output'
    and p.content->>'authored_by'='frontier_expert'
    and (original.created_at,original.id)<(p.created_at,p.id)
on conflict do nothing;

do $$
begin
    if exists(select 1 from jarvis_messages where metadata->>'response_kind'='expert_tts'
              and source_payload_id is null) then
        raise exception 'Unmapped expert history requires source inventory before migration' using errcode='23514';
    end if;
    if exists(select 1 from jarvis_context_payloads p join jarvis_context_events e
              on e.owner_id=p.owner_id and e.payload_id=p.id
              where e.source_namespace like 'voice:%' and e.kind='assistant_output'
                and p.content->>'authored_by'='frontier_expert'
                and not exists(select 1 from jarvis_context_payload_dependencies d
                    where d.owner_id=p.owner_id and d.dependent_payload_id=p.id)) then
        raise exception 'Unmapped expert context requires source inventory before migration' using errcode='23514';
    end if;
end $$;
alter table jarvis_messages add constraint jarvis_expert_message_requires_source
    check(metadata->>'response_kind' is distinct from 'expert_tts' or source_payload_id is not null);

create function lock_jarvis_copy_source(p_owner text,p_source uuid) returns text
language plpgsql set search_path from current as $$
declare source_state text; intent record; has_intent boolean:=false; has_current boolean:=false;
begin
    select availability into source_state from jarvis_context_payloads
    where owner_id=p_owner and id=p_source for share;
    if not found then return 'not_found'; end if;
    if source_state<>'available' then return 'removed'; end if;
    -- Match revocation ordering and prevent a late copy after an intent was
    -- superseded while its original remains retained as historical evidence.
    for intent in select state from jarvis_attention_intents
        where owner_id=p_owner and source_payload_id=p_source order by id for share loop
        has_intent:=true;
        has_current:=has_current or intent.state<>'revoked';
    end loop;
    if has_intent and not has_current then return 'revoked'; end if;
    return 'available';
end $$;

create function guard_jarvis_context_dependency() returns trigger
language plpgsql set search_path from current as $$
begin
    if lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        raise exception 'A copy requires an available owned source' using errcode='23514';
    end if;
    -- Immutable database creation ordering makes this a DAG, including the
    -- rare equal-timestamp case. New materializations depend on older originals.
    if not exists(select 1 from jarvis_context_payloads original
        join jarvis_context_payloads copy on copy.owner_id=original.owner_id
        where original.owner_id=new.owner_id and original.id=new.source_payload_id
            and copy.id=new.dependent_payload_id and copy.availability in ('available','not_finalized')
            and (original.created_at,original.id)<(copy.created_at,copy.id)) then
        raise exception 'A dependency must point from an older original to a new owned copy' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_dependency_guard before insert on jarvis_context_payload_dependencies
for each row execute function guard_jarvis_context_dependency();

create function guard_jarvis_message_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and old.source_payload_id is not null then
        if new.source_payload_id is distinct from old.source_payload_id or new.owner_id<>old.owner_id then
            raise exception 'A source-bound message cannot detach or change its source' using errcode='23514';
        end if;
        if new.content is distinct from old.content and new.content<>'' then
            raise exception 'A source-bound message can only redact its copied text' using errcode='23514';
        end if;
        if new.content='' then return new; end if;
    end if;
    if new.source_payload_id is not null
       and lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        raise exception 'A message copy requires an available owned source' using errcode='23514';
    end if;
    if new.metadata->>'response_kind'='expert_tts' and not exists(
        select 1 from jarvis_attention_intents i where i.owner_id=new.owner_id
            and i.id::text=new.metadata->>'intent_id' and i.source_payload_id=new.source_payload_id) then
        raise exception 'Expert message provenance must match its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_message_source_guard before insert or update of source_payload_id,owner_id,content,metadata on jarvis_messages
for each row execute function guard_jarvis_message_source();

create function guard_jarvis_expert_context_source() returns trigger
language plpgsql set search_path from current as $$
declare payload jsonb;
begin
    if new.kind<>'assistant_output' or new.source_namespace not like 'voice:%' then return new; end if;
    select content into payload from jarvis_context_payloads where owner_id=new.owner_id and id=new.payload_id;
    if payload->>'authored_by'='frontier_expert' and not exists(
        select 1 from jarvis_context_payload_dependencies d
        join jarvis_attention_attempts a on a.owner_id=d.owner_id and a.id::text=payload->>'attempt_id'
        join jarvis_attention_intents i on i.owner_id=a.owner_id and i.id=a.intent_id
        where d.owner_id=new.owner_id and d.dependent_payload_id=new.payload_id
            and d.source_payload_id=i.source_payload_id) then
        raise exception 'Expert context requires exact owned source lineage' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_expert_context_source_guard before insert on jarvis_context_events
for each row execute function guard_jarvis_expert_context_source();

create function revoke_jarvis_source_bound_copies(p_owner text,p_source uuid,p_availability text) returns void
language plpgsql set search_path from current as $$
begin
    if p_availability not in ('deleted','expired','unavailable') then
        raise exception 'Invalid source-copy removal state' using errcode='23514';
    end if;
    update jarvis_messages set content='',metadata=jsonb_build_object(
        'response_kind','source_bound_unavailable','source_status',p_availability,'source_payload_id',p_source)
    where owner_id=p_owner and source_payload_id=p_source;
    -- Existing payload-removal triggers invalidate task manifests, source
    -- mappings and Hindsight outbox records. Each descendant follows this path.
    update jarvis_context_payloads p set availability=p_availability,content=null
    where p.owner_id=p_owner and exists(select 1 from jarvis_context_payload_dependencies d
        where d.owner_id=p.owner_id and d.source_payload_id=p_source and d.dependent_payload_id=p.id)
        and (p.availability in ('available','not_finalized')
             or (p_availability='deleted' and p.availability<>'deleted'));
end $$;

create function revoke_jarvis_payload_copies() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
       and (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        perform revoke_jarvis_source_bound_copies(new.owner_id,new.id,new.availability);
    end if;
    return new;
end $$;
create trigger jarvis_source_bound_payload_removal after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_payload_copies();

create function revoke_jarvis_attention_copies() returns trigger
language plpgsql set search_path from current as $$
declare removal_state text;
begin
    if new.state='revoked' and old.state is distinct from new.state then
        select case when availability in ('deleted','expired','unavailable') then availability
                    else 'unavailable' end into removal_state
        from jarvis_context_payloads where owner_id=new.owner_id and id=new.source_payload_id;
        perform revoke_jarvis_source_bound_copies(new.owner_id,new.source_payload_id,removal_state);
    end if;
    return new;
end $$;
create trigger jarvis_attention_source_copy_revocation after update of state on jarvis_attention_intents
for each row execute function revoke_jarvis_attention_copies();

create function propagate_jarvis_context_manifest_removal() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.status<>'active' and old.status is distinct from new.status then
        -- Reuse the exact result/attempt lineage already used by memory-source
        -- invalidation. Corrections retain originals; removals purge originals.
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,array[new.id],new.status='revoked');
    end if;
    return new;
end $$;
create trigger jarvis_context_manifest_result_invalidation after update of status on jarvis_task_context_manifests
for each row execute function propagate_jarvis_context_manifest_removal();

-- Retire already-invalid copies only after all exact links exist. Any unmapped
-- historical expert data above aborts the migration instead of guessing.
do $$
declare source record; source_state text; availability text;
begin
    for source in select owner_id,source_payload_id from jarvis_context_payload_dependencies
                  union select owner_id,source_payload_id from jarvis_messages where source_payload_id is not null loop
        source_state:=lock_jarvis_copy_source(source.owner_id,source.source_payload_id);
        if source_state<>'available' then
            select case when p.availability in ('deleted','expired','unavailable') then p.availability
                        else 'unavailable' end into availability from jarvis_context_payloads p
            where p.owner_id=source.owner_id and p.id=source.source_payload_id;
            perform revoke_jarvis_source_bound_copies(source.owner_id,source.source_payload_id,availability);
        end if;
    end loop;
end $$;
