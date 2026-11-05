-- Source lifecycle recovery metadata, not another personal memory engine.
-- Backups must include this journal epoch. No pre-migration history is invented.
create table jarvis_source_journal_streams (
    owner_id text primary key check(length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    epoch uuid not null default gen_random_uuid(),
    last_sequence bigint not null default 0 check(last_sequence>=0),
    initialized_at timestamptz not null default clock_timestamp(),
    unique(owner_id,epoch)
);

create table jarvis_source_journal_events (
    owner_id text not null references jarvis_source_journal_streams(owner_id),
    sequence bigint not null check(sequence>0),
    id uuid not null default gen_random_uuid(),
    kind text not null check(kind in ('source','payload','area','cleanup')),
    object_id uuid not null,
    detail jsonb not null check(jsonb_typeof(detail)='object' and octet_length(detail::text)<=8192),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,sequence),
    unique(owner_id,id)
);
create trigger jarvis_source_journal_immutable before update or delete on jarvis_source_journal_events
for each row execute function reject_jarvis_foundation_update();

create function append_jarvis_source_journal(p_owner text,p_kind text,p_object uuid,p_detail jsonb)
returns void language plpgsql set search_path from current as $$
declare next_sequence bigint;
begin
    -- This row lock serializes committed owner order. Rollback also rolls back
    -- the counter; a cursor cannot skip a still-uncommitted lower sequence.
    insert into jarvis_source_journal_streams(owner_id,last_sequence) values(p_owner,1)
    on conflict(owner_id) do update
        set last_sequence=jarvis_source_journal_streams.last_sequence+1
    returning last_sequence into next_sequence;
    insert into jarvis_source_journal_events(owner_id,sequence,kind,object_id,detail)
        values(p_owner,next_sequence,p_kind,p_object,p_detail);
end; $$;

create function record_jarvis_source_lifecycle() returns trigger
language plpgsql set search_path from current as $$
declare detail jsonb;
begin
    -- Hash the stable external identity instead of copying a private path/name.
    detail:=jsonb_build_object(
        'area_id',new.area_id,'revision',new.current_revision,'generation',new.generation,'status',new.status,
        'identity_digest',encode(sha256(convert_to(
            jsonb_build_array(new.owner_id,new.area_id,new.namespace,new.source_key)::text,'UTF8')),'hex'));
    perform append_jarvis_source_journal(new.owner_id,'source',new.id,detail);
    return new;
end; $$;
create trigger jarvis_source_lifecycle_journal after insert or update on jarvis_knowledge_sources
for each row execute function record_jarvis_source_lifecycle();

create function record_jarvis_payload_lifecycle() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and new.availability is not distinct from old.availability
        and new.content_digest is not distinct from old.content_digest then return new; end if;
    perform append_jarvis_source_journal(new.owner_id,'payload',new.id,jsonb_build_object(
        'availability',new.availability,'content_digest',new.content_digest,
        'prior_content_digest',case when tg_op='UPDATE' then old.content_digest else null end));
    return new;
end; $$;
create trigger jarvis_payload_lifecycle_journal after insert or update on jarvis_context_payloads
for each row execute function record_jarvis_payload_lifecycle();

create function record_jarvis_area_lifecycle() returns trigger
language plpgsql set search_path from current as $$
begin
    perform append_jarvis_source_journal(new.owner_id,'area',new.id,
        jsonb_build_object('status',new.status,'revision',new.revision));
    return new;
end; $$;
create trigger jarvis_area_lifecycle_journal after insert or update on jarvis_knowledge_areas
for each row execute function record_jarvis_area_lifecycle();

create function record_jarvis_cleanup_lifecycle() returns trigger
language plpgsql set search_path from current as $$
declare detail jsonb;
begin
    if new.operation<>'delete' then return new; end if;
    if tg_op='UPDATE' and not (
        (new.status in ('done','failed','cancelled','outcome_unknown') and new.status is distinct from old.status)
        or (old.status in ('done','failed','cancelled') and new.status is distinct from old.status)
        or new.generation is distinct from old.generation
        or new.cleanup_operation_ids is distinct from old.cleanup_operation_ids
    ) then return new; end if;
    -- Preserve the deletion intent and terminal operation inventory. Do not
    -- copy provider errors, credentials, source text or every lease/poll update.
    select jsonb_build_object(
        'source_id',v.source_id,'source_revision',v.revision,'document_id',v.document_id,
        'bank_name',b.bank_name,'generation',new.generation,'status',new.status,
        'native_operation_ids',new.cleanup_operation_ids,
        'retain_operation_id',r.id,'retain_dispatch_count',coalesce(r.dispatch_count,0))
    into detail from jarvis_knowledge_source_versions v
    join jarvis_memory_banks b on b.owner_id=v.owner_id and b.id=v.bank_id
    left join jarvis_memory_outbox r on r.owner_id=v.owner_id and r.document_id=v.document_id
        and r.operation='retain'
    where v.owner_id=new.owner_id and v.document_id=new.document_id;
    perform append_jarvis_source_journal(new.owner_id,'cleanup',new.id,detail);
    return new;
end; $$;
create trigger jarvis_cleanup_lifecycle_journal after insert or update on jarvis_memory_outbox
for each row execute function record_jarvis_cleanup_lifecycle();

-- Establish an epoch for existing owners without manufacturing historical
-- events or scanning/copying their source contents.
insert into jarvis_source_journal_streams(owner_id)
select owner_id from jarvis_knowledge_sources
union select owner_id from jarvis_context_payloads
union select owner_id from jarvis_knowledge_areas
on conflict do nothing;
