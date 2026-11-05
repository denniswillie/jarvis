-- Keep M03's frozen legacy row shapes. These exact links are replay tombstones,
-- not another notification queue or memory store.
create table jarvis_visual_delivery_sources (
    owner_id text not null,
    delivery_kind text not null check(delivery_kind in ('visual','prospective')),
    delivery_id uuid not null,
    session_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,delivery_kind,delivery_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_visual_delivery_payload_idx on jarvis_visual_delivery_sources(owner_id,payload_id);
create trigger jarvis_visual_delivery_sources_immutable before update on jarvis_visual_delivery_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_visual_delivery_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not lock_jarvis_copy_sources(new.owner_id,array[new.payload_id]) or not exists(
        select 1 from jarvis_observation_sources
        where owner_id=new.owner_id and session_id=new.session_id and payload_id=new.payload_id
    ) then
        raise exception 'Visual delivery requires its current owned observation' using errcode='23514';
    end if;
    if new.delivery_kind='visual' then
        if not exists(select 1 from visual_interrupts
            where owner_id=new.owner_id and id=new.delivery_id and session_id=new.session_id) then
            raise exception 'Visual delivery identity does not match its owned record' using errcode='23514';
        end if;
    else
        if not exists(select 1 from jarvis_prospective_delivery_events
            where owner_id=new.owner_id and id=new.delivery_id and session_id=new.session_id) then
            raise exception 'Prospective delivery identity does not match its owned record' using errcode='23514';
        end if;
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_source_guard before insert on jarvis_visual_delivery_sources
for each row execute function guard_jarvis_visual_delivery_source();

create function revoke_jarvis_visual_delivery_copies(p_owner text,p_payload uuid) returns void
language plpgsql set search_path from current as $$
declare event record;
begin
    update visual_interrupts v set message='',reason='source_unavailable',confidence=0,dedupe_bucket=''
    where v.owner_id=p_owner and exists(select 1 from jarvis_visual_delivery_sources b
        where b.owner_id=p_owner and b.delivery_kind='visual'
            and b.delivery_id=v.id and b.payload_id=p_payload);
    for event in select e.* from jarvis_prospective_delivery_events e
        join jarvis_visual_delivery_sources b on b.owner_id=e.owner_id
            and b.delivery_kind='prospective' and b.delivery_id=e.id
        where b.owner_id=p_owner and b.payload_id=p_payload order by e.id for update of e
    loop
        update jarvis_prospective_delivery_events set
            message='[source unavailable]',evidence='{}'::jsonb,
            event_type=case when event_type in ('generated','dispatched_to_session') then 'failed' else event_type end,
            metadata=jsonb_build_object('reason','source_unavailable',
                'source_payload_id',p_payload,'prior_event_type',event.event_type,
                'playback_status',coalesce(metadata->>'playback_status','unconfirmed')),
            updated_at=clock_timestamp()
        where owner_id=p_owner and id=event.id;
        if event.event_type in ('generated','dispatched_to_session') then
            update jarvis_visual_focuses set trigger_claim_token=null,trigger_claimed_at=null
            where owner_id=p_owner and id=event.visual_focus_id and trigger_claim_token=event.claim_token;
            update jarvis_reminders set trigger_claim_token=null,trigger_claimed_at=null,
                last_trigger_error='source_unavailable',updated_at=clock_timestamp()
            where owner_id=p_owner and id=event.reminder_id and trigger_claim_token=event.claim_token;
        end if;
    end loop;
    -- A historical find-once delivery is not undone or replayed. Its discarded
    -- evidence cannot continue masquerading as a current matching observation.
    update jarvis_visual_focuses set completion_evidence=jsonb_build_object(
        'source_payload_id',p_payload,'status','source_unavailable',
        'meaning','Past search notification bookkeeping, not current match evidence')
    where owner_id=p_owner and completion_evidence->>'source_payload_id'=p_payload::text;
end $$;

create function revoke_jarvis_visual_delivery_payload() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
        and (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        perform revoke_jarvis_visual_delivery_copies(new.owner_id,new.id);
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_payload_revocation after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_visual_delivery_payload();

create function revoke_jarvis_visual_delivery_revision() returns trigger
language plpgsql set search_path from current as $$
declare original record;
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        for original in select distinct v.payload_id from jarvis_knowledge_source_versions v
            join jarvis_visual_delivery_sources b on b.owner_id=v.owner_id and b.payload_id=v.payload_id
            where v.owner_id=new.owner_id and v.source_id=new.id
        loop
            perform revoke_jarvis_visual_delivery_copies(new.owner_id,original.payload_id);
        end loop;
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_source_revocation after update of generation,status on jarvis_knowledge_sources
for each row execute function revoke_jarvis_visual_delivery_revision();
