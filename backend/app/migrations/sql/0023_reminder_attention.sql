-- Scheduled notifications share the existing inbox/attempt/PCM ledger.
-- Exact bindings are revocation tombstones, not another queue or memory engine.
create table jarvis_reminder_attention_sources (
    owner_id text not null,
    reminder_id uuid not null,
    event_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,event_id),
    unique(owner_id,payload_id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_reminder_attention_subject_idx on jarvis_reminder_attention_sources(owner_id,reminder_id);
create trigger jarvis_reminder_attention_identity_immutable before update on jarvis_reminder_attention_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_reminder_attention_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(select 1 from jarvis_reminders r
        join jarvis_reminder_events e on e.owner_id=r.owner_id and e.reminder_id=r.id
        join jarvis_context_payloads p on p.owner_id=r.owner_id and p.id=new.payload_id
        where r.owner_id=new.owner_id and r.id=new.reminder_id and r.status<>'deleted'
          and e.id=new.event_id and e.event_type='generated'
          and p.availability='available'
          and p.content->>'format'='jarvis.attention.notification.v1'
          and p.content->>'source_namespace'='reminder_event'
          and p.content->>'source_id'=e.id::text and p.content->>'reminder_id'=r.id::text
          and p.content->>'authored_by'='stored_reminder'
          and p.content->>'title'=r.title and p.content->>'details'=r.details
          and p.content->>'message'=e.message) then
        raise exception 'Reminder attention requires its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_reminder_attention_source_guard before insert on jarvis_reminder_attention_sources
for each row execute function guard_jarvis_reminder_attention_source();

create function revoke_jarvis_reminder_attention_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or
       (new.title,new.details,new.remind_at,new.condition) is distinct from
       (old.title,old.details,old.remind_at,old.condition) or new.status='deleted' or
       new.metadata->>'attention_cancelled_at' is distinct from old.metadata->>'attention_cancelled_at' then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_reminder_attention_sources b
              where b.owner_id=p.owner_id and b.payload_id=p.id and b.reminder_id=old.id);
    end if;
    return null;
end $$;
create trigger jarvis_reminder_attention_subject_changed
after update of title,details,remind_at,condition,status,metadata or delete on jarvis_reminders
for each row execute function revoke_jarvis_reminder_attention_source();

create function revoke_jarvis_reminder_attention_event() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or new.message is distinct from old.message then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_reminder_attention_sources b
              where b.owner_id=p.owner_id and b.payload_id=p.id and b.event_id=old.id);
    end if;
    return null;
end $$;
create trigger jarvis_reminder_attention_event_changed
after update of message or delete on jarvis_reminder_events
for each row execute function revoke_jarvis_reminder_attention_event();

create function revoke_jarvis_reminder_attention_copy() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable') and
       (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        update jarvis_reminder_events e set message='[source unavailable]',
            metadata=jsonb_build_object('reason','source_unavailable',
                'playback_status','unconfirmed','physical_action_status','not_established')
        where e.owner_id=new.owner_id and exists(select 1 from jarvis_reminder_attention_sources b
            where b.owner_id=e.owner_id and b.event_id=e.id and b.payload_id=new.id);
    end if;
    return new;
end $$;
create trigger jarvis_reminder_attention_payload_changed
after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_reminder_attention_copy();

-- The original event records generation, not provider/client delivery. The
-- existing attention attempts are the sole delivery truth; do not project them
-- back while holding the source fence (that reverses source/event lock order).
-- No historical events are retroactively enqueued or replayed by installation.
