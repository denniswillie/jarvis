-- A narrow operational fence, not another search queue or memory store. It lets
-- phone writes serialize with subject changes without locking raw subject rows.
create table jarvis_attention_visual_subjects (
    owner_id text not null,
    subject_type text not null check(subject_type in ('visual_focus','conditional_reminder')),
    subject_id uuid not null,
    generation bigint not null default 1 check(generation>0),
    descriptor_digest text not null check(descriptor_digest ~ '^[0-9a-f]{64}$'),
    state text not null check(state in ('active','completed','cancelled')),
    expires_at timestamptz,
    claim_token uuid,
    completed_event_id text,
    updated_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,subject_type,subject_id)
);

create function sync_jarvis_attention_visual_subject(
    p_owner text,p_kind text,p_id uuid,p_descriptor jsonb,p_state text,
    p_expiry timestamptz,p_claim uuid,p_completed text) returns void
language plpgsql set search_path from current as $$
begin
    insert into jarvis_attention_visual_subjects
        (owner_id,subject_type,subject_id,descriptor_digest,state,expires_at,claim_token,completed_event_id)
    values(p_owner,p_kind,p_id,encode(sha256(convert_to(p_descriptor::text,'UTF8')),'hex'),
           p_state,p_expiry,p_claim,p_completed)
    on conflict(owner_id,subject_type,subject_id) do update set
        generation=jarvis_attention_visual_subjects.generation+
            case when jarvis_attention_visual_subjects.descriptor_digest<>excluded.descriptor_digest
                 or (excluded.state='cancelled' and jarvis_attention_visual_subjects.state<>'cancelled')
                 then 1 else 0 end,
        descriptor_digest=excluded.descriptor_digest,state=excluded.state,expires_at=excluded.expires_at,
        claim_token=excluded.claim_token,completed_event_id=excluded.completed_event_id,updated_at=clock_timestamp();
end $$;

create function update_jarvis_attention_visual_subject() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' then
        update jarvis_attention_visual_subjects set state='cancelled',generation=generation+1,
            claim_token=null,completed_event_id=null,updated_at=clock_timestamp()
        where owner_id=old.owner_id and subject_id=old.id
          and subject_type=case when tg_table_name='jarvis_visual_focuses' then 'visual_focus' else 'conditional_reminder' end;
    elsif tg_table_name='jarvis_visual_focuses' then
        perform sync_jarvis_attention_visual_subject(new.owner_id,'visual_focus',new.id,
            jsonb_build_object('target',new.target,'description',new.description,'purpose',new.purpose,
                               'mode',new.mode,'expires_at',new.expires_at),
            case when new.cleared_at is not null then 'cancelled' when new.completed_at is not null then 'completed' else 'active' end,
            new.expires_at,new.trigger_claim_token,new.completion_evidence->>'attention_event_id');
    else
        perform sync_jarvis_attention_visual_subject(new.owner_id,'conditional_reminder',new.id,
            jsonb_build_object('title',new.title,'details',new.details,'condition',new.condition,
                               'remind_at',new.remind_at,'cancelled_at',new.metadata->>'attention_cancelled_at'),
            case when new.status in ('deleted','failed') or new.metadata ? 'attention_cancelled_at' then 'cancelled'
                 when new.status='completed' or new.triggered_at is not null then 'completed' else 'active' end,
            null,new.trigger_claim_token,new.metadata->>'attention_completed_event_id');
    end if;
    return null;
end $$;
create trigger jarvis_visual_attention_focus_changed after insert or update or delete on jarvis_visual_focuses
for each row execute function update_jarvis_attention_visual_subject();
-- This sorts AFTER the reminder source-invalidation trigger from migration 23.
create trigger jarvis_visual_attention_reminder_changed after insert or update or delete on jarvis_reminders
for each row execute function update_jarvis_attention_visual_subject();

select sync_jarvis_attention_visual_subject(owner_id,'visual_focus',id,
    jsonb_build_object('target',target,'description',description,'purpose',purpose,'mode',mode,'expires_at',expires_at),
    case when cleared_at is not null then 'cancelled' when completed_at is not null then 'completed' else 'active' end,
expires_at,trigger_claim_token,completion_evidence->>'attention_event_id') from jarvis_visual_focuses;
select sync_jarvis_attention_visual_subject(owner_id,'conditional_reminder',id,
    jsonb_build_object('title',title,'details',details,'condition',condition,'remind_at',remind_at,
                      'cancelled_at',metadata->>'attention_cancelled_at'),
    case when status in ('deleted','failed') or metadata ? 'attention_cancelled_at' then 'cancelled'
         when status='completed' or triggered_at is not null then 'completed' else 'active' end,
    null,trigger_claim_token,metadata->>'attention_completed_event_id') from jarvis_reminders;

create unique index jarvis_attention_one_visual_observation
    on jarvis_attention_intents(owner_id,coalesce_key) where source_namespace='visual_alert';

create function guard_jarvis_visual_attention_intent() returns trigger
language plpgsql set search_path from current as $$
declare body jsonb; kind text; valid boolean;
begin
    if new.source_namespace not in ('visual_alert','prospective_event') then return new; end if;
    select content into body from jarvis_context_payloads
        where owner_id=new.owner_id and id=new.source_payload_id and availability='available';
    kind:=case when new.source_namespace='visual_alert' then 'visual' else 'prospective' end;
    if not coalesce(body->>'format'='jarvis.attention.visual_notification.v1'
        and body->>'source_namespace'=new.source_namespace and body->>'source_id'=new.source_id::text
        and body->>'session_id'=new.preferred_session_id::text and body->>'authored_by'='visual_analyzer'
        and new.task_id is null and new.source_revision=1
        and (body->>'expires_at')::timestamptz=new.expires_at, false) then
        raise exception 'Visual attention requires its exact notification envelope' using errcode='23514';
    end if;
    valid:=exists(select 1 from jarvis_visual_delivery_sources b
        join jarvis_context_payload_dependencies d on d.owner_id=b.owner_id and d.source_payload_id=b.payload_id
        where b.owner_id=new.owner_id and b.delivery_kind=kind and b.delivery_id=new.source_id
          and b.session_id=new.preferred_session_id and b.payload_id::text=body->>'observation_payload_id'
          and d.dependent_payload_id=new.source_payload_id);
    if new.source_namespace='visual_alert' then
        valid:=valid and exists(select 1 from visual_interrupts
            where owner_id=new.owner_id and id=new.source_id and message=body->>'message')
            and new.coalesce_key='observation:' || (body->>'observation_payload_id');
    else
        valid:=valid and exists(select 1 from jarvis_prospective_delivery_events e
            join jarvis_attention_visual_subjects s on s.owner_id=e.owner_id
              and s.subject_id=coalesce(e.visual_focus_id,e.reminder_id)
            where e.owner_id=new.owner_id and e.id=new.source_id and e.message=body->>'message'
              and s.subject_type=body->>'subject_type' and s.subject_id::text=body->>'subject_id'
              and s.generation::text=body->>'subject_generation'
              and s.claim_token=e.claim_token and s.claim_token::text=body->>'claim_token' and s.state='active');
    end if;
    if not coalesce(valid,false) then
        raise exception 'Visual attention requires its exact owned observation and subject' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_visual_attention_intent_guard before insert on jarvis_attention_intents
for each row execute function guard_jarvis_visual_attention_intent();

create function revoke_jarvis_visual_attention_original() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or new.message is distinct from old.message then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_attention_intents i
              where i.owner_id=p.owner_id and i.source_payload_id=p.id and i.source_id=old.id
                and i.source_namespace=case when tg_table_name='visual_interrupts' then 'visual_alert' else 'prospective_event' end);
    end if;
    return null;
end $$;
create trigger jarvis_visual_attention_original_changed after update of message or delete on visual_interrupts
for each row execute function revoke_jarvis_visual_attention_original();
create trigger jarvis_prospective_attention_original_changed after update of message or delete on jarvis_prospective_delivery_events
for each row execute function revoke_jarvis_visual_attention_original();

create function revoke_jarvis_visual_attention_copies() returns trigger
language plpgsql set search_path from current as $$
declare intent record; event record;
begin
    if new.availability not in ('deleted','expired','unavailable')
       or old.content->>'format' is distinct from 'jarvis.attention.visual_notification.v1' then return new; end if;
    for intent in select source_namespace,source_id,id from jarvis_attention_intents
        where owner_id=new.owner_id and source_payload_id=new.id order by id loop
        if intent.source_namespace='visual_alert' then
            update visual_interrupts set message='',reason='source_unavailable',confidence=0,dedupe_bucket=''
            where owner_id=new.owner_id and id=intent.source_id;
        else
            select * into event from jarvis_prospective_delivery_events where owner_id=new.owner_id and id=intent.source_id for update;
            if not found then continue; end if;
            update jarvis_prospective_delivery_events set message='[source unavailable]',evidence='{}'::jsonb,
                event_type=case when event_type in ('generated','dispatched_to_session') then 'failed' else event_type end,
                metadata=jsonb_build_object('delivery_route','attention','attention_owned',true,
                    'reason','source_unavailable','playback_status','unconfirmed','physical_action_status','not_established'),
                updated_at=clock_timestamp()
            where owner_id=new.owner_id and id=event.id;
            if event.event_type in ('generated','dispatched_to_session') then
                update jarvis_visual_focuses set trigger_claim_token=null,trigger_claimed_at=null
                where owner_id=new.owner_id and id=event.visual_focus_id and trigger_claim_token=event.claim_token;
                update jarvis_reminders set trigger_claim_token=null,trigger_claimed_at=null
                where owner_id=new.owner_id and id=event.reminder_id and trigger_claim_token=event.claim_token;
            end if;
            update jarvis_visual_focuses set completion_evidence=jsonb_build_object(
                'attention_intent_id',intent.id,'status','source_unavailable',
                'meaning','Historical search notification, not current physical evidence')
            where owner_id=new.owner_id and completion_evidence->>'attention_intent_id'=intent.id::text;
        end if;
    end loop;
    return new;
end $$;
create trigger jarvis_visual_attention_payload_changed after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_visual_attention_copies();
-- No prior visual notification is enqueued or replayed by this migration.
