-- Rules produce ordinary reminders; notification delivery never proves an action.
alter table jarvis_reminders add column not_before timestamptz;
create unique index if not exists jarvis_reminders_responsibility_owner_idx
    on jarvis_reminders(owner_id,id);

create table jarvis_responsibilities (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid,
    revision bigint not null default 1 check(revision > 0),
    status text not null default 'active' check(status in ('active','completed','cancelled')),
    spec jsonb not null,
    next_due_at timestamptz,
    snoozed_until timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id)
);
create index jarvis_responsibilities_due_idx on jarvis_responsibilities(owner_id,next_due_at)
    where status='active' and next_due_at is not null;

create table jarvis_responsibility_occurrences (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    responsibility_id uuid not null,
    rule_revision bigint not null,
    scheduled_for timestamptz not null,
    reminder_id uuid not null,
    status text not null default 'open' check(status in ('open','closed','superseded','cancelled')),
    report_state text not null default 'unknown'
        check(report_state in ('unknown','user_reported_done','user_reported_not_done')),
    report_event_id uuid,
    report_revision integer,
    reported_at timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,responsibility_id,rule_revision,scheduled_for),
    foreign key(owner_id,responsibility_id) references jarvis_responsibilities(owner_id,id),
    foreign key(owner_id,reminder_id) references jarvis_reminders(owner_id,id),
    foreign key(owner_id,report_event_id,report_revision) references jarvis_context_events(owner_id,id,revision),
    check((report_event_id is null) = (report_revision is null))
);
create index jarvis_responsibility_occurrences_rule_idx
    on jarvis_responsibility_occurrences(owner_id,responsibility_id,scheduled_for desc);

create table jarvis_responsibility_requests (
    owner_id text not null,
    request_event_id uuid not null,
    request_revision integer not null,
    operation text not null,
    request_slot integer not null check(request_slot between 1 and 20),
    arguments_hash text not null,
    responsibility_id uuid not null,
    occurrence_id uuid,
    report_state text,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_event_id,request_revision,operation,request_slot),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,responsibility_id) references jarvis_responsibilities(owner_id,id),
    foreign key(owner_id,occurrence_id) references jarvis_responsibility_occurrences(owner_id,id)
);


-- Reversible policy delay is separate from the owner's own inbox snooze.
create table jarvis_responsibility_attention_delays (
    owner_id text not null,
    intent_id uuid not null,
    base_not_before timestamptz not null,
    applied_not_before timestamptz not null,
    primary key(owner_id,intent_id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id) on delete cascade
);

create function jarvis_responsibility_notice_time(p_owner text,p_reminder uuid,p_now timestamptz)
returns timestamptz language plpgsql stable set search_path from current as $$
declare policy jsonb; candidate timestamptz; local_now timestamp; starts time; ends time; zone text;
begin
    if not exists(select 1 from jarvis_responsibility_occurrences
        where owner_id=p_owner and reminder_id=p_reminder) then return p_now; end if;
    select case when p.availability='available' then p.content else null end into policy
    from jarvis_capability_preferences q join jarvis_context_payloads p
        on p.owner_id=q.owner_id and p.id=q.policy_payload_id
    where q.owner_id=p_owner and q.capability='responsibility_notifications'
      and q.knowledge_area_id is null order by q.revision desc limit 1;
    if not found then return p_now; end if;
    if policy is null then return 'infinity'::timestamptz; end if;
    candidate:=greatest(p_now,(policy->>'paused_until')::timestamptz);
    starts:=(policy->>'quiet_start')::time;
    ends:=(policy->>'quiet_end')::time;
    if starts is null or ends is null then return candidate; end if;
    zone:=policy->>'timezone';
    local_now:=candidate at time zone zone;
    if starts<ends and local_now::time>=starts and local_now::time<ends then
        return (local_now::date+ends) at time zone zone;
    elsif starts>ends and local_now::time>=starts then
        return ((local_now::date+1)+ends) at time zone zone;
    elsif starts>ends and local_now::time<ends then
        return (local_now::date+ends) at time zone zone;
    end if;
    return candidate;
end $$;

create function refresh_jarvis_responsibility_quiet(p_owner text,p_now timestamptz)
returns void language plpgsql set search_path from current as $$
declare item record; prior record; reminder uuid; allowed timestamptz; base_time timestamptz; chosen timestamptz;
begin
    for item in select * from jarvis_attention_intents
        where owner_id=p_owner and state in ('pending','unknown','failed')
          and source_namespace in ('reminder_event','prospective_event')
        order by id for update loop
        reminder:=null;
        if item.source_namespace='reminder_event' then
            select reminder_id into reminder from jarvis_reminder_events
                where owner_id=p_owner and id=item.source_id;
        else
            select reminder_id into reminder from jarvis_prospective_delivery_events
                where owner_id=p_owner and id=item.source_id;
        end if;
        if reminder is null then continue; end if;
        allowed:=jarvis_responsibility_notice_time(p_owner,reminder,p_now);
        select * into prior from jarvis_responsibility_attention_delays
            where owner_id=p_owner and intent_id=item.id;
        if not found and allowed<=p_now then continue; end if;
        base_time:=case when prior.intent_id is not null and item.not_before=prior.applied_not_before
                        then prior.base_not_before else item.not_before end;
        chosen:=greatest(base_time,allowed);
        insert into jarvis_responsibility_attention_delays(owner_id,intent_id,base_not_before,applied_not_before)
            values(p_owner,item.id,base_time,chosen)
            on conflict(owner_id,intent_id) do update set
                base_not_before=excluded.base_not_before,applied_not_before=excluded.applied_not_before;
        if chosen is distinct from item.not_before then
            update jarvis_attention_intents set not_before=chosen where owner_id=p_owner and id=item.id;
        end if;
    end loop;
end $$;
