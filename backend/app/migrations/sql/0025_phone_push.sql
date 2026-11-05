-- APNs is a channel of the existing inbox, never proof of hearing.
alter table jarvis_attention_attempts drop constraint jarvis_attention_attempts_phase_check;
alter table jarvis_attention_attempts add constraint jarvis_attention_attempts_phase_check
    check(phase in ('claimed','submitted','forwarding','awaiting_receipt','played',
                   'interrupted','failed','unknown','revoked','accepted'));
alter table jarvis_attention_attempts add constraint jarvis_attention_phone_phase_check
    check((phase<>'accepted' or channel='phone') and
          (channel<>'phone' or phase not in ('forwarding','awaiting_receipt','played','interrupted')));
alter table jarvis_attention_intents
    add column phone_notice_generation bigint check(phone_notice_generation>0),
    add column phone_notice_state text check(phone_notice_state in ('claimed','submitted','accepted','unknown')),
    add constraint jarvis_attention_phone_projection_check
        check((phone_notice_generation is null)=(phone_notice_state is null));

create table jarvis_push_devices (
    id uuid primary key,
    owner_id text not null,
    revision bigint not null check(revision>0),
    environment text not null check(environment in ('sandbox','production')),
    topic text not null check(length(topic) between 1 and 200),
    token text check(length(token) between 2 and 4096 and token ~ '^[0-9a-f]+$'),
    token_digest text check(token_digest ~ '^[0-9a-f]{64}$'),
    permission text not null check(permission in ('not_determined','denied','authorized','provisional','ephemeral')),
    enabled boolean not null,
    quiet_until timestamptz check(isfinite(quiet_until)),
    invalid_reason text,
    registered_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    check((token is null)=(token_digest is null)),
    check(not enabled or token is not null)
);
create unique index jarvis_push_active_address on jarvis_push_devices(environment,topic,token_digest)
    where enabled and token_digest is not null;
create index jarvis_push_owned_devices on jarvis_push_devices(owner_id,enabled);

create table jarvis_push_device_requests (
    owner_id text not null,
    request_id uuid not null,
    device_id uuid not null,
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_revision bigint not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,device_id) references jarvis_push_devices(owner_id,id)
);
create trigger jarvis_push_device_requests_append_only before update on jarvis_push_device_requests
    for each row execute function reject_jarvis_foundation_update();

create table jarvis_push_submissions (
    owner_id text not null,
    attempt_id uuid not null,
    device_id uuid not null,
    device_revision bigint not null,
    provider_id uuid,
    provider_status integer,
    provider_reason text,
    retry_at timestamptz check(isfinite(retry_at)),
    primary key(owner_id,attempt_id),
    foreign key(owner_id,attempt_id) references jarvis_attention_attempts(owner_id,id),
    foreign key(owner_id,device_id) references jarvis_push_devices(owner_id,id)
);
create index jarvis_push_submission_device on jarvis_push_submissions(owner_id,device_id,device_revision);

create index jarvis_attention_phone_history on jarvis_attention_attempts(owner_id,intent_id,generation,created_at)
    where channel='phone';
