create table jarvis_background_tasks (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid,
    knowledge_area_id uuid,
    backend_kind text not null check(backend_kind in ('frontier','sandbox')),
    model text not null check(length(model) between 1 and 100),
    status text not null default 'queued' check(status in ('queued','running','waiting_for_input','paused','succeeded',
        'failed','cancel_requested','cancelled','outcome_unknown')),
    revision bigint not null default 1 check(revision>0),
    context_watermark bigint not null default 0 check(context_watermark>=0),
    description_payload_id uuid not null,
    request_event_id uuid not null,
    request_revision integer not null,
    current_attempt_id uuid,
    fence bigint not null default 0 check(fence>=0),
    attempt_count integer not null default 0 check(attempt_count>=0),
    active_seconds double precision not null default 0 check(active_seconds>=0 and active_seconds<'Infinity'::float8),
    budget jsonb not null check(jsonb_typeof(budget)='object' and octet_length(budget::text)<=8192),
    result_payload_id uuid,
    reason text check(length(reason)<=100),
    pending_status text check(pending_status in ('queued','cancelled','paused')),
    last_event_sequence bigint not null default 0 check(last_event_sequence>=0),
    available_at timestamptz not null default clock_timestamp(),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,knowledge_area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,description_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,result_payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_tasks_claim_idx on jarvis_background_tasks(status,available_at,created_at,id);
create index jarvis_tasks_owner_status_idx on jarvis_background_tasks(owner_id,status,created_at,id);
create index jarvis_tasks_area_idx on jarvis_background_tasks(owner_id,knowledge_area_id);

create table jarvis_task_attempts (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    attempt_number integer not null check(attempt_number>0),
    fence bigint not null check(fence>0),
    worker_id text not null check(length(worker_id) between 1 and 200),
    status text not null check(status in ('running','recovering','succeeded','failed','interrupted','cancelled','outcome_unknown')),
    provider_state text not null default 'not_started' check(provider_state in ('not_started','submitting','live','terminal')),
    provider_handle text check(length(provider_handle) between 1 and 512),
    checkpoint_payload_id uuid,
    lease_until timestamptz not null check(isfinite(lease_until)),
    charged_until timestamptz not null default clock_timestamp(),
    started_at timestamptz not null default clock_timestamp(),
    ended_at timestamptz,
    unique(owner_id,id),
    unique(owner_id,task_id,id),
    unique(owner_id,task_id,attempt_number),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,checkpoint_payload_id) references jarvis_context_payloads(owner_id,id),
    check(provider_state<>'live' or provider_handle is not null)
);
alter table jarvis_background_tasks add constraint jarvis_tasks_current_attempt_scope
    foreign key(owner_id,id,current_attempt_id) references jarvis_task_attempts(owner_id,task_id,id);
create index jarvis_task_attempts_recovery_idx on jarvis_task_attempts(status,lease_until);

create table jarvis_task_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    sequence bigint not null check(sequence>0),
    task_revision bigint not null check(task_revision>0),
    attempt_id uuid,
    event_type text not null check(length(event_type) between 1 and 80),
    payload_id uuid,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,sequence),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_task_events_append_only before update on jarvis_task_events
for each row execute function reject_jarvis_foundation_update();

create table jarvis_task_requests (
    owner_id text not null,
    request_id uuid not null,
    task_id uuid not null,
    operation text not null check(operation in ('create','update','cancel','resume')),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_revision bigint not null check(accepted_revision>0),
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id)
);
create trigger jarvis_task_requests_append_only before update on jarvis_task_requests
for each row execute function reject_jarvis_foundation_update();

create table jarvis_task_grants (
    owner_id text not null,
    task_id uuid not null,
    grant_id uuid not null,
    primary key(owner_id,task_id,grant_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,grant_id) references jarvis_operation_grants(owner_id,id)
);

create table jarvis_task_owner_scheduling (
    owner_id text primary key,
    last_claimed_at timestamptz not null default '-infinity'
);

-- Shared admission limits do not depend on which process wins a claim.
create table jarvis_task_scheduler_limits (
    singleton boolean primary key default true check(singleton),
    global_concurrency integer not null check(global_concurrency>0),
    owner_concurrency integer not null check(owner_concurrency>0 and owner_concurrency<=global_concurrency),
    updated_at timestamptz not null default clock_timestamp()
);

create table jarvis_task_reservations (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    attempt_id uuid not null,
    operation_key text not null check(length(operation_key) between 1 and 200),
    status text not null check(status in ('reserved','submitted','settled','released','unknown')),
    is_provider boolean not null default false,
    input_tokens bigint not null check(input_tokens>=0),
    output_tokens bigint not null check(output_tokens>=0),
    tool_steps integer not null check(tool_steps>=0),
    actual_input_tokens bigint check(actual_input_tokens>=0),
    actual_output_tokens bigint check(actual_output_tokens>=0),
    actual_tool_steps integer check(actual_tool_steps>=0),
    accounting_day date not null default (clock_timestamp() at time zone 'UTC')::date,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,operation_key),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    check((status='settled')=(actual_input_tokens is not null and actual_output_tokens is not null and actual_tool_steps is not null))
);
create index jarvis_task_reservations_daily_idx on jarvis_task_reservations(owner_id,accounting_day);

create table jarvis_task_actions (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    operation_key text not null check(length(operation_key) between 1 and 200),
    grant_id uuid not null,
    scope_revision integer not null check(scope_revision>0),
    parameters_payload_id uuid not null,
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    status text not null check(status in ('prepared','authorized','submitting','accepted','rejected','outcome_unknown','reconciled','revoked')),
    submission_token uuid,
    receipt_payload_id uuid,
    reconciled_outcome text check(reconciled_outcome in ('accepted','rejected','not_submitted')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,operation_key),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    unique(owner_id,grant_id),
    foreign key(owner_id,task_id,grant_id) references jarvis_task_grants(owner_id,task_id,grant_id),
    foreign key(owner_id,parameters_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,receipt_payload_id) references jarvis_context_payloads(owner_id,id),
    check((status='reconciled')=(reconciled_outcome is not null)),
    check(status not in ('submitting','accepted','rejected','outcome_unknown') or submission_token is not null)
);
create index jarvis_task_actions_unresolved_idx on jarvis_task_actions(owner_id,task_id,status);
