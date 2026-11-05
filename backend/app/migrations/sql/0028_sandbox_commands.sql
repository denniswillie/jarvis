-- Source-bound command intent/results. Native execution is never replayed from an unknown receipt.
create table jarvis_sandbox_commands (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    attempt_id uuid not null,
    allocation_id uuid not null,
    operation_digest text not null check(operation_digest ~ '^[0-9a-f]{64}$'),
    request_digest text not null check(request_digest ~ '^[0-9a-f]{64}$'),
    parameters_payload_id uuid not null,
    result_payload_id uuid,
    result_digest text check(result_digest ~ '^[0-9a-f]{64}$'),
    reservation_id uuid not null,
    phase text not null check(phase in ('prepared','submitting','running','unknown','completed')),
    submission_token uuid,
    pid integer check(pid>0),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,allocation_id,operation_digest),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    foreign key(owner_id,allocation_id) references jarvis_sandbox_allocations(owner_id,id),
    foreign key(owner_id,parameters_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,result_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,reservation_id) references jarvis_task_reservations(owner_id,id),
    check((phase='prepared')=(submission_token is null)),
    check((phase='completed')=(result_digest is not null)),
    check(result_payload_id is null or phase='completed')
);
create function guard_jarvis_sandbox_command() returns trigger
language plpgsql set search_path from current as $$
begin
    if (new.id,new.owner_id,new.task_id,new.task_revision,new.attempt_id,new.allocation_id,
        new.operation_digest,new.request_digest,new.parameters_payload_id,new.reservation_id)
        is distinct from
       (old.id,old.owner_id,old.task_id,old.task_revision,old.attempt_id,old.allocation_id,
        old.operation_digest,old.request_digest,old.parameters_payload_id,old.reservation_id)
       or (old.submission_token is not null and new.submission_token is distinct from old.submission_token)
       or (old.pid is not null and new.pid is distinct from old.pid)
       or (old.phase='completed' and (new.phase,new.result_digest,new.result_payload_id)
           is distinct from (old.phase,old.result_digest,old.result_payload_id)) then
        raise exception 'Sandbox command ownership or receipt is immutable' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_command_guard before update on jarvis_sandbox_commands
for each row execute function guard_jarvis_sandbox_command();
