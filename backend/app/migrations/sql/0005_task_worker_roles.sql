-- API admission metadata is separate from execution. No live feature is enabled.
create table jarvis_task_workers (
    worker_id text not null check (length(worker_id) between 1 and 100),
    owner_id text not null check (length(owner_id) between 1 and 200),
    host_id text not null check (length(host_id) between 1 and 200),
    backend_kind text not null check (backend_kind in ('frontier','sandbox')),
    model text not null check (length(model) between 1 and 100),
    draining boolean not null default false,
    updated_at timestamptz not null default clock_timestamp(),
    expires_at timestamptz not null,
    primary key(worker_id,backend_kind)
);
create index jarvis_task_workers_readiness on jarvis_task_workers(owner_id,backend_kind,model,expires_at)
    where not draining;

alter table jarvis_task_attempts add column runtime_host_id text
    check (runtime_host_id is null or length(runtime_host_id) between 1 and 200);

create function guard_jarvis_task_runtime_host() returns trigger language plpgsql as $$
begin
    if old.runtime_host_id is not null and new.runtime_host_id is distinct from old.runtime_host_id then
        raise exception 'Native runtime host cannot be reassigned' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_task_runtime_host_guard before update on jarvis_task_attempts
for each row execute function guard_jarvis_task_runtime_host();
