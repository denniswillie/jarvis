-- Recovery is resource ownership, not another model/task execution engine.
alter table jarvis_sandbox_allocations
    add column cleanup_token uuid,
    add column cleanup_worker text check(length(cleanup_worker) between 1 and 100),
    add column cleanup_generation bigint not null default 0 check(cleanup_generation>=0),
    add column cleanup_lease_until timestamptz check(isfinite(cleanup_lease_until)),
    add column next_cleanup_at timestamptz not null default clock_timestamp() check(isfinite(next_cleanup_at)),
    add column cleanup_error text check(length(cleanup_error) between 1 and 100),
    add column cleanup_observed_ids text[] not null default '{}' check(cardinality(cleanup_observed_ids)<=100),
    add constraint jarvis_sandbox_cleanup_claim_pair
        check((cleanup_token is null)=(cleanup_lease_until is null)),
    add constraint jarvis_sandbox_cleanup_claim_phase
        check(cleanup_token is null or phase='cleanup_requested');
create index jarvis_sandbox_cleanup_poll on jarvis_sandbox_allocations(next_cleanup_at,owner_id)
    where phase<>'cleaned';

create function guard_jarvis_sandbox_ownership() returns trigger language plpgsql as $$
begin
    if row(new.id,new.owner_id,new.task_id,new.task_revision,new.attempt_id,new.grant_id,
           new.policy,new.policy_digest,new.created_at)
       is distinct from row(old.id,old.owner_id,old.task_id,old.task_revision,old.attempt_id,old.grant_id,
           old.policy,old.policy_digest,old.created_at)
       or new.deadline>old.deadline then
        raise exception 'Sandbox ownership and authority cannot be reassigned or extended' using errcode='23514';
    end if;
    if old.submission_token is not null and
       row(new.submission_token,new.submission_fence) is distinct from row(old.submission_token,old.submission_fence) then
        raise exception 'Sandbox submission identity is immutable' using errcode='23514';
    end if;
    if old.sandbox_id is not null and row(new.sandbox_id,new.template_id) is distinct from row(old.sandbox_id,old.template_id) then
        raise exception 'Sandbox native identity is immutable' using errcode='23514';
    end if;
    if not old.cleanup_observed_ids <@ new.cleanup_observed_ids
       or (old.phase in ('cleanup_requested','cleaned') and new.phase not in ('cleanup_requested','cleaned')) then
        raise exception 'Sandbox cleanup evidence cannot be erased or execution restored' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_sandbox_ownership_guard before update on jarvis_sandbox_allocations
    for each row execute function guard_jarvis_sandbox_ownership();
