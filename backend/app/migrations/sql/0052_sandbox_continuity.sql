-- Opt-in reuse on the existing allocation ledger. No extra workspace/transcript store.
-- Native metadata stays immutable while a confirmed-paused allocation changes holder.
alter table jarvis_sandbox_allocations
    add column reusable boolean not null default false,
    add column native_scope jsonb;
update jarvis_sandbox_allocations set native_scope=jsonb_build_object(
    'owner_id',owner_id,'task_id',task_id,'task_revision',task_revision,
    'attempt_id',attempt_id,'allocation_id',id);
alter table jarvis_sandbox_allocations alter column native_scope set not null;
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_allocations_phase_check;
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_allocations_phase_check
    check(phase in ('reserved','creating','uncertain','preparing','ready','pausing','paused','resuming','cleanup_requested','cleaned'));
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_cleanup_claim_phase;
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_cleanup_claim_phase
    check(cleanup_token is null or phase in ('cleanup_requested','pausing'));
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_continuity_handle
    check(phase not in ('pausing','paused','resuming') or
        (reusable and sandbox_id is not null and submission_token is not null));

create function initialize_jarvis_sandbox_native_scope() returns trigger language plpgsql as $$
begin
    new.native_scope := jsonb_build_object('owner_id',new.owner_id,'task_id',new.task_id,
        'task_revision',new.task_revision,'attempt_id',new.attempt_id,'allocation_id',new.id);
    return new;
end; $$;
create trigger jarvis_sandbox_native_scope before insert on jarvis_sandbox_allocations
    for each row execute function initialize_jarvis_sandbox_native_scope();

create or replace function guard_jarvis_sandbox_ownership() returns trigger language plpgsql as $$
begin
    if row(new.id,new.owner_id,new.policy,new.policy_digest,new.created_at,new.native_scope,new.reusable)
       is distinct from row(old.id,old.owner_id,old.policy,old.policy_digest,old.created_at,old.native_scope,old.reusable)
       or new.deadline>old.deadline then
        raise exception 'Sandbox identity and policy cannot change or extend' using errcode='23514';
    end if;
    if row(new.task_id,new.task_revision,new.attempt_id,new.grant_id)
       is distinct from row(old.task_id,old.task_revision,old.attempt_id,old.grant_id)
       and not (old.reusable and old.phase='paused' and new.phase='resuming'
                and old.cleanup_token is null and old.deadline>clock_timestamp()) then
        raise exception 'Sandbox holder can move only from confirmed pause' using errcode='23514';
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
