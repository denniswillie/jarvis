-- Explicit source expiry. No source is enrolled by age, inference or default.
create table jarvis_source_retention (
    owner_id text not null,
    area_id uuid not null,
    source_id uuid not null,
    policy_revision bigint not null check(policy_revision>0),
    source_revision bigint not null check(source_revision>0),
    source_generation bigint not null check(source_generation>0),
    expires_at timestamptz check(expires_at is null or isfinite(expires_at)),
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision>0),
    status text not null check(status in ('scheduled','disabled','expired','requires_review')),
    reason text check(reason is null or length(reason) between 1 and 100),
    next_check_at timestamptz not null default clock_timestamp(),
    checked_at timestamptz,
    updated_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,request_event_id) references jarvis_context_events(owner_id,id),
    check(status<>'scheduled' or expires_at is not null)
);
create index jarvis_source_retention_due_idx on jarvis_source_retention(owner_id,next_check_at,source_id)
where status='scheduled';

-- Metadata-only replay identity; provider call IDs cannot duplicate or renew expiry.
create table jarvis_source_retention_requests (
    owner_id text not null,
    source_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision>0),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_policy_revision bigint not null check(accepted_policy_revision>0),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,request_event_id,request_revision),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id),
    foreign key(owner_id,request_event_id) references jarvis_context_events(owner_id,id)
);
create trigger jarvis_source_retention_requests_immutable before update or delete
on jarvis_source_retention_requests for each row execute function guard_jarvis_knowledge_scope_identity();
