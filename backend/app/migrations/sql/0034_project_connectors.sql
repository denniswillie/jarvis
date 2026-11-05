-- Outbound read-only Mac connections. Originals remain canonical removable sources.
create table jarvis_project_connectors (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    label text not null check (length(label) between 1 and 100),
    token_digest text not null check (token_digest ~ '^[0-9a-f]{64}$'),
    expires_at timestamptz not null,
    status text not null default 'active' check (status in ('active','revoked')),
    last_seen_at timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    check(isfinite(expires_at) and expires_at>created_at)
);
create table jarvis_project_roots (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    connector_id uuid not null,
    local_root_id uuid not null,
    label text not null check (length(label) between 1 and 100),
    fingerprint text not null check(fingerprint ~ '^[0-9a-f]{64}$'),
    authorization_payload_id uuid not null,
    status text not null default 'active' check(status in ('active','revoked')),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    unique(connector_id,local_root_id),
    foreign key(owner_id,area_id,connector_id) references jarvis_project_connectors(owner_id,area_id,id),
    foreign key(owner_id,authorization_payload_id) references jarvis_context_payloads(owner_id,id)
);
create table jarvis_project_requests (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    root_id uuid not null,
    request_payload_id uuid not null,
    purpose text not null default 'inspection' check(purpose in ('inspection','refresh')),
    status text not null default 'pending' check(status in ('pending','leased','completed','expired','revoked')),
    lease_id uuid,
    lease_until timestamptz,
    response_digest text check(response_digest ~ '^[0-9a-f]{64}$'),
    source_id uuid,
    created_at timestamptz not null default clock_timestamp(),
    deadline timestamptz not null default clock_timestamp()+interval '90 seconds',
    unique(owner_id,id),
    foreign key(owner_id,area_id,root_id) references jarvis_project_roots(owner_id,area_id,id),
    foreign key(owner_id,request_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    check((status='completed')=(source_id is not null)),
    check((lease_id is null)=(lease_until is null))
);
create index jarvis_project_requests_poll_idx on jarvis_project_requests(owner_id,root_id,created_at)
    where status in ('pending','leased');

-- Only already-read sources are refreshed, not every file in an approved root.
-- Paths remain in removable originals rather than a second permanent text copy.
create table jarvis_project_file_watches (
    owner_id text not null,
    root_id uuid not null,
    source_id uuid not null,
    checked_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,root_id,source_id),
    foreign key(owner_id,root_id) references jarvis_project_roots(owner_id,id),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id)
);

-- Root identity cannot be silently rebound to a different host path or scope.
create function guard_jarvis_project_root() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.area_id<>old.area_id
        or new.connector_id<>old.connector_id or new.local_root_id<>old.local_root_id
        or new.fingerprint<>old.fingerprint or new.authorization_payload_id<>old.authorization_payload_id
        or (old.status='revoked' and new.status<>'revoked') then
        raise exception 'Project root enrollment is immutable except revocation' using errcode='23514';
    end if;
    if new.status='revoked' and old.status<>'revoked' then
        -- Existing copy/source triggers invalidate originals, task evidence and results.
        update jarvis_context_payloads set availability='deleted',content=null
            where owner_id=new.owner_id and id=new.authorization_payload_id and availability<>'deleted';
        update jarvis_project_requests set status='revoked'
            where owner_id=new.owner_id and root_id=new.id and status in ('pending','leased');
    end if;
    return new;
end; $$;
create trigger jarvis_project_roots_guard before update on jarvis_project_roots
for each row execute function guard_jarvis_project_root();

create function revoke_jarvis_project_connector() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.area_id<>old.area_id
        or new.token_digest<>old.token_digest or new.expires_at<>old.expires_at
        or (old.status='revoked' and new.status<>'revoked') then
        raise exception 'Connector identity and credential are immutable' using errcode='23514';
    end if;
    if new.status='revoked' then
        update jarvis_project_roots set status='revoked'
            where owner_id=new.owner_id and connector_id=new.id and status<>'revoked';
    end if;
    return new;
end; $$;
create trigger jarvis_project_connectors_guard before update on jarvis_project_connectors
for each row execute function revoke_jarvis_project_connector();
