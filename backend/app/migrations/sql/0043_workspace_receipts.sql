-- C0 Workspace receipts. Original note content and versions remain in their
-- existing source stores; these rows do not create a second content authority.
create table jarvis_note_operations (
    owner_id text not null,
    operation_id uuid not null,
    session_id uuid not null,
    area_id uuid not null,
    operation_kind text not null check (operation_kind in ('create','patch','restore','delete')),
    parameters_digest text not null check (parameters_digest ~ '^[0-9a-f]{64}$'),
    artifact_id uuid not null,
    result_version_id text not null,
    outcome text not null check (outcome = 'completed'),
    committed_at timestamptz not null default clock_timestamp(),
    primary key (owner_id, operation_id)
);
create trigger jarvis_note_operations_immutable before update on jarvis_note_operations
for each row execute function reject_jarvis_foundation_update();

-- No FK to original notes/source versions: withdrawal must leave a receipt so
-- replay cannot silently recreate a deleted original.
create table jarvis_workspace_requests (
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id),
    request_id uuid not null,
    parameters_digest text not null check (parameters_digest ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null,
    primary key (owner_id, request_id),
    check (jsonb_typeof(payload) = 'object'),
    check (payload->>'request_id' = request_id::text),
    check (payload->>'state' in ('pending','claimed','displayed','saved_local','dismissed','failed','expired'))
);
create index jarvis_workspace_requests_session_idx
    on jarvis_workspace_requests(owner_id, session_id, created_at desc, request_id desc);
