-- Application scope identity only; no semantic memory or provider configuration.
create table jarvis_session_knowledge_selections (
    owner_id text not null,
    session_id uuid not null,
    revision bigint not null check (revision > 0),
    area_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check (request_revision > 0),
    arguments_digest text not null check (arguments_digest ~ '^[0-9a-f]{64}$'),
    receipt_event_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,session_id,revision),
    unique (owner_id,request_event_id,request_revision),
    foreign key (owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,request_event_id) references jarvis_context_events(owner_id,id),
    foreign key (owner_id,receipt_event_id) references jarvis_context_events(owner_id,id)
);

create table jarvis_knowledge_context_scopes (
    owner_id text not null,
    source_namespace text not null check (length(source_namespace) between 1 and 100),
    source_event_id text not null check (length(source_event_id) between 1 and 512),
    area_id uuid not null,
    bound_event_id uuid not null,
    origin_sequence bigint not null check (origin_sequence > 0),
    created_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,source_namespace,source_event_id),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,bound_event_id) references jarvis_context_events(owner_id,id)
);

create function guard_jarvis_knowledge_scope_identity() returns trigger language plpgsql as $$
begin
    raise exception 'Knowledge scope history is immutable' using errcode='23514';
end; $$;
create trigger jarvis_session_knowledge_selections_immutable before update or delete
on jarvis_session_knowledge_selections for each row execute function guard_jarvis_knowledge_scope_identity();
create trigger jarvis_knowledge_context_scopes_immutable before update or delete
on jarvis_knowledge_context_scopes for each row execute function guard_jarvis_knowledge_scope_identity();
