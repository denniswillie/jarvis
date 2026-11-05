-- Session context is attached only by an accepted, source-backed voice operation.
-- This is task authorization/lineage, not another conversation or memory store.
create table jarvis_task_session_context (
    owner_id text not null,
    task_id uuid not null,
    session_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision > 0),
    primary key(owner_id,task_id,session_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision)
        references jarvis_context_events(owner_id,id,revision)
);
