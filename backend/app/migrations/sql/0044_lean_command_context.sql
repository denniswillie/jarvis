-- 0001_second_brain_foundations.sql declares an inline CHECK on kind.
-- Its PostgreSQL-generated name is jarvis_context_events_kind_check.
-- Representation only: explicit-request/grant authority is unchanged.
alter table jarvis_context_events
    drop constraint jarvis_context_events_kind_check;

alter table jarvis_context_events
    add constraint jarvis_context_events_kind_check check(kind in (
        'user_request','user_transcript','assistant_output','played_output_boundary',
        'tool_outcome','observation','task_update','correction','context_gap',
        'command_request'
    ));
