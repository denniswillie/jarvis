-- Descriptive metadata only. This column grants no capability or source access.
alter table jarvis_knowledge_areas add column description text not null default ''
    check (length(description)<=4000);
