-- Existing instruction text and timestamps remain unchanged. Revision zero is
-- reserved for a missing profile; all existing rows begin at revision one.
alter table jarvis_behavior_profiles
    add column revision bigint not null default 1
        constraint jarvis_behavior_profiles_positive_revision check (revision > 0);
