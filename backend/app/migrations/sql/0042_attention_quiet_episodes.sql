-- Coalescing metadata on the canonical inbox, not a queue or preference authority.
-- No backfill or delivery is performed by installing this migration.
alter table jarvis_attention_intents
    add column quiet_episode_id uuid,
    add column quiet_summary_intent_id uuid,
    add constraint jarvis_attention_quiet_pair check (
        (quiet_episode_id is null) = (quiet_summary_intent_id is null)),
    add constraint jarvis_attention_quiet_summary_owner foreign key (owner_id,quiet_summary_intent_id)
        references jarvis_attention_intents(owner_id,id),
    add constraint jarvis_attention_quiet_not_self check (quiet_summary_intent_id is distinct from id);
create index jarvis_attention_quiet_members
    on jarvis_attention_intents(owner_id,quiet_summary_intent_id)
    where quiet_episode_id is not null;
-- Summary idempotency uses the existing unique
-- (owner_id,source_namespace,source_id,source_revision): source_id = episode UUID.
