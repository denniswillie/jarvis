-- Receipt fingerprints are action identity metadata, not retained receipt text.
-- Existing rows remain nullable: never invent a fingerprint for removed content.
alter table jarvis_task_actions add column receipt_digest text
    check(receipt_digest ~ '^[0-9a-f]{64}$');
