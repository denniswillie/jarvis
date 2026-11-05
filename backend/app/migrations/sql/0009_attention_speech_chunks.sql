-- Immutable network-submission identities and bounded playback evidence.
create table jarvis_attention_speech_chunks (
    owner_id text not null,
    attempt_id uuid not null,
    chunk_index integer not null check(chunk_index between 0 and 63),
    input_digest text not null check(input_digest ~ '^[0-9a-f]{64}$'),
    response_id text not null,
    item_id text not null,
    submitted_at timestamptz,
    generated_at timestamptz,
    forwarded_samples bigint not null default 0 check(forwarded_samples >= 0),
    played_samples bigint check(played_samples between 0 and forwarded_samples),
    playback_completed_at timestamptz,
    first_bytes_ms double precision check(first_bytes_ms >= 0),
    provider_request_id text,
    primary key(owner_id,attempt_id,chunk_index),
    unique(owner_id,item_id),
    foreign key(owner_id,attempt_id) references jarvis_attention_attempts(owner_id,id)
);

create function jarvis_attention_chunk_identity_immutable() returns trigger language plpgsql as $$
begin
    if (new.owner_id,new.attempt_id,new.chunk_index,new.input_digest,new.response_id,new.item_id)
        is distinct from (old.owner_id,old.attempt_id,old.chunk_index,old.input_digest,old.response_id,old.item_id)
        or (old.submitted_at is not null and new.submitted_at is distinct from old.submitted_at)
        or new.forwarded_samples < old.forwarded_samples
        or (old.played_samples is not null and (new.played_samples is null or new.played_samples < old.played_samples)) then
        raise exception 'attention chunk identity or receipt cannot be rewritten';
    end if;
    return new;
end $$;
create trigger jarvis_attention_chunk_identity_immutable before update on jarvis_attention_speech_chunks
    for each row execute function jarvis_attention_chunk_identity_immutable();
