-- Additive v1 native/chained evidence in the existing attention authority.
-- No old claims, plans, chunks, intent states or replay generations are changed.
alter table jarvis_attention_attempts
    add column delivery_transport text not null default 'legacy_pcm_tts',
    add column voice_route_id uuid,
    add column voice_route_generation bigint,
    add column voice_offer_id uuid,
    add column voice_route_protocol_version integer,
    add column voice_audio_session_id uuid;

alter table jarvis_attention_attempts add constraint jarvis_attention_route_binding check (
    (delivery_transport='legacy_pcm_tts' and voice_route_id is null and voice_route_generation is null
        and voice_offer_id is null and voice_route_protocol_version is null and voice_audio_session_id is null)
    or (delivery_transport in ('native_voice','chained_voice') and channel='voice'
        and voice_route_id is not null and voice_route_generation is not null and voice_route_generation>0
        and voice_offer_id is not null and voice_route_protocol_version is not null and voice_route_protocol_version=1
        and voice_audio_session_id is not null and plan_payload_id is not null
        and forwarded_samples=0 and played_samples is null
        and phase not in ('forwarding','awaiting_receipt','played'))
);
alter table jarvis_attention_attempts add constraint jarvis_attention_route_attempt_identity
    unique(owner_id,id,voice_route_id,voice_route_generation,voice_offer_id);
create unique index jarvis_attention_route_offer_identity on jarvis_attention_attempts
    (owner_id,voice_route_id,voice_route_generation,voice_offer_id) where voice_offer_id is not null;

create function jarvis_attention_route_identity_immutable() returns trigger
language plpgsql set search_path from current as $$
begin
    if old.delivery_transport<>'legacy_pcm_tts' and (
        (new.owner_id,new.id,new.intent_id,new.generation,new.session_id,new.worker_id,
         new.delivery_transport,new.voice_route_id,new.voice_route_generation,new.voice_offer_id,
         new.voice_route_protocol_version,new.voice_audio_session_id,new.plan_payload_id,new.plan_digest)
        is distinct from
        (old.owner_id,old.id,old.intent_id,old.generation,old.session_id,old.worker_id,
         old.delivery_transport,old.voice_route_id,old.voice_route_generation,old.voice_offer_id,
         old.voice_route_protocol_version,old.voice_audio_session_id,old.plan_payload_id,old.plan_digest)
        or (old.submitted_at is not null and new.submitted_at is distinct from old.submitted_at)) then
        raise exception 'Native delivery identity and submitted boundary are immutable' using errcode='23514';
    end if;
    if old.delivery_transport='legacy_pcm_tts' and new.delivery_transport<>'legacy_pcm_tts' and (
        old.phase<>'claimed' or old.submitted_at is not null or old.plan_payload_id is not null
        or old.forwarded_samples<>0 or old.played_samples is not null
        or exists(select 1 from jarvis_attention_speech_chunks c
                  where c.owner_id=old.owner_id and c.attempt_id=old.id)) then
        raise exception 'An existing legacy output cannot change transport' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_attention_route_identity_immutable before update on jarvis_attention_attempts
for each row execute function jarvis_attention_route_identity_immutable();

create table jarvis_attention_route_evidence (
    owner_id text not null,
    evidence_id uuid not null,
    attempt_id uuid not null,
    voice_route_id uuid not null,
    voice_route_generation bigint not null check(voice_route_generation>0),
    offer_id uuid not null,
    output_id uuid,
    evidence_state text not null check(evidence_state in ('offered','submitted','provider_received',
        'output_started','output_finished','client_playback_observed','interrupted','failed','unknown')),
    observed_at timestamptz not null check(isfinite(observed_at)),
    evidence_payload_id uuid not null,
    evidence_digest text not null check(evidence_digest ~ '^[0-9a-f]{64}$'),
    playback_receipt_id uuid,
    primary key(owner_id,evidence_id),
    foreign key(owner_id,attempt_id,voice_route_id,voice_route_generation,offer_id)
        references jarvis_attention_attempts(owner_id,id,voice_route_id,voice_route_generation,voice_offer_id),
    foreign key(owner_id,evidence_payload_id) references jarvis_context_payloads(owner_id,id),
    check(evidence_state not in ('output_started','output_finished','client_playback_observed') or output_id is not null),
    check((evidence_state='client_playback_observed')=(playback_receipt_id is not null))
);
create index jarvis_attention_route_evidence_attempt on jarvis_attention_route_evidence(owner_id,attempt_id,observed_at);
create index jarvis_attention_route_evidence_output on jarvis_attention_route_evidence
    (owner_id,voice_route_id,voice_route_generation,offer_id,output_id) where output_id is not null;
create unique index jarvis_attention_route_receipt_identity on jarvis_attention_route_evidence
    (owner_id,attempt_id,playback_receipt_id) where playback_receipt_id is not null;
create trigger jarvis_attention_route_evidence_immutable before update on jarvis_attention_route_evidence
for each row execute function reject_jarvis_foundation_update();

-- Guard against accidental legacy-PCM writer use on a native/chained claim.
create function guard_jarvis_attention_legacy_chunk() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(select 1 from jarvis_attention_attempts a
        where a.owner_id=new.owner_id and a.id=new.attempt_id and a.delivery_transport='legacy_pcm_tts') then
        raise exception 'PCM chunks require a legacy delivery attempt' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_attention_legacy_chunk_guard before insert or update on jarvis_attention_speech_chunks
for each row execute function guard_jarvis_attention_legacy_chunk();
