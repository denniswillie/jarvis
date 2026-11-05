-- Recording admission permits bounded capture; it does not assert that capture
-- started, Jarvis was audible, or a file was saved. No runtime/media gate opens.
create table jarvis_recordings (
    owner_id text not null,
    recording_id uuid not null,
    session_id uuid not null,
    area_id uuid not null,
    include_jarvis_audio boolean not null,
    transfer_policy text not null check(transfer_policy='mp4_capture_v1'),
    max_bytes bigint not null check(max_bytes between 1 and 67108864),
    max_duration_ms integer not null check(max_duration_ms between 1 and 60000),
    command_digest text not null check(command_digest ~ '^[0-9a-f]{64}$'),
    admission_event_id uuid not null,
    admission_payload_id uuid not null,
    capture_closed boolean not null default false,
    revoked boolean not null default false,
    final_manifest_json text,
    final_manifest_digest text,
    publication_id uuid,
    file_id uuid,
    source_id uuid,
    version_id text,
    completion_event_id uuid,
    completion_payload_id uuid,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,recording_id),
    unique(owner_id,admission_event_id),
    unique(owner_id,admission_payload_id),
    unique(owner_id,publication_id),
    unique(owner_id,file_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,admission_event_id) references jarvis_context_events(owner_id,id),
    foreign key(owner_id,admission_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,publication_id) references jarvis_file_publications(owner_id,id),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id),
    foreign key(owner_id,version_id) references jarvis_knowledge_source_versions(owner_id,document_id),
    foreign key(owner_id,completion_event_id) references jarvis_context_events(owner_id,id),
    foreign key(owner_id,completion_payload_id) references jarvis_context_payloads(owner_id,id),
    check((final_manifest_json is null and final_manifest_digest is null) or
          (final_manifest_json is not null and final_manifest_digest is not null and capture_closed
           and octet_length(final_manifest_json) between 1 and 16384
           and jsonb_typeof(final_manifest_json::jsonb)='object'
           and final_manifest_digest ~ '^[0-9a-f]{64}$')),
    check((num_nonnulls(publication_id,file_id,source_id,version_id,completion_event_id,completion_payload_id)=0) or
          (num_nonnulls(publication_id,file_id,source_id,version_id,completion_event_id,completion_payload_id)=6
           and final_manifest_digest is not null and capture_closed))
);
alter table jarvis_recordings enable row level security;

create table jarvis_recording_requests (
    owner_id text not null,
    operation_id uuid not null,
    session_id uuid not null,
    recording_id uuid not null,
    capability text not null check(capability in ('recordings.start','recordings.stop')),
    command_digest text not null check(command_digest ~ '^[0-9a-f]{64}$'),
    command_event_id uuid not null,
    command_payload_id uuid not null,
    outcome text not null check(outcome in ('admitted','capture_closed')),
    receipt_event_id uuid not null,
    receipt_payload_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,operation_id),
    foreign key(owner_id,recording_id) references jarvis_recordings(owner_id,recording_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,command_event_id) references jarvis_context_events(owner_id,id),
    foreign key(owner_id,command_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,receipt_event_id) references jarvis_context_events(owner_id,id),
    foreign key(owner_id,receipt_payload_id) references jarvis_context_payloads(owner_id,id),
    check((capability='recordings.start' and outcome='admitted' and operation_id=recording_id)
          or (capability='recordings.stop' and outcome='capture_closed'))
);
alter table jarvis_recording_requests enable row level security;
create trigger jarvis_recording_requests_immutable before update on jarvis_recording_requests
    for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_recording() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['capture_closed','revoked','final_manifest_json','final_manifest_digest',
                           'publication_id','file_id','source_id','version_id','completion_event_id','completion_payload_id','updated_at'])
       is distinct from
       (to_jsonb(old)-array['capture_closed','revoked','final_manifest_json','final_manifest_digest',
                           'publication_id','file_id','source_id','version_id','completion_event_id','completion_payload_id','updated_at'])
       or (old.capture_closed and not new.capture_closed) or (old.revoked and not new.revoked) then
        raise check_violation using message='Recording admission identity and closed authority are immutable';
    end if;
    if old.final_manifest_json is not null and
       (new.final_manifest_json,new.final_manifest_digest) is distinct from (old.final_manifest_json,old.final_manifest_digest) then
        raise check_violation using message='Recording finalization cannot substitute bytes or metadata';
    end if;
    if old.completion_event_id is not null and
       (new.publication_id,new.file_id,new.source_id,new.version_id,new.completion_event_id,new.completion_payload_id)
       is distinct from
       (old.publication_id,old.file_id,old.source_id,old.version_id,old.completion_event_id,old.completion_payload_id) then
        raise check_violation using message='Recording completion cannot substitute an effect';
    end if;
    if old.revoked and ((old.final_manifest_json is null and new.final_manifest_json is not null)
                       or (old.completion_event_id is null and new.completion_event_id is not null)) then
        raise check_violation using message='Revoked recording admission cannot authorize a new finalization';
    end if;
    new.updated_at := clock_timestamp();
    return new;
end $$;
create trigger jarvis_recording_guard before update on jarvis_recordings
    for each row execute function guard_jarvis_recording();

create function revoke_jarvis_recording_admissions() returns trigger language plpgsql as $$
begin
    if new.availability<>'available' or new.content is distinct from old.content then
        -- Preserve committed identities/truth; no invented physical deletion.
        update jarvis_recordings set revoked=true,capture_closed=true
            where owner_id=new.owner_id and admission_payload_id=new.id and not revoked;
    end if;
    return new;
end $$;
create trigger jarvis_recording_admission_revocation after update of availability,content on jarvis_context_payloads
    for each row execute function revoke_jarvis_recording_admissions();
