-- Durable storage effects and adoption; no automatic source indexing or live cutover.
create table jarvis_file_publications (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    operation_key text not null check (length(operation_key) between 1 and 256),
    parameters_digest text not null check (parameters_digest ~ '^[0-9a-f]{64}$'),
    file_id uuid not null,
    metadata_payload_id uuid not null,
    content_sha256 text not null check (content_sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint not null check (size_bytes between 0 and 8388608),
    storage_bucket text,
    storage_path text,
    phase text not null default 'prepared' check
        (phase in ('prepared','uploading','uploaded','available','outcome_unknown','quarantined','delete_pending','deleted')),
    upload_token uuid,
    upload_settled boolean not null default false,
    delete_requested boolean not null default false,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,operation_key),
    unique(owner_id,file_id),
    unique(owner_id,metadata_payload_id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,metadata_payload_id) references jarvis_context_payloads(owner_id,id),
    check (phase <> 'available' or (upload_settled and not delete_requested)),
    check (phase <> 'deleted' or delete_requested),
    check ((phase='deleted' and storage_bucket is null and storage_path is null) or
           (phase<>'deleted' and storage_bucket is not null and storage_path is not null
            and length(storage_bucket) between 1 and 200 and length(storage_path) between 1 and 1200)),
    check (upload_token is not null or not upload_settled)
);
alter table jarvis_file_publications enable row level security;
create index jarvis_file_publications_pending on jarvis_file_publications(owner_id,updated_at,id)
    where phase not in ('available','deleted');

create function guard_jarvis_file_publication() returns trigger language plpgsql as $$
begin
    if (new.id,new.owner_id,new.area_id,new.operation_key,new.parameters_digest,new.file_id,
        new.metadata_payload_id,new.content_sha256,new.size_bytes,new.created_at)
       is distinct from
       (old.id,old.owner_id,old.area_id,old.operation_key,old.parameters_digest,old.file_id,
        old.metadata_payload_id,old.content_sha256,old.size_bytes,old.created_at) then
        raise check_violation using message='File publication identity is immutable';
    end if;
    if old.upload_token is not null and new.upload_token is distinct from old.upload_token then
        raise check_violation using message='File upload may cross the submission boundary only once';
    end if;
    if (old.upload_settled and not new.upload_settled) or (old.delete_requested and not new.delete_requested)
       or (old.phase='deleted' and new.phase<>'deleted') then
        raise check_violation using message='File publication cannot erase settled effects or deletion';
    end if;
    if (new.storage_bucket,new.storage_path) is distinct from (old.storage_bucket,old.storage_path)
       and not (new.phase='deleted' and new.storage_bucket is null and new.storage_path is null) then
        raise check_violation using message='File storage identity cannot be substituted';
    end if;
    new.updated_at := clock_timestamp();
    return new;
end $$;
create trigger jarvis_file_publication_guard before update on jarvis_file_publications
    for each row execute function guard_jarvis_file_publication();

create function revoke_jarvis_file_publications() returns trigger language plpgsql as $$
begin
    if new.availability<>'available' or new.content is distinct from old.content then
        -- Correction/forgetting invalidates ordinary access, not the historical file.
        -- The source-aware privacy adapter must explicitly request physical deletion.
        update jarvis_file_publications
            set phase=case when phase='deleted' then phase
                           when delete_requested then 'delete_pending' else 'quarantined' end
            where owner_id=new.owner_id and metadata_payload_id=new.id;
        delete from jarvis_files f using jarvis_file_publications p
            where p.owner_id=new.owner_id and p.metadata_payload_id=new.id and p.delete_requested
              and f.owner_id=p.owner_id and f.id=p.file_id;
    end if;
    return new;
end $$;
create trigger jarvis_file_publication_revocation after update of availability,content
    on jarvis_context_payloads for each row execute function revoke_jarvis_file_publications();
