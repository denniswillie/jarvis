-- Keep document limits unchanged; explicitly bind each media publication to its
-- admitted transfer policy and effective deployment limit. This enables no lane.
alter table jarvis_file_publications
    add column transfer_policy text not null default 'document_v1',
    add column transfer_limit_bytes bigint not null default 8388608;

alter table jarvis_file_publications
    drop constraint jarvis_file_publications_size_bytes_check,
    add constraint jarvis_file_publications_transfer_policy_check check (
        (transfer_policy = 'document_v1' and transfer_limit_bytes between 1 and 8388608)
        or (transfer_policy = 'mp4_capture_v1' and transfer_limit_bytes between 1 and 67108864)
        or (transfer_policy = 'mp4_output_v1' and transfer_limit_bytes between 1 and 33554432)
    ),
    add constraint jarvis_file_publications_size_bytes_check
        check (size_bytes between 0 and transfer_limit_bytes);

create or replace function guard_jarvis_file_publication() returns trigger language plpgsql as $$
begin
    if (new.id,new.owner_id,new.area_id,new.operation_key,new.parameters_digest,new.file_id,
        new.metadata_payload_id,new.content_sha256,new.size_bytes,new.created_at,
        new.transfer_policy,new.transfer_limit_bytes)
       is distinct from
       (old.id,old.owner_id,old.area_id,old.operation_key,old.parameters_digest,old.file_id,
        old.metadata_payload_id,old.content_sha256,old.size_bytes,old.created_at,
        old.transfer_policy,old.transfer_limit_bytes) then
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
