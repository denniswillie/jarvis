-- Five-minute capture bounds. Existing admissions and their smaller grants stay
-- immutable; document and generated-output limits are unchanged.
alter table jarvis_file_publications
    drop constraint jarvis_file_publications_transfer_policy_check,
    add constraint jarvis_file_publications_transfer_policy_check check (
        (transfer_policy = 'document_v1' and transfer_limit_bytes between 1 and 8388608)
        or (transfer_policy = 'mp4_capture_v1' and transfer_limit_bytes between 1 and 268435456)
        or (transfer_policy = 'mp4_output_v1' and transfer_limit_bytes between 1 and 33554432)
    );

alter table jarvis_recordings
    drop constraint jarvis_recordings_max_bytes_check,
    add constraint jarvis_recordings_max_bytes_check
        check (max_bytes between 1 and 268435456),
    drop constraint jarvis_recordings_max_duration_ms_check,
    add constraint jarvis_recordings_max_duration_ms_check
        check (max_duration_ms between 1 and 300000);
