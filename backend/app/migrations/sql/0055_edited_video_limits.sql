-- Widen publication capacity for newly admitted edited MP4s only. Previously
-- saved transfer limits remain attached to their original publications.
alter table jarvis_file_publications
    drop constraint jarvis_file_publications_transfer_policy_check,
    add constraint jarvis_file_publications_transfer_policy_check check (
        (transfer_policy = 'document_v1' and transfer_limit_bytes between 1 and 8388608)
        or (transfer_policy = 'mp4_capture_v1' and transfer_limit_bytes between 1 and 1073741824)
        or (transfer_policy = 'mp4_output_v1' and transfer_limit_bytes between 1 and 500000000)
    );
