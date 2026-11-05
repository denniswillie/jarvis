-- Disposable previews live beside the immutable original manifest, never in
-- source evidence or model-visible metadata. Originals are not rewritten.
alter table jarvis_file_publications
    add column preview_jpeg bytea,
    add column preview_sha256 text,
    add column preview_width integer,
    add column preview_height integer,
    add constraint jarvis_file_preview_bound check (
        (preview_jpeg is null and preview_sha256 is null and preview_width is null and preview_height is null)
        or (preview_jpeg is not null and octet_length(preview_jpeg) between 1 and 65536
            and preview_sha256 is not null and preview_sha256 ~ '^[0-9a-f]{64}$'
            and preview_width is not null and preview_width between 1 and 480
            and preview_height is not null and preview_height between 1 and 480));

create function retire_jarvis_file_preview() returns trigger language plpgsql as $$
begin
    if new.delete_requested or new.phase in ('deleted','quarantined') then
        new.preview_jpeg := null;
        new.preview_sha256 := null;
        new.preview_width := null;
        new.preview_height := null;
    end if;
    return new;
end; $$;
create trigger retire_jarvis_file_preview before update on jarvis_file_publications
    for each row execute function retire_jarvis_file_preview();
