-- Preserve the existing text-file contract without enlarging ordinary context payloads.
do $$
declare bound_name text;
begin
    select conname into strict bound_name from pg_constraint
        where conrelid='jarvis_context_payloads'::regclass and contype='c'
          and position('octet_length' in pg_get_constraintdef(oid)) > 0
          and position('262144' in pg_get_constraintdef(oid)) > 0;
    execute format('alter table jarvis_context_payloads drop constraint %I', bound_name);
end $$;

alter table jarvis_context_payloads add constraint jarvis_payload_bounded_original check (
    content is null or (jsonb_typeof(content)='object' and
        (octet_length(content::text)<=262144 or coalesce(
            octet_length(content::text)<=2097152
            and content->>'format'='jarvis.text_file_original.v1'
            and jsonb_typeof(content->'record')='object'
            and jsonb_typeof(content->'text')='string'
            and char_length(content->>'text') between 1 and 200000
            and (content->>'content_sha256') ~ '^[0-9a-f]{64}$'
            and content - array['format','record','text','content_sha256'] = '{}'::jsonb,
        false)))
);
