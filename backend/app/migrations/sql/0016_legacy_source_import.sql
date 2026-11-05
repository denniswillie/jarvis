-- Durable metadata for a maintenance-window import. No original bodies or credentials.
create table jarvis_legacy_import_runs (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    mapping_version integer not null check(mapping_version>0),
    input_digest text not null check(input_digest ~ '^[0-9a-f]{64}$'),
    population_counts jsonb not null check(jsonb_typeof(population_counts)='object'
        and octet_length(population_counts::text)<=32768),
    status text not null default 'pending' check(status in ('pending','complete','requires_attention')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id)
);
create function guard_jarvis_legacy_import_run() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['status','updated_at']) <> (to_jsonb(old)-array['status','updated_at']) then
        raise exception 'Import snapshot identity is immutable' using errcode='23514';
    end if;
    new.updated_at:=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_legacy_import_run_identity before update on jarvis_legacy_import_runs
for each row execute function guard_jarvis_legacy_import_run();

create table jarvis_legacy_import_records (
    owner_id text not null,
    run_id uuid not null,
    population text not null check(length(population) between 1 and 80),
    origin_key text not null check(length(origin_key) between 1 and 512),
    input_digest text not null check(input_digest ~ '^[0-9a-f]{64}$'),
    disposition text not null check(disposition in (
        'imported','already_present','attached','suppressed','quarantined',
        'external_only','retained_operational','retained_rebuildable','excluded_credentials'
    )),
    reason text,
    source_id uuid,
    source_revision bigint,
    source_generation bigint,
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,run_id,population,origin_key),
    foreign key(owner_id,run_id) references jarvis_legacy_import_runs(owner_id,id),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id),
    foreign key(owner_id,source_id,source_revision)
        references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    check((source_id is null and source_revision is null and source_generation is null)
        or (source_id is not null and source_revision>0 and source_generation>0)),
    check(reason is null or length(reason) between 1 and 200)
);
create trigger jarvis_legacy_import_record_immutable before update or delete on jarvis_legacy_import_records
for each row execute function reject_jarvis_foundation_update();
