create extension if not exists pgcrypto;
create extension if not exists vector;

create table jarvis_sessions (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    title text not null default 'Jarvis session',
    status text not null check (status in ('active', 'closed')) default 'active',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    closed_at timestamptz,
    last_activity_at timestamptz not null default now(),
    behavior_instructions text not null default '',
    metadata jsonb not null default '{}'::jsonb
);

create index jarvis_sessions_owner_status_updated_idx
    on jarvis_sessions(owner_id, status, updated_at desc);

create table jarvis_behavior_profiles (
    owner_id text primary key,
    behavior_instructions text not null default '',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table jarvis_messages (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    role text not null check (role in ('user', 'assistant', 'tool', 'system')),
    modality text not null check (modality in ('text', 'audio', 'visual', 'tool')),
    content text not null default '',
    provider_item_id text,
    provider_response_id text,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index jarvis_messages_session_created_idx
    on jarvis_messages(session_id, created_at asc);

create index jarvis_messages_owner_created_idx
    on jarvis_messages(owner_id, created_at desc);

create table jarvis_user_preferences (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    key text not null,
    value text not null,
    source text not null default 'assistant',
    confidence double precision not null default 1.0
        check (confidence >= 0 and confidence <= 1),
    evidence_episode_ids uuid[] not null default '{}'::uuid[],
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (array_position(evidence_episode_ids, null) is null),
    unique(owner_id, key)
);

create table visual_frames (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    audio_session_id text,
    visual_session_id uuid not null,
    frame_sequence bigint not null,
    source text not null check (source in ('glasses', 'iphone_camera')),
    captured_at timestamptz not null,
    server_received_at timestamptz not null default now(),
    content_type text not null check (content_type = 'image/jpeg'),
    width integer not null check (width > 0),
    height integer not null check (height > 0),
    storage_bucket text not null,
    storage_path text not null,
    raw_image_status text not null check (
        raw_image_status in ('available', 'expiring', 'expired', 'deleted', 'failed')
    ) default 'available',
    raw_image_expires_at timestamptz not null,
    analyzer_status text not null check (analyzer_status in ('pending', 'skipped', 'completed', 'failed')) default 'skipped',
    frame_description text,
    memory_tags text[] not null default '{}'::text[],
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique(session_id, visual_session_id, frame_sequence)
);

create index visual_frames_session_captured_idx
    on visual_frames(session_id, captured_at desc);

create index visual_frames_raw_expiry_idx
    on visual_frames(raw_image_expires_at)
    where raw_image_status = 'available';

create index visual_frames_expiring_claim_idx
    on visual_frames(updated_at)
    where raw_image_status = 'expiring';

create index visual_frames_owner_captured_idx
    on visual_frames(owner_id, captured_at desc);

create index visual_frames_expired_cleanup_idx
    on visual_frames(owner_id, captured_at asc, id asc)
    where raw_image_status = 'expired';

create table visual_interrupts (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    visual_frame_id uuid references visual_frames(id) on delete set null,
    message text not null,
    reason text,
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    dedupe_bucket text not null,
    spoken_response_id text,
    created_at timestamptz not null default now()
);

create index visual_interrupts_session_created_idx
    on visual_interrupts(session_id, created_at desc);

create index visual_interrupts_dedupe_idx
    on visual_interrupts(session_id, dedupe_bucket, created_at desc);

create index jarvis_user_preferences_owner_updated_idx
    on jarvis_user_preferences(owner_id, updated_at desc);

create table jarvis_notes (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    title text not null,
    body text not null default '',
    source text not null default 'assistant',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index jarvis_notes_owner_updated_idx
    on jarvis_notes(owner_id, updated_at desc);

create index jarvis_notes_session_updated_idx
    on jarvis_notes(session_id, updated_at desc)
    where session_id is not null;

create table jarvis_reminders (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    title text not null,
    details text not null default '',
    remind_at timestamptz,
    condition text,
    status text not null check (status in ('active', 'completed', 'failed', 'deleted')) default 'active',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    completed_at timestamptz,
    triggered_at timestamptz,
    trigger_claimed_at timestamptz,
    trigger_claim_token uuid,
    trigger_attempt_count integer not null default 0 check (trigger_attempt_count >= 0),
    last_trigger_error text,
    metadata jsonb not null default '{}'::jsonb,
    check (remind_at is not null or nullif(btrim(condition), '') is not null)
);

create index jarvis_reminders_owner_status_updated_idx
    on jarvis_reminders(owner_id, status, updated_at desc);

create index jarvis_reminders_active_time_idx
    on jarvis_reminders(owner_id, remind_at asc)
    where status = 'active' and remind_at is not null and triggered_at is null;

create table jarvis_reminder_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    reminder_id uuid not null references jarvis_reminders(id) on delete cascade,
    session_id uuid references jarvis_sessions(id) on delete set null,
    event_type text not null check (
        event_type in (
            'generated',
            'dispatched_to_session',
            'delivered_to_session',
            'missed',
            'failed',
            'interrupted'
        )
    ),
    message text not null default '',
    provider_response_id text,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index jarvis_reminder_events_owner_created_idx
    on jarvis_reminder_events(owner_id, created_at desc);

create index jarvis_reminder_events_reminder_created_idx
    on jarvis_reminder_events(reminder_id, created_at desc);

create table jarvis_visual_focuses (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    target text not null check (nullif(btrim(target), '') is not null),
    description text not null check (nullif(btrim(description), '') is not null),
    purpose text not null check (nullif(btrim(purpose), '') is not null),
    mode text not null check (mode in ('find_once', 'monitor')),
    created_from_message_id uuid references jarvis_messages(id) on delete set null,
    created_at timestamptz not null default now(),
    expires_at timestamptz not null,
    completed_at timestamptz,
    cleared_at timestamptz,
    completion_evidence jsonb check (
        completion_evidence is null or jsonb_typeof(completion_evidence) = 'object'
    ),
    trigger_claim_token uuid,
    trigger_claimed_at timestamptz,
    check (expires_at > created_at),
    check (completed_at is null or cleared_at is null)
);

create unique index jarvis_visual_focuses_owner_open_unique_idx
    on jarvis_visual_focuses(owner_id)
    where completed_at is null and cleared_at is null;

create index jarvis_visual_focuses_owner_expiry_idx
    on jarvis_visual_focuses(owner_id, expires_at desc);

create index jarvis_visual_focuses_stale_claim_idx
    on jarvis_visual_focuses(trigger_claimed_at)
    where trigger_claimed_at is not null;

create table jarvis_prospective_delivery_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    visual_focus_id uuid references jarvis_visual_focuses(id) on delete cascade,
    reminder_id uuid references jarvis_reminders(id) on delete cascade,
    claim_token uuid not null,
    occurrence_key text not null check (nullif(btrim(occurrence_key), '') is not null),
    message text not null check (nullif(btrim(message), '') is not null),
    event_type text not null default 'generated' check (
        event_type in (
            'generated',
            'dispatched_to_session',
            'delivered_to_session',
            'missed',
            'failed',
            'interrupted'
        )
    ),
    evidence jsonb not null default '{}'::jsonb check (jsonb_typeof(evidence) = 'object'),
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    provider_response_id text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check ((visual_focus_id is null) <> (reminder_id is null))
);

create unique index jarvis_prospective_focus_occurrence_unique_idx
    on jarvis_prospective_delivery_events(owner_id, visual_focus_id, occurrence_key)
    where visual_focus_id is not null;

create unique index jarvis_prospective_reminder_occurrence_unique_idx
    on jarvis_prospective_delivery_events(owner_id, reminder_id, occurrence_key)
    where reminder_id is not null;

create index jarvis_prospective_events_owner_created_idx
    on jarvis_prospective_delivery_events(owner_id, created_at desc);

create table google_oauth_tokens (
    owner_id text primary key,
    access_token text not null,
    refresh_token text,
    token_type text,
    scope text,
    expires_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table web_fetch_cache (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    url text not null,
    title text,
    text_content text not null,
    fetched_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb,
    unique(owner_id, url)
);

create index web_fetch_cache_owner_fetched_idx
    on web_fetch_cache(owner_id, fetched_at desc);

create table jarvis_files (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    filename text not null,
    content_type text not null,
    storage_bucket text not null,
    storage_path text not null,
    size_bytes integer check (size_bytes >= 0),
    description text not null default '',
    source text not null check (source in ('assistant', 'visual_frame', 'upload', 'web_fetch')) default 'assistant',
    source_visual_frame_id uuid references visual_frames(id) on delete set null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb,
    unique(storage_bucket, storage_path)
);

create index jarvis_files_owner_created_idx
    on jarvis_files(owner_id, created_at desc);

create index jarvis_files_session_created_idx
    on jarvis_files(session_id, created_at desc)
    where session_id is not null;

create table perception_observations (
    id uuid primary key,
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    source text not null check (source in ('glasses', 'iphone_camera')),
    started_at timestamptz not null,
    ended_at timestamptz not null,
    primary_visual_frame_id uuid references visual_frames(id) on delete set null,
    summary text not null check (nullif(btrim(summary), '') is not null),
    scene text,
    activity text,
    scene_confidence double precision check (
        scene_confidence is null or (scene_confidence >= 0 and scene_confidence <= 1)
    ),
    activity_confidence double precision check (
        activity_confidence is null or (activity_confidence >= 0 and activity_confidence <= 1)
    ),
    entities jsonb not null default '[]'::jsonb check (jsonb_typeof(entities) = 'array'),
    visible_text jsonb not null default '[]'::jsonb check (jsonb_typeof(visible_text) = 'array'),
    state_changes jsonb not null default '[]'::jsonb check (jsonb_typeof(state_changes) = 'array'),
    memory_tags text[] not null default '{}'::text[],
    keyframe_candidates jsonb not null default '[]'::jsonb check (jsonb_typeof(keyframe_candidates) = 'array'),
    interrupt_candidate jsonb check (
        interrupt_candidate is null or jsonb_typeof(interrupt_candidate) = 'object'
    ),
    starts_new_event boolean not null default false,
    boundary_reason text,
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    analyzer_provider text not null check (nullif(btrim(analyzer_provider), '') is not null),
    analyzer_model text not null check (nullif(btrim(analyzer_model), '') is not null),
    analyzer_contract_version integer not null check (analyzer_contract_version > 0),
    compacted_at timestamptz,
    created_at timestamptz not null default now(),
    check (ended_at >= started_at),
    check (
        compacted_at is null
        or (
            entities = '[]'::jsonb
            and visible_text = '[]'::jsonb
            and state_changes = '[]'::jsonb
            and keyframe_candidates = '[]'::jsonb
            and interrupt_candidate is null
        )
    )
);

create index perception_observations_owner_session_ended_idx
    on perception_observations(owner_id, session_id, ended_at desc);

create index perception_observations_session_started_idx
    on perception_observations(session_id, started_at asc, id asc);

create table perception_observation_frames (
    observation_id uuid not null references perception_observations(id) on delete cascade,
    visual_frame_id uuid not null references visual_frames(id) on delete cascade,
    ordinal integer not null check (ordinal >= 0),
    role text not null check (role in ('context', 'primary', 'key_candidate')),
    created_at timestamptz not null default now(),
    primary key (observation_id, visual_frame_id),
    unique (observation_id, ordinal)
);

create unique index perception_observation_frames_primary_idx
    on perception_observation_frames(observation_id)
    where role = 'primary';

create or replace function jarvis_validate_perception_observation_frame()
returns trigger
language plpgsql
as $$
declare
    observation_owner_id text;
    observation_session_id uuid;
    observation_primary_frame_id uuid;
    frame_owner_id text;
    frame_session_id uuid;
begin
    select owner_id, session_id, primary_visual_frame_id
    into observation_owner_id, observation_session_id, observation_primary_frame_id
    from perception_observations
    where id = new.observation_id;

    select owner_id, session_id
    into frame_owner_id, frame_session_id
    from visual_frames
    where id = new.visual_frame_id;

    if observation_owner_id is distinct from frame_owner_id
       or observation_session_id is distinct from frame_session_id then
        raise exception 'perception observation frame must share owner and session';
    end if;

    if new.role = 'primary'
       and observation_primary_frame_id is distinct from new.visual_frame_id then
        raise exception 'primary observation frame must match primary_visual_frame_id';
    end if;

    if new.role <> 'primary'
       and observation_primary_frame_id = new.visual_frame_id then
        raise exception 'primary_visual_frame_id must use the primary frame role';
    end if;

    return new;
end;
$$;

create trigger perception_observation_frames_scope_trigger
before insert or update on perception_observation_frames
for each row execute function jarvis_validate_perception_observation_frame();

create or replace function jarvis_enforce_perception_observation_lifecycle()
returns trigger
language plpgsql
as $$
begin
    if old.id is distinct from new.id
       or old.owner_id is distinct from new.owner_id
       or old.session_id is distinct from new.session_id
       or old.source is distinct from new.source
       or old.started_at is distinct from new.started_at
       or old.ended_at is distinct from new.ended_at
       or old.summary is distinct from new.summary
       or old.scene is distinct from new.scene
       or old.activity is distinct from new.activity
       or old.memory_tags is distinct from new.memory_tags
       or old.starts_new_event is distinct from new.starts_new_event
       or old.boundary_reason is distinct from new.boundary_reason
       or old.confidence is distinct from new.confidence
       or old.analyzer_provider is distinct from new.analyzer_provider
       or old.analyzer_model is distinct from new.analyzer_model
       or old.analyzer_contract_version is distinct from new.analyzer_contract_version
       or old.created_at is distinct from new.created_at then
        raise exception 'perception observation identity and provenance are immutable';
    end if;

    if old.primary_visual_frame_id is distinct from new.primary_visual_frame_id
       and not (old.primary_visual_frame_id is not null and new.primary_visual_frame_id is null) then
        raise exception 'primary_visual_frame_id may only be cleared by evidence deletion';
    end if;

    if old.compacted_at is not distinct from new.compacted_at then
        if old.entities is distinct from new.entities
           or old.visible_text is distinct from new.visible_text
           or old.state_changes is distinct from new.state_changes
           or old.keyframe_candidates is distinct from new.keyframe_candidates
           or old.interrupt_candidate is distinct from new.interrupt_candidate then
            raise exception 'perception observation details are immutable before compaction';
        end if;
        return new;
    end if;

    if old.compacted_at is not null or new.compacted_at is null then
        raise exception 'perception observation compaction is a one-way transition';
    end if;

    if new.entities <> '[]'::jsonb
       or new.visible_text <> '[]'::jsonb
       or new.state_changes <> '[]'::jsonb
       or new.keyframe_candidates <> '[]'::jsonb
       or new.interrupt_candidate is not null then
        raise exception 'compaction must clear bulky perception details';
    end if;

    return new;
end;
$$;

create trigger perception_observations_lifecycle_trigger
before update on perception_observations
for each row execute function jarvis_enforce_perception_observation_lifecycle();

create table jarvis_session_working_memory (
    session_id uuid primary key references jarvis_sessions(id) on delete cascade,
    owner_id text not null,
    revision bigint not null default 0 check (revision >= 0),
    last_observation_id uuid references perception_observations(id),
    last_observation_ended_at timestamptz,
    state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
    expires_at timestamptz not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (
        (last_observation_id is null and last_observation_ended_at is null)
        or (last_observation_id is not null and last_observation_ended_at is not null)
    )
);

create index jarvis_session_working_memory_owner_expiry_idx
    on jarvis_session_working_memory(owner_id, expires_at);

create table jarvis_owner_working_memory (
    owner_id text primary key,
    revision bigint not null default 0 check (revision >= 0),
    state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
    expires_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index jarvis_owner_working_memory_expiry_idx
    on jarvis_owner_working_memory(expires_at)
    where expires_at is not null;

create table jarvis_tool_executions (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    provider_call_id text not null check (nullif(btrim(provider_call_id), '') is not null),
    tool_name text not null check (nullif(btrim(tool_name), '') is not null),
    mutates_state boolean not null default false,
    arguments jsonb not null check (jsonb_typeof(arguments) = 'object'),
    result jsonb check (result is null or jsonb_typeof(result) = 'object'),
    status text not null check (status in ('started', 'completed', 'failed')) default 'started',
    error jsonb check (error is null or jsonb_typeof(error) = 'object'),
    started_at timestamptz not null,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    unique (session_id, provider_call_id),
    check (octet_length(arguments::text) <= 65536),
    check (result is null or octet_length(result::text) <= 65536),
    check (error is null or octet_length(error::text) <= 16384),
    check (
        (status = 'started' and result is null and error is null and completed_at is null)
        or (status = 'completed' and result is not null and error is null and completed_at is not null)
        or (status = 'failed' and error is not null and completed_at is not null)
    ),
    check (completed_at is null or completed_at >= started_at)
);

create index jarvis_tool_executions_owner_session_started_idx
    on jarvis_tool_executions(owner_id, session_id, started_at asc, id asc);

create or replace function jarvis_enforce_tool_execution_lifecycle()
returns trigger
language plpgsql
as $$
begin
    if old.id is distinct from new.id
       or old.owner_id is distinct from new.owner_id
       or old.session_id is distinct from new.session_id
       or old.provider_call_id is distinct from new.provider_call_id
       or old.tool_name is distinct from new.tool_name
       or old.mutates_state is distinct from new.mutates_state
       or old.arguments is distinct from new.arguments
       or old.started_at is distinct from new.started_at
       or old.created_at is distinct from new.created_at then
        raise exception 'tool execution identity and arguments are immutable';
    end if;

    if old.status <> 'started' then
        raise exception 'terminal tool execution cannot be modified';
    end if;

    if new.status not in ('completed', 'failed') then
        raise exception 'tool execution may only transition to a terminal status';
    end if;

    return new;
end;
$$;

create trigger jarvis_tool_executions_lifecycle_trigger
before update on jarvis_tool_executions
for each row execute function jarvis_enforce_tool_execution_lifecycle();

create table experience_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id) on delete cascade,
    status text not null check (
        status in ('open', 'closed', 'consolidating', 'consolidated', 'failed')
    ) default 'open',
    event_kind text not null check (
        event_kind in ('mixed', 'visual', 'interaction', 'tool_activity')
    ),
    modalities text[] not null default '{}'::text[] check (
        modalities <@ array['audio', 'visual', 'text', 'tool']::text[]
    ),
    started_at timestamptz not null,
    ended_at timestamptz,
    title text not null default '',
    summary text not null default '',
    scene text,
    activity text,
    state jsonb not null default '{}'::jsonb check (jsonb_typeof(state) = 'object'),
    boundary_reason text,
    importance double precision not null default 0 check (importance >= 0 and importance <= 1),
    confidence double precision not null default 0 check (confidence >= 0 and confidence <= 1),
    keyframes_finalized_at timestamptz,
    consolidation_ready_at timestamptz,
    evidence_reduced_at timestamptz,
    retention_expires_at timestamptz,
    consolidation_result jsonb check (
        consolidation_result is null or jsonb_typeof(consolidation_result) = 'object'
    ),
    consolidation_provider text,
    consolidation_model text,
    consolidated_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (ended_at is null or ended_at >= started_at),
    check (
        (status = 'open'
            and ended_at is null
            and keyframes_finalized_at is null
            and consolidation_ready_at is null)
        or (status <> 'open'
            and ended_at is not null
            and keyframes_finalized_at is not null
            and consolidation_ready_at is not null)
    ),
    check (
        status <> 'consolidated'
        or (
            consolidation_result is not null
            and nullif(btrim(consolidation_provider), '') is not null
            and nullif(btrim(consolidation_model), '') is not null
            and consolidated_at is not null
        )
    )
);

create unique index experience_events_one_open_per_session_idx
    on experience_events(session_id)
    where status = 'open';

create index experience_events_owner_session_started_idx
    on experience_events(owner_id, session_id, started_at desc);

create index experience_events_consolidation_ready_idx
    on experience_events(consolidation_ready_at asc)
    where status = 'closed';

create table experience_event_evidence (
    id uuid primary key default gen_random_uuid(),
    event_id uuid not null references experience_events(id) on delete cascade,
    evidence_type text not null check (
        evidence_type in ('observation', 'message', 'tool_execution')
    ),
    evidence_ref_id uuid not null,
    occurred_at timestamptz not null,
    source_sequence bigint,
    ingested_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    unique (event_id, evidence_type, evidence_ref_id),
    check (source_sequence is null or source_sequence >= 0),
    check (octet_length(metadata::text) <= 8192)
);

create index experience_event_evidence_order_idx
    on experience_event_evidence(
        event_id,
        occurred_at asc,
        source_sequence asc,
        evidence_type asc,
        evidence_ref_id asc
    );

create or replace function jarvis_reject_experience_evidence_update()
returns trigger
language plpgsql
as $$
begin
    raise exception 'experience event evidence is append-only';
end;
$$;

create trigger experience_event_evidence_append_only_trigger
before update on experience_event_evidence
for each row execute function jarvis_reject_experience_evidence_update();

create table experience_event_keyframes (
    event_id uuid not null references experience_events(id) on delete cascade,
    visual_frame_id uuid not null references visual_frames(id) on delete cascade,
    rank integer not null check (rank >= 1),
    reason text not null check (nullif(btrim(reason), '') is not null),
    quality_score double precision not null check (quality_score >= 0 and quality_score <= 1),
    selected_by text not null check (selected_by in ('deterministic', 'gpt', 'user')),
    created_at timestamptz not null default now(),
    primary key (event_id, visual_frame_id),
    unique (event_id, rank)
);

create table experience_event_retention_reasons (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    experience_event_id uuid not null references experience_events(id) on delete cascade,
    reason_type text not null check (nullif(btrim(reason_type), '') is not null),
    reason_ref_id text not null check (nullif(btrim(reason_ref_id), '') is not null),
    expires_at timestamptz,
    created_at timestamptz not null default now(),
    unique (experience_event_id, reason_type, reason_ref_id)
);

create index experience_event_retention_reasons_owner_expiry_idx
    on experience_event_retention_reasons(owner_id, expires_at);

create index experience_event_retention_reasons_event_idx
    on experience_event_retention_reasons(experience_event_id);

create table visual_frame_retention_reasons (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    visual_frame_id uuid not null references visual_frames(id) on delete cascade,
    reason_type text not null check (
        reason_type in (
            'event_candidate',
            'ordinary_event_keyframe',
            'important_event_keyframe',
            'user_capture',
            'explicit_memory',
            'visual_interrupt'
        )
    ),
    reason_ref_id uuid not null,
    expires_at timestamptz,
    created_at timestamptz not null default now(),
    unique (visual_frame_id, reason_type, reason_ref_id),
    check (
        expires_at is not null
        or reason_type in ('important_event_keyframe', 'user_capture', 'explicit_memory')
    )
);

create index visual_frame_retention_reasons_owner_expiry_idx
    on visual_frame_retention_reasons(owner_id, expires_at);

create index visual_frame_retention_reasons_frame_idx
    on visual_frame_retention_reasons(visual_frame_id);

create table memory_jobs (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    job_type text not null check (nullif(btrim(job_type), '') is not null),
    dedupe_key text not null check (nullif(btrim(dedupe_key), '') is not null),
    payload jsonb not null default '{}'::jsonb check (jsonb_typeof(payload) = 'object'),
    status text not null default 'queued'
        check (status in ('queued', 'claimed', 'completed', 'failed', 'dead')),
    priority integer not null default 0,
    attempt_count integer not null default 0 check (attempt_count >= 0),
    max_attempts integer not null default 3 check (max_attempts >= 1),
    available_at timestamptz not null default now(),
    claimed_at timestamptz,
    claimed_by text,
    claim_token uuid,
    result jsonb check (result is null or jsonb_typeof(result) = 'object'),
    last_error text,
    completed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (attempt_count <= max_attempts),
    check (
        (status = 'queued'
            and attempt_count < max_attempts
            and claimed_at is null
            and claimed_by is null
            and claim_token is null
            and result is null
            and last_error is null
            and completed_at is null)
        or
        (status = 'claimed'
            and attempt_count >= 1
            and claimed_at is not null
            and nullif(btrim(claimed_by), '') is not null
            and claim_token is not null
            and result is null
            and last_error is null
            and completed_at is null)
        or
        (status = 'completed'
            and attempt_count >= 1
            and claimed_at is null
            and claimed_by is null
            and claim_token is null
            and last_error is null
            and completed_at is not null)
        or
        (status = 'failed'
            and attempt_count >= 1
            and attempt_count < max_attempts
            and claimed_at is null
            and claimed_by is null
            and claim_token is null
            and result is null
            and nullif(btrim(last_error), '') is not null
            and completed_at is null)
        or
        (status = 'dead'
            and attempt_count = max_attempts
            and claimed_at is null
            and claimed_by is null
            and claim_token is null
            and result is null
            and nullif(btrim(last_error), '') is not null
            and completed_at is null)
    )
);

create unique index memory_jobs_active_dedupe_idx
    on memory_jobs(owner_id, job_type, dedupe_key)
    where status in ('queued', 'claimed', 'failed');

create index memory_jobs_claim_idx
    on memory_jobs(status, available_at, priority desc, created_at, id);

create index memory_jobs_owner_created_idx
    on memory_jobs(owner_id, created_at desc);

create index memory_jobs_session_created_idx
    on memory_jobs(session_id, created_at desc)
    where session_id is not null;

create table memory_items (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid references jarvis_sessions(id) on delete set null,
    memory_type text not null
        check (memory_type in ('episode', 'semantic_fact', 'reflection', 'procedure', 'goal')),
    status text not null default 'active'
        check (status in ('active', 'superseded', 'retracted', 'forgotten')),
    title text not null check (nullif(btrim(title), '') is not null),
    content text not null check (nullif(btrim(content), '') is not null),
    summary text not null default '',
    source_experience_event_id uuid references experience_events(id),
    occurred_at timestamptz,
    valid_from timestamptz,
    valid_to timestamptz,
    importance double precision not null check (importance >= 0 and importance <= 1),
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    source text not null check (nullif(btrim(source), '') is not null),
    is_explicit boolean not null default false,
    superseded_by_memory_id uuid references memory_items(id),
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (valid_to is null or valid_from is null or valid_to >= valid_from),
    check (superseded_by_memory_id is null or superseded_by_memory_id <> id),
    check (memory_type <> 'episode' or source_experience_event_id is not null),
    check (
        (
            status = 'superseded'
            and valid_to is not null
            and superseded_by_memory_id is not null
        )
        or (
            status = 'retracted'
            and valid_to is not null
            and superseded_by_memory_id is null
        )
        or (status in ('active', 'forgotten') and superseded_by_memory_id is null)
    ),
    unique (owner_id, id)
);

create unique index memory_items_episode_source_unique_idx
    on memory_items(source_experience_event_id)
    where memory_type = 'episode';

create index memory_items_owner_type_status_created_idx
    on memory_items(owner_id, memory_type, status, created_at desc);

create index memory_items_session_created_idx
    on memory_items(session_id, created_at desc)
    where session_id is not null;

create index memory_items_validity_idx
    on memory_items(owner_id, valid_from, valid_to);

create table memory_forgetting_tombstones (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (nullif(btrim(owner_id), '') is not null),
    memory_type text not null check (
        memory_type in ('episode', 'semantic_fact', 'reflection', 'procedure', 'goal')
    ),
    identity_fingerprint text not null check (
        identity_fingerprint ~ '^[0-9a-f]{64}$'
    ),
    source_memory_id uuid not null,
    reason text not null check (nullif(btrim(reason), '') is not null),
    forgotten_at timestamptz not null default now(),
    overridden_at timestamptz,
    override_memory_id uuid references memory_items(id) on delete set null,
    override_metadata jsonb not null default '{}'::jsonb check (
        jsonb_typeof(override_metadata) = 'object'
    ),
    suppressed_recreation_count integer not null default 0 check (
        suppressed_recreation_count >= 0
    ),
    last_suppressed_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    foreign key (owner_id, source_memory_id)
        references memory_items(owner_id, id) on delete restrict,
    check (
        (overridden_at is null and override_memory_id is null)
        or overridden_at is not null
    ),
    unique (owner_id, source_memory_id)
);

create unique index memory_forgetting_tombstones_active_identity_idx
    on memory_forgetting_tombstones(owner_id, memory_type, identity_fingerprint)
    where overridden_at is null;

create index memory_forgetting_tombstones_owner_forgotten_idx
    on memory_forgetting_tombstones(owner_id, forgotten_at desc, id desc);

alter table experience_events
    add column promotion_decision text not null default 'pending'
        check (promotion_decision in ('pending', 'promoted', 'skipped')),
    add column promotion_reason text,
    add column promoted_episode_memory_id uuid references memory_items(id),
    add column promotion_decided_at timestamptz,
    add column semantic_reconciled_at timestamptz,
    add check (
        (promotion_decision = 'pending'
            and promotion_reason is null
            and promoted_episode_memory_id is null
            and promotion_decided_at is null)
        or
        (promotion_decision = 'promoted'
            and nullif(btrim(promotion_reason), '') is not null
            and promoted_episode_memory_id is not null
            and promotion_decided_at is not null)
        or
        (promotion_decision = 'skipped'
            and nullif(btrim(promotion_reason), '') is not null
            and promoted_episode_memory_id is null
            and promotion_decided_at is not null)
    );

create table calendar_event_snapshots (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    provider text not null,
    calendar_id text not null,
    provider_event_id text not null,
    event_title text not null default '',
    starts_at timestamptz,
    ends_at timestamptz,
    observed_at timestamptz not null,
    snapshot jsonb not null,
    content_hash text not null,
    created_at timestamptz not null default now(),
    check (btrim(owner_id) <> ''),
    check (btrim(provider) <> ''),
    check (btrim(calendar_id) <> ''),
    check (btrim(provider_event_id) <> ''),
    check (char_length(content_hash) = 64),
    check (jsonb_typeof(snapshot) = 'object'),
    check (ends_at is null or starts_at is null or ends_at >= starts_at),
    unique (owner_id, provider, calendar_id, provider_event_id, content_hash)
);

create index calendar_event_snapshots_owner_event_idx
    on calendar_event_snapshots (
        owner_id,
        provider,
        calendar_id,
        provider_event_id,
        observed_at desc
    );

create function reject_calendar_event_snapshot_update()
returns trigger
language plpgsql
as $$
begin
    raise exception 'calendar event snapshots are append-only';
end;
$$;

create trigger calendar_event_snapshots_append_only_trigger
before update on calendar_event_snapshots
for each row execute function reject_calendar_event_snapshot_update();

create table memory_evidence (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    memory_item_id uuid not null references memory_items(id) on delete cascade,
    evidence_type text not null check (evidence_type in (
        'frame',
        'observation',
        'experience_event',
        'message',
        'tool_execution',
        'reminder',
        'calendar_event',
        'file',
        'memory'
    )),
    evidence_ref_id text not null check (nullif(btrim(evidence_ref_id), '') is not null),
    evidence_time timestamptz not null,
    excerpt text not null default '',
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    unique(memory_item_id, evidence_type, evidence_ref_id)
);

create index memory_evidence_memory_time_idx
    on memory_evidence(memory_item_id, evidence_time asc, id asc);

create index memory_evidence_owner_type_ref_idx
    on memory_evidence(owner_id, evidence_type, evidence_ref_id);

create function reject_memory_evidence_update()
returns trigger
language plpgsql
as $$
begin
    raise exception 'memory evidence rows are append-only';
end;
$$;

create trigger memory_evidence_append_only_trigger
before update on memory_evidence
for each row execute function reject_memory_evidence_update();

create table memory_links (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    source_memory_id uuid not null references memory_items(id) on delete cascade,
    target_memory_id uuid not null references memory_items(id) on delete cascade,
    relationship text not null check (nullif(btrim(relationship), '') is not null),
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    valid_from timestamptz not null default now(),
    valid_to timestamptz,
    supporting_memory_id uuid references memory_items(id) on delete set null,
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (source_memory_id <> target_memory_id),
    check (valid_to is null or valid_to >= valid_from)
);

create unique index memory_links_active_unique_idx
    on memory_links(owner_id, source_memory_id, target_memory_id, relationship)
    where valid_to is null;

create index memory_links_source_idx
    on memory_links(owner_id, source_memory_id, valid_to);

create index memory_links_target_idx
    on memory_links(owner_id, target_memory_id, valid_to);

create table memory_item_embeddings (
    memory_item_id uuid not null references memory_items(id) on delete cascade,
    embedding_kind text not null check (nullif(btrim(embedding_kind), '') is not null),
    provider text not null check (nullif(btrim(provider), '') is not null),
    model text not null check (nullif(btrim(model), '') is not null),
    dimensions integer not null check (dimensions = 1536),
    embedded_content text not null,
    content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
    embedding vector(1536) not null,
    embedded_at timestamptz not null,
    updated_at timestamptz not null default now(),
    primary key (memory_item_id, embedding_kind, provider, model)
);

create index memory_item_embeddings_vector_idx
    on memory_item_embeddings using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create table experience_event_embeddings (
    experience_event_id uuid not null references experience_events(id) on delete cascade,
    embedding_kind text not null check (nullif(btrim(embedding_kind), '') is not null),
    provider text not null check (nullif(btrim(provider), '') is not null),
    model text not null check (nullif(btrim(model), '') is not null),
    dimensions integer not null check (dimensions = 1536),
    embedded_content text not null,
    content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
    embedding vector(1536) not null,
    embedded_at timestamptz not null,
    updated_at timestamptz not null default now(),
    primary key (experience_event_id, embedding_kind, provider, model)
);

create index experience_event_embeddings_vector_idx
    on experience_event_embeddings using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create table memory_entities (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (nullif(btrim(owner_id), '') is not null),
    entity_type text not null check (nullif(btrim(entity_type), '') is not null),
    canonical_name text not null check (nullif(btrim(canonical_name), '') is not null),
    normalized_name text not null check (nullif(btrim(normalized_name), '') is not null),
    aliases text[] not null default '{}'::text[],
    description text not null default '',
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    external_namespace text,
    external_id text,
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (cardinality(aliases) <= 50),
    check (array_position(aliases, null) is null),
    check (
        (external_namespace is null and external_id is null)
        or (
            nullif(btrim(external_namespace), '') is not null
            and nullif(btrim(external_id), '') is not null
        )
    ),
    check (octet_length(metadata::text) <= 32768),
    unique (owner_id, id)
);

create index memory_entities_owner_name_type_idx
    on memory_entities(owner_id, normalized_name, entity_type, created_at, id);

create index memory_entities_aliases_idx
    on memory_entities using gin (aliases);

create unique index memory_entities_external_identity_unique_idx
    on memory_entities(owner_id, external_namespace, external_id)
    where external_namespace is not null and external_id is not null;

create table memory_entity_mentions (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (nullif(btrim(owner_id), '') is not null),
    entity_id uuid not null,
    memory_item_id uuid not null,
    role text not null check (nullif(btrim(role), '') is not null),
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    created_at timestamptz not null default now(),
    unique (entity_id, memory_item_id, role),
    foreign key (owner_id, entity_id)
        references memory_entities(owner_id, id) on delete cascade,
    foreign key (owner_id, memory_item_id)
        references memory_items(owner_id, id) on delete cascade
);

create index memory_entity_mentions_owner_memory_idx
    on memory_entity_mentions(owner_id, memory_item_id, created_at, id);

create index memory_entity_mentions_owner_entity_idx
    on memory_entity_mentions(owner_id, entity_id, created_at, id);

create table memory_entity_embeddings (
    entity_id uuid not null references memory_entities(id) on delete cascade,
    embedding_kind text not null check (nullif(btrim(embedding_kind), '') is not null),
    provider text not null check (nullif(btrim(provider), '') is not null),
    model text not null check (nullif(btrim(model), '') is not null),
    dimensions integer not null check (dimensions = 1536),
    embedded_content text not null,
    content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
    embedding vector(1536) not null,
    embedded_at timestamptz not null,
    updated_at timestamptz not null default now(),
    primary key (entity_id, embedding_kind, provider, model)
);

create index memory_entity_embeddings_vector_idx
    on memory_entity_embeddings using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create table memory_entity_relations (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (nullif(btrim(owner_id), '') is not null),
    subject_entity_id uuid not null,
    predicate text not null check (nullif(btrim(predicate), '') is not null),
    object_entity_id uuid,
    object_value jsonb,
    status text not null default 'active'
        check (status in ('active', 'superseded', 'retracted')),
    confidence double precision not null check (confidence >= 0 and confidence <= 1),
    valid_from timestamptz not null,
    valid_to timestamptz,
    supporting_memory_id uuid not null,
    superseded_by_relation_id uuid references memory_entity_relations(id) on delete restrict,
    metadata jsonb not null default '{}'::jsonb check (jsonb_typeof(metadata) = 'object'),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check ((object_entity_id is not null) <> (object_value is not null)),
    check (object_entity_id is null or object_entity_id <> subject_entity_id),
    check (valid_to is null or valid_to >= valid_from),
    check (superseded_by_relation_id is null or superseded_by_relation_id <> id),
    check (octet_length(metadata::text) <= 32768),
    unique (owner_id, id),
    foreign key (owner_id, subject_entity_id)
        references memory_entities(owner_id, id) on delete restrict,
    foreign key (owner_id, object_entity_id)
        references memory_entities(owner_id, id) on delete restrict,
    foreign key (owner_id, supporting_memory_id)
        references memory_items(owner_id, id) on delete restrict,
    check (
        (status = 'active' and valid_to is null and superseded_by_relation_id is null)
        or (
            status = 'superseded'
            and valid_to is not null
            and superseded_by_relation_id is not null
        )
        or (
            status = 'retracted'
            and valid_to is not null
            and superseded_by_relation_id is null
        )
    )
);

create unique index memory_entity_relations_active_entity_unique_idx
    on memory_entity_relations(
        owner_id,
        subject_entity_id,
        predicate,
        object_entity_id
    )
    where status = 'active' and object_entity_id is not null;

create unique index memory_entity_relations_active_value_unique_idx
    on memory_entity_relations(
        owner_id,
        subject_entity_id,
        predicate,
        object_value
    )
    where status = 'active' and object_value is not null;

create index memory_entity_relations_subject_history_idx
    on memory_entity_relations(
        owner_id,
        subject_entity_id,
        predicate,
        valid_from desc,
        id
    );

create index memory_entity_relations_object_history_idx
    on memory_entity_relations(owner_id, object_entity_id, valid_from desc, id)
    where object_entity_id is not null;

create table memory_retrieval_runs (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (nullif(btrim(owner_id), '') is not null),
    session_id uuid references jarvis_sessions(id) on delete set null,
    query text not null default '',
    constraints jsonb not null default '{}'::jsonb check (jsonb_typeof(constraints) = 'object'),
    candidate_ids uuid[] not null default '{}'::uuid[],
    result_ids uuid[] not null default '{}'::uuid[],
    reinspection_performed boolean not null default false,
    provider_spaces text[] not null default '{}'::text[],
    latency_ms integer not null check (latency_ms >= 0),
    created_at timestamptz not null default now(),
    check (octet_length(query) <= 8192),
    check (octet_length(constraints::text) <= 32768),
    check (cardinality(candidate_ids) <= 200),
    check (cardinality(result_ids) <= 50),
    check (cardinality(provider_spaces) <= 16),
    check (octet_length(array_to_string(provider_spaces, ',')) <= 4096),
    check (array_position(candidate_ids, null) is null),
    check (array_position(result_ids, null) is null),
    check (array_position(provider_spaces, null) is null),
    check (result_ids <@ candidate_ids)
);

create index memory_retrieval_runs_owner_created_idx
    on memory_retrieval_runs(owner_id, created_at desc, id);

create index memory_retrieval_runs_session_created_idx
    on memory_retrieval_runs(session_id, created_at desc, id)
    where session_id is not null;

create function reject_memory_retrieval_run_update()
returns trigger
language plpgsql
as $$
begin
    raise exception 'memory retrieval audit rows are append-only';
end;
$$;

create trigger memory_retrieval_runs_append_only_trigger
before update on memory_retrieval_runs
for each row execute function reject_memory_retrieval_run_update();

create index if not exists perception_observations_pending_compaction_idx
    on perception_observations(owner_id, ended_at, id)
    where compacted_at is null;

create index if not exists perception_observations_compacted_retention_idx
    on perception_observations(owner_id, ended_at, id)
    where compacted_at is not null;

create index if not exists session_working_memory_expiry_idx
    on jarvis_session_working_memory(owner_id, expires_at, session_id);

create index if not exists owner_working_memory_expiry_idx
    on jarvis_owner_working_memory(expires_at, owner_id)
    where expires_at is not null;

create index if not exists experience_events_retention_expiry_idx
    on experience_events(owner_id, retention_expires_at, id)
    where promotion_decision = 'skipped' and retention_expires_at is not null;

create index if not exists memory_retrieval_runs_result_ids_idx
    on memory_retrieval_runs using gin(result_ids);

create table memory_profile_projections (
    owner_id text primary key check (nullif(btrim(owner_id), '') is not null),
    source_revision bigint not null default 0 check (source_revision >= 0),
    projected_revision bigint not null default -1 check (projected_revision >= -1),
    markdown text,
    content_hash text check (
        content_hash is null or content_hash ~ '^[0-9a-f]{64}$'
    ),
    generated_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    check (projected_revision <= source_revision),
    check (
        (markdown is null and content_hash is null and generated_at is null)
        or (markdown is not null and content_hash is not null and generated_at is not null)
    )
);

create index memory_profile_projections_stale_idx
    on memory_profile_projections(owner_id, source_revision, projected_revision)
    where markdown is null or projected_revision < source_revision;

create function mark_memory_profile_source_changed()
returns trigger
language plpgsql
as $$
declare
    changed_owner_id text;
    changed_revision bigint;
begin
    if tg_op = 'DELETE' then
        changed_owner_id := old.owner_id;
    else
        changed_owner_id := new.owner_id;
    end if;

    insert into memory_profile_projections (owner_id, source_revision)
    values (changed_owner_id, 1)
    on conflict (owner_id) do update
    set source_revision = memory_profile_projections.source_revision + 1,
        updated_at = now()
    returning source_revision into changed_revision;

    insert into memory_jobs (
        owner_id,
        session_id,
        job_type,
        dedupe_key,
        payload,
        priority,
        available_at
    )
    values (
        changed_owner_id,
        null,
        'refresh_memory_profile',
        'memory-profile:' || changed_owner_id || ':' || changed_revision::text,
        jsonb_build_object('source_revision', changed_revision),
        -20,
        now()
    )
    on conflict do nothing;

    if tg_op = 'DELETE' then
        return old;
    end if;
    return new;
end;
$$;

create trigger jarvis_behavior_profiles_memory_profile_trigger
after insert or update or delete on jarvis_behavior_profiles
for each row execute function mark_memory_profile_source_changed();

create trigger jarvis_user_preferences_memory_profile_trigger
after insert or update or delete on jarvis_user_preferences
for each row execute function mark_memory_profile_source_changed();

create trigger jarvis_visual_focuses_memory_profile_trigger
after insert or update or delete on jarvis_visual_focuses
for each row execute function mark_memory_profile_source_changed();

create trigger memory_items_memory_profile_trigger
after insert or update or delete on memory_items
for each row execute function mark_memory_profile_source_changed();

create trigger memory_entities_memory_profile_trigger
after insert or update or delete on memory_entities
for each row execute function mark_memory_profile_source_changed();

