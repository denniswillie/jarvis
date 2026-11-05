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

-- BEGIN GENERATED VERSIONED MIGRATIONS (do not edit below this line)
create table if not exists jarvis_schema_migrations (
    version integer primary key check (version > 0),
    name text not null unique,
    checksum text not null check (checksum ~ '^[0-9a-f]{64}$'),
    applied_at timestamptz not null default clock_timestamp()
);

-- Additive foundations. No routines, workers, providers, or grants are enabled.
alter table jarvis_sessions add constraint jarvis_sessions_owner_identity unique (owner_id, id);

create table jarvis_knowledge_areas (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    name text not null check (length(btrim(name)) between 1 and 200),
    domain text not null check (length(btrim(domain)) between 1 and 200),
    status text not null default 'active' check (status in ('active','archived','deleted')),
    revision bigint not null default 1 check (revision > 0),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique (owner_id, id)
);
create index jarvis_knowledge_areas_owner_status_idx on jarvis_knowledge_areas(owner_id,status,id);

create function guard_jarvis_knowledge_area_update() returns trigger language plpgsql as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.created_at<>old.created_at
        or new.revision<>old.revision+1 or (old.status='deleted' and new.status<>'deleted') then
        raise exception 'Knowledge area identity is immutable and updates require the next revision' using errcode='23514';
    end if;
    new.updated_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_knowledge_areas_identity before update on jarvis_knowledge_areas
for each row execute function guard_jarvis_knowledge_area_update();

create table jarvis_context_streams (
    owner_id text primary key check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    last_sequence bigint not null default 0 check (last_sequence >= 0)
);

create table jarvis_context_payloads (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null check (length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    availability text not null check (availability in ('available','unavailable','expired','deleted','not_finalized')),
    content jsonb,
    content_digest text check(content_digest ~ '^[0-9a-f]{64}$'),
    created_at timestamptz not null default clock_timestamp(),
    removed_at timestamptz,
    unique(owner_id,id),
    check ((content is null)=(content_digest is null)),
    check (content is null or (jsonb_typeof(content)='object' and octet_length(content::text)<=262144)),
    check ((availability in ('available','not_finalized') and content is not null and removed_at is null)
        or (availability in ('unavailable','expired','deleted') and content is null))
);

create function guard_jarvis_context_payload_update() returns trigger language plpgsql as $$
begin
    if tg_op='UPDATE' then
        if new.id<>old.id or new.owner_id<>old.owner_id or new.created_at<>old.created_at
            or new.content is not null or new.availability not in ('unavailable','expired','deleted')
            or (old.availability='deleted' and new.availability<>'deleted') then
            raise exception 'Context payloads are immutable except for irreversible removal' using errcode='23514';
        end if;
        new.removed_at := coalesce(old.removed_at, clock_timestamp());
    else
        new.created_at := clock_timestamp();
    end if;
    -- convert_to is STABLE, so this must not be a generated-column expression.
    -- Compute in the same write and ignore any caller-supplied digest.
    new.content_digest := encode(sha256(convert_to(new.content::text,'UTF8')),'hex');
    return new;
end; $$;
create trigger jarvis_context_payload_removal_guard before insert or update on jarvis_context_payloads
for each row execute function guard_jarvis_context_payload_update();

create table jarvis_context_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null references jarvis_context_streams(owner_id),
    sequence bigint not null check(sequence>0),
    session_id uuid,
    source_namespace text not null check(length(btrim(source_namespace)) between 1 and 100),
    source_event_id text not null check(length(btrim(source_event_id)) between 1 and 512),
    revision integer not null default 1 check(revision>0),
    supersedes_event_id uuid,
    kind text not null check(kind in ('user_request','user_transcript','assistant_output','played_output_boundary',
        'tool_outcome','observation','task_update','correction','context_gap')),
    finality text not null check(finality in ('provisional','final','revised','unavailable')),
    trust text not null check(trust in ('user_input','sensor','external_content','assistant_output','tool_receipt','derived')),
    source_time timestamptz,
    recorded_at timestamptz not null default clock_timestamp(),
    payload_id uuid not null,
    unique(owner_id,id),
    unique(owner_id,id,revision),
    unique(owner_id,sequence),
    unique(owner_id,source_namespace,source_event_id,revision),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,supersedes_event_id) references jarvis_context_events(owner_id,id),
    check ((revision=1 and supersedes_event_id is null) or (revision>1 and supersedes_event_id is not null))
);
create index jarvis_context_events_session_sequence_idx on jarvis_context_events(owner_id,session_id,sequence);
create index jarvis_context_events_payload_idx on jarvis_context_events(owner_id,payload_id);
create index jarvis_context_events_supersedes_idx on jarvis_context_events(owner_id,supersedes_event_id);

create function assign_jarvis_context_sequence() returns trigger language plpgsql set search_path from current as $$
begin
    if new.sequence is not null then
        raise exception 'Context sequence is server assigned' using errcode='23514';
    end if;
    insert into jarvis_context_streams(owner_id,last_sequence) values(new.owner_id,1)
        on conflict(owner_id) do update set last_sequence=jarvis_context_streams.last_sequence+1
        returning last_sequence into new.sequence;
    if new.revision>1 and not exists (
        select 1 from jarvis_context_events e where e.owner_id=new.owner_id and e.id=new.supersedes_event_id
        and e.source_namespace=new.source_namespace and e.source_event_id=new.source_event_id
        and e.session_id is not distinct from new.session_id and e.revision=new.revision-1
    ) then
        raise exception 'Context revision must reference its preceding source revision' using errcode='23514';
    end if;
    new.recorded_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_context_sequence before insert on jarvis_context_events
for each row execute function assign_jarvis_context_sequence();

create function reject_jarvis_foundation_update() returns trigger language plpgsql as $$
begin
    raise exception 'Historical record is append only; append a revision or remove its payload' using errcode='23514';
end; $$;
create trigger jarvis_context_events_append_only before update on jarvis_context_events
for each row execute function reject_jarvis_foundation_update();

create table jarvis_capability_preferences (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    capability text not null check(length(btrim(capability)) between 1 and 100),
    knowledge_area_id uuid,
    revision bigint not null check(revision>0),
    request_event_id uuid not null,
    request_revision integer not null,
    policy_payload_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique nulls not distinct(owner_id,capability,knowledge_area_id,revision),
    foreign key(owner_id,knowledge_area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,policy_payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_capability_preferences_append_only before update on jarvis_capability_preferences
for each row execute function reject_jarvis_foundation_update();

create table jarvis_operation_grants (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    capability text not null check(length(btrim(capability)) between 1 and 100),
    operation_key text not null check(length(btrim(operation_key)) between 1 and 200),
    request_event_id uuid not null,
    request_revision integer not null,
    scope_revision integer not null check(scope_revision>0),
    scope_payload_id uuid not null,
    status text not null default 'active' check(status in ('active','revoked','expired')),
    created_at timestamptz not null default clock_timestamp(),
    expires_at timestamptz not null,
    revoked_at timestamptz,
    unique(owner_id,id),
    unique(owner_id,request_event_id,request_revision,operation_key,scope_revision),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,scope_payload_id) references jarvis_context_payloads(owner_id,id),
    check(isfinite(created_at) and isfinite(expires_at) and expires_at>created_at),
    check((status='revoked')=(revoked_at is not null))
);
create index jarvis_operation_grants_expiry_idx on jarvis_operation_grants(owner_id,expires_at) where status='active';

create function guard_jarvis_explicit_request() returns trigger language plpgsql set search_path from current as $$
declare target_payload uuid;
begin
    -- Serialize authority creation against revisions of this owner's inputs.
    perform 1 from jarvis_context_streams where owner_id=new.owner_id for update;
    if not exists (
        select 1 from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=new.request_revision
            and e.trust='user_input' and e.kind in ('user_request','user_transcript')
            and e.finality in ('final','revised') and p.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
    end if;
    if tg_table_name='jarvis_operation_grants' then target_payload:=new.scope_payload_id;
    else target_payload:=new.policy_payload_id; end if;
    if not exists(select 1 from jarvis_context_payloads where owner_id=new.owner_id
        and id=target_payload and availability='available') then
        raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
    end if;
    new.created_at := clock_timestamp();
    return new;
end; $$;
create trigger jarvis_operation_grants_explicit_request before insert on jarvis_operation_grants
for each row execute function guard_jarvis_explicit_request();
create trigger jarvis_capability_preferences_explicit_request before insert on jarvis_capability_preferences
for each row execute function guard_jarvis_explicit_request();

create function guard_jarvis_grant_update() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['status','revoked_at'])<>(to_jsonb(old)-array['status','revoked_at'])
        or old.status<>'active' or new.status not in ('revoked','expired') then
        raise exception 'Operation grants can only be revoked or expired, never broadened' using errcode='23514';
    end if;
    if new.status='revoked' then new.revoked_at:=clock_timestamp(); end if;
    return new;
end; $$;
create trigger jarvis_operation_grants_immutable_scope before update on jarvis_operation_grants
for each row execute function guard_jarvis_grant_update();
insert into jarvis_schema_migrations (version, name, checksum) values (1, 'second_brain_foundations', '4b75e77e1aa67149e9877f5b427b3cb0c889d9e167fc32610b3b7beb197773fe');

create table jarvis_background_tasks (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid,
    knowledge_area_id uuid,
    backend_kind text not null check(backend_kind in ('frontier','sandbox')),
    model text not null check(length(model) between 1 and 100),
    status text not null default 'queued' check(status in ('queued','running','waiting_for_input','paused','succeeded',
        'failed','cancel_requested','cancelled','outcome_unknown')),
    revision bigint not null default 1 check(revision>0),
    context_watermark bigint not null default 0 check(context_watermark>=0),
    description_payload_id uuid not null,
    request_event_id uuid not null,
    request_revision integer not null,
    current_attempt_id uuid,
    fence bigint not null default 0 check(fence>=0),
    attempt_count integer not null default 0 check(attempt_count>=0),
    active_seconds double precision not null default 0 check(active_seconds>=0 and active_seconds<'Infinity'::float8),
    budget jsonb not null check(jsonb_typeof(budget)='object' and octet_length(budget::text)<=8192),
    result_payload_id uuid,
    reason text check(length(reason)<=100),
    pending_status text check(pending_status in ('queued','cancelled','paused')),
    last_event_sequence bigint not null default 0 check(last_event_sequence>=0),
    available_at timestamptz not null default clock_timestamp(),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,knowledge_area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key(owner_id,description_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,result_payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_tasks_claim_idx on jarvis_background_tasks(status,available_at,created_at,id);
create index jarvis_tasks_owner_status_idx on jarvis_background_tasks(owner_id,status,created_at,id);
create index jarvis_tasks_area_idx on jarvis_background_tasks(owner_id,knowledge_area_id);

create table jarvis_task_attempts (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    attempt_number integer not null check(attempt_number>0),
    fence bigint not null check(fence>0),
    worker_id text not null check(length(worker_id) between 1 and 200),
    status text not null check(status in ('running','recovering','succeeded','failed','interrupted','cancelled','outcome_unknown')),
    provider_state text not null default 'not_started' check(provider_state in ('not_started','submitting','live','terminal')),
    provider_handle text check(length(provider_handle) between 1 and 512),
    checkpoint_payload_id uuid,
    lease_until timestamptz not null check(isfinite(lease_until)),
    charged_until timestamptz not null default clock_timestamp(),
    started_at timestamptz not null default clock_timestamp(),
    ended_at timestamptz,
    unique(owner_id,id),
    unique(owner_id,task_id,id),
    unique(owner_id,task_id,attempt_number),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,checkpoint_payload_id) references jarvis_context_payloads(owner_id,id),
    check(provider_state<>'live' or provider_handle is not null)
);
alter table jarvis_background_tasks add constraint jarvis_tasks_current_attempt_scope
    foreign key(owner_id,id,current_attempt_id) references jarvis_task_attempts(owner_id,task_id,id);
create index jarvis_task_attempts_recovery_idx on jarvis_task_attempts(status,lease_until);

create table jarvis_task_events (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    sequence bigint not null check(sequence>0),
    task_revision bigint not null check(task_revision>0),
    attempt_id uuid,
    event_type text not null check(length(event_type) between 1 and 80),
    payload_id uuid,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,sequence),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_task_events_append_only before update on jarvis_task_events
for each row execute function reject_jarvis_foundation_update();

create table jarvis_task_requests (
    owner_id text not null,
    request_id uuid not null,
    task_id uuid not null,
    operation text not null check(operation in ('create','update','cancel','resume')),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_revision bigint not null check(accepted_revision>0),
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id)
);
create trigger jarvis_task_requests_append_only before update on jarvis_task_requests
for each row execute function reject_jarvis_foundation_update();

create table jarvis_task_grants (
    owner_id text not null,
    task_id uuid not null,
    grant_id uuid not null,
    primary key(owner_id,task_id,grant_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,grant_id) references jarvis_operation_grants(owner_id,id)
);

create table jarvis_task_owner_scheduling (
    owner_id text primary key,
    last_claimed_at timestamptz not null default '-infinity'
);

-- Shared admission limits do not depend on which process wins a claim.
create table jarvis_task_scheduler_limits (
    singleton boolean primary key default true check(singleton),
    global_concurrency integer not null check(global_concurrency>0),
    owner_concurrency integer not null check(owner_concurrency>0 and owner_concurrency<=global_concurrency),
    updated_at timestamptz not null default clock_timestamp()
);

create table jarvis_task_reservations (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    attempt_id uuid not null,
    operation_key text not null check(length(operation_key) between 1 and 200),
    status text not null check(status in ('reserved','submitted','settled','released','unknown')),
    is_provider boolean not null default false,
    input_tokens bigint not null check(input_tokens>=0),
    output_tokens bigint not null check(output_tokens>=0),
    tool_steps integer not null check(tool_steps>=0),
    actual_input_tokens bigint check(actual_input_tokens>=0),
    actual_output_tokens bigint check(actual_output_tokens>=0),
    actual_tool_steps integer check(actual_tool_steps>=0),
    accounting_day date not null default (clock_timestamp() at time zone 'UTC')::date,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,operation_key),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    check((status='settled')=(actual_input_tokens is not null and actual_output_tokens is not null and actual_tool_steps is not null))
);
create index jarvis_task_reservations_daily_idx on jarvis_task_reservations(owner_id,accounting_day);

create table jarvis_task_actions (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    operation_key text not null check(length(operation_key) between 1 and 200),
    grant_id uuid not null,
    scope_revision integer not null check(scope_revision>0),
    parameters_payload_id uuid not null,
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    status text not null check(status in ('prepared','authorized','submitting','accepted','rejected','outcome_unknown','reconciled','revoked')),
    submission_token uuid,
    receipt_payload_id uuid,
    reconciled_outcome text check(reconciled_outcome in ('accepted','rejected','not_submitted')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,operation_key),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    unique(owner_id,grant_id),
    foreign key(owner_id,task_id,grant_id) references jarvis_task_grants(owner_id,task_id,grant_id),
    foreign key(owner_id,parameters_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,receipt_payload_id) references jarvis_context_payloads(owner_id,id),
    check((status='reconciled')=(reconciled_outcome is not null)),
    check(status not in ('submitting','accepted','rejected','outcome_unknown') or submission_token is not null)
);
create index jarvis_task_actions_unresolved_idx on jarvis_task_actions(owner_id,task_id,status);
insert into jarvis_schema_migrations (version, name, checksum) values (2, 'durable_tasks', '97be7db0b6783e744e07ecd3594728191fb9fbd34f5c457936aad302e775bc3b');

create table jarvis_context_ingress_receipts (
    owner_id text not null,
    source_namespace text not null,
    source_event_id text not null,
    receipt_id text not null check(length(receipt_id) between 1 and 512),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    event_id uuid not null,
    primary key(owner_id,source_namespace,source_event_id,receipt_id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);
create trigger jarvis_context_ingress_receipts_append_only before update on jarvis_context_ingress_receipts
for each row execute function reject_jarvis_foundation_update();

create table jarvis_context_pins (
    owner_id text not null,
    session_id uuid not null,
    event_id uuid not null,
    reason text not null check(length(reason) between 1 and 200),
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,session_id,event_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);

create table jarvis_task_context_manifests (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    context_watermark bigint not null check(context_watermark>=0),
    payload_id uuid not null,
    status text not null default 'active' check(status in ('active','superseded','revoked')),
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,task_id,id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);

create table jarvis_task_context_sources (
    owner_id text not null,
    manifest_id uuid not null,
    event_id uuid not null,
    event_revision integer not null,
    source_usage text not null default 'current' check(source_usage in ('current','historical')),
    payload_id uuid not null,
    content_digest text,
    primary key(owner_id,manifest_id,event_id),
    foreign key(owner_id,manifest_id) references jarvis_task_context_manifests(owner_id,id),
    foreign key(owner_id,event_id,event_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_task_context_sources_payload_idx on jarvis_task_context_sources(owner_id,payload_id);
alter table jarvis_task_attempts add column context_manifest_id uuid;
alter table jarvis_task_attempts add constraint jarvis_task_attempt_context_scope
    foreign key(owner_id,task_id,context_manifest_id) references jarvis_task_context_manifests(owner_id,task_id,id);

create function invalidate_jarvis_context_revision() returns trigger language plpgsql set search_path from current as $$
begin
    if new.revision>1 then
        update jarvis_task_context_manifests m set status='superseded'
        where m.owner_id=new.owner_id and m.status='active' and exists (
            select 1 from jarvis_task_context_sources s join jarvis_context_events e
                on e.owner_id=s.owner_id and e.id=s.event_id
            where s.owner_id=m.owner_id and s.manifest_id=m.id
                and e.source_namespace=new.source_namespace and e.source_event_id=new.source_event_id
                and e.revision<new.revision and s.source_usage='current'
        );
    end if;
    return new;
end; $$;
create trigger jarvis_context_revision_invalidates_manifests after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_revision();

create function revoke_jarvis_context_payload_dependents() returns trigger language plpgsql set search_path from current as $$
declare derived_payloads uuid[];
begin
    if new.availability in ('deleted','expired','unavailable') and old.availability is distinct from new.availability then
        with revoked as (
            update jarvis_task_context_manifests m set status='revoked'
            where m.owner_id=new.owner_id and m.status<>'revoked' and exists (
                select 1 from jarvis_task_context_sources s where s.owner_id=m.owner_id
                    and s.manifest_id=m.id and s.payload_id=new.id
            ) returning m.payload_id
        ) select array_agg(payload_id) into derived_payloads from revoked;
        update jarvis_context_payloads set availability='deleted',content=null
        where owner_id=new.owner_id and id=any(derived_payloads) and id<>new.id
            and availability in ('available','not_finalized');
    end if;
    return new;
end; $$;
create trigger jarvis_context_removal_revokes_manifests after update on jarvis_context_payloads
for each row execute function revoke_jarvis_context_payload_dependents();
insert into jarvis_schema_migrations (version, name, checksum) values (3, 'task_context', '8ea6966a31a926ddb0df4b9ece124d0caa26840011ed5aa0a419168ee2de1d4c');

-- M02 source substrate only. No vendor calls, live route or ingestion worker.
create table jarvis_memory_banks (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    provider text not null default 'hindsight' check(provider='hindsight'),
    bank_name text not null unique default ('jarvis_' || replace(gen_random_uuid()::text,'-','')),
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,area_id),
    unique(owner_id,area_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id)
);
create trigger jarvis_memory_banks_immutable before update on jarvis_memory_banks
for each row execute function reject_jarvis_foundation_update();

create table jarvis_knowledge_sources (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    namespace text not null check(length(btrim(namespace)) between 1 and 100),
    source_key text not null check(length(btrim(source_key)) between 1 and 512),
    current_revision bigint not null check(current_revision>0),
    generation bigint not null check(generation>0),
    status text not null default 'active' check(status in ('active','unavailable','deleted')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    unique(owner_id,area_id,namespace,source_key),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id)
);

create table jarvis_knowledge_source_versions (
    owner_id text not null,
    area_id uuid not null,
    source_id uuid not null,
    revision bigint not null check(revision>0),
    generation bigint not null check(generation>0),
    payload_id uuid not null,
    bank_id uuid not null,
    document_id text not null unique default ('source_' || replace(gen_random_uuid()::text,'-','')),
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_type text not null check(length(btrim(source_type)) between 1 and 80),
    trust text not null check(trust in ('user_input','sensor','external_content','assistant_output','tool_receipt','derived')),
    source_time timestamptz check(source_time is null or isfinite(source_time)),
    time_basis text not null check(length(btrim(time_basis)) between 1 and 200),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,revision),
    unique(owner_id,document_id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,bank_id) references jarvis_memory_banks(owner_id,area_id,id)
);
alter table jarvis_knowledge_sources add constraint jarvis_source_current_version
foreign key(owner_id,id,current_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision)
deferrable initially deferred;
create index jarvis_source_versions_payload_idx on jarvis_knowledge_source_versions(owner_id,payload_id);
create trigger jarvis_source_versions_immutable before update on jarvis_knowledge_source_versions
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_source_update() returns trigger language plpgsql as $$
begin
    if (to_jsonb(new)-array['current_revision','generation','status','updated_at'])
        <> (to_jsonb(old)-array['current_revision','generation','status','updated_at'])
        or old.status='deleted' or new.generation<>old.generation+1
        or (new.status='active' and new.current_revision<>old.current_revision+1)
        or (new.status<>'active' and new.current_revision<>old.current_revision) then
        raise exception 'Source updates require immutable identity and the next generation' using errcode='23514';
    end if;
    new.updated_at:=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_source_identity before update on jarvis_knowledge_sources
for each row execute function guard_jarvis_source_update();

create table jarvis_memory_outbox (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    document_id text not null,
    generation bigint not null check(generation>0),
    operation text not null check(operation in ('retain','delete')),
    status text not null default 'pending'
        check(status in ('pending','leased','accepted','searchable','outcome_unknown','done','failed','cancelled')),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,document_id,operation),
    foreign key(owner_id,document_id) references jarvis_knowledge_source_versions(owner_id,document_id)
);
create index jarvis_memory_outbox_pending_idx on jarvis_memory_outbox(owner_id,status,created_at,id);

create function invalidate_jarvis_memory_payload() returns trigger language plpgsql set search_path from current as $$
begin
    if new.availability in ('available','not_finalized') then return new; end if;
    update jarvis_knowledge_sources s set
        status=case when new.availability='deleted' then 'deleted' else 'unavailable' end,
        generation=s.generation+1
    where s.owner_id=new.owner_id and s.status<>'deleted' and exists(
        select 1 from jarvis_knowledge_source_versions v where v.owner_id=s.owner_id
        and v.source_id=s.id and v.revision=s.current_revision and v.payload_id=new.id);
    update jarvis_memory_outbox o set status='cancelled',updated_at=clock_timestamp()
    where o.owner_id=new.owner_id and o.operation='retain' and o.status='pending'
        and exists(select 1 from jarvis_knowledge_source_versions v
            where v.owner_id=o.owner_id and v.document_id=o.document_id and v.payload_id=new.id);
    insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
    select v.owner_id,v.document_id,s.generation,'delete' from jarvis_knowledge_source_versions v
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    where v.owner_id=new.owner_id and v.payload_id=new.id
    on conflict(owner_id,document_id,operation) do update
        set generation=excluded.generation,status='pending',updated_at=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_memory_payload_removal after update on jarvis_context_payloads
for each row execute function invalidate_jarvis_memory_payload();

-- Durable execution ownership is separate from source revision/generation.
alter table jarvis_memory_outbox
    add column fence bigint not null default 0 check(fence>=0),
    add column worker_id text,
    add column lease_until timestamptz,
    add column dispatch_count integer not null default 0 check(dispatch_count>=0),
    add column failure_count integer not null default 0 check(failure_count>=0),
    add column next_attempt_at timestamptz not null default clock_timestamp(),
    add column last_error text,
    add column cleanup_operation_ids uuid[]
        check(cleanup_operation_ids is null or (
            operation='delete' and cardinality(cleanup_operation_ids) between 0 and 64
            and coalesce(array_ndims(cleanup_operation_ids),1)=1
            and array_position(cleanup_operation_ids,null) is null
        ));

-- This is a terminal-operation inventory, persisted before deleting any remote
-- records. NULL means unprepared; an empty array means no retain was dispatched.
-- Keep the IDs after cleanup so lost acknowledgements and rescheduled tombstones
-- do not turn intentional operation deletion into an unknown retention outcome.
create function guard_jarvis_memory_cleanup_inventory() returns trigger language plpgsql as $$
begin
    if old.cleanup_operation_ids is not null
       and new.cleanup_operation_ids is distinct from old.cleanup_operation_ids then
        raise exception 'A prepared memory cleanup inventory is immutable' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_memory_cleanup_inventory_immutable before update on jarvis_memory_outbox
for each row execute function guard_jarvis_memory_cleanup_inventory();
create index jarvis_memory_outbox_due_idx on jarvis_memory_outbox(owner_id,next_attempt_at,id)
    where status in ('pending','accepted','outcome_unknown','leased');

-- Exact canonical lineage. Original contents stay in jarvis_context_payloads.
create table jarvis_knowledge_context_sources (
    owner_id text not null,
    source_id uuid not null,
    source_revision bigint not null,
    event_id uuid not null,
    event_revision integer not null,
    primary key(owner_id,source_id,source_revision),
    foreign key(owner_id,source_id,source_revision)
        references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,event_id,event_revision) references jarvis_context_events(owner_id,id,revision)
);
create index jarvis_knowledge_context_event_idx on jarvis_knowledge_context_sources(owner_id,event_id);
create trigger jarvis_knowledge_context_sources_immutable before update on jarvis_knowledge_context_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_knowledge_context_source() returns trigger language plpgsql set search_path from current as $$
begin
    if not exists(select 1 from jarvis_context_events e
        join jarvis_knowledge_source_versions v on v.owner_id=e.owner_id and v.payload_id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.event_id and e.revision=new.event_revision
        and e.trust=v.trust and e.kind=v.source_type and e.source_time is not distinct from v.source_time
        and v.source_id=new.source_id and v.revision=new.source_revision
        and e.finality in ('final','revised')
        and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
            and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
            and newer.revision>e.revision)) then
        raise exception 'Knowledge lineage requires the exact current canonical original' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_knowledge_context_source_guard before insert on jarvis_knowledge_context_sources
for each row execute function guard_jarvis_knowledge_context_source();

create function invalidate_jarvis_knowledge_context_revision() returns trigger language plpgsql set search_path from current as $$
begin
    if new.revision=1 then return new; end if;
    update jarvis_knowledge_sources s set status='unavailable',generation=s.generation+1
    where s.owner_id=new.owner_id and s.status='active' and exists(
        select 1 from jarvis_knowledge_context_sources link
        join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
        where link.owner_id=s.owner_id and link.source_id=s.id and link.source_revision=s.current_revision
        and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
        and prior.revision<new.revision);
    update jarvis_memory_outbox o set status='cancelled',updated_at=clock_timestamp()
    where o.owner_id=new.owner_id and o.operation='retain' and o.status='pending' and exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_context_sources link on link.owner_id=v.owner_id
            and link.source_id=v.source_id and link.source_revision=v.revision
        join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
        where v.owner_id=o.owner_id and v.document_id=o.document_id
        and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
        and prior.revision<new.revision);
    insert into jarvis_memory_outbox(owner_id,document_id,generation,operation)
    select v.owner_id,v.document_id,s.generation,'delete'
    from jarvis_knowledge_source_versions v
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_knowledge_context_sources link on link.owner_id=v.owner_id
        and link.source_id=v.source_id and link.source_revision=v.revision
    join jarvis_context_events prior on prior.owner_id=link.owner_id and prior.id=link.event_id
    where v.owner_id=new.owner_id and prior.source_namespace=new.source_namespace
        and prior.source_event_id=new.source_event_id and prior.revision<new.revision
    on conflict(owner_id,document_id,operation) do update set generation=excluded.generation,
        status='pending',next_attempt_at=clock_timestamp(),updated_at=clock_timestamp();
    return new;
end; $$;
create trigger jarvis_knowledge_context_revision after insert on jarvis_context_events
for each row execute function invalidate_jarvis_knowledge_context_revision();
insert into jarvis_schema_migrations (version, name, checksum) values (4, 'memory_sources', 'ba24f50bb2e5cb58a9634737155bce7338ab3587a77796f675800563124bf129');

-- API admission metadata is separate from execution. No live feature is enabled.
create table jarvis_task_workers (
    worker_id text not null check (length(worker_id) between 1 and 100),
    owner_id text not null check (length(owner_id) between 1 and 200),
    host_id text not null check (length(host_id) between 1 and 200),
    backend_kind text not null check (backend_kind in ('frontier','sandbox')),
    model text not null check (length(model) between 1 and 100),
    draining boolean not null default false,
    updated_at timestamptz not null default clock_timestamp(),
    expires_at timestamptz not null,
    primary key(worker_id,backend_kind)
);
create index jarvis_task_workers_readiness on jarvis_task_workers(owner_id,backend_kind,model,expires_at)
    where not draining;

alter table jarvis_task_attempts add column runtime_host_id text
    check (runtime_host_id is null or length(runtime_host_id) between 1 and 200);

create function guard_jarvis_task_runtime_host() returns trigger language plpgsql as $$
begin
    if old.runtime_host_id is not null and new.runtime_host_id is distinct from old.runtime_host_id then
        raise exception 'Native runtime host cannot be reassigned' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_task_runtime_host_guard before update on jarvis_task_attempts
for each row execute function guard_jarvis_task_runtime_host();
insert into jarvis_schema_migrations (version, name, checksum) values (5, 'task_worker_roles', '46b11eb1d94e209473b5c838fb647f2089e6abf26c6b77d7a2f0e397f651b4e2');

-- Task dependencies on M02 originals. No provider call or live memory cutover.
create table jarvis_task_memory_sources (
    owner_id text not null,
    manifest_id uuid not null,
    area_id uuid not null,
    source_id uuid not null,
    source_revision bigint not null,
    source_generation bigint not null check(source_generation>0),
    payload_id uuid not null,
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,manifest_id,source_id,source_revision),
    foreign key(owner_id,manifest_id) references jarvis_task_context_manifests(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_task_memory_payload_idx on jarvis_task_memory_sources(owner_id,payload_id);
create index jarvis_task_memory_source_idx on jarvis_task_memory_sources(owner_id,source_id);
create trigger jarvis_task_memory_sources_immutable before update on jarvis_task_memory_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_task_memory_source() returns trigger language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
        join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
        join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
        where v.owner_id=new.owner_id and v.area_id=new.area_id and v.source_id=new.source_id
            and v.revision=new.source_revision and s.generation=new.source_generation
            and v.payload_id=new.payload_id and v.content_digest=new.content_digest
            and p.availability='available' and s.status='active' and a.status='active'
            and ((new.source_usage='current' and s.current_revision=v.revision)
                 or (new.source_usage='historical' and s.current_revision>v.revision))
    ) then
        raise exception 'Task memory dependency must match available original evidence' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_source_guard before insert on jarvis_task_memory_sources
for each row execute function guard_jarvis_task_memory_source();

create function invalidate_jarvis_task_memory_manifests(p_owner text,p_manifests uuid[],p_removed boolean)
returns void language plpgsql set search_path from current as $$
declare derived_payloads uuid[];
begin
    if p_manifests is null then return; end if;
    update jarvis_task_context_manifests set status=case when p_removed then 'revoked' else 'superseded' end
    where owner_id=p_owner and id=any(p_manifests)
        and (status='active' or (p_removed and status<>'revoked'));

    -- Completed results have no current_attempt_id. Use their exact event/attempt
    -- lineage, so an older invalidated attempt cannot hide a newer answer.
    update jarvis_background_tasks t set result_payload_id=null
    where t.owner_id=p_owner and t.result_payload_id is not null and exists(
        select 1 from jarvis_task_events e
        join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
        where e.owner_id=t.owner_id and e.task_id=t.id and e.payload_id=t.result_payload_id
            and a.context_manifest_id=any(p_manifests)
    );
    if p_removed then
        select array_agg(distinct payload_id) into derived_payloads from (
            select m.payload_id from jarvis_task_context_manifests m
                where m.owner_id=p_owner and m.id=any(p_manifests)
            union
            select a.checkpoint_payload_id from jarvis_task_attempts a
                where a.owner_id=p_owner and a.context_manifest_id=any(p_manifests)
            union
            select e.payload_id from jarvis_task_events e
                join jarvis_task_attempts a on a.owner_id=e.owner_id and a.id=e.attempt_id
                where a.owner_id=p_owner and a.context_manifest_id=any(p_manifests)
        ) derived where payload_id is not null;
        update jarvis_context_payloads set availability='deleted',content=null
        where owner_id=p_owner and id=any(derived_payloads) and availability in ('available','not_finalized');
    end if;
end; $$;

create function invalidate_jarvis_task_memory_revision() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
        where owner_id=new.owner_id and source_id=new.id
            and (source_generation<>new.generation or new.status<>'active');
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,new.status<>'active');
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_revision after update on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_task_memory_revision();

create function revoke_jarvis_task_memory_payload() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if new.availability in ('deleted','expired','unavailable') and old.availability is distinct from new.availability then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
            where owner_id=new.owner_id and payload_id=new.id;
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,true);
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_payload after update on jarvis_context_payloads
for each row execute function revoke_jarvis_task_memory_payload();

create function revoke_jarvis_task_memory_area() returns trigger language plpgsql set search_path from current as $$
declare manifests uuid[];
begin
    if new.status<>'active' and old.status is distinct from new.status then
        select array_agg(distinct manifest_id) into manifests from jarvis_task_memory_sources
            where owner_id=new.owner_id and area_id=new.id;
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,manifests,true);
    end if;
    return new;
end; $$;
create trigger jarvis_task_memory_area after update on jarvis_knowledge_areas
for each row execute function revoke_jarvis_task_memory_area();
insert into jarvis_schema_migrations (version, name, checksum) values (6, 'task_memory_evidence', '4630a4dabab909fa3cc0205cfb6d3d8cd3cda5d838a8a04be523c4fab483ca53');

-- Session context is attached only by an accepted, source-backed voice operation.
-- This is task authorization/lineage, not another conversation or memory store.
create table jarvis_task_session_context (
    owner_id text not null,
    task_id uuid not null,
    session_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision > 0),
    primary key(owner_id,task_id,session_id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,request_event_id,request_revision)
        references jarvis_context_events(owner_id,id,revision)
);
insert into jarvis_schema_migrations (version, name, checksum) values (7, 'task_session_context', '2d2c3bb5cc07f2563600a4be7ece543a8d11b02537e6f94f225bd3df328cbf5c');

-- Product delivery records, not inferred memories or a second agent runtime.
create table jarvis_attention_intents (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    source_namespace text not null check(length(source_namespace) between 1 and 80),
    source_id uuid not null,
    source_revision bigint not null check(source_revision>0),
    source_payload_id uuid not null,
    task_id uuid,
    task_revision bigint check(task_revision>0),
    preferred_session_id uuid,
    state text not null default 'pending' check(state in
        ('pending','claimed','awaiting_receipt','played','unknown','failed','dismissed','revoked','expired')),
    generation bigint not null default 1 check(generation>0),
    urgency text not null default 'normal' check(urgency in ('normal','high')),
    privacy text not null default 'private' check(privacy in ('private','sensitive')),
    coalesce_key text check(length(coalesce_key) between 1 and 200),
    not_before timestamptz not null default clock_timestamp(),
    expires_at timestamptz check(isfinite(expires_at)),
    attempt_count integer not null default 0 check(attempt_count>=0),
    reason text,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,source_namespace,source_id,source_revision),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,task_id) references jarvis_background_tasks(owner_id,id),
    foreign key(owner_id,preferred_session_id) references jarvis_sessions(owner_id,id),
    check((task_id is null)=(task_revision is null))
);
create index jarvis_attention_pending on jarvis_attention_intents(owner_id,state,not_before,created_at);

create table jarvis_attention_attempts (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    intent_id uuid not null,
    generation bigint not null check(generation>0),
    channel text not null check(channel in ('voice','phone')),
    session_id uuid,
    worker_id text not null check(length(worker_id) between 1 and 200),
    phase text not null default 'claimed' check(phase in
        ('claimed','submitted','forwarding','awaiting_receipt','played','interrupted','failed','unknown','revoked')),
    plan_payload_id uuid,
    plan_digest text check(plan_digest ~ '^[0-9a-f]{64}$'),
    forwarded_samples bigint not null default 0 check(forwarded_samples>=0),
    played_samples bigint check(played_samples>=0 and played_samples<=forwarded_samples),
    first_audio_at timestamptz,
    submitted_at timestamptz,
    generation_finished_at timestamptz,
    lease_until timestamptz not null check(isfinite(lease_until)),
    reason text,
    created_at timestamptz not null default clock_timestamp(),
    ended_at timestamptz,
    unique(owner_id,id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,plan_payload_id) references jarvis_context_payloads(owner_id,id),
    check((plan_payload_id is null)=(plan_digest is null)),
    check(channel<>'voice' or session_id is not null)
);
create unique index jarvis_attention_one_active_channel on jarvis_attention_attempts(owner_id,channel)
    where phase in ('claimed','submitted','forwarding','awaiting_receipt');
create unique index jarvis_attention_one_active_attempt on jarvis_attention_attempts(owner_id,intent_id)
    where phase in ('claimed','submitted','forwarding','awaiting_receipt');

create table jarvis_attention_requests (
    owner_id text not null,
    request_id uuid not null,
    intent_id uuid not null,
    operation text not null check(operation in ('dismiss','replay')),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_generation bigint not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id)
);
create trigger jarvis_attention_requests_append_only before update on jarvis_attention_requests
for each row execute function reject_jarvis_foundation_update();

create function jarvis_attention_revoke_task_results() returns trigger language plpgsql as $$
begin
    update jarvis_attention_intents set state='revoked',generation=generation+1,
        reason='task_result_changed',updated_at=clock_timestamp()
    where owner_id=new.owner_id and task_id=new.id and state<>'revoked'
      and (task_revision<>new.revision or source_payload_id is distinct from new.result_payload_id);
    update jarvis_attention_attempts a set phase='revoked',reason='task_result_changed',ended_at=clock_timestamp()
    from jarvis_attention_intents i where a.owner_id=i.owner_id and a.intent_id=i.id
      and i.owner_id=new.owner_id and i.task_id=new.id and i.state='revoked'
      and a.phase in ('claimed','submitted','forwarding','awaiting_receipt');
    if new.result_payload_id is not null then
        insert into jarvis_attention_intents(owner_id,source_namespace,source_id,source_revision,
            source_payload_id,task_id,task_revision,preferred_session_id)
        select new.owner_id,'task_result',p.id,new.revision,p.id,new.id,new.revision,new.session_id
        from jarvis_context_payloads p where p.owner_id=new.owner_id and p.id=new.result_payload_id
          and p.availability='available' and p.content->>'format'='jarvis.hermes.task_result.v2'
          and p.content->'authorship'->>'kind'='expert'
          and p.content->'authorship'->>'owner_id'=new.owner_id
          and p.content->'authorship'->>'task_id'=new.id::text
          and p.content->'authorship'->>'task_revision'=new.revision::text
        on conflict(owner_id,source_namespace,source_id,source_revision) do nothing;
    end if;
    return new;
end $$;
create trigger jarvis_attention_task_result after update of result_payload_id,revision on jarvis_background_tasks
for each row execute function jarvis_attention_revoke_task_results();

create function jarvis_attention_revoke_payload() returns trigger language plpgsql as $$
begin
    if new.availability<>'available' or new.content is null then
        update jarvis_attention_intents set state='revoked',generation=generation+1,
            reason='source_payload_unavailable',updated_at=clock_timestamp()
        where owner_id=new.owner_id and source_payload_id=new.id and state<>'revoked';
        update jarvis_attention_attempts a set phase='revoked',reason='source_payload_unavailable',ended_at=clock_timestamp()
        from jarvis_attention_intents i where a.owner_id=i.owner_id and a.intent_id=i.id
          and i.owner_id=new.owner_id and i.source_payload_id=new.id
          and a.phase in ('claimed','submitted','forwarding','awaiting_receipt');
    end if;
    return new;
end $$;
create trigger jarvis_attention_payload_changed after update of availability,content on jarvis_context_payloads
for each row execute function jarvis_attention_revoke_payload();

-- Existing completed results are not automatically replayed by an upgrade.
-- Intents are atomically created on subsequent result publications.
insert into jarvis_schema_migrations (version, name, checksum) values (8, 'attention_inbox', '8ff5236a7b01f486fef5701aa7d65c0a0ddc238ce6046e39626cebd697abe40b');

-- Immutable network-submission identities and bounded playback evidence.
create table jarvis_attention_speech_chunks (
    owner_id text not null,
    attempt_id uuid not null,
    chunk_index integer not null check(chunk_index between 0 and 63),
    input_digest text not null check(input_digest ~ '^[0-9a-f]{64}$'),
    response_id text not null,
    item_id text not null,
    submitted_at timestamptz,
    generated_at timestamptz,
    forwarded_samples bigint not null default 0 check(forwarded_samples >= 0),
    played_samples bigint check(played_samples between 0 and forwarded_samples),
    playback_completed_at timestamptz,
    first_bytes_ms double precision check(first_bytes_ms >= 0),
    provider_request_id text,
    primary key(owner_id,attempt_id,chunk_index),
    unique(owner_id,item_id),
    foreign key(owner_id,attempt_id) references jarvis_attention_attempts(owner_id,id)
);

create function jarvis_attention_chunk_identity_immutable() returns trigger language plpgsql as $$
begin
    if (new.owner_id,new.attempt_id,new.chunk_index,new.input_digest,new.response_id,new.item_id)
        is distinct from (old.owner_id,old.attempt_id,old.chunk_index,old.input_digest,old.response_id,old.item_id)
        or (old.submitted_at is not null and new.submitted_at is distinct from old.submitted_at)
        or new.forwarded_samples < old.forwarded_samples
        or (old.played_samples is not null and (new.played_samples is null or new.played_samples < old.played_samples)) then
        raise exception 'attention chunk identity or receipt cannot be rewritten';
    end if;
    return new;
end $$;
create trigger jarvis_attention_chunk_identity_immutable before update on jarvis_attention_speech_chunks
    for each row execute function jarvis_attention_chunk_identity_immutable();
insert into jarvis_schema_migrations (version, name, checksum) values (9, 'attention_speech_chunks', 'd0b2d7f3535cdb02826050b0c24d96e46e3ebc2e75190319058adb8ef259463b');

-- Original-backed copies, not another memory engine. Only explicit migration
-- execution applies this to a live schema; all acceptance uses manifested data.
create index jarvis_attention_source_payload_idx on jarvis_attention_intents(owner_id,source_payload_id);

create table jarvis_context_payload_dependencies (
    owner_id text not null,
    source_payload_id uuid not null,
    dependent_payload_id uuid not null,
    primary key(owner_id,source_payload_id,dependent_payload_id),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    check(source_payload_id<>dependent_payload_id)
);
create index jarvis_context_dependency_child_idx on jarvis_context_payload_dependencies(owner_id,dependent_payload_id);
create trigger jarvis_context_dependencies_immutable before update on jarvis_context_payload_dependencies
for each row execute function reject_jarvis_foundation_update();

alter table jarvis_messages add column source_payload_id uuid;
alter table jarvis_messages add constraint jarvis_message_source_scope
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id);
create index jarvis_messages_source_payload_idx on jarvis_messages(owner_id,source_payload_id)
    where source_payload_id is not null;

-- Bind only exact, owned historical references. No text matching or inference.
update jarvis_messages m set source_payload_id=i.source_payload_id
from jarvis_attention_intents i
where m.owner_id=i.owner_id and m.metadata->>'response_kind'='expert_tts'
    and m.metadata->>'intent_id'=i.id::text;

insert into jarvis_context_payload_dependencies(owner_id,source_payload_id,dependent_payload_id)
select distinct p.owner_id,i.source_payload_id,p.id
from jarvis_context_payloads p
join jarvis_context_events e on e.owner_id=p.owner_id and e.payload_id=p.id
join jarvis_attention_attempts a on a.owner_id=p.owner_id and a.id::text=p.content->>'attempt_id'
join jarvis_attention_intents i on i.owner_id=a.owner_id and i.id=a.intent_id
join jarvis_context_payloads original on original.owner_id=i.owner_id and original.id=i.source_payload_id
where e.source_namespace like 'voice:%' and e.kind='assistant_output'
    and p.content->>'authored_by'='frontier_expert'
    and (original.created_at,original.id)<(p.created_at,p.id)
on conflict do nothing;

do $$
begin
    if exists(select 1 from jarvis_messages where metadata->>'response_kind'='expert_tts'
              and source_payload_id is null) then
        raise exception 'Unmapped expert history requires source inventory before migration' using errcode='23514';
    end if;
    if exists(select 1 from jarvis_context_payloads p join jarvis_context_events e
              on e.owner_id=p.owner_id and e.payload_id=p.id
              where e.source_namespace like 'voice:%' and e.kind='assistant_output'
                and p.content->>'authored_by'='frontier_expert'
                and not exists(select 1 from jarvis_context_payload_dependencies d
                    where d.owner_id=p.owner_id and d.dependent_payload_id=p.id)) then
        raise exception 'Unmapped expert context requires source inventory before migration' using errcode='23514';
    end if;
end $$;
alter table jarvis_messages add constraint jarvis_expert_message_requires_source
    check(metadata->>'response_kind' is distinct from 'expert_tts' or source_payload_id is not null);

create function lock_jarvis_copy_source(p_owner text,p_source uuid) returns text
language plpgsql set search_path from current as $$
declare source_state text; intent record; has_intent boolean:=false; has_current boolean:=false;
begin
    select availability into source_state from jarvis_context_payloads
    where owner_id=p_owner and id=p_source for share;
    if not found then return 'not_found'; end if;
    if source_state<>'available' then return 'removed'; end if;
    -- Match revocation ordering and prevent a late copy after an intent was
    -- superseded while its original remains retained as historical evidence.
    for intent in select state from jarvis_attention_intents
        where owner_id=p_owner and source_payload_id=p_source order by id for share loop
        has_intent:=true;
        has_current:=has_current or intent.state<>'revoked';
    end loop;
    if has_intent and not has_current then return 'revoked'; end if;
    return 'available';
end $$;

create function guard_jarvis_context_dependency() returns trigger
language plpgsql set search_path from current as $$
begin
    if lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        raise exception 'A copy requires an available owned source' using errcode='23514';
    end if;
    -- Immutable database creation ordering makes this a DAG, including the
    -- rare equal-timestamp case. New materializations depend on older originals.
    if not exists(select 1 from jarvis_context_payloads original
        join jarvis_context_payloads copy on copy.owner_id=original.owner_id
        where original.owner_id=new.owner_id and original.id=new.source_payload_id
            and copy.id=new.dependent_payload_id and copy.availability in ('available','not_finalized')
            and (original.created_at,original.id)<(copy.created_at,copy.id)) then
        raise exception 'A dependency must point from an older original to a new owned copy' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_dependency_guard before insert on jarvis_context_payload_dependencies
for each row execute function guard_jarvis_context_dependency();

create function guard_jarvis_message_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and old.source_payload_id is not null then
        if new.source_payload_id is distinct from old.source_payload_id or new.owner_id<>old.owner_id then
            raise exception 'A source-bound message cannot detach or change its source' using errcode='23514';
        end if;
        if new.content is distinct from old.content and new.content<>'' then
            raise exception 'A source-bound message can only redact its copied text' using errcode='23514';
        end if;
        if new.content='' then return new; end if;
    end if;
    if new.source_payload_id is not null
       and lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        raise exception 'A message copy requires an available owned source' using errcode='23514';
    end if;
    if new.metadata->>'response_kind'='expert_tts' and not exists(
        select 1 from jarvis_attention_intents i where i.owner_id=new.owner_id
            and i.id::text=new.metadata->>'intent_id' and i.source_payload_id=new.source_payload_id) then
        raise exception 'Expert message provenance must match its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_message_source_guard before insert or update of source_payload_id,owner_id,content,metadata on jarvis_messages
for each row execute function guard_jarvis_message_source();

create function guard_jarvis_expert_context_source() returns trigger
language plpgsql set search_path from current as $$
declare payload jsonb;
begin
    if new.kind<>'assistant_output' or new.source_namespace not like 'voice:%' then return new; end if;
    select content into payload from jarvis_context_payloads where owner_id=new.owner_id and id=new.payload_id;
    if payload->>'authored_by'='frontier_expert' and not exists(
        select 1 from jarvis_context_payload_dependencies d
        join jarvis_attention_attempts a on a.owner_id=d.owner_id and a.id::text=payload->>'attempt_id'
        join jarvis_attention_intents i on i.owner_id=a.owner_id and i.id=a.intent_id
        where d.owner_id=new.owner_id and d.dependent_payload_id=new.payload_id
            and d.source_payload_id=i.source_payload_id) then
        raise exception 'Expert context requires exact owned source lineage' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_expert_context_source_guard before insert on jarvis_context_events
for each row execute function guard_jarvis_expert_context_source();

create function revoke_jarvis_source_bound_copies(p_owner text,p_source uuid,p_availability text) returns void
language plpgsql set search_path from current as $$
begin
    if p_availability not in ('deleted','expired','unavailable') then
        raise exception 'Invalid source-copy removal state' using errcode='23514';
    end if;
    update jarvis_messages set content='',metadata=jsonb_build_object(
        'response_kind','source_bound_unavailable','source_status',p_availability,'source_payload_id',p_source)
    where owner_id=p_owner and source_payload_id=p_source;
    -- Existing payload-removal triggers invalidate task manifests, source
    -- mappings and Hindsight outbox records. Each descendant follows this path.
    update jarvis_context_payloads p set availability=p_availability,content=null
    where p.owner_id=p_owner and exists(select 1 from jarvis_context_payload_dependencies d
        where d.owner_id=p.owner_id and d.source_payload_id=p_source and d.dependent_payload_id=p.id)
        and (p.availability in ('available','not_finalized')
             or (p_availability='deleted' and p.availability<>'deleted'));
end $$;

create function revoke_jarvis_payload_copies() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
       and (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        perform revoke_jarvis_source_bound_copies(new.owner_id,new.id,new.availability);
    end if;
    return new;
end $$;
create trigger jarvis_source_bound_payload_removal after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_payload_copies();

create function revoke_jarvis_attention_copies() returns trigger
language plpgsql set search_path from current as $$
declare removal_state text;
begin
    if new.state='revoked' and old.state is distinct from new.state then
        select case when availability in ('deleted','expired','unavailable') then availability
                    else 'unavailable' end into removal_state
        from jarvis_context_payloads where owner_id=new.owner_id and id=new.source_payload_id;
        perform revoke_jarvis_source_bound_copies(new.owner_id,new.source_payload_id,removal_state);
    end if;
    return new;
end $$;
create trigger jarvis_attention_source_copy_revocation after update of state on jarvis_attention_intents
for each row execute function revoke_jarvis_attention_copies();

create function propagate_jarvis_context_manifest_removal() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.status<>'active' and old.status is distinct from new.status then
        -- Reuse the exact result/attempt lineage already used by memory-source
        -- invalidation. Corrections retain originals; removals purge originals.
        perform invalidate_jarvis_task_memory_manifests(new.owner_id,array[new.id],new.status='revoked');
    end if;
    return new;
end $$;
create trigger jarvis_context_manifest_result_invalidation after update of status on jarvis_task_context_manifests
for each row execute function propagate_jarvis_context_manifest_removal();

-- Retire already-invalid copies only after all exact links exist. Any unmapped
-- historical expert data above aborts the migration instead of guessing.
do $$
declare source record; source_state text; availability text;
begin
    for source in select owner_id,source_payload_id from jarvis_context_payload_dependencies
                  union select owner_id,source_payload_id from jarvis_messages where source_payload_id is not null loop
        source_state:=lock_jarvis_copy_source(source.owner_id,source.source_payload_id);
        if source_state<>'available' then
            select case when p.availability in ('deleted','expired','unavailable') then p.availability
                        else 'unavailable' end into availability from jarvis_context_payloads p
            where p.owner_id=source.owner_id and p.id=source.source_payload_id;
            perform revoke_jarvis_source_bound_copies(source.owner_id,source.source_payload_id,availability);
        end if;
    end loop;
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (10, 'source_bound_outputs', '137f74351c4bfef7802caaf1703d7f08fdd30c42e3e2be53ea05c59c1eb25e42');

-- Batch the existing copy-source fence without per-source network round trips.
-- Creation ordering matches the dependency DAG, including ancestor/copy pairs.
create function lock_jarvis_copy_sources(p_owner text,p_sources uuid[]) returns boolean
language plpgsql set search_path from current as $$
declare source record; locked integer:=0;
begin
    if p_sources is null or cardinality(p_sources)=0 then return false; end if;
    for source in select id from jarvis_context_payloads
        where owner_id=p_owner and id=any(p_sources) order by created_at,id loop
        if lock_jarvis_copy_source(p_owner,source.id)<>'available' then return false; end if;
        locked:=locked+1;
    end loop;
    -- Missing, foreign, repeated and NULL references are not an authorized set.
    return locked=cardinality(p_sources);
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (11, 'live_response_source_fence', '261e3478a17296c3816ac0451502158da97a7a043685223135e3d03f40b2c581');

-- Selected-memory read materializations. Originals are not replaced or duplicated
-- in generic execution logs; current claims and explicit historical reads differ.
create table jarvis_context_memory_sources (
    owner_id text not null,
    dependent_payload_id uuid not null,
    area_id uuid not null,
    source_id uuid not null,
    source_revision bigint not null,
    source_generation bigint not null check(source_generation>0),
    source_payload_id uuid not null,
    content_digest text not null check(content_digest ~ '^[0-9a-f]{64}$'),
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,dependent_payload_id,source_id,source_revision),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_context_memory_source_idx on jarvis_context_memory_sources(owner_id,source_id);
create index jarvis_context_memory_area_idx on jarvis_context_memory_sources(owner_id,area_id);
create trigger jarvis_context_memory_sources_immutable before update on jarvis_context_memory_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_context_memory_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_knowledge_source_versions v
        join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
        join jarvis_knowledge_areas a on a.owner_id=v.owner_id and a.id=v.area_id
        join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
        join jarvis_context_payload_dependencies d on d.owner_id=p.owner_id and d.source_payload_id=p.id
        join jarvis_context_payloads dependent on dependent.owner_id=d.owner_id and dependent.id=d.dependent_payload_id
        left join jarvis_knowledge_context_sources cs on cs.owner_id=v.owner_id
            and cs.source_id=v.source_id and cs.source_revision=v.revision
        left join jarvis_context_events ce on ce.owner_id=cs.owner_id and ce.id=cs.event_id
        where v.owner_id=new.owner_id and v.area_id=new.area_id and v.source_id=new.source_id
            and v.revision=new.source_revision and s.generation=new.source_generation
            and v.payload_id=new.source_payload_id and v.content_digest=new.content_digest
            and p.availability='available' and s.status='active' and a.status='active'
            and d.dependent_payload_id=new.dependent_payload_id and dependent.availability='available'
            and ((new.source_usage='historical' and (s.current_revision>v.revision or exists(
                select 1 from jarvis_context_events newer where ce.id is not null
                    and newer.owner_id=ce.owner_id and newer.source_namespace=ce.source_namespace
                    and newer.source_event_id=ce.source_event_id and newer.revision>ce.revision))) or (
                new.source_usage='current' and
                s.current_revision=v.revision and not exists(
                    select 1 from jarvis_context_events newer where ce.id is not null
                        and newer.owner_id=ce.owner_id and newer.source_namespace=ce.source_namespace
                        and newer.source_event_id=ce.source_event_id and newer.revision>ce.revision)))
    ) then
        raise exception 'Context memory evidence must match its owned original and dependency' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_source_guard before insert on jarvis_context_memory_sources
for each row execute function guard_jarvis_context_memory_source();

create function invalidate_jarvis_context_memory_revision() returns trigger
language plpgsql set search_path from current as $$
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status
       or old.current_revision is distinct from new.current_revision then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r where r.owner_id=p.owner_id
                and r.dependent_payload_id=p.id and r.source_id=new.id
                and (new.status<>'active' or (r.source_usage='current'
                    and (r.source_generation<>new.generation or r.source_revision<>new.current_revision))));
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_revision after update on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_context_memory_revision();

create function invalidate_jarvis_context_memory_area() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.status<>'active' and old.status is distinct from new.status then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r where r.owner_id=p.owner_id
                and r.dependent_payload_id=p.id and r.area_id=new.id);
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_area after update on jarvis_knowledge_areas
for each row execute function invalidate_jarvis_context_memory_area();

create function invalidate_jarvis_context_memory_event() returns trigger
language plpgsql set search_path from current as $$
begin
    -- Canonical correction wins even before asynchronous publication catches up.
    if new.revision>1 then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_memory_sources r
            join jarvis_knowledge_context_sources cs on cs.owner_id=r.owner_id
                and cs.source_id=r.source_id and cs.source_revision=r.source_revision
            join jarvis_context_events prior on prior.owner_id=cs.owner_id and prior.id=cs.event_id
            where r.owner_id=p.owner_id and r.dependent_payload_id=p.id and r.source_usage='current'
                and prior.source_namespace=new.source_namespace and prior.source_event_id=new.source_event_id
                and prior.revision<new.revision);
    end if;
    return new;
end $$;
create trigger jarvis_context_memory_event after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_memory_event();
insert into jarvis_schema_migrations (version, name, checksum) values (12, 'context_memory_sources', 'acc660d2e041fb33cb4c9393880a5a5d74c40d0ed4c8a17c1fc349e9acb3d0a3');

-- Source lifecycle recovery metadata, not another personal memory engine.
-- Backups must include this journal epoch. No pre-migration history is invented.
create table jarvis_source_journal_streams (
    owner_id text primary key check(length(owner_id) between 1 and 200 and owner_id=btrim(owner_id)),
    epoch uuid not null default gen_random_uuid(),
    last_sequence bigint not null default 0 check(last_sequence>=0),
    initialized_at timestamptz not null default clock_timestamp(),
    unique(owner_id,epoch)
);

create table jarvis_source_journal_events (
    owner_id text not null references jarvis_source_journal_streams(owner_id),
    sequence bigint not null check(sequence>0),
    id uuid not null default gen_random_uuid(),
    kind text not null check(kind in ('source','payload','area','cleanup')),
    object_id uuid not null,
    detail jsonb not null check(jsonb_typeof(detail)='object' and octet_length(detail::text)<=8192),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,sequence),
    unique(owner_id,id)
);
create trigger jarvis_source_journal_immutable before update or delete on jarvis_source_journal_events
for each row execute function reject_jarvis_foundation_update();

create function append_jarvis_source_journal(p_owner text,p_kind text,p_object uuid,p_detail jsonb)
returns void language plpgsql set search_path from current as $$
declare next_sequence bigint;
begin
    -- This row lock serializes committed owner order. Rollback also rolls back
    -- the counter; a cursor cannot skip a still-uncommitted lower sequence.
    insert into jarvis_source_journal_streams(owner_id,last_sequence) values(p_owner,1)
    on conflict(owner_id) do update
        set last_sequence=jarvis_source_journal_streams.last_sequence+1
    returning last_sequence into next_sequence;
    insert into jarvis_source_journal_events(owner_id,sequence,kind,object_id,detail)
        values(p_owner,next_sequence,p_kind,p_object,p_detail);
end; $$;

create function record_jarvis_source_lifecycle() returns trigger
language plpgsql set search_path from current as $$
declare detail jsonb;
begin
    -- Hash the stable external identity instead of copying a private path/name.
    detail:=jsonb_build_object(
        'area_id',new.area_id,'revision',new.current_revision,'generation',new.generation,'status',new.status,
        'identity_digest',encode(sha256(convert_to(
            jsonb_build_array(new.owner_id,new.area_id,new.namespace,new.source_key)::text,'UTF8')),'hex'));
    perform append_jarvis_source_journal(new.owner_id,'source',new.id,detail);
    return new;
end; $$;
create trigger jarvis_source_lifecycle_journal after insert or update on jarvis_knowledge_sources
for each row execute function record_jarvis_source_lifecycle();

create function record_jarvis_payload_lifecycle() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and new.availability is not distinct from old.availability
        and new.content_digest is not distinct from old.content_digest then return new; end if;
    perform append_jarvis_source_journal(new.owner_id,'payload',new.id,jsonb_build_object(
        'availability',new.availability,'content_digest',new.content_digest,
        'prior_content_digest',case when tg_op='UPDATE' then old.content_digest else null end));
    return new;
end; $$;
create trigger jarvis_payload_lifecycle_journal after insert or update on jarvis_context_payloads
for each row execute function record_jarvis_payload_lifecycle();

create function record_jarvis_area_lifecycle() returns trigger
language plpgsql set search_path from current as $$
begin
    perform append_jarvis_source_journal(new.owner_id,'area',new.id,
        jsonb_build_object('status',new.status,'revision',new.revision));
    return new;
end; $$;
create trigger jarvis_area_lifecycle_journal after insert or update on jarvis_knowledge_areas
for each row execute function record_jarvis_area_lifecycle();

create function record_jarvis_cleanup_lifecycle() returns trigger
language plpgsql set search_path from current as $$
declare detail jsonb;
begin
    if new.operation<>'delete' then return new; end if;
    if tg_op='UPDATE' and not (
        (new.status in ('done','failed','cancelled','outcome_unknown') and new.status is distinct from old.status)
        or (old.status in ('done','failed','cancelled') and new.status is distinct from old.status)
        or new.generation is distinct from old.generation
        or new.cleanup_operation_ids is distinct from old.cleanup_operation_ids
    ) then return new; end if;
    -- Preserve the deletion intent and terminal operation inventory. Do not
    -- copy provider errors, credentials, source text or every lease/poll update.
    select jsonb_build_object(
        'source_id',v.source_id,'source_revision',v.revision,'document_id',v.document_id,
        'bank_name',b.bank_name,'generation',new.generation,'status',new.status,
        'native_operation_ids',new.cleanup_operation_ids,
        'retain_operation_id',r.id,'retain_dispatch_count',coalesce(r.dispatch_count,0))
    into detail from jarvis_knowledge_source_versions v
    join jarvis_memory_banks b on b.owner_id=v.owner_id and b.id=v.bank_id
    left join jarvis_memory_outbox r on r.owner_id=v.owner_id and r.document_id=v.document_id
        and r.operation='retain'
    where v.owner_id=new.owner_id and v.document_id=new.document_id;
    perform append_jarvis_source_journal(new.owner_id,'cleanup',new.id,detail);
    return new;
end; $$;
create trigger jarvis_cleanup_lifecycle_journal after insert or update on jarvis_memory_outbox
for each row execute function record_jarvis_cleanup_lifecycle();

-- Establish an epoch for existing owners without manufacturing historical
-- events or scanning/copying their source contents.
insert into jarvis_source_journal_streams(owner_id)
select owner_id from jarvis_knowledge_sources
union select owner_id from jarvis_context_payloads
union select owner_id from jarvis_knowledge_areas
on conflict do nothing;
insert into jarvis_schema_migrations (version, name, checksum) values (13, 'source_lifecycle_journal', '50a45688dae61734382df539e69edbd47e69fc8485559055943cbbf49f13f190');

-- Receipt fingerprints are action identity metadata, not retained receipt text.
-- Existing rows remain nullable: never invent a fingerprint for removed content.
alter table jarvis_task_actions add column receipt_digest text
    check(receipt_digest ~ '^[0-9a-f]{64}$');
insert into jarvis_schema_migrations (version, name, checksum) values (14, 'action_receipt_fingerprint', '2977d6de70a90ce6c07aff3355dc049145da9de5f8f174fc7de9abc05a1a280c');

-- Logical forgetting suppresses source-backed recall without erasing originals.
alter table jarvis_knowledge_sources drop constraint jarvis_knowledge_sources_status_check;
alter table jarvis_knowledge_sources add constraint jarvis_knowledge_sources_status_check
    check(status in ('active','unavailable','forgotten','deleted'));

create or replace function guard_jarvis_source_update() returns trigger language plpgsql as $$
begin
    -- Upstream invalidation must not remove a user's suppression marker.
    -- Physical deletion remains allowed; no automatic unforget is supported.
    if old.status='forgotten' then
        if new.status='unavailable' then new.status:='forgotten'; end if;
        if new.status not in ('forgotten','deleted') then
            raise exception 'Forgotten sources cannot be automatically republished' using errcode='23514';
        end if;
    end if;
    if (to_jsonb(new)-array['current_revision','generation','status','updated_at'])
        <> (to_jsonb(old)-array['current_revision','generation','status','updated_at'])
        or old.status='deleted' or new.generation<>old.generation+1
        or (new.status='active' and new.current_revision<>old.current_revision+1)
        or (new.status<>'active' and new.current_revision<>old.current_revision) then
        raise exception 'Source updates require immutable identity and the next generation' using errcode='23514';
    end if;
    new.updated_at:=clock_timestamp();
    return new;
end; $$;
insert into jarvis_schema_migrations (version, name, checksum) values (15, 'logical_source_forgetting', '4d493e695c5d8424195310d195a9bdf96617aca46bea91fad76b8337bf833fe0');

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
insert into jarvis_schema_migrations (version, name, checksum) values (16, 'legacy_source_import', '7ddb1357a74f72dab2dc0a5e2aa9be973985511c1653747eb350a460ac1bc108');

-- Trusted per-version session association. No evidence-body or vendor-document rewrite.
create table jarvis_knowledge_source_sessions (
    owner_id text not null,
    source_id uuid not null,
    source_revision bigint not null check(source_revision>0),
    session_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,source_revision),
    foreign key(owner_id,source_id,source_revision)
        references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id) on delete cascade
);
create index jarvis_source_sessions_by_session
    on jarvis_knowledge_source_sessions(owner_id,session_id,source_id,source_revision);
create trigger jarvis_source_session_immutable before update on jarvis_knowledge_source_sessions
for each row execute function reject_jarvis_foundation_update();

-- Existing canonical associations were already admitted by the context boundary.
insert into jarvis_knowledge_source_sessions(owner_id,source_id,source_revision,session_id)
select cs.owner_id,cs.source_id,cs.source_revision,e.session_id
from jarvis_knowledge_context_sources cs
join jarvis_context_events e on e.owner_id=cs.owner_id and e.id=cs.event_id and e.revision=cs.event_revision
join jarvis_sessions j on j.owner_id=e.owner_id and j.id=e.session_id
on conflict(owner_id,source_id,source_revision) do nothing;

-- Only persisted importer receipts establish this provenance. Do not infer
-- session ownership from arbitrary model text, a vendor result or a changed legacy row.
insert into jarvis_knowledge_source_sessions(owner_id,source_id,source_revision,session_id)
select distinct r.owner_id,r.source_id,r.source_revision,j.id
from jarvis_legacy_import_records r
join jarvis_legacy_import_runs run on run.owner_id=r.owner_id and run.id=r.run_id
join jarvis_knowledge_source_versions v
    on v.owner_id=r.owner_id and v.source_id=r.source_id and v.revision=r.source_revision
join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id and p.availability='available'
join jarvis_sessions j on j.owner_id=r.owner_id and j.id::text=p.content#>>'{record,session_id}'
where r.disposition in ('imported','already_present','suppressed')
    and p.content#>>'{origin,system}'='jarvis_legacy'
    and p.content#>>'{origin,table}'=r.population
    and p.content#>>'{origin,key}'=r.origin_key
    and p.content#>>'{origin,mapping_version}'=run.mapping_version::text
on conflict(owner_id,source_id,source_revision) do nothing;
insert into jarvis_schema_migrations (version, name, checksum) values (17, 'source_session_association', 'f3a3fd1ab46e02db76208e5a429180f111022635628fd8c4d39c5a5f329ec2c0');

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
insert into jarvis_schema_migrations (version, name, checksum) values (18, 'file_publication', '5d4bfcc057310ade6e76332fb58b6eb7536221bae31344ff739384feabaad233');

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
insert into jarvis_schema_migrations (version, name, checksum) values (19, 'file_source_originals', '2b35f3e1a859aecfb6d148845b9fdfdf7ecefcb74b10aa7a1360be764d6abd97');

-- Bind ongoing analyzer originals to canonical sources without changing the
-- frozen M03 legacy row shapes. Existing observations are not implicitly imported.
create table jarvis_observation_sources (
    owner_id text not null,
    observation_id uuid not null,
    session_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,observation_id),
    unique(owner_id,payload_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
-- Keep the identity after removing the disposable operational original, so a
-- delayed analyzer replay cannot recreate it under the same observation ID.
create trigger jarvis_observation_sources_immutable before update on jarvis_observation_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_observation_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from perception_observations o
        join jarvis_context_events e on e.owner_id=o.owner_id and e.session_id=o.session_id
            and e.source_namespace='perception:analyzer:v1' and e.source_event_id=o.id::text
        join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
        where o.owner_id=new.owner_id and o.id=new.observation_id and o.session_id=new.session_id
            and p.id=new.payload_id and p.availability='available'
            and e.kind='observation' and e.trust='derived' and e.finality='final'
            and p.content->>'origin'='perception_observations' and p.content->'record'=to_jsonb(o)
    ) then
        raise exception 'Observation binding must match the owned canonical original' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_observation_source_guard before insert on jarvis_observation_sources
for each row execute function guard_jarvis_observation_source();

create function guard_jarvis_observation_replay() returns trigger
language plpgsql set search_path from current as $$
begin
    if exists(select 1 from jarvis_observation_sources
              where owner_id=new.owner_id and observation_id=new.id) then
        raise exception 'A published observation identity cannot be recreated' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_observation_replay_guard before insert on perception_observations
for each row execute function guard_jarvis_observation_replay();

create function invalidate_jarvis_observation_original(p_owner text,p_payload uuid) returns void
language plpgsql set search_path from current as $$
declare binding record;
begin
    for binding in select * from jarvis_observation_sources
        where owner_id=p_owner and payload_id=p_payload
    loop
        -- A reducer may have combined older observations into the latest cache.
        -- Invalidate the session cache, not just its last-observation pointer.
        -- Advance its revision so an in-flight stale CAS cannot restore it.
        update jarvis_session_working_memory set state='{}'::jsonb,revision=revision+1,
            last_observation_id=null,last_observation_ended_at=null,
            expires_at=clock_timestamp(),updated_at=clock_timestamp()
        where owner_id=p_owner and session_id=binding.session_id;
        update visual_frames f set frame_description=null,memory_tags='{}'::text[],
            updated_at=clock_timestamp()
        where f.owner_id=p_owner and exists(select 1 from perception_observation_frames l
            where l.observation_id=binding.observation_id and l.visual_frame_id=f.id);
        -- These are operational aggregates, not independent original evidence.
        -- Keep other evidence rows and timing; force subsequent aggregation to
        -- start from empty state using the existing updated_at conflict check.
        update experience_events e set title='',summary='',scene=null,activity=null,state='{}'::jsonb,
            consolidation_result=case when consolidation_result is null then null else '{}'::jsonb end,
            updated_at=clock_timestamp()
        where e.owner_id=p_owner and exists(select 1 from experience_event_evidence evidence
            where evidence.event_id=e.id and evidence.evidence_type='observation'
                and evidence.evidence_ref_id=binding.observation_id);
        delete from experience_event_evidence evidence using experience_events e
        where e.id=evidence.event_id and e.owner_id=p_owner
            and evidence.evidence_type='observation' and evidence.evidence_ref_id=binding.observation_id;
        delete from perception_observations where owner_id=p_owner and id=binding.observation_id;
    end loop;
end $$;

create function invalidate_jarvis_observation_payload() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
        and old.availability is distinct from new.availability then
        perform invalidate_jarvis_observation_original(new.owner_id,new.id);
    end if;
    return new;
end $$;
create trigger jarvis_observation_payload_invalidation after update of availability on jarvis_context_payloads
for each row execute function invalidate_jarvis_observation_payload();

create function invalidate_jarvis_observation_source_revision() returns trigger
language plpgsql set search_path from current as $$
declare original record;
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        for original in select v.payload_id from jarvis_knowledge_source_versions v
            join jarvis_observation_sources b on b.owner_id=v.owner_id and b.payload_id=v.payload_id
            where v.owner_id=new.owner_id and v.source_id=new.id
        loop
            perform invalidate_jarvis_observation_original(new.owner_id,original.payload_id);
        end loop;
    end if;
    return new;
end $$;
create trigger jarvis_observation_source_invalidation after update of generation,status on jarvis_knowledge_sources
for each row execute function invalidate_jarvis_observation_source_revision();
insert into jarvis_schema_migrations (version, name, checksum) values (20, 'observation_source_publication', '5fd488b7076c1d3b4b13fe405e68a7ac806809e577fa26c30cd43b20a03e9242');

-- Removable canonical-input copies also work while memory publication is paused.
-- This is a provenance fence, not an additional memory engine or model loop.
create table jarvis_context_event_dependencies (
    owner_id text not null,
    dependent_payload_id uuid not null,
    event_id uuid not null,
    event_revision bigint not null check(event_revision>0),
    source_payload_id uuid not null,
    source_usage text not null check(source_usage in ('current','historical')),
    primary key(owner_id,dependent_payload_id,event_id),
    foreign key(owner_id,dependent_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,source_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,event_id) references jarvis_context_events(owner_id,id)
);
create index jarvis_context_event_dependency_event_idx on jarvis_context_event_dependencies(owner_id,event_id);
create trigger jarvis_context_event_dependencies_immutable before update on jarvis_context_event_dependencies
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_context_event_dependency() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(
        select 1 from jarvis_context_events e
        join jarvis_context_payloads original on original.owner_id=e.owner_id and original.id=e.payload_id
        join jarvis_context_payload_dependencies d on d.owner_id=e.owner_id and d.source_payload_id=e.payload_id
        join jarvis_context_payloads copy on copy.owner_id=d.owner_id and copy.id=d.dependent_payload_id
        where e.owner_id=new.owner_id and e.id=new.event_id and e.revision=new.event_revision
            and e.payload_id=new.source_payload_id and copy.id=new.dependent_payload_id
            and original.availability='available' and copy.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        raise exception 'Canonical input dependency must match the available current original' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_context_event_dependency_guard before insert on jarvis_context_event_dependencies
for each row execute function guard_jarvis_context_event_dependency();

create function invalidate_jarvis_context_event_dependencies() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.revision>1 then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=new.owner_id and p.availability in ('available','not_finalized') and exists(
            select 1 from jarvis_context_event_dependencies dependency
            join jarvis_context_events original on original.owner_id=dependency.owner_id
                and original.id=dependency.event_id
            where dependency.owner_id=p.owner_id and dependency.dependent_payload_id=p.id
                and dependency.source_usage='current'
                and original.source_namespace=new.source_namespace
                and original.source_event_id=new.source_event_id and original.revision<new.revision
        );
    end if;
    return new;
end $$;
create trigger jarvis_context_event_dependency_revision after insert on jarvis_context_events
for each row execute function invalidate_jarvis_context_event_dependencies();
insert into jarvis_schema_migrations (version, name, checksum) values (21, 'canonical_input_dependencies', '1cb8727f7c8042ed2cfe1a95676ae8143c4ca039b481fec51dd0b51cc5240321');

-- Keep M03's frozen legacy row shapes. These exact links are replay tombstones,
-- not another notification queue or memory store.
create table jarvis_visual_delivery_sources (
    owner_id text not null,
    delivery_kind text not null check(delivery_kind in ('visual','prospective')),
    delivery_id uuid not null,
    session_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,delivery_kind,delivery_id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_visual_delivery_payload_idx on jarvis_visual_delivery_sources(owner_id,payload_id);
create trigger jarvis_visual_delivery_sources_immutable before update on jarvis_visual_delivery_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_visual_delivery_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not lock_jarvis_copy_sources(new.owner_id,array[new.payload_id]) or not exists(
        select 1 from jarvis_observation_sources
        where owner_id=new.owner_id and session_id=new.session_id and payload_id=new.payload_id
    ) then
        raise exception 'Visual delivery requires its current owned observation' using errcode='23514';
    end if;
    if new.delivery_kind='visual' then
        if not exists(select 1 from visual_interrupts
            where owner_id=new.owner_id and id=new.delivery_id and session_id=new.session_id) then
            raise exception 'Visual delivery identity does not match its owned record' using errcode='23514';
        end if;
    else
        if not exists(select 1 from jarvis_prospective_delivery_events
            where owner_id=new.owner_id and id=new.delivery_id and session_id=new.session_id) then
            raise exception 'Prospective delivery identity does not match its owned record' using errcode='23514';
        end if;
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_source_guard before insert on jarvis_visual_delivery_sources
for each row execute function guard_jarvis_visual_delivery_source();

create function revoke_jarvis_visual_delivery_copies(p_owner text,p_payload uuid) returns void
language plpgsql set search_path from current as $$
declare event record;
begin
    update visual_interrupts v set message='',reason='source_unavailable',confidence=0,dedupe_bucket=''
    where v.owner_id=p_owner and exists(select 1 from jarvis_visual_delivery_sources b
        where b.owner_id=p_owner and b.delivery_kind='visual'
            and b.delivery_id=v.id and b.payload_id=p_payload);
    for event in select e.* from jarvis_prospective_delivery_events e
        join jarvis_visual_delivery_sources b on b.owner_id=e.owner_id
            and b.delivery_kind='prospective' and b.delivery_id=e.id
        where b.owner_id=p_owner and b.payload_id=p_payload order by e.id for update of e
    loop
        update jarvis_prospective_delivery_events set
            message='[source unavailable]',evidence='{}'::jsonb,
            event_type=case when event_type in ('generated','dispatched_to_session') then 'failed' else event_type end,
            metadata=jsonb_build_object('reason','source_unavailable',
                'source_payload_id',p_payload,'prior_event_type',event.event_type,
                'playback_status',coalesce(metadata->>'playback_status','unconfirmed')),
            updated_at=clock_timestamp()
        where owner_id=p_owner and id=event.id;
        if event.event_type in ('generated','dispatched_to_session') then
            update jarvis_visual_focuses set trigger_claim_token=null,trigger_claimed_at=null
            where owner_id=p_owner and id=event.visual_focus_id and trigger_claim_token=event.claim_token;
            update jarvis_reminders set trigger_claim_token=null,trigger_claimed_at=null,
                last_trigger_error='source_unavailable',updated_at=clock_timestamp()
            where owner_id=p_owner and id=event.reminder_id and trigger_claim_token=event.claim_token;
        end if;
    end loop;
    -- A historical find-once delivery is not undone or replayed. Its discarded
    -- evidence cannot continue masquerading as a current matching observation.
    update jarvis_visual_focuses set completion_evidence=jsonb_build_object(
        'source_payload_id',p_payload,'status','source_unavailable',
        'meaning','Past search notification bookkeeping, not current match evidence')
    where owner_id=p_owner and completion_evidence->>'source_payload_id'=p_payload::text;
end $$;

create function revoke_jarvis_visual_delivery_payload() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable')
        and (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        perform revoke_jarvis_visual_delivery_copies(new.owner_id,new.id);
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_payload_revocation after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_visual_delivery_payload();

create function revoke_jarvis_visual_delivery_revision() returns trigger
language plpgsql set search_path from current as $$
declare original record;
begin
    if old.generation is distinct from new.generation or old.status is distinct from new.status then
        for original in select distinct v.payload_id from jarvis_knowledge_source_versions v
            join jarvis_visual_delivery_sources b on b.owner_id=v.owner_id and b.payload_id=v.payload_id
            where v.owner_id=new.owner_id and v.source_id=new.id
        loop
            perform revoke_jarvis_visual_delivery_copies(new.owner_id,original.payload_id);
        end loop;
    end if;
    return new;
end $$;
create trigger jarvis_visual_delivery_source_revocation after update of generation,status on jarvis_knowledge_sources
for each row execute function revoke_jarvis_visual_delivery_revision();
insert into jarvis_schema_migrations (version, name, checksum) values (22, 'visual_delivery_sources', 'dfd68da3ad833b797087f35272b7ae2d7109fe00c7ff341233a94feccef0df05');

-- Scheduled notifications share the existing inbox/attempt/PCM ledger.
-- Exact bindings are revocation tombstones, not another queue or memory engine.
create table jarvis_reminder_attention_sources (
    owner_id text not null,
    reminder_id uuid not null,
    event_id uuid not null,
    payload_id uuid not null,
    primary key(owner_id,event_id),
    unique(owner_id,payload_id),
    foreign key(owner_id,payload_id) references jarvis_context_payloads(owner_id,id)
);
create index jarvis_reminder_attention_subject_idx on jarvis_reminder_attention_sources(owner_id,reminder_id);
create trigger jarvis_reminder_attention_identity_immutable before update on jarvis_reminder_attention_sources
for each row execute function reject_jarvis_foundation_update();

create function guard_jarvis_reminder_attention_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if not exists(select 1 from jarvis_reminders r
        join jarvis_reminder_events e on e.owner_id=r.owner_id and e.reminder_id=r.id
        join jarvis_context_payloads p on p.owner_id=r.owner_id and p.id=new.payload_id
        where r.owner_id=new.owner_id and r.id=new.reminder_id and r.status<>'deleted'
          and e.id=new.event_id and e.event_type='generated'
          and p.availability='available'
          and p.content->>'format'='jarvis.attention.notification.v1'
          and p.content->>'source_namespace'='reminder_event'
          and p.content->>'source_id'=e.id::text and p.content->>'reminder_id'=r.id::text
          and p.content->>'authored_by'='stored_reminder'
          and p.content->>'title'=r.title and p.content->>'details'=r.details
          and p.content->>'message'=e.message) then
        raise exception 'Reminder attention requires its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_reminder_attention_source_guard before insert on jarvis_reminder_attention_sources
for each row execute function guard_jarvis_reminder_attention_source();

create function revoke_jarvis_reminder_attention_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or
       (new.title,new.details,new.remind_at,new.condition) is distinct from
       (old.title,old.details,old.remind_at,old.condition) or new.status='deleted' or
       new.metadata->>'attention_cancelled_at' is distinct from old.metadata->>'attention_cancelled_at' then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_reminder_attention_sources b
              where b.owner_id=p.owner_id and b.payload_id=p.id and b.reminder_id=old.id);
    end if;
    return null;
end $$;
create trigger jarvis_reminder_attention_subject_changed
after update of title,details,remind_at,condition,status,metadata or delete on jarvis_reminders
for each row execute function revoke_jarvis_reminder_attention_source();

create function revoke_jarvis_reminder_attention_event() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or new.message is distinct from old.message then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_reminder_attention_sources b
              where b.owner_id=p.owner_id and b.payload_id=p.id and b.event_id=old.id);
    end if;
    return null;
end $$;
create trigger jarvis_reminder_attention_event_changed
after update of message or delete on jarvis_reminder_events
for each row execute function revoke_jarvis_reminder_attention_event();

create function revoke_jarvis_reminder_attention_copy() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.availability in ('deleted','expired','unavailable') and
       (old.availability is distinct from new.availability or old.content is distinct from new.content) then
        update jarvis_reminder_events e set message='[source unavailable]',
            metadata=jsonb_build_object('reason','source_unavailable',
                'playback_status','unconfirmed','physical_action_status','not_established')
        where e.owner_id=new.owner_id and exists(select 1 from jarvis_reminder_attention_sources b
            where b.owner_id=e.owner_id and b.event_id=e.id and b.payload_id=new.id);
    end if;
    return new;
end $$;
create trigger jarvis_reminder_attention_payload_changed
after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_reminder_attention_copy();

-- The original event records generation, not provider/client delivery. The
-- existing attention attempts are the sole delivery truth; do not project them
-- back while holding the source fence (that reverses source/event lock order).
-- No historical events are retroactively enqueued or replayed by installation.
insert into jarvis_schema_migrations (version, name, checksum) values (23, 'reminder_attention', 'ab56ab9ee67091d86a2fe3b3768e6ed347ad4974f80feb06d039bb96500e9497');

-- A narrow operational fence, not another search queue or memory store. It lets
-- phone writes serialize with subject changes without locking raw subject rows.
create table jarvis_attention_visual_subjects (
    owner_id text not null,
    subject_type text not null check(subject_type in ('visual_focus','conditional_reminder')),
    subject_id uuid not null,
    generation bigint not null default 1 check(generation>0),
    descriptor_digest text not null check(descriptor_digest ~ '^[0-9a-f]{64}$'),
    state text not null check(state in ('active','completed','cancelled')),
    expires_at timestamptz,
    claim_token uuid,
    completed_event_id text,
    updated_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,subject_type,subject_id)
);

create function sync_jarvis_attention_visual_subject(
    p_owner text,p_kind text,p_id uuid,p_descriptor jsonb,p_state text,
    p_expiry timestamptz,p_claim uuid,p_completed text) returns void
language plpgsql set search_path from current as $$
begin
    insert into jarvis_attention_visual_subjects
        (owner_id,subject_type,subject_id,descriptor_digest,state,expires_at,claim_token,completed_event_id)
    values(p_owner,p_kind,p_id,encode(sha256(convert_to(p_descriptor::text,'UTF8')),'hex'),
           p_state,p_expiry,p_claim,p_completed)
    on conflict(owner_id,subject_type,subject_id) do update set
        generation=jarvis_attention_visual_subjects.generation+
            case when jarvis_attention_visual_subjects.descriptor_digest<>excluded.descriptor_digest
                 or (excluded.state='cancelled' and jarvis_attention_visual_subjects.state<>'cancelled')
                 then 1 else 0 end,
        descriptor_digest=excluded.descriptor_digest,state=excluded.state,expires_at=excluded.expires_at,
        claim_token=excluded.claim_token,completed_event_id=excluded.completed_event_id,updated_at=clock_timestamp();
end $$;

create function update_jarvis_attention_visual_subject() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' then
        update jarvis_attention_visual_subjects set state='cancelled',generation=generation+1,
            claim_token=null,completed_event_id=null,updated_at=clock_timestamp()
        where owner_id=old.owner_id and subject_id=old.id
          and subject_type=case when tg_table_name='jarvis_visual_focuses' then 'visual_focus' else 'conditional_reminder' end;
    elsif tg_table_name='jarvis_visual_focuses' then
        perform sync_jarvis_attention_visual_subject(new.owner_id,'visual_focus',new.id,
            jsonb_build_object('target',new.target,'description',new.description,'purpose',new.purpose,
                               'mode',new.mode,'expires_at',new.expires_at),
            case when new.cleared_at is not null then 'cancelled' when new.completed_at is not null then 'completed' else 'active' end,
            new.expires_at,new.trigger_claim_token,new.completion_evidence->>'attention_event_id');
    else
        perform sync_jarvis_attention_visual_subject(new.owner_id,'conditional_reminder',new.id,
            jsonb_build_object('title',new.title,'details',new.details,'condition',new.condition,
                               'remind_at',new.remind_at,'cancelled_at',new.metadata->>'attention_cancelled_at'),
            case when new.status in ('deleted','failed') or new.metadata ? 'attention_cancelled_at' then 'cancelled'
                 when new.status='completed' or new.triggered_at is not null then 'completed' else 'active' end,
            null,new.trigger_claim_token,new.metadata->>'attention_completed_event_id');
    end if;
    return null;
end $$;
create trigger jarvis_visual_attention_focus_changed after insert or update or delete on jarvis_visual_focuses
for each row execute function update_jarvis_attention_visual_subject();
-- This sorts AFTER the reminder source-invalidation trigger from migration 23.
create trigger jarvis_visual_attention_reminder_changed after insert or update or delete on jarvis_reminders
for each row execute function update_jarvis_attention_visual_subject();

select sync_jarvis_attention_visual_subject(owner_id,'visual_focus',id,
    jsonb_build_object('target',target,'description',description,'purpose',purpose,'mode',mode,'expires_at',expires_at),
    case when cleared_at is not null then 'cancelled' when completed_at is not null then 'completed' else 'active' end,
expires_at,trigger_claim_token,completion_evidence->>'attention_event_id') from jarvis_visual_focuses;
select sync_jarvis_attention_visual_subject(owner_id,'conditional_reminder',id,
    jsonb_build_object('title',title,'details',details,'condition',condition,'remind_at',remind_at,
                      'cancelled_at',metadata->>'attention_cancelled_at'),
    case when status in ('deleted','failed') or metadata ? 'attention_cancelled_at' then 'cancelled'
         when status='completed' or triggered_at is not null then 'completed' else 'active' end,
    null,trigger_claim_token,metadata->>'attention_completed_event_id') from jarvis_reminders;

create unique index jarvis_attention_one_visual_observation
    on jarvis_attention_intents(owner_id,coalesce_key) where source_namespace='visual_alert';

create function guard_jarvis_visual_attention_intent() returns trigger
language plpgsql set search_path from current as $$
declare body jsonb; kind text; valid boolean;
begin
    if new.source_namespace not in ('visual_alert','prospective_event') then return new; end if;
    select content into body from jarvis_context_payloads
        where owner_id=new.owner_id and id=new.source_payload_id and availability='available';
    kind:=case when new.source_namespace='visual_alert' then 'visual' else 'prospective' end;
    if not coalesce(body->>'format'='jarvis.attention.visual_notification.v1'
        and body->>'source_namespace'=new.source_namespace and body->>'source_id'=new.source_id::text
        and body->>'session_id'=new.preferred_session_id::text and body->>'authored_by'='visual_analyzer'
        and new.task_id is null and new.source_revision=1
        and (body->>'expires_at')::timestamptz=new.expires_at, false) then
        raise exception 'Visual attention requires its exact notification envelope' using errcode='23514';
    end if;
    valid:=exists(select 1 from jarvis_visual_delivery_sources b
        join jarvis_context_payload_dependencies d on d.owner_id=b.owner_id and d.source_payload_id=b.payload_id
        where b.owner_id=new.owner_id and b.delivery_kind=kind and b.delivery_id=new.source_id
          and b.session_id=new.preferred_session_id and b.payload_id::text=body->>'observation_payload_id'
          and d.dependent_payload_id=new.source_payload_id);
    if new.source_namespace='visual_alert' then
        valid:=valid and exists(select 1 from visual_interrupts
            where owner_id=new.owner_id and id=new.source_id and message=body->>'message')
            and new.coalesce_key='observation:' || (body->>'observation_payload_id');
    else
        valid:=valid and exists(select 1 from jarvis_prospective_delivery_events e
            join jarvis_attention_visual_subjects s on s.owner_id=e.owner_id
              and s.subject_id=coalesce(e.visual_focus_id,e.reminder_id)
            where e.owner_id=new.owner_id and e.id=new.source_id and e.message=body->>'message'
              and s.subject_type=body->>'subject_type' and s.subject_id::text=body->>'subject_id'
              and s.generation::text=body->>'subject_generation'
              and s.claim_token=e.claim_token and s.claim_token::text=body->>'claim_token' and s.state='active');
    end if;
    if not coalesce(valid,false) then
        raise exception 'Visual attention requires its exact owned observation and subject' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_visual_attention_intent_guard before insert on jarvis_attention_intents
for each row execute function guard_jarvis_visual_attention_intent();

create function revoke_jarvis_visual_attention_original() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='DELETE' or new.message is distinct from old.message then
        update jarvis_context_payloads p set availability='unavailable',content=null
        where p.owner_id=old.owner_id and p.availability='available'
          and exists(select 1 from jarvis_attention_intents i
              where i.owner_id=p.owner_id and i.source_payload_id=p.id and i.source_id=old.id
                and i.source_namespace=case when tg_table_name='visual_interrupts' then 'visual_alert' else 'prospective_event' end);
    end if;
    return null;
end $$;
create trigger jarvis_visual_attention_original_changed after update of message or delete on visual_interrupts
for each row execute function revoke_jarvis_visual_attention_original();
create trigger jarvis_prospective_attention_original_changed after update of message or delete on jarvis_prospective_delivery_events
for each row execute function revoke_jarvis_visual_attention_original();

create function revoke_jarvis_visual_attention_copies() returns trigger
language plpgsql set search_path from current as $$
declare intent record; event record;
begin
    if new.availability not in ('deleted','expired','unavailable')
       or old.content->>'format' is distinct from 'jarvis.attention.visual_notification.v1' then return new; end if;
    for intent in select source_namespace,source_id,id from jarvis_attention_intents
        where owner_id=new.owner_id and source_payload_id=new.id order by id loop
        if intent.source_namespace='visual_alert' then
            update visual_interrupts set message='',reason='source_unavailable',confidence=0,dedupe_bucket=''
            where owner_id=new.owner_id and id=intent.source_id;
        else
            select * into event from jarvis_prospective_delivery_events where owner_id=new.owner_id and id=intent.source_id for update;
            if not found then continue; end if;
            update jarvis_prospective_delivery_events set message='[source unavailable]',evidence='{}'::jsonb,
                event_type=case when event_type in ('generated','dispatched_to_session') then 'failed' else event_type end,
                metadata=jsonb_build_object('delivery_route','attention','attention_owned',true,
                    'reason','source_unavailable','playback_status','unconfirmed','physical_action_status','not_established'),
                updated_at=clock_timestamp()
            where owner_id=new.owner_id and id=event.id;
            if event.event_type in ('generated','dispatched_to_session') then
                update jarvis_visual_focuses set trigger_claim_token=null,trigger_claimed_at=null
                where owner_id=new.owner_id and id=event.visual_focus_id and trigger_claim_token=event.claim_token;
                update jarvis_reminders set trigger_claim_token=null,trigger_claimed_at=null
                where owner_id=new.owner_id and id=event.reminder_id and trigger_claim_token=event.claim_token;
            end if;
            update jarvis_visual_focuses set completion_evidence=jsonb_build_object(
                'attention_intent_id',intent.id,'status','source_unavailable',
                'meaning','Historical search notification, not current physical evidence')
            where owner_id=new.owner_id and completion_evidence->>'attention_intent_id'=intent.id::text;
        end if;
    end loop;
    return new;
end $$;
create trigger jarvis_visual_attention_payload_changed after update of availability,content on jarvis_context_payloads
for each row execute function revoke_jarvis_visual_attention_copies();
-- No prior visual notification is enqueued or replayed by this migration.
insert into jarvis_schema_migrations (version, name, checksum) values (24, 'visual_attention', '890047392a929b8986036a6b8bce1c6ff22bb8abeec39028ab8c0c93de2351d3');

-- APNs is a channel of the existing inbox, never proof of hearing.
alter table jarvis_attention_attempts drop constraint jarvis_attention_attempts_phase_check;
alter table jarvis_attention_attempts add constraint jarvis_attention_attempts_phase_check
    check(phase in ('claimed','submitted','forwarding','awaiting_receipt','played',
                   'interrupted','failed','unknown','revoked','accepted'));
alter table jarvis_attention_attempts add constraint jarvis_attention_phone_phase_check
    check((phase<>'accepted' or channel='phone') and
          (channel<>'phone' or phase not in ('forwarding','awaiting_receipt','played','interrupted')));
alter table jarvis_attention_intents
    add column phone_notice_generation bigint check(phone_notice_generation>0),
    add column phone_notice_state text check(phone_notice_state in ('claimed','submitted','accepted','unknown')),
    add constraint jarvis_attention_phone_projection_check
        check((phone_notice_generation is null)=(phone_notice_state is null));

create table jarvis_push_devices (
    id uuid primary key,
    owner_id text not null,
    revision bigint not null check(revision>0),
    environment text not null check(environment in ('sandbox','production')),
    topic text not null check(length(topic) between 1 and 200),
    token text check(length(token) between 2 and 4096 and token ~ '^[0-9a-f]+$'),
    token_digest text check(token_digest ~ '^[0-9a-f]{64}$'),
    permission text not null check(permission in ('not_determined','denied','authorized','provisional','ephemeral')),
    enabled boolean not null,
    quiet_until timestamptz check(isfinite(quiet_until)),
    invalid_reason text,
    registered_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    check((token is null)=(token_digest is null)),
    check(not enabled or token is not null)
);
create unique index jarvis_push_active_address on jarvis_push_devices(environment,topic,token_digest)
    where enabled and token_digest is not null;
create index jarvis_push_owned_devices on jarvis_push_devices(owner_id,enabled);

create table jarvis_push_device_requests (
    owner_id text not null,
    request_id uuid not null,
    device_id uuid not null,
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_revision bigint not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_id),
    foreign key(owner_id,device_id) references jarvis_push_devices(owner_id,id)
);
create trigger jarvis_push_device_requests_append_only before update on jarvis_push_device_requests
    for each row execute function reject_jarvis_foundation_update();

create table jarvis_push_submissions (
    owner_id text not null,
    attempt_id uuid not null,
    device_id uuid not null,
    device_revision bigint not null,
    provider_id uuid,
    provider_status integer,
    provider_reason text,
    retry_at timestamptz check(isfinite(retry_at)),
    primary key(owner_id,attempt_id),
    foreign key(owner_id,attempt_id) references jarvis_attention_attempts(owner_id,id),
    foreign key(owner_id,device_id) references jarvis_push_devices(owner_id,id)
);
create index jarvis_push_submission_device on jarvis_push_submissions(owner_id,device_id,device_revision);

create index jarvis_attention_phone_history on jarvis_attention_attempts(owner_id,intent_id,generation,created_at)
    where channel='phone';
insert into jarvis_schema_migrations (version, name, checksum) values (25, 'phone_push', '3cea85a834cd004731b5392589a106d8eff34dcb7cdb2126169766a6644237e7');

-- Durable resource intent precedes E2B creation. No command text or credentials.
create table jarvis_sandbox_allocations (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    attempt_id uuid not null,
    grant_id uuid not null,
    policy jsonb not null check(jsonb_typeof(policy)='object' and octet_length(policy::text)<=8192),
    policy_digest text not null check(policy_digest ~ '^[0-9a-f]{64}$'),
    phase text not null check(phase in ('reserved','creating','uncertain','preparing','ready','cleanup_requested','cleaned')),
    submission_token uuid,
    submission_fence bigint check(submission_fence>0),
    sandbox_id text check(length(sandbox_id) between 1 and 200),
    template_id text check(length(template_id) between 1 and 200),
    provider_expires_at timestamptz check(isfinite(provider_expires_at)),
    deadline timestamptz not null check(isfinite(deadline)),
    reason text check(length(reason) between 1 and 100),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,grant_id),
    unique(owner_id,task_id,attempt_id),
    unique(sandbox_id),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    foreign key(owner_id,task_id,grant_id) references jarvis_task_grants(owner_id,task_id,grant_id),
    check((submission_token is null)=(submission_fence is null)),
    check((sandbox_id is null)=(template_id is null)),
    check((sandbox_id is null)=(provider_expires_at is null)),
    check(phase not in ('creating','uncertain','preparing','ready') or submission_token is not null),
    check(phase not in ('preparing','ready') or sandbox_id is not null),
    check(phase<>'reserved' or submission_token is null)
);
create index jarvis_sandbox_cleanup_due on jarvis_sandbox_allocations(deadline,owner_id)
    where phase<>'cleaned';
insert into jarvis_schema_migrations (version, name, checksum) values (26, 'sandbox_allocations', 'dbc506bcec2f0b3f43671a2a342fb3ec11bece55714b1bc94a64d9e078bbf4a1');

-- Recovery is resource ownership, not another model/task execution engine.
alter table jarvis_sandbox_allocations
    add column cleanup_token uuid,
    add column cleanup_worker text check(length(cleanup_worker) between 1 and 100),
    add column cleanup_generation bigint not null default 0 check(cleanup_generation>=0),
    add column cleanup_lease_until timestamptz check(isfinite(cleanup_lease_until)),
    add column next_cleanup_at timestamptz not null default clock_timestamp() check(isfinite(next_cleanup_at)),
    add column cleanup_error text check(length(cleanup_error) between 1 and 100),
    add column cleanup_observed_ids text[] not null default '{}' check(cardinality(cleanup_observed_ids)<=100),
    add constraint jarvis_sandbox_cleanup_claim_pair
        check((cleanup_token is null)=(cleanup_lease_until is null)),
    add constraint jarvis_sandbox_cleanup_claim_phase
        check(cleanup_token is null or phase='cleanup_requested');
create index jarvis_sandbox_cleanup_poll on jarvis_sandbox_allocations(next_cleanup_at,owner_id)
    where phase<>'cleaned';

create function guard_jarvis_sandbox_ownership() returns trigger language plpgsql as $$
begin
    if row(new.id,new.owner_id,new.task_id,new.task_revision,new.attempt_id,new.grant_id,
           new.policy,new.policy_digest,new.created_at)
       is distinct from row(old.id,old.owner_id,old.task_id,old.task_revision,old.attempt_id,old.grant_id,
           old.policy,old.policy_digest,old.created_at)
       or new.deadline>old.deadline then
        raise exception 'Sandbox ownership and authority cannot be reassigned or extended' using errcode='23514';
    end if;
    if old.submission_token is not null and
       row(new.submission_token,new.submission_fence) is distinct from row(old.submission_token,old.submission_fence) then
        raise exception 'Sandbox submission identity is immutable' using errcode='23514';
    end if;
    if old.sandbox_id is not null and row(new.sandbox_id,new.template_id) is distinct from row(old.sandbox_id,old.template_id) then
        raise exception 'Sandbox native identity is immutable' using errcode='23514';
    end if;
    if not old.cleanup_observed_ids <@ new.cleanup_observed_ids
       or (old.phase in ('cleanup_requested','cleaned') and new.phase not in ('cleanup_requested','cleaned')) then
        raise exception 'Sandbox cleanup evidence cannot be erased or execution restored' using errcode='23514';
    end if;
    return new;
end; $$;
create trigger jarvis_sandbox_ownership_guard before update on jarvis_sandbox_allocations
    for each row execute function guard_jarvis_sandbox_ownership();
insert into jarvis_schema_migrations (version, name, checksum) values (27, 'sandbox_cleanup', 'b31ddf1917d1c21e2dda93dec3af0997395dedca71a5e75a2b8e3610f1fe2b58');

-- Source-bound command intent/results. Native execution is never replayed from an unknown receipt.
create table jarvis_sandbox_commands (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    task_id uuid not null,
    task_revision bigint not null check(task_revision>0),
    attempt_id uuid not null,
    allocation_id uuid not null,
    operation_digest text not null check(operation_digest ~ '^[0-9a-f]{64}$'),
    request_digest text not null check(request_digest ~ '^[0-9a-f]{64}$'),
    parameters_payload_id uuid not null,
    result_payload_id uuid,
    result_digest text check(result_digest ~ '^[0-9a-f]{64}$'),
    reservation_id uuid not null,
    phase text not null check(phase in ('prepared','submitting','running','unknown','completed')),
    submission_token uuid,
    pid integer check(pid>0),
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,allocation_id,operation_digest),
    foreign key(owner_id,task_id,attempt_id) references jarvis_task_attempts(owner_id,task_id,id),
    foreign key(owner_id,allocation_id) references jarvis_sandbox_allocations(owner_id,id),
    foreign key(owner_id,parameters_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,result_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,reservation_id) references jarvis_task_reservations(owner_id,id),
    check((phase='prepared')=(submission_token is null)),
    check((phase='completed')=(result_digest is not null)),
    check(result_payload_id is null or phase='completed')
);
create function guard_jarvis_sandbox_command() returns trigger
language plpgsql set search_path from current as $$
begin
    if (new.id,new.owner_id,new.task_id,new.task_revision,new.attempt_id,new.allocation_id,
        new.operation_digest,new.request_digest,new.parameters_payload_id,new.reservation_id)
        is distinct from
       (old.id,old.owner_id,old.task_id,old.task_revision,old.attempt_id,old.allocation_id,
        old.operation_digest,old.request_digest,old.parameters_payload_id,old.reservation_id)
       or (old.submission_token is not null and new.submission_token is distinct from old.submission_token)
       or (old.pid is not null and new.pid is distinct from old.pid)
       or (old.phase='completed' and (new.phase,new.result_digest,new.result_payload_id)
           is distinct from (old.phase,old.result_digest,old.result_payload_id)) then
        raise exception 'Sandbox command ownership or receipt is immutable' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_command_guard before update on jarvis_sandbox_commands
for each row execute function guard_jarvis_sandbox_command();
insert into jarvis_schema_migrations (version, name, checksum) values (28, 'sandbox_commands', '33dac9f242e07dba2f8c7132af2f9efae6cbcca3e02f3a999ffd4b0d1b4edbcb');

-- Ready workers advertise optional computation support; unknown old resources fence replacements.
alter table jarvis_task_workers add column capabilities text[] not null default '{}'
    check(cardinality(capabilities)<=16);
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_allocations_owner_id_grant_id_key;
create unique index jarvis_sandbox_one_unfinished_grant on jarvis_sandbox_allocations(owner_id,grant_id)
    where phase<>'cleaned';
create function guard_jarvis_sandbox_grant_task() returns trigger
language plpgsql set search_path from current as $$
begin
    if exists(select 1 from jarvis_sandbox_allocations
              where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'A sandbox grant cannot move between tasks' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_grant_task_guard before insert on jarvis_sandbox_allocations
for each row execute function guard_jarvis_sandbox_grant_task();

-- Bind scope at attachment, not only after the first VM allocation.
do $$
begin
    if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants g
              on g.owner_id=tg.owner_id and g.id=tg.grant_id where g.capability='sandbox.execute'
              group by tg.owner_id,tg.grant_id having count(*)>1) then
        raise exception 'Existing sandbox grants require unambiguous task ownership' using errcode='23514';
    end if;
end $$;
create function guard_jarvis_sandbox_grant_attachment() returns trigger
language plpgsql set search_path from current as $$
declare selected_capability text;
begin
    select capability into selected_capability from jarvis_operation_grants
        where owner_id=new.owner_id and id=new.grant_id for update;
    if selected_capability='sandbox.execute' and exists(select 1 from jarvis_task_grants
        where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'A sandbox grant is bound to one task' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_sandbox_grant_attachment before insert or update on jarvis_task_grants
for each row execute function guard_jarvis_sandbox_grant_attachment();
insert into jarvis_schema_migrations (version, name, checksum) values (29, 'sandbox_worker_activation', '92395cababda468411ce53e7066e232c46d141ef7d41255dc83264d412965216');

-- Generated-output allowances and exact derived effects are bound to one task.
create or replace function guard_jarvis_sandbox_grant_attachment() returns trigger
language plpgsql set search_path from current as $$
declare selected_capability text;
begin
    select capability into selected_capability from jarvis_operation_grants
        where owner_id=new.owner_id and id=new.grant_id for update;
    if selected_capability in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
       and exists(select 1 from jarvis_task_grants
                  where owner_id=new.owner_id and grant_id=new.grant_id and task_id<>new.task_id) then
        raise exception 'An execution or generated-output grant is bound to one task' using errcode='23514';
    end if;
    return new;
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (30, 'generated_artifact_grants', '05f3d4a23d4f7fe4f02976fa9a0510820469561e435b6d20a691934b12cfb3e5');

-- Descriptive metadata only. This column grants no capability or source access.
alter table jarvis_knowledge_areas add column description text not null default ''
    check (length(description)<=4000);
insert into jarvis_schema_migrations (version, name, checksum) values (31, 'knowledge_area_descriptions', 'e9d10a261785cdc28eecf9a0e6fc23a7d423ec21b66607324a80e6908828f868');

-- Application scope identity only; no semantic memory or provider configuration.
create table jarvis_session_knowledge_selections (
    owner_id text not null,
    session_id uuid not null,
    revision bigint not null check (revision > 0),
    area_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check (request_revision > 0),
    arguments_digest text not null check (arguments_digest ~ '^[0-9a-f]{64}$'),
    receipt_event_id uuid not null,
    created_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,session_id,revision),
    unique (owner_id,request_event_id,request_revision),
    foreign key (owner_id,session_id) references jarvis_sessions(owner_id,id),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,request_event_id) references jarvis_context_events(owner_id,id),
    foreign key (owner_id,receipt_event_id) references jarvis_context_events(owner_id,id)
);

create table jarvis_knowledge_context_scopes (
    owner_id text not null,
    source_namespace text not null check (length(source_namespace) between 1 and 100),
    source_event_id text not null check (length(source_event_id) between 1 and 512),
    area_id uuid not null,
    bound_event_id uuid not null,
    origin_sequence bigint not null check (origin_sequence > 0),
    created_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,source_namespace,source_event_id),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,bound_event_id) references jarvis_context_events(owner_id,id)
);

create function guard_jarvis_knowledge_scope_identity() returns trigger language plpgsql as $$
begin
    raise exception 'Knowledge scope history is immutable' using errcode='23514';
end; $$;
create trigger jarvis_session_knowledge_selections_immutable before update or delete
on jarvis_session_knowledge_selections for each row execute function guard_jarvis_knowledge_scope_identity();
create trigger jarvis_knowledge_context_scopes_immutable before update or delete
on jarvis_knowledge_context_scopes for each row execute function guard_jarvis_knowledge_scope_identity();
insert into jarvis_schema_migrations (version, name, checksum) values (32, 'session_knowledge_scopes', 'd15de8dc6e6df59fcd2626b2e6c4e7978b2d588f94015aba98eb9523feaf397c');

-- Area metadata history uses removable originals, not permanent private audit prose.
create table jarvis_knowledge_area_versions (
    owner_id text not null,
    area_id uuid not null,
    revision bigint not null check (revision > 0),
    status text not null check (status in ('active','archived','deleted')),
    origin text not null check (origin in ('migration_baseline','created','updated')),
    metadata_payload_id uuid,
    area_updated_at timestamptz not null,
    recorded_at timestamptz not null default clock_timestamp(),
    primary key (owner_id,area_id,revision),
    foreign key (owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    foreign key (owner_id,metadata_payload_id) references jarvis_context_payloads(owner_id,id)
);
create trigger jarvis_knowledge_area_versions_immutable before update or delete
on jarvis_knowledge_area_versions for each row execute function guard_jarvis_knowledge_scope_identity();

create function retain_jarvis_knowledge_area_revision(
    area_row jarvis_knowledge_areas, revision_origin text
) returns void language plpgsql set search_path from current as $$
declare original_id uuid;
begin
    if area_row.status <> 'deleted' then
        insert into jarvis_context_payloads(owner_id,availability,content)
        values(area_row.owner_id,'available',jsonb_build_object(
            'name',area_row.name,'domain',area_row.domain,'description',area_row.description))
        returning id into original_id;
    else
        -- Deletion must not leave earlier private names/descriptions in this new copy.
        update jarvis_context_payloads p set availability='deleted',content=null
        where p.owner_id=area_row.owner_id and p.availability<>'deleted' and exists(
            select 1 from jarvis_knowledge_area_versions h
            where h.owner_id=p.owner_id and h.area_id=area_row.id and h.metadata_payload_id=p.id);
    end if;
    insert into jarvis_knowledge_area_versions(
        owner_id,area_id,revision,status,origin,metadata_payload_id,area_updated_at)
    values(area_row.owner_id,area_row.id,area_row.revision,area_row.status,
        revision_origin,original_id,area_row.updated_at);
end; $$;

create function capture_jarvis_knowledge_area_revision() returns trigger
language plpgsql set search_path from current as $$
begin
    perform retain_jarvis_knowledge_area_revision(
        new,case when tg_op='INSERT' then 'created' else 'updated' end);
    return new;
end; $$;
create trigger jarvis_knowledge_areas_capture_revision after insert or update
on jarvis_knowledge_areas for each row execute function capture_jarvis_knowledge_area_revision();

-- Do not invent metadata from revisions that predate this history.
do $$
declare area_row jarvis_knowledge_areas;
begin
    for area_row in select * from jarvis_knowledge_areas loop
        perform retain_jarvis_knowledge_area_revision(area_row,'migration_baseline');
    end loop;
end; $$;
insert into jarvis_schema_migrations (version, name, checksum) values (33, 'knowledge_area_history', '24dc8767fc8bf8e49a13fe911436132e0289045ef5f6ba06476267d8fc4c34b1');

-- Outbound read-only Mac connections. Originals remain canonical removable sources.
create table jarvis_project_connectors (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    label text not null check (length(label) between 1 and 100),
    token_digest text not null check (token_digest ~ '^[0-9a-f]{64}$'),
    expires_at timestamptz not null,
    status text not null default 'active' check (status in ('active','revoked')),
    last_seen_at timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    foreign key(owner_id,area_id) references jarvis_knowledge_areas(owner_id,id),
    check(isfinite(expires_at) and expires_at>created_at)
);
create table jarvis_project_roots (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    connector_id uuid not null,
    local_root_id uuid not null,
    label text not null check (length(label) between 1 and 100),
    fingerprint text not null check(fingerprint ~ '^[0-9a-f]{64}$'),
    authorization_payload_id uuid not null,
    status text not null default 'active' check(status in ('active','revoked')),
    unique(owner_id,id),
    unique(owner_id,area_id,id),
    unique(connector_id,local_root_id),
    foreign key(owner_id,area_id,connector_id) references jarvis_project_connectors(owner_id,area_id,id),
    foreign key(owner_id,authorization_payload_id) references jarvis_context_payloads(owner_id,id)
);
create table jarvis_project_requests (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    area_id uuid not null,
    root_id uuid not null,
    request_payload_id uuid not null,
    purpose text not null default 'inspection' check(purpose in ('inspection','refresh')),
    status text not null default 'pending' check(status in ('pending','leased','completed','expired','revoked')),
    lease_id uuid,
    lease_until timestamptz,
    response_digest text check(response_digest ~ '^[0-9a-f]{64}$'),
    source_id uuid,
    created_at timestamptz not null default clock_timestamp(),
    deadline timestamptz not null default clock_timestamp()+interval '90 seconds',
    unique(owner_id,id),
    foreign key(owner_id,area_id,root_id) references jarvis_project_roots(owner_id,area_id,id),
    foreign key(owner_id,request_payload_id) references jarvis_context_payloads(owner_id,id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    check((status='completed')=(source_id is not null)),
    check((lease_id is null)=(lease_until is null))
);
create index jarvis_project_requests_poll_idx on jarvis_project_requests(owner_id,root_id,created_at)
    where status in ('pending','leased');

-- Only already-read sources are refreshed, not every file in an approved root.
-- Paths remain in removable originals rather than a second permanent text copy.
create table jarvis_project_file_watches (
    owner_id text not null,
    root_id uuid not null,
    source_id uuid not null,
    checked_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,root_id,source_id),
    foreign key(owner_id,root_id) references jarvis_project_roots(owner_id,id),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id)
);

-- Root identity cannot be silently rebound to a different host path or scope.
create function guard_jarvis_project_root() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.area_id<>old.area_id
        or new.connector_id<>old.connector_id or new.local_root_id<>old.local_root_id
        or new.fingerprint<>old.fingerprint or new.authorization_payload_id<>old.authorization_payload_id
        or (old.status='revoked' and new.status<>'revoked') then
        raise exception 'Project root enrollment is immutable except revocation' using errcode='23514';
    end if;
    if new.status='revoked' and old.status<>'revoked' then
        -- Existing copy/source triggers invalidate originals, task evidence and results.
        update jarvis_context_payloads set availability='deleted',content=null
            where owner_id=new.owner_id and id=new.authorization_payload_id and availability<>'deleted';
        update jarvis_project_requests set status='revoked'
            where owner_id=new.owner_id and root_id=new.id and status in ('pending','leased');
    end if;
    return new;
end; $$;
create trigger jarvis_project_roots_guard before update on jarvis_project_roots
for each row execute function guard_jarvis_project_root();

create function revoke_jarvis_project_connector() returns trigger
language plpgsql set search_path from current as $$
begin
    if new.id<>old.id or new.owner_id<>old.owner_id or new.area_id<>old.area_id
        or new.token_digest<>old.token_digest or new.expires_at<>old.expires_at
        or (old.status='revoked' and new.status<>'revoked') then
        raise exception 'Connector identity and credential are immutable' using errcode='23514';
    end if;
    if new.status='revoked' then
        update jarvis_project_roots set status='revoked'
            where owner_id=new.owner_id and connector_id=new.id and status<>'revoked';
    end if;
    return new;
end; $$;
create trigger jarvis_project_connectors_guard before update on jarvis_project_connectors
for each row execute function revoke_jarvis_project_connector();
insert into jarvis_schema_migrations (version, name, checksum) values (34, 'project_connectors', '707606a3808b519f2637be61c978df9740555d6a071da1997ba9aa49357d370c');

-- Explicit source expiry. No source is enrolled by age, inference or default.
create table jarvis_source_retention (
    owner_id text not null,
    area_id uuid not null,
    source_id uuid not null,
    policy_revision bigint not null check(policy_revision>0),
    source_revision bigint not null check(source_revision>0),
    source_generation bigint not null check(source_generation>0),
    expires_at timestamptz check(expires_at is null or isfinite(expires_at)),
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision>0),
    status text not null check(status in ('scheduled','disabled','expired','requires_review')),
    reason text check(reason is null or length(reason) between 1 and 100),
    next_check_at timestamptz not null default clock_timestamp(),
    checked_at timestamptz,
    updated_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id),
    foreign key(owner_id,area_id,source_id) references jarvis_knowledge_sources(owner_id,area_id,id),
    foreign key(owner_id,source_id,source_revision) references jarvis_knowledge_source_versions(owner_id,source_id,revision),
    foreign key(owner_id,request_event_id) references jarvis_context_events(owner_id,id),
    check(status<>'scheduled' or expires_at is not null)
);
create index jarvis_source_retention_due_idx on jarvis_source_retention(owner_id,next_check_at,source_id)
where status='scheduled';

-- Metadata-only replay identity; provider call IDs cannot duplicate or renew expiry.
create table jarvis_source_retention_requests (
    owner_id text not null,
    source_id uuid not null,
    request_event_id uuid not null,
    request_revision bigint not null check(request_revision>0),
    parameters_digest text not null check(parameters_digest ~ '^[0-9a-f]{64}$'),
    accepted_policy_revision bigint not null check(accepted_policy_revision>0),
    recorded_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,source_id,request_event_id,request_revision),
    foreign key(owner_id,source_id) references jarvis_knowledge_sources(owner_id,id),
    foreign key(owner_id,request_event_id) references jarvis_context_events(owner_id,id)
);
create trigger jarvis_source_retention_requests_immutable before update or delete
on jarvis_source_retention_requests for each row execute function guard_jarvis_knowledge_scope_identity();
insert into jarvis_schema_migrations (version, name, checksum) values (35, 'source_retention', 'ec933499a31595b1dec6dddc951ffcad4995dee3534de35ee19bfc1f015cda58');

-- Remove per-descendant trigger recursion without weakening atomic read fences.
-- This is still one transaction, not bounded/resumable graph cleanup.
create or replace function revoke_jarvis_source_bound_copies(
    p_owner text,p_source uuid,p_availability text) returns void
language plpgsql set search_path from current as $$
begin
    if p_availability not in ('deleted','expired','unavailable') then
        raise exception 'Invalid source-copy removal state' using errcode='23514';
    end if;
    update jarvis_messages set content='',metadata=jsonb_build_object(
        'response_kind','source_bound_unavailable','source_status',p_availability,'source_payload_id',p_source)
    where owner_id=p_owner and source_payload_id=p_source;

    -- One statement updates the reachable mutable descendants. Its AFTER-row
    -- triggers still run, including message, task, source and outbox fences,
    -- but ordinarily find no mutable children left to recursively update.
    -- Do not disable those triggers: a writer which held a descendant's share
    -- lock may commit a new child after this statement's snapshot was taken.
    -- The descendant's ordinary trigger then catches that late committed copy.
    with recursive affected(id) as (
        select d.dependent_payload_id
        from jarvis_context_payload_dependencies d
        join jarvis_context_payloads p on p.owner_id=d.owner_id and p.id=d.dependent_payload_id
        where d.owner_id=p_owner and d.source_payload_id=p_source
            and (p.availability in ('available','not_finalized')
                 or (p_availability='deleted' and p.availability<>'deleted'))
        union
        select d.dependent_payload_id
        from affected parent
        join jarvis_context_payload_dependencies d on d.owner_id=p_owner and d.source_payload_id=parent.id
        join jarvis_context_payloads p on p.owner_id=d.owner_id and p.id=d.dependent_payload_id
        where p.availability in ('available','not_finalized')
            or (p_availability='deleted' and p.availability<>'deleted')
    )
    update jarvis_context_payloads p set availability=p_availability,content=null
    where p.owner_id=p_owner and p.id in (select id from affected)
        and (p.availability in ('available','not_finalized')
             or (p_availability='deleted' and p.availability<>'deleted'));
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (36, 'source_copy_graph', '3474793439b2a94d88e916d7ea646675c0b507de460926949eb964e11400c31c');

-- Existing imported-message identities permit content-free historical repair.
-- No original table rewrite, inferred mapping or change to import snapshot hashes.
create or replace function guard_jarvis_message_source() returns trigger
language plpgsql set search_path from current as $$
begin
    if tg_op='UPDATE' and old.source_payload_id is not null then
        if new.source_payload_id is distinct from old.source_payload_id or new.owner_id<>old.owner_id then
            raise exception 'A source-bound message cannot detach or change its source' using errcode='23514';
        end if;
        if new.content is distinct from old.content and new.content<>'' then
            raise exception 'A source-bound message can only redact its copied text' using errcode='23514';
        end if;
        if new.content='' then return new; end if;
    end if;
    if new.source_payload_id is not null
       and lock_jarvis_copy_source(new.owner_id,new.source_payload_id)<>'available' then
        -- A historical original may outlive an earlier deletion. Permit only
        -- content-free repair of that exact imported row, never a new copy or
        -- restoration of removed evidence. The host checks its full recorded
        -- input digest before choosing to erase it.
        if tg_op='UPDATE' and old.source_payload_id is null and new.content=''
           and (to_jsonb(new)-array['content','metadata','source_payload_id'])
               = (to_jsonb(old)-array['content','metadata','source_payload_id'])
           and exists(
               select 1 from jarvis_legacy_import_records r
               join jarvis_knowledge_source_versions v on v.owner_id=r.owner_id
                   and v.source_id=r.source_id and v.revision=r.source_revision
               join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
               where r.owner_id=new.owner_id and r.population='jarvis_messages'
                   and r.origin_key=new.id::text and r.disposition in ('imported','already_present')
                   and p.id=new.source_payload_id and p.availability in ('deleted','expired','unavailable')
                   and new.metadata=jsonb_build_object(
                       'response_kind','source_bound_unavailable','source_status',p.availability,
                       'source_payload_id',p.id)
           ) then
            return new;
        end if;
        raise exception 'A message copy requires an available owned source' using errcode='23514';
    end if;
    if new.metadata->>'response_kind'='expert_tts' and not exists(
        select 1 from jarvis_attention_intents i where i.owner_id=new.owner_id
            and i.id::text=new.metadata->>'intent_id' and i.source_payload_id=new.source_payload_id) then
        raise exception 'Expert message provenance must match its exact owned source' using errcode='23514';
    end if;
    return new;
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (37, 'legacy_message_original_removal', '161986ff6472b789e0a434e2da358a09f0a9959482829ca74815e109cee91806');

-- Rules produce ordinary reminders; notification delivery never proves an action.
alter table jarvis_reminders add column not_before timestamptz;
create unique index if not exists jarvis_reminders_responsibility_owner_idx
    on jarvis_reminders(owner_id,id);

create table jarvis_responsibilities (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    session_id uuid,
    revision bigint not null default 1 check(revision > 0),
    status text not null default 'active' check(status in ('active','completed','cancelled')),
    spec jsonb not null,
    next_due_at timestamptz,
    snoozed_until timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    updated_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    foreign key(owner_id,session_id) references jarvis_sessions(owner_id,id)
);
create index jarvis_responsibilities_due_idx on jarvis_responsibilities(owner_id,next_due_at)
    where status='active' and next_due_at is not null;

create table jarvis_responsibility_occurrences (
    id uuid primary key default gen_random_uuid(),
    owner_id text not null,
    responsibility_id uuid not null,
    rule_revision bigint not null,
    scheduled_for timestamptz not null,
    reminder_id uuid not null,
    status text not null default 'open' check(status in ('open','closed','superseded','cancelled')),
    report_state text not null default 'unknown'
        check(report_state in ('unknown','user_reported_done','user_reported_not_done')),
    report_event_id uuid,
    report_revision integer,
    reported_at timestamptz,
    created_at timestamptz not null default clock_timestamp(),
    unique(owner_id,id),
    unique(owner_id,responsibility_id,rule_revision,scheduled_for),
    foreign key(owner_id,responsibility_id) references jarvis_responsibilities(owner_id,id),
    foreign key(owner_id,reminder_id) references jarvis_reminders(owner_id,id),
    foreign key(owner_id,report_event_id,report_revision) references jarvis_context_events(owner_id,id,revision),
    check((report_event_id is null) = (report_revision is null))
);
create index jarvis_responsibility_occurrences_rule_idx
    on jarvis_responsibility_occurrences(owner_id,responsibility_id,scheduled_for desc);

create table jarvis_responsibility_requests (
    owner_id text not null,
    request_event_id uuid not null,
    request_revision integer not null,
    operation text not null,
    request_slot integer not null check(request_slot between 1 and 20),
    arguments_hash text not null,
    responsibility_id uuid not null,
    occurrence_id uuid,
    report_state text,
    created_at timestamptz not null default clock_timestamp(),
    primary key(owner_id,request_event_id,request_revision,operation,request_slot),
    foreign key(owner_id,request_event_id,request_revision) references jarvis_context_events(owner_id,id,revision),
    foreign key(owner_id,responsibility_id) references jarvis_responsibilities(owner_id,id),
    foreign key(owner_id,occurrence_id) references jarvis_responsibility_occurrences(owner_id,id)
);


-- Reversible policy delay is separate from the owner's own inbox snooze.
create table jarvis_responsibility_attention_delays (
    owner_id text not null,
    intent_id uuid not null,
    base_not_before timestamptz not null,
    applied_not_before timestamptz not null,
    primary key(owner_id,intent_id),
    foreign key(owner_id,intent_id) references jarvis_attention_intents(owner_id,id) on delete cascade
);

create function jarvis_responsibility_notice_time(p_owner text,p_reminder uuid,p_now timestamptz)
returns timestamptz language plpgsql stable set search_path from current as $$
declare policy jsonb; candidate timestamptz; local_now timestamp; starts time; ends time; zone text;
begin
    if not exists(select 1 from jarvis_responsibility_occurrences
        where owner_id=p_owner and reminder_id=p_reminder) then return p_now; end if;
    select case when p.availability='available' then p.content else null end into policy
    from jarvis_capability_preferences q join jarvis_context_payloads p
        on p.owner_id=q.owner_id and p.id=q.policy_payload_id
    where q.owner_id=p_owner and q.capability='responsibility_notifications'
      and q.knowledge_area_id is null order by q.revision desc limit 1;
    if not found then return p_now; end if;
    if policy is null then return 'infinity'::timestamptz; end if;
    candidate:=greatest(p_now,(policy->>'paused_until')::timestamptz);
    starts:=(policy->>'quiet_start')::time;
    ends:=(policy->>'quiet_end')::time;
    if starts is null or ends is null then return candidate; end if;
    zone:=policy->>'timezone';
    local_now:=candidate at time zone zone;
    if starts<ends and local_now::time>=starts and local_now::time<ends then
        return (local_now::date+ends) at time zone zone;
    elsif starts>ends and local_now::time>=starts then
        return ((local_now::date+1)+ends) at time zone zone;
    elsif starts>ends and local_now::time<ends then
        return (local_now::date+ends) at time zone zone;
    end if;
    return candidate;
end $$;

create function refresh_jarvis_responsibility_quiet(p_owner text,p_now timestamptz)
returns void language plpgsql set search_path from current as $$
declare item record; prior record; reminder uuid; allowed timestamptz; base_time timestamptz; chosen timestamptz;
begin
    for item in select * from jarvis_attention_intents
        where owner_id=p_owner and state in ('pending','unknown','failed')
          and source_namespace in ('reminder_event','prospective_event')
        order by id for update loop
        reminder:=null;
        if item.source_namespace='reminder_event' then
            select reminder_id into reminder from jarvis_reminder_events
                where owner_id=p_owner and id=item.source_id;
        else
            select reminder_id into reminder from jarvis_prospective_delivery_events
                where owner_id=p_owner and id=item.source_id;
        end if;
        if reminder is null then continue; end if;
        allowed:=jarvis_responsibility_notice_time(p_owner,reminder,p_now);
        select * into prior from jarvis_responsibility_attention_delays
            where owner_id=p_owner and intent_id=item.id;
        if not found and allowed<=p_now then continue; end if;
        base_time:=case when prior.intent_id is not null and item.not_before=prior.applied_not_before
                        then prior.base_not_before else item.not_before end;
        chosen:=greatest(base_time,allowed);
        insert into jarvis_responsibility_attention_delays(owner_id,intent_id,base_not_before,applied_not_before)
            values(p_owner,item.id,base_time,chosen)
            on conflict(owner_id,intent_id) do update set
                base_not_before=excluded.base_not_before,applied_not_before=excluded.applied_not_before;
        if chosen is distinct from item.not_before then
            update jarvis_attention_intents set not_before=chosen where owner_id=p_owner and id=item.id;
        end if;
    end loop;
end $$;
insert into jarvis_schema_migrations (version, name, checksum) values (38, 'responsibilities', '44345c7ce2eb1bab43ca58ae749debe1a4818e424f30d148eb42bb5e5274e7b8');

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
insert into jarvis_schema_migrations (version, name, checksum) values (39, 'attention_voice_routes', 'a7d3110480b8b4678d1ae6243a1bfdec17617baad753d68501417f3012e1a6f3');

-- Extend the existing owner. No alternate check table or placeholder reminders.
alter table jarvis_responsibilities add constraint jarvis_responsibility_spec_kind check (
    jsonb_typeof(spec)='object' and coalesce(spec->>'kind','scheduled') in ('scheduled','episode_check')
    and (coalesce(spec->>'kind','scheduled')<>'episode_check' or next_due_at is null)
);

alter table jarvis_responsibility_occurrences
    alter column reminder_id drop not null,
    add column check_revision bigint not null default 1 check(check_revision>0),
    add column episode_id uuid,
    add column episode_revision bigint,
    add column check_payload jsonb,
    add column check_lifecycle text,
    add column check_resolution jsonb;

-- PostgreSQL CHECK accepts SQL NULL. JSON-null fields must not bypass the shape.
alter table jarvis_responsibility_occurrences add constraint jarvis_responsibility_episode_shape check ((
    (episode_id is null and reminder_id is not null and episode_revision is null
        and check_payload is null and check_lifecycle is null and check_resolution is null)
    or (episode_id is not null and episode_revision is not null and episode_revision>0
        and check_payload is not null and jsonb_typeof(check_payload)='object'
        and check_payload ?& array['question','origin','dependencies','admission_request','last_request','episode_started_at']
        and check_payload->>'origin'='explicit_request'
        and check_lifecycle is not null and check_lifecycle in ('tracking','resolved','dismissed','expired')
        and ((check_lifecycle='resolved' and check_resolution is not null and jsonb_typeof(check_resolution)='object'
            and check_resolution ?& array['basis','source_refs','resolved_at','applicable_episode_or_object_ref']
            and check_resolution->>'basis' in ('user_report','visual_observation','trusted_source'))
            or (check_lifecycle<>'resolved' and check_resolution is null)))
) is true);
create unique index jarvis_responsibility_episode_identity
    on jarvis_responsibility_occurrences(owner_id,episode_id) where episode_id is not null;

create function guard_jarvis_responsibility_kind() returns trigger
language plpgsql set search_path from current as $$
begin
    if coalesce(new.spec->>'kind','scheduled')<>coalesce(old.spec->>'kind','scheduled') then
        raise exception 'Responsibility kind cannot be repurposed' using errcode='23514';
    end if;
    return new;
end $$;
create trigger jarvis_responsibility_kind_immutable before update on jarvis_responsibilities
for each row execute function guard_jarvis_responsibility_kind();

create function guard_jarvis_responsibility_episode() returns trigger
language plpgsql set search_path from current as $$
declare kind text;
begin
    select coalesce(spec->>'kind','scheduled') into kind from jarvis_responsibilities
        where owner_id=new.owner_id and id=new.responsibility_id for share;
    if kind is null or (kind='episode_check')<>(new.episode_id is not null) then
        raise exception 'Episode shape must match its canonical rule kind' using errcode='23514';
    end if;
    if tg_op='UPDATE' and old.episode_id is not null then
        if (new.id,new.owner_id,new.responsibility_id,new.episode_id,new.created_at)
            is distinct from (old.id,old.owner_id,old.responsibility_id,old.episode_id,old.created_at) then
            raise exception 'A new episode requires a new occurrence identity' using errcode='23514';
        end if;
        if new.episode_revision<>old.episode_revision and new.episode_revision<>old.episode_revision+1 then
            raise exception 'Episode correction must advance its revision once' using errcode='23514';
        end if;
        if (new.check_payload,new.check_lifecycle,new.check_resolution,new.episode_revision,
            new.status,new.report_state,new.report_event_id,new.report_revision)
            is distinct from (old.check_payload,old.check_lifecycle,old.check_resolution,old.episode_revision,
                old.status,old.report_state,old.report_event_id,old.report_revision)
            and new.check_revision<>old.check_revision+1 then
            raise exception 'Semantic check changes require a new check revision' using errcode='23514';
        end if;
        if new.check_revision<old.check_revision then
            raise exception 'Check revisions cannot go backwards' using errcode='23514';
        end if;
    end if;
    return new;
end $$;
create trigger jarvis_responsibility_episode_guard before insert or update on jarvis_responsibility_occurrences
for each row execute function guard_jarvis_responsibility_episode();
insert into jarvis_schema_migrations (version, name, checksum) values (40, 'responsibility_episode_checks', '7cde56e525b749addbb6eb5aa19ec5c1feb0d3652718a68b0248fc6abcc77a99');

-- Additive owner metadata only. No backfilled visual proof or queued delivery.
alter table jarvis_visual_focuses
    add column revision bigint not null default 1 check (revision > 0),
    add column episode_revision bigint not null default 1 check (episode_revision > 0),
    add column check_resolution jsonb check (check_resolution is null or coalesce((
        jsonb_typeof(check_resolution) = 'object'
        and check_resolution ?& array['basis','source_refs','resolved_at','applicable_episode_or_object_ref']
        and jsonb_typeof(check_resolution->'basis') = 'string'
        and check_resolution->>'basis' in ('user_report','visual_observation','trusted_source')
        and case when jsonb_typeof(check_resolution->'source_refs') = 'array'
            then jsonb_array_length(check_resolution->'source_refs') between 1 and 8
            else false end
        and jsonb_typeof(check_resolution->'applicable_episode_or_object_ref') = 'object'
        and jsonb_typeof(check_resolution->'resolved_at') = 'string'), false));

create function version_jarvis_visual_focus_check() returns trigger
language plpgsql set search_path from current as $$
declare rebound boolean; changed boolean;
begin
    -- A corrected descriptor starts a new episode version; old evidence must
    -- never be silently repurposed. Ordinary delivery/claim churn is excluded.
    rebound := row(new.target,new.description,new.purpose,new.mode,new.created_at)
        is distinct from row(old.target,old.description,old.purpose,old.mode,old.created_at);
    new.episode_revision := old.episode_revision + case when rebound then 1 else 0 end;
    if rebound then new.check_resolution := null; end if;
    changed := rebound or row(new.expires_at,new.cleared_at,new.completed_at,new.check_resolution)
        is distinct from row(old.expires_at,old.cleared_at,old.completed_at,old.check_resolution);
    new.revision := old.revision + case when changed then 1 else 0 end;
    return new;
end $$;
create trigger jarvis_visual_focus_check_version before update on jarvis_visual_focuses
for each row execute function version_jarvis_visual_focus_check();
insert into jarvis_schema_migrations (version, name, checksum) values (41, 'visual_focus_checks', 'f957a3a818d6833060eb8ffdb2f70258b02cac1f5348112f300ac3648cf25ccc');

-- Coalescing metadata on the canonical inbox, not a queue or preference authority.
-- No backfill or delivery is performed by installing this migration.
alter table jarvis_attention_intents
    add column quiet_episode_id uuid,
    add column quiet_summary_intent_id uuid,
    add constraint jarvis_attention_quiet_pair check (
        (quiet_episode_id is null) = (quiet_summary_intent_id is null)),
    add constraint jarvis_attention_quiet_summary_owner foreign key (owner_id,quiet_summary_intent_id)
        references jarvis_attention_intents(owner_id,id),
    add constraint jarvis_attention_quiet_not_self check (quiet_summary_intent_id is distinct from id);
create index jarvis_attention_quiet_members
    on jarvis_attention_intents(owner_id,quiet_summary_intent_id)
    where quiet_episode_id is not null;
-- Summary idempotency uses the existing unique
-- (owner_id,source_namespace,source_id,source_revision): source_id = episode UUID.
insert into jarvis_schema_migrations (version, name, checksum) values (42, 'attention_quiet_episodes', '9ec581e69fd1c18c337b02f5e95e9d84f7636325837ec8c15017abf23142bcdc');

-- C0 Workspace receipts. Original note content and versions remain in their
-- existing source stores; these rows do not create a second content authority.
create table jarvis_note_operations (
    owner_id text not null,
    operation_id uuid not null,
    session_id uuid not null,
    area_id uuid not null,
    operation_kind text not null check (operation_kind in ('create','patch','restore','delete')),
    parameters_digest text not null check (parameters_digest ~ '^[0-9a-f]{64}$'),
    artifact_id uuid not null,
    result_version_id text not null,
    outcome text not null check (outcome = 'completed'),
    committed_at timestamptz not null default clock_timestamp(),
    primary key (owner_id, operation_id)
);
create trigger jarvis_note_operations_immutable before update on jarvis_note_operations
for each row execute function reject_jarvis_foundation_update();

-- No FK to original notes/source versions: withdrawal must leave a receipt so
-- replay cannot silently recreate a deleted original.
create table jarvis_workspace_requests (
    owner_id text not null,
    session_id uuid not null references jarvis_sessions(id),
    request_id uuid not null,
    parameters_digest text not null check (parameters_digest ~ '^[0-9a-f]{64}$'),
    payload jsonb not null,
    created_at timestamptz not null,
    primary key (owner_id, request_id),
    check (jsonb_typeof(payload) = 'object'),
    check (payload->>'request_id' = request_id::text),
    check (payload->>'state' in ('pending','claimed','displayed','saved_local','dismissed','failed','expired'))
);
create index jarvis_workspace_requests_session_idx
    on jarvis_workspace_requests(owner_id, session_id, created_at desc, request_id desc);
insert into jarvis_schema_migrations (version, name, checksum) values (43, 'workspace_receipts', '91c4c8ea4a78a47a260f58de53c2367691896e5097cb14eec9729c50d0dfe307');

-- 0001_second_brain_foundations.sql declares an inline CHECK on kind.
-- Its PostgreSQL-generated name is jarvis_context_events_kind_check.
-- Representation only: explicit-request/grant authority is unchanged.
alter table jarvis_context_events
    drop constraint jarvis_context_events_kind_check;

alter table jarvis_context_events
    add constraint jarvis_context_events_kind_check check(kind in (
        'user_request','user_transcript','assistant_output','played_output_boundary',
        'tool_outcome','observation','task_update','correction','context_gap',
        'command_request'
    ));
insert into jarvis_schema_migrations (version, name, checksum) values (44, 'lean_command_context', 'cc6ee8d40b9d4f06be874182b15e47a487b08e5a1c8b85285eda5d5f8908489c');

-- Existing instruction text and timestamps remain unchanged. Revision zero is
-- reserved for a missing profile; all existing rows begin at revision one.
alter table jarvis_behavior_profiles
    add column revision bigint not null default 1
        constraint jarvis_behavior_profiles_positive_revision check (revision > 0);
insert into jarvis_schema_migrations (version, name, checksum) values (45, 'behavior_profile_revision', 'b20e41f373b8296377e90035c454ea46baff25dff7306eec2eae98341bfaa1be');

-- Admit only the account-note grant family from a durable native-command
-- create receipt. Ordinary grants and capability preferences retain their
-- existing finalized-user-input requirement. No trigger is disabled.
create or replace function guard_jarvis_explicit_request() returns trigger
language plpgsql set search_path from current as $$
declare
    target_payload uuid;
    request_event record;
    task_record record;
    parent_record record;
    body jsonb;
    command jsonb;
    request jsonb;
    scope jsonb;
    expected_parent jsonb;
    owner_hash bytea;
    canonical_note uuid;
begin
    -- Preserve authority/input-revision serialization from the original guard.
    perform 1 from jarvis_context_streams where owner_id=new.owner_id for update;
    if not exists (
        select 1 from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=new.request_revision
            and e.trust='user_input' and e.kind in ('user_request','user_transcript')
            and e.finality in ('final','revised') and p.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        -- Preferences never receive the command-derived exception. Keep this
        -- branch separate: their NEW record has no grant-only scope columns.
        if tg_table_name<>'jarvis_operation_grants' then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;
        if new.capability not in ('notes.edit_account','notes.publish_account_edit')
            or new.request_revision<>1 or new.scope_revision<>1 or new.status<>'active'
            or new.expires_at<=clock_timestamp() then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;

        select e.*,p.content as body into request_event
        from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=1
            and e.source_namespace='voice_command' and e.kind='command_request'
            and e.trust='derived' and e.finality='final' and p.availability='available'
            and e.session_id is not null
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
        for share of e,p;
        if not found then
            raise exception 'Account-note grants require the original available command' using errcode='23514';
        end if;
        body:=request_event.body;
        command:=body->'command';
        request:=body->'request';
        if jsonb_typeof(body) is distinct from 'object'
            or not (body ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest'])
            or body-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest']<>'{}'::jsonb
            or body->>'format' is distinct from 'jarvis.task_command.v1'
            or body->>'operation' is distinct from 'create'
            or jsonb_typeof(body->'parameters_digest') is distinct from 'string'
            or coalesce(body->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
            or body->'base_revision' is distinct from '0'::jsonb
            or body->>'base_revision' is distinct from '0'
            or (body->'mode' is distinct from 'null'::jsonb and body->'mode' is distinct from '"background"'::jsonb)
            or jsonb_typeof(command) is distinct from 'object'
            or not (command ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
            or command-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
            or command->'owner_id' is distinct from to_jsonb(new.owner_id)
            or command->>'session_id' is distinct from request_event.session_id::text
            or command->>'operation_id' is distinct from request_event.source_event_id
            or command->>'tool_name' is distinct from 'live_note'
            or command->>'capability' is distinct from 'tasks.create'
            or jsonb_typeof(command->'provider_session_key') is distinct from 'string'
            or jsonb_typeof(command->'provider_call_id') is distinct from 'string'
            or coalesce(length(btrim(command->>'provider_session_key')),0) not between 1 and 512
            or coalesce(length(btrim(command->>'provider_call_id')),0) not between 1 and 200
            or coalesce(command->>'provider_session_key','') !~ '[^[:space:]]'
            or coalesce(command->>'provider_call_id','') !~ '[^[:space:]]'
            or jsonb_typeof(command->'arguments_digest') is distinct from 'string'
            or coalesce(command->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
            or jsonb_typeof(command->'arguments') is distinct from 'object'
            or (command->'arguments')-array['instruction']<>'{}'::jsonb
            or jsonb_typeof(command#>'{arguments,instruction}') is distinct from 'string'
            or coalesce(length(command#>>'{arguments,instruction}'),0) not between 1 and 8000
            or coalesce(command#>>'{arguments,instruction}','') !~ '[^[:space:]]'
            or jsonb_typeof(request) is distinct from 'object'
            or not (request ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
            or request-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
            or request->>'request_id' is distinct from request_event.source_event_id
            or request->>'session_id' is distinct from request_event.session_id::text
            or jsonb_typeof(request->'requested_outcome') is distinct from 'string'
            or request->>'requested_outcome' is distinct from command#>>'{arguments,instruction}'
            or jsonb_typeof(request->'title') is distinct from 'string'
            or coalesce(length(request->>'title'),0) not between 1 and 200
            or coalesce(request->>'title','') !~ '[^[:space:]]'
            or request->>'backend_kind' is distinct from 'frontier'
            or request->>'execution_environment' is distinct from 'none'
            or request->'publish_artifacts' is distinct from 'false'::jsonb
            or request->'edit_memory' is distinct from 'false'::jsonb
            or request->'edit_note' is distinct from 'true'::jsonb
            or request->'grant_ids' is distinct from '[]'::jsonb then
            raise exception 'Account-note grants require an exact English note create command' using errcode='23514';
        end if;

        -- The task and immutable create receipt exist before the parent grant
        -- is inserted. An unreceipted context event cannot mint authority.
        select t.*,d.content as description into task_record
        from jarvis_background_tasks t
        join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id
        join jarvis_context_payloads d on d.owner_id=t.owner_id and d.id=t.description_payload_id
        join jarvis_sessions s on s.owner_id=t.owner_id and s.id=t.session_id
        join jarvis_knowledge_areas a on a.owner_id=t.owner_id and a.id=t.knowledge_area_id
        where t.owner_id=new.owner_id and t.id::text=body->>'task_id'
            and t.session_id=request_event.session_id and t.backend_kind='frontier'
            and t.knowledge_area_id::text=body->>'area_id' and a.status='active'
            and request->>'knowledge_area_id'=t.knowledge_area_id::text
            and r.request_id::text=request_event.source_event_id and r.operation='create'
            and r.accepted_revision=1 and r.accepted_revision<=t.revision
            and r.parameters_digest=body->>'parameters_digest'
            and d.availability='available'
            and d.content->>'requested_outcome'=request->>'requested_outcome'
            and d.content->>'title'=request->>'title'
            and d.content->'mode'=body->'mode'
            and d.content->'instruction_request'=jsonb_build_object(
                'event_id',request_event.id::text,'revision',1,'session_id',t.session_id::text)
        for share of t,d,s,a;
        if not found then
            raise exception 'Account-note grants require the matching task create receipt and effective instruction' using errcode='23514';
        end if;
        if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants g
            on g.owner_id=tg.owner_id and g.id=tg.grant_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and (g.capability not in ('notes.edit_account','notes.publish_account_edit')
                    or g.request_event_id<>request_event.id or g.request_revision<>1)) then
            raise exception 'Account-note grants cannot carry unrelated task authority' using errcode='23514';
        end if;

        select p.content into scope from jarvis_context_payloads p
        where p.owner_id=new.owner_id and p.id=new.scope_payload_id and p.availability='available'
        for share;
        if not found or jsonb_typeof(scope) is distinct from 'object' then
            raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
        end if;
        -- Python uuid5(NAMESPACE_URL, 'jarvis.account-note:v1:' +
        -- canonical_json([owner], ensure_ascii=False)). A one-string JSON array
        -- needs no separator rewriting; preserve Unicode and exact owner bytes.
        owner_hash:=digest(decode('6ba7b8119dad11d180b400c04fd430c8','hex') ||
            convert_to('jarvis.account-note:v1:' || jsonb_build_array(new.owner_id)::text,'UTF8'),'sha1');
        owner_hash:=set_byte(owner_hash,6,(get_byte(owner_hash,6) & 15) | 80);
        owner_hash:=set_byte(owner_hash,8,(get_byte(owner_hash,8) & 63) | 128);
        canonical_note:=encode(substring(owner_hash from 1 for 16),'hex')::uuid;
        expected_parent:=jsonb_build_object('task_id',task_record.id::text,
            'artifact_id',canonical_note::text,'area_id',task_record.knowledge_area_id::text,'max_publications',1);

        if new.capability='notes.edit_account' then
            if scope<>expected_parent or scope->>'max_publications'<>'1'
                or new.operation_key<>'account-note-task:' || replace(task_record.id::text,'-','')
                or task_record.revision<>1 or task_record.status<>'queued'
                or task_record.request_event_id<>request_event.id or task_record.request_revision<>1
                or exists(select 1 from jarvis_task_grants tg where tg.owner_id=new.owner_id and tg.task_id=task_record.id) then
                raise exception 'Account-note parent scope must match its fresh accepted task' using errcode='23514';
            end if;
        else
            -- Do not duplicate the document/patch validator in SQL. The child
            -- is a frozen bounded scope; its exact candidate and evidence are
            -- checked again by the task/action and note publication owners.
            if not (scope ?& array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest'])
                or scope-array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest']<>'{}'::jsonb
                or scope->>'task_id' is distinct from task_record.id::text
                or scope->>'artifact_id' is distinct from canonical_note::text
                or (scope->'base_version_id' is distinct from 'null'::jsonb and (
                    jsonb_typeof(scope->'base_version_id') is distinct from 'string'
                    or length(scope->>'base_version_id') not between 1 and 200))
                or jsonb_typeof(scope->'changes') is distinct from 'array'
                or jsonb_typeof(scope->'evidence') is distinct from 'array'
                or jsonb_typeof(scope->'candidate_digest') is distinct from 'string'
                or coalesce(scope->>'candidate_digest','') !~ '^[0-9a-f]{64}$'
                or new.operation_key !~ '^account-note-edit:[0-9a-f]{64}$' then
                raise exception 'Account-note publication scope must name its exact canonical target' using errcode='23514';
            end if;
            if jsonb_array_length(scope->'changes') not between 1 and 32
                or jsonb_array_length(scope->'evidence')>128 then
                raise exception 'Account-note publication scope exceeds its bounds' using errcode='23514';
            end if;
            select g.id,g.expires_at into parent_record
            from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
            join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and g.capability='notes.edit_account' and g.request_event_id=request_event.id and g.request_revision=1
                and g.scope_revision=1 and g.status='active' and g.expires_at>clock_timestamp()
                and g.operation_key='account-note-task:' || replace(task_record.id::text,'-','')
                and p.availability='available' and p.content=expected_parent and p.content->>'max_publications'='1'
            for share of g,p;
            if not found or new.expires_at>parent_record.expires_at then
                raise exception 'Account-note publication requires its attached active exact parent' using errcode='23514';
            end if;
            if (select count(*) from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
                where tg.owner_id=new.owner_id and tg.task_id=task_record.id and g.capability='notes.edit_account')<>1 then
                raise exception 'Account-note publication has competing parents' using errcode='23514';
            end if;
        end if;
    end if;
    -- Original scope-availability and creation-time behavior for every caller.
    if tg_table_name='jarvis_operation_grants' then target_payload:=new.scope_payload_id;
    else target_payload:=new.policy_payload_id; end if;
    if not exists(select 1 from jarvis_context_payloads where owner_id=new.owner_id
        and id=target_payload and availability='available') then
        raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
    end if;
    new.created_at:=clock_timestamp();
    return new;
end;
$$;
insert into jarvis_schema_migrations (version, name, checksum) values (46, 'account_note_command_grants', 'ac956f89163a468c267a86ca482edec2d7417fa73e07053debd1360c6b64d417');

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
insert into jarvis_schema_migrations (version, name, checksum) values (47, 'file_publication_transfer_policy', '921e0e075d0aaec42350ec0c1c707202bbd882095bd2f15e386747ee2c2c7d16');

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
insert into jarvis_schema_migrations (version, name, checksum) values (48, 'recording_admission', 'f4c9590fd6b6ff595809c98b3b8867b4c13b566e86517739b9fc1dcc82a15e5e');

-- Forward-only: an exact native video command may mint only its managed
-- offline execution and two-file recipe/output grant family. Notes and ordinary
-- explicit user requests retain the 0046 guard verbatim below the new branch.
create function jarvis_media_command_grant_valid(g jarvis_operation_grants) returns boolean
language plpgsql set search_path from current as $$
declare
    e record; t record; r record; parent record;
    b jsonb; c jsonb; q jsonb; pin jsonb; scope jsonb;
begin
    if g.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
        or g.request_revision<>1 or g.scope_revision<>1 or g.status<>'active'
        or g.expires_at<=clock_timestamp() then return false; end if;
    select ce.*,p.content as body into e from jarvis_context_events ce
    join jarvis_context_payloads p on p.owner_id=ce.owner_id and p.id=ce.payload_id
    where ce.owner_id=g.owner_id and ce.id=g.request_event_id and ce.revision=1
        and ce.source_namespace='voice_command' and ce.kind='command_request' and ce.trust='derived'
        and ce.finality='final' and ce.session_id is not null and p.availability='available'
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ce.owner_id
            and n.source_namespace=ce.source_namespace and n.source_event_id=ce.source_event_id and n.revision>ce.revision)
    for share of ce,p;
    if not found then return false; end if;
    b:=e.body; c:=b->'command'; q:=b->'request'; pin:=b->'media_input';
    if jsonb_typeof(b) is distinct from 'object'
        or not (b ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input'])
        or b-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input']<>'{}'::jsonb
        or b->>'format' is distinct from 'jarvis.task_command.v1' or b->>'operation' is distinct from 'create'
        or b->'base_revision' is distinct from '0'::jsonb
        or b->'mode' is distinct from '"background"'::jsonb
        or coalesce(b->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c) is distinct from 'object'
        or not(c ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
        or c-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
        or c->'owner_id' is distinct from to_jsonb(g.owner_id) or c->>'session_id' is distinct from e.session_id::text
        or c->>'operation_id' is distinct from e.source_event_id or c->>'tool_name' is distinct from 'start_media_task'
        or c->>'capability' is distinct from 'tasks.create'
        or jsonb_typeof(c->'provider_session_key') is distinct from 'string'
        or jsonb_typeof(c->'provider_call_id') is distinct from 'string'
        or coalesce(c->>'provider_session_key','') !~ '[^[:space:]]'
        or coalesce(c->>'provider_call_id','') !~ '[^[:space:]]'
        or coalesce(c->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c->'arguments') is distinct from 'object'
        or (c->'arguments')-array['instruction','recording_id','saved']<>'{}'::jsonb
        or jsonb_typeof(c#>'{arguments,instruction}') is distinct from 'string'
        or coalesce(length(c#>>'{arguments,instruction}'),0) not between 1 and 8000
        or coalesce(c#>>'{arguments,instruction}','') !~ '[^[:space:]]'
        or jsonb_typeof(q) is distinct from 'object'
        or not(q ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
        or q-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
        or q->>'request_id' is distinct from e.source_event_id or q->>'session_id' is distinct from e.session_id::text
        or q->>'requested_outcome' is distinct from c#>>'{arguments,instruction}'
        or q->>'backend_kind' is distinct from 'frontier' or q->>'execution_environment' is distinct from 'offline_sandbox'
        or q->'publish_artifacts' is distinct from 'true'::jsonb or q->'edit_note' is distinct from 'false'::jsonb
        or q->'edit_memory' is distinct from 'false'::jsonb or q->'grant_ids' is distinct from '[]'::jsonb
        or q->>'knowledge_area_id' is distinct from b->>'area_id' then return false; end if;
    select bt.*,d.content as description into t from jarvis_background_tasks bt
    join jarvis_task_requests tr on tr.owner_id=bt.owner_id and tr.task_id=bt.id
    join jarvis_context_payloads d on d.owner_id=bt.owner_id and d.id=bt.description_payload_id
    where bt.owner_id=g.owner_id and bt.id::text=b->>'task_id' and bt.session_id=e.session_id
        and bt.backend_kind='frontier' and bt.knowledge_area_id::text=b->>'area_id'
        and tr.request_id::text=e.source_event_id and tr.operation='create'
        and tr.accepted_revision=1 and tr.accepted_revision<=bt.revision and tr.parameters_digest=b->>'parameters_digest'
        and d.availability='available' and d.content->>'requested_outcome'=q->>'requested_outcome'
        and d.content->>'title'=q->>'title' and d.content->'mode'=b->'mode'
        and d.content->'media_input'=pin
        and d.content->'instruction_request'=jsonb_build_object('event_id',e.id::text,'revision',1,'session_id',bt.session_id::text)
        and d.content#>>'{managed_plugin,format}'='jarvis.managed-media.v1'
        and d.content#>>'{managed_plugin,binding,owner_id}'=g.owner_id
        and d.content#>>'{managed_plugin,binding,plugin_id}'='video_editor'
        and d.content#>>'{managed_plugin,binding,configuration_id}'=d.content#>>'{managed_plugin,bundle,configuration_id}'
    for share of bt,d;
    if not found then return false; end if;
    select rec.*,v.payload_id as source_payload_id,fp.content_sha256,fp.size_bytes into r
    from jarvis_recordings rec
    join jarvis_knowledge_areas a on a.owner_id=rec.owner_id and a.id=rec.area_id
    join jarvis_knowledge_source_versions v on v.owner_id=rec.owner_id and v.document_id=rec.version_id
        and v.source_id=rec.source_id and v.area_id=rec.area_id
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
    join jarvis_context_payloads cp on cp.owner_id=rec.owner_id and cp.id=rec.completion_payload_id
    join jarvis_context_events ce on ce.owner_id=rec.owner_id and ce.id=rec.completion_event_id and ce.payload_id=cp.id
    join jarvis_context_payloads ap on ap.owner_id=rec.owner_id and ap.id=rec.admission_payload_id
    join jarvis_context_events ae on ae.owner_id=rec.owner_id and ae.id=rec.admission_event_id and ae.payload_id=ap.id
    join jarvis_file_publications fp on fp.owner_id=rec.owner_id and fp.id=rec.publication_id
        and fp.file_id=rec.file_id and fp.area_id=rec.area_id
    join jarvis_context_payloads fpp on fpp.owner_id=fp.owner_id and fpp.id=fp.metadata_payload_id
    where rec.owner_id=g.owner_id and rec.session_id=e.session_id and rec.area_id=t.knowledge_area_id
        and rec.recording_id::text=c#>>'{arguments,recording_id}' and not rec.revoked and rec.capture_closed
        and a.status='active' and s.status='active' and v.revision=s.current_revision
        and p.availability='available' and cp.availability='available' and ap.availability='available'
        and fpp.availability='available' and fp.phase='available' and fp.upload_settled and not fp.delete_requested
        and fp.operation_key='recording:' || rec.recording_id::text and fp.transfer_policy='mp4_capture_v1'
        and p.content->>'recording_id'=rec.recording_id::text and p.content->>'content_sha256'=fp.content_sha256
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ae.owner_id
            and n.source_namespace=ae.source_namespace and n.source_event_id=ae.source_event_id and n.revision>ae.revision)
    for share of rec,a,v,s,p,cp,ap,fp,fpp,ce,ae;
    if not found then return false; end if;
    if c#>'{arguments,saved}' is distinct from jsonb_build_object('file_id',r.file_id::text,
        'source_id',r.source_id::text,'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text)
        or pin is distinct from jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text,'owner_id',r.owner_id,
            'session_id',r.session_id::text,'recording_id',r.recording_id::text,'area_id',r.area_id::text,
            'source_payload_id',r.source_payload_id::text,'receipt_payload_id',r.completion_payload_id::text,
            'admission_payload_id',r.admission_payload_id::text,'content_sha256',r.content_sha256,'size_bytes',r.size_bytes)
        then return false; end if;
    if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants og
        on og.owner_id=tg.owner_id and og.id=tg.grant_id where tg.owner_id=g.owner_id and tg.task_id=t.id
        and (og.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            or og.request_event_id<>e.id or og.request_revision<>1)) then return false; end if;
    select p.content into scope from jarvis_context_payloads p
    where p.owner_id=g.owner_id and p.id=g.scope_payload_id and p.availability='available' for share;
    if not found or jsonb_typeof(scope) is distinct from 'object' then return false; end if;
    if g.capability in ('sandbox.execute','files.publish_generated') then
        if t.revision<>1 or t.status<>'queued' or t.request_event_id<>e.id or t.request_revision<>1
            then return false; end if;
        if g.capability='sandbox.execute' then
            return g.operation_key='managed-video-editor:' || t.id::text
                and scope-array['template','lifetime_seconds','command_seconds','output_bytes','allowed_domains','internet_access']='{}'::jsonb
                and scope->>'template'='base' and scope->'allowed_domains'='[]'::jsonb
                and coalesce(scope->'internet_access','false'::jsonb)='false'::jsonb
                and jsonb_typeof(scope->'lifetime_seconds')='number' and (scope->>'lifetime_seconds')::integer between 60 and 3600
                and jsonb_typeof(scope->'command_seconds')='number' and (scope->>'command_seconds')::integer between 1 and 300
                and jsonb_typeof(scope->'output_bytes')='number' and (scope->>'output_bytes')::integer between 1024 and 256000;
        end if;
        return g.operation_key='artifact-task:' || t.id::text
            and scope-array['max_files','max_file_bytes','max_total_bytes','formats','mp4_max_file_bytes']='{}'::jsonb
            and scope->'formats'='["json","mp4"]'::jsonb and scope->'max_files'='2'::jsonb
            and jsonb_typeof(scope->'max_file_bytes')='number' and (scope->>'max_file_bytes')::bigint between 1 and 8388608
            and jsonb_typeof(scope->'max_total_bytes')='number' and (scope->>'max_total_bytes')::bigint between 1 and 134217728
            and jsonb_typeof(scope->'mp4_max_file_bytes')='number' and (scope->>'mp4_max_file_bytes')::bigint between 1 and 33554432;
    end if;
    if not(scope ?& array['request','inspection','parent_grant_id','task_id','evidence_payload_id'])
        or scope-array['request','inspection','parent_grant_id','task_id','evidence_payload_id','transfer_limit_bytes']<>'{}'::jsonb
        or scope->>'task_id' is distinct from t.id::text
        or scope#>>'{request,format}' not in ('json','mp4')
        or scope#>>'{request,format}' is distinct from scope#>>'{inspection,format}'
        or g.operation_key !~ ('^artifact-snapshot:' || replace(t.id::text,'-','') || ':[0-9a-f]{64}$')
        then return false; end if;
    select og.*,p.content as scope into parent from jarvis_task_grants tg
    join jarvis_operation_grants og on og.owner_id=tg.owner_id and og.id=tg.grant_id
    join jarvis_context_payloads p on p.owner_id=og.owner_id and p.id=og.scope_payload_id
    where tg.owner_id=g.owner_id and tg.task_id=t.id and og.id::text=scope->>'parent_grant_id'
        and og.capability='files.publish_generated' and og.operation_key='artifact-task:' || t.id::text
        and og.request_event_id=e.id and og.request_revision=1 and og.scope_revision=1
        and og.status='active' and og.expires_at>clock_timestamp() and p.availability='available' for share of og,p;
    if not found or g.expires_at>parent.expires_at then return false; end if;
    return exists(select 1 from jarvis_context_payloads p where p.owner_id=g.owner_id
        and p.id::text=scope->>'evidence_payload_id' and p.availability='available');
end;
$$;

-- Admit only the account-note grant family from a durable native-command
-- create receipt. Ordinary grants and capability preferences retain their
-- existing finalized-user-input requirement. No trigger is disabled.
create or replace function guard_jarvis_explicit_request() returns trigger
language plpgsql set search_path from current as $$
declare
    target_payload uuid;
    request_event record;
    task_record record;
    parent_record record;
    body jsonb;
    command jsonb;
    request jsonb;
    scope jsonb;
    expected_parent jsonb;
    owner_hash bytea;
    canonical_note uuid;
begin
    -- Preserve authority/input-revision serialization from the original guard.
    perform 1 from jarvis_context_streams where owner_id=new.owner_id for update;
    if not exists (
        select 1 from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=new.request_revision
            and e.trust='user_input' and e.kind in ('user_request','user_transcript')
            and e.finality in ('final','revised') and p.availability='available'
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
    ) then
        -- Preferences never receive the command-derived exception. Keep this
        -- branch separate: their NEW record has no grant-only scope columns.
        if tg_table_name<>'jarvis_operation_grants' then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;
        if new.capability in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            and jarvis_media_command_grant_valid(new) is true then
            new.created_at:=clock_timestamp();
            return new;
        end if;
        if new.capability not in ('notes.edit_account','notes.publish_account_edit')
            or new.request_revision<>1 or new.scope_revision<>1 or new.status<>'active'
            or new.expires_at<=clock_timestamp() then
            raise exception 'Explicit authority requires a current finalized user request' using errcode='23514';
        end if;

        select e.*,p.content as body into request_event
        from jarvis_context_events e join jarvis_context_payloads p
            on p.owner_id=e.owner_id and p.id=e.payload_id
        where e.owner_id=new.owner_id and e.id=new.request_event_id and e.revision=1
            and e.source_namespace='voice_command' and e.kind='command_request'
            and e.trust='derived' and e.finality='final' and p.availability='available'
            and e.session_id is not null
            and not exists(select 1 from jarvis_context_events newer where newer.owner_id=e.owner_id
                and newer.source_namespace=e.source_namespace and newer.source_event_id=e.source_event_id
                and newer.revision>e.revision)
        for share of e,p;
        if not found then
            raise exception 'Account-note grants require the original available command' using errcode='23514';
        end if;
        body:=request_event.body;
        command:=body->'command';
        request:=body->'request';
        if jsonb_typeof(body) is distinct from 'object'
            or not (body ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest'])
            or body-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest']<>'{}'::jsonb
            or body->>'format' is distinct from 'jarvis.task_command.v1'
            or body->>'operation' is distinct from 'create'
            or jsonb_typeof(body->'parameters_digest') is distinct from 'string'
            or coalesce(body->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
            or body->'base_revision' is distinct from '0'::jsonb
            or body->>'base_revision' is distinct from '0'
            or (body->'mode' is distinct from 'null'::jsonb and body->'mode' is distinct from '"background"'::jsonb)
            or jsonb_typeof(command) is distinct from 'object'
            or not (command ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
            or command-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
            or command->'owner_id' is distinct from to_jsonb(new.owner_id)
            or command->>'session_id' is distinct from request_event.session_id::text
            or command->>'operation_id' is distinct from request_event.source_event_id
            or command->>'tool_name' is distinct from 'live_note'
            or command->>'capability' is distinct from 'tasks.create'
            or jsonb_typeof(command->'provider_session_key') is distinct from 'string'
            or jsonb_typeof(command->'provider_call_id') is distinct from 'string'
            or coalesce(length(btrim(command->>'provider_session_key')),0) not between 1 and 512
            or coalesce(length(btrim(command->>'provider_call_id')),0) not between 1 and 200
            or coalesce(command->>'provider_session_key','') !~ '[^[:space:]]'
            or coalesce(command->>'provider_call_id','') !~ '[^[:space:]]'
            or jsonb_typeof(command->'arguments_digest') is distinct from 'string'
            or coalesce(command->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
            or jsonb_typeof(command->'arguments') is distinct from 'object'
            or (command->'arguments')-array['instruction']<>'{}'::jsonb
            or jsonb_typeof(command#>'{arguments,instruction}') is distinct from 'string'
            or coalesce(length(command#>>'{arguments,instruction}'),0) not between 1 and 8000
            or coalesce(command#>>'{arguments,instruction}','') !~ '[^[:space:]]'
            or jsonb_typeof(request) is distinct from 'object'
            or not (request ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
            or request-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
            or request->>'request_id' is distinct from request_event.source_event_id
            or request->>'session_id' is distinct from request_event.session_id::text
            or jsonb_typeof(request->'requested_outcome') is distinct from 'string'
            or request->>'requested_outcome' is distinct from command#>>'{arguments,instruction}'
            or jsonb_typeof(request->'title') is distinct from 'string'
            or coalesce(length(request->>'title'),0) not between 1 and 200
            or coalesce(request->>'title','') !~ '[^[:space:]]'
            or request->>'backend_kind' is distinct from 'frontier'
            or request->>'execution_environment' is distinct from 'none'
            or request->'publish_artifacts' is distinct from 'false'::jsonb
            or request->'edit_memory' is distinct from 'false'::jsonb
            or request->'edit_note' is distinct from 'true'::jsonb
            or request->'grant_ids' is distinct from '[]'::jsonb then
            raise exception 'Account-note grants require an exact English note create command' using errcode='23514';
        end if;

        -- The task and immutable create receipt exist before the parent grant
        -- is inserted. An unreceipted context event cannot mint authority.
        select t.*,d.content as description into task_record
        from jarvis_background_tasks t
        join jarvis_task_requests r on r.owner_id=t.owner_id and r.task_id=t.id
        join jarvis_context_payloads d on d.owner_id=t.owner_id and d.id=t.description_payload_id
        join jarvis_sessions s on s.owner_id=t.owner_id and s.id=t.session_id
        join jarvis_knowledge_areas a on a.owner_id=t.owner_id and a.id=t.knowledge_area_id
        where t.owner_id=new.owner_id and t.id::text=body->>'task_id'
            and t.session_id=request_event.session_id and t.backend_kind='frontier'
            and t.knowledge_area_id::text=body->>'area_id' and a.status='active'
            and request->>'knowledge_area_id'=t.knowledge_area_id::text
            and r.request_id::text=request_event.source_event_id and r.operation='create'
            and r.accepted_revision=1 and r.accepted_revision<=t.revision
            and r.parameters_digest=body->>'parameters_digest'
            and d.availability='available'
            and d.content->>'requested_outcome'=request->>'requested_outcome'
            and d.content->>'title'=request->>'title'
            and d.content->'mode'=body->'mode'
            and d.content->'instruction_request'=jsonb_build_object(
                'event_id',request_event.id::text,'revision',1,'session_id',t.session_id::text)
        for share of t,d,s,a;
        if not found then
            raise exception 'Account-note grants require the matching task create receipt and effective instruction' using errcode='23514';
        end if;
        if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants g
            on g.owner_id=tg.owner_id and g.id=tg.grant_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and (g.capability not in ('notes.edit_account','notes.publish_account_edit')
                    or g.request_event_id<>request_event.id or g.request_revision<>1)) then
            raise exception 'Account-note grants cannot carry unrelated task authority' using errcode='23514';
        end if;

        select p.content into scope from jarvis_context_payloads p
        where p.owner_id=new.owner_id and p.id=new.scope_payload_id and p.availability='available'
        for share;
        if not found or jsonb_typeof(scope) is distinct from 'object' then
            raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
        end if;
        -- Python uuid5(NAMESPACE_URL, 'jarvis.account-note:v1:' +
        -- canonical_json([owner], ensure_ascii=False)). A one-string JSON array
        -- needs no separator rewriting; preserve Unicode and exact owner bytes.
        owner_hash:=digest(decode('6ba7b8119dad11d180b400c04fd430c8','hex') ||
            convert_to('jarvis.account-note:v1:' || jsonb_build_array(new.owner_id)::text,'UTF8'),'sha1');
        owner_hash:=set_byte(owner_hash,6,(get_byte(owner_hash,6) & 15) | 80);
        owner_hash:=set_byte(owner_hash,8,(get_byte(owner_hash,8) & 63) | 128);
        canonical_note:=encode(substring(owner_hash from 1 for 16),'hex')::uuid;
        expected_parent:=jsonb_build_object('task_id',task_record.id::text,
            'artifact_id',canonical_note::text,'area_id',task_record.knowledge_area_id::text,'max_publications',1);

        if new.capability='notes.edit_account' then
            if scope<>expected_parent or scope->>'max_publications'<>'1'
                or new.operation_key<>'account-note-task:' || replace(task_record.id::text,'-','')
                or task_record.revision<>1 or task_record.status<>'queued'
                or task_record.request_event_id<>request_event.id or task_record.request_revision<>1
                or exists(select 1 from jarvis_task_grants tg where tg.owner_id=new.owner_id and tg.task_id=task_record.id) then
                raise exception 'Account-note parent scope must match its fresh accepted task' using errcode='23514';
            end if;
        else
            -- Do not duplicate the document/patch validator in SQL. The child
            -- is a frozen bounded scope; its exact candidate and evidence are
            -- checked again by the task/action and note publication owners.
            if not (scope ?& array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest'])
                or scope-array['task_id','artifact_id','base_version_id','changes','evidence','candidate_digest']<>'{}'::jsonb
                or scope->>'task_id' is distinct from task_record.id::text
                or scope->>'artifact_id' is distinct from canonical_note::text
                or (scope->'base_version_id' is distinct from 'null'::jsonb and (
                    jsonb_typeof(scope->'base_version_id') is distinct from 'string'
                    or length(scope->>'base_version_id') not between 1 and 200))
                or jsonb_typeof(scope->'changes') is distinct from 'array'
                or jsonb_typeof(scope->'evidence') is distinct from 'array'
                or jsonb_typeof(scope->'candidate_digest') is distinct from 'string'
                or coalesce(scope->>'candidate_digest','') !~ '^[0-9a-f]{64}$'
                or new.operation_key !~ '^account-note-edit:[0-9a-f]{64}$' then
                raise exception 'Account-note publication scope must name its exact canonical target' using errcode='23514';
            end if;
            if jsonb_array_length(scope->'changes') not between 1 and 32
                or jsonb_array_length(scope->'evidence')>128 then
                raise exception 'Account-note publication scope exceeds its bounds' using errcode='23514';
            end if;
            select g.id,g.expires_at into parent_record
            from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
            join jarvis_context_payloads p on p.owner_id=g.owner_id and p.id=g.scope_payload_id
            where tg.owner_id=new.owner_id and tg.task_id=task_record.id
                and g.capability='notes.edit_account' and g.request_event_id=request_event.id and g.request_revision=1
                and g.scope_revision=1 and g.status='active' and g.expires_at>clock_timestamp()
                and g.operation_key='account-note-task:' || replace(task_record.id::text,'-','')
                and p.availability='available' and p.content=expected_parent and p.content->>'max_publications'='1'
            for share of g,p;
            if not found or new.expires_at>parent_record.expires_at then
                raise exception 'Account-note publication requires its attached active exact parent' using errcode='23514';
            end if;
            if (select count(*) from jarvis_task_grants tg join jarvis_operation_grants g
                on g.owner_id=tg.owner_id and g.id=tg.grant_id
                where tg.owner_id=new.owner_id and tg.task_id=task_record.id and g.capability='notes.edit_account')<>1 then
                raise exception 'Account-note publication has competing parents' using errcode='23514';
            end if;
        end if;
    end if;
    -- Original scope-availability and creation-time behavior for every caller.
    if tg_table_name='jarvis_operation_grants' then target_payload:=new.scope_payload_id;
    else target_payload:=new.policy_payload_id; end if;
    if not exists(select 1 from jarvis_context_payloads where owner_id=new.owner_id
        and id=target_payload and availability='available') then
        raise exception 'Authority scope requires an available immutable payload' using errcode='23514';
    end if;
    new.created_at:=clock_timestamp();
    return new;
end;
$$;
insert into jarvis_schema_migrations (version, name, checksum) values (49, 'media_command_grants', 'e816e0ede8fb4307b386a1f966a4c2b8486df58f6b4ffdaec3f72bd5f9828252');

-- 0050: exact retained/corrected targets on the existing media grant predicate.
-- Shared task/source/grant checks; explicit legacy and continuation target branches.
-- No tables, new ledger, policy defaults or changes to the explicit-request guard.
create function jarvis_media_accepted_proof(o text, area uuid, task uuid, action uuid,
    plan text, ref jsonb, kind text, edit uuid, lock_rows boolean default true) returns jsonb
language plpgsql set search_path from current as $$
declare
    result jsonb;
    -- Only the fixed lock suffix varies; all input values remain bound parameters.
    lookup_sql constant text := $media_proof$
    select jsonb_build_object('action_id',a.id::text,'grant_id',g.id::text,
        'scope_payload_id',s.id::text,'receipt_payload_id',p.id::text,
        'parent_scope_payload_id',parentp.id::text,'source_payload_id',original.id::text,
        'metadata_payload_id',metadata.id::text,'parameters_digest',a.parameters_digest)
    from jarvis_task_actions a
    join jarvis_background_tasks t on t.owner_id=a.owner_id and t.id=a.task_id
    join jarvis_task_grants attached on attached.owner_id=a.owner_id and attached.task_id=a.task_id and attached.grant_id=a.grant_id
    join jarvis_operation_grants g on g.owner_id=a.owner_id and g.id=a.grant_id
    join jarvis_context_payloads s on s.owner_id=g.owner_id and s.id=g.scope_payload_id
    join jarvis_context_payloads p on p.owner_id=a.owner_id and p.id=a.receipt_payload_id
    join jarvis_task_grants parent_attached on parent_attached.owner_id=t.owner_id and parent_attached.task_id=t.id
        and parent_attached.grant_id::text=s.content->>'parent_grant_id'
    join jarvis_operation_grants parentg on parentg.owner_id=parent_attached.owner_id and parentg.id=parent_attached.grant_id
        and parentg.capability='files.publish_generated'
    join jarvis_context_payloads parentp on parentp.owner_id=parentg.owner_id and parentp.id=parentg.scope_payload_id
    join jarvis_files f on f.owner_id=t.owner_id and f.id::text=$6->>'file_id'
    join jarvis_file_publications publication on publication.owner_id=f.owner_id and publication.file_id=f.id
        and publication.area_id=t.knowledge_area_id and publication.operation_key='task-generated-file:' || a.id::text
    join jarvis_context_payloads metadata on metadata.owner_id=publication.owner_id and metadata.id=publication.metadata_payload_id
    join jarvis_knowledge_sources source on source.owner_id=t.owner_id and source.area_id=t.knowledge_area_id
        and source.namespace='product:files' and source.source_key=f.id::text
    join jarvis_knowledge_source_versions v on v.owner_id=source.owner_id and v.source_id=source.id
        and v.revision=source.current_revision and v.document_id=$6->>'version_id'
    join jarvis_context_payloads original on original.owner_id=v.owner_id and original.id=v.payload_id
    join jarvis_knowledge_areas ar on ar.owner_id=t.owner_id and ar.id=t.knowledge_area_id
    where a.owner_id=$1 and a.task_id=$3 and a.id=$4 and a.status='accepted' and a.submission_token is not null
        and t.knowledge_area_id=$2 and ar.status='active' and source.status='active'
        and g.capability='files.publish_snapshot' and a.parameters_payload_id=g.scope_payload_id
        and a.operation_key='tool-grant:' || replace(g.id::text,'-','')
        and p.content->>'parameters_digest'=a.parameters_digest and p.content->>'tool'='publish_generated_file'
        and p.content#>>'{result,status}'='success' and p.content#>>'{result,file,id}'=$6->>'file_id'
        and p.content#>>'{result,file,memory_source,document_id}'=$6->>'version_id'
        and s.content->>'task_id'=$3::text and s.content#>>'{request,format}'=$7
        and s.content#>>'{inspection,format}'=$7
        and s.content#>>'{request,operation_key}'=(case when $7='json' then 'media-recipe:' else 'media-output:' end) || $5
        and s.content#>>'{inspection,content_sha256}'=$6->>'content_sha256'
        and s.content#>>'{inspection,size_bytes}'=f.size_bytes::text
        and f.filename=s.content#>>'{request,filename}' and f.description=s.content#>>'{request,description}'
        and f.filename='edit-' || replace($8::text,'-','') || '-' || left($5,12) || '.' || $7
        and f.content_type=(case when $7='json' then 'application/json' else 'video/mp4' end)
        and f.metadata#>>'{task_artifact,task_id}'=$3::text and f.metadata#>>'{task_artifact,action_id}'=$4::text
        and f.metadata#>>'{task_artifact,area_id}'=$2::text
        and f.metadata#>>'{generated_artifact,operation_key}'=s.content#>>'{request,operation_key}'
        and f.metadata#>>'{generated_artifact,evidence_payload_id}'=s.content->>'evidence_payload_id'
        and original.content->>'content_sha256'=$6->>'content_sha256'
        and original.content#>>'{record,id}'=f.id::text
        and publication.content_sha256=$6->>'content_sha256' and publication.size_bytes=f.size_bytes
        and publication.storage_bucket=f.storage_bucket and publication.storage_path=f.storage_path
        and publication.phase='available' and publication.upload_settled and not publication.delete_requested
        and original.availability='available' and metadata.availability='available'
        and s.availability='available' and p.availability='available' and parentp.availability='available'
        and ($7<>'json' or ($6->>'content_sha256'=$5 and f.size_bytes<=131072))
    $media_proof$;
begin
    if kind not in ('json','mp4') or plan !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(ref) is distinct from 'object'
        or not(ref ?& array['file_id','version_id','content_sha256'])
        or ref-array['file_id','version_id','content_sha256']<>'{}'::jsonb then return null; end if;
    execute lookup_sql || (case when lock_rows then
        ' for share of a,t,g,s,p,parentg,parentp,f,publication,metadata,source,v,original,ar' else '' end)
        into result using o,area,task,action,plan,ref,kind,edit;
    return result;
end;
$$;

create function jarvis_media_correction_lineage(o text, capture uuid, recipe uuid, output uuid) returns boolean
language sql set search_path from current as $$
    with recursive ancestors(root,id) as (
        values(recipe,recipe),(output,output)
        union
        select a.root,d.source_payload_id from ancestors a
        join jarvis_context_payload_dependencies d on d.owner_id=o and d.dependent_payload_id=a.id
    )
    select exists(select 1 from ancestors where root=recipe and id=capture)
       and exists(select 1 from ancestors where root=output and id=recipe)
       and not exists(select 1 from ancestors a left join jarvis_context_payloads p on p.owner_id=o and p.id=a.id
                      where p.id is null or p.availability<>'available');
$$;

create or replace function jarvis_media_command_grant_valid(g jarvis_operation_grants) returns boolean
language plpgsql set search_path from current as $$
declare
    e record; t record; r record; parent record;
    b jsonb; c jsonb; q jsonb; pin jsonb; scope jsonb; original_pin jsonb;
    capture_session text; recording_key text;
begin
    if g.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
        or g.request_revision<>1 or g.scope_revision<>1 or g.status<>'active'
        or g.expires_at<=clock_timestamp() then return false; end if;
    select ce.*,p.content as body into e from jarvis_context_events ce
    join jarvis_context_payloads p on p.owner_id=ce.owner_id and p.id=ce.payload_id
    where ce.owner_id=g.owner_id and ce.id=g.request_event_id and ce.revision=1
        and ce.source_namespace='voice_command' and ce.kind='command_request' and ce.trust='derived'
        and ce.finality='final' and ce.session_id is not null and p.availability='available'
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ce.owner_id
            and n.source_namespace=ce.source_namespace and n.source_event_id=ce.source_event_id and n.revision>ce.revision)
    for share of ce,p;
    if not found then return false; end if;
    b:=e.body; c:=b->'command'; q:=b->'request'; pin:=b->'media_input';
    if jsonb_typeof(b) is distinct from 'object'
        or not (b ?& array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input'])
        or b-array['format','command','operation','task_id','base_revision','area_id','request','mode','parameters_digest','media_input']<>'{}'::jsonb
        or b->>'format' is distinct from 'jarvis.task_command.v1' or b->>'operation' is distinct from 'create'
        or b->'base_revision' is distinct from '0'::jsonb
        or b->'mode' is distinct from '"background"'::jsonb
        or coalesce(b->>'parameters_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c) is distinct from 'object'
        or not(c ?& array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments'])
        or c-array['owner_id','session_id','provider_session_key','provider_call_id','operation_id','tool_name','capability','arguments_digest','arguments']<>'{}'::jsonb
        or c->'owner_id' is distinct from to_jsonb(g.owner_id) or c->>'session_id' is distinct from e.session_id::text
        or c->>'operation_id' is distinct from e.source_event_id or c->>'tool_name' is distinct from 'start_media_task'
        or c->>'capability' is distinct from 'tasks.create'
        or jsonb_typeof(c->'provider_session_key') is distinct from 'string'
        or jsonb_typeof(c->'provider_call_id') is distinct from 'string'
        or coalesce(c->>'provider_session_key','') !~ '[^[:space:]]'
        or coalesce(c->>'provider_call_id','') !~ '[^[:space:]]'
        or coalesce(c->>'arguments_digest','') !~ '^[0-9a-f]{64}$'
        or jsonb_typeof(c->'arguments') is distinct from 'object'
        or jsonb_typeof(c#>'{arguments,instruction}') is distinct from 'string'
        or coalesce(length(c#>>'{arguments,instruction}'),0) not between 1 and 8000
        or coalesce(c#>>'{arguments,instruction}','') !~ '[^[:space:]]'
        or jsonb_typeof(q) is distinct from 'object'
        or not(q ?& array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids'])
        or q-array['request_id','title','requested_outcome','session_id','knowledge_area_id','backend_kind','execution_environment','publish_artifacts','edit_memory','edit_note','grant_ids']<>'{}'::jsonb
        or q->>'request_id' is distinct from e.source_event_id or q->>'session_id' is distinct from e.session_id::text
        or q->>'requested_outcome' is distinct from c#>>'{arguments,instruction}'
        or q->>'backend_kind' is distinct from 'frontier' or q->>'execution_environment' is distinct from 'offline_sandbox'
        or q->'publish_artifacts' is distinct from 'true'::jsonb or q->'edit_note' is distinct from 'false'::jsonb
        or q->'edit_memory' is distinct from 'false'::jsonb or q->'grant_ids' is distinct from '[]'::jsonb
        or q->>'knowledge_area_id' is distinct from b->>'area_id' then return false; end if;
    if jsonb_typeof(pin) is distinct from 'object' then return false; end if;
    if not(pin ? 'profile') then
        -- Original 0049 profile: same-session recording and exact original shape.
        if (c->'arguments')-array['instruction','recording_id','saved']<>'{}'::jsonb then return false; end if;
        capture_session:=e.session_id::text;
        recording_key:=c#>>'{arguments,recording_id}';
        original_pin:=pin;
    else
        if jsonb_typeof(pin) is distinct from 'object'
            or not(pin ?& array['profile','original','capture_session_id','command_session_id'])
            or pin->>'command_session_id' is distinct from e.session_id::text
            or pin#>>'{original,session_id}' is distinct from pin->>'capture_session_id' then return false; end if;
        if pin->>'profile'='retained_recording.v1' then
            if pin-array['profile','original','capture_session_id','command_session_id']<>'{}'::jsonb
                or (c->'arguments')-array['instruction','recording_id','saved','recording_profile']<>'{}'::jsonb
                or c#>>'{arguments,recording_profile}' is distinct from 'retained_recording.v1'
                or c#>>'{arguments,recording_id}' is distinct from pin#>>'{original,recording_id}'
                or c#>'{arguments,saved}' is distinct from ((pin->'original')-array['owner_id','session_id','recording_id',
                    'area_id','source_payload_id','receipt_payload_id','admission_payload_id','content_sha256','size_bytes'])
                then return false; end if;
        elsif pin->>'profile'='accepted_output.v1' then
            if not(pin ?& array['target','previous_recipe','edit_id','recipe_action_id','recipe_proof','output_proof'])
                or pin-array['profile','original','capture_session_id','command_session_id','target','previous_recipe',
                             'edit_id','recipe_action_id','recipe_proof','output_proof']<>'{}'::jsonb
                or (c->'arguments')-array['instruction','accepted_output']<>'{}'::jsonb
                or c#>'{arguments,accepted_output}' is distinct from pin->'target'
                or jsonb_typeof(pin->'target') is distinct from 'object'
                or not((pin->'target') ?& array['task_id','action_id','render_plan_sha256','file'])
                or (pin->'target')-array['task_id','action_id','render_plan_sha256','file']<>'{}'::jsonb
                or pin#>>'{previous_recipe,content_sha256}' is distinct from pin#>>'{target,render_plan_sha256}'
                or pin->>'recipe_action_id'=pin#>>'{target,action_id}' then return false; end if;
        else return false; end if;
        capture_session:=pin->>'capture_session_id';
        recording_key:=pin#>>'{original,recording_id}';
        original_pin:=pin->'original';
    end if;
    select bt.*,d.content as description into t from jarvis_background_tasks bt
    join jarvis_task_requests tr on tr.owner_id=bt.owner_id and tr.task_id=bt.id
    join jarvis_context_payloads d on d.owner_id=bt.owner_id and d.id=bt.description_payload_id
    where bt.owner_id=g.owner_id and bt.id::text=b->>'task_id' and bt.session_id=e.session_id
        and bt.backend_kind='frontier' and bt.knowledge_area_id::text=b->>'area_id'
        and tr.request_id::text=e.source_event_id and tr.operation='create'
        and tr.accepted_revision=1 and tr.accepted_revision<=bt.revision and tr.parameters_digest=b->>'parameters_digest'
        and d.availability='available' and d.content->>'requested_outcome'=q->>'requested_outcome'
        and d.content->>'title'=q->>'title' and d.content->'mode'=b->'mode'
        and d.content->'media_input'=pin
        and d.content->'instruction_request'=jsonb_build_object('event_id',e.id::text,'revision',1,'session_id',bt.session_id::text)
        and d.content#>>'{managed_plugin,format}'='jarvis.managed-media.v1'
        and d.content#>>'{managed_plugin,binding,owner_id}'=g.owner_id
        and d.content#>>'{managed_plugin,binding,plugin_id}'='video_editor'
        and d.content#>>'{managed_plugin,binding,configuration_id}'=d.content#>>'{managed_plugin,bundle,configuration_id}'
    for share of bt,d;
    if not found then return false; end if;
    select rec.*,v.payload_id as source_payload_id,fp.content_sha256,fp.size_bytes into r
    from jarvis_recordings rec
    join jarvis_knowledge_areas a on a.owner_id=rec.owner_id and a.id=rec.area_id
    join jarvis_knowledge_source_versions v on v.owner_id=rec.owner_id and v.document_id=rec.version_id
        and v.source_id=rec.source_id and v.area_id=rec.area_id
    join jarvis_knowledge_sources s on s.owner_id=v.owner_id and s.id=v.source_id
    join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
    join jarvis_context_payloads cp on cp.owner_id=rec.owner_id and cp.id=rec.completion_payload_id
    join jarvis_context_events ce on ce.owner_id=rec.owner_id and ce.id=rec.completion_event_id and ce.payload_id=cp.id
    join jarvis_context_payloads ap on ap.owner_id=rec.owner_id and ap.id=rec.admission_payload_id
    join jarvis_context_events ae on ae.owner_id=rec.owner_id and ae.id=rec.admission_event_id and ae.payload_id=ap.id
    join jarvis_file_publications fp on fp.owner_id=rec.owner_id and fp.id=rec.publication_id
        and fp.file_id=rec.file_id and fp.area_id=rec.area_id
    join jarvis_context_payloads fpp on fpp.owner_id=fp.owner_id and fpp.id=fp.metadata_payload_id
    where rec.owner_id=g.owner_id and rec.session_id::text=capture_session and rec.area_id=t.knowledge_area_id
        and rec.recording_id::text=recording_key and not rec.revoked and rec.capture_closed
        and a.status='active' and s.status='active' and v.revision=s.current_revision
        and p.availability='available' and cp.availability='available' and ap.availability='available'
        and fpp.availability='available' and fp.phase='available' and fp.upload_settled and not fp.delete_requested
        and fp.operation_key='recording:' || rec.recording_id::text and fp.transfer_policy='mp4_capture_v1'
        and p.content->>'recording_id'=rec.recording_id::text and p.content->>'content_sha256'=fp.content_sha256
        and not exists(select 1 from jarvis_context_events n where n.owner_id=ae.owner_id
            and n.source_namespace=ae.source_namespace and n.source_event_id=ae.source_event_id and n.revision>ae.revision)
    for share of rec,a,v,s,p,cp,ap,fp,fpp,ce,ae;
    if not found then return false; end if;
    if not(pin ? 'profile') and c#>'{arguments,saved}' is distinct from
        jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text) then return false; end if;
    if original_pin is distinct from jsonb_build_object('file_id',r.file_id::text,'source_id',r.source_id::text,
            'version_id',r.version_id,'receipt_event_id',r.completion_event_id::text,'owner_id',r.owner_id,
            'session_id',r.session_id::text,'recording_id',r.recording_id::text,'area_id',r.area_id::text,
            'source_payload_id',r.source_payload_id::text,'receipt_payload_id',r.completion_payload_id::text,
            'admission_payload_id',r.admission_payload_id::text,'content_sha256',r.content_sha256,'size_bytes',r.size_bytes)
        then return false; end if;
    if pin->>'profile'='accepted_output.v1' then
        if pin#>>'{target,task_id}'=t.id::text
            or jarvis_media_accepted_proof(g.owner_id,t.knowledge_area_id,(pin#>>'{target,task_id}')::uuid,
                (pin->>'recipe_action_id')::uuid,pin#>>'{target,render_plan_sha256}',pin->'previous_recipe',
                'json',(pin->>'edit_id')::uuid) is distinct from pin->'recipe_proof'
            or jarvis_media_accepted_proof(g.owner_id,t.knowledge_area_id,(pin#>>'{target,task_id}')::uuid,
                (pin#>>'{target,action_id}')::uuid,pin#>>'{target,render_plan_sha256}',pin#>'{target,file}',
                'mp4',(pin->>'edit_id')::uuid) is distinct from pin->'output_proof'
            or not jarvis_media_correction_lineage(g.owner_id,r.source_payload_id,
                (pin#>>'{recipe_proof,source_payload_id}')::uuid,(pin#>>'{output_proof,source_payload_id}')::uuid)
            then return false; end if;
    end if;
    if exists(select 1 from jarvis_task_grants tg join jarvis_operation_grants og
        on og.owner_id=tg.owner_id and og.id=tg.grant_id where tg.owner_id=g.owner_id and tg.task_id=t.id
        and (og.capability not in ('sandbox.execute','files.publish_generated','files.publish_snapshot')
            or og.request_event_id<>e.id or og.request_revision<>1)) then return false; end if;
    select p.content into scope from jarvis_context_payloads p
    where p.owner_id=g.owner_id and p.id=g.scope_payload_id and p.availability='available' for share;
    if not found or jsonb_typeof(scope) is distinct from 'object' then return false; end if;
    if g.capability in ('sandbox.execute','files.publish_generated') then
        if t.revision<>1 or t.status<>'queued' or t.request_event_id<>e.id or t.request_revision<>1
            then return false; end if;
        if g.capability='sandbox.execute' then
            return g.operation_key='managed-video-editor:' || t.id::text
                and scope-array['template','lifetime_seconds','command_seconds','output_bytes','allowed_domains','internet_access']='{}'::jsonb
                and scope->>'template'='base' and scope->'allowed_domains'='[]'::jsonb
                and coalesce(scope->'internet_access','false'::jsonb)='false'::jsonb
                and jsonb_typeof(scope->'lifetime_seconds')='number' and (scope->>'lifetime_seconds')::integer between 60 and 3600
                and jsonb_typeof(scope->'command_seconds')='number' and (scope->>'command_seconds')::integer between 1 and 300
                and jsonb_typeof(scope->'output_bytes')='number' and (scope->>'output_bytes')::integer between 1024 and 256000;
        end if;
        return g.operation_key='artifact-task:' || t.id::text
            and scope-array['max_files','max_file_bytes','max_total_bytes','formats','mp4_max_file_bytes']='{}'::jsonb
            and scope->'formats'='["json","mp4"]'::jsonb and scope->'max_files'='2'::jsonb
            and jsonb_typeof(scope->'max_file_bytes')='number' and (scope->>'max_file_bytes')::bigint between 1 and 8388608
            and jsonb_typeof(scope->'max_total_bytes')='number' and (scope->>'max_total_bytes')::bigint between 1 and 134217728
            and jsonb_typeof(scope->'mp4_max_file_bytes')='number' and (scope->>'mp4_max_file_bytes')::bigint between 1 and 33554432;
    end if;
    if not(scope ?& array['request','inspection','parent_grant_id','task_id','evidence_payload_id'])
        or scope-array['request','inspection','parent_grant_id','task_id','evidence_payload_id','transfer_limit_bytes']<>'{}'::jsonb
        or scope->>'task_id' is distinct from t.id::text
        or scope#>>'{request,format}' not in ('json','mp4')
        or scope#>>'{request,format}' is distinct from scope#>>'{inspection,format}'
        or g.operation_key !~ ('^artifact-snapshot:' || replace(t.id::text,'-','') || ':[0-9a-f]{64}$')
        then return false; end if;
    select og.*,p.content as scope into parent from jarvis_task_grants tg
    join jarvis_operation_grants og on og.owner_id=tg.owner_id and og.id=tg.grant_id
    join jarvis_context_payloads p on p.owner_id=og.owner_id and p.id=og.scope_payload_id
    where tg.owner_id=g.owner_id and tg.task_id=t.id and og.id::text=scope->>'parent_grant_id'
        and og.capability='files.publish_generated' and og.operation_key='artifact-task:' || t.id::text
        and og.request_event_id=e.id and og.request_revision=1 and og.scope_revision=1
        and og.status='active' and og.expires_at>clock_timestamp() and p.availability='available' for share of og,p;
    if not found or g.expires_at>parent.expires_at then return false; end if;
    return exists(select 1 from jarvis_context_payloads p where p.owner_id=g.owner_id
        and p.id::text=scope->>'evidence_payload_id' and p.availability='available');
end;
$$;
insert into jarvis_schema_migrations (version, name, checksum) values (50, 'media_correction_commands', 'a166af398aca1fb142a9c079f91f575799413edbb14855e95e3395abe3c4956b');

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
insert into jarvis_schema_migrations (version, name, checksum) values (51, 'recording_capture_limits', 'a9516ae313022ac2f9ea8820ea4b4e5c4590454bc5ee6b89754e979427eecb6a');

-- Opt-in reuse on the existing allocation ledger. No extra workspace/transcript store.
-- Native metadata stays immutable while a confirmed-paused allocation changes holder.
alter table jarvis_sandbox_allocations
    add column reusable boolean not null default false,
    add column native_scope jsonb;
update jarvis_sandbox_allocations set native_scope=jsonb_build_object(
    'owner_id',owner_id,'task_id',task_id,'task_revision',task_revision,
    'attempt_id',attempt_id,'allocation_id',id);
alter table jarvis_sandbox_allocations alter column native_scope set not null;
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_allocations_phase_check;
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_allocations_phase_check
    check(phase in ('reserved','creating','uncertain','preparing','ready','pausing','paused','resuming','cleanup_requested','cleaned'));
alter table jarvis_sandbox_allocations drop constraint jarvis_sandbox_cleanup_claim_phase;
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_cleanup_claim_phase
    check(cleanup_token is null or phase in ('cleanup_requested','pausing'));
alter table jarvis_sandbox_allocations add constraint jarvis_sandbox_continuity_handle
    check(phase not in ('pausing','paused','resuming') or
        (reusable and sandbox_id is not null and submission_token is not null));

create function initialize_jarvis_sandbox_native_scope() returns trigger language plpgsql as $$
begin
    new.native_scope := jsonb_build_object('owner_id',new.owner_id,'task_id',new.task_id,
        'task_revision',new.task_revision,'attempt_id',new.attempt_id,'allocation_id',new.id);
    return new;
end; $$;
create trigger jarvis_sandbox_native_scope before insert on jarvis_sandbox_allocations
    for each row execute function initialize_jarvis_sandbox_native_scope();

create or replace function guard_jarvis_sandbox_ownership() returns trigger language plpgsql as $$
begin
    if row(new.id,new.owner_id,new.policy,new.policy_digest,new.created_at,new.native_scope,new.reusable)
       is distinct from row(old.id,old.owner_id,old.policy,old.policy_digest,old.created_at,old.native_scope,old.reusable)
       or new.deadline>old.deadline then
        raise exception 'Sandbox identity and policy cannot change or extend' using errcode='23514';
    end if;
    if row(new.task_id,new.task_revision,new.attempt_id,new.grant_id)
       is distinct from row(old.task_id,old.task_revision,old.attempt_id,old.grant_id)
       and not (old.reusable and old.phase='paused' and new.phase='resuming'
                and old.cleanup_token is null and old.deadline>clock_timestamp()) then
        raise exception 'Sandbox holder can move only from confirmed pause' using errcode='23514';
    end if;
    if old.submission_token is not null and
       row(new.submission_token,new.submission_fence) is distinct from row(old.submission_token,old.submission_fence) then
        raise exception 'Sandbox submission identity is immutable' using errcode='23514';
    end if;
    if old.sandbox_id is not null and row(new.sandbox_id,new.template_id) is distinct from row(old.sandbox_id,old.template_id) then
        raise exception 'Sandbox native identity is immutable' using errcode='23514';
    end if;
    if not old.cleanup_observed_ids <@ new.cleanup_observed_ids
       or (old.phase in ('cleanup_requested','cleaned') and new.phase not in ('cleanup_requested','cleaned')) then
        raise exception 'Sandbox cleanup evidence cannot be erased or execution restored' using errcode='23514';
    end if;
    return new;
end; $$;
insert into jarvis_schema_migrations (version, name, checksum) values (52, 'sandbox_continuity', '00e507a83659d5b3e2ff854dea5614310df7d0749af47fc662b8a324fa4d2e6c');

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
insert into jarvis_schema_migrations (version, name, checksum) values (53, 'file_previews', '0189d8082269d5723cb7b7c4495fc86e80f190e40f44d570d2a0f8df22b7b277');

-- Widen new capture admissions only. Existing grants and recordings retain
-- their original limits; document and generated-output policies are unchanged.
alter table jarvis_file_publications
    drop constraint jarvis_file_publications_transfer_policy_check,
    add constraint jarvis_file_publications_transfer_policy_check check (
        (transfer_policy = 'document_v1' and transfer_limit_bytes between 1 and 8388608)
        or (transfer_policy = 'mp4_capture_v1' and transfer_limit_bytes between 1 and 1073741824)
        or (transfer_policy = 'mp4_output_v1' and transfer_limit_bytes between 1 and 33554432)
    );

alter table jarvis_recordings
    drop constraint jarvis_recordings_max_bytes_check,
    add constraint jarvis_recordings_max_bytes_check check (max_bytes between 1 and 1073741824),
    drop constraint jarvis_recordings_max_duration_ms_check,
    add constraint jarvis_recordings_max_duration_ms_check check (max_duration_ms between 1 and 1800000);
insert into jarvis_schema_migrations (version, name, checksum) values (54, 'recording_thirty_minute_limits', '67d8a0cb0ae04769a3fb399fc4c02dada958a22b1ae6fbf03e465070464b701c');

-- Widen publication capacity for newly admitted edited MP4s only. Previously
-- saved transfer limits remain attached to their original publications.
alter table jarvis_file_publications
    drop constraint jarvis_file_publications_transfer_policy_check,
    add constraint jarvis_file_publications_transfer_policy_check check (
        (transfer_policy = 'document_v1' and transfer_limit_bytes between 1 and 8388608)
        or (transfer_policy = 'mp4_capture_v1' and transfer_limit_bytes between 1 and 1073741824)
        or (transfer_policy = 'mp4_output_v1' and transfer_limit_bytes between 1 and 500000000)
    );
insert into jarvis_schema_migrations (version, name, checksum) values (55, 'edited_video_limits', 'f6e84df5be31b74c42c722d58c4a906a61abbe4d5cf9e30bfe2ca905bc8c8c5f');
