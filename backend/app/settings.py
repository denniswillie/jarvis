from __future__ import annotations

import json
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PLACEHOLDER_VALUES = {"", "replace_me", "changeme", "todo", "none", "null"}


def is_configured_secret(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in PLACEHOLDER_VALUES


class JarvisSettings(BaseSettings):
    """Runtime settings for the Jarvis backend.

    Only core process settings are mandatory at startup. Feature-specific
    settings are validated by the feature that uses them.
    """

    APP_ENV: str = "development"
    # Operator-controlled outside database backups. Keep enabled throughout a
    # restore until journal replay and the documented release checks complete.
    JARVIS_RESTORE_IN_PROGRESS: bool = False
    APP_NAME: str = "Jarvis"
    PUBLIC_BASE_URL: str = "http://localhost:9000"
    CORS_ALLOW_ORIGINS: list[str] = Field(default_factory=list)
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False
    JARVIS_OWNER_ID: str = "default"
    JARVIS_STATIC_BEARER_TOKEN: str | None = None
    JARVIS_MAX_TOOL_CALLS_PER_RESPONSE: int = 12
    JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE: int = 3
    JARVIS_REALTIME_MAX_INTERRUPT_CHAIN: int = 2
    MODEL_PROVIDER_MAX_CONCURRENCY: int | None = Field(default=None, ge=1)

    SUPABASE_URL: str | None = None
    SUPABASE_SERVICE_ROLE_KEY: str | None = None
    SUPABASE_DB_URL: str | None = None
    SUPABASE_DB_POOL_MIN_SIZE: int = Field(default=1, ge=1)
    SUPABASE_DB_POOL_MAX_SIZE: int = Field(default=5, ge=1)
    SUPABASE_DB_COMMAND_TIMEOUT_SECONDS: float = Field(default=30, gt=0)
    SUPABASE_DB_CONNECT_TIMEOUT_SECONDS: float = Field(default=10, gt=0)
    SUPABASE_STORAGE_BUCKET: str = "jarvis-visual-frames"
    JARVIS_FILES_STORAGE_BUCKET: str = "jarvis-files"

    # Resolve provider values at audio admission, not during process startup.
    AUDIO_PROVIDER_DEFAULT: str = "openai"
    # Requires a matching native-client capability offer; legacy stays unchanged.
    RESPONSIVE_VOICE_ENABLED: bool = False
    # Explicit lean session opt-in; ordinary turns belong to the native provider.
    LEAN_VOICE_ENABLED: bool = False
    # Explicit per-session opt-in only. Invalid unused Live flags do not affect
    # the existing Realtime path; admission validates this selected branch.
    OPENAI_LIVE_ENABLED: bool | str = False
    OPENAI_LIVE_MODEL: str | None = "gpt-live-1"
    OPENAI_LIVE_VOICE: str | None = "vesper"
    OPENAI_API_KEY: str | None = None
    OPENAI_REALTIME_MODEL: str | None = None
    OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    OPENAI_REALTIME_VOICE: str | None = "cedar"
    OPENAI_REALTIME_TRANSCRIPTION_MODEL: str | None = "gpt-realtime-whisper"
    OPENAI_REMINDER_MODEL: str | None = "gpt-5.6"
    OPENAI_CONSOLIDATION_MODEL: str | None = "gpt-5.6"
    OPENAI_CONSOLIDATION_TIMEOUT_SECONDS: float = Field(default=60, gt=0)
    JARVIS_AUDIO_SAMPLE_RATE: int = 24_000
    JARVIS_AUDIO_FORMAT: str = "pcm16"
    JARVIS_REALTIME_MAX_SESSION_SECONDS: int = 3_300
    JARVIS_REALTIME_MAX_CLIENT_MESSAGE_BYTES: int = 1_000_000
    JARVIS_REALTIME_MAX_AUDIO_CHUNK_BYTES: int = 262_144
    JARVIS_REALTIME_RECENT_MESSAGES_LIMIT: int = 12
    PHONE_CAMERA_DEBUG_ENABLED: bool = False
    VISUAL_RAW_FRAME_TTL_SECONDS: int = 300
    VISUAL_FRAME_MAX_BYTES: int = 512_000
    VISUAL_FRAME_MAX_WIDTH: int = 1_024
    VISUAL_FRAME_MAX_HEIGHT: int = 1_024
    VISUAL_LATEST_FRAME_TTL_SECONDS: int = 10
    VISUAL_LATEST_CAPTURE_MAX_AGE_SECONDS: int = 10
    VISUAL_SENSORY_WINDOW_MAX_FRAMES: int = Field(default=4, ge=1, le=16)
    VISUAL_SENSORY_WINDOW_TTL_SECONDS: int = Field(default=10, ge=1)
    VISUAL_REALTIME_ENABLED: bool = True
    VISUAL_REALTIME_MIN_INTERVAL_SECONDS: float = Field(
        default=1.0, gt=0, allow_inf_nan=False
    )
    VISUAL_REALTIME_MAX_CONTEXT_AGE_SECONDS: float = Field(
        default=5.0, gt=0, allow_inf_nan=False
    )
    VISUAL_REALTIME_SYNC_TIMEOUT_SECONDS: float = Field(
        default=0.5, gt=0, allow_inf_nan=False
    )
    VISUAL_REALTIME_MAX_IMAGE_ITEMS: int = Field(default=4, ge=4)
    GEMINI_API_KEY: str | None = None
    # Gemini audio is parked; Gemini visual analysis remains independently enabled.
    # Keep raw environment strings so an invalid unused Live flag cannot break GPT.
    # The selected Gemini branch performs boolean validation.
    GEMINI_LIVE_ENABLED: bool | str = False
    GEMINI_LIVE_MODEL: str | None = "gemini-3.1-flash-live-preview"
    GEMINI_LIVE_VOICE: str = "Puck"
    # Parse only in the Gemini branch so malformed unused Live configuration is inert.
    GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS: float | str = 0.75
    GEMINI_VISUAL_MODEL: str | None = "gemini-3.1-flash-lite"
    GEMINI_MEMORY_EMBEDDING_MODEL: str = "gemini-embedding-2"
    MEMORY_EMBEDDING_DIMENSIONS: int = Field(default=1_536, ge=1)
    MEMORY_EMBEDDING_TIMEOUT_SECONDS: float = Field(default=20, gt=0)
    MEMORY_RETRIEVAL_CANDIDATE_LIMIT: int = Field(default=40, ge=1, le=200)
    MEMORY_RETRIEVAL_RESULT_LIMIT: int = Field(default=12, ge=1, le=50)
    MEMORY_RETRIEVAL_DEFAULT_SECONDS: int = Field(default=600, ge=1)
    MEMORY_RETRIEVAL_MAX_DAYS: int = Field(default=3_650, ge=1)
    VISUAL_IMAGE_RETRIEVAL_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0)
    MEMORY_REINSPECTION_MAX_KEYFRAMES: int = Field(default=6, ge=1, le=6)
    MEMORY_JOB_RUNNER_ENABLED: bool = True
    # Keep serving-only releases from implicitly starting maintenance loops.
    MEMORY_VISUAL_FRAME_RETENTION_ENABLED: bool = True
    EXPERIENCE_EVENT_SWEEPER_ENABLED: bool = True
    MEMORY_JOB_POLL_INTERVAL_SECONDS: float = Field(default=2, gt=0)
    MEMORY_JOB_MAX_CONCURRENCY: int = Field(default=2, ge=1)
    MEMORY_JOB_CLAIM_TIMEOUT_SECONDS: float = Field(default=120, gt=0)
    MEMORY_JOB_DEFAULT_MAX_ATTEMPTS: int = Field(default=3, ge=1)
    MEMORY_JOB_RECONCILE_INTERVAL_SECONDS: float = Field(default=60, gt=0)
    MEMORY_MAINTENANCE_ENABLED: bool = True
    MEMORY_MAINTENANCE_INTERVAL_SECONDS: int = Field(default=300, ge=1)
    MEMORY_VISUAL_FRAME_EXPIRY_INTERVAL_SECONDS: int = Field(default=60, ge=1)
    MEMORY_MAINTENANCE_BATCH_SIZE: int = Field(default=500, ge=1, le=5_000)
    MEMORY_OBSERVATION_DETAIL_RETENTION_DAYS: int = Field(default=7, ge=1)
    MEMORY_UNREFERENCED_OBSERVATION_RETENTION_DAYS: int = Field(default=30, ge=1)
    MEMORY_UNREFERENCED_VISUAL_FRAME_RETENTION_DAYS: int = Field(default=30, ge=1)
    MEMORY_COMPLETED_JOB_RETENTION_DAYS: int = Field(default=14, ge=1)
    MEMORY_RETRIEVAL_AUDIT_RETENTION_DAYS: int = Field(default=30, ge=1)
    MEMORY_CONSOLIDATION_MAX_KEYFRAMES: int = Field(default=6, ge=1, le=6)
    MEMORY_CONSOLIDATION_MAX_TRANSCRIPTS: int = Field(default=20, ge=1)
    MEMORY_EPISODE_PROMOTION_MIN_IMPORTANCE: float = Field(
        default=0.55,
        ge=0,
        le=1,
    )
    MEMORY_IMPORTANT_KEYFRAME_MIN_IMPORTANCE: float = Field(
        default=0.80,
        ge=0,
        le=1,
    )
    MEMORY_SKIPPED_EVENT_RETENTION_DAYS: int = Field(default=90, ge=1)
    MEMORY_SEMANTIC_INFERENCE_MIN_CONFIDENCE: float = Field(
        default=0.85, ge=0, le=1
    )
    MEMORY_SEMANTIC_SUPERSEDE_MIN_CONFIDENCE: float = Field(
        default=0.90, ge=0, le=1
    )
    MEMORY_REFLECTION_MIN_NEW_EPISODES: int = Field(default=8, ge=1)
    MEMORY_REFLECTION_MAX_EPISODES: int = Field(default=30, ge=1)
    MEMORY_INFERRED_PREFERENCE_MIN_SUPPORTING_EPISODES: int = Field(
        default=3, ge=1
    )
    MEMORY_INFERRED_PREFERENCE_MIN_CONFIDENCE: float = Field(
        default=0.85, ge=0, le=1
    )
    MEMORY_PROCEDURE_MIN_SUPPORTING_EPISODES: int = Field(default=3, ge=1)
    MEMORY_PROFILE_MAX_CHARACTERS: int = Field(
        default=16_000, ge=2_000, le=100_000
    )
    MEMORY_PROFILE_SECTION_ITEM_LIMIT: int = Field(default=20, ge=1, le=100)
    MEMORY_PROFILE_RECENT_EPISODE_LIMIT: int = Field(default=12, ge=1, le=100)
    MEMORY_PROFILE_SUPERSEDED_HISTORY_LIMIT: int = Field(
        default=10, ge=1, le=100
    )
    MEMORY_PROFILE_RECOVERY_BATCH_SIZE: int = Field(default=100, ge=1, le=1_000)
    VISUAL_ANALYZER_MAX_CONCURRENCY: int = 6
    VISUAL_ANALYZER_MAX_PENDING_FRAMES: int = 8
    VISUAL_FAST_PERCEPTION_ENABLED: bool = True
    VISUAL_FAST_PERCEPTION_MAX_WINDOW_FRAMES: int = Field(default=4, ge=1, le=16)
    VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS: int = Field(default=20, ge=1)
    WORKING_MEMORY_TTL_SECONDS: int = Field(default=120, ge=1)
    WORKING_MEMORY_ENTITY_TTL_SECONDS: int = Field(default=30, ge=1)
    WORKING_MEMORY_VISIBLE_TEXT_TTL_SECONDS: int = Field(default=60, ge=1)
    WORKING_MEMORY_MAX_ENTITIES: int = Field(default=30, ge=1)
    WORKING_MEMORY_MAX_RECENT_CHANGES: int = Field(default=12, ge=1)
    PROSPECTIVE_FIND_ONCE_DEFAULT_TTL_SECONDS: int = Field(default=900, ge=1)
    PROSPECTIVE_MONITOR_DEFAULT_TTL_SECONDS: int = Field(default=3_600, ge=1)
    PROSPECTIVE_TRIGGER_CLAIM_LEASE_SECONDS: int = Field(default=120, ge=1)
    PROSPECTIVE_MAX_CONDITIONAL_REMINDERS: int = Field(default=12, ge=1)
    PROSPECTIVE_MAX_GOALS: int = Field(default=8, ge=1)
    PROSPECTIVE_MAX_COMMITMENTS: int = Field(default=8, ge=1)
    PROSPECTIVE_MAX_CANDIDATES_PER_ANALYSIS: int = Field(default=3, ge=1, le=8)
    PROSPECTIVE_MAX_EVIDENCE_AGE_SECONDS: int = Field(default=10, ge=1)
    PERCEPTION_EVENT_IDLE_CLOSE_SECONDS: int = Field(default=12, ge=1)
    PERCEPTION_EVENT_MAX_SECONDS: int = Field(default=120, ge=1)
    PERCEPTION_EVENT_SWEEP_INTERVAL_SECONDS: int = Field(default=5, ge=1)
    EVENT_KEYFRAME_MAX_COUNT: int = Field(default=3, ge=1, le=3)
    EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT: int = Field(default=6, ge=1)
    EVENT_KEYFRAME_CANDIDATE_MAX_COUNT: int = Field(default=12, ge=1)
    EVENT_CANDIDATE_RETENTION_SECONDS: int = Field(default=1_800, ge=1)
    EVENT_ORDINARY_KEYFRAME_RETENTION_SECONDS: int = Field(default=86_400, ge=1)
    VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD: float = 0.85
    VISUAL_INTERRUPT_DEDUPE_SECONDS: int = 10
    REMINDER_SCHEDULER_ENABLED: bool = True
    REMINDER_SCHEDULER_INTERVAL_SECONDS: int = 10
    REMINDER_TRIGGER_CLAIM_TIMEOUT_SECONDS: int = 120
    REMINDER_TRIGGER_MAX_ATTEMPTS: int = 3
    GOOGLE_OAUTH_CLIENT_ID: str | None = None
    GOOGLE_OAUTH_CLIENT_SECRET: str | None = None
    GOOGLE_OAUTH_REDIRECT_URI: str = "https://your-jarvis.example.invalid/oauth/google/callback"
    GOOGLE_OAUTH_STATE_SECRET: str | None = None
    GOOGLE_CALENDAR_SCOPES: list[str] = Field(
        default_factory=lambda: ["https://www.googleapis.com/auth/calendar"]
    )
    JARVIS_TIMEZONE: str = "Europe/Dublin"
    WEB_FETCH_MAX_BYTES: int = 2_000_000
    WEB_FETCH_TIMEOUT_SECONDS: int = 10
    WEB_FETCH_MAX_REDIRECTS: int = 3
    WEB_FETCH_CACHE_TTL_SECONDS: int = 900
    OPENAI_WEB_SEARCH_MODEL: str | None = "gpt-5.6"
    WEB_SEARCH_TIMEOUT_SECONDS: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @field_validator("CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("CORS_ALLOW_ORIGINS JSON value must be a list")
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        raise ValueError("CORS_ALLOW_ORIGINS must be a list or comma-separated string")

    @field_validator("JARVIS_TIMEZONE")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("JARVIS_TIMEZONE must be a valid IANA timezone") from error
        return value

    @model_validator(mode="after")
    def validate_database_pool_size(self) -> "JarvisSettings":
        if self.SUPABASE_DB_POOL_MAX_SIZE < self.SUPABASE_DB_POOL_MIN_SIZE:
            raise ValueError(
                "SUPABASE_DB_POOL_MAX_SIZE must be greater than or equal to "
                "SUPABASE_DB_POOL_MIN_SIZE"
            )
        if self.EVENT_KEYFRAME_MAX_COUNT > self.EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT:
            raise ValueError(
                "EVENT_KEYFRAME_MAX_COUNT cannot exceed "
                "EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT"
            )
        if (
            self.EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT
            > self.EVENT_KEYFRAME_CANDIDATE_MAX_COUNT
        ):
            raise ValueError(
                "EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT cannot exceed "
                "EVENT_KEYFRAME_CANDIDATE_MAX_COUNT"
            )
        if (
            self.MEMORY_CONSOLIDATION_MAX_KEYFRAMES
            > self.EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT
        ):
            raise ValueError(
                "MEMORY_CONSOLIDATION_MAX_KEYFRAMES cannot exceed "
                "EVENT_CONSOLIDATION_KEYFRAME_MAX_COUNT"
            )
        if self.MEMORY_EMBEDDING_DIMENSIONS != 1_536:
            raise ValueError("MEMORY_EMBEDDING_DIMENSIONS must be 1536")
        if self.MEMORY_RETRIEVAL_RESULT_LIMIT > self.MEMORY_RETRIEVAL_CANDIDATE_LIMIT:
            raise ValueError(
                "MEMORY_RETRIEVAL_RESULT_LIMIT cannot exceed "
                "MEMORY_RETRIEVAL_CANDIDATE_LIMIT"
            )
        if self.MEMORY_REFLECTION_MAX_EPISODES < self.MEMORY_REFLECTION_MIN_NEW_EPISODES:
            raise ValueError(
                "MEMORY_REFLECTION_MAX_EPISODES must be greater than or equal to "
                "MEMORY_REFLECTION_MIN_NEW_EPISODES"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> JarvisSettings:
    return JarvisSettings()
