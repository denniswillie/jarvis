from __future__ import annotations

from dataclasses import dataclass

from app.errors import JarvisHTTPError
from app.settings import JarvisSettings, is_configured_secret


@dataclass(frozen=True)
class SupabaseStorageConfig:
    supabase_url: str
    service_role_key: str
    visual_frame_bucket: str
    files_bucket: str


def get_supabase_storage_config(settings: JarvisSettings) -> SupabaseStorageConfig:
    if not is_configured_secret(settings.SUPABASE_URL) or not is_configured_secret(
        settings.SUPABASE_SERVICE_ROLE_KEY
    ):
        raise JarvisHTTPError(
            503,
            "jarvis_supabase_storage_not_configured",
            "Jarvis Supabase storage configuration is missing.",
        )

    return SupabaseStorageConfig(
        supabase_url=settings.SUPABASE_URL,
        service_role_key=settings.SUPABASE_SERVICE_ROLE_KEY,
        visual_frame_bucket=settings.SUPABASE_STORAGE_BUCKET,
        files_bucket=settings.JARVIS_FILES_STORAGE_BUCKET,
    )
