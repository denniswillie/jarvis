"""Optional recording retention composition; never implies physical capture.

Reuse the existing selected source/session/task ownership and file storage.
An unavailable optional dependency leaves recording explicitly unavailable;
it does not switch memory providers or start another voice/camera pipeline.
"""
import asyncio

import asyncpg

from app.files.media_inspection import MediaInspectionError
from app.files.media_runtime import prepare_media_inspector
from app.files.recording_ingress import RecordingIngress
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY
from app.migrations.runner import read_schema_readiness
from app.recordings.admission import RecordingAdmission
from app.recordings.source_access import RecordingSourceAccess
from app.knowledge.models import SourceError
from app.runtime.feature_config import Capability, MemoryAuthority, get_second_brain_settings


async def initialize_recording_runtime(services, *, flags=None):
    previous = getattr(services, "recording_source_access", None)
    if previous is not None:
        await previous.close()
    services.recording_source_access = None
    services.recording_admission = None
    services.recording_ingress = None
    services.recording_policy = None
    services.recording_issue = None
    selection = (flags or get_second_brain_settings()).resolve(
        Capability.RECORDINGS, database_url=services.settings.SUPABASE_DB_URL)
    if not selection.enabled:
        services.recording_issue = selection.issue
        return
    if not selection.configured:
        services.recording_issue = selection.issue
        return
    runtime = services.knowledge_runtime
    pool, storage = services.db_pool, services.jarvis_file_storage
    if pool is None or storage is None:
        services.recording_issue = "recording_dependencies_unavailable"
        return
    if services.memory_authority not in (MemoryAuthority.LEGACY, MemoryAuthority.HINDSIGHT):
        services.recording_issue = "recording_scope_configuration_invalid"
        return
    if services.memory_authority == MemoryAuthority.HINDSIGHT:
        if runtime is None:
            services.recording_issue = "recording_dependencies_unavailable"
            return
        sources, tasks = runtime.sources, runtime.context_repository.tasks
        if sources.pool is not pool or tasks.pool is not pool or runtime.area_selections.sources is not sources:
            services.recording_issue = "recording_scope_configuration_invalid"
            return
    try:
        async with asyncio.timeout(10):
            async with pool.acquire() as connection:
                schema = await connection.fetchval("select current_schema()")
                ready = await read_schema_readiness(connection, schema=schema)
                tables = await connection.fetchval("""select
                    to_regclass('jarvis_recordings') is not null and
                    to_regclass('jarvis_recording_requests') is not null""")
        if not ready.ready or not tables:
            services.recording_issue = "recording_schema_not_ready"
            return
    except (asyncpg.PostgresError, asyncpg.InterfaceError, TimeoutError, OSError):
        services.recording_issue = "recording_database_unavailable"
        return
    configuration = selection.configuration
    policy = FileTransferPolicy(MP4_CAPTURE_POLICY, configuration.media.storage_max_file_bytes)
    try:
        inspector = await prepare_media_inspector(configuration.media, policy)
    except MediaInspectionError:
        services.recording_issue = "recording_inspector_not_ready"
        return
    if services.memory_authority == MemoryAuthority.HINDSIGHT:
        selections, access_check = runtime.area_selections, runtime.context_for
    else:
        try:
            async with asyncio.timeout(10):
                access = await RecordingSourceAccess.create(services,
                    area_id=configuration.area_id, policy=policy, inspector=inspector)
        except (SourceError, asyncpg.PostgresError, asyncpg.InterfaceError, TimeoutError, OSError) as error:
            services.recording_issue = error.code if isinstance(error, SourceError) else "recording_database_unavailable"
            return
        services.recording_source_access = access
        sources, tasks = access.sources, access.tasks
        selections, access_check = access.area_selections, access.check
    admissions = RecordingAdmission(sources, tasks, selections,
        owner_id=services.settings.JARVIS_OWNER_ID, access_check=access_check)
    ingress = RecordingIngress(sources, storage, admissions, policy=policy, inspect_file=inspector)
    services.recording_admission = admissions
    services.recording_ingress = ingress
    services.recording_policy = policy
