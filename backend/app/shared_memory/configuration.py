"""Opt-in owner-bound memory composition; no model context injection/backfill."""
import asyncio
import logging
import os
from pathlib import Path

from .client import BankScope, NativeMemory
from .intake import AsyncpgCaptionSource, LiveDialogueIntake


LOGGER = logging.getLogger(__name__)


def configured_memory(*, owner_id, environ=None):
    env = os.environ if environ is None else environ
    if env.get("JARVIS_NATIVE_MEMORY_ENABLED", "false").lower() != "true":
        return None
    generation = env.get("JARVIS_HINDSIGHT_BANK_GENERATION", "")
    key_file = Path(env.get("JARVIS_HINDSIGHT_API_KEY_FILE", ""))
    if not generation or not key_file.is_absolute() or not key_file.is_file():
        raise ValueError("Native memory requires a provisioned bank generation and credential file")
    return NativeMemory(scope=BankScope.for_owner(owner_id, generation=generation),
        base_url=env.get("JARVIS_HINDSIGHT_URL", "http://127.0.0.1:8888"),
        api_key=key_file.read_text().strip())


async def initialize_shared_memory(services, *, environ=None):
    """Called after pool construction. Provisioning/seeding is explicit elsewhere.

    The callback is intentionally resolved at call time: later composition can
    supply the real Live State/reminder invalidator without rebuilding intake.
    Missing invalidation fails honestly and leaves durable pending tombstones.
    """
    if getattr(services, "shared_memory_intake", None) is not None:
        return
    env = os.environ if environ is None else environ
    memory = configured_memory(owner_id=services.settings.JARVIS_OWNER_ID, environ=env)
    services.shared_memory = memory
    if memory is None:
        return
    if services.db_pool is None:
        raise ValueError("Native memory intake requires the canonical caption database")
    default_path = (Path.home() / ".local/share/jarvis/shared-memory"
                    / memory.scope.bank_id / "cursor.sqlite")
    checkpoint = Path(env.get("JARVIS_NATIVE_MEMORY_CHECKPOINT_FILE", str(default_path)))
    if not checkpoint.is_absolute():
        raise ValueError("Native memory checkpoint must be a private absolute path")

    async def invalidate(scope):
        callback = getattr(services, "shared_memory_invalidator", None)
        if callback is None:
            raise RuntimeError("native_memory_invalidator_not_configured")
        await callback(scope)

    intake = LiveDialogueIntake(
        memory, AsyncpgCaptionSource(services.db_pool, services.settings.JARVIS_OWNER_ID),
        checkpoint, invalidate=invalidate,
    )
    services.shared_memory_intake = intake
    await intake.initialize()
    services.shared_memory_stop = asyncio.Event()
    services.shared_memory_task = asyncio.create_task(
        intake.run(services.shared_memory_stop), name="native-caption-memory-intake",
    )

    def completed(task):
        if not task.cancelled() and task.exception() is not None:
            services.shared_memory_issue = intake.issue or "native_memory_intake_stopped"
            LOGGER.error("Native memory intake stopped; canonical source cursor retained")

    services.shared_memory_task.add_done_callback(completed)


async def close_shared_memory(services):
    """Drain after canonical producers, before pool closure, with bounded shutdown."""
    stop = getattr(services, "shared_memory_stop", None)
    task = getattr(services, "shared_memory_task", None)
    intake = getattr(services, "shared_memory_intake", None)
    memory = getattr(services, "shared_memory", None)
    if stop is not None:
        stop.set()
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except Exception:
            services.shared_memory_issue = "native_memory_intake_stopped"
    try:
        if intake is not None:
            try:
                await asyncio.wait_for(intake.close(), timeout=10)
            except Exception:
                services.shared_memory_issue = "native_memory_flush_pending"
                LOGGER.warning("Native memory shutdown incomplete; pending cursor/receipt retained")
            finally:
                intake.abort()
    finally:
        if memory is not None:
            await memory.aclose()
        services.shared_memory_task = None
        services.shared_memory_stop = None
        services.shared_memory_intake = None
        services.shared_memory = None
