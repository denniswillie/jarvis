"""Memory transport + real isolated SQL + clean OS process restarts.

Controlled is the default. --actual additionally requires an explicit release
environment flag and file-based Hindsight credentials; it never starts services.
No audio/Realtime client is used in either mode.
Virtual time reaches existing Live State and working-memory freshness readers;
it does not replace SQL time, authorization, leases, transport or monotonic time.
Source-date filtering and explicit version reads are not a full as-of engine.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.observations import ObservationPublication
from app.knowledge.retrieval import SourceRecallScope
from app.live_state.canonical import context_dependency, memory_dependency
from app.live_state.projection import select_voice_context
from app.live_state.runtime import create_session_live_state
from app.migrations.runner import apply_migrations, quote_schema
from app.perception.models import ObservationFrameLink, VisualObservation
from app.perception.repository import PerceptionRepository
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy
from app.runtime.events import DomainEventBus
from app.runtime.feature_config import KnowledgeConfiguration, MemoryAuthority
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskError
from evals.responsive_voice.memory_lifecycle_qualification import (
    Bounds, INDEPENDENT, ORIGINAL, REPLACEMENT, SOURCE_DAY_ZERO, LifecycleCase,
    assert_recall, failure, new_report, require, save,
)
from evals.responsive_voice.memory_temporal_transport import PersistentControlledMemory
from evals.responsive_voice.memory_temporal_native import (
    PersistentActualMemory, actual_inputs, require_actual_release,
)
from evals.responsive_voice.memory_live_state_diagnostics import CanonicalRefreshDiagnostics
from evals.second_brain.fixtures import DatabaseFixture


STAGES = ("day0", "day1", "day7", "day30", "day90_delete", "day90_restart")
CHILD_MODULE = "evals.responsive_voice.memory_temporal_restart_qualification"
DAYS = (0, 1, 7, 30, 90, 90)
PROGRESS_0 = "Training log: bench press 30 kg for 5 repetitions."
PROGRESS_30 = "Training log: bench press 45 kg for 5 repetitions."
CURRENT_VIEW = "TEMPORARY_BENCH_DISPLAY_73"
BOUNDS = Bounds(drain_seconds=60, max_steps=180, lifecycle_seconds=480, cleanup_seconds=180)
LIMITATIONS = (
    "Controlled lexical candidate transport, not actual Hindsight/model quality.",
    "Clean Python worker process exits/restarts, not crash recovery or restarting Hindsight/PostgreSQL.",
    "No frontend, audio, Realtime creation, reminder scheduler or retention scheduler qualification.",
    "Dated progress is exact source retention/filtering, not reasoning about which measurement is latest.",
    "Controlled analyzer observation with frame metadata; no pixels, camera analysis or physical-action proof.",
)


def exception_frames(error):
    # Private diagnostics contain locations, never exception messages or locals.
    return [{"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(error.__traceback__)[-12:]]


class StageTrace:
    """Bounded read-only instrumentation; never changes a deadline or result."""

    def __init__(self, report, path, state, *, limit=96):
        self.report, self.path, self.state, self.limit = report, path, state, limit
        report.setdefault("boundaries", [])

    @contextmanager
    def boundary(self, name):
        require(len(self.report["boundaries"]) < self.limit, "stage_diagnostic_budget")
        record = {"name": name, "status": "running"}
        self.report["boundaries"].append(record)
        save(self.path, self.report)
        started = time.monotonic()
        try:
            yield
        except BaseException as error:
            record.update(status="failed", failure=failure(error), frames=exception_frames(error))
            self.report.setdefault("failed_boundary", name)
            try:
                record["state"] = self.state()
            except Exception as diagnostic_error:
                record["diagnostic_failure"] = failure(diagnostic_error)
            raise
        else:
            record["status"] = "passed"
        finally:
            record["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
            save(self.path, self.report)


def mode_limitations(mode):
    require(mode in {"controlled", "actual"}, "temporal_mode_invalid")
    return list(LIMITATIONS) if mode == "controlled" else [
        "Actual Hindsight ingestion and candidate retrieval; no model-authored answer or general quality score.",
        *LIMITATIONS[1:],
        "Provider-internal model usage and dollar cost are not inferred from HTTP counts."]


def new_child_report(stage, clock, advanced, *, mode="controlled"):
    return new_report("controlled_hindsight_real_postgres") | {
        "format": "jarvis.temporal_memory_restart.child.v1",
        "evidence_mode": ("actual_hindsight_real_postgres_fresh_subprocesses" if mode == "actual"
                          else "controlled_memory_real_postgres_fresh_subprocesses"),
        "stage": stage, "pid": os.getpid(), "parent_pid": os.getppid(), "status": "running",
        "day": clock.day, "domain_time": clock.now().isoformat(), "domain_advance_seconds": advanced,
        "wall_started_at": datetime.now(UTC).isoformat(), "selections": [], "stages": {},
        "limitations": mode_limitations(mode),
        "clock": {"source_day_zero": SOURCE_DAY_ZERO.isoformat(), "domain_day_labels": list(DAYS),
            "virtual_scope": ["Live State freshness", "working-memory expiry", "source-date filters"],
            "elapsed_clock": "time.monotonic", "scheduler_time_travel": False,
            "database_clock_changed": False}}


class DomainClock:
    def __init__(self, day=0):
        require(type(day) is int and 0 <= day <= 90, "domain_day_invalid")
        self.day = day

    def now(self):
        return SOURCE_DAY_ZERO + timedelta(days=self.day)

    def advance(self, day):
        require(type(day) is int and self.day <= day <= 90, "domain_time_cannot_reverse")
        delta = day - self.day
        self.day = day
        return delta * 86400


def configuration(area):
    return KnowledgeConfiguration(base_url="http://127.0.0.1:19999",
        api_key="controlled-fixture-only", area_id=area)


def transport_for(directory, bank, mode, config, *, create=False):
    require(mode in {"controlled", "actual"}, "temporal_mode_invalid")
    if mode == "actual":
        require_actual_release()
        return PersistentActualMemory(directory / "native-budget.json", bank, config.base_url, create=create)
    return PersistentControlledMemory(directory / "controlled-memory.json", bank, create=create)


async def pool_for(dsn, schema):
    async def setup(connection):
        await connection.execute(f"set search_path to {quote_schema(schema)},pg_catalog,extensions")
    return await asyncpg.create_pool(dsn, min_size=1, max_size=3, timeout=8,
                                    command_timeout=30, setup=setup)


async def until(predicate, *, timeout=20):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(.01)


class TemporalCase(LifecycleCase):
    def __init__(self, *args, clock):
        super().__init__(*args)
        self.clock, self.live = clock, None
        self.bus = DomainEventBus()
        self.repository = None
        self.working = None
        self.expected_sources = []
        self.refresh_diagnostics = CanonicalRefreshDiagnostics(limit=32)
        self.trace = StageTrace(self.report, self.path, self.state_diagnostics)

    def state_diagnostics(self):
        live = self.live
        result = {"domain_time": self.clock.now().isoformat(), "live_present": live is not None,
            "expected_sources": self.expected_sources[:8],
            "canonical_refresh": self.refresh_diagnostics.document()}
        if live is None:
            return result
        snapshot = live.service.snapshot()
        result.update(readiness=snapshot.readiness, epoch=str(snapshot.cursor.epoch),
            revision=snapshot.cursor.revision, requires_reconstruction=live.requires_reconstruction,
            needs_resync=live.service.needs_resync,
            runtime_failure=type(live.failure).__name__ if live.failure else None,
            capabilities=snapshot.capabilities.model_dump(mode="json"),
            entries=[{"id": entry.entry_id, "kind": entry.kind, "validity": entry.validity,
                "sources": [source.model_dump(mode="json") for source in entry.sources[:16]],
                "observed_at": entry.observed_at.isoformat() if entry.observed_at else None,
                "fresh_until": entry.fresh_until.isoformat() if entry.fresh_until else None,
                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None}
                for entry in snapshot.entries[:32]],
            omitted_entries=max(0, len(snapshot.entries) - 32))
        result["workers"] = {}
        for name, worker in (("canonical", live.worker),
                ("perception", live.perception.worker if live.perception else None),
                ("enrichment", live.enrichment.worker if live.enrichment else None)):
            if worker is not None:
                result["workers"][name] = {field: getattr(worker, field, None)
                    for field in ("refresh_count", "failure_count", "last_error")}
        if live.enrichment is not None:
            enrichment = live.enrichment
            result["candidate_admission"] = enrichment.candidate_diagnostics()
            result["enrichment"] = {field: getattr(enrichment, field, None) for field in (
                "status", "last_error", "lookup_count", "stale_results", "omitted_sources",
                "publication_attempts", "publication_completed", "publication_failures", "last_publication_ms")}
            result["enrichment"]["selected_originals"] = [
                {"id": str(item.source_id), "revision": item.revision, "generation": item.generation}
                for item in enrichment._selected[:8]]
        result["live_tasks"] = [{"name": task.get_name(), "done": task.done(),
            "frames": [{"file": frame.f_code.co_filename, "line": frame.f_lineno,
                        "function": frame.f_code.co_name} for frame in task.get_stack(limit=4)]}
            for task in sorted(asyncio.all_tasks(), key=lambda task: task.get_name())
            if task.get_name().startswith("live")][:16]
        return result

    async def wait(self, name, predicate):
        with self.trace.boundary(name):
            await until(predicate)

    async def capture(self, text, day, *, kind="user_transcript"):
        with self.trace.boundary("ingress." + kind):
            return await super().capture(text, day, kind=kind)

    async def drain(self):
        with self.trace.boundary("publication.drain"):
            return await super().drain()

    async def invoke(self, name, arguments, request):
        with self.trace.boundary("owner." + name):
            return await super().invoke(name, arguments, request)

    async def fresh(self, day):
        with self.trace.boundary("knowledge.fresh_runtime"):
            await super().fresh(day)
        recall = self.runtime.retrieval.recall

        async def observed_recall(*args, **kwargs):
            with self.trace.boundary("retrieval.canonical_sources"):
                result = await recall(*args, **kwargs)
                receipts = self.report.setdefault("retrieval_results", [])
                require(len(receipts) < 12, "retrieval_diagnostic_budget")
                receipts.append({"query_digest": hashlib.sha256(args[2].encode()).hexdigest(),
                    "status": result.get("status"),
                    "sources": [{key: item.get(key) for key in ("source_id", "revision", "generation",
                        "source_time", "content_included")} for item in result.get("sources", ())[:8]],
                    "rejected_candidates": result.get("rejected_candidates"),
                    "omitted_sources": result.get("omitted_sources")})
                return result
        self.runtime.retrieval.recall = observed_recall
        self.repository = PerceptionRepository(self.pool, observation_publication=ObservationPublication(
            self.runtime.context_for(self.owner), owner_id=self.owner, access_check=self.runtime.context_for))
        self.working = WorkingMemoryService(repository=self.repository,
            policy=WorkingMemoryPolicy(ttl_seconds=30, entity_ttl_seconds=30,
                visible_text_ttl_seconds=30, max_entities=8, max_recent_changes=4),
            domain_event_bus=self.bus, now=self.clock.now)

    async def start_live(self, *, session=None):
        from app.live_state.models import SessionScope
        require(self.live is None, "old_live_runtime_not_closed")
        services = SimpleNamespace(knowledge_runtime=self.runtime, memory_authority=MemoryAuthority.HINDSIGHT,
            domain_event_bus=self.bus, working_memory_service=self.working,
            responsibility_issue="responsibilities_disabled", background_task_service=None)
        self.live = create_session_live_state(services,
            scope=SessionScope(owner_id=self.owner, session_id=session or self.session, area_id=self.area),
            enabled=True, clock=self.clock.now, interval_seconds=.1, timeout_seconds=2,
            perception_brief_enabled=False)
        require(not self.live.service.snapshot().entries, "new_process_inherited_live_state")
        if self.live.enrichment is not None:
            self.refresh_diagnostics.observe(self.live.enrichment)
        with self.trace.boundary("live.start"):
            await self.live.start()
        await self.wait("live.await_ready", lambda: self.live.service.snapshot().readiness == "ready")
        return self.live

    async def close_live(self):
        if self.live is not None:
            self.report.setdefault("live_runs", []).append(self.state_diagnostics())
            live, self.live = self.live, None
            with self.trace.boundary("live.close_and_join"):
                await live.close()

    async def close_runtime(self):
        await self.close_live()
        await super().close_runtime()

    async def selected(self, expected, request, *, absent=(), forbidden=()):
        self.expected_sources = [memory_dependency(source).model_dump(mode="json") for source in expected]
        with self.trace.boundary("selection.validate_provenance_and_projection"):
            await self._selected(expected, request, absent=absent, forbidden=forbidden)

    async def _selected(self, expected, request, *, absent=(), forbidden=()):
        live = self.live
        expected_dependencies = {memory_dependency(item) for item in expected}
        def ready():
            present = {source for entry in live.service.snapshot().entries
                       if entry.kind == "selected_memory" for source in entry.sources}
            return expected_dependencies.issubset(present)
        await self.wait("selection.await_expected_sources", ready)
        snapshot = live.service.snapshot()
        entries = [entry for entry in snapshot.entries if entry.kind == "selected_memory"]
        require(not any(source.source_id in absent for entry in entries for source in entry.sources),
                "deleted_source_in_live_state")
        for source in expected:
            entry, = [entry for entry in entries if memory_dependency(source) in entry.sources]
            require(entry.observed_at == source.source_time and context_dependency(request) in entry.sources,
                    "source_time_or_intent_changed")
            require(json.dumps(source.content, ensure_ascii=True, sort_keys=True) in entry.summary,
                    "original_content_not_selected")
            require(await live.binder.available(entry.guard_binding), "selected_source_not_current")
        selection = select_voice_context(snapshot, now=self.clock.now())
        require(not any(text in selection.text for text in forbidden), "obsolete_projection_text")
        for source in expected:
            require(memory_dependency(source) in selection.dependencies, "selected_source_omitted_from_projection")
        binding = await live.bind(selection)
        async with live.guard(binding):
            pass
        self.report.setdefault("selections", []).append({"state_epoch": str(snapshot.cursor.epoch),
            "domain_as_of": snapshot.as_of.isoformat(), "canonical_guard_passed": True,
            "source_versions": [{"id": str(item.source_id), "revision": item.revision,
                "generation": item.generation, "source_time": item.source_time.isoformat()} for item in expected],
            "voice_projection_only": True, "audio_or_provider_delivery": False})

    async def observation(self):
        captured = self.clock.now()
        # Controlled sensor metadata is input, not a precomputed Live State.
        # No pixels are uploaded or claimed available.
        async with self.pool.acquire() as connection:
            frame = await connection.fetchval("""insert into visual_frames
                (owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
                 content_type,width,height,storage_bucket,storage_path,raw_image_status,raw_image_expires_at)
                values($1,$2,$3,1,'glasses',$4,'image/jpeg',640,480,
                       'fixture-no-storage','not-uploaded','failed',$5) returning id""",
                self.owner, self.session, uuid4(), captured, captured + timedelta(seconds=30))
        observation = VisualObservation(id=uuid4(), owner_id=self.owner, session_id=self.session,
            source="glasses", started_at=captured, ended_at=captured, primary_visual_frame_id=frame,
            summary=CURRENT_VIEW, scene="controlled workbench", activity="reading a display",
            entities=(), visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
            interrupt_candidate=None, starts_new_event=False, boundary_reason=None, confidence=.9,
            analyzer_provider="controlled", analyzer_model="fixture-no-model", analyzer_contract_version=1,
            compacted_at=None, created_at=datetime.now(UTC))
        await self.repository.insert_observation(observation=observation,
            frame_links=(ObservationFrameLink(frame, 0, "primary"),))
        applied = await self.working.apply_observation(observation)
        require(applied.applied, "observation_not_applied")
        return self.session


async def child(stage, manifest_path):
    started = time.monotonic()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
    fixture = DatabaseFixture.load(dsn, Path(manifest["fixture_manifest"]))
    require(manifest["owner"] == fixture.owners[0] and manifest["schema"] in fixture.manifest["schemas"]
            and manifest["bank"] == "jarvis_temporal_eval_" + fixture.run_id, "child_fixture_scope")
    require(stage in STAGES, "child_stage_invalid")
    state_path = root / "handoff.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"last_day": 0, "sources": {}}
    day = DAYS[STAGES.index(stage)]
    clock = DomainClock(state["last_day"])
    advanced = clock.advance(day)
    mode = manifest.get("mode", "controlled")
    config = (actual_inputs()[1].model_copy(update={"area_id": UUID(manifest["area"])})
              if mode == "actual" else configuration(UUID(manifest["area"])))
    provider = transport_for(root, manifest["bank"], mode, config)
    if mode == "actual":
        provider.claim_child(stage, STAGES.index(stage))
    report = new_child_report(stage, clock, advanced, mode=mode)
    path = root / (stage + ".json")
    pool = case = None
    def source(name):
        return SourceVersion.model_validate(state["sources"][name])
    def remember(name, value):
        state["sources"][name] = value.model_dump(mode="json")
    try:
        pool = await pool_for(dsn, manifest["schema"])
        async with pool.acquire() as connection:
            report["database_wall_started_at"] = (await connection.fetchval("select clock_timestamp()")).isoformat()
        with patch("app.knowledge.runtime.HindsightClient", side_effect=provider.client):
            case = TemporalCase(fixture, pool, UUID(manifest["area"]), manifest["bank"],
                config, report, path, BOUNDS, clock=clock)
            await case.fresh(day)
            if stage == "day0":
                event, _, original = await case.capture(ORIGINAL, 0)
                remember("original", original)
                state["original_event"] = event.model_dump(mode="json")
                _, _, independent = await case.capture(INDEPENDENT, 0)
                remember("independent", independent)
                _, _, progress = await case.capture(PROGRESS_0, 0)
                remember("progress0", progress)
                with case.trace.boundary("observation.ingest_and_apply"):
                    state["observation_session"] = str(await case.observation())
                _, request, _ = await case.capture("workshop bicycle training", 0, kind="user_request")
                await case.drain()
                await case.start_live()
                await case.selected((original, independent, progress), request)
                await case.wait("observation.await_current_projection", lambda: any(entry.kind == "observation" and entry.summary == CURRENT_VIEW
                    for entry in case.live.service.snapshot().entries))
                require(CURRENT_VIEW in select_voice_context(case.live.service.snapshot(), now=clock.now()).text,
                        "fresh_observation_not_projected")
                report["current_evidence"] = {"fresh_observation_projected": True}
            elif stage == "day1":
                old_session = UUID(state["observation_session"])
                raw = await case.repository.get_session_working_memory(owner_id=case.owner, session_id=old_session)
                require(raw is not None and raw.expires_at <= clock.now(), "persisted_evidence_not_expired")
                require(await case.working.get_session_working_memory(owner_id=case.owner, session_id=old_session) is None,
                        "expired_working_evidence_returned_as_current")
                await case.start_live(session=old_session)
                await case.wait("perception.await_expired_session_refresh",
                    lambda: case.live.perception.worker.refresh_count > 0)
                require(not any(entry.kind == "observation" for entry in case.live.service.snapshot().entries),
                        "expired_observation_rehydrated_as_current")
                report["current_evidence"] = {"persisted_row_retained": True, "reader_expired": True,
                    "resumed_old_session_has_no_current_observation": True,
                    "original_expires_at": raw.expires_at.isoformat()}
                await case.close_live()
                _, request, _ = await case.capture("bicycle LARCH serial", 1, kind="user_request")
                await case.drain()
                await case.start_live()
                await case.selected((source("independent"),), request)
            elif stage == "day7":
                original = source("original")
                _, request, command = await case.capture(f"Correct source {original.source_id} revision 1. "
                    + "Replacement: " + REPLACEMENT, 7, kind="user_request")
                await case.invoke("correct_memory_source", {"source_id": str(original.source_id),
                    "expected_revision": 1, "replacement": REPLACEMENT}, request)
                corrected = await case.runtime.sources.read(case.owner, case.area, original.source_id)
                require(corrected.revision == 2 and corrected.trust == "derived", "correction_revision_missing")
                remember("corrected", corrected)
                remember("correction_command", command)
                _, request, _ = await case.capture("workshop pouch key", 7, kind="user_request")
                await case.drain()
                await case.start_live()
                await case.selected((corrected,), request, forbidden=(ORIGINAL,))
            elif stage == "day30":
                _, _, progress = await case.capture(PROGRESS_30, 30)
                remember("progress30", progress)
                _, request, _ = await case.capture("training bench repetitions", 30, kind="user_request")
                await case.drain()
                await case.start_live()
                await case.selected((source("progress0"), progress), request)
                historical = await case.runtime.sources.read(case.owner, case.area,
                    source("original").source_id, revision=1)
                current = await case.runtime.sources.read(case.owner, case.area, source("original").source_id)
                require(not historical.is_current and historical.content == source("original").content
                    and historical.source_time == source("original").source_time
                    and current.revision == 2 and current.content == source("corrected").content,
                    "current_and_historical_revision_confused")
                report["dated_windows"] = []
                for first, last, expected, excluded in ((0, 2, source("progress0"), progress),
                                                       (30, 31, progress, source("progress0"))):
                    result = await case.runtime.retrieval.recall(case.owner, case.area, "training bench repetitions",
                        scope=SourceRecallScope(start_time=SOURCE_DAY_ZERO + timedelta(days=first),
                                               end_time=SOURCE_DAY_ZERO + timedelta(days=last)))
                    assert_recall(result, (expected,), absent=(excluded.source_id,))
                    report["dated_windows"].append({"start_day": first, "end_day_exclusive": last,
                        "expected_source": str(expected.source_id), "other_date_excluded": True})
                report["explicit_revision_history"] = {"historical_revision": 1, "current_revision": 2}
            elif stage == "day90_delete":
                targets = (source("corrected"), source("correction_command"))
                _, request, _ = await case.capture("Permanently delete exact originals and historical copies: "
                    + "; ".join(f"source {item.source_id} revision {item.revision}" for item in targets)
                    + ". Keep other facts.", 90, kind="user_request")
                await case.invoke("delete_memory_source_group", {"sources": [
                    {"source_id": str(item.source_id), "expected_revision": item.revision} for item in targets]}, request)
                await case.drain()
                await case.unavailable(source("original"), revision=1)
                await case.unavailable(source("corrected"))
            else:
                try:
                    await case.runtime.context_for(case.owner).append(case.owner,
                        ContextIngress.model_validate(state["original_event"]))
                except (SourceError, TaskError):
                    report["old_receipt_replay"] = "rejected"
                else:
                    report["old_receipt_replay"] = "returned_without_original"
                await case.unavailable(source("original"), revision=1)
                await case.unavailable(source("corrected"))
                _, request, _ = await case.capture("LARCH-492", 90, kind="user_request")
                await case.drain()
                await case.start_live()
                absent = (source("original").source_id, source("correction_command").source_id)
                await case.selected((source("independent"),), request, absent=absent,
                                    forbidden=(ORIGINAL, REPLACEMENT))
                result = await case.runtime.retrieval.recall(case.owner, case.area, "workshop key")
                assert_recall(result, (), absent=absent, forbidden=(ORIGINAL, REPLACEMENT))
            state["last_day"] = day
            save(state_path, state)
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failure=failure(error), failure_frames=exception_frames(error))
    finally:
        if case is not None:
            try:
                await case.close_runtime()
            except Exception as error:
                report.update(status="failed", close_failure=failure(error))
        if pool is not None:
            await asyncio.wait_for(pool.close(), 10)
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        report["wall_finished_at"] = datetime.now(UTC).isoformat()
        report["accepted"] = report["status"] == "passed"
        save(path, report)
    return report


async def launch_child(manifest, stage, dsn):
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", CHILD_MODULE,
        "--child", str(manifest), "--stage", stage,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=dict(os.environ, JARVIS_SECOND_BRAIN_TEST_DB_URL=dsn),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(process.communicate(), 90)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.communicate(), 5)
        raise
    require(len(output) <= 65536, "child_output_budget")
    save(manifest.parent / (stage + "-process.json"), {"pid": process.pid,
        "returncode": process.returncode, "output": output.decode(errors="replace")})
    require(process.returncode == 0, "child_qualification_failed_" + stage)
    result = json.loads((manifest.parent / (stage + ".json")).read_text())
    require(result["pid"] == process.pid and result["status"] == "passed", "child_receipt_mismatch")
    return result


async def run_controlled(dsn, directory):
    return await run_temporal(dsn, directory)


async def run_actual(dsn, configuration, directory):
    require_actual_release()
    return await run_temporal(dsn, directory, mode="actual", actual_configuration=configuration)


async def run_temporal(dsn, directory, *, mode="controlled", actual_configuration=None):
    require(mode in {"controlled", "actual"}, "temporal_mode_invalid")
    if mode == "actual":
        require_actual_release()
        require(actual_configuration is not None, "native_configuration_required")
    directory = Path(directory)
    require(directory.is_absolute() and not directory.exists(), "new_report_directory_required")
    directory.mkdir(parents=True, mode=0o700)
    fixture = DatabaseFixture(dsn, directory, test_name="temporal-memory-clean-process-restarts")
    bank = "jarvis_temporal_eval_" + fixture.run_id
    transport = transport_for(directory, bank, mode, actual_configuration, create=True)
    report = {"format": "jarvis.temporal_memory_restart.v1", "status": "preparing", "accepted": False,
        "evidence_mode": ("actual_hindsight_real_postgres_fresh_subprocesses" if mode == "actual"
                          else "controlled_memory_real_postgres_fresh_subprocesses"), "parent_pid": os.getpid(),
        "stages": {stage: {"status": "unreached"} for stage in STAGES}, "children": [],
        "fixture_manifest": str(fixture.path), "bank": bank, "limits": {
            "main_seconds": 480, "child_seconds": 90, "cleanup_seconds": 180,
            "banks": 1, "recalls": 12, "main_http": 1200, "cleanup_http": 400, "whole_run_retries": 0},
        "clock_scope": {"virtual": ["Live State freshness", "working-memory expiry", "source-date filters"],
            "real": ["monotonic latency/deadlines", "SQL time", "authorization", "leases", "transport"],
            "scheduler_time_travel": False, "database_clock_changed": False},
        "limitations": mode_limitations(mode), "actual_memory_lifecycle_qualified": False}
    if mode == "actual":
        report["limits"].update(retain_requests=32, submitted_retain_characters=128000,
            recall_max_tokens=2400, note="Recall response budget is not billed model-token usage.")
    path = directory / "qualification.json"
    save(path, report)
    started, pool, area = time.monotonic(), None, None
    try:
        async with asyncio.timeout(480):
            if mode == "actual":
                async with asyncio.timeout(8):
                    await transport.preflight(actual_configuration)
                report["native_preflight"] = {"existing_service_reachable": True, "new_bank_absent": True}
                save(path, report)
            connection = await asyncpg.connect(dsn, timeout=8, command_timeout=120)
            try:
                schema = await fixture.create_schema(connection)
                await apply_migrations(connection, schema=schema, fresh=True)
            finally:
                await connection.close(timeout=5)
            pool = await pool_for(dsn, schema)
            async with pool.acquire() as connection:
                area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
                    "values($1,'Temporal qualification','test') returning id", fixture.owners[0])
                await connection.execute("insert into jarvis_memory_banks(owner_id,area_id,bank_name) values($1,$2,$3)",
                                         fixture.owners[0], area, bank)
            manifest = directory / "child-input.json"
            save(manifest, {"fixture_manifest": str(fixture.path), "schema": schema,
                "owner": fixture.owners[0], "area": str(area), "bank": bank, "mode": mode})
            for stage in STAGES:
                report["stages"][stage] = {"status": "running"}
                save(path, report)
                result = await launch_child(manifest, stage, dsn)
                report["children"].append(result)
                report["stages"][stage] = {"status": "passed", "pid": result["pid"], "elapsed_ms": result["elapsed_ms"]}
                save(path, report)
            pids = [item["pid"] for item in report["children"]]
            require(len(set(pids)) == 6 and os.getpid() not in pids, "process_restart_not_proven")
            report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", failure=failure(error))
        for stage, result in report["stages"].items():
            if result["status"] == "running":
                result["status"] = "failed"
    finally:
        report["cleanup"] = {}
        provider_clean = area is None
        if pool is not None and area is not None:
            transport.begin_cleanup()
            cleanup_report = new_report("actual_hindsight" if mode == "actual" else "controlled_hindsight_real_postgres")
            cleanup_config = (actual_configuration.model_copy(update={"area_id": area})
                              if mode == "actual" else configuration(area))
            cleanup_key = "native_bank" if mode == "actual" else "controlled_bank"
            case = LifecycleCase(fixture, pool, area, bank, cleanup_config, cleanup_report,
                directory / "cleanup-operations.json", BOUNDS)
            try:
                with patch("app.knowledge.runtime.HindsightClient", side_effect=transport.client), \
                     patch("evals.responsive_voice.memory_lifecycle_qualification.HindsightClient", side_effect=transport.client):
                    async with asyncio.timeout(180):
                        report["cleanup"][cleanup_key] = await case.cleanup_native()
                if mode == "controlled":
                    independent = PersistentControlledMemory(directory / "controlled-memory.json", bank)
                    require(not independent.read()["banks"], "independent_controlled_bank_residue")
                provider_clean = True
            except Exception as error:
                report["cleanup"][cleanup_key] = {"failure": failure(error)}
            finally:
                try:
                    await asyncio.wait_for(case.close_runtime(), 10)
                except Exception as error:
                    report["cleanup"]["runtime_close"] = failure(error)
                save(directory / "cleanup-operations.json", cleanup_report)
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), 10)
            except Exception as error:
                pool.terminate()
                report["cleanup"]["pool_close"] = failure(error)
        if provider_clean:
            try:
                report["cleanup"]["database"] = await fixture.cleanup()
            except Exception as error:
                report["cleanup"]["database"] = {"failure": failure(error)}
        else:
            report["cleanup"]["database"] = {"status": "preserved_for_cleanup_retry"}
        report["transport"] = (transport.report() if mode == "actual"
            else {key: value for key, value in transport.read().items() if key != "banks"})
        report["elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        report["accepted"] = (report["status"] == "passed" and provider_clean
            and report["cleanup"].get("database", {}).get("remaining_schemas") == 0
            and report["cleanup"]["database"].get("remaining_public_owner_rows") == 0
            and "runtime_close" not in report["cleanup"] and "pool_close" not in report["cleanup"])
        report["actual_memory_lifecycle_qualified"] = report["accepted"] and mode == "actual"
        save(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--controlled", action="store_true")
    group.add_argument("--actual", action="store_true")
    group.add_argument("--child", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    if args.child:
        result = asyncio.run(child(args.stage, args.child))
    elif args.actual:
        if args.report_dir is None:
            parser.error("--report-dir is required")
        dsn, config = actual_inputs()
        result = asyncio.run(run_actual(dsn, config, args.report_dir))
    else:
        if args.report_dir is None:
            parser.error("--report-dir is required")
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        dsn = os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") or os.environ["SUPABASE_DB_URL"]
        result = asyncio.run(run_controlled(dsn, args.report_dir))
    print(json.dumps({"status": result["status"], "accepted": result.get("accepted"), "pid": os.getpid()}))
    raise SystemExit(0 if result["status"] == "passed" and result.get("accepted", True) else 1)


if __name__ == "__main__":
    main()
