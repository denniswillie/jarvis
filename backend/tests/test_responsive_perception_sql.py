"""Bounded real-PostgreSQL applicability qualification in a disposable v39 schema.

Opt in with the existing second-brain DB URL and manifest-directory gates.
No public migration, model, service, pixels, or production owner is used.
Each named check is saved independently; teardown audits committed removal.
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.attention.models import AttentionError
from app.attention.notification_ledger.repository import VoiceNotificationLedger
from app.attention.repository import AttentionRepository
from app.knowledge.models import SourceError
from app.knowledge.observations import ObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope
from app.live_state.perception import DurableResponsibilityCapability, PerceptionBrief
from app.live_state.perception_focus import CanonicalFocusInputs
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.live_state.task_context import TaskContextProjector
from app.memory.prospective.repository import ProspectiveRepository
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.perception.applicability import PerceptionEffectGuard, PerceptionEffectUnavailable
from app.perception.models import ObservationFrameLink, VisualObservation
from app.perception.repository import PerceptionRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskCreate, TaskMutation, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.visual_memory.repository import VisualFrameRepository
from app.voice_routes.contracts import RouteIdentity
from evals.second_brain.fixtures import DatabaseFixture
from test_notification_ledger_database import cleanup_fixture_resources, native_capabilities


def now():
    return datetime.now(UTC)


def sanitized(error):
    return {
        "error_type": type(error).__name__, "code": getattr(error, "code", None),
        "sqlstate": getattr(error, "sqlstate", None),
        "frames": [{"file": Path(frame.filename).name, "line": frame.lineno,
                    "function": frame.name}
                   for frame in traceback.extract_tb(error.__traceback__)[-6:]],
    }


class ObservedEffect:
    """Instrument the real guard, without replacing its transaction or checks."""

    def __init__(self, guard, after=None):
        self.guard, self.after, self.pids = guard, after, []

    @property
    def focus_fence(self):
        return self.guard.focus_fence

    @asynccontextmanager
    async def __call__(self):
        async with self.guard() as connection:
            if not connection.is_in_transaction():
                raise AssertionError("Effect PID must be observed inside its real transaction")
            self.pids.append(await connection.fetchval("select pg_backend_pid()"))
            yield connection
            if self.after is not None:
                await self.after()


class ObservedTransactionPool:
    """Borrow the actual revoker lease and observe its own transaction's BEGIN.

    The repository still opens its transaction and acquires every lock. This
    wrapper adds only a read after that BEGIN, before the first repository query.
    Protocol/session PIDs outside a transaction are not physical backend proof
    when the configured URL leads through a transaction pooler.
    """

    def __init__(self, pool):
        self.pool, self.pid, self.ready = pool, None, asyncio.Event()

    @asynccontextmanager
    async def acquire(self, *args, **kwargs):
        async with self.pool.acquire(*args, **kwargs) as connection:
            observer = self

            class Connection:
                def __getattr__(self, name):
                    return getattr(connection, name)

                @asynccontextmanager
                async def transaction(self, *args, **kwargs):
                    async with connection.transaction(*args, **kwargs):
                        if not connection.is_in_transaction():
                            raise AssertionError("Revoker transaction did not start")
                        observer.pid = await connection.fetchval("select pg_backend_pid()")
                        observer.ready.set()
                        yield

            yield Connection()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class ResponsivePerceptionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_applicability_effects_and_task_sources(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"responsive-perception-{fixture.run_id}.json"
        report = {"format": "jarvis.responsive_perception.sql.v1", "run_id": fixture.run_id,
                  "status": "running", "migration_ceiling": 39, "checks": [],
                  "cleanup": None, "network_retries": 0}
        connection = pool = rival_pool = None
        failures = []

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            try:
                async with asyncio.timeout(50):
                    evidence = await operation()
            except Exception as error:
                entry = {"name": name, "status": "failed", **sanitized(error)}
                failures.append(entry)
                # TimeoutError subclasses OSError. A local assertion/deadline
                # failure is not evidence of a lost transport and must not skip
                # unrelated checks. Actual transport failures are not retried.
                if isinstance(error, TimeoutError):
                    entry["failure_kind"] = "local_deadline"
                elif isinstance(error, (OSError, asyncpg.ConnectionDoesNotExistError,
                                        asyncpg.CannotConnectNowError)):
                    report["checks"].append(entry)
                    save()
                    raise
            else:
                entry = {"name": name, "status": "passed", "evidence": evidence}
            report["checks"].append(entry)
            save()

        save()
        try:
            # Exactly one initial network attempt. No fixture SQL precedes it.
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(item for item in load_migrations() if item.version <= 39)
            self.assertEqual(migrations[-1].version, 39)
            self.assertEqual(migrations[-1].name, "attention_voice_routes")
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), timeout=240)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            # One lease proves an effect repository borrows the guarded lease.
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            rival_pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                                  command_timeout=25, setup=setup)
            owner, task_owner = fixture.owners
            areas = {}
            for current_owner in fixture.owners:
                areas[current_owner] = await connection.fetchval(
                    "insert into jarvis_knowledge_areas(owner_id,name,domain) "
                    "values($1,'SQL fixture','test') returning id", current_owner)
            sources = SourceRepository(pool)
            tasks = TaskRepository(pool, TaskLimits(lease_seconds=300))
            context = ContextRepository(tasks,
                publication=CanonicalSourcePublication(sources, areas[owner]))
            publisher = ObservationPublication(context, owner_id=owner)
            perceptions = PerceptionRepository(pool, observation_publication=publisher)
            visuals = VisualFrameRepository(pool)
            prospective = ProspectiveRepository(pool)
            attention = AttentionRepository(tasks, lease_seconds=300)
            ledger = VoiceNotificationLedger(attention)

            async def source_id(payload_id, selected_owner=owner):
                identity = await connection.fetchval(
                    "select source_id from jarvis_knowledge_source_versions "
                    "where owner_id=$1 and payload_id=$2", selected_owner, payload_id)
                self.assertIsNotNone(identity)
                return identity

            async def scenario(*, claim=False):
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner)
                focus = await prospective.set_visual_focus(owner_id=owner, target="synthetic marker",
                    description="A synthetic square; no image was submitted to a model.",
                    purpose="Bounded SQL qualification", mode="find_once",
                    expires_at=now() + timedelta(minutes=10))
                token = (await prospective.try_claim_subject(owner_id=owner,
                    subject_type="visual_focus", subject_id=focus.id, claim_timeout_seconds=60)) if claim else None
                receipt = await context.append(owner, ContextIngress(session_id=session,
                    source_namespace="qualification:focus", source_event_id=str(focus.id),
                    receipt_id=str(uuid4()), kind="tool_outcome", trust="tool_receipt",
                    payload={"tool_name": "set_visual_focus", "result": {
                        "status": "success", "visual_focus": focus.as_prompt_payload()}}))
                scope = SessionScope(owner_id=owner, session_id=session, area_id=areas[owner])
                canonical = CanonicalContextBinder(sources, scope, clock=now, timeout_seconds=18)
                focus_inputs = CanonicalFocusInputs(canonical)
                selected = await focus_inputs.read()
                self.assertIsNotNone(selected)
                when = now()
                brief = PerceptionBrief(id=uuid4(), version=1, scope=scope, scope_epoch=uuid4(),
                    based_on_live_revision=1, questions=(selected.question,), dependencies=selected.dependencies,
                    scene_change_cues=("Notice a meaningful activity transition.",),
                    durable_responsibility=DurableResponsibilityCapability(
                        owner_namespace="fixture", status="unavailable"),
                    created_at=when, fresh_until=when + timedelta(minutes=4),
                    expires_at=when + timedelta(minutes=5))
                binder = CanonicalPerceptionBinder(canonical, focus=focus_inputs)
                binding = await binder.bind(brief)
                captured = now()
                frame = await connection.fetchval("""insert into visual_frames
                    (owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
                     content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                    values($1,$2,$3,0,'glasses',$4,'image/jpeg',32,32,'fixture','not-uploaded',
                           clock_timestamp()+interval '1 hour') returning id""",
                    owner, session, uuid4(), captured)
                observation = VisualObservation(id=uuid4(), owner_id=owner, session_id=session,
                    source="glasses", started_at=captured, ended_at=captured, primary_visual_frame_id=frame,
                    summary="Synthetic marker visibility is uncertain.", scene=None, activity=None,
                    entities=(), visible_text=(), state_changes=(), memory_tags=(), keyframe_candidates=(),
                    interrupt_candidate=None, starts_new_event=False, boundary_reason=None, confidence=0.5,
                    analyzer_provider="fixture", analyzer_model="no-model-call", analyzer_contract_version=1,
                    compacted_at=None, created_at=captured)
                await perceptions.insert_observation(observation=observation,
                    frame_links=(ObservationFrameLink(frame, 0, "primary"),))
                original = await connection.fetchval("select payload_id from jarvis_observation_sources "
                    "where owner_id=$1 and observation_id=$2", owner, observation.id)
                live = [True]
                guard = PerceptionEffectGuard(brief, binding, binder.guard, lambda: live[0], original)
                return SimpleNamespace(session=session, focus=focus, token=token, receipt=receipt,
                    scope=scope, canonical=canonical, binder=binder, brief=brief, binding=binding,
                    frame=frame, observation=observation, original=original, live=live, guard=guard)

            async def effect(s, kind, guard):
                if kind == "working_memory":
                    result = await perceptions.compare_and_swap_session_working_memory(owner_id=owner,
                        session_id=s.session, expected_revision=0, observation_id=s.observation.id,
                        state={"summary": "Synthetic visibility uncertain"},
                        expires_at=now() + timedelta(minutes=3), effect_guard=guard)
                    self.assertTrue(result.applied)
                    return result
                if kind == "visual":
                    return await visuals.create_visual_interrupt(owner_id=owner, session_id=s.session,
                        visual_frame_id=s.frame, message="A possible synthetic marker is visible.",
                        reason="fixture", confidence=0.5, dedupe_bucket=str(uuid4()),
                        source_payload_id=s.original, attention=attention, max_age_seconds=600,
                        effect_guard=guard)
                result = await prospective.reserve_delivery_event(owner_id=owner, session_id=s.session,
                    subject_type="visual_focus", subject_id=s.focus.id, claim_token=s.token,
                    occurrence_key=str(uuid4()), message="A possible synthetic marker is visible.",
                    evidence={"supporting_frame_ids": [str(s.frame)]}, dedupe_window_seconds=1,
                    source_payload_id=s.original, attention=attention, max_age_seconds=600,
                    effect_guard=guard)
                self.assertIsNotNone(result)
                return result

            async def counts(s):
                return dict(await connection.fetchrow("""select
                    (select count(*) from jarvis_session_working_memory where owner_id=$1 and session_id=$2) as wm,
                    (select count(*) from visual_interrupts where owner_id=$1 and session_id=$2) as visual,
                    (select count(*) from jarvis_prospective_delivery_events where owner_id=$1 and session_id=$2) as prospective,
                    (select count(*) from jarvis_attention_intents where owner_id=$1 and preferred_session_id=$2) as intents,
                    (select count(*) from jarvis_visual_delivery_sources where owner_id=$1 and session_id=$2) as bindings""",
                    owner, s.session))

            async def valid_effect(kind):
                s = await scenario(claim=kind == "prospective")
                observed = ObservedEffect(s.guard)
                await effect(s, kind, observed)
                self.assertEqual(len(observed.pids), 1)
                expected = {"wm": 0, "visual": 0, "prospective": 0, "intents": 0, "bindings": 0}
                expected[{"working_memory": "wm", "visual": "visual", "prospective": "prospective"}[kind]] = 1
                if kind != "working_memory":
                    expected.update(intents=1, bindings=1)
                    content = json.loads(await connection.fetchval("""select p.content from jarvis_attention_intents i
                        join jarvis_context_payloads p on p.owner_id=i.owner_id and p.id=i.source_payload_id
                        where i.owner_id=$1 and i.preferred_session_id=$2""", owner, s.session))
                    self.assertEqual(content["perception_focus"], s.guard.focus_fence)
                self.assertEqual(await counts(s), expected)
                self.assertIsNone(await connection.fetchval(
                    "select completed_at from jarvis_visual_focuses where owner_id=$1 and id=$2", owner, s.focus.id))
                return {"one_connection_pool": True, "committed": expected, "physical_completion": False}

            async def rolled_back_effect(kind):
                s = await scenario(claim=kind == "prospective")
                async def close_after_write():
                    s.live[0] = False
                observed = ObservedEffect(s.guard, close_after_write)
                with self.assertRaises(PerceptionEffectUnavailable):
                    await effect(s, kind, observed)
                self.assertEqual(len(observed.pids), 1)
                remaining = await counts(s)
                self.assertTrue(all(value == 0 for value in remaining.values()))
                self.assertTrue(await connection.fetchval(
                    "select exists(select 1 from perception_observations where owner_id=$1 and id=$2)",
                    owner, s.observation.id))
                return {"rolled_back": remaining, "historical_observation_retained": True}

            async def historical_focus_change(operation):
                s = await scenario()
                if operation == "replace":
                    await prospective.set_visual_focus(owner_id=owner, target="replacement marker",
                        description="New target", purpose="fixture replacement", mode="find_once",
                        expires_at=now() + timedelta(minutes=10))
                else:
                    await prospective.clear_active_visual_focus(owner_id=owner, expected_focus_id=s.focus.id)
                # An intact historical payload alone must not grant applicability.
                self.assertEqual(await connection.fetchval(
                    "select availability from jarvis_context_payloads where owner_id=$1 and id=$2",
                    owner, s.binding.payload_root_ids[0]), "available")
                with self.assertRaises(PerceptionEffectUnavailable):
                    await effect(s, "working_memory", s.guard)
                self.assertTrue(all(value == 0 for value in (await counts(s)).values()))
                return {"historical_brief_available": True, "stale_effect_rejected": True}

            async def removed_receipt():
                s = await scenario()
                identity = await source_id(s.receipt.payload_id)
                await sources.delete_source(owner, areas[owner], identity, expected_revision=1)
                with self.assertRaises(SourceError):
                    async with s.binder.guard(s.brief, s.binding):
                        self.fail("Deleted receipt admitted a brief")
                return {"real_source_delete": True, "guard_rejected": True}

            async def removed_observation():
                s = await scenario()
                identity = await source_id(s.original)
                await sources.delete_source(owner, areas[owner], identity, expected_revision=1)
                with self.assertRaises(PerceptionEffectUnavailable):
                    await effect(s, "visual", s.guard)
                self.assertTrue(all(value == 0 for value in (await counts(s)).values()))
                return {"real_source_delete": True, "new_delivery_rows": 0}

            async def wait_blocked(waiter, holder):
                async with asyncio.timeout(4):
                    while not await connection.fetchval(
                            "select $1::int=any(pg_blocking_pids($2::int))", holder, waiter):
                        await asyncio.sleep(0.05)

            async def concurrent_effect_revocation(operation):
                s = await scenario()
                identity = await source_id(s.original)
                observed_rival = ObservedTransactionPool(rival_pool)
                written, release = asyncio.Event(), asyncio.Event()
                async def hold_commit():
                    written.set()
                    await asyncio.wait_for(release.wait(), timeout=10)
                observed = ObservedEffect(s.guard, hold_commit)
                writer = asyncio.create_task(effect(s, "working_memory", observed))
                revoker = None
                try:
                    await asyncio.wait_for(written.wait(), timeout=15)
                    if operation == "cancel":
                        revoker = asyncio.create_task(ProspectiveRepository(observed_rival).clear_active_visual_focus(
                            owner_id=owner, expected_focus_id=s.focus.id))
                    else:
                        revoker = asyncio.create_task(SourceRepository(observed_rival).delete_source(
                            owner, areas[owner], identity, expected_revision=1))
                    await asyncio.wait_for(observed_rival.ready.wait(), timeout=4)
                    self.assertIsInstance(observed_rival.pid, int)
                    self.assertNotEqual(observed_rival.pid, observed.pids[0])
                    await wait_blocked(observed_rival.pid, observed.pids[0])
                    self.assertFalse(revoker.done())
                    release.set()
                    await asyncio.wait_for(writer, timeout=10)
                    await asyncio.wait_for(revoker, timeout=15)
                    with self.assertRaises(PerceptionEffectUnavailable):
                        async with s.guard():
                            self.fail("Revoked assignment admitted another effect")
                    return {"postgres_blocking_pids_confirmed": True, "both_transactions_completed": True,
                            "subsequent_effect_rejected": True}
                finally:
                    release.set()
                    for task in (writer, revoker):
                        if task is not None:
                            if not task.done():
                                task.cancel()
                            try:
                                await task
                            except BaseException:
                                pass

            async def queued_notification(operation):
                s = await scenario()
                event = await effect(s, "visual", s.guard)
                intent = await connection.fetchval("select id from jarvis_attention_intents "
                    "where owner_id=$1 and source_id=$2 and source_namespace='visual_alert'", owner, event.id)
                claim = await attention.claim_next(owner, s.session, "responsive-sql",
                    admission=lambda row, output: None if row["id"] == intent else "fixture_other_intent")
                self.assertIsNotNone(claim)
                route = RouteIdentity(scope=s.scope, route_id=uuid4(), route_generation=1)
                offer = await ledger.prepare(claim, route=route, audio_session_id=uuid4(),
                    capabilities=native_capabilities(), offer_id=uuid4(), expires_at=now()+timedelta(minutes=3),
                    submission_boundary="fixture.nonexecuted.audio_write")
                ticket = await ledger.commit_submission(claim, offer)
                self.assertIsNotNone(ticket)
                if operation == "replace":
                    await prospective.set_visual_focus(owner_id=owner, target="different marker",
                        description="replacement", purpose="fixture", mode="find_once",
                        expires_at=now()+timedelta(minutes=10))
                elif operation == "cancel":
                    await prospective.clear_active_visual_focus(owner_id=owner, expected_focus_id=s.focus.id)
                else:
                    await sources.delete_source(owner, areas[owner], await source_id(s.original), expected_revision=1)
                with self.assertRaises(AttentionError):
                    async with ledger.guarded_transaction(ticket):
                        self.fail("Obsolete queued notification crossed the output boundary")
                with self.assertRaises(AttentionError):
                    async with ledger.output_guard(claim, offer):
                        self.fail("Obsolete notification resumed output")
                await attention.settle(claim, reason="fixture_obsolete_notification")
                return {"submission_accepted_before_change": True, "initial_write_rejected": True,
                        "followup_write_rejected": True, "audio_generated": False}

            async def task_scenario():
                session = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", task_owner)
                task = await tasks.create(task_owner, TaskCreate(request_id=uuid4(),
                    title="Synthetic task", requested_outcome="Return a synthetic test result; no external action.",
                    session_id=session, knowledge_area_id=areas[task_owner]), model="fixture-"+uuid4().hex)
                scope = SessionScope(owner_id=task_owner, session_id=session, area_id=areas[task_owner])
                canonical = CanonicalContextBinder(sources, scope, clock=now, timeout_seconds=18)
                projector = TaskContextProjector(canonical, tasks)
                return task, canonical, projector

            async def project(projector, task):
                entries = await projector.project((task.id,), received_at=now(),
                    expires_at=now()+timedelta(minutes=3))
                self.assertEqual(len(entries), 1)
                async with projector.binder.guard(entries[0].guard_binding):
                    pass
                return entries[0]

            async def task_status_change():
                task, canonical, projector = await task_scenario()
                entry = await project(projector, task)
                claim = await tasks.claim_next("responsive-task-sql", ("frontier",), owner_id=task_owner,
                    provider_models={"frontier": task.model})
                self.assertIsNotNone(claim)
                try:
                    self.assertEqual(claim.task.revision, task.revision)
                    with self.assertRaises(SourceError):
                        async with canonical.guard(entry.guard_binding):
                            self.fail("Queued status remained current after claim")
                    fresh = await project(projector, task)
                    self.assertEqual(fresh.status, "running")
                finally:
                    await tasks.abandon(claim, reason="fixture_finished", target=TaskStatus.CANCELLED)
                return {"intent_revision_unchanged": True, "status_change_rejected": True,
                        "fresh_running_snapshot_admitted": True}

            async def task_intent_change():
                task, canonical, projector = await task_scenario()
                entry = await project(projector, task)
                changed = await tasks.mutate(task_owner, task.id, "update", TaskMutation(
                    request_id=uuid4(), expected_revision=task.revision, message="Use the replacement test goal."))
                self.assertGreater(changed.revision, task.revision)
                with self.assertRaises(SourceError):
                    async with canonical.guard(entry.guard_binding):
                        self.fail("Old intent remained current")
                await project(projector, changed)
                return {"canonical_intent_mutation": True, "old_snapshot_rejected": True}

            async def task_manifest_change():
                task, canonical, projector = await task_scenario()
                claim = await tasks.claim_next("responsive-result-sql", ("frontier",), owner_id=task_owner,
                    provider_models={"frontier": task.model})
                self.assertIsNotNone(claim)
                task_context = ContextRepository(tasks)
                request = await task_context._record(connection, task_owner, task.request_event_id)
                manifest = await task_context.store_manifest(claim, watermark=task.context_watermark,
                    records=(request,), content={"fixture": True, "records": [request.as_context()]})
                outcome = await HermesTaskResult(status="succeeded", answer="Synthetic verified result 37.",
                    speech=[{"text": "Synthetic verified result 37."}], evidence=[],
                    completed_operation_refs=[], uncertainties=[]).verified(
                        TaskRunner(tasks, {}).context(claim), SimpleNamespace())
                self.assertTrue(await tasks.finish(claim, outcome))
                entry = await project(projector, task)
                self.assertIn("Synthetic verified result 37.", entry.summary)
                self.assertEqual(entry.status, "succeeded")
                await connection.execute("update jarvis_task_context_manifests set status='superseded' "
                    "where owner_id=$1 and id=$2", task_owner, manifest.id)
                with self.assertRaises(SourceError):
                    async with canonical.guard(entry.guard_binding):
                        self.fail("Result with a superseded manifest remained current")
                return {"real_manifest_and_finish": True, "worker_result_projected": True,
                        "manifest_change_rejected": True}

            for kind in ("working_memory", "visual", "prospective"):
                await check(f"current_{kind}_one_connection_commit", lambda kind=kind: valid_effect(kind))
                await check(f"closed_during_{kind}_rolls_back", lambda kind=kind: rolled_back_effect(kind))
            for operation in ("replace", "cancel"):
                await check(f"focus_{operation}_after_analysis", lambda operation=operation: historical_focus_change(operation))
            await check("deleted_focus_receipt_blocks_bound_brief", removed_receipt)
            await check("deleted_observation_blocks_new_alert", removed_observation)
            for operation in ("cancel", "delete_source"):
                await check(f"concurrent_{operation}_serializes_with_effect", lambda operation=operation: concurrent_effect_revocation(operation))
            for operation in ("replace", "cancel", "delete_source"):
                await check(f"queued_notification_{operation}_blocked_at_write", lambda operation=operation: queued_notification(operation))
            await check("task_status_changes_without_intent_change", task_status_change)
            await check("task_intent_change_revalidates_snapshot", task_intent_change)
            await check("task_result_manifest_change_revalidates_snapshot", task_manifest_change)
            report["status"] = "failed" if failures else "passed"
            save()
        except Exception as error:
            report.update(status="failed", fatal=sanitized(error))
            save()
            raise AssertionError("SQL qualification stopped; see sanitized private report") from None
        finally:
            # No reconnect after an initial unreachable endpoint: no schema was
            # ever requested. Otherwise cleanup always uses the manifested guard
            # and independent post-commit audit, including on failed checks.
            try:
                if rival_pool is not None:
                    try:
                        await asyncio.wait_for(rival_pool.close(), timeout=10)
                    except Exception as error:
                        rival_pool.terminate()
                        report.update(status="failed", rival_close=sanitized(error))
                if fixture.manifest["schemas"]:
                    await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
                else:
                    if connection is not None:
                        await connection.close(timeout=5)
                    report["cleanup"] = {"status": "not_required_no_resources_allocated",
                        "schemas_requested": 0, "independent_remote_audit": "not_attempted"}
            finally:
                save()
                print(json.dumps({"report": str(report_path), "status": report["status"],
                    "checks": [{"name": item["name"], "status": item["status"],
                                **({key: item.get(key) for key in ("error_type", "code", "sqlstate")}
                                   if item["status"] == "failed" else {})}
                               for item in report["checks"]], "cleanup": report["cleanup"]}))
        self.assertFalse(failures, json.dumps(failures))
