"""Opt-in PostgreSQL qualification of visual decisions and owner settlement.

Only the database is remote. Images, camera identity and model responses are
controlled fixtures, not recognition or appliance-safety qualification. All DDL
and writes use one manifested disposable schema and generated fixture owners.
The established cleanup helper audits removal through a new connection.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import uuid4

import asyncpg

from app.knowledge.observations import ObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.episode_context import EpisodeContextRuntime
from app.live_state.models import SessionScope
from app.live_state.perception import (
    CheckResolution, DurableResponsibilityCapability, ObservationalQuestion,
    PerceptionBrief, PerceptionFinding,
)
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from app.perception.applicability import PerceptionEffectGuard
from app.perception.contracts import FastPerceptionFrame, FastPerceptionOutput, FastPerceptionRequest
from app.perception.evidence_consumer import PerceptionEvidenceConsumer, canonical_episode_snapshots
from app.perception.models import ObservationFrameLink
from app.perception.repository import PerceptionRepository
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import (
    AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec,
    VisualResolutionPolicy,
)
from app.reminders.visual_resolution import DECISION_NAMESPACE, EpisodeVisualResolver
from app.runtime.concurrency import PriorityAsyncLimiter
from app.runtime.events import DomainEventBus, DomainEventName
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.tools.invocation import ToolInvocationContext
from app.visual_memory.pipeline import _observation_from_fast_output
from evals.second_brain.fixtures import DatabaseFixture
from test_episode_check_sql import DiagnosticPool
from test_notification_ledger_database import cleanup_fixture_resources
from test_responsive_perception_sql import sanitized


REPORT_FIELDS = ("report_state", "report_event_id", "report_revision", "reported_at")


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested database required",
)
class VisualResolutionSQLTests(unittest.IsolatedAsyncioTestCase):
    async def test_visual_owner_receipt_replay_and_rejected_effects(self):
        dsn = os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"]
        directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        fixture = DatabaseFixture(dsn, directory, test_name=self.id())
        report_path = directory / f"visual-resolution-{fixture.run_id}.json"
        report = {
            "format": "jarvis.visual_resolution.sql.v1", "run_id": fixture.run_id,
            "status": "running", "migration_ceiling": 40, "checks": [],
            "cleanup": None, "provider_calls": False, "public_migration": False,
            "network_retries": 0,
        }
        connection = pool = None
        failures, sql_errors, resolvers, runtimes = [], [], [], []

        def save():
            fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")

        async def check(name, operation):
            before_errors = len(sql_errors)
            try:
                async with asyncio.timeout(80):
                    evidence = await operation()
                self.assertEqual(sql_errors[before_errors:], [], "Unexpected PostgreSQL error")
            except Exception as error:
                entry = {"name": name, "status": "failed", **sanitized(error),
                         "sql_errors": sql_errors[before_errors:]}
                failures.append(entry)
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
            connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
            schema = await fixture.create_schema(connection)
            migrations = tuple(m for m in load_migrations() if m.version <= 40)
            self.assertEqual(migrations[-1].version, 40)
            await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True,
                                                   migrations=migrations), timeout=240)
            path_sql = f"set search_path to {quote_schema(schema)},pg_catalog,extensions"
            await connection.execute(path_sql)
            await connection.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            async def setup(c):
                await c.execute(path_sql)
                await c.execute("set lock_timeout='8s'; set statement_timeout='20s'")

            # Allocate after DDL. A single lease also proves model execution and
            # post-commit events can borrow storage without an outstanding guard.
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=10,
                                            command_timeout=25, setup=setup)
            inspected = DiagnosticPool(pool, sql_errors)
            owner_id = fixture.owners[0]
            area_id = await connection.fetchval(
                "insert into jarvis_knowledge_areas(owner_id,name,domain) "
                "values($1,'Visual predicate SQL fixture','test') returning id", owner_id)
            sources = SourceRepository(inspected)
            tasks = TaskRepository(inspected, TaskLimits())
            publication = CanonicalSourcePublication(sources, area_id)
            context = ContextRepository(tasks, publication=publication)
            publisher = ObservationPublication(context, owner_id=owner_id)
            perceptions = PerceptionRepository(inspected, observation_publication=publisher)
            repository = ResponsibilityRepository(inspected)

            async def capture(scope, when, text):
                record = await context.append(owner_id, ContextIngress(
                    session_id=scope.session_id, source_namespace="qualification:visual-resolution",
                    source_event_id=str(uuid4()), receipt_id=str(uuid4()),
                    kind="user_transcript", trust="user_input", finality="final",
                    source_time=when, payload={"text": text}))
                return ToolInvocationContext(request_event_id=record.id,
                                             request_revision=record.revision), record

            async def row(s, c=None):
                return dict(await (c or connection).fetchrow(
                    "select * from jarvis_responsibility_occurrences where owner_id=$1 and id=$2",
                    owner_id, s.admitted.check.id))

            async def decisions(s, c=None):
                return await (c or connection).fetchval(
                    "select count(*) from jarvis_context_events "
                    "where owner_id=$1 and session_id=$2 and source_namespace=$3",
                    owner_id, s.scope.session_id, DECISION_NAMESPACE)

            async def requests(s):
                return await connection.fetchval(
                    "select count(*) from jarvis_responsibility_requests "
                    "where owner_id=$1 and responsibility_id=$2",
                    owner_id, s.admitted.check.canonical_owner_ref.source_id)

            async def scenario(*, policy=True, visibility="visible"):
                base = datetime.now(UTC) - timedelta(seconds=10)
                clock = [base]
                session_id = await connection.fetchval(
                    "insert into jarvis_sessions(owner_id) values($1) returning id", owner_id)
                scope = SessionScope(owner_id=owner_id, session_id=session_id, area_id=area_id)
                actor = EpisodeCheckOwner(repository, enabled=True, clock=lambda: clock[0])
                self.assertEqual((await actor.initialize()).status, "available")
                source, admission_record = await capture(scope, base,
                    "For this synthetic fixture, track READY and the arm inside its marked zone; "
                    "use visual evidence to resolve only when explicitly configured.")
                configured = VisualResolutionPolicy(authorization="explicit_user_request",
                    condition="The status display reads READY and the arm is inside the marked zone.") if policy else None
                draft = EpisodeCheckDraft(question="What does the display and arm position indicate?",
                    observation_window_expires_at=base + timedelta(minutes=5),
                    visual_resolution=configured)
                admitted = await actor.admit(scope, source, AdmitEpisodeCheck(
                    spec=EpisodeCheckSpec(title="Synthetic readiness condition"), check=draft))
                self.assertIsNotNone(admitted.check)
                clock[0] = base + timedelta(seconds=5)
                canonical = CanonicalContextBinder(sources, scope, clock=lambda: clock[0], timeout_seconds=20)
                canonical.context.publication = publication
                bus = DomainEventBus()
                # No cue/background projection is started for these unconditional
                # checks. The actual runtime's assignment validator and owner
                # guards still execute, rather than a fabricated owner proof.
                episodes = EpisodeContextRuntime(service=None, canonical=canonical, owner=actor,
                    bus=bus, clock=lambda: clock[0])
                runtimes.append(episodes)
                canonical.episode_context = episodes
                binder = CanonicalPerceptionBinder(canonical)
                question = ObservationalQuestion(id=uuid4(), question=draft.question,
                    check_ref=admitted.outcome.check_ref,
                    episode_or_object_ref=admitted.check.episode_or_object_ref)
                brief = PerceptionBrief(id=uuid4(), version=1, scope=scope, scope_epoch=uuid4(),
                    based_on_live_revision=1, questions=(question,),
                    dependencies=(admitted.outcome.canonical_owner_ref, admitted.outcome.check_ref,
                                  admitted.check.episode_or_object_ref),
                    durable_responsibility=DurableResponsibilityCapability(
                        owner_namespace="responsibility", status="available"),
                    created_at=base + timedelta(seconds=1), fresh_until=base + timedelta(minutes=1),
                    expires_at=base + timedelta(minutes=2))
                binding = await binder.bind(brief)
                stream, camera_id = uuid4(), uuid4()
                frames = []
                for ordinal in range(2):
                    captured_at = base + timedelta(seconds=ordinal + 2)
                    identity = await connection.fetchval("""insert into visual_frames
                        (owner_id,session_id,visual_session_id,frame_sequence,source,captured_at,
                         content_type,width,height,storage_bucket,storage_path,raw_image_expires_at)
                        values($1,$2,$3,$4,'glasses',$5,'image/jpeg',32,32,'fixture','not-uploaded',
                               clock_timestamp()+interval '1 hour') returning id""",
                        owner_id, session_id, stream, ordinal, captured_at)
                    frames.append(FastPerceptionFrame(identity, ordinal, captured_at,
                        "glasses", "image/jpeg", b"controlled-offline-frame"))
                request = FastPerceptionRequest(owner_id=owner_id, session_id=session_id,
                    window_id=uuid4(), frames=tuple(frames), previous_working_state={},
                    existing_entity_ids=frozenset(), recent_audio_transcripts=(),
                    behavior_instructions="Observe the synthetic assignment only.",
                    active_reminders=(), active_focus=None, recent_interrupts=(),
                    requested_at=base + timedelta(seconds=3), perception_brief=brief,
                    perception_brief_binding=binding, camera_visual_session_id=stream,
                    camera_connection_id=camera_id)
                request, analysis_input = await publisher.prepare_inputs(request, SimpleNamespace(
                    owner_id=owner_id, session_id=session_id,
                    recent_visual_interrupts=(), recent_audio_transcripts=()))
                finding = PerceptionFinding(question_id=question.id, check_ref=question.check_ref,
                    visibility=visibility, literal_finding="The display reads READY and the arm is inside the zone.",
                    confidence=.99, evidence_frame_ids=tuple(f.frame_id for f in frames))
                output = FastPerceptionOutput(summary="Synthetic readiness panel.", starts_new_event=False,
                    confidence=.99, perception_findings=[finding])
                observation = replace(_observation_from_fast_output(request=request, output=output,
                    provider="offline", model="controlled-fixture"), created_at=base + timedelta(seconds=4))
                await perceptions.insert_observation(observation=observation, analysis_input=analysis_input,
                    frame_links=tuple(ObservationFrameLink(f.frame_id, i,
                        "primary" if f.frame_id == request.primary_frame.frame_id else "context")
                        for i, f in enumerate(frames)))
                original = await connection.fetchval(
                    "select payload_id from jarvis_observation_sources where owner_id=$1 and observation_id=$2",
                    owner_id, observation.id)
                s = SimpleNamespace(scope=scope, clock=clock, actor=actor, canonical=canonical,
                    admitted=admitted, draft=draft, admission_record=admission_record, request=request,
                    question=question, live=True, camera_id=camera_id, stream=stream,
                    calls=0, during_model=None, events=[], rollback_seen=False)
                s.camera = lambda req, active_stream: s.live and (
                    req.camera_connection_id == s.camera_id
                    and req.camera_visual_session_id == active_stream == s.stream)
                s.guard = PerceptionEffectGuard(brief, binding, binder.guard, lambda: s.live, original)
                consumer = PerceptionEvidenceConsumer(publisher, enabled=True, clock=lambda: clock[0],
                    timeout_seconds=30, camera_is_current=s.camera,
                    episode_snapshots=canonical_episode_snapshots(canonical))
                s.evidence = await consumer.consume(request=request, output=output, observation=observation,
                                                   analysis_input=analysis_input, effect_guard=s.guard)
                self.assertEqual(s.evidence.status, "recorded", s.evidence.reason)
                s.limiter = PriorityAsyncLimiter(1)
                s.answer = dict(verdict="satisfied", basis="condition_observed",
                    observed_condition="The status reads READY and the arm is visibly inside the outlined zone.",
                    confidence=.99, uncertainties=[], supporting_frame_ids=["frame:0", "frame:1"])

                async def generate(**kwargs):
                    s.calls += 1
                    self.assertEqual(s.limiter.active_count, 1)
                    self.assertEqual(len(kwargs["images"]), 2)
                    async with pool.acquire() as c:
                        self.assertFalse(c.is_in_transaction(), "The model must run outside guarded SQL")
                    if s.during_model is not None:
                        await s.during_model()
                    return json.dumps(s.answer)

                async def changed(event):
                    # An independent DB connection must observe the committed
                    # result before publication; no transaction-local success.
                    current = await row(s)
                    s.events.append((event, current["check_lifecycle"], await decisions(s)))

                bus.subscribe(DomainEventName.MEMORY_ITEM_CHANGED, changed)
                s.model = SimpleNamespace(provider="offline", model="controlled-predicate",
                                          generate_temporal_json=generate)
                s.resolver = EpisodeVisualResolver(owner=actor, canonical=canonical, model=s.model,
                    limiter=s.limiter, bus=bus, enabled=True, deadline_seconds=60)
                resolvers.append(s.resolver)
                return s

            async def evaluate(s, guard=None, resolver=None):
                return await (resolver or s.resolver).evaluate(request=s.request, evidence=s.evidence,
                    question_id=s.question.id, effect_guard=guard or s.guard, camera_is_current=s.camera)

            async def change(s, operation, **extra):
                source, _ = await capture(s.scope, s.clock[0], "Synthetic explicit " + operation)
                current = s.admitted.check
                return await s.actor.change(s.scope, source, ChangeEpisodeCheck(
                    responsibility_id=current.canonical_owner_ref.source_id,
                    expected_revision=current.canonical_owner_ref.revision, occurrence_id=current.id,
                    expected_check_revision=current.revision,
                    episode_id=current.episode_or_object_ref.source_id,
                    expected_episode_revision=current.episode_or_object_ref.revision,
                    operation=operation, **extra))

            async def withdraw(s):
                source_id = await connection.fetchval(
                    "select source_id from jarvis_knowledge_source_versions where owner_id=$1 and payload_id=$2",
                    owner_id, s.admission_record.payload_id)
                self.assertIsNotNone(source_id)
                await sources.delete_source(owner_id, area_id, source_id,
                                            expected_revision=s.admission_record.revision)

            async def success_and_replay():
                s = await scenario()
                before, before_requests = await row(s), await requests(s)
                first = await evaluate(s)
                self.assertEqual(first.status, "resolved", first.reason)
                settled = await row(s)
                self.assertEqual(settled["check_lifecycle"], "resolved")
                self.assertEqual(settled["check_revision"], before["check_revision"] + 1)
                self.assertIsInstance(settled["check_resolution"], str,
                                      "This qualification must exercise native asyncpg JSON text")
                resolution = CheckResolution.model_validate_json(settled["check_resolution"])
                self.assertEqual(resolution.basis, "visual_observation")
                self.assertEqual({ref.frame_id: ref.captured_at for ref in resolution.source_refs
                                  if ref.frame_id is not None},
                                 {f.frame_id: f.captured_at for f in s.request.frames})
                self.assertEqual({key: settled[key] for key in REPORT_FIELDS},
                                 {key: before[key] for key in REPORT_FIELDS})
                self.assertEqual(settled["report_state"], "unknown")
                self.assertEqual(await requests(s), before_requests)
                self.assertEqual([(state, count) for _, state, count in s.events], [("resolved", 1)])
                # A newly constructed resolver proves replay is not an in-memory
                # cache. Keep original frozen evidence; do not recapture/relabel.
                replacement = EpisodeVisualResolver(owner=s.actor, canonical=s.canonical, model=s.model,
                    limiter=s.limiter, bus=s.resolver.bus, enabled=True, deadline_seconds=60)
                resolvers.append(replacement)
                replay = await evaluate(s, resolver=replacement)
                self.assertEqual(replay.status, "replayed", replay.reason)
                self.assertTrue(replay.owner_result.replayed)
                self.assertEqual(replay.decision_receipt.id, first.decision_receipt.id)
                self.assertEqual(replay.decision_receipt.payload_id, first.decision_receipt.payload_id)
                self.assertEqual(await row(s), settled)
                self.assertEqual(await decisions(s), 1)
                self.assertEqual(await requests(s), before_requests)
                self.assertEqual(s.calls, 1)
                self.assertEqual(len(s.events), 1)
                return {"native_json_text_replay": True, "model_calls": 1, "decision_receipts": 1,
                        "owner_mutations": 1, "original_frame_times": True, "user_report_fields_unchanged": True,
                        "new_user_operation_receipts": 0, "wake_observed_committed_state": True}

            async def ineligible(kind):
                s = await scenario(policy=kind != "no_policy",
                                   visibility=kind if kind in {"not_visible", "unclear"} else "visible")
                before, before_requests = await row(s), await requests(s)
                if kind == "stale":
                    s.clock[0] += timedelta(seconds=21)
                result = await evaluate(s)
                self.assertIn(result.status, {"not_eligible", "not_applied"}, result.reason)
                self.assertEqual(await row(s), before)
                self.assertEqual(await requests(s), before_requests)
                self.assertEqual(await decisions(s), 0)
                self.assertEqual(s.calls, 0)
                self.assertEqual(s.events, [])
                return {"status": result.status, "reason": result.reason, "model_calls": 0,
                        "owner_unchanged": True, "decision_receipts": 0}

            async def changed_while_model_runs(kind):
                s = await scenario()
                expected = {}

                async def mutate():
                    if kind == "cancel":
                        await change(s, "cancel")
                    elif kind == "replace":
                        await change(s, "revise", check=s.draft.model_copy(update={
                            "object_association": "fixture:replacement",
                            "question": "Inspect the replacement instrument."}))
                    elif kind == "withdraw":
                        await withdraw(s)
                    elif kind == "expire":
                        s.clock[0] += timedelta(seconds=21)
                    elif kind == "camera":
                        s.camera_id = uuid4()
                    expected.update(row=await row(s), requests=await requests(s))

                s.during_model = mutate
                result = await evaluate(s)
                self.assertIn(result.status, {"not_applied", "unavailable"}, result.reason)
                self.assertIn("row", expected, "The mutation must actually run while the model owns no DB lock")
                self.assertEqual(await row(s), expected["row"])
                self.assertNotEqual(expected["row"]["check_lifecycle"], "resolved")
                self.assertEqual(await requests(s), expected["requests"])
                self.assertEqual(await decisions(s), 0)
                self.assertEqual(s.calls, 1)
                self.assertEqual(s.events, [])
                return {"status": result.status, "reason": result.reason, "model_calls": 1,
                        "changed_owner_preserved": True, "decision_receipts": 0}

            async def inconclusive():
                s = await scenario()
                before, before_requests = await row(s), await requests(s)
                s.answer.update(verdict="uncertain", basis="insufficient_evidence",
                                uncertainties=["The required position marking is obscured."])
                result = await evaluate(s)
                replay = await evaluate(s)
                self.assertEqual(result.status, "observed_without_resolution", result.reason)
                self.assertEqual(replay.status, "observed_without_resolution", replay.reason)
                self.assertEqual(result.decision_receipt.id, replay.decision_receipt.id)
                self.assertEqual(result.decision_receipt.content["decision"]["uncertainties"],
                                 s.answer["uncertainties"])
                self.assertEqual(await row(s), before)
                self.assertEqual(await requests(s), before_requests)
                self.assertEqual(await decisions(s), 1)
                self.assertEqual(s.calls, 1)
                self.assertEqual(s.events, [])
                return {"uncertainty_preserved": True, "model_calls": 1,
                        "decision_receipts": 1, "owner_unchanged": True}

            async def rollback_after_owner_write():
                s = await scenario()
                before, before_requests = await row(s), await requests(s)

                class FenceBeforeCommit:
                    brief, binding = s.guard.brief, s.guard.binding

                    @asynccontextmanager
                    async def __call__(self):
                        async with s.guard() as c:
                            yield c
                            if await decisions(s, c):
                                # Both writes really happened inside the actual
                                # effect transaction, then its original session
                                # fence rejects commit. This is not a mock error.
                                self_outer.assertTrue(c.is_in_transaction())
                                self_outer.assertEqual((await row(s, c))["check_lifecycle"], "resolved")
                                s.rollback_seen = True
                                s.live = False

                self_outer = self
                result = await evaluate(s, guard=FenceBeforeCommit())
                self.assertTrue(s.rollback_seen, "Reach the post-owner-write transaction boundary")
                self.assertIn(result.status, {"not_applied", "unavailable"}, result.reason)
                self.assertEqual(await row(s), before)
                self.assertEqual(await decisions(s), 0)
                self.assertEqual(await requests(s), before_requests)
                self.assertEqual(s.events, [])
                return {"owner_and_receipt_existed_in_transaction": True,
                        "owner_and_receipt_rolled_back": True, "wake_events": 0}

            await check("native_json_text_replay_exactly_once", success_and_replay)
            for kind in ("no_policy", "not_visible", "unclear", "stale"):
                await check("ineligible_" + kind, lambda kind=kind: ineligible(kind))
            for kind in ("cancel", "replace", "withdraw", "expire", "camera"):
                await check("during_model_" + kind, lambda kind=kind: changed_while_model_runs(kind))
            await check("uncertainty_receipt_without_settlement", inconclusive)
            await check("actual_transaction_rollback_after_owner_write", rollback_after_owner_write)
            report["status"] = "failed" if failures else "passed"
        except Exception as error:
            report.update(status="failed", fatal=sanitized(error), sql_errors=sql_errors)
            save()
            raise AssertionError("Visual resolution SQL qualification stopped; inspect sanitized private report") from None
        finally:
            try:
                for resolver in resolvers:
                    await resolver.close()
                for runtime in runtimes:
                    await runtime.close()
            finally:
                if fixture.manifest["schemas"]:
                    await cleanup_fixture_resources(fixture, report, save, pool=pool, connection=connection)
                else:
                    if connection is not None:
                        await connection.close(timeout=5)
                    report["cleanup"] = {"status": "not_required_no_resources_allocated",
                        "schemas_requested": 0, "independent_remote_audit": "not_attempted"}
                save()
                print(json.dumps({"report": str(report_path), "status": report["status"],
                                  "checks": report["checks"], "cleanup": report["cleanup"]}))
        self.assertFalse(failures, json.dumps(failures))


if __name__ == "__main__":
    unittest.main()
