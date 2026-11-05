"""Offline production owner/binder/consumer/ingress integration.

Only SQL storage is simulated. These tests do not qualify PostgreSQL locks,
provider perception accuracy or physical-action resolution.
"""
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.knowledge.observation_inputs import ObservationInputSnapshot, request_document
from app.knowledge.observations import ObservationPublication, OBSERVATION_NAMESPACE
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope, SourceDependency
from app.live_state.perception import DurableResponsibilityCapability, ObservationalQuestion, PerceptionBrief, PerceptionFinding
from app.live_state.perception_runtime import CanonicalPerceptionBinder
from app.perception.applicability import PerceptionEffectGuard
from app.perception.contracts import FastPerceptionFrame, FastPerceptionOutput, FastPerceptionRequest
from app.perception.evidence_consumer import EVIDENCE_NAMESPACE, PerceptionEvidenceConsumer, canonical_episode_snapshots
from app.perception.live_brief import seal_live_brief_input
from app.reminders.duties import ResponsibilityRepository
from app.reminders.episode_checks import EpisodeCheckOwner
from app.reminders.episode_models import AdmitEpisodeCheck, ChangeEpisodeCheck, EpisodeCheckDraft, EpisodeCheckSpec
from app.runtime.concurrency import PriorityAsyncLimiter
from app.tasks.context_events import ContextRecord
from app.tasks.models import digest
from app.visual_memory.pipeline import VisualAnalysisPipeline, _observation_from_fast_output
from tests.test_live_state_episode_context import IntegrationStore
from tests.test_episode_check_owner import NOW


class EvidenceStore(IntegrationStore):
    """Transaction rollback plus the real ContextRepository's SQL surface."""

    def __init__(self):
        super().__init__()
        self.ingress_receipts, self.context_rows, self.frames = {}, {}, {}
        self.observation = None
        self.on_append = None

    @asynccontextmanager
    async def transaction(self, **kwargs):
        saved = deepcopy((self.copies, self.edges, self.ingress_receipts, self.context_rows))
        try:
            async with super().transaction(**kwargs):
                yield self
        except BaseException:
            self.copies, self.edges, self.ingress_receipts, self.context_rows = saved
            raise

    def original_payload(self, owner, content, parents=()):
        identity = uuid4()
        self.copies[identity] = dict(owner=owner, content=deepcopy(content), parents=set(parents))
        return identity

    def record(self, identity):
        if identity not in self.context_rows:
            return super().record(identity)
        row = self.context_rows[identity]
        payload = self.copies[row["payload_id"]]
        return ContextRecord(**row, availability="available" if payload["content"] is not None else "deleted",
            content=deepcopy(payload["content"]), content_digest="a" * 64, is_current=True)

    async def fetchrow(self, query, *args):
        q = " ".join(query.lower().split())
        if "join jarvis_observation_sources b" in q:
            self.sql(query)
            record = self.observation
            if (record.owner_id, record.session_id, record.source_event_id, record.source_namespace) != (
                    args[0], args[1], str(args[2]), args[3]):
                return None
            return record.model_dump()
        if q.startswith("select * from jarvis_context_ingress_receipts"):
            self.sql(query)
            return self.ingress_receipts.get(args)
        if ("from jarvis_context_events e join jarvis_context_payloads" in q
                and q.endswith("where e.owner_id=$1 and e.source_namespace=$2 "
                               "and e.source_event_id=$3 order by e.revision desc limit 1")):
            self.sql(query)
            rows = [self.record(identity) for identity, row in self.context_rows.items()
                    if (row["owner_id"], row["source_namespace"], row["source_event_id"]) == args]
            return rows[-1].model_dump() if rows else None
        return await super().fetchrow(query, *args)

    async def fetchval(self, query, *args):
        q = " ".join(query.lower().split())
        if q.startswith("select lock_jarvis_copy_source("):
            self.sql(query)
            return "available" if self.available(*args) else "unavailable"
        if q.startswith("insert into jarvis_context_payloads"):
            self.sql(query)
            assert args[1] == "available"
            return self.original_payload(args[0], json.loads(args[2]))
        if q.startswith("insert into jarvis_context_events"):
            self.sql(query)
            identity = uuid4()
            keys = ("owner_id", "session_id", "source_namespace", "source_event_id", "revision",
                    "supersedes_event_id", "kind", "finality", "trust", "source_time", "payload_id")
            self.context_rows[identity] = dict(zip(keys, args, strict=True), id=identity,
                sequence=len(self.context_rows) + 1, origin_sequence=len(self.context_rows) + 1, recorded_at=NOW)
            if self.on_append is not None:
                await self.on_append()
            return identity
        return await super().fetchval(query, *args)

    async def execute(self, query, *args):
        q = " ".join(query.lower().split())
        if q.startswith("insert into jarvis_context_ingress_receipts"):
            self.sql(query)
            self.ingress_receipts[args[:4]] = dict(parameters_digest=args[4], event_id=args[5])
            return
        return await super().execute(query, *args)

    async def fetch(self, query, *args):
        if "from visual_frames" in query:
            self.sql(query)
            return [dict(self.frames[identity]) for identity in args[2] if identity in self.frames
                    and (self.frames[identity]["owner_id"], self.frames[identity]["session_id"]) == args[:2]]
        return await super().fetch(query, *args)


class PerceptionEvidenceConsumerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db, self.now = EvidenceStore(), NOW
        self.scope = SessionScope(owner_id="evidence-owner", session_id=uuid4())
        self.owner = EpisodeCheckOwner(ResponsibilityRepository(self.db), enabled=True, clock=lambda:self.now)
        await self.owner.initialize()
        self.admission = self.db.event(self.scope.owner_id, self.scope.session_id, at=NOW)
        self.check = await self.owner.admit(self.scope, self.admission, AdmitEpisodeCheck(
            spec=EpisodeCheckSpec(title="Check control"),
            check=EpisodeCheckDraft(question="What does the control indicate?",
                observation_window_expires_at=NOW + timedelta(minutes=10))))
        # Admission precedes the brief and captured frames; advancing the test
        # clock is not permission to relabel older images as this episode.
        self.now = NOW + timedelta(seconds=5)

        @asynccontextmanager
        async def transaction(owner):
            assert owner == self.scope.owner_id
            async with self.db.transaction():
                yield self.db

        self.sources = SimpleNamespace(pool=self.db, transaction=transaction)
        self.canonical = CanonicalContextBinder(self.sources, self.scope, clock=lambda:self.now)
        self.db.bind_io(self.canonical)
        self.canonical.episode_context = SimpleNamespace(owner=self.owner, limit=8,
            require_owner=lambda:self.owner, validate_assignment=AsyncMock())
        self.binder = CanonicalPerceptionBinder(self.canonical)
        self.dependencies = (self.check.outcome.canonical_owner_ref, self.check.outcome.check_ref,
                             self.check.check.episode_or_object_ref)
        self.question = ObservationalQuestion(id=uuid4(), question=self.check.check.question,
            check_ref=self.check.outcome.check_ref, episode_or_object_ref=self.check.check.episode_or_object_ref)
        self.brief = PerceptionBrief(id=uuid4(), version=2, scope=self.scope, scope_epoch=uuid4(),
            based_on_live_revision=2, questions=(self.question,), dependencies=self.dependencies,
            durable_responsibility=DurableResponsibilityCapability(owner_namespace="responsibility", status="available"),
            created_at=NOW + timedelta(seconds=1), fresh_until=NOW + timedelta(seconds=10),
            expires_at=NOW + timedelta(seconds=30))
        self.binding = await self.binder.bind(self.brief)
        self.stream, self.stream_live, self.session_live = uuid4(), True, True
        self.request = FastPerceptionRequest(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            window_id=uuid4(), frames=tuple(FastPerceptionFrame(uuid4(), index,
                NOW + timedelta(seconds=index + 1), "glasses", "image/jpeg", b"offline-frame") for index in (1, 2)),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="Observe only.", active_reminders=(), active_focus=None, recent_interrupts=(),
            requested_at=NOW + timedelta(seconds=3), perception_brief=self.brief, perception_brief_binding=self.binding,
            input_context_gaps=({"component":"memory", "reason":"unavailable"},))
        self.document = request_document(self.request)
        self.input_root = self.db.original_payload(self.scope.owner_id,
            {"format":"jarvis.visual_input.v1", "request_digest":digest(self.document), "request":self.document},
            self.binding.payload_root_ids)
        self.capture = ObservationInputSnapshot(self.scope.owner_id, self.scope.session_id,
                                              self.input_root, digest(self.document))
        self.request = seal_live_brief_input(self.request, self.capture)
        self.finding = PerceptionFinding(question_id=self.question.id, check_ref=self.question.check_ref,
            visibility="visible", literal_finding="The controls are visible; the setting cannot be read.",
            confidence=.7, evidence_frame_ids=tuple(frame.frame_id for frame in self.request.frames))
        self.output = FastPerceptionOutput(summary="A control panel.", starts_new_event=False,
            confidence=.8, perception_findings=[self.finding])
        self.observation = replace(_observation_from_fast_output(request=self.request, output=self.output,
            provider="offline", model="fixture"), created_at=NOW + timedelta(seconds=4))
        original = dict(id=str(self.observation.id), owner_id=self.scope.owner_id, session_id=str(self.scope.session_id),
            primary_visual_frame_id=str(self.request.primary_frame.frame_id), summary=self.output.summary,
            started_at=self.observation.started_at.isoformat(), ended_at=self.observation.ended_at.isoformat(),
            created_at=self.observation.created_at.isoformat())
        content = {"record":original, "analysis_input_id":str(self.input_root),
            "frame_links":[{"frame_id":str(frame.frame_id)} for frame in self.request.frames]}
        self.original_root = self.db.original_payload(self.scope.owner_id, content, (self.input_root,))
        self.db.observation = ContextRecord(id=uuid4(), owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            sequence=10, origin_sequence=10, source_namespace=OBSERVATION_NAMESPACE,
            source_event_id=str(self.observation.id), revision=1, supersedes_event_id=None,
            kind="observation", trust="derived", finality="final", source_time=self.observation.ended_at,
            recorded_at=self.observation.created_at, payload_id=self.original_root, availability="available",
            content_digest="b" * 64, content=content, is_current=True)
        self.db.frames = {frame.frame_id:dict(id=frame.frame_id, owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, visual_session_id=self.stream, frame_sequence=frame.frame_sequence,
            captured_at=frame.captured_at, source=frame.source, content_type=frame.content_type)
            for frame in self.request.frames}
        publication = SimpleNamespace(sources=self.sources, area_id=None,
            transaction=lambda tasks, owner:transaction(owner), publish=AsyncMock())
        self.canonical.context.publication = publication
        self.publication = ObservationPublication(self.canonical.context, owner_id=self.scope.owner_id)
        self.guard = PerceptionEffectGuard(self.brief, self.binding, self.binder.guard,
            lambda:self.session_live, self.original_root)
        self.consumer = PerceptionEvidenceConsumer(self.publication, enabled=True, clock=lambda:self.now,
            episode_snapshots=canonical_episode_snapshots(self.canonical),
            stream_is_current=lambda owner, session, stream:self.stream_live and
                (owner, session, stream) == (self.scope.owner_id, self.scope.session_id, self.stream))

    async def consume(self, **changes):
        return await self.consumer.consume(**(dict(request=self.request, output=self.output,
            observation=self.observation, analysis_input=self.capture, effect_guard=self.guard) | changes))

    async def change_check(self, operation, **changes):
        source = self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now)
        return await self.owner.change(self.scope, source, ChangeEpisodeCheck(
            responsibility_id=self.check.outcome.canonical_owner_ref.source_id,
            occurrence_id=self.check.check.id, expected_revision=self.check.outcome.canonical_owner_ref.revision,
            expected_check_revision=self.check.check.revision,
            episode_id=self.check.check.episode_or_object_ref.source_id,
            expected_episode_revision=self.check.check.episode_or_object_ref.revision,
            operation=operation, **changes))

    def first_frame_at(self, captured_at):
        """Arrange an exact synthetic original BEFORE consuming its first result."""
        self.assertFalse(self.db.context_rows)
        first, second = self.request.frames
        self.request = replace(self.request, frames=(replace(first, captured_at=captured_at), second),
                               perception_brief_input_digest=None)
        self.document = request_document(self.request)
        self.capture = replace(self.capture, request_digest=digest(self.document))
        self.db.copies[self.input_root]["content"] = {
            "format":"jarvis.visual_input.v1", "request_digest":self.capture.request_digest,
            "request":self.document}
        self.request = seal_live_brief_input(self.request, self.capture)
        self.db.frames[first.frame_id]["captured_at"] = captured_at
        self.observation = replace(self.observation, started_at=captured_at)
        content = deepcopy(self.db.observation.content)
        content["record"]["started_at"] = captured_at.isoformat()
        self.db.observation = self.db.observation.model_copy(update={"content":content})
        self.db.copies[self.original_root]["content"] = deepcopy(content)

    async def test_all_pre_episode_support_is_rejected_even_with_current_primary_frame(self):
        self.first_frame_at(NOW - timedelta(microseconds=1))
        output = self.output.model_copy(update={"perception_findings":[self.finding.model_copy(
            update={"evidence_frame_ids":(self.request.frames[0].frame_id,)})]})
        result = await self.consume(output=output)
        self.assertEqual(result.reason, "perception_evidence_frame_predates_episode")
        self.assertFalse(self.db.context_rows)

    async def test_mixed_age_support_is_rejected_without_silently_dropping_old_frame(self):
        self.first_frame_at(NOW - timedelta(microseconds=1))
        result = await self.consume()
        self.assertEqual(result.reason, "perception_evidence_frame_predates_episode")
        self.assertFalse(self.db.context_rows)
        self.assertEqual(self.db.occurrences[self.check.check.id]["check_lifecycle"], "tracking")

    async def test_current_episode_support_includes_exact_start_not_brief_creation_cutoff(self):
        self.first_frame_at(NOW)
        result = await self.consume()
        self.assertEqual(result.status, "recorded", result.reason)
        self.assertEqual(result.owner_findings[0].source_refs[0].captured_at, NOW)
        self.assertLess(NOW, self.brief.created_at)

    async def test_old_uncited_context_frame_does_not_disqualify_current_support(self):
        self.first_frame_at(NOW - timedelta(seconds=1))
        output = self.output.model_copy(update={"perception_findings":[self.finding.model_copy(
            update={"evidence_frame_ids":(self.request.primary_frame.frame_id,)})]})
        result = await self.consume(output=output)
        self.assertEqual(result.status, "recorded", result.reason)
        self.assertEqual(len(result.owner_findings[0].source_refs), 1)

    async def test_episode_callback_is_required_and_borrows_actual_effect_connection(self):
        self.consumer.episode_snapshots = None
        self.assertEqual((await self.consume()).reason, "perception_evidence_episode_snapshot_unavailable")
        actual = canonical_episode_snapshots(self.canonical)
        async def snapshots(connection, brief):
            self.assertIs(connection, self.db)
            self.assertTrue(connection.is_in_transaction())
            self.assertIs(brief, self.brief)
            return await actual(connection, brief)
        self.consumer.episode_snapshots = AsyncMock(side_effect=snapshots)
        result = await self.consume()
        self.assertEqual(result.status, "recorded", result.reason)
        self.consumer.episode_snapshots.assert_awaited_once()

    async def test_callback_cannot_substitute_another_scope_revision_or_episode(self):
        actual = canonical_episode_snapshots(self.canonical)
        for changes in ({"scope":self.scope.model_copy(update={"owner_id":"other-owner"})},
                        {"revision":99}, {"episode_or_object_ref":self.question.episode_or_object_ref.model_copy(
                            update={"revision":99})}):
            with self.subTest(changes=tuple(changes)):
                async def snapshots(connection, brief):
                    snapshot, = await actual(connection, brief)
                    return (snapshot.model_copy(update={"check":snapshot.check.model_copy(update=changes)}),)
                self.consumer.episode_snapshots = snapshots
                self.assertEqual((await self.consume()).reason, "perception_evidence_episode_snapshot_mismatch")
                self.assertFalse(self.db.context_rows)

    async def test_withdrawn_or_superseded_owner_admission_source_rejects_evidence(self):
        event = self.db.events[self.admission.request_event_id]
        event["current"] = False
        self.assertEqual((await self.consume()).status, "not_applied")
        event["current"] = True
        event["content"] = None
        self.assertEqual((await self.consume()).status, "not_applied")
        self.assertFalse(self.db.context_rows)

    async def test_actual_owner_guard_and_ingress_preserve_exact_provenance_without_settlement(self):
        before = deepcopy((self.db.rules, self.db.occurrences, self.db.reminders))
        result = await self.consume()
        self.assertEqual(result.status, "recorded", result.reason)
        self.assertEqual(result.evidence.brief, self.brief)
        self.assertEqual(result.evidence.window_id, self.request.window_id)
        self.assertEqual(result.evidence.camera_stream_epoch, self.stream)
        self.assertEqual(result.evidence.analyzed_at, self.observation.created_at)
        self.assertEqual(result.record.source_namespace, EVIDENCE_NAMESPACE)
        self.assertEqual(result.record.source_time, self.observation.ended_at)
        self.assertEqual(self.db.copies[result.record.payload_id]["parents"],
                         {self.original_root, self.input_root, *self.binding.payload_root_ids})
        self.assertEqual(result.record.content["input_context_gaps"], list(self.request.input_context_gaps))
        references = result.owner_findings[0].source_refs
        self.assertEqual(tuple(r.captured_at for r in references), tuple(f.captured_at for f in self.request.frames))
        self.assertTrue(all(r.episode_or_object_ref == self.question.episode_or_object_ref for r in references))
        self.assertTrue(all(r.source.source_id == result.record.id for r in references))
        self.assertEqual(result.owner_findings[0].resolution_status, "validator_unavailable")
        self.assertEqual(before, (self.db.rules, self.db.occurrences, self.db.reminders))
        self.assertTrue(any("for update" in q for q in self.db.queries))
        self.canonical.episode_context.validate_assignment.assert_awaited()

    async def test_each_visibility_retains_evidence_without_completing_any_action(self):
        for visibility in ("not_visible", "unclear", "visible"):
            with self.subTest(visibility=visibility):
                # Separate offline fixture, not a new identity for a retry.
                await self.asyncSetUp()
                output = self.output.model_copy(update={"perception_findings":[
                    self.finding.model_copy(update={"visibility":visibility})]})
                result = await self.consume(output=output)
                self.assertEqual(result.status, "recorded", result.reason)
                row = self.db.occurrences[self.check.check.id]
                self.assertEqual(row["check_lifecycle"], "tracking")
                self.assertIsNone(row["check_resolution"])
                self.assertEqual(row["report_state"], "unknown")
                self.assertEqual(result.owner_findings[0].source_refs[0].visibility, visibility)
                self.assertFalse(self.db.reminders)

    async def test_retry_is_idempotent_but_changed_findings_conflict(self):
        first = await self.consume()
        self.assertEqual(first.status, "recorded", first.reason)
        again = await self.consume()
        self.assertEqual(again.status, "recorded", again.reason)
        self.assertEqual(first.record.id, again.record.id)
        changed = self.output.model_copy(update={"perception_findings":[
            self.finding.model_copy(update={"literal_finding":"Different observation."})]})
        conflict = await self.consume(output=changed)
        self.assertEqual(conflict.reason, "context_receipt_conflict")
        self.assertEqual(len(self.db.context_rows), 1)

    async def test_original_source_withdrawal_and_withdrawn_receipt_never_resurrect(self):
        first = await self.consume()
        self.assertEqual(first.status, "recorded", first.reason)
        self.db.copies[self.original_root]["content"] = None
        self.assertFalse(self.db.available(self.scope.owner_id, first.record.payload_id))
        again = await self.consume()
        self.assertEqual(again.status, "not_applied")
        self.assertEqual(len(self.db.context_rows), 1)

    async def test_cancel_snooze_and_episode_replacement_reject_late_evidence(self):
        for operation, changes in (("cancel", {}), ("snooze", {"until":NOW + timedelta(minutes=2)}),
                ("revise", {"check":EpisodeCheckDraft(question="A replacement question?"), "rebind_episode":True})):
            with self.subTest(operation=operation):
                await self.asyncSetUp()
                await self.change_check(operation, **changes)
                result = await self.consume()
                self.assertEqual(result.status, "not_applied")
                self.assertFalse(self.db.context_rows)

    async def test_unrelated_canonical_activity_does_not_invalidate_fixed_assignment(self):
        self.db.event(self.scope.owner_id, self.scope.session_id, at=self.now, content={"text":"Unrelated task update"})
        result = await self.consume()
        self.assertEqual(result.status, "recorded", result.reason)
        self.assertEqual(result.evidence.brief.based_on_live_revision, 2)

    async def test_cue_authority_failure_is_reused_not_reimplemented(self):
        self.canonical.episode_context.validate_assignment.side_effect = SourceError(
            "live_episode_cue_inapplicable", "Original cue no longer applies.")
        result = await self.consume()
        self.assertEqual(result.reason, "live_episode_cue_inapplicable")
        self.assertFalse(self.db.context_rows)

    async def test_wrong_question_check_revision_frames_and_capture_are_rejected(self):
        variants = (
            {"output":self.output.model_copy(update={"perception_findings":[self.finding.model_copy(update={"question_id":uuid4()})]})},
            {"output":self.output.model_copy(update={"perception_findings":[self.finding.model_copy(update={"check_ref":
                self.question.check_ref.model_copy(update={"revision":99})})]})},
            {"output":self.output.model_copy(update={"perception_findings":[self.finding.model_copy(update={"evidence_frame_ids":(uuid4(),)})]})},
            {"output":self.output.model_copy(update={"perception_findings":[self.finding, self.finding]})},
            {"request":replace(self.request, window_id=uuid4())},
            {"analysis_input":replace(self.capture, session_id=uuid4())},
            {"effect_guard":replace(self.guard, observation_payload_id=uuid4())},
            {"effect_guard":None},
        )
        for changes in variants:
            with self.subTest(changes=tuple(changes)):
                self.assertEqual((await self.consume(**changes)).status, "not_applied")
                self.assertFalse(self.db.context_rows)

    async def test_mixed_old_or_unknown_streams_and_changed_frame_times_fail_closed(self):
        self.stream_live = False
        self.assertEqual((await self.consume()).reason, "perception_evidence_camera_stream_unavailable")
        self.stream_live = True
        self.consumer.stream_is_current = None
        self.assertEqual((await self.consume()).reason, "perception_evidence_camera_stream_unavailable")
        self.consumer.stream_is_current = lambda *args:True
        frame = self.db.frames[self.request.frames[0].frame_id]
        frame["visual_session_id"] = uuid4()
        self.assertEqual((await self.consume()).reason, "perception_evidence_mixed_camera_streams")
        frame["visual_session_id"] = self.stream
        frame["captured_at"] += timedelta(seconds=1)
        self.assertEqual((await self.consume()).reason, "perception_evidence_frame_metadata_changed")
        self.assertFalse(self.db.context_rows)

    async def test_stream_or_session_revocation_during_write_rolls_back(self):
        for attribute in ("stream_live", "session_live"):
            with self.subTest(attribute=attribute):
                await self.asyncSetUp()
                async def revoke():
                    setattr(self, attribute, False)
                self.db.on_append = revoke
                self.assertEqual((await self.consume()).status, "not_applied")
                self.assertFalse(self.db.context_rows)
                self.assertFalse(self.db.ingress_receipts)

    async def test_commit_failure_never_reports_recorded(self):
        self.db.fail_commit = True
        result = await self.consume()
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.record)
        self.assertFalse(self.db.context_rows)

    async def test_deadline_and_cancellation_rollback_without_false_receipt(self):
        entered = asyncio.Event()
        async def wait_forever():
            entered.set()
            await asyncio.Event().wait()
        self.db.on_append = wait_forever
        self.consumer.timeout_seconds = .01
        result = await self.consume()
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(self.db.context_rows)
        self.consumer.timeout_seconds = 1
        entered.clear()
        task = asyncio.create_task(self.consume())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.db.context_rows)

    async def test_disabled_legacy_and_empty_findings_do_not_read_or_write(self):
        before = len(self.db.queries)
        self.consumer.enabled = False
        self.assertEqual((await self.consume()).status, "disabled")
        self.consumer.enabled = True
        legacy = replace(self.request, perception_brief=None, perception_brief_binding=None, perception_brief_input_digest=None)
        self.assertEqual((await self.consume(request=legacy)).status, "no_findings")
        self.assertEqual((await self.consume(output=self.output.model_copy(update={"perception_findings":[]}))).status,
                         "no_findings")
        self.assertEqual(len(self.db.queries), before)

    async def test_expired_assignment_owner_disabled_and_missing_original_reject(self):
        self.now = self.brief.expires_at
        self.assertEqual((await self.consume()).status, "not_applied")
        self.now = NOW + timedelta(seconds=5)
        self.owner._unavailable()
        self.assertEqual((await self.consume()).status, "not_applied")
        self.assertFalse(self.db.context_rows)

    async def test_pipeline_opt_in_runs_real_consumer_and_keeps_existing_scene_flow(self):
        async def prepare(request, context):
            self.assertEqual(request, self.request)
            return request, self.capture
        self.publication.prepare_inputs = prepare
        diagnostics = []
        async def diagnostic(payload):
            diagnostics.append(payload)
        repository = SimpleNamespace(mark_analysis_pending=AsyncMock(), mark_analysis_skipped=AsyncMock(),
            mark_analysis_failed=AsyncMock(), apply_analysis_result=AsyncMock())
        perception = SimpleNamespace(insert_observation=AsyncMock())
        working = SimpleNamespace(apply_observation=AsyncMock())
        analyzer = SimpleNamespace(provider="offline", model="fixture", analyze=AsyncMock(return_value=self.output))
        bridge = SimpleNamespace(attach=lambda request:request, effect_guard_for=lambda request, root:self.guard)
        pipeline = VisualAnalysisPipeline(settings=SimpleNamespace(VISUAL_FAST_PERCEPTION_TIMEOUT_SECONDS=1,
                VISUAL_INTERRUPT_CONFIDENCE_THRESHOLD=.9), repository=repository,
            session_repository=object(), behavior_repository=object(), interrupt_service=SimpleNamespace(maybe_interrupt=AsyncMock()),
            model_limiter=PriorityAsyncLimiter(1), fast_analyzer=analyzer, perception_repository=perception,
            sensory_frame_window=object(), working_memory_service=working,
            live_brief_bridge_for=lambda owner, session:bridge,
            perception_evidence_consumer_for=lambda owner, session:self.consumer)
        pipeline.set_observation_publication(self.publication)
        pipeline._context_builder.build = AsyncMock(return_value=object())
        pipeline._build_fast_request = lambda **kwargs:self.request
        frame = SimpleNamespace(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            id=self.request.primary_frame.frame_id, frame_sequence=2)
        try:
            with patch("app.visual_memory.pipeline.observation_delivery_source", AsyncMock(return_value=self.original_root)), \
                 patch("app.visual_memory.pipeline._observation_from_fast_output", return_value=self.observation):
                await pipeline.enqueue(frame=frame, image_bytes=b"offline", send_diagnostic=diagnostic)
                await pipeline.drain()
            repository.mark_analysis_failed.assert_not_awaited()
            working.apply_observation.assert_awaited_once()
            repository.apply_analysis_result.assert_awaited_once()
            analyzer.analyze.assert_awaited_once()
            self.assertEqual(len(self.db.context_rows), 1)
            self.assertTrue(any(item.get("type") == "visual.perception_evidence" and item["status"] == "recorded"
                                for item in diagnostics))
            pipeline._interrupt_service.maybe_interrupt.assert_not_awaited()
        finally:
            await pipeline.close()
