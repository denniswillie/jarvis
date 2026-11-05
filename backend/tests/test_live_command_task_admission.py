"""Prepared offline admission/source tests. All database/provider readiness is fake.

Uses the actual packet, task request replay/receipt methods and context builder;
no SQL, provider/model, worker process, runtime or command bridge is executed.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

from app.live_state.models import GuardedPayloadBinding, SessionScope, StateCursor
from app.realtime.live_delegation_input import LiveDelegationInput
from app.runtime.feature_config import TaskLimits
from app.tasks.context import TaskContextBuilder
from app.tasks.context_repository import ContextRepository
from app.tasks.live_commands import (
    ASSIGNMENT, FORMAT, PREFIX, PROFILE, TITLE, LiveCommandTaskAdmission,
    live_command_source, request_identity,
)
from app.tasks.models import TaskError, digest, json_object
from app.tasks.readonly import FRAGMENT_FORMAT, readonly_source
from app.tasks.repository import TaskRepository
from app.voice_routes.contracts import RouteIdentity, VoiceContextUpdate
from app.voice_routes.live_adapter import LiveDelegationWindow, LiveTranscriptPart
from app.voice_routes.live_protocol import LiveDelegationRequest, LiveTranscriptFragment


class LiveCommandAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.at = datetime.now(timezone.utc)
        self.scope = SessionScope(owner_id="synthetic-live-command-owner", session_id=UUID(int=1), area_id=UUID(int=2))
        self.route = RouteIdentity(scope=self.scope, route_id=UUID(int=3), route_generation=1)
        self.state_root = UUID(int=4)
        self.context = VoiceContextUpdate(route=self.route, update_id=UUID(int=5),
            cursor=StateCursor(epoch=UUID(int=6), revision=1), mode="snapshot",
            text="Synthetic factual Live State; no selected saved recording yet.",
            guard_binding=GuardedPayloadBinding(scope=self.scope, payload_root_ids=(self.state_root,)),
            created_at=self.at - timedelta(seconds=1), expires_at=self.at + timedelta(seconds=90))
        self.parts = (
            LiveTranscriptPart(UUID(int=10), 1, LiveTranscriptFragment("observed-user", "user",
                "Please edit the recording I just saved.", 10, 90)),
            LiveTranscriptPart(UUID(int=11), 1, LiveTranscriptFragment("observed-assistant", "assistant",
                "I am checking which recording you mean.", 95, 120)),
        )
        self.window = LiveDelegationWindow(LiveDelegationRequest("delegation-event", "delegation-not-a-tool-call", 125),
                                           self.parts, 5)
        self.records, self.payloads, self.requests = {}, {}, {}
        self.inserts, self.sequence, self.next_id = 0, 0, 100
        self.task = None
        self.has_grants = self.orphan = False
        self.session_ok = self.area_ok = True
        self.payloads[self.state_root] = {"availability": "available", "content": {"fixture": "guarded state"}}
        originals = []
        for part in self.parts:
            fragment = part.fragment
            body = {"format": FRAGMENT_FORMAT, "voice_route": "live", "route": self.route.model_dump(mode="json"),
                "audio_provider": "openai", "audio_session_id": "synthetic-audio", "caption_id": str(part.transcript_id),
                "caption_revision": part.revision, "provider_event_id": fragment.event_id,
                "speaker": fragment.speaker, "delta": fragment.delta, "start_ms": fragment.start_ms,
                "end_ms": fragment.end_ms, "speech_finality": "provisional", "request_boundary": "unavailable"}
            originals.append(self.record("live-observation", fragment.event_id, "observation", body))
        self.originals = originals
        self.packet = LiveDelegationInput.from_window(self.route, "synthetic-audio", self.window, originals, self.context)
        self.connection = SimpleNamespace(is_in_transaction=Mock(return_value=True),
            fetchrow=AsyncMock(side_effect=self.fetchrow), fetchval=AsyncMock(side_effect=self.fetchval),
            fetch=AsyncMock(side_effect=self.fetch), execute=AsyncMock(side_effect=self.execute))
        self.tasks = TaskRepository(object(), TaskLimits())
        self.tasks.lock_owner = AsyncMock()
        self.tasks._queue_capacity = AsyncMock()
        self.tasks.load = AsyncMock(side_effect=lambda *a, **kw: self.task)
        self.tasks.payload = AsyncMock(side_effect=self.payload)
        self.tasks.emit = AsyncMock()
        self.registry = SimpleNamespace(require=AsyncMock())
        self.service = SimpleNamespace(repository=self.tasks, registry=self.registry,
            providers={"frontier": SimpleNamespace(model="synthetic-existing-frontier")})
        self.admission = LiveCommandTaskAdmission(self.service)
        self.admission.sources.append_in_transaction = AsyncMock(side_effect=self.append)
        for target, replacement in (
            ("app.tasks.live_commands.now", lambda: self.at),
            ("app.tasks.context_repository.ContextRepository._record", AsyncMock(side_effect=self.get_record)),
            ("app.tasks.context_repository.ContextRepository._records", AsyncMock(side_effect=self.get_records)),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def identity(self):
        self.next_id += 1
        return UUID(int=self.next_id)

    def record(self, namespace, event, kind, body, *, source_time=None):
        self.sequence += 1
        identity, payload = self.identity(), self.identity()
        record = SimpleNamespace(id=identity, payload_id=payload, owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, source_namespace=namespace, source_event_id=event,
            kind=kind, trust="derived", finality="final", revision=1, is_current=True,
            availability="available", content_digest=digest(body), content=body,
            sequence=self.sequence, origin_sequence=self.sequence, source_time=source_time,
            authorizes_request=False)
        record.as_context = lambda r=record: {"event_id": str(r.id), "sequence": r.sequence,
            "origin_sequence": r.origin_sequence, "kind": r.kind, "trust": r.trust,
            "finality": r.finality, "content": r.content}
        self.records[identity] = record
        self.payloads[payload] = {"availability": "available", "content": body}
        return record

    async def get_record(self, connection, owner, identity):
        value = self.records[identity]
        if value.owner_id != owner:
            raise TaskError("context_event_not_found", "Synthetic wrong-owner source", 404)
        return value

    async def get_records(self, connection, owner, identities):
        # Reversed mapping order proves the admission follows the received refs.
        return {identity: await self.get_record(connection, owner, identity) for identity in reversed(identities)}

    async def append(self, connection, owner, event):
        self.assertIs(connection, self.connection)
        self.assertEqual(owner, self.scope.owner_id)
        return self.record(event.source_namespace, event.source_event_id, event.kind, event.payload,
                           source_time=event.source_time)

    async def payload(self, connection, owner, content, **options):
        self.assertIs(connection, self.connection)
        identity = self.identity()
        self.payloads[identity] = {"availability": "available", "content": content}
        return identity

    async def fetchrow(self, query, *args):
        if "from jarvis_task_requests" in query:
            return self.requests.get(args[1])
        if "from jarvis_context_payloads" in query:
            return self.payloads.get(args[1])
        raise AssertionError("Unexpected synthetic fetchrow: " + query)

    async def fetchval(self, query, *args):
        if "insert into jarvis_background_tasks" in query:
            self.inserts += 1
            self.task = SimpleNamespace(id=self.identity(), owner_id=args[0], session_id=args[1],
                knowledge_area_id=args[2], backend_kind=args[3], model=args[4], description_payload_id=args[5],
                request_event_id=args[6], request_revision=args[7], revision=1, context_watermark=args[8],
                budget=TaskLimits.model_validate_json(args[9]), title=TITLE, requested_outcome=ASSIGNMENT,
                mode="consult", current_attempt_id=None)
            return self.task.id
        if "from jarvis_context_events" in query:
            return self.orphan or any(r.source_namespace == args[1] and r.source_event_id == args[2]
                                      for r in self.records.values())
        if "from jarvis_sessions" in query:
            return self.session_ok and args == (self.scope.owner_id, self.scope.session_id)
        if "from jarvis_knowledge_areas" in query:
            return self.area_ok and args == (self.scope.owner_id, self.scope.area_id)
        if "from jarvis_task_grants" in query:
            return self.has_grants
        if "lock_jarvis_copy_source" in query:
            return self.payloads.get(args[1], {}).get("availability", "missing")
        raise AssertionError("Unexpected synthetic fetchval: " + query)

    async def fetch(self, query, *args):
        if "from jarvis_context_payloads" in query:
            return [{"id": identity, "availability": self.payloads[identity]["availability"]}
                    for identity in args[1] if identity in self.payloads]
        if "from jarvis_task_grants" in query:
            return []
        raise AssertionError("Unexpected synthetic fetch: " + query)

    async def execute(self, query, *args):
        if "insert into jarvis_task_requests" not in query:
            raise AssertionError("Unexpected synthetic execute: " + query)
        self.requests[args[1]] = {"owner_id": args[0], "request_id": args[1], "task_id": args[2],
            "operation": args[3], "parameters_digest": args[4], "accepted_revision": args[5]}
        return "INSERT 0 1"

    async def admit(self, packet=None, *, command_catalog=()):
        return await self.admission.admit(self.connection, packet or self.packet, command_catalog=command_catalog)

    async def test_real_request_replay_and_receipt_use_one_connection_and_no_new_capability(self):
        first, again = await self.admit(), await self.admit()
        self.assertEqual(first.task.id, again.task.id)
        self.assertEqual(first.expires_at, again.expires_at)
        self.assertEqual(self.inserts, 1)
        self.admission.sources.append_in_transaction.assert_awaited_once()
        self.registry.require.assert_awaited_once_with(self.scope.owner_id, "frontier", "synthetic-existing-frontier",
                                                     connection=self.connection)
        self.assertNotIn("required_capabilities", self.registry.require.await_args.kwargs)
        self.assertEqual(first.delegation_id, self.window.request.delegation_id)
        self.assertFalse(first.source.authorizes_request)
        self.assertFalse(first.source.content["direct_action_authority"])
        self.assertEqual((first.source.kind, first.source.trust), ("task_update", "derived"))
        self.assertEqual(first.task.budget.max_attempts, 1)
        self.assertLessEqual(first.task.budget.active_seconds, 120)
        self.assertLessEqual(first.task.budget.tool_steps, 12)
        self.assertEqual(first.expires_at, self.at + timedelta(seconds=120))

    async def test_order_omissions_provisional_sources_and_roots_without_second_transcript(self):
        receipt = await self.admit()
        parameters = receipt.source.content["parameters"]
        self.assertEqual([item["id"] for item in parameters["speech_sources"]], [str(r.id) for r in self.originals])
        self.assertEqual(parameters["window"], self.packet.as_context()["window"])
        self.assertEqual(parameters["window"]["omitted_prefix_parts"], 3)
        self.assertEqual(parameters["window"]["speech_finality"], "unavailable")
        self.assertNotIn(self.parts[0].fragment.delta, json.dumps(receipt.source.content))
        self.assertNotIn("provider_call_id", json.dumps(receipt.source.content))
        ingress = self.admission.sources.append_in_transaction.await_args.args[2]
        self.assertEqual(set(ingress.source_payload_ids), {self.state_root, *(r.payload_id for r in self.originals)})
        self.tasks.payload.assert_awaited_once()
        self.assertEqual(self.tasks.payload.await_args.kwargs["source_payload_ids"], (receipt.source.payload_id,))
        self.assertTrue(all(r.is_current and r.availability == "available" for r in self.originals))
        self.assertTrue(all(r.kind != "user_transcript" for r in self.records.values()))

    async def test_same_delegation_different_original_packet_conflicts_without_effect(self):
        await self.admit()
        different = replace(self.window, request=replace(self.window.request, event_id="changed-delegation-event"))
        packet = LiveDelegationInput.from_window(self.route, "synthetic-audio", different, self.originals, self.context)
        with self.assertRaises(TaskError) as error:
            await self.admit(packet)
        self.assertEqual(error.exception.code, "task_request_conflict")
        self.assertEqual(self.inserts, 1)
        self.registry.require.assert_awaited_once()
        self.assertNotEqual(request_identity(self.route, "id"),
                            request_identity(self.route.model_copy(update={"route_generation": 2}), "id"))

    async def test_replay_does_not_depend_on_fresh_provider_readiness(self):
        first = await self.admit()
        self.registry.require.side_effect = AssertionError("No new provider qualification on replay")
        self.assertEqual((await self.admit()).task.id, first.task.id)
        self.assertEqual(self.inserts, 1)

    async def test_uncertain_lookup_never_creates_another_task(self):
        first = await self.admit()
        with patch.object(self.tasks, "_replay", AsyncMock(side_effect=TimeoutError("synthetic lookup uncertainty"))):
            with self.assertRaises(TimeoutError):
                await self.admit()
        self.assertEqual(self.inserts, 1)
        self.assertEqual((await self.admit()).task.id, first.task.id)
        self.admission.sources.append_in_transaction.assert_awaited_once()

    async def test_orphan_source_is_unknown_not_permission_to_create(self):
        self.orphan = True
        with self.assertRaises(TaskError) as error:
            await self.admit()
        self.assertEqual(error.exception.code, "live_command_admission_unknown")
        self.assertEqual(self.inserts, 0)
        self.admission.sources.append_in_transaction.assert_not_awaited()
        self.registry.require.assert_not_awaited()

    async def test_missing_qualification_session_or_area_has_no_admission_effect(self):
        self.registry.require.side_effect = TaskError("worker_not_ready", "Synthetic unqualified provider")
        with self.assertRaises(TaskError):
            await self.admit()
        self.registry.require.side_effect = None
        for flag in ("session_ok", "area_ok"):
            setattr(self, flag, False)
            with self.subTest(flag=flag), self.assertRaises(TaskError):
                await self.admit()
            setattr(self, flag, True)
        self.assertEqual(self.inserts, 0)
        self.admission.sources.append_in_transaction.assert_not_awaited()

    async def test_foreign_changed_or_withdrawn_canonical_observations_fail(self):
        original = self.originals[0]
        for field, changed in (("owner_id", "other"), ("session_id", uuid4()), ("is_current", False),
                               ("availability", "deleted"), ("content_digest", "b" * 64)):
            old = getattr(original, field)
            setattr(original, field, changed)
            with self.subTest(field=field), self.assertRaises(TaskError):
                await self.admit()
            setattr(original, field, old)
        self.assertEqual(self.inserts, 0)

    async def test_no_user_speech_and_forged_packet_text_do_not_gain_authority(self):
        for record in self.originals:
            record.content["speaker"] = "assistant"
        with self.assertRaises(TaskError):
            await self.admit()
        for record, part in zip(self.originals, self.parts, strict=True):
            record.content["speaker"] = part.fragment.speaker
        raw = self.packet.as_context()
        raw["dialogue"][0]["delta"] = "Fabricated request"
        with self.assertRaises(TaskError):
            await self.admit(replace(self.packet, context_json=json.dumps(raw)))
        self.assertEqual(self.inserts, 0)

    async def test_live_state_withdrawal_and_parent_grants_are_rejected(self):
        receipt = await self.admit()
        self.payloads[self.state_root]["availability"] = "deleted"
        with self.assertRaises(TaskError):
            await live_command_source(self.tasks, self.connection, receipt.task)
        self.payloads[self.state_root]["availability"] = "available"
        self.has_grants = True
        with self.assertRaises(TaskError) as error:
            await live_command_source(self.tasks, self.connection, receipt.task)
        self.assertEqual(error.exception.code, "live_command_parent_has_grants")
        self.assertFalse(receipt.source.authorizes_request)

    async def test_context_expiry_does_not_cancel_an_admitted_request(self):
        receipt = await self.admit()
        original_expiry = receipt.source.content["parameters"]["live_state"]["expires_at"]
        self.at = self.context.expires_at + timedelta(seconds=1)
        self.assertLess(self.at, receipt.expires_at)
        self.assertIs(await live_command_source(self.tasks, self.connection, receipt.task), receipt.source)
        self.assertEqual((await self.admit()).task.id, receipt.task.id)
        self.assertEqual(receipt.source.content["parameters"]["live_state"]["expires_at"], original_expiry)
        self.assertEqual(self.inserts, 1)

    async def test_already_stale_context_retains_its_timestamp_not_request_authority(self):
        stale = self.context.model_copy(update={"expires_at": self.at - timedelta(seconds=1)})
        packet = LiveDelegationInput.from_window(self.route, "synthetic-audio", self.window,
                                                self.originals, stale)
        receipt = await self.admit(packet)
        self.assertEqual(receipt.expires_at, self.at + timedelta(seconds=120))
        self.assertEqual(receipt.source.content["parameters"]["live_state"]["expires_at"],
                         stale.expires_at.isoformat())
        self.assertFalse(receipt.source.content["direct_action_authority"])

    async def test_phone_context_bound_matches_the_live_packet(self):
        context = {"visible_resources_description": "x" * 20000}
        receipt = await self.admit(self.application_packet(context))
        self.assertEqual(receipt.source.content["parameters"]["application_context"], context)

    async def test_expiry_does_not_delete_originals_or_renew_on_replay(self):
        receipt = await self.admit()
        self.at = receipt.expires_at + timedelta(seconds=1)
        with self.assertRaises(TaskError):
            await self.admit()
        self.assertEqual(self.inserts, 1)
        self.assertTrue(all(self.payloads[r.payload_id]["availability"] == "available" for r in self.originals))
        self.assertTrue(all(r.is_current for r in self.originals))
        self.assertFalse(hasattr(self.admission, "cancel"))  # Route gating belongs to Core, not original deletion.

    async def test_profile_receipt_description_and_budget_are_bound(self):
        receipt = await self.admit()
        key = request_identity(self.route, receipt.delegation_id)
        original = dict(self.requests[key])
        for field, value in (("task_id", uuid4()), ("parameters_digest", "c" * 64), ("accepted_revision", 2)):
            self.requests[key] = original | {field: value}
            with self.subTest(field=field), self.assertRaises(TaskError):
                await live_command_source(self.tasks, self.connection, receipt.task)
        self.requests[key] = original
        receipt.task.budget = receipt.task.budget.model_copy(update={"max_attempts": 2})
        with self.assertRaises(TaskError):
            await live_command_source(self.tasks, self.connection, receipt.task)

    async def test_readonly_and_other_namespaces_are_not_relaxed(self):
        receipt = await self.admit()
        self.assertIsNone(await readonly_source(self.tasks, self.connection, receipt.task))
        receipt.source.source_namespace = "voice:ordinary"
        self.assertIsNone(await live_command_source(self.tasks, self.connection, receipt.task))

    async def test_source_readonly_flag_and_context_problem_fallback(self):
        receipt = await self.admit()
        self.connection.fetchval.reset_mock()
        self.assertIs(await live_command_source(self.tasks, self.connection, receipt.task, lock=False), receipt.source)
        self.assertFalse(any("for share" in call.args[0].lower() or "lock_jarvis_copy_source" in call.args[0]
                             for call in self.connection.fetchval.await_args_list))
        self.tasks._source_available = AsyncMock(return_value=False)
        with patch("app.tasks.repository.task_command_source", AsyncMock(return_value=None)):
            self.assertIsNone(await self.tasks.context_problem(self.connection, receipt.task, lock=False))
            self.has_grants = True
            self.assertEqual(await self.tasks.context_problem(self.connection, receipt.task, lock=False),
                             "live_command_parent_has_grants")

    async def test_actual_builder_uses_canonical_originals_and_exact_request_pointer(self):
        receipt = await self.admit()
        snapshot = {"watermark": receipt.source.sequence, "total": len(self.records), "pinned_total": 0,
            "pinned": [], "recent": [*self.originals, receipt.source], "selected_pinned_ids": set()}
        repository = SimpleNamespace(snapshot=AsyncMock(return_value=snapshot),
            store_manifest=AsyncMock(side_effect=lambda claim, **kw: SimpleNamespace(content=kw["content"])))
        manifest = await TaskContextBuilder(repository).build(SimpleNamespace(task=receipt.task))
        self.assertEqual(manifest.content["task"]["request_source"]["event_id"], str(receipt.source.id))
        included = {row["event_id"]: row for row in manifest.content["messages"]}
        self.assertEqual(included[str(self.originals[0].id)]["content"]["delta"], self.parts[0].fragment.delta)
        self.assertEqual(included[str(self.originals[0].id)]["content"]["speech_finality"], "provisional")
        self.assertIn("using the supplied typed command tools", receipt.task.requested_outcome)
        self.assertIn("not merely summarizing", receipt.task.requested_outcome)
        self.assertEqual(manifest.content["history"]["source_reader"], "read_task_context_source")


    def catalog(self):
        return [{"type": "function", "name": "start_media_task",
            "description": "Synthetic server-offered command; no real effect in this test.",
            "parameters": {"type": "object", "properties": {"instruction": {"type": "string"}},
                           "required": ["instruction"], "additionalProperties": False}}]

    def application_packet(self, context):
        raw = self.packet.as_context()
        raw["application_context"] = context
        return replace(self.packet, context_json=json.dumps(raw, allow_nan=True))

    async def test_catalog_is_explicit_immutable_server_offer_and_exact_replay_input(self):
        offered = self.catalog()
        receipt = await self.admit(command_catalog=offered)
        persisted = receipt.source.content["parameters"]["command_catalog"]
        self.assertEqual(persisted, offered)
        offered[0]["description"] = "Caller changed its own later catalog"
        self.assertNotEqual(persisted, offered)
        self.assertEqual((await self.admit(command_catalog=self.catalog())).task.id, receipt.task.id)
        for changed in (offered, [], self.catalog() + [{"name": "another_tool", "parameters": {"type": "object"}}]):
            with self.subTest(changed=changed), self.assertRaises(TaskError) as error:
                await self.admit(command_catalog=changed)
            self.assertEqual(error.exception.code, "task_request_conflict")
        self.assertEqual(self.inserts, 1)
        self.assertFalse(self.has_grants)
        self.assertIs(await live_command_source(self.tasks, self.connection, receipt.task), receipt.source)

    async def test_application_context_does_not_supply_command_catalog_or_grants(self):
        application = {"command_catalog": self.catalog(), "grant": "not-authority",
            "selected_file": {"file_id": str(UUID(int=900)), "version_id": "opaque/files:revision 7"}}
        receipt = await self.admit(self.application_packet(application))
        self.assertEqual(receipt.source.content["parameters"]["application_context"], application)
        self.assertEqual(receipt.source.content["parameters"]["command_catalog"], [])
        self.assertFalse(receipt.source.content["direct_action_authority"])
        self.assertFalse(receipt.source.authorizes_request)
        self.assertFalse(self.has_grants)

    async def test_catalog_non_json_oversized_duplicate_and_non_function_rejected_before_effects(self):
        invalid = [
            None, {"name": "not-a-list"}, [*self.catalog()] * 33, self.catalog() * 2,
            [{"name": "bad", "type": "shell", "parameters": {"type": "object"}}],
            [{"name": "bad", "parameters": {"type": "array"}}],
            [{"name": "bad", "parameters": {"type": "object", "value": float("nan")}}],
            [{"name": "bad", "parameters": {"type": "object", "value": float("inf")}}],
            [{"name": "bad", "parameters": {"type": "object", "value": object()}}],
            [{"name": "bad", "parameters": {"type": "object", 3: "non-string-key"}}],
            [{"name": "bad", "parameters": {"type": "object"}, "description": "x" * 32769}],
        ]
        for value in invalid:
            with self.subTest(value=repr(value)[:100]), self.assertRaises(TaskError) as error:
                await self.admit(command_catalog=value)
            self.assertEqual(error.exception.code, "live_command_catalog_invalid")
        self.assertEqual(self.inserts, 0)
        self.tasks.lock_owner.assert_not_awaited()
        self.admission.sources.append_in_transaction.assert_not_awaited()

    async def test_opaque_version_and_application_context_survive_originals_and_replay(self):
        # 256 UTF-8 bytes, not a UUID; preserve whitespace and original spelling.
        version = " " + "\u00e9" * 127 + " "
        context = {"selected_file": {"file_id": str(UUID(int=900)), "version_id": version},
                   "visible_files": [], "recording_hint": {"state": "saved", "receipt_event_id": str(UUID(int=901))}}
        packet = self.application_packet(context)
        receipt = await self.admit(packet, command_catalog=self.catalog())
        self.assertEqual(receipt.source.content["parameters"]["application_context"], context)
        self.assertEqual((await self.admit(packet, command_catalog=self.catalog())).task.id, receipt.task.id)
        self.assertIs(await live_command_source(self.tasks, self.connection, receipt.task, lock=False), receipt.source)
        context["selected_file"]["version_id"] = "another-version"
        with self.assertRaises(TaskError) as error:
            await self.admit(self.application_packet(context), command_catalog=self.catalog())
        self.assertEqual(error.exception.code, "task_request_conflict")
        self.assertEqual(self.inserts, 1)
        self.assertNotIn(self.parts[0].fragment.delta, json.dumps(receipt.source.content))

    async def test_application_context_limits_and_utf8_version_limit_are_fail_closed(self):
        deep = {}
        for _ in range(14):
            deep = {"child": deep}
        values = [
            [], {"version_id": "x" * 257}, {"version_id": "\u00e9" * 129},
            {"version_id": 123}, {"nested": {"value": float("nan")}},
            {"nested": {"value": float("inf")}}, {"large": "x" * 40001}, deep,
        ]
        for context in values:
            with self.subTest(context=repr(context)[:100]), self.assertRaises(TaskError) as error:
                await self.admit(self.application_packet(context))
            self.assertEqual(error.exception.code, "live_command_application_context_invalid")
        self.assertEqual(self.inserts, 0)
        self.admission.sources.append_in_transaction.assert_not_awaited()

    async def test_validated_source_detects_catalog_change_and_builder_keeps_offer_as_context(self):
        receipt = await self.admit(command_catalog=self.catalog())
        snapshot = {"watermark": receipt.source.sequence, "total": len(self.records), "pinned_total": 0,
            "pinned": [], "recent": [*self.originals, receipt.source], "selected_pinned_ids": set()}
        repository = SimpleNamespace(snapshot=AsyncMock(return_value=snapshot),
            store_manifest=AsyncMock(side_effect=lambda claim, **kw: SimpleNamespace(content=kw["content"])))
        manifest = await TaskContextBuilder(repository).build(SimpleNamespace(task=receipt.task))
        admission_message = next(row for row in manifest.content["messages"] if row["event_id"] == str(receipt.source.id))
        self.assertEqual(admission_message["content"]["parameters"]["command_catalog"], self.catalog())
        receipt.source.content["parameters"]["command_catalog"] = []
        with self.assertRaises(TaskError) as error:
            await live_command_source(self.tasks, self.connection, receipt.task)
        self.assertEqual(error.exception.code, "live_command_receipt_unavailable")


if __name__ == "__main__":
    unittest.main()
