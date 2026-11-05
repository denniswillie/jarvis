"""Shared voice task contracts; provider-native acceptance is reported separately."""
from copy import deepcopy
import json
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.assistant.base_prompt import (
    build_base_prompt, CONVERSATION_STYLE_PROMPT_LINES, DELEGATION_PROMPT_LINES,
)
from app.realtime.gemini.instructions import build_gemini_instructions
from app.realtime.gemini.operations import native_declarations
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.models import TaskError, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.service import TaskService
from app.tasks.voice_tools import BackgroundTaskToolModule, TASK_TOOL_NAMES
from app.tools.executor import JarvisToolExecutor
from app.tools.invocation import CanonicalRequestIdentity, ToolInvocationContext, bind_tool_invocation
from app.tools.module import ToolBuildContext
from app.tools.registry import create_tool_registry


class VoiceToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.session = "fixture-owner", uuid4()
        now = datetime.now(UTC)
        self.source = ContextRecord(id=uuid4(), owner_id=self.owner, session_id=self.session,
            sequence=2, origin_sequence=1, source_namespace="fixture-voice", source_event_id="input-1",
            revision=2, supersedes_event_id=uuid4(), kind="user_transcript", trust="user_input",
            finality="final", source_time=now, recorded_at=now, payload_id=uuid4(),
            availability="available", content_digest="a"*64, content={"text": "Check this synthetic code"},
            is_current=True)
        self.task = SimpleNamespace(id=uuid4(), owner_id=self.owner, session_id=self.session,
            knowledge_area_id=None, revision=1, model="fixture-model", title="Synthetic task", requested_outcome="Check synthetic code", mode="consult",
            status=TaskStatus.QUEUED, reason=None, pending_status=None, updated_at=now, budget=TaskLimits())
        self.repository = SimpleNamespace(
            pool=None, reference=TaskRepository.reference, replay_create=AsyncMock(return_value=None),
            create=AsyncMock(return_value=self.task), resolve_reference=AsyncMock(return_value=self.task.id),
            list=AsyncMock(return_value=[self.task]), voice_detail=AsyncMock(),
            replay_mutation=AsyncMock(return_value=None), mutate=AsyncMock(return_value=self.task))
        self.provider = SimpleNamespace(model="fixture", execute=AsyncMock())
        self.service = TaskService(self.repository, {"frontier": self.provider})
        self.module = BackgroundTaskToolModule(self.service)
        self.registry = create_tool_registry(modules=(self.module,),
            context=ToolBuildContext(None, self.owner, self.session))
        self.args = {"title": "Synthetic task", "requested_outcome": "Check synthetic code", "mode": "consult"}

    async def invoke(self, name, args, *, source=True, call_id="call-1"):
        with patch("app.tasks.voice_tools.ContextRepository.get", AsyncMock(return_value=self.source)):
            with bind_tool_invocation(provider_call_id=call_id,
                    request_event_id=self.source.id if source else None,
                    request_revision=self.source.revision if source else None):
                return await self.registry.get(name).handler(args)

    async def test_schema_parity_and_no_model_selected_identity_or_authority(self):
        self.assertEqual({item.name for item in self.registry.list()}, TASK_TOOL_NAMES)
        gpt = {item["name"]: item["parameters"] for item in self.registry.openai_realtime_tools()}
        gemini = {item["name"]: item["parametersJsonSchema"] for item in native_declarations(self.registry)}
        self.assertEqual(gpt, gemini)
        for schema in gpt.values():
            self.assertFalse(schema["additionalProperties"])
            for forbidden in ("owner_id", "session_id", "request_event_id", "request_id", "grant_ids",
                              "provider_url", "api_key", "backend_kind"):
                self.assertNotIn(forbidden, schema["properties"])
        self.assertEqual(self.module.build_tools(ToolBuildContext(None, self.owner, None)), [])

    async def test_acceptance_returns_without_executing_provider(self):
        result = await self.invoke("start_background_task", self.args)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["task"]["state"], "queued")
        self.assertIsNone(result["task"]["knowledge_area_id"])
        self.assertEqual(result["external_action_grants"], [])
        self.provider.execute.assert_not_awaited()
        kwargs = self.repository.create.await_args.kwargs
        self.assertEqual(kwargs["source"].event_id, self.source.id)
        self.assertEqual(kwargs["source"].session_id, self.session)
        self.assertEqual(kwargs["mode"], "consult")

    async def test_network_request_uses_same_task_service_without_external_action_authority(self):
        self.service.create = AsyncMock(return_value=self.task)
        result = await self.invoke("start_background_task", self.args | {
            "requested_capabilities": ["sandbox_network", "publish_files"]})
        request = self.service.create.await_args.args[1]
        self.assertEqual(request.execution_environment, "networked_sandbox")
        self.assertTrue(request.publish_artifacts)
        self.assertEqual(result["capabilities"],
                         ["reasoning", "session_context", "sandbox_compute", "sandbox_network", "publish_files"])
        self.assertEqual(result["external_action_grants"], [])
        self.provider.execute.assert_not_awaited()
        result = await self.invoke("start_background_task", self.args | {
            "requested_capabilities": ["publish_files"]})
        self.assertEqual(self.service.create.await_args.args[1].execution_environment, "offline_sandbox")
        self.assertNotIn("sandbox_network", result["capabilities"])

    async def test_every_task_tool_exposes_known_delivery_unavailability(self):
        self.task.knowledge_area_id = uuid4()
        self.repository.voice_detail.return_value = (self.task, {
            "result": None, "result_unavailable_reason": None, "last_milestone": None})
        reference = {"task_ref": self.repository.reference(self.task.id), "expected_revision": 1}
        expected = {"route": "direct_tts", "state": "unavailable", "reason": "attention_coordinator_unavailable"}
        for name, arguments in (("start_background_task", self.args),
                ("list_background_tasks", {}), ("get_background_task", reference),
                ("update_background_task", reference | {"message": "Updated task"}),
                ("cancel_background_task", reference), ("resume_background_task", reference)):
            with self.subTest(tool=name):
                result = await self.invoke(name, arguments)
                self.assertEqual(result["status"], "success")
                tasks = result["tasks"] if name == "list_background_tasks" else [result["task"]]
                for task in tasks:
                    self.assertEqual(task["knowledge_area_id"], str(self.task.knowledge_area_id))
                    self.assertEqual(task["answer_delivery"], expected)
                    self.assertEqual(task["delivery_state"], "not_established_by_task_status")
                self.assertNotIn("expert_result", result)
        self.provider.execute.assert_not_awaited()

    async def test_computer_use_requests_explicit_desktop_scope_without_application_priming(self):
        self.service.create = AsyncMock(return_value=self.task)
        for capabilities, environment in [(["sandbox_desktop"], "desktop_sandbox"),
                (["sandbox_desktop", "sandbox_network"], "networked_desktop_sandbox")]:
            with self.subTest(environment=environment):
                result = await self.invoke("start_background_task", self.args | {"requested_capabilities": capabilities})
                request = self.service.create.await_args.args[1]
                self.assertEqual(request.execution_environment, environment)
                self.assertFalse(request.publish_artifacts)
                self.assertIn("sandbox_compute", result["capabilities"])
                self.assertIn("sandbox_desktop", result["capabilities"])
                self.assertEqual("sandbox_network" in result["capabilities"], "sandbox_network" in capabilities)
                self.assertEqual(result["external_action_grants"], [])
        description = self.registry.get("start_background_task").input_schema["properties"][
            "requested_capabilities"]["description"].lower()
        self.assertNotIn("email", description)
        self.assertNotIn("gmail", description)
        self.provider.execute.assert_not_awaited()

    async def test_new_provider_call_id_does_not_change_logical_request(self):
        await self.invoke("start_background_task", self.args, call_id="original")
        first = self.repository.create.await_args.args[1].request_id
        await self.invoke("start_background_task", self.args, call_id="reconnected")
        self.assertEqual(first, self.repository.create.await_args.args[1].request_id)
        await self.invoke("start_background_task", self.args | {"request_slot": 2})
        self.assertNotEqual(first, self.repository.create.await_args.args[1].request_id)

    async def test_trusted_native_source_names_avoid_the_second_lookup(self):
        names = CanonicalRequestIdentity.from_record(self.source)
        with patch("app.tasks.voice_tools.ContextRepository.get", AsyncMock()) as lookup:
            with bind_tool_invocation(provider_call_id="native",
                    request_event_id=self.source.id, request_revision=self.source.revision,
                    request_identity=names):
                result = await self.registry.get("start_background_task").handler(self.args)
        self.assertTrue(result["accepted"])
        lookup.assert_not_awaited()
        cached_identity = self.repository.create.await_args.args[1].request_id
        await self.invoke("start_background_task", self.args)
        self.assertEqual(cached_identity, self.repository.create.await_args.args[1].request_id)
        with self.assertRaises(ValueError):
            ToolInvocationContext(request_identity=names)
        with self.assertRaises(ValueError):
            ToolInvocationContext(request_event_id=self.source.id, request_revision=1,
                                  request_identity={"namespace": "not-typed"})

    async def test_same_words_in_a_new_input_are_a_new_request(self):
        await self.invoke("start_background_task", self.args)
        first = self.repository.create.await_args.args[1].request_id
        self.source = self.source.model_copy(update={"id": uuid4(), "source_event_id": "input-2"})
        await self.invoke("start_background_task", self.args)
        self.assertNotEqual(first, self.repository.create.await_args.args[1].request_id)

    async def test_revised_asr_keeps_logical_identity_instead_of_allocating_duplicate(self):
        await self.invoke("start_background_task", self.args)
        first = self.repository.create.await_args.args[1].request_id
        self.source = self.source.model_copy(update={"id": uuid4(), "revision": 3})
        await self.invoke("start_background_task", self.args)
        self.assertEqual(first, self.repository.create.await_args.args[1].request_id)

    async def test_missing_or_unavailable_input_cannot_start_or_cancel_work(self):
        for source in (False, True):
            self.source = self.source.model_copy(update={"availability": "deleted"})
            for name, args in (("start_background_task", self.args),
                ("cancel_background_task", {"task_ref": self.repository.reference(self.task.id), "expected_revision": 1})):
                result = await self.invoke(name, args, source=source)
                self.assertEqual(result["status"], "failed")
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_unavailable_worker_does_not_claim_acceptance(self):
        self.service.providers = {}
        result = await self.invoke("start_background_task", self.args)
        self.assertEqual(result["error"]["code"], "task_provider_not_configured")
        self.assertNotIn("accepted", result)
        self.repository.create.assert_not_awaited()

    async def test_invalid_and_extra_arguments_are_rejected_not_coerced(self):
        for patch_args in ({"owner_id": "foreign"}, {"request_slot": True}, {"request_slot": 0},
                           {"requested_capabilities": ["email"]}, {"grant_ids": [str(uuid4())]}):
            result = await self.invoke("start_background_task", self.args | patch_args)
            self.assertEqual(result["error"]["code"], "invalid_tool_input")
        for ref in ("task_bad", str(self.task.id), "task_abcdef", "the code task"):
            result = await self.invoke("cancel_background_task", {"task_ref": ref, "expected_revision": 1})
            self.assertEqual(result["status"], "failed")
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_revision_conflict_remains_visible(self):
        self.repository.mutate.side_effect = TaskError("task_revision_conflict", "Changed")
        result = await self.invoke("update_background_task", {"task_ref": self.repository.reference(self.task.id),
            "expected_revision": 1, "message": "Full revised objective"})
        self.assertEqual(result["error"]["code"], "task_revision_conflict")

    async def test_read_is_allowed_without_fabricating_user_authority(self):
        result = await self.invoke("list_background_tasks", {}, source=False)
        self.assertEqual(result["tasks"][0]["task_ref"], self.repository.reference(self.task.id))
        self.assertFalse(result["has_more"])
        self.repository.list.assert_awaited_once_with(self.owner, limit=11, offset=0, active_only=True)

    async def test_status_omits_expert_prose_without_changing_stored_answers_at_any_supported_length(self):
        prefix = "mid = (low + high) // 2\n    return mid\n\u03bb\n"
        suffix = "\nUNIQUE-END-OF-EXPERT-ANSWER"
        for length in (7999, 8001, 32000):
            with self.subTest(length=length):
                answer = prefix + "x" * (length - len(prefix) - len(suffix)) + suffix
                evidence = [{"event_id": str(self.source.id), "revision": self.source.revision}]
                result = {"format": "jarvis.hermes.task_result.v1", "status": "succeeded",
                    "answer": answer, "evidence": evidence,
                    "completed_operation_refs": ["operation_fixture"],
                    "uncertainties": ["Indentation is not visible.", "Observed \u03bb, not an ASCII l."],
                    "verified_operations": [{"private_internal_record": "v" * 100000}],
                    "tool_receipts": [{"private_internal_record": "r" * 100000}]}
                original = deepcopy(result)
                self.repository.voice_detail.side_effect = lambda *args: (self.task, {
                    "result": result, "result_unavailable_reason": None,
                    "last_milestone": {"event_type": "succeeded"}})
                self.repository.voice_detail.reset_mock()
                args = {"task_ref": self.repository.reference(self.task.id), "expected_revision": 1}
                payload = await self.invoke("get_background_task", args, source=False)
                self.assertNotIn("expert_result", payload)
                self.assertTrue(payload["expert_output_available"])
                self.assertEqual(payload["answer_delivery"], {
                    "route": "direct_tts", "state": "unavailable", "reason": "attention_coordinator_unavailable"})
                self.assertNotIn(suffix.strip(), json.dumps(payload))
                self.assertNotIn("Indentation is not visible.", json.dumps(payload))
                self.assertNotIn(str(self.source.id), json.dumps(payload))
                self.assertNotIn("private_internal_record", json.dumps(payload))
                self.assertNotIn("result_page", payload)
                self.assertEqual(result, original)
                self.repository.voice_detail.assert_awaited_once_with(self.owner, self.task.id)
        stale = await self.invoke("get_background_task", args | {"expected_revision": 2}, source=False)
        self.assertEqual(stale["error"]["code"], "task_revision_conflict")

    async def test_absent_answer_and_older_minimal_results(self):
        for result in (None, {"answer": "A complete concise expert answer."}):
            with self.subTest(result=result):
                self.repository.voice_detail.side_effect = lambda *args: (self.task, {
                    "result": result, "result_unavailable_reason": None, "last_milestone": None})
                payload = await self.invoke("get_background_task", {
                    "task_ref": self.repository.reference(self.task.id)}, source=False)
                self.assertNotIn("expert_result", payload)
                self.assertFalse(payload["expert_output_available"])
                self.assertEqual(payload["stored_output_kind"], "absent" if result is None else "unverified")
                self.assertFalse(payload["expert_speech_prepared"])
                self.assertEqual(payload["answer_delivery"]["state"], "unavailable")
                if result is not None:
                    self.assertNotIn(result["answer"], json.dumps(payload))

    async def test_malformed_stored_answer_is_not_blame_assigned_to_tool_arguments(self):
        invalid = ("not a result", {}, {"answer": ""}, {"answer": " \n "},
            {"answer": "x" * 32001}, {"answer": "Valid answer", "uncertainties": None},
            {"answer": "Valid answer", "evidence": [{"event_id": "invented", "revision": 1}]},
            {"answer": "Valid answer", "completed_operation_refs": ["op"] * 33})
        for result in invalid:
            with self.subTest(result_type=type(result).__name__):
                self.repository.voice_detail.side_effect = lambda *args: (self.task, {
                    "result": result, "result_unavailable_reason": None, "last_milestone": None})
                payload = await self.invoke("get_background_task", {
                    "task_ref": self.repository.reference(self.task.id)}, source=False)
                self.assertEqual(payload["status"], "failed")
                self.assertEqual(payload["error"]["code"], "task_result_unavailable")
                self.assertNotIn("expert_result", payload)

    async def test_worker_result_schema_keeps_existing_required_fields_and_bound(self):
        schema = HermesTaskResult.model_json_schema()
        self.assertEqual(set(schema["required"]),
            {"status", "answer", "evidence", "completed_operation_refs", "uncertainties", "speech"})
        self.assertEqual(schema["properties"]["answer"]["maxLength"], 32000)
        self.assertFalse(schema["additionalProperties"])
        receipt_help = schema["properties"]["completed_operation_refs"]["description"]
        self.assertIn("task_operation.operation_ref", receipt_help)
        self.assertIn("terminal-only", receipt_help)
        self.assertIn("Never include sandbox command_id", receipt_help)
        voice = self.registry.get("get_background_task").input_schema
        self.assertNotIn("result_offset", voice["properties"])
        result = await self.invoke("get_background_task", {
            "task_ref": self.repository.reference(self.task.id), "result_offset": 0}, source=False)
        self.assertEqual(result["error"]["code"], "invalid_tool_input")

    async def test_content_availability_is_not_execution_or_delivery_state(self):
        for state in (TaskStatus.SUCCEEDED, TaskStatus.WAITING, TaskStatus.PAUSED):
            with self.subTest(state=state):
                self.task.status = state
                self.repository.voice_detail.side_effect = lambda *args: (self.task, {
                    "result": {"format":"jarvis.hermes.task_result.v1", "status":state.value,
                        "answer":"Available expert content.", "evidence":[],
                        "completed_operation_refs":[], "uncertainties":[]}, "result_unavailable_reason": None,
                    "last_milestone": None})
                result = await self.invoke("get_background_task", {
                    "task_ref": self.repository.reference(self.task.id)}, source=False)
                self.assertTrue(result["expert_output_available"])
                self.assertNotIn("expert_result", result)
                self.assertEqual(result["answer_delivery"]["state"], "unavailable")
                self.assertEqual(result["task"]["state"], state.value)
                self.assertEqual(result["task"]["delivery_state"], "not_established_by_task_status")
        self.task.status = TaskStatus.SUCCEEDED
        self.repository.voice_detail.side_effect = lambda *args: (self.task, {
            "result": None, "result_unavailable_reason": "request_evidence_unavailable", "last_milestone": None})
        result = await self.invoke("get_background_task", {
            "task_ref": self.repository.reference(self.task.id)}, source=False)
        self.assertFalse(result["expert_output_available"])
        self.assertEqual(result["task"]["state"], "succeeded")
        self.assertEqual(result["result_unavailable_reason"], "request_evidence_unavailable")

    async def test_runtime_pause_does_not_advertise_an_expert_answer_or_trigger_work(self):
        from app.tasks.hermes import HermesTaskProvider
        outcome = HermesTaskProvider.incomplete("hermes_native_incomplete")
        self.task.status = outcome.status
        self.repository.voice_detail.return_value = (self.task, {
            "result":outcome.payload, "result_unavailable_reason":None, "last_milestone":None})
        original = deepcopy(outcome.payload)
        result = await self.invoke("get_background_task", {
            "task_ref":self.repository.reference(self.task.id)}, source=False)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stored_output_kind"], "runtime")
        self.assertFalse(result["expert_output_available"])
        self.assertFalse(result["expert_speech_prepared"])
        self.assertEqual(result["expert_output_unavailable_reason"], "runtime_outcome_not_expert_answer")
        self.assertNotIn(outcome.payload["answer"], json.dumps(result))
        self.assertEqual(outcome.payload, original)
        self.provider.execute.assert_not_awaited()
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_both_providers_share_generic_conversational_boundaries(self):
        shared = "\n\n".join(CONVERSATION_STYLE_PROMPT_LINES)
        self.assertNotIn("final_answer", shared)
        self.assertNotIn("commentary-phase", shared)
        for enabled in (False, True):
            for prompt in (build_base_prompt(tasks_enabled=enabled), build_gemini_instructions(tasks_enabled=enabled)):
                for paragraph in CONVERSATION_STYLE_PROMPT_LINES:
                    self.assertIn(paragraph, prompt)
        self.assertIn("verbatim", self.registry.get("start_background_task").input_schema[
            "properties"]["requested_outcome"]["description"])

    async def test_audit_projection_keeps_identity_without_duplicate_expert_text(self):
        definition = self.registry.get("start_background_task")
        arguments = definition.audit_arguments(self.args)
        self.assertNotIn("Synthetic task", json.dumps(arguments))
        self.assertNotEqual(arguments["input_digest"], definition.audit_arguments(
            self.args | {"requested_outcome": "A different objective"})["input_digest"])
        original = {"status": "success", "task": {
            "task_ref": self.repository.reference(self.task.id), "revision": 1, "state": "succeeded",
            "title": "Private synthetic project"},
            "expert_result": {"answer": "Private synthetic expert answer"}}
        projected = self.registry.get("get_background_task").audit_result(original)
        self.assertNotIn("Private synthetic", json.dumps(projected))
        self.assertEqual(projected["task"]["task_ref"], original["task"]["task_ref"])
        capture = SimpleNamespace(tool_outcome=unittest.mock.Mock(), request_for_call=AsyncMock(return_value=None))
        executor = JarvisToolExecutor(self.registry, context_capture=capture)
        with patch.object(executor, "_execute", AsyncMock(return_value=original)):
            returned = await executor.execute("get_background_task", {}, provider_call_id="fixture")
        self.assertEqual(returned, original)
        self.assertEqual(capture.tool_outcome.call_args.args[2], projected)

    async def test_selected_voice_delivery_ownership_is_shared_and_feature_gated(self):
        ownership = "Background experts own substantive reasoning; the selected voice route delivers"
        for build in (build_base_prompt, build_gemini_instructions):
            self.assertNotIn(ownership, build())
            enabled = build(tasks_enabled=True)
            self.assertEqual(enabled.count(ownership), 1)
            self.assertIn("Do not re-solve them or substitute your own answer", enabled)
            self.assertIn("Speak a result only when an authorized delivery event supplies it", enabled)
            self.assertIn("route availability alone is not such an event", enabled)
            self.assertIn("do not claim it was delivered or heard", enabled)
            self.assertNotIn("through direct TTS", enabled)
            self.assertNotIn("complete expert-authored answer and its relevant evidence", enabled)

    async def test_selected_route_status_never_delivers_or_exposes_expert_prose(self):
        self.task.status = TaskStatus.SUCCEEDED
        result = {"format": "jarvis.hermes.task_result.v1", "status": "succeeded",
                  "answer": "PRIVATE-EXPERT-RESULT", "evidence": [],
                  "completed_operation_refs": [], "uncertainties": []}
        original = deepcopy(result)
        self.repository.voice_detail.side_effect = lambda *args: (self.task, {
            "result": result, "result_unavailable_reason": None, "last_milestone": None})
        for route in ("realtime", "live", "direct_tts"):
            with self.subTest(route=route):
                status = {"route": route, "state": "available", "reason": None,
                          "hearing": "not_established_by_route_availability"}
                coordinator = SimpleNamespace(voice_status=lambda: status,
                                              offer_notification=AsyncMock())
                self.service.attention_service = coordinator
                payload = await self.invoke("get_background_task", {
                    "task_ref": self.repository.reference(self.task.id)}, source=False)
                self.assertEqual(payload["answer_delivery"], status)
                self.assertEqual(payload["task"]["answer_delivery"], status)
                self.assertTrue(payload["expert_output_available"])
                self.assertEqual(payload["task"]["state"], "succeeded")
                self.assertEqual(payload["task"]["delivery_state"], "not_established_by_task_status")
                self.assertIn("one short sentence", payload["meaning"])
                self.assertIn("Unknown hearing alone is not a task or delivery failure", payload["meaning"])
                self.assertIn("Do not promise imminent playback", payload["meaning"])
                self.assertNotIn("PRIVATE-EXPERT-RESULT", json.dumps(payload))
                self.assertEqual(result, original)
                coordinator.offer_notification.assert_not_awaited()
        self.provider.execute.assert_not_awaited()
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()
        description = self.registry.get("get_background_task").description
        self.assertIn("authorized delivery by the selected voice route", description)
        self.assertIn("this tool cannot deliver them", description)
        self.assertNotIn("direct TTS", description)

    async def test_terse_status_prompt_preserves_execution_and_delivery_boundaries(self):
        for build in (build_base_prompt, build_gemini_instructions):
            with self.subTest(builder=build.__name__):
                prompt = build(tasks_enabled=True)
                self.assertIn("one short sentence about the supported execution state, then yield", prompt)
                self.assertIn("when current task evidence says succeeded: The task has finished", prompt)
                self.assertIn("Status tools intentionally omit expert prose; that omission does not mean the task is unfinished", prompt)
                self.assertIn("Unknown hearing alone is not a task or delivery failure", prompt)
                self.assertIn("Do not hide actual failed, cancelled, paused, or waiting_for_input states", prompt)
                self.assertIn("when they prevent the requested outcome or the user asks about delivery", prompt)
                self.assertIn("must not replace an authorized result or its qualifications", prompt)
                self.assertIn("Do not promise imminent playback from stored output or route availability", prompt)
                self.assertIn("Speak a result only when an authorized delivery event supplies it", prompt)
                self.assertIn("do not claim it was delivered or heard", prompt)
                self.assertNotIn("If a status read has no answer, say it is not ready", prompt)
                self.assertNotIn("Keep routine task-status or continuation acknowledgments", build())

    async def test_terse_status_keeps_all_task_states_and_actual_route_failures(self):
        for state in TaskStatus:
            for route_state in ("available", "unavailable"):
                with self.subTest(task_state=state.value, route_state=route_state):
                    self.task.status = state
                    self.task.reason = "fixture_" + state.value
                    delivery = {"route": "realtime", "state": route_state,
                        "reason": "attention_voice_stopped" if route_state == "unavailable" else None,
                        "hearing": "not_established_by_route_availability"}
                    coordinator = SimpleNamespace(voice_status=lambda: delivery,
                                                  offer_notification=AsyncMock())
                    self.service.attention_service = coordinator
                    self.repository.voice_detail.return_value = (self.task, {
                        "result": None, "result_unavailable_reason": "request_evidence_unavailable",
                        "last_milestone": None})
                    payload = await self.invoke("get_background_task", {
                        "task_ref": self.repository.reference(self.task.id)}, source=False)
                    self.assertEqual(payload["status"], "success")
                    self.assertEqual(payload["task"]["state"], state.value)
                    self.assertEqual(payload["task"]["reason"], self.task.reason)
                    self.assertEqual(payload["task"]["delivery_state"], "not_established_by_task_status")
                    self.assertEqual(payload["task"]["answer_delivery"], delivery)
                    self.assertEqual(payload["answer_delivery"], delivery)
                    self.assertEqual(payload["result_unavailable_reason"], "request_evidence_unavailable")
                    self.assertFalse(payload["expert_output_available"])
                    self.assertNotIn("expert_result", payload)
                    self.assertIn("when they prevent the requested outcome or the user asks about delivery", payload["meaning"])
                    coordinator.offer_notification.assert_not_awaited()
        self.provider.execute.assert_not_awaited()
        self.repository.create.assert_not_awaited()
        self.repository.mutate.assert_not_awaited()

    async def test_feature_gated_prompt_keeps_disabled_paths_unchanged(self):
        self.assertNotIn("## Background experts", build_base_prompt())
        self.assertNotIn("## Background experts", build_gemini_instructions())
        for instructions in (build_base_prompt(tasks_enabled=True),
                             build_gemini_instructions(tasks_enabled=True)):
            for line in DELEGATION_PROMPT_LINES:
                self.assertIn(line, instructions)
            self.assertIn("do not delegate or announce thinking on every turn", instructions)
            self.assertIn("does not cancel accepted work", instructions)
