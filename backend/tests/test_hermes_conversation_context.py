import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.tasks.hermes_child import run_conversation
from app.tasks.hermes_context import native_conversation_context


def message(text="Keep the original audio, and do not crop.", **updates):
    return {
        "id": "source-one", "revision": 2, "origin_sequence": 4, "sequence": 7,
        "kind": "user_transcript", "trust": "user_input", "finality": "revised",
        "availability": "available", "is_current": True,
        "content_digest": "original-content-digest", "content": {"text": text},
        **updates,
    }


class HermesConversationContextTests(unittest.TestCase):
    def test_actual_dialogue_is_forwarded_once_without_changing_stored_manifest(self):
        user = message()
        assistant = message("I can work on that.", id="source-two", kind="assistant_output",
            trust="assistant_output", finality="final", heard_state="unknown")
        manifest = {"format": "jarvis.task_context.v1", "messages": [user, assistant],
                    "task": {"requested_outcome": "Create a portrait edit."},
                    "history": {"omitted_count": 3}, "shared_context": {"focus": "video"}}
        before = copy.deepcopy(manifest)
        native = native_conversation_context(manifest)
        self.assertEqual(manifest, before)
        self.assertEqual(native["conversation_history"], [
            {"role": "user", "content": user["content"]["text"]},
            {"role": "assistant", "content": assistant["content"]["text"]},
        ])
        self.assertEqual(json.dumps(native).count(user["content"]["text"]), 1)
        for index, original in enumerate(before["messages"]):
            projected = native["manifest"]["messages"][index]
            self.assertEqual(projected["native_history_index"], index)
            for field in ("id", "revision", "sequence", "origin_sequence", "content_digest"):
                self.assertEqual(projected[field], original[field])
            self.assertNotIn("text", projected["content"])
        self.assertEqual(native["manifest"]["messages"][1]["heard_state"], "unknown")
        self.assertEqual(native["manifest"]["history"], {"omitted_count": 3})
        self.assertEqual(native["manifest"]["shared_context"], {"focus": "video"})

    def test_derived_command_is_not_promoted_to_user_dialogue(self):
        command = message("Crop the image.", kind="command_request", trust="derived")
        native = native_conversation_context({"messages": [command]})
        self.assertEqual(native["conversation_history"], [])
        self.assertEqual(native["manifest"]["messages"], [command])

    def test_untrusted_nonfinal_missing_and_withdrawn_text_stays_evidence(self):
        for change in (
            {"trust": "external_content"}, {"kind": "observation", "trust": "sensor"},
            {"finality": "provisional"}, {"finality": "unavailable"},
            {"availability": "deleted"}, {"is_current": False},
            {"content": None}, {"content": {"transcript_pending": True}},
            {"content": {"text": "  "}}, {"content": {"text": "bad\x00text"}},
        ):
            with self.subTest(change=change):
                source = message(**change)
                native = native_conversation_context({"messages": [source]})
                self.assertEqual(native["conversation_history"], [])
                self.assertEqual(native["manifest"]["messages"], [source])

    def test_provisional_assistant_output_is_not_claimed_as_completed_dialogue(self):
        source = message("The file is...", kind="assistant_output", trust="assistant_output",
                         finality="provisional", streaming_snapshot="Partial output")
        native = native_conversation_context({"messages": [source]})
        self.assertEqual(native["conversation_history"], [])
        self.assertEqual(native["manifest"]["messages"], [source])

    def test_other_content_fields_and_order_survive_projection(self):
        first = message("First", content={"text": "First", "provider_item_id": "item-1"})
        gap = message(kind="context_gap", trust="derived", content={"reason": "missing transcript"})
        last = message("Later correction", id="source-three", origin_sequence=9, sequence=9)
        native = native_conversation_context({"messages": [first, gap, last]})
        self.assertEqual([item["content"] for item in native["conversation_history"]], ["First", "Later correction"])
        self.assertEqual(native["manifest"]["messages"][0]["content"], {"provider_item_id": "item-1"})
        self.assertEqual(native["manifest"]["messages"][1], gap)
        self.assertEqual(native["manifest"]["messages"][2]["native_history_index"], 1)

    def test_plain_original_request_is_supported_without_synthesizing_missing_text(self):
        source = message("Original written request", kind="user_request", finality="final")
        native = native_conversation_context({"messages": [source]})
        self.assertEqual(native["conversation_history"], [{"role": "user", "content": "Original written request"}])
        self.assertEqual(native_conversation_context({})["conversation_history"], [])

    def test_child_forwards_exact_history_to_native_api(self):
        native = native_conversation_context({"messages": [message()]})
        config = {**native, "system_prompt": "Existing task authority.", "task_id": "task-one"}
        agent = SimpleNamespace(run_conversation=Mock(return_value={"completed": True}))
        self.assertEqual(run_conversation(agent, config), {"completed": True})
        agent.run_conversation.assert_called_once_with(
            json.dumps(native["manifest"]), system_message=config["system_prompt"],
            conversation_history=native["conversation_history"], task_id="task-one")

    def test_child_without_history_keeps_existing_empty_history_behavior(self):
        agent = SimpleNamespace(run_conversation=Mock())
        run_conversation(agent, {"manifest": {}, "system_prompt": "Existing", "task_id": "task-one"})
        self.assertIsNone(agent.run_conversation.call_args.kwargs["conversation_history"])
