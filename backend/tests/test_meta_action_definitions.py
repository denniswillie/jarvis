"""Compact model-facing choices preserve the existing AppCommand contract."""
from copy import deepcopy
import unittest

from app.app_controls import AppCommand
from app.model_providers.structured_router import control_argument_schema, complete_control_arguments


ID = "22222222-2222-4222-8222-222222222222"
URI = "spotify:track:" + "a" * 22


class ActionDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.parameters = AppCommand.model_json_schema()

    def test_every_action_shape_uses_existing_defaults_and_validator(self):
        examples = [
            {"action":"show_link", "url":"https://example.com/video", "description":"Edited video"},
            {"action":"recording.start", "include_jarvis_audio":True},
            {"action":"recording.start", "include_jarvis_audio":False},
            {"action":"recording.start", "include_jarvis_audio":False, "recording_mode":"voiceover"},
            {"action":"recording.stop", "recording_id":ID},
            {"action":"sandbox.start", "allocation_id":ID, "generation":"a" * 64, "resume":False},
            {"action":"sandbox.start", "allocation_id":ID, "generation":"a" * 64, "resume":True},
            {"action":"sandbox.stop", "view_id":ID},
            {"action":"media.start", "song_title":"Lost in Japan", "artist":"Shawn Mendes"},
            {"action":"media.start", "song_title":"null", "artist":None},
            {"action":"media.start", "spotify_uri":URI},
            {"action":"media.start"},
            {"action":"media.stop", "media_id":ID},
            {"action":"media.volume", "media_id":ID, "volume":0.25},
        ]
        for example in examples:
            with self.subTest(example=example):
                before = deepcopy(example)
                self.assertEqual(complete_control_arguments(example, self.parameters),
                                 AppCommand.model_validate(example).model_dump(mode="json"))
                self.assertEqual(example, before)

    def test_invalid_values_and_cross_action_fields_remain_rejected(self):
        examples = [
            {"action":"recording.start", "include_jarvis_audio":"null"},
            {"action":"recording.start", "include_jarvis_audio":True, "recording_mode":"voiceover"},
            {"action":"recording.start", "include_jarvis_audio":True, "media_id":None},
            {"action":"recording.stop"},
            {"action":"recording.stop", "recording_id":"null"},
            {"action":"media.start", "song_title":"Lost in Japan", "artist":None, "spotify_uri":URI},
            {"action":"media.start", "artist":"Shawn Mendes"},
            {"action":"media.start", "spotify_uri":"invented"},
            {"action":"media.stop", "media_id":None},
            {"action":"media.volume", "media_id":ID, "volume":"null"},
            {"action":"media.volume", "media_id":ID, "volume":2.0},
            {"action":"sandbox.start", "allocation_id":ID, "generation":"bad", "resume":False},
            {"action":"sandbox.stop", "view_id":ID, "resume":True},
        ]
        for example in examples:
            with self.subTest(example=example), self.assertRaises(ValueError):
                complete_control_arguments(example, self.parameters)

    def test_schema_retains_constraints_and_does_not_mutate_backend_definition(self):
        before = deepcopy(self.parameters)
        choices = control_argument_schema(self.parameters)["anyOf"]
        self.assertEqual(self.parameters, before)
        self.assertEqual(len(choices), 11)
        self.assertEqual({c["properties"]["action"]["enum"][0] for c in choices},
                         set(self.parameters["properties"]["action"]["enum"]))
        for choice in choices:
            self.assertFalse(choice["additionalProperties"])
            self.assertEqual(set(choice["required"]), set(choice["properties"]))
            self.assertLessEqual(len(choice["properties"]), 4)
        volume = next(c["properties"]["volume"] for c in choices if "volume" in c["properties"])
        self.assertEqual((volume["minimum"], volume["maximum"]), (0, 1))
        self.assertNotIn("anyOf", volume)
        record_stop = next(c for c in choices if c["properties"]["action"]["enum"] == ["recording.stop"])
        self.assertEqual(record_stop["properties"]["recording_id"]["format"], "uuid")
        self.assertNotIn("anyOf", record_stop["properties"]["recording_id"])
