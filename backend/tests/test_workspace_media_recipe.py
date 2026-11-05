"""Pure recipe tests. No renderer, database, source authorization or provider."""
import copy
import json
import unittest
from uuid import UUID

from pydantic import ValidationError

from app.workspace.media_recipe import (
    MediaRecipeError, MediaRecipePolicy, PreviousRecipe, RetainedRef, SourceInspection,
    TemplateInventoryEntry, canonical_recipe, parse_recipe, recipe_digest, validate_recipe,
)


def reference(number=1, digest="a"):
    return {"file_id": str(UUID(int=number)), "version_id": "source_" + str(number).zfill(32),
            "content_sha256": digest * 64}


def recipe():
    return {"schema_version": 1, "edit_id": str(UUID(int=100)), "previous_recipe": None,
        "template": {"id": "jarvis.clip.v1", "version": "fixture-1", "bundle_sha256": "b" * 64},
        "sources": [{"source_id": "original", "file": reference()}],
        "segments": [{"segment_id": "first", "source_id": "original", "in_ms": 2000, "out_ms": 7000},
                     {"segment_id": "second", "source_id": "original", "in_ms": 12000, "out_ms": 17000}],
        "captions": [{"caption_id": "title", "start_frame": 24, "end_frame": 96, "text": "Two moments"}],
        "output_profile": "portrait_720p24_v1", "fit": "contain", "audio": "preserve"}


class MediaRecipeTests(unittest.TestCase):
    def setUp(self):
        self.value = recipe()
        self.inspection = SourceInspection(file=reference(), duration_ms=30000, size_bytes=12 * 1024 * 1024,
            video_decodable=True, audio_streams=1, audio_decodable=True)
        self.inventory = TemplateInventoryEntry(template=self.value["template"],
            output_profile="portrait_720p24_v1", ready=True)
        # Test inputs only; this does not configure a runtime grant/limit.
        self.policy = MediaRecipePolicy(max_source_bytes=64 * 1024 * 1024,
            max_source_duration_ms=60000, max_output_frames=720)

    def validate(self, value=None, **overrides):
        return validate_recipe(self.value if value is None else value, **({
            "inspections": [self.inspection], "inventory": [self.inventory], "policy": self.policy} | overrides))

    def assert_code(self, code, value=None, **overrides):
        with self.assertRaises(MediaRecipeError) as error:
            self.validate(value, **overrides)
        self.assertEqual(error.exception.code, code)

    def test_two_cuts_make_240_frames_and_data_only_props(self):
        result = self.validate()
        self.assertEqual(result.duration_in_frames, 240)
        props = result.template_props()
        self.assertEqual(props["segments"], [
            {"segmentId": "first", "sourceId": "original", "fromFrame": 48, "toFrame": 168,
             "outputFromFrame": 0, "durationInFrames": 120},
            {"segmentId": "second", "sourceId": "original", "fromFrame": 288, "toFrame": 408,
             "outputFromFrame": 120, "durationInFrames": 120}])
        self.assertEqual((props["width"], props["height"], props["fps"]), (720, 1280, 24))
        self.assertNotIn("file", json.dumps(props))
        self.assertNotIn("path", json.dumps(props))

    def test_revision_preserves_original_and_requires_exact_predecessor(self):
        original = copy.deepcopy(self.value)
        first = self.validate()
        revised = copy.deepcopy(self.value)
        revised["previous_recipe"] = reference(2, "c")
        revised["segments"][1]["out_ms"] = 19000
        previous = PreviousRecipe(file=reference(2, "c"), edit_id=revised["edit_id"])
        second = self.validate(revised, previous=previous)
        self.assertEqual(second.duration_in_frames, 288)
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(self.value, original)
        self.assertEqual(first.duration_in_frames, 240)
        self.assertEqual(first.recipe.sources, second.recipe.sources)
        self.assert_code("media_predecessor_mismatch", revised)
        self.assert_code("media_predecessor_mismatch", revised,
            previous=PreviousRecipe(file=reference(3, "c"), edit_id=revised["edit_id"]))
        self.assert_code("media_predecessor_mismatch", revised,
            previous=PreviousRecipe(file=reference(2, "c"), edit_id=UUID(int=101)))
        self.assert_code("media_predecessor_mismatch", previous=previous)

    def test_source_exact_identity_hash_and_inspection_count(self):
        for key, replacement in (("file_id", str(UUID(int=3))), ("version_id", "source_other"),
                                  ("content_sha256", "d" * 64)):
            value = copy.deepcopy(self.value)
            value["sources"][0]["file"][key] = replacement
            self.assert_code("media_source_reference_mismatch", value)
        self.assert_code("media_source_reference_mismatch", inspections=[])
        self.assert_code("media_source_reference_mismatch", inspections=[self.inspection, self.inspection])

    def test_template_pin_readiness_and_ambiguous_inventory(self):
        for key, replacement in (("version", "uninstalled"), ("bundle_sha256", "c" * 64)):
            value = copy.deepcopy(self.value); value["template"][key] = replacement
            self.assert_code("media_template_unavailable", value)
        self.assert_code("media_template_unavailable", inventory=[])
        unavailable = self.inventory.model_dump() | {"ready": False}
        self.assert_code("media_template_unavailable", inventory=[unavailable])
        self.assert_code("media_template_ambiguous", inventory=[self.inventory, self.inventory])
        value = copy.deepcopy(self.value); value["template"]["id"] = "arbitrary.js"
        self.assert_code("media_recipe_invalid", value)

    def test_count_limits_and_duplicate_or_missing_references(self):
        cases = []
        value = recipe(); value["sources"].append(copy.deepcopy(value["sources"][0])); cases.append(value)
        value = recipe(); value["sources"] = []; cases.append(value)
        value = recipe(); value["segments"] = []; cases.append(value)
        value = recipe(); value["segments"] = [value["segments"][0] | {"segment_id": f"s{i}"} for i in range(9)]; cases.append(value)
        value = recipe(); value["segments"][1]["segment_id"] = "first"; cases.append(value)
        value = recipe(); value["segments"][1]["source_id"] = "not-declared"; cases.append(value)
        value = recipe(); value["captions"].append(copy.deepcopy(value["captions"][0])); cases.append(value)
        value = recipe(); value["captions"] = [value["captions"][0] | {"caption_id": f"c{i}"} for i in range(17)]; cases.append(value)
        value = recipe(); value["segments"][0]["segment_id"] = "bad id"; cases.append(value)
        for value in cases:
            with self.subTest(value=value):
                self.assert_code("media_recipe_invalid", value)

    def test_explicit_repeat_and_segment_order_remain_meaningful(self):
        value = recipe(); value["segments"][1] = value["segments"][0] | {"segment_id": "repeat"}
        self.assertEqual(self.validate(value).duration_in_frames, 240)
        reversed_value = recipe(); reversed_value["segments"].reverse()
        self.assertNotEqual(recipe_digest(reversed_value), recipe_digest(self.value))
        self.assertEqual(self.validate(reversed_value).template_props()["segments"][0]["fromFrame"], 288)

    def test_half_open_source_bounds_and_no_silent_frame_rounding(self):
        for start, end in ((2000, 2000), (7000, 2000), (2000, 30125)):
            value = recipe(); value["segments"][0].update(in_ms=start, out_ms=end)
            self.assert_code("media_interval_invalid", value)
        for endpoint in (1, 41, 42, 124, 126):
            value = recipe(); value["segments"][0]["in_ms"] = endpoint
            self.assert_code("media_interval_off_grid", value)
        value = recipe(); value["segments"][0].update(in_ms=125, out_ms=250)
        self.assertEqual(self.validate(value).duration_in_frames, 123)
        value = recipe(); value["segments"][1]["out_ms"] = 30000
        self.assertEqual(self.validate(value).duration_in_frames, 552)

    def test_captions_are_half_open_output_frames_and_cannot_overlap(self):
        value = recipe(); value["captions"] = [
            {"caption_id": "later", "start_frame": 120, "end_frame": 240, "text": "B"},
            {"caption_id": "earlier", "start_frame": 0, "end_frame": 120, "text": "A"}]
        self.assertEqual(len(self.validate(value).template_props()["captions"]), 2)
        for start, end in ((24, 24), (96, 24), (24, 241)):
            bad = recipe(); bad["captions"][0].update(start_frame=start, end_frame=end)
            self.assert_code("media_caption_interval_invalid", bad)
        value["captions"][1]["end_frame"] = 121
        self.assert_code("media_caption_overlap", value)

    def test_audio_is_never_invented_and_mute_is_explicit(self):
        for changes in ({"audio_streams": 0}, {"audio_decodable": False}):
            inspection = self.inspection.model_dump() | changes
            self.assert_code("media_audio_unavailable", inspections=[inspection])
            value = recipe(); value["audio"] = "mute"
            self.assertEqual(self.validate(value, inspections=[inspection]).template_props()["audio"], "mute")
        self.assert_code("media_video_unavailable", inspections=[self.inspection.model_dump() | {"video_decodable": False}])

    def test_html_looking_caption_stays_literal_and_original_immutable(self):
        text = '<img src="https://example.invalid/evil" onerror="run()">\n${shell} <script>code</script>'
        value = recipe(); value["captions"][0]["text"] = text
        before = copy.deepcopy(value)
        result = self.validate(value)
        props = result.template_props()
        self.assertEqual(props["captions"][0]["text"], text)
        props["captions"][0]["text"] = "local change"
        self.assertEqual(result.template_props()["captions"][0]["text"], text)
        self.assertEqual(value, before)
        with self.assertRaises(ValidationError):
            result.recipe.segments[0].in_ms = 0

    def test_executable_inputs_and_mutable_generation_fields_rejected(self):
        cases = []
        for key in ("filtergraph", "output_path", "source_url", "jsx", "owner_id"):
            value = recipe(); value[key] = "not-allowed"; cases.append(value)
        value = recipe(); value["template"]["url"] = "https://example.invalid/template.js"; cases.append(value)
        value = recipe(); value["sources"][0]["file"]["source_generation"] = 2; cases.append(value)
        value = recipe(); value["captions"][0]["html"] = "<b>not-allowed</b>"; cases.append(value)
        value = recipe(); value["fit"] = "cover"; cases.append(value)
        for value in cases:
            self.assert_code("media_recipe_invalid", value)

    def test_canonical_retry_digest_stable_without_trimming_or_reordering(self):
        canonical = canonical_recipe(self.value)
        shuffled = {key: self.value[key] for key in reversed(self.value)}
        encoded = json.dumps(shuffled, indent=4)
        self.assertEqual(canonical_recipe(encoded), canonical)
        self.assertEqual(recipe_digest(encoded), self.validate().digest)
        value = recipe(); value["captions"][0]["text"] += " "
        self.assertNotEqual(recipe_digest(value), self.validate().digest)
        value = recipe(); value["sources"][0]["file"]["version_id"] += " "
        self.assert_code("media_source_reference_mismatch", value)

    def test_strict_fields_unicode_and_duplicate_json_keys(self):
        for field, bad in (("schema_version", True), ("schema_version", 1.0)):
            value = recipe(); value[field] = bad
            self.assert_code("media_recipe_invalid", value)
        for bad in (True, 2000.0, "2000", -1, float("nan")):
            value = recipe(); value["segments"][0]["in_ms"] = bad
            self.assert_code("media_recipe_invalid", value)
        for text in ("x" * 201, "bad\x00text", "bad\ud800text"):
            value = recipe(); value["captions"][0]["text"] = text
            self.assert_code("media_recipe_invalid", value)
        with self.assertRaises(MediaRecipeError):
            parse_recipe('{"schema_version":1,"schema_version":1}')
        value = recipe(); value["captions"][0]["text"] = "你好 🎵"
        self.assertIn("你好 🎵".encode(), canonical_recipe(value))

    def test_explicit_policy_byte_duration_output_and_recipe_limits(self):
        for key, limit, code in (("max_source_bytes", 1, "media_source_limit"),
                                 ("max_source_duration_ms", 1000, "media_source_limit"),
                                 ("max_output_frames", 239, "media_output_limit"),
                                 ("max_recipe_bytes", 100, "media_recipe_too_large")):
            self.assert_code(code, policy=self.policy.model_dump() | {key: limit})
        size = len(canonical_recipe(self.value))
        self.assertEqual(self.validate(policy=self.policy.model_dump() | {"max_recipe_bytes": size}).duration_in_frames, 240)
        self.assert_code("media_recipe_too_large", policy=self.policy.model_dump() | {"max_recipe_bytes": size - 1})
        self.assert_code("media_validation_context_invalid", policy=self.policy.model_dump() | {"max_recipe_bytes": 65537})
        self.assert_code("media_validation_context_invalid", inspections=[self.inspection.model_dump() | {"size_bytes": True}])


if __name__ == "__main__":
    unittest.main()
