"""Negative controls for the matched native evaluator; no model or DB calls."""
import ast
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageFont

from evals.glasses_reliability.quality_cases import BY_NAME, CASES, definition_manifest, render_case
from evals.glasses_reliability.quality_probe import answer_observation_deadline, image_digests, load_assets, native_event_summary, observe_visual_write, select_matrix, summarize
from evals.gemini_live.transport_probe import write_wav


class QualityCaseTests(unittest.TestCase):
    def test_terminal_observation_respects_assumed_pcm_playout_but_is_bounded(self):
        self.assertEqual(answer_observation_deadline(100), 150)
        self.assertEqual(answer_observation_deadline(100, first_audio_at=101, pcm_bytes=55 * 48000), 166)
        self.assertEqual(answer_observation_deadline(100, first_audio_at=101, pcm_bytes=400 * 48000), 220)
        self.assertEqual(answer_observation_deadline(100, pcm_bytes=55 * 48000), 150)
        with self.assertRaises(ValueError):
            answer_observation_deadline(100, pcm_bytes=-1)

    def test_all_cases_have_predeclared_rubrics_and_explicit_provenance(self):
        self.assertEqual(len(CASES), len(BY_NAME))
        for case in CASES:
            with self.subTest(case=case.name):
                self.assertTrue(case.expected)
                self.assertTrue(case.forbidden)
                self.assertIn("synthetic", case.stratum)
                self.assertNotIn("original_frame", case.stratum)

    def test_expected_typing_traps_really_are_in_the_input(self):
        for name in ("wrong_indentation", "duplicate_token", "partial_boundary_update"):
            with self.subTest(case=name), self.assertRaises(SyntaxError):
                ast.parse(BY_NAME[name].code)
        tree = ast.parse(BY_NAME["partial_ret"].code)
        self.assertTrue(any(isinstance(node, ast.Name) and node.id == "ret" for node in ast.walk(tree)))
        self.assertIn("mid = low + high // 2", BY_NAME["midpoint_precedence"].code)
        self.assertNotIn("low = mid + 1", BY_NAME["unfinished_function"].code)

    def test_two_providers_receive_each_case_three_times_with_alternating_order(self):
        names = list(BY_NAME)
        rows = select_matrix(names, 3)
        self.assertEqual(len(rows), len(names) * 6)
        for name in names:
            for provider in ("openai", "gemini"):
                self.assertEqual(sum(c.name == name and p == provider for c, _, p in rows), 3)
        self.assertEqual(rows[0][2], "openai")
        self.assertEqual(rows[len(names) * 2][2], "gemini")

    def test_invalid_or_duplicate_cases_fail_closed(self):
        for names, repeats in ((["absent"], 3), (["partial_ret"] * 2, 3), (["partial_ret"], 0), ([], 3)):
            with self.assertRaises(ValueError):
                select_matrix(names, repeats)
        with self.assertRaises(ValueError):
            select_matrix(["partial_ret"], 3, providers=("openai", "openai"))

    def test_missing_results_cannot_pass_an_empty_all_check(self):
        result = summarize([], expected_runs=6)
        self.assertFalse(result["matrix_complete"])
        self.assertFalse(result["infrastructure_passed"])
        self.assertFalse(summarize([], expected_runs=0)["infrastructure_passed"])

    def test_transport_pass_never_becomes_quality_acceptance(self):
        result = summarize([{"infrastructure_passed": True}], expected_runs=1)
        self.assertTrue(result["infrastructure_passed"])
        self.assertEqual(result["quality_acceptance"], "pending_independent_answer_and_audio_review")

    def test_missing_or_stale_view_requires_no_native_image(self):
        self.assertFalse(BY_NAME["vision_stale"].expects_image)
        self.assertFalse(BY_NAME["vision_unavailable"].expects_image)
        self.assertTrue(BY_NAME["vision_cropped"].expects_image)

    def test_rendered_inputs_fit_and_cropped_view_really_omits_lower_code(self):
        font = ImageFont.load_default(size=20)
        for case in CASES:
            jpeg, size = render_case(case, font=font)
            self.assertEqual(Image.open(BytesIO(jpeg)).size, size)
            self.assertLess(len(jpeg), 512000)
            self.assertEqual(size[1], 170 if case.image_condition == "cropped" else 640)

    def test_case_manifest_excludes_audible_and_original_image_claims(self):
        manifest = definition_manifest()
        self.assertEqual(manifest["scoring"]["minimum_native_repeats_per_provider"], 3)
        self.assertIn("not_original", manifest["cases"][0]["stratum"])

    def test_equivalent_operand_order_is_not_a_wrong_comparison(self):
        for case in CASES:
            if case.family == "continuity":
                expected = " ".join(case.expected)
                self.assertIn("elif target > arr[mid]:", expected)
                self.assertIn("elif arr[mid] < target:", expected)

    def test_manifest_round_trip_and_actual_asset_integrity(self):
        case = BY_NAME["partial_ret"]
        jpeg, size = render_case(case, font=ImageFont.load_default(size=20))
        manifest = definition_manifest()
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        manifest["media"] = {case.name: {"image_sha256": sha256(jpeg).hexdigest(), "size": size}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest))
            image = root / (case.name + ".jpg")
            image.write_bytes(jpeg)
            write_wav(root / (case.name + "-1.wav"), b"\x00\x00" * 2400)
            write_wav(root / "bridge.wav", b"\x00\x00" * 2400)
            assets, hashes = load_assets(root, [case.name])
            self.assertEqual(assets[case.name]["jpeg"], jpeg)
            self.assertEqual(hashes[case.name]["image_sha256"], sha256(jpeg).hexdigest())
            image.write_bytes(b"changed after preparation")
            with self.assertRaisesRegex(ValueError, "Image changed"):
                load_assets(root, [case.name])


class ImageBoundaryTests(unittest.TestCase):
    def test_native_boundary_summary_does_not_include_content_or_invent_completion(self):
        summary = native_event_summary({
            "serverContent": {"generationComplete": True, "outputTranscription": {"text": "private utterance"},
                              "modelTurn": {"parts": [{"inlineData": {"data": "secret bytes"}}]}},
            "toolCall": {"functionCalls": [{"args": {"private": "content"}}]},
        })
        self.assertEqual(summary["content_flags"], {"generationComplete": True})
        self.assertEqual(summary["boundaries"], ["toolCall"])
        self.assertNotIn("private", json.dumps(summary))
        self.assertNotIn("secret", json.dumps(summary))

    def test_both_native_image_shapes_have_the_actual_byte_digest(self):
        encoded = base64.b64encode(b"fixture image").decode()
        digest = sha256(b"fixture image").hexdigest()
        payloads = [
            {"realtimeInput": {"video": {"mimeType": "image/jpeg", "data": encoded}}},
            {"item": {"content": [{"type": "input_image", "image_url": "data:image/jpeg;base64," + encoded}]}},
            {"toolResponse": {"functionResponses": [{"parts": [{"inlineData": {"mimeType": "image/jpeg", "data": encoded}}]}]}},
        ]
        for payload in payloads:
            self.assertEqual(image_digests(payload), [digest])

    def test_audio_and_remote_urls_do_not_fabricate_image_writes(self):
        self.assertEqual(image_digests({"mimeType": "audio/pcm", "data": "Zm9v"}), [])
        self.assertEqual(image_digests({"image_url": "https://example.invalid/image.jpg"}), [])

    def test_invalid_image_encoding_does_not_get_a_plausible_digest(self):
        with self.assertRaises(ValueError):
            image_digests({"mimeType": "image/jpeg", "data": "not-base64"})


class VisualWriterObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_priority_visual_writer_is_observed_without_changing_its_payload(self):
        payload = {"image_url": "data:image/jpeg;base64," + base64.b64encode(b"image").decode()}
        writes = []

        async def send(factory, *, timeout_seconds):
            self.assertEqual(timeout_seconds, 0.5)
            self.assertIs(factory(), payload)
            return True

        self.assertTrue(await observe_visual_write(send, lambda: payload, timeout_seconds=0.5, written_images=writes))
        self.assertEqual([row["sha256"] for row in writes], [sha256(b"image").hexdigest()])

    async def test_declined_or_ambiguous_write_is_not_recorded_as_success(self):
        payload = {"image_url": "data:image/jpeg;base64," + base64.b64encode(b"image").decode()}
        writes = []

        async def declined(factory, **kwargs):
            factory()
            return False

        async def uncertain(factory, **kwargs):
            factory()
            raise TimeoutError()

        self.assertFalse(await observe_visual_write(declined, lambda: payload, timeout_seconds=0.5, written_images=writes))
        with self.assertRaises(TimeoutError):
            await observe_visual_write(uncertain, lambda: payload, timeout_seconds=0.5, written_images=writes)
        self.assertEqual(writes, [])
