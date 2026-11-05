"""Opt-in native Swift decoding/pagination checks for the source-backed phone contract."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from app.knowledge.browser import _item, _evidence


SWIFT = r"""
import Foundation
@main struct SourceMemoryContract {
    static func main() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
        let pages = try JSONDecoder().decode([JarvisMemoryDetail].self, from: data)
        precondition(pages[0].memory.confidence == nil)
        precondition(pages[0].memory.importance == nil)
        precondition(pages[0].memory.sourceEvidence?.sourceTime == nil)
        precondition(pages[0].evidence[0].evidenceTime == nil)
        let complete = try pages[0].appendingOriginalPage(pages[1])
        precondition(complete.memory.content == "{\"text\":\"synthetic original\"}")
        precondition(complete.contentPage?.complete == true)
        precondition(complete.contentPage?.nextOffset == nil)
        precondition(complete.memory.contentTruncated == false)
        for index in 2...4 {
            var rejected = false
            do { _ = try pages[0].appendingOriginalPage(pages[index]) }
            catch { rejected = true }
            precondition(rejected)
        }
        precondition(pages[5].memory.sourceEvidence == nil)
        precondition(pages[5].memory.confidence == 0.8)
        precondition(pages[5].contentPage == nil)
        print("Source memory Swift contract passed")
    }
}
"""


@unittest.skipUnless(os.getenv("JARVIS_IOS_CONTRACT_TESTS") == "1" and shutil.which("swiftc"),
                     "Opt-in native Swift compiler contract")
class SourceMemoryPhoneContractTests(unittest.TestCase):
    def test_source_and_legacy_payloads_and_guarded_page_append(self):
        source = {
            "source_id": "00000000-0000-0000-0000-000000000001",
            "revision": 1, "generation": 1, "content_digest": "a" * 64,
            "source_type": "user_transcript", "trust": "user_input",
            "source_time": None, "time_basis": "Original time is unknown",
            "recorded_at": "2026-09-15T12:00:00Z", "is_current": True, "ingestion_status": "pending",
        }
        text = '{"text":"synthetic original"}'
        def page(offset, end):
            body = text[offset:end]
            return {"memory": _item(source, body, truncated=end < len(text)),
                "evidence": [_evidence(source, body)], "links": [], "related_memories": [],
                "uncertainty": "Original source, not proof",
                "content_page": {"encoding": "canonical-json-ascii-v1", "offset": offset,
                    "total_characters": len(text), "next_offset": end if end < len(text) else None,
                    "complete": offset == 0 and end == len(text)}}
        first, second = page(0, 10), page(10, len(text))
        changed = json.loads(json.dumps(second))
        changed["memory"]["source_evidence"]["content_digest"] = "b" * 64
        gap = json.loads(json.dumps(second))
        gap["content_page"]["offset"] = 11
        incomplete = json.loads(json.dumps(second))
        incomplete["memory"]["content"] = incomplete["memory"]["content"][:-1]
        legacy = json.loads(json.dumps(first))
        legacy["memory"].pop("source_evidence")
        legacy["memory"].pop("content_truncated")
        legacy["memory"]["confidence"] = 0.8
        legacy.pop("content_page")
        with tempfile.TemporaryDirectory(prefix="jarvis-source-memory-swift-") as folder:
            root = Path(folder)
            (root / "Contract.swift").write_text(SWIFT)
            (root / "payloads.json").write_text(json.dumps([first, second, changed, gap, incomplete, legacy]))
            model = Path(__file__).resolve().parents[2] / "ios/Jarvis/Jarvis/Models/JarvisMemoryModels.swift"
            compile_result = subprocess.run(["swiftc", str(model), str(root / "Contract.swift"), "-o", str(root / "check")],
                text=True, capture_output=True, timeout=120)
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            result = subprocess.run([str(root / "check"), str(root / "payloads.json")],
                text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Source memory Swift contract passed", result.stdout)
