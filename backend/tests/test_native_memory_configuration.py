import tempfile
from pathlib import Path
import unittest

from app.shared_memory.configuration import configured_memory


class NativeMemoryConfigurationTests(unittest.IsolatedAsyncioTestCase):
    def test_disabled_composition_needs_no_server_or_secret(self):
        self.assertIsNone(configured_memory(owner_id="owner", environ={}))

    def test_enabled_composition_requires_explicit_bank_and_key_path(self):
        with self.assertRaises(ValueError):
            configured_memory(owner_id="owner", environ={"JARVIS_NATIVE_MEMORY_ENABLED": "true"})

    async def test_enabled_composition_is_lazy_and_owner_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key"
            path.write_text("test-only-credential")
            memory = configured_memory(owner_id="owner", environ={
                "JARVIS_NATIVE_MEMORY_ENABLED": "true", "JARVIS_HINDSIGHT_BANK_GENERATION": "fixture-v1",
                "JARVIS_HINDSIGHT_API_KEY_FILE": str(path.resolve())})
            self.assertEqual(memory.scope.owner_id, "owner")
            await memory.aclose()
