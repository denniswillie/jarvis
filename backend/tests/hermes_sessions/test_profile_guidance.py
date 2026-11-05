import unittest

from app.hermes_sessions.profile_guidance import BEGIN, END, with_profile_guidance


class ProfileGuidanceTests(unittest.TestCase):
    def apply(self, config, **changes):
        return with_profile_guidance(config, **({"callbacks_enabled": True, "sandbox_controls": True} | changes))

    def test_native_config_preserves_other_hints_and_identity(self):
        base = {"model": {"default": "fixture"}, "platform_hints": {
            "cli": {"replace": "User identity", "append": "User preference"}, "other": "Untouched"}}
        output = self.apply(base)
        self.assertEqual(output["platform_hints"]["cli"]["replace"], "User identity")
        self.assertTrue(output["platform_hints"]["cli"]["append"].startswith("User preference"))
        self.assertEqual(output["platform_hints"]["other"], "Untouched")
        self.assertEqual(base["platform_hints"]["cli"]["append"], "User preference")

    def test_reprovision_does_not_duplicate_or_keep_disabled_capability(self):
        first = self.apply({})
        self.assertEqual(self.apply(first), first)
        disabled = self.apply(first, callbacks_enabled=False, sandbox_controls=False)
        text = disabled["platform_hints"]["cli"]["append"]
        self.assertEqual(text.count(BEGIN), 1)
        self.assertEqual(text.count(END), 1)
        self.assertNotIn("wake_agent.py", text)
        self.assertNotIn("Use session_sandbox", text)

    def test_plain_native_hint_remains_appended(self):
        output = self.apply({"platform_hints": {"cli": "User text"}})
        self.assertTrue(output["platform_hints"]["cli"]["append"].startswith("User text\n\n"))

    def test_ambiguous_managed_block_is_not_silently_overwritten(self):
        for text in (BEGIN, END, END + BEGIN, BEGIN + BEGIN + END):
            with self.assertRaises(ValueError):
                self.apply({"platform_hints": {"cli": text}})


if __name__ == "__main__":
    unittest.main()
