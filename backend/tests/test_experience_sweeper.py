import unittest

from app.experience.sweeper import ExperienceEventSweeper


class _FakeExperienceService:
    def __init__(self, closed_count: int = 0) -> None:
        self.closed_count = closed_count
        self.calls = 0

    async def sweep_due_events(self, **kwargs) -> int:
        self.calls += 1
        return self.closed_count


class ExperienceEventSweeperTests(unittest.IsolatedAsyncioTestCase):
    async def test_sweeper_records_success_and_closed_count(self) -> None:
        service = _FakeExperienceService(closed_count=3)
        sweeper = ExperienceEventSweeper(service)  # type: ignore[arg-type]

        result = await sweeper.run_once()

        diagnostics = sweeper.diagnostics()
        self.assertEqual(result, 3)
        self.assertEqual(service.calls, 1)
        self.assertEqual(diagnostics["closed_count"], 3)
        self.assertIsNotNone(diagnostics["last_success_at"])
        self.assertIsNone(diagnostics["last_error"])

    async def test_sweeper_records_failure_and_reraises(self) -> None:
        class _FailingService:
            async def sweep_due_events(self, **kwargs) -> int:
                raise RuntimeError("database unavailable")

        sweeper = ExperienceEventSweeper(_FailingService())  # type: ignore[arg-type]

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            await sweeper.run_once()

        diagnostics = sweeper.diagnostics()
        self.assertEqual(diagnostics["last_error"], "RuntimeError")
        self.assertIsNone(diagnostics["last_success_at"])
