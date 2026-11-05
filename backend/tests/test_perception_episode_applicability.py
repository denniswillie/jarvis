"""Binder-to-episode relevance seam; real owner/pipeline scenarios live separately."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.models import SourceDependency
from app.live_state.perception_focus import FOCUS_SUBJECT
from app.live_state.perception_runtime import CanonicalPerceptionBinder


class EpisodeApplicabilityBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = object()
        self.episodes = (object(),)
        self.calls = []
        self.dependency = SourceDependency(source_namespace="episode", source_id=uuid4(), revision=2)
        self.brief = SimpleNamespace(dependencies=(self.dependency,))

        async def versions(connection, dependencies):
            self.assertIs(connection, self.connection)
            self.assertEqual(dependencies, (self.dependency,))
            self.calls.append("owner_versions")
            return self.episodes

        async def relevance(connection, brief, episodes):
            self.assertIs(connection, self.connection)
            self.assertIs(brief, self.brief)
            self.assertIs(episodes, self.episodes)
            self.calls.append("activity_relevance")

        self.context = SimpleNamespace(validate_assignment=AsyncMock(side_effect=relevance))
        self.canonical = SimpleNamespace(_validate_dependencies=AsyncMock(side_effect=versions),
                                         episode_context=self.context)
        self.binder = CanonicalPerceptionBinder(self.canonical)

    async def test_relevance_follows_owner_proof_on_the_same_connection(self):
        result = await self.binder._validate_dependencies(self.connection, self.brief)
        self.assertIs(result, self.episodes)
        self.assertEqual(self.calls, ["owner_versions", "activity_relevance"])

    async def test_activity_changed_expired_or_withdrawn_cannot_pass_owner_versions_alone(self):
        for code in ("live_episode_cue_inapplicable", "live_episode_activity_changed",
                     "live_episode_activity_mismatch"):
            with self.subTest(code=code):
                self.context.validate_assignment.side_effect = SourceError(code, "Fixture activity is no longer applicable.")
                with self.assertRaises(SourceError) as raised:
                    await self.binder._validate_dependencies(self.connection, self.brief)
                self.assertEqual(raised.exception.code, code)

    async def test_owner_failure_prevents_relevance_from_bypassing_it(self):
        self.canonical._validate_dependencies.side_effect = SourceError(
            "episode_check_changed", "Fixture owner revision changed.")
        with self.assertRaises(SourceError):
            await self.binder._validate_dependencies(self.connection, self.brief)
        self.context.validate_assignment.assert_not_awaited()

    async def test_non_episode_context_does_not_require_an_episode_runtime(self):
        self.canonical._validate_dependencies.side_effect = None
        self.canonical._validate_dependencies.return_value = ()
        self.canonical.episode_context = None
        self.brief = SimpleNamespace(dependencies=())
        self.assertEqual(await self.binder._validate_dependencies(self.connection, self.brief), ())
        self.context.validate_assignment.assert_not_awaited()

    async def test_existing_focus_guard_is_preserved(self):
        focus = SourceDependency(source_namespace=FOCUS_SUBJECT, source_id=uuid4(), revision=1, generation=1)
        self.brief = SimpleNamespace(dependencies=(self.dependency, focus))
        self.binder.focus = SimpleNamespace(validate=AsyncMock())
        await self.binder._validate_dependencies(self.connection, self.brief)
        self.binder.focus.validate.assert_awaited_once_with(self.connection, self.brief)
        self.context.validate_assignment.assert_awaited_once()
