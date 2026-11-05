from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from pydantic import ValidationError

from app.evidence.models import Claim, ClaimBasis, EvidenceRef, OwnedReference, SourceAvailability, SourceTrust
from app.migrations.runner import Migration, MigrationError, load_migrations, quote_schema, render_fresh_schema
from app.runtime.feature_config import Capability, SecondBrainSettings, TaskLimits
from app.settings import JarvisSettings
from evals.second_brain.fixtures import DatabaseFixture, validate_manifest


class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.base = dict(owner_id="owner-a", source_type="visual_frame", source_id=str(uuid4()),
                         source_version="1", event_time=self.now, recorded_at=self.now,
                         availability=SourceAvailability.AVAILABLE, trust=SourceTrust.SENSOR)

    def claim(self, **kwargs):
        return Claim(**(dict(owner_id="owner-a", id=uuid4(), content="The switch appeared off.",
                             basis=ClaimBasis.OBSERVED, evidence_refs=(EvidenceRef(**self.base),)) | kwargs))

    def test_available_observation_retains_basis_and_provenance(self):
        claim = self.claim(confidence=0.8)
        self.assertEqual(claim.basis, ClaimBasis.OBSERVED)
        self.assertEqual(len(claim.available_evidence(self.now)), 1)
        self.assertEqual(Claim.model_validate_json(claim.model_dump_json()), claim)

    def test_records_and_nested_references_are_immutable(self):
        claim = self.claim()
        with self.assertRaises(ValidationError):
            claim.content = "Something else"
        with self.assertRaises(ValidationError):
            claim.evidence_refs[0].owner_id = "owner-b"

    def test_every_timestamp_rejects_naive_values(self):
        for key in ("event_time", "recorded_at", "captured_at"):
            with self.subTest(field=key), self.assertRaises(ValidationError):
                EvidenceRef(**(self.base | {key: datetime.now()}))
        for key in ("valid_from", "valid_to"):
            with self.subTest(field=key), self.assertRaises(ValidationError):
                self.claim(**{key: datetime.now()})

    def test_future_source_clock_is_preserved_but_not_available_yet(self):
        for key in ("event_time", "recorded_at", "captured_at"):
            ref = EvidenceRef(**(self.base | {key: self.now + timedelta(seconds=1)}))
            self.assertFalse(ref.is_available_at(self.now))
            self.assertTrue(ref.is_available_at(self.now + timedelta(seconds=2)))
        with self.assertRaises(ValueError):
            EvidenceRef(**self.base).is_available_at(datetime.now())

    def test_unavailable_states_remain_distinct_and_do_not_support_current_claim(self):
        for status in SourceAvailability:
            with self.subTest(status=status):
                ref = EvidenceRef(**(self.base | {"availability": status}))
                self.assertEqual(ref.is_available_at(self.now), status == SourceAvailability.AVAILABLE)
                claim = self.claim(evidence_refs=(ref,))
                self.assertEqual(claim.evidence_refs[0].availability, status)

    def test_unknown_does_not_require_invented_evidence(self):
        claim = self.claim(basis="unknown", evidence_refs=(), uncertainty="No view of the switch.")
        self.assertEqual(claim.available_evidence(self.now), ())
        with self.assertRaises(ValidationError):
            self.claim(basis="unknown", evidence_refs=())
        with self.assertRaises(ValidationError):
            self.claim(evidence_refs=())

    def test_identical_source_id_does_not_cross_owner_boundary(self):
        other = EvidenceRef(**(self.base | {"owner_id": "owner-b"}))
        with self.assertRaises(ValidationError):
            self.claim(evidence_refs=(other,))
        with self.assertRaises(ValueError):
            other.require_scope(owner_id="owner-a")
        other.require_scope(owner_id="owner-b")

    def test_area_scope_is_explicit_and_not_inferred_from_owner(self):
        area = uuid4()
        ref = EvidenceRef(**(self.base | {"knowledge_area_id": area}))
        with self.assertRaises(ValueError):
            ref.require_scope(owner_id="owner-a", allowed_area_ids=frozenset())
        ref.require_scope(owner_id="owner-a", allowed_area_ids=frozenset({area}))

    def test_assistant_confidence_does_not_become_observed_success(self):
        ref = EvidenceRef(**(self.base | {"trust": "assistant_output"}))
        with self.assertRaises(ValidationError):
            self.claim(evidence_refs=(ref,), confidence=1)
        inferred = self.claim(evidence_refs=(ref,), basis="inferred", confidence=1)
        self.assertEqual(inferred.basis, ClaimBasis.INFERRED)

    def test_reports_require_matching_source_origin(self):
        for basis, trust in (("user_reported", "user_input"), ("source_reported", "external_content")):
            with self.subTest(basis=basis):
                with self.assertRaises(ValidationError):
                    self.claim(basis=basis)
                self.claim(basis=basis, evidence_refs=(EvidenceRef(**(self.base | {"trust": trust})),))

    def test_invalid_intervals_confidence_and_identity_rejected(self):
        for fields in ({"valid_to": self.now}, {"valid_from": self.now, "valid_to": self.now},
                       {"confidence": float("nan")}, {"confidence": 1.1}, {"content": " "},
                       {"owner_id": " owner-a"}, {"unexpected": "silently discarded?"}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                self.claim(**fields)
        for fields in ({"source_id": "x" * 513}, {"source_type": " "}, {"source_version": ""}):
            with self.subTest(fields=fields), self.assertRaises(ValidationError):
                EvidenceRef(**(self.base | fields))

    def test_replacement_is_a_different_same_owner_record(self):
        identity = uuid4()
        for reference in (OwnedReference(owner_id="owner-b", id=uuid4()), OwnedReference(owner_id="owner-a", id=identity)):
            with self.assertRaises(ValidationError):
                self.claim(id=identity, superseded_by=reference)
        self.claim(superseded_by=OwnedReference(owner_id="owner-a", id=uuid4()))


class CapabilityConfigurationTests(unittest.TestCase):
    def settings(self, **kwargs):
        return SecondBrainSettings(_env_file=None, **kwargs)

    def test_all_capabilities_default_off(self):
        settings = self.settings()
        for capability in Capability:
            self.assertFalse(settings.resolve(capability).enabled)

    def test_development_frontier_defaults_to_sol_without_changing_voice(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = self.settings(enabled="true", frontier_enabled="true")
            self.assertEqual(settings.frontier_selection().model, "gpt-5.6-sol")
            self.assertEqual(settings.frontier_selection().reasoning_effort, "medium")
            voice = JarvisSettings(_env_file=None)
            self.assertEqual(voice.AUDIO_PROVIDER_DEFAULT, "openai")
            self.assertIs(voice.GEMINI_LIVE_ENABLED, False)
            self.assertEqual(voice.GEMINI_VISUAL_MODEL, "gemini-3.1-flash-lite")

    def test_explicit_frontier_model_override_remains_available(self):
        settings = self.settings(enabled="true", frontier_enabled="true", frontier_model="fixture-model")
        self.assertEqual(settings.frontier_selection().model, "fixture-model")

    def test_unused_malformed_flags_limits_and_models_are_inert(self):
        settings = self.settings(tasks_enabled="bad", frontier_enabled="bad", owner_concurrency="not-a-number",
                                 frontier_model=" invalid model ")
        for capability in Capability:
            resolution = settings.resolve(capability)
            self.assertIsNone(resolution.issue)
            self.assertFalse(resolution.enabled)

    def test_selected_bad_flag_reports_unavailability_without_raising(self):
        for kwargs in ({"enabled": "bad"}, {"enabled": "true", "tasks_enabled": "bad"}):
            self.assertEqual(self.settings(**kwargs).resolve(Capability.TASKS).issue, "invalid_feature_flag")

    def test_selected_tasks_validate_budgets_but_not_unused_frontier_config(self):
        settings = self.settings(enabled="true", tasks_enabled="true", frontier_model="not a model")
        resolution = settings.resolve(Capability.TASKS, database_url="postgresql://configured")
        self.assertTrue(resolution.configured)
        self.assertIsInstance(resolution.configuration, TaskLimits)
        self.assertEqual(resolution.configuration.task_output_tokens, 32_000)

    def test_selected_invalid_limits_are_sanitized(self):
        settings = self.settings(enabled="true", tasks_enabled="true", owner_concurrency="private-invalid-value")
        resolution = settings.resolve(Capability.TASKS, database_url="postgresql://configured")
        self.assertEqual(resolution.invalid_fields, ("owner_concurrency",))
        self.assertNotIn("private-invalid-value", repr(resolution))
        self.assertFalse(resolution.configured)

    def test_aggregate_budget_and_concurrency_invariants(self):
        for kwargs in ({"owner_concurrency": 5}, {"request_output_tokens": 33_000}, {"task_input_tokens": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                TaskLimits(**kwargs)

    def test_missing_database_or_placeholder_provider_key_is_not_ready(self):
        settings = self.settings(enabled="true", frontier_enabled="true")
        self.assertEqual(settings.resolve(Capability.FRONTIER).issue, "database_not_configured")
        self.assertEqual(settings.resolve(Capability.FRONTIER, database_url="configured", provider_key="replace_me").issue,
                         "provider_key_not_configured")

    def test_provider_secret_is_not_exposed_by_configuration_representation(self):
        settings = self.settings(enabled="true", frontier_enabled="true")
        resolution = settings.resolve(Capability.FRONTIER, database_url="configured", provider_key="private-key-fixture")
        self.assertTrue(resolution.configured)
        self.assertNotIn("private-key-fixture", repr(resolution))
        self.assertNotIn("private-key-fixture", resolution.configuration.model_dump_json())

    def test_unimplemented_native_integrations_are_not_reported_ready(self):
        settings = self.settings(enabled="true", sandbox_enabled="true", push_enabled="true", mail_enabled="true")
        for capability in (Capability.MAIL,):
            result = settings.resolve(capability, database_url="configured", provider_key="configured")
            self.assertFalse(result.configured)
            self.assertEqual(result.issue, "integration_not_implemented")

    def test_bad_optional_environment_does_not_change_legacy_settings_or_imports(self):
        with patch.dict("os.environ", {"SECOND_BRAIN_ENABLED": "false", "SECOND_BRAIN_OWNER_CONCURRENCY": "bad",
                                      "SECOND_BRAIN_SANDBOX_ENABLED": "bad"}):
            settings = JarvisSettings(_env_file=None)
            self.assertEqual(settings.AUDIO_PROVIDER_DEFAULT, "openai")
            self.assertFalse(self.settings().resolve(Capability.SANDBOX).enabled)
            from app.main import create_app
            self.assertIsNotNone(create_app())


class MigrationArtifactTests(unittest.TestCase):
    def test_canonical_schema_matches_packaged_migrations(self):
        from importlib.resources import files
        self.assertEqual(files("app").joinpath("schema.sql").read_text(encoding="utf-8"), render_fresh_schema())
        self.assertEqual(tuple(item.version for item in load_migrations()), tuple(range(1, len(load_migrations()) + 1)))

    def test_checksum_covers_exact_migration_bytes(self):
        self.assertNotEqual(Migration(1, "fixture", "select 1;").checksum,
                            Migration(1, "fixture", "select 1;\n").checksum)

    def test_noncontiguous_duplicate_or_unsafe_migrations_rejected(self):
        for migrations in ((Migration(2, "fixture", ""),),
                           (Migration(1, "fixture", ""), Migration(2, "fixture", "")),
                           (Migration(1, "quote'", ""),)):
            with self.assertRaises(MigrationError):
                render_fresh_schema(migrations)
        for schema in ("public;drop schema public", "", "../public", "PUBLIC"):
            with self.assertRaises(ValueError):
                quote_schema(schema)

    def test_manifest_is_prepared_before_database_access_and_has_exact_scope(self):
        with TemporaryDirectory() as directory:
            fixture = DatabaseFixture("never-connect", Path(directory), test_name=self.id())
            validate_manifest(fixture.manifest)
            self.assertEqual(fixture.path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("never-connect", fixture.path.read_text())
            for alteration in ({"owners": ["real-owner"]}, {"schemas": ["public"]}, {"run_id": "not-a-uuid"}):
                with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                    validate_manifest(fixture.manifest | alteration)
