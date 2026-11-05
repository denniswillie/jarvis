"""The native service refuses unsafe startup without loading it in voice."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from app.knowledge.service_entrypoint import deployment_environment, main, secret_file
from app.knowledge.service_policy import (
    FALSE_FIELDS, HINDSIGHT_VERSION, POLICY_ID, TENANT_EXTENSION,
    ServicePolicyError, validate_bank_configuration, validate_service_configuration,
)


def config(**changes):
    return SimpleNamespace(**(dict.fromkeys(FALSE_FIELDS, False) | {
        "database_url": "postgresql://fixture:private@127.0.0.1/hindsight",
        "worker_max_slots": 2, "worker_slot_reservations": {"consolidation": 1},
        "llm_max_concurrent": 1, "llm_timeout": 90,
    } | changes))


def environment():
    return {"HINDSIGHT_API_TENANT_EXTENSION": TENANT_EXTENSION,
            "HINDSIGHT_API_TENANT_API_KEY": "synthetic-authentication-key-123"}


class ServicePolicyTests(unittest.TestCase):
    def test_private_pinned_bounded_configuration_returns_only_safe_receipt(self):
        receipt = validate_service_configuration(config(), environment(), HINDSIGHT_VERSION)
        self.assertEqual(receipt["policy"], POLICY_ID)
        self.assertEqual(receipt["shared_worker_slots"], 1)
        self.assertNotIn("private", str(receipt))
        self.assertNotIn("synthetic-authentication-key", str(receipt))
        self.assertEqual(receipt["bank_overrides"], "must_be_checked_by_adapter")

    def test_every_payload_history_flag_is_strictly_disabled(self):
        for field in FALSE_FIELDS:
            for value in (True, None, 0, "false"):
                with self.subTest(field=field, value=value), self.assertRaises(ServicePolicyError):
                    validate_service_configuration(config(**{field: value}), environment(), HINDSIGHT_VERSION)

    def test_unaccepted_versions_authentication_and_capacity_fail_closed(self):
        with self.assertRaises(ServicePolicyError):
            validate_service_configuration(config(), environment(), "unreviewed-version")
        for env in ({}, environment() | {"HINDSIGHT_API_TENANT_EXTENSION": "another-extension"},
                    environment() | {"HINDSIGHT_API_TENANT_API_KEY": "short"}):
            with self.subTest(env=env), self.assertRaises(ServicePolicyError):
                validate_service_configuration(config(), env, HINDSIGHT_VERSION)
        for change in ({"database_url": "pg0"}, {"worker_max_slots": 99},
                       {"worker_slot_reservations": {"consolidation": 2}},
                       {"worker_slot_reservations": {"consolidation": True}},
                       {"llm_max_concurrent": 32}, {"llm_timeout": float("inf")}):
            with self.subTest(change=change), self.assertRaises(ServicePolicyError):
                validate_service_configuration(config(**change), environment(), HINDSIGHT_VERSION)

    def test_missing_filtered_bank_fields_are_not_assumed_safe(self):
        for payload in ({}, {"bank_id": "bank", "config": {"audit_log_enabled": False}},
                        {"bank_id": "foreign", "config": {"audit_log_enabled": False, "store_document_text": True}}):
            with self.subTest(payload=payload), self.assertRaises(ServicePolicyError):
                validate_bank_configuration(payload, "bank")

    def test_check_only_verifies_native_config_without_starting_the_server(self):
        output = io.StringIO()
        with patch.dict(os.environ, environment(), clear=True), \
             patch("app.knowledge.service_entrypoint.native_configuration", return_value=(config(), HINDSIGHT_VERSION)), \
             redirect_stdout(output):
            self.assertEqual(main(["--check-only"]), 0)
        self.assertIn('"policy_verified": true', output.getvalue())
        self.assertNotIn("synthetic-authentication-key", output.getvalue())

    def test_native_import_or_configuration_failure_is_redacted(self):
        output = io.StringIO()
        with patch.dict(os.environ, environment(), clear=True), \
             patch("app.knowledge.service_entrypoint.native_configuration", side_effect=ValueError("PRIVATE CONFIGURATION")), \
             redirect_stderr(output):
            self.assertEqual(main(["--check-only"]), 1)
        self.assertNotIn("PRIVATE", output.getvalue())
        self.assertIn("hindsight_startup_policy_unavailable", output.getvalue())

    def test_docker_secret_files_do_not_require_values_in_compose_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory)/"key"
            secret.write_text("fixture-secret\n")
            loaded = deployment_environment({"HINDSIGHT_API_LLM_API_KEY_FILE": str(secret)})
            self.assertEqual(loaded["HINDSIGHT_API_LLM_API_KEY"], "fixture-secret")
            self.assertNotIn("HINDSIGHT_API_LLM_API_KEY_FILE", loaded)
            with self.assertRaises(ServicePolicyError):
                deployment_environment({"HINDSIGHT_API_LLM_API_KEY_FILE": str(secret), "HINDSIGHT_API_LLM_API_KEY": "ambiguous"})
            password = Path(directory)/"password"
            password.write_text("p@:/?#")
            loaded = deployment_environment({"JARVIS_HINDSIGHT_DB_PASSWORD_FILE": str(password)})
            self.assertEqual(loaded["HINDSIGHT_API_DATABASE_URL"], "postgresql://hindsight:p%40%3A%2F%3F%23@hindsight-db:5432/hindsight")
            with self.assertRaises(ServicePolicyError):
                deployment_environment({"JARVIS_HINDSIGHT_DB_PASSWORD_FILE": str(password), "HINDSIGHT_API_DATABASE_URL": "configured"})

    def test_maximum_secret_with_one_trailing_newline_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory)/"key"
            secret.write_text("a"*8192+"\n")
            self.assertEqual(secret_file(str(secret)), "a"*8192)

    def test_malformed_or_missing_secret_files_do_not_echo_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = Path(directory)/"key"
            for value in ("", "private\nsecond-line", "a"*8193, "a"*8192+"\nextra"):

                secret.write_text(value)
                with self.subTest(length=len(value)), self.assertRaises(ServicePolicyError):
                    secret_file(str(secret))
            with self.assertRaises(ServicePolicyError):
                secret_file(str(Path(directory)/"missing"))
            with self.assertRaises(ServicePolicyError):
                secret_file("relative-private-path")
