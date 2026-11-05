"""Model-envelope schema depth is separate from persisted memory-job limits."""
from types import SimpleNamespace
import unittest

from app.memory.jobs.models import InvalidMemoryJobPayloadError, normalize_job_json_object
from app.tasks.model_gateway import ModelGatewayError, TaskModelGateway
import test_second_brain_model_gateway as fixtures


class TaskModelSchemaDepthTests(unittest.TestCase):
    def test_registered_generic_nested_schema_survives_transport_normalization(self):
        schemas = [{"name": "generic_nested_document", "description": "Generic structured file input", "parameters":
            {"type": "object", "properties": {"rows": {"type": "array", "items":
                {"type": "object", "properties": {"cells": {"type": "array", "items": {"type": "string"}}}}}}}}]
        gateway = TaskModelGateway(fixtures.context(), fixtures.Upstream(), tools=schemas)
        payload, _ = gateway.prepare({"model": "fixture-model", "input": "Synthetic structured file request",
            "tools": [{"type": "function", **schema} for schema in schemas]})
        self.assertEqual(payload["tools"][0]["parameters"], schemas[0]["parameters"])
        self.assertFalse(payload["tools"][0]["strict"])

    def test_memory_job_default_depth_is_not_relaxed(self):
        value = 1
        for _ in range(14):
            value = {"nested": value}
        with self.assertRaises(InvalidMemoryJobPayloadError):
            normalize_job_json_object(value, label="memory fixture")
        self.assertEqual(normalize_job_json_object(value, label="transport fixture", max_depth=32), value)

    def test_transport_still_rejects_excessive_schema_depth(self):
        value = 1
        for _ in range(34):
            value = {"nested": value}
        schema = {"name": "fixture", "description": "Synthetic", "parameters": value}
        gateway = TaskModelGateway(fixtures.context(), fixtures.Upstream(), tools=[schema])
        with self.assertRaises(ModelGatewayError):
            gateway.prepare({"model": "fixture-model", "input": "Synthetic", "tools": [{"type": "function", **schema}]})

    def test_explicit_normalizer_depth_must_be_bounded_integer(self):
        for value in (-1, 129, True, 32.0, "32"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_job_json_object({}, label="fixture", max_depth=value)
