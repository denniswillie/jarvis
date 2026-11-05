"""Isolated command identity tests with no runtime or service dependencies."""

import hashlib
import unittest
from collections import UserDict
from datetime import date
from decimal import Decimal
from types import MappingProxyType
from uuid import NAMESPACE_URL, UUID, uuid5

from app.live_state.command_identity import (
    command_arguments_digest,
    command_operation_id,
)


class CommandOperationIdTests(unittest.TestCase):
    def setUp(self):
        self.components = {
            "owner_id": "opaque-owner",
            "session_id": UUID("F47AC10B-58CC-4372-A567-0E02B2C3D479"),
            "provider_session_key": "provider-session",
            "provider_call_id": "call-42",
        }

    def operation(self, **overrides):
        return command_operation_id(**{**self.components, **overrides})

    def test_exact_reference_encoding_with_unicode_and_whitespace(self):
        reference_name = (
            '[" Owner \\u00c9 \\u96ea ",'
            '"f47ac10b-58cc-4372-a567-0e02b2c3d479",'
            '" Provider\\tKey "," Call/\\u03b1\\n "]'
        )
        expected = uuid5(uuid5(NAMESPACE_URL, "jarvis.command.v1"), reference_name)
        components = {
            **self.components,
            "owner_id": " Owner \u00c9 \u96ea ",
            "provider_session_key": " Provider\tKey ",
            "provider_call_id": " Call/\u03b1\n ",
        }
        self.assertEqual(command_operation_id(**components), expected)
        self.assertEqual(command_operation_id(**components), expected)
        self.assertEqual(components["provider_call_id"], " Call/\u03b1\n ")

    def test_each_namespace_component_isolates_calls(self):
        baseline = self.operation()
        alternatives = {
            "owner_id": "another-owner",
            "session_id": UUID("f47ac10b-58cc-4372-a567-0e02b2c3d480"),
            "provider_session_key": "another-provider-session",
            "provider_call_id": "another-call",
        }
        identities = {baseline}
        for field, value in alternatives.items():
            with self.subTest(field=field):
                identity = self.operation(**{field: value})
                self.assertNotEqual(identity, baseline)
                identities.add(identity)
        self.assertEqual(len(identities), 5)

    def test_session_uuid_is_canonical_and_operation_is_uuid5(self):
        session = self.components["session_id"]
        baseline = self.operation()
        self.assertIsInstance(baseline, UUID)
        self.assertEqual(baseline.version, 5)
        self.assertEqual(
            self.operation(session_id=UUID(str(session).upper())), baseline
        )
        self.assertEqual(self.operation(session_id=UUID(hex=session.hex)), baseline)

    def test_owner_is_an_opaque_non_uuid_string(self):
        self.assertIsInstance(self.operation(owner_id="auth0|tenant:user/alpha"), UUID)

    def test_string_components_preserve_whitespace_and_case(self):
        baseline = self.operation()
        for field in ("owner_id", "provider_session_key", "provider_call_id"):
            value = self.components[field]
            for variant in (" " + value, value + " ", value.upper(), value + "\n"):
                with self.subTest(field=field, variant=variant):
                    self.assertNotEqual(self.operation(**{field: variant}), baseline)

    def test_string_components_preserve_unicode_representation(self):
        for field in ("owner_id", "provider_session_key", "provider_call_id"):
            with self.subTest(field=field):
                self.assertNotEqual(
                    self.operation(**{field: "caf\u00e9"}),
                    self.operation(**{field: "cafe\u0301"}),
                )

    def test_json_encoding_keeps_component_boundaries_distinct(self):
        self.assertNotEqual(
            self.operation(provider_session_key="a:b", provider_call_id="c"),
            self.operation(provider_session_key="a", provider_call_id="b:c"),
        )

    def test_identifiers_reject_non_strings(self):
        invalid = (None, True, 42, b"id", ["id"], {"id": 1}, self.components["session_id"])
        for field in ("owner_id", "provider_session_key", "provider_call_id"):
            for value in invalid:
                with self.subTest(field=field, value=value):
                    with self.assertRaises(TypeError):
                        self.operation(**{field: value})

    def test_identifiers_reject_blank_strings(self):
        for field in ("owner_id", "provider_session_key", "provider_call_id"):
            for value in ("", " ", "\t\r\n", "\u2003"):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.operation(**{field: value})

    def test_session_rejects_non_uuid_values_including_uuid_strings(self):
        session = self.components["session_id"]
        for value in (None, True, 42, str(session), session.bytes, object()):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    self.operation(session_id=value)


class CommandArgumentsDigestTests(unittest.TestCase):
    def test_exact_canonical_json_reference(self):
        arguments = {
            "z": 2,
            "a": {"flags": [True, None, False], "accent": "\u00e9"},
        }
        reference = (
            b'{"arguments":{"a":{"accent":"\\u00e9",'
            b'"flags":[true,null,false]},"z":2},"tool":"search\\u96ea"}'
        )
        digest = command_arguments_digest("search\u96ea", arguments)
        self.assertEqual(digest, hashlib.sha256(reference).hexdigest())
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_object_order_is_stable_at_every_depth(self):
        first = {"b": [{"y": 2, "x": 1}], "a": {"d": 4, "c": 3}}
        second = {"a": {"c": 3, "d": 4}, "b": [{"x": 1, "y": 2}]}
        self.assertEqual(
            command_arguments_digest("lookup", first),
            command_arguments_digest("lookup", second),
        )

    def test_mapping_inputs_and_repeated_non_circular_containers(self):
        shared = MappingProxyType({"value": [1, 2.5, None, True, "text"]})
        arguments = UserDict({"left": shared, "right": shared})
        plain = {"left": dict(shared), "right": dict(shared)}
        self.assertEqual(
            command_arguments_digest("lookup", arguments),
            command_arguments_digest("lookup", plain),
        )
        self.assertIs(arguments["left"], shared)
        self.assertEqual(shared["value"], [1, 2.5, None, True, "text"])

    def test_changed_tool_or_arguments_conflict_without_changing_identity(self):
        components = (
            "opaque-owner",
            UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            "provider-session",
            "raw-call-id",
        )
        commands = (
            ("lookup", {"query": "first"}),
            ("write", {"query": "first"}),
            ("lookup", {"query": "second"}),
        )
        identities = [command_operation_id(*components) for _ in commands]
        digests = [command_arguments_digest(tool, args) for tool, args in commands]
        self.assertEqual(len(set(identities)), 1)
        self.assertEqual(len(set(digests)), 3)
        self.assertEqual(components[-1], "raw-call-id")

    def test_array_order_and_json_value_types_are_significant(self):
        arguments = (
            {"value": [1, 2]},
            {"value": [2, 1]},
            {"value": "1"},
            {"value": 1},
            {"value": True},
            {"value": None},
        )
        digests = {command_arguments_digest("lookup", args) for args in arguments}
        self.assertEqual(len(digests), len(arguments))

    def test_tool_and_argument_strings_are_not_normalized(self):
        tools = ("lookup", " lookup ", "LOOKUP", "caf\u00e9", "cafe\u0301")
        self.assertEqual(len({command_arguments_digest(tool, {}) for tool in tools}), 5)
        values = ("value", " value ", "VALUE", "caf\u00e9", "cafe\u0301")
        self.assertEqual(
            len({command_arguments_digest("lookup", {"x": value}) for value in values}),
            5,
        )

    def test_tool_name_rejects_invalid_types_and_blank_values(self):
        for value in (None, True, 42, b"tool", [], {}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    command_arguments_digest(value, {})
        for value in ("", " ", "\t\n", "\u2003"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    command_arguments_digest(value, {})

    def test_arguments_must_be_a_mapping(self):
        for value in (None, True, 42, "{}", b"{}", [], (), [("key", "value")]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    command_arguments_digest("lookup", value)

    def test_nonfinite_numbers_are_rejected_at_every_depth(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for payload in (value, [value], {"nested": [value]}):
                with self.subTest(value=value, payload=payload):
                    with self.assertRaises(ValueError):
                        command_arguments_digest("lookup", {"payload": payload})

    def test_non_json_values_are_rejected_at_every_depth(self):
        invalid = (
            (1, 2),
            {1, 2},
            frozenset({1}),
            b"bytes",
            bytearray(b"bytes"),
            Decimal("1.25"),
            complex(1, 2),
            date(2026, 11, 1),
            UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            object(),
        )
        for value in invalid:
            for payload in (value, [value], {"nested": [value]}):
                with self.subTest(value_type=type(value), payload=payload):
                    with self.assertRaises(TypeError):
                        command_arguments_digest("lookup", {"payload": payload})

    def test_non_string_object_keys_are_rejected_at_every_depth(self):
        for key in (1, 1.5, None, False, ("key",), b"key"):
            invalid = {"valid": 1, key: "value"}
            for arguments in (invalid, {"nested": invalid}, {"nested": [invalid]}):
                with self.subTest(key=key, arguments=arguments):
                    with self.assertRaises(TypeError):
                        command_arguments_digest("lookup", arguments)

    def test_circular_objects_and_arrays_are_rejected(self):
        circular_object = {}
        circular_object["self"] = circular_object
        circular_array = []
        circular_array.append(circular_array)
        for arguments in (circular_object, {"array": circular_array}):
            with self.subTest(container_type=type(arguments.get("array", arguments))):
                with self.assertRaises(ValueError):
                    command_arguments_digest("lookup", arguments)
