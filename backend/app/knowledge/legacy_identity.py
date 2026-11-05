"""Stable identities shared by frozen legacy import and original cleanup."""
import hashlib
import json


MAPPING_VERSION = 1
NAMESPACE = "legacy:m03:v1"


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def quoted(name):
    # Only the fixed population registry supplies SQL identifiers.
    return '"' + name.replace('"', '""') + '"'

