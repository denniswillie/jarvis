"""Owned deployment cleanup, including failed startup and teardown."""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from evals.second_brain.hindsight_deployment_trace import deployment_service


class DeploymentCleanupTests(unittest.TestCase):
    def test_all_exit_paths_record_cleanup_and_protect_unrelated_projects(self):
        for failure in (None, "up", "body", "down"):
            with self.subTest(failure=failure):
                root = Path("/tmp")/("jarvis-m02-deployment.unit-"+uuid4().hex)
                calls = []
                def command(arguments, **kwargs):
                    calls.append(arguments)
                    if arguments[:3] == ["docker", "ps", "-a"]:
                        return ""
                    if arguments[:3] == ["docker", "image", "inspect"]:
                        return "sha256:fixture-image"
                    if arguments[:2] == ["docker", "inspect"]:
                        return json.dumps([
                            {"Id": "api", "Name": "/owned-api", "Image": "sha256:fixture-image",
                             "Config": {"Labels": {"com.docker.compose.project": "jarvis-m02-owned",
                                                   "com.docker.compose.service": "hindsight-api"}}},
                            {"Id": "db", "Name": "/owned-db", "Image": "sha256:fixture-db",
                             "Config": {"Labels": {"com.docker.compose.project": "jarvis-m02-owned",
                                                   "com.docker.compose.service": "hindsight-db"}}},
                        ])
                    self.assertEqual(arguments[arguments.index("-p")+1], "jarvis-m02-owned")
                    if "config" in arguments:
                        return json.dumps({"services": {"hindsight-api": {"ports": [{"host_ip": "127.0.0.1"}]},
                                                       "hindsight-db": {}}})
                    if "up" in arguments and failure == "up":
                        raise RuntimeError("injected-startup-failure")
                    if "down" in arguments and failure == "down":
                        raise RuntimeError("injected-cleanup-failure")
                    if "ps" in arguments:
                        return "api db"
                    if "port" in arguments:
                        return "127.0.0.1:19999"
                    if "logs" in arguments:
                        return "synthetic-api-key synthetic-native-token"
                    return ""
                try:
                    with patch("evals.second_brain.hindsight_deployment_trace.command", side_effect=command), \
                         patch("evals.second_brain.hindsight_deployment_trace.secrets.token_hex", return_value="owned"), \
                         patch("evals.second_brain.hindsight_deployment_trace.secrets.token_urlsafe",
                               return_value="synthetic-native-token"), \
                         patch("evals.second_brain.hindsight_deployment_trace.httpx.Client") as client:
                        client.return_value.__enter__.return_value.get.return_value.status_code = 200
                        try:
                            with deployment_service(root, "synthetic-api-key") as service:
                                self.assertEqual(service["url"], "http://127.0.0.1:19999")
                                if failure == "body":
                                    raise LookupError("injected-body-failure")
                        except (RuntimeError, LookupError):
                            self.assertIsNotNone(failure)
                    self.assertEqual(sum("down" in command for command in calls), 1)
                    manifest = json.loads((root/"deployment-resources.json").read_text())
                    if failure == "down":
                        self.assertEqual(manifest["status"], "cleanup_unverified")
                        self.assertFalse(manifest["secrets_removed"])
                        self.assertTrue((root/"secrets/openai-api-key").exists())
                    else:
                        self.assertEqual(manifest["status"], "cleaned")
                        self.assertTrue(manifest["secrets_removed"])
                        self.assertFalse((root/"secrets").exists())
                    self.assertNotIn("synthetic-api-key", json.dumps(manifest))
                    self.assertNotIn("synthetic-native-token", (root/"deployment.log").read_text())
                finally:
                    if root.exists():
                        shutil.rmtree(root)
