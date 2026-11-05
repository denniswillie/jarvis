"""Explicit project enrollment, local inspection and outbound read-only connection."""
import argparse
import json
import os
from pathlib import Path
import stat
import sys

from pydantic import ValidationError

from .filesystem import ProjectReader, register_root
from .navigation import ProjectInspector
from .protocol import ConnectorError, ReadRequest, RootRegistration
from .transport import Connection, OutboundConnector, load_connection, reserved_connection


def load_registry(path: Path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ConnectorError("registry_permissions")
        data = json.load(handle)
    return [RootRegistration.model_validate(row) for row in data]


def save_registry(path: Path, roots):
    # Exclusive creation avoids silently replacing unrelated files or a symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump([root.model_dump(mode="json") for root in roots], handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    enroll = commands.add_parser("enroll")
    enroll.add_argument("--root", type=Path, action="append", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--root-id", required=True)
    inspect.add_argument("--request", required=True, help="JSON read operation, never a shell command")
    inspect.add_argument("--node", type=Path)
    inspect.add_argument("--typescript", type=Path)
    register = commands.add_parser("register")
    register.add_argument("--url", required=True)
    register.add_argument("--area-id", required=True)
    register.add_argument("--label", default="Mac")
    register.add_argument("--credentials", type=Path, required=True)
    register.add_argument("--bootstrap-token-env", default="JARVIS_PROJECT_BOOTSTRAP_TOKEN")
    serve = commands.add_parser("serve")
    serve.add_argument("--credentials", type=Path, required=True)
    serve.add_argument("--node", type=Path)
    serve.add_argument("--typescript", type=Path)
    serve.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "enroll":
            roots = [register_root(path, path.name) for path in args.root]
            save_registry(args.registry, roots)
            print(json.dumps([{"root_id": str(root.root_id), "label": root.label} for root in roots]))
        elif args.command == "register":
            token = os.environ.pop(args.bootstrap_token_env, "")
            if not token:
                raise ConnectorError("bootstrap_token_required")
            client = OutboundConnector(args.url, token)
            enrolled = None
            try:
                with reserved_connection(args.credentials) as save:
                    enrolled = client.register(load_registry(args.registry), area_id=args.area_id, label=args.label)
                    save(Connection(url=args.url, connector_id=enrolled["connector_id"], token=enrolled["token"]))
                print(json.dumps({key: value for key, value in enrolled.items() if key != "token"}))
            except BaseException:
                if enrolled is not None:
                    try:
                        client.post("/projects/connectors/" + enrolled["connector_id"], {}, method="DELETE")
                    except ConnectorError:
                        print(json.dumps({"status": "cleanup_required", "connector_id": enrolled["connector_id"]}))
                raise
            finally:
                client.close()
        elif args.command == "serve":
            connection = load_connection(args.credentials)
            client = OutboundConnector(connection.url, connection.token.get_secret_value(),
                                       node=args.node, typescript=args.typescript)
            try:
                if args.once:
                    client.once(lambda: load_registry(args.registry))
                else:
                    client.run(lambda: load_registry(args.registry))
            finally:
                client.close()
        else:
            roots = load_registry(args.registry)
            root = next((root for root in roots if str(root.root_id) == args.root_id), None)
            if root is None:
                raise ConnectorError("root_not_enrolled")
            reader = ProjectReader(root)
            request = ReadRequest.model_validate_json(args.request)
            result = ProjectInspector(reader, node=args.node, typescript=args.typescript).inspect(request)
            # Removing enrollment during inspection revokes the result.
            if root not in load_registry(args.registry):
                raise ConnectorError("root_not_enrolled")
            print(json.dumps(result))
        return 0
    except (ConnectorError, ValidationError, OSError, ValueError) as error:
        code = error.code if isinstance(error, ConnectorError) else "invalid_configuration_or_request"
        print(json.dumps({"status": "unavailable", "error": code}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
