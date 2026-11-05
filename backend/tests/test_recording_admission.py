"""Offline domain tests. Fake connection/locks are not real SQL qualification."""
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileTransferPolicy
from app.files.repository import _file_from_record
from app.knowledge.models import SourceError
from app.knowledge.product_files import FILE_NAMESPACE, RECORDING_ORIGINAL_FORMAT
from app.live_state.admission import CommandAdmission
from app.recordings.admission import RecordingAdmission
from app.recordings.models import FINALIZATION_FORMAT, freeze_manifest
from app.tasks.context_events import ContextRecord
from app.tasks.models import digest


class Connection:
    def __init__(self):
        self.state = {name: {} for name in ("recordings", "requests", "events", "publications", "files", "sources", "versions", "availability")}
        self.state["dependencies"] = set()
        self.active = False
        self.order = []

    def is_in_transaction(self): return self.active

    async def fetchval(self, sql, *args):
        if sql.startswith("select lock_jarvis_copy_source"):
            return self.state["availability"].get(args[1], "available")
        if sql.startswith("select exists(select 1 from jarvis_context_payload_dependencies"):
            return args in self.state["dependencies"]
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        sql = " ".join(sql.split())
        if sql.startswith("select * from jarvis_recordings"):
            row = self.state["recordings"].get((args[0], args[2]))
            return deepcopy(row) if row and row["session_id"] == args[1] else None
        tables = {"select * from jarvis_recording_requests": "requests", "select f.*,p.availability": "publications",
                  "select * from jarvis_files": "files", "select * from jarvis_knowledge_sources": "sources"}
        for prefix, table in tables.items():
            if sql.startswith(prefix): return deepcopy(self.state[table].get((args[0], args[1])))
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        sql = " ".join(sql.split())
        if sql.startswith("insert into jarvis_recordings "):
            keys = ("owner_id", "recording_id", "session_id", "area_id", "include_jarvis_audio", "transfer_policy",
                    "max_bytes", "max_duration_ms", "command_digest", "admission_event_id", "admission_payload_id")
            row = dict(zip(keys, args)) | dict(capture_closed=False, revoked=False, final_manifest_json=None,
                final_manifest_digest=None, publication_id=None, file_id=None, source_id=None, version_id=None,
                completion_event_id=None, completion_payload_id=None)
            self.state["recordings"][(args[0], args[1])] = row
        elif sql.startswith("insert into jarvis_recording_requests "):
            keys = ("owner_id", "operation_id", "session_id", "recording_id", "capability", "command_digest",
                    "command_event_id", "command_payload_id", "outcome", "receipt_event_id", "receipt_payload_id")
            self.state["requests"][(args[0], args[1])] = dict(zip(keys, args))
        elif sql.startswith("update jarvis_recordings set "):
            row = self.state["recordings"][(args[0], args[1])]
            row["capture_closed"] = True
            if "final_manifest_json=$3" in sql:
                row.update(final_manifest_json=args[2], final_manifest_digest=args[3])
            elif "publication_id=$3" in sql:
                row.update(zip(("publication_id", "file_id", "source_id", "version_id", "completion_event_id", "completion_payload_id"), args[2:]))
        else: raise AssertionError(sql)


class Sources:
    def __init__(self, connection):
        self.pool = object()
        self.connection = connection
        self.areas = set()

    @asynccontextmanager
    async def transaction(self, owner_id):
        self.assert_idle()
        before = deepcopy(self.connection.state)
        self.connection.active = True
        self.connection.order.append("source")
        try: yield self.connection
        except BaseException:
            self.connection.state = before
            raise
        finally: self.connection.active = False

    def assert_idle(self):
        if self.connection.active: raise AssertionError("nested transaction")

    async def _area(self, connection, owner_id, area_id):
        if (owner_id, area_id) not in self.areas: raise SourceError("knowledge_area_unavailable", "Unavailable", 404)

    async def resolve_document_in_transaction(self, connection, owner, area, version, *, historical=False):
        row = connection.state["versions"].get((owner, area, version))
        if row is None: raise SourceError("knowledge_source_not_found", "Unavailable", 404)
        return row


class Tasks:
    def __init__(self, sources): self.pool = sources.pool
    async def lock_owner(self, connection, owner):
        if not connection.active: raise AssertionError("missing transaction")
        connection.order.append("task")


class Selections:
    def __init__(self, sources, tasks, owner, session, area):
        self.sources, self.requests = sources, tasks
        self.owner, self.session, self.area = owner, session, area
    async def _session(self, connection, owner, session):
        if (owner, session) != (self.owner, self.session): raise SourceError("knowledge_session_unavailable", "Unavailable", 404)
    async def read_in_transaction(self, connection, owner, session):
        await self._session(connection, owner, session)
        return {"area_id": str(self.area), "area_status": "active"}


class Context:
    async def append_in_transaction(self, connection, owner, event):
        if not connection.active: raise AssertionError("missing transaction")
        identity, payload = uuid4(), uuid4()
        record = ContextRecord(id=identity, payload_id=payload, owner_id=owner, session_id=event.session_id,
            sequence=len(connection.state["events"]) + 1, origin_sequence=len(connection.state["events"]) + 1,
            source_namespace=event.source_namespace, source_event_id=event.source_event_id, revision=1,
            supersedes_event_id=None, kind=event.kind, trust=event.trust, finality=event.finality,
            source_time=None, recorded_at=datetime.now(UTC), availability="available", content_digest=digest(event.payload),
            content=deepcopy(event.payload), is_current=True)
        connection.state["events"][(owner, identity)] = record
        return record
    async def _record(self, connection, owner, identity): return connection.state["events"][(owner, identity)]


class RecordingAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.db = Connection(); self.owner = "recording-test-owner"; self.session = uuid4(); self.area = uuid4()
        self.sources = Sources(self.db); self.sources.areas.add((self.owner, self.area))
        self.tasks = Tasks(self.sources)
        self.selections = Selections(self.sources, self.tasks, self.owner, self.session, self.area)
        self.admissions = RecordingAdmission(self.sources, self.tasks, self.selections, owner_id=self.owner)
        self.admissions.context = Context()
        self.policy = FileTransferPolicy("mp4_capture_v1", 1024)
        self.start = self.command("recordings.start", {"include_jarvis_audio": True}, call="start")

    def command(self, capability, arguments, *, call=None, session=None, owner=None):
        return CommandAdmission(owner_id=owner or self.owner, session_id=session or self.session,
            provider_session_key="actual-server-session", provider_call_id=call or str(uuid4()),
            tool_name="live_recording", capability=capability, normalized_arguments=arguments)

    async def admit(self):
        return await self.admissions.admit(self.start, area_id=self.area, policy=self.policy)

    def manifest(self, **changes):
        return {"format": FINALIZATION_FORMAT, "size_bytes": 100, "content_sha256": "a" * 64, "duration_ms": 1234,
                "filename": "recording.mp4", "description": "A requested recording", "media_inspection": {"measured": True},
                "client_report": {"timeline_end_ms": 1234, "missing_sources": []}} | changes

    @asynccontextmanager
    async def transaction(self):
        async with self.sources.transaction(self.owner) as connection:
            await self.tasks.lock_owner(connection, self.owner)
            yield connection

    async def bind(self, scope):
        async with self.transaction() as connection:
            return await self.admissions.bind_finalization(connection, scope, self.manifest())

    def completion_fixture(self, scope, manifest):
        publication_id, file_id, source_id, payload = uuid4(), uuid4(), uuid4(), uuid4()
        version = "recording_version_exact"
        self.db.state["publications"][(self.owner, publication_id)] = dict(
            id=publication_id, operation_key="recording:" + str(scope.recording_id), area_id=scope.area_id, file_id=file_id,
            phase="uploaded", upload_settled=True, upload_token=uuid4(), delete_requested=False, availability="available",
            transfer_policy=scope.transfer_policy, transfer_limit_bytes=scope.max_bytes, size_bytes=manifest["size_bytes"],
            content_sha256=manifest["content_sha256"], metadata_payload_id=uuid4(), storage_bucket="fixture", storage_path="exact")
        self.db.state["files"][(self.owner, file_id)] = dict(id=file_id, session_id=scope.session_id, content_type="video/mp4",
            size_bytes=manifest["size_bytes"], storage_bucket="fixture", storage_path="exact", filename=manifest["filename"],
            description=manifest["description"], source="upload", source_visual_frame_id=None,
            created_at=datetime.now(UTC), metadata={})
        self.db.state["sources"][(self.owner, source_id)] = dict(namespace=FILE_NAMESPACE, source_key=str(file_id))
        self.db.state["versions"][(self.owner, scope.area_id, version)] = SimpleNamespace(source_id=source_id,
            session_id=scope.session_id, payload_id=payload, source_type="recording_file", trust="user_input",
            content=dict(format=RECORDING_ORIGINAL_FORMAT, media_origin="client_reported", recording_id=str(scope.recording_id),
                finalization_manifest=deepcopy(manifest), media_inspection=deepcopy(manifest["media_inspection"]),
                record=_file_from_record(self.db.state["files"][(self.owner, file_id)]).model_dump(mode="json"),
                content_sha256=manifest["content_sha256"]))
        metadata = self.db.state["publications"][(self.owner, publication_id)]["metadata_payload_id"]
        self.db.state["dependencies"].update({(self.owner, scope.admission_payload_id, metadata), (self.owner, metadata, payload)})
        return dict(publication_id=publication_id, file_id=file_id, source_id=source_id, version_id=version)

    async def test_admission_is_permission_with_exact_canonical_scope_and_lock_order(self):
        receipt = await self.admit()
        self.assertEqual(receipt.outcome, "admitted"); self.assertIsNone(receipt.file_id)
        self.assertEqual(receipt.scope.recording_id, self.start.operation_id)
        self.assertTrue(receipt.capture_allowed); self.assertEqual(receipt.scope.max_bytes, 1024)
        self.assertEqual(receipt.scope.max_duration_ms, CAPTURE_MAX_DURATION_MS)
        self.assertEqual(self.db.order, ["source", "task"])
        async with self.transaction() as connection:
            self.assertEqual(await self.admissions.require_admitted_recording(connection, self.owner, self.session,
                receipt.scope.recording_id, for_capture=True), receipt.scope)

    async def test_replay_preserves_original_area_and_limit_but_changed_command_conflicts(self):
        first = await self.admit(); self.selections.area = uuid4()
        again = await self.admissions.admit(self.start, area_id=self.selections.area, policy=FileTransferPolicy("mp4_capture_v1", 64 * 1024 * 1024))
        self.assertEqual(first.scope, again.scope); self.assertTrue(again.replayed)
        self.assertEqual(len(self.db.state["requests"]), 1)
        changed = self.command("recordings.start", {"include_jarvis_audio": False}, call="start")
        with self.assertRaisesRegex(SourceError, "different recording command"):
            await self.admissions.admit(changed, area_id=self.area, policy=self.policy)

    async def test_strict_args_selection_policy_owner_and_session(self):
        for args in ({}, {"include_jarvis_audio": 1}, {"include_jarvis_audio": True, "max_bytes": 9999}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                await self.admissions.admit(self.command("recordings.start", args), area_id=self.area, policy=self.policy)
        for policy in (None, FileTransferPolicy(), FileTransferPolicy("mp4_output_v1", 1024)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                await self.admissions.admit(self.start, area_id=self.area, policy=policy)
        with self.assertRaises(SourceError): await self.admissions.admit(self.start, area_id=uuid4(), policy=self.policy)
        for command in (self.command("recordings.start", {"include_jarvis_audio": True}, owner="foreign"),
                        self.command("recordings.start", {"include_jarvis_audio": True}, session=uuid4())):
            with self.assertRaises(SourceError): await self.admissions.admit(command, area_id=self.area, policy=self.policy)
        self.assertEqual(self.db.state["recordings"], {})

    async def test_stop_before_finalization_closes_only_new_capture_and_replays(self):
        scope = (await self.admit()).scope
        command = self.command("recordings.stop", {"recording_id": str(scope.recording_id)})
        stopped = await self.admissions.stop(command)
        self.assertEqual(stopped.outcome, "capture_closed"); self.assertFalse(stopped.capture_allowed)
        self.assertIsNone(stopped.file_id); self.assertTrue((await self.admissions.stop(command)).replayed)
        async with self.transaction() as connection:
            with self.assertRaises(SourceError):
                await self.admissions.require_admitted_recording(connection, self.owner, self.session, scope.recording_id, for_capture=True)
            self.assertEqual(await self.admissions.require_admitted_recording(connection, self.owner, self.session, scope.recording_id), scope)
            frozen = await self.admissions.bind_finalization(connection, scope, self.manifest())
            self.assertEqual(frozen, await self.admissions.get_finalization(connection, scope))

    async def test_unknown_foreign_revoked_changed_original_and_forged_scope_denied(self):
        scope = (await self.admit()).scope
        async with self.transaction() as connection:
            for owner, session, recording in (("other", self.session, scope.recording_id), (self.owner, uuid4(), scope.recording_id),
                                             (self.owner, self.session, uuid4())):
                with self.assertRaises(SourceError): await self.admissions.require_admitted_recording(connection, owner, session, recording)
            with self.assertRaises(SourceError): await self.admissions.bind_finalization(connection, scope.model_copy(update={"max_bytes": 2048}), self.manifest())
        self.db.state["recordings"][(self.owner, scope.recording_id)]["revoked"] = True
        async with self.transaction() as connection:
            with self.assertRaises(SourceError): await self.admissions.bind_finalization(connection, scope, self.manifest())
        self.assertFalse((await self.admit()).capture_allowed)
        self.db.state["recordings"][(self.owner, scope.recording_id)]["revoked"] = False
        event = self.db.state["events"][(self.owner, scope.admission_event_id)]
        self.db.state["events"][(self.owner, scope.admission_event_id)] = event.model_copy(update={"is_current": False})
        async with self.transaction() as connection:
            with self.assertRaises(SourceError): await self.admissions.require_admitted_recording(connection, self.owner, self.session, scope.recording_id)

    async def test_finalization_is_immutable_bounded_and_does_not_nest_transactions(self):
        scope = (await self.admit()).scope; frozen = await self.bind(scope)
        async with self.transaction() as connection:
            self.assertEqual(await self.admissions.bind_finalization(connection, scope, dict(reversed(list(self.manifest().items())))), frozen)
            with self.assertRaises(SourceError): await self.admissions.bind_finalization(connection, scope, self.manifest(description="changed"))
        for changes in ({"size_bytes": 1025}, {"size_bytes": True}, {"duration_ms": CAPTURE_MAX_DURATION_MS + 1}, {"duration_ms": 0},
                        {"content_sha256": "bad"}, {"client_report": {"text": "x" * 16384}}, {"client_report": {1: "bad"}},
                        {"client_report": {"bad": float("nan")}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): freeze_manifest(scope, self.manifest(**changes))
        with self.assertRaises(RuntimeError): await self.admissions.get_finalization(self.db, scope)
        frozen.manifest["size_bytes"] = 999
        self.assertEqual(frozen.manifest["size_bytes"], 100)

    async def test_completion_requires_matching_frozen_binding_and_exact_settled_publication(self):
        scope = (await self.admit()).scope; binding = await self.bind(scope)
        identities = self.completion_fixture(scope, binding.manifest)
        publication = self.db.state["publications"][(self.owner, identities["publication_id"])]
        for key, bad in (("operation_key", "unrelated"), ("phase", "uploading"), ("upload_settled", False),
                         ("upload_token", None), ("file_id", uuid4()), ("content_sha256", "b" * 64),
                         ("size_bytes", 99), ("transfer_limit_bytes", 2048), ("delete_requested", True)):
            before = publication[key]; publication[key] = bad
            async with self.transaction() as connection:
                with self.subTest(key=key), self.assertRaises(SourceError):
                    await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **identities)
            publication[key] = before
        async with self.transaction() as connection:
            with self.assertRaises(SourceError): await self.admissions.complete_in_transaction(connection, scope, "0" * 64, **identities)
            wrong = dict(identities, publication_id=uuid4())
            with self.assertRaises(SourceError): await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **wrong)

    async def test_completion_exact_receipt_replay_and_outer_transaction_rollback(self):
        scope = (await self.admit()).scope; binding = await self.bind(scope)
        identities = self.completion_fixture(scope, binding.manifest)
        before = deepcopy(self.db.state)
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            async with self.transaction() as connection:
                saved = await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **identities)
                self.assertEqual(saved.outcome, "saved")
                raise RuntimeError("rollback")
        self.assertEqual(self.db.state, before)
        async with self.transaction() as connection:
            saved = await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **identities)
            self.db.state["publications"][(self.owner, identities["publication_id"])]["phase"] = "available"
        async with self.transaction() as connection:
            replay = await self.admissions.get_completion(connection, scope)
            self.assertTrue(replay.replayed); self.assertEqual(replay.receipt_event_id, saved.receipt_event_id)
            self.assertFalse(replay.capture_allowed)
            with self.assertRaises(SourceError):
                await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **dict(identities, file_id=uuid4()))
            with self.assertRaises(SourceError):
                await self.admissions.require_admitted_recording(connection, self.owner, self.session, scope.recording_id, for_capture=True)

    async def test_source_original_cannot_borrow_other_recording_or_changed_manifest(self):
        scope = (await self.admit()).scope; binding = await self.bind(scope)
        identities = self.completion_fixture(scope, binding.manifest)
        original = self.db.state["versions"][(self.owner, scope.area_id, identities["version_id"])].content
        for key, bad in (("recording_id", str(uuid4())), ("finalization_manifest", self.manifest(description="different")),
                         ("format", "jarvis.generated_file_original.v1"), ("media_origin", "sensor")):
            before = original[key]; original[key] = bad
            async with self.transaction() as connection:
                with self.subTest(key=key), self.assertRaises(SourceError):
                    await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **identities)
            original[key] = before
        self.db.state["dependencies"].clear()
        async with self.transaction() as connection:
            with self.assertRaises(SourceError):
                await self.admissions.complete_in_transaction(connection, scope, binding.manifest_digest, **identities)


if __name__ == "__main__":
    unittest.main()
