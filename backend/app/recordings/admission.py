"""Durable recording authority and receipts, never a capture/transfer engine.

In-transaction methods require the caller's existing source-owner -> task-owner
transaction. No provider/storage I/O is performed here. Completion is called by
the existing FilePublication adoption callback: its outer transaction must mark
the publication available before committing, atomically with this saved receipt.
"""
from uuid import UUID

from app.evidence.models import OwnedReference
from app.files.repository import _file_from_record
from app.files.transfer import FileTransferPolicy, MP4_CAPTURE_POLICY
from app.knowledge.models import SourceError
from app.knowledge.product_files import FILE_NAMESPACE, RECORDING_ORIGINAL_FORMAT
from app.live_state.admission import CommandAdmission
from app.recordings.models import FrozenFinalization, RecordingReceipt, RecordingScope, freeze_manifest
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import digest, json_object


COMMAND_FORMAT = "jarvis.recording_command.v1"


class RecordingAdmission:
    def __init__(self, sources, tasks, selections, *, owner_id, access_check=None):
        OwnedReference.validate_owner(owner_id)
        if (sources.pool is not tasks.pool or selections.sources.pool is not sources.pool
                or selections.requests.pool is not tasks.pool):
            raise ValueError("Recording owners must share the canonical source/task pool")
        self.sources, self.tasks, self.selections = sources, tasks, selections
        self.owner_id, self.access_check = owner_id, access_check
        self.context = ContextRepository(tasks)

    def _authorize(self, owner_id):
        if owner_id != self.owner_id:
            raise SourceError("recording_owner_mismatch", "Recording is outside this owner scope.", 403)
        if self.access_check is not None:
            self.access_check(owner_id)

    def _command(self, command, capability):
        if type(command) is not CommandAdmission or command.capability != capability:
            raise SourceError("recording_command_invalid", "An exact admitted recording command is required.", 403)
        self._authorize(command.owner_id)
        if not isinstance(command.session_id, UUID) or not isinstance(command.operation_id, UUID):
            raise ValueError("Recording command identities must be exact UUIDs")
        args = command.normalized_arguments
        if capability == "recordings.start":
            if (type(args) is not dict or set(args) not in ({"include_jarvis_audio"},
                    {"include_jarvis_audio", "recording_mode", "include_media_audio"})
                    or type(args["include_jarvis_audio"]) is not bool):
                raise ValueError("Recording start requires only an explicit include_jarvis_audio boolean")
            if "recording_mode" in args and (args["recording_mode"] not in ("ordinary", "voiceover")
                    or type(args["include_media_audio"]) is not bool
                    or (args["recording_mode"] == "voiceover"
                        and (args["include_jarvis_audio"] or args["include_media_audio"]))):
                raise ValueError("Invalid recording media selection")
        else:
            if (type(args) is not dict or set(args) != {"recording_id"}
                    or not isinstance(args["recording_id"], str)
                    or str(UUID(args["recording_id"])) != args["recording_id"]):
                raise ValueError("Recording stop requires only an exact recording UUID string")
        return args, digest(command.parameters())

    async def _row(self, connection, owner_id, session_id, recording_id):
        if not connection.is_in_transaction():
            raise RuntimeError("Recording composition requires an existing owner transaction")
        self._authorize(owner_id)
        if not isinstance(session_id, UUID) or not isinstance(recording_id, UUID):
            raise ValueError("Recording and session identities must be exact UUIDs")
        await self.selections._session(connection, owner_id, session_id)
        row = await connection.fetchrow("""select * from jarvis_recordings
            where owner_id=$1 and session_id=$2 and recording_id=$3 for update""", owner_id, session_id, recording_id)
        if row is None:
            raise SourceError("recording_not_found", "No recording exists in this owner/session scope.", 404)
        return row

    @staticmethod
    def _scope(row):
        return RecordingScope.model_validate({key: row[key] for key in RecordingScope.model_fields})

    async def _proof(self, connection, row):
        if row["revoked"]:
            raise SourceError("recording_revoked", "Recording capture authority is revoked.", 403)
        await self.sources._area(connection, row["owner_id"], row["area_id"])
        record = await self.context._record(connection, row["owner_id"], row["admission_event_id"])
        request = await connection.fetchrow("""select * from jarvis_recording_requests
            where owner_id=$1 and operation_id=$2""", row["owner_id"], row["recording_id"])
        scope = self._scope(row)
        try:
            body = record.content
            raw = body["command"]
            command = CommandAdmission(owner_id=raw["owner_id"], session_id=UUID(raw["session_id"]),
                provider_session_key=raw["provider_session_key"], provider_call_id=raw["provider_call_id"],
                tool_name=raw["tool_name"], capability=raw["capability"], normalized_arguments=raw["arguments"])
            _, fingerprint = self._command(command, "recordings.start")
            expected = scope.model_dump(mode="json", exclude={"admission_event_id", "admission_payload_id"})
            if (request is None or request["command_digest"] != row["command_digest"]
                    or request["capability"] != "recordings.start" or request["recording_id"] != row["recording_id"]
                    or request["session_id"] != row["session_id"] or request["outcome"] != "admitted"
                    or request["command_event_id"] != record.id or request["command_payload_id"] != record.payload_id
                    or raw != command.parameters() or fingerprint != row["command_digest"]
                    or body != {"format": COMMAND_FORMAT, "command": raw, "scope": expected}
                    or command.operation_id != row["recording_id"] or command.session_id != row["session_id"]
                    or command.normalized_arguments["include_jarvis_audio"] != row["include_jarvis_audio"]
                    or record.owner_id != row["owner_id"] or record.session_id != row["session_id"]
                    or record.payload_id != row["admission_payload_id"] or record.revision != 1
                    or record.source_namespace != "voice_command" or record.source_event_id != str(row["recording_id"])
                    or record.kind != "command_request" or record.trust != "derived" or record.finality != "final"
                    or not record.is_current or record.availability != "available"):
                raise ValueError()
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise SourceError("recording_admission_changed", "The canonical recording admission is unavailable.", 403) from error
        if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", row["owner_id"], row["admission_payload_id"]) != "available":
            raise SourceError("recording_admission_changed", "The original recording authority is unavailable.", 403)
        return scope

    async def _prior(self, connection, command, fingerprint):
        prior = await connection.fetchrow("""select * from jarvis_recording_requests
            where owner_id=$1 and operation_id=$2""", command.owner_id, command.operation_id)
        if prior is not None and (prior["command_digest"] != fingerprint or prior["capability"] != command.capability
                or prior["session_id"] != command.session_id):
            raise SourceError("recording_request_conflict", "This operation already has a different recording command.")
        return prior

    def _receipt(self, row, request, *, replayed):
        return RecordingReceipt(scope=self._scope(row), operation_id=request["operation_id"],
            outcome=request["outcome"], receipt_event_id=request["receipt_event_id"],
            receipt_payload_id=request["receipt_payload_id"], replayed=replayed,
            capture_allowed=not (row["capture_closed"] or row["revoked"] or row["completion_event_id"] is not None))

    async def _append_command(self, connection, command, payload):
        return await self.context.append_in_transaction(connection, command.owner_id, ContextIngress(
            session_id=command.session_id, source_namespace="voice_command", source_event_id=str(command.operation_id),
            receipt_id=command.journal_call_key, kind="command_request", trust="derived", expected_revision=0,
            payload={"format": COMMAND_FORMAT, "command": command.parameters(), **payload}))

    async def _request_receipt(self, connection, command, fingerprint, recording_id, event, outcome):
        receipt = await self.context.append_in_transaction(connection, command.owner_id, ContextIngress(
            session_id=command.session_id, source_namespace="recording-receipt", source_event_id=str(command.operation_id),
            receipt_id="accepted", kind="tool_outcome", trust="tool_receipt", expected_revision=0,
            payload={"recording_id": str(recording_id), "operation_id": str(command.operation_id), "outcome": outcome},
            source_payload_ids=(event.payload_id,)))
        await connection.execute("""insert into jarvis_recording_requests
            (owner_id,operation_id,session_id,recording_id,capability,command_digest,command_event_id,command_payload_id,
             outcome,receipt_event_id,receipt_payload_id) values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
            command.owner_id, command.operation_id, command.session_id, recording_id, command.capability, fingerprint,
            event.id, event.payload_id, outcome, receipt.id, receipt.payload_id)
        return {"operation_id": command.operation_id, "outcome": outcome,
                "receipt_event_id": receipt.id, "receipt_payload_id": receipt.payload_id}

    async def admit(self, command, *, area_id, policy):
        args, fingerprint = self._command(command, "recordings.start")
        async with self.sources.transaction(command.owner_id) as connection:
            await self.tasks.lock_owner(connection, command.owner_id)
            await self.selections._session(connection, command.owner_id, command.session_id)
            prior = await self._prior(connection, command, fingerprint)
            if prior is not None:
                row = await self._row(connection, command.owner_id, command.session_id, prior["recording_id"])
                # The receipt pins the first server scope; changed current area or
                # configuration cannot silently mint a second capture allowance.
                if not row["revoked"]:
                    await self._proof(connection, row)
                return self._receipt(row, prior, replayed=True)
            if not isinstance(area_id, UUID) or type(policy) is not FileTransferPolicy or policy.policy_id != MP4_CAPTURE_POLICY:
                raise ValueError("A server-selected area and verified capture policy are required")
            selected = await self.selections.read_in_transaction(connection, command.owner_id, command.session_id)
            if selected["area_id"] != str(area_id) or selected["area_status"] != "active":
                raise SourceError("recording_area_changed", "The selected recording area changed.")
            await self.sources._area(connection, command.owner_id, area_id)
            values = {"recording_id": str(command.operation_id), "owner_id": command.owner_id,
                "session_id": str(command.session_id), "area_id": str(area_id),
                "include_jarvis_audio": args["include_jarvis_audio"], "transfer_policy": policy.policy_id,
                "max_bytes": policy.max_bytes, "max_duration_ms": policy.max_duration_ms}
            event = await self._append_command(connection, command, {"scope": values})
            await connection.execute("""insert into jarvis_recordings
                (owner_id,recording_id,session_id,area_id,include_jarvis_audio,transfer_policy,max_bytes,max_duration_ms,
                 command_digest,admission_event_id,admission_payload_id) values($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)""",
                command.owner_id, command.operation_id, command.session_id, area_id, args["include_jarvis_audio"],
                policy.policy_id, policy.max_bytes, policy.max_duration_ms, fingerprint, event.id, event.payload_id)
            request = await self._request_receipt(connection, command, fingerprint, command.operation_id, event, "admitted")
            row = await self._row(connection, command.owner_id, command.session_id, command.operation_id)
            return self._receipt(row, request, replayed=False)

    async def stop(self, command):
        args, fingerprint = self._command(command, "recordings.stop")
        recording_id = UUID(args["recording_id"])
        async with self.sources.transaction(command.owner_id) as connection:
            await self.tasks.lock_owner(connection, command.owner_id)
            prior = await self._prior(connection, command, fingerprint)
            row = await self._row(connection, command.owner_id, command.session_id, recording_id)
            if prior is not None:
                return self._receipt(row, prior, replayed=True)
            # Closing authority remains safe when the original was revoked. It
            # does not restore capture, promise media exists or erase a saved file.
            event = await self._append_command(connection, command, {"recording_id": str(recording_id)})
            await connection.execute("""update jarvis_recordings set capture_closed=true
                where owner_id=$1 and recording_id=$2""", command.owner_id, recording_id)
            request = await self._request_receipt(connection, command, fingerprint, recording_id, event, "capture_closed")
            row = await self._row(connection, command.owner_id, command.session_id, recording_id)
            return self._receipt(row, request, replayed=False)

    async def require_admitted_recording(self, connection, owner_id, session_id, recording_id, *, for_capture=False):
        if type(for_capture) is not bool:
            raise ValueError("Capture intent must be explicit")
        row = await self._row(connection, owner_id, session_id, recording_id)
        scope = await self._proof(connection, row)
        if for_capture and (row["capture_closed"] or row["final_manifest_json"] is not None or row["completion_event_id"] is not None):
            raise SourceError("recording_capture_closed", "This admission cannot start new capture.", 403)
        return scope

    async def _bound_row(self, connection, scope):
        if type(scope) is not RecordingScope:
            raise ValueError("An exact recording scope is required")
        row = await self._row(connection, scope.owner_id, scope.session_id, scope.recording_id)
        if self._scope(row) != scope:
            raise SourceError("recording_scope_changed", "The recording differs from its immutable admission.", 403)
        await self._proof(connection, row)
        return row

    async def get_finalization(self, connection, scope):
        row = await self._bound_row(connection, scope)
        if row["final_manifest_json"] is None:
            return None
        return FrozenFinalization(row["final_manifest_json"], row["final_manifest_digest"])

    async def get_completion(self, connection, scope):
        """Return an available exact saved receipt; never creates a publication."""
        row = await self._bound_row(connection, scope)
        if row["completion_event_id"] is None:
            return None
        return await self.complete_in_transaction(connection, scope, row["final_manifest_digest"],
            publication_id=row["publication_id"], file_id=row["file_id"], source_id=row["source_id"], version_id=row["version_id"])

    async def bind_finalization(self, connection, scope, manifest):
        frozen = freeze_manifest(scope, manifest)
        row = await self._bound_row(connection, scope)
        if row["final_manifest_json"] is not None:
            if (row["final_manifest_json"], row["final_manifest_digest"]) != (frozen.manifest_json, frozen.manifest_digest):
                raise SourceError("recording_finalization_conflict", "This recording already binds different final bytes or metadata.")
            return frozen
        if row["completion_event_id"] is not None:
            raise SourceError("recording_already_saved", "Saved recordings cannot acquire a new finalization.")
        await connection.execute("""update jarvis_recordings set final_manifest_json=$3,final_manifest_digest=$4,
            capture_closed=true where owner_id=$1 and recording_id=$2""",
            scope.owner_id, scope.recording_id, frozen.manifest_json, frozen.manifest_digest)
        return frozen

    async def complete_in_transaction(self, connection, scope, manifest_digest, *, publication_id, file_id, source_id, version_id):
        """Only inside existing Files adoption; outer commit sets available too.

        uploaded+settled is accepted for NEW atomic adoption. An exact replay
        requires available. No committed saved receipt is fabricated from bytes,
        inspection, a client report or an agent's claim alone.
        """
        row = await self._bound_row(connection, scope)
        if row["final_manifest_json"] is None or row["final_manifest_digest"] != manifest_digest:
            raise SourceError("recording_finalization_missing", "Completion requires this recording's exact frozen finalization.")
        if any(not isinstance(item, UUID) for item in (publication_id, file_id, source_id)) or not isinstance(version_id, str):
            raise ValueError("Completion requires exact publication, file and source identities")
        identities = (publication_id, file_id, source_id, version_id)
        replayed = row["completion_event_id"] is not None
        if replayed and identities != tuple(row[key] for key in ("publication_id", "file_id", "source_id", "version_id")):
            raise SourceError("recording_completion_conflict", "This recording already has a different saved result.")
        manifest = json_object(row["final_manifest_json"])
        publication = await connection.fetchrow("""select f.*,p.availability,p.content as specification
            from jarvis_file_publications f join jarvis_context_payloads p on p.owner_id=f.owner_id and p.id=f.metadata_payload_id
            where f.owner_id=$1 and f.id=$2 for update of f""", scope.owner_id, publication_id)
        if (publication is None or publication["operation_key"] != "recording:" + str(scope.recording_id)
                or publication["area_id"] != scope.area_id or publication["file_id"] != file_id
                or publication["phase"] not in ({"available"} if replayed else {"uploaded", "available"})
                or not publication["upload_settled"] or publication["upload_token"] is None or publication["delete_requested"]
                or publication["availability"] != "available" or publication["transfer_policy"] != scope.transfer_policy
                or publication["transfer_limit_bytes"] != scope.max_bytes
                or publication["size_bytes"] != manifest["size_bytes"] or publication["content_sha256"] != manifest["content_sha256"]):
            raise SourceError("recording_publication_mismatch", "No settled publication matches this exact recording.")
        file = await connection.fetchrow("select * from jarvis_files where owner_id=$1 and id=$2", scope.owner_id, file_id)
        source = await self.sources.resolve_document_in_transaction(connection, scope.owner_id, scope.area_id, version_id, historical=True)
        mapping = await connection.fetchrow("select * from jarvis_knowledge_sources where owner_id=$1 and id=$2", scope.owner_id, source_id)
        original = source.content
        if (file is None or file["session_id"] != scope.session_id or file["content_type"] != "video/mp4"
                or file["size_bytes"] != manifest["size_bytes"]
                or (file["storage_bucket"], file["storage_path"]) != (publication["storage_bucket"], publication["storage_path"])
                or source.source_id != source_id or source.session_id != scope.session_id
                or mapping is None or mapping["namespace"] != FILE_NAMESPACE or mapping["source_key"] != str(file_id)
                or source.source_type != "recording_file" or source.trust != "user_input"
                or original.get("format") != RECORDING_ORIGINAL_FORMAT or original.get("media_origin") != "client_reported"
                or original.get("recording_id") != str(scope.recording_id)
                or original.get("finalization_manifest") != manifest
                or original.get("media_inspection") != manifest.get("media_inspection")
                or original.get("content_sha256") != manifest["content_sha256"]
                or original.get("record") != _file_from_record(file).model_dump(mode="json")
                or file["filename"] != manifest.get("filename") or file["description"] != manifest.get("description")):
            raise SourceError("recording_source_mismatch", "The saved source does not match this recording's publication.")
        for parent, child in ((scope.admission_payload_id, publication["metadata_payload_id"]),
                              (publication["metadata_payload_id"], source.payload_id)):
            if not await connection.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3)""", scope.owner_id, parent, child):
                raise SourceError("recording_source_mismatch", "The saved recording lacks its exact admission/publication lineage.")
        for payload_id in (publication["metadata_payload_id"], source.payload_id):
            if await connection.fetchval("select lock_jarvis_copy_source($1,$2)", scope.owner_id, payload_id) != "available":
                raise SourceError("recording_source_unavailable", "The exact saved recording evidence is unavailable.", 403)
        if not replayed:
            receipt = await self.context.append_in_transaction(connection, scope.owner_id, ContextIngress(
                session_id=scope.session_id, source_namespace="recording-saved", source_event_id=str(scope.recording_id),
                receipt_id=manifest_digest, kind="tool_outcome", trust="tool_receipt", expected_revision=0,
                payload={"recording_id": str(scope.recording_id), "manifest_digest": manifest_digest,
                    "publication_id": str(publication_id), "file_id": str(file_id), "source_id": str(source_id),
                    "version_id": version_id, "outcome": "saved"},
                source_payload_ids=(scope.admission_payload_id, publication["metadata_payload_id"], source.payload_id)))
            await connection.execute("""update jarvis_recordings set publication_id=$3,file_id=$4,source_id=$5,version_id=$6,
                completion_event_id=$7,completion_payload_id=$8,capture_closed=true where owner_id=$1 and recording_id=$2""",
                scope.owner_id, scope.recording_id, publication_id, file_id, source_id, version_id, receipt.id, receipt.payload_id)
            row = await self._row(connection, scope.owner_id, scope.session_id, scope.recording_id)
        return RecordingReceipt(scope=scope, operation_id=scope.recording_id, outcome="saved",
            receipt_event_id=row["completion_event_id"], receipt_payload_id=row["completion_payload_id"],
            replayed=replayed, capture_allowed=False, publication_id=publication_id, file_id=file_id, source_id=source_id, version_id=version_id)
