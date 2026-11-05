"""Durable transport and source publication, not a filesystem or agent runtime."""
from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import hmac
import json
import secrets
from uuid import UUID

from app.knowledge.models import MemorySourceReference, SourceError, SourcePublication
from app.knowledge.read_provenance import SourceReadProvenance
from app.projects.models import Completion, Enrollment, Inspection, Poll, bounded_payload, encoded
from app.tasks.models import digest


def metadata(source):
    result = source.original_evidence()
    result.pop("content")
    return result


class ProjectService:
    def __init__(self, sources):
        self.sources = sources
        self.provenance = SourceReadProvenance(sources)

    async def enroll(self, owner_id, request: Enrollment):
        # Only the owner's administrative HTTP route calls this, never model tools.
        secret = secrets.token_urlsafe(32)
        async with self.sources.transaction(owner_id) as connection:
            await self.sources._area(connection, owner_id, request.area_id)
            count = await connection.fetchval(
                "select count(*) from jarvis_project_connectors where owner_id=$1 and status='active'", owner_id)
            if count >= 16:
                raise SourceError("project_connector_limit", "Revoke an unused connector before enrolling another.")
            row = await connection.fetchrow("""insert into jarvis_project_connectors
                (owner_id,area_id,label,token_digest,expires_at)
                values($1,$2,$3,$4,clock_timestamp()+make_interval(days=>$5)) returning id,expires_at""",
                owner_id, request.area_id, request.label, sha256(secret.encode()).hexdigest(), request.expires_in_days)
            roots = []
            for root in request.roots:
                payload = await self._payload(connection, owner_id, {
                    "kind": "project_root_authorization", "connector_id": str(row["id"]),
                    "local_root_id": str(root.local_root_id), "fingerprint": root.fingerprint})
                identity = await connection.fetchval("""insert into jarvis_project_roots
                    (owner_id,area_id,connector_id,local_root_id,label,fingerprint,authorization_payload_id)
                    values($1,$2,$3,$4,$5,$6,$7) returning id""",
                    owner_id, request.area_id, row["id"], root.local_root_id, root.label, root.fingerprint, payload)
                roots.append({"root_id": str(identity), "local_root_id": str(root.local_root_id), "label": root.label})
        return {"protocol_version": 1, "connector_id": str(row["id"]),
                "token": str(row["id"]) + "." + secret,
                "expires_at": row["expires_at"].isoformat(), "roots": roots}

    async def _credential(self, token):
        try:
            identity, secret = token.split(".", 1)
            identity = UUID(identity)
            if not 32 <= len(secret) <= 100:
                raise ValueError
        except (ValueError, AttributeError):
            raise SourceError("project_auth_invalid", "A valid scoped connector credential is required.", 401) from None
        async with self.sources.pool.acquire() as connection:
            row = await connection.fetchrow(
                "select owner_id,token_digest from jarvis_project_connectors where id=$1", identity)
        if row is None or not hmac.compare_digest(row["token_digest"], sha256(secret.encode()).hexdigest()):
            raise SourceError("project_auth_invalid", "A valid scoped connector credential is required.", 401)
        return row["owner_id"], identity

    async def _active_connector(self, connection, owner_id, identity):
        row = await connection.fetchrow("""select * from jarvis_project_connectors
            where owner_id=$1 and id=$2 and status='active' and expires_at>clock_timestamp() for update""",
            owner_id, identity)
        if row is None:
            raise SourceError("project_connector_unavailable", "The connector credential is expired or revoked.", 403)
        await self.sources._area(connection, owner_id, row["area_id"])
        return row

    async def revoke(self, owner_id, connector_id):
        async with self.sources.transaction(owner_id) as connection:
            row = await connection.fetchval("""update jarvis_project_connectors set status='revoked'
                where owner_id=$1 and id=$2 returning id""", owner_id, connector_id)
            if row is None:
                raise SourceError("project_connector_not_found", "No connector exists in this owner scope.", 404)
        return {"connector_id": str(connector_id), "status": "revoked"}

    async def roots(self, owner_id, area_id):
        async with self.sources.transaction(owner_id) as connection:
            await self.sources._area(connection, owner_id, area_id)
            rows = await connection.fetch("""select r.id,r.label,r.status,c.last_seen_at,
                (c.status='active' and c.expires_at>clock_timestamp()
                    and c.last_seen_at>clock_timestamp()-interval '45 seconds') as online
                from jarvis_project_roots r join jarvis_project_connectors c
                on c.owner_id=r.owner_id and c.id=r.connector_id
                where r.owner_id=$1 and r.area_id=$2 order by r.id limit 256""", owner_id, area_id)
        return {"roots": [{"root_id": str(row["id"]), "label": row["label"], "status": row["status"],
                          "online": bool(row["online"]) and row["status"] == "active",
                          "last_seen_at": row["last_seen_at"].isoformat() if row["last_seen_at"] else None}
                         for row in rows],
                "meaning": "Enrollment and recent contact, not proof of current file content. No host path is exposed."}

    async def _root(self, connection, owner_id, area_id, root_id, *, require_online=False):
        row = await connection.fetchrow("""select r.*,c.last_seen_at,
            (c.expires_at>clock_timestamp() and c.status='active') as connector_active
            from jarvis_project_roots r join jarvis_project_connectors c
            on c.owner_id=r.owner_id and c.id=r.connector_id
            join jarvis_context_payloads p on p.owner_id=r.owner_id and p.id=r.authorization_payload_id
            where r.owner_id=$1 and r.area_id=$2 and r.id=$3
                and r.status='active' and p.availability='available' for update of r""", owner_id, area_id, root_id)
        await self.sources._area(connection, owner_id, area_id)
        if row is None or not row["connector_active"]:
            raise SourceError("project_root_unavailable", "The approved project root is unavailable.", 404)
        if require_online and (row["last_seen_at"] is None or
                (datetime.now(UTC) - row["last_seen_at"]).total_seconds() >= 45):
            raise SourceError("project_connector_offline", "The Mac connector is offline; this is not an empty repository.", 503)
        return row

    async def submit(self, owner_id, area_id, inspection: Inspection):
        async with self.sources.transaction(owner_id) as connection:
            root = await self._root(connection, owner_id, area_id, inspection.root_id, require_online=True)
            await self._expire(connection, owner_id)
            count = await connection.fetchval("""select count(*) from jarvis_project_requests
                where owner_id=$1 and status in ('pending','leased')""", owner_id)
            if count >= 8:
                raise SourceError("project_requests_busy", "Too many project reads are pending; inspect existing requests.", 429)
            identity = await self._enqueue(connection, owner_id, root, inspection.request)
        return {"request_id": str(identity), "project_status": "pending"}

    async def _enqueue(self, connection, owner, root, request, *, purpose="inspection"):
        payload = await self._payload(connection, owner, {"request": request},
                                      (root["authorization_payload_id"],))
        return await connection.fetchval("""insert into jarvis_project_requests
            (owner_id,area_id,root_id,request_payload_id,purpose) values($1,$2,$3,$4,$5) returning id""",
            owner, root["area_id"], root["id"], payload, purpose)

    @staticmethod
    async def _expire(connection, owner):
        rows = await connection.fetch("""update jarvis_project_requests set status='expired'
            where owner_id=$1 and status in ('pending','leased') and deadline<=clock_timestamp()
            returning request_payload_id""", owner)
        if rows:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=any($2::uuid[]) and availability<>'deleted'""",
                owner, [row["request_payload_id"] for row in rows])

    async def _refresh_one(self, connection, owner, connector_id):
        """Refresh one previously read file on idle polls; the Mac still enforces policy."""
        row = await connection.fetchrow("""select w.*,p.content->>'path' as path,r.area_id,
                r.id,r.authorization_payload_id
            from jarvis_project_file_watches w
            join jarvis_project_roots r on r.owner_id=w.owner_id and r.id=w.root_id
            join jarvis_knowledge_sources s on s.owner_id=w.owner_id and s.id=w.source_id
            join jarvis_knowledge_source_versions v on v.owner_id=s.owner_id
                and v.source_id=s.id and v.revision=s.current_revision
            join jarvis_context_payloads p on p.owner_id=v.owner_id and p.id=v.payload_id
            where w.owner_id=$1 and r.connector_id=$2 and r.status='active'
                and s.status='active' and p.availability='available'
                and w.checked_at<clock_timestamp()-interval '10 seconds'
                and not exists(select 1 from jarvis_project_requests q where q.owner_id=w.owner_id
                    and q.root_id=w.root_id and q.status in ('pending','leased'))
            order by w.checked_at,w.source_id limit 1""", owner, connector_id)
        if row is None:
            return
        await connection.execute("""update jarvis_project_file_watches set checked_at=clock_timestamp()
            where owner_id=$1 and root_id=$2 and source_id=$3""", owner, row["root_id"], row["source_id"])
        await self._enqueue(connection, owner, row,
                            {"operation": "read", "path": row["path"], "max_chars": 1, "limit": 30}, purpose="refresh")

    async def poll(self, token, request: Poll):
        owner, connector_id = await self._credential(token)
        async with self.sources.transaction(owner) as connection:
            connector = await self._active_connector(connection, owner, connector_id)
            enrolled = await connection.fetch("""select * from jarvis_project_roots
                where owner_id=$1 and connector_id=$2 and status='active' order by id""", owner, connector_id)
            offered = {root.local_root_id: root for root in request.roots}
            for root in enrolled:
                offer = offered.get(root["local_root_id"])
                if offer is None or offer.fingerprint != root["fingerprint"]:
                    await connection.execute(
                        "update jarvis_project_roots set status='revoked' where owner_id=$1 and id=$2",
                        owner, root["id"])
            await connection.execute("update jarvis_project_connectors set last_seen_at=clock_timestamp() where id=$1", connector_id)
            await self._expire(connection, owner)
            await self._refresh_one(connection, owner, connector_id)
            row = await connection.fetchrow("""select q.*,r.local_root_id,r.fingerprint,p.content as request_content
                from jarvis_project_requests q join jarvis_project_roots r on r.owner_id=q.owner_id and r.id=q.root_id
                join jarvis_context_payloads p on p.owner_id=q.owner_id and p.id=q.request_payload_id
                where q.owner_id=$1 and r.connector_id=$2 and r.status='active' and q.deadline>clock_timestamp()
                    and p.availability='available'
                    and (q.status='pending' or (q.status='leased' and q.lease_until<=clock_timestamp()))
                    and not exists(select 1 from jarvis_project_requests busy
                        where busy.owner_id=q.owner_id and busy.root_id=q.root_id and busy.id<>q.id
                            and busy.status='leased' and busy.lease_until>clock_timestamp()
                            and busy.deadline>clock_timestamp())
                order by q.created_at,q.id limit 1 for update of q""", owner, connector_id)
            if row is None:
                return {"protocol_version": 1, "request": None}
            leased = await connection.fetchrow("""update jarvis_project_requests
                set status='leased',lease_id=gen_random_uuid(),
                    lease_until=least(deadline,clock_timestamp()+interval '40 seconds')
                where id=$1 returning lease_id,lease_until""", row["id"])
            return {"protocol_version": 1, "request": {"request_id": str(row["id"]),
                "lease_id": str(leased["lease_id"]), "lease_until": leased["lease_until"].isoformat(),
                "local_root_id": str(row["local_root_id"]), "fingerprint": row["fingerprint"],
                "read": (json.loads(row["request_content"]) if isinstance(row["request_content"], str)
                         else row["request_content"])["request"]}}

    @staticmethod
    async def _payload(connection, owner, content, dependencies=()):
        bounded_payload(content)
        if dependencies and not await connection.fetchval(
                "select lock_jarvis_copy_sources($1,$2::uuid[])", owner, list(dict.fromkeys(dependencies))):
            raise SourceError("project_source_unavailable", "The original project authorization or source was removed.")
        identity = await connection.fetchval("""insert into jarvis_context_payloads(owner_id,availability,content)
            values($1,'available',$2::jsonb) returning id""", owner, encoded(content))
        if dependencies:
            await connection.executemany("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                [(owner, source, identity) for source in dict.fromkeys(dependencies)])
        return identity

    async def _publish(self, connection, owner, root, *, namespace, key, content, captured_at, dependencies):
        prior = await connection.fetchrow("""select id,status,current_revision from jarvis_knowledge_sources
            where owner_id=$1 and area_id=$2 and namespace=$3 and source_key=$4 for update""",
            owner, root["area_id"], namespace, key)
        if prior is not None:
            if prior["status"] != "active":
                raise SourceError("project_source_suppressed", "A deleted or forgotten project source cannot be republished.")
            current = await self.sources._read(connection, owner, root["area_id"], prior["id"])
            if current.content == content:
                return current
        payload = await self._payload(connection, owner, content, dependencies)
        return await self.sources.publish_in_transaction(connection, owner, SourcePublication(
            area_id=root["area_id"], namespace=namespace, source_key=key, payload_id=payload,
            source_type="project_file" if namespace == "project:file" else "project_inspection",
            trust="external_content" if namespace == "project:file" else "tool_receipt", source_time=captured_at,
            time_basis="Mac capture of saved on-disk source; not unsaved editor state or future checkout",
            expected_revision=prior["current_revision"] if prior else 0))

    @staticmethod
    def _validate_result(request, root, response):
        result = response.result
        if result.get("status") == "unavailable":
            if response.originals or set(result) != {"status", "error"} or not isinstance(result["error"], str):
                raise SourceError("project_response_invalid", "Invalid unavailable response.")
            return
        if (result.get("status") != "available" or result.get("root_id") != str(root["local_root_id"])
                or result.get("operation") != request["operation"]):
            raise SourceError("project_response_invalid", "The result does not match the leased project read.")
        originals = {original.path: original for original in response.originals}
        if request["operation"] == "read":
            source = result.get("source", {})
            original = originals.get(source.get("path"))
            start = request.get("offset", 0)
            if (original is None or len(originals) != 1 or original.path != request.get("path")
                    or source.get("content_digest") != original.content_digest
                    or result.get("content") != original.text[start:start + request.get("max_chars", 16000)]):
                raise SourceError("project_response_invalid", "The read page is not bound to its actual original.")
        elif request["operation"] in {"search", "symbols", "definition", "references"}:
            items = result.get("items", [])
            if not isinstance(items, list) or len(items) > request.get("limit", 30):
                raise SourceError("project_response_invalid", "The result exceeds the requested bound.")
            for item in items:
                original = originals.get(item.get("path"))
                if original is None or original.content_digest != item.get("content_digest"):
                    raise SourceError("project_response_invalid", "A navigation result has no matching original.")
            if set(originals) != {item["path"] for item in items}:
                raise SourceError("project_response_invalid", "Unrequested originals were supplied.")
        elif originals:
            raise SourceError("project_response_invalid", "This metadata operation does not request full source text.")

    async def complete(self, token, response: Completion):
        owner, connector_id = await self._credential(token)
        response_digest = digest(response.model_dump(mode="json"))
        async with self.sources.transaction(owner) as connection:
            connector = await self._active_connector(connection, owner, connector_id)
            request = await connection.fetchrow("""select q.* from jarvis_project_requests q
                join jarvis_project_roots r on r.owner_id=q.owner_id and r.id=q.root_id
                where q.owner_id=$1 and q.id=$2 and r.connector_id=$3 for update of q""",
                owner, response.request_id, connector_id)
            if request is None:
                raise SourceError("project_request_unavailable", "No request exists for this connector.", 404)
            root = await self._root(connection, owner, connector["area_id"], request["root_id"])
            if request["status"] == "completed":
                if request["lease_id"] != response.lease_id or request["response_digest"] != response_digest:
                    raise SourceError("project_response_conflict", "This request already accepted a different response.")
                await self.sources._read(connection, owner, root["area_id"], request["source_id"])
                return {"request_id": str(request["id"]), "accepted": True, "replayed": True}
            valid = await connection.fetchval("""select status='leased' and lease_id=$2
                and lease_until>clock_timestamp() and deadline>clock_timestamp()
                from jarvis_project_requests where id=$1""", request["id"], response.lease_id)
            if not valid:
                raise SourceError("project_lease_expired", "This read lease is no longer current.")
            request_content = await connection.fetchval("""select content from jarvis_context_payloads
                where owner_id=$1 and id=$2 and availability='available'""", owner, request["request_payload_id"])
            if request_content is None:
                raise SourceError("project_request_unavailable", "The original project request was removed.")
            read = (json.loads(request_content) if isinstance(request_content, str) else request_content)["request"]
            self._validate_result(read, root, response)
            originals = []
            for original in response.originals:
                content = {"kind": "project_file", "root_id": str(root["id"]), "root_label": root["label"],
                           "path": original.path, "text": original.text, "file_digest": original.content_digest,
                           "git": response.result.get("git", response.result.get("snapshot", {}).get("git", {}))}
                source = await self._publish(connection, owner, root, namespace="project:file",
                    key=str(root["id"]) + ":" + sha256(original.path.encode()).hexdigest(), content=content,
                    captured_at=response.captured_at, dependencies=(root["authorization_payload_id"],))
                originals.append(metadata(source))
                await connection.execute("""insert into jarvis_project_file_watches(owner_id,root_id,source_id)
                    values($1,$2,$3) on conflict(owner_id,root_id,source_id)
                    do update set checked_at=clock_timestamp()""", owner, root["id"], source.source_id)
            if not originals:
                unavailable = await self._record_file_unavailable(connection, owner, root, read, response)
                if unavailable is not None:
                    originals.append(metadata(unavailable))
            if request["purpose"] == "refresh":
                await self._complete_refresh(connection, owner, root, request, read, response, originals, response_digest)
                return {"request_id": str(request["id"]), "accepted": True, "replayed": False}
            result = {"kind": "project_inspection", "request_id": str(request["id"]),
                      "root_id": str(root["id"]), "captured_at": response.captured_at.isoformat(),
                      "result": (response.result | {"root_id": str(root["id"])}
                                 if response.result.get("status") == "available" else response.result),
                      "originals": originals,
                      "meaning": "Captured on-disk evidence, not continuous visibility, execution or instructions."}
            async def materialize(conn, payloads):
                source = await self._publish(conn, owner, root, namespace="project:inspection",
                    key=str(request["id"]), content=result, captured_at=response.captured_at,
                    dependencies=(root["authorization_payload_id"], *payloads))
                return source, source.payload_id
            receipt = await self.provenance.bind_in_transaction(
                connection, owner, root["area_id"], originals, materialize)
            await connection.execute("""update jarvis_project_requests set status='completed',
                source_id=$2,response_digest=$3 where id=$1""", request["id"], receipt.source_id, response_digest)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", owner, request["request_payload_id"])
            return {"request_id": str(request["id"]), "accepted": True, "replayed": False}

    async def _record_file_unavailable(self, connection, owner, root, read, response):
        if read["operation"] != "read" or response.result.get("error") not in {
                "source_unavailable", "excluded_path", "excluded_file_type", "file_too_large",
                "binary_or_non_utf8", "source_storage_limit", "response_too_large", "source_not_in_snapshot"}:
            return None
        key = str(root["id"]) + ":" + sha256(read["path"].encode()).hexdigest()
        # Do not create phantom files or undo explicit forgetting while recording an access failure.
        exists = await connection.fetchval("""select true from jarvis_knowledge_sources
            where owner_id=$1 and area_id=$2 and namespace='project:file' and source_key=$3 and status='active'""",
            owner, root["area_id"], key)
        if not exists:
            return None
        return await self._publish(connection, owner, root, namespace="project:file", key=key,
            content={"kind": "project_file", "root_id": str(root["id"]), "root_label": root["label"],
                     "path": read["path"], "availability": "unavailable", "reason": response.result["error"],
                     "meaning": "The connector could not read this file; no current code or deletion is inferred."},
            captured_at=response.captured_at, dependencies=(root["authorization_payload_id"],))

    async def _complete_refresh(self, connection, owner, root, request, read, response, originals, response_digest):
        source_id = UUID(originals[0]["source_id"]) if originals else None
        if source_id is None:
            # Offline/replaced roots and transient analysis failures never overwrite known code.
            await connection.execute("update jarvis_project_requests set status='expired' where id=$1", request["id"])
        if source_id is not None:
            await connection.execute("""update jarvis_project_requests set status='completed',
                source_id=$2,response_digest=$3 where id=$1""", request["id"], source_id, response_digest)
        await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
            where owner_id=$1 and id=$2""", owner, request["request_payload_id"])

    async def result(self, owner_id, area_id, request_id, *, task_evidence=None):
        async with self.sources.transaction(owner_id) as connection:
            request = await connection.fetchrow("""select * from jarvis_project_requests
                where owner_id=$1 and area_id=$2 and id=$3 for update""", owner_id, area_id, request_id)
            if request is None:
                raise SourceError("project_request_unavailable", "No project request exists in this scope.", 404)
            if request["purpose"] != "inspection":
                raise SourceError("project_request_unavailable", "Background refresh is not a user inspection result.", 404)
            await self._root(connection, owner_id, area_id, request["root_id"])
            if request["status"] != "completed":
                if request["status"] in {"expired", "revoked"} or request["deadline"] <= datetime.now(UTC):
                    raise SourceError("project_request_expired", "This read expired or was revoked; no repository state was inferred.")
                return {"request_id": str(request_id), "project_status": "pending"}
            source = await self.sources._read(connection, owner_id, area_id, request["source_id"])
            # Validate all original revisions under the same owner transaction.
            originals = source.content["originals"]
            for original in originals:
                await self.provenance._resolve(connection, owner_id, area_id,
                    MemorySourceReference.from_metadata(original))
            evidence = [*originals, metadata(source)]
            if task_evidence is not None:
                citations = await task_evidence.bind_in_transaction(connection, evidence)
                evidence = [item | {"task_evidence": citation} for item, citation in zip(evidence, citations, strict=True)]
            return {"request_id": str(request_id), "project_status": "complete",
                    "inspection": source.content, "sources": evidence, "provider_contacted": False}
