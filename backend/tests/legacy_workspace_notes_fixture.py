"""Thin Workspace service; no new task engine, content store or model loop."""
from __future__ import annotations

from app.notes.workspace_content import parse_command
from app.notes.workspace_repository import WorkspaceNoteRepository


class WorkspaceNoteService:
    def __init__(self, repository: WorkspaceNoteRepository):
        self.repository = repository

    async def read_account_note(self, ctx, version_id=None):
        snapshot = await self.read_account_snapshot(ctx, version_id)
        return snapshot.view().model_dump(mode="json") if snapshot is not None else None

    async def read_current_account_note(self, owner_id):
        snapshot = await self.repository.read_current_account_note(owner_id)
        return snapshot.view().model_dump(mode="json") if snapshot is not None else None

    async def read_account_snapshot(self, ctx, version_id=None):
        return await self.repository.read_account_note(ctx, version_id)

    async def read_account_snapshot_in_transaction(self, connection, ctx, version_id=None):
        return await self.repository.read_account_note_in_transaction(connection, ctx, version_id)

    async def publish_account_edit(self, ctx, base_version_id, changes, publication_guard, *, receipt_recorder=None):
        return await self.repository.publish_account_edit(ctx, base_version_id, changes, publication_guard,
                                                          receipt_recorder=receipt_recorder)

    async def create_note(self, ctx, title, content):
        return await self.repository.mutate(ctx, parse_command({"action": "create", "title": title, "content": content}))

    async def patch_note(self, ctx, artifact_id, base_version_id, changes):
        return await self.repository.mutate(ctx, parse_command({"action": "patch", "artifact_id": artifact_id,
            "base_version_id": base_version_id, "changes": changes}))

    async def restore_note(self, ctx, artifact_id, base_version_id, restore_version_id):
        return await self.repository.mutate(ctx, parse_command({"action": "restore", "artifact_id": artifact_id,
            "base_version_id": base_version_id, "restore_version_id": restore_version_id}))

    async def delete_note(self, ctx, artifact_id, base_version_id):
        return await self.repository.mutate(ctx, parse_command({"action": "delete", "artifact_id": artifact_id,
            "base_version_id": base_version_id}))

    async def execute_note(self, ctx, arguments):
        return await self.repository.mutate(ctx, parse_command(arguments))

    async def read_note(self, ctx, artifact_id, version_id=None):
        return (await self.repository.read_note(ctx, artifact_id, version_id)).view().model_dump(mode="json")

    async def read_snapshot(self, ctx, artifact_id, version_id=None):
        """Trusted export composition gets exact source provenance, not wire data."""
        return await self.repository.read_note(ctx, artifact_id, version_id)

    async def read_snapshot_in_transaction(self, connection, ctx, artifact_id, version_id=None):
        """Revalidate exact evidence inside the caller's existing source guard."""
        return await self.repository.read_note_in_transaction(connection, ctx, artifact_id, version_id)

    async def list_notes(self, ctx, query=None, limit=8, cursor=None):
        return (await self.repository.list_notes(ctx, query, limit, cursor)).model_dump(mode="json")

    async def list_versions(self, ctx, artifact_id, limit=8, cursor=None):
        return (await self.repository.list_versions(ctx, artifact_id, limit, cursor)).model_dump(mode="json")

    async def operation_receipt(self, ctx, operation_id):
        return await self.repository.operation_receipt(ctx, operation_id)
