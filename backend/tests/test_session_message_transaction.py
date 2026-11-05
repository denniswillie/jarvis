from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.sessions.repository import SessionRepository


class NoNestedPool:
    def acquire(self):
        raise AssertionError("A composed write must not acquire a second pool connection")


@pytest.mark.asyncio
async def test_message_insert_and_session_touch_reuse_the_guarded_connection():
    session_id, message_id = uuid4(), uuid4()
    row = dict(id=message_id, session_id=session_id, role="assistant", modality="audio",
        content="Synthetic fixture.", provider_item_id=None, provider_response_id=None,
        created_at=datetime.now(UTC), metadata={})

    class Connection:
        fetchrow = AsyncMock(return_value=row)
        execute = AsyncMock()

        def is_in_transaction(self):
            return True

    connection = Connection()
    message = await SessionRepository(NoNestedPool()).create_message(connection=connection,
        owner_id="fixture", session_id=session_id, role="assistant", modality="audio", content="Synthetic fixture.")
    assert message.id == message_id
    connection.fetchrow.assert_awaited_once()
    assert "source_payload_id" not in connection.fetchrow.await_args.args[0]
    connection.execute.assert_awaited_once()
    assert "update jarvis_sessions" in connection.execute.await_args.args[0]


@pytest.mark.asyncio
async def test_composed_message_write_requires_a_transaction():
    class Connection:
        def is_in_transaction(self):
            return False

    with pytest.raises(RuntimeError, match="active transaction"):
        await SessionRepository(NoNestedPool()).create_message(connection=Connection(),
            owner_id="fixture", session_id=uuid4(), role="assistant", modality="audio", content="Synthetic fixture.")
