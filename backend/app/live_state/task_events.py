"""Best-effort task-event hints on the existing canonical refresh worker.

No polling task or durable authority is created here. Per-task sequence cursors
avoid assuming event timestamps reflect transaction commit order. Consumers
reread their domain receipt; reconnecting clients still read current state.
"""
import asyncio
import logging

LOGGER = logging.getLogger(__name__)


class TaskEventObserver:
    def __init__(self, repository, owner_id, consume, *, event_types=("action_accepted",)):
        self.repository, self.owner_id, self.consume = repository, owner_id, consume
        if (not isinstance(event_types, (tuple, list, set, frozenset)) or not event_types
                or any(not isinstance(kind, str) or not 1 <= len(kind) <= 80 for kind in event_types)):
            raise ValueError("Task event consumers require explicit bounded event types")
        self.event_types = frozenset(event_types)
        self._cursors = {}
        self._lock = asyncio.Lock()

    async def refresh(self):
        async with self._lock:
            async with self.repository.pool.acquire() as connection:
                heads = await connection.fetch("""select id,last_event_sequence
                    from jarvis_background_tasks where owner_id=$1
                    order by updated_at desc,id limit 32""", self.owner_id)
            current = {row["id"] for row in heads}
            self._cursors = {key: value for key, value in self._cursors.items() if key in current}
            pages = 0
            for head in heads:
                task_id = head["id"]
                after = self._cursors.get(task_id, 0)
                if head["last_event_sequence"] <= after:
                    continue
                if pages >= 4:
                    break
                pages += 1
                try:
                    events = await self.repository.events(self.owner_id, task_id, after=after, limit=16)
                    for event in events:
                        if event["sequence"] <= self._cursors.get(task_id, 0):
                            continue
                        if event["event_type"] in self.event_types:
                            await self.consume(event["id"])
                        self._cursors[task_id] = event["sequence"]
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    # Failed events remain unconsumed for the next refresh;
                    # other tasks can still make progress.
                    LOGGER.warning("Task event hint unavailable error_type=%s", type(error).__name__)
