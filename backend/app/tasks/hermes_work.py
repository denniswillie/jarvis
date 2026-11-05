"""Worker-side mailbox pump; no audio callbacks or interpretation model."""
import asyncio
import json

from app.tasks.models import ClaimLost, TaskError


class HermesWorkSession:
    def __init__(self, context, mailbox, *, poll_seconds=0.1):
        self.context, self.mailbox = context, mailbox
        self.poll_seconds = poll_seconds
        self.cursor = 0
        self.roots = set()
        self.initial = []
        self.pending_progress = None
        self.wake = asyncio.Event()

    async def prepare(self):
        # The ordinary manifest already supplies canonical conversation history.
        # This adds exact work-specific follow-ups, not a generated summary.
        for _ in range(8):
            messages = await self.mailbox.read_inputs(self.context.claim, after=self.cursor, limit=64)
            for message in messages:
                self.initial.append({"sequence": message.sequence, "input": message.content})
                self.roots.add(message.payload_id)
                self.cursor = message.sequence
            if len(messages) < 64:
                history = await self.mailbox.history(self.context.claim) if callable(getattr(self.mailbox, "history", None)) else []
                from uuid import UUID
                self.roots.update(UUID(item["payload_id"]) for item in history)
                return {"messages": self.initial, "earlier_tool_receipts_and_results": history,
                    "meaning": "Conversation for this same work. Earlier messages are history, not requests to repeat effects. "
                               "Original user input is primary; live observation windows are not finalized turns."}
        raise TaskError("work_history_transport_limit", "This work history exceeds the bounded native handoff.")

    async def offer_progress(self, text):
        # Called by the stdout reader: never wait for storage or model work here.
        # Intermediate status is coalescible; results use the existing result path.
        self.pending_progress = text
        self.wake.set()

    async def guard(self, connection):
        roots = frozenset(self.roots)
        if not roots:
            return
        rows = await connection.fetch("""select id,availability from jarvis_context_payloads
            where owner_id=$1 and id=any($2::uuid[])""", self.context.task.owner_id, list(roots))
        if {row["id"] for row in rows if row["availability"] == "available"} != roots:
            raise ClaimLost()

    async def pump(self, run):
        while True:
            self.wake.clear()
            messages = await self.mailbox.read_inputs(self.context.claim, after=self.cursor, limit=64)
            for message in messages:
                # Register the source before it can enter another provider request.
                self.roots.add(message.payload_id)
                text = json.dumps({"work_followup": message.content, "sequence": message.sequence},
                                  ensure_ascii=False, separators=(",", ":"))
                try:
                    await run.steer(text)
                except TaskError as error:
                    if error.code == "hermes_input_run_ended":
                        return
                    raise
                self.cursor = message.sequence
            progress = self.pending_progress
            if progress is not None:
                self.pending_progress = None
                await self.mailbox.publish_progress(self.context.claim, progress,
                                                    source_payload_ids=tuple(self.roots))
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass
