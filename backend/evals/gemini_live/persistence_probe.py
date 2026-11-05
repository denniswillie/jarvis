"""Opt-in committed message/evidence checks using disposable tracked owners.

The database credential arrives on stdin. No model API, production owner, or
production deployment is used. Cleanup runs even when an assertion fails.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import logging
from pathlib import Path
import sys
from uuid import uuid4

import asyncpg

from app.experience.repository import ExperienceRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.memory.consolidation.service import ExperienceConsolidationService
from app.perception.repository import PerceptionRepository
from app.realtime.gemini.persistence import FinalTranscript, GeminiConversationJournal
from app.sessions.repository import SessionRepository
from evals.gemini_live.interruption import DatabaseFixtures


class LostInsertAcknowledgement(SessionRepository):
    lost = False

    async def create_or_reuse_gemini_message(self, **kwargs):
        message = await super().create_or_reuse_gemini_message(**kwargs)
        if not self.lost:
            self.lost = True
            raise ConnectionError("Controlled insert acknowledgement loss")
        return message


class InterruptedAttachment:
    def __init__(self, service, *, after_commit):
        self.service, self.after_commit, self.failed = service, after_commit, False

    async def attach_message(self, **kwargs):
        if not self.failed:
            self.failed = True
            if self.after_commit:
                await self.service.attach_message(**kwargs)
            raise ConnectionError("Controlled attachment interruption")
        return await self.service.attach_message(**kwargs)


class NoModelCalls:
    provider = "fixture"
    model = "not_called"

    async def generate(self, *args, **kwargs):
        raise AssertionError("This probe must not run uncontrolled consolidation")

    async def close(self):
        pass


async def run(args, keys):
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = DatabaseFixtures(keys["database"], args.output)
    report = {"cases": [], "passed": False, "model_api_used": False}
    stage = "connect"
    services = []
    try:
        await fixtures.connect()
        async with asyncio.timeout(180):
            stage = "concurrent_message_create_or_reuse"
            owner, session_id = await fixtures.create(stage)
            repository = SessionRepository(fixtures.pool)
            run_id = str(uuid4())
            values = dict(
                owner_id=owner, session_id=session_id, audio_session_id=run_id,
                logical_item_id="turn:1:user", role="user", content="I prefer morning walks.",
                occurred_at=datetime.now(UTC), provider_item_id="native-input-one",
                metadata={"completion_state": "finalized", "experience_eligible": True},
            )
            replayed = await asyncio.gather(*(
                repository.create_or_reuse_gemini_message(**values) for _ in range(8)
            ))
            assert len({message.id for message in replayed}) == 1
            repeated = await repository.create_or_reuse_gemini_message(**{
                **values, "logical_item_id": "turn:2:user", "provider_item_id": "native-input-two",
            })
            assert repeated.id != replayed[0].id
            foreign_owner, _ = await fixtures.create("foreign_message_owner")
            try:
                await repository.create_or_reuse_gemini_message(**{**values, "owner_id": foreign_owner})
            except ValueError:
                pass
            else:
                raise AssertionError("Foreign owner message insertion was accepted")
            report["cases"].append({"name": stage, "same_item_replays": 8, "distinct_rows": 2,
                                    "foreign_owner_rejected": True, "passed": True})

            for mode in ("ordinary", "insert_ack_loss", "before_attachment", "attachment_ack_loss", "late_input"):
                stage = mode
                owner, session_id = await fixtures.create(mode)
                repository = (LostInsertAcknowledgement(fixtures.pool) if mode == "insert_ack_loss"
                              else SessionRepository(fixtures.pool))
                service = ExperienceEventService(
                    repository=ExperienceRepository(fixtures.pool),
                    segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300, 900)),
                )
                services.append(service)
                experience = (InterruptedAttachment(service, after_commit=mode == "attachment_ack_loss")
                              if mode in {"before_attachment", "attachment_ack_loss"} else service)
                errors = []

                async def failed(code):
                    errors.append(code)

                run_id = str(uuid4())

                def journal():
                    return GeminiConversationJournal(
                        repository=repository, experience_service=experience,
                        owner_id=owner, session_id=session_id, audio_session_id=run_id,
                        on_failure=failed,
                    )

                writer = journal()
                transcript = FinalTranscript(
                    logical_item_id="turn:1:user", role="user",
                    content="I work best in the morning and prefer to plan my day before breakfast.",
                    occurred_at=datetime.now(UTC), provider_item_id="original-input",
                    metadata={"completion_state": "finalized", "experience_eligible": True},
                )
                expected_rows = 1
                if mode == "late_input":
                    earlier_speech_time = datetime.now(UTC) - timedelta(seconds=30)
                    writer.submit(FinalTranscript(
                        logical_item_id="turn:1:assistant", role="assistant", content="We can plan together.",
                        occurred_at=datetime.now(UTC), provider_item_id="first-answer",
                        metadata={"experience_eligible": True, "completion_state": "completed",
                                  "delivery_status": "forwarded", "playback_status": "unconfirmed"},
                    ))
                    await writer.drain(timeout_seconds=12)
                    writer = journal()
                    transcript = replace(transcript, occurred_at=datetime.now(UTC), metadata={
                        **transcript.metadata, "input_cutoff_at": earlier_speech_time.isoformat(),
                        "segment_id": "original-segment", "arrived_after_assistant": True,
                    })
                    expected_rows = 2
                writer.submit(transcript)
                writer.submit(transcript)  # Duplicate final event, not another utterance.
                await writer.drain(timeout_seconds=12)
                assert not errors and not writer.diagnostics()["degraded"]

                # Rebuild the journal as a recovery replay, after closing the first
                # experience event. Readback must find its existing evidence link.
                event = await service.repository.get_open_event(owner_id=owner, session_id=session_id)
                assert event is not None
                await service._close_event(event, ended_at=datetime.now(UTC), reason="fixture_recovery_boundary")
                # Follow the ordinary, shared consolidation reader to its actual
                # cited model input. No invented memory item or model extraction
                # substitutes for the source message. Long-term promotion remains
                # the existing downstream policy, not a promise made by this probe.
                consolidation = ExperienceConsolidationService(
                    experience_repository=service.repository,
                    perception_repository=PerceptionRepository(fixtures.pool),
                    client=NoModelCalls(), image_loader=None,
                    owner_working_memory_reader=None, active_reminder_reader=None,
                    max_keyframes=1, max_transcripts=10,
                )
                request = await consolidation.build_request(
                    owner_id=owner, session_id=session_id, event_id=event.id,
                )
                candidates = [item for item in request.evidence
                              if item.evidence_type == "message" and item.payload.get("role") == "user"
                              and item.payload.get("content") == transcript.content]
                assert len(candidates) == 1 and candidates[0].citation_allowed
                assert candidates[0].payload["metadata"]["audio_provider"] == "gemini"
                retrieved = await repository.get_recent_messages(owner, session_id, limit=10)
                assert any(message.id == candidates[0].evidence_ref_id and message.content == transcript.content
                           for message in retrieved)
                replay = journal()
                replay.submit(transcript)
                await replay.drain(timeout_seconds=12)
                assert not errors and not replay.diagnostics()["degraded"]

                connection = await asyncpg.connect(keys["database"], timeout=15, command_timeout=20)
                try:
                    rows = await connection.fetch(
                        "select id, role, metadata from jarvis_messages where owner_id=$1 and session_id=$2",
                        owner, session_id,
                    )
                    evidence = await connection.fetch(
                        """select evidence.evidence_ref_id, evidence.metadata
                        from experience_event_evidence evidence
                        join experience_events events on events.id=evidence.event_id
                        where events.owner_id=$1 and events.session_id=$2 and evidence.evidence_type='message'""",
                        owner, session_id,
                    )
                    memories = await connection.fetchval(
                        "select count(*) from memory_items where owner_id=$1 and session_id=$2", owner, session_id,
                    )
                    assert len(rows) == expected_rows
                    assert len(evidence) == expected_rows
                    assert len({record["evidence_ref_id"] for record in evidence}) == expected_rows
                    assert memories == 0  # No explicit save tool or uncontrolled consolidation.
                finally:
                    await connection.close()
                report["cases"].append({"name": mode, "message_rows": len(rows), "evidence_links": len(evidence),
                                        "shared_consolidation_input_verified": True, "shared_message_retrieval_verified": True,
                                        "explicit_memory_writes": memories, "recovery_replay": True, "passed": True})
            report["passed"] = True
    except Exception as error:
        report.update({"failed_stage": stage, "error_type": type(error).__name__})
    finally:
        for service in services:
            await service.close()
        try:
            await fixtures.cleanup()
            report["cleanup"] = fixtures.cleanup_result
        except Exception as error:
            report.update({"passed": False, "cleanup_error_type": type(error).__name__})
        (args.output / "persistence-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(arguments, json.load(sys.stdin))))
