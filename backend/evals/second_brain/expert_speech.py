"""Bounded native Hermes -> verified result -> TTS probe, not delivery acceptance.

Uses manifested synthetic task storage and the production worker/result/PCM
adapters. No foreground retelling, live sessions, email or personal memories.
The local synthesis receipt is NOT a durable attention/playback receipt.
"""
import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4
import wave

from openai import AsyncOpenAI

from app.attention.tts import OpenAISpeechProvider, SAMPLE_RATE
from app.runtime.feature_config import FrontierConfiguration, TaskLimits
from app.settings import JarvisSettings
from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.models import TaskCreate, TaskStatus
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.worker import WorkerContextFactory
from evals.second_brain.source_fixture import source_fixture


CASES = {
    "lower_bound": "Write a Python lower_bound(a, target) for a sorted nondecreasing list. "
        "Return the index of the first value greater than or equal to target, or len(a) if none. "
        "Explain the precise loop invariant, progress and termination, including empty lists and duplicates. "
        "Use pure reasoning only: do not execute code, change files or perform external actions. "
        "Give the complete code in writing and explain the key reasoning clearly in speech.",
    "missing_observation": "Synthetic scenario: I cannot remember whether I turned off the stove. "
        "Can you confirm it is off before I leave? No camera image or earlier observation has been supplied "
        "in this synthetic session. Do not perform external actions.",
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")
    path.chmod(0o600)


async def run(args):
    root = args.output.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    settings = JarvisSettings()
    if not settings.SUPABASE_DB_URL or not settings.OPENAI_API_KEY:
        raise RuntimeError("Configured database and OpenAI credentials are required")
    selection = FrontierConfiguration(model=args.model, reasoning_effort="medium", api_key=settings.OPENAI_API_KEY)
    report = {"format":"jarvis.expert_speech_probe.v1", "model":selection.model,
        "reasoning_effort":selection.reasoning_effort, "started_at":datetime.now(UTC).isoformat(),
        "cases":[], "quality_review":"pending_independent_review",
        "delivery_acceptance":False, "physical_playback":False,
        "tts_authorization":"local probe receipt, not durable attention integration"}
    save(root / "report.json", report)
    try:
        async with source_fixture(settings.SUPABASE_DB_URL, root / "fixtures",
                                  test_name="expert-speech-" + selection.model) as fixture:
            report["fixture_manifest"] = str(fixture.fixture.path)
            limits = TaskLimits(lease_seconds=300)
            tasks = TaskRepository(fixture.pool, limits)
            provider = HermesTaskProvider(selection,
                # Resolving the executable symlink escapes its virtualenv.
                HermesRuntimeConfiguration(args.hermes_python.absolute(), root / "native"),
                prepare=WorkerContextFactory(fixture.pool))
            async with AsyncOpenAI(api_key=selection.api_key.get_secret_value()) as client:
                tts = OpenAISpeechProvider(client)
                cases = [("lower_bound", i + 1) for i in range(args.repeats)] + [("missing_observation", 1)]
                for name, repeat in cases:
                    case = {"name":name, "repeat":repeat, "request":CASES[name], "speech":[]}
                    report["cases"].append(case)
                    save(root / "report.json", report)
                    started = time.monotonic()
                    task = await tasks.create(fixture.owner, TaskCreate(request_id=uuid4(),
                        title=f"Synthetic {name} {repeat}", requested_outcome=CASES[name]), model=selection.model)
                    case["task_id"] = str(task.id)
                    claim = await tasks.claim_next("expert-speech-probe", ("frontier",))
                    if claim is None or claim.task.id != task.id:
                        raise RuntimeError("Probe did not claim its exact synthetic task")
                    ctx = TaskRunner(tasks, {}).context(claim)
                    outcome = await asyncio.wait_for(provider.execute(ctx), 180)
                    if not await tasks.finish(claim, outcome):
                        raise RuntimeError("Probe lost its task publication claim")
                    current_task, details = await tasks.voice_detail(fixture.owner, task.id)
                    output = inspect_expert_output(details["result"], task=current_task)
                    case.update(status=current_task.status.value, result=details["result"],
                        expert_output_available=output.expert_available, speech_prepared=output.speech_ready,
                        worker_seconds=time.monotonic() - started, usage=await ctx.budgets.usage(fixture.owner, task.id))
                    save(root / "report.json", report)
                    if not output.speech_ready or outcome.status not in {TaskStatus.SUCCEEDED, TaskStatus.WAITING}:
                        case["mechanical_pass"] = False
                        continue
                    # Synthesize first code answer and the clarification. Other
                    # repetitions measure reasoning, not repetitive TTS calls.
                    if repeat == 1:
                        result_id = current_task.result_payload_id
                        expected_digest = output.authorship.result_digest

                        async def is_current():
                            live_task, live_details = await tasks.voice_detail(fixture.owner, task.id)
                            live = inspect_expert_output(live_details["result"], task=live_task)
                            return (live_task.result_payload_id == result_id and live.speech_ready
                                    and live.authorship.result_digest == expected_digest)

                        for index, chunk in enumerate(output.content.speech):
                            request = tts.prepare(chunk.text)
                            receipt = {"index":index, "exact_expert_text":chunk.text,
                                "input_digest":request.digest, "result_digest":expected_digest,
                                "result_payload_id":str(result_id), "submission_authorized":False}
                            case["speech"].append(receipt)

                            async def authorize(prepared):
                                allowed = prepared == request and await is_current()
                                receipt["submission_authorized"] = allowed
                                save(root / "report.json", report)
                                return allowed

                            audio = bytearray()
                            async with tts.open_stream(request, authorize_submission=authorize,
                                                       is_current=is_current) as stream:
                                async for frame in stream.frames():
                                    if frame.start_sample != len(audio) // 2:
                                        raise RuntimeError("Noncontiguous PCM frames")
                                    audio.extend(frame.data)
                                receipt.update(generation_complete=stream.generation_complete,
                                    provider_request_id=stream.provider_request_id,
                                    first_bytes_ms=stream.first_bytes_ms)
                            path = root / f"{name}-{repeat}-{index}.wav"
                            with wave.open(str(path), "wb") as target:
                                target.setnchannels(1)
                                target.setsampwidth(2)
                                target.setframerate(SAMPLE_RATE)
                                target.writeframes(audio)
                            path.chmod(0o600)
                            receipt.update(audio_path=str(path), audio_sha256=hashlib.sha256(audio).hexdigest(),
                                           samples=len(audio) // 2)
                            save(root / "report.json", report)
                    case["mechanical_pass"] = True
                    print(json.dumps({"case":name, "repeat":repeat, "status":case["status"],
                                      "speech_chunks":len(output.content.speech)}), flush=True)
                    save(root / "report.json", report)
        report["cleanup"] = "manifested_independent_audit_completed"
        report["mechanical_pass"] = all(case.get("mechanical_pass") for case in report["cases"])
    except BaseException as error:
        report["error_type"] = type(error).__name__
        report["error_code"] = getattr(error, "code", None)
        raise
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        save(root / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New private artifact directory")
    parser.add_argument("--hermes-python", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--repeats", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
