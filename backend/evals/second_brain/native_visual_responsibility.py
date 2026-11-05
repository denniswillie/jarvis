"""Native visual cue -> owned attention inbox; receipt is explicitly simulated.

Called after a real foreground conversation teaches the duty. Image pixels are
controlled, not a glasses capture. Gemini output and producer routing are real.
F1 separately qualifies actual TTS and simulator playback; this does not replace it.
"""
from io import BytesIO
from types import SimpleNamespace
from uuid import UUID

from PIL import Image, ImageDraw, ImageFont

from app.attention.repository import AttentionRepository
from app.attention.tts import OpenAISpeechProvider
from app.knowledge.observations import ObservationPublication
from app.knowledge.publication import CanonicalSourcePublication
from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.perception.fast_analyzer import FastPerceptionAnalyzer
from app.reminders.duties import ResponsibilityRepository
from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from app.visual_memory.gemini_client import GeminiVisualAnalyzerClient, gemini_visual_config_from_settings
from evals.second_brain.stored_image_fixture import stored_image_fixture
from evals.second_brain.visual_observation_fixture import run_visual_fixture


async def qualify_visual_duty(fixture, settings, session_id, root, duty):
    config = gemini_visual_config_from_settings(settings)
    if config is None:
        raise RuntimeError("Native visual model is not configured")
    client = GeminiVisualAnalyzerClient(config)
    analyzer = FastPerceptionAnalyzer(GeminiFastVisualModel(client))
    tasks = TaskRepository(fixture.pool, TaskLimits())
    context = ContextRepository(tasks, publication=CanonicalSourcePublication(fixture.sources, fixture.area))
    publisher = ObservationPublication(context, owner_id=fixture.owner)
    inbox = AttentionRepository(tasks)
    reminder_id = UUID(duty["occurrences"][0]["reminder_id"])
    image = Image.new("RGB", (1000, 480), "white")
    ImageDraw.Draw(image).multiline_text((30, 35),
        "def binary_search(values, target):\n    low = 0\n    high = len(values) - 1\n"
        "    while low <= high:\n        mid = (low + high) // 2",
        font=ImageFont.load_default(size=34), fill="black", spacing=18)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    pixels = buffer.getvalue()
    (root/"visual-cue.jpg").write_bytes(pixels)
    result = {"provider":analyzer.provider, "model":analyzer.model, "analyses":[],
              "scope":"actual Gemini pixels and production pipeline; controlled image, simulated playback receipt"}
    try:
        for index in range(2):
            # Upload each repeat just before analysis so freshness is not fabricated.
            async with stored_image_fixture(fixture, root, settings, session_id=session_id,
                                             images=[pixels]) as frames:
                trace = await run_visual_fixture(fixture, publisher, frames[0].id, "",
                    native_analyzer=analyzer, image_bytes=pixels, settings=settings,
                    attention_repository=inbox)
                result["analyses"].append({
                    "frame_id":str(frames[0].id),
                    "outputs":[output.model_dump(mode="json") for output in trace["outputs"]],
                    "active_reminder_counts":[len(request.active_reminders) for request in trace["requests"]],
                    "diagnostics":trace["diagnostics"],
                    "legacy_notifications":trace["notifications"],
                })
                if index == 0:
                    first = await inbox.list(fixture.owner)
                    result["first_inbox"] = first
                    claim = await inbox.claim_next(fixture.owner, session_id, "native-visual-fixture")
                    result["simulated_receipt_accepted"] = False
                    if claim is not None:
                        provider = OpenAISpeechProvider(SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace()))
                        requests = await inbox.prepare(claim, provider)
                        if len(requests) != 1:
                            raise RuntimeError("Controlled notification unexpectedly needs several speech chunks")
                        await inbox.submit_chunk(claim, 0, requests[0])
                        async with inbox.write_guard(claim, 0, start_sample=0, samples=480) as mark_sent:
                            mark_sent()
                        await inbox.chunk_generated(claim, 0, samples=480, first_bytes_ms=1,
                                                     provider_request_id="simulated-not-native-tts")
                        await inbox.generation_finished(claim)
                        result["simulated_receipt_accepted"] = await inbox.playback_receipt(
                            claim, item_id=f"tts_{claim.attempt_id.hex}_0",
                            response_id=f"tts_{claim.attempt_id.hex}", samples=480, completed=True)
                        await inbox.reconcile_notifications(fixture.owner)
        async with fixture.pool.acquire() as connection:
            intents = await connection.fetch(
                "select source_namespace,state from jarvis_attention_intents where owner_id=$1", fixture.owner)
            observations = await connection.fetchval(
                "select count(*) from jarvis_observation_sources where owner_id=$1", fixture.owner)
        stored = await ResponsibilityRepository(fixture.pool).list(fixture.owner, include_inactive=True)
        analyses = result["analyses"]
        matches = [match for output in analyses[0]["outputs"] for match in output["prospective_matches"]]
        result["final_inbox"] = [dict(row) for row in intents]
        result["checks"] = {
            "two_actual_visual_analyses":all(len(row["outputs"])==1 for row in analyses) and len(analyses)==2,
            "native_match_uses_taught_reminder":any(
                match["subject_type"]=="conditional_reminder" and match["subject_id"]==str(reminder_id)
                for match in matches),
            "observations_have_canonical_sources":observations==2,
            "one_notification_no_competing_route":len(intents)==1 and intents[0]["source_namespace"]=="prospective_event",
            "simulated_receipt_accepted":result["simulated_receipt_accepted"],
            "repeat_does_not_reactivate_cue":analyses[1]["active_reminder_counts"]==[0],
            "notification_does_not_claim_action_completion":len(stored["duties"])==1 and
                stored["duties"][0]["occurrences"][0]["report_state"]=="unknown",
        }
        return result
    finally:
        await client.close()
