"""Bounded native OpenAI duty conversation; synthetic microphone, not physical acceptance."""
import argparse
import asyncio
import json
import logging
from pathlib import Path
import subprocess
from uuid import uuid4

from app.experience.repository import ExperienceRepository, ToolExecutionRepository
from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.reminders.duties import ResponsibilityRepository
from app.reminders.duty_tools import ResponsibilityToolModule
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.gemini_live.transport_probe import read_pcm, write_wav
from evals.second_brain.native_voice_tasks import (
    ObservedPhone, ObservedExecutor, answer, foreground_context_providers, wait_until,
)
from evals.second_brain.source_fixture import source_fixture

UTTERANCES = {
    "teach": "Please save three responsibilities. Remind me to do laundry every day at seven p.m. "
        "Dublin time. Remind me once tomorrow at noon to buy toilet paper. And remind me to take "
        "my umbrella when I am at the office doorway. Keep the umbrella reminder one-off.",
    "preferences": "For duty reminders, keep quiet from ten p.m. until eight a.m., Europe Dublin time. "
        "Continue to ask me before you infer a recurring obligation.",
    "preference_recall": "What duty reminder quiet hours and inference policy are currently saved?",
    "visual": "When Python code is visible, remind me to take a short break. Activate this now and keep it one-off.",
    "update": "Change only the laundry responsibility to every seven days at seven p.m. Dublin time.",
    "snooze": "Snooze the umbrella reminder for one hour. Do not change the other duties.",
    "done": "I did the laundry just now. Please mark this occurrence done.",
    "attribution": "Did you actually see me do the laundry, or are you going on what I told you?",
    "correction": "Correction: I have not done the laundry. Correct my completion report; do not send another alert now.",
    "cancel": "Cancel only the toilet paper errand.",
    "recall": "What responsibilities are still pending? Did I actually finish the laundry?",
    "unknown": "Do you know whether I took my medication, turned off the stove, and locked the door today?",
}


def prepare(directory, names):
    directory.mkdir(parents=True, mode=0o700)
    for name in names:
        text = UTTERANCES[name]
        original, aiff, wav = (directory/(name+suffix) for suffix in (".txt",".aiff",".wav"))
        original.write_text(text)
        subprocess.run(["/usr/bin/say","-r","185","-f",str(original),"-o",str(aiff)], check=True)
        subprocess.run(["/usr/bin/afconvert","-f","WAVE","-d","LEI16@24000","-c","1",str(aiff),str(wav)], check=True)
        read_pcm(wav)
        aiff.unlink()


async def run(args):
    settings = JarvisSettings()
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    scenarios = {"controls":(("teach","preferences","update","snooze","done","attribution","correction","cancel"),
                             ("recall","unknown")),
                 "preferences":(("preferences",),("preference_recall",)), "visual":(("visual",),)}
    sessions = scenarios[args.case]
    prepare(args.output/"speech", [name for names in sessions for name in names])
    result = {"format":"jarvis.focused.duties.native.v1", "case":args.case, "model":settings.OPENAI_REALTIME_MODEL,
              "answers":[], "tools":[], "sessions":[], "ok":False, "semantic_review":"pending",
              "scope":"native foreground and owned duty tools; synthetic microphone, no physical hearing"}
    try:
        async with source_fixture(settings.SUPABASE_DB_URL, args.output, test_name="focused-native-duties") as fixture:
            tasks = TaskRepository(fixture.pool, TaskLimits())
            sources = ContextRepository(tasks)
            repository = SessionRepository(fixture.pool)
            duties = ResponsibilityRepository(fixture.pool)
            experience = ExperienceEventService(repository=ExperienceRepository(fixture.pool),
                segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(300,900)))
            try:
                for names in sessions:
                    async with fixture.pool.acquire() as c:
                        identity = await c.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", fixture.owner)
                    result["sessions"].append(str(identity))
                    phone = ObservedPhone(lambda: [])
                    capture = SessionContextCapture(sources, owner_id=fixture.owner, session_id=identity,
                        provider="openai", audio_session_id=str(uuid4()))
                    session = RealtimeAudioSession(websocket=phone, context=RealtimeSessionContext(
                        audio_session_id=capture.audio_session_id, owner_id=fixture.owner,
                        jarvis_session_id=identity, client_kind="phone_audio",
                        session=await repository.get_session(fixture.owner, identity)),
                        settings=settings, repository=repository, tool_modules=(ResponsibilityToolModule(duties),),
                        context_providers=foreground_context_providers(), context_capture=capture,
                        tool_execution_repository=ToolExecutionRepository(fixture.pool),
                        experience_event_service=experience)
                    session.tool_executor = ObservedExecutor(session.tool_executor, result["tools"], phone.started)
                    job = asyncio.create_task(session.run())
                    try:
                        await wait_until(lambda: any(e["type"]=="session.created" for e in phone.events),
                                         running=(job,), timeout=30)
                        for name in names:
                            spoken = await answer(phone, args.output/"speech", name, running=(job,))
                            result["answers"].append(spoken)
                            print(json.dumps({"utterance":name,"answer":spoken["text"]}), flush=True)
                        await capture.flush()
                        result.setdefault("capture_issues",[]).append(capture.issue)
                        result.setdefault("application_errors",[]).extend(e for e in phone.events if e["type"]=="error")
                        for index,(item,pcm) in enumerate(phone.audio.items()):
                            write_wav(args.output/(str(identity)+"-"+str(index)+".wav"), bytes(pcm))
                    finally:
                        await session.close("focused_native_duties_finished")
                        try:
                            await asyncio.wait_for(job,20)
                        finally:
                            await capture.close()
                result["stored"] = await duties.list(fixture.owner, include_inactive=True)
                rows = result["stored"]["duties"]
                recurring = [r for r in rows if r["responsibility"]["spec"]["every_days"] is not None]
                situational = [r for r in rows if r["responsibility"]["spec"]["condition"]]
                once = [r for r in rows if r not in recurring and r not in situational]
                policy = result["stored"]["notification_preferences"]["policy"]
                result["checks"] = {
                    "three_rules_not_duplicate_updates":len(rows)==3,
                    "weekly_change_persisted":len(recurring)==1 and recurring[0]["responsibility"]["spec"]["every_days"]==7,
                    "correction_reopened_current_occurrence":len(recurring)==1 and
                        recurring[0]["occurrences"][0]["status"]=="open" and
                        recurring[0]["occurrences"][0]["report_state"]=="user_reported_not_done",
                    "situational_snooze_persisted":len(situational)==1 and bool(situational[0]["responsibility"]["snoozed_until"]),
                    "only_one_off_cancelled":len(once)==1 and once[0]["responsibility"]["status"]=="cancelled" and
                        all(r["responsibility"]["status"]=="active" for r in recurring+situational),
                    "quiet_preferences_persisted":bool(policy) and policy["quiet_start"]=="22:00:00" and
                        policy["quiet_end"]=="08:00:00" and policy["timezone"]=="Europe/Dublin",
                    "fresh_session":len(set(result["sessions"]))==2,
                    "capture_healthy":not any(result["capture_issues"]),
                    "no_application_errors":not result["application_errors"],
                    "all_questions_received_audio":all(r["response_ids"] for r in result["answers"]),
                }
                if args.case != "controls":
                    common = {key:result["checks"][key] for key in (
                        "capture_healthy", "no_application_errors", "all_questions_received_audio")}
                    if args.case == "preferences":
                        result["checks"] = common | {
                            "quiet_preferences_persisted":result["checks"]["quiet_preferences_persisted"],
                            "fresh_session":result["checks"]["fresh_session"],
                            "inference_confirmation_preserved":bool(policy) and
                                policy["confirm_inferred_obligations"] is True,
                            "preferences_do_not_create_duties":not rows,
                        }
                    else:
                        result["checks"] = common | {"one_situational_duty":len(rows)==len(situational)==1}
                        if result["checks"]["one_situational_duty"]:
                            from evals.second_brain.native_visual_responsibility import qualify_visual_duty
                            visual = await qualify_visual_duty(fixture, settings, identity, args.output, rows[0])
                            result["visual"] = visual
                            result["checks"].update(visual["checks"])
                result["ok"] = all(result["checks"].values())
            finally:
                await experience.close()
        result["fixture_cleanup"] = "completed"
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error_code"] = getattr(error,"code",None)
        result["ok"] = False
    finally:
        (args.output/"result.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps({k:result.get(k) for k in ("ok","checks","error_type","error_code","fixture_cleanup")}), flush=True)
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-native-models",action="store_true",required=True)
    parser.add_argument("--case",choices=("controls","preferences","visual"),default="controls")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
