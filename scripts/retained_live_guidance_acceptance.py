#!/usr/bin/env python3
"""Real retained-Live adapter + Hermes/E2B/Files test; no microphone or Voice API.

Reuses the running demo profile, uses only synthetic media in a unique folder,
leaves sandbox/editor and artifacts intact, and cancels only this test's inputs
on failure. No provider prompt changes, packages, or services are installed.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import shutil
import sqlite3
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

from retained_media_acceptance import (Evidence, Failure, json_request, lookup_file,
    private_write, require, verify_file)
from app.hermes_sessions.client import HermesGatewayClient
from app.hermes_sessions.domain_service import compose_domain_work
from app.hermes_sessions.sandbox_provider import RetainedE2BProvider
from app.realtime.live_retained import RetainedLiveWork
from app.sandbox.models import E2BConfiguration


async def run(args):
    import httpx
    from dotenv import load_dotenv
    from e2b import Sandbox
    load_dotenv('backend/.env')
    load_dotenv('.jarvis-dev/phone-viewer-integration.gu1kvk/serving.env', override=True)
    evidence = Evidence(args.output)
    evidence.report['test_kind'] = 'retained_live_adapter_native_guidance'
    evidence.report['voice_audio_tested'] = False
    evidence.report['real_gui_editing_tested'] = False
    key = args.e2b_api_key_file.read_text().strip()
    owner = json.loads((args.profile / 'jarvis-adapter.json').read_text())['owner_id']
    provider = RetainedE2BProvider(E2BConfiguration(api_key=key))
    run_id = uuid4().hex[:10]
    folder = 'guidance-' + run_id
    interim_name, final_name = folder + '-interim.mp4', folder + '-final.mp4'
    sid = None
    work = None
    admissions = []
    completed = False
    async with httpx.AsyncClient(base_url=args.base_url,
            headers={'Authorization': 'Bearer ' + os.environ['JARVIS_STATIC_BEARER_TOKEN']},
            timeout=20, follow_redirects=False, trust_env=False) as http:
        async with HermesGatewayClient(args.profile, timeout=15) as client:
            service = compose_domain_work(args.profile, client).service
            master = await service.master()
            try:
                require((await json_request(http, 'GET', '/sessions/active')).get('session') is None,
                        'user_session_is_active')
                snapshot = await client.snapshot(master.native_id)
                require(not snapshot.get('running') and not snapshot.get('pending'), 'master_is_busy')
                catalog = json.loads((args.profile / 'jarvis' / 'domain-tools.json').read_text())
                allowed = frozenset(tool['name'] for tool in catalog['tools'])
                started = await json_request(http, 'POST', '/sessions/start',
                    json={'title': folder + ' synthetic acceptance', 'resume_active': False})
                sid = UUID(started['session']['id'])
                route = NS(scope=NS(owner_id=owner, session_id=sid), route_id=uuid4())
                async def ready(**_):
                    return None
                async def prepare(**kw):
                    await service.bind_input_scope(kw['session_id'], input_id=kw['input_id'],
                        jarvis_session_id=kw['jarvis_session_id'], allowed_tools=allowed)
                runtime = NS(bridge=NS(service=service), wait_ready=ready, prepare_voice_input=prepare)
                work = RetainedLiveWork(NS(services=NS(), ready=True, request=NS(route=route)), runtime)
                parts = []
                async def publish(result, **_):
                    evidence.event('result_received', text=result.text)
                async def submit(label, text):
                    parts.append({'context_event_id': str(uuid4()), 'sequence': len(parts)+1,
                        'speaker': 'user', 'delta': text, 'start_ms': 0, 'end_ms': 1})
                    packet = NS(route=route, window=NS(request=NS(delegation_id=folder+'-'+label)),
                        as_context=lambda: {'dialogue': list(parts), 'window': {'speech_finality': 'unavailable'}})
                    evidence.event('user_input', label=label, text=text)
                    await work(packet, publish=publish, is_current=lambda: True)
                    admission = work.last_receipt['admission_id']
                    admissions.append(admission)
                    value = await service.receipt(master.logical_id, admission)
                    evidence.event('input_admitted', label=label, receipt=value,
                                   adapter_issue=work.last_issue)
                    return admission
                async def binding():
                    with sqlite3.connect(args.profile / 'jarvis' / 'sandboxes.sqlite') as db:
                        row = db.execute('SELECT sandbox_id FROM sandboxes WHERE logical_session_id=?',
                                         (master.logical_id,)).fetchone()
                    require(row and row[0], 'sandbox_missing')
                    return row[0]
                before = await binding()
                info = await asyncio.to_thread(Sandbox.get_info, sandbox_id=before, api_key=key)
                require(getattr(info.state, 'value', info.state) == 'running', 'sandbox_must_already_be_running')
                sandbox = await asyncio.to_thread(Sandbox.connect, sandbox_id=before, api_key=key)
                async def command(text):
                    wrapped = ('env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/jarvis/workspace '
                        'USER=jarvis LOGNAME=jarvis DISPLAY=:0 XDG_RUNTIME_DIR=/home/jarvis/workspace/.runtime '
                        '/bin/bash --noprofile --norc -c ' + shlex.quote(text))
                    return await asyncio.to_thread(sandbox.commands.run, wrapped,
                        user='jarvis', cwd='/home/jarvis/workspace', timeout=10)
                async def wait_marker(name):
                    async with asyncio.timeout(180):
                        while True:
                            found = await command('if test -f ' + shlex.quote(folder+'/'+name) +
                                                  '; then printf ready; fi')
                            if 'ready' in found.stdout:
                                return
                            value = await service.receipt(master.logical_id, root)
                            require(value.get('status') != 'terminal', 'work_finished_before_'+name)
                            await asyncio.sleep(1)
                evidence.report.update(jarvis_session_id=str(sid), master=master.logical_id,
                    sandbox_before=before, run_id=run_id, folder=folder)
                async with asyncio.timeout(360):
                    root = await submit('start',
                        'This is a bounded synthetic live-guidance test in your CURRENT master sandbox, not a child. '
                        'Do not use personal media, install anything, close/pause the sandbox, or change existing files. '
                        f'Create {folder}/source.mp4, a six-second 320x180 red video labeled DRAFT, '
                        'with a quiet generated sine-tone audio track, H.264 yuv420p/AAC. Use installed FFmpeg. '
                        'Open the source in installed Shotcut on DISPLAY=:0, leaving the editor open without autoplay. '
                        f'Then create {folder}/ready and run a foreground sleep 25 with a sufficient terminal timeout. '
                        'This pause is an intentional window for me to guide you while working. '
                        'After that tool returns, follow incoming guidance before doing anything else. '
                        'If there is none, finish by reporting ready; do not publish yet.')
                    await wait_marker('ready')
                    require((await service.receipt(master.logical_id, root))['status'] == 'started', 'root_not_working')
                    await submit('change',
                        'Change of direction while you are working: the edited video must have a BLUE background '
                        'and the visible words LIVE GUIDE instead of DRAFT. Preserve the generated audio. '
                        'Keep Shotcut and the same sandbox open. I am sending an interim-export request next.')
                    await submit('publish',
                        f'Publish just the first TWO seconds of that updated blue LIVE GUIDE video now as '
                        f'{folder}/interim.mp4, filename {interim_name}, using sandbox_publish_file. '
                        'Do not wait until all editing is finished to publish it. '
                        f'After its saved-file receipt, create {folder}/published and run another foreground '
                        'sleep 25 before continuing. Leave the editor and sandbox open throughout; '
                        'this is a test window for the next instruction.')
                    await wait_marker('published')
                    interim = await lookup_file(http, sid, interim_name)
                    require((await service.receipt(master.logical_id, root))['status'] == 'started',
                            'interim_not_published_during_original_work')
                    evidence.event('interim_available_while_working', file_id=interim['id'])
                    desktop = await command('pgrep -x shotcut >/dev/null && printf editor_open')
                    require('editor_open' in desktop.stdout, 'editor_closed')
                    await verify_file(http, provider, args, evidence, session_id=sid, owner=owner,
                        master=master, file=interim, path=folder+'/interim.mp4', duration=2, label='interim')
                    await submit('continue',
                        f'The interim export is available. Continue the SAME edit: publish the full SIX-second '
                        f'blue LIVE GUIDE version as {folder}/final.mp4, filename {final_name}. '
                        'Keep Shotcut and the sandbox open, then finish with one brief final reply.')
                    async with asyncio.timeout(180):
                        while True:
                            result = await service.receipt(master.logical_id, root)
                            if result['status'] == 'terminal':
                                break
                            await asyncio.sleep(1)
                    require(result.get('outcome') == 'completed', 'original_work_not_completed')
                    final = await lookup_file(http, sid, final_name)
                    await verify_file(http, provider, args, evidence, session_id=sid, owner=owner,
                        master=master, file=final, path=folder+'/final.mp4', duration=6, label='final')
                    for admission in admissions[1:]:
                        value = await service.receipt(master.logical_id, admission)
                        evidence.event('guidance_settled', admission_id=admission, receipt=value)
                        require(value.get('outcome') == 'steered', 'guidance_was_queued_not_steered')
                    after = await binding()
                    require(after == before, 'sandbox_identity_changed')
                    info = await asyncio.to_thread(Sandbox.get_info, sandbox_id=after, api_key=key)
                    require(getattr(info.state, 'value', info.state) == 'running', 'sandbox_closed')
                    require('editor_open' in (await command('pgrep -x shotcut >/dev/null && printf editor_open')).stdout,
                            'editor_closed_after_publication')
                    evidence.report.update(status='passed', sandbox_after=after, editor_still_open=True,
                                           guidance_count=len(admissions)-1)
                    completed = True
                    evidence.event('acceptance_passed')
            except BaseException as error:
                evidence.report.update(status='failed', error_type=type(error).__name__,
                    error_code=str(error) if isinstance(error, Failure) else 'inspect_private_trace')
                evidence.event('acceptance_failed', error_type=type(error).__name__,
                               error_code=evidence.report['error_code'])
                import traceback
                (args.output/'failure.txt').write_text(traceback.format_exc())
            finally:
                if not completed:
                    for admission in admissions:
                        try:
                            value = await service.receipt(master.logical_id, admission)
                            if value.get('status') != 'terminal':
                                await client.request('prompt.cancel', session_id=master.native_id, admission_id=admission)
                        except Exception:
                            evidence.report['cleanup_uncertain'] = True
                if work:
                    await work.close()
                if sid:
                    await json_request(http, 'POST', f'/sessions/{sid}/close')
                private_write(args.output/'report.json', evidence.report)
    return 0 if completed else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=Path('.jarvis-dev/physical-demo-v1/profile'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:9000')
    p.add_argument('--e2b-api-key-file', type=Path, default=Path('.jarvis-dev/sandbox-view-runtime/provider-api-key'))
    p.add_argument('--execute-paid', action='store_true')
    args = p.parse_args()
    require(args.execute_paid, 'explicit_paid_run_required')
    args.profile = args.profile.resolve()
    args.output = args.output.absolute()
    args.ffmpeg, args.ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    args.max_file_bytes = 8*1024**2
    os.umask(0o077)
    raise SystemExit(asyncio.run(run(args)))
