"""One isolated native replay, one authorization, one audio admission.

No execution occurs on import. --execute-once and a fresh Main-authored risk
authorization are required. The installed app and booted simulator are borrowed;
this runner never builds, installs, boots, retries a launch, or calls a provider.
The legacy voice_pass=False is an unqualified sentinel, not a failed assertion.
voice_qualification makes that distinction explicit without grading the answer.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from uuid import UUID, uuid4

from evals.responsive_voice.frontend_host_qualification import (
    ChildOutputDrain, NATIVE_MEDIA, cleanup_prepared_host, qualify_frontend_host,
    require, sanitized, serve_prepared_host, native_host_options, NO_WORKLOAD, REASONING_REMINDER, VISUAL_MEMORY,
    sanitize_live_finalization,
)
from evals.responsive_voice import native_interruption_scenarios as scenarios

SCENARIO = 'current_activity_baseline'
HISTORICAL_UNKNOWN = 'ae630165-2db9-4f9f-938b-6d1d83d143e2'
PREPARED = 'prepared-scenario'
MAX_DURATION = 180
# Scene length and trigger waits are independent limits, matching the iOS plan.
MAX_TRIGGER_TIMEOUT = 120
CAPTURE_GRACE = 30
# The qualified client allocates bounded pre-media capture as well as the
# scene/tail. This is storage capacity, never extra media/deadline authority.
MAX_CAPTURE_PCM_BYTES = (CAPTURE_GRACE + MAX_DURATION) * scenarios.RATE * 2
# Before lease expiry: ready 75s, container 20s, launch 20s, capture grace
# 30s, app termination 20s, export 60s, plus 15s staging/scheduling margin.
LEASE_OVERHEAD = 240
WORKER_READINESS_SECONDS = 45
WORKER_CLOSE_RESERVE = 60
WORKER_LEASE_OVERHEAD = 120
TASK_CONTROL_TIMEOUT_SECONDS = 10


def now():
    return datetime.now(UTC).isoformat()


def unassessed_voice_qualification():
    """Lifecycle/media collection does not evaluate semantic voice acceptance."""
    return {'status':'not_evaluated',
            'reason':'lifecycle_runner_does_not_evaluate_voice_acceptance',
            'voice_pass_semantics':'legacy_unqualified_sentinel_not_failure_verdict'}


def save(path, value):
    """Private atomic journal; no bearer value or credential digest is stored."""
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    fd = os.open(temporary, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path, limit=512*1024):
    with Path(path).open('rb') as stream:
        data = stream.read(limit+1)
    return decode_json(data, limit)


def decode_json(data, limit=512*1024):
    require(len(data)<=limit, 'input_json_bound')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def inside(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'relative_asset_required')
    base = Path(root).resolve()
    path = (base/relative).resolve()
    require(path.is_relative_to(base) and path!=base, 'asset_outside_media_root')
    return path


def copy_bounded(source, destination, limit, checksum=None):
    with Path(source).open('rb') as stream:
        data = stream.read(limit+1)
    require(len(data)<=limit, 'artifact_byte_bound')
    digest = hashlib.sha256(data).hexdigest()
    if checksum is not None:
        require(digest==checksum, 'media_checksum_mismatch')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(destination, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(data)
    return {'path':str(destination), 'bytes':len(data), 'sha256':digest}


@dataclass(frozen=True)
class RunOptions:
    input_manifest: Path
    media_root: Path
    speech_manifest: Path
    artifact_dir: Path
    ledger_dir: Path
    risk_authorization: Path
    canonical_export_command: Path
    hindsight_url: str
    hindsight_api_key_file: Path
    simulator_id: str = '7642363B-3872-44E7-B5CE-09802F2AF3A7'
    bundle_id: str = 'com.symbiote.jarvis'
    scenario: str = SCENARIO
    lease_seconds: int | None = None
    input_mode: str = 'baseline'
    cuts: int | None = None
    native_pool_max_size: int | None = None
    workload: str = NO_WORKLOAD
    worker_model: str | None = None
    worker_reasoning_effort: str = 'low'
    worker_api_key_file: Path | None = None
    worker_python: Path | None = None
    voice_route: str = 'realtime'
    live_model: str | None = None
    live_voice: str | None = None
    task_aware: bool = False
    result_hold_seconds: float | None = None
    result_release_policy: str = 'all_sdk_inputs'
    retained_sequence: Path | None = None


def declared_capture_mode(template):
    require(isinstance(template, dict), 'invalid_capture_mode')
    mode = template.get('capture_mode', 'auto')
    if mode is None:
        return 'auto'
    require(type(mode) is str and mode in {'auto', 'manual'}, 'invalid_capture_mode')
    return mode


def require_manual_capture_scope(options, *, retained_phase):
    # The existing retained path alone supplies a fresh, grant-bound invocation.
    # Do not synthesize an invocation, task authority or manual Live capability.
    require(retained_phase is True and options.input_mode == PREPARED
        and options.scenario == 'baseline' and type(options.cuts) is int and options.cuts == 0
        and options.voice_route == 'realtime' and options.task_aware is False
        and options.workload in {NO_WORKLOAD, VISUAL_MEMORY}
        and options.result_hold_seconds is None and options.worker_model is None
        and options.worker_api_key_file is None and options.worker_python is None,
        'manual_capture_requires_retained_prepared_baseline')


def stage_media(options, *, expected_manifest_sha256=None, allow_empty_frames=False,
                expected_capture_mode='auto', expected_memory_readiness=False):
    if options.input_mode == PREPARED:
        return stage_prepared_media(options, expected_manifest_sha256,
                                    allow_empty_frames=allow_empty_frames,
                                    expected_capture_mode=expected_capture_mode,
                                    expected_memory_readiness=expected_memory_readiness)
    require(expected_memory_readiness is False, 'memory_readiness_requires_prepared_manual')
    require(not allow_empty_frames, 'zero_camera_requires_retained_prepared_phase')
    template = read_json(options.input_manifest)
    require(declared_capture_mode(template) == 'auto' and expected_capture_mode == 'auto',
            'manual_capture_requires_retained_prepared_baseline')
    require('post_media_capture_seconds' not in template, 'explicit_capture_tail_requires_prepared_input')
    require(options.scenario==SCENARIO and template.get('version')==1
        and template.get('voice_route')==options.voice_route and template.get('duration_seconds')==45,
        'only_reviewed_baseline_supported')
    frames, utterances = template.get('frames'), template.get('utterances')
    require(isinstance(frames, list) and [item.get('at_seconds') for item in frames]==[20,22,24],
            'baseline_frames_mismatch')
    require(utterances==[{'clip_id':'current_activity','trigger':{'kind':'at_time','at_seconds':25}}],
            'baseline_utterance_mismatch')
    root = options.artifact_dir/'inputs'
    root.mkdir(mode=0o700)
    staged, records = [], []
    for index, item in enumerate(frames):
        relative = f'frames/frame-{index}.jpg'
        record = copy_bounded(inside(options.media_root, item['relative_path']), root/relative,
                              4*1024*1024, item['sha256'])
        records.append(record)
        staged.append({'at_seconds':item['at_seconds'], 'relative_path':relative, 'sha256':record['sha256']})
    speech = read_json(options.speech_manifest)
    require(speech.get('format')=='jarvis.speech_assets.v1', 'speech_format_mismatch')
    clips = [item for item in speech.get('clips', []) if item.get('id')=='current_activity']
    require(len(clips)==1, 'baseline_speech_identity_mismatch')
    clip = clips[0]
    require((clip.get('sample_rate_hz'),clip.get('channels'),clip.get('sample_width_bytes'))==(48000,1,2),
            'baseline_speech_format_mismatch')
    allowed = ('id','sha256','sample_rate_hz','channels','sample_width_bytes','sample_count',
               'first_active_sample','last_active_sample_exclusive','energy_threshold_abs_pcm16')
    selected = {key:clip[key] for key in allowed}
    selected['relative_path'] = 'current_activity.wav'
    records.append(copy_bounded(inside(options.speech_manifest.parent, clip['relative_path']),
                               root/'speech/current_activity.wav', 6*1024*1024, clip['sha256']))
    save(root/'speech/speech.json', {'format':'jarvis.speech_assets.v1','clips':[selected]})
    # Never copy old run/owner/session/backend/token fields, nor old token files.
    plan = {'version':1,'voice_route':options.voice_route,'duration_seconds':45,
            'speech_manifest':'speech/speech.json','frames':staged,'utterances':utterances}
    save(root/'media-plan.json', plan)
    return plan, records


def stage_prepared_media(options, expected_manifest_sha256, *, allow_empty_frames=False,
                         expected_capture_mode='auto', expected_memory_readiness=False):
    """Consume compiler runtime fields only; never load its expectations file.

    Compiler scenario revisions use the existing Swift version=1 wire contract.
    Case labels and cut counts are authorization scope, not semantic acceptance.
    """
    validate_workload_options(options)
    raw = scenarios.bounded_read(options.input_manifest, 512*1024)
    require(scenarios.digest(raw)==expected_manifest_sha256, 'authorized_manifest_hash_mismatch')
    template = decode_json(raw)
    fields = {'version','run_id','backend_base_url','owner_id','area_id','memory_bank_id',
              'voice_route','bearer_token_file','duration_seconds','speech_manifest','frames','utterances'}
    if options.task_aware:
        fields.add('task_barrier')
    if isinstance(template,dict) and 'capture_mode' in template:
        fields.add('capture_mode')
    if isinstance(template,dict) and 'memory_readiness' in template:
        fields.add('memory_readiness')
    if isinstance(template,dict) and 'post_media_capture_seconds' in template:
        fields.add('post_media_capture_seconds')
    require(isinstance(template,dict) and set(template)==fields and type(template['version']) is int
            and template['version']==1 and template['voice_route']==options.voice_route, 'unsupported_prepared_manifest')
    capture_mode = declared_capture_mode(template)
    require(type(expected_capture_mode) is str and expected_capture_mode in {'auto','manual'}
        and capture_mode == expected_capture_mode, 'capture_mode_authorization_mismatch')
    if capture_mode == 'manual':
        require_manual_capture_scope(options, retained_phase=allow_empty_frames)
    from evals.responsive_voice.native_memory_readiness_control import declared_memory_readiness
    memory_enabled = declared_memory_readiness(template)
    require(type(expected_memory_readiness) is bool and memory_enabled is expected_memory_readiness,
            'memory_readiness_authorization_mismatch')
    if memory_enabled:
        require_manual_capture_scope(options, retained_phase=allow_empty_frames)
    if options.task_aware:
        declaration = template['task_barrier']
        require(isinstance(declaration,dict) and set(declaration)=={'version','scope_file','request_input_ordinal'}
            and type(declaration['version']) is int and declaration['version']==1
            and declaration['scope_file']=='task-scope.json'
            and type(declaration['request_input_ordinal']) is int and declaration['request_input_ordinal']==0,
            'invalid_task_barrier_declaration')
    duration = scenarios.finite(template['duration_seconds'], 'invalid_duration', 1, MAX_DURATION)
    if 'post_media_capture_seconds' in template:
        tail = scenarios.finite(template['post_media_capture_seconds'], 'invalid_capture_tail', 0, 20)
        require(duration+tail<=MAX_DURATION, 'scene_and_capture_tail_exceed_bound')
    source_root = options.input_manifest.parent.resolve()
    require(options.media_root.resolve()==source_root and options.speech_manifest.resolve()==
            scenarios.local_path(source_root,template['speech_manifest']), 'prepared_media_root_mismatch')
    frames, utterances = template['frames'], template['utterances']
    require(type(allow_empty_frames) is bool and (not allow_empty_frames or (
        options.input_mode==PREPARED and options.scenario=='baseline' and options.cuts==0
        and not options.task_aware and options.voice_route=='realtime'
        and options.workload in {NO_WORKLOAD,VISUAL_MEMORY})), 'zero_camera_requires_retained_prepared_phase')
    require(isinstance(frames,list) and (0 if allow_empty_frames else 1)<=len(frames)<=240, 'invalid_frames')
    if allow_empty_frames:
        require(isinstance(utterances,list) and utterances
            and utterances[0].get('trigger',{}).get('kind')=='at_time'
            and all(isinstance(item,dict) and item.get('trigger',{}).get('kind') in
                    ({'at_time','after_manual_response'} if memory_enabled else {'at_time'})
                    for item in utterances), 'retained_phase_requires_timed_speech')
    # A no-duty visual control is not a dummy audio turn. Keep this exception
    # narrower than the general prepared schema and every task/cut recipe.
    allow_empty_input = (getattr(options,'input_mode',None)==PREPARED
        and getattr(options,'scenario',None)=='baseline' and type(options.cuts) is int
        and options.cuts==0 and options.task_aware is False)
    minimum_inputs = 0 if allow_empty_input else 1
    require(isinstance(utterances,list) and minimum_inputs<=len(utterances)<=20, 'invalid_utterances')
    task_cancel = options.task_aware is True and options.scenario == 'explicit_cancel'
    if task_cancel:
        require(options.voice_route=='realtime' and capture_mode=='auto'
            and options.workload==REASONING_REMINDER and type(options.cuts) is int and options.cuts==1
            and duration<=120 and len(utterances)==4, 'task_cancel_profile_mismatch')
        expected_kinds = ['at_time','after_task_claim','response_output_start',
                          'after_provider_input_finalized']
        require([item.get('trigger',{}).get('kind') if isinstance(item,dict) else None for item in utterances]
                ==expected_kinds, 'task_cancel_sequence_mismatch')
        require(all(item['trigger'].get('after_input_ordinal')==i-1 for i,item in enumerate(utterances) if i)
            and utterances[2]['trigger'].get('require_task_claim') is True,
            'task_cancel_claim_and_predecessor_required')
    speech = read_json(options.speech_manifest)
    require(isinstance(speech,dict) and speech.get('format')=='jarvis.speech_assets.v1'
        and isinstance(speech.get('clips'),list) and minimum_inputs<=len(speech['clips'])<=20,
        'invalid_prepared_speech')
    require(bool(utterances)==bool(speech['clips']), 'empty_prepared_input_mismatch')
    # The existing compiler validator exposes named bundles, not a standalone
    # clip loader. Alias this one neutral bundle and retain only its baseline
    # namespace instead of duplicating its WAV/hash/activity validation rules.
    validated = scenarios.load_bundles({name:options.speech_manifest
                                       for name in ('baseline','conversation','state')})
    clips = {clip.source_id:clip for key,clip in validated.items() if key.startswith('baseline/')}
    scheduled, used, last_time, last_cursor, cut_count, samples = [], set(), -1.0, -1, 0, 0
    completion_bounds = []
    latest_start = 0.0
    claim_gates = 0
    trigger_timeout_limit = min(duration, MAX_TRIGGER_TIMEOUT)
    for item in utterances:
        require(isinstance(item,dict) and set(item)=={'clip_id','trigger'}, 'invalid_prepared_utterance')
        identity, trigger = item['clip_id'], item['trigger']
        require(isinstance(identity,str) and identity in clips and isinstance(trigger,dict), 'invalid_clip_reference')
        clip = clips[identity]
        kind = trigger.get('kind')
        if kind=='at_time':
            require(set(trigger)=={'kind','at_seconds'}, 'unsupported_trigger_fields')
            at = scenarios.finite(trigger['at_seconds'], 'invalid_trigger_time', 0, duration)
            require(last_time<at<duration and at>=latest_start, 'nonmonotonic_or_late_utterance')
            last_time = latest_start = at
        elif kind=='after_manual_response':
            require(memory_enabled and capture_mode=='manual' and scheduled,
                    'manual_memory_trigger_not_authorized')
            # Original scene clock, never prior SDK completion or nonce time.
            at = scenarios.finite(trigger['at_seconds'], 'invalid_trigger_time', 0, duration)
            timeout = scenarios.finite(trigger['timeout_seconds'], 'invalid_trigger_timeout', .001, MAX_TRIGGER_TIMEOUT)
            require(at>=completion_bounds[-1], 'manual_memory_gate_overlaps_prior_input')
            latest_start = at+timeout
        elif kind=='output_start':
            require(set(trigger)=={'kind','after_cursor','delay_seconds','timeout_seconds'} and scheduled,
                    'unsupported_trigger_fields')
            cursor = trigger['after_cursor']
            require(type(cursor) is int and last_cursor<cursor<=19, 'nonmonotonic_output_cursor')
            delay = scenarios.finite(trigger['delay_seconds'], 'invalid_trigger_delay', 0, duration)
            timeout = scenarios.finite(trigger['timeout_seconds'], 'invalid_trigger_timeout', .01, trigger_timeout_limit)
            require(0<delay<timeout, 'invalid_output_trigger_window')
            latest_start += timeout
            last_cursor = cursor
            cut_count += 1
        elif kind=='response_output_start':
            require(options.voice_route=='realtime', 'response_trigger_requires_realtime')
            required = {'kind','after_input_ordinal','delay_seconds','timeout_seconds','max_pcm_age_seconds'}
            require(required<=set(trigger)<=required|{'response_id','connection_id','require_task_claim'},
                    'unsupported_response_trigger_fields')
            if 'require_task_claim' in trigger:
                require(type(trigger['require_task_claim']) is bool
                    and (options.task_aware or trigger['require_task_claim'] is False), 'task_claim_mode_required')
            if options.task_aware:
                require(trigger.get('require_task_claim') is True, 'task_cut_requires_claim')
            ordinal = trigger['after_input_ordinal']
            require(type(ordinal) is int and 0<=ordinal<len(scheduled),
                    'response_trigger_requires_earlier_input')
            require(all(type(trigger[key]) in (int,float) for key in
                        ('delay_seconds','timeout_seconds','max_pcm_age_seconds')),
                    'invalid_response_trigger_number')
            delay = scenarios.finite(trigger['delay_seconds'], 'invalid_trigger_delay', 0, duration)
            timeout = scenarios.finite(trigger['timeout_seconds'], 'invalid_trigger_timeout', 0, trigger_timeout_limit)
            age = scenarios.finite(trigger['max_pcm_age_seconds'], 'invalid_response_pcm_age', 0, timeout)
            require(0<=delay<timeout and 0<age<=timeout, 'invalid_response_trigger_window')
            if 'response_id' in trigger:
                response_id = trigger['response_id']
                require(isinstance(response_id,str)
                        and re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}',response_id) is not None,
                        'invalid_response_trigger_identity')
            if 'connection_id' in trigger:
                connection_id = trigger['connection_id']
                require(isinstance(connection_id,str) and re.fullmatch(
                    r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',connection_id) is not None,
                    'invalid_response_trigger_connection')
                UUID(connection_id)
            # Timeout starts after the named input's complete SDK submission,
            # not its nominal start. This is only a conservative staging bound;
            # actual completion, response/PCM selection and enqueue fencing
            # remain native responsibilities and are not acceptance evidence.
            latest_start = max(completion_bounds[-1],completion_bounds[ordinal]+timeout)
            cut_count += 1
        elif kind=='after_provider_input_finalized':
            require(options.task_aware is True and options.voice_route=='realtime'
                and capture_mode=='auto' and (options.scenario=='pause_continue' or task_cancel)
                and type(options.cuts) is int and 1<=options.cuts<=3
                and options.workload==REASONING_REMINDER,
                'provider_input_finalization_requires_task_realtime_pause_continue')
            # Only the explicitly declared three-cut fixture extends its scene.
            # The prior one/two-cut bounds and all trigger waits stay unchanged.
            extended_three_cut = (options.cuts==3 and duration==150 and
                [row.get('trigger',{}).get('kind') if isinstance(row,dict) else None for row in utterances] ==
                ['at_time','after_task_claim'] + ['response_output_start','after_provider_input_finalized']*3)
            require(duration<=120 or extended_three_cut, 'provider_input_finalization_scene_exceeds_task_bound')
            require(set(trigger)=={'kind','after_input_ordinal','timeout_seconds'},
                    'unsupported_provider_input_finalization_fields')
            ordinal = trigger['after_input_ordinal']
            require(type(ordinal) is int and scheduled and ordinal==len(scheduled)-1
                and scheduled[ordinal]['trigger']['kind']=='response_output_start'
                and scheduled[ordinal]['trigger'].get('require_task_claim') is True,
                'provider_input_finalization_requires_preceding_task_cut')
            timeout = scenarios.finite(trigger['timeout_seconds'],
                'invalid_provider_input_finalization_timeout', .001, trigger_timeout_limit)
            # Reserve the whole wait after predecessor SDK completion. This
            # admits a plan, not a provider/canonical identity or semantic cut.
            # Native evidence must prove finalization AND full SDK submission.
            latest_start = completion_bounds[ordinal]+timeout
        elif kind in {'after_task_claim','after_input_complete'}:
            require(options.task_aware, 'task_trigger_mode_required')
            required = {'kind','after_input_ordinal','timeout_seconds'}
            if kind=='after_input_complete':
                required.add('delay_seconds')
            require(set(trigger)==required, 'unsupported_task_trigger_fields')
            ordinal = trigger['after_input_ordinal']
            require(type(ordinal) is int and 0<=ordinal<len(scheduled), 'task_trigger_requires_earlier_input')
            timeout = scenarios.finite(trigger['timeout_seconds'],'invalid_task_trigger_timeout',.001,trigger_timeout_limit)
            if kind=='after_task_claim':
                require(ordinal==0, 'task_claim_requires_setup_input')
                latest_start = max(completion_bounds[-1],completion_bounds[ordinal]+timeout)
                claim_gates += 1
            else:
                delay = scenarios.finite(trigger['delay_seconds'],'invalid_task_trigger_delay',0,timeout)
                latest_start = max(completion_bounds[-1],completion_bounds[ordinal]+delay)
        else:
            require(False, 'unsupported_prepared_trigger')
        require(latest_start+clip.sample_count/scenarios.RATE+5<=duration, 'scenario_cannot_fit_deadline')
        completion_bounds.append(latest_start+clip.sample_count/scenarios.RATE)
        samples += clip.sample_count
        require(samples<=60*scenarios.RATE, 'input_budget_exceeded')
        used.add(identity)
        scheduled.append({'clip_id':identity,'trigger':dict(trigger)})
    require(cut_count==options.cuts, 'prepared_cut_count_mismatch')
    if options.task_aware:
        require(claim_gates>=1 and scheduled[0]['trigger']['kind']=='at_time', 'task_setup_and_claim_gate_required')
    require(used==set(clips), 'unreferenced_prepared_speech')
    root = options.artifact_dir/'inputs'
    root.mkdir(mode=0o700)
    records, staged_frames, total_frames, last_frame = [], [], 0, -1.0
    for index,item in enumerate(frames):
        require(isinstance(item,dict) and set(item)=={'at_seconds','relative_path','sha256'}, 'invalid_frame')
        at = scenarios.finite(item['at_seconds'], 'invalid_frame_time', 0, duration)
        require(last_frame<at<duration, 'nonmonotonic_or_late_frame')
        source = scenarios.local_path(source_root,item['relative_path'])
        data = scenarios.bounded_read(source,4*1024*1024)
        require(data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9'), 'invalid_jpeg_signature')
        require(scenarios.digest(data)==item['sha256'], 'frame_hash_mismatch')
        total_frames += len(data)
        require(total_frames<=64*1024*1024, 'frame_budget_exceeded')
        relative = f'frames/frame-{index}.jpg'
        records.append(copy_bounded(source,root/relative,4*1024*1024,item['sha256']))
        staged_frames.append({'at_seconds':item['at_seconds'],'relative_path':relative,'sha256':item['sha256']})
        last_frame = at
    # Empty-input controls still carry a real, empty selected-speech manifest.
    (root/'speech').mkdir(mode=0o700)
    selected = []
    for raw_clip in speech['clips']:
        clip = clips[raw_clip['id']]
        relative = f'{clip.source_id}.wav'
        records.append(copy_bounded(scenarios.local_path(options.speech_manifest.parent,raw_clip['relative_path']),
            root/'speech'/relative,60*scenarios.RATE*2+65536,clip.sha256))
        selected.append({'id':clip.source_id,'relative_path':relative,'sha256':clip.sha256,
            'sample_rate_hz':scenarios.RATE,'channels':1,'sample_width_bytes':2,'sample_count':clip.sample_count,
            'first_active_sample':clip.first,'last_active_sample_exclusive':clip.end,
            'energy_threshold_abs_pcm16':clip.threshold})
    save(root/'speech/speech.json',{'format':'jarvis.speech_assets.v1','clips':selected})
    plan = {'version':1,'voice_route':options.voice_route,'duration_seconds':template['duration_seconds'],
            'speech_manifest':'speech/speech.json','frames':staged_frames,'utterances':scheduled}
    if 'capture_mode' in template:
        plan['capture_mode'] = template['capture_mode']
    if memory_enabled:
        plan['memory_readiness'] = dict(template['memory_readiness'])
    if 'post_media_capture_seconds' in template:
        plan['post_media_capture_seconds'] = template['post_media_capture_seconds']
    if options.task_aware:
        plan['task_barrier'] = dict(template['task_barrier'])
    save(root/'media-plan.json',plan)
    return plan, records


def capture_budget(duration, lease_seconds, *, workload=NO_WORKLOAD, post_media_capture_seconds=0):
    duration = scenarios.finite(duration, 'invalid_duration', 1, MAX_DURATION)
    tail = scenarios.finite(post_media_capture_seconds, 'invalid_capture_tail', 0, 20)
    require(duration+tail<=MAX_DURATION, 'scene_and_capture_tail_exceed_bound')
    require(workload in {NO_WORKLOAD,REASONING_REMINDER,VISUAL_MEMORY}, 'invalid_native_workload')
    minimum = max(240,math.ceil(duration+tail)+LEASE_OVERHEAD+
                  (WORKER_LEASE_OVERHEAD if workload==REASONING_REMINDER else 0))
    effective = max(300,minimum) if lease_seconds is None else lease_seconds
    require(type(effective) is int and minimum<=effective<=900, 'lease_too_short_for_capture_and_export')
    return {'duration_seconds':duration,'capture_wait_seconds':duration+tail+CAPTURE_GRACE,
            'lease_seconds':effective,'minimum_lease_seconds':minimum,
            'decoded_pcm_byte_limit':min(MAX_CAPTURE_PCM_BYTES,
                int((CAPTURE_GRACE+duration+tail)*scenarios.RATE)*2),
            'capture_json_byte_limit':16*1024*1024}


def validate_workload_options(options):
    require(type(options.result_release_policy) is str
        and options.result_release_policy in {'all_sdk_inputs','no_auto_release'},
        'invalid_result_release_policy')
    if options.result_release_policy == 'no_auto_release':
        require(options.input_mode==PREPARED and options.scenario=='explicit_cancel'
            and options.cuts==1 and options.task_aware is True
            and options.voice_route=='realtime' and options.workload==REASONING_REMINDER
            and options.retained_sequence is None
            and type(options.result_hold_seconds) in (int,float)
            and math.isfinite(options.result_hold_seconds) and 0<options.result_hold_seconds<=60,
            'held_pending_cancel_scope_required')
    require(options.workload in {NO_WORKLOAD,REASONING_REMINDER,VISUAL_MEMORY}, 'invalid_native_workload')
    if options.workload==VISUAL_MEMORY:
        require(options.voice_route=='realtime' and options.task_aware is False
            and options.result_hold_seconds is None and options.worker_model is None
            and options.worker_api_key_file is None and options.worker_python is None
            and options.worker_reasoning_effort=='low'
            and options.live_model is None and options.live_voice is None,
            'visual_memory_requires_realtime_without_worker')


def host_options(options):
    validate_workload_options(options)
    return native_host_options(NATIVE_MEDIA,native_pool_max_size=options.native_pool_max_size,
        workload=options.workload,frontier_model=options.worker_model,
        frontier_reasoning_effort=options.worker_reasoning_effort if options.workload==REASONING_REMINDER else None,
        voice_route=options.voice_route,live_model=options.live_model,live_voice=options.live_voice)


def worker_closed(receipt):
    return (isinstance(receipt,dict) and receipt.get('status')=='closed' and all(
        receipt.get(key) is True for key in ('worker_joined','readiness_inactive','pipes_complete','secrets_removed')))


def require_worker_readiness(receipt, resources, options):
    require(isinstance(receipt,dict) and receipt.get('status')=='ready' and receipt.get('registry_ready') is True
        and all(receipt.get(key)==resources[key] for key in ('run_id','owner_id','database_schema','area_id'))
        and receipt.get('memory_bank_id')==resources['memory_bank_id']
        and receipt.get('model')==options.worker_model
        and receipt.get('reasoning_effort')==options.worker_reasoning_effort
        and isinstance(receipt.get('worker_id'),str) and bool(receipt['worker_id'])
        and isinstance(receipt.get('host_id'),str) and bool(receipt['host_id'])
        and type(receipt.get('pid')) is int and receipt['pid']>0, 'native_task_worker_registry_not_ready')


def validate_risk(options):
    validate_workload_options(options)
    document = read_json(options.risk_authorization)
    require(document.get('schema_version')==1 and document.get('allow_one_launch') is True
        and document.get('exclusive_simulator_app') is True, 'explicit_one_attempt_authorization_required')
    identity = str(UUID(document['authorization_id']))
    expires = datetime.fromisoformat(document['expires_at'])
    require(expires.tzinfo is not None and datetime.now(UTC)<expires, 'risk_authorization_expired')
    require(document.get('scenario')==options.scenario
        and UUID(document['simulator_id'])==UUID(options.simulator_id)
        and document.get('bundle_id')==options.bundle_id, 'risk_authorization_scope_mismatch')
    if options.input_mode==PREPARED:
        require(document.get('input_mode')==PREPARED and type(document.get('cuts')) is int
            and document['cuts']==options.cuts and isinstance(document.get('input_manifest_sha256'),str)
            and re.fullmatch(r'[0-9a-f]{64}',document['input_manifest_sha256']) is not None,
            'prepared_input_authorization_required')
    require(document.get('workload',NO_WORKLOAD)==options.workload, 'workload_authorization_scope_mismatch')
    if options.workload==VISUAL_MEMORY:
        require(all(document.get(key) is None for key in
            ('frontier_model','frontier_reasoning_effort','result_hold_seconds')),
            'visual_memory_grant_worker_escalation')
    require(document.get('voice_route','realtime')==options.voice_route, 'voice_route_authorization_scope_mismatch')
    require(document.get('task_aware',False) is options.task_aware, 'task_replay_authorization_scope_mismatch')
    require(type(document.get('result_release_policy','all_sdk_inputs')) is str
        and document.get('result_release_policy','all_sdk_inputs')==options.result_release_policy,
        'result_release_policy_authorization_mismatch')
    capture_mode = document.get('capture_mode', 'auto')
    require(type(capture_mode) is str and capture_mode in {'auto','manual'},
            'invalid_capture_mode_authorization')
    if capture_mode == 'manual':
        require_manual_capture_scope(options,
            retained_phase=isinstance(document.get('retained_sequence'), dict))
    memory_enabled = document.get('memory_readiness', False)
    require(type(memory_enabled) is bool, 'invalid_memory_readiness_authorization')
    if memory_enabled:
        require(capture_mode=='manual', 'memory_readiness_requires_manual_authorization')
        require_manual_capture_scope(options, retained_phase=isinstance(document.get('retained_sequence'), dict))
    if options.task_aware:
        require(type(document.get('result_hold_seconds')) in (int,float)
            and document['result_hold_seconds']==options.result_hold_seconds, 'task_hold_authorization_required')
    if options.voice_route=='live':
        require(document.get('voice_route')=='live' and document.get('live_model')==options.live_model
            and document.get('live_voice')==options.live_voice, 'explicit_live_model_voice_authorization_required')
    if options.workload==REASONING_REMINDER:
        require(document.get('frontier_model')==options.worker_model
            and document.get('frontier_reasoning_effort')==options.worker_reasoning_effort,
            'explicit_workload_model_authorization_required')
    history = document.get('historical_attempts', [])
    matches = [item for item in history if item.get('audio_session_id')==HISTORICAL_UNKNOWN]
    require(len(matches)==1 and matches[0].get('decision') in {'risk_accepted','resolved'}
        and isinstance(matches[0].get('evidence_or_rationale'), str)
        and 10<=len(matches[0]['evidence_or_rationale'])<=2048, 'historical_unknown_risk_gate_unresolved')
    return identity, document


def provider_evidence(records, launched, *, voice_route='realtime', native_events=(),
                      live_finalizations=(), live_finalizations_dropped=0):
    if voice_route=='live':
        # Media preparation proves creation, not activation or remote closure.
        observed=[item for item in native_events if item.get('type') in
                  {'audio.provider.selected','voice_route.selected','live.media.prepare','live.media.activate'}]
        sessions={item['provider_session_id'] for item in observed
            if item.get('type')=='live.media.prepare' and isinstance(item.get('provider_session_id'),str)
            and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',item['provider_session_id'])
            and not item['provider_session_id'].lower().startswith('sk-')}
        diagnostics, invalid, conflicts = [], 0, set()
        bindings, finalized = {}, set()
        anchors=[item for item in observed if any(item.get(key) is not None
            for key in ('audio_session_id','provider_session_id','route_id'))]
        bounded = live_finalizations if isinstance(live_finalizations,(tuple,list)) else ()
        if bounded is not live_finalizations:
            invalid += 1
        overflow = max(0,len(bounded)-16)
        for raw in bounded[:16]:
            item=sanitize_live_finalization(raw)
            if item is None:
                invalid += 1
                continue
            if item not in diagnostics:
                diagnostics.append(item)
            session_id=item['provider_session_id']
            if session_id is not None:
                sessions.add(session_id)
            related=[event for event in anchors if any(event.get(key) is not None
                and event.get(key)==item[key] for key in ('audio_session_id','provider_session_id','route_id'))]
            matching=(not anchors or bool(related) and all(
                all(event.get(key) is None or event[key]==item[key]
                    for key in ('audio_session_id','provider_session_id','route_id','route_generation'))
                for event in related))
            binding=tuple(item[key] for key in ('audio_session_id','jarvis_session_id','route_id','route_generation'))
            if session_id is not None:
                previous=bindings.setdefault(session_id,(binding,item))
                if previous!=(binding,item) or not matching:
                    conflicts.add(session_id)
            if not matching:
                invalid += 1
            elif item['remote_closure_confirmed']:
                finalized.add(session_id)
        loss = (type(live_finalizations_dropped) is not int or live_finalizations_dropped!=0 or overflow>0)
        unbound = any(item['creation_submitted'] is not False and item['provider_session_id'] is None
                      for item in diagnostics)
        closed = bool(sessions) and sessions<=finalized and not (invalid or conflicts or loss or unbound)
        outcome = ('confirmed' if closed else 'unknown' if diagnostics or invalid or loss
                   else 'unavailable' if launched else 'not_requested')
        if (diagnostics and not sessions and not invalid and not loss
                and all(item['finalization_outcome']=='not_requested' for item in diagnostics)):
            outcome='not_requested'
        return {'voice_route':'live','native_route_events':observed,'provider_session_ids':sorted(sessions),
            'creation_outcome':'created' if sessions else 'unknown' if launched else 'not_requested',
            'finalization_outcome':outcome,'remote_closure_confirmed':closed,'realtime_hangup_applicable':False,
            'live_finalization_diagnostics':diagnostics,
            'finalization_by_session':{identity:'confirmed' if identity in finalized and identity not in conflicts
                                       else 'unknown' for identity in sorted(sessions)},
            'finalization_binding_conflicts':sorted(conflicts),'invalid_finalization_diagnostics':invalid,
            'finalization_evidence_dropped':live_finalizations_dropped if type(live_finalizations_dropped) is int
                                           and live_finalizations_dropped>=0 else None,
            'ignored_realtime_diagnostic_count':len(records),'provider_attempt_budget_violated':len(sessions)>1,
            'meaning':'Confirmed requires the owned Live adapter session.closed receipt for every known session, '
                'with matching route identity and no evidence loss. Local cleanup and Realtime hangup are not finalization.',
            'historical_attempts':[{'audio_session_id':HISTORICAL_UNKNOWN,'outcome':'unknown','changed_by_this_run':False}]}
    calls = sorted({item['call_id'] for item in records if item.get('call_id')})
    closures = {}
    for item in records:
        if item.get('stage')=='hangup_http' and item.get('call_id'):
            state = item.get('closure_outcome')
            closures[item['call_id']] = ('accepted' if state=='accepted' and item.get('http_status')==200
                                         else 'failed' if state=='failed' else 'unknown')
    return {'voice_route':'realtime','records':records, 'created_call_ids':calls, 'closure_by_call':closures,
        'creation_outcome':('created' if calls else 'rejected' if any(item.get('creation_outcome')=='rejected'
            for item in records) else 'unknown' if launched else 'not_requested'),
        'remote_closure_confirmed':False, 'provider_attempt_budget_violated':len(calls)>1,
        'meaning':'HTTP 200 hangup is accepted teardown, never confirmed remote closure.',
        'historical_attempts':[{'audio_session_id':HISTORICAL_UNKNOWN, 'outcome':'unknown', 'changed_by_this_run':False}]}


class _CommandDrain(ChildOutputDrain):
    def __init__(self, child, log):
        self.stdout_prefix = bytearray()
        super().__init__(child, log)

    def _control_bytes(self, chunk):
        self.stdout_prefix.extend(chunk[:max(0,8192-len(self.stdout_prefix))])
        super()._control_bytes(chunk)


class ProcessBoundary:
    """Only fixed simctl commands and the explicitly supplied exporter argv."""
    async def command(self, argv, log_path, timeout=20):
        log = os.fdopen(os.open(log_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb')
        child = drain = None
        try:
            child = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
                env={key:os.environ[key] for key in ('PATH','HOME','TMPDIR','LANG','SSL_CERT_FILE','SSL_CERT_DIR') if key in os.environ})
            drain = _CommandDrain(child, log)
            await asyncio.wait_for(child.wait(), timeout)
            output = await drain.wait_closed()
            require(output['complete'] and not output['errors'], 'command_log_drain_incomplete')
            return {'returncode':child.returncode, 'stdout':drain.stdout_prefix.decode(errors='replace'),
                    'log':str(log_path), 'output_drain':output}
        finally:
            if child is not None and child.returncode is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(child.wait(), 5)
                except TimeoutError:
                    os.killpg(child.pid, signal.SIGKILL)
                    await child.wait()
            if drain is not None:
                await drain.wait_closed()
            log.close()

    async def wait_capture(self, path, seconds):
        async with asyncio.timeout(seconds):
            while not path.is_file():
                await asyncio.sleep(.25)


def sample_replay_source_clock():
    """Use the app's exact v1 clock API on the simulator's Darwin host.

    Never compare Python monotonic/UTC epochs to ProcessInfo.systemUptime.
    Unsupported platforms or missing boot identity cannot authorize a launch.
    """
    require(sys.platform == 'darwin', 'retained_source_clock_unavailable')
    import ctypes
    try:
        system = ctypes.CDLL(None)
        sysctl = system.sysctlbyname
        sysctl.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                           ctypes.c_void_p, ctypes.c_size_t]
        sysctl.restype = ctypes.c_int
        size = ctypes.c_size_t(0)
        require(sysctl(b'kern.bootsessionuuid', None, ctypes.byref(size), None, 0) == 0
            and 0 < size.value <= 128, 'retained_source_clock_unavailable')
        buffer = ctypes.create_string_buffer(size.value)
        require(sysctl(b'kern.bootsessionuuid', buffer, ctypes.byref(size), None, 0) == 0,
                'retained_source_clock_unavailable')
        boot = str(UUID(buffer.value.decode('ascii')))
        foundation = ctypes.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            ('objc_msgSend', objc))
        send_double = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p)(
            ('objc_msgSend', objc))
        klass = objc.objc_getClass(b'NSProcessInfo')
        require(bool(klass), 'retained_source_clock_unavailable')
        info = send_object(klass, objc.sel_registerName(b'processInfo'))
        require(bool(info), 'retained_source_clock_unavailable')
        uptime = send_double(info, objc.sel_registerName(b'systemUptime'))
        return {'clock':'ProcessInfo.systemUptime', 'boot_session_id':boot,
                'uptime_seconds':uptime}
    except (OSError, AttributeError, ValueError, UnicodeError):
        require(False, 'retained_source_clock_unavailable')


def retained_source_deadline(plan, retained, *, sample=None):
    """Exact app v1 deadline; not a promise that remaining startup will fit."""
    require(retained is not None and retained['index'] > 0, 'retained_continuation_required')
    phase = retained['phases'][retained['index']]
    evidence = retained['source_anchor']
    raw = scenarios.bounded_read(evidence['path'], 16384)
    require(scenarios.digest(raw) == evidence['sha256'], 'retained_source_anchor_changed')
    anchor = decode_json(raw, 16384)
    expected = {'format','run_id','origin_invocation_id','monotonic_before_seconds',
                'monotonic_after_seconds','utc_unix_seconds','boot_session_id'}
    require(isinstance(anchor,dict) and set(anchor) == expected
        and anchor['format'] == 'jarvis.replay.source_clock.v1'
        and UUID(anchor['run_id']) == UUID(retained['scope']['run_id'])
        and anchor['origin_invocation_id'] == retained['phases'][0]['invocation_id'],
        'retained_source_anchor_identity_mismatch')
    require(plan.get('replay_invocation_id') == phase['invocation_id']
        and plan.get('replay_source_clock') == {'version':1,
            'phase_start_seconds':phase['phase_start_seconds'],
            'anchor_file':'source-clock.json', 'anchor_sha256':evidence['sha256']},
        'retained_source_clock_plan_mismatch')
    require(all(type(anchor[key]) in (int,float) for key in
        ('monotonic_before_seconds','monotonic_after_seconds','utc_unix_seconds')),
        'invalid_source_anchor')
    before = scenarios.finite(anchor['monotonic_before_seconds'],'invalid_source_anchor',0,1e12)
    after = scenarios.finite(anchor['monotonic_after_seconds'],'invalid_source_anchor',before,1e12)
    utc = scenarios.finite(anchor['utc_unix_seconds'],'invalid_source_anchor',1,4102444800)
    require(after-before <= 1 and utc < 4102444800, 'source_anchor_sampling_uncertainty')
    sampled = sample_replay_source_clock() if sample is None else sample
    require(isinstance(sampled,dict) and sampled.get('clock') == 'ProcessInfo.systemUptime'
        and type(sampled.get('uptime_seconds')) in (int,float), 'retained_source_clock_unavailable')
    require(UUID(str(sampled.get('boot_session_id'))) == UUID(anchor['boot_session_id']),
            'retained_source_clock_boot_mismatch')
    uptime = scenarios.finite(sampled['uptime_seconds'],'retained_source_clock_unavailable',0,1e12)
    require(uptime >= after, 'retained_source_clock_future_anchor')
    duration = scenarios.finite(plan['duration_seconds'],'invalid_duration',1,MAX_DURATION)
    phase_start = scenarios.finite(phase['phase_start_seconds'],'invalid_phase_start',0,900)
    times = [scenarios.finite(item['trigger']['at_seconds'],'invalid_trigger_time',0,duration)
             for item in plan['utterances'] if item['trigger'].get('kind') == 'at_time']
    first = min(times, default=duration)
    phase_reference = ((before+phase_start)+(after+phase_start))/2
    limit = min(first,duration)
    deadline = phase_reference + limit
    return {'status':'expired' if uptime-phase_reference >= limit else 'not_yet_expired',
            'clock':'ProcessInfo.systemUptime', 'boot_session_id':anchor['boot_session_id'],
            'run_id':retained['scope']['run_id'], 'replay_invocation_id':phase['invocation_id'],
            'anchor_sha256':evidence['sha256'], 'phase_start_seconds':phase_start,
            'first_at_time_seconds':first, 'deadline_uptime_seconds':deadline,
            'sampled_uptime_seconds':uptime, 'remaining_seconds':deadline-uptime,
            'meaning':'Same-boot v1 monotonic deadline only; no UTC equality or startup-fit guarantee.'}


class RetainedCaptureFailure(RuntimeError):
    """A scoped terminal app failure, not a provider or semantic verdict."""
    def __init__(self, evidence):
        self.code = evidence['failure_code']
        self.evidence = evidence
        super().__init__(self.code)


def retained_capture_failure(path, scope, *, observation_stage):
    try:
        raw = scenarios.bounded_read(path, 16*1024*1024)
    except FileNotFoundError:
        return None
    value = decode_json(raw, 16*1024*1024)
    if not isinstance(value, dict) or value.get('final_ui_status') != 'Evaluation replay capture failed.':
        return None
    clock = value.get('replay_source_clock')
    if not isinstance(clock, dict) or clock.get('status') not in {'failed', 'bound'}:
        return None
    try:
        same_run = UUID(str(value.get('run_id'))) == UUID(str(scope['run_id']))
    except (ValueError, TypeError, AttributeError):
        same_run = False
    require(same_run and value.get('replay_invocation_id') == scope['replay_invocation_id']
        and type(clock.get('version')) is int and clock['version'] == 1
        and type(clock.get('phase_start_seconds')) in (int, float)
        and clock['phase_start_seconds'] == scope['phase_start_seconds'],
        'retained_failure_capture_scope_mismatch')
    # An origin-phase capture may arrive alongside its source anchor. Do not
    # trust the capture's hash before retain_source_anchor validates that file.
    if (scope['anchor_sha256'] is None and scope['phase_start_seconds'] == 0
            and clock.get('anchor_sha256') is not None):
        return None
    require(clock.get('anchor_sha256') == scope['anchor_sha256'],
            'retained_failure_capture_scope_mismatch')
    failures = []

    def record_code(code, field):
        require(isinstance(code, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}', code) is not None,
                'retained_failure_capture_code_invalid')
        failures.append({'field': field, 'failure_code': code})

    if clock['status'] == 'failed':
        record_code(clock.get('failure_code'), 'replay_source_clock.failure_code')
    for field in ('callback_failure', 'task_barrier_failure_code'):
        if value.get(field) is not None:
            record_code(value[field], field)
    manual = value.get('manual_input')
    if isinstance(manual, dict) and manual.get('failure_code') is not None:
        record_code(manual['failure_code'], 'manual_input.failure_code')
    utterances = value.get('utterances', [])
    require(isinstance(utterances, list) and len(utterances) <= 20,
            'retained_failure_capture_utterance_bound')
    for ordinal, item in enumerate(utterances):
        require(isinstance(item, dict), 'retained_failure_capture_utterance_invalid')
        if item.get('status') == 'failed' and item.get('failure_code') is not None:
            record_code(item['failure_code'], f'utterances[{ordinal}].failure_code')
    codes = list(dict.fromkeys(item['failure_code'] for item in failures))
    code = (clock['failure_code'] if clock['status'] == 'failed'
            else codes[0] if len(codes) == 1 else 'native_capture_failed')
    metadata = Path(path).stat()
    return {'failure_code':code, 'path':str(path), 'sha256':scenarios.digest(raw), 'bytes':len(raw),
            'run_id':scope['run_id'], 'replay_invocation_id':scope['replay_invocation_id'],
            'failure_fields':failures, 'terminal_status':value['final_ui_status'],
            'source_clock':dict(clock), 'observed_at':now(),
            'observation_stage':observation_stage, 'filesystem_mtime_ns':metadata.st_mtime_ns,
            'meaning':'Scoped terminal app failure; observation time is not first-persistence proof.'}


async def wait_retained_artifact(process, path, seconds, *, capture_path, scope, observations):
    """Watch failure capture even while waiting for a successful source anchor.

    Keep the injected process boundary used by offline runner tests. Every
    pending wait is cancelled/joined before the caller can stop the app.
    """
    def check_failure():
        evidence = retained_capture_failure(capture_path, scope, observation_stage='during_capture_wait')
        if evidence is not None:
            observations.append(evidence)
            raise RetainedCaptureFailure(evidence)
    check_failure()
    waiter = asyncio.create_task(process.wait_capture(path, seconds))
    try:
        while not waiter.done():
            check_failure()
            await asyncio.wait((waiter,), timeout=.05)
        # A terminal artifact wins over a simultaneous generic wait timeout.
        check_failure()
        await waiter
    except TimeoutError:
        observations.append({'path':str(capture_path), 'observed_at':now(),
            'observation_stage':'capture_wait_timeout_before_app_stop',
            'terminal_failure_observed':False})
        raise
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


class HostBoundary:
    def __init__(self, dsn):
        self.dsn = dsn

    async def prepare(self, options):
        return await qualify_frontend_host(dsn=self.dsn, directory=options.artifact_dir,
            hindsight_url=options.hindsight_url, api_key_file=options.hindsight_api_key_file,
            profile=NATIVE_MEDIA, prepare_only=True, lifetime_seconds=options.lease_seconds,**host_options(options))

    async def start_worker(self, options, resources):
        from evals.responsive_voice.native_task_worker import NativeTaskWorkerOptions, start
        worker_options = NativeTaskWorkerOptions(model=options.worker_model,
            api_key_file=options.worker_api_key_file,hermes_python=options.worker_python,
            artifact_dir=options.artifact_dir,reasoning_effort=options.worker_reasoning_effort,
            readiness_timeout_seconds=WORKER_READINESS_SECONDS,
            result_hold_seconds=options.result_hold_seconds if options.task_aware else None)
        return await start(options=worker_options,resources=resources,dsn=self.dsn)

    async def serve(self, manifest, stop, ready):
        return await serve_prepared_host(manifest=manifest, dsn=self.dsn, stop_event=stop,
                                          ready_callback=ready, auto_cleanup=False)

    async def cleanup(self, manifest):
        # The helper owns this single deadline, including its shutdown reserve.
        return await cleanup_prepared_host(manifest=manifest, dsn=self.dsn,
            deadline=asyncio.get_running_loop().time()+180)


def task_replay_metadata(options):
    metadata = {'format':'jarvis.native_task_replay.host.v1','barrier_id':str(uuid4()),
        'model':options.worker_model, **{name:str(options.artifact_dir/filename) for name,filename in (
            ('configuration_path','task-replay-configuration.json'),
            ('status_path','task-replay-status.json'),('stop_path','task-replay-stop.json'))}}
    require(all(not Path(metadata[key]).exists() for key in ('configuration_path','status_path','stop_path')),
            'task_control_paths_preexist')
    if options.result_release_policy != 'all_sdk_inputs':
        metadata.update(result_release_policy=options.result_release_policy,scenario=options.scenario)
    return metadata


def memory_replay_metadata(options, authorization, plan, retained):
    from evals.responsive_voice.native_memory_readiness_control import HOST_FORMAT, config_from_metadata, memory_gate_rows
    require(retained is not None and authorization.get('memory_readiness') is True,
            'memory_readiness_requires_retained_authorization')
    phase = retained['phases'][retained['index']]
    metadata = {'format':HOST_FORMAT, 'invocation_id':phase['invocation_id'],
        'manifest_sha256':authorization['input_manifest_sha256'], 'approved_gates':memory_gate_rows(plan),
        'origin_invocation_id':retained['phases'][0]['invocation_id'],
        'phase_start_seconds':phase['phase_start_seconds'],
        'source_anchor_sha256':retained['source_anchor']['sha256'] if retained['index'] else None,
        'configuration_path':str(options.artifact_dir/'memory-readiness-configuration.json'),
        'status_path':str(options.artifact_dir/'memory-readiness-status.json')}
    config_from_metadata(metadata)
    require(all(not Path(metadata[k]).exists() for k in ('configuration_path','status_path')),
            'memory_control_paths_preexist')
    return metadata


def configure_memory_replay(metadata, *, resources, staged, controls, plan, artifact_dir):
    from dataclasses import asdict
    from evals.responsive_voice.native_memory_replay_control import Scope
    from evals.responsive_voice.native_memory_readiness_control import CONFIG_FORMAT, SCOPE_NAME
    # Read actual manifested fixture fields. No schema/bank inference from run ID.
    scope = Scope(**{key:resources[key] for key in
        ('run_id','owner_id','database_schema','area_id','memory_bank_id')})
    save(staged/SCOPE_NAME, asdict(scope))
    save(Path(artifact_dir)/'inputs'/SCOPE_NAME, asdict(scope))
    configuration = {'format':CONFIG_FORMAT, 'run_id':scope.run_id,
        'invocation_id':metadata['invocation_id'], 'manifest_sha256':metadata['manifest_sha256'],
        'scope_file':str(staged/SCOPE_NAME), 'runtime_manifest_file':str(staged/'replay.json'),
        'runtime_manifest_sha256':scenarios.digest(scenarios.bounded_read(staged/'replay.json',512*1024)),
        'control_root':str(controls), 'source_anchor_path':str(Path(artifact_dir)/'source-clock.json')}
    save(Path(metadata['configuration_path']),configuration)
    return configuration


async def wait_memory_replay_status(metadata, run_id, *, timeout=10):
    from evals.responsive_voice.frontend_host import read_task_replay_file
    from evals.responsive_voice.native_memory_readiness_control import STATUS_FORMAT
    async with asyncio.timeout(timeout):
        while True:
            try:
                value = decode_json(read_task_replay_file(Path(metadata['status_path'])),16384)
            except FileNotFoundError:
                await asyncio.sleep(.05)
                continue
            require(value.get('format')==STATUS_FORMAT and value.get('run_id')==run_id
                and value.get('invocation_id')==metadata['invocation_id']
                and value.get('manifest_sha256')==metadata['manifest_sha256'], 'memory_control_status_scope')
            require(value.get('status')!='failed', 'memory_control_configuration_failed')
            if value.get('status')=='ready' and value.get('configured') is True:
                return value
            await asyncio.sleep(.05)


def configure_task_replay(metadata, *, resources, readiness, staged, controls, plan):
    from evals.responsive_voice.native_task_replay_control import NativeTaskScope
    scope = NativeTaskScope(run_id=UUID(resources['run_id']),barrier_id=UUID(metadata['barrier_id']),
        owner_id=resources['owner_id'],database_schema=resources['database_schema'],
        worker_id=readiness['worker_id'],runtime_host_id=readiness['host_id'],model=readiness['model'],
        host_expires_at=datetime.fromisoformat(readiness['host_expires_at']))
    require(isinstance(readiness.get('result_hold_control'),dict)
        and readiness['result_hold_control'].get('format')=='jarvis.native_task_result_hold.control.v1',
        'task_result_hold_not_ready')
    save(staged/'task-scope.json',scope.model_dump(mode='json'))
    speech = read_json(staged/plan['speech_manifest'])
    counts = {clip['id']:clip['sample_count'] for clip in speech['clips']}
    configuration = {'format':'jarvis.native_task_replay.configuration.v1',
        'run_id':resources['run_id'],'barrier_id':metadata['barrier_id'],
        'worker_readiness':readiness,'control_root':str(controls),
        'manifest_file':str(staged/'replay.json'),'scope_file':str(staged/'task-scope.json'),
        'input_sample_counts':[counts[item['clip_id']] for item in plan['utterances']]}
    if metadata.get('result_release_policy','all_sdk_inputs') != 'all_sdk_inputs':
        configuration.update(result_release_policy=metadata['result_release_policy'],scenario=metadata['scenario'])
    save(Path(metadata['configuration_path']),configuration)
    return configuration


async def wait_task_replay_status(metadata, run_id, phase, *, timeout=None):
    from evals.responsive_voice.native_task_release_control import read_private
    path = Path(metadata['status_path'])
    async with asyncio.timeout(TASK_CONTROL_TIMEOUT_SECONDS if timeout is None else timeout):
        while True:
            try:
                value = decode_json(read_private(path.parent,path.name),8192)
            except FileNotFoundError:
                await asyncio.sleep(.05)
                continue
            require(value.get('format')=='jarvis.native_task_replay.status.v1'
                and UUID(str(value.get('run_id')))==UUID(str(run_id))
                and UUID(str(value.get('barrier_id')))==UUID(metadata['barrier_id']), 'task_control_status_binding_mismatch')
            if phase=='ready':
                require(value.get('status')!='failed', 'task_control_configuration_failed')
            if value.get('status')==phase:
                if phase=='closed':
                    require(value.get('observer_joined') is True and value.get('release_loop_joined') is True,
                            'task_control_join_not_confirmed')
                return value
            await asyncio.sleep(.05)


def capture_replay_controls(report, controls, artifact_dir):
    """Retain declared control files without confusing memory and task scopes."""
    for section, directory, error_key, files in (
        ('task_replay', 'task-replay-observations', 'task_control_capture_error',
         (('session-binding.json', 8192), ('task-barrier.json', 16384),
          ('stimulus-completion.json', 8192))),
        ('memory_readiness', 'memory-readiness-observations', 'memory_control_capture_error',
         (('memory-readiness-request.json', 16384), ('memory-readiness-receipt.json', 32768))),
    ):
        if section not in report:
            continue
        try:
            report[section]['observations'] = [copy_bounded(controls/name,
                artifact_dir/directory/name, limit) for name, limit in files if (controls/name).is_file()]
        except Exception as error:
            report[error_key] = sanitized(error)


def retained_task_capture_evidence(report, artifact_dir):
    """Read joined producer evidence verbatim; never reconstruct bound JSON."""
    from evals.responsive_voice.frontend_host import read_task_replay_file, _private_directory_identity
    from evals.responsive_voice.native_task_barrier import MAX_RECEIPT_BYTES, decode_receipt
    from evals.responsive_voice.native_task_barrier_host import MAX_HISTORY_RECEIPTS
    from evals.responsive_voice.native_task_release_control import write_private

    root = Path(artifact_dir)
    task = report['task_replay']
    metadata = task['metadata']
    run_id, barrier_id = UUID(report['run_id']), UUID(metadata['barrier_id'])
    host = report.get('host_result', {})
    child = host.get('child', {})
    require(report.get('cleanup', {}).get('app_termination', {}).get('confirmed') is True
        and worker_closed(report.get('cleanup', {}).get('worker'))
        and 'host_stop_error' not in report and host.get('status')=='passed'
        and host.get('child_output_drain', {}).get('complete') is True
        and child.get('normal_shutdown') is True and child.get('runtime_closed') is True,
        'task_capture_producers_not_joined')
    for closed in (task.get('closed'), child.get('native_task_replay')):
        require(isinstance(closed, dict)
            and closed.get('format')=='jarvis.native_task_replay.status.v1'
            and UUID(str(closed.get('run_id')))==run_id
            and UUID(str(closed.get('barrier_id')))==barrier_id
            and closed.get('status')=='closed' and closed.get('observer_joined') is True
            and closed.get('release_loop_joined') is True and closed.get('failure_code') is None,
            'task_capture_control_not_closed_cleanly')
    require('task_control_capture_error' not in report, 'task_capture_control_archive_failed')
    configuration = Path(metadata['configuration_path'])
    require(configuration.parent==root, 'task_capture_configuration_path')
    config = decode_json(read_task_replay_file(configuration), 16384)
    staged = Path(report['staged_inputs'])
    require(UUID(str(config.get('run_id')))==run_id
        and UUID(str(config.get('barrier_id')))==barrier_id
        and config.get('manifest_file')==str(staged/'replay.json')
        and config.get('scope_file')==str(staged/'task-scope.json')
        and config.get('control_root')==task['control_root'], 'task_capture_configuration_binding')

    references = {}
    def original(path, limit):
        raw = read_task_replay_file(path, limit=limit)
        references[str(path)] = {'path':str(path), 'bytes':len(raw), 'sha256':scenarios.digest(raw)}
        return raw

    documents = {'manifest_bytes':original(staged/'replay.json', 256*1024),
                 'scope_bytes':original(staged/'task-scope.json', 8192)}
    scope = decode_json(documents['scope_bytes'], 8192)
    launch = decode_json(documents['manifest_bytes'], 256*1024)
    require(UUID(str(scope.get('run_id')))==run_id and UUID(str(launch.get('run_id')))==run_id
        and UUID(str(scope.get('barrier_id')))==barrier_id
        and scope.get('owner_id')==launch.get('owner_id'), 'task_capture_input_binding')
    # These are the already archived client control bytes, not the mutable
    # exchange or a JSON projection of its final status.
    for key, name in (('session_binding_bytes','session-binding.json'),
                      ('stimulus_completion_bytes','stimulus-completion.json')):
        path = root/'task-replay-observations'/name
        recorded = [item for item in task.get('observations', []) if item.get('path')==str(path)]
        require(len(recorded)==1, 'task_capture_control_missing')
        documents[key] = original(path, 8192)
        require(references[str(path)]['sha256']==recorded[0].get('sha256'),
                'task_capture_control_archive_changed')

    history = root/'task-barrier-history'
    history_identity = _private_directory_identity(history)
    index_bytes = original(history/'index.json', 16384)
    index = decode_json(index_bytes, 16384)
    require(isinstance(index, dict) and index.get('format')=='jarvis.native_task_barrier_history.v1'
        and index.get('root')==str(history) and UUID(str(index.get('run_id')))==run_id
        and UUID(str(index.get('barrier_id')))==barrier_id
        and index.get('filename_pattern')=='{sequence:020d}.json'
        and index.get('closed') is True and index.get('complete') is True
        and all(key in index and index[key] is None for key in ('error','pending_file','unconfirmed_sequence')),
        'task_capture_history_not_complete')
    count, maximum = index.get('receipt_count'), index.get('max_receipts')
    require(type(maximum) is int and 1<=maximum<=MAX_HISTORY_RECEIPTS
        and type(count) is int and 0<=count<=maximum
        and type(index.get('max_receipt_bytes')) is int and index['max_receipt_bytes']==MAX_RECEIPT_BYTES
        and all(type(index.get(key)) is int and index[key]==count
                for key in ('last_published_sequence','last_archived_sequence')),
        'task_capture_history_count_mismatch')
    names = set()
    with os.scandir(history) as entries:
        for entry in entries:
            require(len(names)<MAX_HISTORY_RECEIPTS+1, 'task_capture_history_inventory_bound')
            names.add(entry.name)
    expected = [f'{sequence:020d}.json' for sequence in range(1,count+1)]
    require(names=={'index.json',*expected}, 'task_capture_history_inventory_mismatch')
    receipts = []
    for sequence, name in enumerate(expected, 1):
        raw = original(history/name, MAX_RECEIPT_BYTES)
        receipt = decode_receipt(raw)
        require(receipt.sequence==sequence and receipt.scope.run_id==run_id
            and receipt.scope.barrier_id==barrier_id, 'task_capture_history_sequence_or_scope')
        receipts.append(raw)
    require(_private_directory_identity(history)==history_identity, 'task_capture_history_root_changed')
    documents['barrier_receipts'] = receipts
    retained = root/'task-replay-inputs'
    retained.mkdir(mode=0o700)
    for key, name in (('manifest_bytes','replay.json'), ('scope_bytes','task-scope.json')):
        write_private(retained, name, documents[key], exclusive=True)
        references[str(retained/name)] = {'path':str(retained/name), 'bytes':len(documents[key]),
                                         'sha256':scenarios.digest(documents[key])}
    task['analysis_inputs'] = {'manifest':str(retained/'replay.json'),
        'scope':str(retained/'task-scope.json'), 'history_index':str(history/'index.json'),
        'ordered_receipts':[str(history/name) for name in expected], 'receipt_count':count,
        'files':list(references.values()), 'literal_bytes':True,
        'canonical_association':'reader_must_validate_not_inferred_by_runner'}
    return documents


async def _run_session(options, *, host, process, retained=None):
    """Lifecycle result is NOT native/model/answer acceptance. No launch retries."""
    require((options.input_mode=='baseline' and options.scenario==SCENARIO and options.cuts is None)
        or (options.input_mode==PREPARED and options.scenario in scenarios.CASES and type(options.cuts) is int
            and (options.cuts==0 if options.scenario=='baseline' else 1<=options.cuts<=3)), 'invalid_replay_scope')
    expected_host_options = host_options(options)
    require(type(options.task_aware) is bool, 'invalid_task_replay_mode')
    if options.task_aware:
        require(options.input_mode==PREPARED
            and (options.scenario=='pause_continue' or options.scenario=='explicit_cancel' and options.cuts==1)
            and options.voice_route=='realtime' and options.workload==REASONING_REMINDER
            and type(options.result_hold_seconds) in (int,float)
            and math.isfinite(options.result_hold_seconds) and 0<options.result_hold_seconds<=60,
            'task_replay_configuration_required')
    else:
        require(options.result_hold_seconds is None, 'result_hold_requires_task_replay')
    if options.voice_route=='live':
        require(options.workload==NO_WORKLOAD and (options.input_mode=='baseline'
            or options.scenario=='baseline' and options.cuts==0), 'live_baseline_only_no_task_authority_qualification')
    if options.workload==REASONING_REMINDER:
        require(options.input_mode==PREPARED and isinstance(options.worker_api_key_file,Path)
            and options.worker_api_key_file.is_file() and isinstance(options.worker_python,Path)
            and options.worker_python.is_file(), 'prepared_workload_worker_configuration_required')
    else:
        require(options.worker_api_key_file is None and options.worker_python is None,
                'worker_configuration_without_workload')
    simulator = str(UUID(options.simulator_id)).upper()
    require(re.fullmatch(r'[A-Za-z0-9_.-]+', options.bundle_id) is not None, 'invalid_bundle_id')
    identity, authorization = validate_risk(options)
    require(retained is not None or 'retained_sequence' not in authorization,
            'sequence_authorization_requires_retained_runner')
    exporter = read_json(options.canonical_export_command)
    require(isinstance(exporter, list) and exporter and all(isinstance(arg,str) for arg in exporter)
        and any('{resource_manifest}' in arg for arg in exporter) and any('{output}' in arg for arg in exporter)
        and Path(exporter[0]).is_absolute() and Path(exporter[0]).is_file(), 'canonical_exporter_command_required')
    options.artifact_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    options.ledger_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal = options.artifact_dir/'native-run.json'
    require(not journal.exists(), 'run_journal_already_exists')
    lock = (retained['lock'] if retained is not None else
            os.open(options.ledger_dir/f'simulator-{simulator}.lock', os.O_RDWR|os.O_CREAT, 0o600))
    active = options.ledger_dir/f'simulator-{simulator}.active.json'
    report = {'format':'jarvis.native_replay.lifecycle.v1','status':'starting','started_at':now(),
        'journal':str(journal),'scenario':options.scenario,'input_mode':options.input_mode,'cuts':options.cuts,
        'native_options':expected_host_options,
        'simulator_id':simulator,'bundle_id':options.bundle_id,
        'authorization_id':identity,'risk_authorization':authorization,'launch_attempts':0,'events':[],
        'provider_evidence':provider_evidence([],False,voice_route=options.voice_route),'cleanup':{},'voice_pass':False,
        'voice_qualification':unassessed_voice_qualification()}
    if options.result_release_policy != 'all_sdk_inputs':
        report['result_release_policy'] = options.result_release_policy
        report['cancellation_qualification'] = 'unmeasured'
    def record(event, **values):
        report['events'].append({'at':now(),'event':event,**values})
        save(journal, report)
    stop, ready, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    serving = None
    manifest = retained.get('manifest') if retained is not None else None
    launch_info = None
    staged = capture_dir = container = None
    token_paths = []
    locked = False
    worker = None
    worker_stop_confirmed = True
    task_metadata = controls = None
    memory_metadata = None
    task_control_stop_confirmed = True
    capture_failure_scope = None
    try:
        if retained is None:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked = True
        if retained is None:
            require(not active.exists(), 'previous_run_cleanup_pending')
        else:
            require(active.exists() and read_json(active).get('journal')==str(retained['journal']),
                    'retained_sequence_lock_ownership_changed')
            require(identity==retained['authorization_ids'][retained['index']],
                    'retained_authorization_identity_changed')
            validate_sequence_authorization(authorization, retained, retained['index'])
        spent = options.ledger_dir/f'authorization-{identity}.spent'
        fd = os.open(spent, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,'w') as stream:
            json.dump({'journal':str(journal),'consumed_at':now()},stream)
            stream.flush()
            os.fsync(stream.fileno())
        if retained is None:
            save(active, {'journal':str(journal),'authorization_id':identity})
        report['active_ledger'] = str(active)
        record('authorization_consumed_before_allocation')
        plan, report['media'] = stage_media(options,
            expected_manifest_sha256=authorization.get('input_manifest_sha256'),
            expected_capture_mode=authorization.get('capture_mode', 'auto'),
            expected_memory_readiness=authorization.get('memory_readiness', False),
            # Both retained phases may be audio-only after the exact sequence
            # grant was revalidated above. One-shot still requires camera input.
            allow_empty_frames=retained is not None)
        tail = effective_capture_tail(plan)
        budget = capture_budget(plan['duration_seconds'],options.lease_seconds,workload=options.workload,
                                post_media_capture_seconds=tail)
        report['post_media_capture_seconds'] = tail
        report['capture_budget'] = budget
        options = replace(options,lease_seconds=budget['lease_seconds'])
        if manifest is None:
            result = await host.prepare(options)
        else:
            require(retained_scope(read_json(manifest))==retained['scope'],
                    'retained_fixture_identity_changed')
            result = {'status':'prepared','resource_manifest':manifest}
        manifest = result.get('resource_manifest')
        if retained is not None:
            retained['manifest'] = manifest
        report['resource_manifest'] = manifest
        if manifest:
            resources = read_json(manifest)
            report['cleanup_command'] = resources.get('cleanup_command')
            report['run_id'] = resources['run_id']
            if resources.get('bearer_token_file'):
                if retained is None:
                    token_paths.append(Path(resources['bearer_token_file']))
                else:
                    retained['tokens'].add(Path(resources['bearer_token_file']))
            if retained is not None:
                if 'scope' not in retained:
                    retained['scope'] = retained_scope(resources)
                require(retained_scope(resources)==retained['scope'], 'retained_fixture_identity_changed')
            report['token_files'] = [str(path) for path in token_paths]
            record('fixture_manifested')
        require(result.get('status')=='prepared' and manifest, 'host_prepare_failed')
        if authorization.get('memory_readiness') is True:
            memory_metadata = memory_replay_metadata(options,authorization,plan,retained)
            resources['native_memory_readiness'] = memory_metadata
            save(Path(manifest),resources)
            report['memory_readiness'] = {'metadata':memory_metadata,'semantic_acceptance':'not_evaluated'}
        elif 'native_memory_readiness' in resources:
            resources.pop('native_memory_readiness')
            save(Path(manifest),resources)
        if options.task_aware:
            task_metadata = task_replay_metadata(options)
            resources['native_task_replay'] = task_metadata
            save(Path(manifest),resources)
            report['task_replay'] = {'metadata':task_metadata,'semantic_acceptance':'unmeasured'}
            task_control_stop_confirmed = False
        async def on_ready(info):
            nonlocal launch_info
            launch_info = info
            ready.set()
            await release.wait()
        if retained is not None:
            retained['cleanup_safe'] = False
        serving = asyncio.create_task(host.serve(manifest, stop, on_ready))
        waiter = asyncio.create_task(ready.wait())
        try:
            done, _ = await asyncio.wait((serving,waiter),timeout=75,return_when=asyncio.FIRST_COMPLETED)
            require(waiter in done and ready.is_set(), 'host_did_not_become_ready')
        finally:
            waiter.cancel()
            await asyncio.gather(waiter,return_exceptions=True)
        require(launch_info['run_id']==report['run_id'] and launch_info.get('audio_attempt_limit')==1
            and launch_info.get('capability_profile')==NATIVE_MEDIA, 'one_attempt_native_host_required')
        if retained is not None:
            require(all(launch_info.get(key)==retained['scope'][key]
                        for key in ('run_id','owner_id','area_id','memory_bank_id')),
                    'retained_launch_scope_changed')
        require(launch_info.get('native_options')==expected_host_options
            and launch_info.get('database_pool')=={'min_size':1,'max_size':expected_host_options['native_pool_max_size']},
            'native_host_configuration_receipt_mismatch')
        if options.voice_route=='live':
            configuration=launch_info.get('provider_configuration',{})
            require(configuration.get('voice_route')=='live' and configuration.get('selected_model')==options.live_model
                and configuration.get('selected_voice')==options.live_voice and configuration.get('live_enabled') is True,
                'native_live_configuration_receipt_mismatch')
        record('native_host_ready', backend_base_url=launch_info['backend_base_url'])
        if options.workload==REASONING_REMINDER:
            worker_stop_confirmed = False
            record('task_worker_start_requested')
            try:
                worker = await host.start_worker(options,resources)
            except BaseException as error:
                failure_receipt = getattr(error,'receipt',None)
                if isinstance(failure_receipt,dict):
                    closed = failure_receipt.get('close_receipt',failure_receipt)
                    report['cleanup']['worker'] = closed
                    worker_stop_confirmed = worker_closed(closed)
                report['worker_start_resource_manifest'] = str(getattr(error,'resource_manifest',''))
                if not worker_stop_confirmed:
                    report['worker_stop_error'] = 'startup_cleanup_not_confirmed'
                raise
            report['worker_readiness'] = worker.readiness_receipt
            require_worker_readiness(worker.readiness_receipt,resources,options)
            record('task_worker_registry_ready')
        located = await process.command(['xcrun','simctl','get_app_container',simulator,options.bundle_id,'data'],
                                        options.artifact_dir/'simulator-container.log')
        require(located['returncode']==0, 'installed_app_container_unavailable')
        container = Path(located['stdout'].strip()).resolve(strict=True)
        require(container.is_dir() and container.is_absolute(), 'invalid_simulator_container')
        staged_root = container/'Documents'/'ReplayInputs'/report['run_id']
        if retained is not None:
            staged_root = staged_root/retained['phases'][retained['index']]['invocation_id']
        staged = staged_root/'inputs' if options.task_aware or memory_metadata is not None else staged_root
        capture_dir = container/'Documents'/'Replay'/str(UUID(report['run_id']))
        if retained is not None:
            capture_dir = capture_dir/retained['phases'][retained['index']]['invocation_id']
        require(not staged_root.exists() and not capture_dir.exists(), 'native_run_paths_preexist')
        staged.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        shutil.copytree(options.artifact_dir/'inputs', staged)
        if options.task_aware or memory_metadata is not None:
            controls = staged_root/'controls'
            controls.mkdir(mode=0o700)
        plan.update({key:launch_info[key] for key in ('run_id','backend_base_url','owner_id','area_id','memory_bank_id')})
        plan['bearer_token_file'] = 'isolated-token'
        if retained is not None:
            phase = retained['phases'][retained['index']]
            plan['replay_invocation_id'] = phase['invocation_id']
            clock = {'version':1, 'phase_start_seconds':phase['phase_start_seconds'],
                     'anchor_file':None, 'anchor_sha256':None}
            if retained['index']:
                anchor = retained['source_anchor']
                for destination in (options.artifact_dir/'inputs/source-clock.json', staged/'source-clock.json'):
                    copy_bounded(anchor['path'],destination,16384,anchor['sha256'])
                clock.update(anchor_file='source-clock.json',anchor_sha256=anchor['sha256'])
            plan['replay_source_clock'] = clock
            report['retained_sequence'] = {'sequence_id':retained['sequence_id'],
                'phase_index':retained['index'],'replay_invocation_id':phase['invocation_id'],
                'source_clock_binding':clock,'fixture_cleanup_owner':str(retained['journal'])}
        if memory_metadata is not None:
            plan['memory_readiness']['manifest_sha256'] = memory_metadata['manifest_sha256']
        save(options.artifact_dir/'inputs/replay.json',plan)
        save(staged/'replay.json',plan)
        if memory_metadata is not None:
            report['memory_readiness']['configuration'] = configure_memory_replay(memory_metadata,
                resources=resources,staged=staged,controls=controls,plan=plan,artifact_dir=options.artifact_dir)
            report['memory_readiness']['ready'] = await wait_memory_replay_status(memory_metadata,report['run_id'])
            record('memory_control_ready_before_launch')
        token_paths.append(staged/'isolated-token')
        report.update(token_files=[str(path) for path in token_paths], staged_inputs=str(staged),
                      native_capture_directory=str(capture_dir))
        record('token_copy_manifested_before_write')
        copy_bounded(launch_info['bearer_token_file'],token_paths[-1],4096)
        if options.task_aware:
            configure_task_replay(task_metadata,resources=resources,readiness=worker.readiness_receipt,
                                  staged=staged,controls=controls,plan=plan)
            report['task_replay']['control_root'] = str(controls)
            report['task_replay']['ready'] = await wait_task_replay_status(task_metadata,report['run_id'],'ready')
            record('task_control_ready_before_launch')
        expires = datetime.fromisoformat(launch_info['expires_at'])
        require(expires.tzinfo is not None and (expires-datetime.now(UTC)).total_seconds()
            >budget['capture_wait_seconds']+20+20+60+5+
                (WORKER_CLOSE_RESERVE if options.workload==REASONING_REMINDER else 0),
            'remaining_lease_cannot_cover_capture_and_export')
        if retained is not None:
            capture_failure_scope = {'run_id':report['run_id'],
                'replay_invocation_id':phase['invocation_id'],
                'phase_start_seconds':phase['phase_start_seconds'],
                'anchor_sha256':retained['source_anchor']['sha256'] if retained['index'] else None}
            report['capture_wait_observations'] = []
            if retained['index']:
                deadline = retained_source_deadline(plan, retained)
                report['source_clock_prelaunch'] = deadline
                record('source_clock_checked_before_launch_commit')
                require(deadline['status'] == 'not_yet_expired', 'retained_stimulus_deadline_expired')
        report['launch_attempts'] = 1
        record('single_launch_committed_before_spawn')
        launched = await process.command(['xcrun','simctl','launch',simulator,options.bundle_id,
            '--jarvis-replay-manifest',str(staged/'replay.json'),'--jarvis-replay-run'],
            options.artifact_dir/'native-launch.log')
        require(launched['returncode']==0, 'native_launch_failed_no_retry')
        if retained is not None:
            wait_started = time.monotonic()
            await wait_retained_artifact(process, capture_dir/'source-clock.json',
                budget['capture_wait_seconds'], capture_path=capture_dir/'capture.json',
                scope=capture_failure_scope, observations=report['capture_wait_observations'])
            anchor = retain_source_anchor(capture_dir/'source-clock.json',
                options.artifact_dir/'source-clock.json', retained)
            if retained['index']==0:
                retained['source_anchor'] = anchor
            report['source_anchor'] = anchor
            capture_failure_scope['anchor_sha256'] = anchor['sha256']
            remaining = budget['capture_wait_seconds']-(time.monotonic()-wait_started)
            require(remaining>0, 'retained_capture_wait_expired')
            await wait_retained_artifact(process, capture_dir/'capture.json',remaining,
                capture_path=capture_dir/'capture.json', scope=capture_failure_scope,
                observations=report['capture_wait_observations'])
            capture = read_json(capture_dir/'capture.json',16*1024*1024)
            require(UUID(capture.get('run_id',''))==UUID(report['run_id'])
                and capture.get('replay_invocation_id')==phase['invocation_id'],
                'retained_capture_identity_mismatch')
            clock_evidence = capture.get('replay_source_clock')
            report['source_clock_evidence'] = clock_evidence
            require(isinstance(clock_evidence,dict) and clock_evidence.get('version')==1
                and clock_evidence.get('status')=='bound'
                and clock_evidence.get('phase_start_seconds')==phase['phase_start_seconds']
                and clock_evidence.get('anchor_sha256')==anchor['sha256'],
                'retained_native_source_clock_not_bound')
        else:
            await process.wait_capture(capture_dir/'capture.json',budget['capture_wait_seconds'])
        report['status'] = 'capture_available'
    except asyncio.CancelledError:
        report.update(status='interrupted', interrupted=True)
    except RetainedCaptureFailure as error:
        report.update(status='failed', fatal=sanitized(error), native_capture_failure=error.evidence)
    except Exception as error:
        report.update(status='failed', fatal=sanitized(error))
    finally:
        async def finish():
            nonlocal worker_stop_confirmed, task_control_stop_confirmed
            export_ok = manifest is None
            if report['launch_attempts']:
                try:
                    outcome = await process.command(['xcrun','simctl','terminate',simulator,options.bundle_id],
                                                    options.artifact_dir/'native-terminate.log')
                    report['cleanup']['app_termination'] = {'returncode':outcome['returncode'],
                        'confirmed':outcome['returncode']==0}
                except Exception as error:
                    report['cleanup']['app_termination'] = {'error':sanitized(error),'confirmed':False}
            if task_metadata is not None:
                try:
                    save(Path(task_metadata['stop_path']),{'format':'jarvis.native_task_replay.stop.v1',
                        'run_id':report['run_id'],'barrier_id':task_metadata['barrier_id'],'requested_at':now()})
                    report['task_replay']['closed'] = await wait_task_replay_status(task_metadata,report['run_id'],'closed')
                    task_control_stop_confirmed = True
                    record('task_control_stopped_before_worker')
                except Exception as error:
                    report['task_control_stop_error'] = sanitized(error)
            if worker is not None:
                try:
                    receipt = await worker.close()
                    report['cleanup']['worker'] = receipt
                    worker_stop_confirmed = worker_closed(receipt)
                    require(worker_stop_confirmed, 'task_worker_close_not_confirmed')
                    record('task_worker_stopped_before_export')
                except Exception as error:
                    report['worker_stop_error'] = sanitized(error)
            if controls is not None:
                capture_replay_controls(report, controls, options.artifact_dir)
            if capture_dir is not None:
                try:
                    retained_outputs = []
                    for name, limit in (('capture.json',16*1024*1024),
                                        ('decoded.pcm',budget['decoded_pcm_byte_limit'])):
                        path = capture_dir/name
                        if path.is_file():
                            retained_outputs.append(copy_bounded(path,options.artifact_dir/'native-capture'/name,limit))
                    report['native_output'] = retained_outputs
                    if (options.artifact_dir/'native-capture/capture.json').is_file():
                        if capture_failure_scope is not None and 'native_capture_failure' not in report:
                            failure = retained_capture_failure(options.artifact_dir/'native-capture/capture.json',
                                capture_failure_scope, observation_stage='cleanup_after_termination_attempt')
                            if failure is not None:
                                report['native_capture_failure_after_stop'] = failure
                                report.update(status='failed', fatal=sanitized(RetainedCaptureFailure(failure)))
                        if not options.task_aware:
                            from evals.responsive_voice.native_capture_evidence import read_capture
                            try:
                                analysis = read_capture(options.artifact_dir/'native-capture/capture.json')
                                save(options.artifact_dir/'native-evidence-report.json',analysis)
                            except Exception as error:
                                report['capture_analysis_error'] = sanitized(error)
                except Exception as error:
                    report['native_output_error'] = sanitized(error)
            if manifest is not None and worker_stop_confirmed and task_control_stop_confirmed:
                try:
                    output = options.artifact_dir/'canonical-evidence.json'
                    record('canonical_export_before_host_stop')
                    argv = [arg.replace('{resource_manifest}',str(manifest)).replace('{output}',str(output)) for arg in exporter]
                    outcome = await process.command(argv,options.artifact_dir/'canonical-export.log',timeout=60)
                    evidence = read_json(output,16*1024*1024)
                    resources = read_json(manifest)
                    require(outcome['returncode']==0 and evidence.get('status')=='captured'
                        and evidence.get('run_id')==resources['run_id'] and evidence.get('owner_id')==resources['owner_id']
                        and evidence.get('schema')==resources['database_schema']
                        and evidence.get('transaction_read_only')=='on', 'canonical_export_not_qualified')
                    report['canonical_evidence'] = str(output)
                    export_ok = True
                except Exception as error:
                    report['canonical_export_error'] = sanitized(error)
            elif manifest is not None:
                report['canonical_export_error'] = ('task_worker_join_not_confirmed_export_blocked'
                    if not worker_stop_confirmed else 'task_control_join_not_confirmed_export_blocked')
            stop.set()
            release.set()
            if serving is not None:
                try:
                    result = await asyncio.wait_for(asyncio.shield(serving),65)
                    report['host_result'] = result
                except TimeoutError:
                    serving.cancel()
                    try:
                        report['host_result'] = await asyncio.wait_for(serving,65)
                    except BaseException as error:
                        report['host_stop_error'] = sanitized(error)
                except BaseException as error:
                    report['host_stop_error'] = sanitized(error)
            record('host_stop_joined')
            if options.task_aware and task_metadata is None:
                report['capture_analysis'] = {'status':'not_available',
                    'reason':'task_metadata_not_created', 'semantic_acceptance':'not_evaluated'}
            if options.task_aware and task_metadata is not None:
                report['task_replay']['capture_analysis'] = {'status':'unqualified',
                    'semantic_acceptance':'not_evaluated'}
                try:
                    require(serving is not None and serving.done()
                        and task_control_stop_confirmed and worker_stop_confirmed,
                        'task_capture_producers_not_joined')
                    from evals.responsive_voice.native_capture_evidence import read_capture
                    documents = retained_task_capture_evidence(report, options.artifact_dir)
                    analysis_path = options.artifact_dir/'native-evidence-report.json'
                    require(not analysis_path.exists(), 'task_capture_analysis_path_preexists')
                    analysis = read_capture(options.artifact_dir/'native-capture/capture.json',
                                            task_evidence=documents)
                    save(analysis_path, analysis)
                    report['task_replay']['capture_analysis'].update(status='reader_completed',
                        report_path=str(analysis_path))
                except Exception as error:
                    # Analysis never releases authority or disables the existing
                    # export, shutdown, fixture-cleanup and token-removal gates.
                    report['capture_analysis_error'] = sanitized(error)
            records = report.get('host_result',{}).get('child_output_drain',{}).get('provider_diagnostics',[])
            native_events=[event for channel in report.get('host_result',{}).get('child',{}).get('websocket_trace',[])
                if channel.get('path')=='/realtime/audio/ws' for event in channel.get('events',[])]
            report['provider_evidence'] = provider_evidence(records,bool(report['launch_attempts']),
                voice_route=options.voice_route,native_events=native_events,
                live_finalizations=report.get('host_result',{}).get('child_output_drain',{}).get('live_finalization_diagnostics',[]),
                live_finalizations_dropped=report.get('host_result',{}).get('child_output_drain',{}).get('live_finalization_diagnostics_dropped',0))
            if retained is not None and manifest is not None:
                report['cleanup']['fixture'] = {'status':'retained_for_sequence',
                    'cleanup_owner':str(retained['journal'])}
            elif manifest is not None and export_ok and worker_stop_confirmed and task_control_stop_confirmed and 'host_stop_error' not in report:
                try:
                    report['cleanup']['fixture'] = await host.cleanup(manifest)
                except Exception as error:
                    report['cleanup']['fixture'] = {'status':'failed','error':sanitized(error)}
            elif manifest is not None:
                report['cleanup']['fixture'] = {'status':'pending','reason':
                    'preserve_unjoined_worker_fixture' if not worker_stop_confirmed
                    else 'preserve_unexported_or_unstopped_fixture'}
            # Remove only the exact fresh host/staged token paths manifested by
            # this process. Cleanup does not need the frontend bearer token.
            removed = []
            for path in token_paths:
                try:
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
                except OSError as error:
                    report.setdefault('token_cleanup_errors',[]).append(sanitized(error))
            report['cleanup']['removed_token_files'] = removed
            clean = ((manifest is None or export_ok) if retained is not None else
                     (manifest is None or report['cleanup'].get('fixture',{}).get('status')=='passed'))
            clean = clean and worker_stop_confirmed and task_control_stop_confirmed and not report.get('token_cleanup_errors') and 'host_stop_error' not in report
            if report['launch_attempts']:
                clean = clean and report['cleanup'].get('app_termination',{}).get('confirmed',False)
            if retained is not None:
                clean = clean and (serving is None or report.get('host_result',{}).get('status')=='passed')
                retained['cleanup_safe'] = clean
                report['session_cleanup_complete'] = clean
                report['local_cleanup_complete'] = False
                if clean and report['status']=='capture_available':
                    try:
                        accepted = [item for item in report.get('host_result',{}).get('child',{}).get('websocket_trace',[])
                                    if item.get('path')=='/realtime/audio/ws' and item.get('accepted') is True]
                        require(len(accepted)==1 and len(accepted[0].get('session_id',[]))==1,
                                'retained_fresh_audio_admission_missing')
                        session_id = str(UUID(accepted[0]['session_id'][0]))
                        require(session_id not in retained['session_ids'], 'retained_session_reused')
                        require(not report['provider_evidence'].get('provider_attempt_budget_violated'),
                                'retained_provider_attempt_budget_violated')
                        retained['session_ids'].add(session_id)
                        report['retained_sequence']['session_id'] = session_id
                    except Exception as error:
                        report.update(status='failed',fatal=sanitized(error))
                report['status'] = ('session_complete' if clean and report['status']=='capture_available'
                                    else 'cleanup_pending' if not clean else report['status'])
            else:
                report['local_cleanup_complete'] = clean
            if retained is None and clean and active.exists() and read_json(active).get('journal')==str(journal):
                active.unlink()
            if retained is None:
                report['status'] = ('lifecycle_complete' if clean and report['status']=='capture_available'
                                    else 'cleanup_pending' if not clean else report['status'])
            record('finished', local_cleanup_complete=report['local_cleanup_complete'])
        if locked:
            cleanup_task = asyncio.create_task(finish())
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
            finally:
                if retained is None:
                    fcntl.flock(lock,fcntl.LOCK_UN)
        if retained is None:
            os.close(lock)
    return report


def effective_capture_tail(plan):
    duration = scenarios.finite(plan['duration_seconds'], 'invalid_duration', 1, MAX_DURATION)
    tail = scenarios.finite(plan.get('post_media_capture_seconds',min(20,MAX_DURATION-duration)),
                            'invalid_capture_tail', 0, 20)
    require(duration+tail<=MAX_DURATION, 'scene_and_capture_tail_exceed_bound')
    return tail


def retained_scope(resources):
    keys = ('run_id','owner_id','database_schema','area_id','memory_bank_id','bearer_token_file')
    value = {key:resources[key] for key in keys}
    run = UUID(value['run_id']).hex
    require(value['run_id']==run and value['owner_id']==f'direct-image-eval-second-brain-{run}-a'
        and value['database_schema']==f'jarvis_sb_eval_{run}_0'
        and re.fullmatch(r'jarvis_[0-9a-f]{32}',value['memory_bank_id'])
        and Path(value['bearer_token_file']).is_absolute(), 'invalid_retained_fixture_scope')
    UUID(value['area_id'])
    return value


def validate_sequence_authorization(document, retained, index):
    phase = retained['phases'][index]
    binding = document.get('retained_sequence')
    expected = {
        'sequence_id':retained['sequence_id'],
        'sequence_plan_sha256':retained['sequence_sha256'],
        'phase_index':index, 'replay_invocation_id':phase['invocation_id'],
        'phase_start_seconds':phase['phase_start_seconds'],
        'post_media_capture_seconds':phase['post_media_capture_seconds'],
        'previous_authorization_id':retained['authorization_ids'][index-1] if index else None,
        # Current provider evidence NEVER promotes HTTP hangup to remote closure.
        'accept_unconfirmed_prior_provider_closure':bool(index),
    }
    require(isinstance(binding,dict) and binding==expected
        and type(binding.get('phase_index')) is int
        and type(binding.get('phase_start_seconds')) in (int,float)
        and type(binding.get('post_media_capture_seconds')) in (int,float)
        and type(binding.get('accept_unconfirmed_prior_provider_closure')) is bool,
        'retained_sequence_authorization_mismatch')


def retain_source_anchor(source, destination, retained):
    """Retain actual app bytes, never synthesize an old clock or historical hash."""
    raw = scenarios.bounded_read(source,16384)
    anchor = decode_json(raw,16384)
    required = {'format','run_id','origin_invocation_id','monotonic_before_seconds',
                'monotonic_after_seconds','utc_unix_seconds','boot_session_id'}
    require(isinstance(anchor,dict) and set(anchor)==required
        and anchor['format']=='jarvis.replay.source_clock.v1'
        and UUID(anchor['run_id'])==UUID(retained['scope']['run_id'])
        and anchor['origin_invocation_id']==retained['phases'][0]['invocation_id'],
        'retained_source_anchor_identity_mismatch')
    UUID(anchor['boot_session_id'])  # A missing OS boot identity cannot authorize reuse.
    before = scenarios.finite(anchor['monotonic_before_seconds'],'invalid_source_anchor',0,1e12)
    after = scenarios.finite(anchor['monotonic_after_seconds'],'invalid_source_anchor',before,1e12)
    scenarios.finite(anchor['utc_unix_seconds'],'invalid_source_anchor',1,1e12)
    require(after-before<=1, 'source_anchor_sampling_uncertainty')
    checksum = scenarios.digest(raw)
    if retained['index']:
        require(checksum==retained['source_anchor']['sha256'], 'retained_source_anchor_changed')
    result = copy_bounded(source,destination,16384,checksum)
    result['origin_invocation_id'] = anchor['origin_invocation_id']
    result['boot_session_id'] = anchor['boot_session_id']
    return result


def load_retained_sequence(options):
    """Only a two-admission, same-day native sequence; no virtual-clock override."""
    require(options.input_mode==PREPARED and options.scenario=='baseline' and options.cuts==0
        and options.voice_route=='realtime' and options.workload in {NO_WORKLOAD,VISUAL_MEMORY}
        and not options.task_aware and options.result_hold_seconds is None
        and options.worker_api_key_file is None and options.worker_python is None,
        'retained_sequence_requires_prepared_realtime_baseline')
    raw = scenarios.bounded_read(options.retained_sequence,16384)
    document = decode_json(raw,16384)
    require(isinstance(document,dict) and set(document)=={'format','sequence_id','phases'}
        and document['format']=='jarvis.native_replay.retained_sequence.v1'
        and isinstance(document['phases'],list) and len(document['phases'])==2,
        'invalid_retained_sequence')
    sequence_id = str(UUID(document['sequence_id']))
    require(document['sequence_id']==sequence_id, 'canonical_sequence_uuid_required')
    phases, phase_options, authorizations, ids, invocation_ids = [], [], [], [], set()
    end = 0
    for index,item in enumerate(document['phases']):
        require(isinstance(item,dict) and set(item)=={
            'invocation_id','phase_start_seconds','input_manifest','media_root','speech_manifest',
            'risk_authorization'}, 'invalid_retained_phase')
        invocation = str(UUID(item['invocation_id']))
        require(item['invocation_id']==invocation and invocation not in invocation_ids,
                'unique_canonical_invocation_required')
        invocation_ids.add(invocation)
        start = scenarios.finite(item['phase_start_seconds'],'invalid_phase_start',0,900)
        require((index==0 and start==0) or (index>0 and start>=end), 'overlapping_or_rebased_phase')
        paths = {}
        for key in ('input_manifest','media_root','speech_manifest','risk_authorization'):
            require(isinstance(item[key],str) and Path(item[key]).is_absolute(), 'absolute_phase_path_required')
            paths[key] = Path(item[key]).resolve()
        if index==0:
            require(all(paths[key]==getattr(options,key).resolve() for key in paths),
                    'first_phase_cli_binding_mismatch')
        selected = replace(options,**paths,retained_sequence=None,
            artifact_dir=options.artifact_dir/'sessions'/f'{index:02d}-{invocation}')
        template = read_json(selected.input_manifest)
        capture_mode = declared_capture_mode(template)
        from evals.responsive_voice.native_memory_readiness_control import declared_memory_readiness
        memory_enabled = declared_memory_readiness(template)
        if capture_mode == 'manual':
            require_manual_capture_scope(selected, retained_phase=True)
        duration = scenarios.finite(template['duration_seconds'],'invalid_duration',1,MAX_DURATION)
        require(isinstance(template.get('utterances'),list) and template['utterances']
            and template['utterances'][0].get('trigger',{}).get('kind')=='at_time'
            and all(isinstance(row,dict) and row.get('trigger',{}).get('kind') in
                    ({'at_time','after_manual_response'} if memory_enabled else {'at_time'})
                    for row in template['utterances']), 'retained_phase_requires_timed_speech')
        tail = effective_capture_tail(template)
        end = start+duration
        identity, authorization = validate_risk(selected)
        require(authorization.get('capture_mode', 'auto') == capture_mode,
                'capture_mode_authorization_mismatch')
        require(authorization.get('memory_readiness', False) is memory_enabled,
                'memory_readiness_authorization_mismatch')
        require(identity not in ids, 'retained_authorization_reused')
        ids.append(identity)
        phases.append({'invocation_id':invocation,'phase_start_seconds':start,
                       'duration_seconds':duration,'post_media_capture_seconds':tail})
        phase_options.append(selected)
        authorizations.append(authorization)
    minimum = math.ceil(end+sum(p['post_media_capture_seconds'] for p in phases))+2*LEASE_OVERHEAD+180
    lease = minimum if options.lease_seconds is None else options.lease_seconds
    require(type(lease) is int and minimum<=lease<=900, 'retained_sequence_lease_bound')
    retained = {'sequence_id':sequence_id,'sequence_sha256':scenarios.digest(raw),'phases':phases,
        'authorization_ids':ids,'manifest':None,'tokens':set(),'session_ids':set(),'cleanup_safe':True,
        'lease_seconds':lease,'minimum_lease_seconds':minimum}
    for index,document in enumerate(authorizations):
        validate_sequence_authorization(document,retained,index)
    return retained,[replace(selected,lease_seconds=lease) for selected in phase_options]


async def run_retained_sequence(options, *, host, process):
    """Run the SAME native session operation twice, allocate/clean the fixture once.

    No semantic acceptance, alternate provider path, new ledger or hidden retries.
    Source-anchored timing is enforced by the debug app, never shifted by this loop.
    """
    retained, phases = load_retained_sequence(options)
    options.artifact_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
    options.ledger_dir.mkdir(mode=0o700,parents=True,exist_ok=True)
    journal = options.artifact_dir/'native-sequence.json'
    require(not journal.exists(), 'sequence_journal_already_exists')
    simulator = str(UUID(options.simulator_id)).upper()
    lock = os.open(options.ledger_dir/f'simulator-{simulator}.lock',os.O_RDWR|os.O_CREAT,0o600)
    active = options.ledger_dir/f'simulator-{simulator}.active.json'
    retained.update(lock=lock,journal=journal)
    report = {'format':'jarvis.native_replay.retained_sequence.lifecycle.v1','status':'starting',
        'journal':str(journal),'sequence_id':retained['sequence_id'],
        'sequence_plan_sha256':retained['sequence_sha256'],'lease_seconds':retained['lease_seconds'],
        'minimum_lease_seconds':retained['minimum_lease_seconds'],'phases':[],
        'launch_attempts':0,'provider_evidence':{'phases':[],'remote_closure_confirmed':False},
        'cleanup':{},'local_cleanup_complete':False,'voice_pass':False,
        'voice_qualification':unassessed_voice_qualification(),'memory_semantics':'not_evaluated',
        'calendar_simulation':'not_enabled','fixed_source_deadlines':'never_rebased'}
    locked = False
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        locked = True
        require(not active.exists(), 'previous_run_cleanup_pending')
        save(active,{'journal':str(journal),'sequence_id':retained['sequence_id'],
                     'authorization_ids':retained['authorization_ids']})
        save(journal,report)
        async with asyncio.timeout(retained['lease_seconds']-180):
            for index,selected in enumerate(phases):
                retained['index'] = index
                result = await _run_session(selected,host=host,process=process,retained=retained)
                report['phases'].append(result)
                report['launch_attempts'] += result['launch_attempts']
                report['provider_evidence']['phases'].append(result['provider_evidence'])
                report['resource_manifest'] = retained.get('manifest')
                save(journal,report)
                if (result['status']!='session_complete' or result.get('native_capture_failure')
                        or result.get('native_capture_failure_after_stop')):
                    report['status'] = 'failed'
                    break
            else:
                report['status'] = 'sessions_complete'
    except asyncio.CancelledError:
        report.update(status='interrupted',interrupted=True)
    except Exception as error:
        report.update(status='failed',fatal=sanitized(error))
    finally:
        async def finish():
            manifest = retained.get('manifest')
            if manifest is not None:
                if retained['cleanup_safe']:
                    try:
                        report['cleanup']['fixture'] = await host.cleanup(manifest)
                    except Exception as error:
                        report['cleanup']['fixture'] = {'status':'failed','error':sanitized(error)}
                else:
                    report['cleanup']['fixture'] = {'status':'pending',
                        'reason':'preserve_unjoined_or_unexported_retained_fixture'}
            removed = []
            for path in retained['tokens']:
                try:
                    path.unlink(missing_ok=True)
                    removed.append(str(path))
                except OSError as error:
                    report.setdefault('token_cleanup_errors',[]).append(sanitized(error))
            report['cleanup']['removed_token_files'] = removed
            clean = retained['cleanup_safe'] and not report.get('token_cleanup_errors') and (
                manifest is None or report['cleanup'].get('fixture',{}).get('status')=='passed')
            report['local_cleanup_complete'] = clean
            if clean and active.exists() and read_json(active).get('journal')==str(journal):
                active.unlink()
            report['status'] = ('lifecycle_complete' if clean and report['status']=='sessions_complete'
                                else 'cleanup_pending' if not clean else report['status'])
            save(journal,report)
        if locked:
            cleanup_task = asyncio.create_task(finish())
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)
        os.close(lock)
    return report


async def run_once(options, *, host, process):
    if options.retained_sequence is not None:
        return await run_retained_sequence(options,host=host,process=process)
    return await _run_session(options,host=host,process=process)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-manifest','media-root','speech-manifest','artifact-dir','risk-authorization','canonical-export-command',
                 'hindsight-api-key-file'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--hindsight-url',required=True)
    parser.add_argument('--ledger-dir',type=Path,
        default=Path(__file__).resolve().parents[3]/'.jarvis-dev/native-replay-authorization-ledger')
    parser.add_argument('--input-mode',choices=('baseline',PREPARED),default='baseline',
                        help='baseline preserves old input mode; prepared-scenario accepts compiler runtime manifests')
    parser.add_argument('--scenario',choices=(SCENARIO,*scenarios.CASES),default=SCENARIO)
    parser.add_argument('--cuts',type=int,help='prepared-scenario only: 0 for baseline, otherwise 1, 2 or 3')
    parser.add_argument('--native-pool-max-size',type=int,choices=range(1,6),
                        help='native pool defaults to min1/max5; explicit 1 reproduces contention')
    parser.add_argument('--workload',choices=(NO_WORKLOAD,REASONING_REMINDER,VISUAL_MEMORY),default=NO_WORKLOAD)
    parser.add_argument('--worker-model',help='explicit Main-approved frontier model; matched by host and worker')
    parser.add_argument('--worker-reasoning-effort',choices=('low','medium','high','xhigh','max','ultra'),default='low')
    parser.add_argument('--worker-api-key-file',type=Path,help='caller-owned source; helper copies privately, never deletes source')
    parser.add_argument('--worker-python',type=Path,help='pinned Hermes interpreter, not the p0 supervisor interpreter')
    parser.add_argument('--voice-route',choices=('realtime','live'),default='realtime',
                        help='explicit Live baseline opt-in; no automatic route fallback or extra admission')
    parser.add_argument('--live-model',help='must match fresh Live authorization and isolated host receipt')
    parser.add_argument('--live-voice',help='must match fresh Live authorization and isolated host receipt')
    parser.add_argument('--task-aware',action='store_true',help='explicit committed-claim/SDK-completion replay opt-in')
    parser.add_argument('--result-hold-seconds',type=float,help='task-aware only: bounded returned-result hold, up to 60 seconds')
    parser.add_argument('--result-release-policy',choices=('all_sdk_inputs','no_auto_release'),default='all_sdk_inputs',
                        help='no_auto_release requires a fresh matching explicit_cancel task-aware grant; no cancellation is injected')
    parser.add_argument('--retained-sequence',type=Path,
        help='explicit two-session source-anchored plan; fresh existing authorization per admission')
    parser.add_argument('--simulator-id',default='7642363B-3872-44E7-B5CE-09802F2AF3A7')
    parser.add_argument('--bundle-id',default='com.symbiote.jarvis')
    parser.add_argument('--lease-seconds',type=int,
                        help='default covers media + bounded app capture tail +240; worker/sequence reserves additional')
    parser.add_argument('--execute-once',action='store_true')
    args = vars(parser.parse_args(argv))
    require(args.pop('execute_once'), 'explicit_execute_once_required_no_automatic_launch')
    # Dereferencing a virtualenv's Python symlink selects the base interpreter
    # and loses its pinned Hermes packages. Canonicalize data paths, not Python.
    options = RunOptions(**{
        key: (value.absolute() if key == 'worker_python' else value.resolve())
        if isinstance(value, Path) else value
        for key, value in args.items()
    })
    from app.settings import JarvisSettings
    async def execute():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        cancelled = False
        def interrupt():
            nonlocal cancelled
            if not cancelled:
                cancelled = True
                task.cancel()
        for signum in (signal.SIGINT,signal.SIGTERM):
            loop.add_signal_handler(signum,interrupt)
        try:
            return await run_once(options, host=HostBoundary(JarvisSettings().SUPABASE_DB_URL), process=ProcessBoundary())
        finally:
            for signum in (signal.SIGINT,signal.SIGTERM):
                loop.remove_signal_handler(signum)
    result = asyncio.run(execute())
    print(json.dumps({'status':result['status'],'journal':result['journal'],
        'launch_attempts':result['launch_attempts'],'local_cleanup_complete':result.get('local_cleanup_complete'),
        'provider_evidence':result['provider_evidence']}))
    return 0 if result['status']=='lifecycle_complete' else 1


if __name__=='__main__':
    raise SystemExit(main())
