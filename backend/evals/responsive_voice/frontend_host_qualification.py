"""Bounded real-SQL/HTTP qualification of the normal frontend replay host.

Owns one manifested schema, one previously absent native Hindsight bank, and one
loopback child server. Default qualification never enables model credentials.
The explicit native profile passes configured credentials to a private child;
only the native client initiates provider work. No production monkeypatch or
public migration. Configuration/admission is NOT a voice or model-quality pass.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import socket
import sys
import tempfile
import traceback
from urllib.parse import parse_qs, quote, urlsplit
from uuid import UUID, uuid4

import asyncpg
import httpx
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.attention.admission_diagnostics import shutdown_snapshot
from app.knowledge.hindsight import HindsightClient
from app.knowledge.service_entrypoint import secret_file
from app.migrations.runner import apply_migrations, load_migrations, quote_schema
from evals.responsive_voice.frontend_host import (
    ReplayHostScope, NativeTaskReplayMetadata, create_replay_application, scoped_database_url,
    _private_directory_identity, _task_document,
)
from evals.second_brain.fixtures import DatabaseFixture
from evals.responsive_voice.model_wire_trace import install_from_environment

install_from_environment()


NO_MEDIA = 'admission_only_no_media'
NATIVE_MEDIA = 'native_realtime_media'
NO_WORKLOAD = 'none'
REASONING_REMINDER = 'reasoning-reminder'
VISUAL_MEMORY = 'visual-memory'
VISUAL_MEMORY_REQUIRED_TOOLS = ('track_visual_memory', 'list_episode_checks', 'change_episode_check')


def native_host_options(profile, *, native_pool_max_size=None, workload=NO_WORKLOAD,
                        frontier_model=None, frontier_reasoning_effort=None,
                        voice_route='realtime', live_model=None, live_voice=None):
    """Explicit fixture configuration, never inherited production capability flags."""
    require(profile in {NO_MEDIA,NATIVE_MEDIA}, 'invalid_capability_profile')
    maximum = (5 if profile==NATIVE_MEDIA else 1) if native_pool_max_size is None else native_pool_max_size
    require(type(maximum) is int and 1<=maximum<=5 and (profile==NATIVE_MEDIA or maximum==1),
            'invalid_native_pool_capacity')
    require(workload in {NO_WORKLOAD,REASONING_REMINDER,VISUAL_MEMORY}, 'invalid_native_workload')
    require(voice_route in {'realtime','live'}, 'invalid_native_voice_route')
    if workload==VISUAL_MEMORY:
        require(profile==NATIVE_MEDIA and voice_route=='realtime',
                'visual_memory_requires_native_realtime')
    if voice_route=='live':
        from app.settings import is_configured_secret
        require(profile==NATIVE_MEDIA and workload==NO_WORKLOAD and all(
            isinstance(value,str) and 0<len(value)<=256 and is_configured_secret(value)
            and not any(char.isspace() or ord(char)<32 for char in value)
            for value in (live_model,live_voice)), 'explicit_live_baseline_configuration_required')
    else:
        require(live_model is None and live_voice is None, 'live_configuration_without_live_route')
    if workload==REASONING_REMINDER:
        require(profile==NATIVE_MEDIA and isinstance(frontier_model,str)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}',frontier_model) is not None
            and frontier_reasoning_effort in {'low','medium','high','xhigh','max','ultra'},
            'explicit_native_frontier_configuration_required')
    else:
        require(frontier_model is None and frontier_reasoning_effort is None,
                'frontier_configuration_without_workload')
    return {'native_pool_max_size':maximum,'workload':workload,'frontier_model':frontier_model,
            'frontier_reasoning_effort':frontier_reasoning_effort,'voice_route':voice_route,
            'live_model':live_model,'live_voice':live_voice}


def visual_memory_runtime_qualification(runtime, *, flags, owner_id):
    """Check normal initialized ownership; never substitute for session tool advertisement."""
    from app.reminders.episode_checks import EpisodeCheckOwner
    from app.reminders.episode_tools import EpisodeCheckToolModule
    from app.runtime.feature_config import Capability

    settings = runtime.settings
    selected = flags.resolve(Capability.RESPONSIBILITIES, database_url=settings.SUPABASE_DB_URL)
    require(selected.enabled and selected.configured
        and settings.RESPONSIVE_VOICE_ENABLED is True,
        'native_visual_memory_configuration_unavailable')
    require(settings.JARVIS_OWNER_ID==owner_id and runtime.db_pool is not None,
        'native_visual_memory_owner_scope_mismatch')
    require(settings.REMINDER_SCHEDULER_ENABLED is False and all(
        not flags.resolve(capability, database_url=settings.SUPABASE_DB_URL).enabled
        for capability in (Capability.FRONTIER, Capability.ATTENTION_VOICE, Capability.PUSH,
                           Capability.MAIL, Capability.SANDBOX, Capability.ARTIFACTS)),
        'native_visual_memory_unexpected_delivery_or_worker')
    modules = tuple(module for module in getattr(runtime, 'tool_modules', ())
                    if isinstance(module, EpisodeCheckToolModule))
    require(len(modules)==1, 'native_visual_memory_episode_module_missing_or_ambiguous')
    owner = modules[0].owner
    require(isinstance(owner, EpisodeCheckOwner) and owner.enabled
        and owner.capability.status=='available', 'native_visual_memory_episode_owner_unavailable')
    require(owner.repository.pool is runtime.db_pool, 'native_visual_memory_owner_pool_mismatch')
    # No synthetic session ID, new registry, tool invocation or provider write.
    # The ASGI observer sees client control traffic, not the provider's tool list.
    # In particular, a ready module must not manufacture an advertisement pass.
    return {'schema_version':1, 'episode_owner_status':'available',
        'registered_module':'episode_checks', 'owner_and_pool_match':True,
        'required_session_tool_names':list(VISUAL_MEMORY_REQUIRED_TOOLS),
        'advertised_session_tool_names':None, 'session_advertisement_status':'unobserved',
        'session_advertisement_qualified':False, 'frontier_worker_required':False,
        'meaning':'Normal owner/module readiness only. Actual session tool advertisement remains unqualified.'}


def configure_native_diagnostic_logging(profile):
    if profile==NATIVE_MEDIA:
        # Only existing sanitized transport/lifecycle metadata loggers. No
        # global INFO, DEBUG, HTTP headers, provider bodies or SDP logging.
        logging.getLogger('app.realtime.webrtc').setLevel(logging.INFO)
        logging.getLogger('app.realtime.telemetry').setLevel(logging.INFO)
        logging.getLogger('app.realtime.live_session.finalization').setLevel(logging.INFO)
        logging.getLogger('app.realtime.request_source_diagnostics').setLevel(logging.INFO)
        logging.getLogger('app.attention.native_voice').setLevel(logging.INFO)



def sanitize_live_finalization(value):
    """Closed schema for an owned child's authenticated Live-reader summary."""
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version']!=1 or value.get('stage')!='live_finalization'
            or value.get('voice_route')!='live'):
        return None
    result = {'schema_version':1, 'stage':'live_finalization', 'voice_route':'live'}
    for key in ('audio_session_id','provider_session_id'):
        candidate = value.get(key)
        result[key] = candidate if (isinstance(candidate,str)
            and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',candidate)
            and not candidate.lower().startswith('sk-')) else None
    for key in ('jarvis_session_id','route_id'):
        candidate = value.get(key)
        try:
            result[key] = str(UUID(candidate)) if (isinstance(candidate,str)
                and re.fullmatch(r'[a-fA-F0-9-]{36}',candidate)) else None
        except ValueError:
            result[key] = None
    candidate = value.get('route_generation')
    result['route_generation'] = candidate if type(candidate) is int and 1<=candidate<=2**31-1 else None
    for key in ('creation_submitted','session_closed_received'):
        candidate = value.get(key)
        result[key] = candidate if type(candidate) is bool else None
    reasons = {'close_requested','expired','content','remote_hangup','connection_lost','unknown',
               'receipt_unavailable','provider_not_requested','receipt_binding_mismatch'}
    candidate = value.get('reason')
    result['reason'] = candidate if isinstance(candidate,str) and candidate in reasons else None
    result['receipt_kind'] = 'session.closed' if value.get('receipt_kind')=='session.closed' else None
    for key, maximum in (('usage_seconds',86_400),('shutdown_elapsed_ms',120_000)):
        candidate = value.get(key)
        result[key] = candidate if type(candidate) in (int,float) and 0<=candidate<=maximum else None
    confirmed = (value.get('finalization_outcome')=='confirmed'
        and result['creation_submitted'] is True and result['session_closed_received'] is True
        and result['receipt_kind']=='session.closed' and result['reason'] in {
            'close_requested','expired','content','remote_hangup','connection_lost','unknown'}
        and all(result[key] is not None for key in (
            'audio_session_id','jarvis_session_id','route_id','route_generation','provider_session_id')))
    not_requested = (value.get('finalization_outcome')=='not_requested'
        and result['creation_submitted'] is False and result['session_closed_received'] is False
        and result['provider_session_id'] is None and result['receipt_kind'] is None
        and result['reason']=='provider_not_requested')
    result['finalization_outcome'] = 'confirmed' if confirmed else 'not_requested' if not_requested else 'unknown'
    # Ignore any caller-provided boolean, local shutdown state or HTTP status.
    result['remote_closure_confirmed'] = confirmed
    return result


def sanitized_live_finalization_diagnostic(line):
    if not isinstance(line,str):
        return None
    try:
        if len(line.encode('utf-8'))>8192:
            return None
    except UnicodeError:
        return None
    if (not re.search(r'(?:^|\s)app\.realtime\.live_session\.finalization(?:\s|:)',line)):
        return None
    match = re.search(r'live_finalization_diagnostic=(\{.*\})\s*$',line)
    if not match:
        return None
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_diagnostic_key')
            result[key] = value
        return result
    try:
        value = json.loads(match[1],object_pairs_hook=unique_keys)
    except (ValueError,RecursionError):
        return None
    return sanitize_live_finalization(value)


def sanitized_webrtc_diagnostic(line):
    # The pipe reader already caps lines at 8 KiB. Apply the same character
    # bound to direct callers before parsing; never project arbitrary log text.
    if not isinstance(line, str) or len(line)>8192:
        return None
    if not any(name in line for name in ('app.realtime.webrtc', 'app.realtime.session')):
        return None
    match = re.search(r'(?:transport_diagnostic|diagnostic)=(\{.*\})\s*$', line)
    if not match:
        return None
    try:
        value = json.loads(match[1])
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    result = {'remote_closure_confirmed':False}
    for key, allowed in {
        'stage':{'create_http','validate_call_identity','validate_sdp_answer','sideband_connect',
                 'provider_control_error','post_attach_setup','hangup_http'},
        'creation_outcome':{'not_requested','requested','created','rejected','unknown'},
        'closure_outcome':{'not_requested','requested','accepted','failed','unknown'},
        'reason':{'connection_is_single_use','request_started','request_cancelled_without_response',
                  'request_failed_without_response','unexpected_http_status','call_identity_unavailable',
                  'invalid_sdp_answer','attach_cancelled','attach_failed','session_setup_failed',
                  'provider_error_event','call_identity_or_transport_unavailable',
                  'request_cancelled_after_response','request_failed_after_response',
                  'teardown_accepted_not_confirmed_closed','undocumented_success_status',
                  'hangup_http_failure','hangup_deadline_without_response','hangup_deadline_after_response'},
        'http_phase':{'not_started','request_dispatch','connect_tcp','tls','request_headers',
                      'request_body','response_headers','response_body'},
        'http_phase_state':{'not_started','started','complete','failed'},
        'deadline_source':{'http_phase','hangup_total','cancelled'},
        'exception_type':{'ConnectTimeout','ReadTimeout','WriteTimeout','PoolTimeout','ConnectError',
                          'ReadError','WriteError','CloseError','RemoteProtocolError','LocalProtocolError',
                          'ProxyError','UnsupportedProtocol','DecodingError','TooManyRedirects',
                          'TimeoutError','CancelledError','other'},
        'response_body_outcome':{'not_read','not_needed','read','too_large','invalid_json','failed','cancelled'},
        'response_close_outcome':{'not_requested','closed','deadline','failed','cancelled'},
        'http_close_outcome':{'not_requested','borrowed','closed','deadline','failed','cancelled'},
    }.items():
        candidate = value.get(key)
        result[key] = candidate if isinstance(candidate, str) and candidate in allowed else None
    if result['stage'] is None:
        return None
    for key in ('request_headers_sent','request_body_sent','response_headers_received','caller_cancelled'):
        candidate = value.get(key)
        result[key] = candidate if type(candidate) is bool else None
    for key in ('hangup_elapsed_ms','hangup_total_timeout_ms','http_phase_timeout_ms'):
        candidate = value.get(key)
        # Reject, never clamp. NaN/infinity, bools, strings and implausibly large
        # timings are unavailable evidence, not fabricated measured durations.
        result[key] = candidate if type(candidate) in (int,float) and 0<=candidate<=60_000 else None
    for key in ('http_status','creation_http_status'):
        candidate = value.get(key)
        result[key] = candidate if type(candidate) is int and 100<=candidate<=599 else None
    for key in ('call_id','provider_request_id','creation_request_id'):
        candidate = value.get(key)
        pattern = r'rtc_[A-Za-z0-9_-]{1,180}' if key=='call_id' else r'(?:req_[A-Za-z0-9_-]{1,128}|[a-fA-F0-9-]{32,36})'
        result[key] = candidate if isinstance(candidate, str) and re.fullmatch(pattern, candidate) and 'sk-' not in candidate.lower() else None
    return result
COUNTED_RELATIONS = ('jarvis_knowledge_sources', 'jarvis_context_events',
    'jarvis_background_tasks', 'visual_frames', 'perception_observations', 'jarvis_attention_intents')


async def count_owned_rows(connection, owner, diagnostics, relations=COUNTED_RELATIONS):
    counts = {}
    for relation in relations:
        require(re.fullmatch(r'[a-z_]+', relation) is not None, 'invalid_inventory_relation')
        query = f'select count(*) from "{relation}" where owner_id=$1'
        try:
            counts[relation] = await connection.fetchval(query, owner)
        except asyncpg.PostgresError as error:
            # Written only to the private artifact, never the stdout fatal summary.
            diagnostics['private_sql_error'] = {'relation':relation, 'query':query,
                'schema':await connection.fetchval('select current_schema()'),
                'message':error.message, 'sqlstate':error.sqlstate,
                'server_table':getattr(error, 'table_name', None)}
            raise
    return counts


class QualificationFailure(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise QualificationFailure(code)


def sanitized(error):
    return {'error_type':type(error).__name__, 'code':getattr(error, 'code', None),
        'sqlstate':getattr(error, 'sqlstate', None),
        'frames':[{'file':Path(frame.filename).name, 'line':frame.lineno, 'function':frame.name}
                  for frame in traceback.extract_tb(error.__traceback__)[-6:]]}


def save_json(path, value):
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write('\n')


async def native_bank_inventory(client):
    # Same supported protocol as memory_lifecycle_qualification.py. A bank-root
    # GET is not supported and a 405 must NEVER be interpreted as absence.
    response = await client._call('GET', '/v1/default/banks')
    rows = response.get('banks')
    require(isinstance(rows, list) and all(isinstance(row, dict)
        and isinstance(row.get('bank_id'), str) and row['bank_id'] for row in rows),
        'native_bank_inventory_invalid')
    identities = {row['bank_id'] for row in rows}
    require(len(identities)==len(rows), 'native_bank_inventory_ambiguous')
    return identities


async def cleanup_owned_native_bank(bank, *, hindsight_url, api_key_file, client_factory=HindsightClient,
                                    budget=None):
    require(re.fullmatch(r'jarvis_[0-9a-f]{32}', bank) is not None, 'invalid_owned_bank_identity')
    key = secret_file(str(api_key_file))
    deleter = client_factory(hindsight_url, key, timeout=10, private_http_origin=hindsight_url)
    try:
        await _cleanup_step(budget, 'bank_delete',
            lambda: deleter._call('DELETE', deleter._bank(bank), missing=True), limit=10)
    finally:
        await _cleanup_close_all(budget, [('bank_delete_client_close', deleter.close)],
                                 primary_error=sys.exception())
    audit = client_factory(hindsight_url, key, timeout=10, private_http_origin=hindsight_url)
    try:
        identities = await _cleanup_step(budget, 'bank_independent_audit',
            lambda: native_bank_inventory(audit), limit=10)
        require(bank not in identities, 'independent_native_bank_cleanup_failed')
    finally:
        await _cleanup_close_all(budget, [('bank_audit_client_close', audit.close)],
                                 primary_error=sys.exception())
    return {'native_bank_absent':True, 'independent_client_audit':True}


async def storage_inventory(connection, owner):
    rows = await connection.fetch('select id,session_id,storage_bucket,storage_path from visual_frames '
                                  'where owner_id=$1 order by id', owner)
    items = []
    for row in rows:
        path = row['storage_path']
        items.append({'frame_id':str(row['id']), 'session_id':str(row['session_id']),
            'bucket':row['storage_bucket'], 'key':path,
            'matches_owned_prefix':isinstance(path, str) and path.startswith(f'owners/{owner}/sessions/')})
    return {'status':'recorded_objects' if items else 'no_recorded_objects',
        'objects':items, 'deletion_performed':False}


def native_settings(*, voice_route='realtime', live_model=None, live_voice=None):
    from app.settings import JarvisSettings, is_configured_secret
    from app.storage.supabase_config import get_supabase_storage_config
    settings = JarvisSettings()
    require(settings.AUDIO_PROVIDER_DEFAULT=='openai', 'native_profile_requires_configured_openai_route')
    require(is_configured_secret(settings.OPENAI_API_KEY)
        and (voice_route=='live' or is_configured_secret(settings.OPENAI_REALTIME_MODEL))
        and is_configured_secret(settings.GEMINI_API_KEY), 'native_provider_configuration_missing')
    if voice_route=='live':
        from app.realtime.live_selection import select_live_voice_route
        from app.realtime.provider_selection import AudioProvider
        effective=settings.model_copy(update={'RESPONSIVE_VOICE_ENABLED':True,'OPENAI_LIVE_ENABLED':True,
                                             'OPENAI_LIVE_MODEL':live_model,'OPENAI_LIVE_VOICE':live_voice})
        select_live_voice_route(settings=effective,requested_values=['live'],
                               provider=AudioProvider.OPENAI,transport='webrtc')
    get_supabase_storage_config(settings)
    return settings


def native_provider_configuration(settings, *, voice_route):
    """Typed host receipt after normal route validation, not provider acceptance."""
    require(voice_route in {'realtime','live'}, 'invalid_native_voice_route')
    if voice_route=='live':
        from app.realtime.live_selection import select_live_voice_route
        from app.realtime.provider_selection import AudioProvider
        # Settings deliberately accepts bool|string. Use the existing selector's
        # strict boolean parser rather than truthiness or weakening runner checks.
        select_live_voice_route(settings=settings,requested_values=['live'],
                                provider=AudioProvider.OPENAI,transport='webrtc')
    return {'openai_key_present':bool(settings.OPENAI_API_KEY),
        'gemini_key_present':bool(settings.GEMINI_API_KEY),
        'realtime_model':settings.OPENAI_REALTIME_MODEL,
        'audio_provider':settings.AUDIO_PROVIDER_DEFAULT,
        'responsive_voice_enabled':settings.RESPONSIVE_VOICE_ENABLED,
        'voice_route':voice_route,
        'selected_model':settings.OPENAI_LIVE_MODEL if voice_route=='live' else settings.OPENAI_REALTIME_MODEL,
        'selected_voice':settings.OPENAI_LIVE_VOICE if voice_route=='live' else settings.OPENAI_REALTIME_VOICE,
        'live_enabled':True if voice_route=='live' else settings.OPENAI_LIVE_ENABLED,
        'provider_execution_qualified':False}



def configured_storage(resources):
    from app.settings import JarvisSettings
    from app.storage.supabase_config import get_supabase_storage_config
    config = get_supabase_storage_config(JarvisSettings())
    require(config.supabase_url.rstrip('/')==resources['storage']['origin']
        and config.visual_frame_bucket==resources['storage']['bucket'], 'storage_configuration_changed')
    return config


def storage_http(config):
    return httpx.AsyncClient(base_url=config.supabase_url.rstrip('/'), timeout=20, trust_env=False,
        headers={'Authorization':'Bearer '+config.service_role_key, 'apikey':config.service_role_key})


def owned_storage_key(resources, key):
    owner, run = resources['owner_id'], resources['run_id']
    require(re.fullmatch(r'[0-9a-f]{32}', run) is not None
        and owner in {f'direct-image-eval-second-brain-{run}-a', f'direct-image-eval-second-brain-{run}-b'},
        'storage_owner_not_fixture_scoped')
    prefix = f'owners/{owner}/sessions/'
    require(resources['storage']['prefix']==prefix, 'storage_prefix_mismatch')
    require(isinstance(key, str) and key.startswith(prefix), 'storage_key_outside_owned_prefix')
    parts = key[len(prefix):].split('/')
    require(len(parts)==4 and parts[1]=='frames' and re.fullmatch(r'[0-9]{12}\.jpg', parts[3]) is not None,
        'storage_key_not_normal_frame_path')
    for identity in (parts[0], parts[2]):
        try:
            valid = str(UUID(identity))==identity
        except ValueError:
            valid = False
        require(valid, 'storage_frame_identity_invalid')
    return key


async def list_owned_storage(client, resources):
    """Bounded recursive listing only below this never-shared fixture owner."""
    prefix = resources['storage']['prefix']
    # Validate scope even when the remote listing is empty.
    owned_storage_key(resources, prefix+str(UUID(int=0))+'/frames/'+str(UUID(int=0))+'/000000000000.jpg')
    bucket = resources['storage']['bucket']
    require(isinstance(bucket, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', bucket) is not None,
        'storage_bucket_invalid')
    queue, seen, keys, requests = [prefix.rstrip('/')], set(), set(), 0
    while queue:
        folder = queue.pop()
        require(folder not in seen and len(seen)<512, 'storage_listing_directory_bound')
        seen.add(folder)
        offset = 0
        while True:
            requests += 1
            require(requests<=1024, 'storage_listing_request_bound')
            response = await client.post('/storage/v1/object/list/'+quote(bucket, safe=''), json={
                'prefix':folder, 'limit':100, 'offset':offset, 'sortBy':{'column':'name','order':'asc'}})
            require(response.status_code==200, 'storage_inventory_http_'+str(response.status_code))
            rows = response.json()
            require(isinstance(rows, list) and len(rows)<=100, 'storage_inventory_invalid')
            for row in rows:
                name = row.get('name') if isinstance(row, dict) else None
                require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', name) is not None
                    and name not in {'.','..'}, 'storage_inventory_unsafe_name')
                key = folder+'/'+name
                if row.get('id') is None:
                    require(len(key.split('/'))<=7, 'storage_directory_depth_bound')
                    queue.append(key)
                else:
                    owned_storage_key(resources, key)
                    require(key not in keys and len(keys)<2048, 'storage_inventory_object_bound')
                    keys.add(key)
            if len(rows)<100:
                break
            offset += len(rows)
    return sorted(keys)


async def cleanup_owned_storage(resources, sql_items, persist, *, config=None, client_factory=storage_http,
                                budget=None):
    """Persist exact identities before deleting; audit using an independent client."""
    if not resources.get('storage'):
        require(not sql_items, 'storage_configuration_missing_for_recorded_objects')
        return {'status':'no_media_profile', 'objects':[], 'deletion_performed':False}
    bucket = resources['storage']['bucket']
    combined = {}
    for item in [*resources.get('storage_cleanup', {}).get('objects', []), *sql_items]:
        key = owned_storage_key(resources, item['key'])
        require(item['bucket']==bucket, 'storage_key_bucket_mismatch')
        combined[key] = {**item, 'key':key, 'bucket':bucket}
    config = config or configured_storage(resources)
    result = {'status':'enumerating', 'objects':list(combined.values()), 'deletion_performed':False,
        'prefix':resources['storage']['prefix'], 'independent_client_audit':False}
    resources['storage_cleanup'] = result
    persist()
    try:
        async with _cleanup_storage_client(client_factory, config, budget, 'storage_delete_client') as deleter:
            listed = await _cleanup_step(budget, 'storage_inventory',
                lambda: list_owned_storage(deleter, resources))
            sql_keys = {item['key'] for item in sql_items}
            for key in listed:
                combined.setdefault(key, {'key':key, 'bucket':bucket, 'origin':'prefix_only_no_sql_row'})
            result.update(status='manifested_before_delete', objects=sorted(combined.values(), key=lambda item:item['key']),
                sql_key_count=len(sql_keys), prefix_key_count=len(listed),
                prefix_only_key_count=len(set(listed)-sql_keys))
            persist()
            keys = sorted(combined)
            for start in range(0, len(keys), 100):
                batch = keys[start:start+100]
                response = await _cleanup_step(budget, 'storage_delete_batch',
                    lambda: deleter.request('DELETE', '/storage/v1/object/'+quote(bucket, safe=''),
                                            json={'prefixes':batch}),
                    limit=20, batch_start=start, object_count=len(batch))
                require(response.status_code==200, 'storage_delete_http_'+str(response.status_code))
            result.update(status='deleted_awaiting_audit', deletion_performed=bool(keys))
            persist()
        async with _cleanup_storage_client(client_factory, config, budget, 'storage_audit_client') as audit:
            remaining = await _cleanup_step(budget, 'storage_independent_audit',
                lambda: list_owned_storage(audit, resources))
        result.update(remaining_keys=remaining, independent_client_audit=True)
        require(not remaining, 'independent_storage_absence_audit_failed')
        result['status'] = 'passed'
        persist()
        return result
    except (Exception, asyncio.CancelledError) as error:
        result.update(status='incomplete', fatal=sanitized(error))
        persist()
        raise


CLEANUP_TIMEOUT_SECONDS = 180
CLEANUP_SHUTDOWN_SECONDS = 10
CLEANUP_CLOSE_SECONDS = 3  # At most source client, source pool and outer SQL connection.
CLEANUP_WORKER_BATCH_SIZE = 8
CLEANUP_WORKER_STEP_LIMIT = 180


class _CleanupDeadline:
    """One process-local deadline, with a reserved allowance for owned I/O joins."""

    def __init__(self, result, persist, *, deadline=None, shutdown_seconds=None):
        self.clock = asyncio.get_running_loop().time
        self.started = self.clock()
        self.deadline = self.started + CLEANUP_TIMEOUT_SECONDS if deadline is None else deadline
        reserve = CLEANUP_SHUTDOWN_SECONDS if shutdown_seconds is None else shutdown_seconds
        require(type(self.deadline) in {int, float} and math.isfinite(self.deadline)
                and self.deadline <= self.started + CLEANUP_TIMEOUT_SECONDS,
                'invalid_cleanup_deadline')
        require(type(reserve) in {int, float} and math.isfinite(reserve)
                and 0 < reserve <= CLEANUP_SHUTDOWN_SECONDS, 'invalid_cleanup_shutdown_allowance')
        self.work_deadline = self.deadline - reserve
        self.persist = persist
        self.shutdown_failed = False
        self.timing = result['timing'] = {
            'clock': 'same_process_event_loop_monotonic', 'started_monotonic': self.started,
            'deadline_monotonic': self.deadline, 'work_deadline_monotonic': self.work_deadline,
            'budget_seconds': max(0, self.deadline-self.started),
            'shutdown_allowance_seconds': reserve, 'step_count': 0,
            'steps': [], 'dropped_steps': 0, 'active_step': None,
            'failure_step': None, 'terminal_status': None,
        }
        self.persist()

    def progress(self, **values):
        self.timing['progress'] = values
        self.persist()

    async def step(self, name, factory, *, limit=None, shutdown=False, **identity):
        started = self.clock()
        if shutdown:
            limit = min(CLEANUP_CLOSE_SECONDS, limit if limit is not None else CLEANUP_CLOSE_SECONDS)
        deadline = self.deadline if shutdown else self.work_deadline
        if limit is not None:
            deadline = min(deadline, started+limit)
        self.timing['step_count'] += 1
        step = {
            'ordinal': self.timing['step_count'], 'name': name, 'status': 'started',
            'started_monotonic': started, 'deadline_monotonic': deadline,
            'remaining_seconds': deadline-started, 'shutdown': shutdown, **identity,
        }
        self.timing['steps'].append(step)
        if len(self.timing['steps']) > 64:
            self.timing['steps'].pop(0)
            self.timing['dropped_steps'] += 1
        self.timing['active_step'] = step
        try:
            self.persist()  # Durable before the next await or any side effect.
            if deadline <= self.clock():
                raise TimeoutError('cleanup_budget_exhausted')
            async with asyncio.timeout_at(deadline):
                # Deliver cancellation requested between steps before starting I/O.
                await asyncio.sleep(0)
                value = await factory()
                if self.clock() >= deadline:
                    raise TimeoutError('cleanup_step_deadline_exhausted')
            step['status'] = 'completed'
            return value
        except (Exception, asyncio.CancelledError) as error:
            step['status'] = ('cancelled' if isinstance(error, asyncio.CancelledError)
                              else 'timed_out' if isinstance(error, TimeoutError) else 'failed')
            step['error'] = sanitized(error)
            self.shutdown_failed |= shutdown
            if self.timing['failure_step'] is None:
                self.timing['failure_step'] = step
            self.timing['terminal_status'] = step['status']
            raise
        finally:
            step['finished_monotonic'] = self.clock()
            step['elapsed_seconds'] = step['finished_monotonic']-started
            self.timing['active_step'] = None
            self.persist()


async def _cleanup_step(budget, name, factory, **kwargs):
    return await budget.step(name, factory, **kwargs) if budget is not None else await factory()


async def _cleanup_close_all(budget, entries, *, primary_error):
    first = None
    for name, close in entries:
        try:
            await _cleanup_step(budget, name, close, limit=5, shutdown=True)
        except (Exception, asyncio.CancelledError) as error:
            first = first or error
    if first is not None and primary_error is None:
        raise first


@asynccontextmanager
async def _cleanup_storage_client(factory, config, budget, name):
    manager = factory(config)
    client = await _cleanup_step(budget, name+'_open', manager.__aenter__, limit=10)
    try:
        yield client
    finally:
        info = sys.exc_info()
        await _cleanup_close_all(budget, [(name+'_close', lambda: manager.__aexit__(*info))],
                                 primary_error=info[1])


async def cleanup_source_originals(dsn, resources, *, budget=None):
    """Use canonical deletion/outbox owners; never start a retention/model job."""
    from app.knowledge.deletion import SourceDeletionService
    from app.knowledge.sources import SourceRepository
    from app.knowledge.outbox import OutboxRepository
    from app.knowledge.ingestion import IngestionWorker
    budget = budget or _CleanupDeadline({}, lambda: None)
    pool = client = None
    try:
        pool = await budget.step('source_pool_open',
            lambda: asyncpg.create_pool(scoped_database_url(dsn, resources['database_schema']),
                                       min_size=1, max_size=1, command_timeout=30), limit=10)
        sources = SourceRepository(pool)
        deletions = SourceDeletionService(sources)
        owner, area = resources['owner_id'], UUID(resources['area_id'])
        async def inventory():
            async with pool.acquire() as connection:
                return await connection.fetch('select id,area_id,current_revision,status from jarvis_knowledge_sources '
                                              'where owner_id=$1 order by id limit 257', owner)
        rows = await budget.step('source_inventory', inventory, limit=30)
        require(len(rows)<=256 and all(row['area_id']==area for row in rows), 'source_cleanup_scope_or_bound')
        budget.progress(source_count=len(rows), iteration=0)
        if not rows:
            return {'status':'no_sources', 'source_count':0, 'model_submission':False}
        for row in rows:
            if row['status']!='deleted':
                await budget.step('source_delete_request',
                    lambda: deletions.request(owner, area, row['id'], expected_revision=row['current_revision']),
                    limit=30, source_id=str(row['id']), revision=row['current_revision'])
        client = HindsightClient(resources['hindsight_url'], secret_file(resources['hindsight_api_key_file']),
            timeout=20, private_http_origin=resources['hindsight_url'])
        worker = IngestionWorker(OutboxRepository(sources), client)
        worker_id = 'frontend-cleanup-'+resources['run_id']
        worker_steps = inspection_passes = 0
        while True:
            inspection_passes += 1
            receipts = []
            for row in rows:
                receipt = await budget.step('source_inspect',
                    lambda: deletions.inspect(owner, area, row['id']), limit=30,
                    source_id=str(row['id']), revision=row['current_revision'],
                    iteration=worker_steps, inspection_pass=inspection_passes)
                require(receipt.get('source_id') == str(row['id'])
                    and type(receipt.get('revision')) is int
                    and receipt['revision'] == row['current_revision'],
                    'source_cleanup_target_changed')
                receipts.append(receipt)
            states = {}
            for receipt in receipts:
                states[receipt['state']] = states.get(receipt['state'], 0)+1
            budget.progress(source_count=len(rows), iteration=worker_steps,
                inspection_passes=inspection_passes, worker_steps=worker_steps,
                worker_step_limit=CLEANUP_WORKER_STEP_LIMIT,
                worker_batch_limit=CLEANUP_WORKER_BATCH_SIZE, source_states=states)
            # No cached "done" set: every target, including already-deleted rows,
            # must pass its original canonical inspection in this final scan.
            if all(receipt['state']=='scoped_cleanup_recorded' for receipt in receipts):
                return {'status':'scoped_cleanup_recorded', 'source_count':len(rows),
                    'receipts':receipts, 'model_submission':False}
            require(not any(receipt['state']=='requires_attention' for receipt in receipts),
                    'source_cleanup_requires_attention')
            require(worker_steps < CLEANUP_WORKER_STEP_LIMIT, 'source_cleanup_drain_bound')
            # Keep normal claim/lease/fence/unknown-outcome handling in the worker.
            # Eight serial steps amortize the expensive full-target audit without
            # multiplying the existing 180-step total or changing its deadline.
            for position in range(min(CLEANUP_WORKER_BATCH_SIZE,
                                      CLEANUP_WORKER_STEP_LIMIT-worker_steps)):
                outcome = await budget.step('source_worker_step',
                    lambda: worker.step(owner, worker_id), limit=30,
                    iteration=worker_steps, inspection_pass=inspection_passes,
                    batch_position=position)
                worker_steps += 1
                if outcome == 'failed':
                    break  # Inspect exact targets immediately; never hide errors.
                if outcome is None:
                    # No eligible work is not completion. Preserve bounded idle
                    # polling while avoiding one full-source scan per empty claim.
                    await budget.step('source_poll_interval', lambda: asyncio.sleep(.5),
                        limit=1, iteration=worker_steps, inspection_pass=inspection_passes)
            if outcome is not None and outcome != 'failed':
                await budget.step('source_poll_interval', lambda: asyncio.sleep(.5),
                    limit=1, iteration=worker_steps, inspection_pass=inspection_passes)
    finally:
        entries = []
        if client is not None:
            entries.append(('source_client_close', client.close))
        if pool is not None:
            entries.append(('source_pool_close', pool.close))
        await _cleanup_close_all(budget, entries, primary_error=sys.exception())


class ChildOutputDrain:
    """One reader per pipe for the entire child lifetime, even after log cap.

    Control notices are parsed independently of retained bytes. File writes run
    off the event loop; stdout/stderr share one bounded private log budget.
    """
    TRUNCATION_MARKER = b'\n[FRONTEND_HOST_LOG_TRUNCATED: middle omitted; see discarded_bytes; pipe drainage continues]\n'

    def __init__(self, child, log, *, byte_limit=1024*1024):
        require(type(byte_limit) is int and byte_limit>=len(self.TRUNCATION_MARKER), 'invalid_child_log_bound')
        require(child.stdout is not None and child.stderr is not None, 'child_output_pipes_required')
        self.child, self.log, self.byte_limit = child, log, byte_limit
        self._payload_limit = byte_limit-len(self.TRUNCATION_MARKER)
        self._head_limit = min(256*1024, byte_limit//4, self._payload_limit)
        self._tail_limit = self._payload_limit-self._head_limit
        self._head, self._tail = bytearray(), bytearray()
        self._tail_starts_line = True
        self._retention_finalized = False
        self._boundary_discarded = 0
        self._retained = self._discarded = self._written = 0
        self._truncated = self._log_failed = self._joined = False
        self._bytes = {'stdout':0, 'stderr':0}
        self._eof = {'stdout':False, 'stderr':False}
        self._errors = []
        self._line = b''
        self._discard_line = False
        self._lock = asyncio.Lock()
        self.ready = asyncio.get_running_loop().create_future()
        self.exit_status = None
        self.provider_diagnostics = []
        self.provider_diagnostics_dropped = 0
        self.live_finalization_diagnostics = []
        self.live_finalization_diagnostics_dropped = 0
        self._tasks = tuple(asyncio.create_task(self._drain(name, stream), name='frontend-child-'+name)
            for name, stream in (('stdout',child.stdout), ('stderr',child.stderr)))

    def _control_bytes(self, chunk):
        pieces = chunk.split(b'\n')
        for index, part in enumerate(pieces):
            if not self._discard_line:
                if len(self._line)+len(part)<=8192:
                    self._line += part
                else:
                    self._line = b''
                    self._discard_line = True
            if index==len(pieces)-1:
                continue
            line = self._line if not self._discard_line else b''
            self._line, self._discard_line = b'', False
            diagnostic = sanitized_webrtc_diagnostic(line.decode('utf-8', errors='replace'))
            if diagnostic is not None:
                if len(self.provider_diagnostics)==64:
                    self.provider_diagnostics.pop(32)
                    self.provider_diagnostics_dropped += 1
                self.provider_diagnostics.append(diagnostic)
            live = sanitized_live_finalization_diagnostic(line.decode('utf-8', errors='replace'))
            if live is not None:
                if len(self.live_finalization_diagnostics)==16:
                    self.live_finalization_diagnostics.pop(8)
                    self.live_finalization_diagnostics_dropped += 1
                self.live_finalization_diagnostics.append(live)
            if line.startswith(b'FRONTEND_HOST_READY '):
                try:
                    notice = json.loads(line[len(b'FRONTEND_HOST_READY '):])
                    require(type(notice.get('pid')) is int and notice['pid']==self.child.pid,
                            'child_ready_pid_mismatch')
                except Exception:
                    if not self.ready.done():
                        self.ready.set_exception(QualificationFailure('invalid_child_ready_notice'))
                else:
                    if not self.ready.done():
                        self.ready.set_result(notice)
            elif line.startswith(b'FRONTEND_HOST_EXIT '):
                try:
                    notice = json.loads(line[len(b'FRONTEND_HOST_EXIT '):])
                    self.exit_status = notice.get('status')
                except (ValueError, AttributeError):
                    self.exit_status = 'invalid_exit_notice'
                if not self.ready.done():
                    self.ready.set_exception(QualificationFailure('child_exited_before_ready'))

    def _write(self, data):
        count = self.log.write(data)
        self.log.flush()
        return count

    async def _retain(self, chunk):
        async with self._lock:
            take = min(len(chunk), self._head_limit-len(self._head))
            self._head.extend(chunk[:take])
            if take:
                self._tail_starts_line = self._head[-1:] == b'\n'
            self._tail.extend(chunk[take:])
            dropped = max(0, len(self._tail)-self._tail_limit)
            if dropped:
                self._tail_starts_line = self._tail[dropped-1] == 10
                del self._tail[:dropped]
                self._discarded += dropped
                self._truncated = True
            self._retained = len(self._head)+len(self._tail)

    async def _flush_retained(self):
        if self._retention_finalized:
            return
        self._retention_finalized = True
        head, tail = bytes(self._head), bytes(self._tail)
        if self._truncated:
            # Never join two partial JSON/log lines across the omitted middle.
            head = head[:head.rfind(b'\n')+1]
            if not self._tail_starts_line:
                newline = tail.find(b'\n')
                tail = tail[newline+1:] if newline >= 0 else b''
            self._boundary_discarded = self._retained-len(head)-len(tail)
            self._discarded += self._boundary_discarded
        self._retained = len(head)+len(tail)
        data = head+(self.TRUNCATION_MARKER if self._truncated else b'')+tail
        self._head.clear()
        self._tail.clear()
        try:
            written = await asyncio.to_thread(self._write, data)
            self._written += written
            require(written==len(data), 'short_child_log_write')
        except Exception as error:
            self._log_failed = True
            self._errors.append({'stage':'private_log_write', **sanitized(error)})
            # No retry of a possibly partial write; both pipes are already joined.

    async def _drain(self, name, stream):
        try:
            while True:
                chunk = await stream.read(16384)
                if not chunk:
                    self._eof[name] = True
                    break
                self._bytes[name] += len(chunk)
                if name=='stdout':
                    self._control_bytes(chunk)
                await self._retain(chunk)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._errors.append({'stage':name+'_read', **sanitized(error)})
        finally:
            if name=='stdout' and not self.ready.done():
                self.ready.set_exception(QualificationFailure('child_stdout_ended_before_ready'))

    async def wait_ready(self, timeout=70):
        return await asyncio.wait_for(asyncio.shield(self.ready), timeout)

    async def wait_closed(self, timeout=10):
        if not self._joined:
            try:
                await asyncio.wait_for(asyncio.gather(*self._tasks), timeout)
            except TimeoutError as error:
                self._errors.append({'stage':'pipe_eof_timeout', **sanitized(error)})
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._joined = True
            if self.ready.done() and not self.ready.cancelled():
                self.ready.exception()
        await self._flush_retained()
        return self.diagnostics()

    def diagnostics(self):
        return {'byte_limit':self.byte_limit, 'bytes_read':dict(self._bytes),
            'retained_log_bytes':self._written, 'discarded_bytes':self._discarded,
            'retention':'fixed_head_rolling_tail', 'head_byte_limit':self._head_limit,
            'tail_byte_limit':self._tail_limit, 'retained_source_bytes':self._retained,
            'line_boundary_discarded_bytes':self._boundary_discarded,
            'retention_finalized':self._retention_finalized,
            'retention_complete':self._retention_finalized and not self._log_failed,
            'truncated':self._truncated, 'drained_to_eof':dict(self._eof),
            'complete':self._joined and all(self._eof.values()), 'errors':list(self._errors),
            'exit_notice_status':self.exit_status,
            'provider_diagnostics':list(self.provider_diagnostics),
            'provider_diagnostics_dropped':self.provider_diagnostics_dropped,
            'live_finalization_diagnostics':list(self.live_finalization_diagnostics),
            'live_finalization_diagnostics_dropped':self.live_finalization_diagnostics_dropped,
            'meaning':'Log retention is bounded; discarded output is still drained. Local exit is not provider closure.'}


def native_route_observation(payload):
    """Bounded server-side transport metadata, never SDP, captions or credentials."""
    def identifier(value):
        return value if (isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,256}',value)
                         and not value.startswith('sk-')) else None
    result={'type':identifier(payload.get('type')),'code':identifier(payload.get('code')),
            'observed_at':datetime.now(UTC).isoformat()}
    if payload.get('type') == 'voice_route.event':
        event = payload.get('payload')
        if isinstance(event, dict):
            from app.voice_routes.contracts import (
                CommandReceipt, ContextReceipt, RouteConnectionEvent, RouteError,
                TranscriptRevision,
            )
            from app.realtime.voice_route_runtime import context_receipt_diagnostic
            models = {'connection_state': RouteConnectionEvent, 'context_receipt': ContextReceipt,
                'command_receipt': CommandReceipt, 'route_error': RouteError,
                'transcript_revision': TranscriptRevision}
            kind = event.get('kind')
            if isinstance(kind, str) and kind in models:
                # A schema check is not current-route authority, playback or finality.
                # Only fixed enums, typed identities and local event times survive.
                detail = {'kind': kind, 'schema_valid': False}
                result['route_event'] = detail
                try:
                    typed = models[kind].model_validate(event)
                except (ValueError, TypeError):
                    if kind == 'context_receipt':
                        result['context_receipt_invalid'] = True
                else:
                    detail.update(schema_valid=True, route_id=str(typed.route.route_id),
                        route_generation=typed.route.route_generation,
                        event_observed_at=typed.observed_at.isoformat())
                    codes = {'live_start_cancelled', 'live_start_unconfirmed',
                        'context_replacement_requires_fenced_reconnect', 'notification_source_replaced',
                        'delegation_result_submission_unconfirmed', 'delegation_result_revoked_after_submission',
                        'append_capacity_exceeded', 'append_cancelled', 'append_write_unconfirmed',
                        'uncorrelated_live_provider_error', 'live_reader_failed',
                        'remote_context_expired_requires_replacement', 'live_deadline_monitor_failed',
                        'live_append_acknowledgment_unknown', 'live_event_backpressure',
                        'provider_finalized', 'provider_finalization_unconfirmed', 'route_closed',
                        'source_retirement', 'close_or_output_fence_unconfirmed'}
                    if isinstance(typed, ContextReceipt):
                        result['context_receipt'] = context_receipt_diagnostic(typed)
                    elif isinstance(typed, TranscriptRevision):
                        detail.update(transcript_id=str(typed.transcript_id), revision=typed.revision,
                            role=typed.role, final=typed.final, character_count=len(typed.text))
                    elif isinstance(typed, RouteError):
                        detail.update(code=typed.code if typed.code in codes else 'unrecognized_code',
                            recoverable=typed.recoverable)
                    else:
                        detail['reason'] = (typed.reason if typed.reason in codes else
                            None if typed.reason is None else 'unrecognized_code')
                        if isinstance(typed, RouteConnectionEvent):
                            detail['state'] = typed.state
                        elif isinstance(typed, CommandReceipt):
                            detail.update(command_id=str(typed.command_id), operation=typed.operation,
                                status=typed.status, output_fenced=typed.output_fenced,
                                provider_stopped=typed.provider_stopped)
    if payload.get('type') in {'voice_route.close', 'voice_route.interrupt'}:
        from app.voice_routes.contracts import CloseRouteCommand, InterruptSpeechCommand
        model = CloseRouteCommand if payload['type'] == 'voice_route.close' else InterruptSpeechCommand
        detail = {'schema_valid': False}
        result['route_control'] = detail
        try:
            command = model.model_validate(payload.get('payload'))
        except (ValueError, TypeError):
            pass
        else:
            reasons = {'remote_context_expired_requires_replacement', 'context_replacement_requires_fenced_reconnect',
                'live_start_cancelled', 'live_start_unconfirmed', 'live_reader_failed',
                'live_append_acknowledgment_unknown', 'live_deadline_monitor_failed', 'live_event_backpressure',
                'append_write_unconfirmed', 'append_cancelled', 'append_capacity_exceeded',
                'notification_source_replaced', 'delegation_result_submission_unconfirmed',
                'delegation_result_revoked_after_submission', 'uncorrelated_live_provider_error',
                'source_retirement', 'route_closed', 'live_session_closed', 'live_source_context_revoked',
                'live_context_requires_reconstruction', 'live_session_worker_failed', 'client_stop',
                'client_requested_close', 'client_disconnected', 'source_retirement_requires_reconnect',
                'live_control_capacity_exceeded', 'live_interrupt_requires_fenced_replacement',
                'live_readonly_remote_source_revoked'}
            detail.update(schema_valid=True, route_id=str(command.route.route_id),
                route_generation=command.route.route_generation, command_id=str(command.command_id),
                requested_at=command.requested_at.isoformat(),
                reason=command.reason if command.reason in reasons else 'unrecognized_code')
    if payload.get('type') in {'audio.provider.selected','voice_route.selected','live.media.prepare','live.media.activate'}:
        for key in ('audio_session_id','provider','model','provider_session_id','command_id'):
            if key in payload:
                result[key]=identifier(payload[key])
        route=payload.get('route')
        if isinstance(route,dict):
            result['route_id']=identifier(route.get('route_id'))
            generation=route.get('route_generation')
            result['route_generation']=generation if type(generation) is int and generation>0 else None
        if payload.get('type')=='voice_route.selected':
            from app.voice_routes.contracts import RouteCapabilities
            try:
                result['capabilities']=RouteCapabilities.model_validate(payload.get('capabilities')).model_dump(mode='json')
            except (ValueError,TypeError):
                result['capabilities_invalid']=True
    return result


class AdmissionTrace:
    """Observe real ASGI transport metadata; forward every message unchanged."""
    def __init__(self, host, *, audio_attempt_limit=None, expected_voice_route=None):
        self.host, self.websockets, self.lifecycle = host, [], {}
        self.audio_attempt_limit, self.audio_attempts = audio_attempt_limit, 0
        self.expected_voice_route=expected_voice_route

    async def __call__(self, scope, receive, send):
        if scope['type']=='websocket' and scope['path']=='/realtime/audio/ws' and self.audio_attempt_limit is not None:
            if self.audio_attempts>=self.audio_attempt_limit:
                await send({'type':'websocket.close', 'code':1008, 'reason':'evaluation_audio_attempt_limit'})
                return
            # Reserve before delegation. Even rejected/ambiguous admission burns
            # the one budget; normal host authentication and owner gates remain.
            self.audio_attempts += 1
            requested=parse_qs(scope.get('query_string',b'').decode('ascii')).get('voice_route',[])
            if self.expected_voice_route is not None and not (
                    requested==[self.expected_voice_route]
                    or self.expected_voice_route=='realtime' and not requested):
                await send({'type':'websocket.close','code':1008,'reason':'evaluation_voice_route_mismatch'})
                return
        record = None
        if scope['type']=='websocket':
            record = {'path':scope['path'], 'accepted':False, 'events':[],
                'session_id':parse_qs(scope.get('query_string', b'').decode('ascii')).get('session_id', []),
                'requested_voice_route':parse_qs(scope.get('query_string',b'').decode('ascii')).get('voice_route',[]),
                'close_code':None, 'close_reason':None}
            self.websockets.append(record)
        async def observed_send(message):
            kind = message['type']
            if kind.startswith('lifespan.'):
                self.lifecycle[kind] = True
                if kind=='lifespan.startup.failed':
                    codes = re.findall(r'(?:ReplayHostError|SourceError|TaskError): ([a-z_]+)',
                                       message.get('message', ''))
                    self.lifecycle['startup_failure_code'] = codes[-1] if codes else None
            if record is not None:
                if kind=='websocket.accept':
                    record['accepted'] = True
                elif kind=='websocket.close':
                    record.update(close_code=message.get('code'), close_reason=message.get('reason'))
                elif kind=='websocket.send':
                    if message.get('bytes') is not None:
                        record['events'].append({'type':'binary'})
                    elif message.get('text'):
                        payload = json.loads(message['text'])
                        record['events'].append(native_route_observation(payload))
            await send(message)
        await self.host(scope, receive, observed_send)


def task_replay_child_environment(resources, scope, artifact_directory):
    """Forward only the frozen opt-in metadata, never reread a mutating manifest in the child."""
    if 'native_task_replay' not in resources:
        return {}
    options = resources.get('native_options', {})
    require(resources.get('capability_profile') == NATIVE_MEDIA
        and options.get('workload') == REASONING_REMINDER
        and options.get('voice_route', 'realtime') == 'realtime', 'task_replay_profile_mismatch')
    require((resources.get('run_id'), resources.get('owner_id'), resources.get('database_schema')) ==
            (scope.run_id, scope.owner_id, scope.database_schema), 'task_replay_resource_scope_mismatch')
    metadata = _task_document(NativeTaskReplayMetadata, json.dumps(resources['native_task_replay']).encode())
    require(metadata.model == options.get('frontier_model'), 'task_replay_model_mismatch')
    directory = Path(artifact_directory)
    require(directory.is_absolute() and metadata.configuration_path.parent.is_relative_to(directory),
            'task_replay_operator_path_outside_artifacts')
    _private_directory_identity(metadata.configuration_path.parent)
    return {'JARVIS_FRONTEND_REPLAY_TASK_CONTROL': metadata.model_dump_json()}


def memory_readiness_child_environment(resources, scope, artifact_directory):
    if 'native_memory_readiness' not in resources:
        return {}
    from evals.responsive_voice.native_memory_readiness_control import config_from_metadata
    options = resources.get('native_options', {})
    require(resources.get('capability_profile') == NATIVE_MEDIA
        and options.get('workload') in {NO_WORKLOAD, VISUAL_MEMORY}
        and options.get('voice_route', 'realtime') == 'realtime'
        and 'native_task_replay' not in resources, 'memory_readiness_profile_mismatch')
    require(all(str(resources.get(key)) == str(getattr(scope, key)) for key in
        ('run_id','owner_id','database_schema','area_id','memory_bank_id')), 'memory_readiness_resource_scope_mismatch')
    config = config_from_metadata(resources['native_memory_readiness'])
    directory = Path(artifact_directory)
    if config.configuration_path.parent != directory:
        # A retained two-admission sequence keeps its fixture in phase 00,
        # while each invocation owns separate control files. Accept only that
        # exact identity-bound sibling, not arbitrary descendants or siblings.
        require(directory.is_absolute() and directory.parent.name == 'sessions'
            and directory.name == f'00-{config.origin_invocation_id}'
            and config.invocation_id != config.origin_invocation_id
            and config.phase_start_seconds > 0 and config.source_anchor_sha256 is not None
            and config.configuration_path == directory.parent / f'01-{config.invocation_id}'
                / 'memory-readiness-configuration.json'
            and config.status_path.name == 'memory-readiness-status.json',
            'memory_readiness_metadata_path')
        _private_directory_identity(directory)
    _private_directory_identity(config.configuration_path.parent)
    return {'JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL': json.dumps(resources['native_memory_readiness'])}


def child_replay_application(scope):
    raw = os.environ.get('JARVIS_FRONTEND_REPLAY_TASK_CONTROL')
    memory_raw = os.environ.get('JARVIS_FRONTEND_REPLAY_MEMORY_CONTROL')
    if memory_raw is not None:
        from evals.responsive_voice.native_memory_readiness_control import config_from_metadata
        from evals.responsive_voice.native_task_replay_control import _object
        require(raw is None and os.environ.get('JARVIS_FRONTEND_REPLAY_PROFILE') == NATIVE_MEDIA
            and os.environ.get('JARVIS_FRONTEND_REPLAY_WORKLOAD') in {NO_WORKLOAD, VISUAL_MEMORY}
            and os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE') == 'realtime',
            'memory_readiness_child_profile_mismatch')
        require(len(memory_raw.encode()) <= 16384, 'memory_readiness_metadata_bound')
        return create_replay_application(scope, native_memory_readiness=config_from_metadata(_object(memory_raw.encode())))
    if raw is None:
        return create_replay_application(scope)
    require(os.environ.get('JARVIS_FRONTEND_REPLAY_PROFILE') == NATIVE_MEDIA
        and os.environ.get('JARVIS_FRONTEND_REPLAY_WORKLOAD') == REASONING_REMINDER
        and os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE') == 'realtime',
        'task_replay_child_profile_mismatch')
    metadata = _task_document(NativeTaskReplayMetadata, raw.encode())
    require(metadata.model == os.environ.get('SECOND_BRAIN_FRONTIER_MODEL'), 'task_replay_child_model_mismatch')
    return create_replay_application(scope, native_task_replay=metadata)


async def child_server(scope_path, result_path, socket_fd):
    """Normal create_app/lifespan runs only after private child environment exists."""
    scope = ReplayHostScope.model_validate_json(Path(scope_path).read_text())
    profile = os.environ.get('JARVIS_FRONTEND_REPLAY_PROFILE', NO_MEDIA)
    require(profile in {NO_MEDIA, NATIVE_MEDIA}, 'invalid_capability_profile')
    report = {'status':'starting', 'pid':os.getpid(), 'scope':scope.public(), 'voice_pass':False,
              'capability_profile':profile}
    host = trace = server = serve_task = runtime = knowledge = None
    sock = socket.socket(fileno=socket_fd)
    try:
        host = child_replay_application(scope)
        trace = AdmissionTrace(host, audio_attempt_limit=1 if profile==NATIVE_MEDIA else None,
            expected_voice_route=os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE','realtime')
                if profile==NATIVE_MEDIA else None)
        from evals.responsive_voice.native_preferences_fixture import attach_preferences_trace
        trace = attach_preferences_trace(trace, scope)
        server = uvicorn.Server(uvicorn.Config(trace, host='127.0.0.1', port=sock.getsockname()[1],
            log_level='warning', access_log=False, lifespan='on', timeout_graceful_shutdown=15))
        serve_task = asyncio.create_task(server.serve(sockets=[sock]))
        async with asyncio.timeout(60):
            while not server.started:
                if serve_task.done():
                    await serve_task
                    raise QualificationFailure('normal_lifespan_startup_failed')
                await asyncio.sleep(0.025)
        from fastapi import FastAPI
        from app.runtime import JarvisRuntimeServices
        from app.knowledge.runtime import KnowledgeRuntime
        from app.runtime.feature_config import MemoryAuthority, get_second_brain_settings
        runtime = host.app.state.runtime_services
        knowledge = runtime.knowledge_runtime
        configure_native_diagnostic_logging(profile)
        require(isinstance(host.app, FastAPI), 'normal_fastapi_missing')
        require(isinstance(runtime, JarvisRuntimeServices), 'normal_runtime_type_missing')
        require(isinstance(runtime.db_pool, asyncpg.Pool), 'normal_sql_pool_missing')
        require(isinstance(knowledge, KnowledgeRuntime), 'normal_knowledge_runtime_missing')
        require(runtime.memory_authority==MemoryAuthority.HINDSIGHT, 'normal_memory_authority_mismatch')
        require(runtime.background_task_service is not None, 'normal_task_owner_missing')
        require(runtime.background_task_service.repository.pool is runtime.db_pool, 'normal_task_pool_mismatch')
        report['database_pool'] = {'min_size':runtime.db_pool.get_min_size(),
                                   'max_size':runtime.db_pool.get_max_size()}
        report['native_options'] = native_host_options(profile,
            native_pool_max_size=runtime.settings.SUPABASE_DB_POOL_MAX_SIZE,
            workload=os.environ.get('JARVIS_FRONTEND_REPLAY_WORKLOAD',NO_WORKLOAD),
            frontier_model=os.environ.get('SECOND_BRAIN_FRONTIER_MODEL')
                if os.environ.get('JARVIS_FRONTEND_REPLAY_WORKLOAD')==REASONING_REMINDER else None,
            frontier_reasoning_effort=os.environ.get('SECOND_BRAIN_FRONTIER_REASONING_EFFORT')
                if os.environ.get('JARVIS_FRONTEND_REPLAY_WORKLOAD')==REASONING_REMINDER else None,
            voice_route=os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE','realtime'),
            live_model=runtime.settings.OPENAI_LIVE_MODEL
                if os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE')=='live' else None,
            live_voice=runtime.settings.OPENAI_LIVE_VOICE
                if os.environ.get('JARVIS_FRONTEND_REPLAY_VOICE_ROUTE')=='live' else None)
        require(report['database_pool']=={'min_size':1,'max_size':report['native_options']['native_pool_max_size']},
                'normal_pool_capacity_mismatch')
        integrated = report['native_options']['workload']==REASONING_REMINDER
        report['workload_configuration'] = {
            'frontier_enabled':os.environ.get('SECOND_BRAIN_FRONTIER_ENABLED')=='true',
            'attention_voice_enabled':os.environ.get('SECOND_BRAIN_ATTENTION_VOICE_ENABLED')=='true',
            'reminder_scheduler_enabled':runtime.settings.REMINDER_SCHEDULER_ENABLED,
            'external_worker_registry_readiness_required':integrated}
        require(all(report['workload_configuration'][key] is integrated for key in
                    ('frontier_enabled','attention_voice_enabled','reminder_scheduler_enabled')),
                'normal_workload_configuration_mismatch')
        if report['native_options']['workload']==VISUAL_MEMORY:
            report['visual_memory_qualification'] = visual_memory_runtime_qualification(
                runtime, flags=get_second_brain_settings(), owner_id=scope.owner_id)
        if integrated:
            expected_limits = native_task_limit_environment()
            actual_limits = {key:os.environ.get(key) for key in expected_limits}
            require(actual_limits==expected_limits, 'normal_workload_task_limit_mismatch')
            report['workload_configuration']['task_limit_environment'] = actual_limits
        require(type(runtime.attention_service).__module__.startswith('app.attention.'), 'normal_attention_owner_missing')
        require(runtime.attention_service.push is None, 'push_unexpectedly_configured')
        if profile==NO_MEDIA:
            require(not runtime.settings.OPENAI_API_KEY and not runtime.settings.GEMINI_API_KEY,
                    'provider_credential_unexpectedly_configured')
        else:
            selected_route=report['native_options']['voice_route']
            native_settings(voice_route=selected_route,live_model=report['native_options']['live_model'],
                            live_voice=report['native_options']['live_voice'])
            require(runtime.visual_frame_storage is not None, 'native_storage_not_configured')
            require(runtime.settings.RESPONSIVE_VOICE_ENABLED is True, 'native_responsive_voice_route_disabled')
            report['provider_configuration'] = native_provider_configuration(
                runtime.settings,voice_route=selected_route)
            if selected_route=='live':
                report['live_evidence_limits']={
                    'native_client_capabilities':'normal LiveClientChannel negotiation remains required',
                    'provider_finalization':'owned Live adapter session.closed only; see child_output_drain.live_finalization_diagnostics',
                    'task_authority_qualified':False,'playback_or_hearing_qualified':False}
        require(knowledge.task is not None and not knowledge.task.done(), 'knowledge_worker_not_running')
        host.check_live()
        async with runtime.db_pool.acquire() as connection:
            schema = await connection.fetchval('select current_schema()')
            paths = await connection.fetchval('select current_schemas(false)')
            bank = dict(await connection.fetchrow('select owner_id,area_id,provider,bank_name from jarvis_memory_banks '
                'where owner_id=$1 and area_id=$2', scope.owner_id, scope.area_id))
            report['schema_relations'] = {name:await connection.fetchval('select to_regclass($1)::text', name)
                for name in (*COUNTED_RELATIONS, 'jarvis_tasks')}
        require(schema==scope.database_schema and 'public' not in paths, 'normal_pool_not_isolated')
        require(bank['bank_name']==scope.memory_bank_id and bank['owner_id']==scope.owner_id,
                'normal_pool_bank_mismatch')
        report.update(status='ready', normal_create_app=True, normal_lifespan=True,
            runtime_type=type(runtime).__module__+'.'+type(runtime).__qualname__,
            knowledge_type=type(knowledge).__module__+'.'+type(knowledge).__qualname__,
            attention_type=type(runtime.attention_service).__module__+'.'+type(runtime.attention_service).__qualname__,
            schema=schema, search_path=paths, canonical_bank=bank, started_at=host.started_at.isoformat(),
            knowledge_worker_running=True, knowledge_http_requests_at_start=knowledge.client.diagnostics()['requests'],
            storage_configured=runtime.visual_frame_storage is not None,
            audio_attempt_limit=trace.audio_attempt_limit,
            routes=sorted({route.path for route in host.app.routes if hasattr(route, 'path')}))
        save_json(result_path, report)
        print('FRONTEND_HOST_READY '+json.dumps({'pid':os.getpid()}), flush=True)
        stop_task = asyncio.create_task(asyncio.to_thread(sys.stdin.readline))
        done, _ = await asyncio.wait((stop_task, serve_task), return_when=asyncio.FIRST_COMPLETED)
        require(stop_task in done, 'normal_server_exited_before_shutdown')
        require(await stop_task in {'STOP\n', ''}, 'invalid_child_control_message')
        async with runtime.db_pool.acquire() as connection:
            report['empty_data_counts'] = await count_owned_rows(connection, scope.owner_id, report)
        report['knowledge_http_requests_after_probes'] = knowledge.client.diagnostics()['requests']
        if profile==NO_MEDIA:
            require(report['knowledge_http_requests_after_probes']==0, 'unexpected_runtime_memory_request')
            require(not any(report['empty_data_counts'].values()), 'unexpected_model_or_ingestion_work')
        server.should_exit = True
        await asyncio.wait_for(asyncio.shield(serve_task), 20)
        require(not host.ready and runtime._closed and knowledge._closed and knowledge.task.done(),
                'normal_lifespan_did_not_close')
        report.update(status='passed', normal_shutdown=True)
    except Exception as error:
        report.update(status='failed', fatal=sanitized(error))
    finally:
        if server is not None:
            server.should_exit = True
        if serve_task is not None and not serve_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(serve_task), 20)
            except Exception as error:
                report.update(status='failed', shutdown_error=sanitized(error))
                serve_task.cancel()
                await asyncio.gather(serve_task, return_exceptions=True)
        if host is not None:
            actual = getattr(host.app.state, 'runtime_services', None)
            report['host_ready_after_shutdown'] = host.ready
            report['runtime_closed'] = getattr(actual, '_closed', None)
            report['knowledge_issue'] = getattr(actual, 'knowledge_issue', None)
            report['attention_admission_diagnostics'] = shutdown_snapshot(
                getattr(actual, 'attention_service', None))
            if host.task_replay is not None:
                report['native_task_replay'] = host.task_replay.status()
                receipt = report['native_task_replay'] or {}
                if (receipt.get('status') != 'closed' or receipt.get('observer_joined') is not True
                        or receipt.get('release_loop_joined') is not True or receipt.get('failure_code')):
                    report['status'] = 'failed'
            if host.memory_readiness is not None:
                report['native_memory_readiness'] = host.memory_readiness.status()
                memory_receipt = report['native_memory_readiness']
                if (memory_receipt.get('observer_joined') is not True
                        or memory_receipt.get('io_loop_joined') is not True
                        or memory_receipt.get('failure_code')):
                    report['status'] = 'failed'
        if trace is not None:
            report.update(lifecycle=trace.lifecycle, websocket_trace=trace.websockets)
            report['audio_attempts_reserved'] = trace.audio_attempts
        sock.close()
        save_json(result_path, report)
        print('FRONTEND_HOST_EXIT '+json.dumps({'status':report['status'], 'fatal':report.get('fatal')}), flush=True)
    return report['status']=='passed'


def native_task_limit_environment():
    """Use the worker's canonical defaults, without constructing or starting it."""
    from evals.responsive_voice.native_task_worker import NativeTaskWorkerOptions
    limits = NativeTaskWorkerOptions.model_fields['task_limits'].get_default(call_default_factory=True)
    return {'SECOND_BRAIN_'+key.upper():str(value) for key,value in limits.model_dump().items()}


def child_environment(*, dsn, scope, token, origin, hindsight_url, api_key_file,
                      profile=NO_MEDIA, configured=None, native_pool_max_size=None, workload=NO_WORKLOAD,
                      frontier_model=None, frontier_reasoning_effort=None,
                      voice_route='realtime', live_model=None, live_voice=None):
    """Whitelist process environment; a temporary cwd prevents .env inheritance."""
    env = {key:os.environ[key] for key in ('PATH','HOME','LANG','TMPDIR','SSL_CERT_FILE') if key in os.environ}
    require(profile in {NO_MEDIA, NATIVE_MEDIA}, 'invalid_capability_profile')
    options = native_host_options(profile,native_pool_max_size=native_pool_max_size,workload=workload,
        frontier_model=frontier_model,frontier_reasoning_effort=frontier_reasoning_effort,
        voice_route=voice_route,live_model=live_model,live_voice=live_voice)
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONUNBUFFERED='1',
        APP_ENV='frontend-replay-qualification', LOG_LEVEL='WARNING', LOG_JSON='false',
        PUBLIC_BASE_URL=origin, JARVIS_OWNER_ID=scope.owner_id, JARVIS_STATIC_BEARER_TOKEN=token,
        SUPABASE_DB_URL=dsn, SUPABASE_DB_POOL_MIN_SIZE='1',
        SUPABASE_DB_POOL_MAX_SIZE=str(options['native_pool_max_size']),
        SUPABASE_URL='', SUPABASE_SERVICE_ROLE_KEY='', OPENAI_API_KEY='', GEMINI_API_KEY='',
        OPENAI_REALTIME_MODEL='', AUDIO_PROVIDER_DEFAULT='openai', OPENAI_LIVE_ENABLED='false',
        GEMINI_LIVE_ENABLED='false', RESPONSIVE_VOICE_ENABLED='false', MEMORY_JOB_RUNNER_ENABLED='false',
        MEMORY_MAINTENANCE_ENABLED='false', REMINDER_SCHEDULER_ENABLED='false',
        SECOND_BRAIN_ENABLED='true', SECOND_BRAIN_TASKS_ENABLED='true', SECOND_BRAIN_ATTENTION_ENABLED='true',
        SECOND_BRAIN_KNOWLEDGE_ENABLED='true', SECOND_BRAIN_MEMORY_AUTHORITY='hindsight',
        SECOND_BRAIN_KNOWLEDGE_AREA_ID=str(scope.area_id), SECOND_BRAIN_FRONTIER_ENABLED='false',
        SECOND_BRAIN_ATTENTION_VOICE_ENABLED='false', SECOND_BRAIN_RESPONSIBILITIES_ENABLED='false',
        SECOND_BRAIN_PUSH_ENABLED='false', SECOND_BRAIN_MAIL_ENABLED='false',
        SECOND_BRAIN_SANDBOX_ENABLED='false', SECOND_BRAIN_ARTIFACTS_ENABLED='false',
        SECOND_BRAIN_HINDSIGHT_URL=hindsight_url, SECOND_BRAIN_HINDSIGHT_PRIVATE_HTTP_ORIGIN=hindsight_url,
        SECOND_BRAIN_HINDSIGHT_API_KEY='', SECOND_BRAIN_HINDSIGHT_API_KEY_FILE=str(api_key_file))
    env['JARVIS_FRONTEND_REPLAY_PROFILE'] = profile
    env['JARVIS_FRONTEND_REPLAY_WORKLOAD'] = workload
    env['JARVIS_FRONTEND_REPLAY_VOICE_ROUTE'] = voice_route
    if profile == NATIVE_MEDIA and os.environ.get('JARVIS_EVAL_REALTIME_WIRE_DIR'):
        env['JARVIS_EVAL_REALTIME_WIRE_DIR'] = os.environ['JARVIS_EVAL_REALTIME_WIRE_DIR']
    if profile==NATIVE_MEDIA:
        # WebRTC control uses the child's default SSL context, unlike HTTPX's
        # certifi-backed default. Preserve explicit trust configuration exactly;
        # only the isolated native profile supplies a bundle when neither knob
        # was provided. Certificate and hostname verification stay enabled.
        for key in ('SSL_CERT_FILE', 'SSL_CERT_DIR'):
            if key in os.environ:
                env[key] = os.environ[key]
        if not any(key in os.environ for key in ('SSL_CERT_FILE', 'SSL_CERT_DIR')):
            import certifi
            bundle = Path(certifi.where()).resolve()
            require(bundle.is_file(), 'native_ca_bundle_unavailable')
            env['SSL_CERT_FILE'] = str(bundle)
        settings = configured or native_settings(voice_route=voice_route,live_model=live_model,live_voice=live_voice)
        # Deliberately do not inherit normal owner, database, worker or external
        # notification configuration along with the provider settings.
        for key, value in settings.model_dump(mode='json').items():
            if key.startswith(('OPENAI_', 'GEMINI_', 'VISUAL_', 'PERCEPTION_')) or key in {
                'AUDIO_PROVIDER_DEFAULT', 'RESPONSIVE_VOICE_ENABLED', 'SUPABASE_URL',
                'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_STORAGE_BUCKET'}:
                env[key] = '' if value is None else json.dumps(value) if not isinstance(value, str) else value
        # This profile is the explicit isolated opt-in for native voice_route
        # offers. Do not copy the normal deployment's deliberately disabled gate
        # back over this opt-in, or install_responsive_voice rejects before run().
        # This is independent of OPENAI_LIVE_ENABLED/GEMINI_LIVE_ENABLED, whose
        # configured values remain unchanged; no alternate route is enabled here.
        env['RESPONSIVE_VOICE_ENABLED'] = 'true'
        # Explicit isolated-profile opt-in; legacy offers still select legacy.
        env['LEAN_VOICE_ENABLED'] = ('true' if voice_route == 'realtime'
            and getattr(settings, 'LEAN_VOICE_ENABLED', False) is True else 'false')
        if voice_route=='live':
            # This opt-in changes only this isolated child's explicitly selected
            # route. Realtime configuration and the normal default are untouched.
            env.update(OPENAI_LIVE_ENABLED='true',OPENAI_LIVE_MODEL=live_model,OPENAI_LIVE_VOICE=live_voice)
        env['JARVIS_REALTIME_MAX_SESSION_SECONDS'] = str(max(1, min(settings.JARVIS_REALTIME_MAX_SESSION_SECONDS,
            int((scope.expires_at-datetime.now(UTC)).total_seconds()))))
    if workload==REASONING_REMINDER:
        env.update(native_task_limit_environment())
        env.update(SECOND_BRAIN_FRONTIER_ENABLED='true', SECOND_BRAIN_FRONTIER_MODEL=frontier_model,
            SECOND_BRAIN_FRONTIER_REASONING_EFFORT=frontier_reasoning_effort,
            SECOND_BRAIN_ATTENTION_VOICE_ENABLED='true', REMINDER_SCHEDULER_ENABLED='true')
    elif workload==VISUAL_MEMORY:
        env['SECOND_BRAIN_RESPONSIBILITIES_ENABLED'] = 'true'
    return env


async def qualify_frontend_host(*, dsn, directory, hindsight_url, api_key_file,
                                prepare_only=False, lifetime_seconds=300, profile=NO_MEDIA,
                                native_pool_max_size=None, workload=NO_WORKLOAD,
                                frontier_model=None, frontier_reasoning_effort=None,
                                voice_route='realtime', live_model=None, live_voice=None):
    directory = Path(directory).resolve()
    require(profile in {NO_MEDIA, NATIVE_MEDIA}, 'invalid_capability_profile')
    require(profile==NO_MEDIA or prepare_only, 'native_profile_requires_prepare_then_serve_no_provider_probe')
    options = native_host_options(profile,native_pool_max_size=native_pool_max_size,workload=workload,
        frontier_model=frontier_model,frontier_reasoning_effort=frontier_reasoning_effort,
        voice_route=voice_route,live_model=live_model,live_voice=live_voice)
    configured = native_settings(voice_route=voice_route,live_model=live_model,live_voice=live_voice) if profile==NATIVE_MEDIA else None
    require(type(lifetime_seconds) is int and 60<=lifetime_seconds<=3600, 'invalid_scope_lease')
    require(urlsplit(hindsight_url).scheme=='http' and urlsplit(hindsight_url).hostname=='127.0.0.1',
            'qualification_requires_explicit_loopback_memory')
    fixture = DatabaseFixture(dsn, directory, test_name='frontend_host_normal_child_sql')
    report_path = directory / ('frontend-host-'+fixture.run_id+'.json')
    resource_path = directory / ('frontend-host-resources-'+fixture.run_id+'.json')
    child_path = directory / ('frontend-host-child-'+fixture.run_id+'.json')
    scope_path = directory / ('frontend-host-scope-'+fixture.run_id+'.json')
    log_path = directory / ('frontend-host-child-'+fixture.run_id+'.log')
    token_path = directory / ('frontend-host-bearer-'+fixture.run_id+'.token')
    control_path = directory / ('frontend-host-control-'+fixture.run_id+'.json')
    report = {'format':'jarvis.frontend_host.sql.v1', 'run_id':fixture.run_id, 'status':'running',
        'checks':[], 'cleanup':{}, 'voice_pass':False, 'provider_calls':False, 'media_calls':False,
        'public_migration':False, 'report_path':str(report_path), 'child_report':str(child_path),
        'child_log':str(log_path), 'fixture_manifest':str(fixture.path), 'resource_manifest':str(resource_path),
        'native_options':options}
    resources = {'format':'jarvis.frontend_host.resources.v1', 'run_id':fixture.run_id,
        'fixture_manifest':str(fixture.path), 'scope_file':str(scope_path),
        'hindsight_url':hindsight_url, 'hindsight_api_key_file':str(Path(api_key_file).resolve()),
        'bank_name':None, 'bank_creation_attempted':False, 'bank_created':False,
        'child_pid':None, 'child_returncode':None, 'status':'preparing',
        'bearer_token_file':None, 'control_file':str(control_path), 'lease_seconds':lifetime_seconds,
        'capability_profile':profile, 'storage_cleanup_implemented':True, 'native_options':options,
        'storage':None,
        'cleanup_command':[sys.executable, '-m', 'evals.responsive_voice.frontend_host_qualification',
                           '--mode', 'cleanup', '--resource-manifest', str(resource_path)]}
    connection = native = child = sock = output_drain = None
    preserve_resources = False
    temporary = tempfile.TemporaryDirectory(prefix='jarvis-frontend-host-child-')
    token = secrets.token_urlsafe(36)
    child_log = os.fdopen(os.open(log_path, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600), 'wb')
    def save():
        save_json(report_path, report)
        save_json(resource_path, resources)
    def passed(name, **evidence):
        report['checks'].append({'name':name, 'status':'passed', **evidence})
        save()
    save()
    try:
        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=90)
        schema = await fixture.create_schema(connection)
        migrations = load_migrations()
        await asyncio.wait_for(apply_migrations(connection, schema=schema, fresh=True, migrations=migrations), 240)
        await connection.execute(f'set search_path to {quote_schema(schema)},pg_catalog,extensions')
        owner = fixture.owners[0]
        area = await connection.fetchval("insert into jarvis_knowledge_areas(owner_id,name,domain) "
            "values($1,'Frontend host qualification','test') returning id", owner)
        bank = await connection.fetchval('insert into jarvis_memory_banks(owner_id,area_id) '
                                        'values($1,$2) returning bank_name', owner, area)
        require(re.fullmatch(r'jarvis_[0-9a-f]{32}', bank) is not None, 'canonical_default_bank_name_invalid')
        resources.update(database_schema=schema, owner_id=owner, area_id=str(area), bank_name=bank)
        if configured is not None:
            from app.storage.supabase_config import get_supabase_storage_config
            config = get_supabase_storage_config(configured)
            resources['storage'] = {'origin':config.supabase_url.rstrip('/'),
                'bucket':config.visual_frame_bucket, 'prefix':f'owners/{owner}/sessions/'}
            save()
            async with storage_http(config) as storage:
                require(not await list_owned_storage(storage, resources), 'new_run_storage_prefix_not_empty')
            passed('native_storage_prefix_preflight_empty', prefix=resources['storage']['prefix'])
        save()
        key = secret_file(str(api_key_file))
        native = HindsightClient(hindsight_url, key, timeout=10, private_http_origin=hindsight_url)
        require(bank not in await native_bank_inventory(native),
                'native_bank_preexists_cleanup_not_authorized')
        resources['bank_creation_attempted'] = True
        save()
        await native.ensure_bank(bank)
        require(bank in await native_bank_inventory(native), 'native_bank_not_present')
        resources['bank_created'] = True
        passed('canonical_area_bank_and_native_empty_bank', bank_name=bank, native_policy_checked=True,
               schema=schema, migration_versions=[migration.version for migration in migrations])
        scope = ReplayHostScope(run_id=fixture.run_id, owner_id=owner, database_schema=schema,
            area_id=area, memory_bank_id=bank, expires_at=datetime.now(UTC)+timedelta(seconds=lifetime_seconds))
        save_json(scope_path, scope.model_dump(mode='json'))
        resources.update(memory_bank_id=bank, expires_at=scope.expires_at.isoformat(),
                         backend_base_url=None)
        scoped_dsn = scoped_database_url(dsn, schema)
        proof = await asyncpg.connect(scoped_dsn, timeout=10, command_timeout=20)
        try:
            actual_schema = await proof.fetchval('select current_schema()')
            paths = await proof.fetchval('select current_schemas(false)')
            actual_bank = await proof.fetchval('select bank_name from jarvis_memory_banks where owner_id=$1 and area_id=$2', owner, area)
            require(actual_schema==schema and paths[0]==schema and 'public' not in paths and actual_bank==bank,
                    'actual_dsn_search_path_binding_failed')
        finally:
            await proof.close(timeout=5)
        passed('actual_scoped_dsn_without_public_fallback', schema=actual_schema, search_path=paths)
        if prepare_only:
            descriptor = os.open(token_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w') as secret:
                secret.write(token+'\n')
            resources.update(bearer_token_file=str(token_path), status='prepared')
            report.update(status='prepared', prepared_scope=scope.public(), bearer_token_file=str(token_path),
                          admission_qualified=False)
            preserve_resources = True
            save()
            return report
        stale = await connection.fetchval('insert into jarvis_sessions(owner_id,title) values($1,$2) returning id',
                                          owner, 'Pre-startup qualification session')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', 0))
        sock.listen(128)
        sock.set_inheritable(True)
        origin = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        env = child_environment(dsn=scoped_dsn, scope=scope, token=token, origin=origin,
                                hindsight_url=hindsight_url, api_key_file=api_key_file)
        child = await asyncio.create_subprocess_exec(sys.executable, '-m',
            'evals.responsive_voice.frontend_host_qualification', '--child', '--scope-file', str(scope_path),
            '--child-result', str(child_path), '--socket-fd', str(sock.fileno()),
            cwd=temporary.name, env=env, pass_fds=(sock.fileno(),), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output_drain = ChildOutputDrain(child, child_log)
        resources.update(child_pid=child.pid, origin=origin, status='child_starting')
        sock.close()
        sock = None
        save()
        await output_drain.wait_ready()
        ready = json.loads(child_path.read_text())
        require(ready['status']=='ready' and ready['pid']==child.pid and child.pid!=os.getpid(), 'actual_child_startup_missing')
        require(ready['schema']==schema and ready['canonical_bank']['bank_name']==bank,
                'actual_child_canonical_binding_mismatch')
        passed('normal_child_create_app_and_lifespan', pid=child.pid, runtime_type=ready['runtime_type'],
               knowledge_type=ready['knowledge_type'], attention_type=ready['attention_type'])
        headers = [('authorization','Bearer '+token), ('x-jarvis-evaluation-run', fixture.run_id)]
        endpoint = '/eval/replay/'+fixture.run_id+'/scope'
        expected_ws = []
        async with httpx.AsyncClient(base_url=origin, timeout=10, trust_env=False) as http:
            response = await http.get(endpoint, headers=headers)
            require(response.status_code==200 and response.json()==scope.public(), 'authenticated_scope_metadata_mismatch')
            passed('authenticated_scope_endpoint', owner_id=owner, area_id=str(area), bank_name=bank)
            header_cases = (
                ('missing_run', headers[:1], 'replay_header_missing_or_ambiguous', 403),
                ('wrong_run', [headers[0],('x-jarvis-evaluation-run',uuid4().hex)], 'replay_run_mismatch', 403),
                ('duplicate_run', headers+[headers[1]], 'replay_header_missing_or_ambiguous', 403),
                ('missing_auth', headers[1:], 'replay_header_missing_or_ambiguous', 403),
                ('duplicate_auth', headers+[headers[0]], 'replay_header_missing_or_ambiguous', 403),
                ('wrong_auth', [('authorization','Bearer invalid'),headers[1]], 'jarvis_auth_invalid', 403),
                ('malformed_auth', [('authorization','Basic invalid'),headers[1]], 'jarvis_auth_required', 401),
            )
            for name, supplied, code, status in header_cases:
                response = await http.get(endpoint, headers=supplied)
                require(response.status_code==status and response.json()['error']['code']==code, 'header_rejection_'+name)
                passed('http_'+name, status_code=status, code=code)
            response = await http.post(endpoint, headers=headers)
            require(response.status_code==405 and response.json()['error']['code']=='replay_scope_method', 'scope_method_not_rejected')
            passed('scope_method_rejected', status_code=405)
            before = await connection.fetchval('select count(*) from jarvis_sessions where owner_id=$1', owner)
            for name, payload in (('resume_true',{'resume_active':True}), ('resume_missing',{}),
                                  ('resume_string',{'resume_active':'false'})):
                response = await http.post('/sessions/start', headers=headers, json=payload)
                require(response.status_code==403 and response.json()['error']['code']=='replay_fresh_session_required',
                        'fresh_body_rejection_'+name)
                passed('http_'+name, code='replay_fresh_session_required')
            require(await connection.fetchval('select count(*) from jarvis_sessions where owner_id=$1', owner)==before,
                    'rejected_session_body_created_row')
            fresh_ids = []
            for index in range(2):
                response = await http.post('/sessions/start', headers=headers,
                    json={'resume_active':False, 'title':'Frontend qualification '+str(index)})
                require(response.status_code==200 and bool(response.headers.get('x-request-id')), 'normal_session_start_failed')
                payload = response.json()
                identity = UUID(payload['session']['id'])
                row = await connection.fetchrow('select owner_id,status,created_at,metadata from jarvis_sessions where id=$1', identity)
                require(row is not None and row['owner_id']==owner and row['status']=='active'
                    and row['created_at']>=datetime.fromisoformat(ready['started_at'])
                    and row['created_at']==datetime.fromisoformat(payload['session']['created_at']),
                    'normal_session_owner_or_freshness_mismatch')
                require(identity!=stale and identity not in fresh_ids, 'normal_session_was_resumed')
                fresh_ids.append(identity)
                for name, path in (('audio_websocket_url','/realtime/audio/ws'), ('visual_websocket_url','/realtime/visual/ws')):
                    parsed = urlsplit(payload['realtime'][name])
                    require(parsed.netloc==urlsplit(origin).netloc and parsed.path==path
                            and parse_qs(parsed.query)['session_id']==[str(identity)], 'normal_realtime_route_rewritten')
                detail = await http.get('/sessions/'+str(identity), headers=headers)
                require(detail.status_code==200 and detail.json()['session']['id']==str(identity), 'normal_session_get_failed')
            passed('normal_sessions_start_fresh_false_and_sql_owner', session_ids=[str(x) for x in fresh_ids],
                   owner_id=owner, existing_session_not_resumed=True, downstream_routes_preserved=True)
            response = await http.post('/sessions/'+str(fresh_ids[1])+'/close', headers=headers)
            require(response.status_code==200 and response.json()['session']['status']=='closed', 'normal_session_close_failed')
            foreign = await connection.fetchval('insert into jarvis_sessions(owner_id,title) values($1,$2) returning id',
                                                fixture.owners[1], 'Foreign fixture session')
            async def denied_ws(name, identity, supplied, reason):
                uri = origin.replace('http://','ws://')+'/realtime/audio/ws?session_id='+str(identity)
                try:
                    async with websockets.connect(uri, additional_headers=supplied, proxy=None,
                                                  open_timeout=5, close_timeout=2):
                        raise QualificationFailure('websocket_policy_unexpectedly_accepted_'+name)
                except InvalidStatus as error:
                    require(error.response.status_code==403, 'unexpected_websocket_http_denial_'+name)
                expected_ws.append({'accepted':False, 'reason':reason})
                passed('websocket_'+name, handshake_status=403, expected_policy_reason=reason)
            await denied_ws('missing_run', fresh_ids[0], headers[:1], 'replay_header_missing_or_ambiguous')
            await denied_ws('stale_session', stale, headers, 'replay_fresh_session_required')
            await denied_ws('foreign_owner', foreign, headers, 'replay_fresh_session_required')
            await denied_ws('closed_session', fresh_ids[1], headers, 'replay_fresh_session_required')
            await denied_ws('wrong_run', fresh_ids[0], [headers[0],('x-jarvis-evaluation-run',uuid4().hex)], 'replay_run_mismatch')
            uri = origin.replace('http://','ws://')+'/realtime/audio/ws?session_id='+str(fresh_ids[0])
            async with websockets.connect(uri, additional_headers=headers, proxy=None,
                                          open_timeout=5, close_timeout=2) as ws:
                event = json.loads(await asyncio.wait_for(ws.recv(), 5))
                require(event.get('type')=='error' and event.get('code')=='realtime_not_configured',
                        'fresh_websocket_did_not_reach_expected_downstream_refusal')
                try:
                    await asyncio.wait_for(ws.recv(), 5)
                    raise QualificationFailure('unexpected_followup_voice_event')
                except ConnectionClosed:
                    require(ws.close_code==1008 and ws.close_reason=='realtime_not_configured',
                            'fresh_websocket_close_mismatch')
            expected_ws.append({'accepted':True, 'reason':'realtime_not_configured'})
            passed('fresh_websocket_admitted_then_normal_downstream_refused', code='realtime_not_configured',
                   remote_voice_started=False, voice_pass=False)
        child.stdin.write(b'STOP\n')
        await child.stdin.drain()
        child.stdin.close()
        async with asyncio.timeout(30):
            await child.wait()
        report['child_output_drain'] = await output_drain.wait_closed()
        require(report['child_output_drain']['complete'] and not report['child_output_drain']['errors'],
                'child_output_drain_incomplete')
        result = json.loads(child_path.read_text())
        report['child'] = result
        require(child.returncode==0 and result['status']=='passed' and result['normal_shutdown'],
                'normal_child_shutdown_not_qualified')
        require(len(result['websocket_trace'])==len(expected_ws), 'websocket_trace_count_mismatch')
        for actual, expected in zip(result['websocket_trace'], expected_ws):
            require(actual['accepted']==expected['accepted'] and actual['close_code']==1008
                    and actual['close_reason']==expected['reason'], 'actual_websocket_policy_trace_mismatch')
            require(actual['events']==([{'type':'error','code':'realtime_not_configured'}]
                    if expected['accepted'] else []), 'unexpected_websocket_media_or_voice_event')
        passed('actual_ws_policy_trace_and_normal_shutdown', knowledge_runtime_http_requests=0,
               empty_data_counts=result['empty_data_counts'])
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed', fatal=sanitized(error))
    finally:
        if sock is not None:
            sock.close()
        if child is not None:
            try:
                if child.returncode is None:
                    try:
                        child.stdin.write(b'STOP\n')
                        await child.stdin.drain()
                        child.stdin.close()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    try:
                        await asyncio.wait_for(child.wait(), 30)
                    except TimeoutError:
                        child.terminate()
                        try:
                            await asyncio.wait_for(child.wait(), 10)
                        except TimeoutError:
                            child.kill()
                            await child.wait()
                        report.update(status='failed', forced_child_shutdown=True)
                report['cleanup']['child_exited'] = child.returncode is not None
                report['cleanup']['child_returncode'] = child.returncode
                if child_path.exists():
                    report['child'] = json.loads(child_path.read_text())
            except Exception as error:
                report.update(status='failed', child_cleanup_error=sanitized(error))
        if output_drain is not None:
            report['child_output_drain'] = await output_drain.wait_closed()
            if not report['child_output_drain']['complete'] or report['child_output_drain']['errors']:
                report.update(status='failed', child_output_drain_incomplete=True)
        native_safe = not resources['bank_creation_attempted']
        try:
            if preserve_resources:
                report['cleanup']['native_bank_cleanup'] = 'retained_for_bounded_serve'
            elif resources['bank_creation_attempted']:
                if connection is not None and resources.get('database_schema'):
                    inventory = await storage_inventory(connection, resources['owner_id'])
                    report['cleanup']['storage'] = await cleanup_owned_storage(resources, inventory['objects'], save)
                report['cleanup'].update(await cleanup_owned_native_bank(resources['bank_name'],
                    hindsight_url=hindsight_url, api_key_file=api_key_file))
                native_safe = True
            else:
                report['cleanup']['native_bank_cleanup'] = 'not_allocated'
        except Exception as error:
            report.update(status='failed', bank_cleanup_error=sanitized(error))
        finally:
            if native is not None:
                await native.close()
        try:
            if connection is not None:
                try:
                    await connection.close(timeout=5)
                except Exception as error:
                    connection.terminate()
                    report.update(status='failed', connection_close_error=sanitized(error))
        finally:
            try:
                if preserve_resources:
                    report['cleanup']['sql'] = {'status':'retained_for_bounded_serve'}
                elif native_safe:
                    report['cleanup']['sql'] = await fixture.cleanup()
                else:
                    report['cleanup']['sql'] = {'status':'preserved_for_cleanup_retry'}
            except Exception as error:
                report.update(status='failed', sql_cleanup_error=sanitized(error))
        temporary.cleanup()
        child_log.close()
        resources.update(status='prepared' if preserve_resources else 'cleaned'
                         if report.get('cleanup', {}).get('sql', {}).get('remaining_schemas')==0
                         and not any(key.endswith('cleanup_error') for key in report) else 'cleanup_incomplete',
                         cleanup=report['cleanup'])
        save()
    return report


def prepared_resources(manifest, dsn):
    path = Path(manifest).resolve(strict=True)
    resources = json.loads(path.read_text())
    require(resources.get('format')=='jarvis.frontend_host.resources.v1', 'invalid_resource_manifest')
    run = resources['run_id']
    require(path.name==f'frontend-host-resources-{run}.json', 'resource_manifest_path_mismatch')
    fixture_path = path.parent / f'second-brain-{run}.json'
    require(Path(resources['fixture_manifest']).resolve()==fixture_path, 'fixture_manifest_path_mismatch')
    fixture = DatabaseFixture.load(dsn, fixture_path)
    require(resources['run_id']==fixture.run_id and resources['owner_id'] in fixture.owners
            and resources['database_schema'] in fixture.manifest['schemas'], 'resource_fixture_identity_mismatch')
    require(re.fullmatch(r'jarvis_[0-9a-f]{32}', resources['bank_name']) is not None, 'invalid_owned_bank_identity')
    for key, name in (('scope_file',f'frontend-host-scope-{run}.json'),
                      ('control_file',f'frontend-host-control-{run}.json')):
        require(Path(resources[key]).resolve()==path.parent/name, 'resource_owned_path_mismatch')
    if resources.get('bearer_token_file'):
        require(Path(resources['bearer_token_file']).resolve()==path.parent/f'frontend-host-bearer-{run}.token',
                'bearer_file_path_mismatch')
    return path, resources, fixture


async def cleanup_prepared_host(*, manifest, dsn, deadline=None):
    path, resources, fixture = prepared_resources(manifest, dsn)
    result_path = path.parent / f'frontend-host-cleanup-{fixture.run_id}-{uuid4().hex}.json'
    result = {'status':'running', 'run_id':fixture.run_id, 'report_path':str(result_path), 'cleanup':{}}
    persist = lambda: save_json(result_path, result)
    budget = _CleanupDeadline(result, persist, deadline=deadline)
    lock_fd = os.open(path.parent/f'frontend-host-{fixture.run_id}.lock', os.O_RDWR|os.O_CREAT, 0o600)
    connection = None
    locked = False
    async def acquire_lock():
        nonlocal locked
        while not locked:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                # Only our manifest-controlled child; never the shared service.
                save_json(Path(resources['control_file']), {'run_id':fixture.run_id, 'action':'stop'})
                await asyncio.sleep(.2)
    try:
        await budget.step('owned_host_lock', acquire_lock, limit=35)
        resources = json.loads(path.read_text())
        pid = resources.get('child_pid')
        if pid and resources.get('child_returncode') is None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise QualificationFailure('owned_child_exit_not_confirmed')
        connection = await budget.step('cleanup_sql_connect',
            lambda: asyncpg.connect(dsn, timeout=10, command_timeout=30), limit=10)
        schema_exists = await budget.step('cleanup_schema_lookup',
            lambda: connection.fetchval('select exists(select 1 from pg_namespace where nspname=$1)',
                                         resources['database_schema']), limit=30)
        if schema_exists:
            await budget.step('cleanup_schema_binding',
                lambda: connection.execute(f"set search_path to {quote_schema(resources['database_schema'])},pg_catalog,extensions"),
                limit=30)
            bank = await budget.step('cleanup_bank_binding',
                lambda: connection.fetchval('select bank_name from jarvis_memory_banks where owner_id=$1 and area_id=$2',
                                             resources['owner_id'], UUID(resources['area_id'])), limit=30)
            require(bank==resources['bank_name'], 'cleanup_canonical_bank_mismatch')
            inventory = await budget.step('storage_sql_inventory',
                lambda: storage_inventory(connection, resources['owner_id']), limit=30)
            result['cleanup']['storage'] = await cleanup_owned_storage(resources, inventory['objects'],
                lambda: save_json(path, resources), budget=budget)
            result['cleanup']['source_originals'] = await cleanup_source_originals(dsn, resources, budget=budget)
        else:
            # Recovery still independently audits persisted exact-owned identities.
            result['cleanup']['storage'] = await cleanup_owned_storage(resources, [],
                lambda: save_json(path, resources), budget=budget)
        if resources['bank_creation_attempted']:
            result['cleanup'].update(await cleanup_owned_native_bank(resources['bank_name'],
                hindsight_url=resources['hindsight_url'], api_key_file=resources['hindsight_api_key_file'],
                budget=budget))
        closing, connection = connection, None
        await budget.step('cleanup_sql_close', lambda: closing.close(timeout=5), limit=5, shutdown=True)
        result['cleanup']['sql'] = await budget.step('fixture_cleanup_and_independent_audit', fixture.cleanup)
        require(not budget.shutdown_failed, 'cleanup_shutdown_unjoined')
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        if resources.get('bearer_token_file'):
            Path(resources['bearer_token_file']).unlink(missing_ok=True)
            result['cleanup']['bearer_token_removed'] = True
        result['status'] = 'passed'
    except asyncio.CancelledError as error:
        result.update(status='cancelled', failure_code='cleanup_cancelled',
                      fatal=sanitized(error), sql_preserved=True)
        raise
    except Exception as error:
        result.update(status='failed', failure_code='cleanup_deadline_exhausted' if isinstance(error, TimeoutError)
                      else 'cleanup_step_failed', fatal=sanitized(error), sql_preserved=True)
    finally:
        if connection is not None:
            try:
                await budget.step('cleanup_sql_close', lambda: connection.close(timeout=5),
                                  limit=5, shutdown=True)
            except (Exception, asyncio.CancelledError) as error:
                result.setdefault('shutdown_errors', []).append(sanitized(error))
                if result['status'] == 'passed':
                    result.update(status='failed', failure_code='cleanup_shutdown_unjoined')
        budget.timing.update(terminal_status=result['status'], finished_monotonic=budget.clock(),
                             elapsed_seconds=budget.clock()-budget.started)
        resources.update(status='cleaned' if result['status']=='passed' else 'cleanup_incomplete',
                         cleanup=result['cleanup'], cleanup_report=str(result_path))
        try:
            if locked:
                save_json(path, resources)
        finally:
            if locked:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            persist()
    return result


async def serve_prepared_host(*, manifest, dsn, port=0, stop_event=None, ready_callback=None, auto_cleanup=True):
    path, resources, fixture = prepared_resources(manifest, dsn)
    require(resources['status'] in {'prepared','stopped','serve_failed'}, 'prepared_scope_not_servable')
    scope = ReplayHostScope.model_validate_json(Path(resources['scope_file']).read_text())
    require(scope.run_id==fixture.run_id and scope.owner_id==resources['owner_id']
        and scope.database_schema==resources['database_schema'] and str(scope.area_id)==resources['area_id']
        and scope.memory_bank_id==resources['bank_name'], 'prepared_scope_identity_mismatch')
    require(0<(scope.expires_at-datetime.now(UTC)).total_seconds()<=3600, 'prepared_scope_expired')
    require(type(port) is int and 0<=port<=65535, 'invalid_loopback_port')
    token = secret_file(resources['bearer_token_file'])
    attempt = uuid4().hex
    child_path = path.parent/f'frontend-host-child-{fixture.run_id}-{attempt}.json'
    log_path = path.parent/f'frontend-host-child-{fixture.run_id}-{attempt}.log'
    result_path = path.parent/f'frontend-host-serve-{fixture.run_id}-{attempt}.json'
    result = {'status':'starting', 'run_id':fixture.run_id, 'report_path':str(result_path),
              'child_report':str(child_path), 'child_log':str(log_path), 'voice_pass':False}
    lock_fd = os.open(path.parent/f'frontend-host-{fixture.run_id}.lock', os.O_RDWR|os.O_CREAT, 0o600)
    temporary = tempfile.TemporaryDirectory(prefix='jarvis-prepared-host-child-')
    log = os.fdopen(os.open(log_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'wb')
    child = sock = output_drain = None
    locked = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked = True
        except BlockingIOError:
            raise QualificationFailure('prepared_scope_already_serving') from None
        Path(resources['control_file']).unlink(missing_ok=True)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        sock.listen(128)
        sock.set_inheritable(True)
        origin = 'http://127.0.0.1:'+str(sock.getsockname()[1])
        env = child_environment(dsn=scoped_database_url(dsn, scope.database_schema), scope=scope, token=token,
            origin=origin, hindsight_url=resources['hindsight_url'], api_key_file=resources['hindsight_api_key_file'],
            profile=resources['capability_profile'], **resources.get('native_options',{}))
        env.update(task_replay_child_environment(resources, scope, path.parent))
        env.update(memory_readiness_child_environment(resources, scope, path.parent))
        child = await asyncio.create_subprocess_exec(sys.executable, '-m',
            'evals.responsive_voice.frontend_host_qualification', '--child', '--scope-file', resources['scope_file'],
            '--child-result', str(child_path), '--socket-fd', str(sock.fileno()), cwd=temporary.name,
            env=env, pass_fds=(sock.fileno(),), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        output_drain = ChildOutputDrain(child, log)
        sock.close()
        sock = None
        resources.update(status='starting', backend_base_url=origin, child_pid=child.pid, child_returncode=None,
            parent_pid=os.getpid(), child_report=str(child_path), serve_report=str(result_path),
            control_handle={'kind':'owned_stop_file', 'path':resources['control_file'], 'run_id':fixture.run_id})
        save_json(path, resources)
        await output_drain.wait_ready()
        actual = json.loads(child_path.read_text())
        require(actual['status']=='ready' and actual['pid']==child.pid, 'prepared_normal_startup_missing')
        resources.update(status='serving', normal_lifespan_qualified=True)
        save_json(path, resources)
        info = {'resource_manifest':str(path), 'run_id':scope.run_id, 'backend_base_url':origin,
            'owner_id':scope.owner_id, 'area_id':str(scope.area_id), 'memory_bank_id':scope.memory_bank_id,
            'bearer_token_file':resources['bearer_token_file'], 'child_pid':child.pid,
            'control_handle':resources['control_handle'], 'cleanup_command':resources['cleanup_command'],
            'expires_at':scope.expires_at.isoformat(), 'lease_seconds':resources['lease_seconds'],
            'capability_profile':resources['capability_profile'], 'storage_cleanup_implemented':True,
            'audio_attempt_limit':actual.get('audio_attempt_limit'),
            'native_options':actual.get('native_options'), 'database_pool':actual.get('database_pool'),
            'workload_configuration':actual.get('workload_configuration'),
            'provider_configuration':actual.get('provider_configuration'),
            'live_evidence_limits':actual.get('live_evidence_limits'),
            'storage_prefix':resources['storage']['prefix'] if resources.get('storage') else None,
            'session_policy':'Native client must POST /sessions/start with resume_active=false; no provider session prestarted.'}
        result.update(status='serving', private_launch=info)
        save_json(result_path, result)
        print('FRONTEND_HOST_SERVING '+json.dumps(info), flush=True)
        if ready_callback is not None:
            await ready_callback(info)
        while datetime.now(UTC)<scope.expires_at and child.returncode is None:
            if stop_event is not None and stop_event.is_set():
                result['stop_reason'] = 'test_stop_event'
                break
            control = Path(resources['control_file'])
            if control.exists():
                command = json.loads(control.read_text())
                require(command=={'run_id':fixture.run_id, 'action':'stop'}, 'invalid_owned_control_message')
                result['stop_reason'] = 'owned_control_file'
                break
            await asyncio.sleep(.2)
        result.setdefault('stop_reason', 'lease_expired' if datetime.now(UTC)>=scope.expires_at else 'child_exit')
        result['status'] = 'stopped'
    except asyncio.CancelledError:
        # Finish the owned shutdown and cleanup instead of leaving a live lease
        # when the serve CLI receives cancellation/KeyboardInterrupt.
        result.update(status='failed', cancellation_received=True)
    except Exception as error:
        result.update(status='failed', fatal=sanitized(error))
    finally:
        if sock is not None:
            sock.close()
        if child is not None:
            try:
                if child.returncode is None:
                    try:
                        child.stdin.write(b'STOP\n')
                        await child.stdin.drain()
                        child.stdin.close()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    await asyncio.wait_for(child.wait(), 30)
            except TimeoutError:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 10)
                except TimeoutError:
                    child.kill()
                    await child.wait()
                result.update(status='failed', forced_child_shutdown=True)
            if output_drain is not None:
                result['child_output_drain'] = await output_drain.wait_closed()
                if not result['child_output_drain']['complete'] or result['child_output_drain']['errors']:
                    result['status'] = 'failed'
            resources['child_returncode'] = child.returncode
            if child_path.exists():
                result['child'] = json.loads(child_path.read_text())
            if child.returncode!=0 or result.get('child', {}).get('status')!='passed':
                result['status'] = 'failed'
            elif result['status']=='stopped':
                result['status'] = 'passed'
        if locked:
            resources['status'] = 'stopped' if result['status']=='passed' else 'serve_failed'
            save_json(path, resources)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
        log.close()
        temporary.cleanup()
        save_json(result_path, result)
    if auto_cleanup and locked:
        task_receipt = result.get('child', {}).get('native_task_replay') or {}
        if ('native_task_replay' in resources and (task_receipt.get('status') != 'closed'
                or task_receipt.get('observer_joined') is not True
                or task_receipt.get('release_loop_joined') is not True)):
            result['cleanup'] = {'status': 'blocked_task_replay_shutdown'}
        else:
            result['cleanup'] = await cleanup_prepared_host(manifest=path, dsn=dsn)
        if result['cleanup']['status']!='passed':
            result['status'] = 'failed'
        save_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--scope-file', type=Path)
    parser.add_argument('--child-result', type=Path)
    parser.add_argument('--socket-fd', type=int)
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--hindsight-url')
    parser.add_argument('--hindsight-api-key-file', type=Path)
    parser.add_argument('--mode', choices=('qualify','prepare','serve','cleanup'), default='qualify')
    parser.add_argument('--resource-manifest', type=Path)
    parser.add_argument('--lease-seconds', type=int, default=300)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--profile', choices=(NO_MEDIA,NATIVE_MEDIA), default=NO_MEDIA)
    parser.add_argument('--native-pool-max-size',type=int,choices=range(1,6),
                        help='native default 5/min 1; explicit 1 reproduces contention; no-media stays 1')
    parser.add_argument('--workload',choices=(NO_WORKLOAD,REASONING_REMINDER,VISUAL_MEMORY),default=NO_WORKLOAD)
    parser.add_argument('--frontier-model')
    parser.add_argument('--frontier-reasoning-effort',choices=('low','medium','high','xhigh','max','ultra'))
    parser.add_argument('--voice-route',choices=('realtime','live'),default='realtime')
    parser.add_argument('--live-model',help='explicit configured model for isolated GPT Live baseline')
    parser.add_argument('--live-voice',help='explicit configured voice for isolated GPT Live baseline')
    args = parser.parse_args()
    if args.child:
        passed = asyncio.run(child_server(args.scope_file, args.child_result, args.socket_fd))
    else:
        from app.settings import JarvisSettings
        dsn = os.getenv('JARVIS_SECOND_BRAIN_TEST_DB_URL') or JarvisSettings().SUPABASE_DB_URL
        if args.mode in {'qualify','prepare'}:
            require(args.artifact_dir and args.hindsight_url and args.hindsight_api_key_file,
                    'explicit_qualification_arguments_required')
            report = asyncio.run(qualify_frontend_host(dsn=dsn, directory=args.artifact_dir,
                hindsight_url=args.hindsight_url, api_key_file=args.hindsight_api_key_file,
                prepare_only=args.mode=='prepare', lifetime_seconds=args.lease_seconds, profile=args.profile,
                native_pool_max_size=args.native_pool_max_size,workload=args.workload,
                frontier_model=args.frontier_model,frontier_reasoning_effort=args.frontier_reasoning_effort,
                voice_route=args.voice_route,live_model=args.live_model,live_voice=args.live_voice))
        elif args.mode=='serve':
            require(args.resource_manifest, 'resource_manifest_required')
            report = asyncio.run(serve_prepared_host(manifest=args.resource_manifest, dsn=dsn, port=args.port))
        else:
            require(args.resource_manifest, 'resource_manifest_required')
            report = asyncio.run(cleanup_prepared_host(manifest=args.resource_manifest, dsn=dsn))
        print(json.dumps({'status':report['status'], 'report':report['report_path'],
            'resource_manifest':report.get('resource_manifest'), 'checks':report.get('checks'),
            'cleanup':report.get('cleanup'), 'fatal':report.get('fatal')}, default=str))
        passed = report['status'] in {'passed','prepared'}
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':
    main()
