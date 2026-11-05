"""Opt-in q03 canonical pipeline qualification. Default CLI only prepares files.

No replacement perception engine: run_replay -> PipelineReplayPort -> existing
camera/brief/analyzer/publication/evidence/focus/attention owners. Hosted execution
requires a NEW, exact, expiring, one-use grant. The old fifteen-image diagnostic
grant is deliberately incompatible. No Hindsight worker or delivery is started.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx

from app.model_providers.gemini_fast_visual import GeminiFastVisualModel
from app.visual_memory.gemini_client import (
    GEMINI_GENERATE_CONTENT_URL, GeminiVisualAnalyzerClient,
    gemini_visual_config_from_settings,
)
from evals.responsive_voice.focus_pipeline_replay import FocusPipelineReplay
from evals.responsive_voice.video_replay import ReplayCase, read_frame_asset, run_replay
from evals.second_brain.source_fixture import source_fixture
from evals.second_brain.fixtures import _owner_tables, _quote, validate_manifest


GRANT_FORMAT = 'jarvis.pov.hosted_pipeline_permission.v1'
CASE_IDS = ('q03-general', 'q03-focused')
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / (
    '.jarvis-dev/pov-assets-20261031-B7oCGPHS/qualification-15f-20261031-01/manifest.json')
REVIEW = {
    'revision': 'q03-provisional-cross-cut-v2',
    'status': 'agent_review_not_human_gold',
    'source': 'Main reviewed all three original-resolution q03 frames; 2026-10-31',
    'observations': [
        'Existing long orange/yellow-orange pepper strips are visible.',
        'The blade across their ends changes position; the hand occludes contact.',
        'No finished pile or overall completion is established.',
    ],
    'allowed_paraphrases': [
        'cutting existing strips', 'cutting across the ends of the pepper strips',
        'ongoing pepper cutting with the contact/result partly hidden',
    ],
    'scoring_rules': [
        'The word strips alone is not a contradiction.',
        'Do not require the word dice.',
        'Only an explicit unsupported claim of making new long strips is the stated contradiction.',
        'Do not infer a finished pile or completion from an occluded contact point.',
        'Semantic assessment requires review, not keyword matching.',
    ],
    'historical_labels_and_results': 'unchanged',
}


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def write_private(path: Path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf8') as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write('\n')


def safe_error(error):
    return {'error_type': type(error).__name__, 'code': getattr(error, 'code', None)}


@dataclass(frozen=True)
class Limits:
    calls: int = 12
    request_seconds: float = 25
    trial_seconds: float = 300
    cleanup_seconds: float = 30

    def __post_init__(self):
        require(type(self.calls) is int and 1 <= self.calls <= 12, 'invalid_shared_call_cap')
        for value, maximum in ((self.request_seconds, 25), (self.trial_seconds, 300),
                               (self.cleanup_seconds, 60)):
            require(type(value) in (int, float) and 0 < value <= maximum, 'invalid_deadline')

    def document(self):
        return {'max_model_calls': self.calls, 'max_http_attempts': self.calls,
            'request_seconds': self.request_seconds, 'trial_seconds': self.trial_seconds,
            'cleanup_seconds': self.cleanup_seconds, 'concurrency': 1, 'automatic_retries': 0}


@dataclass(frozen=True)
class Payload:
    manifest_path: Path
    manifest_sha256: str
    asset_root: Path
    cases: tuple[ReplayCase, ...]
    case_hashes: dict[str, str]
    image_hashes: tuple[str, ...]

    def binding(self):
        return {'payload_manifest_sha256': self.manifest_sha256,
            'case_sha256s': self.case_hashes, 'image_sha256s': list(self.image_hashes),
            'review_revision_sha256': sha(encoded(REVIEW)),
            'context_policy': 'isolated_canonical_observational_focus_monitor',
            'context_sha256s': {c.case_id: sha(encoded(c.inputs.model_dump(mode='json')))
                               for c in self.cases}}


def load_payload(manifest_path: Path) -> Payload:
    """Read only the pinned q03 pair. No clips, extra frames or remote data."""
    manifest_path = manifest_path.resolve(strict=True)
    raw = manifest_path.read_bytes()
    require(len(raw) <= 2_000_000, 'manifest_too_large')
    manifest = json.loads(raw)
    root = Path(manifest['asset_root']).resolve(strict=True)
    group, = [group for group in manifest['cases'] if group['id'] == 'q03']

    def hashes(value):
        if isinstance(value, dict):
            return set().union(*(hashes(v) for v in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(hashes(v) for v in value)) if value else set()
        return {value} if isinstance(value, str) and len(value) == 64 else set()

    reviewed = hashes(group['frames'])
    cases, case_hashes, images = [], {}, set()
    for identity in CASE_IDS:
        case_path = (manifest_path.parent / 'cases' / (identity + '.json')).resolve(strict=True)
        require(case_path.is_relative_to(manifest_path.parent), 'case_path_outside_package')
        case_bytes = case_path.read_bytes()
        require(len(case_bytes) <= 100_000, 'case_too_large')
        case = ReplayCase.model_validate_json(case_bytes)
        require(case.case_id == identity and len(case.inputs.frames) == 3, 'unexpected_q03_case')
        require(case.permission.local_evaluation and case.permission.retain_derived_results,
                'local_permission_missing')
        require(bool(case.inputs.initial_context), 'canonical_initial_context_required')
        require(not case.inputs.events, 'initial_pair_has_no_authorized_dynamic_events')
        for frame in case.inputs.frames:
            require(frame.sha256 in reviewed, 'frame_not_in_reviewed_q03_group')
            read_frame_asset(root, frame)
            images.add(frame.sha256)
        cases.append(case)
        case_hashes[identity] = sha(case_bytes)
    require(cases[0].inputs.frames == cases[1].inputs.frames, 'paired_frame_schedule_differs')
    return Payload(manifest_path, sha(raw), root, tuple(cases), case_hashes, tuple(sorted(images)))


def grant_template(payload: Payload, *, model='gemini-3.1-flash-lite', limits=Limits()):
    return {'format': GRANT_FORMAT, 'grant_id': str(uuid4()), **payload.binding(),
        'model': model, 'provider': 'google_gemini', 'limits': limits.document(),
        'hosted_inference': False, 'retain_local_results': True, 'public_demo': False,
        'other_personal_context': False, 'request_store': False,
        'approved_by': '', 'approval_reference': '', 'expires_at': None,
        'permission_reference': '', 'provider_policy_reference': '',
        'note': 'NOT APPROVAL. A fresh exact grant is required; the old diagnostic grant is consumed.'}


def validate_grant(payload: Payload, grant: dict, *, now=None) -> Limits:
    require(grant.get('format') == GRANT_FORMAT, 'fresh_pipeline_grant_required')
    UUID(grant['grant_id'])
    for key, value in payload.binding().items():
        require(grant.get(key) == value, 'authorization_binding_mismatch')
    require(grant.get('hosted_inference') is True and grant.get('retain_local_results') is True,
            'hosted_authorization_missing')
    require(grant.get('public_demo') is False and grant.get('other_personal_context') is False
            and grant.get('request_store') is False, 'authorization_scope_invalid')
    for key in ('approved_by', 'approval_reference', 'permission_reference', 'provider_policy_reference'):
        require(isinstance(grant.get(key), str) and bool(grant[key].strip()), 'authorization_reference_missing')
    require(grant.get('provider') == 'google_gemini' and isinstance(grant.get('model'), str)
            and bool(grant['model']), 'provider_not_authorized')
    expiry = datetime.fromisoformat(grant['expires_at'])
    require(expiry.tzinfo is not None and expiry > (now or datetime.now(UTC)), 'authorization_expired')
    limits = grant['limits']
    result = Limits(limits['max_model_calls'], limits['request_seconds'],
                    limits['trial_seconds'], limits['cleanup_seconds'])
    require(limits == result.document(), 'authorization_limits_invalid')
    return result


_CONSUMPTION_AUTHORITY = object()


@dataclass(frozen=True)
class ConsumedGrant:
    """Opaque in-process receipt; the immutable ledger remains the durable proof."""
    marker: Path
    grant_sha256: str
    _authority: object = field(repr=False, compare=False)


def consume_grant(payload, grant, ledger: Path) -> ConsumedGrant:
    validate_grant(payload, grant)
    ledger.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = ledger / (str(UUID(grant['grant_id'])) + '.json')
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'status': 'admitted_once_not_a_completion_receipt',
            'grant_sha256': sha(encoded(grant)), 'binding': payload.binding(),
            'at': datetime.now(UTC).isoformat()}, handle)
    return ConsumedGrant(marker.resolve(strict=True), sha(encoded(grant)), _CONSUMPTION_AUTHORITY)


def claim_consumed_grant(payload, grant, admission: ConsumedGrant | None) -> Path:
    """Claim once before any runtime work, including after an earlier failed run.

    A declaration or copied JSON is not a consumption receipt. Neither completion,
    cancellation nor failed setup removes this claim; ambiguous work is not replayed.
    """
    require(isinstance(admission, ConsumedGrant)
            and admission._authority is _CONSUMPTION_AUTHORITY, 'consumed_grant_receipt_required')
    require(admission.grant_sha256 == sha(encoded(grant)), 'consumed_grant_binding_mismatch')
    require(admission.marker.name == str(UUID(grant['grant_id'])) + '.json',
            'consumed_grant_identity_mismatch')
    raw = admission.marker.read_bytes()
    require(len(raw) <= 100_000, 'consumption_record_invalid')
    record = json.loads(raw)
    require(record.get('status') == 'admitted_once_not_a_completion_receipt'
            and record.get('grant_sha256') == admission.grant_sha256
            and record.get('binding') == payload.binding(), 'consumption_record_invalid')
    execution = admission.marker.with_name(admission.marker.name + '.execution.json')
    fd = os.open(execution, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'status': 'execution_claimed_once_not_a_completion_receipt',
            'grant_sha256': admission.grant_sha256, 'consumption_marker': str(admission.marker),
            'at': datetime.now(UTC).isoformat()}, handle)
    return execution


class SharedModelGate:
    """One budget and serialization lock for BOTH analyzer and owner predicates."""
    def __init__(self, model, *, limits=Limits(), execution_kind='synthetic'):
        require(execution_kind in {'synthetic', 'hosted_model'}, 'invalid_execution_kind')
        self.raw, self.limits, self.execution_kind = model, limits, execution_kind
        self.lock = asyncio.Lock()
        self.calls, self.http_attempts, self.released = [], [], {}
        self.current_case = self.ticket = None
        self.closed, self.halted = False, False

    @property
    def provider(self):
        return self.raw.provider

    @property
    def model(self):
        return self.raw.model

    def start_case(self, case):
        require(not self.lock.locked() and self.ticket is None, 'previous_model_call_still_owned')
        self.current_case, self.released = case, {}

    def release(self, frame):
        require(self.current_case is not None, 'case_not_started')
        expected = [asset for asset in self.current_case.inputs.frames
                    if asset.offset_ms == frame.offset_ms]
        require(len(expected) == 1 and expected[0].sha256 == sha(frame.image_bytes), 'unbound_frame')
        require(frame.captured_at <= datetime.now(UTC), 'future_frame')
        self.released[sha(frame.image_bytes)] = {'frame_id': str(frame.frame_id),
            'captured_at': frame.captured_at.isoformat(), 'offset_ms': frame.offset_ms,
            'released_monotonic': time.monotonic()}

    async def generate_temporal_json(self, *, prompt, images):
        async with self.lock:
            require(not self.closed and not self.halted, 'model_gate_retired')
            require(len(self.calls) < self.limits.calls, 'shared_model_budget_exhausted')
            identities = [sha(image.image_bytes) for image in images]
            require(bool(identities) and all(identity in self.released for identity in identities),
                    'image_not_yet_released_in_current_case')
            started = time.monotonic()
            entry = {'case_id': self.current_case.case_id,
                'phase': 'predicate' if prompt.startswith('Evaluate whether these original images') else 'analysis',
                'prompt_sha256': sha(prompt.encode()), 'image_sha256s': identities,
                'frames': [dict(self.released[identity]) for identity in identities], 'status': 'started'}
            self.calls.append(entry)
            self.ticket = {'prompt': prompt, 'images': tuple(images), 'used': False}
            try:
                async with asyncio.timeout(self.limits.request_seconds):
                    result = await self.raw.generate_temporal_json(prompt=prompt, images=images)
                entry['status'] = 'returned'
                return result
            except BaseException as error:
                self.halted = True
                entry.update(status='cancelled' if isinstance(error, asyncio.CancelledError) else 'failed',
                             **safe_error(error))
                raise
            finally:
                self.ticket = None
                entry['model_call_elapsed_ms'] = (time.monotonic() - started) * 1000


class ExactWireTransport(httpx.AsyncBaseTransport):
    """Reject altered payloads, non-store-false requests and retries before I/O."""
    def __init__(self, gate, delegate, *, model, actual_network):
        self.gate, self.delegate, self.model = gate, delegate, model
        self.actual_network, self.closed = actual_network, False

    async def handle_async_request(self, request):
        ticket = self.gate.ticket
        try:
            require(ticket is not None and not ticket['used'], 'missing_or_reused_request_ticket')
            ticket['used'] = True
            require(len(self.gate.http_attempts) < self.gate.limits.calls, 'http_budget_exhausted')
            require(request.method == 'POST' and str(request.url) ==
                    GEMINI_GENERATE_CONTENT_URL.format(model=self.model), 'endpoint_not_authorized')
            body = json.loads(request.content)
            require(set(body) == {'contents', 'generationConfig', 'store'} and body['store'] is False,
                    'request_store_false_required')
            parts = [{'text': ticket['prompt']}]
            for image in ticket['images']:
                parts.extend([{'text': image.label}, {'inlineData': {'mimeType': image.content_type,
                    'data': base64.b64encode(image.image_bytes).decode('ascii')}}])
            require(body['contents'] == [{'role': 'user', 'parts': parts}], 'request_payload_changed')
            require(body['generationConfig'] == {'temperature': 0.2, 'responseMimeType': 'application/json'},
                    'generation_config_changed')
        except BaseException:
            self.gate.halted = True
            raise
        row = {'case_id': self.gate.current_case.case_id, 'store': False,
            'actual_network': self.actual_network, 'status': 'dispatched',
            'request_content_sha256': sha(request.content),
            'request_content_bytes': len(request.content),
            'prompt_characters': len(ticket['prompt']),
            'prompt_utf8_bytes': len(ticket['prompt'].encode('utf-8')),
            'image_count': len(ticket['images'])}
        self.gate.http_attempts.append(row)
        started = time.monotonic()
        try:
            response = await self.delegate.handle_async_request(request)
            row.update(status='response_transport_returned', http_status=response.status_code)
            return response
        except BaseException as error:
            row.update(status='failed_or_cancelled', **safe_error(error))
            raise
        finally:
            # This is the transport-return boundary, NOT body/token/audio latency.
            row['dispatch_to_transport_return_or_error_ms'] = (time.monotonic() - started) * 1000

    async def aclose(self):
        if not self.closed:
            self.closed = True
            await self.delegate.aclose()


async def create_gemini_gate(config, *, limits, controlled_transport=None):
    """Only the explicit hosted CLI passes no controlled transport."""
    mode = 'hosted_model' if controlled_transport is None else 'synthetic'
    gate = SharedModelGate(None, limits=limits, execution_kind=mode)
    delegate = controlled_transport if controlled_transport is not None else httpx.AsyncHTTPTransport(retries=0)
    wire = ExactWireTransport(gate, delegate, model=config.model, actual_network=controlled_transport is None)
    client = GeminiVisualAnalyzerClient(config)
    await client._client.aclose()
    client._client = httpx.AsyncClient(transport=wire, timeout=httpx.Timeout(limits.request_seconds),
                                     follow_redirects=False)
    gate.raw = GeminiFastVisualModel(client, store=False)

    async def close():
        gate.closed = True
        await client.close()
    return gate, close


class HostedPipelineComposition(FocusPipelineReplay):
    """Execution labels differ; the existing canonical owners do not."""
    def __init__(self, *, gate, binding, **kwargs):
        self.gate, self.notification_callbacks, self.analysis_timing = gate, [], []
        super().__init__(model=gate, execution_kind=gate.execution_kind,
            evaluation_provenance=binding, **kwargs)
        original = self.analyzer.analyze

        async def measured(request):
            started = time.monotonic()
            row = {'window_id': str(request.window_id), 'primary_frame_id': str(request.primary_frame.frame_id),
                'captured_at': request.primary_frame.captured_at.isoformat(), 'status': 'started'}
            self.analysis_timing.append(row)
            try:
                result = await original(request)
                row.update(status='parsed', question_count=len(request.perception_brief.questions)
                           if request.perception_brief else 0)
                return result
            except BaseException as error:
                row.update(status='failed_or_cancelled', **safe_error(error))
                raise
            finally:
                row['analyzer_elapsed_ms'] = (time.monotonic() - started) * 1000
        self.analyzer.analyze = measured

    async def forbid_speech(self, *args, **kwargs):
        # Record callback occurrence only. No fake delivery or second authority.
        self.notification_callbacks.append({'at': datetime.now(UTC).isoformat(),
            'meaning': 'callback_recorded_not_spoken_or_delivered'})
        return None

    async def make_frame(self, frame):
        result = await super().make_frame(frame)
        self.gate.release(frame)
        return result

    async def collect(self):
        result = await super().collect()
        result.update(analyzer_outputs=[value.model_dump(mode='json') for value in self.analyzer.outputs],
            analyzer_timing=self.analysis_timing, recording_only_callbacks=self.notification_callbacks,
            general_visual_alert_policy_qualified=False, actual_delivery=False,
            semantic_quality_qualified=False, human_gold=False)
        return result


async def audit_fixtures(dsn, manifests):
    """Independent connection; inspect only exact manifested fixture identities."""
    import asyncpg
    owners, schemas = [], []
    for manifest in manifests:
        validate_manifest(manifest)
        owners.extend(manifest['owners'])
        schemas.extend(manifest['schemas'])
    if not manifests:
        return {'status': 'no_fixture_allocated', 'remaining_schemas': 0, 'remaining_public_owner_rows': 0}
    connection = await asyncpg.connect(dsn, timeout=10, command_timeout=10)
    try:
        remaining = await connection.fetchval('select count(*) from pg_namespace where nspname=any($1::text[])', schemas)
        rows, tables = 0, 0
        for table in await _owner_tables(connection, 'public'):
            rows += await connection.fetchval(
                f"select count(*) from public.{_quote(table['relname'])} where owner_id::text=any($1::text[])", owners)
            tables += 1
        return {'status': 'passed' if not remaining and not rows else 'failed',
            'independent_connection': True, 'remaining_schemas': remaining,
            'remaining_public_owner_rows': rows, 'public_tables_checked': tables,
            'exact_schemas': schemas, 'exact_owners': owners}
    finally:
        await connection.close(timeout=5)


async def run_pair(payload, gate, *, dsn, output_dir, grant=None, admission=None):
    """Fresh fixture/session per condition, common bounded model budget, no delivery."""
    import asyncpg
    from app.migrations.runner import quote_schema

    execution_marker = None
    if gate.execution_kind == 'hosted_model':
        require(grant is not None, 'hosted_grant_required')
        require(validate_grant(payload, grant) == gate.limits, 'grant_budget_mismatch')
        require(gate.model == grant['model'], 'configured_model_mismatch')
        execution_marker = claim_consumed_grant(payload, grant, admission)
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    report = {'format': 'jarvis.pov.canonical_pipeline.v1', 'status': 'running',
        'execution_kind': gate.execution_kind, 'binding': payload.binding(), 'limits': gate.limits.document(),
        'cases': [], 'fixture_manifests': [], 'cleanup': None, 'audit': None,
        'semantic_quality_qualified': False, 'human_gold': False, 'actual_delivery': False,
        'remote_hindsight_bank_created': False, 'hindsight_calls': 0, 'automatic_retries': 0}
    if execution_marker is not None:
        report.update(grant_consumption_marker=str(admission.marker),
                      grant_execution_marker=str(execution_marker))
    manifests = []
    started = time.monotonic()

    def save():
        report.update(model_calls=gate.calls, http_attempts=gate.http_attempts)
        write_private(output_dir / 'report.json', report)

    save()
    try:
        async with asyncio.timeout(gate.limits.trial_seconds):
            for case in payload.cases:
                require(not gate.halted, 'previous_model_failure_no_retry')
                gate.start_case(case)
                connection = composition = None
                async with source_fixture(dsn, output_dir / 'fixtures', test_name='hosted_pipeline:' + case.case_id) as fixture:
                    manifests.append(fixture.fixture.manifest)
                    report['fixture_manifests'].append(str(fixture.fixture.path))
                    schema, = fixture.fixture.manifest['schemas']
                    try:
                        connection = await asyncpg.connect(dsn, timeout=10, command_timeout=20)
                        await connection.execute(f'set search_path to {quote_schema(schema)},pg_catalog,extensions')
                        composition = HostedPipelineComposition(pool=fixture.pool, connection=connection,
                            owner=fixture.owner, area=fixture.area, schema=schema, gate=gate,
                            binding={'payload_manifest_sha256': payload.manifest_sha256,
                                'case_sha256': payload.case_hashes[case.case_id],
                                'review_revision': REVIEW['revision'],
                                'authorization_sha256': sha(encoded(grant)) if grant else None})
                        await composition.prepare(case.inputs.initial_context)
                        admitted = case
                        if gate.execution_kind == 'hosted_model':
                            admitted = case.model_copy(update={'permission': case.permission.model_copy(update={
                                'hosted_inference': True, 'evidence': 'Fresh exact pipeline grant ' + grant['grant_id']})})
                        result = await run_replay(admitted, asset_root=payload.asset_root,
                            port=composition.port(), origin=datetime.now(UTC), realtime=True,
                            max_model_calls=gate.limits.calls, timeout_seconds=gate.limits.trial_seconds,
                            cleanup_seconds=gate.limits.cleanup_seconds)
                        report['cases'].append(result)
                        save()
                        require(result['execution_status'] == 'completed' and result['port_closed'],
                                'pipeline_replay_incomplete')
                        require(bool(composition.analyzer.outputs), 'no_analyzer_output')
                        require(all(request.perception_brief and request.perception_brief.questions
                                    and request.perception_brief_input_digest
                                    and request.camera_connection_id
                                    for request in composition.analyzer.requests), 'canonical_assignment_missing')
                        require(not gate.halted, 'model_failure_not_acceptance')
                    finally:
                        try:
                            if composition is not None:
                                async with asyncio.timeout(gate.limits.cleanup_seconds):
                                    await composition.pipeline.close()
                                    await composition.close()
                        finally:
                            if connection is not None:
                                await connection.close(timeout=5)
                save()
        report['status'] = 'completed_unscored'
    except Exception as error:
        report.update(status='failed', failure=safe_error(error))
    finally:
        report['main_elapsed_ms'] = (time.monotonic() - started) * 1000
        report['cleanup'] = {'manifest_statuses': [m['status'] for m in manifests]}
        try:
            async with asyncio.timeout(60):
                report['audit'] = await audit_fixtures(dsn, manifests)
            require(report['audit']['status'] in {'passed', 'no_fixture_allocated'}, 'fixture_residue')
        except Exception as error:
            report['status'] = 'failed'
            report['audit'] = {'status': 'failed_or_unknown', **safe_error(error)}
        report['total_elapsed_ms'] = (time.monotonic() - started) * 1000
        save()
    return report


async def execute(payload, grant_path, *, output_dir, settings_env_file):
    from dotenv import dotenv_values
    from app.settings import JarvisSettings
    grant = json.loads(grant_path.read_bytes())
    limits = validate_grant(payload, grant)
    values = dotenv_values(settings_env_file)
    require(bool(values.get('SUPABASE_DB_URL')), 'database_not_configured')
    settings = JarvisSettings(_env_file=settings_env_file)
    config = gemini_visual_config_from_settings(settings)
    require(config is not None and config.model == grant['model'], 'configured_model_mismatch')
    ledger = Path(__file__).resolve().parents[3] / '.jarvis-dev/hosted-pipeline-admissions'
    admission = consume_grant(payload, grant, ledger)
    gate = close = None
    try:
        gate, close = await create_gemini_gate(config, limits=limits)
        report = await run_pair(payload, gate, dsn=values['SUPABASE_DB_URL'], output_dir=output_dir,
                                grant=grant, admission=admission)
        return report
    finally:
        if close is not None:
            async with asyncio.timeout(limits.cleanup_seconds):
                await close()
            if 'report' in locals():
                report['provider_client_closed'] = True
                write_private(output_dir / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='gemini-3.1-flash-lite')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--settings-env-file', type=Path)
    parser.add_argument('--execute-hosted', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    payload = load_payload(args.manifest)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    write_private(args.output_dir / 'review-q03-v2.json', REVIEW)
    if not args.execute_hosted:
        write_private(args.output_dir / 'grant-template.json', grant_template(payload, model=args.model))
        write_private(args.output_dir / 'plan.json', {'status': 'prepared_not_authorized_or_executed',
            'binding': payload.binding(), 'limits': Limits().document(), 'case_ids': list(CASE_IDS),
            'model_calls': 0, 'database_calls': 0, 'public_demo': False,
            'source_calendar_capture_dates': 'unknown; preserve clip offsets under an explicit replay origin',
            'scoring': 'manual semantic review; no keyword verdict or human-gold claim'})
        print(json.dumps({'status': 'prepared_not_executed', 'output_dir': str(args.output_dir)}))
        return 0
    require(args.authorization is not None and args.settings_env_file is not None,
            'fresh_authorization_and_settings_file_required')
    report = asyncio.run(execute(payload, args.authorization, output_dir=args.output_dir,
                                settings_env_file=args.settings_env_file))
    print(json.dumps({'status': report['status'], 'report': str(args.output_dir / 'report.json')}))
    return 0 if report['status'] == 'completed_unscored' else 1


if __name__ == '__main__':
    raise SystemExit(main())
