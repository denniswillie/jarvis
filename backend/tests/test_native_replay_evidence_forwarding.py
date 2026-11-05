"""Literal archived evidence and cleanup ordering, using offline fixtures only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from evals.responsive_voice import native_capture_evidence as reader
from evals.responsive_voice import native_replay_run as runner
from evals.responsive_voice.native_task_barrier import AtomicReceiptFile, decode_receipt
from evals.responsive_voice.native_task_barrier_host import BarrierReceiptHistory
from evals.responsive_voice.native_task_release_control import write_private
from test_native_task_capture_controls import task_fixture
from test_native_task_replay_run import TaskFixture
from test_native_replay_run import Fixture


@pytest.fixture(autouse=True)
def no_external_execution():
    with patch('socket.socket.connect', side_effect=AssertionError('network forbidden')), \
         patch('subprocess.Popen', side_effect=AssertionError('process forbidden')), \
         patch('asyncpg.connect', side_effect=AssertionError('SQL forbidden')), \
         patch('asyncpg.create_pool', side_effect=AssertionError('SQL forbidden')):
        yield


@pytest.fixture
def original_documents():
    capture, pcm, docs = task_fixture()
    # Deliberately noncanonical JSON bytes: reparsing/reserializing would break
    # the real reader's literal manifest/scope digest checks.
    docs['manifest_bytes'] = b' \n' + docs['manifest_bytes'] + b'\n '
    docs['scope_bytes'] = b'\n' + docs['scope_bytes'] + b'\n'
    for key in ('session_binding_bytes', 'stimulus_completion_bytes'):
        value = json.loads(docs[key])
        value.update(manifest_file_sha256=hashlib.sha256(docs['manifest_bytes']).hexdigest(),
                     scope_file_sha256=hashlib.sha256(docs['scope_bytes']).hexdigest())
        docs[key] = json.dumps(value, indent=1).encode() + b'\n'
    return capture, pcm, docs


@pytest.fixture
def archived(tmp_path, original_documents):
    capture, pcm, docs = original_documents
    tmp_path.chmod(0o700)
    staged, controls, observations = (tmp_path/name for name in ('staged', 'controls', 'task-replay-observations'))
    for path in (staged, controls, observations):
        path.mkdir(mode=0o700)
    scope = json.loads(docs['scope_bytes'])
    run_id, barrier_id = scope['run_id'], scope['barrier_id']
    write_private(staged, 'replay.json', docs['manifest_bytes'], exclusive=True)
    write_private(staged, 'task-scope.json', docs['scope_bytes'], exclusive=True)
    configuration = tmp_path/'task-replay-configuration.json'
    runner.save(configuration, {'run_id':run_id, 'barrier_id':barrier_id,
        'manifest_file':str(staged/'replay.json'), 'scope_file':str(staged/'task-scope.json'),
        'control_root':str(controls)})
    records = []
    for key, name in (('session_binding_bytes','session-binding.json'),
                      ('stimulus_completion_bytes','stimulus-completion.json')):
        write_private(observations, name, docs[key], exclusive=True)
        records.append({'path':str(observations/name), 'sha256':hashlib.sha256(docs[key]).hexdigest()})
    first = decode_receipt(docs['barrier_receipts'][0])
    binding = {key:getattr(first.scope, key) for key in
        ('run_id','barrier_id','owner_id','database_schema','model','host_expires_at')}
    history = BarrierReceiptHistory(tmp_path/'task-barrier-history', binding=binding)
    with AtomicReceiptFile(controls) as exchange:
        for raw in docs['barrier_receipts']:
            receipt = decode_receipt(raw)
            history.publish(exchange, receipt, now=receipt.observed_at)
    index = history.close()
    closed = {'format':'jarvis.native_task_replay.status.v1','run_id':run_id,'barrier_id':barrier_id,
        'status':'closed','observer_joined':True,'release_loop_joined':True,'failure_code':None}
    report = {'run_id':UUID(run_id).hex,'staged_inputs':str(staged),
        'task_replay':{'metadata':{'configuration_path':str(configuration),'barrier_id':barrier_id},
            'control_root':str(controls),'observations':records,'closed':closed},
        'cleanup':{'app_termination':{'confirmed':True},'worker':{'status':'closed',
            'worker_joined':True,'readiness_inactive':True,'pipes_complete':True,'secrets_removed':True}},
        'host_result':{'status':'passed','child_output_drain':{'complete':True},'child':{
            'normal_shutdown':True,'runtime_closed':True,'native_task_replay':deepcopy(closed)}}}
    return tmp_path, report, capture, pcm, docs, index


def test_actual_reader_receives_literal_bound_bytes_and_ordered_producer_history(archived):
    root, report, capture, pcm, docs, _ = archived
    forwarded = runner.retained_task_capture_evidence(report, root)
    assert forwarded == docs
    assert (root/'task-replay-inputs/replay.json').read_bytes() == docs['manifest_bytes']
    assert (root/'task-replay-inputs/task-scope.json').read_bytes() == docs['scope_bytes']
    capture_path = root/'capture.json'
    runner.save(capture_path, capture)
    (root/'decoded.pcm').write_bytes(pcm)
    result = reader.read_capture(capture_path, task_evidence=forwarded)
    assert result['task_control_evidence']['verdict'] == 'validated_control_metadata'
    assert result['actual_cuts'] is None and not result['hearing_measured']
    assert not result['answer_quality_judged']
    assert report['task_replay']['analysis_inputs']['receipt_count'] == len(docs['barrier_receipts'])


@pytest.mark.parametrize('fault', ['missing_control','changed_control','missing_scope','oversize_scope',
    'unclosed','incomplete','index_error','published_count','extra_receipt','missing_receipt',
    'pending_file','wrong_sequence','wrong_index_scope','symlink_receipt'])
def test_missing_truncated_or_foreign_archives_never_become_complete_evidence(archived, fault):
    root, report, _, _, docs, index = archived
    history = root/'task-barrier-history'
    first = history/'00000000000000000001.json'
    if fault == 'missing_control':
        (root/'task-replay-observations/session-binding.json').unlink()
    elif fault == 'changed_control':
        (root/'task-replay-observations/session-binding.json').write_bytes(b'{}')
    elif fault == 'missing_scope':
        (root/'staged/task-scope.json').unlink()
    elif fault == 'oversize_scope':
        (root/'staged/task-scope.json').write_bytes(b' ' * 8193)
    elif fault == 'extra_receipt':
        (history/'00000000000000000003.json').write_bytes(docs['barrier_receipts'][0])
    elif fault == 'missing_receipt':
        first.unlink()
    elif fault == 'pending_file':
        (history/'.pending-incomplete').write_bytes(b'not a published receipt')
    elif fault == 'wrong_sequence':
        first.write_bytes(docs['barrier_receipts'][1])
    elif fault == 'symlink_receipt':
        first.unlink()
        first.symlink_to(history/'00000000000000000002.json')
    else:
        key, value = {'unclosed':('closed',False),'incomplete':('complete',False),
            'index_error':('error','history_archive_failed'),
            'published_count':('last_published_sequence',index['receipt_count']+1),
            'wrong_index_scope':('run_id','00000000-0000-4000-8000-000000000000')}[fault]
        runner.save(history/'index.json',dict(index, **{key:value}))
    with pytest.raises(Exception):
        runner.retained_task_capture_evidence(report, root)
    assert 'analysis_inputs' not in report['task_replay']


@pytest.mark.parametrize('fault', ['observer','release_loop','host_failure','control_failure','host_unjoined'])
def test_reader_inputs_require_every_producer_join_and_error_free_control(archived, fault):
    root, report, *_ = archived
    if fault in ('observer','release_loop'):
        report['task_replay']['closed'][fault+'_joined'] = False
    elif fault == 'host_failure':
        report['host_result']['child']['native_task_replay']['failure_code'] = 'receipt_history_failed'
    elif fault == 'control_failure':
        report['task_control_capture_error'] = {'error_type':'OSError'}
    else:
        report['host_result']['child']['runtime_closed'] = False
    with pytest.raises(Exception):
        runner.retained_task_capture_evidence(report, root)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [None,'missing_evidence','reader_error'])
async def test_task_runner_forwards_exact_documents_after_join_and_always_cleans(tmp_path, original_documents, failure):
    f = TaskFixture(tmp_path)
    docs = original_documents[2]
    def evidence(report, artifact_dir):
        assert f.child_stopped
        assert f.events.index('control_closed') < f.events.index('worker_closed') < f.events.index('host_stopped')
        assert f.events.index('canonical_export') < f.events.index('host_stopped')
        f.events.append('analysis_inputs')
        if failure == 'missing_evidence':
            raise FileNotFoundError('offline missing retained evidence')
        return docs
    def analyze(path, **kwargs):
        assert path == f.options.artifact_dir/'native-capture/capture.json'
        assert set(kwargs) == {'task_evidence'}
        assert kwargs['task_evidence'] is docs
        f.events.append('reader')
        if failure == 'reader_error':
            raise reader.EvidenceError('offline malformed retained evidence')
        return {'actual_cuts':None,'hearing_measured':False,'answer_quality_judged':False}
    with patch.object(runner,'retained_task_capture_evidence',side_effect=evidence), \
         patch.object(reader,'read_capture',side_effect=analyze) as called:
        result = await runner.run_once(f.options,host=f,process=f)
    assert f.cleanups == 1 and result['local_cleanup_complete']
    assert result['status'] == 'lifecycle_complete' and result['voice_pass'] is False
    assert f.events.index('analysis_inputs') < f.events.index('fixture_cleanup')
    assert all(not Path(path).exists() for path in result['token_files'])
    assert not Path(result['active_ledger']).exists()
    assert called.call_count == (0 if failure == 'missing_evidence' else 1)
    assert ('capture_analysis_error' in result) == (failure is not None)
    assert result['task_replay']['capture_analysis']['status'] == ('unqualified' if failure else 'reader_completed')


@pytest.mark.asyncio
async def test_unjoined_task_observer_does_not_analyze_or_weaken_existing_cleanup_gate(tmp_path):
    f = TaskFixture(tmp_path)
    f.control_bad_join = True
    with patch.object(runner,'retained_task_capture_evidence') as evidence, patch.object(reader,'read_capture') as analyze:
        result = await runner.run_once(f.options,host=f,process=f)
    evidence.assert_not_called()
    analyze.assert_not_called()
    assert result['status'] == 'cleanup_pending' and f.cleanups == 0
    assert result['task_replay']['capture_analysis']['status'] == 'unqualified'


@pytest.mark.asyncio
async def test_ordinary_reader_path_keeps_original_arguments_and_order(tmp_path):
    f = Fixture(tmp_path)
    def analyze(path):
        assert not f.child_stopped
        assert path == f.options.artifact_dir/'native-capture/capture.json'
        return {'actual_cuts':None,'hearing_measured':False}
    with patch.object(reader,'read_capture',side_effect=analyze) as called:
        result = await runner.run_once(f.options,host=f,process=f)
    called.assert_called_once_with(f.options.artifact_dir/'native-capture/capture.json')
    assert result['status'] == 'lifecycle_complete' and f.cleanups == 1
