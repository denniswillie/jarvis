"""Exact producer bytes and lifecycle evidence; entirely offline controlled SQL rows."""
import asyncio
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest

from evals.responsive_voice.native_task_barrier import (
    AtomicReceiptFile, BarrierError, CanonicalTaskObserver, decode_receipt,
)
from evals.responsive_voice.native_task_barrier_host import BarrierReceiptHistory
from test_native_task_barrier import NOW, case
from test_native_task_barrier_host import setup, attach, readiness, invoke, until
from test_frontend_task_replay_host import fixture, running, publish_configuration, eventually


def roots(tmp_path):
    root = tmp_path.resolve()
    root.chmod(0o700)
    controls, operator = root / 'controls', root / 'operator'
    controls.mkdir(mode=0o700)
    operator.mkdir(mode=0o700)
    return controls, operator / 'task-barrier-history'


def history(root, scope, **kwargs):
    return BarrierReceiptHistory(root, binding={key: getattr(scope, key) for key in (
        'run_id', 'barrier_id', 'owner_id', 'database_schema', 'model', 'host_expires_at')}, **kwargs)


async def receipt(s):
    return await CanonicalTaskObserver(s.pool, s.scope, s.witness,
        ttl_seconds=2, clock=lambda: NOW).observe()


def entries(root):
    return sorted(root.glob('[0-9]' * 20 + '.json'))


@pytest.mark.asyncio
async def test_exact_raw_bytes_blocked_and_eligible_are_immutable_and_ordered(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    first = await receipt(setup)
    expected = []
    with AtomicReceiptFile(controls) as exchange:
        for sequence in range(1, 4):
            value = first.model_copy(update={'sequence': sequence})
            if sequence == 2:
                value = value.model_copy(update={'status': 'blocked', 'claim': None, 'reason': 'canonical_unavailable'})
            archived.publish(exchange, value, now=NOW)
            expected.append(exchange.read())
            assert (root / f'{sequence:020d}.json').read_bytes() == expected[-1]
            assert [p.read_bytes() for p in entries(root)] == expected
    status = archived.close()
    assert status['closed'] and status['complete'] and status['receipt_count'] == 3
    assert [decode_receipt(data).sequence for data in expected] == [1, 2, 3]
    assert [decode_receipt(data).status for data in expected] == ['eligible', 'blocked', 'eligible']
    assert all(p.stat().st_mode & 0o777 == 0o600 and p.stat().st_nlink == 1 for p in entries(root))
    assert json.loads((root / 'index.json').read_bytes()) == status
    shutil.rmtree(controls)
    assert [p.read_bytes() for p in entries(root)] == expected  # Simulator cleanup cannot remove host evidence.


@pytest.mark.asyncio
async def test_actual_host_poll_and_close_withdrawal_all_have_original_bytes(setup, tmp_path):
    controls, root = roots(tmp_path)
    host = attach(setup)
    published = []
    original = AtomicReceiptFile.publish

    def record(exchange, value, **kwargs):
        original(exchange, value, **kwargs)
        published.append(exchange.read())

    with patch.object(AtomicReceiptFile, 'publish', record):
        try:
            await host.configure(worker_readiness=readiness(setup), control_root=controls, history_root=root)
            await invoke(setup)
            await until(lambda: host.sequence >= 3)
        finally:
            closed = await host.close()
    assert closed['joined'] and closed['receipt_withdrawal'] == 'blocked_or_expired'
    assert closed['receipt_history']['closed'] and closed['receipt_history']['complete']
    assert [p.read_bytes() for p in entries(root)] == published
    assert len(published) == closed['sequence'] >= 4
    assert decode_receipt(published[-1]).status == 'blocked'
    assert all(decode_receipt(data).admission == host.admission for data in published)
    assert all(b'sensitive' not in data for data in published)
    assert {p.name for p in controls.iterdir()} == {'task-barrier.json', 'task-barrier.json.lock'}


@pytest.mark.asyncio
async def test_stage_failure_prevents_live_publication_and_is_not_hidden(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange, patch.object(exchange, 'publish') as publish:
        with patch.object(archived, '_stage', side_effect=OSError('private-error-text')):
            with pytest.raises(BarrierError, match='receipt_history_failed'):
                archived.publish(exchange, value, now=NOW)
        publish.assert_not_called()
    status = archived.close()
    assert status['error'] == 'history_stage_failed' and not status['complete']
    assert status['last_published_sequence'] == 0 and not entries(root)
    assert 'private-error-text' not in (root / 'index.json').read_text()


@pytest.mark.asyncio
async def test_archive_failure_preserves_live_visibility_and_pending_exact_bytes(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange:
        with patch.object(archived, '_commit', side_effect=OSError('full')):
            with pytest.raises(BarrierError, match='receipt_history_failed'):
                archived.publish(exchange, value, now=NOW)
        live = exchange.read()
        assert archived.last_published == 1 and archived.last_archived == 0
        assert (root / archived.pending_file).read_bytes() == live
        with patch.object(exchange, 'publish') as publish:
            with pytest.raises(BarrierError):
                archived.publish(exchange, value.model_copy(update={'sequence': 2}), now=NOW)
            publish.assert_not_called()
    status = archived.close()
    assert not status['complete'] and status['error'] == 'history_archive_failed'


@pytest.mark.asyncio
async def test_publish_error_after_actual_visibility_archives_but_does_not_claim_success(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange:
        original = exchange.publish

        def uncertain(value, **kwargs):
            original(value, **kwargs)
            raise BarrierError('control_publication_failed')

        with patch.object(exchange, 'publish', uncertain), pytest.raises(BarrierError):
            archived.publish(exchange, value, now=NOW)
        assert entries(root)[0].read_bytes() == exchange.read()
    status = archived.close()
    assert status['last_published_sequence'] == status['last_archived_sequence'] == 1
    assert not status['complete'] and status['error'] == 'live_publication_failed_after_visibility'


@pytest.mark.asyncio
async def test_unconfirmed_publication_does_not_fabricate_actual_history(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange, patch.object(exchange, 'publish', side_effect=BarrierError('failed')):
        with pytest.raises(BarrierError):
            archived.publish(exchange, value, now=NOW)
    status = archived.close()
    assert status['unconfirmed_sequence'] == 1 and not entries(root)
    assert status['last_published_sequence'] == 0 and not status['complete']


@pytest.mark.asyncio
async def test_index_failure_retains_receipt_but_never_complete_success(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange, patch.object(archived, '_index', side_effect=OSError('index')):
        with pytest.raises(BarrierError):
            archived.publish(exchange, value, now=NOW)
        assert entries(root)[0].read_bytes() == exchange.read()
        assert not json.loads((root / 'index.json').read_bytes())['closed']
    status = archived.close()
    assert status['error'] == 'history_index_failed' and not status['complete']


@pytest.mark.asyncio
async def test_capacity_is_checked_before_live_write_without_overwrite(setup, tmp_path):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope, max_receipts=2)
    value = await receipt(setup)
    with AtomicReceiptFile(controls) as exchange:
        for seq in (1, 2):
            archived.publish(exchange, value.model_copy(update={'sequence': seq}), now=NOW)
        old = exchange.read()
        with patch.object(exchange, 'publish') as publish, pytest.raises(BarrierError):
            archived.publish(exchange, value.model_copy(update={'sequence': 3}), now=NOW)
        publish.assert_not_called()
        assert exchange.read() == old and len(entries(root)) == 2
    assert not archived.close()['complete']


@pytest.mark.asyncio
@pytest.mark.parametrize('defect', ['owner', 'run', 'barrier', 'gap', 'preexisting', 'symlink', 'replaced_root'])
async def test_scope_sequence_and_filesystem_fences_before_live_write(setup, tmp_path, defect):
    controls, root = roots(tmp_path)
    archived = history(root, setup.scope)
    value = await receipt(setup)
    if defect in {'owner', 'run', 'barrier'}:
        key = {'owner': 'owner_id', 'run': 'run_id', 'barrier': 'barrier_id'}[defect]
        value = value.model_copy(update={'scope': value.scope.model_copy(update={key: 'other' if defect == 'owner' else uuid4()})})
    elif defect == 'gap':
        value = value.model_copy(update={'sequence': 2})
    elif defect == 'preexisting':
        (root / '00000000000000000001.json').write_bytes(b'do-not-overwrite')
    elif defect == 'symlink':
        (root / '00000000000000000001.json').symlink_to(controls / 'outside')
    else:
        root.rename(root.with_name('retained-history'))
        root.mkdir(mode=0o700)
    with AtomicReceiptFile(controls) as exchange, patch.object(exchange, 'publish') as publish:
        with pytest.raises(BarrierError):
            archived.publish(exchange, value, now=NOW)
        publish.assert_not_called()
    assert not archived.close()['complete']
    if defect == 'preexisting':
        assert (root / '00000000000000000001.json').read_bytes() == b'do-not-overwrite'


@pytest.mark.parametrize('bound', [0, 4097, True])
def test_explicit_capacity_bound_rejects_unbounded_history(setup, tmp_path, bound):
    _, root = roots(tmp_path)
    with pytest.raises(BarrierError, match='invalid_history_bound'):
        history(root, setup.scope, max_receipts=bound)
    assert not root.exists()


@pytest.mark.asyncio
async def test_host_archival_failure_stops_polling_and_close_reports_incomplete(setup, tmp_path):
    controls, root = roots(tmp_path)
    host = attach(setup)
    await host.configure(worker_readiness=readiness(setup), control_root=controls, history_root=root)
    try:
        with patch.object(host.history, '_commit', side_effect=OSError('disk')):
            await invoke(setup)
            await until(lambda: host.poll_task.done())
        assert host.reason == 'receipt_history_failed'
        assert host.sequence == 1 and decode_receipt(host.exchange.read()).sequence == 1
        count = len(setup.pool.queries)
        await asyncio.sleep(.03)
        assert len(setup.pool.queries) == count
    finally:
        closed = await host.close()
    assert closed['joined'] and closed['receipt_withdrawal'] == 'failed_existing_receipt_expires'
    assert closed['receipt_history']['error'] == 'history_archive_failed'
    assert not closed['receipt_history']['complete']


@pytest.mark.asyncio
async def test_frontend_fences_release_on_history_error_and_keeps_closed_failure(fixture):
    async with running(fixture) as (control, bridge, _):
        publish_configuration(fixture)
        await eventually(lambda: control.accepting_inputs)
        bridge.history = SimpleNamespace(error='history_archive_failed')
        with patch.object(control.release, 'step') as release:
            await eventually(lambda: control.task.done())
            closed = await control.close()
            release.assert_not_called()
        assert closed['status'] == 'closed' and closed['failure_code'] == 'receipt_history_failed'
        assert closed['observer_joined'] and closed['release_loop_joined']
