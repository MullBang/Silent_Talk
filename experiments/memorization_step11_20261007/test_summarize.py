"""Test fixed-budget accounting and rejection of selected or incomplete results."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize


def evaluation():
    ids = [str(i) for i in range(8)]
    return {'schema': 'lipnet-step11-train-memorization-evaluation-v1', 'status': 'completed',
            'seed': 42, 'arm': 'A_ctc', 'ids': ids, 'record_count': 56,
            'optimization_steps': 0, 'dev_inputs_loaded': False, 'held_inputs_loaded': False,
            'test_inputs_loaded': False, 'service_weights_updated': False,
            'milestones': [{'completed_updates': update, 'conditions': {
                c: {'records': [{'id': sid, 'split': 'train'} for sid in ids],
                    'groups': {g: {} for g in summarize.GROUPS}}
                for c in (('normal', 'static', 'shuffled') if update == 1200 else ('normal',))}}
                for update in summarize.MILESTONES]}


def run():
    return {'status': 'completed', 'start_update': 0, 'completed_updates': 1200, 'restored_from': None,
            'elapsed_seconds': 123., 'initial_state_hashes': {}, 'final_state_hashes': {},
            'updates': [{'update': i + 1, 'ids': [str(2 * (i % 4)), str(2 * (i % 4) + 1)],
                         'frames': [100, 120], 'optimizer_updated': True, 'student_changed': True,
                         'loss': 4. if i < 100 else 1., 'ctc_loss': 3. if i < 100 else .8,
                         'feature_loss': 1. if i < 100 else .2, 'gradient_norm_before_clipping': 2.,
                         'gradient_norm_after_clipping': 1., 'cuda_peak_allocated_bytes': 100,
                         'cuda_peak_reserved_bytes': 200} for i in range(1200)]}


def test_complete_trajectory():
    summarize.validate_evaluation(evaluation(), 42, 'A_ctc')


@pytest.mark.parametrize('change', ('milestone', 'row', 'condition', 'group', 'dev', 'seed', 'scope'))
def test_incomplete_or_wrong_trajectory_rejected(change):
    value = evaluation()
    if change == 'milestone':
        value['milestones'].pop(1)
    elif change == 'row':
        value['milestones'][0]['conditions']['normal']['records'].pop()
    elif change == 'condition':
        value['milestones'][-1]['conditions'].pop('static')
    elif change == 'group':
        value['milestones'][-1]['conditions']['normal']['groups'].pop('new4')
    elif change == 'dev':
        value['milestones'][0]['conditions']['normal']['records'][0]['split'] = 'dev'
    elif change == 'seed':
        value['seed'] = 43
    else:
        value['test_inputs_loaded'] = True
    with pytest.raises(ValueError):
        summarize.validate_evaluation(value, 42, 'A_ctc')


def test_exposure_losses_and_actual_resources():
    value = summarize.summarize_training(run())
    assert value['processed_frames'] == 264000
    assert value['presentations'] == 2400
    assert set(value['exposure_per_clip'].values()) == {300}
    assert value['loss']['first100']['ctc_loss'] == 3.
    assert value['loss']['last100']['ctc_loss'] == .8
    assert value['elapsed_seconds'] == 123.


@pytest.mark.parametrize('change', ('skipped', 'exposure', 'budget', 'nonfinite'))
def test_invalid_training_accounting_rejected(change):
    value = run()
    if change == 'skipped':
        value['updates'][100]['optimizer_updated'] = False
    elif change == 'exposure':
        value['updates'][0]['ids'] = ['2', '3']
    elif change == 'budget':
        value['completed_updates'] = 1100
    else:
        value['updates'][-1]['loss'] = float('nan')
    with pytest.raises(ValueError):
        summarize.summarize_training(value)
