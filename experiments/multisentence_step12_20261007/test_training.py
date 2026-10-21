"""Reject accidentally changed data before the expensive optimizer path."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training


def fixture_data():
    clips = {sid: {'id': sid, 'split': 'train',
                   'pixels': np.zeros((3, 96, 96, 3), dtype=np.uint8),
                   'tokens': [1, 2], 'teacher_features': np.ones((3, 512), dtype=np.float32)}
             for sid in ('a', 'b')}
    plan = {'seed': 42, 'updates': 1, 'batch_size': 2, 'checkpoint_interval': 1,
            'schedule': [['a', 'b']]}
    actual = training.runner.validate_plan(plan, clips, 'B_feature')
    plan['training_data'] = {k: actual[k] for k in ('data_sha256', 'clips')}
    return plan, {'train_clips': clips, 'holdout_ids': ['hold']}


@pytest.mark.parametrize('arm', training.ARMS)
def test_exact_tensor_binding_both_arms(arm):
    plan, loaded = fixture_data()
    assert training.validate_loaded(plan, loaded, arm)['data_sha256'] == plan['training_data']['data_sha256']


@pytest.mark.parametrize('part', ('pixels', 'tokens', 'teacher_features'))
def test_changed_training_input_fails(part):
    plan, loaded = fixture_data()
    if part == 'tokens':
        loaded['train_clips']['a'][part][0] = 3
    else:
        loaded['train_clips']['a'][part].flat[0] = 2
    with pytest.raises(ValueError, match='bytes changed'):
        training.validate_loaded(plan, loaded, 'B_feature')


def test_development_record_rejected_before_training():
    plan, loaded = fixture_data()
    loaded['train_clips']['a']['split'] = 'dev'
    with pytest.raises(ValueError, match='Non-training'):
        training.validate_loaded(plan, loaded, 'A_ctc')


def test_extra_unscheduled_record_rejected():
    plan, loaded = fixture_data()
    loaded['train_clips']['c'] = copy.deepcopy(loaded['train_clips']['a'])
    loaded['train_clips']['c']['id'] = 'c'
    with pytest.raises(ValueError, match='scheduled training scope'):
        training.validate_loaded(plan, loaded, 'A_ctc')


@pytest.mark.parametrize('seed,arm', ((41, 'A_ctc'), (42, 'other')))
def test_unknown_run_cannot_write_output(seed, arm):
    with pytest.raises(ValueError, match='Unknown run'):
        training.run_path(seed, arm)


def test_internal_holdout_cannot_enter_optimizer_scope():
    plan, loaded = fixture_data()
    loaded['holdout_ids'] = ['a']
    with pytest.raises(ValueError, match='Internal holdout entered'):
        training.validate_loaded(plan, loaded, 'B_feature')
