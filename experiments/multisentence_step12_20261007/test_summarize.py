"""Test fixed-budget accounting and rejection of selected or incomplete results."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize


def evaluation():
    ids = [str(i) for i in range(32)]
    return {'schema': 'lipnet-step12-multisentence-evaluation-v1', 'status': 'completed',
            'seed': 42, 'arm': 'A_ctc', 'ids': ids, 'record_count': 224, 'diagnostic_groups': {'train24': ids[:24], 'hold8': ids[24:]},
            'optimization_steps': 0, 'dev_inputs_loaded': False, 'original_held27_inputs_loaded': False,
            'test_inputs_loaded': False, 'service_weights_updated': False,
            'milestones': [{'completed_updates': update, 'conditions': {
                c: {'records': [{'id': sid, 'source_split': 'train', 'split': 'train' if int(sid) < 24 else 'internal_holdout', 'diagnostic_role': 'train' if int(sid) < 24 else 'internal_holdout'} for sid in ids],
                    'groups': {g: {} for g in summarize.GROUPS}}
                for c in (('normal', 'static', 'shuffled') if update == 3600 else ('normal',))}}
                for update in summarize.MILESTONES]}


def run():
    return {'status': 'completed', 'start_update': 0, 'completed_updates': 3600, 'restored_from': None,
            'elapsed_seconds': 123., 'initial_state_hashes': {}, 'final_state_hashes': {},
            'updates': [{'update': i + 1, 'ids': [str(2 * (i % 12)), str(2 * (i % 12) + 1)],
                         'frames': [100, 120], 'optimizer_updated': True, 'student_changed': True,
                         'loss': 4. if i < 100 else 1., 'ctc_loss': 3. if i < 100 else .8,
                         'feature_loss': 1. if i < 100 else .2, 'gradient_norm_before_clipping': 2.,
                         'gradient_norm_after_clipping': 1., 'cuda_peak_allocated_bytes': 100,
                         'cuda_peak_reserved_bytes': 200} for i in range(3600)]}


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
        value['milestones'][-1]['conditions']['normal']['groups'].pop('hold_new4')
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
    assert value['processed_frames'] == 792000
    assert value['presentations'] == 7200
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


def baseline_policy():
    speakers = [f's{i}' for i in range(8)]
    train = [f'a{i}' for i in range(8)]
    hold = [f'h{i}' for i in range(8)]
    return {'speaker_anchor_baseline': {'kind': 'known_speaker_fixed_training_sentence',
            'anchor_ids_by_speaker': dict(zip(speakers, train)), 'evaluation_ids': hold,
            'choice_uses_holdout_targets': False},
            'selection': {'train_ids': train, 'holdout_ids': hold,
                'records': [{'id': sid, 'speaker_id': speaker, 'text': '나는 가'} for sid, speaker in zip(train, speakers)] +
                           [{'id': sid, 'speaker_id': speaker, 'text': '나는 나'} for sid, speaker in zip(hold, speakers)],
                'evaluation_groups': {'hold8': hold, 'hold_old4': hold[:4], 'hold_new4': hold[4:]}}}


def test_baseline_uses_train_anchor_and_fixed_denominator():
    result = summarize.speaker_baseline(baseline_policy())
    assert {r['prediction'] for r in result['records']} == {'나는 가'}
    assert result['groups']['hold8']['character_edits'] == 8
    assert result['groups']['hold8']['reference_characters'] == 32
    assert result['groups']['hold8']['cer_percent'] == 25
    assert result['neural_inference_rows'] == 0


def test_changing_holdout_truth_does_not_select_different_baseline_prediction():
    policy = baseline_policy()
    before = summarize.speaker_baseline(policy)
    policy['selection']['records'][-1]['text'] = '다른 정답'
    after = summarize.speaker_baseline(policy)
    assert [r['prediction'] for r in before['records']] == [r['prediction'] for r in after['records']]
    assert before['groups']['hold8'] != after['groups']['hold8']


@pytest.mark.parametrize('changed', ('anchor_role', 'speaker', 'cohort'))
def test_invalid_baseline_binding_rejected(changed):
    policy = baseline_policy()
    if changed == 'anchor_role':
        policy['speaker_anchor_baseline']['anchor_ids_by_speaker']['s0'] = 'h0'
    elif changed == 'speaker':
        policy['selection']['records'][0]['speaker_id'] = 's1'
    else:
        policy['speaker_anchor_baseline']['evaluation_ids'] = policy['selection']['holdout_ids'][1:]
    with pytest.raises(ValueError):
        summarize.speaker_baseline(policy)


def test_paired_differences_and_seed_means():
    models = []
    for seed in summarize.SEEDS:
        for arm in summarize.ARMS:
            models.append({'seed': seed, 'arm': arm, 'final_conditions': {
                condition: {group: {'aggregate': {
                    'cer_percent': (10 if arm == 'A_ctc' else 5) + seed - 42,
                    'wer_percent': 20 if arm == 'A_ctc' else 15,
                    'exact_sentences': 1 if arm == 'A_ctc' else 2}}
                    for group in summarize.GROUPS} for condition in ('normal', 'static', 'shuffled')}})
    paired, means = summarize.comparisons(models)
    assert paired[0]['B_minus_A']['normal']['hold8'] == {'cer_percentage_points': -5, 'exact_sentences': 1}
    assert means['A_ctc']['normal']['hold8']['cer_percent'] == 10.5
    assert means['B_feature']['normal']['train24']['cer_percent'] == 5.5
    with pytest.raises(ValueError):
        summarize.comparisons(models[:-1])
