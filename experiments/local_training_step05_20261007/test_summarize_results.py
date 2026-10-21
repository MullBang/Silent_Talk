import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('step05_summary_test', Path(__file__).with_name('summarize_results.py'))
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)


def record(sid, reference, prediction, chars, words, n_chars=None, status='ok'):
    n_chars = chars if n_chars is None else n_chars
    return {'id': sid, 'reference': reference, 'prediction': prediction, 'status': status,
            'metrics': {'character_edits': chars, 'reference_characters': len(reference),
                        'word_edits': words, 'reference_words': len(reference.split()),
                        'exact': chars == 0, 'normalized_character_edits': n_chars,
                        'normalized_reference_characters': len(' '.join(reference.split())), 'normalized_exact': n_chars == 0}}


def test_pooled_counts_keep_failures_and_allow_over100_cer():
    rows = [record('a', '가', '가나다라', 3, 1), record('b', '나다', '', 2, 1, status='failed')]
    score = s.score_records(rows)
    assert score['cer_percent'] == pytest.approx(100 * 5 / 3)
    assert score['failed_inputs'] == 1 and score['clips'] == 2
    assert score['reference_characters'] == 3 and score['exact_sentences'] == 0


def test_primary_metric_preserves_internal_spaces():
    score = s.score_records([record('a', '가  나', '가 나', 1, 0, 0)])
    assert score['cer_percent'] == 25
    assert score['normalized_cer_percent'] == 0
    assert score['wer_percent'] == 0
    assert score['normalized_exact_sentences'] == 1 and score['exact_sentences'] == 0


def test_nfc_prediction_matches_composed_reference():
    row = record('a', '가', '\u1100\u1161', 0, 0)
    assert s.score_records([row])['exact_sentences'] == 1


@pytest.mark.parametrize('mutation', ['duplicate', 'metrics', 'failure_prediction'])
def test_invalid_evaluation_rows_rejected(mutation):
    rows = [record('a', '가', '가', 0, 0)]
    if mutation == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == 'metrics':
        rows[0]['metrics']['character_edits'] = 1
    else:
        rows[0]['status'] = 'failed'
    with pytest.raises(ValueError):
        s.score_records(rows)


def four_evaluations():
    result = {}
    for seed in s.SEEDS:
        for arm in s.ARMS:
            conditions = {}
            normal = {(42, 'A_ctc'): 90, (42, 'B_feature'): 80, (43, 'A_ctc'): 60, (43, 'B_feature'): 80}[(seed, arm)]
            for condition, extra in [('normal', 0), ('static', 20), ('shuffled', 5)]:
                conditions[condition] = {'aggregate': {'clips': 71, 'reference_characters': 100, 'reference_words': 20,
                    'normalized_reference_characters': 100, 'cer_percent': normal + extra,
                    'wer_percent': normal + extra + 10, 'normalized_cer_percent': normal + extra,
                    'exact_sentences': 1 if arm == 'A_ctc' else 3, 'failed_inputs': 0}}
            result[(seed, arm)] = {'conditions': conditions}
    return result


def test_pairing_uses_both_seeds_and_correct_difference_direction():
    result = s.paired_metrics(four_evaluations())
    first, second = result['per_seed']
    assert first['conditions']['normal']['B_minus_A_cer_percentage_points'] == -10
    assert second['conditions']['normal']['B_minus_A_cer_percentage_points'] == 20
    mean = result['arithmetic_mean_of_two_seeds']['normal']
    assert mean['A_ctc']['cer_percent'] == 75
    assert mean['B_feature']['cer_percent'] == 80
    assert mean['B_minus_A_cer_percentage_points'] == 5
    assert mean['B_minus_A_exact_sentences'] == 2
    assert result['mean_ablation_gaps']['B_feature']['static_minus_normal_cer_percentage_points'] == 20
    assert result['mean_ablation_gaps']['A_ctc']['shuffled_minus_normal_cer_percentage_points'] == 5


@pytest.mark.parametrize('mutation', ['missing_seed', 'denominator'])
def test_incomparable_pair_rejected(mutation):
    records = four_evaluations()
    if mutation == 'missing_seed':
        del records[(43, 'B_feature')]
    else:
        records[(43, 'B_feature')]['conditions']['normal']['aggregate']['reference_characters'] = 99
    with pytest.raises(ValueError):
        s.paired_metrics(records)


def fixture_history():
    plan = {'schedule': [['a', 'b'] for _ in range(3200)]}
    rows = []
    for index in range(3200):
        rows.append({'update': index + 1, 'ids': ['a', 'b'], 'ctc_loss': 2., 'feature_loss': 0., 'loss': 2.,
                     'gradient_norm_before_clipping': 2., 'gradient_norm_after_clipping': 1.,
                     'seconds_optimizer_update': .25, 'optimizer_updated': True, 'student_changed': True,
                     'feature_hook_count_before': 0, 'feature_hook_count_after': 0, 'lambda': 0.,
                     'adapter_changed': False, 'cuda_peak_allocated_bytes': 100, 'cuda_peak_reserved_bytes': 200})
    return rows, plan


def test_all_update_totals_and_timing_scope():
    rows, plan = fixture_history()
    result = s.validate_history(rows, plan, 'A_ctc')
    assert result['finite_updates'] == 3200
    assert result['clip_presentations'] == 6400
    assert result['optimizer_update_seconds'] == 800
    assert result['mean_optimizer_update_seconds'] == .25
    assert result['last100_mean_ctc_loss'] == 2.


@pytest.mark.parametrize('mutation', ['nan', 'missing_update', 'schedule', 'adapter', 'loss'])
def test_training_history_failure_rejected(mutation):
    rows, plan = fixture_history()
    if mutation == 'nan':
        rows[100]['ctc_loss'] = float('nan')
    elif mutation == 'missing_update':
        rows.pop()
    elif mutation == 'schedule':
        rows[100]['ids'] = ['b', 'a']
    elif mutation == 'adapter':
        rows[100]['adapter_changed'] = True
    else:
        rows[100]['loss'] = 3.
    with pytest.raises(ValueError):
        s.validate_history(rows, plan, 'A_ctc')
