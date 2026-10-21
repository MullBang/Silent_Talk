"""Meaningful safeguards for the bounded paired data-diversity experiment."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('step15_design_under_test', Path(__file__).with_name('experiment_plan.py'))
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def fixture():
    rows, small, large, dev = [], [], [], []
    for speaker in range(8):
        for i in range(9):
            sid = f's{speaker}_train{i}'
            rows.append({'id': sid, 'speaker_id': f's{speaker}', 'frames': 50+speaker+i,
                         'text': '가 나' if i < 3 else '다 라', 'tokens': [1, 2] if i < 3 else [2, 3]})
            large.append(sid)
            if i < 3:
                small.append(sid)
        for i in range(2):
            sid = f's{speaker}_dev{i}'
            rows.append({'id': sid, 'speaker_id': f's{speaker}', 'frames': 60, 'text': '가 다 마', 'tokens': [1, 3, 4]})
            dev.append(sid)
    data = {'selection': {'records': rows, 'train24_ids': small, 'train72_ids': large, 'dev16_ids': dev},
            'pool_profile': {'prior_probe_ids': [f'probe{i}' for i in range(14)],
                             'records': [{'id': f'probe{i}', 'speaker_id': f's{i}'} for i in range(14)]}}
    prior = {'initial_state_hashes_expected': {str(seed): {'B_feature': {'model': f'initial{seed}'}} for seed in (42, 43)},
             'runtime': {'device': 'reference only'}}
    return data, prior


@pytest.mark.parametrize('update,expected', [(1,.001), (2400,.001), (2401,.0003), (3000,.0003), (3001,.0001), (3600,.0001)])
def test_learning_rate_global_update_boundaries(update, expected):
    assert subject.learning_rate(update) == expected


@pytest.mark.parametrize('update', [0, -1, 3601, 1.5, True])
def test_out_of_budget_learning_rate_rejected(update):
    with pytest.raises(ValueError):
        subject.learning_rate(update)


def test_paired_schedule_exact_exposures_speaker_order_and_frame_budgets():
    data, prior = fixture()
    plan = subject.make_plan(data, prior)
    rows = {row['id']: row for row in data['selection']['records']}
    for seed in (42, 43):
        small, large = [run for run in plan['runs'] if run['seed'] == seed]
        for run, count in ((small, 300), (large, 100)):
            assert len(run['schedule']) == 3600
            occurrences = Counter(sid for batch in run['schedule'] for sid in batch)
            assert occurrences == Counter({sid: count for sid in run['train_ids']})
            assert all(len({rows[sid]['speaker_id'] for sid in batch}) == 2 for batch in run['schedule'])
            assert run['student_frames_presented'] == sum(rows[sid]['frames']*count for sid in run['train_ids'])
            assert not set(occurrences) & set(data['selection']['dev16_ids'])
        assert [[rows[sid]['speaker_id'] for sid in batch] for batch in small['schedule']] == [
            [rows[sid]['speaker_id'] for sid in batch] for batch in large['schedule']]
        assert small['initial_state_hashes_expected'] == large['initial_state_hashes_expected']
    assert sum(plan['budget']['evaluation_rows_breakdown'].values()) == plan['budget']['neural_evaluation_rows_planned'] == 824
    assert plan['budget']['total_updates'] == 14400
    assert plan['budget']['same_updates_not_same_frames_flops_or_wall_time'] is True
    assert plan['optimizer_updates_executed'] == plan['new_model_inferences'] == 0


def test_schedule_is_order_independent_reproducible_and_seed_specific():
    data, _ = fixture()
    records = [row for row in data['selection']['records'] if row['id'] in data['selection']['train24_ids']]
    schedule = subject.make_schedule(records, 42)
    assert subject.make_schedule(list(reversed(records)), 42) == schedule
    assert subject.make_schedule(records, 43) != schedule


def test_unbalanced_speakers_rejected():
    data, _ = fixture()
    records = [deepcopy(row) for row in data['selection']['records'] if row['id'] in data['selection']['train24_ids']]
    records[0]['speaker_id'] = 's1'
    with pytest.raises(ValueError, match='Unbalanced'):
        subject.make_schedule(records, 42)


@pytest.mark.parametrize('change', ['train_dev_overlap', 'missing_large', 'duplicate_record', 'bad_frame', 'wrong_dev_speaker'])
def test_invalid_group_plan_rejected(change):
    data, prior = fixture()
    selection = data['selection']
    if change == 'train_dev_overlap':
        selection['dev16_ids'][0] = selection['train24_ids'][0]
    elif change == 'missing_large':
        selection['train72_ids'].pop()
    elif change == 'duplicate_record':
        # Remove a selected record's key while preserving the list size.
        selection['records'][1]['id'] = selection['records'][0]['id']
    elif change == 'bad_frame':
        selection['records'][0]['frames'] = -10
    else:
        next(row for row in selection['records'] if row['id'] == selection['dev16_ids'][0])['speaker_id'] = 's9'
    with pytest.raises(ValueError):
        subject.make_plan(data, prior)


@pytest.mark.parametrize('field', ['schedule', 'learning_rate_segments', 'final_comparison_update'])
def test_mutated_training_design_rejected(field):
    data, prior = fixture()
    plan = subject.make_plan(data, prior)
    if field == 'schedule':
        plan['runs'][0][field][0][0] = data['selection']['dev16_ids'][0]
    elif field == 'learning_rate_segments':
        plan['runs'][0][field][-1]['lr'] = .001
    else:
        plan['runs'][0][field] = 2400
    with pytest.raises(ValueError, match='differs'):
        subject.validate_plan(plan, data, prior)


def test_final_selection_and_limits_do_not_promise_improvement():
    data, prior = fixture()
    plan = subject.make_plan(data, prior)
    gate = plan['evaluation']['followup_expansion_gate']
    assert gate['both_seed_deltas_strictly_below_zero'] is True
    assert gate['mean_delta_at_most_percentage_points'] == -5
    assert gate['statistical_significance_claim'] is False
    assert plan['evaluation']['dev_used_for_training_early_stopping_or_model_choice'] is False
    assert all(run['final_comparison_update'] == 3600 for run in plan['runs'])
    assert plan['budget']['automatic_extension'] is False


def test_post_selection_coverage_does_not_claim_word_recognition():
    data, prior = fixture()
    profile = subject.make_plan(data, prior)['post_selection_corpus_profile']
    assert profile['groups']['train24']['unique_token_classes'] == 2
    assert profile['groups']['train72']['unique_token_classes'] == 3
    assert profile['dev16_values_absent_from_training']['train24']['tokens'] == [3, 4]
    assert profile['dev16_values_absent_from_training']['train72']['tokens'] == [4]
    assert profile['dev16_values_absent_from_training']['train72']['words'] == ['마']
    assert profile['used_to_select_or_tune'] is False


@pytest.mark.parametrize('change', ['duplicate_record', 'missing_probe', 'probe_overlap'])
def test_duplicate_or_unbound_historical_probe_rejected(change):
    data, prior = fixture()
    if change == 'duplicate_record':
        data['selection']['records'].append(deepcopy(data['selection']['records'][0]))
    elif change == 'missing_probe':
        data['pool_profile']['prior_probe_ids'].pop()
    else:
        sid = data['selection']['train72_ids'][0]
        data['pool_profile']['records'][0]['id'] = sid
        data['pool_profile']['prior_probe_ids'][0] = sid
    with pytest.raises(ValueError):
        subject.make_plan(data, prior)
