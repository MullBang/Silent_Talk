"""Guard fixed denominators and non-accuracy interpretation of candidate scores."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('step14_summary_test_subject', Path(__file__).with_name('summarize.py'))
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def score_row(key='a', rank=1, tied=False, failed=False, infeasible=0, margin=2.0):
    ranking = {'reference_rank': rank, 'top_candidate_ids': ['a', 'b'] if tied else ['a'],
               'reference_minus_best_training': margin, 'same_minus_other_training_score': None,
               'best_training_relation': None, 'feasible_candidates': 25-infeasible}
    return {'source_key': key, 'status': 'failed' if failed else 'ok',
            'candidates': [{'status': 'source_failed' if failed else ('infeasible' if i < infeasible else 'ok')}
                           for i in range(25)],
            'rankings': None if failed else {name: deepcopy(ranking) for name in subject.SCORE_KEYS},
            'reference_minus_greedy_total_log_probability': margin,
            'greedy_score': {'target_tokens': 2}, 'candidate_pairs': 25, 'auxiliary_greedy_pairs': 1}


def fixture():
    groups = {}
    for cohort in ('same', 'different'):
        ids = [f'{cohort}{i}' for i in range(6)]
        groups.update({cohort+'6': ids, cohort+'_old4': ids[:4], cohort+'_new2': ids[4:]})
    sources, rows = [], []
    for seed in subject.SEEDS:
        for arm in subject.ARMS:
            for condition in subject.CONDITIONS:
                for cohort in ('same', 'different'):
                    for sid in groups[cohort+'6']:
                        key = f'{seed}/{arm}/{condition}/{sid}'
                        source = {'key': key, 'seed': seed, 'arm': arm, 'condition': condition,
                                  'cohort': cohort, 'id': sid, 'speaker_id': sid[-1], 'frames': 100,
                                  'source_record_sha256': 'hash', 'source_evaluation': {'sha256': 'a'},
                                  'posterior_archive': {'sha256': 'b'}, 'posterior_key': key,
                                  'source_record': {'metrics': {'reference_characters': 10, 'reference_words': 3,
                                                               'character_edits': 12, 'word_edits': 4, 'exact': False}}}
                        row = score_row(key)
                        row.update({k: source[k] for k in ('seed', 'arm', 'condition', 'cohort', 'id', 'speaker_id',
                                    'frames', 'source_record_sha256', 'source_evaluation', 'posterior_archive', 'posterior_key')})
                        sources.append(source)
                        rows.append(row)
    value = {'schema': 'lipnet-step14-posterior-candidate-scores-v1', 'status': 'completed', 'rows': rows,
             'source_rows': 144, 'candidate_pairs': 3600, 'auxiliary_greedy_pairs': 144, 'total_ctc_pairs': 3744,
             'new_model_inferences': 0, 'new_optimizer_updates': 0, 'checkpoint_payloads_loaded': 0,
             'visual_input_caches_loaded': 0, 'audio_used': False, 'teacher_used': False, 'service_weights_updated': False,
             'reference_inclusive_candidate_ranks_are_recognition_accuracy': False,
             'token_length_normalized_scores_are_actual_log_probabilities': False,
             'failed_source_rows': 0, 'infeasible_candidate_pairs': 0, 'source_failed_candidate_pairs': 0}
    return value, {'sources': sources, 'groups': groups}


def test_failures_remain_in_denominator_and_do_not_become_zero_scores():
    result = subject.aggregate_rows([score_row(), score_row('b', failed=True), score_row('c', rank=3, infeasible=4)])
    assert (result['rows_total'], result['rows_scored'], result['rows_failed']) == (3, 2, 1)
    assert (result['candidate_pairs'], result['feasible_candidate_pairs'], result['infeasible_candidate_pairs'],
            result['source_failed_candidate_pairs']) == (75, 46, 4, 25)
    ranking = result['rankings']['total_log_probability']
    assert ranking['reference_rank_over_scored_rows']['mean'] == 2
    assert ranking['reference_rank_histogram'] == {'1': 1, '3': 1}
    assert ranking['same_minus_other_training_over_scored_rows']['mean'] is None
    assert ranking['best_training_relation_counts']['unavailable'] == 2


def test_ties_count_as_top_but_not_unique_top():
    result = subject.aggregate_rows([score_row(tied=True, margin=0), score_row('b', margin=1e-9)])
    for rank in result['rankings'].values():
        assert rank['reference_top_including_ties_count'] == 2
        assert rank['reference_unique_top_count'] == 1
        assert rank['reference_minus_best_training_over_scored_rows']['tied_count'] == 2


def test_margin_epsilon_and_unavailable_are_explicit():
    result = subject.margin_summary([None, -2, -1e-8, 0, 1e-8, 2])
    assert (result['count'], result['unavailable']) == (5, 1)
    assert (result['positive_count'], result['tied_count'], result['negative_count']) == (1, 3, 1)
    assert result['mean'] == 0


def test_all_failed_rows_have_no_fabricated_rank_or_margin():
    result = subject.aggregate_rows([score_row(failed=True)])
    assert result['rows_total'] == result['rows_failed'] == 1
    assert result['rankings']['total_log_probability']['reference_rank_over_scored_rows']['mean'] is None
    assert result['reference_minus_greedy_total_log_probability_over_scored_rows']['count'] == 0
    assert result['ranks_are_recognition_accuracy'] is False


@pytest.mark.parametrize('bad', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_not_silently_averaged(bad):
    with pytest.raises(ValueError, match='Nonfinite'):
        subject.numeric_summary([bad])


@pytest.mark.parametrize('rows', [[], [score_row(), score_row()]])
def test_missing_or_duplicate_rows_rejected(rows):
    with pytest.raises(ValueError):
        subject.aggregate_rows(rows)


def test_empty_greedy_sequence_is_reported():
    row = score_row()
    row['greedy_score']['target_tokens'] = 0
    assert subject.aggregate_rows([row])['empty_greedy_sequences'] == 1


def test_full_comparison_keeps_groups_and_original_error_rates_over_100():
    value, policy = fixture()
    result = subject.make_comparison(value, policy)
    assert result['source_rows'] == 144 and result['total_ctc_pairs'] == 3744
    assert len(result['models']) == 4
    for model in result['models']:
        for condition in subject.CONDITIONS:
            groups = model['conditions'][condition]
            for cohort in ('same', 'different'):
                assert groups[cohort+'6']['rows_total'] == 6
                assert groups[cohort+'_old4']['rows_total'] == 4
                assert groups[cohort+'_new2']['rows_total'] == 2
                assert groups[cohort+'6']['original_greedy_metrics']['cer_percent'] == 120
    assert result['across_models_condition_cohort_counts']['normal']['same']['rows_total'] == 24
    assert result['ranking_is_accuracy'] is False and result['reference_candidate_is_oracle'] is True
    assert result['source_predictions_changed'] is False


def test_failed_scoring_does_not_drop_original_greedy_metrics():
    value, policy = fixture()
    old = value['rows'][0]
    old.update(score_row(old['source_key'], failed=True))
    value.update(failed_source_rows=1, source_failed_candidate_pairs=25)
    result = subject.make_comparison(value, policy)['models'][0]['conditions']['normal']['same6']
    assert result['rows_scored'] == 5 and result['rows_total'] == 6
    assert result['original_greedy_metrics']['rows'] == 6
    assert result['original_greedy_metrics']['reference_characters'] == 60


@pytest.mark.parametrize('field,new_value', [('candidate_pairs', 3599), ('auxiliary_greedy_pairs', 0),
    ('new_model_inferences', 1), ('new_optimizer_updates', 1), ('audio_used', True),
    ('service_weights_updated', True), ('failed_source_rows', 1),
    ('reference_inclusive_candidate_ranks_are_recognition_accuracy', True),
    ('token_length_normalized_scores_are_actual_log_probabilities', True)])
def test_changed_budget_scope_or_semantics_rejected(field, new_value):
    value, policy = fixture()
    value[field] = new_value
    with pytest.raises(ValueError):
        subject.validate_scores(value, policy)


def test_selected_or_reordered_sources_rejected():
    value, policy = fixture()
    value['rows'][0], value['rows'][1] = value['rows'][1], value['rows'][0]
    with pytest.raises(ValueError, match='reordered'):
        subject.validate_scores(value, policy)


def test_changed_provenance_rejected():
    value, policy = fixture()
    value['rows'][0]['source_record_sha256'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        subject.validate_scores(value, policy)


def test_incomplete_candidate_set_rejected():
    value, policy = fixture()
    value['rows'][0]['candidates'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        subject.validate_scores(value, policy)


def test_missing_group_member_rejected():
    value, policy = fixture()
    policy['groups']['same6'].append('nonexistent')
    with pytest.raises(ValueError, match='denominator'):
        subject.make_comparison(value, policy)
