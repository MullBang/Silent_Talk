"""Reject scope drift and preserve paired denominators and baseline rules."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize


def selection():
    same, different = [f's{i}' for i in range(6)], [f'd{i}' for i in range(6)]
    groups = {}
    for name, ids in (('same', same), ('different', different)):
        groups.update({name+'6': ids, name+'_old4': ids[:4], name+'_new2': ids[4:]})
    return {'same_ids': same, 'different_ids': different, 'groups': groups,
            'records': [{'id': sid, 'speaker_id': f'p{i}', 'text': '나는 나' if kind == 'same' else '나는 거기 가'}
                        for kind, ids in (('same', same), ('different', different)) for i, sid in enumerate(ids)],
            'pairs': [{'speaker_id': f'p{i}', 'same_id': same[i], 'different_id': different[i]} for i in range(6)]}


def evaluation(seed=42, arm='A_ctc'):
    cohort = selection()
    metrics = summarize.metrics_module()
    by_id = {r['id']: r for r in cohort['records']}
    conditions = {}
    for condition in summarize.CONDITIONS:
        branches = {}
        for kind in ('same', 'different'):
            rows = []
            for i, sid in enumerate(cohort[kind+'_ids']):
                text = by_id[sid]['text']
                prediction = text if kind == 'same' else '나는 나'
                rows.append({'id': sid, 'speaker_id': f'p{i}', 'reference': text, 'prediction': prediction,
                             'condition': condition, 'source_split': 'train', 'status': 'ok',
                             'metrics': metrics.counts(text, prediction)})
            branches[kind] = {'records': rows, 'groups': {
                name: {'aggregate': metrics.aggregate([r for r in rows if r['id'] in ids]), 'diagnostics': {}}
                for name, ids in cohort['groups'].items() if name.startswith(kind)}}
        conditions[condition] = branches
    return {'schema': 'lipnet-step13-recording-probe-evaluation-v1', 'status': 'completed',
            'seed': seed, 'arm': arm, 'checkpoint': {'path': 'fake.pt', 'sha256': 'dummy'},
            'conditions': conditions, 'new_inference_rows': 18, 'reused_inference_rows': 18, 'record_count': 36,
            'optimization_steps': 0, 'audio_used': False, 'service_weights_updated': False}


def test_matched_cohorts_keep_separate_pooled_denominators():
    value = evaluation()
    result = summarize.model_summary(value, {}, selection())
    same = result['final_conditions']['normal']['same6']['aggregate']
    different = result['final_conditions']['normal']['different6']['aggregate']
    assert same['reference_characters'] == 24
    assert different['reference_characters'] == 42
    assert same['cer_percent'] == 0
    assert different['cer_percent'] > 0
    assert result['different_minus_same']['normal']['6']['cer_percentage_points'] == different['cer_percent']
    assert len(result['speaker_pairs']['normal']) == 6


@pytest.mark.parametrize('change', ('count', 'reuse_count', 'condition', 'row', 'duplicate', 'speaker', 'text',
                                    'group', 'denominator', 'train', 'audio', 'service', 'source', 'model'))
def test_scope_and_denominator_drift_rejected(change):
    value = evaluation()
    branch = value['conditions']['normal']['same']
    if change == 'count': value['record_count'] = 72
    elif change == 'reuse_count': value['reused_inference_rows'] = 0
    elif change == 'condition': value['conditions'].pop('static')
    elif change == 'row': branch['records'].pop()
    elif change == 'duplicate': branch['records'][1] = copy.deepcopy(branch['records'][0])
    elif change == 'speaker': branch['records'][0]['speaker_id'] = 'other'
    elif change == 'text': branch['records'][0]['reference'] = '바뀐 정답'
    elif change == 'group': branch['groups']['same8'] = branch['groups'].pop('same6')
    elif change == 'denominator': branch['groups']['same6']['aggregate']['clips'] = 8
    elif change == 'train': value['optimization_steps'] = 1
    elif change == 'audio': value['audio_used'] = True
    elif change == 'service': value['service_weights_updated'] = True
    elif change == 'source': branch['records'][0]['source_split'] = 'dev'
    elif change == 'model': value['arm'] = 'B_feature'
    with pytest.raises(ValueError):
        summarize.validate_evaluation(value, 42, 'A_ctc', selection())


def policy():
    anchors = {f'p{i}': {'id': f'a{i}', 'text': '나는 가'} for i in range(6)}
    return {'selection': selection(), 'speaker_anchor_baseline': {
        'choice_uses_evaluation_targets': False, 'baseline_rows': 12, 'neural_inference_rows': 0,
        'anchors': anchors, 'anchor_ids_by_speaker': {k: v['id'] for k, v in anchors.items()}}}


def test_fixed_speaker_baseline_is_separate_from_neural_inference():
    result = summarize.baseline(policy())
    assert result['neural_inference_rows'] == 0 and len(result['records']) == 12
    assert result['groups']['same6']['reference_characters'] == 24
    assert result['groups']['different6']['reference_characters'] == 42
    assert result['groups']['same6']['character_edits'] == 6
    assert {r['prediction'] for r in result['records']} == {'나는 가'}


def test_baseline_choice_does_not_use_evaluation_targets():
    value = policy()
    before = summarize.baseline(value)
    value['selection']['records'][0]['text'] = '새로운 평가 문장'
    after = summarize.baseline(value)
    assert [r['prediction'] for r in before['records']] == [r['prediction'] for r in after['records']]
    assert before['groups']['same6']['reference_characters'] != after['groups']['same6']['reference_characters']


def test_reusing_evaluation_target_as_baseline_anchor_is_rejected():
    value = policy()
    value['speaker_anchor_baseline']['anchors']['p0']['id'] = 's0'
    value['speaker_anchor_baseline']['anchor_ids_by_speaker']['p0'] = 's0'
    with pytest.raises(ValueError): summarize.baseline(value)


def test_seed_pairs_and_averages_preserve_conditions_and_cohorts():
    models = []
    for seed in summarize.SEEDS:
        for arm in summarize.ARMS:
            m = summarize.model_summary(evaluation(seed, arm), {}, selection())
            if arm == 'B_feature':
                for condition in summarize.CONDITIONS:
                    m['final_conditions'][condition]['same6']['aggregate']['cer_percent'] = 10.
            models.append(m)
    paired, means = summarize.comparisons(models)
    assert len(paired) == 2
    assert paired[0]['B_minus_A']['normal']['same6']['cer_percentage_points'] == 10.
    assert paired[0]['B_minus_A']['normal']['different6']['cer_percentage_points'] == 0.
    assert means['B_feature']['normal']['same6']['cer_percent'] == 10.
    with pytest.raises(ValueError): summarize.comparisons(models[:-1])


def test_failed_inferences_remain_in_the_fixed_cohort():
    value = evaluation()
    value['conditions']['normal']['same']['records'][0]['status'] = 'failed'
    result = summarize.model_summary(value, {}, selection())
    assert result['failed_new_inferences'] == 1
    assert result['new_inference_rows'] == 18 and result['reused_inference_rows'] == 18
