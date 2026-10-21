"""Adversarial independent selection and leakage checks for Step12."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('step12_audit_under_test', Path(__file__).with_name('independent_audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture():
    anchors, eligible, rows, entries, provenance = [], [], {}, {}, {}
    for generation, prefix in [('old', 'O'), ('new', 'N')]:
        for number in range(4):
            speaker = prefix + str(number)
            for position in range(6):
                sid = f'{speaker}_{position}'
                stem = f'{speaker}_train' if position < 4 else f'{speaker}_hold'
                if position == 0:
                    anchors.append(sid)
                eligible.append(sid)
                rows[sid] = {'id': sid, 'speaker_id': speaker, 'source_stem': stem,
                             'text': sid, 'split': 'train', 'frames': 20, 'token_ids': [1, 20]}
                entries[sid] = {'role': 'old_train' if generation == 'old' else 'new_provisional_train',
                                'generation': generation}
                provenance[sid] = {key: hashlib.sha256((stem + key).encode()).hexdigest()
                                   for key in ('video_sha256', 'label_sha256')}
    return anchors, eligible, rows, entries, provenance


def selected(data):
    return audit.independent_selection(*data)


def test_retains_anchors_and_three_train_one_hold_per_speaker():
    data = fixture()
    train, hold, groups = selected(data)
    anchors, _, rows, _, _ = data
    assert train[::3] == anchors
    assert len(train) == len(set(train)) == 24
    assert len(hold) == len(set(hold)) == 8
    assert not set(train) & set(hold)
    for index, anchor in enumerate(anchors):
        assert all(rows[sid]['speaker_id'] == rows[anchor]['speaker_id'] for sid in train[3*index:3*index+3])
        assert rows[hold[index]]['source_stem'] != rows[anchor]['source_stem']
    assert [len(groups[key]) for key in audit.GROUPS] == [24, 8, 12, 12, 4, 4]


def test_input_metadata_order_cannot_change_selection():
    data = fixture()
    changed = list(data)
    changed[1] = data[1][::-1]
    assert selected(data) == selected(changed)


def test_hash_salts_are_exact_and_purpose_specific():
    for purpose in ('train', 'holdout'):
        assert audit.digest_rank('abc', purpose) == (
            hashlib.sha256(f'step12-multisentence-v1:{purpose}:abc'.encode()).hexdigest(), 'abc')
    assert audit.digest_rank('abc', 'train') != audit.digest_rank('abc', 'holdout')


def test_unknown_rank_purpose_rejected():
    with pytest.raises(ValueError):
        audit.digest_rank('abc', 'development')


@pytest.mark.parametrize(('source', 'expected'), [(' 가\t나\n', '가나'), ('\u1100\u1161 나', '가나'), ('가\u3000나', '가나')])
def test_duplicate_key_normalizes_jamos_and_all_whitespace(source, expected):
    assert audit.normalized_text(source) == expected


def test_anchor_duplicate_rejected_without_replacement():
    data = fixture()
    data[2][data[0][1]]['text'] = data[2][data[0][0]]['text'] + ' '
    with pytest.raises(ValueError):
        selected(data)


def test_all_anchors_reserved_before_extra_selection():
    data = fixture()
    candidate = sorted(['O0_1', 'O0_2', 'O0_3'], key=lambda sid: audit.digest_rank(sid, 'train'))[0]
    data[2][candidate]['text'] = data[2]['N3_0']['text']
    train, _, _ = selected(data)
    assert candidate not in train and 'N3_0' in train


def test_infeasible_extra_skipped_deterministically():
    data = fixture()
    candidate = sorted(['O0_1', 'O0_2', 'O0_3'], key=lambda sid: audit.digest_rank(sid, 'train'))[0]
    data[2][candidate]['frames'] = 1
    train, _, _ = selected(data)
    assert candidate not in train


def test_infeasible_anchor_rejected_not_replaced():
    data = fixture()
    data[2]['O0_0']['frames'] = 1
    with pytest.raises(ValueError):
        selected(data)


def test_too_few_additions_rejected_not_different_speaker():
    data = fixture()
    data[1].remove('O0_1')
    data[1].remove('O0_2')
    with pytest.raises(ValueError):
        selected(data)


@pytest.mark.parametrize('key', ['video_sha256', 'label_sha256'])
def test_alias_recording_holdout_rejected_by_content_hash(key):
    data = fixture()
    for sid in ['O0_4', 'O0_5']:
        data[4][sid][key] = data[4]['O0_0'][key]
    with pytest.raises(ValueError):
        selected(data)


def test_holdout_target_duplicate_skipped():
    data = fixture()
    first = sorted(['O0_4', 'O0_5'], key=lambda sid: audit.digest_rank(sid, 'holdout'))[0]
    data[2][first]['text'] = data[2]['N3_0']['text']
    _, hold, _ = selected(data)
    assert first not in hold


def test_duplicate_holdout_targets_cannot_both_enter():
    data = fixture()
    first_o0 = sorted(['O0_4', 'O0_5'], key=lambda sid: audit.digest_rank(sid, 'holdout'))[0]
    first_o1 = sorted(['O1_4', 'O1_5'], key=lambda sid: audit.digest_rank(sid, 'holdout'))[0]
    data[2][first_o1]['text'] = data[2][first_o0]['text']
    _, hold, _ = selected(data)
    assert first_o0 in hold and first_o1 not in hold


def test_same_stem_but_changed_bytes_not_selected():
    data = fixture()
    first = sorted(['O0_1', 'O0_2', 'O0_3'], key=lambda sid: audit.digest_rank(sid, 'train'))[0]
    data[4][first]['video_sha256'] = '1' * 64
    train, _, _ = selected(data)
    assert first not in train


@pytest.mark.parametrize('tokens', [[0], [69], [True], [], [1, 1]])
def test_ctc_feasibility_rejects_invalid_or_short(tokens):
    assert not audit.ctc_feasible({'token_ids': tokens, 'frames': 1})


def test_normal_ctc_feasibility():
    assert audit.ctc_feasible({'token_ids': [1, 1], 'frames': 3})


def baseline_policy():
    data = fixture()
    train, hold, groups = selected(data)
    anchors, _, rows, _, _ = data
    return {'selection': {'train_ids': train, 'holdout_ids': hold, 'retained_step11_ids': anchors,
                          'records': [rows[sid] for sid in train + hold], 'evaluation_groups': groups},
            'speaker_anchor_baseline': {'kind': 'known_speaker_fixed_training_sentence',
                                       'evaluation_ids': hold, 'choice_uses_holdout_targets': False,
                                       'anchor_ids_by_speaker': {rows[sid]['speaker_id']: sid for sid in anchors}}}


def test_speaker_baseline_is_eight_fixed_text_outputs_not_neural_inference():
    result = audit.independent_baseline(baseline_policy())
    assert len(result['records']) == 8
    assert result['groups']['hold8']['clips'] == 8
    assert result['groups']['hold_old4']['clips'] == 4
    assert result['groups']['hold_new4']['clips'] == 4
    assert result['groups']['hold8']['exact_sentences'] == 0
    assert result['neural_inference_rows'] == 0
    assert result['requires_known_metadata_speaker_id'] is True
    assert result['deployable_unseen_speaker_recognizer'] is False


def test_speaker_baseline_cannot_choose_holdout_transcript():
    policy = baseline_policy()
    policy['speaker_anchor_baseline']['anchor_ids_by_speaker']['O0'] = policy['selection']['holdout_ids'][0]
    with pytest.raises(ValueError):
        audit.independent_baseline(policy)


def test_speaker_baseline_cannot_switch_to_an_extra_training_sentence():
    policy = baseline_policy()
    policy['speaker_anchor_baseline']['anchor_ids_by_speaker']['O0'] = policy['selection']['train_ids'][1]
    with pytest.raises(ValueError):
        audit.independent_baseline(policy)


def test_speaker_baseline_cannot_use_another_speaker_anchor():
    policy = baseline_policy()
    policy['speaker_anchor_baseline']['anchor_ids_by_speaker']['O0'] = 'O1_0'
    with pytest.raises(ValueError):
        audit.independent_baseline(policy)


def test_speaker_baseline_cannot_filter_holdout_by_target():
    policy = baseline_policy()
    policy['speaker_anchor_baseline']['choice_uses_holdout_targets'] = True
    with pytest.raises(ValueError):
        audit.independent_baseline(policy)


def test_speaker_baseline_cannot_drop_a_holdout():
    policy = baseline_policy()
    policy['speaker_anchor_baseline']['evaluation_ids'] = policy['selection']['holdout_ids'][:-1]
    with pytest.raises(ValueError):
        audit.independent_baseline(policy)


def test_independent_pairing_and_means_keep_both_seeds_all_groups():
    evaluations = {}
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            delta = 2 if arm == 'B_feature' else 0
            groups = {name: {'aggregate': {'cer_percent': float(seed + delta),
                                          'wer_percent': float(seed + delta + 10),
                                          'exact_sentences': 2 + delta}}
                      for name in audit.GROUPS}
            evaluations[seed, arm] = {'milestones': {'3600': {condition: copy.deepcopy(groups)
                                                             for condition in ('normal', 'static', 'shuffled')}}}
    paired, means = audit.independent_comparisons(evaluations)
    assert [value['seed'] for value in paired] == [42, 43]
    for value in paired:
        for groups in value['B_minus_A'].values():
            assert set(groups) == set(audit.GROUPS)
            assert all(item == {'cer_percentage_points': 2., 'exact_sentences': 2} for item in groups.values())
    assert means['A_ctc']['normal']['hold8']['cer_percent'] == 42.5
    assert means['B_feature']['normal']['train24']['cer_percent'] == 44.5


def test_complete_evaluator_provenance_matches_frozen_current_policy():
    policy = audit.read(audit.BASE / 'diagnostic_plan.json', audit.POLICY_SHA)
    frozen = {row['path']: row for row in policy['execution_code']}
    expected = audit.expected_evaluation_sources(frozen)
    assert len(expected) == 13
    audit.verify_evaluation_sources(expected, frozen)


@pytest.mark.parametrize('alteration', ['empty', 'omitted', 'duplicate', 'reordered'])
def test_incomplete_evaluator_provenance_rejected(alteration):
    policy = audit.read(audit.BASE / 'diagnostic_plan.json', audit.POLICY_SHA)
    frozen = {row['path']: row for row in policy['execution_code']}
    expected = audit.expected_evaluation_sources(frozen)
    broken = {'empty': [], 'omitted': expected[1:], 'duplicate': expected + expected[:1],
              'reordered': expected[::-1]}[alteration]
    with pytest.raises(ValueError):
        audit.verify_evaluation_sources(broken, frozen)
