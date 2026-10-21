"""Tests only the new post-hoc eligibility/token-retention analysis boundary."""
import copy
import importlib.util
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('posthoc_summary_test', BASE / 'summarize_decode_correction.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
tok = s.module_at('posthoc_summary_test_tokenizer', BASE.parents[1] / 'backend/models/tokenizer.py')


def record_pair(invalid=False):
    text = '' if invalid else '가'
    raw = [1, 1, 0, 1] if invalid else [1, 0, 20]
    old = {'id': 'test', 'speaker_id': 'C002', 'source_stem': 'clip', 'split': 'dev', 'condition': 'normal',
           'reference': '가', 'prediction': text, 'status': 'failed' if invalid else 'ok',
           'error': s.INVALID_JAMO_PREFIX + ': test' if invalid else None,
           'metrics': {}, 'frames': len(raw), 'temporal': {'source_positions': list(range(len(raw)))}}
    row = {**copy.deepcopy(old), 'status': 'ok', 'error': None, 'prediction': '\ufffd\ufffd' if invalid else '가',
           'raw_argmax_ids': raw, 'collapsed_ids': [1, 1] if invalid else [1, 20],
           'replacement_characters': 2 if invalid else 0, 'strict_replay_matches_original': True,
           'original_success_prediction_match': None if invalid else True,
           'correction': 'invalid_predicted_jamo_retained_as_replacement_character' if invalid else 'unchanged_valid_prediction',
           'strict_replay': {'status': old['status'], 'prediction': None if invalid else text, 'error': old['error']},
           'original': {key: old[key] for key in ['status', 'prediction', 'error', 'metrics']}}
    return row, old


def test_blank_separates_repeated_tokens_before_removal():
    assert s.collapse_ids([1, 1, 0, 1, 0, 0, 20, 20]) == [1, 1, 20]


@pytest.mark.parametrize('invalid', [False, True])
def test_valid_original_or_explicit_invalid_jamo_correction(invalid):
    row, old = record_pair(invalid)
    s.validate_record(row, old, tok)


@pytest.mark.parametrize('mutation', ['erase_bad_tokens', 'erase_replacement', 'wrong_count', 'wrong_error', 'changed_input', 'alter_valid_prediction'])
def test_silent_posthoc_changes_are_rejected(mutation):
    row, old = record_pair(mutation != 'alter_valid_prediction')
    if mutation == 'erase_bad_tokens':
        row['collapsed_ids'] = [1]
    elif mutation == 'erase_replacement':
        row['prediction'] = ''
    elif mutation == 'wrong_count':
        row['replacement_characters'] = 0
    elif mutation == 'wrong_error':
        row['strict_replay']['error'] = 'different'
    elif mutation == 'changed_input':
        row['temporal'] = {'source_positions': [0]}
    else:
        row['prediction'] = '나'
    with pytest.raises(ValueError):
        s.validate_record(row, old, tok)


def test_unrelated_failure_cannot_be_claimed_corrected():
    row, old = record_pair(True)
    old['error'] = 'RuntimeError: unrelated failure'
    row['original']['error'] = old['error']
    with pytest.raises(ValueError):
        s.validate_record(row, old, tok)


def test_replacement_characters_remain_in_text_lengths_frequency_and_examples():
    ids = [f'id{i}' for i in range(71)]
    conditions = {}
    for condition in s.CONDITIONS:
        records = [{'id': sid, 'speaker_id': 'C002', 'source_stem': 'clip', 'split': 'dev', 'condition': condition,
                    'reference': '가나', 'prediction': '\ufffd가', 'status': 'ok'} for sid in ids]
        conditions[condition] = {'records': records}
    result = s.diagnostics({'schema': 'lipnet-step05-posthoc-prediction-decode-evaluation-v1', 'status': 'completed',
        'seed': 42, 'arm': 'A_ctc', 'completed_updates': 3200, 'record_count': 71, 'ids': ids,
        'checkpoint': {}, 'plan': {}, 'conditions': conditions})
    normal = result['conditions']['normal']
    assert normal['replacement_characters'] == normal['sentences_with_replacement_characters'] == 71
    assert normal['total_prediction_characters'] == 142
    assert normal['most_frequent_predictions'] == [{'prediction': '\ufffd가', 'count': 71}]
    assert result['examples_first_five_fixed_development_rows'][0]['predictions']['normal'] == '\ufffd가'
