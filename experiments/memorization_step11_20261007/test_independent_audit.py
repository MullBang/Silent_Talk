"""Adversarial checks for independent train-only selection and decoding audit."""
from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('step11_audit_under_test', Path(__file__).with_name('independent_audit.py'))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize(('raw', 'expected'), [([1, 1, 0, 1], [1, 1]), ([0, 0], []),
                                               ([1, 20, 20, 68, 0, 68], [1, 20, 68, 68])])
def test_ctc_collapse_order(raw, expected):
    assert audit.collapse(raw, len(raw)) == expected


@pytest.mark.parametrize('raw', [[69], [-1], [True], [1.0]])
def test_invalid_tokens_rejected(raw):
    with pytest.raises(ValueError):
        audit.collapse(raw, len(raw))


def test_frame_count_must_match():
    with pytest.raises(ValueError):
        audit.collapse([1], 2)


@pytest.mark.parametrize(('ids', 'expected'), [([1, 20], '가'), ([1, 20, 41], '각'),
                                               ([12, 20, 68, 12, 20], '아 아'),
                                               ([20, 1, 20], '\ufffd가'),
                                               ([1, 1, 20, 41, 42], '\ufffd각\ufffd'), ([], '')])
def test_independent_unicode_composition(ids, expected):
    result = audit.compose(ids)
    assert result['prediction'] == expected
    assert result['replacement_characters'] == expected.count('\ufffd')
    assert result['strict']['status'] == ('failed' if '\ufffd' in expected else 'ok')


@pytest.mark.parametrize(('text', 'expected'), [('', (0, 0)), ('아아아 아아', (3, 0)),
                                               ('아아아아아 아아아아아아', (6, 2)),
                                               ('     ', (0, 0)), ('aabbbccc', (3, 0))])
def test_repetition_respects_whitespace_boundaries(text, expected):
    assert audit.repeated_runs(text) == expected


def sources():
    entries, rows = [], {}
    for role, prefix in [('old_train', 'O'), ('new_provisional_train', 'N')]:
        for speaker in range(4):
            for clip in range(4):
                sid = f'{prefix}{speaker}_{clip}'
                entries.append({'id': sid, 'role': role, 'speaker_id': f'{prefix}{speaker}'})
                rows[sid] = {'frames': 10 + clip, 'speaker_id': f'{prefix}{speaker}'}
    return entries, rows


def test_lower_median_and_group_sizes():
    entries, rows = sources()
    ids, groups = audit.selection_ids(entries, rows)
    assert sorted(ids) == sorted(f'{p}{s}_1' for p in ('O', 'N') for s in range(4))
    assert set(groups) == {'old4', 'new4'}
    assert all(len(value) == 4 for value in groups.values())


def test_held_and_dev_cannot_enter_selection():
    entries, rows = sources()
    expected = audit.selection_ids(entries, rows)
    for i, role in enumerate(['old_development', 'new_development', 'new_train_hold']):
        entries.append({'id': f'excluded{i}', 'role': role, 'speaker_id': f'Z{i}'})
        rows[f'excluded{i}'] = {'frames': 1, 'speaker_id': f'Z{i}'}
    assert audit.selection_ids(entries, rows) == expected


def test_source_order_cannot_change_selection():
    entries, rows = sources()
    assert audit.selection_ids(entries[::-1], rows) == audit.selection_ids(entries, rows)


def test_duplicate_source_ids_rejected():
    entries, rows = sources()
    with pytest.raises(ValueError):
        audit.selection_ids(entries + entries[:1], rows)


def test_insufficient_speakers_rejected():
    entries, rows = sources()
    with pytest.raises(ValueError):
        audit.selection_ids([e for e in entries if e['speaker_id'] != 'O0'], rows)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 2.0])
def test_nonfinite_or_changed_metric_rejected(value):
    with pytest.raises(ValueError):
        audit.assert_near(value, 1.0, 'changed')


@pytest.mark.parametrize('target', [[1], [1, 1], [1, 2], [2, 1, 2]])
def test_independent_ctc_matches_exhaustive_path_enumeration(target):
    rng = np.random.default_rng(41)
    probabilities = rng.uniform(.1, 1., (4, 3))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    logp = np.full((4, 69), -1000., dtype=np.float64)
    logp[:, :3] = np.log(probabilities)
    probability_sum = 0.
    for path in itertools.product(range(3), repeat=4):
        if audit.collapse(list(path), 4) == target:
            probability_sum += math.prod(probabilities[t, token] for t, token in enumerate(path))
    expected = -math.log(probability_sum) / len(target)
    assert audit.ctc_mean_numpy(logp, target) == pytest.approx(expected, abs=1e-12)


def test_ctc_repeated_target_requires_intervening_blank():
    probabilities = np.full((2, 69), 1. / 69)
    assert math.isinf(audit.ctc_mean_numpy(np.log(probabilities), [1, 1]))


def test_ctc_rejects_blank_target():
    with pytest.raises(ValueError):
        audit.ctc_mean_numpy(np.zeros((3, 69)), [0])
