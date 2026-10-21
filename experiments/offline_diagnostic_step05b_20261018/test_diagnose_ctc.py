"""Check probability math against exhaustive tiny alignments, not implementation details."""
import itertools

import numpy as np
import pytest

from diagnose_ctc import collapse, ctc_nll, minimum_frames


@pytest.mark.parametrize('T,labels', [(1, []), (1, [1]), (2, [1, 1]),
                                    (3, [1, 1]), (3, [1, 2]), (4, [2, 1, 2]),
                                    (4, []), (0, []), (0, [1])])
def test_ctc_matches_exhaustive_paths(T, labels):
    rng = np.random.default_rng(318)
    p = rng.uniform(.05, 1, size=(T, 3))
    p /= p.sum(axis=1, keepdims=True)
    probability = sum(float(np.prod([p[t, symbol] for t, symbol in enumerate(path)]))
                      for path in itertools.product(range(3), repeat=T) if collapse(path) == labels)
    expected = -np.log(probability) if probability else np.inf
    actual = ctc_nll(np.log(p), labels)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_blank_separates_repeated_labels():
    assert collapse([1, 1, 0, 1, 2, 2, 0]) == [1, 1, 2]
    assert minimum_frames([1, 1, 2, 2]) == 6


@pytest.mark.parametrize('labels', [[0], [-1], [3]])
def test_invalid_target_rejected(labels):
    with pytest.raises(ValueError):
        ctc_nll(np.log(np.full((3, 3), 1/3)), labels)


def test_nonfinite_posterior_rejected():
    with pytest.raises(ValueError):
        ctc_nll(np.full((3, 3), np.nan), [1])
