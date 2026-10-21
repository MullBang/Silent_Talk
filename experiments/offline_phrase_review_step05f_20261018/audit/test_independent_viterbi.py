"""Small exhaustive checks calibrate the scalar auditor before real posteriors."""
from __future__ import annotations

import itertools
import math
import numpy as np
import pytest

from independent_viterbi import align, collapse


@pytest.mark.parametrize("target", [[1], [1, 1], [1, 2]])
def test_score_matches_enumerated_paths(target: list[int]) -> None:
    probs = np.array([[.2, .5, .3], [.5, .3, .2], [.1, .2, .7], [.4, .3, .3]])
    possible = [sum(math.log(probs[t, token]) for t, token in enumerate(path))
                for path in itertools.product(range(3), repeat=4)
                if collapse(list(path), 0) == target]
    actual = align(np.log(probs), target, 0)
    assert actual["best_log_score"] == pytest.approx(max(possible), abs=1e-12)


def test_identical_adjacent_targets_require_blank_time() -> None:
    with pytest.raises(ValueError, match="No complete"):
        align(np.log(np.array([[.1, .9], [.2, .8]])), [1, 1], 0)
