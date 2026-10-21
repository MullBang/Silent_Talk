"""Independent exhaustive-path tests for the small NumPy CTC reference aligner."""

from __future__ import annotations

import itertools
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctc_alignment import CTCAlignmentError, collapse_ctc, forced_align


def exhaustive_scores(emissions: np.ndarray, blank: int) -> dict[tuple[int, ...], float]:
    """Enumerate raw token paths and collapse them independently of production helpers."""
    try:
        frames, vocabulary = emissions.shape
        results: dict[tuple[int, ...], float] = {}
        for path in itertools.product(range(vocabulary), repeat=frames):
            compressed = [key for key, _group in itertools.groupby(path)]
            target = tuple(token for token in compressed if token != blank)
            score = sum(float(emissions[frame, token]) for frame, token in enumerate(path))
            if math.isfinite(score) and score > results.get(target, -math.inf):
                results[target] = score
        return results
    except Exception as exc:
        raise RuntimeError("Exhaustive independent CTC path enumeration failed") from exc


class AlignmentTests(unittest.TestCase):
    """Check exact CTC semantics and failure behavior without Torch or media."""

    def assert_valid_result(self, result: dict, emissions: np.ndarray, target: list[int], blank: int) -> None:
        """Independently check returned transitions, score, transcript and token spans."""
        try:
            path = result["path_states"]
            expanded = result["expanded_tokens"]
            self.assertEqual(len(path), len(emissions))
            self.assertIn(path[0], (0, 1))
            self.assertIn(path[-1], (2 * len(target) - 1, 2 * len(target)))
            for left, right in zip(path, path[1:]):
                self.assertIn(right - left, (0, 1, 2))
                if right - left == 2:
                    self.assertNotEqual(expanded[right], blank)
                    self.assertNotEqual(expanded[right], expanded[left])
            recovered = [expanded[state] for state in path]
            collapsed = [key for key, _group in itertools.groupby(recovered) if key != blank]
            self.assertEqual(collapsed, target)
            self.assertAlmostEqual(sum(float(emissions[t, token]) for t, token in enumerate(recovered)),
                                   result["best_log_score"], places=12)
            self.assertEqual(len(result["token_spans"]), len(target))
            for index, span in enumerate(result["token_spans"]):
                occupied = [i for i, state in enumerate(path) if state == 2 * index + 1]
                self.assertEqual(span, {"index": index, "token_id": target[index],
                                        "start_frame": min(occupied), "end_frame": max(occupied) + 1})
            json.dumps(result, allow_nan=False)
        except Exception:
            raise

    def test_exhaustive_all_small_paths(self) -> None:
        """Compare every target of lengths one to three against V=3,T=1..6 paths."""
        try:
            rng = np.random.default_rng(20261012)
            for blank in (0, 2):
                labels = [i for i in range(3) if i != blank]
                for frames in range(1, 7):
                    probabilities = rng.uniform(0.01, 1.0, size=(frames, 3))
                    probabilities /= probabilities.sum(axis=1, keepdims=True)
                    emissions = np.log(probabilities)
                    reference = exhaustive_scores(emissions, blank)
                    for length in range(1, 4):
                        for sequence in itertools.product(labels, repeat=length):
                            with self.subTest(blank=blank, frames=frames, sequence=sequence):
                                target = list(sequence)
                                if sequence not in reference:
                                    with self.assertRaises(CTCAlignmentError):
                                        forced_align(emissions, target, blank)
                                    continue
                                result = forced_align(emissions, target, blank)
                                self.assertAlmostEqual(result["best_log_score"], reference[sequence], places=12)
                                self.assert_valid_result(result, emissions, target, blank)
        except Exception:
            raise

    def test_first_last_and_adjacent_repeated_tokens(self) -> None:
        """Repeated targets must consume a blank, including exact first and final frames."""
        try:
            emissions = np.full((3, 3), -np.inf)
            emissions[0, 0] = 0.0
            emissions[1, 2] = 0.0
            emissions[2, 0] = 0.0
            result = forced_align(emissions, [0, 0], blank=2)
            self.assertEqual(result["path_states"], [1, 2, 3])
            self.assertEqual(result["token_spans"], [{"index": 0, "token_id": 0, "start_frame": 0, "end_frame": 1},
                                                    {"index": 1, "token_id": 0, "start_frame": 2, "end_frame": 3}])
            self.assert_valid_result(result, emissions, [0, 0], 2)
        except Exception:
            raise

    def test_initial_and_final_blank_are_consumed(self) -> None:
        """The path covers blank margins rather than aligning an arbitrary inner subsequence."""
        try:
            emissions = np.full((4, 3), -np.inf)
            emissions[np.arange(4), [2, 0, 1, 2]] = 0.0
            result = forced_align(emissions, [0, 1], 2)
            self.assertEqual(result["path_states"], [0, 1, 3, 4])
            self.assert_valid_result(result, emissions, [0, 1], 2)
        except Exception:
            raise

    def test_impossible_emissions_and_lengths(self) -> None:
        """Length-feasible but zero-probability alignments are still rejected."""
        try:
            for emissions, target in ((np.full((2, 3), -math.log(3)), [1, 1]),
                                      (np.full((4, 3), -np.inf), [1]),
                                      (np.array([[0., -np.inf, -np.inf]] * 3), [1])):
                with self.subTest(target=target, frames=len(emissions)):
                    with self.assertRaises(CTCAlignmentError):
                        forced_align(emissions, target, 0)
        except Exception:
            raise

    def test_nonfinite_and_shape_validation(self) -> None:
        """NaN, positive infinity, empty shapes, non-real arrays and wrong rank fail."""
        try:
            bad_arrays = [np.array([[0., np.nan]]), np.array([[0., np.inf]]), np.empty((0, 3)),
                          np.empty((3, 0)), np.zeros((3, 1)), np.zeros(3), np.zeros((1, 2, 3)),
                          np.array([[1j, 0j]]), np.array([["0", "1"]]), np.array([[False, True]])]
            for emissions in bad_arrays:
                with self.subTest(shape=emissions.shape, dtype=str(emissions.dtype)):
                    with self.assertRaises(CTCAlignmentError):
                        forced_align(emissions, [1], 0)
            with self.assertRaises(CTCAlignmentError):
                forced_align([[0.0, -1.0]], [1], 0)
        except Exception:
            raise

    def test_target_and_blank_validation(self) -> None:
        """Reject empty targets, unknown IDs, blank targets and noninteger IDs explicitly."""
        try:
            emissions = np.full((4, 3), -math.log(3))
            for target in ([], [0], [3], [-1], [1.0], [True], "a", (1,)):
                with self.subTest(target=target):
                    with self.assertRaises(CTCAlignmentError):
                        forced_align(emissions, target, 0)
            for blank in (-1, 3, 0.0, False):
                with self.subTest(blank=blank):
                    with self.assertRaises(CTCAlignmentError):
                        forced_align(emissions, [1], blank)
            self.assert_valid_result(forced_align(emissions, [np.int64(1)], np.int64(0)), emissions, [1], 0)
        except Exception:
            raise

    def test_one_frame_and_deterministic_ties(self) -> None:
        """A one-token one-frame path and a documented tied path remain deterministic."""
        try:
            one = np.array([[-0.7, -0.8, -2.0]], dtype=np.float32)
            result = forced_align(one, [1], 2)
            self.assertEqual(result["path_states"], [1])
            self.assert_valid_result(result, one, [1], 2)
            equal = np.full((4, 3), -math.log(3))
            result = forced_align(equal, [1], 0)
            self.assertEqual(result["path_states"], [1, 1, 1, 1])
            self.assertEqual(result, forced_align(equal, [1], 0))
        except Exception:
            raise

    def test_greedy_collapse_order(self) -> None:
        """Blanks separate equal labels and are removed only after adjacent collapse."""
        try:
            self.assertEqual(collapse_ctc([2, 0, 0, 2, 0, 1, 1, 2], blank=2), [0, 0, 1])
            self.assertEqual(collapse_ctc([], 2), [])
            for path, blank in (([True], 2), ([-1], 2), ([0.0], 2), ([0], -1), ([0], True)):
                with self.subTest(path=path, blank=blank):
                    with self.assertRaises(CTCAlignmentError):
                        collapse_ctc(path, blank)
        except Exception:
            raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
