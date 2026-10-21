"""Known-signal regression tests for the reused Step 04 correlator.

No cloud source, dataset label, recognition model or precomputed alignment is
read.  These tests establish lag direction and rejection signals; production
validation (finite arrays, minimum length, RMS and search-boundary checks)
belongs to audit_sources.py, not to this legacy correlation function.

Run with .venv-olkavs-roi/Scripts/python.exe and unittest (no pytest required).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "olkavs_official" / "align_step04_audio.py"
SPEC = importlib.util.spec_from_file_location("step04_signal_contract_subject", MODULE_PATH)
SUBJECT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUBJECT)
RATE = SUBJECT.RATE


class SignalContractTests(unittest.TestCase):
    def make_signal(self, seconds: float = 36.0, seed: int = 9317) -> np.ndarray:
        return np.random.default_rng(seed).normal(0.0, 0.1, round(seconds * RATE))

    def known_match(self, offset_seconds: float, gain: float = 1.0,
                    dc: float = 0.0) -> tuple[dict, int]:
        # WAV anchor=30, MP4 search start=15, so a negative source offset
        # puts the matching start before search-local time 15 seconds.
        search = self.make_signal()
        start = round((15.0 + offset_seconds) * RATE)
        reference = search[start:start + 6 * RATE].copy()
        return SUBJECT.correlate_windows(reference, gain * search + dc), start

    def assert_diagnostic_strong(self, score: dict) -> None:
        self.assertGreaterEqual(score["absolute_correlation"], 0.6)
        self.assertGreaterEqual(score["peak_ratio"], 1.5)

    def test_negative_offset_sign_and_sample_grid(self) -> None:
        score, expected = self.known_match(-8.362)
        self.assertEqual(score["best_start_sample_in_search"], expected)
        actual = 15.0 + score["best_start_seconds_in_search"] - 30.0
        self.assertAlmostEqual(actual, -8.362, places=10)
        self.assert_diagnostic_strong(score)

    def test_positive_offset_sign_and_sample_grid(self) -> None:
        score, expected = self.known_match(0.046375)
        self.assertEqual(score["best_start_sample_in_search"], expected)
        actual = 15.0 + score["best_start_seconds_in_search"] - 30.0
        self.assertAlmostEqual(actual, 0.046375, places=10)
        self.assert_diagnostic_strong(score)

    def test_gain_and_dc_do_not_change_lag(self) -> None:
        baseline, expected = self.known_match(-8.362)
        score, _ = self.known_match(-8.362, gain=0.23, dc=8.0)
        self.assertEqual(score["best_start_sample_in_search"], expected)
        self.assertAlmostEqual(score["absolute_correlation"],
                               baseline["absolute_correlation"], places=10)

    def test_polarity_inversion_retains_negative_signed_correlation(self) -> None:
        score, expected = self.known_match(0.046375, gain=-0.73, dc=4.0)
        self.assertEqual(score["best_start_sample_in_search"], expected)
        self.assertLess(score["signed_correlation"], -0.99)
        self.assertGreater(score["absolute_correlation"], 0.99)
        self.assert_diagnostic_strong(score)

    def test_moderate_noise_is_diagnostic_but_not_strict_point99(self) -> None:
        search = self.make_signal()
        start = 8 * RATE
        reference = search[start:start + 6 * RATE].copy()
        noise = self.make_signal(seed=7) * 0.3
        score = SUBJECT.correlate_windows(reference, search + noise)
        self.assertEqual(score["best_start_sample_in_search"], start)
        self.assert_diagnostic_strong(score)
        self.assertLess(score["absolute_correlation"], 0.99)

    def test_unrelated_noise_does_not_pass_correlation_gate(self) -> None:
        score = SUBJECT.correlate_windows(self.make_signal(6.0, 123),
                                          self.make_signal(36.0, 789))
        self.assertLess(score["absolute_correlation"], 0.1)

    def test_zero_reference_is_not_strong(self) -> None:
        score = SUBJECT.correlate_windows(np.zeros(6 * RATE), self.make_signal())
        self.assertEqual(score["absolute_correlation"], 0.0)
        self.assertEqual(score["reference_rms_filtered"], 0.0)

    def test_zero_search_is_not_strong(self) -> None:
        score = SUBJECT.correlate_windows(self.make_signal(6.0), np.zeros(36 * RATE))
        self.assertEqual(score["absolute_correlation"], 0.0)
        self.assertEqual(score["matched_search_rms_filtered_centered"], 0.0)

    def test_constant_dc_is_not_evidence(self) -> None:
        score = SUBJECT.correlate_windows(np.full(6 * RATE, 0.2),
                                          np.full(36 * RATE, -0.5))
        self.assertLess(score["absolute_correlation"], 0.6)
        self.assertLess(score["reference_rms_filtered"], 1e-8)

    def test_two_far_matches_rejected_by_peak_ratio(self) -> None:
        reference = self.make_signal(6.0, 12)
        search = np.zeros(36 * RATE)
        search[4 * RATE:10 * RATE] = reference
        search[20 * RATE:26 * RATE] = reference
        score = SUBJECT.correlate_windows(reference, search)
        self.assertGreater(score["absolute_correlation"], 0.99)
        self.assertLess(score["peak_ratio"], 1.5)

    def test_boundary_maximum_requires_external_boundary_guard(self) -> None:
        search = self.make_signal()
        score = SUBJECT.correlate_windows(search[:6 * RATE].copy(), search)
        self.assertEqual(score["best_start_sample_in_search"], 0)
        # This intentionally demonstrates why a strong correlation alone is
        # insufficient: a maximum at the bounded search edge needs review.
        self.assert_diagnostic_strong(score)

    def test_input_arrays_are_unchanged(self) -> None:
        search = self.make_signal()
        reference = search[8 * RATE:14 * RATE].copy()
        saved_search, saved_reference = search.copy(), reference.copy()
        SUBJECT.correlate_windows(reference, search)
        np.testing.assert_array_equal(search, saved_search)
        np.testing.assert_array_equal(reference, saved_reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
