"""Synthetic numerical contracts for the separate finer-grid follow-up."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from refine_alignment import FINE_RATE, RATE, global_score, refine_source


def analytic_pair(delay, *, polarity=1., dc=0.):
    rng = np.random.default_rng(6017)
    frequency = rng.uniform(150, 3200, 83)
    phase = rng.uniform(-np.pi, np.pi, len(frequency))
    amplitude = rng.uniform(.3, 1, len(frequency))
    def signal(times):
        out = np.zeros_like(times)
        for f, p, a in zip(frequency, phase, amplitude):
            out += a*np.sin(2*np.pi*f*times+p)
        return out/len(frequency)
    return signal(np.arange(6*RATE)/RATE+delay), polarity*signal(np.arange(10*RATE)/RATE)+dc


class RefinementSignalTests(unittest.TestCase):
    def test_global_fractional_delay_and_all_gates(self):
        known = 2+1024/48000
        reference, search = analytic_pair(known)
        score = global_score(reference, search)
        self.assertAlmostEqual(score["best_start_seconds_in_search"], known, places=12)
        self.assertEqual(score["global_candidate_count"], 4*FINE_RATE+1)
        self.assertGreater(score["absolute_correlation"], .995)
        self.assertTrue(score["strict_anchor_pass"])

    def test_negative_gain_and_dc_preserve_signed_pearson(self):
        known = 1+1337/48000
        reference, search = analytic_pair(known, polarity=-1.75, dc=.3)
        score = global_score(reference, search)
        self.assertAlmostEqual(score["best_start_seconds_in_search"], known, places=12)
        self.assertLess(score["signed_correlation"], -.995)
        self.assertTrue(score["strict_anchor_pass"])

    def test_silence_cannot_pass(self):
        score = global_score(np.zeros(6*RATE), np.zeros(10*RATE))
        self.assertFalse(score["strict_anchor_pass"])
        self.assertFalse(score["strict_gates"]["reference_energy"])
        self.assertFalse(score["strict_gates"]["matched_energy"])
        self.assertEqual(score["absolute_correlation"], 0.)

    def test_boundary_peak_cannot_pass(self):
        known = .1+1/48000
        reference, search = analytic_pair(known)
        score = global_score(reference, search)
        self.assertAlmostEqual(score["best_start_seconds_in_search"], known, places=12)
        self.assertFalse(score["strict_gates"]["boundary_clearance"])
        self.assertFalse(score["strict_anchor_pass"])

    def test_repeated_signal_cannot_pass_unique_peak(self):
        rng = np.random.default_rng(6048)
        tile = rng.normal(size=2*RATE)
        search = np.tile(tile, 6)
        reference = search[2*RATE:5*RATE].copy()
        score = global_score(reference, search)
        self.assertGreater(score["absolute_correlation"], .99)
        self.assertFalse(score["strict_gates"]["unique_far_peak"])
        self.assertFalse(score["strict_anchor_pass"])

    def test_late_evidence_change_clears_mapping_and_frame_impact(self):
        source = {"source_stem": "synthetic", "speaker_id": "C000", "phase": "screen8",
                  "branch_sample_ids": {}, "training_records": [{"id": "synthetic_001",
                  "start_time": 40., "end_time": 42., "video_fps": 30., "frames": 50}]}
        anchors = []
        for anchor in (30., 150., 270.):
            anchors.append({"reference": {"path": "ref.wav", "sha256": "ref", "start_seconds": anchor, "frames": 48000},
                            "mp4_search": {"path": "search.wav", "sha256": "search", "start_seconds": anchor-15, "frames": 288000}})
        original = {"plan_sha256": "oldplan", "source_stem": "synthetic", "status": "supported",
                    "summary": {"video_duration_seconds": 305.}, "anchors": anchors}
        plan = {"original_plan": {"sha256": "oldplan"}, "anchors_seconds": [30., 150., 270.]}
        def score(*_):
            return {"best_start_seconds_in_search": 15.+1024/48000, "strict_anchor_pass": True,
                    "absolute_correlation": .999, "peak_ratio": 4.}
        with patch("refine_alignment.read", return_value=original), \
             patch("refine_alignment.bind", return_value={"path": "old.json", "sha256": "before"}), \
             patch("refine_alignment.sha", return_value="after"), \
             patch("refine_alignment.pcm", return_value=np.zeros(8)), \
             patch("refine_alignment.global_score", side_effect=score):
            result = refine_source("old.json", source, plan)
        self.assertEqual(result["status"], "quarantined")
        self.assertIn("Original result changed", result["reason"])
        self.assertNotIn("training_frame_selection_impact", result)
        self.assertFalse(result["summary"]["constant_mapping_supported"])
        self.assertFalse(result["summary"]["original_integrity_checks_completed"])


if __name__ == "__main__":
    unittest.main()
