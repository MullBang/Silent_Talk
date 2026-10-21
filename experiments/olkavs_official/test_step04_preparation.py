"""Replay validation prevents unnoticed cohort, frame and pixel changes."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_step04_inputs import (geometry_stats, rectangle_control,
                                   require_same_frames, validate_trace, verify_control)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.row = {"id": "sample", "text": "가 나", "split": "val", "start_time": 1.0, "end_time": 1.1}
        self.prior = {"id": "sample", "text": "가 나", "native_fps": 30.0,
                      "start_frame": 30, "stop_frame_exclusive": 33, "status": "ok"}
        self.trace = {"id": "sample", "frames": [{"source_frame_index": i,
            "automatic_failed": False, "automatic_xyxy": [2, 3, 7, 8]} for i in range(30, 33)]}

    def test_exact_cohort_and_trace_accepted(self):
        validate_trace(self.row, self.prior, self.trace)

    def test_reordered_or_missing_source_frames_rejected(self):
        for indices in ([30, 32, 31], [30, 31], [31, 32, 33]):
            trace = copy.deepcopy(self.trace)
            trace["frames"] = [{"source_frame_index": i, "automatic_failed": False,
                                "automatic_xyxy": [2, 3, 7, 8]} for i in indices]
            with self.assertRaises(ValueError):
                validate_trace(self.row, self.prior, trace)

    def test_missing_or_failed_detector_box_rejected(self):
        for name, value in (("automatic_xyxy", None), ("automatic_failed", True)):
            trace = copy.deepcopy(self.trace)
            trace["frames"][0][name] = value
            with self.assertRaises(ValueError):
                validate_trace(self.row, self.prior, trace)

    def test_different_transcript_or_time_rejected(self):
        for name, value in (("text", "다 라"), ("start_time", 1.02)):
            row = {**self.row, name: value}
            with self.assertRaises(ValueError):
                validate_trace(row, self.prior, self.trace)

    def test_source_stream_mismatch_stops_run(self):
        require_same_frames("a", "a")
        with self.assertRaises(ValueError):
            require_same_frames("a", "b")

    def test_control_preserves_exact_original_rectangle(self):
        frame = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
        actual, telemetry = rectangle_control(frame, [4, 7, 30, 16])
        expected = cv2.resize(frame[7:16, 4:30], (96, 96), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(telemetry["final_rectangle_xyxy"], [4, 7, 30, 16])

    def test_decoded_control_equality_required(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "original.mp4"
            path.write_bytes(b"fixture")
            prior = {"path": str(path), "sha256": hashlib.sha256(b"fixture").hexdigest(),
                     "shape": [3, 96, 96, 3], "decoded_bgr_stream_sha256": "pixels"}
            stats = {"shape": [3, 96, 96, 3], "decoded_bgr_stream_sha256": "pixels"}
            self.assertTrue(verify_control(path, stats, prior)["encoded_bytes_identical"])
            with self.assertRaises(ValueError):
                verify_control(path, {**stats, "decoded_bgr_stream_sha256": "different"}, prior)

    def test_geometry_summary_is_descriptive(self):
        stats = geometry_stats([{"raw_bbox_xyxy": [2, 3, 12, 8],
            "final_square_xyxy": [2, 1, 12, 11], "boundary_shifted": False},
            {"raw_bbox_xyxy": [4, 3, 14, 8], "final_square_xyxy": [4, 1, 14, 11], "boundary_shifted": False}])
        self.assertEqual(stats["median_crop_area_to_original_detector_box"], 2)
        self.assertEqual(stats["center_step_pixels_mean"], 2)


if __name__ == "__main__":
    unittest.main()
