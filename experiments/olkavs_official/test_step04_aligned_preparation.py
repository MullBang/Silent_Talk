"""Independent timeline offsets cannot silently alter targets or selection."""
import copy
import unittest

import cv2
import numpy as np

from prepare_step04_aligned_inputs import (ARMS, aligned_bounds, annotation_crop,
                                           offsets_by_source)
from prepare_step03_inputs import native_bounds


class AlignedPreparationTests(unittest.TestCase):
    def test_negative_offset_selects_earlier_video_without_changing_target(self):
        original = {"id": "same-id", "text": "원래 정답", "start": 10.069333333333333, "end": 11.477333333333334}
        before = original.copy()
        fps = 30000 / 1001
        first, stop = aligned_bounds(original["start"], original["end"], -8.362, fps)
        self.assertEqual((first, stop), (52, 94))
        self.assertEqual(original, before)
        self.assertLess(first, native_bounds(original["start"], original["end"], fps)[0])

    def test_positive_offset_uses_native_half_open_timestamp_interval(self):
        start, end, offset, fps = 30.25, 32.125, .046375, 30000 / 1001
        first, stop = aligned_bounds(start, end, offset, fps)
        self.assertGreaterEqual(first / fps, start + offset)
        self.assertLess((first - 1) / fps, start + offset)
        self.assertLess((stop - 1) / fps, end + offset)
        self.assertGreaterEqual(stop / fps, end + offset)

    def test_zero_offset_exactly_preserves_stage03_native_selection(self):
        self.assertEqual(aligned_bounds(246.997333, 249.216, 0, 30000 / 1001),
                         native_bounds(246.997333, 249.216, 30000 / 1001))

    def test_before_video_start_and_nonfinite_offsets_rejected(self):
        for offset in (-10, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                aligned_bounds(1.0, 2.0, offset, 30)

    def test_sidecar_cannot_omit_duplicate_or_add_sources(self):
        valid = {"status": "validated", "sources": [
            {"source_name": "a.mp4", "label_to_video_offset_seconds": -8.362},
            {"source_name": "b.mp4", "label_to_video_offset_seconds": .046375}]}
        self.assertEqual(offsets_by_source(valid, {"a.mp4", "b.mp4"}),
                         {"a.mp4": -8.362, "b.mp4": .046375})
        invalids = [valid["sources"][:1], valid["sources"] + valid["sources"][:1],
                    valid["sources"] + [{"source_name": "c.mp4", "label_to_video_offset_seconds": 0}]]
        for rows in invalids:
            with self.assertRaises(ValueError):
                offsets_by_source({"status": "validated", "sources": rows}, {"a.mp4", "b.mp4"})
        unvalidated = copy.deepcopy(valid)
        unvalidated["status"] = "pending"
        with self.assertRaises(ValueError):
            offsets_by_source(unvalidated, {"a.mp4", "b.mp4"})

    def test_five_arms_include_annotation_control_and_three_candidates(self):
        self.assertEqual(ARMS, ("annotation_bbox", "rectangle_control", "square",
                               "context_square", "stable_context_square"))
        self.assertEqual(len({(sample, arm) for sample in range(20) for arm in ARMS}), 100)

    def test_annotation_box_axis_order_and_full_slice_preserved(self):
        frame = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
        actual, telemetry = annotation_crop(frame, [4, 9, 13, 28])
        expected = cv2.resize(frame[4:13, 9:28], (96, 96), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(telemetry["final_rectangle_xyxy"], [9, 4, 28, 13])
        with self.assertRaises(ValueError):
            annotation_crop(frame, [4, 9, 35, 28])


if __name__ == "__main__":
    unittest.main()
