"""Geometry contract tests requiring no detector, model, or source videos."""
import json
import unittest

import cv2
import numpy as np

from roi_geometry_step04 import ARMS, RoiGeometry, profile


class TestRoiGeometryStep04(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((160, 240, 3), dtype=np.uint8)
        self.frame[..., 0] = np.arange(240)[None, :]
        self.frame[..., 1] = np.arange(160)[:, None]
        self.frame[..., 2] = 211

    def test_square_preserves_source_aspect_and_bgr(self):
        crop, row = RoiGeometry("square").process(self.frame, (70, 60, 110, 80))
        self.assertEqual(row["final_square_xyxy"], [70, 50, 110, 90])
        self.assertEqual(crop.shape, (96, 96, 3))
        self.assertEqual(crop.dtype, np.uint8)
        expected = cv2.resize(self.frame[50:90, 70:110], (96, 96), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(crop, expected)
        self.assertTrue((crop[..., 2] == 211).all())

    def test_context_factor_and_fractional_rounding(self):
        _, square = RoiGeometry("square").process(self.frame, (70, 60, 111, 80))
        _, context = RoiGeometry("context_square").process(self.frame, (70, 60, 111, 80))
        self.assertEqual(square["integer_side"], 41)
        self.assertEqual(context["pre_smoothing_side"], 61.5)
        self.assertEqual(context["integer_side"], 62)
        self.assertEqual(context["requested_square_xyxy"], [59, 39, 121, 101])

    def test_all_edges_shift_square_without_truncation_or_padding(self):
        boxes = [(0, 0, 40, 20), (200, 0, 240, 20),
                 (0, 140, 40, 160), (200, 140, 240, 160)]
        expected = [[0, 0, 60, 60], [180, 0, 240, 60],
                    [0, 100, 60, 160], [180, 100, 240, 160]]
        for box, bounds in zip(boxes, expected):
            with self.subTest(box=box):
                crop, row = RoiGeometry("context_square").process(self.frame, box)
                self.assertEqual(row["final_square_xyxy"], bounds)
                self.assertTrue(row["boundary_shifted"])
                x1, y1, x2, y2 = bounds
                self.assertEqual(x2 - x1, y2 - y1)
                np.testing.assert_array_equal(crop, cv2.resize(self.frame[y1:y2, x1:x2], (96, 96)))

    def test_fit_equal_shorter_dimension_and_reject_oversize(self):
        _, row = RoiGeometry("square").process(self.frame, (0, 0, 160, 40))
        self.assertEqual(row["final_square_xyxy"], [0, 0, 160, 160])
        for arm, box in [("square", (0, 0, 161, 40)),
                         ("context_square", (0, 0, 107, 40)),
                         ("stable_context_square", (0, 0, 107, 40))]:
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                RoiGeometry(arm).process(self.frame, box)

    def test_stable_first_frame_and_exact_causal_ema(self):
        geometry = RoiGeometry("stable_context_square")
        _, first = geometry.process(self.frame, (70, 60, 110, 80))
        _, second = geometry.process(self.frame, (80, 70, 140, 90))
        self.assertFalse(first["smoothing_applied"])
        self.assertEqual(first["pre_smoothing_center_xy"], [90, 70])
        self.assertEqual(first["post_smoothing_center_xy"], [90, 70])
        self.assertEqual(first["post_smoothing_side"], 60)
        self.assertTrue(second["smoothing_applied"])
        self.assertEqual(second["post_smoothing_center_xy"], [98, 74])
        self.assertEqual(second["post_smoothing_side"], 72)
        self.assertEqual(second["final_square_xyxy"], [62, 38, 134, 110])

    def test_no_smoothing_in_two_other_arms(self):
        for arm in ("square", "context_square"):
            geometry = RoiGeometry(arm)
            geometry.process(self.frame, (70, 60, 110, 80))
            _, row = geometry.process(self.frame, (80, 70, 140, 90))
            self.assertFalse(row["smoothing_applied"])
            self.assertEqual(row["post_smoothing_center_xy"], [110, 80])
            self.assertEqual(row["post_smoothing_side"], row["pre_smoothing_side"])

    def test_future_input_cannot_change_emitted_prefix(self):
        a, b = RoiGeometry("stable_context_square"), RoiGeometry("stable_context_square")
        boxes = [(70, 60, 110, 80), (80, 70, 140, 90)]
        prefix_a = [a.process(self.frame, box) for box in boxes]
        prefix_b = [b.process(self.frame, box) for box in boxes]
        snapshots = [(crop.copy(), json.dumps(row, sort_keys=True)) for crop, row in prefix_a]
        a.process(self.frame, (0, 0, 30, 20))
        b.process(self.frame, (180, 110, 230, 140))
        for (crop_a, row_a), (crop_b, row_b), (saved_crop, saved_row) in zip(prefix_a, prefix_b, snapshots):
            np.testing.assert_array_equal(crop_a, crop_b)
            np.testing.assert_array_equal(crop_a, saved_crop)
            self.assertEqual(row_a, row_b)
            self.assertEqual(json.dumps(row_a, sort_keys=True), saved_row)

    def test_current_pixels_used_even_when_box_repeated(self):
        for arm in ARMS:
            geometry = RoiGeometry(arm)
            geometry.process(self.frame, (70, 60, 110, 80))
            current = np.full_like(self.frame, (17, 29, 83))
            crop, _ = geometry.process(current, (70, 60, 110, 80))
            np.testing.assert_array_equal(crop, np.full((96, 96, 3), (17, 29, 83), np.uint8))

    def test_reset_is_new_utterance_and_allows_new_dimensions(self):
        geometry = RoiGeometry("stable_context_square")
        geometry.process(self.frame, (70, 60, 110, 80))
        geometry.reset()
        frame = self.frame[:150, :200]
        crop, row = geometry.process(frame, (80, 70, 140, 90))
        other_crop, other_row = RoiGeometry("stable_context_square").process(frame, (80, 70, 140, 90))
        self.assertEqual(row, other_row)
        self.assertEqual(row["frame_index"], 0)
        np.testing.assert_array_equal(crop, other_crop)

    def test_boundary_shift_does_not_feed_back_into_ema(self):
        geometry = RoiGeometry("stable_context_square")
        _, first = geometry.process(self.frame, (0, 0, 40, 20))
        self.assertEqual(first["post_smoothing_center_xy"], [20, 10])
        self.assertEqual(first["final_square_xyxy"], [0, 0, 60, 60])
        _, second = geometry.process(self.frame, (10, 10, 50, 30))
        self.assertEqual(second["post_smoothing_center_xy"], [24, 14])

    def test_invalid_boxes_and_frame_shapes_rejected_without_state_update(self):
        invalid_boxes = [None, [], [1, 2, 3], [[1, 2, 3, 4]],
                         [1, 2, 1, 4], [1, 2, 3, 2], [-1, 2, 3, 4],
                         [1, 2, 241, 4], [1, 2, 3, 161], [1, np.nan, 3, 4],
                         [1, 2, np.inf, 4], ["bad", 2, 3, 4]]
        geometry = RoiGeometry("stable_context_square")
        for box in invalid_boxes:
            with self.subTest(box=box), self.assertRaises(ValueError):
                geometry.process(self.frame, box)
        _, first = geometry.process(self.frame, (70, 60, 110, 80))
        self.assertEqual(first["frame_index"], 0)
        bad_frames = [None, self.frame.astype(np.float32), self.frame[..., 0],
                      self.frame[:0], self.frame[..., :2], self.frame[:100]]
        for frame in bad_frames:
            with self.subTest(frame_shape=getattr(frame, "shape", None)), self.assertRaises(ValueError):
                geometry.process(frame, (70, 60, 110, 80))
        _, second = geometry.process(self.frame, (80, 70, 140, 90))
        self.assertEqual(second["frame_index"], 1)
        self.assertEqual(second["post_smoothing_center_xy"], [98, 74])

    def test_unknown_arm_rejected_and_profile_serializable(self):
        with self.assertRaises(ValueError):
            RoiGeometry("rectangular")
        self.assertEqual(profile()["arms"], list(ARMS))
        json.dumps(profile(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
