"""Input-boundary, failure-policy and color tests independent of MediaPipe."""
from types import SimpleNamespace as NS
import unittest

import cv2
import numpy as np

from automatic_roi import AutomaticLipRoi, LIP_INDICES, bbox_with_margin


def face(x1=.3, y1=.4, x2=.7, y2=.6):
    points = [NS(x=0., y=0.) for _ in range(468)]
    for j, index in enumerate(LIP_INDICES):
        points[index] = NS(x=x1 if j % 2 == 0 else x2, y=y1 if j % 3 == 0 else y2)
    return NS(landmark=points)


class FakeDetector:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.images = []
        self.closed = False

    def process(self, image):
        self.images.append(image.copy())
        value = next(self.frames)
        if isinstance(value, Exception):
            raise value
        return NS(multi_face_landmarks=value)

    def close(self):
        self.closed = True


class TestAutomaticLipRoi(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((100, 200, 3), (9, 71, 203), dtype=np.uint8)

    def test_detection_rgb_but_output_original_bgr(self):
        detector = FakeDetector([[face()]])
        with AutomaticLipRoi(detector) as tracker:
            roi = tracker.process(self.frame)
            self.assertEqual(detector.images[0].shape, (320, 640, 3))
            np.testing.assert_array_equal(detector.images[0][0, 0], [203, 71, 9])
            self.assertEqual(roi.shape, (96, 96, 3))
            np.testing.assert_array_equal(roi[0, 0], [9, 71, 203])
            self.assertFalse(tracker.failed)
        self.assertTrue(detector.closed)

    def test_margin_clamps_all_image_boundaries(self):
        self.assertEqual(bbox_with_margin(np.array([[-5, -2], [105, 55]]), 100, 50),
                         (0, 0, 100, 50))

    def test_exact_original_pixel_crop(self):
        frame_bgr = np.zeros((100, 200, 3), np.uint8)
        frame_bgr[:, :, 0] = np.arange(200)[None, :]
        frame_bgr[:, :, 1] = np.arange(100)[:, None]
        with AutomaticLipRoi(FakeDetector([[face()]])) as tracker:
            roi = tracker.process(frame_bgr)
            x1, y1, x2, y2 = tracker.last_box_xyxy
            expected = cv2.resize(frame_bgr[y1:y2, x1:x2], (96, 96), interpolation=cv2.INTER_LINEAR)
            np.testing.assert_array_equal(roi, expected)
            self.assertLess(x1, 60)
            self.assertGreater(x2, 140)
            self.assertLess(y1, 40)
            self.assertGreater(y2, 60)

    def test_largest_mouth_selected_not_first_face(self):
        with AutomaticLipRoi(FakeDetector([[face(.1, .1, .2, .2), face(.4, .4, .8, .8)]])) as tracker:
            tracker.process(self.frame)
            self.assertGreater(tracker.last_box_xyxy[0], 50)

    def test_initial_miss_never_backfilled_or_recovered(self):
        detector = FakeDetector([[], [face()]])
        with AutomaticLipRoi(detector) as tracker:
            self.assertIsNone(tracker.process(self.frame))
            self.assertIsNone(tracker.process(self.frame))
            self.assertTrue(tracker.failed)
            self.assertEqual(tracker.failure_reason, "initial_detection_miss")
            self.assertEqual(len(detector.images), 1)
            self.assertEqual(tracker.stats()["frames_skipped_after_failure"], 1)

    def test_later_miss_reuses_bbox_on_current_frame_not_old_pixels(self):
        with AutomaticLipRoi(FakeDetector([[face()], []])) as tracker:
            first = tracker.process(self.frame)
            second = tracker.process(np.full_like(self.frame, (10, 20, 30)))
            np.testing.assert_array_equal(first[0, 0], [9, 71, 203])
            np.testing.assert_array_equal(second[0, 0], [10, 20, 30])
            self.assertEqual(tracker.stats()["frames_previous_bbox"], 1)

    def test_twelfth_consecutive_miss_rejects_whole_utterance(self):
        with AutomaticLipRoi(FakeDetector([[face()]] + [[]] * 12)) as tracker:
            tracker.process(self.frame)
            for _ in range(11):
                self.assertIsNotNone(tracker.process(self.frame))
            self.assertIsNone(tracker.process(self.frame))
            self.assertTrue(tracker.failed)
            self.assertEqual(tracker.failure_reason, "consecutive_detection_misses")
            self.assertEqual(tracker.stats()["max_consecutive_miss"], 12)

    def test_detection_resets_consecutive_miss_counter(self):
        values = [[face()]] + [[]] * 11 + [[face()]] + [[]] * 11
        with AutomaticLipRoi(FakeDetector(values)) as tracker:
            for _ in values:
                self.assertIsNotNone(tracker.process(self.frame))
            self.assertFalse(tracker.failed)
            self.assertEqual(tracker.stats()["max_consecutive_miss"], 11)

    def test_nonfinite_landmarks_fail_without_fallback(self):
        broken = face()
        broken.landmark[LIP_INDICES[0]].x = float("nan")
        with AutomaticLipRoi(FakeDetector([[broken]])) as tracker:
            self.assertIsNone(tracker.process(self.frame))
            self.assertTrue(tracker.failed)

    def test_outside_or_degenerate_bounds_rejected(self):
        for points in ([[200, 200], [210, 210]], [[1, 1], [1, 2]]):
            with self.assertRaises(ValueError):
                bbox_with_margin(np.array(points), 100, 100)

    def test_detector_error_counted_as_failure(self):
        with AutomaticLipRoi(FakeDetector([RuntimeError("broken")])) as tracker:
            self.assertIsNone(tracker.process(self.frame))
            self.assertEqual(tracker.failure_reason, "detector_exception")
            self.assertEqual(tracker.frame_records[0]["error_type"], "RuntimeError")

    def test_input_dtype_shape_and_size_changes_rejected(self):
        with AutomaticLipRoi(FakeDetector([[face()]])) as tracker:
            for bad in (self.frame.astype(float), self.frame[:, :, 0], np.zeros((0, 1, 3), np.uint8)):
                with self.assertRaises(ValueError):
                    tracker.process(bad)
            tracker.process(self.frame)
            with self.assertRaises(ValueError):
                tracker.process(self.frame[:50])

    def test_context_close_and_no_further_frames(self):
        detector = FakeDetector([[face()]])
        with AutomaticLipRoi(detector) as tracker:
            tracker.process(self.frame)
        with self.assertRaises(RuntimeError):
            tracker.process(self.frame)


if __name__ == "__main__":
    unittest.main()
