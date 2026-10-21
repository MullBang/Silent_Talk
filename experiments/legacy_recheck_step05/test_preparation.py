"""Failure gates and legacy temporal/spatial behavior, using generated frames."""
import unittest
import cv2
import numpy as np
from prepare_inputs import extract, offsets, require_exact_reproduction, target_selection
from models.tokenizer import encode_text


class FakeCapture:
    def __init__(self, fps=30., broken_pts=False):
        self.fps, self.position, self.broken_pts = fps, 0, broken_pts
        y, x = np.indices((60, 100))
        self.frames = [np.stack(((x + i) % 255, y, x * 2 % 255), axis=-1).astype(np.uint8) for i in range(100)]

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
                cv2.CAP_PROP_POS_FRAMES: self.position,
                cv2.CAP_PROP_POS_MSEC: (self.position - 1) * 1000 / self.fps + (50 if self.broken_pts else 0)}[prop]

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError("Original helper must not set FPS")
        self.position = int(value)
        return True

    def read(self):
        frame = self.frames[self.position]
        self.position += 1
        return True, frame


class PreparationTests(unittest.TestCase):
    def test_contract_sampling_rgb_margin(self):
        cap = FakeCapture()
        pixels, trace = extract(cap, .2, .6, [[30, 20, 50, 30]] * 100, 30.)
        self.assertEqual(pixels.shape, (10, 96, 96, 3))
        self.assertEqual([r["source_frame_index"] for r in trace["selected"]], [6, 7, 8, 10, 11, 12, 13, 14, 16, 17])
        # Independent expected margin: width20*.2=4, height10*.2=2.
        rgb = cv2.cvtColor(cap.frames[6], cv2.COLOR_BGR2RGB)
        expected = cv2.resize(rgb[18:32, 26:54], (96, 96), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(pixels[0], expected)

    def test_corrected_offset_preserves_t_count_and_real_video_indices(self):
        old_times, old = target_selection(10.069333333333333, 11.477333333333334, 30000 / 1001)
        new_times, new = target_selection(10.069333333333333 - 8.362, 11.477333333333334 - 8.362, 30000 / 1001)
        self.assertEqual(len(old), 35)
        self.assertEqual(len(new), 35)
        self.assertEqual(new[0], 51)
        self.assertAlmostEqual(old_times[0] - new_times[0], 8.362)

    def test_original_helper_rejects_wrong_pts(self):
        with self.assertRaisesRegex(ValueError, "PTS"):
            extract(FakeCapture(broken_pts=True), .2, .6, [[30, 20, 50, 30]] * 100, 30.)

    def test_original_helper_rejects_box_count_mismatch(self):
        with self.assertRaises(ValueError):
            extract(FakeCapture(), .2, .6, [[30, 20, 50, 30]] * 99, 30.)

    def test_original_helper_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            extract(FakeCapture(), -1, .6, [[30, 20, 50, 30]] * 100, 30.)

    def test_validated_offsets_require_all_sources(self):
        valid = {"status": "validated", "sources": [{"source_name": "a.mp4", "label_to_video_offset_seconds": -8.362}]}
        self.assertEqual(offsets(valid, {"a.mp4"}), {"a.mp4": -8.362})
        with self.assertRaises(ValueError):
            offsets(valid, {"a.mp4", "b.mp4"})
        valid["sources"].append(valid["sources"][0])
        with self.assertRaises(ValueError):
            offsets(valid, {"a.mp4"})

    def test_nonfinite_offset_rejected(self):
        with self.assertRaises(ValueError):
            offsets({"status": "validated", "sources": [{"source_name": "a", "label_to_video_offset_seconds": float("nan")}]}, {"a"})

    def test_reproduction_one_pixel_difference_is_failure(self):
        pixels = np.zeros((2, 96, 96, 3), np.uint8)
        cached = {"frames": pixels.copy(), "label": np.array(encode_text("가")), "text": "가", "source_text": "가."}
        require_exact_reproduction(pixels, cached, "가", "가.")
        pixels[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "exactly"):
            require_exact_reproduction(pixels, cached, "가", "가.")

    def test_reproduction_transcript_or_token_change_is_failure(self):
        pixels = np.zeros((2, 96, 96, 3), np.uint8)
        cached = {"frames": pixels.copy(), "label": np.array(encode_text("나")), "text": "가", "source_text": "가."}
        with self.assertRaises(ValueError):
            require_exact_reproduction(pixels, cached, "가", "가.")
        cached["label"] = np.array(encode_text("가"))
        with self.assertRaises(ValueError):
            require_exact_reproduction(pixels, cached, "가", "가!")


if __name__ == "__main__":
    unittest.main()
