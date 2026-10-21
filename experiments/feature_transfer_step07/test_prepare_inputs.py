"""Synthetic contract checks; reads no project dataset or model weights."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from prepare_inputs import (extract_teacher, interval_contract, legacy,
                            teacher_crop, verify_payload)


class FakeCapture:
    def __init__(self, frames, fps=30.0, pts_error=0.0, corrupt_at=None):
        self.frames, self.fps, self.pos = frames, fps, 0
        self.pts_error, self.corrupt_at = pts_error, corrupt_at

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(prop)
        self.pos = int(value)
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
                cv2.CAP_PROP_POS_FRAMES: self.pos,
                cv2.CAP_PROP_POS_MSEC: ((self.pos - 1) / self.fps + self.pts_error) * 1000}[prop]

    def read(self):
        if self.pos >= len(self.frames) or self.pos == self.corrupt_at:
            return False, None
        out = self.frames[self.pos].copy()
        self.pos += 1
        return True, out


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.arange(60 * 24 * 40 * 3, dtype=np.uint32).reshape(60, 24, 40, 3).astype(np.uint8)
        self.boxes = [[7, 5, 29, 18] for _ in self.frames]

    def test_round_before_ceil_is_included_in_teacher_union(self):
        interval = interval_contract(.01, .21, 30, 60)
        self.assertEqual(interval["official_first"], 1)
        self.assertEqual(interval["student_indices"][0], 0)
        self.assertEqual(interval["teacher_first"], 0)
        self.assertTrue(interval["context_clipped_left"])
        np.testing.assert_array_equal(interval["teacher_indices"][interval["student_teacher_rows"]], interval["student_indices"])

    def test_fractional_native_rate_has_exact_integer_connection(self):
        interval = interval_contract(12.021333333, 16.432, 30000 / 1001, 10000)
        expected = [round((12.021333333 + k / 25) * (30000 / 1001)) for k in range(round((16.432 - 12.021333333) * 25))]
        np.testing.assert_array_equal(interval["student_indices"], expected)
        self.assertTrue((interval["left_real_context"] == 2).all())
        self.assertTrue((interval["right_real_context"] == 2).all())
        np.testing.assert_array_equal(interval["teacher_indices"][interval["student_teacher_rows"]], expected)

    def test_physical_end_context_is_clipped_and_declared(self):
        interval = interval_contract(1.6, 2, 30, 60)
        self.assertEqual(interval["teacher_stop"], 60)
        self.assertTrue(interval["context_clipped_right"])
        self.assertEqual(interval["right_real_context"][-1], 0)
        self.assertEqual(interval["student_indices"][-1], 59)

    def test_invalid_times_and_context_fail_closed(self):
        for args in ((-.1, .2, 30, 60), (.2, .2, 30, 60), (.2, 2.1, 30, 60),
                     (.1, .2, float("nan"), 60), (.1, .2, 30, 60, 3)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                interval_contract(*args)

    def test_teacher_crop_retains_bgr_and_exact_margin_zero(self):
        frame = self.frames[7]
        crop, rectangle = teacher_crop(frame, self.boxes[7])
        expected = cv2.resize(frame[5:18, 7:29], (96, 96), interpolation=cv2.INTER_LINEAR)
        self.assertEqual(rectangle, [7, 5, 29, 18])
        np.testing.assert_array_equal(crop, expected)
        self.assertFalse(np.array_equal(crop, cv2.cvtColor(expected, cv2.COLOR_BGR2RGB)))

    def test_student_uses_exact_original_helper_and_source_identity(self):
        interval = interval_contract(.3, .7, 30, 60)
        student, trace_s = legacy.extract(FakeCapture(self.frames), .3, .7, self.boxes, 30)
        teacher, trace_t = extract_teacher(FakeCapture(self.frames), interval["teacher_indices"], self.boxes, 30)
        direct = legacy._roi_seq_from_boxes(FakeCapture(self.frames), .3, .7, self.boxes, 30)
        np.testing.assert_array_equal(student, direct)
        for i, item in enumerate(trace_s["selected"]):
            matched = trace_t["selected"][int(interval["student_teacher_rows"][i])]
            self.assertEqual(item["source_frame_index"], matched["source_frame_index"])
            self.assertEqual(item["source_bgr_pixel_sha256"], matched["source_bgr_pixel_sha256"])
        self.assertFalse(np.array_equal(student[0], teacher[int(interval["student_teacher_rows"][0])]))

    def test_teacher_rejects_wrong_pts_and_truncated_decode(self):
        indices = np.arange(4, 10, dtype=np.int64)
        for cap in (FakeCapture(self.frames, pts_error=.01), FakeCapture(self.frames, corrupt_at=7)):
            with self.assertRaises(ValueError):
                extract_teacher(cap, indices, self.boxes, 30)

    def test_teacher_rejects_sparse_or_reordered_indices(self):
        for selected in ([2, 4], [4, 3], []):
            with self.assertRaises(ValueError):
                extract_teacher(FakeCapture(self.frames), np.asarray(selected, dtype=np.int64), self.boxes, 30)

    def test_saved_payload_roundtrip_and_frame_mapping_tamper_detection(self):
        start, end = .3, .7
        interval = interval_contract(start, end, 30, 60)
        student, trace_s = legacy.extract(FakeCapture(self.frames), start, end, self.boxes, 30)
        teacher, trace_t = extract_teacher(FakeCapture(self.frames), interval["teacher_indices"], self.boxes, 30)
        row = {"id": "synthetic", "start_time": start, "end_time": end, "video_fps": 30,
               "frames": len(student), "token_ids": [2, 3]}
        meta = {"id": row["id"], "record": row, "plan_sha256": "synthetic-plan", "source_frame_count": 60,
                "student_rgb_sha256": hashlib.sha256(student.tobytes()).hexdigest(),
                "teacher_bgr_sha256": hashlib.sha256(teacher.tobytes()).hexdigest(),
                "student_trace": trace_s, "teacher_trace": trace_t}
        arrays = {"student_rgb": student, "teacher_bgr": teacher, "student_source_indices": interval["student_indices"],
                  "teacher_source_indices": interval["teacher_indices"], "student_teacher_rows": interval["student_teacher_rows"],
                  "label_tokens": np.array([2, 3], dtype=np.int64), "metadata_json": json.dumps(meta)}
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "good.npz", Path(tmp) / "bad.npz"
            np.savez_compressed(good, **arrays)
            self.assertEqual(verify_payload(good, row, "synthetic-plan"), meta)
            arrays["student_teacher_rows"] = arrays["student_teacher_rows"].copy()
            arrays["student_teacher_rows"][0] += 1
            np.savez_compressed(bad, **arrays)
            with self.assertRaises(ValueError):
                verify_payload(bad, row, "synthetic-plan")


if __name__ == "__main__":
    unittest.main()
