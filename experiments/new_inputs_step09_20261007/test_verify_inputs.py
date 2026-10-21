"""Contract rejection tests for the independent auditor."""
import copy
import unittest
import numpy as np
import verify_inputs as audit


def fixture(candidate=True):
    row = {"id": "synthetic", "split": "train" if candidate else "val", "comparison_group": "both_supported",
           "teacher_candidate": candidate, "start_time": 1., "end_time": 1.4, "video_fps": 30.,
           "source_frame_count": 100, "frames": 10, "text": "가", "token_ids": [1, 20], "ctc_min_frames": 2}
    source = {"video_frame_count": 100, "video_fps": 30., "video_metadata": {"width": 12, "height": 12}}
    boxes = np.tile(np.asarray([2., 3., 8., 9.]), (100, 1))
    student, teacher, mapping, context = audit.interval(1., 1.4, 30., 100)
    def trace(indices, is_student):
        values = []
        for k, native in enumerate(indices):
            r = {"source_frame_index": int(native), "annotation_box_index": int(native),
                 "actual_pts_seconds": int(native) / 30., "source_bgr_pixel_sha256": "0" * 64,
                 "annotation_box_xyxy": boxes[int(native)].tolist()}
            if is_student:
                r["target_video_seconds"] = 1. + k / 25
            else:
                r["final_rectangle_xyxy"] = [2, 3, 8, 9]
            values.append(r)
        value = {"selected": values, "selected_source_pixels_sha256": "0" * 64}
        if is_student:
            value.update(native_fps=30., target_fps=25, selected_count=10, source_frames_decoded=12)
        return value
    arrays = {"student_rgb": np.zeros((10, 96, 96, 3), np.uint8), "student_source_indices": student,
              "label_tokens": np.array([1, 20], np.int64)}
    meta = {"record": row, "id": row["id"], "teacher_pixels_persisted": candidate,
            "schema": "lipnet-step09-paired-input-v1" if candidate else "lipnet-step09-student-input-v1",
            "source_frame_count": 100, "student_trace": trace(student, True)}
    for key in ("training_ready", "full_transcript_human_approved", "human_boundary_approved", "audio_used_as_model_input", "teacher_model_run"):
        meta[key] = False
    entry = {"teacher_frames": len(teacher) if candidate else 0}
    if candidate:
        arrays.update(teacher_bgr=np.zeros((len(teacher), 96, 96, 3), np.uint8), teacher_source_indices=teacher, student_teacher_rows=mapping)
        meta.update(teacher_trace=trace(teacher, False), teacher_context=context)
    for key in ("student_rgb", "teacher_bgr") if candidate else ("student_rgb",):
        meta[key + "_sha256"] = audit.pixel_sha(arrays[key])
        entry[key + "_sha256"] = meta[key + "_sha256"]
    return arrays, meta, row, entry, boxes, source


class AuditorTests(unittest.TestCase):
    def test_independent_hangul_arithmetic_and_punctuation(self):
        self.assertEqual(audit.tokens("가, 각"), ("가 각", [1, 20, 68, 1, 20, 41]))

    def test_unsupported_number_rejected(self):
        with self.assertRaises(ValueError):
            audit.tokens("30분")

    def test_ctc_adjacent_repeat_cost(self):
        self.assertEqual(audit.ctc_min([1, 1, 2, 2, 2]), 8)

    def test_native_interval_rounding_real_context(self):
        s, t, m, c = audit.interval(1, 1.4, 30, 100)
        self.assertEqual(s.tolist(), [30, 31, 32, 34, 35, 36, 37, 38, 40, 41])
        self.assertEqual(t.tolist(), list(range(28, 44)))
        self.assertEqual(t[m].tolist(), s.tolist())
        self.assertFalse(c["clipped_left"])

    def test_interval_clips_real_context_at_source_start(self):
        _, t, _, c = audit.interval(0, .4, 30, 100)
        self.assertEqual(t[0], 0)
        self.assertTrue(c["clipped_left"])

    def test_interval_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            audit.interval(float("nan"), 1, 30, 100)

    def test_interval_source_overrun_rejected(self):
        with self.assertRaises(ValueError):
            audit.interval(3, 4, 30, 100)

    def test_student_margin_uses_floor_ceil_and_clamp(self):
        self.assertEqual(audit.rectangle([2, 3, 8, 9], 12, 12, .2), [0, 1, 10, 11])
        self.assertEqual(audit.rectangle([0, 0, 2, 2], 12, 12, .2), [0, 0, 3, 3])

    def test_teacher_margin_zero_truncates(self):
        self.assertEqual(audit.rectangle([2.9, 3.9, 8.9, 9.9], 12, 12, 0), [2, 3, 8, 9])

    def test_bad_box_rejected(self):
        with self.assertRaises(ValueError):
            audit.rectangle([-1, 0, 2, 2], 12, 12, .2)

    def test_paired_arrays_pass(self):
        audit.verify_arrays(*fixture())

    def test_student_only_arrays_pass(self):
        audit.verify_arrays(*fixture(False))

    def test_development_teacher_arrays_rejected(self):
        args = fixture(False)
        args[0]["teacher_bgr"] = np.zeros((1, 96, 96, 3), np.uint8)
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_development_promoted_to_candidate_rejected(self):
        args = fixture()
        args[2]["split"] = "val"
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_bad_mapping_rejected(self):
        args = fixture()
        args[0]["student_teacher_rows"][0] += 1
        with self.assertRaises(AssertionError):
            audit.verify_arrays(*args)

    def test_frame_tamper_rejected(self):
        args = fixture()
        args[0]["student_rgb"][0, 0, 0, 0] = 7
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_token_tamper_rejected(self):
        args = fixture()
        args[0]["label_tokens"][1] = 21
        with self.assertRaises(AssertionError):
            audit.verify_arrays(*args)

    def test_trace_bbox_tamper_rejected(self):
        args = fixture()
        args[1]["student_trace"]["selected"][0]["annotation_box_xyxy"][0] += 1
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_paired_raw_hash_tamper_rejected(self):
        args = fixture()
        args[1]["teacher_trace"]["selected"][2]["source_bgr_pixel_sha256"] = "1" * 64
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_pts_drift_rejected(self):
        args = fixture()
        args[1]["student_trace"]["selected"][0]["actual_pts_seconds"] += .03
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_approval_escalation_rejected(self):
        args = fixture()
        args[1]["full_transcript_human_approved"] = True
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_held_teacher_metadata_rejected(self):
        args = fixture(False)
        args[1]["teacher_bgr_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            audit.verify_arrays(*args)

    def test_source_direct_recrop_rgb_order(self):
        frame = np.zeros((12, 12, 3), np.uint8)
        frame[:, :, 0] = 1; frame[:, :, 1] = 2; frame[:, :, 2] = 3
        self.assertEqual(audit.recrop(frame, [2, 3, 8, 9], True)[0, 0].tolist(), [3, 2, 1])
        self.assertEqual(audit.recrop(frame, [2, 3, 8, 9], False)[0, 0].tolist(), [1, 2, 3])

    def test_actual_frozen_selection_and_overlap(self):
        result = audit.selection.audit()
        self.assertEqual(result["candidate_train"]["records"], 69)
        self.assertEqual(result["held_train"]["records"], 27)
        self.assertEqual(result["development_all"]["records"], 24)
        self.assertEqual(len(result["candidate_vs_dev_text_overlap"]), 1)


if __name__ == "__main__":
    unittest.main()
