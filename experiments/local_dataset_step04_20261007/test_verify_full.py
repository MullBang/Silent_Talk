"""Fault-injection tests of the independent D: pilot input auditor."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("step02_independent_auditor_tests", HERE / "verify_full.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


class TrainInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="audit-test-", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.inputs = Path(self.temp.name)
        (self.inputs / "records").mkdir()
        (self.inputs / "metadata").mkdir()
        self.row = {
            "id": "synthetic_001", "source_stem": "synthetic", "speaker_id": "speaker",
            "split": "train", "start_time": 1.0, "end_time": 2.0,
            "original_start_time": 1.0, "original_end_time": 2.0, "offset_seconds": 0.0,
            "video_fps": 25.0, "frames": 25, "source_text": "가.", "text": "가",
            "token_ids": [1, 20], "video_size_bytes": 1000, "label_sha256": "a" * 64,
        }
        self.source = {"stem": "synthetic", "members": {
            "video": {"size": 1000, "sha256": "b" * 64, "member_name": "TS1/synthetic.mp4"},
            "label": {"size": 500, "sha256": "a" * 64, "member_name": "TL1/synthetic.json"},
            "audio": {"size": 100, "sha256": "c" * 64, "member_name": "TS1/synthetic.wav"},
        }}
        self.transport = [{"path": "binding", "sha256": "d" * 64}]
        self.runner = {"path": str(HERE / "prepare_full.py"), "sha256": "e" * 64}
        provenance = {
            "transport": "audited-local-tar-v1", "source_plan_path": str(audit.SOURCE_PLAN),
            "source_plan_sha256": audit.SOURCE_SHA, "parent_plan_sha256": audit.PARENT_SHA,
            "source_stem": "synthetic", "members": {k: copy.deepcopy(self.source["members"][k]) for k in ("video", "label")},
            "temporary_paths_persisted": False, "archive_stats_checked_before_and_after": True,
            "member_hashes_checked_on_materialization": True, "audio_used_as_model_input": False,
            "transport_code": self.transport,
            "temporary_payload_verification": {"video_size_bytes": 1000, "video_mtime_ns": 123456,
                "label_sha256_before_and_after": "a" * 64, "source_video_unchanged": True},
        }
        indices = np.arange(25, 50, dtype=np.int64)
        dense = np.arange(23, 52, dtype=np.int64)
        mapping = indices - 23
        context = {
            "requested_real_frames_each_side": 2, "official_first": 25, "official_stop_exclusive": 50,
            "student_union_first": 25, "student_union_stop_exclusive": 50,
            "actual_first": 23, "actual_stop_exclusive": 52, "clipped_left": False, "clipped_right": False,
            "student_left_real_context": [2] * 25, "student_right_real_context": [2] * 25,
            "boundary_policy": "clip to available native frames; teacher front retains its own configured zero padding",
        }
        def trace(index):
            return {"source_frame_index": int(index), "annotation_box_index": int(index),
                    "actual_pts_seconds": int(index) / 25, "annotation_box_xyxy": [10., 20., 100., 110.],
                    "source_bgr_pixel_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
        student_trace = [{**trace(i), "target_video_seconds": float(i) / 25} for i in indices]
        teacher_trace = [{**trace(i), "final_rectangle_xyxy": [10, 20, 100, 110]} for i in dense]
        self.arrays = {
            "student_rgb": np.zeros((25, 96, 96, 3), np.uint8),
            "teacher_bgr": np.ones((29, 96, 96, 3), np.uint8),
            "student_source_indices": indices, "teacher_source_indices": dense,
            "student_teacher_rows": mapping, "label_tokens": np.array([1, 20], np.int64),
        }
        self.meta = {"record": copy.deepcopy(self.row), "plan_sha256": audit.PARENT_SHA,
                     "id": self.row["id"], "local_full_preparation": self.runner,
                     "source_frame_count": 100, "teacher_context": context,
                     "student_trace": {"selected": student_trace}, "teacher_trace": {"selected": teacher_trace},
                     "source_verification": provenance}
        self.entry = {"id": self.row["id"], "split": "train", "source_stem": "synthetic",
                      "student_frames": 25, "teacher_frames": 29, "video_fps": 25.0,
                      "source_frame_count": 100, "plan_sha256": audit.PARENT_SHA,
                      "teacher_context": copy.deepcopy(context), "source_verification": copy.deepcopy(provenance)}

    def write(self):
        for key in ("student_rgb", "teacher_bgr"):
            self.meta[key + "_sha256"] = digest(self.arrays[key])
            self.entry[key + "_sha256"] = digest(self.arrays[key])
        path = self.inputs / "records" / (self.row["id"] + ".npz")
        np.savez_compressed(path, **self.arrays, metadata_json=json.dumps(self.meta))
        self.entry["npz_path"] = str(path)
        self.entry["npz_sha256"] = audit.sha(path)
        (self.inputs / "metadata" / (self.row["id"] + ".json")).write_text(json.dumps(self.entry), encoding="utf-8")

    def verify(self):
        return audit.verify_train_record(self.row, self.entry, audit.PARENT_SHA, self.inputs,
            self.source, audit.SOURCE_PLAN, audit.SOURCE_SHA, self.transport, self.runner)

    def expect_rejection(self, pattern=None):
        self.write()
        if pattern:
            with self.assertRaisesRegex((ValueError, AssertionError), pattern):
                self.verify()
        else:
            with self.assertRaises((ValueError, AssertionError)):
                self.verify()

    def test_valid_exact_frame_identity_and_tokens(self):
        self.write()
        result = self.verify()
        self.assertEqual(result["student_frames"], 25)
        self.assertEqual(result["teacher_frames"], 29)

    def test_frame_shift_rejected_even_when_npz_hash_rebound(self):
        self.arrays["student_source_indices"][10] += 1
        self.expect_rejection()

    def test_sparse_teacher_sequence_rejected(self):
        self.arrays["teacher_source_indices"][10] += 1
        self.expect_rejection()

    def test_ratio_based_teacher_mapping_rejected(self):
        self.arrays["student_teacher_rows"] -= 1
        self.expect_rejection()

    def test_changed_tokens_rejected_independently_of_hash(self):
        self.arrays["label_tokens"][0] = 2
        self.expect_rejection()

    def test_changed_frozen_logical_record_rejected(self):
        self.meta["record"]["split"] = "dev"
        self.expect_rejection("Payload fixed record")

    def test_wrong_raw_source_frame_hash_rejected(self):
        self.meta["student_trace"]["selected"][4]["source_bgr_pixel_sha256"] = "f" * 64
        self.expect_rejection("Raw source pixel identity")

    def test_changed_frame_context_rejected(self):
        self.meta["teacher_context"]["student_left_real_context"][0] = 0
        self.expect_rejection("Every student context")

    def test_teacher_margin_change_rejected(self):
        self.meta["teacher_trace"]["selected"][0]["final_rectangle_xyxy"][0] = 9
        self.expect_rejection("margin-zero")

    def test_wrong_tar_member_hash_rejected(self):
        self.meta["source_verification"]["members"]["video"]["sha256"] = "f" * 64
        self.entry["source_verification"] = copy.deepcopy(self.meta["source_verification"])
        self.expect_rejection("Source provenance members")

    def test_fake_legacy_provenance_rejected(self):
        self.meta["source_verification"] = {"video_size_bytes": 1000, "source_video_unchanged": True}
        self.entry["source_verification"] = copy.deepcopy(self.meta["source_verification"])
        self.expect_rejection("Transport evidence fields")

    def test_unexpected_audio_materialization_rejected(self):
        self.meta["source_verification"]["members"]["audio"] = self.source["members"]["audio"]
        self.entry["source_verification"] = copy.deepcopy(self.meta["source_verification"])
        self.expect_rejection("Source provenance members")

    def test_missing_payload_hash_check_claim_rejected(self):
        self.meta["source_verification"]["member_hashes_checked_on_materialization"] = False
        self.entry["source_verification"] = copy.deepcopy(self.meta["source_verification"])
        self.expect_rejection("member_hashes_checked")

    def test_runner_change_rejected(self):
        self.meta["local_full_preparation"] = {**self.runner, "sha256": "0" * 64}
        self.expect_rejection("Payload preparation runner")

    def test_changed_npz_bytes_rejected(self):
        self.write()
        path = Path(self.entry["npz_path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "NPZ hash"):
            self.verify()

    def test_outside_workspace_binding_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Auditor must not read"):
            audit.old.verify_binding({"path": "D:/source.tar", "sha256": "a" * 64})

    def test_bound_code_change_rejected(self):
        code = self.inputs / "code.py"
        code.write_text("original", encoding="utf-8")
        binding = {"path": str(code), "sha256": audit.sha(code)}
        code.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Bound local evidence/code hash"):
            audit.old.verify_binding(binding)

    def test_hangul_token_boundaries_and_normalization(self):
        self.assertEqual(audit.tokens_from_syllables("가 각, 힣."), ("가 각 힣", [1, 20, 68, 1, 20, 41, 68, 19, 40, 67]))
        with self.assertRaises(ValueError):
            audit.tokens_from_syllables("가A")


class DevInputAuditTests(unittest.TestCase):
    def setUp(self):
        fixture = TrainInputAuditTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for name in ("temp", "inputs", "row", "source", "transport", "runner", "arrays", "meta", "entry"):
            setattr(self, name, getattr(fixture, name))
        self.row["split"] = "dev"
        self.entry["split"] = "dev"
        self.entry["teacher_frames"] = 0
        self.entry.pop("teacher_context")
        for name in ("teacher_bgr", "teacher_source_indices", "student_teacher_rows"):
            self.arrays.pop(name)
        self.meta.pop("teacher_context")
        self.meta.pop("teacher_trace")
        self.meta.update(schema="local-tar-step04-dev-input-v1", preparation_seconds=.1,
                         record=copy.deepcopy(self.row))
        self.meta["student_trace"].update(native_fps=25.0, target_fps=25,
            source_frames_decoded=25, selected_count=25, selected_source_pixels_sha256="a" * 64,
            selection="round((video_start + k / 25) * actual_fps), k < round(duration * 25)")
        self.source_metadata = {"frame_count": 100, "fps": 25.0, "width": 1920, "height": 1080}

    def write(self):
        self.meta["student_rgb_sha256"] = digest(self.arrays["student_rgb"])
        self.entry["student_rgb_sha256"] = digest(self.arrays["student_rgb"])
        path = self.inputs / "records" / (self.row["id"] + ".npz")
        np.savez_compressed(path, **self.arrays, metadata_json=json.dumps(self.meta))
        self.entry["npz_path"] = str(path)
        self.entry["npz_sha256"] = audit.sha(path)
        (self.inputs / "metadata" / (self.row["id"] + ".json")).write_text(json.dumps(self.entry), encoding="utf-8")

    def verify(self):
        return audit.verify_dev_record(self.row, self.entry, self.inputs, self.source,
            self.transport, self.runner, self.source_metadata)

    def reject(self, pattern=None):
        self.write()
        with self.assertRaisesRegex((ValueError, AssertionError), pattern or ".*"):
            self.verify()

    def test_valid_student_only_dev(self):
        self.write()
        result = self.verify()
        self.assertEqual(result["teacher_frames"], 0)
        self.assertFalse(result["teacher_target_allowed"])

    def test_dev_teacher_array_injection_rejected_with_rebound_hash(self):
        self.arrays["teacher_bgr"] = np.ones((25, 96, 96, 3), np.uint8)
        self.reject("student-only payload keys")

    def test_dev_teacher_sidecar_injection_rejected(self):
        self.entry["teacher_bgr_sha256"] = "a" * 64
        self.reject("contains teacher targets")

    def test_dev_teacher_metadata_injection_rejected(self):
        self.meta["teacher_context"] = {}
        self.reject("metadata fields")

    def test_dev_frame_index_shift_rejected_with_rebound_hash(self):
        self.arrays["student_source_indices"][3] += 1
        self.reject()

    def test_dev_target_token_change_rejected_with_rebound_hash(self):
        self.arrays["label_tokens"][0] = 2
        self.reject()

    def test_dev_source_bound_change_rejected(self):
        self.meta["source_frame_count"] = 99
        self.reject("Audited physical source frame count")

    def test_dev_source_interval_outside_bound_rejected(self):
        self.source_metadata["frame_count"] = 49
        self.reject("interval outside audited source bounds")

    def test_dev_trace_bounds_rejected(self):
        self.meta["student_trace"]["selected"][0]["annotation_box_xyxy"][2] = 1921
        self.reject("outside physical image bounds")

    def test_dev_raw_frame_digest_rejected(self):
        self.meta["student_trace"]["selected"][0]["source_bgr_pixel_sha256"] = "g" * 64
        self.reject("Invalid source-pixel digest")

    def test_dev_incorrect_time_trace_rejected(self):
        self.meta["student_trace"]["selected"][0]["target_video_seconds"] += .04
        self.reject("Trace target time")

    def test_dev_incorrect_source_binding_rejected(self):
        self.meta["source_verification"]["members"]["video"]["sha256"] = "f" * 64
        self.entry["source_verification"] = copy.deepcopy(self.meta["source_verification"])
        self.reject("Source provenance members")

    def test_dev_pre_normalized_pixels_rejected(self):
        self.arrays["student_rgb"] = self.arrays["student_rgb"].astype(np.float32)
        self.reject("RGB uint8 storage")


class FullCohortTests(unittest.TestCase):
    def setUp(self):
        self.plan = audit.read_json(audit.PARENT, audit.PARENT_SHA)
        self.pilot = audit.read_json(audit.PILOT_BASE / "inputs/pilot_manifest.json", audit.PILOT_SHA)
        pilot_by_id = {r["id"]: r for r in self.pilot["records"]}
        self.manifest = {"ids": [r["id"] for r in self.plan["records"]], "record_count": 512, "source_recordings": 59,
            "records": [copy.deepcopy(pilot_by_id[r["id"]]) if r["id"] in pilot_by_id else
                        {k:r[k] for k in ("id", "split", "source_stem")} for r in self.plan["records"]]}

    def test_exact_frozen_cohort_and_real_pilot_preservation(self):
        audit.verify_cohort(self.plan, self.manifest, self.pilot)

    def test_shifted_record_order_rejected(self):
        self.manifest["records"][0], self.manifest["records"][1] = self.manifest["records"][1], self.manifest["records"][0]
        with self.assertRaisesRegex(ValueError, "exact record coverage"):
            audit.verify_cohort(self.plan, self.manifest, self.pilot)

    def test_split_change_rejected(self):
        self.manifest["records"][0]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Frozen split"):
            audit.verify_cohort(self.plan, self.manifest, self.pilot)

    def test_reused_pilot_copy_path_change_rejected(self):
        row = next(r for r in self.manifest["records"] if r["id"] == self.pilot["ids"][0])
        row["npz_path"] = str(HERE / "copied.npz")
        with self.assertRaisesRegex(ValueError, "Original8 entry preserved"):
            audit.verify_cohort(self.plan, self.manifest, self.pilot)


if __name__ == "__main__":
    unittest.main()

