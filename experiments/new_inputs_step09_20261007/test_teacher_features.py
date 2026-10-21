"""Synthetic contract checks only; no checkpoint construction or GPU forward."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import teacher_features as target


DATA_REF = {"path": "synthetic_data.json", "sha256": "a" * 64}
PLAN_REF = {"path": "synthetic_teacher_plan.json", "sha256": "b" * 64}
INPUT_EXEC = {"path": "synthetic_input_plan.json", "sha256": "c" * 64}
STATE = "d" * 64


def cohort_data():
    rows, inputs = [], []
    for i in range(120):
        sid, candidate, split = f"synthetic_{i:03d}", i < 69, "train" if i < 96 else "val"
        row = {"id": sid, "split": split, "speaker_id": f"speaker_{i // 4}", "source_stem": f"source_{i // 4}",
               "frames": 4, "teacher_candidate": candidate, "comparison_group": "both_supported" if candidate or split == "val" else "neither_supported",
               "training_ready": False}
        item = {**{key: row[key] for key in ("id", "split", "speaker_id", "source_stem", "teacher_candidate")},
                "plan_sha256": DATA_REF["sha256"], "npz_path": str(target.BASE / "inputs" / (sid + ".npz")),
                "npz_sha256": "e" * 64, "student_frames": 4, "teacher_frames": 6 if candidate else 0,
                "student_rgb_sha256": "f" * 64, "training_ready": False}
        if candidate:
            item["teacher_bgr_sha256"] = "1" * 64
        rows.append(row)
        inputs.append(item)
    data = {"schema": "lipnet-step09-provisional-data-plan-v1", "training_ready": False, "records": rows,
            "ids": [row["id"] for row in rows], "teacher_candidate_ids": [row["id"] for row in rows[:69]],
            "held_train_ids": [row["id"] for row in rows[69:96]], "development_ids": [row["id"] for row in rows[96:]]}
    manifest = {"schema": "lipnet-step09-input-manifest-v1", "status": "passed", "data_plan": DATA_REF,
                "execution_plan": INPUT_EXEC, "ids": data["ids"], "records": inputs}
    return data, manifest


def fixture_arrays():
    native = np.arange(100, 106, dtype=np.int64)
    student = np.array([100, 102, 104, 104], dtype=np.int64)
    return {"teacher_source_indices": native, "student_source_indices": student,
            "gather_indices": np.array([0, 2, 4, 4], dtype=np.int64),
            "teacher_bgr": np.arange(6 * 96 * 96 * 3, dtype=np.uint32).astype(np.uint8).reshape(6, 96, 96, 3)}


def fixture_plan():
    return {"data_plan": DATA_REF, "input_manifest": {"path": "synthetic_manifest", "sha256": "2" * 64},
            "input_execution_plan": INPUT_EXEC, "input_audit": {"path": "synthetic_audit", "sha256": "3" * 64},
            "selection_audit": {"path": "synthetic_selection", "sha256": "4" * 64},
            "checkpoint": {"path": "synthetic_checkpoint", "sha256": "5" * 64}, "runtime": {"dtype": "float32"}}


def write_fixture(base, change_meta=None, change_arrays=None):
    data, manifest = cohort_data()
    cohort = target.validate_cohort(data, manifest, DATA_REF)
    sid, plan, arrays = cohort["selected_ids"][0], fixture_plan(), fixture_arrays()
    features = np.arange(4 * 512, dtype=np.float32).reshape(4, 512) + 1
    _, normalization = target.teacher.normalize_teacher(arrays["teacher_bgr"])
    row, item = cohort["rows"][sid], cohort["inputs"][sid]
    meta = {"schema": "lipnet-step09-teacher-record-v1", "id": sid, "split": "train", "speaker_id": row["speaker_id"],
            "source_stem": row["source_stem"], "teacher_plan": PLAN_REF, **plan, "input_record": item,
            "data_record_sha256": target.teacher.canonical_sha(row), "state_sha256_before": STATE, "state_sha256_after": STATE,
            "audio_used": False, "decoder_instantiated": False, "optimizer_instantiated": False,
            "transcript_used_for_teacher_forward": False, "training_ready": False, "optimization_steps": 0, "trainable_parameters": 0,
            "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(), "feature_diagnostics": target.pilot.feature_diagnostics(features),
            "native_feature_shape": [1, 6, 512], "aligned_feature_shape": [4, 512], "feature_dtype": "float32", "normalization": normalization,
            "warm_repeat": {"bitwise_equal": True, "max_absolute_difference": 0., "same_input_sha256": item["npz_sha256"]}}
    if change_meta:
        change_meta(meta)
    payload = {"features": features, **{key: arrays[key].copy() for key in ("teacher_source_indices", "student_source_indices", "gather_indices")}}
    if change_arrays:
        change_arrays(payload)
    payload["metadata_json"] = np.array(json.dumps(meta))
    fp, rp = base / "features" / (sid + ".npz"), base / "records" / (sid + ".json")
    target.publish(fp, lambda stream: np.savez_compressed(stream, **payload))
    fr = target.teacher.binding(fp)
    target.save_json(rp, {**meta, "feature_file": fr, "status": "completed"})
    rr = target.teacher.binding(rp)
    return cohort, sid, plan, arrays, fr, rr


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        target.torch.set_num_threads(4)

    def test_roles_exact69_and_no_dev_or_held(self):
        cohort = target.validate_cohort(*cohort_data(), DATA_REF)
        self.assertEqual(len(cohort["selected_ids"]), 69)
        self.assertEqual(len(cohort["held_train_ids"]), 27)
        self.assertEqual(len(cohort["development_ids"]), 24)

    def test_candidate_is_train_and_both_supported(self):
        data, manifest = cohort_data()
        data["records"][0]["comparison_group"] = "neither_supported"
        with self.assertRaises(ValueError):
            target.validate_cohort(data, manifest, DATA_REF)

    def test_development_and_held_teacher_arrays_rejected(self):
        for i in (69, 96):
            data, manifest = cohort_data()
            manifest["records"][i]["teacher_frames"] = 6
            with self.assertRaises(ValueError):
                target.validate_cohort(data, manifest, DATA_REF)
            data, manifest = cohort_data()
            manifest["records"][i]["teacher_bgr_sha256"] = "a" * 64
            with self.assertRaises(ValueError):
                target.validate_cohort(data, manifest, DATA_REF)

    def test_role_order_path_and_recording_leakage_rejected(self):
        changes = [lambda d, m: m["records"].reverse(),
                   lambda d, m: m["records"][0].update(npz_path=str(target.BASE.parent / "elsewhere.npz")),
                   lambda d, m: d["records"][96].update(source_stem=d["records"][0]["source_stem"]),
                   lambda d, m: d["teacher_candidate_ids"].reverse()]
        for change in changes:
            data, manifest = cohort_data()
            change(data, manifest)
            with self.assertRaises(ValueError):
                target.validate_cohort(data, manifest, DATA_REF)

    def test_non_candidate_rejected_before_any_npz_open(self):
        cohort = target.validate_cohort(*cohort_data(), DATA_REF)
        with patch.object(target.teacher, "read_prepared_npz", side_effect=AssertionError("NPZ must not open")) as mocked:
            for sid in cohort["held_train_ids"] + cohort["development_ids"]:
                with self.assertRaises(ValueError):
                    target.read_candidate(cohort, sid, DATA_REF, INPUT_EXEC)
            mocked.assert_not_called()

    def test_input_audit_wrong_binding_or_teacher_role_rejected(self):
        cohort = target.validate_cohort(*cohort_data(), DATA_REF)
        refs = fixture_plan()
        records = [{"id": sid, "teacher_candidate": item["teacher_candidate"], "student_frames": 4,
                    "teacher_frames": item["teacher_frames"], "npz": {"path": item["npz_path"], "sha256": item["npz_sha256"]}}
                   for sid, item in cohort["inputs"].items()]
        value = {"schema": "lipnet-step09-independent-input-audit-v1", "status": "passed", "data_plan": DATA_REF,
                 "input_manifest": refs["input_manifest"], "execution_plan": INPUT_EXEC, "ids": cohort["ids"], "records": records,
                 "code": {"path": "fake", "sha256": "a" * 64}}
        with patch.object(target.teacher, "verify_binding"):
            target.validate_audit(value, refs, cohort)
            value["records"][96]["teacher_candidate"] = True
            with self.assertRaises(ValueError):
                target.validate_audit(value, refs, cohort)

    def test_feature_artifact_roundtrip_exact_gather(self):
        with tempfile.TemporaryDirectory(dir=target.BASE, prefix="synthetic_teacher_") as tmp:
            cohort, sid, plan, arrays, fr, rr = write_fixture(Path(tmp))
            record = target.check_feature(fr, rr, PLAN_REF, cohort, sid, STATE, arrays, plan)
            self.assertEqual(record["status"], "completed")
            self.assertIs(record["training_ready"], False)

    def test_state_plan_normalization_repeat_and_false_claim_rejected(self):
        changes = [lambda m: m.update(state_sha256_after="x" * 64), lambda m: m.update(teacher_plan=DATA_REF),
                   lambda m: m["normalization"].update(std_correction=0), lambda m: m.update(warm_repeat=None),
                   lambda m: m.update(audio_used=True), lambda m: m.update(optimization_steps=1)]
        for change in changes:
            with tempfile.TemporaryDirectory(dir=target.BASE, prefix="synthetic_teacher_") as tmp:
                cohort, sid, plan, arrays, fr, rr = write_fixture(Path(tmp), change_meta=change)
                with self.assertRaises(ValueError):
                    target.check_feature(fr, rr, PLAN_REF, cohort, sid, STATE, arrays, plan)

    def test_feature_nonfinite_shape_and_source_gather_tamper_rejected(self):
        changes = [lambda p: p["features"].__setitem__((0, 0), float("nan")),
                   lambda p: p.update(features=p["features"][:, :511]),
                   lambda p: p["gather_indices"].__setitem__(0, 1),
                   lambda p: p["student_source_indices"].__setitem__(0, 101)]
        for change in changes:
            with tempfile.TemporaryDirectory(dir=target.BASE, prefix="synthetic_teacher_") as tmp:
                cohort, sid, plan, arrays, fr, rr = write_fixture(Path(tmp), change_arrays=change)
                with self.assertRaises(ValueError):
                    target.check_feature(fr, rr, PLAN_REF, cohort, sid, STATE, arrays, plan)

    def test_resume_requires_checksums_and_fixed_prefix(self):
        with tempfile.TemporaryDirectory(dir=target.BASE, prefix="synthetic_teacher_") as tmp:
            base = Path(tmp)
            cohort, sid, plan, arrays, fr, rr = write_fixture(base)
            progress_path = base / "teacher_progress.json"
            with self.assertRaises(ValueError):
                target.load_progress(PLAN_REF, cohort, plan, STATE, progress_path, base / "features", base / "records")
            progress = {"schema": "lipnet-step09-teacher-progress-v1", "teacher_plan": PLAN_REF,
                        "records": [{"id": sid, "feature_file": fr, "record_file": rr}]}
            target.save_json(progress_path, progress)
            with patch.object(target, "read_candidate", return_value=arrays):
                self.assertEqual(len(target.load_progress(PLAN_REF, cohort, plan, STATE, progress_path, base / "features", base / "records")[1]), 1)
                with Path(rr["path"]).open("a", encoding="utf-8") as stream:
                    stream.write(" ")
                with self.assertRaises(ValueError):
                    target.load_progress(PLAN_REF, cohort, plan, STATE, progress_path, base / "features", base / "records")

    def test_immutable_publication_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=target.BASE, prefix="synthetic_teacher_") as tmp:
            path = Path(tmp) / "record.json"
            target.save_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                target.save_json(path, {"a": 2})
            self.assertEqual(target.read(path), {"a": 1})


if __name__ == "__main__":
    unittest.main()
