"""Read-only scope and corruption tests for the mixed cached artifact loader."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import mixed_artifacts as mod


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = mod.verify_handoff()

    def test_metadata_api_never_opens_npz(self):
        with patch.object(np, "load", side_effect=AssertionError("NPZ not permitted")):
            metadata = mod.evaluation_metadata()
        self.assertEqual(len(metadata["old71_ids"]), 71)
        self.assertEqual(len(metadata["new24_ids"]), 24)
        self.assertEqual(len(metadata["new23_ids"]), 23)
        self.assertEqual(metadata["excluded_duplicate_ids"], ["lip_J_1_F_05_C059_A_002_036"])

    def test_frozen_roles_and_spacing_only_overlap(self):
        groups = self.bundle["groups"]
        self.assertEqual(len(groups["expanded510"]), 510)
        self.assertEqual(len(groups["held27"]), 27)
        self.assertFalse(set(groups["expanded510"]) & set(groups["old71"] + groups["new24"] + groups["held27"]))
        overlap = self.bundle["overlap_evidence"][0]
        self.assertNotEqual(overlap["train_text"], overlap["val_text"])
        self.assertEqual(overlap["train_text"].replace(" ", ""), overlap["val_text"].replace(" ", ""))

    def test_held_or_custom_training_scope_blocked_before_any_loader(self):
        with patch.object(mod, "old_loader", side_effect=AssertionError("Must reject before file access")):
            for bad in ("held27", "new96", "new69", "new24", "train", "", None):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    mod.load_training(bad)

    def test_filtered_dev_or_training_request_blocked_before_loader(self):
        with patch.object(mod, "old_loader", side_effect=AssertionError("Must reject before file access")):
            for bad in ("new23", "held27", "expanded510", "dev", "", None):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    mod.load_development(bad)

    def test_unknown_new_id_subset_and_held_id_rejected_before_npz(self):
        with patch.object(np, "load", side_effect=AssertionError("Must not open forbidden payload")):
            for ids in ([self.bundle["groups"]["held27"][0]], self.bundle["groups"]["new24"][:-1], ["unknown"]):
                with self.subTest(ids=ids[:1]), self.assertRaises(ValueError):
                    mod._load_new(self.bundle, ids, "dev", None)

    def test_role_or_record_hash_tampering_rejected(self):
        for key, value in (("role", "new_provisional_train"), ("logical_record_sha256", "0" * 64),
                           ("new_human_approval_added", True)):
            handoff = copy.deepcopy(self.bundle["handoff"])
            handoff["records"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                mod.validate_handoff(handoff, self.bundle["documents"])

    def test_duplicate_handoff_record_rejected(self):
        handoff = copy.deepcopy(self.bundle["handoff"])
        handoff["records"][1] = copy.deepcopy(handoff["records"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            mod.validate_handoff(handoff, self.bundle["documents"])

    def test_extra_feature_record_rejected(self):
        docs = copy.deepcopy(self.bundle["documents"])
        injected = dict(docs["new_features"]["records"][0], id="unexpected")
        docs["new_features"]["records"].append(injected)
        with self.assertRaisesRegex(ValueError, "extra feature"):
            mod.validate_handoff(self.bundle["handoff"], docs)

    def test_dev_cannot_become_teacher_candidate(self):
        docs = copy.deepcopy(self.bundle["documents"])
        next(r for r in docs["new_data_plan"]["records"] if r["split"] == "val")["teacher_candidate"] = True
        with self.assertRaisesRegex(ValueError, "Development"):
            mod.validate_handoff(self.bundle["handoff"], docs)

    def test_workspace_binding_refuses_source_drives(self):
        checked = mod.old_loader().LocalBindings()
        for name in ("D:/source.tar", "G:/video.mp4"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                checked.path(name)


class NewPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = mod.verify_handoff()
        sid = next(s for s in cls.bundle["groups"]["expanded510"] if cls.bundle["entries"][s]["generation"] == "new")
        cls.row = cls.bundle["rows"][sid]
        cls.item = cls.bundle["inputs"]["new"][sid]
        cls.feature = cls.bundle["features"]["new"][sid]
        with np.load(cls.item["npz_path"], allow_pickle=False) as saved:
            cls.arrays = {key: saved[key] for key in mod.PAIRED_KEYS - {"metadata_json", "teacher_bgr"}}
            cls.meta = json.loads(str(saved["metadata_json"].item()))
        with np.load(cls.feature["feature_file"]["path"], allow_pickle=False) as saved:
            cls.target = {key: saved[key] for key in mod.FEATURE_KEYS - {"metadata_json"}}
            cls.feature_meta = json.loads(str(saved["metadata_json"].item()))

    def student(self, arrays=None, meta=None, row=None, item=None, scope="train"):
        mod.validate_new_student(row or self.row, item or self.item, arrays or self.arrays,
                                 meta or self.meta, self.bundle, scope)

    def teacher(self, target=None, meta=None, feature=None, arrays=None, row=None):
        mod.validate_new_target(row or self.row, self.item, feature or self.feature, arrays or self.arrays,
                                target or self.target, meta or self.feature_meta, self.bundle)

    def test_actual_new_candidate_schema_compatible(self):
        self.student()
        self.teacher()

    def test_student_pixel_mutation_rejected(self):
        arrays = dict(self.arrays, student_rgb=self.arrays["student_rgb"].copy())
        arrays["student_rgb"].flat[0] ^= 1
        with self.assertRaisesRegex(ValueError, "pixels changed"):
            self.student(arrays=arrays)

    def test_student_dtype_shape_mutations_rejected(self):
        for pixels in (self.arrays["student_rgb"].astype(np.float32), self.arrays["student_rgb"][:-1]):
            with self.subTest(shape=pixels.shape), self.assertRaisesRegex(ValueError, "Invalid student"):
                self.student(arrays=dict(self.arrays, student_rgb=pixels))

    def test_canonical_tokens_and_frame_indices_rejected(self):
        for key in ("label_tokens", "student_source_indices"):
            arrays = dict(self.arrays)
            arrays[key] = arrays[key].copy()
            arrays[key].flat[0] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frame/token"):
                self.student(arrays=arrays)

    def test_wrong_source_split_or_held_candidate_rejected(self):
        for row in (dict(self.row, split="val"), dict(self.row, teacher_candidate=False)):
            with self.assertRaisesRegex(ValueError, "held role"):
                self.student(row=row)
            with self.assertRaisesRegex(ValueError, "forbidden"):
                self.teacher(row=row)

    def test_upstream_plan_or_approval_tampering_rejected(self):
        for key, value in (("plan_sha256", "0" * 64), ("training_ready", True),
                           ("full_transcript_human_approved", True), ("teacher_model_run", True)):
            meta = copy.deepcopy(self.meta)
            meta[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.student(meta=meta)

    def test_nan_zero_constant_or_wrong_feature_shape_rejected(self):
        bad_values = [np.full_like(self.target["features"], np.nan), np.zeros_like(self.target["features"]),
                      np.ones_like(self.target["features"]), self.target["features"][:-1]]
        for value in bad_values:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                self.teacher(target=dict(self.target, features=value))

    def test_feature_value_mutation_rejected(self):
        target = dict(self.target, features=self.target["features"].copy())
        target["features"].flat[0] += np.float32(1.)
        with self.assertRaisesRegex(ValueError, "features changed"):
            self.teacher(target=target)

    def test_gather_off_by_one_and_float_indices_rejected(self):
        target = dict(self.target, gather_indices=self.target["gather_indices"].copy())
        target["gather_indices"][0] += 1
        with self.assertRaisesRegex(ValueError, "mapping changed"):
            self.teacher(target=target)
        with self.assertRaises(ValueError):
            self.teacher(target=dict(self.target, student_source_indices=self.target["student_source_indices"].astype(np.float64)))

    def test_teacher_state_audio_or_transcript_usage_rejected(self):
        for key, value in (("state_sha256_before", "0" * 64), ("audio_used", True),
                           ("transcript_used_for_teacher_forward", True), ("optimization_steps", 1)):
            meta = copy.deepcopy(self.feature_meta)
            meta[key] = value
            feature = {**meta, "feature_file": self.feature["feature_file"], "status": "completed"}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.teacher(meta=meta, feature=feature)

    def test_new24_development_reads_no_teacher_or_held_npz(self):
        opened = []
        original = np.load
        expected = {str(mod.STEP9 / "inputs/records" / (sid + ".npz")) for sid in self.bundle["groups"]["new24"]}
        def guarded(path, *args, **kwargs):
            self.assertIn(str(Path(path)), expected)
            opened.append(str(Path(path)))
            return original(path, *args, **kwargs)
        with patch.object(np, "load", side_effect=guarded):
            loaded = mod.load_development("new24")
        self.assertEqual(len(opened), 24)
        self.assertEqual(loaded["ids"], self.bundle["groups"]["new24"])
        self.assertEqual(len(loaded["supplemental_ids"]), 23)
        self.assertTrue(all(c["split"] == "dev" and c["source_split"] == "val" and "teacher_features" not in c
                            for c in loaded["clips"].values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
