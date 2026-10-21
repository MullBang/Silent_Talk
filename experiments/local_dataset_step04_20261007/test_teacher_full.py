"""Scope, immutable reuse and recovery checks without a GPU or checkpoint."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import teacher_full as full


class FullCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((full.PARENT / "data_plan.json").read_text(encoding="utf-8"))
        cls.pilot_inputs = json.loads((full.PILOT / "inputs/pilot_manifest.json").read_text(encoding="utf-8"))
        cls.pilot_features = json.loads((full.PILOT / "teacher_results/pilot_manifest.json").read_text(encoding="utf-8"))
        cls.recipe = json.loads(full.RECIPE.read_text(encoding="utf-8"))
        cls.source = {"records": cls.data["records"], "parent_plan": {"sha256": full.pilot.EXPECTED_PARENT_SHA}}
        prior = {r["id"]: r for r in cls.pilot_inputs["records"]}
        records = [prior.get(row["id"], {"id": row["id"], "split": row["split"],
                   "source_stem": row["source_stem"], "student_frames": row["frames"],
                   "teacher_frames": 0 if row["split"] == "dev" else 150,
                   "video_fps": row["video_fps"], "plan_sha256": full.pilot.EXPECTED_PARENT_SHA})
                   for row in cls.data["records"]]
        cls.manifest = {"schema": "feature-transfer-step07-input-manifest-v1", "status": "passed",
                        "phase": "all", "plan_sha256": full.pilot.EXPECTED_PARENT_SHA,
                        "ids": [r["id"] for r in records], "records": records, "record_count": 512}

    def select(self, manifest=None, features=None, recipe=None):
        return full.validate_cohort(self.data, self.source, manifest or self.manifest,
                                    self.pilot_inputs, features or self.pilot_features, recipe or self.recipe)

    def test_selects_exact_parent_order_and_433_new_training_only(self):
        selected = self.select()
        self.assertEqual(len(selected["train_ids"]), 441)
        self.assertEqual(len(selected["new_ids"]), 433)
        self.assertEqual(len(selected["dev_ids"]), 71)
        self.assertFalse(set(selected["train_ids"]) & set(selected["dev_ids"]))
        self.assertEqual(set(selected["new_ids"]) | set(selected["pilot_ids"]), set(selected["train_ids"]))
        self.assertEqual(selected["train_ids"], self.recipe["train_ids"])

    def test_refuses_changed_reordered_or_duplicated_cohort(self):
        changes = []
        reordered = copy.deepcopy(self.manifest)
        reordered["records"].reverse()
        changes.append(reordered)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["records"][-1] = duplicate["records"][0]
        changes.append(duplicate)
        split = copy.deepcopy(self.manifest)
        split["records"][0]["split"] = "test"
        changes.append(split)
        for changed in changes:
            with self.assertRaises(ValueError):
                self.select(manifest=changed)

    def test_pilot_record_dictionary_and_feature_provenance_are_immutable(self):
        changed = copy.deepcopy(self.manifest)
        sid = self.data["preparation"]["pilot_ids"][0]
        next(r for r in changed["records"] if r["id"] == sid)["npz_path"] += ".copy"
        with self.assertRaises(ValueError):
            self.select(manifest=changed)
        changed_features = copy.deepcopy(self.pilot_features)
        changed_features["records"][0]["teacher_plan_sha256"] = "new-plan-is-not-old-provenance"
        with self.assertRaises(ValueError):
            self.select(features=changed_features)

    def test_dev_has_no_teacher_frames_and_recipe_cannot_change_split(self):
        changed = copy.deepcopy(self.manifest)
        next(r for r in changed["records"] if r["split"] == "dev")["teacher_frames"] = 12
        with self.assertRaises(ValueError):
            self.select(manifest=changed)
        recipe = copy.deepcopy(self.recipe)
        recipe["train_ids"].reverse()
        with self.assertRaises(ValueError):
            self.select(recipe=recipe)

    def test_dev_refused_before_any_input_or_feature_file_read(self):
        row = next(r for r in self.data["records"] if r["split"] == "dev")
        item = {"id": row["id"], "split": "dev", "npz_path": "must never open"}
        with patch.object(full.teacher, "read_prepared_npz", side_effect=AssertionError("read dev")) as read:
            with self.assertRaises(ValueError):
                full.read_train_input(row, item, {}, {})
            read.assert_not_called()
        with patch.object(full.np, "load", side_effect=AssertionError("read dev")) as read:
            with self.assertRaises(ValueError):
                full.verify_feature(Path("dev.npz"), Path("dev.json"), "plan", item, row, "state")
            read.assert_not_called()

    def test_full_audit_binds_scope_and_train_dev_permission(self):
        rows = {r["id"]: r for r in self.data["records"]}
        bindings = {"data_plan": {"sha256": "data"}, "source_plan": {"sha256": "source"},
                    "input_manifest": {"sha256": "inputs"}}
        with tempfile.TemporaryDirectory() as directory:
            code = Path(directory) / "auditor.py"
            code.write_text("# test", encoding="utf-8")
            audit = {"schema": "local-tar-step04-independent-input-audit-v1", "status": "passed",
                     "scope": "all512", "record_count": 512, "plan": bindings["data_plan"],
                     "source_plan": bindings["source_plan"], "manifest": bindings["input_manifest"],
                     "code": full.teacher.binding(code), "records": [
                         {"id": r["id"], "teacher_target_allowed": r["split"] == "train",
                          "teacher_frames": 100 if r["split"] == "train" else 0} for r in self.data["records"]]}
            full.validate_audit(audit, bindings, self.manifest["ids"], rows)
            for key, value in [("status", "failed"), ("scope", "pilot8"), ("record_count", 511)]:
                changed = copy.deepcopy(audit)
                changed[key] = value
                with self.assertRaises(ValueError):
                    full.validate_audit(changed, bindings, self.manifest["ids"], rows)
            changed = copy.deepcopy(audit)
            next(r for r in changed["records"] if not r["teacher_target_allowed"])["teacher_target_allowed"] = True
            with self.assertRaises(ValueError):
                full.validate_audit(changed, bindings, self.manifest["ids"], rows)


class ImmutableFeatureTests(unittest.TestCase):
    def fixture(self, directory):
        path, sidecar = Path(directory) / "train.npz", Path(directory) / "train.json"
        row = {"id": "train", "split": "train", "frames": 3}
        item = {"id": "train", "split": "train"}
        rng = np.random.default_rng(9)
        features = rng.normal(size=(3, 512)).astype(np.float32)
        arrays = {"features": features, "student_source_indices": np.array([2, 3, 5], dtype=np.int64),
                  "teacher_source_indices": np.arange(7, dtype=np.int64), "gather_indices": np.array([2, 3, 5], dtype=np.int64)}
        meta = {"id": "train", "split": "train", "teacher_plan_sha256": "plan", "input_record": item,
                "data_record_sha256": full.teacher.canonical_sha(row), "state_sha256_before": "state",
                "state_sha256_after": "state", "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                "feature_diagnostics": full.pilot.feature_diagnostics(features), "audio_used": False,
                "decoder_instantiated": False, "transcript_used_for_teacher_forward": False,
                "optimization_steps": 0, "trainable_parameters": 0}
        full.teacher.save_new_npz(path, **arrays, metadata_json=np.array(json.dumps(meta)))
        return path, sidecar, row, item, arrays

    def test_recovers_missing_sidecar_and_revalidates_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sidecar, row, item, arrays = self.fixture(directory)
            first = full.verify_feature(path, sidecar, "plan", item, row, "state", arrays=arrays)
            self.assertTrue(sidecar.is_file())
            second = full.verify_feature(path, sidecar, "plan", item, row, "state", expected=first, arrays=arrays)
            self.assertEqual(first, second)
            self.assertEqual(first["feature_file"], full.teacher.binding(path))

    def test_rejects_resume_from_other_plan_or_state_or_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sidecar, row, item, arrays = self.fixture(directory)
            for plan, state in [("wrong", "state"), ("plan", "wrong")]:
                with self.assertRaises(ValueError):
                    full.verify_feature(path, sidecar, plan, item, row, state)
            changed = {**arrays, "student_source_indices": np.array([1, 3, 5], dtype=np.int64)}
            with self.assertRaises(ValueError):
                full.verify_feature(path, sidecar, "plan", item, row, "state", arrays=changed)
            self.assertFalse(sidecar.exists())

    def test_immutable_manifest_refuses_changed_feature_or_absent_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sidecar, row, item, arrays = self.fixture(directory)
            expected = full.verify_feature(path, sidecar, "plan", item, row, "state")
            changed = copy.deepcopy(expected)
            changed["feature_file"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                full.verify_feature(path, sidecar, "plan", item, row, "state", expected=changed)
            sidecar.unlink()
            with self.assertRaises(ValueError):
                full.verify_feature(path, sidecar, "plan", item, row, "state", expected=expected)
            self.assertFalse(sidecar.exists())

    def test_no_dev_or_pilot_artifacts_allowed_in_new_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "new_train.npz").write_bytes(b"fixture")
            full.validate_output_inventory(target, ["new_train"])
            (target / "dev.npz").write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                full.validate_output_inventory(target, ["new_train"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
