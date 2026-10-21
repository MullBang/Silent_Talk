"""Fault injection for independent target-value and normalization checks."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test_independent_step04_feature_audit", HERE / "verify_features.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IndependentFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = audit.read_json(audit.PILOT / "teacher_results/pilot_manifest.json", audit.PILOT_FEATURE_SHA)
        cls.plan = audit.read_json(audit.PILOT / "teacher_plan.json", audit.PILOT_PLAN_SHA)
        cls.rows = {r["id"]: r for r in audit.read_json(audit.inputs.PARENT, audit.inputs.PARENT_SHA)["records"]}
        first = cls.manifest["records"][0]
        with np.load(first["feature_file"]["path"], allow_pickle=False) as data:
            cls.features = data["features"]
        with np.load(first["input_record"]["npz_path"], allow_pickle=False) as data:
            cls.bgr = data["teacher_bgr"]

    def setUp(self):
        self.meta = copy.deepcopy(self.manifest["records"][0])

    def numeric(self, features):
        self.meta["feature_values_sha256"] = hashlib.sha256(features.tobytes()).hexdigest()
        return audit.verify_numeric_features(features, self.meta, len(self.features))

    def test_real_original_eight_features_and_mapping(self):
        for entry in self.manifest["records"]:
            result = audit.verify_record(self.rows[entry["id"]], entry, entry["input_record"], self.plan,
                audit.PILOT_PLAN_SHA, self.manifest["state_sha256_before"], True)
            self.assertTrue(result["same_integer_source_frames"])

    def test_nonfinite_targets_rejected_even_with_rebound_hash(self):
        changed = self.features.copy()
        changed[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.numeric(changed)

    def test_zero_row_rejected_even_with_rebound_hash(self):
        changed = self.features.copy()
        changed[0] = 0
        with self.assertRaisesRegex(ValueError, "Zero or near-zero"):
            self.numeric(changed)

    def test_temporal_collapse_rejected_even_with_rebound_hash(self):
        changed = np.repeat(self.features[:1], len(self.features), axis=0)
        with self.assertRaisesRegex(ValueError, "constant temporal"):
            self.numeric(changed)

    def test_changed_values_rejected_by_independent_diagnostics(self):
        changed = self.features.copy()
        changed[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Independent feature diagnostic"):
            self.numeric(changed)

    def test_wrong_normalization_channel_contract_rejected(self):
        self.meta["normalization"]["channel_order"] = "RGB"
        with self.assertRaisesRegex(ValueError, "channel_order"):
            audit.verify_normalization(self.bgr, self.meta)

    def test_wrong_sample_correction_rejected(self):
        self.meta["normalization"]["std_correction"] = 0
        with self.assertRaisesRegex(ValueError, "std_correction"):
            audit.verify_normalization(self.bgr, self.meta)

    def test_perframe_or_wrong_window_mean_rejected(self):
        self.meta["normalization"]["raw_mean"] += .1
        with self.assertRaisesRegex(ValueError, "raw mean"):
            audit.verify_normalization(self.bgr, self.meta)


if __name__ == "__main__":
    unittest.main()
