"""Scope, provenance and noncollapse tests, without CUDA or checkpoint reads."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import teacher_pilot as pilot


class PilotScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((pilot.PARENT / "data_plan.json").read_text(encoding="utf-8"))
        rows = {r["id"]: r for r in cls.data["records"]}
        cls.ids = cls.data["preparation"]["pilot_ids"]
        cls.manifest = {"schema": "feature-transfer-step07-input-manifest-v1", "status": "passed",
                        "phase": "pilot", "plan_sha256": pilot.EXPECTED_PARENT_SHA,
                        "ids": cls.ids, "record_count": 8,
                        "records": [{"id": sid, "split": rows[sid]["split"],
                                     "source_stem": rows[sid]["source_stem"],
                                     "student_frames": rows[sid]["frames"], "video_fps": rows[sid]["video_fps"],
                                     "plan_sha256": pilot.EXPECTED_PARENT_SHA} for sid in cls.ids]}
        cls.source = {"records": cls.data["records"], "parent_plan": {"sha256": pilot.EXPECTED_PARENT_SHA}}

    def test_accepts_only_original_fixed_eight(self):
        result = pilot.validate_pilot(self.data, self.manifest, self.source)
        self.assertEqual(result["ids"], self.ids)
        self.assertEqual(len(result["inputs"]), 8)

    def test_rejects_expansion_duplication_dev_or_other_phase(self):
        mutants = []
        expanded = copy.deepcopy(self.manifest)
        expanded["records"].append(expanded["records"][0])
        expanded["record_count"] = 9
        mutants.append(expanded)
        duplicated = copy.deepcopy(self.manifest)
        duplicated["ids"][-1] = duplicated["ids"][0]
        mutants.append(duplicated)
        dev = copy.deepcopy(self.manifest)
        dev["records"][0]["split"] = "dev"
        mutants.append(dev)
        other_phase = copy.deepcopy(self.manifest)
        other_phase["phase"] = "all"
        mutants.append(other_phase)
        for manifest in mutants:
            with self.subTest(manifest=manifest["phase"]), self.assertRaises(ValueError):
                pilot.validate_pilot(self.data, manifest, self.source)

    def test_rejects_altered_logical_records_or_source_parent(self):
        source = copy.deepcopy(self.source)
        source["records"][0]["start_time"] += .1
        with self.assertRaises(ValueError):
            pilot.validate_pilot(self.data, self.manifest, source)
        source = copy.deepcopy(self.source)
        source["parent_plan"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            pilot.validate_pilot(self.data, self.manifest, source)

    def test_audit_must_bind_exact_manifest_source_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            code = Path(folder) / "auditor.py"
            code.write_text("# immutable auditor fixture\n", encoding="utf-8")
            binding = pilot.teacher.binding(code)
            audit = {"schema": "local-tar-step02-independent-input-audit-v1", "status": "passed",
                     "scope": "pilot8", "record_count": 8, "plan": {"sha256": "data"},
                     "manifest": {"sha256": "manifest"}, "source_plan": {"sha256": "source"},
                     "code": binding, "records": [{"id": sid} for sid in self.ids]}
            args = (audit["plan"], audit["manifest"], audit["source_plan"], self.ids)
            pilot.validate_audit(audit, *args)
            for field, value in [("status", "failed"), ("record_count", 7),
                                 ("manifest", {"sha256": "different"}), ("source_plan", {})]:
                changed = copy.deepcopy(audit)
                changed[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    pilot.validate_audit(changed, *args)
            changed = copy.deepcopy(audit)
            changed["records"].reverse()
            with self.assertRaises(ValueError):
                pilot.validate_audit(changed, *args)
            code.write_text("# altered\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                pilot.validate_audit(audit, *args)


class TargetDiagnosticTests(unittest.TestCase):
    def test_reports_finite_time_variation_and_cosines(self):
        rng = np.random.default_rng(27)
        features = rng.normal(size=(11, 512)).astype(np.float32)
        result = pilot.feature_diagnostics(features)
        self.assertEqual(result["dimensions_with_temporal_variation"], 512)
        self.assertFalse(result["exactly_constant_across_time"])
        self.assertGreater(result["row_l2_norm_min"], 0)
        self.assertLessEqual(result["adjacent_cosine_max"], 1.000001)

    def test_refuses_collapsed_zero_nonfinite_and_wrong_dtype(self):
        bad = np.ones((5, 512), dtype=np.float32)
        nonfinite = bad.copy()
        nonfinite[2, 1] = np.nan
        for features in [bad, np.zeros_like(bad), nonfinite, bad.astype(np.float64), bad[:, :511]]:
            with self.subTest(shape=features.shape, dtype=features.dtype), self.assertRaises(ValueError):
                pilot.feature_diagnostics(features)


if __name__ == "__main__":
    unittest.main(verbosity=2)
