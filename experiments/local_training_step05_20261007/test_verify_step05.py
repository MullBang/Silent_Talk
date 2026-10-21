"""Independent final-audit compatibility, metric and tampering tests."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

import verify_step05 as verify


class DigestAndStateTests(unittest.TestCase):
    def test_hash_is_typed_and_sensitive_to_state_values(self):
        variants = [True, 1, 1., "1", [1], (1,), np.array([1], np.int64), np.array([1], np.int32), torch.tensor([1])]
        self.assertEqual(len({verify.nested_hash(value) for value in variants}), len(variants))
        self.assertEqual(verify.nested_hash({"z": 2, "a": 1}), verify.nested_hash({"a": 1, "z": 2}))
        with self.assertRaises(ValueError):
            verify.nested_hash(float("nan"))

    def test_restricted_numpy_unpack_refuses_shape_and_object_dtype(self):
        array = np.arange(7, dtype=np.uint32)
        packed = {verify.NUMPY_MARKER: True, "dtype": array.dtype.str, "shape": list(array.shape),
                  "bytes": torch.from_numpy(np.frombuffer(array.tobytes(), np.uint8).copy())}
        np.testing.assert_array_equal(verify.unpack(packed), array)
        for key, value in [("shape", [8]), ("dtype", "object"), ("bytes", torch.ones(28))]:
            changed = {**packed, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.unpack(changed)

    def test_cpu_reader_matches_existing_frozen_step03_checkpoint_hashes(self):
        path = verify.ROOT / "experiments/local_training_step03_20261007/runs/B_feature/continuous/run_result.json"
        record = verify.read_json(path)
        payload = verify.load_checkpoint(record["checkpoint"])
        self.assertEqual(payload["completed_updates"], 4)
        self.assertEqual(payload["state_hashes"], record["final_state_hashes"])
        self.assertEqual(verify.tensor_hash(payload["state"]["model"]), payload["history"][-1]["model_sha256"])
        verify.validate_optimizer(payload["state"], "B_feature", 4)
        changed = copy.deepcopy(record["checkpoint"])
        changed["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify.load_checkpoint(changed)

    def test_nonfinite_tensor_state_is_rejected(self):
        with self.assertRaises(ValueError):
            verify.finite_tree({"optimizer": {"exp_avg": torch.tensor([1., float("inf")])}})

    def test_payload_scope_refuses_missing_extra_teacher_and_duplicate_refs(self):
        expected = [{"path": str(verify.BASE / "dev_fixture.npz"), "sha256": "a" * 64}]
        verify.validate_npz_binding_scope(expected, expected)
        other = {"path": str(verify.BASE / "teacher_fixture.npz"), "sha256": "b" * 64}
        for refs in [[], [*expected, other], [*expected, *expected]]:
            with self.assertRaises(ValueError):
                verify.validate_npz_binding_scope(refs, expected)

    def test_payload_scope_refuses_nonworkspace_media_drive_reference(self):
        with self.assertRaises(ValueError):
            verify.validate_npz_binding_scope([{"path": "D:/must_not_read.npz", "sha256": "a" * 64}], [])

    def test_schedule_independently_matches_frozen_handoff_and_both_seed_plans(self):
        parent = verify.read_json(verify.PARENT, verify.PARENT_SHA)
        handoff = verify.read_json(verify.STEP4 / "full_data_plan.json", verify.HANDOFF_SHA)
        schedules = []
        for seed in verify.SEEDS:
            plan, ref = verify.validate_plan_file(seed, parent, handoff)
            self.assertEqual(ref, verify.binding(verify.BASE / "plans" / f"seed{seed}.json"))
            schedules.append(plan["schedule"])
            self.assertEqual(len(plan["schedule"]), 3200)
            self.assertEqual({sid for batch in plan["schedule"] for sid in batch}, set(handoff["train_ids"]))
        self.assertNotEqual(*schedules)


def history_fixture(arm="B_feature"):
    plan = {"updates": 2, "schedule": [["a", "b"], ["b", "a"]]}
    rows = {"a": {"frames": 10}, "b": {"frames": 12}}
    records = []
    for index, ids in enumerate(plan["schedule"]):
        feature = .25 if arm == "B_feature" else 0.
        adapter_norm = .8 if arm == "B_feature" else 0.
        record = {"update": index + 1, "ids": ids, "frames": [rows[sid]["frames"] for sid in ids],
                  "optimizer_updated": True, "student_changed": True, "adapter_changed": arm == "B_feature",
                  "ctc_loss": 2., "feature_loss": feature, "loss": 2. + feature,
                  "lambda": 1. if arm == "B_feature" else 0., "gradient_norm_before_clipping": 8.,
                  "gradient_norm_after_clipping": math.hypot(.6, adapter_norm), "seconds_optimizer_update": .04,
                  "student_gradients": {"parameter_tensors_with_gradient": 31,
                      "parameter_tensors_with_nonzero_gradient": 31, "norm_after_clipping": .6},
                  "adapter_gradients": {"parameter_tensors_with_gradient": 2 if arm == "B_feature" else 0,
                      "parameter_tensors_with_nonzero_gradient": 2 if arm == "B_feature" else 0,
                      "norm_after_clipping": adapter_norm}, "feature_hook_count_before": 0, "feature_hook_count_after": 0,
                  "cuda_peak_allocated_bytes": 400, "cuda_peak_reserved_bytes": 500,
                  "model_sha256": "1" * 64, "adapter_sha256": "2" * 64,
                  "state_hashes": {key: "3" * 64 for key in ("model", "adapter", "optimizer", "rng", "complete")}}
        records.append(record)
    return records, plan, rows


class HistoryTamperTests(unittest.TestCase):
    def test_valid_both_arm_histories_have_no_skips(self):
        for arm in verify.ARMS:
            records, plan, rows = history_fixture(arm)
            result = verify.validate_history(records, plan, arm, rows)
            self.assertEqual(result["updates"], 2)
            self.assertEqual(result["clip_presentations"], 4)
            self.assertEqual(result["per_clip_presentations"], {"a": 2, "b": 2})

    def test_rejects_missing_duplicate_reordered_updates_and_wrong_frame_count(self):
        records, plan, rows = history_fixture()
        for changed in [records[:1], [records[0], records[0]], list(reversed(records))]:
            with self.assertRaises(ValueError):
                verify.validate_history(changed, plan, "B_feature", rows)
        changed = copy.deepcopy(records)
        changed[0]["ids"].reverse()
        with self.assertRaises(ValueError):
            verify.validate_history(changed, plan, "B_feature", rows)
        changed = copy.deepcopy(records)
        changed[0]["frames"][0] += 1
        with self.assertRaises(ValueError):
            verify.validate_history(changed, plan, "B_feature", rows)

    def test_rejects_nonfinite_skipped_zerochange_and_leakedhook(self):
        records, plan, rows = history_fixture()
        for key, value in [("loss", float("nan")), ("optimizer_updated", False), ("student_changed", False),
                           ("adapter_changed", False), ("feature_hook_count_after", 1), ("gradient_norm_after_clipping", 1.1)]:
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_history(changed, plan, "B_feature", rows)

    def test_ctc_arm_cannot_include_adapter_or_feature_loss(self):
        records, plan, rows = history_fixture("A_ctc")
        for key, value in [("adapter_changed", True), ("feature_loss", .1), ("lambda", 1.)]:
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_history(changed, plan, "A_ctc", rows)


class OptimizerTamperTests(unittest.TestCase):
    def test_wrong_adam_step_arm_or_hyperparameter_is_rejected(self):
        record = verify.read_json(verify.ROOT / "experiments/local_training_step03_20261007/runs/A_ctc/continuous/run_result.json")
        state = verify.load_checkpoint(record["checkpoint"])["state"]
        verify.validate_optimizer(state, "A_ctc", 4)
        with self.assertRaises(ValueError):
            verify.validate_optimizer(state, "B_feature", 4)
        with self.assertRaises(ValueError):
            verify.validate_optimizer(state, "A_ctc", 3200)
        changed = copy.deepcopy(state)
        changed["optimizer"]["param_groups"][0]["lr"] = .002
        with self.assertRaises(ValueError):
            verify.validate_optimizer(changed, "A_ctc", 4)
        changed["optimizer"]["param_groups"][0]["lr"] = .001
        changed["optimizer"]["state"][0]["exp_avg_sq"].flatten()[0] = -1
        with self.assertRaises(ValueError):
            verify.validate_optimizer(changed, "A_ctc", 4)


def condition_fixture(condition="normal"):
    rows = [{"id": "one", "speaker_id": "s1", "source_stem": "rec1", "split": "dev", "text": "가  나", "frames": 5},
            {"id": "two", "speaker_id": "s2", "source_stem": "rec2", "split": "dev", "text": "다", "frames": 4}]
    records = []
    for row, prediction, failed in zip(rows, ["가 나", ""], [False, True]):
        order = verify.temporal_order(row["frames"], condition, row["id"])
        records.append({"id": row["id"], "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                        "split": "dev", "condition": condition, "reference": row["text"], "prediction": prediction,
                        "status": "failed" if failed else "ok", "error": "RuntimeError: fixture" if failed else None,
                        "frames": row["frames"], "seconds": .02,
                        "temporal": {"frames": row["frames"], "source_positions": order, "source_positions_sha256": verify.nested_hash(order),
                                     "static_position": row["frames"] // 2 if condition == "static" else None},
                        "metrics": verify.metric_counts(row["text"], prediction)})
    result = {"records": records, "aggregate": verify.aggregate_metrics(records),
              "per_speaker": {row["speaker_id"]: verify.aggregate_metrics([record]) for row, record in zip(rows, records)}}
    return result, rows


class MetricAndConditionTests(unittest.TestCase):
    def test_nfc_internal_whitespace_and_empty_failure_denominator(self):
        counts = verify.metric_counts("가  나", "가 나")
        self.assertEqual(counts["character_edits"], 1)
        self.assertEqual(counts["normalized_character_edits"], 0)
        self.assertTrue(counts["normalized_exact"])
        result, rows = condition_fixture()
        metrics = verify.validate_condition(result, "normal", rows)["aggregate"]
        self.assertEqual(metrics["clips"], 2)
        self.assertEqual(metrics["failed_inputs"], 1)
        self.assertEqual(metrics["reference_characters"], 5)
        self.assertEqual(metrics["cer_percent"], 40.)

    def test_all_three_temporal_conditions_validate(self):
        for condition in verify.CONDITIONS:
            result, rows = condition_fixture(condition)
            verify.validate_condition(result, condition, rows)

    def test_failed_row_cannot_be_omitted_or_have_nonempty_prediction(self):
        result, rows = condition_fixture()
        omitted = copy.deepcopy(result)
        omitted["records"].pop()
        with self.assertRaises(ValueError):
            verify.validate_condition(omitted, "normal", rows)
        changed = copy.deepcopy(result)
        changed["records"][1]["prediction"] = "다"
        with self.assertRaises(ValueError):
            verify.validate_condition(changed, "normal", rows)

    def test_metric_reference_prediction_and_aggregate_tampering_are_detected(self):
        result, rows = condition_fixture()
        for mutate in [lambda x: x["records"][0].update(reference="wrong"),
                       lambda x: x["records"][0].update(prediction="wrong"),
                       lambda x: x["records"][0]["metrics"].update(character_edits=0),
                       lambda x: x["aggregate"].update(cer_percent=0.),
                       lambda x: x["per_speaker"]["s1"].update(clips=0)]:
            changed = copy.deepcopy(result)
            mutate(changed)
            with self.assertRaises(ValueError):
                verify.validate_condition(changed, "normal", rows)

    def test_static_must_use_middle_and_shuffle_exact_fixed_permutation(self):
        for condition in ("static", "shuffled"):
            result, rows = condition_fixture(condition)
            changed = copy.deepcopy(result)
            changed["records"][0]["temporal"]["source_positions"][0] = 99
            with self.assertRaises(ValueError):
                verify.validate_condition(changed, condition, rows)

    def test_paired_difference_keeps_both_seeds_and_direction(self):
        evals = {}
        for seed in verify.SEEDS:
            for arm in verify.ARMS:
                cer = {42: {"A_ctc": 50., "B_feature": 40.}, 43: {"A_ctc": 20., "B_feature": 24.}}[seed][arm]
                evals[(seed, arm)] = {condition: {"aggregate": {"cer_percent": cer, "exact_sentences": 2}}
                                     for condition in verify.CONDITIONS}
        result = verify.compare_conditions(evals)
        self.assertEqual(result["normal"]["per_seed"]["42"]["B_minus_A_cer_percentage_points"], -10.)
        self.assertEqual(result["normal"]["per_seed"]["43"]["B_minus_A_cer_percentage_points"], 4.)
        self.assertEqual(result["normal"]["mean_B_minus_A_cer_percentage_points"], -3.)


class PairedInitializationTests(unittest.TestCase):
    def fixture(self):
        runs = {}
        for seed in verify.SEEDS:
            for arm in verify.ARMS:
                runs[(seed, arm)] = {"initial_state_hashes": {"model": f"model-{seed}", "adapter": f"adapter-{seed}",
                    "rng": f"rng-{seed}", "optimizer": arm}, "data_sha256": "data", "schedule_sha256": f"order-{seed}",
                    "per_clip_presentations": {"a": 15}, "updates": 3200, "clip_presentations": 6400,
                    "final_state_hashes": {"rng": f"final-rng-{seed}"}}
        return runs

    def test_matched_initial_model_adapter_rng_but_distinct_optimizer(self):
        pairs = verify.validate_paired_runs(self.fixture())
        self.assertEqual(set(pairs), {"42", "43"})
        self.assertTrue(pairs["42"]["final_rng_hashes_equal"])

    def test_mismatched_initialization_data_schedule_and_budget_fail(self):
        for key in ("data_sha256", "schedule_sha256", "updates"):
            runs = self.fixture()
            runs[(42, "B_feature")][key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify.validate_paired_runs(runs)
        runs = self.fixture()
        runs[(42, "B_feature")]["initial_state_hashes"]["model"] = "different"
        with self.assertRaises(ValueError):
            verify.validate_paired_runs(runs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
