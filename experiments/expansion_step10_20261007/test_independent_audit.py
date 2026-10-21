"""Reject misleading token paths, metric denominators and evaluation subsets."""
import copy
import unittest
import numpy as np
import independent_audit as a


def example():
    row = {"id": "example", "speaker_id": "s", "source_stem": "source", "split": "val", "text": "가", "frames": 4}
    order = [0, 1, 2, 3]
    record = {"id": "example", "speaker_id": "s", "source_stem": "source", "source_split": "val", "split": "dev",
              "reference": "가", "frames": 4, "condition": "normal", "status": "ok", "error": None,
              "raw_argmax_ids": [1, 1, 20, 0], "collapsed_ids": [1, 20], "prediction": "가", "replacement_characters": 0,
              "strict_label_decoder_diagnostic": {"status": "ok", "prediction": "가", "error": None},
              "temporal": {"frames": 4, "source_positions": order, "source_positions_sha256": a.num.nested_hash(order), "static_position": None},
              "metrics": a.num.metric_counts("가", "가")}
    return row, record


class IndependentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = a.read(a.BASE / "comparison_plan.json")
        cls.plan = a.read(a.BASE / "plans/seed42.json")
        cls.old_plan = a.read(cls.policy["controls"]["42"]["plan_file"]["path"])
        cls.old_run = a.read(cls.policy["controls"]["42"]["run_result"]["path"])
        cls.handoff = a.read(a.HANDOFF, a.HANDOFF_SHA)

    def check_plan(self, plan=None, policy=None, old_run=None):
        a.validate_plan_contract(policy or self.policy, plan or self.plan, self.handoff, self.old_plan,
                                 old_run or self.old_run,
                                 {"control_old441": self.old_plan["training_data"], "expanded510": self.plan["training_data"]})

    def test_actual_frozen_plan_contract_passes(self):
        self.check_plan()

    def test_hold_substitution_rejected(self):
        plan = copy.deepcopy(self.plan); plan["ids"][0] = self.handoff["held_new_train_ids"][0]
        plan["plan_sha256"] = a.num.canonical_hash(plan)
        with self.assertRaises(ValueError): self.check_plan(plan=plan)

    def test_schedule_swap_rejected_even_with_updated_digest(self):
        plan = copy.deepcopy(self.plan); plan["schedule"][0], plan["schedule"][1] = plan["schedule"][1], plan["schedule"][0]
        plan["plan_sha256"] = a.num.canonical_hash(plan)
        with self.assertRaises(ValueError): self.check_plan(plan=plan)

    def test_changed_runtime_rejected(self):
        plan = copy.deepcopy(self.plan); plan["runtime"]["torch"] = "other"
        plan["plan_sha256"] = a.num.canonical_hash(plan)
        with self.assertRaises(ValueError): self.check_plan(plan=plan)

    def test_warm_start_rejected(self):
        plan = copy.deepcopy(self.plan); plan["initial_state_hashes_expected"]["model"] = self.old_run["final_state_hashes"]["model"]
        plan["plan_sha256"] = a.num.canonical_hash(plan)
        with self.assertRaises(ValueError): self.check_plan(plan=plan)

    def test_incomplete_initial_state_gate_rejected(self):
        policy = copy.deepcopy(self.policy); policy["reuse_gates"]["42"]["fresh_initial_state_hashes"].pop("optimizer")
        with self.assertRaises(ValueError): self.check_plan(policy=policy)

    def test_early_optimization_gate_rejected(self):
        policy = copy.deepcopy(self.policy); policy["reuse_gates"]["42"]["student_updates_executed"] = 1
        with self.assertRaises(ValueError): self.check_plan(policy=policy)

    def test_altered_training_pixels_digest_rejected(self):
        plan = copy.deepcopy(self.plan); sid = plan["ids"][0]; plan["training_data"]["clips"][sid]["pixels_sha256"] = "0" * 64
        plan["plan_sha256"] = a.num.canonical_hash(plan)
        with self.assertRaises(ValueError): self.check_plan(plan=plan)

    def test_strict_composer_diagnostic_cannot_hide_failure(self):
        row, record = example(); record["strict_label_decoder_diagnostic"]["status"] = "failed"
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_failed_row_still_cannot_keep_token_evidence(self):
        row, record = example(); record.update(status="failed", error="failure", prediction="", metrics=a.num.metric_counts("가", ""))
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_ctc_collapse_before_blank_removal(self):
        self.assertEqual(a.collapse([1, 1, 0, 1, 20, 20], 6), [1, 1, 20])

    def test_argmax_wrong_length_rejected(self):
        with self.assertRaises(ValueError): a.collapse([0], 2)

    def test_boolean_token_rejected(self):
        with self.assertRaises(ValueError): a.collapse([True], 1)

    def test_class69_rejected(self):
        with self.assertRaises(ValueError): a.collapse([69], 1)

    def test_unicode_arithmetic(self):
        self.assertEqual(a.compose([1, 20, 41, 68, 1, 20])["prediction"], "각 가")

    def test_invalid_jamos_not_erased(self):
        value = a.compose([20, 1, 1, 20, 67])
        self.assertEqual(value["prediction"], "��갛")
        self.assertEqual(value["replacement_characters"], 2)
        self.assertEqual(value["strict"]["status"], "failed")

    def test_empty_ctc_output_preserved(self):
        self.assertEqual(a.compose(a.collapse([0, 0], 2))["prediction"], "")

    def test_unknown_collapsed_blank_rejected(self):
        with self.assertRaises(ValueError): a.compose([0])

    def test_metric_space_is_primary_character(self):
        m = a.num.metric_counts("가 나", "가나")
        self.assertEqual((m["character_edits"], m["reference_characters"]), (1, 3))

    def test_failed_input_stays_in_denominator(self):
        rows = [{"id": "one", "reference": "가나", "prediction": "", "status": "failed"},
                {"id": "two", "reference": "가", "prediction": "가", "status": "ok"}]
        m = a.num.aggregate_metrics(rows)
        self.assertEqual(m["character_edits"], 2)
        self.assertEqual(m["reference_characters"], 3)
        self.assertAlmostEqual(m["cer_percent"], 200 / 3)
        self.assertEqual(m["failed_inputs"], 1)

    def test_duplicate_metric_rows_rejected(self):
        x = {"id": "same", "reference": "가", "prediction": "가", "status": "ok"}
        with self.assertRaises(ValueError): a.num.aggregate_metrics([x, x])

    def test_static_middle_frame(self):
        self.assertEqual(a.num.temporal_order(5, "static", "a"), [2] * 5)

    def test_shuffle_is_fixed_per_id(self):
        x = a.num.temporal_order(25, "shuffled", "a")
        self.assertEqual(x, a.num.temporal_order(25, "shuffled", "a"))
        self.assertEqual(sorted(x), list(range(25)))
        self.assertNotEqual(x, list(range(25)))

    def test_normal_prediction_pass(self):
        row, record = example(); a.validate_prediction(record, row, "normal")

    def test_deleted_bad_jamo_prediction_rejected(self):
        row, record = example(); record["raw_argmax_ids"] = [1, 0, 1, 20]
        record["collapsed_ids"] = [1, 1, 20]
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_reference_substitution_rejected(self):
        row, record = example(); record["reference"] = "나"
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_false_success_error_rejected(self):
        row, record = example(); record["error"] = "failed inference"
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_temporal_changed_order_rejected(self):
        row, record = example(); record["temporal"]["source_positions"] = [0, 2, 1, 3]
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_metric_deletion_rejected(self):
        row, record = example(); record["metrics"]["reference_characters"] = 0
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_failed_row_must_have_empty_prediction(self):
        row, record = example(); record["status"] = "failed"; record["error"] = "failure"
        with self.assertRaises(ValueError): a.validate_prediction(record, row, "normal")

    def test_model_scope_selection(self):
        handoff, rows, groups = a.cohort_source()
        self.assertEqual(len(handoff["provisional_train_ids"]), 510)
        self.assertEqual(len(groups["old71"]), 71)
        self.assertEqual(len(groups["new24"]), 24)
        self.assertEqual(len(groups["new23"]), 23)

    def test_typed_digest_distinguishes_numpy_and_list(self):
        self.assertNotEqual(a.num.nested_hash([1, 2]), a.num.nested_hash(np.array([1, 2], np.int64)))

    def test_two_seed_expanded_minus_old_sign(self):
        values = {}
        for variant in a.VARIANTS:
            for seed in a.SEEDS:
                metric = {"cer_percent": 80. if variant == "control_old441" else 70., "exact_sentences": 1 if variant == "control_old441" else 2}
                values[(variant, seed)] = {c: {condition: dict(metric) for condition in a.CONDITIONS} for c in ("old71", "new24", "new23")}
        out = a.compare(values)
        self.assertEqual(out["new24"]["normal"]["mean_expanded_minus_old_cer_percentage_points"], -10.)

    def test_preflight_exact_snapshots_retained(self):
        value = a.preserved_preflight()
        self.assertTrue(all(x["byte_exact"] for x in value["code_snapshots"]))

    def test_rounding_only_tolerance(self):
        a.close_tree({"x": [0.1 + 0.2]}, {"x": [0.3]}, "rounding")
        with self.assertRaises(ValueError): a.close_tree({"x": [0.301]}, {"x": [0.3]}, "changed score")

    def test_derived_comparison_integer_may_not_change(self):
        with self.assertRaises(ValueError): a.close_tree({"edits": 30}, {"edits": 31}, "integer")

    def test_comparison_nonfinite_rejected(self):
        with self.assertRaises(ValueError): a.close_tree(float("nan"), 1.0, "nan")

    def test_prediction_change_uses_edit_count_direction(self):
        row, old = example(); new = copy.deepcopy(old)
        new["prediction"] = "나"; new["metrics"] = a.num.metric_counts(row["text"], "나")
        value = a.changed_predictions([old], [new])
        self.assertEqual(value["totals"]["worsened"], 1)
        self.assertEqual(value["totals"]["net_character_edits_change"], 1)
        self.assertEqual(value["totals"]["changed_predictions"], 1)

    def test_prediction_change_does_not_call_equal_error_improvement(self):
        row, old = example(); old["prediction"] = "나"; old["metrics"] = a.num.metric_counts(row["text"], "나")
        new = copy.deepcopy(old); new["prediction"] = "다"; new["metrics"] = a.num.metric_counts(row["text"], "다")
        value = a.changed_predictions([old], [new])
        self.assertEqual(value["totals"]["tied"], 1)
        self.assertEqual(value["totals"]["changed_predictions"], 1)

    def test_prediction_change_reordered_ids_rejected(self):
        _, old = example(); new = copy.deepcopy(old); new["id"] = "different"
        with self.assertRaises(ValueError): a.changed_predictions([old], [new])

    def test_inference_binding_rejects_teacher_target(self):
        ref = {"path": str(a.BASE / "student.npz"), "sha256": "0" * 64}
        teacher = {"path": str(a.BASE / "teacher.npz"), "sha256": "1" * 64}
        with self.assertRaises(ValueError): a.num.validate_npz_binding_scope([ref, teacher], [ref])


if __name__ == "__main__": unittest.main()
