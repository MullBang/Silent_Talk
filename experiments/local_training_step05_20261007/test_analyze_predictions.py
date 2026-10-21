"""Synthetic saved-JSON diagnostics tests; no model or inference imported."""
import copy
import unittest

import analyze_predictions as subject


def example(seed=42, arm="A_ctc"):
    ids = [f"id_{index:03d}" for index in range(71)]
    conditions = {}
    for condition in subject.CONDITIONS:
        records = [{"id": sample_id, "speaker_id": "C002", "source_stem": "recording", "split": "dev",
                    "condition": condition, "reference": "가  나", "prediction": "", "status": "ok"}
                   for sample_id in ids]
        conditions[condition] = {"records": records}
    return {"schema": "lipnet-step05-final-development-evaluation-v1", "status": "completed",
            "seed": seed, "arm": arm, "completed_updates": 3200, "record_count": 71, "ids": ids,
            "conditions": conditions, "checkpoint": {"path": f"{seed}/{arm}/update_3200.pt"},
            "plan": {"path": f"seed{seed}.json"}, "full_data_plan": {"path": "same_handoff.json", "sha256": "same"}}


class PredictionDiagnosticTests(unittest.TestCase):
    def test_empty_outputs_have_full_denominators(self):
        result = subject.analyze_evaluation(example())
        normal = result["conditions"]["normal"]
        self.assertEqual(normal["empty_prediction_count"], 71)
        self.assertEqual(normal["unique_prediction_count"], 1)
        self.assertEqual(normal["unique_nonempty_prediction_count"], 0)
        self.assertEqual(normal["total_reference_characters"], 284)
        self.assertEqual(normal["mean_reference_characters"], 4.)
        self.assertEqual(normal["prediction_to_reference_character_ratio"], 0.)
        self.assertEqual(result["prediction_equality"]["all_three_equal"]["sentences"], 71)

    def test_nfc_strip_preserves_internal_spaces_and_frequency(self):
        value = example()
        rows = value["conditions"]["normal"]["records"]
        rows[0]["prediction"] = " 가  나 "
        rows[1]["prediction"] = "가  나"
        rows[2]["prediction"] = "가 나"
        result = subject.analyze_evaluation(value)
        normal = result["conditions"]["normal"]
        self.assertEqual(normal["unique_prediction_count"], 3)
        self.assertEqual(normal["total_prediction_characters"], 11)
        self.assertEqual(normal["mean_prediction_characters"], 11 / 71)
        self.assertEqual(normal["prediction_to_reference_character_ratio"], 11 / 284)
        self.assertEqual(normal["most_frequent_predictions"], [
            {"prediction": "", "count": 68}, {"prediction": "가  나", "count": 2}, {"prediction": "가 나", "count": 1}])

    def test_condition_equality_counts_all_rows(self):
        value = example()
        value["conditions"]["normal"]["records"][0]["prediction"] = "가"
        value["conditions"]["static"]["records"][0]["prediction"] = "가"
        value["conditions"]["shuffled"]["records"][1]["prediction"] = "나"
        matches = subject.analyze_evaluation(value)["prediction_equality"]
        self.assertEqual(matches["normal_equals_static"]["sentences"], 71)
        self.assertEqual(matches["normal_equals_shuffled"]["sentences"], 69)
        self.assertEqual(matches["all_three_equal"]["sentences"], 69)

    def test_examples_are_first_five_even_when_later_output_matches(self):
        value = example()
        value["conditions"]["normal"]["records"][-1]["prediction"] = "가  나"
        result = subject.analyze_evaluation(value)
        self.assertEqual([row["id"] for row in result["examples_first_five_fixed_development_rows"]], value["ids"][:5])
        self.assertTrue(all(row["predictions"]["normal"] == "" for row in result["examples_first_five_fixed_development_rows"]))

    def test_missing_reordered_duplicate_and_changed_reference_rejected(self):
        for mutation in ("missing", "reordered", "duplicate", "reference"):
            value = example()
            records = value["conditions"]["static"]["records"]
            if mutation == "missing":
                records.pop()
            elif mutation == "reordered":
                records[0], records[1] = records[1], records[0]
            elif mutation == "duplicate":
                records[0] = copy.deepcopy(records[1])
            else:
                records[0]["reference"] = "changed"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.analyze_evaluation(value)

    def test_failed_empty_prediction_retained_and_nonempty_failed_rejected(self):
        value = example()
        value["conditions"]["normal"]["records"][0]["status"] = "failed"
        result = subject.analyze_evaluation(value)
        self.assertEqual(result["conditions"]["normal"]["failed_inputs"], 1)
        self.assertEqual(result["conditions"]["normal"]["empty_prediction_count"], 71)
        value["conditions"]["normal"]["records"][0]["prediction"] = "bad"
        with self.assertRaises(ValueError):
            subject.analyze_evaluation(value)

    def test_four_runs_required_and_sorted_fixed_pairs(self):
        values = [example(seed, arm) for seed, arm in reversed(subject.PAIRS)]
        report = subject.build_report(values)
        self.assertEqual([(value["seed"], value["arm"]) for value in report["models"]], list(subject.PAIRS))
        with self.assertRaises(ValueError):
            subject.build_report(values[:3])
        values[-1] = values[0]
        with self.assertRaises(ValueError):
            subject.build_report(values)

    def test_cross_model_reference_and_handoff_change_rejected(self):
        values = [example(seed, arm) for seed, arm in subject.PAIRS]
        for condition in subject.CONDITIONS:
            values[-1]["conditions"][condition]["records"][0]["reference"] = "달라"
        with self.assertRaises(ValueError):
            subject.build_report(values)
        values = [example(seed, arm) for seed, arm in subject.PAIRS]
        values[-1]["full_data_plan"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            subject.build_report(values)


if __name__ == "__main__":
    unittest.main()
