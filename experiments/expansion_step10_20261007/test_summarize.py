"""Synthetic accounting tests; no models, real cache reads or optimization."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize as subject


def policy_fixture():
    old, new = [f"olddev{i}" for i in range(71)], [f"newdev{i}" for i in range(24)]
    control, added = [f"oldtrain{i}" for i in range(441)], [f"newtrain{i}" for i in range(69)]
    return {"schema": "lipnet-step10-expansion-comparison-plan-v1", "status": "frozen_before_expanded_training_and_new_evaluation",
            "seeds": [42, 43], "variants": list(subject.VARIANTS), "method": "B_feature", "updates_per_model": 3200,
            "clips_per_update": 2, "new_optimizer_updates_planned": 6400, "reused_historical_optimizer_updates": 6400,
            "new_full_transcript_approvals": 0, "training_ready_flags_promoted": False, "service_weights_updated": False,
            "audio_used": False, "control_train_ids": control, "provisional_train_ids": control + added,
            "held_train_ids": [f"hold{i}" for i in range(27)], "runtime": {"device": "synthetic"},
            "evaluation_contract": {"conditions": list(subject.CONDITIONS), "declared_before_training": True,
                "development": {"old71_ids": old, "new24_ids": new, "new23_ids": new[1:], "excluded_duplicate_ids": new[:1],
                    "overlap_evidence": [{"train_id": control[0], "val_id": new[0], "train_text": "가 나", "val_text": "가나"}]}}}


def row(sid, group="new24", condition="normal", reference="가 나", prediction="가 나", speaker="speaker0", status="ok"):
    return {"id": sid, "split": "dev", "source_split": "dev" if group == "old71" else "val", "group": group,
            "condition": condition, "speaker_id": speaker, "source_stem": "recording_" + sid,
            "reference": reference, "prediction": prediction, "metrics": subject.counts(reference, prediction),
            "status": status, "error": None if status == "ok" else "synthetic error", "frames": 2,
            "raw_argmax_ids": [1, 1], "collapsed_ids": [1], "replacement_characters": prediction.count("\ufffd"),
            "temporal": {"frames": 2}, "seconds": .1, "strict_label_decoder_diagnostic": {}}


def part(records):
    return {"records": records, "aggregate": subject.aggregate(records),
            "per_speaker": {speaker: subject.aggregate([r for r in records if r["speaker_id"] == speaker])
                            for speaker in sorted({r["speaker_id"] for r in records})},
            "replacement_characters": sum(r["prediction"].count("\ufffd") for r in records),
            "sentences_with_replacement_characters": sum("\ufffd" in r["prediction"] for r in records)}


def evaluation_fixture(seed=42, variant="control_old441"):
    policy = policy_fixture()
    meta = policy["evaluation_contract"]["development"]
    cohorts = {}
    for group in ("old71", "new24"):
        ids = meta[group + "_ids"]
        cohorts[group] = {"ids": ids, "record_count": len(ids), "model_state_before": "model", "model_state_after": "model",
                          "model_state_unchanged": True, "aggregation_only": False,
                          "conditions": {condition: part([row(sid, group, condition, speaker=f"speaker{i % 2}",
                             prediction="가 나" if variant == "expanded510" else "가 다") for i, sid in enumerate(ids)])
                             for condition in subject.CONDITIONS}}
    cohorts["new23"] = {**cohorts["new24"], "ids": meta["new23_ids"], "record_count": 23,
                        "aggregation_only": True, "parent_group": "new24", "excluded_ids": meta["excluded_duplicate_ids"],
                        "conditions": {condition: part(cohorts["new24"]["conditions"][condition]["records"][1:])
                                       for condition in subject.CONDITIONS}}
    return {"schema": "lipnet-step10-development-evaluation-v1", "status": "completed", "seed": seed, "variant": variant,
            "arm": "B_feature", "completed_updates": 3200, "inferred_distinct_records": 95,
            "inferred_condition_records": 285, "cohorts": cohorts, "new_optimizer_updates": 0,
            "audio_read": False, "original_media_read": False, "teacher_targets_read": False, "teacher_executed": False,
            "optimizer_executed": False, "held_training_records_read": False, "service_weights_updated": False,
            "old_control_replay": {"status": "passed" if variant == "control_old441" else "not_applicable_to_new_model",
                                   "mismatches": [], "compared_records": 213}}


def training_fixture(variant="expanded510"):
    policy = policy_fixture()
    ids = policy["control_train_ids"] if variant == "control_old441" else policy["provisional_train_ids"]
    clips = {sid: {"frames": 3 + i % 5} for i, sid in enumerate(ids)}
    schedule = [[ids[2 * i % len(ids)], ids[(2 * i + 1) % len(ids)]] for i in range(3200)]
    plan = {"seed": 42, "updates": 3200, "batch_size": 2, "ids": ids, "schedule": schedule,
            "training_data": {"clips": clips}, "plan_sha256": "frozen", "runtime": policy["runtime"]}
    updates = [{"update": i + 1, "ids": batch, "frames": [clips[sid]["frames"] for sid in batch],
                "optimizer_updated": True, "student_changed": True, "adapter_changed": True,
                "loss": i + 2., "ctc_loss": i + 1., "feature_loss": 1., "lambda": 1.,
                "seconds_optimizer_update": .2, "cuda_peak_allocated_bytes": i + 100,
                "cuda_peak_reserved_bytes": i + 200, "state_hashes": {"model": f"state{i}"}}
               for i, batch in enumerate(schedule)]
    result = {"status": "completed", "arm": "B_feature", "seed": 42, "variant": variant,
              "completed_updates": 3200, "start_update": 0, "restored_from": None,
              "plan_sha256": "frozen", "runtime": policy["runtime"], "updates": updates,
              "dev_inputs_loaded": False, "audio_used": False, "teacher_model_loaded": False,
              "service_weights_updated": False, "held_inputs_loaded": False, "elapsed_seconds": 800.,
              "final_state_hashes": updates[-1]["state_hashes"],
              "checkpoint": {"completed_updates": 3200, "state_hashes": updates[-1]["state_hashes"]}}
    return result, plan, policy


class MetricsTests(unittest.TestCase):
    def test_independent_character_word_and_nfc_counts(self):
        value = subject.counts("\u1100\u1161  나", "가 다")
        self.assertEqual((value["character_edits"], value["reference_characters"]), (2, 4))
        self.assertEqual((value["word_edits"], value["reference_words"]), (1, 2))
        self.assertEqual((value["normalized_character_edits"], value["normalized_reference_characters"]), (1, 3))

    def test_failed_record_keeps_full_denominator(self):
        value = subject.aggregate([row("a"), row("b", reference="가나다", prediction="", status="failed")])
        self.assertEqual((value["character_edits"], value["reference_characters"], value["cer_percent"]), (3, 6, 50.))
        self.assertEqual((value["clips"], value["failed_inputs"], value["exact_sentences"]), (2, 1, 1))

    def test_saved_metrics_tampering_is_rejected(self):
        value = row("a")
        value["metrics"]["character_edits"] = 99
        with self.assertRaisesRegex(ValueError, "recount"):
            subject.aggregate([value])

    def test_duplicate_or_empty_cohort_rejected(self):
        for records in ([], [row("a"), row("a")]):
            with self.assertRaises(ValueError):
                subject.aggregate(records)

    def test_failed_prediction_cannot_be_kept_as_success_text(self):
        with self.assertRaisesRegex(ValueError, "empty prediction"):
            subject.aggregate([row("a", status="failed")])

    def test_ctc_repeats_collapse_before_blank_removal(self):
        self.assertEqual(subject.collapse_ids([1, 1, 0, 1, 20, 20]), [1, 1, 20])
        for values in ([True], [69], [], [-1]):
            with self.assertRaises(ValueError):
                subject.collapse_ids(values)

    def test_prediction_changes_preserve_tied_but_changed_output(self):
        one = [row("a", prediction="가 다"), row("b", prediction="가 다"), row("c")]
        two = [row("a", prediction="가 마"), row("b"), row("c", prediction="")]
        result = subject.prediction_changes(one, two)
        self.assertEqual(result["totals"], {"improved": 1, "worsened": 1, "tied": 1,
                                          "clips": 3, "changed_predictions": 3,
                                          "unchanged_predictions": 0, "net_character_edits_change": 2})

    def test_two_seed_mean_is_not_denominator_weighted_pool(self):
        value = subject.mean_values([{"cer_percent": 100.}, {"cer_percent": 0.}])
        self.assertEqual(value, {"cer_percent": 50.})
        self.assertNotEqual(value["cer_percent"], 100 * 1 / (1 + 9))
        with self.assertRaises(ValueError):
            subject.mean_values([{"cer_percent": 0.}])


class EvaluationTests(unittest.TestCase):
    def test_full_fixed_fixture_valid(self):
        result = subject.validate_evaluation(evaluation_fixture(), policy_fixture())
        self.assertEqual(result["old71"]["normal"]["aggregate"]["reference_characters"], 213)
        self.assertEqual(result["new24"]["normal"]["aggregate"]["clips"], 24)
        self.assertEqual(result["new23"]["normal"]["aggregate"]["clips"], 23)

    def test_missing_duplicate_or_reordered_id_rejected(self):
        for mode in ("missing", "duplicate", "reordered"):
            value = evaluation_fixture()
            rows = value["cohorts"]["old71"]["conditions"]["normal"]["records"]
            if mode == "missing": rows.pop()
            elif mode == "duplicate": rows[-1] = rows[0]
            else: rows.reverse()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                subject.validate_evaluation(value, policy_fixture())

    def test_saved_aggregate_and_speaker_tampering_rejected(self):
        for target in ("aggregate", "per_speaker"):
            value = evaluation_fixture()
            part_value = value["cohorts"]["old71"]["conditions"]["normal"]
            if target == "aggregate": part_value[target]["cer_percent"] = 0
            else: part_value[target]["speaker0"]["cer_percent"] = 0
            with self.subTest(target=target), self.assertRaises(ValueError):
                subject.validate_evaluation(value, policy_fixture())

    def test_failed_replay_and_missing_condition_rejected(self):
        value = evaluation_fixture()
        value["old_control_replay"]["status"] = "failed"
        with self.assertRaises(ValueError): subject.validate_evaluation(value, policy_fixture())
        value = evaluation_fixture()
        del value["cohorts"]["old71"]["conditions"]["static"]
        with self.assertRaises(ValueError): subject.validate_evaluation(value, policy_fixture())

    def test_new23_is_exact_subset_without_reinference(self):
        value = evaluation_fixture()
        value["cohorts"]["new23"] = copy.deepcopy(value["cohorts"]["new23"])
        value["cohorts"]["new23"]["conditions"]["normal"]["records"][0]["seconds"] = 99
        with self.assertRaisesRegex(ValueError, "reuse exact"):
            subject.validate_evaluation(value, policy_fixture())

    def test_reference_cannot_change_by_condition(self):
        value = evaluation_fixture()
        rows = value["cohorts"]["old71"]["conditions"]["static"]["records"]
        rows[0]["reference"] = "다 라"
        rows[0]["metrics"] = subject.counts("다 라", rows[0]["prediction"])
        value["cohorts"]["old71"]["conditions"]["static"] = part(rows)
        with self.assertRaisesRegex(ValueError, "References"):
            subject.validate_evaluation(value, policy_fixture())

    def test_role_or_approval_promotion_rejected(self):
        for key in ("training_ready_flags_promoted", "service_weights_updated"):
            policy = policy_fixture(); policy[key] = True
            with self.assertRaises(ValueError): subject.validate_metadata(policy)
        policy = policy_fixture(); policy["held_train_ids"][0] = policy["provisional_train_ids"][0]
        with self.assertRaises(ValueError): subject.validate_metadata(policy)
        policy = policy_fixture(); policy["evaluation_contract"]["development"]["new23_ids"].reverse()
        with self.assertRaises(ValueError): subject.validate_metadata(policy)

    def test_four_model_comparison_has_no_pooled95_or_best_seed(self):
        evaluations = {key: evaluation_fixture(*key) for key in subject.PAIRS}
        result = subject.compare(evaluations, policy_fixture())
        self.assertEqual(set(result["two_seed_arithmetic_means"]), set(subject.GROUPS))
        self.assertEqual(len(result["per_model"]), 4)
        self.assertEqual(len(result["integer_accounting_rows"]), 4 * (71 + 24 + 23) * 3)
        paired = result["paired_seed_comparisons"][0]["groups"]["new24"]["normal"]
        self.assertAlmostEqual(paired["expanded_minus_control"]["cer_percentage_points"], -100 / 3)
        self.assertEqual(paired["prediction_changes"]["totals"]["improved"], 24)
        del evaluations[(43, "expanded510")]
        with self.assertRaises(ValueError): subject.compare(evaluations, policy_fixture())


class TrainingTests(unittest.TestCase):
    def test_actual_exposure_and_loss_windows(self):
        run, plan, policy = training_fixture()
        value = subject.training_summary(run, plan, policy, "expanded510")
        self.assertEqual((value["new_optimizer_updates"], value["reused_historical_optimizer_updates"]), (3200, 0))
        self.assertEqual(value["clip_presentations"], 6400)
        self.assertEqual(value["processed_frames"], sum(sum(r["frames"]) for r in run["updates"]))
        self.assertEqual(value["exposure_by_role"]["old441"]["unique_clips"], 441)
        self.assertEqual(value["exposure_by_role"]["new69"]["unique_clips"], 69)
        self.assertEqual(sum(r["presentations"] for r in value["per_clip_exposures"]), 6400)
        self.assertEqual(value["loss_window_means"]["first100"]["loss"], 51.5)
        self.assertEqual(value["loss_window_means"]["last100"]["loss"], 3151.5)
        self.assertEqual(value["cuda_peak_allocated_bytes"], 3299)

    def test_historical_control_is_not_counted_as_new_training(self):
        run, plan, policy = training_fixture("control_old441")
        value = subject.training_summary(run, plan, policy, "control_old441")
        self.assertEqual((value["new_optimizer_updates"], value["reused_historical_optimizer_updates"]), (0, 3200))
        self.assertEqual(value["exposure_by_role"]["new69"]["presentations"], 0)
        self.assertEqual(value["exposure_by_role"]["new69"]["unique_clips"], 0)

    def test_actual_schedule_frame_or_final_state_mismatch_rejected(self):
        for change in ("ids", "frames", "final_state"):
            run, plan, policy = training_fixture()
            if change == "ids": run["updates"][0]["ids"] = ["forbidden", "forbidden"]
            elif change == "frames": run["updates"][0]["frames"][0] += 1
            else: run["final_state_hashes"] = {"model": "different"}
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.training_summary(run, plan, policy, "expanded510")

    def test_partial_history_or_nonfinite_loss_rejected(self):
        for change in ("missing", "resumed", "nan"):
            run, plan, policy = training_fixture()
            if change == "missing": run["updates"].pop()
            elif change == "resumed": run["start_update"] = 100; run["restored_from"] = "checkpoint"
            else: run["updates"][0]["loss"] = float("nan")
            with self.subTest(change=change), self.assertRaises(ValueError):
                subject.training_summary(run, plan, policy, "expanded510")

    def test_hash_binding_rejects_changed_json_and_workspace_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(subject, "ROOT", root):
                file = root / "test.json"; file.write_text(json.dumps({"value": 1}), encoding="utf-8")
                ref = subject.bind(file)
                self.assertEqual(subject.read(ref), {"value": 1})
                file.write_text(json.dumps({"value": 2}), encoding="utf-8")
                with self.assertRaises(ValueError): subject.read(ref)
                with self.assertRaises(ValueError): subject.local(root.parent / "outside.json")


if __name__ == "__main__":
    unittest.main()
