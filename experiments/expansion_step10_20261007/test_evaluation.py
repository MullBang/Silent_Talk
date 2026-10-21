"""Synthetic tests only: no real student forward, GPU use, or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import evaluation as subject


def posterior(ids):
    value = np.full((len(ids), 69), -10., dtype=np.float32)
    value[np.arange(len(ids)), ids] = 0.
    return value


class ToyStudent(nn.Module):
    def __init__(self, mode="invalid"):
        super().__init__()
        self.register_buffer("marker", torch.tensor(1.))
        self.mode = mode

    def forward(self, pixels):
        if self.mode == "fail":
            raise RuntimeError("synthetic failure")
        if self.mode == "mutate":
            self.marker.add_(1.)
        ids = [0, 0, 0, 0] if self.mode == "blank" else [20, 20, 0, 20]
        output = torch.from_numpy(posterior(ids)).unsqueeze(1)
        return output * float("nan") if self.mode == "nan" else output


def fixture(count=3, source_split="val"):
    rows = [{"id": f"dev_{i}", "speaker_id": f"speaker_{i % 2}", "source_stem": f"recording_{i}",
             "split": "dev", "source_split": source_split, "text": "가 나"} for i in range(count)]
    clips = {row["id"]: {**row, "pixels": np.zeros((4, 96, 96, 3), dtype=np.uint8), "tokens": [1, 20]}
             for row in rows}
    return {"ids": [row["id"] for row in rows], "rows": rows, "clips": clips}


def evaluate(loaded=None, mode="invalid"):
    return subject.evaluate_clips(ToyStudent(mode).eval(), loaded or fixture(), "new24", torch.device("cpu"))


def previous(result):
    result = copy.deepcopy(result)
    for condition in result["conditions"].values():
        for row in condition["records"]:
            row["strict_replay"] = row.pop("strict_label_decoder_diagnostic")
    return result


def metadata():
    old = [f"old{i}" for i in range(71)]
    new = [f"new{i}" for i in range(24)]
    return {"old71_ids": old, "new24_ids": new, "new23_ids": new[1:], "excluded_duplicate_ids": new[:1],
            "overlap_evidence": [{"train_id": "train1", "val_id": new[0], "train_text": "가 나", "val_text": "가나"}],
            "bindings": []}


def target_fixture(variant="expanded510"):
    count = 510 if variant == "expanded510" else 441
    ids = [f"train{i}" for i in range(count)]
    plan = {"seed": 42, "updates": 3200, "batch_size": 2, "variant": variant,
            "schedule": [[ids[2 * i % count], ids[(2 * i + 1) % count]] for i in range(3200)],
            "runtime": {"device": "synthetic"},
            "training_data": {"data_sha256": "data_digest", "clips": {sid: {} for sid in ids}}}
    plan["plan_sha256"] = subject.runner.canonical_plan_hash(plan)
    state = {"model": {"weight": torch.tensor([1.])}, "adapter": {}, "optimizer": {}, "rng": {}}
    hashes = subject.runner.state_hashes(state)
    identity = {"seed": 42, "total_updates": 3200, "batch_size": 2, "arm": "B_feature",
                "plan_sha256": plan["plan_sha256"], "schedule_sha256": subject.runner.nested_hash(plan["schedule"]),
                "data_sha256": "data_digest", "clip_bindings": plan["training_data"]["clips"],
                "code_bindings": subject.runner._code_bindings(), "runtime": plan["runtime"],
                "optimization": {"optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
                    "weight_decay": 0., "gradient_clip": 1., "accumulation_clips": 2, "lambda": 1.,
                    "ctc_device": "cpu", "precision": "float32"}}
    history = [{"update": i + 1, "ids": batch, "state_hashes": hashes} for i, batch in enumerate(plan["schedule"])]
    payload = {"schema": subject.runner.SCHEMA, "completed_updates": 3200, "identity": identity,
               "history": history, "state": state, "state_hashes": hashes, "initial_state_hashes": {"model": "fresh"}}
    result = {"schema": "local-paired-training-segment-result-v1", "completed_updates": 3200, "variant": variant,
              "arm": "B_feature", "plan_sha256": plan["plan_sha256"], "identity_sha256": subject.runner.nested_hash(identity),
              "runtime": identity["runtime"], "final_state_hashes": hashes, "plan_file": {"path": "plan", "sha256": "plan_sha"},
              "checkpoint": {"state_hashes": hashes, "completed_updates": 3200}, "initial_state_hashes": {"model": "fresh"},
              "experiment_plan": {"path": "experiment", "sha256": "experiment_sha"}}
    target = {"plan": plan, "result": result, "plan_file": copy.deepcopy(result["plan_file"]),
              "experiment_plan": copy.deepcopy(result["experiment_plan"]), "initial_state_hashes_expected": {"model": "fresh"}}
    return target, payload


class EvaluationTests(unittest.TestCase):
    def test_valid_jamo_matches_strict_and_invalids_are_retained(self):
        valid = subject.decoder.decode_prediction(posterior([1, 1, 20, 20, 0, 1, 20]))
        self.assertEqual(valid["prediction"], "가가")
        self.assertEqual(valid["prediction"], valid["strict_replay"]["prediction"])
        invalid = subject.decoder.decode_prediction(posterior([20, 20, 0, 20]))
        self.assertEqual(invalid["collapsed_ids"], [20, 20])
        self.assertEqual(invalid["prediction"], "\ufffd\ufffd")
        self.assertEqual(invalid["replacement_characters"], 2)
        self.assertEqual(invalid["strict_replay"]["status"], "failed")

    def test_nfc_internal_spaces_wer_and_pooled_denominators(self):
        a, b = subject.core.counts(" 가  나 ", "가 나"), subject.core.counts("가나다라마", "가나다라마")
        total = subject.core.aggregate([{"id": "a", "status": "ok", "metrics": a}, {"id": "b", "status": "ok", "metrics": b}])
        self.assertEqual((a["character_edits"], a["reference_characters"], a["word_edits"], a["normalized_character_edits"]), (1, 4, 0, 0))
        self.assertAlmostEqual(total["cer_percent"], 100 / 9)
        self.assertEqual((total["exact_sentences"], total["normalized_exact_sentences"]), (1, 2))
        self.assertEqual(subject.core.counts("가", "\ufffd\ufffd")["character_edits"], 2)

    def test_conditions_are_id_determined_and_seed_independent(self):
        pixels = np.arange(19)[:, None]
        normal = subject.original.condition_metadata(pixels, "normal", "same")
        static = subject.original.condition_metadata(pixels, "static", "same")
        shuffle = subject.original.condition_metadata(pixels, "shuffled", "same")
        np.random.seed(43)
        self.assertEqual(shuffle, subject.original.condition_metadata(pixels, "shuffled", "same"))
        self.assertEqual(normal["source_positions"], list(range(19)))
        self.assertEqual(static["source_positions"], [9] * 19)
        self.assertEqual(sorted(shuffle["source_positions"]), list(range(19)))
        self.assertNotEqual(shuffle["source_positions"], list(range(19)))

    def test_all_rows_conditions_raw_tokens_and_state_survive(self):
        result = evaluate(fixture(24))
        self.assertTrue(result["model_state_unchanged"])
        for condition in result["conditions"].values():
            self.assertEqual(condition["aggregate"]["clips"], 24)
            self.assertEqual(condition["aggregate"]["failed_inputs"], 0)
            self.assertEqual(condition["replacement_characters"], 48)
            self.assertEqual(set(condition["per_speaker"]), {"speaker_0", "speaker_1"})
            for row in condition["records"]:
                self.assertEqual(row["source_split"], "val")
                self.assertEqual(row["raw_argmax_ids"], [20, 20, 0, 20])

    def test_all_blank_is_successful_empty_not_failure(self):
        for condition in evaluate(mode="blank")["conditions"].values():
            self.assertEqual(condition["aggregate"]["failed_inputs"], 0)
            self.assertTrue(all(row["prediction"] == "" and row["status"] == "ok" for row in condition["records"]))

    def test_failed_and_nonfinite_inference_keeps_all_denominators(self):
        for mode in ("fail", "nan"):
            for value in evaluate(mode=mode)["conditions"].values():
                self.assertEqual(value["aggregate"]["failed_inputs"], 3)
                self.assertEqual(value["aggregate"]["reference_characters"], 9)
                self.assertEqual(value["aggregate"]["cer_percent"], 100.)
                self.assertTrue(all(row["prediction"] == "" and row["raw_argmax_ids"] == [] for row in value["records"]))

    def test_missing_and_malformed_inputs_remain_rows(self):
        loaded = fixture()
        loaded["clips"].pop("dev_0")
        loaded["clips"]["dev_1"]["pixels"] = np.zeros((2, 3), dtype=np.uint8)
        for value in evaluate(loaded)["conditions"].values():
            self.assertEqual(value["aggregate"]["failed_inputs"], 2)
            self.assertEqual([row["id"] for row in value["records"]], loaded["ids"])

    def test_mutation_training_mode_and_grad_rejected(self):
        with self.assertRaisesRegex(ValueError, "changed student state"):
            evaluate(mode="mutate")
        for model in (ToyStudent(), nn.Linear(1, 1).eval()):
            with self.assertRaisesRegex(ValueError, "frozen eval"):
                subject.evaluate_clips(model, fixture(), "new24", torch.device("cpu"))

    def test_fixed_subset_reaggregates_without_inference(self):
        parent = evaluate(fixture(24))
        excluded, selected = parent["ids"][:1], parent["ids"][1:]
        subset = subject.subset_cohort(parent, selected, excluded)
        self.assertTrue(subset["aggregation_only"])
        for key in subject.CONDITIONS:
            self.assertEqual(subset["conditions"][key]["aggregate"]["clips"], 23)
            self.assertEqual(subset["conditions"][key]["records"], parent["conditions"][key]["records"][1:])
        with self.assertRaises(ValueError):
            subject.subset_cohort(parent, selected[::-1], excluded)

    def test_both_primary_cohorts_infer95_and_subset_adds_no_forward(self):
        model = ToyStudent().eval()
        old = subject.evaluate_clips(model, fixture(71, "dev"), "old71", torch.device("cpu"))
        new = subject.evaluate_clips(model, fixture(24, "val"), "new24", torch.device("cpu"))
        self.assertEqual(old["record_count"] + new["record_count"], 95)
        self.assertEqual(sum(len(v["records"]) for c in (old, new) for v in c["conditions"].values()), 285)
        self.assertEqual(old["model_state_after"], new["model_state_before"])

    def test_failed_replay_is_saved_before_raising_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            value = {"status": "failed_control_replay", "old_control_replay": {"mismatches": [{"field": "raw_argmax_ids"}]}}
            with self.assertRaisesRegex(ValueError, "new results preserved"):
                subject.save_result(output, value)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)
            with self.assertRaisesRegex(ValueError, "already exists"):
                subject.save_result(output, {"status": "completed"})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), value)

    def test_replay_compares_raw_ids_status_text_metrics_temporal(self):
        current = evaluate()
        prior = previous(current)
        self.assertEqual(subject.control_replay(current, prior)["status"], "passed")
        for key, replacement in (("raw_argmax_ids", []), ("collapsed_ids", []), ("status", "failed"),
                                 ("prediction", "changed"), ("metrics", {}), ("temporal", {}), ("strict_replay", {})):
            changed = copy.deepcopy(prior)
            changed["conditions"]["normal"]["records"][0][key] = replacement
            result = subject.control_replay(current, changed)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["prior_predictions_used_as_outputs"])
            self.assertEqual(current["conditions"]["normal"]["records"][0]["prediction"], "\ufffd\ufffd")

    def test_metadata_enforces_fixed_roles_order_and_overlap(self):
        subject.validate_metadata(metadata())
        for field, replacement in (("new23_ids", ["bad"] * 23), ("excluded_duplicate_ids", ["old0"]),
                                   ("old71_ids", ["old0"] * 71)):
            changed = metadata()
            changed[field] = replacement
            with self.assertRaises(ValueError):
                subject.validate_metadata(changed)
        changed = metadata()
        changed["overlap_evidence"][0]["val_text"] = "다른 문장"
        with self.assertRaisesRegex(ValueError, "overlap evidence"):
            subject.validate_metadata(changed)

    def test_loaded_guard_rejects_teacher_audio_and_source_role(self):
        loaded = fixture()
        contract = {"development": {"new24_ids": loaded["ids"], "new23_ids": loaded["ids"][1:], "excluded_duplicate_ids": loaded["ids"][:1]}}
        loaded.update(supplemental_ids=loaded["ids"][1:], excluded_duplicate_ids=loaded["ids"][:1])
        subject.validate_loaded(loaded, "new24", contract)
        for key, value in (("teacher_features", None), ("audio", None), ("source_split", "train")):
            changed = copy.deepcopy(loaded)
            changed["clips"]["dev_0"][key] = value
            with self.assertRaises(ValueError):
                subject.validate_loaded(changed, "new24", contract)

    def test_exact_final_targets_for_both_variants(self):
        for variant in subject.VARIANTS:
            target, payload = target_fixture(variant)
            subject.validate_final_target(target, variant, 42, payload)

    def test_early_checkpoint_wrong_seed_changed_state_and_init_rejected(self):
        for mutation in ("early", "seed", "state", "init", "plan_file", "experiment", "history", "runtime"):
            target, payload = target_fixture()
            if mutation == "early": payload["completed_updates"] = 3199
            elif mutation == "seed": payload["identity"]["seed"] = 43
            elif mutation == "state": payload["state"]["model"]["weight"].add_(1)
            elif mutation == "init": target["initial_state_hashes_expected"] = {"model": "different"}
            elif mutation == "plan_file": target["plan_file"]["sha256"] = "different"
            elif mutation == "experiment": target["experiment_plan"]["sha256"] = "different"
            elif mutation == "history": payload["history"].pop()
            else: target["result"]["runtime"] = {"changed": True}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.validate_final_target(target, "expanded510", 42, payload)

    def test_wrong_cohort_variant_rejected_even_if_payload_itself_valid(self):
        target, payload = target_fixture("control_old441")
        with self.assertRaisesRegex(ValueError, "training cohort"):
            subject.validate_final_target(target, "expanded510", 42, payload)


if __name__ == "__main__":
    unittest.main()
