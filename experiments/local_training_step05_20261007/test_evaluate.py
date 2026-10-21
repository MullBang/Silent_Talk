"""Focused tests for final-only, full-cohort development evaluation."""
from __future__ import annotations

import copy
import unittest

import numpy as np
import torch
from torch import nn

import evaluate as subject


class EmptyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("identity", torch.tensor(1.))

    def forward(self, pixels):
        posterior = torch.full((pixels.shape[2], 1, 69), -20.)
        posterior[:, :, 0] = 0.
        return posterior


class FailingStudent(EmptyStudent):
    def forward(self, pixels):
        raise RuntimeError("synthetic inference failure")


class MutatingStudent(EmptyStudent):
    def forward(self, pixels):
        self.identity.add_(1.)
        return super().forward(pixels)


def fixture():
    rows = [{"id": "one", "speaker_id": "C001", "source_stem": "recording_one", "split": "dev", "text": "가  나"},
            {"id": "two", "speaker_id": "C002", "source_stem": "recording_two", "split": "dev", "text": "가나다라마"}]
    clips = {row["id"]: {**row, "pixels": np.repeat(np.arange(7, dtype=np.uint8)[:, None, None, None],
                                                  96 * 96 * 3, axis=1).reshape(7, 96, 96, 3), "tokens": [1, 2]}
             for row in rows}
    return rows, clips


def payload_fixture():
    plan = {"seed": 42, "updates": 3200, "batch_size": 2,
            "schedule": [["train_one", "train_two"] for _ in range(3200)],
            "runtime": {"device": "synthetic"},
            "training_data": {"data_sha256": "data_digest", "clips": {"train_one": {}, "train_two": {}}}}
    plan["plan_sha256"] = subject.runner.canonical_plan_hash(plan)
    state = {"model": {"weight": torch.tensor([1.])}, "adapter": {}, "optimizer": {}, "rng": {}}
    hashes = subject.runner.state_hashes(state)
    identity = {"seed": 42, "total_updates": 3200, "batch_size": 2, "arm": "A_ctc",
                "plan_sha256": plan["plan_sha256"], "schedule_sha256": subject.runner.nested_hash(plan["schedule"]),
                "data_sha256": "data_digest", "clip_bindings": plan["training_data"]["clips"],
                "code_bindings": subject.runner._code_bindings(), "runtime": {"device": "synthetic"},
                "optimization": {"optimizer": "Adam", "lr": .001, "betas": [.9, .999], "eps": 1e-8,
                    "weight_decay": 0., "gradient_clip": 1., "accumulation_clips": 2, "lambda": 0.,
                    "ctc_device": "cpu", "precision": "float32"}}
    history = [{"update": i + 1, "ids": batch, "state_hashes": hashes}
               for i, batch in enumerate(plan["schedule"])]
    payload = {"schema": subject.runner.SCHEMA, "completed_updates": 3200, "identity": identity,
               "history": history, "state": state, "state_hashes": hashes}
    result = {"schema": "local-paired-training-segment-result-v1", "completed_updates": 3200,
              "arm": "A_ctc", "plan_sha256": plan["plan_sha256"], "identity_sha256": subject.runner.nested_hash(identity),
              "runtime": identity["runtime"], "final_state_hashes": hashes,
              "checkpoint": {"state_hashes": hashes, "completed_updates": 3200}}
    return plan, result, payload


class EvaluationTests(unittest.TestCase):
    def test_primary_metric_preserves_spaces_and_nfc_and_is_pooled(self):
        first = subject.core.counts("가  나", "가 나")
        second = subject.core.counts("가나다라마", "가나다라마")
        total = subject.core.aggregate([{"id": "1", "status": "ok", "metrics": first},
                                        {"id": "2", "status": "ok", "metrics": second}])
        self.assertEqual(first["character_edits"], 1)
        self.assertEqual(first["reference_characters"], 4)
        self.assertEqual(first["normalized_character_edits"], 0)
        self.assertEqual(first["word_edits"], 0)
        self.assertEqual(total["reference_characters"], 9)
        self.assertAlmostEqual(total["cer_percent"], 100 / 9)
        self.assertEqual(total["exact_sentences"], 1)
        self.assertEqual(total["normalized_exact_sentences"], 2)

    def test_conditions_preserve_length_and_use_same_seed_independent_order(self):
        pixels = np.arange(19)[:, None]
        normal = subject.condition_metadata(pixels, "normal", "same_sentence")
        static = subject.condition_metadata(pixels, "static", "same_sentence")
        shuffle = subject.condition_metadata(pixels, "shuffled", "same_sentence")
        np.random.seed(13)
        repeated = subject.condition_metadata(pixels, "shuffled", "same_sentence")
        self.assertEqual(normal["source_positions"], list(range(19)))
        self.assertEqual(static["source_positions"], [9] * 19)
        self.assertEqual(shuffle, repeated)
        self.assertEqual(sorted(shuffle["source_positions"]), list(range(19)))
        self.assertNotEqual(shuffle["source_positions"], normal["source_positions"])
        with self.assertRaises(ValueError):
            subject.condition_metadata(pixels, "unregistered", "same_sentence")

    def test_all_records_and_conditions_retained_with_empty_predictions(self):
        rows, clips = fixture()
        result = subject.evaluate_clips(EmptyStudent().eval(), clips, rows, torch.device("cpu"))
        self.assertTrue(result["model_state_unchanged"])
        for condition in subject.CONDITIONS:
            value = result["conditions"][condition]
            self.assertEqual([row["id"] for row in value["records"]], ["one", "two"])
            self.assertEqual(value["aggregate"]["clips"], 2)
            self.assertEqual(value["aggregate"]["failed_inputs"], 0)
            self.assertEqual(value["aggregate"]["cer_percent"], 100.)
            self.assertEqual(set(value["per_speaker"]), {"C001", "C002"})
            self.assertTrue(all(row["prediction"] == "" for row in value["records"]))

    def test_failed_inference_remains_in_every_denominator(self):
        rows, clips = fixture()
        result = subject.evaluate_clips(FailingStudent().eval(), clips, rows, torch.device("cpu"))
        for value in result["conditions"].values():
            self.assertEqual(value["aggregate"]["failed_inputs"], 2)
            self.assertEqual(value["aggregate"]["reference_characters"], 9)
            self.assertEqual(value["aggregate"]["cer_percent"], 100.)
            self.assertTrue(all(row["prediction"] == "" and row["error"] for row in value["records"]))

    def test_missing_input_and_invalid_shape_are_not_dropped(self):
        rows, clips = fixture()
        clips.pop("one")
        clips["two"]["pixels"] = np.zeros((2, 3), dtype=np.uint8)
        result = subject.evaluate_clips(EmptyStudent().eval(), clips, rows, torch.device("cpu"))
        self.assertTrue(all(value["aggregate"]["failed_inputs"] == 2 for value in result["conditions"].values()))

    def test_model_state_mutation_and_training_mode_rejected(self):
        rows, clips = fixture()
        with self.assertRaisesRegex(ValueError, "changed student"):
            subject.evaluate_clips(MutatingStudent().eval(), clips, rows, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "eval mode"):
            subject.evaluate_clips(EmptyStudent(), clips, rows, torch.device("cpu"))

    def test_dev_only_and_no_unexpected_or_duplicate_rows(self):
        rows, clips = fixture()
        with self.assertRaises(ValueError):
            subject.evaluate_clips(EmptyStudent().eval(), clips, rows + [rows[0]], torch.device("cpu"))
        changed = copy.deepcopy(rows)
        changed[0]["split"] = "train"
        with self.assertRaises(ValueError):
            subject.evaluate_clips(EmptyStudent().eval(), clips, changed, torch.device("cpu"))
        clips["third"] = clips["one"]
        with self.assertRaises(ValueError):
            subject.evaluate_clips(EmptyStudent().eval(), clips, rows, torch.device("cpu"))

    def test_exact_final_checkpoint_contract_accepted(self):
        subject.validate_final_payload(*payload_fixture())

    def test_early_or_other_seed_arm_checkpoint_rejected(self):
        for target, key, value in (("payload", "completed_updates", 3199), ("identity", "seed", 43),
                                    ("identity", "arm", "B_feature"), ("identity", "total_updates", 3300)):
            with self.subTest(key=key):
                plan, result, payload = payload_fixture()
                (payload if target == "payload" else payload["identity"])[key] = value
                with self.assertRaises(ValueError):
                    subject.validate_final_payload(plan, result, payload)

    def test_changed_schedule_data_code_and_state_rejected(self):
        for mutation in ("schedule", "data", "code", "state", "history", "result_state"):
            with self.subTest(mutation=mutation):
                plan, result, payload = payload_fixture()
                if mutation == "schedule":
                    payload["identity"]["schedule_sha256"] = "wrong"
                elif mutation == "data":
                    payload["identity"]["data_sha256"] = "wrong"
                elif mutation == "code":
                    payload["identity"]["code_bindings"] = []
                elif mutation == "state":
                    payload["state"]["model"]["weight"].add_(.5)
                elif mutation == "history":
                    payload["history"] = payload["history"][:-1]
                else:
                    result["final_state_hashes"] = {}
                with self.assertRaises(ValueError):
                    subject.validate_final_payload(plan, result, payload)

    def test_declared_plan_digest_change_rejected(self):
        plan, result, payload = payload_fixture()
        plan["plan_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            subject.validate_final_payload(plan, result, payload)

    def test_plan_runtime_must_match_checkpoint_runtime(self):
        plan, result, payload = payload_fixture()
        payload["identity"]["runtime"] = {"device": "altered"}
        result["runtime"] = payload["identity"]["runtime"]
        result["identity_sha256"] = subject.runner.nested_hash(payload["identity"])
        with self.assertRaisesRegex(ValueError, "runtime"):
            subject.validate_final_payload(plan, result, payload)

    def test_optimization_change_rejected(self):
        plan, result, payload = payload_fixture()
        payload["identity"]["optimization"]["lr"] = .01
        result["identity_sha256"] = subject.runner.nested_hash(payload["identity"])
        with self.assertRaises(ValueError):
            subject.validate_final_payload(plan, result, payload)


if __name__ == "__main__":
    unittest.main()
