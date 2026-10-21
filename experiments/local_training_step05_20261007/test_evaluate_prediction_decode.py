"""Synthetic CTC/post-hoc decoder tests; no real evaluation or optimization."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

import evaluate_prediction_decode as subject


def posterior(ids):
    value = np.full((len(ids), 69), -10., dtype=np.float32)
    value[np.arange(len(ids)), ids] = 0.
    return value


def old_row(prediction="", status="ok", error=None):
    return {"prediction": prediction, "status": status, "error": error}


class BadJamoStudent(nn.Module):
    def __init__(self, mutate=False):
        super().__init__()
        self.register_buffer("marker", torch.tensor(1.))
        self.mutate = mutate

    def forward(self, inputs):
        if self.mutate:
            self.marker.add_(1.)
        ids = [20, 20, 0, 20]
        return torch.from_numpy(posterior(ids)).unsqueeze(1)


def full_fixture():
    rows, clips = [], {}
    conditions = {condition: {"records": []} for condition in subject.original.CONDITIONS}
    decoded = subject.decode_prediction(posterior([20, 20, 0, 20]))
    for index in range(71):
        row = {"id": f"dev_{index:03d}", "speaker_id": "C002", "source_stem": "source", "split": "dev", "text": "가"}
        rows.append(row)
        clip = {**row, "pixels": np.zeros((4, 96, 96, 3), dtype=np.uint8), "tokens": [1, 20]}
        clips[row["id"]] = clip
        for condition in subject.original.CONDITIONS:
            conditions[condition]["records"].append({"id": row["id"], "reference": "가", "prediction": "", "status": "failed",
                "error": decoded["strict_replay"]["error"], "metrics": subject.core.counts("가", ""),
                "temporal": subject.original.condition_metadata(clip["pixels"], condition, row["id"])})
    for condition in conditions.values():
        condition["aggregate"] = subject.core.aggregate(condition["records"])
    previous = {"ids": [row["id"] for row in rows], "conditions": conditions}
    return rows, clips, previous


class PredictionDecodeTests(unittest.TestCase):
    def test_valid_jamo_matches_original_strict_exactly(self):
        value = subject.decode_prediction(posterior([1, 1, 20, 20, 0, 1, 20]))
        self.assertEqual(value["raw_argmax_ids"], [1, 1, 20, 20, 0, 1, 20])
        self.assertEqual(value["collapsed_ids"], [1, 20, 1, 20])
        self.assertEqual(value["prediction"], "가가")
        self.assertEqual(value["replacement_characters"], 0)
        result = subject.compare_original(value, old_row("가가"))
        self.assertTrue(result["strict_replay_matches_original"])
        self.assertTrue(result["original_success_prediction_match"])

    def test_blank_separates_repeat_and_invalid_tokens_are_not_dropped(self):
        value = subject.decode_prediction(posterior([20, 20, 0, 20]))
        self.assertEqual(value["collapsed_ids"], [20, 20])
        self.assertEqual(value["prediction"], "\ufffd\ufffd")
        self.assertEqual(value["replacement_characters"], 2)
        previous = old_row("", "failed", value["strict_replay"]["error"])
        result = subject.compare_original(value, previous)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["strict_replay_matches_original"])
        self.assertEqual(subject.core.counts("가", result["prediction"])["character_edits"], 2)

    def test_all_blank_is_valid_empty_prediction(self):
        value = subject.decode_prediction(posterior([0, 0, 0]))
        self.assertEqual(value["collapsed_ids"], [])
        self.assertEqual(value["prediction"], "")
        self.assertEqual(value["strict_replay"]["status"], "ok")
        self.assertEqual(subject.compare_original(value, old_row())["status"], "ok")

    def test_internal_spaces_and_unknowns_remain(self):
        value = subject.decode_prediction(posterior([1, 20, 68, 0, 68, 20, 2]))
        self.assertEqual(value["prediction"], "가  \ufffd\ufffd")
        self.assertEqual(subject.core.counts("가 나", value["prediction"])["reference_characters"], 3)

    def test_original_success_difference_is_fatal(self):
        value = subject.decode_prediction(posterior([1, 20]))
        with self.assertRaises(subject.ReplayMismatch):
            subject.compare_original(value, old_row("나"))

    def test_original_jamo_failure_must_reproduce_error_exactly(self):
        value = subject.decode_prediction(posterior([20]))
        with self.assertRaises(subject.ReplayMismatch):
            subject.compare_original(value, old_row("", "failed", subject.INVALID_JAMO_PREFIX + " changed"))

    def test_other_original_failures_are_never_automatically_corrected(self):
        value = subject.decode_prediction(posterior([1, 20]))
        result = subject.compare_original(value, old_row("", "failed", "RuntimeError: unavailable input"))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["prediction"], "")
        self.assertFalse(result["strict_replay_matches_original"])

    def test_nonfinite_empty_wrong_shape_rejected(self):
        for value in (np.full((3, 69), np.nan, dtype=np.float32), np.zeros((0, 69)), np.zeros((2, 68))):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                subject.decode_prediction(value)

    def test_full71_three_conditions_preserve_raw_ids_and_model_state(self):
        rows, clips, previous = full_fixture()
        result = subject.evaluate_rows(BadJamoStudent().eval(), clips, rows, previous, torch.device("cpu"))
        self.assertTrue(result["model_state_unchanged"])
        for condition in result["conditions"].values():
            self.assertEqual(condition["aggregate"]["clips"], 71)
            self.assertEqual(condition["aggregate"]["failed_inputs"], 0)
            self.assertEqual(condition["corrected_invalid_jamo_failures"], 71)
            self.assertEqual(condition["replacement_characters"], 142)
            self.assertEqual(condition["aggregate"]["cer_percent"], 200.)
            self.assertEqual(condition["posthoc_minus_original_cer_pp"], 100.)
            self.assertTrue(all(row["raw_argmax_ids"] == [20, 20, 0, 20] for row in condition["records"]))

    def test_changed_state_or_changed_original_error_stops(self):
        rows, clips, previous = full_fixture()
        with self.assertRaises(subject.ReplayMismatch):
            subject.evaluate_rows(BadJamoStudent(mutate=True).eval(), clips, rows, previous, torch.device("cpu"))
        changed = copy.deepcopy(previous)
        changed["conditions"]["normal"]["records"][0]["error"] += " altered"
        with self.assertRaises(subject.ReplayMismatch):
            subject.evaluate_rows(BadJamoStudent().eval(), clips, rows, changed, torch.device("cpu"))

    def test_new_input_failure_stops_instead_of_silently_losing_raw_path(self):
        rows, clips, previous = full_fixture()
        clips.pop(rows[0]["id"])
        with self.assertRaisesRegex(subject.ReplayMismatch, "Cannot reproduce original inference"):
            subject.evaluate_rows(BadJamoStudent().eval(), clips, rows, previous, torch.device("cpu"))

    def test_original_non_decoder_failure_can_remain_failed(self):
        rows, clips, previous = full_fixture()
        clips.pop(rows[0]["id"])
        for condition in previous["conditions"].values():
            condition["records"][0]["error"] = "RuntimeError: original missing input"
        result = subject.evaluate_rows(BadJamoStudent().eval(), clips, rows, previous, torch.device("cpu"))
        for condition in result["conditions"].values():
            self.assertEqual(condition["aggregate"]["clips"], 71)
            self.assertEqual(condition["aggregate"]["failed_inputs"], 1)
            self.assertEqual(condition["records"][0]["prediction"], "")


if __name__ == "__main__":
    unittest.main()
