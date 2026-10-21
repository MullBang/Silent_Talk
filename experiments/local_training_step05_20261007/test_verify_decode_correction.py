"""Independent saved-evidence audit tests, including deliberate tampering."""
import copy
import sys
import unittest
import unicodedata

import verify_decode_correction as subject


def fixture(condition="normal"):
    sources, records, previous = [], [], []
    for index in range(3):
        source = {"id": f"dev_{index}", "speaker_id": "C001" if index < 2 else "C002",
                  "source_stem": "recording", "frames": 6, "text": "가가"}
        sources.append(source)
        raw = [1, 20, 0, 1, 20, 0] if index != 1 else [20, 20, 0, 20, 0, 0]
        prediction = "가가" if index != 1 else "\ufffd\ufffd"
        strict = {"status": "ok", "prediction": "가가", "error": None} if index != 1 else {
            "status": "failed", "prediction": None, "error": "ValueError: 잘못된 자모 순서/토큰 (위치 0): 'ᅡ'"}
        positions = subject.old_audit.temporal_order(6, condition, source["id"])
        temporal = {"frames": 6, "source_positions": positions,
                    "source_positions_sha256": subject.old_audit.nested_hash(positions),
                    "static_position": 3 if condition == "static" else None,
                    "shuffle_policy": "stable"}
        old = {"id": source["id"], "speaker_id": source["speaker_id"], "source_stem": "recording",
               "split": "dev", "condition": condition, "reference": "가가", "frames": 6,
               "temporal": temporal, "seconds": 0.01, "status": strict["status"],
               "prediction": "가가" if index != 1 else "", "error": strict["error"]}
        old["metrics"] = subject.old_audit.metric_counts("가가", old["prediction"])
        previous.append(old)
        record = {**copy.deepcopy(old), "raw_argmax_ids": raw,
                  "collapsed_ids": [1, 20, 1, 20] if index != 1 else [20, 20],
                  "prediction": prediction, "replacement_characters": 0 if index != 1 else 2,
                  "strict_replay": strict, "strict_replay_matches_original": True,
                  "original_success_prediction_match": True if index != 1 else None,
                  "status": "ok", "error": None,
                  "correction": "unchanged_valid_prediction" if index != 1 else
                                "invalid_predicted_jamo_retained_as_replacement_character",
                  "original": {key: old[key] for key in ("status", "prediction", "error", "metrics")},
                  "metrics": subject.old_audit.metric_counts("가가", prediction)}
        records.append(record)
    def condition_data(rows):
        return {"records": rows, "aggregate": subject.old_audit.aggregate_metrics(rows),
                "per_speaker": {sp: subject.old_audit.aggregate_metrics([r for r in rows if r["speaker_id"] == sp])
                                for sp in ("C001", "C002")}}
    old_data, data = condition_data(previous), condition_data(records)
    data.update({"original_aggregate": old_data["aggregate"], "posthoc_minus_original_cer_pp": 0.,
                 "corrected_invalid_jamo_failures": 1, "unchanged_original_successes": 2,
                 "sentences_with_replacement_characters": 1, "replacement_characters": 2})
    return data, old_data, sources


class IndependentTokenTests(unittest.TestCase):
    def test_blank_and_repeat_order(self):
        self.assertEqual(subject.collapse([0, 20, 20, 0, 20, 0, 1, 1], 8), [20, 20, 1])
        self.assertEqual(subject.collapse([0, 0, 0], 3), [])

    def test_raw_length_type_and_range(self):
        for raw, frames in (([], 0), ([1], 2), ([True], 1), ([-1], 1), ([69], 1), ([1.], 1), ((1,), 1)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                subject.collapse(raw, frames)

    def test_syllable_boundaries_and_invalids(self):
        value = subject.reconstruct([1, 20, 41, 68, 19, 40, 67, 20, 1, 68, 41])
        self.assertEqual(value["prediction"], "각 힣\ufffd\ufffd \ufffd")
        self.assertEqual(value["replacement_characters"], 3)
        self.assertEqual(value["strict_replay"], {"status": "failed", "prediction": None,
            "error": "ValueError: 잘못된 자모 순서/토큰 (위치 7): 'ᅡ'"})

    def test_all_modern_hangul_syllables_against_unicode_decomposition(self):
        for point in range(0xAC00, 0xD7A4):
            parts = unicodedata.normalize("NFD", chr(point))
            ids = [ord(parts[0]) - 0x1100 + 1, ord(parts[1]) - 0x1161 + 20]
            if len(parts) == 3:
                ids.append(ord(parts[2]) - 0x11A8 + 41)
            value = subject.reconstruct(ids)
            self.assertEqual(value["prediction"], chr(point))
            self.assertEqual(value["strict_replay"]["status"], "ok")

    def test_empty_spaces_and_repeated_unknowns_not_deleted(self):
        self.assertEqual(subject.reconstruct([])["prediction"], "")
        self.assertEqual(subject.reconstruct([68, 68])["prediction"], "  ")
        self.assertEqual(subject.reconstruct([20, 21, 20])["prediction"], "\ufffd\ufffd\ufffd")
        self.assertEqual(subject.old_audit.metric_counts("가", "\ufffd\ufffd")["character_edits"], 2)

    def test_collapsed_blank_and_wrong_type_rejected(self):
        for tokens in ([0], [69], [True], [1.], [-1]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                subject.reconstruct(tokens)


class RowAuditTests(unittest.TestCase):
    def test_all_three_conditions_and_metrics(self):
        for condition in subject.CONDITIONS:
            value, previous, sources = fixture(condition)
            result = subject.validate_condition(value, previous, sources, condition)
            self.assertEqual(result["aggregate"]["clips"], 3)
            self.assertEqual(result["aggregate"]["failed_inputs"], 0)
            self.assertEqual(result["replacement_characters"], 2)

    def test_tampered_raw_collapse_prediction_replacement_and_strict_evidence(self):
        alterations = {
            "raw_argmax_ids": [20, 20, 0, 21, 0, 0], "collapsed_ids": [20],
            "prediction": "\ufffd", "replacement_characters": 1,
            "strict_replay": {"status": "ok", "prediction": "", "error": None},
            "strict_replay_matches_original": False, "original_success_prediction_match": True,
            "correction": "unchanged_valid_prediction"}
        for key, bad in alterations.items():
            value, previous, sources = fixture()
            value["records"][1][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_row(value["records"][1], previous["records"][1], sources[1], "normal")

    def test_original_success_cannot_change_or_fail(self):
        for changed in ({"prediction": "나"}, {"status": "failed", "error": "RuntimeError"},
                        {"original_success_prediction_match": None}, {"raw_argmax_ids": []}):
            value, previous, sources = fixture()
            value["records"][0].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                subject.validate_row(value["records"][0], previous["records"][0], sources[0], "normal")

    def test_original_error_and_snapshot_cannot_change(self):
        value, previous, sources = fixture()
        value["records"][1]["original"]["error"] += " changed"
        with self.assertRaises(ValueError):
            subject.validate_row(value["records"][1], previous["records"][1], sources[1], "normal")
        value, previous, sources = fixture()
        previous["records"][1]["error"] += " changed"
        value["records"][1]["original"]["error"] = previous["records"][1]["error"]
        with self.assertRaises(ValueError):
            subject.validate_row(value["records"][1], previous["records"][1], sources[1], "normal")

    def test_unknown_original_runtime_failure_rejected_from_this_audit_scope(self):
        value, previous, sources = fixture()
        previous["records"][1]["error"] = "RuntimeError: missing input"
        value["records"][1]["original"]["error"] = previous["records"][1]["error"]
        with self.assertRaisesRegex(ValueError, "separate audit scope"):
            subject.validate_row(value["records"][1], previous["records"][1], sources[1], "normal")

    def test_missing_reordered_rows_metrics_and_diagnostics_tamper(self):
        for change in (lambda v: v["records"].pop(), lambda v: v["records"].reverse(),
                       lambda v: v["aggregate"].update(cer_percent=0),
                       lambda v: v["records"][1]["metrics"].update(character_edits=0),
                       lambda v: v.update(corrected_invalid_jamo_failures=0),
                       lambda v: v.update(posthoc_minus_original_cer_pp=-10),
                       lambda v: v["per_speaker"]["C001"].update(wer_percent=0)):
            value, previous, sources = fixture()
            change(value)
            with self.assertRaises(ValueError):
                subject.validate_condition(value, previous, sources, "normal")

    def test_changed_source_temporal_reference_or_denominator(self):
        for changed in ({"reference": "나"}, {"frames": 5}, {"condition": "static"}, {"split": "train"}):
            value, previous, sources = fixture()
            value["records"][1].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                subject.validate_condition(value, previous, sources, "normal")


class ActualArtifactTests(unittest.TestCase):
    def test_original_audit_still_exact_and_only_independent_helper_imported(self):
        audit = subject.read(subject.ORIGINAL_AUDIT, subject.ORIGINAL_AUDIT_SHA)
        subject.verify(audit["code"], subject.BASE / "verify_step05.py")
        self.assertEqual(audit["status"], "passed")
        self.assertNotIn("evaluate_prediction_decode", sys.modules)
        self.assertNotIn("backend.models.tokenizer", sys.modules)

    def test_actual_new_provenance_and_tampered_bindings(self):
        plan = subject.read(subject.PLAN)
        audit = subject.read(subject.ORIGINAL_AUDIT, subject.ORIGINAL_AUDIT_SHA)
        entry = plan["models"][0]
        value = subject.read(entry["output"])
        previous = subject.read(entry["original_evaluation"]["path"])
        run = audit["runs"][0]
        args = previous, entry, subject.bind(subject.PLAN), run, plan["created_utc"]
        subject.validate_provenance(value, *args)
        for key, bad in (("model_state_after", "bad"), ("new_optimizer_updates", 1),
                         ("official_frozen_results_replaced", True), ("checkpoint", {}),
                         ("ids", value["ids"][:-1]), ("data_bindings", []), ("runtime", {}),
                         ("conditions", {"normal": value["conditions"]["normal"]}),
                         ("created_utc", "2020-01-01T00:00:00+00:00")):
            changed = copy.deepcopy(value)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                subject.validate_provenance(changed, *args)

    def test_plan_missing_model_and_rebound_code_rejected(self):
        plan = subject.read(subject.PLAN)
        audit = subject.read(subject.ORIGINAL_AUDIT, subject.ORIGINAL_AUDIT_SHA)
        changed = copy.deepcopy(plan)
        changed["models"].pop()
        with self.assertRaises(ValueError):
            subject.validate_posthoc_plan(changed, audit)
        changed = copy.deepcopy(plan)
        changed["code_bindings"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            subject.validate_posthoc_plan(changed, audit)


if __name__ == "__main__":
    subject.old_audit.torch.set_num_threads(1)
    unittest.main()
