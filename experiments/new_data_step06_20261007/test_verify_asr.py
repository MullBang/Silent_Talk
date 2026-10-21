"""Focused mutation tests for the independent saved-ASR consistency audit."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import verify_asr as v


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = v.read(v.BASE / "asr_plan.json")
        cls.plan_ref = v.bind(v.BASE / "asr_plan.json")
        cls.results = v.read(v.BASE / "results/asr_results.json")
        cls.media = v.read(v.BASE / "results/media_manifest.json")
        cls.decoder = v.ByteTokens(v.read(Path(cls.plan["model"]["directory"]) / "tokenizer.json"))

    def record(self):
        return copy.deepcopy(self.results["records"][0])

    def check(self, row):
        v.check_record(row, self.media["clips"][0], self.plan_ref, self.decoder)

    def test_independent_import_boundary(self):
        parsed = ast.parse(Path(v.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.add((node.module or "").split(".")[0])
        self.assertFalse(names & {"asr_audit", "classify_asr_differences", "torch", "numpy", "tokenizers", "faster_whisper", "ctranslate2"})

    def test_normalization_retains_symbols_and_numbers(self):
        self.assertEqual(v.normalize("가,\t나! 12 + �"), "가 나 12 + �")
        self.assertNotEqual(v.normalize("두 개"), v.normalize("2개"))

    def test_edit_metric_known_and_empty_prediction(self):
        self.assertEqual(v.edit_distance("kitten", "sitting"), 3)
        item = v.comparison("가 나", "가나")
        self.assertEqual(item["metrics"]["character_edits"], 1)
        self.assertTrue(item["metrics"]["without_spaces_exact"])
        self.assertEqual(v.comparison("가 나", "")["metrics"]["cer_percent"], 100.)

    def test_nonfinite_duplicate_json(self):
        with tempfile.TemporaryDirectory(dir=v.BASE) as directory:
            path = Path(directory) / "bad.json"
            for content in ('{"value":NaN}', '{"value":1,"value":2}'):
                path.write_text(content, encoding="utf8")
                with self.assertRaises(ValueError):
                    v.read(path)

    def test_boolean_is_not_count(self):
        with self.assertRaises(ValueError):
            v.same(True, 1, "type")

    def test_frozen_plan(self):
        v.check_plan(self.plan)
        for key, value in (("initial_prompt", "정답"), ("condition_on_previous_text", True), ("temperature", .1)):
            modified = copy.deepcopy(self.plan)
            modified["transcribe_options"][key] = value
            with self.assertRaises(ValueError):
                v.check_plan(modified)

    def test_threshold_cannot_relax(self):
        modified = copy.deepcopy(self.plan)
        modified["review_thresholds"]["boundary_margin_seconds"] = 0.
        with self.assertRaises(ValueError):
            v.check_plan(modified)

    def test_real_record_and_byte_tokens(self):
        self.check(self.record())
        self.assertEqual(self.decoder.decode(self.record()["raw_segments"][0]["tokens"]), self.record()["raw_segments"][0]["text"])

    def test_token_or_timestamp_tamper(self):
        for mutation in ("tokens", "start"):
            row = self.record()
            if mutation == "tokens":
                row["raw_segments"][0]["tokens"][1] = 1
            else:
                row["raw_segments"][0]["start"] += .01
            with self.assertRaises(ValueError):
                self.check(row)

    def test_prompt_options_and_duration_tamper(self):
        for field, value in (("initial_prompt", "라벨"), ("prefix", "정답"), ("hotwords", "안녕"), ("suppress_tokens", [])):
            row = self.record()
            row["transcription_info"]["transcription_options"][field] = value
            with self.assertRaises(ValueError):
                self.check(row)
        row = self.record()
        row["transcription_info"]["duration"] += .01
        with self.assertRaises(ValueError):
            self.check(row)

    def test_audio_reference_role_mutations(self):
        for field, value in (("reference", "다른 문장"), ("split", "val"), ("id", "wrong")):
            row = self.record(); row[field] = value
            with self.assertRaises(ValueError):
                self.check(row)
        row = self.record(); row["audio"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.check(row)

    def test_metrics_flags_and_manual_claims(self):
        for case in range(4):
            row = self.record()
            if case == 0:
                row["metrics"]["character_edits"] += 1
            elif case == 1:
                row["boundary_flags"] = []
            elif case == 2:
                row["human_listening_confirmed"] = True
            else:
                row["review_status"] = "automatic_support_only"
            with self.assertRaises(ValueError):
                self.check(row)

    def test_flags_at_exact_thresholds(self):
        segment = {"start": .08, "end": 1.92, "avg_logprob": -1., "no_speech_prob": .6, "compression_ratio": 2.4}
        b, c = v.segment_flags([segment], 2.)
        self.assertEqual(len(b), 2)
        self.assertEqual(c, ["high_segment_no_speech_probability"])
        self.assertEqual(v.segment_flags([], 2.), ([], ["no_segments"]))
        segment.update(start=.081, end=1.919, no_speech_prob=.599)
        self.assertEqual(v.segment_flags([segment], 2.), ([], []))

    def test_nonmonotonic_timestamp_flag(self):
        a = {"start": .1, "end": 1., "avg_logprob": -.1, "no_speech_prob": 0., "compression_ratio": 1.}
        b = dict(a, start=.9, end=1.5)
        self.assertIn("asr_segment_timestamp_outside_or_nonmonotonic", v.segment_flags([a, b], 2.)[0])

    def test_sha_tamper(self):
        with tempfile.TemporaryDirectory(dir=v.BASE) as directory:
            path = Path(directory) / "evidence.bin"
            path.write_bytes(b"original")
            ref = v.bind(path)
            v.verify(ref)
            path.write_bytes(b"changed!")
            with self.assertRaises(ValueError):
                v.verify(ref)

    def test_original_failed_row_denominator(self):
        row = self.record()
        row.update(status="asr_failed", asr_text="", raw_segments=[], transcription_info=None, error="RuntimeError: synthetic", boundary_flags=[], confidence_flags=[])
        row.update(v.comparison(row["reference"], ""))
        row.update(review_reasons=["asr_not_completed", "normalized_transcript_mismatch"], review_status="human_listening_required")
        self.check(row)
        self.assertEqual(v.summary([row])["pooled_cer_percent_including_failed_empty"], 100.)

    def test_saved_full_summary(self):
        v.same(v.summary(self.results["records"]), self.results["summary"], "summary")
        changed = copy.deepcopy(self.results["summary"])
        changed["normalized_exact_matches"] += 1
        with self.assertRaises(ValueError):
            v.same(v.summary(self.results["records"]), changed, "summary")

    def test_number_hints_do_not_replace_content(self):
        self.assertIn("스무", v.number_forms("20"))
        self.assertNotIn("십이", v.number_forms("0012"))
        self.assertEqual(v.number_forms("1.5"), set())
        for reference, prediction, expected in (("2개", "두 개", "possible_number_expression_difference"), ("2.5개", "두 점 오 개", "possible_content_difference"), ("가 나", "가나", "spacing_only_difference")):
            row = {"reference": reference, "asr_text": prediction, "status": "ok", **v.comparison(reference, prediction)}
            self.assertEqual(v.category(row)[0], expected)

    def test_triage_link_and_mutation(self):
        triage = v.read(v.BASE / "results/asr_difference_review.json")
        result_ref = v.bind(v.BASE / "results/asr_results.json")
        v.check_triage(triage, self.results, result_ref)
        triage["records"][0]["original_asr_review_status"] = "automatic_support_only"
        with self.assertRaises(ValueError):
            v.check_triage(triage, self.results, result_ref)


if __name__ == "__main__":
    unittest.main()
