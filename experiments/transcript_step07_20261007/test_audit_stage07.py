"""Independent scanner, target, timing and mutation audit tests; no inference."""
import ast
import copy
from pathlib import Path
import unittest

import audit_stage07 as a


class Stage7AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = a.read(a.STEP6 / "source_plan.json")
        cls.media = a.read(a.STEP6 / "results/media_manifest.json")
        cls.baseline = a.read(a.STEP6 / "results/asr_results.json")
        cls.context = a.read(a.BASE / "results/context_asr_results.json")
        cls.targets = a.read(a.BASE / "results/label_targets.json")
        cls.overrides = {row["id"]: row for row in a.read(a.BASE / "label_overrides.json")}
        cls.plan = a.bind(a.BASE / "context_plan.json")
        cls.decoder = a.old.ByteTokens(a.read(a.ROOT / "models/asr_audit/faster-whisper-large-v3/tokenizer.json"))
        cls.changed = next(i for i, row in enumerate(cls.targets["records"]) if row["target_status"] == "resolved_alternative_choice")

    def context_check(self, row):
        a.check_context(row, self.media["clips"][0], self.baseline["records"][0], self.plan, self.decoder)

    def target_check(self, row, index=0):
        a.check_target_record(row, self.source["records"][index], self.overrides)

    def test_no_producer_or_model_import(self):
        tree = ast.parse(Path(a.__file__).read_text(encoding="utf8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(set(imports) & {"context_asr", "label_policy", "torch", "tokenizers", "faster_whisper", "numpy"})
        self.assertEqual(a.bind(a.HELPER)["sha256"], a.HELPER_SHA)

    def test_scanner_spans_and_exact_roundtrip(self):
        text = "앞 (애써는)/(애쓰는) 뒤 (하나)/(한 번)."
        parsed = a.parse_independent(text)
        self.assertEqual("".join(row["raw"] for row in parsed["segments"]), text)
        self.assertEqual(len(parsed["alternatives"]), 2)
        for row in parsed["alternatives"]:
            self.assertEqual(text[slice(*row["span"])], row["raw"])
            self.assertEqual([text[slice(*span)] for span in row["option_spans"]], row["options"])

    def test_nested_malformed_and_unicode_annotation_rejected(self):
        for text in ("(가)/(나", "()/ (나)", "(가)/()", "((가))/(나)", "(가)/(나/다)", "（가）／（나）", "가/나", "[가]"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                a.parse_independent(text)

    def test_arithmetic_jamo_and_ctc(self):
        value = a.target("각 가", 100)
        self.assertEqual(value["tokens"], [1, 20, 41, 68, 1, 20])
        self.assertEqual(value["ctc_min_frames"], 6)
        self.assertEqual(a.target("가", 1)["ctc_feasible"], False)
        self.assertEqual(a.target("가!", 10)["tokens"], [1, 20])

    def test_unsupported_text_cannot_be_removed(self):
        for text in ("가 2", "가 A", "가 +", "가 ᄀ", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                a.target(text, 100)

    def test_ordinary_target_real(self):
        self.target_check(self.targets["records"][0])

    def test_user_choice_real_and_scope(self):
        row = self.targets["records"][self.changed]
        self.target_check(row, self.changed)
        self.assertEqual(row["canonical_target"].count("애쓰는"), 1)
        self.assertNotIn("애써는", row["canonical_target"])
        self.assertEqual(row["previous_target"]["ctc_min_frames"], 51)
        self.assertEqual(row["ctc_min_frames"], 43)
        self.assertFalse(row["full_transcript_human_approved"])

    def test_wrong_candidate_or_repeated_alternatives_rejected(self):
        for mutation in (lambda row: row.update(canonical_target=row["previous_target"]["normalized_text"]),
                         lambda row: row.update(selected_candidate_id="candidate_0"),
                         lambda row: row["canonical_tokens"].append(1)):
            row = copy.deepcopy(self.targets["records"][self.changed]); mutation(row)
            with self.assertRaises(ValueError):
                self.target_check(row, self.changed)

    def test_ordinary_token_and_frame_mutations(self):
        for mutation in (lambda row: row["canonical_tokens"].append(1), lambda row: row.update(student_frames=999),
                         lambda row: row.update(full_transcript_human_approved=True), lambda row: row.update(split="val")):
            row = copy.deepcopy(self.targets["records"][0]); mutation(row)
            with self.assertRaises(ValueError):
                self.target_check(row)

    def test_actual_context_row(self):
        self.context_check(self.context["records"][0])

    def test_half_open_midpoint_and_strict_crossing(self):
        words = [{"word": str(i), "start": x, "end": y, "probability": .8} for i, (x, y) in enumerate(((.4, .6), (.5, .8), (1.4, 1.6)))]
        result = a.interpret([{"words": words}], {"start_seconds": .5, "end_seconds": 1.5, "context_duration_seconds": 2.})
        self.assertEqual(result["selected_word_indices"], [0, 1])
        self.assertEqual(result["boundary_cues"]["crosses_label_start"], [0])
        self.assertEqual(result["boundary_cues"]["crosses_label_end"], [2])

    def test_zero_duration_and_no_selected_word_flags(self):
        result = a.interpret([{"words": [{"word": "밖", "start": .2, "end": .2, "probability": .9}]}],
                             {"start_seconds": .5, "end_seconds": 1.5, "context_duration_seconds": 2.})
        self.assertEqual(set(result["word_timing_flags"]), {"zero_duration_word", "no_midpoint_selected_words"})

    def test_word_nonfinite_or_inverted_rejected(self):
        for left, right in ((.6, .5), (float("nan"), 1.)):
            with self.assertRaises(ValueError):
                a.interpret([{"words": [{"word": "가", "start": left, "end": right, "probability": .9}]}],
                            {"start_seconds": .5, "end_seconds": 1.5, "context_duration_seconds": 2.})

    def test_context_option_token_and_selection_mutations(self):
        for mutation in (lambda row: row["transcription_info"]["transcription_options"].update(word_timestamps=False),
                         lambda row: row["transcription_info"]["transcription_options"].update(initial_prompt="정답"),
                         lambda row: row["raw_segments"][0]["tokens"].insert(1, 1),
                         lambda row: row.update(selected_text="정답으로 바꾼 문장"),
                         lambda row: row.update(automatic_approval=True)):
            row = copy.deepcopy(self.context["records"][0]); mutation(row)
            with self.assertRaises(ValueError):
                self.context_check(row)

    def test_context_clock_mutation(self):
        clip = copy.deepcopy(self.media["clips"][0])
        clip["context_audio"]["start_seconds"] += .1
        with self.assertRaises(ValueError):
            a.interval(clip)

    def test_context_word_cue_and_metric_mutations(self):
        for mutation in (lambda row: row["words"][0].update(midpoint=100.),
                         lambda row: row["boundary_cues"].update(crosses_label_start=[]),
                         lambda row: row["selected_comparison"]["metrics"].update(character_edits=0)):
            row = copy.deepcopy(self.context["records"][0]); mutation(row)
            with self.assertRaises(ValueError):
                self.context_check(row)

    def test_summary_and_denominators(self):
        a.same(a.context_summary(self.context["records"]), self.context["summary"], "Actual summary")
        self.assertEqual(self.context["summary"]["records"], 120)
        self.assertEqual(self.context["summary"]["automatic_approval_count"], 0)
        modified = copy.deepcopy(self.context["summary"])
        modified["comparison_metrics"]["selected_comparison"]["reference_characters"] -= 1
        with self.assertRaises(ValueError):
            a.same(a.context_summary(self.context["records"]), modified, "Tampered denominator")


if __name__ == "__main__":
    unittest.main()
