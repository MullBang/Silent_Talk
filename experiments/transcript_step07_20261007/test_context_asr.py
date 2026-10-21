"""Synthetic-only context ASR contract tests; no model or dataset inference."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import context_asr as ctx


REF = {"path": "synthetic-plan.json", "sha256": "a" * 64}


def word(text, start, end, probability=0.9):
    return {"word": text, "start": start, "end": end, "probability": probability}


def segment(words):
    return {"id": 0, "seek": 0, "start": 0.2, "end": 1.8, "text": "".join(row["word"] for row in words),
            "tokens": [1, 2, 3], "temperature": 0.0, "avg_logprob": -0.2, "compression_ratio": 1.1,
            "no_speech_prob": 0.01, "words": words}


def clip():
    original = {"path": "synthetic_label.wav", "sha256": "b" * 64, "clock": "original_label_wav", "source_sha256": "c" * 64,
                "frames": 16000, "start_seconds": 2., "end_seconds": 3.}
    context = {**original, "path": "synthetic_context.wav", "frames": 32000, "start_seconds": 1.5, "end_seconds": 3.5}
    return {"id": "synthetic_001", "split": "train", "speaker_id": "C001", "source_stem": "source_001",
            "reference": "안녕 해", "start_time": 2., "end_time": 3., "context_audio": context, "audio": original,
            "timing_status": "verified", "artifacts_ready": True}


def baseline(reference="안녕 해"):
    return {"status": "ok", "asr_text": "안녕 해", "metrics": ctx.audit.compare_text(reference, "안녕 해")["metrics"],
            "boundary_flags": ["first_asr_segment_near_audio_start"], "confidence_flags": []}


class Model:
    def __init__(self, segments=None, fail=False):
        self.segments = segments if segments is not None else [segment([word(" 밖", .1, .3), word("안녕", .6, .9), word(" 해", 1., 1.4), word(" 밖", 1.6, 1.8)])]
        self.fail = fail
        self.calls = []
        self.exhausted = False

    def transcribe(self, samples, **options):
        self.calls.append((samples.copy(), options))
        if self.fail:
            raise RuntimeError("synthetic recognizer failure")
        def generate():
            yield from copy.deepcopy(self.segments)
            self.exhausted = True
        info = {"language": "ko", "duration": len(samples) / 16000, "duration_after_vad": len(samples) / 16000,
                "transcription_options": {**options, "temperatures": [options["temperature"]]}}
        return generate(), info


def audio_loader(_):
    return np.zeros(32000, dtype=np.float32)


class ContextTests(unittest.TestCase):
    def test_only_option_difference_is_word_timestamps(self):
        old = copy.deepcopy(ctx.audit.OPTIONS)
        self.assertFalse(old["word_timestamps"])
        old["word_timestamps"] = True
        self.assertEqual(old, ctx.OPTIONS)
        self.assertFalse(ctx.audit.OPTIONS["word_timestamps"])

    def test_audio_only_no_reference_prompt(self):
        one, two = Model(), Model()
        a, b = clip(), clip()
        b["reference"] = "정답을 모델에 주면 안 됨"
        ctx.make_record(one, a, baseline(), REF, audio_loader)
        ctx.make_record(two, b, baseline(b["reference"]), REF, audio_loader)
        self.assertEqual(one.calls[0][1], two.calls[0][1])
        np.testing.assert_array_equal(one.calls[0][0], two.calls[0][0])
        for key in ("initial_prompt", "prefix", "hotwords"):
            self.assertIsNone(one.calls[0][1][key])
        self.assertTrue(one.exhausted and two.exhausted)

    def test_fixed_midpoint_half_open_selection(self):
        words = [word("below", .1, .3), word("at-start", .4, .6), word("inside", .7, 1.2), word("at-end", 1.4, 1.6)]
        interpreted = ctx.interpret_words([segment(words)], ctx.interval_for(clip()))
        self.assertEqual(interpreted["selected_word_indices"], [1, 2])
        self.assertEqual(interpreted["selected_text"], "at-startinside")
        self.assertEqual(interpreted["boundary_cues"]["crosses_label_start"], [1])
        self.assertEqual(interpreted["boundary_cues"]["crosses_label_end"], [3])

    def test_touching_boundary_is_near_but_not_crossing(self):
        words = [word("touch", .5, .8), word("end", 1.2, 1.5)]
        value = ctx.interpret_words([segment(words)], ctx.interval_for(clip()))
        self.assertEqual(value["boundary_cues"]["crosses_label_start"], [])
        self.assertEqual(value["boundary_cues"]["crosses_label_end"], [])
        self.assertEqual(value["boundary_cues"]["near_label_start"], [0])
        self.assertEqual(value["boundary_cues"]["near_label_end"], [1])

    def test_approximate_word_flags_are_preserved_not_approval(self):
        words = [word("a", -.03, .3), word("b", .2, .2), word("c", 1.8, 2.03)]
        value = ctx.interpret_words([segment(words)], ctx.interval_for(clip()))
        self.assertEqual(set(value["word_timing_flags"]), {"word_timestamp_outside_context", "word_timestamps_overlap_or_out_of_order",
                                                         "zero_duration_word", "no_midpoint_selected_words"})

    def test_success_preserves_context_and_no_human_approval(self):
        row = ctx.make_record(Model(), clip(), baseline(), REF, audio_loader)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["full_context_text"], " 밖안녕 해 밖")
        self.assertEqual(row["selected_text"], "안녕 해")
        self.assertTrue(row["selected_comparison"]["metrics"]["exact"])
        self.assertFalse(row["full_context_comparison"]["metrics"]["exact"])
        for key in ("human_listening_confirmed", "automatic_approval", "label_modified", "training_ready"):
            self.assertIs(row[key], False)
        ctx.validate_record(row, clip(), baseline(), REF)

    def test_inference_failure_kept_with_empty_predictions(self):
        row = ctx.make_record(Model(fail=True), clip(), baseline(), REF, audio_loader)
        self.assertEqual(row["status"], "asr_failed")
        self.assertEqual(row["selected_text"], "")
        self.assertEqual(row["selected_comparison"]["metrics"]["cer_percent"], 100.)
        ctx.validate_record(row, clip(), baseline(), REF)

    def test_audio_integrity_failure_aborts_not_empty_prediction(self):
        def corrupt(_):
            raise ctx.audit.AudioEvidenceError("changed audio")
        with self.assertRaises(ctx.audit.AudioEvidenceError):
            ctx.make_record(Model(), clip(), baseline(), REF, corrupt)

    def test_invalid_words_fail_without_omission(self):
        for words in ([word("a", .8, .7)], [word("a", .8, .9, 1.1)], [word("a", float("nan"), .9)]):
            with self.subTest(words=words):
                row = ctx.make_record(Model([segment(words)]), clip(), baseline(), REF, audio_loader)
                self.assertEqual(row["status"], "asr_failed")
                self.assertEqual(row["raw_segments"], [])

    def test_no_segments_is_success_but_no_selected_words(self):
        row = ctx.make_record(Model([]), clip(), baseline(), REF, audio_loader)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["selected_text"], "")
        self.assertIn("no_word_timestamps", row["word_timing_flags"])
        self.assertIn("no_segments", row["decoder_confidence_flags"])
        ctx.validate_record(row, clip(), baseline(), REF)

    def test_changed_context_clock_or_padding_rejected(self):
        for change in (lambda row: row["context_audio"].update(start_seconds=1.4),
                       lambda row: row["context_audio"].update(source_sha256="d" * 64),
                       lambda row: row["context_audio"].update(frames=31900)):
            row = clip()
            change(row)
            with self.assertRaises(ValueError):
                ctx.interval_for(row)

    def test_saved_row_selection_metrics_and_claim_tamper_rejected(self):
        original = ctx.make_record(Model(), clip(), baseline(), REF, audio_loader)
        for change in (lambda row: row.update(selected_text="바뀜"), lambda row: row.update(automatic_approval=True),
                       lambda row: row["selected_word_indices"].append(0), lambda row: row.pop("error")):
            row = copy.deepcopy(original)
            change(row)
            with self.assertRaises(ValueError):
                ctx.validate_record(row, clip(), baseline(), REF)

    def test_saved_options_prompt_duration_and_vad_tamper_rejected(self):
        original = ctx.make_record(Model(), clip(), baseline(), REF, audio_loader)
        for change in (lambda info: info['transcription_options'].update(word_timestamps=False),
                       lambda info: info['transcription_options'].update(initial_prompt='label'),
                       lambda info: info.update(duration=1.8), lambda info: info.update(duration_after_vad=1.7),
                       lambda info: info['transcription_options'].update(temperatures=[0., 0.2])):
            row = copy.deepcopy(original)
            change(row['transcription_info'])
            with self.assertRaises(ValueError):
                ctx.validate_record(row, clip(), baseline(), REF)

    def test_progress_checksum_orphan_and_order_guards(self):
        with tempfile.TemporaryDirectory(dir=ctx.BASE, prefix="synthetic_context_") as tmp:
            base = Path(tmp)
            directory, path = base / "records", base / "progress.json"
            row = ctx.make_record(Model(), clip(), baseline(), REF, audio_loader)
            ctx.save_new(directory / "synthetic_001.json", row)
            with self.assertRaises(ValueError):
                ctx.load_progress([clip()], [baseline()], REF, path, directory)
            progress = {"schema": "step07-context-asr-progress-v1", "context_plan": REF,
                        "records": [{"id": "synthetic_001", **ctx.bind(directory / "synthetic_001.json")}]}
            ctx.save_progress(progress, path)
            self.assertEqual(ctx.load_progress([clip()], [baseline()], REF, path, directory)[1], [row])
            (directory / "synthetic_001.json").write_text(json.dumps({**row, "selected_text": "tamper"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ctx.load_progress([clip()], [baseline()], REF, path, directory)
            with self.assertRaises(ValueError):
                ctx.load_progress([clip()], [baseline()], {**REF, "sha256": "e" * 64}, path, directory)

    def test_summary_retains120_and_failed_denominator(self):
        rows = []
        for i in range(120):
            item = clip()
            item["id"] = "synthetic_" + str(i)
            rows.append(ctx.make_record(Model(fail=i == 119), item, baseline(), REF, audio_loader))
        value = ctx.summarize(rows)
        self.assertEqual(value["records"], 120)
        self.assertEqual(value["status_counts"], {"ok": 119, "asr_failed": 1})
        self.assertEqual(value["comparison_metrics"]["selected_comparison"]["exact"], 119)
        self.assertEqual(value["human_listening_confirmed_count"], 0)
        self.assertEqual(value["automatic_approval_count"], 0)


if __name__ == "__main__":
    unittest.main()
