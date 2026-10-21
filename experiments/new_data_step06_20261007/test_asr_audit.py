"""Synthetic ASR-audit contracts; no real audio, model loading or network."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

import asr_audit as subject


def clip(index=0):
    return {"id": f"synthetic_{index:03d}", "split": "train" if index < 96 else "val",
        "speaker_id": "S", "source_stem": f"source_{index // 4:02d}", "reference": "안녕, 세상",
        "start_time": 10., "end_time": 12., "timing_status": "verified", "artifacts_ready": True,
        "training_ready": False, "human_transcript_review": "not_reviewed",
        "audio": {"path": "synthetic.wav", "sha256": "synthetic", "sample_rate": 16000,
                  "frames": 32000, "channels": 1, "sample_width_bytes": 2}}


class FakeASR:
    def __init__(self, text="안녕 세상", segments=None, failure=False):
        self.text, self.segments, self.failure = text, segments, failure
        self.calls, self.iterated = [], False

    def transcribe(self, samples, **options):
        self.calls.append((samples, copy.deepcopy(options)))
        if self.failure:
            raise RuntimeError("synthetic inference failure")
        duration = len(samples) / 16000
        segments = self.segments if self.segments is not None else [{"id": 0, "seek": 0, "start": .3, "end": duration - .3,
            "text": self.text, "tokens": [1, 2, 3], "avg_logprob": -.2, "compression_ratio": 1., "no_speech_prob": .1,
            "words": None, "temperature": 0.0}]
        actual = {name: options[name] for name in ("beam_size", "condition_on_previous_text", "initial_prompt", "prefix", "hotwords", "word_timestamps")}
        actual["temperatures"] = [0.0]
        info = {"language": "ko", "language_probability": 1., "duration": duration, "duration_after_vad": duration,
                "all_language_probs": None, "transcription_options": actual, "vad_options": None}
        def generate():
            self.iterated = True
            yield from segments
        return generate(), info


def samples(_spec):
    return np.zeros(32000, dtype=np.float32)


class ASRAuditTests(unittest.TestCase):
    def test_normalization_retains_letters_numbers_and_internal_spacing_rule(self):
        self.assertEqual(subject.normalize_for_comparison(" 가,\t나!  10분 X "), "가 나 10분 X")
        value = subject.compare_text("가  나", "가나")
        self.assertFalse(value["metrics"]["exact"])
        self.assertTrue(value["metrics"]["without_spaces_exact"])
        self.assertEqual(value["metrics"]["character_edits"], 1)

    def test_reference_is_never_passed_to_model_or_prompt(self):
        model = FakeASR()
        record = subject.audit_clip(model, clip(), {"sha256": "plan"}, audio_loader=samples)
        audio, options = model.calls[0]
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(options, subject.OPTIONS)
        self.assertIsNone(options["initial_prompt"])
        self.assertIsNone(options["hotwords"])
        self.assertIsNone(options["prefix"])
        self.assertFalse(options["condition_on_previous_text"])
        self.assertFalse(options["vad_filter"])
        self.assertTrue(model.iterated)
        self.assertEqual(record["review_status"], "automatic_support_only")
        self.assertFalse(record["human_listening_confirmed"])
        self.assertFalse(record["training_ready"])
        self.assertFalse(record["label_modified"])

    def test_same_audio_options_do_not_depend_on_reference(self):
        first, second = clip(), clip()
        second["reference"] = "완전히 다른 정답"
        a, b = FakeASR(), FakeASR()
        subject.audit_clip(a, first, {}, audio_loader=samples)
        record = subject.audit_clip(b, second, {}, audio_loader=samples)
        self.assertEqual(a.calls[0][1], b.calls[0][1])
        np.testing.assert_array_equal(a.calls[0][0], b.calls[0][0])
        self.assertEqual(record["review_status"], "human_listening_required")
        self.assertIn("normalized_transcript_mismatch", record["review_reasons"])

    def test_quarantined_media_retained_without_model_call(self):
        row = clip()
        row.update(timing_status="quarantined", artifacts_ready=False, audio=None)
        model = FakeASR()
        result = subject.audit_clip(model, row, {}, audio_loader=samples)
        self.assertEqual(model.calls, [])
        self.assertEqual(result["status"], "not_transcribed")
        self.assertEqual(result["metrics"]["cer_percent"], 100.)
        self.assertEqual(result["review_status"], "human_listening_required")

    def test_inference_failure_preserves_reference_and_empty_prediction(self):
        row = clip()
        record = subject.audit_clip(FakeASR(failure=True), row, {}, audio_loader=samples)
        self.assertEqual(record["status"], "asr_failed")
        self.assertEqual(record["asr_text"], "")
        self.assertEqual(record["reference"], row["reference"])
        self.assertEqual(record["metrics"]["cer_percent"], 100.)
        self.assertIn("synthetic inference failure", record["error"])

    def test_exact_match_with_boundary_or_low_confidence_needs_listening(self):
        segment = {"id": 0, "seek": 0, "start": 0., "end": 2., "text": "안녕 세상", "tokens": [1],
                   "avg_logprob": -1.1, "no_speech_prob": .6, "compression_ratio": 2.5, "words": None, "temperature": 0.}
        record = subject.audit_clip(FakeASR(segments=[segment]), clip(), {}, audio_loader=samples)
        self.assertTrue(record["metrics"]["exact"])
        self.assertEqual(record["review_status"], "human_listening_required")
        self.assertEqual(len(record["boundary_flags"]), 2)
        self.assertEqual(len(record["confidence_flags"]), 3)
        self.assertEqual(record["raw_segments"], [segment])

    def test_empty_transcription_never_counts_as_support(self):
        record = subject.audit_clip(FakeASR(segments=[]), clip(), {}, audio_loader=samples)
        self.assertEqual(record["status"], "ok")
        self.assertIn("no_segments", record["confidence_flags"])
        self.assertIn("empty_normalized_transcription", record["confidence_flags"])
        self.assertEqual(record["review_status"], "human_listening_required")

    def test_bad_timestamps_and_numeric_metadata_flag_or_reject(self):
        row = {"start": -.04, "end": 2.1, "avg_logprob": -.1, "no_speech_prob": 0., "compression_ratio": 1.}
        boundary, _ = subject.assess_segments([row], 2.)
        self.assertIn("asr_segment_timestamp_outside_or_nonmonotonic", boundary)
        with self.assertRaises(ValueError):
            subject.json_value({"avg_logprob": float("nan")})

    def test_full_cohort_split_and_speaker_records_not_dropped(self):
        media = {"clips": [clip(i) for i in range(120)]}
        self.assertEqual(len(subject.validate_media(media)), 120)
        media["clips"].pop()
        with self.assertRaises(ValueError):
            subject.validate_media(media)
        media = {"clips": [clip(i) for i in range(120)]}
        media["clips"][-1]["split"] = "train"
        with self.assertRaises(ValueError):
            subject.validate_media(media)

    def test_audio_hash_and_header_mismatch_are_fatal_evidence_failures(self):
        with tempfile.TemporaryDirectory(prefix="asr_synthetic_test_", dir=subject.BASE) as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(subject.BASE.resolve()))
            path = folder / "clip.wav"
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(np.zeros(160, dtype="<i2").tobytes())
            spec = {**subject.bind(path), "sample_rate": 16000, "frames": 160, "channels": 1, "sample_width_bytes": 2}
            self.assertEqual(subject.load_audio(spec).shape, (160,))
            changed = {**spec, "sha256": "wrong"}
            with self.assertRaises(subject.AudioEvidenceError):
                subject.load_audio(changed)
            with self.assertRaises(subject.AudioEvidenceError):
                subject.load_audio({**spec, "frames": 159})
            row = clip()
            row["audio"] = changed
            with self.assertRaises(subject.AudioEvidenceError):
                subject.audit_clip(FakeASR(), row, {})

    def test_summary_keeps120_denominators_and_zero_manual_confirmation(self):
        records = [subject.audit_clip(FakeASR(failure=i == 119), clip(i), {}, audio_loader=samples) for i in range(120)]
        summary = subject.summarize(records)
        self.assertEqual(summary["records"], 120)
        self.assertEqual(summary["asr_status_counts"], {"ok": 119, "asr_failed": 1})
        self.assertEqual(summary["normalized_exact_matches"], 119)
        self.assertEqual(summary["human_listening_confirmed_count"], 0)
        self.assertAlmostEqual(summary["pooled_cer_percent_including_failed_empty"], 100 / 120)
        with self.assertRaises(ValueError):
            subject.summarize(records[:-1])

    def test_progress_requires_record_sha_and_complete_saved_fields(self):
        for mutation in ("unchanged", "changed_bytes", "missing_field"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="asr_progress_test_", dir=subject.BASE) as directory:
                folder = Path(directory).resolve()
                self.assertTrue(folder.is_relative_to(subject.BASE.resolve()))
                record_dir, progress_path = folder / "records", folder / "progress.json"
                row, plan_ref = clip(), {"path": "plan.json", "sha256": "expected"}
                value = subject.audit_clip(FakeASR(), row, plan_ref, audio_loader=samples)
                record_path = record_dir / (row["id"] + ".json")
                subject.save_new(record_path, value)
                index = {"schema": "step06-local-asr-progress-v1", "asr_plan": plan_ref,
                         "records": [{"id": row["id"], **subject.bind(record_path)}]}
                subject.save_progress(progress_path, index)
                if mutation == "changed_bytes":
                    value["asr_text"] = "tampered"
                    record_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                elif mutation == "missing_field":
                    del value["raw_segments"]
                    record_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                    index["records"][0] = {"id": row["id"], **subject.bind(record_path)}
                    subject.save_progress(progress_path, index)
                if mutation == "unchanged":
                    _, restored = subject.load_progress(progress_path, [row], plan_ref, record_dir)
                    self.assertEqual(restored, [value])
                else:
                    with self.assertRaises(ValueError):
                        subject.load_progress(progress_path, [row], plan_ref, record_dir)

    def test_unindexed_record_and_wrong_plan_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="asr_progress_test_", dir=subject.BASE) as directory:
            folder = Path(directory).resolve()
            self.assertTrue(folder.is_relative_to(subject.BASE.resolve()))
            record_dir, progress_path = folder / "records", folder / "progress.json"
            row, plan_ref = clip(), {"path": "plan.json", "sha256": "expected"}
            value = subject.audit_clip(FakeASR(), row, plan_ref, audio_loader=samples)
            record_path = record_dir / (row["id"] + ".json")
            subject.save_new(record_path, value)
            with self.assertRaisesRegex(ValueError, "Unindexed"):
                subject.load_progress(progress_path, [row], plan_ref, record_dir)
            subject.save_progress(progress_path, {"schema": "step06-local-asr-progress-v1", "asr_plan": {},
                                  "records": [{"id": row["id"], **subject.bind(record_path)}]})
            with self.assertRaisesRegex(ValueError, "plan/schema"):
                subject.load_progress(progress_path, [row], plan_ref, record_dir)


if __name__ == "__main__":
    unittest.main()
