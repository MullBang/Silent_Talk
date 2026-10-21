"""Synthetic-only safety and binding checks for explicit review submissions."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from validate_review import (EXPORT_SCHEMA, MANIFEST_SCHEMA, ReviewValidationError,
                             validate, write_receipt)

NOW = "2026-10-13T01:00:00+00:00"


class ReviewValidationTests(unittest.TestCase):
    """Use temporary artificial data only, never saved human evidence."""

    def setUp(self) -> None:
        """Build two fake records, one with a deliberately too-short word."""
        try:
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.root = Path(self.temp.name)
            self.manifest_path = self.root / "manifest.json"
            self.submission_path = self.root / "synthetic_submission.json"
            self.manifest: dict[str, Any] = {
                "schema": MANIFEST_SCHEMA, "training_ready": False,
                "sources": {str((self.root / "source.json").resolve()): "a" * 64},
                "records": [{
                    "id": "SYNTHETIC_A", "text": "안녕 나", "source_text": "안녕 나",
                    "review_target_sha256": "b" * 64, "context_duration_seconds": 2.0,
                    "sentence_start_in_context": 0.0, "sentence_end_in_context": 2.0,
                    "segments": [
                        {"segment_id": "w0", "kind": "word", "text": "안녕",
                         "start_sample_16k": 1600, "end_sample_16k": 9600,
                         "start_in_context": 0.1, "end_in_context": 0.6,
                         "frames_25fps": 12, "ctc_min_frames": 6, "ctc_feasible": True},
                        {"segment_id": "w1", "kind": "word", "text": "나",
                         "start_sample_16k": 9600, "end_sample_16k": 9920,
                         "start_in_context": 0.6, "end_in_context": 0.62,
                         "frames_25fps": 0, "ctc_min_frames": 2, "ctc_feasible": False},
                    ]}, {
                    "id": "SYNTHETIC_OOV", "text": "탔어", "source_text": "탔어",
                    "review_target_sha256": "c" * 64, "context_duration_seconds": 1.0,
                    "sentence_start_in_context": 0.0, "sentence_end_in_context": 1.0,
                    "segments": [], "alignment_status": "oov_held",
                }],
            }
            self.write_json(self.manifest_path, self.manifest)
            self.manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
            self.export: dict[str, Any] = {
                "schema": EXPORT_SCHEMA, "reviewer": "user", "training_ready": False,
                "review_manifest_sha256": self.manifest_sha, "created_utc": NOW,
                "provenance_note": "SYNTHETIC TEST ONLY", "records": [],
            }
            self.write_json(self.submission_path, self.export)
        except Exception:
            raise

    def write_json(self, path: Path, value: dict[str, Any]) -> None:
        """Write a temporary test input."""
        try:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            raise

    def record(self, record_index: int = 0, positive: bool = False) -> dict[str, Any]:
        """Create an explicitly synthetic export record from the temporary manifest."""
        try:
            source = self.manifest["records"][record_index]
            return {
                "id": source["id"], "manifest_sha256": self.manifest_sha,
                "target_sha256": source["review_target_sha256"],
                "transcript_decision": "match" if positive else None,
                "sentence_boundary_decision": "complete" if positive else None,
                "notes": "SYNTHETIC TEST ONLY", "reviewed_at": NOW if positive else None,
                "segments": {
                    row["segment_id"]: {
                        "segment_id": row["segment_id"], "text": row["text"],
                        "original_start_sample_16k": row["start_sample_16k"],
                        "original_end_sample_16k": row["end_sample_16k"],
                        "transcript_decision": "match" if positive else None,
                        "audio_boundary_decision": "acceptable" if positive else None,
                        "visual_context_decision": "acceptable" if positive else None,
                        "notes": "SYNTHETIC TEST ONLY", "reviewed_at": NOW if positive else None,
                    } for row in source["segments"]
                },
            }
        except Exception:
            raise

    def check(self, export: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate a selected synthetic input."""
        try:
            self.write_json(self.submission_path, self.export if export is None else export)
            return validate(self.manifest_path, self.submission_path)
        except Exception:
            raise

    def test_empty_export_creates_no_approvals(self) -> None:
        """An untouched export is valid but has zero decisions."""
        try:
            result = self.check()
            self.assertEqual(result["submitted_sentence_records"], 0)
            self.assertEqual(result["submitted_segment_decision_fields"], 0)
            self.assertFalse(result["training_ready"])
            self.assertFalse(result["actual_human_provenance_verified"])
        except Exception:
            raise

    def test_all_null_review_stays_unanswered(self) -> None:
        """Default UI records cannot silently become positive reviews."""
        try:
            self.export["records"] = [self.record(), self.record(1)]
            result = self.check()
            self.assertEqual(result["submitted_positive_segment_records"], 0)
            self.assertEqual(result["submitted_sentence_decision_fields"], 0)
            self.assertEqual(result["provisional_candidates_if_actual_human_provenance_is_verified"], [])
        except Exception:
            raise

    def test_positive_claims_are_not_authenticated(self) -> None:
        """Even all-positive fabricated input cannot create an actual approval."""
        try:
            self.export["records"] = [self.record(positive=True)]
            result = self.check()
            self.assertEqual(result["submitted_positive_segment_records"], 2)
            self.assertEqual(result["actual_human_approved_training_segments"], 0)
            self.assertFalse(result["actual_human_provenance_verified"])
            self.assertFalse(result["training_ready"])
            self.assertEqual(len(result["provisional_candidates_if_actual_human_provenance_is_verified"]), 1)
        except Exception:
            raise

    def test_positive_length_failure_keeps_blocker(self) -> None:
        """Positive review does not override zero frames or CTC length failure."""
        try:
            self.export["records"] = [self.record(positive=True)]
            result = self.check()
            blocked = result["positive_segment_records_with_other_blockers"]
            self.assertEqual(blocked[0]["segment_id"], "w1")
            self.assertIn("insufficient_visual_ctc_length", blocked[0]["reasons"])
        except Exception:
            raise

    def test_parent_sentence_gate_remains_required(self) -> None:
        """Segment positives alone do not erase missing whole-sentence review."""
        try:
            record = self.record(positive=True)
            record["transcript_decision"] = None
            self.export["records"] = [record]
            result = self.check()
            self.assertEqual(result["provisional_candidates_if_actual_human_provenance_is_verified"], [])
            self.assertEqual(len(result["positive_segment_records_with_other_blockers"]), 2)
        except Exception:
            raise

    def test_oov_sentence_has_no_approved_segments(self) -> None:
        """An OOV sentence can be reviewed without manufacturing word candidates."""
        try:
            self.export["records"] = [self.record(1, positive=True)]
            result = self.check()
            self.assertEqual(result["submitted_positive_complete_sentence_records"], 1)
            self.assertEqual(result["submitted_positive_segment_records"], 0)
            self.assertFalse(result["training_ready"])
        except Exception:
            raise

    def test_corrupt_identity_fields_rejected(self) -> None:
        """Detect record, hash, segment, text and exact-boundary substitutions."""
        try:
            mutations = [
                ("manifest", lambda e: e.update(review_manifest_sha256="d" * 64)),
                ("record_id", lambda e: e["records"][0].update(id="foreign")),
                ("record_hash", lambda e: e["records"][0].update(manifest_sha256="d" * 64)),
                ("target_hash", lambda e: e["records"][0].update(target_sha256="d" * 64)),
                ("segment_id", lambda e: e["records"][0]["segments"]["w0"].update(segment_id="foreign")),
                ("segment_text", lambda e: e["records"][0]["segments"]["w0"].update(text="변조")),
                ("start", lambda e: e["records"][0]["segments"]["w0"].update(original_start_sample_16k=1601)),
                ("end", lambda e: e["records"][0]["segments"]["w0"].update(original_end_sample_16k=9601)),
                ("bool_start", lambda e: e["records"][0]["segments"]["w0"].update(original_start_sample_16k=True)),
                ("float_end", lambda e: e["records"][0]["segments"]["w0"].update(original_end_sample_16k=9600.0)),
            ]
            for name, mutate in mutations:
                with self.subTest(name=name):
                    export = copy.deepcopy(self.export)
                    export["records"] = [self.record()]
                    mutate(export)
                    with self.assertRaises(ReviewValidationError):
                        self.check(export)
        except Exception:
            raise

    def test_duplicate_record_id_rejected(self) -> None:
        """The same identity cannot be counted twice in a submission."""
        try:
            self.export["records"] = [self.record(), self.record()]
            with self.assertRaises(ReviewValidationError):
                self.check()
        except Exception:
            raise

    def test_unknown_segment_key_rejected(self) -> None:
        """Foreign candidates cannot be inserted into a bound record."""
        try:
            record = self.record()
            record["segments"]["foreign"] = record["segments"]["w0"]
            self.export["records"] = [record]
            with self.assertRaises(ReviewValidationError):
                self.check()
        except Exception:
            raise

    def test_approval_forgery_and_unknown_fields_rejected(self) -> None:
        """Export claims cannot add provenance verification or training approval."""
        try:
            mutations = [
                lambda e: e.update(training_ready=True),
                lambda e: e.update(training_ready=0),
                lambda e: e.update(actual_human_provenance_verified=True),
                lambda e: e.update(reviewer="assistant"),
                lambda e: e["records"][0].update(approved=True),
                lambda e: e["records"][0]["segments"]["w0"].update(approved=True),
                lambda e: e["records"][0]["segments"]["w0"].update(adjusted_start=0.0),
            ]
            for i, mutate in enumerate(mutations):
                with self.subTest(case=i):
                    export = copy.deepcopy(self.export)
                    export["records"] = [self.record()]
                    mutate(export)
                    with self.assertRaises(ReviewValidationError):
                        self.check(export)
        except Exception:
            raise

    def test_unknown_enum_and_type_rejected(self) -> None:
        """Unsupported decisions, booleans and list values cannot pass enums."""
        try:
            for value in ("approved", True, [], 1):
                with self.subTest(value=value):
                    record = self.record()
                    record["segments"]["w0"]["audio_boundary_decision"] = value
                    self.export["records"] = [record]
                    with self.assertRaises(ReviewValidationError):
                        self.check()
        except Exception:
            raise

    def test_decisions_need_timezone_timestamp(self) -> None:
        """Recorded non-null decisions need a syntactically valid dated record."""
        try:
            for value in (None, "yesterday", "2026-10-13T01:00:00"):
                with self.subTest(value=value):
                    record = self.record(positive=True)
                    record["segments"]["w0"]["reviewed_at"] = value
                    self.export["records"] = [record]
                    with self.assertRaises(ReviewValidationError):
                        self.check()
        except Exception:
            raise

    def test_duplicate_json_keys_rejected_at_every_depth(self) -> None:
        """JSON parser rejects shadowed fields even when values are identical."""
        try:
            for raw in ('{"schema":"x","schema":"x"}', '{"nested":{"a":1,"a":2}}'):
                with self.subTest(raw=raw):
                    self.submission_path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ReviewValidationError):
                        validate(self.manifest_path, self.submission_path)
        except Exception:
            raise

    def test_nonfinite_numbers_rejected(self) -> None:
        """Reject both nonstandard constants and overflowed JSON numeric values."""
        try:
            for literal in ("NaN", "Infinity", "-Infinity", "1e999"):
                with self.subTest(literal=literal):
                    self.submission_path.write_text('{"x":' + literal + '}', encoding="utf-8")
                    with self.assertRaises(ReviewValidationError):
                        validate(self.manifest_path, self.submission_path)
        except Exception:
            raise

    def test_manifest_bytes_changed_invalidates_binding(self) -> None:
        """Even semantically equivalent manifest edits invalidate the export SHA."""
        try:
            with self.manifest_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(ReviewValidationError):
                self.check()
        except Exception:
            raise

    def test_manifest_frame_or_ctc_corruption_rejected(self) -> None:
        """The trusted manifest must still satisfy integer-sample geometry."""
        try:
            for key, value in (("frames_25fps", 13), ("ctc_feasible", False),
                               ("start_in_context", 0.2), ("end_sample_16k", 999999)):
                with self.subTest(key=key):
                    altered = copy.deepcopy(self.manifest)
                    altered["records"][0]["segments"][0][key] = value
                    self.write_json(self.manifest_path, altered)
                    self.export["review_manifest_sha256"] = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
                    with self.assertRaises(ReviewValidationError):
                        self.check()
        except Exception:
            raise

    def test_existing_output_and_input_overwrite_rejected(self) -> None:
        """Receipt output cannot destroy a manifest, submission or prior receipt."""
        try:
            result = self.check()
            for path in (self.manifest_path, self.submission_path):
                previous = path.read_bytes()
                with self.assertRaises(ReviewValidationError):
                    write_receipt(path, result, (self.manifest_path, self.submission_path))
                self.assertEqual(path.read_bytes(), previous)
            output = self.root / "receipt.json"
            write_receipt(output, result, (self.manifest_path, self.submission_path))
            previous = output.read_bytes()
            with self.assertRaises(ReviewValidationError):
                write_receipt(output, result, (self.manifest_path, self.submission_path))
            self.assertEqual(output.read_bytes(), previous)
        except Exception:
            raise

    def test_cli_default_is_read_only(self) -> None:
        """The default CLI emits a receipt only on stdout and changes no file."""
        try:
            before = {p.name: p.read_bytes() for p in self.root.iterdir()}
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("validate_review.py")),
                                  "--manifest", str(self.manifest_path),
                                  "--submission", str(self.submission_path)],
                                 capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(json.loads(run.stdout)["training_ready"])
            self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, before)
        except Exception:
            raise

    def test_cli_invalid_input_creates_no_receipt(self) -> None:
        """Failed validation cannot leave an apparently successful output file."""
        try:
            self.export["training_ready"] = True
            self.write_json(self.submission_path, self.export)
            output = self.root / "receipt.json"
            run = subprocess.run([sys.executable, str(Path(__file__).with_name("validate_review.py")),
                                  "--manifest", str(self.manifest_path),
                                  "--submission", str(self.submission_path), "--output", str(output)],
                                 capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(output.exists())
        except Exception:
            raise


if __name__ == "__main__":
    unittest.main()
