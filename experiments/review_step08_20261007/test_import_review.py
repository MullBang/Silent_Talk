"""Synthetic in-memory exports only; these tests create no user review evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import import_review as module


class ImportReviewTests(unittest.TestCase):
    def setUp(self):
        self.upstream = {key: {"path": "C:/fixture/" + key, "sha256": char * 64}
                         for key, char in zip(("summary", "label_targets", "context_results",
                                               "media_manifest", "review_data"), "abcde")}
        self.upstream["records"] = [
            {"id": f"synthetic_{i:03}", "split": "train" if i < 96 else "val",
             "speaker_id": f"synthetic_speaker_{i // 8:02}", "canonical_target": f"시험 문장 {i}"}
            for i in range(120)]
        target = self.upstream["records"][0]
        self.row = {"id": target["id"],
                    "canonical_target_sha256": module.sha_bytes(target["canonical_target"].encode("utf-8")),
                    "transcript_decision": "match", "boundary_decision": None, "notes": "",
                    "reviewed_at": "2026-10-07T04:00:00.000Z",
                    "review_method": "user_interaction_in_local_review_page"}
        self.payload = {"schema": module.EXPORT_SCHEMA, "created_utc": "2026-10-07T04:01:00Z",
                        "reviewer": "user", "label_targets": self.upstream["label_targets"],
                        "context_results": self.upstream["context_results"],
                        "media_manifest_sha256": self.upstream["media_manifest"]["sha256"],
                        "records": [self.row]}

    def validate(self):
        return module.validate_export(self.payload, self.upstream)

    def test_empty_export_is_not_approval(self):
        self.payload["records"] = []
        self.assertEqual(self.validate(), {})

    def test_partial_transcript_retains_all_roles_and_no_boundary(self):
        rows = self.validate()
        receipt = module.make_receipt(self.upstream, rows, {}, "Synthetic test caller context", True)
        self.assertEqual([r["id"] for r in receipt["records"]], [r["id"] for r in self.upstream["records"]])
        self.assertEqual(receipt["counts"]["train"], 96)
        self.assertEqual(receipt["counts"]["val"], 24)
        self.assertTrue(receipt["records"][0]["full_transcript_human_approved"])
        self.assertFalse(receipt["records"][0]["human_boundary_approved"])
        self.assertFalse(any(r["training_ready"] for r in receipt["records"]))
        self.assertFalse(receipt["training_ready"])

    def test_boundary_only_is_not_transcript_approval(self):
        self.row.update(transcript_decision=None, boundary_decision="complete")
        receipt = module.make_receipt(self.upstream, self.validate(), {}, "Synthetic", True)
        self.assertFalse(receipt["records"][0]["full_transcript_human_approved"])
        self.assertTrue(receipt["records"][0]["human_boundary_approved"])

    def test_json_reviewer_is_not_authentication(self):
        self.row["boundary_decision"] = "complete"
        receipt = module.make_receipt(self.upstream, self.validate(), {})
        self.assertFalse(receipt["records"][0]["full_transcript_human_approved"])
        self.assertFalse(receipt["records"][0]["human_boundary_approved"])
        self.assertEqual(receipt["records"][0]["review"]["transcript_decision"], "match")
        self.assertFalse(receipt["provenance"]["human_origin_authenticated_by_json"])

    def test_explicit_origin_requires_context(self):
        with self.assertRaisesRegex(ValueError, "context note"):
            module.make_receipt(self.upstream, self.validate(), {}, caller_confirms_user_submission=True)

    def test_notes_only_creates_no_decision(self):
        self.row.update(transcript_decision=None, notes="Synthetic note")
        receipt = module.make_receipt(self.upstream, self.validate(), {}, "Synthetic", True)
        self.assertEqual(receipt["counts"]["notes_only"], 1)
        self.assertEqual(receipt["counts"]["transcript_decisions"], 0)
        self.assertFalse(receipt["records"][0]["full_transcript_human_approved"])

    def test_negative_and_uncertain_do_not_approve(self):
        for transcript in ("mismatch", "uncertain"):
            for boundary in ("cut_suspected", "uncertain"):
                with self.subTest(transcript=transcript, boundary=boundary):
                    self.row.update(transcript_decision=transcript, boundary_decision=boundary)
                    receipt = module.make_receipt(self.upstream, self.validate(), {}, "Synthetic", True)
                    self.assertFalse(receipt["records"][0]["full_transcript_human_approved"])
                    self.assertFalse(receipt["records"][0]["human_boundary_approved"])

    def test_both_positive_still_do_not_enable_training(self):
        self.row["boundary_decision"] = "complete"
        receipt = module.make_receipt(self.upstream, self.validate(), {}, "Synthetic", True)
        self.assertTrue(receipt["records"][0]["full_transcript_human_approved"])
        self.assertTrue(receipt["records"][0]["human_boundary_approved"])
        self.assertFalse(receipt["records"][0]["training_ready"])
        self.assertFalse(receipt["labels_modified"])

    def test_duplicate_id_rejected(self):
        self.payload["records"].append(dict(self.row))
        with self.assertRaisesRegex(ValueError, "Duplicate review ID"):
            self.validate()

    def test_unknown_id_rejected(self):
        self.row["id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown review ID"):
            self.validate()

    def test_missing_target_hash_rejected(self):
        del self.row["canonical_target_sha256"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_target_hash_rejected(self):
        self.row["canonical_target_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Canonical hash mismatch"):
            self.validate()

    def test_binding_hash_and_path_rejected_without_reading_input_paths(self):
        for key in ("label_targets", "context_results"):
            for field in ("path", "sha256"):
                with self.subTest(key=key, field=field):
                    altered = copy.deepcopy(self.payload)
                    altered[key][field] = "//untrusted-host/must-not-be-read" if field == "path" else "0" * 64
                    with patch.object(Path, "read_bytes", side_effect=AssertionError("Must not read export path")):
                        with self.assertRaisesRegex(ValueError, "binding mismatch"):
                            module.validate_export(altered, self.upstream)

    def test_media_hash_rejected(self):
        self.payload["media_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "media binding mismatch"):
            self.validate()

    def test_unknown_fields_cannot_promote_training_or_replace_label(self):
        for key, value in (("training_ready", True), ("canonical_target", "replacement"), ("split", "test")):
            for location in ("top", "row"):
                with self.subTest(key=key, location=location):
                    altered = copy.deepcopy(self.payload)
                    (altered if location == "top" else altered["records"][0])[key] = value
                    with self.assertRaises(ValueError):
                        module.validate_export(altered, self.upstream)

    def test_decision_types_and_values_rejected(self):
        for key in ("transcript_decision", "boundary_decision"):
            for value in (True, 1, [], {}, "yes", "", "approved"):
                with self.subTest(key=key, value=value):
                    altered = copy.deepcopy(self.payload)
                    altered["records"][0][key] = value
                    with self.assertRaises(ValueError):
                        module.validate_export(altered, self.upstream)

    def test_bad_timestamps_rejected(self):
        for key in ("created_utc", "reviewed_at"):
            for value in (None, "yesterday", "2026-10-07T04:00:00"):
                with self.subTest(key=key, value=value):
                    altered = copy.deepcopy(self.payload)
                    (altered if key == "created_utc" else altered["records"][0])[key] = value
                    with self.assertRaises(ValueError):
                        module.validate_export(altered, self.upstream)

    def test_empty_or_oversized_note_rejected(self):
        self.row["transcript_decision"] = None
        for value in ("", "   ", "x" * 16001):
            self.row["notes"] = value
            with self.assertRaises(ValueError):
                self.validate()

    def test_wrong_method_and_schema_rejected(self):
        for key, value in (("schema", "auto-review"), ("review_method", "automatic_asr")):
            altered = copy.deepcopy(self.payload)
            (altered if key == "schema" else altered["records"][0])[key] = value
            with self.assertRaises(ValueError):
                module.validate_export(altered, self.upstream)

    def test_duplicate_json_key_and_nonfinite_rejected(self):
        for raw in (b'{"records":[],"records":[]}', b'{"a":NaN}', b'{"a":Infinity}',
                    b'{"outer":{"id":1,"id":2}}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    module.parse_json(raw)

    def test_external_file_is_read_only_and_exact_bytes_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic_review.json"
            raw = json.dumps(self.payload, ensure_ascii=False).encode("utf-8")
            source.write_bytes(raw)
            with patch.object(module, "load_upstream", return_value=self.upstream):
                receipt = module.import_file(source)
            self.assertEqual(source.read_bytes(), raw)
            self.assertEqual(receipt["input"]["sha256"], module.sha_bytes(raw))

    def test_size_limit_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic_large.json"
            source.write_bytes(b" " * (module.MAX_INPUT_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "too large"):
                module.import_file(source)

    def test_output_cannot_overwrite_or_escape_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module, "BASE", Path(directory)):
                path = Path(directory) / "results" / "synthetic.json"
                module.write_new_result(path, {"synthetic": True})
                with self.assertRaises(FileExistsError):
                    module.write_new_result(path, {"synthetic": False})
                with self.assertRaisesRegex(ValueError, "Step8 results"):
                    module.write_new_result(Path(directory) / "outside.json", {})

    def test_real_frozen_upstream_preflight(self):
        upstream = module.load_upstream()
        self.assertEqual(upstream["summary"]["sha256"], module.SUMMARY_SHA)
        self.assertEqual(len(upstream["records"]), 120)


if __name__ == "__main__":
    unittest.main(verbosity=2)
