"""Focused metadata tests; no media, model, or result-producing code runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import unicodedata

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
spec = importlib.util.spec_from_file_location("stage5e_selector", HERE / "select_review_queue.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalized(text):
    return "".join(unicodedata.normalize("NFC", text).split())


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.queue = read(HERE / "review_queue.json")
        cls.roles = read(HERE / "role_proposal.json")
        cls.eligible = read(BASE / "data/eligible_sentences.json")["records"]
        cls.parents = {r["parent_id"]: r for r in cls.eligible}
        cls.train = cls.roles["existing_anchors"] + [r for r in cls.queue["records"] if r["proposed_role"] == "proposed_train"]
        cls.probes = cls.roles["probe_records"]

    def test_normalization_preserves_spelling_but_ignores_whitespace_and_nfd(self):
        self.assertEqual(selector.key(unicodedata.normalize("NFD", "잘\t안\n돼")), "잘안돼")
        self.assertNotEqual(selector.key("잘 안 돼"), selector.key("잘 안 돼요"))

    def test_contiguous_eojeol_check_rejects_subword_and_reordered_words(self):
        self.assertTrue(selector.contained("잘 안 돼", "요즘 잘 안 돼 정말"))
        self.assertFalse(selector.contained("잘 안 돼", "잘만 안 돼요"))
        self.assertFalse(selector.contained("잘 안 돼", "안 잘 돼"))
        self.assertFalse(selector.contained("잘 안 돼", "잘 돼"))

    def test_source_bindings_still_exact(self):
        for name in ("review_queue.json", "role_proposal.json", "selection_summary.json"):
            for ref in read(HERE / name)["sources"]:
                path = Path(ref["path"])
                self.assertEqual(path.stat().st_size, ref["bytes"])
                with path.open("rb") as stream:
                    self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), ref["sha256"])

    def test_training_and_probes_share_no_parent_or_recording_or_full_transcript(self):
        for field in ("parent_id", "recording_key", "video_sha256", "label_sha256"):
            train = {self.parents[r["parent_id"]][field] for r in self.train}
            probe = {self.parents[r["parent_id"]][field] for r in self.probes}
            self.assertFalse(train & probe, field)
        for field in ("text", "historical_text", "source_text"):
            train = {normalized(self.parents[r["parent_id"]][field]) for r in self.train}
            probe = {normalized(self.parents[r["parent_id"]][field]) for r in self.probes}
            self.assertFalse(train & probe, field)

    def test_all_reserved_siblings_are_excluded_from_training(self):
        train_ids = {r["parent_id"] for r in self.train}
        reserved_count = 0
        for reservation in self.roles["reserved_recordings"]:
            expected = {r["parent_id"] for r in self.eligible if r["recording_key"] == reservation["recording_key"]}
            actual = set(reservation["reserved_parent_ids_in_current_eligible_pool"])
            selected = set(reservation["selected_parent_ids"])
            self.assertEqual(expected, actual)
            self.assertEqual(expected - selected, set(reservation["unselected_siblings_reserved_from_future_training"]))
            self.assertFalse(actual & train_ids)
            self.assertTrue(reservation["applies_to_all_angles_and_future_discovered_siblings"])
            reserved_count += len(actual)
        self.assertEqual(reserved_count, 61)

    def test_probe_roles_are_distinguished_by_full_phrase_overlap(self):
        train_texts = [normalized(r["text"]) for r in self.train]
        observed_words = {word for r in self.train for word in r["text"].split()}
        subtype_counts = {}
        for row in self.probes:
            text = normalized(row["text"])
            if row["proposed_role"] == "same_expression_probe":
                self.assertIn(text, train_texts)
            else:
                self.assertTrue(all(text not in train_text for train_text in train_texts))
                words = row["text"].split()
                missing = [word for word in words if word not in observed_words]
                self.assertEqual(missing, row["unseen_train_eojeols"])
                self.assertEqual(sum(word in observed_words for word in words) / len(words), row["eojeol_occurrence_coverage"])
                subtype_counts[bool(missing)] = subtype_counts.get(bool(missing), 0) + 1
        self.assertEqual(subtype_counts, {False: 1, True: 3})

    def test_fixed_protection_applies_to_all_twenty_selected_candidates(self):
        scope = read(BASE / "data/protected_scope.json")
        rows = self.queue["records"]
        self.assertEqual(len(rows), len({r["parent_id"] for r in rows}))
        for row in rows:
            parent = self.parents[row["parent_id"]]
            self.assertNotIn(parent["parent_id"], scope["protected_parent_ids"])
            self.assertNotIn(parent["recording_key"], scope["protected_recording_keys"])
            self.assertNotIn(parent["video_sha256"], scope["blocked_video_sha256"])
            self.assertNotIn(parent["label_sha256"], scope["blocked_label_sha256"])
            self.assertNotIn(normalized(row["text"]), scope["protected_candidate_text_keys"])
            self.assertTrue(all(text not in normalized(row["text"]) for text in scope["protected_phrase_text_keys"]))

    def test_first_batch_and_unapproved_state_are_preserved(self):
        rows = self.queue["records"]
        self.assertEqual(self.queue["first_batch"]["parent_ids"], [r["parent_id"] for r in rows[:8]])
        self.assertEqual(len({r["speaker_id"] for r in rows[:8]}), 8)
        self.assertTrue(all(r["proposed_role"] == "proposed_train" for r in rows[:8]))
        for row in rows:
            self.assertIn(row["n_eojeols"], (2, 3, 4))
            self.assertIsNone(row["actual_split"])
            self.assertIsNone(row["frame_count"])
            self.assertIsNone(row["ctc_feasible"])
            self.assertFalse(row["training_ready"])
            self.assertFalse(row["parent_transcript_confirmed"])
            self.assertFalse(row["candidate_boundary_confirmed"])
            self.assertFalse(row["new_human_approval_added"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
