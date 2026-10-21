"""Read-only classification contract tests, with synthetic transcripts only."""
import unittest

import classify_asr_differences as subject


def row(reference, prediction, status="ok", index=0):
    return {"id": f"id_{index:03d}", "split": "train", "speaker_id": "S", "source_stem": "source",
            "reference": reference, "asr_text": prediction, "status": status, "audio": None,
            "boundary_flags": [], "confidence_flags": [], "review_status": "human_listening_required",
            **subject.audit.compare_text(reference, prediction)}


class ListeningClassificationTests(unittest.TestCase):
    def test_spacing_only(self):
        self.assertEqual(subject.classify(row("갈 수 있어", "갈수있어"))[0], "spacing_only_difference")

    def test_sino_and_native_integer_spelling_possible(self):
        for reference, prediction in (("십 분", "10분"), ("세 시간", "3시간"), ("이천이십육 년", "2026년")):
            with self.subTest(reference=reference):
                self.assertEqual(subject.classify(row(reference, prediction))[0], "possible_number_expression_difference")

    def test_changed_quantity_or_other_words_remain_possible_content(self):
        for reference, prediction in (("십 분", "20분"), ("세 시간 뒤", "3시간 전"), ("영점오", "0.5")):
            with self.subTest(reference=reference):
                self.assertEqual(subject.classify(row(reference, prediction))[0], "possible_content_difference")

    def test_failed_asr_is_unavailable_not_content_claim(self):
        self.assertEqual(subject.classify(row("안녕", "", "asr_failed"))[0], "asr_unavailable_requires_listening")

    def test_normalized_match_is_not_manual_approval(self):
        rows = [row("안녕!", "안녕", index=index) for index in range(120)]
        value = subject.build({"schema": "step06-local-asr-audit-results-v1", "status": "completed",
                               "ids": [item["id"] for item in rows], "records": rows}, {"sha256": "source"})
        self.assertEqual(value["category_counts"], {"normalized_match": 120})
        self.assertEqual(value["human_listening_confirmed_count"], 0)
        self.assertTrue(all(item["training_ready"] is False for item in value["records"]))

    def test_mismatch_order_preserved_without_accuracy_ranking(self):
        rows = [row("안녕", "안녕", index=index) for index in range(120)]
        rows[1], rows[90] = row("십 분", "10분", index=1), row("갈 수", "갈수", index=90)
        value = subject.build({"schema": "step06-local-asr-audit-results-v1", "status": "completed",
                               "ids": [item["id"] for item in rows], "records": rows}, {"sha256": "source"})
        self.assertEqual([item["id"] for item in value["mismatches_in_original_order"]], ["id_001", "id_090"])


if __name__ == "__main__":
    unittest.main()
