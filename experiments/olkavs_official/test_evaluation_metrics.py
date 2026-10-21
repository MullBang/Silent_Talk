import unittest

from evaluation_metrics import aggregate, counts


class EvaluationMetricsTests(unittest.TestCase):
    def test_primary_retains_internal_space_error(self):
        value = counts("너무 신나", "너무  신나")
        self.assertEqual(value["character_edits"], 1)
        self.assertEqual(value["normalized_character_edits"], 0)
        self.assertEqual(value["word_edits"], 0)
        self.assertFalse(value["exact"])

    def test_failed_clip_keeps_its_full_denominator(self):
        rows = [dict(id="a", status="ok", metrics=counts("안녕 세상", "안녕 세상")),
                dict(id="b", status="failed", metrics=counts("오늘", ""))]
        result = aggregate(rows)
        self.assertEqual(result["character_edits"], 2)
        self.assertEqual(result["reference_characters"], 7)
        self.assertEqual(result["word_edits"], 1)
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["failed_inputs"], 1)
        self.assertEqual(result["exact_sentences"], 1)

    def test_word_substitution_insertion_and_unicode_nfc(self):
        value = counts("오늘 집에 간다", "오늘 학교에 빨리 간다")
        self.assertEqual(value["word_edits"], 2)
        self.assertEqual(value["reference_words"], 3)
        self.assertTrue(counts("가", "가")["exact"])

    def test_duplicate_or_empty_cohorts_fail(self):
        item = dict(id="a", status="ok", metrics=counts("가", "가"))
        for rows in ([], [item, item]):
            with self.assertRaises(ValueError):
                aggregate(rows)


if __name__ == "__main__":
    unittest.main()
