import unittest

from diagnostic_utils import collapse_ctc, edit_distance, text_counts


class DiagnosticHelpersTests(unittest.TestCase):
    def test_blank_separates_repeated_tokens(self):
        self.assertEqual(collapse_ctc([3, 5, 5, 3, 5, 7, 7, 3, 2, 2]), [5, 5, 7, 2])
        self.assertEqual(collapse_ctc([3, 3]), [])

    def test_edit_counts_include_missing_entire_sentence(self):
        self.assertEqual(edit_distance("안녕", ""), 2)
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance([1, 2, 3], [1, 3]), 1)

    def test_whitespace_and_unicode_contract(self):
        counts = text_counts("  안녕   세상 ", "안녕세상")
        self.assertEqual(counts["character_edits_with_spaces"], 1)
        self.assertEqual(counts["character_edits_without_spaces"], 0)
        self.assertFalse(counts["exact_sentence"])


if __name__ == "__main__":
    unittest.main()
