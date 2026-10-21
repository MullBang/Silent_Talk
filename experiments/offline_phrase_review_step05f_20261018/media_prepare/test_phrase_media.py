"""Exact context conversion and preview-bin checks using synthetic clocks."""
from fractions import Fraction
import unittest

import prepare_phrase_media as phrase


class PhraseMediaTests(unittest.TestCase):
    def test_native_context_origin(self):
        origin=Fraction(13543008,48000)
        self.assertEqual(phrase.integer_sample(origin+Fraction(5600+40,16000),origin),5640)

    def test_off_grid_rejected(self):
        with self.assertRaises(ValueError):
            phrase.integer_sample(Fraction(1,48000),Fraction(0))

    def test_before_origin_rejected(self):
        with self.assertRaises(ValueError):
            phrase.integer_sample(Fraction(0),Fraction(1))

    def test_display_overlap_keeps_boundary_bins(self):
        self.assertEqual(phrase.frame_window(Fraction(21,1000),Fraction(81,1000),Fraction(0),100),(0,3))

    def test_exact_display_edge_excludes_next_bin(self):
        self.assertEqual(phrase.frame_window(Fraction(1,25),Fraction(3,25),Fraction(0),100),(1,3))

    def test_no_silent_preview_clamp(self):
        with self.assertRaises(ValueError):
            phrase.frame_window(Fraction(1,25),Fraction(101,25),Fraction(0),100)


if __name__=="__main__":
    unittest.main()
