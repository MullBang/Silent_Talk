"""Small deterministic checks of native-sample and presentation selection rules."""
from fractions import Fraction
from pathlib import Path
import unittest

import prepare_media as media


class MediaContract(unittest.TestCase):
    def test_half_even_native_sample(self):
        self.assertEqual(media.sample(Fraction(1,96000)),0)
        self.assertEqual(media.sample(Fraction(3,96000)),2)

    def test_decimal_label_time(self):
        self.assertEqual(media.sample(15.402666),739328)

    def test_offset_once(self):
        clock=media.helper(media.CLOCK,"test_clock_offset")
        start=Fraction(739328+1024,48000)
        grid=clock.grid_window(start,start+1,Fraction(0),Fraction(600))
        self.assertTrue(grid["display_start"]<=start<grid["display_start"]+Fraction(1,25))

    def test_nearest_tie_earlier(self):
        clock=media.helper(media.CLOCK,"test_clock_tie")
        self.assertEqual(clock.nearest([Fraction(0),Fraction(1,30)],Fraction(1,60))[0],0)

    def test_native_invalid_range_rejected_before_read(self):
        with self.assertRaises(ValueError):
            media.native_excerpt(Path("missing.wav"),Path("missing-output.wav"),30,20)

    def test_missing_bracket_rejected(self):
        clock=media.helper(media.CLOCK,"test_clock_missing")
        with self.assertRaises(ValueError):
            clock.nearest([Fraction(0),Fraction(1,30)],Fraction(1,10))


if __name__=="__main__":
    unittest.main()
