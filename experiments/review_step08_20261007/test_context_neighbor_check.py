import math
import unittest
from context_neighbor_check import interval, overlap


class IntervalTests(unittest.TestCase):
    def test_touching_is_not_overlap(self):
        self.assertEqual(overlap((1, 2), (2, 3)), 0)

    def test_context_can_overlap_while_target_does_not(self):
        self.assertEqual(overlap((1, 2), (2.2, 4)), 0)
        self.assertAlmostEqual(overlap((.5, 2.5), (2.2, 4)), .3)

    def test_enclosed_and_partial(self):
        self.assertEqual(overlap((1, 10), (3, 5)), 2)
        self.assertEqual(overlap((1, 4), (2, 5)), 2)

    def test_invalid_intervals(self):
        for a in [(1, 1), (2, 1), (-1, 2), (0, math.inf), (math.nan, 2), (False, 1)]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                interval(*a)


if __name__ == '__main__':
    unittest.main()
