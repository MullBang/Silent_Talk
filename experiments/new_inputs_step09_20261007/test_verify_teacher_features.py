import unittest
import numpy as np
from verify_teacher_features import check_arrays


class SavedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.features = np.ones((2, 512), dtype=np.float32)
        self.student = np.array([11, 13], dtype=np.int64)
        self.native = np.array([10, 11, 12, 13], dtype=np.int64)
        self.gather = np.array([1, 3], dtype=np.int64)

    def check(self, values=None, gather=None):
        check_arrays(self.features if values is None else values, self.student, self.native,
                     self.gather if gather is None else gather, self.student, self.native, self.gather)

    def test_exact_native_identity(self):
        self.check()

    def test_reordered_or_out_of_range_gather(self):
        for ids in [[3, 1], [-1, 3], [1, 4]]:
            with self.assertRaises(ValueError):
                self.check(gather=np.array(ids, dtype=np.int64))

    def test_bad_feature_values(self):
        for value in (0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.check(np.full((2, 512), value, dtype=np.float32))

    def test_wrong_shape_or_dtype(self):
        for array in (np.ones((2, 511), dtype=np.float32), self.features.astype(np.float64)):
            with self.assertRaises(ValueError):
                self.check(array)


if __name__ == '__main__':
    unittest.main()
