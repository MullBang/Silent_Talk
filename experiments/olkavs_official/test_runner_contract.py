"""Small CPU tests for failure boundaries in the isolated inference adapter."""
import unittest

import numpy as np
import torch

from verify_model import checked_state, normalize_video


class RunnerContractTests(unittest.TestCase):
    def test_video_matches_published_global_normalization_and_time_axis(self):
        frames = np.random.default_rng(2207).integers(0, 256, (5, 224, 224, 3), dtype=np.uint8)
        result, lengths, stats = normalize_video(frames)
        # Independent copy of published dataset.py operations, before layout conversion.
        published = torch.from_numpy(frames).float()
        published -= torch.mean(published)
        published /= torch.std(published)
        torch.testing.assert_close(result[0].permute(1, 2, 3, 0), published)
        self.assertEqual(lengths.tolist(), [5])
        self.assertEqual(tuple(result.shape), (1, 3, 5, 224, 224))
        self.assertAlmostEqual(stats["normalized_std"], 1.0, places=5)

    def test_constant_and_wrong_profile_are_rejected(self):
        for frames in (np.zeros((5, 224, 224, 3), dtype=np.uint8),
                       np.zeros((5, 96, 96, 3), dtype=np.uint8),
                       np.zeros((5, 224, 224, 3), dtype=np.float32)):
            with self.subTest(shape=frames.shape, dtype=frames.dtype), self.assertRaises(ValueError):
                normalize_video(frames)

    def test_partial_and_mismatched_weights_fail_before_mutation(self):
        model = torch.nn.Linear(3, 2)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        for state in ({"weight": torch.ones(2, 3)},
                      {"weight": torch.ones(3, 2), "bias": torch.ones(2)},
                      {**original, "extra": torch.zeros(1)},
                      {"weight": "not a tensor"}):
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                checked_state(model, state)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key])

    def test_exact_state_is_loaded_completely(self):
        source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
        record = checked_state(target, source.state_dict())
        self.assertTrue(record["strict"])
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[key])


if __name__ == "__main__":
    unittest.main()
