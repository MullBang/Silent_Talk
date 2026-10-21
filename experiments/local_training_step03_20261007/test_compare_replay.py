"""Acceptance bounds must detect corruption without demanding CUDA bit identity."""
import unittest
import numpy as np
import torch
from compare_replay import compare_state, compare_metrics

POLICY = {"model": {"atol": 1e-6, "rtol": 1e-5}, "adapter": {"atol": 1e-6, "rtol": 1e-5},
          "exp_avg": {"atol": 1e-7, "rtol": 1e-4}, "exp_avg_sq": {"atol": 1e-10, "rtol": 1e-4},
          "loss": {"atol": 1e-5, "rtol": 1e-5}, "gradient_norm": {"atol": 1e-4, "rtol": 1e-4}}


class ReplayComparisonTests(unittest.TestCase):
    def test_small_float_difference_is_reported_not_hidden(self):
        left = {"model": {"weight": torch.tensor([.1, -.1])}}
        right = {"model": {"weight": torch.tensor([.1 + 1e-7, -.1])}}
        result = compare_state(left, right, POLICY)
        self.assertTrue(result["passed"])
        self.assertFalse(result["bitwise_equal"])
        self.assertGreater(result["maximum_absolute_difference"], 0)

    def test_parameter_drift_and_nonfinite_fail(self):
        for value in (.11, float("nan"), float("inf")):
            self.assertFalse(compare_state({"model": {"x": torch.tensor([.1])}},
                                           {"model": {"x": torch.tensor([value])}}, POLICY)["passed"])

    def test_rng_and_adam_step_are_exact(self):
        for left, right in (({"rng": {"cuda": [torch.tensor([1], dtype=torch.uint8)]}},
                             {"rng": {"cuda": [torch.tensor([2], dtype=torch.uint8)]}}),
                            ({"optimizer": {"state": {0: {"step": torch.tensor(4.)}}}},
                             {"optimizer": {"state": {0: {"step": torch.tensor(4.000001)}}}}),
                            ({"rng": {"numpy": np.array([1, 2], dtype=np.uint32)}},
                             {"rng": {"numpy": np.array([1, 3], dtype=np.uint32)}})):
            self.assertFalse(compare_state(left, right, POLICY)["passed"])

    def test_adam_squared_moment_uses_its_own_tight_tolerance(self):
        left = {"optimizer": {"state": {0: {"exp_avg_sq": torch.tensor([0.])}}}}
        right = {"optimizer": {"state": {0: {"exp_avg_sq": torch.tensor([1e-8])}}}}
        self.assertFalse(compare_state(left, right, POLICY)["passed"])

    def test_missing_or_wrong_shape_state_fails(self):
        left = {"model": {"x": torch.tensor([.1])}}
        for right in ({"model": {}}, {"model": {"x": torch.tensor([[.1]])}}, {"model": {"x": torch.tensor([.1], dtype=torch.float64)}}):
            self.assertFalse(compare_state(left, right, POLICY)["passed"])

    def test_metrics_do_not_accept_schedule_or_loss_change(self):
        left = {"ids": ["a", "b"], "lambda": 1., "optimizer_updated": True,
                "ctc_loss": 2., "feature_loss": .5, "loss": 2.5, "gradient_norm_before_clipping": 3.}
        self.assertTrue(compare_metrics(left, dict(left), POLICY)["passed"])
        self.assertFalse(compare_metrics(left, dict(left, loss=3.), POLICY)["passed"])
        with self.assertRaises(RuntimeError):
            compare_metrics(left, dict(left, ids=["b", "a"]), POLICY)


if __name__ == "__main__":
    unittest.main()
