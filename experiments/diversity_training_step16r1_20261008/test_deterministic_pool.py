"""CPU operator equivalence, architecture guards and no-state-change checks."""
import copy
import unittest
from unittest.mock import Mock
import torch
import deterministic_pool as p
import engine


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cpu_values_and_gradients_match_including_ties_and_overlaps(self):
        result = p.cpu_equivalence()
        self.assertTrue(result['passed'], result)
        self.assertEqual(len(result['cases']), 8)
        for row in result['cases']:
            if row['kind'].startswith('max'):
                self.assertTrue(row['output_bitwise_equal'])
                self.assertTrue(row['input_gradient_bitwise_equal'])

    def test_framewise_maxpool_preserves_frame_and_batch_independence(self):
        x = torch.arange(2*3*4*8*10, dtype=torch.float32).reshape(2, 3, 4, 8, 10)
        self.assertTrue(torch.equal(p.FramewiseMaxPool()(x), torch.nn.MaxPool3d((1,2,2))(x)))

    def test_adaptive_pool_rejects_other_geometry(self):
        with self.assertRaisesRegex(ValueError, '6x6'):
            p.FramewiseAdaptive6to4()(torch.zeros(1,2,3,8,8))

    def test_invalid_rank_rejected(self):
        with self.assertRaisesRegex(ValueError, 'B,C,T'):
            p.FramewiseMaxPool()(torch.zeros(1,2,3,4))

    def test_configuration_preserves_parameters_buffers_state_and_cpu_rng(self):
        model = engine.core.LipNetBaseline()
        original = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        parameter_ids = [id(value) for value in model.parameters()]
        self.assertIs(p.configure_model(model), model)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(parameter_ids, [id(value) for value in model.parameters()])
        self.assertEqual(list(original), list(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in original.items()))
        self.assertEqual(sum(len(list(module.parameters())) for module in [model.spatial_pool, *[b.pool for b in model.conv]]), 0)
        self.assertEqual(list(model.buffers()), [])
        self.assertIs(p.configure_model(model), model)

    def test_configuration_validates_all_original_pools_before_mutation(self):
        model = engine.core.LipNetBaseline()
        model.conv[2].pool = torch.nn.MaxPool3d((1,3,3))
        with self.assertRaisesRegex(ValueError, 'Unexpected original'):
            p.configure_model(model)
        self.assertIsInstance(model.conv[0].pool, torch.nn.MaxPool3d)

    def test_strict_runtime_cannot_silently_use_warning_mode(self):
        old, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            core = Mock()
            p.setup_runtime(core)
            core.setup_runtime.assert_called_once_with()
            result = p.amend_runtime({'strict_deterministic_algorithms': False, 'deterministic_algorithms': True})
            self.assertTrue(result['strict_deterministic_algorithms'])
            self.assertTrue(result['deterministic_algorithms'])
            self.assertFalse(result['deterministic_warn_only'])
            torch.use_deterministic_algorithms(True, warn_only=True)
            with self.assertRaisesRegex(ValueError, 'Strict'):
                p.amend_runtime({})
        finally:
            torch.use_deterministic_algorithms(old, warn_only=warn)


if __name__ == '__main__':
    unittest.main()
