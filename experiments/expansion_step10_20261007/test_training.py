"""Guard the expansion contract against leakage and silent budget/initialization drift."""
import copy
import unittest
from unittest.mock import patch
import training as t


class PlanContractTests(unittest.TestCase):
    def setUp(self):
        ids = [f'train_{i:03}' for i in range(510)]
        self.ref = {'path': str(t.POLICY), 'sha256': 'frozen-policy'}
        self.policy = {'provisional_train_ids': ids, 'held_train_ids': ['held'],
                       'development_groups': {'old71_same_training_speakers': ['olddev'], 'new24_all_original_items': ['newdev']},
                       'runtime': {'precision': 'fp32'}, 'controls': {'42': {'initial_state_hashes': {'model': 'initial'}}},
                       'optimizer': {'lr': .001}, 'loss': {'feature_lambda': 1.}}
        self.plan = {'schema': 'lipnet-step10-expanded-training-plan-v1', 'seed': 42, 'variant': 'expanded510',
                     'arm': 'B_feature', 'updates': 3200, 'batch_size': 2, 'checkpoint_interval': 100,
                     'ids': ids[:], 'schedule': t.core.fixed_schedule(ids, 42, 3200, 2),
                     'runtime': copy.deepcopy(self.policy['runtime']), 'initial_state_hashes_expected': {'model': 'initial'},
                     'optimizer': copy.deepcopy(self.policy['optimizer']), 'loss': copy.deepcopy(self.policy['loss']),
                     'experiment_plan': self.ref}

    def check(self):
        self.plan['plan_sha256'] = t.runner.canonical_plan_hash(self.plan)
        with patch.object(t, 'bind', return_value=self.ref):
            t.check_contract(self.plan, self.policy)

    def test_complete_fixed_plan_is_accepted(self):
        self.check()

    def test_budget_change_rejected_even_when_rehashed(self):
        for key, value in [('updates', 3201), ('batch_size', 1), ('checkpoint_interval', 200)]:
            original = self.plan[key]; self.plan[key] = value
            with self.assertRaises(ValueError): self.check()
            self.plan[key] = original

    def test_method_change_rejected(self):
        self.plan['arm'] = 'A_ctc'
        with self.assertRaises(ValueError): self.check()

    def test_held_and_development_leakage_rejected(self):
        for sid in ['held', 'olddev', 'newdev']:
            original = self.policy['provisional_train_ids'][0]
            self.policy['provisional_train_ids'][0] = sid
            self.plan['ids'][0] = sid
            self.plan['schedule'] = t.core.fixed_schedule(self.plan['ids'], 42, 3200, 2)
            with self.assertRaises(ValueError): self.check()
            self.policy['provisional_train_ids'][0] = original
            self.plan['ids'][0] = original

    def test_schedule_swap_rejected(self):
        self.plan['schedule'][0], self.plan['schedule'][1] = self.plan['schedule'][1], self.plan['schedule'][0]
        with self.assertRaises(ValueError): self.check()

    def test_learned_initialization_rejected(self):
        self.plan['initial_state_hashes_expected'] = {'model': 'learned'}
        with self.assertRaises(ValueError): self.check()

    def test_runtime_change_rejected(self):
        self.plan['runtime']['precision'] = 'fp16'
        with self.assertRaises(ValueError): self.check()

    def test_optimizer_and_loss_change_rejected(self):
        self.plan['optimizer']['lr'] = .002
        with self.assertRaises(ValueError): self.check()
        self.plan['optimizer'] = copy.deepcopy(self.policy['optimizer'])
        self.plan['loss']['feature_lambda'] = 0.
        with self.assertRaises(ValueError): self.check()

    def test_canonical_hash_and_policy_binding_required(self):
        self.plan['plan_sha256'] = 'wrong'
        with patch.object(t, 'bind', return_value=self.ref), self.assertRaises(ValueError):
            t.check_contract(self.plan, self.policy)
        self.plan['experiment_plan'] = {'path': 'other', 'sha256': 'other'}
        with self.assertRaises(ValueError): self.check()


if __name__ == '__main__':
    unittest.main()
