"""CPU execution, LR-boundary, failure preservation and exact resume contracts."""
from __future__ import annotations
import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import engine as e

torch, np = e.torch, e.np


class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.LayerNorm(256),
                                               torch.nn.ReLU(), torch.nn.Dropout(.1))
        self.fc = torch.nn.Linear(256, 69)
    def forward(self, pixels):
        hidden = self.projection(pixels.mean(dim=(-1, -2)).transpose(1, 2))
        return self.fc(hidden).log_softmax(-1).transpose(0, 1).contiguous()


def cpu_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state().clone(), 'cuda': []}


def restore_cpu_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def tiny_objects(seed, arm, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    model, adapter = TinyStudent().to(device), torch.nn.Linear(256, 512).to(device)
    optimizer = torch.optim.Adam([*model.parameters(), *adapter.parameters()], lr=.001,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    return model, adapter, optimizer


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='step16_cpu_', dir=e.ROOT)
        self.base = Path(self.temp.name)
        generator = np.random.default_rng(13)
        self.clips = {sid: {'id': sid, 'split': 'train', 'pixels': generator.integers(0, 256, (frames, 96, 96, 3), np.uint8),
                            'tokens': [1, 2], 'teacher_features': generator.normal(size=(frames, 512)).astype(np.float32)}
                      for sid, frames in [('train_a', 4), ('train_b', 6)]}
        self.runtime = {'device': 'cpu-only-test', 'precision': 'float32'}
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(e.core, 'make_objects', side_effect=tiny_objects))
        self.stack.enter_context(patch.object(e.core, 'capture_rng', side_effect=cpu_rng))
        self.stack.enter_context(patch.object(e.core, 'restore_rng', side_effect=restore_cpu_rng))
        self.stack.enter_context(patch.object(e.core, 'setup_runtime', side_effect=lambda: torch.set_num_threads(2)))
        self.stack.enter_context(patch.object(e, 'runtime_for', return_value=self.runtime))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.plan = {'seed': 42, 'cohort': 'cpu-smoke', 'arm': 'B_feature', 'run_kind': 'smoke', 'updates': 6,
                     'batch_size': 2, 'checkpoint_interval': 2, 'schedule': [['train_a', 'train_b']] * 6,
                     'lr_segments': copy.deepcopy(e.SMOKE_LR), 'time_budget_seconds': 120,
                     'initial_state_hashes_expected': e.runner.state_hashes(e.core.runtime_state(*tiny_objects(42, 'B_feature', 'cpu'))),
                     'runtime': self.runtime, 'execution_code': e.execution_code()}
        self.rebind()

    def rebind(self):
        self.plan.pop('plan_sha256', None)
        binding = e.runner.validate_plan(self.plan, self.clips, 'B_feature')
        self.plan['training_data'] = {key: binding[key] for key in ('data_sha256', 'clips')}
        self.plan['plan_sha256'] = e.runner.canonical_plan_hash(self.plan)

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def run_segment(self, name, **kwargs):
        return e.run_segment(self.plan, self.clips, self.base / name, device='cpu', **kwargs)

    def test_full_run_lr_history_and_checkpoint_updates(self):
        result = self.run_segment('full')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([row['learning_rate'] for row in result['updates']], [.001, .001, .0003, .0003, .0001, .0001])
        self.assertEqual([ref['completed_updates'] for ref in result['checkpoints']], [2, 4, 6])
        payload = e.load_checkpoint(result['checkpoint']['path'])
        e.validate_checkpoint_payload(self.plan, payload, 6)
        self.assertEqual(payload['next_update'], 7)
        self.assertIsNone(payload['next_learning_rate'])
        self.assertEqual(len(payload['history']), 6)

    def test_resume_across_lr_boundary_matches_continuous_full_state(self):
        full = self.run_segment('full')
        split = self.run_segment('split', stop_update=2)
        saved = e.load_checkpoint(split['checkpoint']['path'])
        self.assertEqual(saved['state']['optimizer']['param_groups'][0]['lr'], .001)
        self.assertEqual(saved['next_learning_rate'], .0003)
        resumed = self.run_segment('resumed', resume_checkpoint=Path(split['checkpoint']['path']))
        self.assertTrue(resumed['restore_verification']['exact'])
        self.assertEqual(resumed['start_update'], 2)
        self.assertEqual(resumed['updates'][0]['learning_rate'], .0003)
        self.assertEqual(full['final_state_hashes'], resumed['final_state_hashes'])
        self.assertEqual(resumed['resume_count'], 1)

    def test_second_resume_rejected(self):
        first = self.run_segment('first', stop_update=2)
        second = self.run_segment('second', stop_update=4, resume_checkpoint=Path(first['checkpoint']['path']))
        with self.assertRaisesRegex(ValueError, 'at most one resume'):
            self.run_segment('third', resume_checkpoint=Path(second['checkpoint']['path']))
        self.assertFalse((self.base / 'third').exists())

    def test_changed_lr_history_rejected(self):
        result = self.run_segment('full')
        payload = e.load_checkpoint(result['checkpoint']['path'])
        payload['history'][2]['learning_rate'] = .001
        with self.assertRaisesRegex(ValueError, 'history/schedule/LR'):
            e.validate_checkpoint_payload(self.plan, payload, 6)

    def test_changed_next_global_lr_rejected(self):
        result = self.run_segment('split', stop_update=2)
        payload = e.load_checkpoint(result['checkpoint']['path'])
        payload['next_learning_rate'] = .001
        with self.assertRaisesRegex(ValueError, 'Next global'):
            e.validate_checkpoint_payload(self.plan, payload, 2)

    def test_changed_saved_optimizer_lr_rejected_even_if_state_rehashed(self):
        result = self.run_segment('split', stop_update=2)
        payload = e.load_checkpoint(result['checkpoint']['path'])
        payload['state']['optimizer']['param_groups'][0]['lr'] = .0003
        payload['state_hashes'] = e.runner.state_hashes(payload['state'])
        payload['history'][-1]['state_hashes'] = payload['state_hashes']
        with self.assertRaisesRegex(ValueError, 'Saved optimizer LR'):
            e.validate_checkpoint_payload(self.plan, payload, 2)

    def test_failed_update_keeps_last_complete_state_for_resume(self):
        original = e.core.optimizer_update
        calls = 0
        def fail_third(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('simulated pre-update resource failure')
            return original(*args, **kwargs)
        with patch.object(e.core, 'optimizer_update', side_effect=fail_third):
            result = self.run_segment('failed')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['completed_updates'], 2)
        self.assertEqual(result['failed_or_unrecorded_update_attempts'], 1)
        saved = e.load_checkpoint(result['checkpoint']['path'])
        e.validate_checkpoint_payload(self.plan, saved, 2)
        continued = self.run_segment('recovered', resume_checkpoint=Path(result['checkpoint']['path']),
                                     prior_elapsed_seconds=result['elapsed_training_seconds'])
        full = self.run_segment('full')
        self.assertEqual(continued['final_state_hashes'], full['final_state_hashes'])

    def test_failure_after_optimizer_mutation_does_not_save_partial_state(self):
        def fail_after_mutation(model, adapter, optimizer, *args):
            with torch.no_grad():
                next(model.parameters()).add_(123.)
            raise RuntimeError('partial state must not be trusted')
        with patch.object(e.core, 'optimizer_update', side_effect=fail_after_mutation):
            result = self.run_segment('failed')
        saved = e.load_checkpoint(result['checkpoint']['path'])
        e.validate_checkpoint_payload(self.plan, saved, 0)
        self.assertEqual(saved['state_hashes'], self.plan['initial_state_hashes_expected'])
        self.assertEqual(saved['history'], [])

    def test_timeout_before_first_update_saves_zero_update_checkpoint(self):
        result = self.run_segment('timeout', remaining_time_seconds=1e-12)
        self.assertEqual(result['status'], 'time_budget_exhausted')
        self.assertEqual(result['completed_updates'], 0)
        e.validate_checkpoint_payload(self.plan, e.load_checkpoint(result['checkpoint']['path']), 0)

    def test_consumed_time_cannot_be_erased_by_resume(self):
        result = self.run_segment('split', stop_update=2)
        with self.assertRaisesRegex(ValueError, 'erase consumed time'):
            self.run_segment('resumed', resume_checkpoint=Path(result['checkpoint']['path']), prior_elapsed_seconds=0)

    def test_resume_cannot_extend_original_budget(self):
        result = self.run_segment('split', stop_update=2)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            self.run_segment('resumed', resume_checkpoint=Path(result['checkpoint']['path']), prior_elapsed_seconds=120.)

    def test_changed_pixels_rejected_before_output(self):
        self.clips['train_a']['pixels'][0, 0, 0, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'input bytes changed'):
            self.run_segment('changed')
        self.assertFalse((self.base / 'changed').exists())

    def test_evaluation_clip_scope_rejected(self):
        self.clips['train_a']['split'] = 'development'
        with self.assertRaisesRegex(ValueError, 'Non-training'):
            self.run_segment('bad')

    def test_missing_teacher_target_rejected(self):
        self.clips['train_a'].pop('teacher_features')
        with self.assertRaisesRegex(ValueError, 'Missing teacher'):
            self.run_segment('bad')

    def test_frozen_source_change_rejected(self):
        self.plan['execution_code'][0]['sha256'] = '0' * 64
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'execution code'):
            self.run_segment('changed')

    def test_wrong_fresh_initial_state_rejected(self):
        self.plan['initial_state_hashes_expected']['model'] = '0' * 64
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'Fresh initialization'):
            self.run_segment('bad')

    def test_existing_output_is_not_overwritten(self):
        output = self.base / 'existing'
        output.mkdir()
        (output / 'keep.txt').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            self.run_segment('existing')
        self.assertEqual((output / 'keep.txt').read_text(), 'preserve')


class LearningRateTests(unittest.TestCase):
    def test_actual_global_boundaries(self):
        plan = {'updates': 3600, 'lr_segments': e.MAIN_LR}
        self.assertEqual([e.learning_rate(plan, update) for update in (1, 2400, 2401, 3000, 3001, 3600)],
                         [.001, .001, .0003, .0003, .0001, .0001])

    def test_gap_or_overlap_lr_policy_rejected(self):
        for start in (2400, 2402):
            segments = copy.deepcopy(e.MAIN_LR)
            segments[1]['start_update'] = start
            with self.assertRaisesRegex(ValueError, 'Noncontiguous'):
                e.validate_lr_segments(segments, 3600)

    def test_out_of_range_update_rejected(self):
        for update in (0, 3601):
            with self.assertRaisesRegex(ValueError, 'outside global'):
                e.learning_rate({'updates': 3600, 'lr_segments': e.MAIN_LR}, update)

    def test_smoke_counts_are_separate_actual_prefixes(self):
        contract = e.smoke_contract()
        self.assertEqual(contract['initial_update'], 0)
        self.assertEqual(contract['continuous_updates'] + contract['split_first_updates'] + contract['split_resumed_updates'], 12)
        self.assertFalse(contract['fake_history_allowed'])
        self.assertEqual(contract['main_optimizer_updates'], 0)


if __name__ == '__main__':
    unittest.main()
