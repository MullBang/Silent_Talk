"""CPU synthetic tests for Step11 metrics, failure accounting and provenance."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import evaluate as subject


def logposterior(ids):
    logits = torch.full((len(ids), 69), -7., dtype=torch.float32)
    logits[torch.arange(len(ids)), ids] = 3.
    return logits.log_softmax(-1).numpy()


class ToyStudent(nn.Module):
    def __init__(self, mode='invalid'):
        super().__init__()
        self.register_buffer('marker', torch.tensor(1.))
        self.mode = mode

    def forward(self, pixels):
        if self.mode == 'fail':
            raise RuntimeError('synthetic inference failure')
        if self.mode == 'mutate':
            self.marker.add_(1.)
        ids = [0, 0, 0, 0] if self.mode == 'blank' else [20, 20, 0, 20]
        if self.mode == 'valid':
            ids = [1, 1, 20, 20]
        result = torch.from_numpy(logposterior(ids)).unsqueeze(1)
        return result * float('nan') if self.mode == 'nan' else result


def clips_fixture():
    clips = {f'train{i}': {'id': f'train{i}', 'split': 'train', 'source_split': 'train',
              'speaker_id': f'speaker{i}', 'source_stem': f'recording{i}', 'text': '가',
              'pixels': np.zeros((4, 96, 96, 3), dtype=np.uint8), 'tokens': [1, 20]}
             for i in range(8)}
    return clips, {'old4': list(clips)[:4], 'new4': list(clips)[4:]}


def evaluate(mode='invalid', conditions=('normal',)):
    clips, groups = clips_fixture()
    arrays = {}
    value = subject.evaluate_clips(ToyStudent(mode).eval(), clips, groups, conditions,
                                  torch.device('cpu'), 1200, arrays)
    return value, arrays


def checkpoint_fixture(arm='A_ctc', update=300):
    ids = [f'train{i}' for i in range(8)]
    plan = {'seed': 42, 'updates': 1200, 'checkpoint_interval': 300, 'batch_size': 2,
            'runtime': {'device': 'synthetic'},
            'schedule': [[ids[2 * i % 8], ids[(2 * i + 1) % 8]] for i in range(1200)]}
    plan['plan_sha256'] = subject.runner.canonical_plan_hash(plan)
    validation = {'plan_sha256': plan['plan_sha256'], 'data_sha256': 'data', 'clips': {sid: {} for sid in ids}}
    state = {'model': {'w': torch.tensor([1.])}, 'adapter': {}, 'optimizer': {}, 'rng': {}}
    hashes = subject.runner.state_hashes(state)
    identity = {'plan_sha256': plan['plan_sha256'], 'seed': 42, 'arm': arm, 'total_updates': 1200,
                'batch_size': 2, 'schedule_sha256': subject.runner.nested_hash(plan['schedule']),
                'data_sha256': 'data', 'clip_bindings': validation['clips'],
                'code_bindings': subject.runner._code_bindings(), 'runtime': plan['runtime'],
                'optimization': subject.optimization_contract(arm)}
    initial = {'model': 'initial', 'adapter': 'initial_adapter', 'optimizer': 'arm_specific', 'rng': 'initial_rng', 'complete': 'initial_all'}
    payload = {'schema': subject.runner.SCHEMA, 'completed_updates': update, 'identity': identity,
               'initial_state_hashes': initial, 'state': state, 'state_hashes': hashes,
               'history': [{'update': i + 1, 'ids': plan['schedule'][i], 'state_hashes': hashes} for i in range(update)]}
    return plan, validation, payload, initial


class EvaluationTests(unittest.TestCase):
    def test_ctc_repeat_before_blank_removal_and_invalid_jamos_preserved(self):
        decoded = subject.decoder.decode_prediction(logposterior([20, 20, 0, 20]))
        self.assertEqual(decoded['collapsed_ids'], [20, 20])
        self.assertEqual(decoded['prediction'], '\ufffd\ufffd')
        self.assertEqual(decoded['strict_replay']['status'], 'failed')
        valid = subject.decoder.decode_prediction(logposterior([1, 1, 20, 20]))
        self.assertEqual(valid['prediction'], '가')

    def test_blank_argmax_is_not_mean_posterior(self):
        lp = np.full((4, 69), np.log(.5 / 68), dtype=np.float32)
        lp[:, 0] = np.log(.5)
        decoded = subject.decoder.decode_prediction(lp)
        value = subject.output_diagnostics(lp, decoded, '가')
        self.assertEqual(value['blank_argmax_frames'], 4)
        self.assertEqual(value['blank_argmax_fraction'], 1.)
        self.assertAlmostEqual(value['mean_blank_posterior'], .5)
        self.assertEqual(sum(value['raw_token_histogram'].values()), 4)

    def test_unnormalized_wrong_dtype_and_nonfinite_rejected(self):
        lp = logposterior([0, 0, 0, 0])
        decoded = subject.decoder.decode_prediction(lp)
        for changed in (lp + 2., lp.astype(np.float64), lp * float('nan')):
            with self.assertRaises(ValueError):
                subject.output_diagnostics(changed, decoded, '가')

    def test_whitespace_separates_runs_and_histograms_keep_spaces(self):
        self.assertEqual(subject.run_lengths(' 아아아아아 아아아아아아\t가 '), [5, 6, 1])
        decoded = subject.decoder.decode_prediction(logposterior([0, 0, 0, 0]))
        decoded['prediction'] = ' 아아아아아 아아아아아아 '
        value = subject.output_diagnostics(logposterior([0, 0, 0, 0]), decoded, ' 가  나 ')
        self.assertEqual(value['longest_nonspace_same_character_run'], 6)
        self.assertEqual(value['same_character_runs_ge5'], 2)
        self.assertEqual(value['character_histogram'], {' ': 1, '아': 11})
        self.assertEqual(value['nonspace_character_histogram'], {'아': 11})
        self.assertEqual(value['reference_characters'], 4)

    def test_actual_ctc_matches_cpu_and_saved_probability_arrays(self):
        value, arrays = evaluate('valid')
        condition = value['conditions']['normal']
        self.assertEqual(condition['groups']['all8']['aggregate']['exact_sentences'], 8)
        self.assertEqual(condition['groups']['all8']['diagnostics']['token_exact_sentences'], 8)
        for row in condition['records']:
            lp = torch.from_numpy(arrays[row['log_posteriors_key']]).unsqueeze(1)
            expected = torch.nn.functional.ctc_loss(lp, torch.tensor([1, 20]), torch.tensor([4]), torch.tensor([2]),
                                                   blank=0, reduction='mean', zero_infinity=False).item()
            self.assertEqual(row['ctc_loss_mean'], expected)
            self.assertEqual(row['output_diagnostics']['raw_token_histogram'], {'1': 2, '20': 2})
        self.assertEqual(len(arrays), 8)
        self.assertTrue(value['model_state_unchanged'])

    def test_eval_all_conditions_and_group_aggregation_without_extra_forward(self):
        value, arrays = evaluate('invalid', subject.CONDITIONS)
        self.assertEqual(len(arrays), 24)
        for condition in value['conditions'].values():
            self.assertEqual(len(condition['records']), 8)
            self.assertEqual(condition['groups']['all8']['aggregate']['reference_characters'], 8)
            self.assertEqual(condition['groups']['all8']['aggregate']['character_edits'], 16)
            self.assertEqual(condition['groups']['old4']['aggregate']['clips'], 4)
            self.assertEqual(condition['groups']['new4']['aggregate']['clips'], 4)
            self.assertEqual(condition['groups']['all8']['diagnostics']['replacement_characters'], 16)
            self.assertEqual(condition['groups']['all8']['diagnostics']['target_tokens'], 16)

    def test_failed_and_nonfinite_inference_keeps_errors_but_not_fabricated_loss(self):
        for mode in ('fail', 'nan'):
            value, arrays = evaluate(mode)
            condition = value['conditions']['normal']
            self.assertEqual(arrays, {})
            self.assertEqual(condition['groups']['all8']['aggregate']['cer_percent'], 100.)
            self.assertEqual(condition['groups']['all8']['diagnostics']['ctc_loss_count'], 0)
            self.assertIsNone(condition['groups']['all8']['diagnostics']['ctc_loss_mean_over_successful_clips'])
            self.assertIsNone(condition['groups']['all8']['diagnostics']['mean_blank_posterior'])
            for row in condition['records']:
                self.assertEqual(row['status'], 'failed')
                self.assertEqual(row['token_metrics']['token_edits'], 2)
                self.assertIsNone(row['output_diagnostics'])
                self.assertIsNone(row['ctc_loss_mean'])

    def test_empty_cohort_or_reference_denominator_is_never_reported_as_perfect(self):
        with self.assertRaises(ValueError):
            subject.core.aggregate([])
        with self.assertRaisesRegex(ValueError, 'Empty reference denominator'):
            subject.core.aggregate([{'id': 'empty', 'status': 'ok', 'metrics': subject.core.counts('', '')}])

    def test_blank_output_is_successful_inference_with_positive_ctc(self):
        value, arrays = evaluate('blank')
        self.assertEqual(len(arrays), 8)
        for row in value['conditions']['normal']['records']:
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['prediction'], '')
            self.assertGreater(row['ctc_loss_mean'], 0.)
            self.assertEqual(row['output_diagnostics']['blank_argmax_fraction'], 1.)

    def test_model_mutation_training_mode_grad_and_dev_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'changed student state'):
            evaluate('mutate')
        clips, groups = clips_fixture()
        with self.assertRaisesRegex(ValueError, 'frozen eval'):
            subject.evaluate_clips(ToyStudent(), clips, groups, ('normal',), 'cpu', 0, {})
        clips['train0']['split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'Nontraining'):
            subject.evaluate_clips(ToyStudent().eval(), clips, groups, ('normal',), 'cpu', 0, {})

    def test_temporal_conditions_reproduce_fixed_source_positions(self):
        pixels = np.arange(19)[:, None]
        static = subject.original.condition_metadata(pixels, 'static', 'same')
        shuffled = subject.original.condition_metadata(pixels, 'shuffled', 'same')
        self.assertEqual(static['source_positions'], [9] * 19)
        self.assertEqual(sorted(shuffled['source_positions']), list(range(19)))
        self.assertNotEqual(shuffled['source_positions'], list(range(19)))
        np.random.seed(900)
        self.assertEqual(shuffled, subject.original.condition_metadata(pixels, 'shuffled', 'same'))

    def test_group_membership_and_transcript_token_source_roles_are_bound(self):
        clips, groups = clips_fixture()
        self.assertEqual(subject.validate_groups({'diagnostic_groups': groups}, clips), groups)
        for field, value in (('split', 'dev'), ('source_split', 'val'), ('tokens', [1, 21]), ('audio', None)):
            changed = copy.deepcopy(clips)
            changed['train0'][field] = value
            with self.assertRaises(ValueError):
                subject.validate_groups({'diagnostic_groups': groups}, changed)
        changed = copy.deepcopy(groups)
        changed['new4'][0] = changed['old4'][0]
        with self.assertRaises(ValueError):
            subject.validate_groups({'diagnostic_groups': changed}, clips)

    def test_every_planned_checkpoint_valid_for_both_arms(self):
        for arm in subject.core.ARMS:
            for update in subject.MILESTONES[1:]:
                plan, validation, payload, initial = checkpoint_fixture(arm, update)
                subject.validate_checkpoint_payload(plan, validation, arm, payload, update, initial)

    def test_checkpoint_changes_in_all_identity_fields_are_rejected(self):
        changes = {'arm': 'B_feature', 'seed': 43, 'total_updates': 3200, 'batch_size': 1,
                   'schedule_sha256': 'changed', 'data_sha256': 'changed', 'clip_bindings': {},
                   'code_bindings': [], 'runtime': {}, 'optimization': {}, 'plan_sha256': 'changed'}
        for field, changed in changes.items():
            plan, validation, payload, initial = checkpoint_fixture()
            payload['identity'][field] = changed
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 300, initial)

    def test_checkpoint_history_state_initial_and_position_are_rejected(self):
        for change in ('history', 'history_state', 'state', 'initial', 'position', 'plan'):
            plan, validation, payload, initial = checkpoint_fixture()
            if change == 'history':
                payload['history'][0]['ids'] = ['wrong', 'wrong']
            elif change == 'history_state':
                payload['history'][-1]['state_hashes'] = {'model': 'other'}
            elif change == 'state':
                payload['state']['model']['w'].add_(1.)
            elif change == 'initial':
                payload['initial_state_hashes'] = {'model': 'other'}
            elif change == 'position':
                payload['completed_updates'] = 299
            else:
                plan['seed'] = 43
            with self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 300, initial)
        plan, validation, payload, initial = checkpoint_fixture()
        with self.assertRaisesRegex(ValueError, 'Unplanned'):
            subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 150, initial)

    def test_checkpoint_reference_milestones_and_bytes_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            refs = []
            for update in subject.MILESTONES[1:]:
                path = Path(directory) / f'{update}.pt'
                sidecar = path.with_suffix('.pt.json')
                path.write_bytes(b'checkpoint')
                sidecar.write_text('{}', encoding='utf-8')
                refs.append({'path': str(path), 'sha256': subject.bind(path)['sha256'],
                             'sidecar': str(sidecar), 'sidecar_sha256': subject.bind(sidecar)['sha256'],
                             'completed_updates': update})
            subject.validate_checkpoint_refs(refs)
            for changed in (refs[:-1], refs[::-1], refs + refs[:1]):
                with self.assertRaisesRegex(ValueError, 'milestone'):
                    subject.validate_checkpoint_refs(changed)
            Path(refs[0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                subject.validate_checkpoint_refs(refs)

    def test_contract_fixed_record_count_and_code_bindings_complete(self):
        contract = subject.evaluation_contract()
        self.assertEqual(contract['milestones'], [0, 300, 600, 900, 1200])
        self.assertEqual(contract['records_per_run'], 56)
        bindings = subject.evaluation_code()
        self.assertIn(subject.bind(Path(subject.__file__)), bindings)
        self.assertIn(subject.bind(Path(__file__)), bindings)
        self.assertEqual(len(bindings), len({ref['path'] for ref in bindings}))


if __name__ == '__main__':
    unittest.main()
