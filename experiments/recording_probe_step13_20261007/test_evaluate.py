"""CPU-only tests of isolated inference and immutable result reuse."""
import copy
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

import evaluate as subject

torch.set_num_threads(2)


def logposterior(ids):
    logits = torch.full((len(ids), 69), -7., dtype=torch.float32)
    logits[torch.arange(len(ids)), ids] = 3.
    return logits.log_softmax(-1).numpy()


class ToyStudent(nn.Module):
    def __init__(self, mode='valid'):
        super().__init__()
        self.register_buffer('marker', torch.tensor(1.))
        self.mode = mode
        self.calls = 0

    def forward(self, pixels):
        self.calls += 1
        if self.mode == 'fail':
            raise RuntimeError('synthetic inference failure')
        if self.mode == 'mutate':
            self.marker.add_(1.)
        ids = [0, 0, 0, 0] if self.mode == 'blank' else [1, 1, 20, 20]
        if self.mode == 'invalid':
            ids = [20, 20, 0, 20]
        result = torch.from_numpy(logposterior(ids)).unsqueeze(1)
        return result * float('nan') if self.mode == 'nan' else result


def fixture():
    groups = {name: [] for name in (*subject.SAME_GROUPS, *subject.DIFFERENT_GROUPS)}
    clips = {}
    for speaker in range(6):
        sid = f'same_s{speaker}'
        clips[sid] = {'id': sid, 'split': subject.ROLE, 'diagnostic_role': subject.ROLE,
                      'source_split': 'train', 'speaker_id': f'speaker{speaker}',
                      'source_stem': f's{speaker}_train_recording', 'text': '가',
                      'pixels': np.zeros((4, 96, 96, 3), dtype=np.uint8),
                      'tokens': subject.decoder.tokenizer.encode_text('가')}
        groups['same6'].append(sid)
        groups['different6'].append(f'different_s{speaker}')
        suffix = 'old4' if speaker < 4 else 'new2'
        groups[f'same_{suffix}'].append(sid)
        groups[f'different_{suffix}'].append(f'different_s{speaker}')
    return clips, groups


def evaluate(mode='valid'):
    clips, groups = fixture()
    arrays = {}
    model = ToyStudent(mode).eval()
    result = subject.evaluate_clips(model, clips, groups, 'cpu', arrays)
    return result, arrays, model


def reuse_fixture():
    result, arrays, _ = evaluate()
    _, groups = fixture()
    conditions, archive, bound = {}, {}, {}
    for condition, values in result['conditions'].items():
        records = []
        for index, original in enumerate(values['records']):
            row = copy.deepcopy(original)
            sid = groups['different6'][index]
            row.update(id=sid, split='internal_holdout', diagnostic_role='internal_holdout',
                       group='hold_old4' if index < 4 else 'hold_new4',
                       source_stem=f's{index}_different_recording')
            records.append(row)
            archive[row['log_posteriors_key']] = arrays[row['log_posteriors_key']]
            bound[sid] = {'speaker_id': row['speaker_id'], 'source_stem': row['source_stem'],
                          'text': row['reference'], 'frames': row['frames'],
                          'tokens_sha256': subject.runner.nested_hash(row['target_token_ids'])}
        for index in range(26):
            row = copy.deepcopy(records[0])
            row['id'] = f'irrelevant{index}'
            records.append(row)
        conditions[condition] = {'records': records}
    reference = {'state_hashes': {'model': 'final-model'}, 'completed_updates': 3600}
    plan = {'seed': 42, 'plan_sha256': 'plan-digest', 'training_data': {'data_sha256': 'training-data'},
            'evaluation_data': {'data_sha256': 'evaluation-data', 'clips': bound},
            'initial_state_hashes_expected': {'A_ctc': {'model': 'initial'}},
            'diagnostic_groups': {'hold8': groups['different6'] + ['extra1', 'extra2']}}
    source = {'schema': 'lipnet-step12-multisentence-evaluation-v1', 'status': 'completed',
              'seed': 42, 'arm': 'A_ctc', 'plan_sha256': 'plan-digest',
              'evaluation_data_sha256': 'evaluation-data', 'data_sha256': 'training-data',
              'initial_state_hashes_verified': {'model': 'initial'}, 'milestones': [
                  {'completed_updates': 3600, 'checkpoint': reference, 'model_state_unchanged': True,
                   'model_state_before': 'final-model', 'model_state_after': 'final-model', 'conditions': conditions}]}
    return source, plan, {'arm': 'A_ctc', 'reference': reference}, {'groups': groups}, archive


class EvaluationTests(unittest.TestCase):
    def test_three_conditions_infer18_rows_once_and_separate_strata(self):
        value, arrays, model = evaluate()
        self.assertEqual(model.calls, 18)
        self.assertEqual(len(arrays), 18)
        for condition, result in value['conditions'].items():
            self.assertEqual(len(result['records']), 6)
            self.assertEqual(set(result['groups']), set(subject.SAME_GROUPS))
            self.assertNotIn('all12', result['groups'])
            for name, size in subject.SAME_GROUPS.items():
                self.assertEqual(result['groups'][name]['aggregate']['clips'], size)
                self.assertEqual(result['groups'][name]['aggregate']['exact_sentences'], size)
            self.assertTrue(all(row['condition'] == condition and row['diagnostic_role'] == subject.ROLE for row in result['records']))

    def test_saved_fp32_posteriors_reproduce_actual_ctc(self):
        value, arrays, _ = evaluate()
        for row in value['conditions']['normal']['records']:
            lp = arrays[row['log_posteriors_key']]
            self.assertEqual(lp.dtype, np.float32)
            self.assertEqual(lp.shape, (4, 69))
            expected = torch.nn.functional.ctc_loss(torch.from_numpy(lp).unsqueeze(1),
                torch.tensor(row['target_token_ids']), torch.tensor([4]), torch.tensor([2]),
                blank=0, reduction='mean', zero_infinity=False).item()
            self.assertEqual(row['ctc_loss_mean'], expected)

    def test_failed_and_nonfinite_rows_keep_denominator_without_fake_ctc(self):
        for mode in ('fail', 'nan'):
            result, arrays, _ = evaluate(mode)
            self.assertEqual(arrays, {})
            for values in result['conditions'].values():
                for name, size in subject.SAME_GROUPS.items():
                    group = values['groups'][name]
                    self.assertEqual(group['aggregate']['failed_inputs'], size)
                    self.assertEqual(group['aggregate']['reference_characters'], size)
                    self.assertEqual(group['aggregate']['cer_percent'], 100.)
                    self.assertEqual(group['diagnostics']['ctc_loss_count'], 0)
                    self.assertIsNone(group['diagnostics']['ctc_loss_mean_over_successful_clips'])

    def test_blank_is_success_and_uncomposable_jamos_are_retained(self):
        blank, arrays, _ = evaluate('blank')
        self.assertEqual(len(arrays), 18)
        for row in blank['conditions']['normal']['records']:
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['prediction'], '')
        invalid, _, _ = evaluate('invalid')
        for row in invalid['conditions']['normal']['records']:
            self.assertEqual(row['collapsed_ids'], [20, 20])
            self.assertEqual(row['prediction'], '\ufffd\ufffd')
            self.assertEqual(row['strict_label_decoder_diagnostic']['status'], 'failed')

    def test_teacher_audio_train_or_global_val_role_rejected_before_model(self):
        for field, value in (('teacher_features', None), ('audio', None), ('split', 'train'),
                             ('diagnostic_role', 'internal_holdout'), ('source_split', 'dev')):
            clips, groups = fixture()
            clips[groups['same6'][0]][field] = value
            model = ToyStudent().eval()
            with self.assertRaises(ValueError):
                subject.evaluate_clips(model, clips, groups, 'cpu', {})
            self.assertEqual(model.calls, 0)

    def test_invalid_tokens_pixels_and_duplicate_speakers_rejected(self):
        for field, value in (('tokens', [1, 21]), ('pixels', np.zeros((4, 96, 96, 3), dtype=np.float32)),
                             ('speaker_id', 'speaker1')):
            clips, groups = fixture()
            clips[groups['same6'][0]][field] = value
            with self.assertRaises(ValueError):
                subject.validate_same_clips(clips, groups)

    def test_strata_duplicates_and_crossovers_rejected(self):
        for mode in ('duplicate', 'crossover', 'missing'):
            clips, groups = fixture()
            if mode == 'duplicate':
                groups['same6'][0] = groups['same6'][1]
            elif mode == 'crossover':
                groups['same_old4'][0] = groups['same_new2'][0]
            else:
                del clips[groups['same6'][0]]
            with self.assertRaises(ValueError):
                subject.validate_same_clips(clips, groups)

    def test_training_mode_gradients_or_state_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'changed student state'):
            evaluate('mutate')
        clips, groups = fixture()
        for model in (ToyStudent(), nn.Linear(1, 1).eval()):
            with self.assertRaisesRegex(ValueError, 'frozen eval'):
                subject.evaluate_clips(model, clips, groups, 'cpu', {})

    def test_reuse_is_exact_unmodified_original_rows_without_model_calls(self):
        source, plan, checkpoint, selection, archive = reuse_fixture()
        before = copy.deepcopy(source)
        with mock.patch.object(subject.core, 'make_objects', side_effect=AssertionError('No model allowed')):
            values = subject.validate_reused_rows(source, plan, checkpoint, selection, archive)
        self.assertEqual(source, before)
        for condition, value in values.items():
            self.assertTrue(value['reused'])
            self.assertEqual(value['records'], source['milestones'][0]['conditions'][condition]['records'][:6])
            self.assertEqual(set(value['groups']), set(subject.DIFFERENT_GROUPS))
            self.assertEqual(value['groups']['different6']['aggregate']['clips'], 6)
            self.assertTrue(all(row['diagnostic_role'] == 'internal_holdout' for row in value['records']))

    def test_reuse_wrong_seed_arm_training_or_evaluation_digest_rejected(self):
        for field, value in (('seed', 43), ('arm', 'B_feature'), ('data_sha256', 'bad'),
                             ('evaluation_data_sha256', 'bad'), ('plan_sha256', 'bad')):
            args = reuse_fixture()
            args[0][field] = value
            with self.assertRaises(ValueError):
                subject.validate_reused_rows(*args)

    def test_reuse_missing_duplicate_or_changed_final_checkpoint_rejected(self):
        for mode in ('missing', 'duplicate', 'checkpoint', 'mutation'):
            args = reuse_fixture()
            finals = args[0]['milestones']
            if mode == 'missing':
                finals[0]['completed_updates'] = 2700
            elif mode == 'duplicate':
                finals.append(copy.deepcopy(finals[0]))
            elif mode == 'checkpoint':
                finals[0]['checkpoint'] = {'state_hashes': {'model': 'other'}}
            else:
                finals[0]['model_state_after'] = 'other'
            with self.assertRaises(ValueError):
                subject.validate_reused_rows(*args)

    def test_reuse_transcript_condition_token_role_or_metrics_tampering_rejected(self):
        for field, value in (('reference', '나'), ('condition', 'shuffled'), ('target_token_ids', [1, 21]),
                             ('split', 'train'), ('source_stem', 'other'), ('metrics', {})):
            args = reuse_fixture()
            args[0]['milestones'][0]['conditions']['normal']['records'][0][field] = value
            with self.assertRaises(ValueError):
                subject.validate_reused_rows(*args)

    def test_reuse_posterior_dtype_or_decode_corruption_rejected(self):
        for mode in ('dtype', 'decode', 'nan'):
            args = reuse_fixture()
            key = args[0]['milestones'][0]['conditions']['normal']['records'][0]['log_posteriors_key']
            if mode == 'dtype':
                args[4][key] = args[4][key].astype(np.float64)
            elif mode == 'decode':
                args[4][key] = logposterior([0, 0, 0, 0])
            else:
                args[4][key] = np.full((4, 69), np.nan, dtype=np.float32)
            with self.assertRaises(ValueError):
                subject.validate_reused_rows(*args)

    def test_reuse_failed_rows_retain_denominator(self):
        args = reuse_fixture()
        for condition in subject.CONDITIONS:
            row = args[0]['milestones'][0]['conditions'][condition]['records'][0]
            row.update(status='failed', prediction='', ctc_loss_mean=None, output_diagnostics=None,
                       log_posteriors_key=None, raw_argmax_ids=[], collapsed_ids=[],
                       metrics=subject.core.counts('가', ''),
                       token_metrics={'token_edits': 2, 'target_tokens': 2, 'exact': False})
        values = subject.validate_reused_rows(*args)
        for value in values.values():
            self.assertEqual(value['groups']['different6']['aggregate']['clips'], 6)
            self.assertEqual(value['groups']['different6']['aggregate']['failed_inputs'], 1)
            self.assertEqual(value['groups']['different6']['diagnostics']['ctc_loss_count'], 5)

    def test_new_and_reused_group_partition_errors_are_not_inference_failures(self):
        args = reuse_fixture()
        args[3]['groups']['different_new2'][0] = args[3]['groups']['different_old4'][0]
        with self.assertRaises(ValueError):
            subject.validate_reused_rows(*args)


if __name__ == '__main__':
    unittest.main()
