"""CPU-only tests of train/hold isolation, checkpoint lineage and metrics."""
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
    def __init__(self, mode='valid'):
        super().__init__()
        self.register_buffer('marker', torch.tensor(1.))
        self.mode = mode

    def forward(self, pixels):
        if self.mode == 'fail':
            raise RuntimeError('synthetic inference failure')
        if self.mode == 'mutate':
            self.marker.add_(1.)
        ids = [0, 0, 0, 0] if self.mode == 'blank' else [1, 1, 20, 20]
        if self.mode == 'invalid':
            ids = [20, 20, 0, 20]
        result = torch.from_numpy(logposterior(ids)).unsqueeze(1)
        return result * float('nan') if self.mode == 'nan' else result


def clips_fixture():
    train, evaluation = {}, {}
    groups = {name: [] for name in subject.GROUP_SIZES}
    for speaker in range(8):
        generation = 'old' if speaker < 4 else 'new'
        for index in range(4):
            role = 'train' if index < 3 else 'internal_holdout'
            sid = f'{role}_s{speaker}_u{index}'
            text = '가' if role == 'train' else '나'
            clip = {'id': sid, 'split': role, 'source_split': 'train', 'diagnostic_role': role,
                    'speaker_id': f'speaker{speaker}', 'source_stem': f's{speaker}_{role}_recording',
                    'text': text, 'pixels': np.zeros((4, 96, 96, 3), dtype=np.uint8),
                    'tokens': subject.decoder.tokenizer.encode_text(text)}
            evaluation[sid] = clip
            if role == 'train':
                train[sid] = {**clip, 'teacher_features': np.ones((4, 512), dtype=np.float32)}
                groups['train24'].append(sid)
                groups[f'train_{generation}12'].append(sid)
            else:
                groups['hold8'].append(sid)
                groups[f'hold_{generation}4'].append(sid)
    return train, evaluation, groups


def evaluate(mode='valid', conditions=('normal',)):
    train, clips, groups = clips_fixture()
    arrays = {}
    value = subject.evaluate_clips(ToyStudent(mode).eval(), clips, groups, conditions,
                                  torch.device('cpu'), 3600, arrays)
    return value, arrays


def checkpoint_fixture(arm='A_ctc', update=900):
    ids = [f'train{i}' for i in range(24)]
    plan = {'seed': 42, 'updates': 3600, 'checkpoint_interval': 900, 'batch_size': 2,
            'runtime': {'device': 'synthetic'},
            'schedule': [[ids[2 * i % 24], ids[(2 * i + 1) % 24]] for i in range(3600)]}
    plan['plan_sha256'] = subject.runner.canonical_plan_hash(plan)
    validation = {'plan_sha256': plan['plan_sha256'], 'data_sha256': 'data24', 'clips': {sid: {} for sid in ids}}
    state = {'model': {'w': torch.tensor([1.])}, 'adapter': {}, 'optimizer': {}, 'rng': {}}
    hashes = subject.runner.state_hashes(state)
    identity = {'plan_sha256': plan['plan_sha256'], 'seed': 42, 'arm': arm, 'total_updates': 3600,
                'batch_size': 2, 'schedule_sha256': subject.runner.nested_hash(plan['schedule']),
                'data_sha256': 'data24', 'clip_bindings': validation['clips'],
                'code_bindings': subject.runner._code_bindings(), 'runtime': plan['runtime'],
                'optimization': subject.prior.optimization_contract(arm)}
    initial = {'model': 'initial', 'adapter': 'initial_adapter', 'optimizer': 'arm_specific', 'rng': 'initial_rng', 'complete': 'initial_all'}
    payload = {'schema': subject.runner.SCHEMA, 'completed_updates': update, 'identity': identity,
               'initial_state_hashes': initial, 'state': state, 'state_hashes': hashes,
               'history': [{'update': i + 1, 'ids': plan['schedule'][i], 'state_hashes': hashes} for i in range(update)]}
    return plan, validation, payload, initial


class EvaluationTests(unittest.TestCase):
    def test_train_and_hold_metrics_are_separate_without_all32_headline(self):
        value, arrays = evaluate()
        condition = value['conditions']['normal']
        self.assertEqual(len(arrays), 32)
        self.assertEqual(set(condition['groups']), set(subject.GROUP_SIZES))
        self.assertNotIn('all32', condition['groups'])
        self.assertEqual(condition['groups']['train24']['aggregate']['exact_sentences'], 24)
        self.assertEqual(condition['groups']['train24']['aggregate']['cer_percent'], 0.)
        self.assertEqual(condition['groups']['hold8']['aggregate']['exact_sentences'], 0)
        self.assertEqual(condition['groups']['hold8']['aggregate']['cer_percent'], 100.)
        self.assertEqual(condition['groups']['train_old12']['aggregate']['clips'], 12)
        self.assertEqual(condition['groups']['hold_new4']['aggregate']['clips'], 4)
        self.assertEqual([row['diagnostic_role'] for row in condition['records']], ['train'] * 24 + ['internal_holdout'] * 8)
        self.assertTrue(all(row['source_split'] == 'train' for row in condition['records']))

    def test_all_temporal_conditions_infer96_rows_once_despite_six_aggregations(self):
        value, arrays = evaluate(conditions=subject.CONDITIONS)
        self.assertEqual(len(arrays), 96)
        self.assertEqual(sum(len(item['records']) for item in value['conditions'].values()), 96)
        self.assertTrue(value['model_state_unchanged'])

    def test_actual_ctc_for_train_and_hold_matches_saved_cpu_posteriors(self):
        value, arrays = evaluate()
        rows = value['conditions']['normal']['records']
        for row in (rows[0], rows[-1]):
            lp = torch.from_numpy(arrays[row['log_posteriors_key']]).unsqueeze(1)
            expected = torch.nn.functional.ctc_loss(lp, torch.tensor(row['target_token_ids']), torch.tensor([4]), torch.tensor([2]),
                                                   blank=0, reduction='mean', zero_infinity=False).item()
            self.assertEqual(row['ctc_loss_mean'], expected)
        self.assertLess(rows[0]['ctc_loss_mean'], rows[-1]['ctc_loss_mean'])

    def test_invalid_jamos_preserved_and_empty_blank_output_is_success(self):
        invalid, _ = evaluate('invalid')
        for row in invalid['conditions']['normal']['records']:
            self.assertEqual(row['collapsed_ids'], [20, 20])
            self.assertEqual(row['prediction'], '\ufffd\ufffd')
            self.assertEqual(row['strict_label_decoder_diagnostic']['status'], 'failed')
        blank, arrays = evaluate('blank')
        self.assertEqual(len(arrays), 32)
        for row in blank['conditions']['normal']['records']:
            self.assertEqual(row['status'], 'ok')
            self.assertEqual(row['prediction'], '')
            self.assertGreater(row['ctc_loss_mean'], 0.)

    def test_failed_inference_retains_both_cohort_denominators_without_fabricated_ctc(self):
        for mode in ('fail', 'nan'):
            value, arrays = evaluate(mode)
            self.assertEqual(arrays, {})
            for name, size in subject.GROUP_SIZES.items():
                group = value['conditions']['normal']['groups'][name]
                self.assertEqual(group['aggregate']['failed_inputs'], size)
                self.assertEqual(group['aggregate']['reference_characters'], size)
                self.assertEqual(group['aggregate']['cer_percent'], 100.)
                self.assertEqual(group['diagnostics']['ctc_loss_count'], 0)
                self.assertIsNone(group['diagnostics']['ctc_loss_mean_over_successful_clips'])

    def test_teacher_audio_and_role_contamination_abort_before_inference(self):
        for field, value in (('teacher_features', None), ('audio', None), ('split', 'train'), ('source_split', 'val')):
            _, clips, groups = clips_fixture()
            clips[groups['hold8'][0]][field] = value
            with self.assertRaises(ValueError):
                subject.evaluate_clips(ToyStudent().eval(), clips, groups, ('normal',), 'cpu', 0, {})

    def test_training_mode_grad_and_model_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'changed student state'):
            evaluate('mutate')
        _, clips, groups = clips_fixture()
        with self.assertRaisesRegex(ValueError, 'frozen eval'):
            subject.evaluate_clips(ToyStudent(), clips, groups, ('normal',), 'cpu', 0, {})
        with self.assertRaisesRegex(ValueError, 'frozen eval'):
            subject.evaluate_clips(nn.Linear(1, 1).eval(), clips, groups, ('normal',), 'cpu', 0, {})

    def test_groups_validate_three_train_and_one_separate_recording_per_speaker(self):
        train, clips, groups = clips_fixture()
        self.assertEqual(subject.validate_groups({'diagnostic_groups': groups}, train, clips), groups)
        for mode in ('speaker', 'recording', 'stratum', 'duplicate', 'optimized_hold', 'pixel_copy'):
            train, clips, groups = clips_fixture()
            if mode == 'speaker':
                clips[groups['hold8'][0]]['speaker_id'] = 'foreign_speaker'
            elif mode == 'recording':
                clips[groups['hold8'][0]]['source_stem'] = clips[groups['train24'][0]]['source_stem']
            elif mode == 'stratum':
                groups['train_old12'][0] = groups['train_new12'][0]
            elif mode == 'duplicate':
                groups['hold8'][0] = groups['hold8'][1]
            elif mode == 'optimized_hold':
                train[groups['hold8'][0]] = clips[groups['hold8'][0]]
            else:
                train[groups['train24'][0]]['pixels'] = np.ones((4, 96, 96, 3), dtype=np.uint8)
            with self.assertRaises(ValueError, msg=mode):
                subject.validate_groups({'diagnostic_groups': groups}, train, clips)

    def test_evaluation_binding_binds_roles_text_speakers_and_tensor_bytes(self):
        _, clips, groups = clips_fixture()
        baseline = subject.evaluation_bindings(clips)
        for field, value in (('speaker_id', 'other'), ('source_stem', 'other_source'), ('pixels', np.ones((4, 96, 96, 3), dtype=np.uint8))):
            changed = copy.deepcopy(clips)
            changed[groups['hold8'][0]][field] = value
            self.assertNotEqual(baseline['data_sha256'], subject.evaluation_bindings(changed)['data_sha256'])
        for field, value in (('source_split', 'dev'), ('diagnostic_role', 'train'), ('teacher_features', None), ('tokens', [1, 21])):
            changed = copy.deepcopy(clips)
            changed[groups['hold8'][0]][field] = value
            with self.assertRaises(ValueError):
                subject.evaluation_bindings(changed)

    def test_schedule_bindings_cannot_include_internal_holdout(self):
        train, clips, groups = clips_fixture()
        plan = {'seed': 42, 'updates': 12, 'checkpoint_interval': 3, 'batch_size': 2,
                'schedule': [groups['train24'][index:index + 2] for index in range(0, 24, 2)]}
        subject.runner.validate_plan(plan, train, 'B_feature')
        plan['schedule'][0][0] = groups['hold8'][0]
        with self.assertRaises(ValueError):
            subject.runner.validate_plan(plan, train, 'B_feature')

    def test_blank_frequency_and_posterior_remain_distinct(self):
        lp = np.full((4, 69), np.log(.5 / 68), dtype=np.float32)
        lp[:, 0] = np.log(.5)
        value = subject.prior.output_diagnostics(lp, subject.decoder.decode_prediction(lp), '가')
        self.assertEqual(value['blank_argmax_fraction'], 1.)
        self.assertAlmostEqual(value['mean_blank_posterior'], .5)
        self.assertEqual(subject.prior.run_lengths('아아아아아 아아아아아아'), [5, 6])

    def test_all_declared_checkpoints_valid_for_both_arms(self):
        for arm in subject.core.ARMS:
            for update in subject.MILESTONES[1:]:
                plan, validation, payload, initial = checkpoint_fixture(arm, update)
                subject.validate_checkpoint_payload(plan, validation, arm, payload, update, initial)

    def test_checkpoint_identity_rejects_arm_budget_runtime_data_and_code_changes(self):
        changes = {'arm': 'B_feature', 'seed': 43, 'total_updates': 1200, 'batch_size': 1,
                   'schedule_sha256': 'changed', 'data_sha256': 'hold_contamination', 'clip_bindings': {},
                   'code_bindings': [], 'runtime': {}, 'optimization': {}, 'plan_sha256': 'changed'}
        for field, changed in changes.items():
            plan, validation, payload, initial = checkpoint_fixture()
            payload['identity'][field] = changed
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 900, initial)

    def test_checkpoint_history_state_init_position_and_plan_changes_rejected(self):
        for change in ('history', 'history_state', 'state', 'initial', 'position', 'plan'):
            plan, validation, payload, initial = checkpoint_fixture()
            if change == 'history':
                payload['history'][0]['ids'] = ['held', 'held']
            elif change == 'history_state':
                payload['history'][-1]['state_hashes'] = {'model': 'other'}
            elif change == 'state':
                payload['state']['model']['w'].add_(1.)
            elif change == 'initial':
                payload['initial_state_hashes'] = {'model': 'other'}
            elif change == 'position':
                payload['completed_updates'] = 899
            else:
                plan['seed'] = 43
            with self.assertRaises(ValueError):
                subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 900, initial)
        plan, validation, payload, initial = checkpoint_fixture()
        with self.assertRaisesRegex(ValueError, 'Unplanned'):
            subject.validate_checkpoint_payload(plan, validation, 'A_ctc', payload, 300, initial)

    def test_checkpoint_refs_require_all_four_exact_hashes(self):
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

    def test_contract_record_count_scope_and_frozen_dependencies(self):
        contract = subject.evaluation_contract()
        self.assertEqual(contract['milestones'], [0, 900, 1800, 2700, 3600])
        self.assertEqual(contract['records_per_run'], 224)
        self.assertEqual(contract['headline_groups'], ['train24', 'hold8'])
        self.assertTrue(contract['internal_holdout_inputs_loaded'])
        self.assertFalse(contract['original_held27_inputs_loaded'])
        refs = subject.evaluation_code()
        self.assertIn(subject.bind(Path(subject.__file__)), refs)
        self.assertIn(subject.bind(subject.PRIOR_PATH), refs)
        self.assertEqual(len(refs), len({ref['path'] for ref in refs}))


if __name__ == '__main__':
    unittest.main()
