"""CPU-only evidence, failure-accounting, and scheduled-cohort tests."""
import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

import evaluate as subject

torch.set_num_threads(2)


class ToyStudent(nn.Module):
    def __init__(self, mode='valid'):
        super().__init__()
        self.register_buffer('marker', torch.tensor(1.))
        self.mode, self.calls = mode, 0

    def forward(self, pixels):
        self.calls += 1
        if self.mode == 'fail':
            raise RuntimeError('synthetic inference failure')
        if self.mode == 'mutate':
            self.marker.add_(1.)
        ids = [0, 0, 0, 0] if self.mode == 'blank' else [1, 1, 20, 20]
        if self.mode == 'invalid':
            ids = [20, 20, 0, 20]
        logits = torch.full((4, 69), -7., dtype=torch.float32)
        logits[torch.arange(4), ids] = 3.
        lp = logits.log_softmax(-1).unsqueeze(1)
        if self.mode == 'nan':
            lp *= float('nan')
        if self.mode == 'unnormalized':
            lp += 1.
        return lp


def fixture(cohort='train24'):
    data = {'train': {}, 'dev': {}, 'probe': {}}
    train24, train72, dev, probe, known, unseen = [], [], [], [], [], []

    def add(pool, sid, speaker):
        data[pool][sid] = {'id': sid, 'speaker_id': f'speaker{speaker}',
                          'source_stem': f'{pool}_recording{speaker}', 'text': '가',
                          'pixels': np.arange(4, dtype=np.uint8)[:, None, None, None]
                                    * np.ones((4, 96, 96, 3), dtype=np.uint8),
                          'tokens': subject.decoder.tokenizer.encode_text('가')}

    for speaker in range(8):
        for index in range(9):
            sid = f'train_s{speaker}_{index}'
            add('train', sid, speaker)
            train72.append(sid)
            if index < 3:
                train24.append(sid)
        for index in range(2):
            sid = f'dev_s{speaker}_{index}'
            add('dev', sid, speaker)
            dev.append(sid)
    for speaker, count in ((0, 2), (1, 2), (2, 2), (3, 2), (8, 2), (9, 2), (10, 1), (11, 1)):
        for index in range(count):
            sid = f'probe_s{speaker}_{index}'
            add('probe', sid, speaker)
            probe.append(sid)
            (known if speaker < 8 else unseen).append(sid)
    plan = {'cohort': cohort, 'train_ids': train24 if cohort == 'train24' else train72,
            'common_train24_ids': train24, 'dev_ids': dev, 'probe_ids': probe,
            'probe_known_ids': known, 'probe_unseen_ids': unseen}
    return data, plan


def evaluate(mode='valid', cohort='train24', update=3600):
    data, plan = fixture(cohort)
    clips = subject.evaluation_inputs(data, plan)
    arrays, model = {}, ToyStudent(mode).eval()
    value = subject.evaluate_milestone(model, clips, plan, update, 'cpu', arrays)
    return value, arrays, model


def checkpoint_fixture():
    identity = {'name': 'the-full-training-identity'}
    identity_sha = subject.runner.nested_hash(identity)
    plan = {'seed': 42, 'cohort': 'train24', 'arm': 'B_feature', 'plan_sha256': 'plan',
            'experiment_plan': {'path': 'policy', 'sha256': 'policysha'}, 'train_ids': ['a'], 'ids': ['a'],
            'initial_state_hashes_expected': {'model': 'initial', 'optimizer': 'optimizer0'}, 'runtime': {'device': 'cpu'}}
    history = [{'update': u, 'ids': ['a', 'b']} for u in range(1, 3601)]
    refs, payloads = [], {}
    for update in subject.ALL_CHECKPOINTS:
        hashes = {'model': f'model{update}', 'adapter': f'adapter{update}', 'optimizer': f'opt{update}', 'rng': f'rng{update}'}
        ref = {'path': f'checkpoint{update}', 'completed_updates': update, 'identity_sha256': identity_sha, 'state_hashes': hashes}
        refs.append(ref)
        payloads[ref['path']] = {'state': {'model': {'weight': torch.tensor(float(update))}},
            'state_hashes': hashes, 'history': history[:update], 'initial_state_hashes': plan['initial_state_hashes_expected'],
            'next_update': update+1, 'next_learning_rate': None if update == 3600 else .001, 'resume_count': 0}
    run = {'schema': 'lipnet-step16-training-run-result-v1', 'status': 'completed', 'run_kind': 'main',
        'seed': 42, 'cohort': 'train24', 'arm': 'B_feature', 'plan_file': {'path': 'plan', 'sha256': 'plansha'},
        'experiment_plan': plan['experiment_plan'], 'plan_sha256': 'plan', 'start_update': 0, 'completed_updates': 3600,
        'updates': history, 'new_successful_optimizer_updates': 3600, 'training_input_ids': ['a'],
        'initial_state_hashes': plan['initial_state_hashes_expected'], 'runtime': plan['runtime'], 'resume_count': 0,
        'checkpoints': refs, 'checkpoint': refs[-1], 'final_checkpoint': refs[-1],
        'identity_sha256': identity_sha, 'final_state_hashes': refs[-1]['state_hashes'],
        **{name: False for name in ('dev_inputs_loaded', 'probe_inputs_loaded', 'audio_used', 'teacher_model_loaded', 'service_weights_updated')}}
    engine = SimpleNamespace(load_checkpoint=mock.Mock(side_effect=lambda path: copy.deepcopy(payloads[str(path)])),
                             validate_checkpoint_payload=mock.Mock(return_value=identity))
    return plan, run, payloads, engine


class EvaluationTests(unittest.TestCase):
    def test_revision_r1_configures_parameter_free_pooling_before_state_validation(self):
        model, adapter, optimizer = ToyStudent(), object(), object()
        before = subject.runner.nested_hash(model.state_dict())
        with mock.patch.object(subject.core, 'make_objects', return_value=(model, adapter, optimizer)) as make, \
             mock.patch.object(subject.pooling, 'configure_model') as configure:
            actual = subject.make_evaluation_objects(42, 'cpu')
        make.assert_called_once_with(42, 'B_feature', 'cpu')
        configure.assert_called_once_with(model)
        self.assertEqual(actual, (model, adapter, optimizer))
        self.assertEqual(subject.runner.nested_hash(model.state_dict()), before)

    def test_pooling_configuration_cannot_silently_change_model_state(self):
        model = ToyStudent()
        def corrupt(value):
            value.marker.add_(1.)
        with mock.patch.object(subject.core, 'make_objects', return_value=(model, None, None)), \
             mock.patch.object(subject.pooling, 'configure_model', side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, 'changed model parameters'):
                subject.make_evaluation_objects(42, 'cpu')

    def test_revision_r1_pooling_helper_and_strict_runtime_are_bound(self):
        refs = subject.evaluation_code()
        self.assertIn(subject.bind(subject.BASE / 'deterministic_pool.py'), refs)
        self.assertTrue(subject.evaluation_contract()['strict_deterministic_algorithms'])
        self.assertEqual(subject.evaluation_contract()['execution_revision'],
                         'step16r1_deterministic_framewise_spatial_pooling')

    def test_exact_824_schedule_without_duplicate_common24_inference(self):
        totals = {}
        for cohort, expected in (('train24', 158), ('train72', 254)):
            data, plan = fixture(cohort)
            clips = subject.evaluation_inputs(data, plan)
            model, arrays, rows = ToyStudent().eval(), {}, 0
            for update in subject.UPDATES:
                value = subject.evaluate_milestone(model, clips, plan, update, 'cpu', arrays)
                self.assertEqual(tuple(value['conditions']), subject.CONDITIONS if update == 3600 else ('normal',))
                for condition, result in value['conditions'].items():
                    ids, groups = subject.condition_groups(plan, update, condition)
                    self.assertEqual([row['id'] for row in result['records']], ids)
                    self.assertEqual(set(result['groups']), set(groups))
                    rows += len(result['records'])
            self.assertEqual((rows, model.calls, len(arrays)), (expected,) * 3)
            totals[cohort] = rows
        self.assertEqual(2 * sum(totals.values()), 824)

    def test_native_common_dev_and_probe_strata_are_separate(self):
        value, arrays, model = evaluate(cohort='train72')
        normal = value['conditions']['normal']
        sizes = {'train_native': 72, 'train_common24': 24, 'dev16': 16,
                 'historical_probe14': 14, 'probe_known8': 8, 'probe_unseen6': 6}
        self.assertEqual(len(normal['records']), 102)
        for group, count in sizes.items():
            self.assertEqual(normal['groups'][group]['aggregate']['clips'], count)
            speakers = normal['per_speaker'][group]
            self.assertEqual(len(speakers), 4 if group in ('probe_known8', 'probe_unseen6') else 8)
            self.assertEqual(sum(v['aggregate']['clips'] for v in speakers.values()), count)
        for condition in ('static', 'shuffled'):
            values = value['conditions'][condition]
            self.assertEqual(set(values['groups']), {'dev16'})
            self.assertTrue(all(row['split'] == 'internal_dev' for row in values['records']))
        self.assertEqual((len(arrays), model.calls), (134, 134))

    def test_saved_fp32_posteriors_reproduce_ctc_and_tolerant_decode(self):
        value, arrays, _ = evaluate(update=1200)
        for row in value['conditions']['normal']['records']:
            lp = arrays[row['log_posteriors_key']]
            self.assertEqual((lp.dtype, lp.shape), (np.float32, (4, 69)))
            expected = torch.nn.functional.ctc_loss(torch.from_numpy(lp).unsqueeze(1),
                torch.tensor(row['target_token_ids']), torch.tensor([4]), torch.tensor([2]),
                blank=0, reduction='mean', zero_infinity=False).item()
            self.assertEqual(row['ctc_loss_mean'], expected)
            self.assertEqual(row['raw_argmax_ids'], [1, 1, 20, 20])
            self.assertEqual(row['collapsed_ids'], [1, 20])
            self.assertEqual(row['prediction'], '가')

    def test_failures_keep_fixed_denominators_and_null_losses(self):
        for mode in ('fail', 'nan', 'unnormalized'):
            value, arrays, _ = evaluate(mode, update=1200)
            self.assertEqual(arrays, {})
            group = value['conditions']['normal']['groups']['dev16']
            self.assertEqual(group['aggregate']['clips'], 16)
            self.assertEqual(group['aggregate']['failed_inputs'], 16)
            self.assertEqual(group['aggregate']['reference_characters'], 16)
            self.assertEqual(group['aggregate']['cer_percent'], 100.)
            self.assertEqual(group['diagnostics']['ctc_loss_count'], 0)
            self.assertIsNone(group['diagnostics']['ctc_loss_mean_over_successful_clips'])
            for row in value['conditions']['normal']['records']:
                self.assertIsNone(row['ctc_loss_mean'])
                self.assertIsNone(row['log_posteriors_key'])
                self.assertEqual(row['collapsed_ids'], [])

    def test_blank_is_success_and_invalid_composition_remains_visible(self):
        for mode in ('blank', 'invalid'):
            value, arrays, _ = evaluate(mode, update=1200)
            self.assertEqual(len(arrays), 16)
            for row in value['conditions']['normal']['records']:
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(row['prediction'], '' if mode == 'blank' else '\ufffd\ufffd')
                if mode == 'invalid':
                    self.assertEqual(row['collapsed_ids'], [20, 20])
                    self.assertEqual(row['replacement_characters'], 2)
                    self.assertEqual(row['strict_label_decoder_diagnostic']['status'], 'failed')

    def test_eval_inputs_are_pixel_only_with_current_and_historical_roles(self):
        data, plan = fixture()
        for clip in data['train'].values():
            clip['teacher_features'] = np.zeros((4, 512), dtype=np.float32)
        clips = subject.evaluation_inputs(data, plan)
        self.assertEqual(len(clips), 54)
        self.assertTrue(all('teacher_features' not in clip and clip['source_split'] == 'train' for clip in clips.values()))
        self.assertTrue(all(clips[sid]['split'] == 'internal_dev' for sid in plan['dev_ids']))
        self.assertTrue(all(clips[sid]['seen_in_current_training'] for sid in plan['probe_known_ids']))
        self.assertTrue(all(not clips[sid]['seen_in_current_training'] for sid in plan['probe_unseen_ids']))

    def test_held_teacher_features_audio_or_wrong_strata_rejected(self):
        for mode in ('teacher', 'audio', 'devspeaker', 'known', 'overlap', 'duplicate', 'common'):
            data, plan = fixture()
            if mode == 'teacher':
                data['dev'][plan['dev_ids'][0]]['teacher_features'] = None
            elif mode == 'audio':
                data['train'][plan['train_ids'][0]]['audio'] = None
            elif mode == 'devspeaker':
                data['dev'][plan['dev_ids'][0]]['speaker_id'] = 'unseen'
            elif mode == 'known':
                plan['probe_known_ids'][0] = plan['probe_unseen_ids'][0]
            elif mode == 'overlap':
                plan['dev_ids'][0] = plan['train_ids'][0]
            elif mode == 'duplicate':
                plan['train_ids'][0] = plan['train_ids'][1]
            else:
                plan['common_train24_ids'][0] = 'outside'
            with self.assertRaises(ValueError):
                subject.evaluation_inputs(data, plan)

    def test_bad_tokens_pixels_or_roles_fail_before_any_model_call(self):
        for field, value in (('tokens', [1, 21]), ('pixels', np.zeros((4, 96, 96, 3), dtype=np.float32)),
                             ('source_split', 'test'), ('audio', None), ('teacher_features', None)):
            data, plan = fixture()
            clips = subject.evaluation_inputs(data, plan)
            clips[plan['dev_ids'][-1]][field] = value
            model = ToyStudent().eval()
            with self.assertRaises(ValueError):
                subject.evaluate_milestone(model, clips, plan, 1200, 'cpu', {})
            self.assertEqual(model.calls, 0)

    def test_binding_detects_pixel_token_and_role_changes_without_mutation(self):
        data, plan = fixture()
        clips = subject.evaluation_inputs(data, plan)
        before = subject.input_bindings(clips)
        snapshot = copy.deepcopy(clips)
        values = subject.evaluate_milestone(ToyStudent().eval(), clips, plan, 3600, 'cpu', {})
        for condition, result in values['conditions'].items():
            for row in result['records']:
                self.assertEqual(row['temporal'], subject.original.condition_metadata(clips[row['id']]['pixels'], condition, row['id']))
        self.assertEqual(subject.input_bindings(clips), before)
        sid = plan['dev_ids'][0]
        clips[sid]['pixels'][0, 0, 0, 0] = 99
        self.assertNotEqual(subject.input_bindings(clips), before)
        self.assertEqual(snapshot[sid]['pixels'][0, 0, 0, 0], 0)

    def test_train_mode_gradients_and_state_mutation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'changed student state'):
            evaluate('mutate', update=1200)
        data, plan = fixture()
        clips = subject.evaluation_inputs(data, plan)
        for model in (ToyStudent(), nn.Linear(1, 1).eval()):
            with self.assertRaisesRegex(ValueError, 'frozen eval'):
                subject.evaluate_milestone(model, clips, plan, 1200, 'cpu', {})

    def test_unplanned_milestones_temporal_conditions_and_duplicate_keys_rejected(self):
        _, plan = fixture()
        for update, condition in ((600, 'normal'), (1200, 'static'), (3000, 'shuffled'), (3600, 'other')):
            with self.assertRaises(ValueError):
                subject.condition_groups(plan, update, condition)
        data, plan = fixture()
        clips = subject.evaluation_inputs(data, plan)
        model = ToyStudent().eval()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            subject.evaluate_milestone(model, clips, plan, 1200, 'cpu', {'u1200_normal_0': None})
        self.assertEqual(model.calls, 0)

    def test_full_checkpoint_manifest_order_and_bindings_required(self):
        refs = [{'completed_updates': u, 'path': f'/dummy/c{u}.pt', 'sidecar': f'/dummy/c{u}.pt.json',
                 'sha256': 'checkpoint', 'sidecar_sha256': 'sidecar'} for u in subject.ALL_CHECKPOINTS]
        def fake_bind(path):
            return {'sha256': 'sidecar' if str(path).endswith('.json') else 'checkpoint'}
        with mock.patch.object(subject, 'bind', side_effect=fake_bind):
            subject.validate_checkpoint_refs(refs)
            for mode in ('missing', 'reordered', 'wrong_sidecar', 'digest'):
                changed = copy.deepcopy(refs)
                if mode == 'missing':
                    changed.pop(0)
                elif mode == 'reordered':
                    changed.reverse()
                elif mode == 'wrong_sidecar':
                    changed[0]['sidecar'] = '/dummy/other.json'
                else:
                    changed[0]['sha256'] = 'bad'
                with self.assertRaises(ValueError):
                    subject.validate_checkpoint_refs(changed)

    def test_all_six_checkpoints_are_verified_but_only_four_model_states_retained(self):
        plan, run, _, engine = checkpoint_fixture()
        states, evidence = subject.validated_model_states(plan, run, engine)
        self.assertEqual(list(states), list(subject.UPDATES))
        self.assertEqual([row['history_updates_verified'] for row in evidence], list(subject.ALL_CHECKPOINTS))
        self.assertEqual(engine.load_checkpoint.call_count, 6)
        self.assertEqual(engine.validate_checkpoint_payload.call_count, 6)
        self.assertTrue(all(row['restricted_cpu_load'] for row in evidence))
        self.assertEqual(states[3600]['weight'].item(), 3600.)

    def test_checkpoint_prefix_initial_identity_or_full_state_tampering_rejected(self):
        for mode in ('history', 'initial', 'identity', 'state', 'final'):
            plan, run, payloads, engine = checkpoint_fixture()
            if mode == 'history':
                payloads['checkpoint600']['history'][5] = {'update': 6, 'ids': ['bad']}
            elif mode == 'initial':
                payloads['checkpoint600']['initial_state_hashes'] = {'model': 'other'}
            elif mode == 'identity':
                run['identity_sha256'] = 'wrong'
            elif mode == 'state':
                payloads['checkpoint600']['state_hashes'] = {'model': 'other'}
            else:
                run['final_state_hashes'] = {'model': 'other'}
            with self.assertRaises(ValueError):
                subject.validated_model_states(plan, run, engine)

    def test_production_run_scope_and_contract_validation(self):
        plan, run, _, _ = checkpoint_fixture()
        plan_ref, policy_ref = run['plan_file'], run['experiment_plan']
        with mock.patch.object(subject.runner, 'canonical_plan_hash', return_value='plan'), \
             mock.patch.object(subject, 'validate_checkpoint_refs'):
            subject.validate_training_result(run, plan, plan_ref, policy_ref)
            for field, value in (('cohort', 'train72'), ('arm', 'A_ctc'), ('run_kind', 'smoke'), ('status', 'failed'),
                                 ('completed_updates', 3000), ('training_input_ids', ['extra']),
                                 ('initial_state_hashes', {}), ('plan_sha256', 'bad'), ('runtime', {}),
                                 ('resume_count', 2), ('teacher_model_loaded', True), ('dev_inputs_loaded', True),
                                 ('probe_inputs_loaded', True), ('audio_used', True), ('service_weights_updated', True)):
                changed = copy.deepcopy(run)
                changed[field] = value
                with self.assertRaises(ValueError, msg=field):
                    subject.validate_training_result(changed, plan, plan_ref, policy_ref)


if __name__ == '__main__':
    unittest.main()
