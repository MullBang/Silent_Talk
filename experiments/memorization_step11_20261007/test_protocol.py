"""Selection and plan rejection tests, with synthetic metadata and no training."""
from __future__ import annotations
import copy
import hashlib
import unittest
from unittest.mock import patch

import protocol as p


def synthetic_bundle():
    entries, rows, ids = {}, {}, []
    for role, generation in p.ROLE_GENERATIONS.items():
        for speaker_number in range(6):
            speaker = generation + str(speaker_number)
            for number, frames in enumerate((8, 4, 6, 6)):
                sid = f'{speaker}_{number}'
                ids.append(sid)
                rows[sid] = {'id': sid, 'speaker_id': speaker, 'split': 'train',
                             'frames': frames, 'token_ids': [1, 2, 1], 'source_stem': speaker,
                             'text': 'synthetic'}
                entries[sid] = {'id': sid, 'role': role, 'generation': generation,
                                'original_split': 'train', 'provisional_training_candidate': True,
                                'speaker_id': speaker, 'student_frames': frames,
                                'logical_record_sha256': 'synthetic',
                                'input': {'path': sid + '.npz', 'sha256': 'input'},
                                'teacher_feature': {'path': sid + '_feature.npz', 'sha256': 'feature'}}
    return {'entries': entries, 'rows': rows,
            'groups': {'expanded510': ids, 'held27': ['held'], 'old71': ['old_dev'], 'new24': ['new_dev']}}


def contract_fixture():
    selection = p.select_from_bundle(synthetic_bundle())
    states = {arm: {'model': 'm', 'adapter': 'a', 'rng': 'r',
                    'optimizer': arm, 'complete': arm} for arm in p.ARMS}
    policy = {'selection': selection, 'runtime': {'fixed': True}, 'execution_code': [],
              'data_bindings': [], 'optimizer': {'fixed': True}, 'loss': {'fixed': True},
              'initial_state_hashes_expected': {'42': states}}
    plan = {'schema': 'lipnet-step11-memorization-training-plan-v1', 'seed': 42,
            'arms': list(p.ARMS), 'updates': 1200, 'batch_size': 2, 'checkpoint_interval': 300,
            'ids': selection['ids'], 'diagnostic_groups': selection['diagnostic_groups'],
            'schedule': p.core.fixed_schedule(selection['ids'], 42, 1200, 2),
            'runtime': policy['runtime'], 'execution_code': [], 'data_bindings': [],
            'optimizer': policy['optimizer'], 'loss': policy['loss'],
            'initial_state_hashes_expected': states, 'experiment_plan': {'path': 'policy', 'sha256': 'fixed'}}
    plan['plan_sha256'] = p.runner.canonical_plan_hash(plan)
    return plan, policy


class SelectionTests(unittest.TestCase):
    def test_exact_eight_distinct_speakers_and_groups(self):
        selected = p.select_from_bundle(synthetic_bundle())
        self.assertEqual(len(selected['ids']), 8)
        self.assertEqual({k: len(v) for k, v in selected['diagnostic_groups'].items()}, {'old4': 4, 'new4': 4})
        self.assertEqual(len({row['speaker_id'] for row in selected['records']}), 8)

    def test_hash_rank_uses_literal_role_and_salt(self):
        result = p.select_from_bundle(synthetic_bundle())
        for role, generation in p.ROLE_GENERATIONS.items():
            expected = sorted((hashlib.sha256(f'step11-memorization-v1:{role}:{generation}{n}'.encode()).hexdigest(),
                               generation + str(n)) for n in range(6))
            self.assertEqual([row['speaker_id'] for row in result['records'] if row['role'] == role],
                             [speaker for _, speaker in expected[:4]])

    def test_lower_median_with_id_tie_break(self):
        selected = p.select_from_bundle(synthetic_bundle())
        self.assertTrue(all(row['id'].endswith('_2') for row in selected['records']))
        self.assertTrue(all(row['lower_median_index'] == 1 for row in selected['records']))

    def test_source_order_does_not_change_selection(self):
        original = synthetic_bundle()
        changed = copy.deepcopy(original)
        changed['groups']['expanded510'].reverse()
        self.assertEqual(p.select_from_bundle(original)['ids'], p.select_from_bundle(changed)['ids'])

    def test_transcript_content_does_not_rank(self):
        original = synthetic_bundle()
        changed = copy.deepcopy(original)
        for index, row in enumerate(changed['rows'].values()):
            row['text'] = 'much longer target ' * (index + 1)
        self.assertEqual(p.select_from_bundle(original)['ids'], p.select_from_bundle(changed)['ids'])

    def test_reject_duplicate_eligible(self):
        bundle = synthetic_bundle()
        bundle['groups']['expanded510'].append(bundle['groups']['expanded510'][0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.select_from_bundle(bundle)

    def test_reject_excluded_contamination(self):
        bundle = synthetic_bundle()
        bundle['groups']['held27'].append(bundle['groups']['expanded510'][0])
        with self.assertRaisesRegex(ValueError, 'contamination'):
            p.select_from_bundle(bundle)

    def test_reject_dev_role(self):
        bundle = synthetic_bundle()
        bundle['entries'][bundle['groups']['expanded510'][0]]['role'] = 'old_development'
        with self.assertRaisesRegex(ValueError, 'Nontraining'):
            p.select_from_bundle(bundle)

    def test_reject_source_split(self):
        bundle = synthetic_bundle()
        bundle['rows'][bundle['groups']['expanded510'][0]]['split'] = 'val'
        with self.assertRaisesRegex(ValueError, 'Nontraining'):
            p.select_from_bundle(bundle)

    def test_reject_false_candidate(self):
        bundle = synthetic_bundle()
        bundle['entries'][bundle['groups']['expanded510'][0]]['provisional_training_candidate'] = False
        with self.assertRaisesRegex(ValueError, 'Nontraining'):
            p.select_from_bundle(bundle)

    def test_reject_frame_metadata_mismatch(self):
        bundle = synthetic_bundle()
        bundle['entries'][bundle['groups']['expanded510'][0]]['student_frames'] += 1
        with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
            p.select_from_bundle(bundle)

    def test_repeated_ctc_tokens_count_additional_frames(self):
        bundle = synthetic_bundle()
        selected = p.select_from_bundle(bundle)['ids'][0]
        bundle['rows'][selected]['token_ids'] = [1, 1, 1]
        result = p.select_from_bundle(bundle)
        record = next(row for row in result['records'] if row['id'] == selected)
        self.assertEqual(record['ctc_min_input_frames'], 5)
        self.assertEqual(record['ctc_slack'], 1)

    def test_infeasible_selected_clip_is_not_replaced(self):
        bundle = synthetic_bundle()
        selected = p.select_from_bundle(bundle)['ids'][0]
        bundle['rows'][selected]['token_ids'] = [1, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, 'CTC infeasible'):
            p.select_from_bundle(bundle)

    def test_blank_target_is_rejected(self):
        bundle = synthetic_bundle()
        selected = p.select_from_bundle(bundle)['ids'][0]
        bundle['rows'][selected]['token_ids'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'Invalid selected CTC'):
            p.select_from_bundle(bundle)

    def test_cross_generation_speaker_collision_is_rejected(self):
        bundle = synthetic_bundle()
        for source in (bundle['rows'], bundle['entries']):
            for row in source.values():
                row['speaker_id'] = row['speaker_id'].replace('new', 'old')
        with self.assertRaisesRegex(ValueError, 'distinct'):
            p.select_from_bundle(bundle)


class PlanTests(unittest.TestCase):
    def check(self, plan, policy):
        with patch.object(p, 'bind', return_value={'path': 'policy', 'sha256': 'fixed'}):
            p.check_plan(plan, policy)

    def test_unchanged_contract_and_equal_exposure_pass(self):
        plan, policy = contract_fixture()
        self.check(plan, policy)

    def test_budget_cannot_be_extended(self):
        plan, policy = contract_fixture()
        plan['updates'] += 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.check(plan, policy)

    def test_schedule_mutation_is_rejected(self):
        plan, policy = contract_fixture()
        plan['schedule'][0][0] = plan['schedule'][0][1]
        with self.assertRaisesRegex(ValueError, 'schedule'):
            self.check(plan, policy)

    def test_unpaired_rng_is_rejected(self):
        plan, policy = contract_fixture()
        plan['initial_state_hashes_expected']['B_feature']['rng'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            self.check(plan, policy)

    def test_optimizer_equality_is_rejected(self):
        plan, policy = contract_fixture()
        plan['initial_state_hashes_expected']['B_feature']['optimizer'] = 'A_ctc'
        with self.assertRaisesRegex(ValueError, 'optimizer parameter scope'):
            self.check(plan, policy)

    def test_arbitrary_plan_mutation_is_bound_by_hash(self):
        plan, policy = contract_fixture()
        plan['unplanned_field'] = 'mutation'
        with self.assertRaisesRegex(ValueError, 'hash binding'):
            self.check(plan, policy)


if __name__ == '__main__':
    unittest.main()
