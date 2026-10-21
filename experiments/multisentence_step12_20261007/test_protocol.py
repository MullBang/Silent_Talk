"""Metadata-selection and train/holdout boundary tests without optimization."""
from __future__ import annotations
import copy
import hashlib
import unittest
from unittest.mock import patch

import numpy as np
import protocol as p


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    rows, entries, ids = {}, {}, []
    inputs = {'old': {}, 'new': {}}
    anchor_ids, anchors = [], []
    for generation in ('old', 'new'):
        for number in range(4):
            speaker = generation + str(number)
            for clip_number in range(8):
                stem = speaker + ('_train' if clip_number < 5 else '_hold' + str(clip_number))
                sid = speaker + '_' + str(clip_number)
                row = {'id': sid, 'speaker_id': speaker, 'split': 'train', 'source_stem': stem,
                       'frames': 12, 'token_ids': [1, 2, 3], 'text': sid}
                rows[sid] = row
                entries[sid] = {'id': sid, 'speaker_id': speaker, 'original_split': 'train',
                                'generation': generation, 'role': 'old_train' if generation == 'old' else 'new_provisional_train',
                                'provisional_training_candidate': True, 'student_frames': 12,
                                'logical_record_sha256': digest(sid), 'input': {'path': sid, 'sha256': digest(sid)},
                                'teacher_feature': {'path': sid + '_feature', 'sha256': digest(sid + '_feature')}}
                inputs[generation][sid] = {'source_verification': {'members':
                    {kind: {'sha256': digest(stem + kind)} for kind in ('video', 'label')}}}
                ids.append(sid)
                if clip_number == 0:
                    anchor_ids.append(sid)
                    anchors.append({'id': sid, 'speaker_id': speaker, 'source_stem': stem,
                                    'logical_record_sha256': digest(sid)})
    bundle = {'rows': rows, 'entries': entries, 'inputs': inputs,
              'groups': {'expanded510': ids, 'held27': ['held_original'], 'old71': ['dev_old'], 'new24': ['dev_new']}}
    return bundle, {'ids': anchor_ids, 'records': anchors}


def plan_fixture():
    selection = p.select_from_bundle(*fixture())
    initial = {arm: {'model': 'model', 'adapter': 'adapter', 'rng': 'rng', 'optimizer': arm, 'complete': arm}
               for arm in p.ARMS}
    policy = {'selection': selection, 'runtime': {}, 'execution_code': [], 'data_bindings': [], 'optimizer': {},
              'loss': {}, 'evaluation_data': {}, 'initial_state_hashes_expected': {'42': initial}}
    plan = {'schema': 'lipnet-step12-multisentence-training-plan-v1', 'seed': 42, 'arms': list(p.ARMS),
            'updates': 3600, 'batch_size': 2, 'checkpoint_interval': 900,
            'ids': selection['train_ids'], 'train_ids': selection['train_ids'], 'holdout_ids': selection['holdout_ids'],
            'diagnostic_groups': selection['evaluation_groups'], 'evaluation_groups': selection['evaluation_groups'],
            'schedule': p.core.fixed_schedule(selection['train_ids'], 42, 3600, 2),
            'initial_state_hashes_expected': initial, 'experiment_plan': {'path': 'policy', 'sha256': 'fixed'},
            **{key: policy[key] for key in ('runtime', 'execution_code', 'data_bindings', 'optimizer', 'loss', 'evaluation_data')}}
    plan['plan_sha256'] = p.runner.canonical_plan_hash(plan)
    return plan, policy


class SelectionTests(unittest.TestCase):
    def test_counts_anchors_and_speaker_scopes(self):
        bundle, anchors = fixture()
        result = p.select_from_bundle(bundle, anchors)
        self.assertEqual(len(result['train_ids']), 24)
        self.assertEqual(len(result['holdout_ids']), 8)
        self.assertTrue(set(anchors['ids']) <= set(result['train_ids']))
        self.assertEqual(len({row['normalized_target'] for row in result['records']}), 32)
        for item in result['speaker_selections']:
            self.assertEqual(len(item['train_ids']), 3)
            self.assertNotEqual(item['training_source_stem'], item['holdout_source_stem'])

    def test_exact_hash_salts_and_rank(self):
        bundle, anchors = fixture()
        result = p.select_from_bundle(bundle, anchors)
        for item in result['speaker_selections']:
            for scope, candidates in [('train', item['train_candidate_ids_ranked']), ('holdout', item['holdout_candidate_ids_ranked'])]:
                self.assertEqual(candidates, sorted(candidates, key=lambda sid: (digest('step12-multisentence-v1:' + scope + ':' + sid), sid)))

    def test_metadata_iteration_order_does_not_choose_different_ids(self):
        bundle, anchors = fixture()
        changed = copy.deepcopy(bundle)
        changed['groups']['expanded510'].reverse()
        original = p.select_from_bundle(bundle, anchors)
        actual = p.select_from_bundle(changed, anchors)
        self.assertEqual(original['train_ids'], actual['train_ids'])
        self.assertEqual(original['holdout_ids'], actual['holdout_ids'])

    def test_normalization_is_nfc_then_all_whitespace_removed(self):
        self.assertEqual(p.normalized_target(' \u1100\u1161\t\n나\u00a0 '), '가나')

    def test_duplicate_future_anchor_is_reserved_before_additions(self):
        bundle, anchors = fixture()
        base = p.select_from_bundle(bundle, anchors)
        candidate = base['speaker_selections'][0]['train_candidate_ids_ranked'][0]
        bundle['rows'][candidate]['text'] = bundle['rows'][anchors['ids'][-1]]['text']
        actual = p.select_from_bundle(bundle, anchors)
        self.assertNotIn(candidate, actual['train_ids'])
        event = next(event for event in actual['selection_events'] if event['id'] == candidate)
        self.assertEqual(event['status'], 'skipped')
        self.assertIn('duplicate_normalized_target', event['reasons'])

    def test_duplicate_anchors_abort(self):
        bundle, anchors = fixture()
        bundle['rows'][anchors['ids'][1]]['text'] = bundle['rows'][anchors['ids'][0]]['text']
        with self.assertRaisesRegex(ValueError, 'Duplicate retained'):
            p.select_from_bundle(bundle, anchors)

    def test_infeasible_addition_is_logged_and_skipped(self):
        bundle, anchors = fixture()
        candidate = p.select_from_bundle(bundle, anchors)['speaker_selections'][0]['train_candidate_ids_ranked'][0]
        bundle['rows'][candidate]['token_ids'] = [1] * 8
        actual = p.select_from_bundle(bundle, anchors)
        self.assertNotIn(candidate, actual['train_ids'])
        self.assertIn('ctc_infeasible', next(event for event in actual['selection_events'] if event['id'] == candidate)['reasons'])

    def test_infeasible_anchor_aborts_without_replacement(self):
        bundle, anchors = fixture()
        bundle['rows'][anchors['ids'][0]]['token_ids'] = [1] * 8
        with self.assertRaisesRegex(ValueError, 'Invalid retained'):
            p.select_from_bundle(bundle, anchors)

    def test_fewer_than_two_same_recording_additions_aborts(self):
        bundle, anchors = fixture()
        speaker = bundle['rows'][anchors['ids'][0]]['speaker_id']
        for sid, row in bundle['rows'].items():
            if row['speaker_id'] == speaker and sid != anchors['ids'][0] and row['source_stem'].endswith('_train'):
                row['token_ids'] = [0]
        with self.assertRaisesRegex(ValueError, 'two same-recording'):
            p.select_from_bundle(bundle, anchors)

    def test_holdout_duplicate_train_text_is_skipped(self):
        bundle, anchors = fixture()
        candidate = p.select_from_bundle(bundle, anchors)['holdout_ids'][0]
        bundle['rows'][candidate]['text'] = bundle['rows'][anchors['ids'][3]]['text']
        actual = p.select_from_bundle(bundle, anchors)
        self.assertNotIn(candidate, actual['holdout_ids'])

    def test_holdout_video_sha_collision_is_skipped(self):
        bundle, anchors = fixture()
        candidate = p.select_from_bundle(bundle, anchors)['holdout_ids'][0]
        bundle['inputs']['old'][candidate]['source_verification']['members']['video']['sha256'] = p.source_hashes(bundle, anchors['ids'][3])['video_sha256']
        actual = p.select_from_bundle(bundle, anchors)
        self.assertNotIn(candidate, actual['holdout_ids'])
        self.assertIn('training_video_sha_overlap', next(event for event in actual['selection_events'] if event['id'] == candidate)['reasons'])

    def test_holdout_label_sha_collision_is_skipped(self):
        bundle, anchors = fixture()
        candidate = p.select_from_bundle(bundle, anchors)['holdout_ids'][0]
        bundle['inputs']['old'][candidate]['source_verification']['members']['label']['sha256'] = p.source_hashes(bundle, anchors['ids'][3])['label_sha256']
        actual = p.select_from_bundle(bundle, anchors)
        self.assertNotIn(candidate, actual['holdout_ids'])

    def test_all_holdouts_invalid_aborts_without_speaker_replacement(self):
        bundle, anchors = fixture()
        item = p.select_from_bundle(bundle, anchors)['speaker_selections'][0]
        for candidate in item['holdout_candidate_ids_ranked']:
            bundle['rows'][candidate]['text'] = bundle['rows'][anchors['ids'][0]]['text']
        with self.assertRaisesRegex(ValueError, 'source-disjoint holdout'):
            p.select_from_bundle(bundle, anchors)

    def test_original_excluded_role_cannot_enter(self):
        bundle, anchors = fixture()
        bundle['groups']['held27'].append(anchors['ids'][0])
        with self.assertRaisesRegex(ValueError, 'contamination'):
            p.select_from_bundle(bundle, anchors)

    def test_original_dev_split_cannot_enter(self):
        bundle, anchors = fixture()
        bundle['rows'][anchors['ids'][0]]['split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'Ineligible'):
            p.select_from_bundle(bundle, anchors)

    def test_holdout_records_have_no_teacher_reference(self):
        result = p.select_from_bundle(*fixture())
        self.assertTrue(all(row['teacher_feature'] is None for row in result['records'] if row['diagnostic_role'] == 'internal_holdout'))


class EvaluationBoundaryTests(unittest.TestCase):
    def clip(self):
        return {'id': 'hold', 'split': 'internal_holdout', 'source_split': 'train',
                'diagnostic_role': 'internal_holdout', 'speaker_id': 'speaker', 'source_stem': 'stem',
                'text': 'text', 'pixels': np.zeros((3, 96, 96, 3), np.uint8), 'tokens': [1]}

    def test_holdout_pixels_bound_without_teacher(self):
        clip = self.clip()
        value = p.evaluation_bindings({'hold': clip})
        self.assertEqual(value['clips']['hold']['pixels_sha256'], p.runner.nested_hash(clip['pixels']))
        self.assertEqual(value['clips']['hold']['diagnostic_role'], 'internal_holdout')

    def test_teacher_leak_rejected(self):
        clip = self.clip()
        clip['teacher_features'] = None
        with self.assertRaisesRegex(ValueError, 'leaked teacher'):
            p.evaluation_binding(clip)

    def test_relabeling_holdout_as_train_is_rejected(self):
        clip = self.clip()
        clip['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'role'):
            p.evaluation_binding(clip)


class PlanTests(unittest.TestCase):
    def check(self, plan, policy):
        with patch.object(p, 'bind', return_value={'path': 'policy', 'sha256': 'fixed'}):
            p.check_plan(plan, policy)

    def test_expected_contract_passes(self):
        self.check(*plan_fixture())

    def test_holdout_in_schedule_rejected(self):
        plan, policy = plan_fixture()
        plan['schedule'][0][0] = plan['holdout_ids'][0]
        with self.assertRaisesRegex(ValueError, 'schedule'):
            self.check(plan, policy)

    def test_adaptive_budget_extension_rejected(self):
        plan, policy = plan_fixture()
        plan['updates'] = 7200
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.check(plan, policy)

    def test_unpaired_rng_rejected(self):
        plan, policy = plan_fixture()
        plan['initial_state_hashes_expected']['B_feature']['rng'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            self.check(plan, policy)

    def test_unbound_extra_change_rejected(self):
        plan, policy = plan_fixture()
        plan['unplanned'] = True
        with self.assertRaisesRegex(ValueError, 'hash binding'):
            self.check(plan, policy)


if __name__ == '__main__':
    unittest.main()
