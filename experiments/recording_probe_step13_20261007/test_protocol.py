"""Scope, leakage, timing and lazy student-array loading tests; no inference."""
from __future__ import annotations
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import protocol as p


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    rows, entries, ids, records = {}, {}, [], []
    inputs = {'old': {}, 'new': {}}
    train_ids, hold_ids, speakers = [], [], []
    for index, speaker in enumerate(p.SPEAKERS):
        generation = 'old' if index < 4 else 'new'
        training = []
        for number in range(6 if speaker in p.EXPECTED_AVAILABLE else 4):
            sid = speaker + '_' + str(number)
            stem = speaker + ('_different' if number == 3 else '_same')
            row = {'id': sid, 'speaker_id': speaker, 'split': 'train', 'source_stem': stem,
                   'frames': 12, 'token_ids': [1, 2, 3], 'text': sid, 'start_time': number * 10.,
                   'end_time': number * 10. + 0.48, 'video_fps': 25.}
            rows[sid] = row
            entries[sid] = {'id': sid, 'speaker_id': speaker, 'original_split': 'train', 'generation': generation,
                            'role': 'old_train' if generation == 'old' else 'new_provisional_train',
                            'provisional_training_candidate': True, 'student_frames': 12,
                            'logical_record_sha256': digest(sid), 'input': {'path': sid, 'sha256': digest(sid)}}
            inputs[generation][sid] = {'source_verification': {'members':
                {kind: {'sha256': digest(stem + kind)} for kind in ('video', 'label')}}}
            ids.append(sid)
            if number < 3:
                training.append(sid)
            elif number == 3:
                hold_ids.append(sid)
            if number < 4:
                records.append({'id': sid, 'speaker_id': speaker, 'source_stem': stem, 'text': sid,
                                'frames': 12, 'tokens': [1, 2, 3], 'logical_record_sha256': digest(sid)})
        train_ids.extend(training)
        speakers.append({'speaker_id': speaker, 'retained_anchor_id': training[0], 'train_ids': training,
                         'training_source_stem': speaker + '_same', 'holdout_id': speaker + '_3'})
    bundle = {'rows': rows, 'entries': entries, 'inputs': inputs,
              'handoff_ref': {'path': 'metadata.json', 'sha256': digest('metadata')},
              'groups': {'expanded510': ids, 'held27': ['held_original'], 'old71': ['dev_old'], 'new24': ['dev_new']}}
    return bundle, {'train_ids': train_ids, 'holdout_ids': hold_ids, 'records': records, 'speaker_selections': speakers}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle, self.previous = fixture()

    def selected(self):
        return p.select_from_bundle(self.bundle, self.previous)

    def first_candidate(self, speaker='C002'):
        return next(item for item in self.selected()['speaker_selections'] if item['speaker_id'] == speaker)['candidate_ids_ranked'][0]

    def event(self, result, sid):
        return next(item for item in result['candidate_events'] if item['id'] == sid)

    def test_exact_scope_disjoint32_and_matched_speakers(self):
        value = self.selected()
        self.assertEqual([row['speaker_id'] for row in value['pairs']], list(p.EXPECTED_AVAILABLE))
        self.assertEqual(len(value['same_ids']), 6)
        self.assertEqual(len(value['different_ids']), 6)
        self.assertFalse(set(value['same_ids']) & set(value['excluded_step12_ids']))
        self.assertTrue(set(value['different_ids']) <= set(self.previous['holdout_ids']))

    def test_exact_rank_salt_and_iteration_order_invariance(self):
        value = self.selected()
        for item in value['speaker_selections']:
            pool = item['candidate_ids_ranked']
            self.assertEqual(pool, sorted(pool, key=lambda sid: (digest('step13-same-recording-v1:' + sid), sid)))
        self.bundle['groups']['expanded510'].reverse()
        self.assertEqual(self.selected()['same_ids'], value['same_ids'])

    def test_unavailable_speakers_remain_unavailable(self):
        items = [item for item in self.selected()['speaker_selections'] if item['selected_id'] is None]
        self.assertEqual([item['speaker_id'] for item in items], ['C058', 'C658'])
        self.assertTrue(all(item['candidate_count'] == 0 for item in items))

    def test_all_prior_targets_reserved_and_unicode_normalized(self):
        sid = self.first_candidate()
        previous_id = self.previous['holdout_ids'][-1]
        self.bundle['rows'][previous_id]['text'] = '가나'
        next(row for row in self.previous['records'] if row['id'] == previous_id)['text'] = '가나'
        self.bundle['rows'][sid]['text'] = ' \u1100\u1161\t나\u00a0'
        result = self.selected()
        self.assertNotIn(sid, result['same_ids'])
        self.assertIn('duplicate_normalized_target', self.event(result, sid)['reasons'])

    def test_duplicate_between_new_selections_is_skipped(self):
        first = self.first_candidate()
        second = self.first_candidate('C032')
        self.bundle['rows'][second]['text'] = self.bundle['rows'][first]['text']
        result = self.selected()
        self.assertIn(first, result['same_ids'])
        self.assertNotIn(second, result['same_ids'])

    def test_empty_candidate_target_skipped(self):
        sid = self.first_candidate()
        self.bundle['rows'][sid]['text'] = '\t \n'
        self.assertIn('empty_normalized_target', self.event(self.selected(), sid)['reasons'])

    def test_ctc_repeated_token_minimum_respected(self):
        sid = self.first_candidate()
        self.bundle['rows'][sid]['token_ids'] = [1] * 7
        self.assertIn('ctc_infeasible', self.event(self.selected(), sid)['reasons'])

    def test_ctc_blank_token_forbidden(self):
        sid = self.first_candidate()
        self.bundle['rows'][sid]['token_ids'] = [0]
        self.assertIn('invalid_ctc_target', self.event(self.selected(), sid)['reasons'])

    def test_video_identity_mismatch_skipped(self):
        sid = self.first_candidate()
        self.bundle['inputs']['old'][sid]['source_verification']['members']['video']['sha256'] = digest('changed')
        self.assertIn('same_stem_source_hash_mismatch', self.event(self.selected(), sid)['reasons'])

    def test_label_identity_mismatch_skipped(self):
        sid = self.first_candidate()
        self.bundle['inputs']['old'][sid]['source_verification']['members']['label']['sha256'] = digest('changed')
        self.assertIn('same_stem_source_hash_mismatch', self.event(self.selected(), sid)['reasons'])

    def test_time_and_native_frame_overlap_skipped(self):
        sid = self.first_candidate()
        self.bundle['rows'][sid].update(start_time=0., end_time=0.48)
        reasons = self.event(self.selected(), sid)['reasons']
        self.assertIn('training_time_interval_overlap:C002_0', reasons)
        self.assertIn('training_native_frame_overlap:C002_0', reasons)

    def test_adjacent_intervals_allowed_without_frame_collision(self):
        sid = self.first_candidate()
        self.bundle['rows'][sid].update(start_time=0.48, end_time=0.96)
        self.assertEqual(self.event(self.selected(), sid)['status'], 'selected')

    def test_rounded_native_frame_collision_despite_nonoverlapping_time(self):
        sid = self.first_candidate()
        self.bundle['rows']['C002_0']['end_time'] = 0.445
        self.bundle['rows'][sid].update(start_time=0.445, end_time=0.925)
        reasons = self.event(self.selected(), sid)['reasons']
        self.assertNotIn('training_time_interval_overlap:C002_0', reasons)
        self.assertIn('training_native_frame_overlap:C002_0', reasons)

    def test_nonfinite_timing_rejected(self):
        self.bundle['rows'][self.first_candidate()]['start_time'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid timing'):
            self.selected()

    def test_prior_training_metadata_change_rejected(self):
        self.bundle['rows']['C002_0']['text'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Previous selected metadata'):
            self.selected()

    def test_historical_development_overlap_rejected(self):
        self.bundle['groups']['old71'].append(self.first_candidate())
        with self.assertRaisesRegex(ValueError, 'contamination'):
            self.selected()

    def test_historical_wrong_split_rejected(self):
        self.bundle['rows'][self.first_candidate()]['split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'Ineligible'):
            self.selected()

    def test_missing_required_speaker_aborts_without_substitution(self):
        ids = next(item for item in self.selected()['speaker_selections'] if item['speaker_id'] == 'C002')['candidate_ids_ranked']
        for sid in ids:
            self.bundle['rows'][sid]['token_ids'] = [0]
        with self.assertRaisesRegex(ValueError, 'Available speaker scope'):
            self.selected()

    def test_different_recording_same_source_hash_rejected(self):
        self.bundle['inputs']['old']['C002_3']['source_verification']['members']['video']['sha256'] = digest('C002_samevideo')
        with self.assertRaisesRegex(ValueError, 'Different-recording identity'):
            self.selected()

    def test_incorrect_old_new_role_rejected(self):
        self.bundle['entries'][self.first_candidate()]['generation'] = 'new'
        with self.assertRaisesRegex(ValueError, 'Ineligible'):
            self.selected()


class LoadingTests(unittest.TestCase):
    def test_only_selected_six_student_arrays_decoded(self):
        bundle, previous = fixture()
        selection = p.select_from_bundle(bundle, previous)
        opened, decoded = [], []
        class Checked:
            def verify(self, ref):
                opened.append(ref['path'])
                return ref['path']
            def finish(self):
                return [bundle['entries'][sid]['input'] for sid in opened]
        class Saved:
            files = p.mixed.PAIRED_KEYS
            def __init__(self, sid): self.sid = sid
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def __getitem__(self, key):
                decoded.append((self.sid, key))
                row = bundle['rows'][self.sid]
                if key == 'student_rgb': return np.zeros((12, 96, 96, 3), np.uint8)
                if key == 'student_source_indices': return np.asarray(p.source_indices(row), np.int64)
                if key == 'label_tokens': return np.asarray(row['token_ids'], np.int64)
                if key == 'metadata_json':
                    item = bundle['inputs'][bundle['entries'][self.sid]['generation']][self.sid]
                    return np.asarray(json.dumps({'source_verification': item['source_verification']}))
                raise AssertionError('Teacher/nonstudent array requested: ' + key)
        loader = p.mixed.old_loader()
        with patch.object(p.mixed, 'verify_handoff', return_value=bundle), patch.dict(p._prior_policy, {'selection': previous}), \
             patch.object(loader, 'LocalBindings', Checked), patch.object(loader, 'validate_student'), \
             patch.object(p.mixed, 'validate_new_student'), patch.object(p.np, 'load', side_effect=lambda path, **kwargs: Saved(path)):
            loaded = p.load_inputs()
        self.assertEqual(opened, selection['same_ids'])
        self.assertEqual(len(decoded), 24)
        self.assertEqual(loaded['counts']['teacher_feature_npz_opened'], 0)
        self.assertEqual(loaded['counts']['teacher_pixel_arrays_decoded'], 0)
        self.assertEqual(set(loaded['clips']), set(selection['same_ids']))
        self.assertTrue(all('teacher_features' not in clip and clip['diagnostic_role'] == 'same_recording_holdout' for clip in loaded['clips'].values()))

    def test_changed_selection_rejected_before_payload_open(self):
        bundle, previous = fixture()
        with patch.object(p.mixed, 'verify_handoff', return_value=bundle), patch.dict(p._prior_policy, {'selection': previous}), \
             patch.object(p.np, 'load') as opened:
            with self.assertRaisesRegex(ValueError, 'Selection changed'):
                p.load_inputs({'same_ids': []})
        opened.assert_not_called()

    def test_teacher_array_in_evaluation_clip_rejected(self):
        clip = {'split': 'same_recording_holdout', 'diagnostic_role': 'same_recording_holdout', 'source_split': 'train', 'teacher_features': None}
        with self.assertRaisesRegex(ValueError, 'Invalid inference clip'):
            p.evaluation_binding(clip)

    def test_wrong_evaluation_role_rejected(self):
        clip = {'split': 'train', 'diagnostic_role': 'train', 'source_split': 'train'}
        with self.assertRaisesRegex(ValueError, 'Invalid inference clip'):
            p.evaluation_binding(clip)


if __name__ == '__main__':
    unittest.main()
