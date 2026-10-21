"""Deterministic balanced selection and data-boundary tests; no cache decoding."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import data_plan as p


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture():
    rows, entries, inputs, features, pool, probes = {}, {}, {'old': {}}, {'old': {}}, [], []
    for speaker in p.SPEAKERS:
        for source in range(5):
            stem = speaker + '_recording' + str(source)
            for clip in range(1 if source == 4 else 8):
                sid = stem + '_' + str(clip)
                row = {'id': sid, 'source_stem': stem, 'speaker_id': speaker, 'split': 'train', 'text': sid,
                       'token_ids': [1, 2, 3], 'frames': 12, 'start_time': clip * 2., 'end_time': clip * 2. + .48, 'video_fps': 25.}
                rows[sid] = row
                entries[sid] = {'id': sid, 'source_stem': stem, 'speaker_id': speaker, 'original_split': 'train',
                                'generation': 'old', 'role': 'old_train', 'student_frames': 12,
                                'provisional_training_candidate': True, 'logical_record_sha256': p.canonical_sha(row),
                                'input': {'path': sid + '.npz', 'sha256': digest(sid)},
                                'teacher_feature': {'path': sid + '.features.npz', 'sha256': digest(sid + '_f')}}
                inputs['old'][sid] = {'source_verification': {'members':
                    {kind: {'sha256': digest(stem + kind)} for kind in ('video', 'label')}}}
                pool.append(sid)
                if source == 4:
                    probes.append(sid)
    excluded = ['original_dev', 'original_held']
    for sid in excluded:
        rows[sid] = {'id': sid, 'text': sid}
    return {'rows': rows, 'entries': entries, 'inputs': inputs, 'features': features, 'pool_ids': pool,
            'historical_excluded_ids': excluded, 'prior_probe_ids': probes, 'pending_listening': ['original_held']}


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.bundle = fixture()

    def selected(self):
        return p.select_from_bundle(self.bundle)['selection']

    def test_balanced_scopes_and_subset_same_recording_ratio(self):
        selected = self.selected()
        self.assertEqual((len(selected['train24_ids']), len(selected['train72_ids']), len(selected['dev16_ids'])), (24, 72, 16))
        self.assertTrue(set(selected['train24_ids']) <= set(selected['train72_ids']))
        self.assertFalse(set(selected['train72_ids']) & set(selected['dev16_ids']))
        for item in selected['speaker_selections']:
            self.assertEqual(len(item['primary_ids']), 6)
            self.assertEqual(len(item['secondary_ids']), 3)
            self.assertEqual(item['train24_ids'], item['primary_ids'][:2] + item['secondary_ids'][:1])
            self.assertEqual(len({item['dev_source_stem'], item['primary_source_stem'], item['secondary_source_stem']}), 3)

    def test_stable_sha_salts_and_metadata_iteration_independence(self):
        before = self.selected()
        self.bundle['pool_ids'].reverse()
        after = self.selected()
        self.assertEqual(before, after)
        for item in before['speaker_selections']:
            self.assertEqual(item['ranked_dev_sources'], sorted(item['ranked_dev_sources'], key=lambda stem: (digest('step15-diversity-v1:dev_source:' + stem), stem)))

    def test_prior_probe_entire_recording_is_blocked(self):
        profile = p.candidate_inventory(self.bundle)
        for item in profile['records']:
            if item['id'] in self.bundle['prior_probe_ids']:
                self.assertFalse(item['eligible'])
                self.assertTrue({'prior_probe_recording_stem', 'prior_probe_video_sha', 'prior_probe_label_sha'} <= set(item['reasons']))

    def test_video_hash_alias_recording_also_blocked(self):
        affected_stem = self.bundle['rows'][self.bundle['pool_ids'][0]]['source_stem']
        probe = self.bundle['prior_probe_ids'][0]
        for sid in self.bundle['pool_ids']:
            if self.bundle['rows'][sid]['source_stem'] == affected_stem:
                self.bundle['inputs']['old'][sid]['source_verification']['members']['video']['sha256'] = p.source_hashes(self.bundle, probe)['video_sha256']
        rows = p.candidate_inventory(self.bundle)['records']
        self.assertTrue(all('prior_probe_video_sha' in row['reasons'] for row in rows if row['source_stem'] == affected_stem))

    def test_label_hash_alias_recording_also_blocked(self):
        affected_stem = self.bundle['rows'][self.bundle['pool_ids'][0]]['source_stem']
        probe = self.bundle['prior_probe_ids'][0]
        for sid in self.bundle['pool_ids']:
            if self.bundle['rows'][sid]['source_stem'] == affected_stem:
                self.bundle['inputs']['old'][sid]['source_verification']['members']['label']['sha256'] = p.source_hashes(self.bundle, probe)['label_sha256']
        rows = p.candidate_inventory(self.bundle)['records']
        self.assertTrue(all('prior_probe_label_sha' in row['reasons'] for row in rows if row['source_stem'] == affected_stem))

    def test_original_dev_transcript_blocked_after_unicode_normalization(self):
        sid = self.bundle['pool_ids'][0]
        self.bundle['rows'][sid]['text'] = ' \u1100\u1161\t나\u00a0'
        self.bundle['rows']['original_dev']['text'] = '가나'
        row = next(row for row in p.candidate_inventory(self.bundle)['records'] if row['id'] == sid)
        self.assertIn('prior_probe_or_excluded_normalized_target', row['reasons'])

    def test_held_transcript_is_blocked(self):
        sid = self.bundle['pool_ids'][0]
        self.bundle['rows'][sid]['text'] = 'original_held'
        row = next(row for row in p.candidate_inventory(self.bundle)['records'] if row['id'] == sid)
        self.assertFalse(row['eligible'])

    def test_historical_dev_or_held_id_cannot_enter_pool(self):
        self.bundle['historical_excluded_ids'].append(self.bundle['pool_ids'][0])
        with self.assertRaisesRegex(ValueError, 'Pool/excluded contamination'):
            self.selected()

    def test_same_recording_inconsistent_identity_rejected(self):
        sid = self.bundle['pool_ids'][0]
        self.bundle['inputs']['old'][sid]['source_verification']['members']['video']['sha256'] = digest('other')
        with self.assertRaisesRegex(ValueError, 'inconsistent identity'):
            self.selected()

    def test_ctc_infeasible_and_blank_targets_excluded(self):
        first, second = self.bundle['pool_ids'][:2]
        self.bundle['rows'][first]['token_ids'] = [1] * 7
        self.bundle['rows'][second]['token_ids'] = [0]
        lookup = {row['id']: row for row in p.candidate_inventory(self.bundle)['records']}
        self.assertIn('ctc_infeasible', lookup[first]['reasons'])
        self.assertIn('invalid_ctc_target', lookup[second]['reasons'])

    def test_empty_target_excluded(self):
        sid = self.bundle['pool_ids'][0]
        self.bundle['rows'][sid]['text'] = '\t\n '
        row = next(row for row in p.candidate_inventory(self.bundle)['records'] if row['id'] == sid)
        self.assertIn('empty_normalized_target', row['reasons'])

    def test_missing_recordings_does_not_relax_selection(self):
        speaker = p.SPEAKERS[0]
        for sid in self.bundle['pool_ids']:
            row = self.bundle['rows'][sid]
            if row['speaker_id'] == speaker and row['source_stem'].endswith(('2', '3')):
                row['token_ids'] = [0]
        with self.assertRaisesRegex(ValueError, 'Cannot satisfy fixed'):
            self.selected()

    def test_dev_source_skipped_when_it_consumes_only_primary_capacity(self):
        original = self.selected()['speaker_selections'][0]
        first_dev = original['ranked_dev_sources'][0]
        speaker = original['speaker_id']
        for stem in original['ranked_dev_sources'][1:]:
            ids = [sid for sid in self.bundle['pool_ids'] if self.bundle['rows'][sid]['source_stem'] == stem]
            for sid in ids[3:]:
                self.bundle['rows'][sid]['token_ids'] = [0]
        result = self.selected()
        actual = result['speaker_selections'][0]
        self.assertNotEqual(actual['dev_source_stem'], first_dev)
        self.assertEqual(actual['primary_source_stem'], first_dev)
        rejected = [row for row in result['events'] if row['kind'] == 'dev_source' and row['speaker_id'] == speaker and row['source_stem'] == first_dev]
        self.assertEqual(rejected[0]['reasons'], ['no_disjoint_primary6_secondary3_combination'])

    def test_cross_speaker_duplicate_target_skipped_and_logged(self):
        original = self.selected()
        first = original['speaker_selections'][0]['primary_ids'][0]
        second = original['speaker_selections'][1]['primary_ids'][0]
        self.bundle['rows'][second]['text'] = self.bundle['rows'][first]['text']
        result = self.selected()
        self.assertNotIn(second, result['train72_ids'])
        self.assertTrue(any(row.get('id') == second and 'duplicate_selected_normalized_target' in row['reasons'] for row in result['events']))

    def test_train_clip_interval_and_native_frame_overlap_skipped(self):
        selected = self.selected()
        first, second = selected['speaker_selections'][0]['primary_ids'][:2]
        self.bundle['rows'][second].update(start_time=self.bundle['rows'][first]['start_time'], end_time=self.bundle['rows'][first]['end_time'])
        result = self.selected()
        self.assertNotIn(second, result['train72_ids'])
        reasons = [reason for row in result['events'] if row.get('id') == second for reason in row['reasons']]
        self.assertTrue(any(reason.startswith('selected_time_overlap:') for reason in reasons))
        self.assertTrue(any(reason.startswith('selected_native_frame_overlap:') for reason in reasons))

    def test_development_clip_overlap_is_also_blocked(self):
        chosen = self.selected()['speaker_selections'][0]
        first, second = chosen['dev_ids']
        self.bundle['rows'][second].update(start_time=self.bundle['rows'][first]['start_time'], end_time=self.bundle['rows'][first]['end_time'])
        self.assertNotIn(second, self.selected()['dev16_ids'])

    def test_adjacent_time_windows_can_still_collide_after_rounding(self):
        row1 = {'id': 'a', 'source_stem': 'same', 'frames': 12, 'start_time': 0., 'end_time': .445, 'video_fps': 25.}
        row2 = {'id': 'b', 'source_stem': 'same', 'frames': 12, 'start_time': .445, 'end_time': .925, 'video_fps': 25.}
        reasons = p.overlap_reasons(row2, [row1])
        self.assertNotIn('selected_time_overlap:a', reasons)
        self.assertIn('selected_native_frame_overlap:a', reasons)

    def test_nonfinite_time_rejected(self):
        self.bundle['rows'][self.bundle['pool_ids'][0]]['start_time'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid timing'):
            self.selected()

    def test_selected_source_rows_and_approval_status_preserved(self):
        for record in self.selected()['records']:
            self.assertEqual(record['logical_record'], self.bundle['rows'][record['id']])
            self.assertEqual(record['source_entry'], self.bundle['entries'][record['id']])
            self.assertFalse(record['new_human_approval_added'])
            self.assertEqual(record['original_split'], 'train')
            if record['role'] == 'development':
                self.assertIsNone(record['teacher_feature'])


class CacheBoundaryTests(unittest.TestCase):
    def test_hashes_only_88_inputs_and_72_training_features(self):
        selected = p.select_from_bundle(fixture())['selection']
        with tempfile.TemporaryDirectory(dir=p.ROOT) as temp:
            root = Path(temp)
            expected = []
            for record in selected['records']:
                for key in ['input'] + (['teacher_feature'] if record['role'] == 'train' else []):
                    path = root / (record['id'] + '_' + key + '.npz')
                    path.write_bytes(b'not a decodable archive; hashing only')
                    record[key] = p.bind(path)
                    expected.append(record[key])
            calls = []
            original_verify = p.verify
            def track(ref):
                calls.append(ref)
                return original_verify(ref)
            with patch.object(p, 'verify', side_effect=track):
                value = p.hash_selected_caches(selected)
            self.assertEqual(len(calls), 160)
            self.assertEqual({ref['path'] for ref in calls}, {ref['path'] for ref in expected})
            self.assertEqual(value['npz_arrays_decoded'], 0)
            self.assertEqual(value['development_teacher_feature_files_opened'], 0)

    def test_dev_teacher_reference_rejected_before_files_read(self):
        selected = p.select_from_bundle(fixture())['selection']
        dev = next(row for row in selected['records'] if row['role'] == 'development')
        dev['teacher_feature'] = dev['source_entry']['teacher_feature']
        with patch.object(p, 'verify') as verify:
            with self.assertRaisesRegex(ValueError, 'Cache scope'):
                p.hash_selected_caches(selected)
        verify.assert_not_called()

    def test_duplicate_input_binding_rejected(self):
        selected = p.select_from_bundle(fixture())['selection']
        selected['records'][1]['input'] = selected['records'][0]['input']
        with self.assertRaisesRegex(ValueError, 'Cache scope'):
            p.hash_selected_caches(selected)

    def test_existing_plan_not_overwritten_or_sources_loaded(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT) as temp:
            path = Path(temp) / 'data_plan.json'
            path.write_text('already frozen', encoding='utf-8')
            with patch.object(p, 'PLAN', path), patch.object(p, 'source_metadata') as source:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    p.build()
            source.assert_not_called()
            self.assertEqual(path.read_text('utf-8'), 'already frozen')

    def test_preservation_does_not_rehash_prior_experiment_payloads(self):
        service = p.ROOT / 'backend/models/weights'
        refs = [{'path': str(path), 'sha256': digest(str(path))} for path in
                [service / 'baseline.pt', p.ROOT / 'experiments/prior/runs/model.pt',
                 p.ROOT / 'experiments/prior/results/posteriors.npz', p.ROOT / 'experiments/prior/plan.json']]
        bundle = {'prior_preservation': {'artifacts': refs, 'weights_directory': str(service),
                                        'weights_inventory': [str(service / 'baseline.pt')], 'absent_paths': []},
                  'source_refs': {}, 'prior_execution_code': []}
        contract = p.preservation_contract(bundle)
        self.assertEqual(contract['artifacts'], [refs[0], refs[3]])
        self.assertEqual(contract['prior_payload_bindings_preserved_without_rehashing'], refs[1:3])
        self.assertEqual(contract['historical_trained_checkpoint_or_posterior_payloads_rehashed'], 0)


if __name__ == '__main__':
    unittest.main()
