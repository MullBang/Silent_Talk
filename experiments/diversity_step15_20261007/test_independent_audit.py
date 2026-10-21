"""Independent leakage, cache and future-budget boundary tests; CPU only."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).with_name('independent_audit.py')
spec = importlib.util.spec_from_file_location('step15_independent_under_test', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = audit.source_metadata()
        cls.selected = audit.independent_selection(cls.bundle)

    def test_real_metadata_reconstructs_fixed_old8_split(self):
        for key, size in [('train24_ids', 24), ('train72_ids', 72), ('dev16_ids', 16)]:
            self.assertEqual(len(self.selected[key]), size)
        self.assertEqual(self.selected['inventory']['experiment_eligible_count'], 368)
        self.assertEqual(len(self.selected['events']), 218)
        audit.verify_selected_separation(self.selected, self.bundle)

    def test_new_provisional_pool_is_preserved_but_excluded(self):
        rows = [row for row in self.selected['inventory']['records'] if row['generation'] == 'new']
        self.assertEqual(len(rows), 69)
        self.assertTrue(all(not row['eligible'] for row in rows))

    def test_old_dev_duplicate_transcript_is_blocked(self):
        row = next(row for row in self.selected['inventory']['records'] if row['id'] == 'lip_J_1_F_02_C004_A_005_004')
        self.assertFalse(row['eligible'])
        self.assertIn('prior_probe_or_excluded_normalized_target', row['reasons'])

    def test_selected_cross_speaker_duplicate_is_skipped(self):
        skipped = [row for row in self.selected['events'] if row['status'] == 'skipped']
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]['id'], 'lip_J_1_F_03_C312_A_003_012')
        self.assertIn('duplicate_selected_normalized_target', skipped[0]['reasons'])

    def test_same_stem_different_video_hash_rejected(self):
        value = copy.deepcopy(self.bundle)
        sid = self.selected['train72_ids'][0]
        value['provenance'][sid]['video_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            audit.candidate_inventory(value)

    def test_development_id_in_training_rejected(self):
        value = copy.deepcopy(self.selected)
        value['train72_ids'][-1] = value['dev16_ids'][0]
        with self.assertRaises(ValueError):
            audit.verify_selected_separation(value, self.bundle)

    def test_control_id_outside_expanded_rejected(self):
        value = copy.deepcopy(self.selected)
        value['train24_ids'][0] = value['dev16_ids'][0]
        with self.assertRaises(ValueError):
            audit.verify_selected_separation(value, self.bundle)

    def test_duplicate_normalized_target_between_splits_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        a, b = self.selected['train72_ids'][0], self.selected['dev16_ids'][0]
        bundle['rows'][b]['text'] = bundle['rows'][a]['text']
        with self.assertRaises(ValueError):
            audit.verify_selected_separation(self.selected, bundle)

    def test_same_recording_native_frame_reuse_rejected(self):
        first = {'source_stem': 's', 'start_time': 1., 'end_time': 1.005, 'frames': 1, 'video_fps': 25.}
        second = {**first, 'start_time': 1.01, 'end_time': 1.015}
        self.assertTrue(audit.overlap(first, second))

    def test_disjoint_recordings_may_share_clock_time(self):
        first = {'source_stem': 's', 'start_time': 1., 'end_time': 1.1, 'frames': 2, 'video_fps': 25.}
        self.assertFalse(audit.overlap(first, {**first, 'source_stem': 't'}))


class ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = {f'{speaker}_{index}': {'id': f'{speaker}_{index}', 'speaker_id': speaker}
                       for speaker in audit.SPEAKERS for index in range(9)}
        cls.small = [sid for sid in cls.records if int(sid.rsplit('_', 1)[1]) < 3]
        cls.large = list(cls.records)

    def test_pair_has_identical_speaker_order_and_distinct_batch_speakers(self):
        first = audit.schedule_for(self.small, self.records, 42)
        second = audit.schedule_for(self.large, self.records, 42)
        speakers = lambda value: [[self.records[sid]['speaker_id'] for sid in batch] for batch in value]
        self.assertEqual(speakers(first), speakers(second))
        self.assertTrue(all(len(set(batch)) == 2 for batch in speakers(first)))

    def test_exposure_counts_differ_300_versus_100(self):
        for ids, count in [(self.small, 300), (self.large, 100)]:
            result = audit.schedule_for(ids, self.records, 43)
            self.assertEqual(len(result), 3600)
            self.assertEqual(audit.Counter(sid for batch in result for sid in batch), audit.Counter({sid: count for sid in ids}))

    def test_different_seeds_change_schedule(self):
        self.assertNotEqual(audit.schedule_for(self.small, self.records, 42), audit.schedule_for(self.small, self.records, 43))

    def test_unbalanced_speaker_count_rejected(self):
        ids = self.small.copy()
        ids[-1] = 'C002_8'
        with self.assertRaises(ValueError):
            audit.schedule_for(ids, self.records, 42)

    def test_duplicate_training_id_rejected(self):
        ids = self.small.copy()
        ids[-1] = ids[0]
        with self.assertRaises(ValueError):
            audit.schedule_for(ids, self.records, 42)

    def test_unknown_seed_rejected(self):
        with self.assertRaises(ValueError):
            audit.schedule_for(self.small, self.records, 7)

    def test_lr_boundaries_use_next_global_update(self):
        self.assertEqual([audit.learning_rate(value) for value in (1, 2400, 2401, 3000, 3001, 3600)],
                         [.001, .001, .0003, .0003, .0001, .0001])

    def test_update_zero_rejected(self):
        with self.assertRaises(ValueError):
            audit.learning_rate(0)

    def test_update_past_budget_rejected(self):
        with self.assertRaises(ValueError):
            audit.learning_rate(3601)

    def test_boolean_update_rejected(self):
        with self.assertRaises(ValueError):
            audit.learning_rate(True)


def cache_fixture():
    row = {'id': 'clip', 'speaker_id': 'C002', 'source_stem': 's', 'frames': 4,
           'token_ids': [1, 20], 'start_time': 0., 'end_time': .16, 'video_fps': 25.}
    pixels = np.zeros((4, 96, 96, 3), dtype=np.uint8)
    indices = np.arange(4, dtype=np.int64)
    arrays = {'student_rgb': pixels, 'student_source_indices': indices, 'label_tokens': np.array([1, 20], dtype=np.int64),
              'teacher_source_indices': indices, 'student_teacher_rows': indices}
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    item = {'student_rgb_sha256': digest, 'source_verification': {'members': {}}}
    student = {'record': row, 'id': 'clip', 'source_frame_count': 10,
        'plan_sha256': 'c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30',
        'source_verification': item['source_verification'], 'student_rgb_sha256': digest}
    values = np.arange(1, 2049, dtype=np.float32).reshape(4, 512)
    target = {'features': values, 'student_source_indices': indices, 'teacher_source_indices': indices, 'gather_indices': indices}
    metadata = {'id': 'clip', 'speaker_id': 'C002', 'source_stem': 's', 'split': 'train', 'input_record': item,
        'data_record_sha256': audit.canonical_sha(row), 'aligned_feature_shape': [4, 512], 'feature_dtype': 'float32',
        'optimization_steps': 0, 'trainable_parameters': 0, 'audio_used': False, 'decoder_instantiated': False,
        'transcript_used_for_teacher_forward': False, 'state_sha256_before': 'fixed', 'state_sha256_after': 'fixed',
        'native_feature_shape': [1, 4, 512], 'feature_values_sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    entry = {**metadata, 'feature_file': {'path': 'feature.npz', 'sha256': 'bound'}, 'status': 'completed'}
    return row, item, arrays, student, target, metadata, entry


class CacheTests(unittest.TestCase):
    def test_valid_student_and_feature_cache(self):
        row, item, arrays, student, target, metadata, entry = cache_fixture()
        self.assertEqual(audit.validate_student_arrays(row, item, arrays, student)['frames'], 4)
        self.assertEqual(audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)['frames'], 4)

    def test_changed_rgb_pixel_is_rejected(self):
        row, item, arrays, student, *_ = cache_fixture()
        arrays['student_rgb'][0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            audit.validate_student_arrays(row, item, arrays, student)

    def test_wrong_student_dtype_rejected(self):
        row, item, arrays, student, *_ = cache_fixture()
        arrays['student_rgb'] = arrays['student_rgb'].astype(np.float32)
        with self.assertRaises(ValueError):
            audit.validate_student_arrays(row, item, arrays, student)

    def test_changed_label_tokens_rejected(self):
        row, item, arrays, student, *_ = cache_fixture()
        arrays['label_tokens'][0] = 2
        with self.assertRaises(ValueError):
            audit.validate_student_arrays(row, item, arrays, student)

    def test_changed_source_frame_mapping_rejected(self):
        row, item, arrays, student, *_ = cache_fixture()
        arrays['student_source_indices'] = np.array([0, 1, 2, 4], dtype=np.int64)
        with self.assertRaises(ValueError):
            audit.validate_student_arrays(row, item, arrays, student)

    def test_nonfinite_feature_rejected(self):
        row, item, arrays, _, target, metadata, entry = cache_fixture()
        target['features'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)

    def test_changed_feature_digest_rejected(self):
        row, item, arrays, _, target, metadata, entry = cache_fixture()
        target['features'][0, 0] = 2.
        with self.assertRaises(ValueError):
            audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)

    def test_zero_feature_rows_rejected(self):
        row, item, arrays, _, target, metadata, entry = cache_fixture()
        target['features'][0] = 0.
        with self.assertRaises(ValueError):
            audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)

    def test_wrong_feature_native_gather_rejected(self):
        row, item, arrays, _, target, metadata, entry = cache_fixture()
        target['gather_indices'] = np.array([0, 1, 2, 2], dtype=np.int64)
        with self.assertRaises(ValueError):
            audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)

    def test_teacher_transcript_forward_flag_rejected(self):
        row, item, arrays, _, target, metadata, entry = cache_fixture()
        metadata['transcript_used_for_teacher_forward'] = True
        entry['transcript_used_for_teacher_forward'] = True
        with self.assertRaises(ValueError):
            audit.validate_feature_arrays(row, item, entry, arrays, target, metadata)


if __name__ == '__main__':
    unittest.main()
