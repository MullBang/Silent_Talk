"""Saved-output boundary and candidate-contract tests without scoring/model work."""
from __future__ import annotations
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import protocol as p


def training_fixture():
    records = []
    for speaker in range(8):
        for index in range(3):
            sid = f'train_{speaker}_{index}'
            records.append({'id': sid, 'speaker_id': f'S{speaker}', 'source_stem': f'source{speaker}',
                            'text': sid, 'tokens': [speaker + 1, index + 10], 'generation': 'old' if speaker < 4 else 'new',
                            'original_split': 'train', 'original_role': 'old_train' if speaker < 4 else 'new_provisional_train',
                            'diagnostic_role': 'train'})
    return records


def source_fixture(cohort='same'):
    row = {'id': 'probe', 'speaker_id': 'S0', 'source_stem': 'source0', 'reference': 'probe text',
           'target_token_ids': [1, 2], 'frames': 3, 'status': 'ok', 'error': None,
           'condition': 'normal', 'split': 'same_recording_holdout' if cohort == 'same' else 'internal_holdout',
           'diagnostic_role': 'same_recording_holdout' if cohort == 'same' else 'internal_holdout',
           'source_split': 'train', 'raw_argmax_ids': [1, 0, 2], 'collapsed_ids': [1, 2], 'log_posteriors_key': 'array0'}
    selected = {'id': 'probe', 'speaker_id': 'S0', 'source_stem': 'source0', 'text': 'probe text', 'tokens': [1, 2], 'frames': 3}
    source = {**selected, 'key': 'source0', 'source_record': row, 'source_record_sha256': p.canonical_sha(row),
              'posterior_archive': {'path': 'probabilities.npz', 'sha256': 'a' * 64}, 'posterior_key': 'array0'}
    logits = np.full((3, 69), -5., np.float64)
    logits[np.arange(3), [1, 0, 2]] = 5.
    values = logits - np.log(np.exp(logits).sum(axis=1))[:, None]
    return source, row, selected, values.astype(np.float32)


class CandidateTests(unittest.TestCase):
    def test_exact_reference1_train24_and_speaker3_other21(self):
        training = training_fixture()
        result = p.candidates_for(source_fixture()[0], training)
        self.assertEqual(len(result), 25)
        self.assertEqual(result[0]['kind'], 'reference')
        self.assertEqual([row['id'] for row in result[1:]], sorted(row['id'] for row in training))
        self.assertEqual(sum(row['same_speaker'] for row in result[1:]), 3)
        self.assertEqual(sum(not row['same_speaker'] for row in result[1:]), 21)

    def test_candidates_are_copies(self):
        training = training_fixture()
        source = source_fixture()[0]
        result = p.candidates_for(source, training)
        result[0]['tokens'][0] = 9
        result[1]['tokens'][0] = 8
        self.assertEqual(source['tokens'][0], 1)
        self.assertEqual(training[0]['tokens'][0], 1)

    def test_candidate_order_stable_when_inventory_order_changes(self):
        source, training = source_fixture()[0], training_fixture()
        expected = p.candidates_for(source, training)
        self.assertEqual(p.candidates_for(source, list(reversed(training))), expected)

    def test_reference_id_in_training_rejected(self):
        source = source_fixture()[0]
        source['id'] = training_fixture()[0]['id']
        with self.assertRaisesRegex(ValueError, 'Reference leaked'):
            p.candidates_for(source, training_fixture())

    def test_unicode_normalized_reference_duplicate_rejected(self):
        source, training = source_fixture()[0], training_fixture()
        source['text'] = ' \u1100\u1161\t나\u00a0'
        training[0]['text'] = '가나'
        with self.assertRaisesRegex(ValueError, 'Reference leaked'):
            p.candidates_for(source, training)

    def test_unknown_speaker_cannot_change_comparison_denominator(self):
        source = source_fixture()[0]
        source['speaker_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'same-speaker3'):
            p.candidates_for(source, training_fixture())

    def test_duplicate_training_ids_rejected(self):
        records = training_fixture()
        records[-1] = copy.deepcopy(records[0])
        with self.assertRaisesRegex(ValueError, 'Expected train24'):
            p.candidates_for(source_fixture()[0], records)

    def test_development_role_is_not_training_candidate(self):
        records = training_fixture()
        ids = [row['id'] for row in records]
        records[0]['original_split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'Nontraining'):
            p.validate_training_candidates(records, ids)

    def test_duplicate_training_targets_rejected(self):
        records = training_fixture()
        records[1]['text'] = records[0]['text']
        with self.assertRaisesRegex(ValueError, 'Duplicate normalized'):
            p.validate_training_candidates(records, [row['id'] for row in records])

    def test_unbalanced_speaker_candidate_counts_rejected(self):
        records = training_fixture()
        records[0]['speaker_id'] = records[3]['speaker_id']
        with self.assertRaisesRegex(ValueError, 'speaker balance'):
            p.validate_training_candidates(records, [row['id'] for row in records])

    def test_infeasibility_counts_adjacent_repeated_tokens(self):
        self.assertEqual(p.validate_tokens([1, 1, 2, 2, 2]), 8)
        self.assertEqual(p.validate_tokens([], allow_empty=True), 0)

    def test_blank_or_boolean_tokens_rejected(self):
        for tokens in ([0], [True], [69], []):
            with self.assertRaisesRegex(ValueError, 'canonical CTC'):
                p.validate_tokens(tokens)


class SourceTests(unittest.TestCase):
    def test_same_and_reused_roles_validated_separately(self):
        for cohort in ('same', 'different'):
            _, row, selected, _ = source_fixture(cohort)
            p.validate_source_record(row, selected, 'normal', cohort)

    def test_failed_source_not_silently_removed(self):
        _, row, selected, _ = source_fixture()
        row['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'role/status'):
            p.validate_source_record(row, selected, 'normal', 'same')

    def test_changed_reference_not_accepted(self):
        _, row, selected, _ = source_fixture()
        row['reference'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'identity'):
            p.validate_source_record(row, selected, 'normal', 'same')

    def test_reused_row_cannot_be_mislabelled_as_new_same(self):
        _, row, selected, _ = source_fixture('different')
        with self.assertRaisesRegex(ValueError, 'role/status'):
            p.validate_source_record(row, selected, 'normal', 'same')

    def test_greedy_collapse_is_checked_before_auxiliary_scoring(self):
        _, row, selected, _ = source_fixture()
        row['collapsed_ids'] = [1]
        with self.assertRaisesRegex(ValueError, 'Greedy CTC collapse'):
            p.validate_source_record(row, selected, 'normal', 'same')

    def test_empty_greedy_valid_when_every_frame_blank(self):
        _, row, selected, _ = source_fixture()
        row['raw_argmax_ids'] = [0, 0, 0]
        row['collapsed_ids'] = []
        p.validate_source_record(row, selected, 'normal', 'same')

    def test_canonical_hash_ignores_mapping_order_not_content(self):
        row = source_fixture()[1]
        expected = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        self.assertEqual(p.canonical_sha(dict(reversed(list(row.items())))), expected)
        row['frames'] = 4
        self.assertNotEqual(p.canonical_sha(row), expected)


class ProbabilityTests(unittest.TestCase):
    def test_pure_scoring_dependencies_are_frozen(self):
        refs = p.code_bindings()
        bound = {ref['path']: ref['sha256'] for ref in refs}
        for path, expected in p.PURE_FILES.items():
            self.assertEqual(bound[str(path)], expected)

    def test_valid_saved_fp32_array(self):
        source, _, _, values = source_fixture()
        p.validate_log_probs(values, source)

    def test_wrong_dtype_rejected(self):
        source, _, _, values = source_fixture()
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            p.validate_log_probs(values.astype(np.float64), source)

    def test_unnormalized_log_probs_rejected(self):
        source, _, _, values = source_fixture()
        with self.assertRaisesRegex(ValueError, 'Unnormalized'):
            p.validate_log_probs(values + np.float32(0.1), source)

    def test_nonfinite_log_probs_rejected(self):
        source, _, _, values = source_fixture()
        values[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finiteness'):
            p.validate_log_probs(values, source)

    def test_changed_greedy_path_rejected(self):
        source, _, _, values = source_fixture()
        changed = values.copy()
        changed[:, [1, 3]] = changed[:, [3, 1]]
        with self.assertRaisesRegex(ValueError, 'greedy path'):
            p.validate_log_probs(changed, source)

    def test_load_decodes_only_declared_saved_posterior_arrays(self):
        source, _, _, values = source_fixture()
        opened, decoded = [], []
        metadata = {'sources': [source], 'training_candidates': training_fixture()}
        class Archive:
            files = ['array0', 'unrequested_training_output']
            def __enter__(self): return self
            def __exit__(self, *args): return None
            def __getitem__(self, key):
                decoded.append(key)
                if key != 'array0': raise AssertionError('Unrequested array decoded')
                return values.copy()
        def fake_load(path, **kwargs):
            opened.append(path)
            self.assertFalse(kwargs['allow_pickle'])
            return Archive()
        with patch.object(p, 'build_metadata', return_value=metadata), patch.object(p, 'verify'), \
             patch.object(p.np, 'load', side_effect=fake_load):
            loaded = p.load_sources()
        self.assertEqual(opened, ['probabilities.npz'])
        self.assertEqual(decoded, ['array0'])
        self.assertFalse(loaded['sources'][0]['log_probs'].flags.writeable)
        self.assertEqual(loaded['counts']['student_input_npz_opened'], 0)
        self.assertEqual(loaded['counts']['checkpoint_tensors_deserialized'], 0)

    def test_changed_policy_rejected_before_npz_open(self):
        metadata = {'sources': [source_fixture()[0]], 'training_candidates': training_fixture()}
        with patch.object(p, 'build_metadata', return_value=metadata), patch.object(p.np, 'load') as opened:
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                p.load_sources({'sources': [], 'training_candidates': training_fixture()})
        opened.assert_not_called()

    def test_same_posterior_key_cannot_be_counted_twice(self):
        source = source_fixture()[0]
        duplicate = copy.deepcopy(source)
        duplicate['key'] = 'other'
        metadata = {'sources': [source, duplicate], 'training_candidates': training_fixture()}
        with patch.object(p, 'build_metadata', return_value=metadata), patch.object(p.np, 'load') as opened:
            with self.assertRaisesRegex(ValueError, 'Reused posterior key'):
                p.load_sources()
        opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
