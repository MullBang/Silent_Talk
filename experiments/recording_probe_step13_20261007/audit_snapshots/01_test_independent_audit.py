"""Negative boundary tests for the independent Step13 audit, CPU only."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('independent_audit.py')
spec = importlib.util.spec_from_file_location('step13_audit_under_test', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class TimingTests(unittest.TestCase):
    def row(self, **changes):
        return {'source_stem': 'one', 'start_time': 1., 'end_time': 1.12,
                'video_fps': 25., 'frames': 3, **changes}

    def test_frame_selection_uses_adjusted_start(self):
        self.assertEqual(audit.native_positions(self.row()), [25, 26, 27])

    def test_distinct_recordings_are_not_time_compared(self):
        self.assertFalse(audit.overlap(self.row(), self.row(source_stem='two')))

    def test_partial_interval_overlap_is_rejected(self):
        self.assertTrue(audit.overlap(self.row(), self.row(start_time=1.10, end_time=1.22)))

    def test_adjacent_intervals_without_shared_frames_are_valid(self):
        self.assertFalse(audit.overlap(self.row(), self.row(start_time=1.12, end_time=1.24)))

    def test_disjoint_intervals_can_still_reuse_native_frame(self):
        first = self.row(start_time=1., end_time=1.005, frames=1)
        second = self.row(start_time=1.01, end_time=1.015, frames=1)
        self.assertTrue(audit.overlap(first, second))

    def test_nonpositive_fps_rejected(self):
        with self.assertRaises(ValueError):
            audit.native_positions(self.row(video_fps=0.))

    def test_nan_timing_rejected(self):
        with self.assertRaises(ValueError):
            audit.native_positions(self.row(start_time=float('nan')))

    def test_reversed_interval_rejected(self):
        with self.assertRaises(ValueError):
            audit.native_positions(self.row(end_time=.9))

    def test_zero_frames_rejected(self):
        with self.assertRaises(ValueError):
            audit.native_positions(self.row(frames=0))

    def test_boolean_frames_rejected(self):
        with self.assertRaises(ValueError):
            audit.native_positions(self.row(frames=True))


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handoff, cls.rows, cls.entries, cls.provenance, _ = audit.previous.source_metadata()
        cls.prior = audit.read(audit.PRIOR / 'diagnostic_plan.json', audit.previous.POLICY_SHA)

    def choose(self, rows=None, provenance=None):
        return audit.independent_selection(self.prior, self.handoff, rows or self.rows,
                                           self.entries, provenance or self.provenance)

    def test_real_selection_retains_six_known_speakers(self):
        selection = self.choose()
        self.assertEqual([self.rows[sid]['speaker_id'] for sid in selection['same_ids']], list(audit.SPEAKERS))
        self.assertEqual(sum(self.rows[sid]['frames'] for sid in selection['same_ids']), 722)
        audit.validate_selected_scope(selection, self.prior, self.rows, self.entries, self.provenance)

    def test_real_selection_reconstructs_all_27_candidate_decisions(self):
        events = self.choose()['candidate_events']
        self.assertEqual(len(events), 27)
        self.assertEqual(sum(row['status'] == 'selected' for row in events), 6)
        self.assertEqual(sum(row['status'] == 'skipped' for row in events), 0)

    def test_two_unavailable_speakers_have_zero_candidates(self):
        missing = [r for r in self.choose()['speaker_selections'] if r['selected_id'] is None]
        self.assertEqual([(r['speaker_id'], r['candidate_count']) for r in missing], [('C058', 0), ('C658', 0)])

    def test_changed_target_to_training_sentence_skips_candidate(self):
        selected = self.choose()['same_ids'][0]
        rows = copy.deepcopy(self.rows)
        rows[selected]['text'] = rows[self.prior['selection']['train_ids'][0]]['text']
        self.assertNotIn(selected, self.choose(rows=rows)['same_ids'])

    def test_source_hash_mismatch_skips_candidate(self):
        selected = self.choose()['same_ids'][0]
        provenance = copy.deepcopy(self.provenance)
        provenance[selected]['video_sha256'] = '0' * 64
        self.assertNotIn(selected, self.choose(provenance=provenance)['same_ids'])

    def test_infeasible_ctc_skips_candidate(self):
        selected = self.choose()['same_ids'][0]
        rows = copy.deepcopy(self.rows)
        rows[selected]['token_ids'] = [1] * 1000
        self.assertNotIn(selected, self.choose(rows=rows)['same_ids'])

    def test_training_interval_overlap_skips_candidate(self):
        selected = self.choose()['same_ids'][0]
        rows = copy.deepcopy(self.rows)
        original = rows[self.prior['selection']['train_ids'][0]]
        rows[selected]['start_time'], rows[selected]['end_time'] = original['start_time'], original['end_time']
        self.assertNotIn(selected, self.choose(rows=rows)['same_ids'])

    def test_training_id_in_probe_rejected(self):
        selected = self.choose()
        selected['same_ids'][0] = self.prior['selection']['train_ids'][0]
        with self.assertRaises(ValueError):
            audit.validate_selected_scope(selected, self.prior, self.rows, self.entries, self.provenance)

    def test_reordered_different_speakers_rejected(self):
        selected = self.choose()
        selected['different_ids'][0], selected['different_ids'][1] = selected['different_ids'][1], selected['different_ids'][0]
        with self.assertRaises(ValueError):
            audit.validate_selected_scope(selected, self.prior, self.rows, self.entries, self.provenance)

    def test_duplicate_probe_id_rejected(self):
        selected = self.choose()
        selected['same_ids'][1] = selected['same_ids'][0]
        with self.assertRaises(ValueError):
            audit.validate_selected_scope(selected, self.prior, self.rows, self.entries, self.provenance)

    def test_nfc_and_whitespace_normalization_detects_equivalence(self):
        self.assertEqual(audit.previous.normalized_text('가 나'), audit.previous.normalized_text('\u1100\u1161\t나'))


class EvidenceTests(unittest.TestCase):
    def test_probe_absent_from_actual_and_planned_updates(self):
        audit.require_zero_exposure([{'ids': ['train']}], [['train']], ['probe'])

    def test_probe_in_actual_history_rejected(self):
        with self.assertRaises(ValueError):
            audit.require_zero_exposure([{'ids': ['probe']}], [['train']], ['probe'])

    def test_probe_in_planned_schedule_rejected(self):
        with self.assertRaises(ValueError):
            audit.require_zero_exposure([{'ids': ['train']}], [['probe']], ['probe'])

    def test_reused_rows_preserved_exactly(self):
        rows = [{'id': 'a', 'prediction': '가'}, {'id': 'b', 'prediction': '나'}]
        audit.validate_reused_records([rows[1]], rows, ['b'])

    def test_modified_reused_prediction_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_reused_records([{'id': 'a', 'prediction': 'changed'}], [{'id': 'a', 'prediction': 'original'}], ['a'])

    def test_duplicate_original_row_rejected(self):
        with self.assertRaises(ValueError):
            audit.validate_reused_records([{'id': 'a'}], [{'id': 'a'}, {'id': 'a'}], ['a'])

    def test_missing_model_pair_rejected(self):
        with self.assertRaises(ValueError):
            audit.paired_comparisons([])

    def test_empty_evaluator_dependency_evidence_rejected(self):
        with self.assertRaises(ValueError):
            audit.verify_frozen_sources([], [])

    def test_duplicated_evaluator_dependency_rejected(self):
        reference = {'path': str(PATH), 'sha256': audit.sha(PATH)}
        with self.assertRaises(ValueError):
            audit.verify_frozen_sources([reference, reference], [reference])


if __name__ == '__main__':
    unittest.main()
