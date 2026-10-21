"""Independent CTC audit boundaries, with exhaustive tiny-path references."""
import importlib.util
import itertools
import math
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).with_name('independent_audit.py')
spec = importlib.util.spec_from_file_location('step14_independent_under_test', PATH)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def probabilities(frames=4):
    values = np.full((frames, 69), 1e-20, dtype=np.float64)
    values[:, :3] = np.array([[.4, .35, .25], [.3, .5, .2], [.6, .1, .3], [.2, .3, .5]])[:frames]
    return np.log(values)


def exhaustive(logp, target):
    return math.log(math.fsum(math.exp(sum(logp[i, token] for i, token in enumerate(path)))
                             for path in itertools.product((0, 1, 2), repeat=len(logp))
                             if audit.collapse(path) == target))


class CtcTests(unittest.TestCase):
    def test_single_token_matches_all_paths(self):
        lp = probabilities(3)
        self.assertAlmostEqual(audit.ctc_log_probability(lp, [1]), exhaustive(lp, [1]), places=12)

    def test_two_distinct_tokens_match_all_paths(self):
        lp = probabilities(3)
        self.assertAlmostEqual(audit.ctc_log_probability(lp, [1, 2]), exhaustive(lp, [1, 2]), places=12)

    def test_repeated_token_requires_intervening_blank(self):
        lp = probabilities(3)
        self.assertAlmostEqual(audit.ctc_log_probability(lp, [1, 1]), float(lp[0, 1] + lp[1, 0] + lp[2, 1]), places=12)

    def test_repeated_token_matches_all_four_frame_paths(self):
        lp = probabilities(4)
        self.assertAlmostEqual(audit.ctc_log_probability(lp, [2, 2]), exhaustive(lp, [2, 2]), places=12)

    def test_infeasible_repeated_target_returns_none(self):
        self.assertIsNone(audit.ctc_log_probability(probabilities(2), [1, 1]))

    def test_exact_minimum_distinct_path(self):
        lp = probabilities(2)
        self.assertAlmostEqual(audit.ctc_log_probability(lp, [1, 2]), float(lp[0, 1] + lp[1, 2]), places=12)

    def test_score_is_log_probability_not_zero_infinity(self):
        score = audit.score_value(probabilities(2), [1, 1])
        self.assertEqual(score['status'], 'infeasible')
        self.assertIsNone(score['total_log_probability'])
        self.assertIsNone(score['mean_log_probability_per_target_token'])

    def test_normalized_score_divides_by_target_token_count(self):
        score = audit.score_value(probabilities(3), [1, 2])
        self.assertEqual(score['mean_log_probability_per_target_token'], score['total_log_probability'] / 2)

    def test_empty_greedy_is_blank_only_path(self):
        lp = probabilities(3)
        score = audit.score_value(lp, [], allow_empty=True)
        self.assertAlmostEqual(score['total_log_probability'], float(lp[:, 0].sum()), places=12)
        self.assertEqual(score['target_tokens'], 0)
        self.assertIsNone(score['mean_log_probability_per_target_token'])

    def test_empty_candidate_target_rejected(self):
        with self.assertRaises(ValueError):
            audit.ctc_log_probability(probabilities(3), [])

    def test_blank_in_target_rejected(self):
        with self.assertRaises(ValueError):
            audit.minimum_frames([0])

    def test_boolean_token_rejected(self):
        with self.assertRaises(ValueError):
            audit.minimum_frames([True])

    def test_out_of_vocabulary_token_rejected(self):
        with self.assertRaises(ValueError):
            audit.minimum_frames([69])

    def test_nonfinite_posterior_rejected(self):
        lp = probabilities(3)
        lp[0, 1] = float('nan')
        with self.assertRaises(ValueError):
            audit.ctc_log_probability(lp, [1])

    def test_wrong_vocabulary_dimension_rejected(self):
        with self.assertRaises(ValueError):
            audit.ctc_log_probability(probabilities(3)[:, :3], [1])

    def test_zero_frames_rejected(self):
        with self.assertRaises(ValueError):
            audit.ctc_log_probability(np.zeros((0, 69)), [1])

    def test_ctc_collapse_removes_repeats_before_blank(self):
        self.assertEqual(audit.collapse([1, 1, 0, 1, 2, 2, 0]), [1, 1, 2])


class IntegrityTests(unittest.TestCase):
    def test_fp32_saved_posterior_is_accepted(self):
        self.assertLess(audit.posterior_valid(probabilities(3).astype(np.float32), 3), 1e-4)

    def test_fp64_source_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.posterior_valid(probabilities(3), 3)

    def test_unnormalized_posterior_rejected(self):
        with self.assertRaises(ValueError):
            audit.posterior_valid((probabilities(3) - 2).astype(np.float32), 3)

    def test_changed_frame_count_rejected(self):
        with self.assertRaises(ValueError):
            audit.posterior_valid(probabilities(3).astype(np.float32), 4)

    def test_raw_jamos_keep_replacement_character(self):
        self.assertEqual(audit.compose([1, 20, 20]), '가\ufffd')

    def test_hangul_composition_keeps_space(self):
        self.assertEqual(audit.compose([1, 20, 68, 3, 20]), '가 나')

    def test_canonical_source_hash_independent_of_dict_order(self):
        self.assertEqual(audit.canonical_sha({'가': 1, '나': 2}), audit.canonical_sha({'나': 2, '가': 1}))

    def test_canonical_source_hash_detects_transcript_change(self):
        self.assertNotEqual(audit.canonical_sha({'text': '가'}), audit.canonical_sha({'text': '나'}))

    def test_near_tree_rejects_changed_identity(self):
        with self.assertRaises(ValueError):
            audit.near_tree({'id': 'a', 'score': .1}, {'id': 'b', 'score': .1}, 'row')

    def test_near_tree_rejects_score_change(self):
        with self.assertRaises(ValueError):
            audit.near_tree({'score': .11}, {'score': .1}, 'row')

    def test_near_tree_rejects_added_candidate(self):
        with self.assertRaises(ValueError):
            audit.near_tree([{'id': 'a'}, {'id': 'b'}], [{'id': 'a'}], 'rows')

    def test_nfc_and_spaces_are_normalized_for_duplicate_checks(self):
        self.assertEqual(audit.normalized_text('가 나'), audit.normalized_text('\u1100\u1161\t나'))


class RankingTests(unittest.TestCase):
    def candidate(self, sid, score, kind='training', own=False, status='ok'):
        return {'id': sid, 'kind': kind, 'same_speaker': own, 'status': status,
                'total_log_probability': score, 'mean_log_probability_per_target_token': score}

    def rank(self, candidates):
        return audit.independent_ranking(candidates, 'total_log_probability')

    def test_rank_uses_strictly_better_outside_epsilon(self):
        rows = [self.candidate('ref', -2., 'reference', True), self.candidate('a', -2. + 1e-9), self.candidate('b', -1.)]
        result = self.rank(rows)
        self.assertEqual(result['reference_rank'], 2)
        self.assertEqual(result['reference_tie_ids'], ['a', 'ref'])

    def test_equal_scores_order_by_stable_id(self):
        rows = [self.candidate('z', -2., 'reference'), self.candidate('a', -2.)]
        self.assertEqual(self.rank(rows)['ordered_candidate_ids'], ['a', 'z'])

    def test_infeasible_candidate_is_never_ranked_as_zero_loss(self):
        rows = [self.candidate('ref', -2., 'reference'), self.candidate('bad', None, status='infeasible')]
        result = self.rank(rows)
        self.assertEqual(result['reference_rank'], 1)
        self.assertEqual(result['ordered_candidate_ids'], ['ref'])
        self.assertIsNone(result['best_training_score'])
        self.assertEqual(result['infeasible_candidate_ids'], ['bad'])

    def test_same_and_other_best_tie_is_reported(self):
        rows = [self.candidate('ref', -4., 'reference'), self.candidate('own', -2., own=True), self.candidate('other', -2.)]
        result = self.rank(rows)
        self.assertEqual(result['best_training_relation'], 'mixed_tie')
        self.assertEqual(result['reference_minus_best_training'], -2.)

    def test_reference_never_becomes_best_training_candidate(self):
        rows = [self.candidate('ref', -1., 'reference', True), self.candidate('own', -2., own=True)]
        result = self.rank(rows)
        self.assertEqual(result['top_candidate_ids'], ['ref'])
        self.assertEqual(result['best_training_ids'], ['own'])

    def test_duplicate_candidate_ids_rejected(self):
        with self.assertRaises(ValueError):
            self.rank([self.candidate('x', -1., 'reference'), self.candidate('x', -2.)])

    def test_two_references_rejected(self):
        with self.assertRaises(ValueError):
            self.rank([self.candidate('x', -1., 'reference'), self.candidate('y', -2., 'reference')])

    def test_nonfinite_feasible_score_rejected(self):
        with self.assertRaises(ValueError):
            self.rank([self.candidate('x', -1., 'reference'), self.candidate('y', float('nan'))])


class AggregateTests(unittest.TestCase):
    def test_all_unavailable_values_keep_denominator(self):
        result = audit.numeric_summary([None, None], margins=True)
        self.assertEqual(result['count'], 0)
        self.assertEqual(result['unavailable'], 2)
        self.assertIsNone(result['mean'])

    def test_margin_sign_uses_declared_epsilon(self):
        result = audit.numeric_summary([-1., -1e-9, 0., 1e-9, 1., None], margins=True)
        self.assertEqual((result['positive_count'], result['tied_count'], result['negative_count']), (1, 3, 1))
        self.assertEqual(result['unavailable'], 1)

    def test_even_median_averages_two_middle_values(self):
        self.assertEqual(audit.numeric_summary([8., 1., 4., 2.])['median'], 3.)

    def test_odd_median_selects_middle_value(self):
        self.assertEqual(audit.numeric_summary([8., 1., 2.])['median'], 2.)

    def test_nonfinite_summary_rejected(self):
        with self.assertRaises(ValueError):
            audit.numeric_summary([float('inf')])

    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):
            audit.aggregate_scored_rows([])


if __name__ == '__main__':
    unittest.main()
