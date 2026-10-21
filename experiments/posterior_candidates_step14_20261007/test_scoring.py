"""CPU-only numerical CTC, tie, failure-denominator, and source replay tests."""
import copy
import itertools
import unittest
from unittest import mock

import numpy as np
import torch

import scoring as subject

torch.set_num_threads(2)


def posterior(path):
    logits = torch.full((len(path), 69), -7., dtype=torch.float32)
    logits[torch.arange(len(path)), path] = 3.
    return logits.log_softmax(-1).numpy()


def fixture(path=None):
    lp = posterior(path or [0, 1, 1, 20, 20, 0, 0, 0])
    tokens = subject.tokenizer.encode_text('가')
    raw = lp.argmax(-1).tolist()
    collapsed = subject.diagnostics.collapse_ctc(raw, blank_id=0)
    prediction = subject.tokenizer.compose_jamos([subject.tokenizer.IDX_TO_GRAPHEME[i] for i in collapsed], strict=False)
    try:
        strict = {'status': 'ok', 'prediction': subject.tokenizer.decode_labels(collapsed), 'error': None}
    except ValueError as exc:
        strict = {'status': 'failed', 'prediction': None, 'error': f'ValueError: {exc}'}
    ctc = float(torch.nn.functional.ctc_loss(torch.from_numpy(lp).unsqueeze(1), torch.tensor(tokens),
                torch.tensor([len(lp)]), torch.tensor([len(tokens)]), blank=0, reduction='mean', zero_infinity=False))
    record = {'id': 'ref', 'speaker_id': 's0', 'condition': 'normal', 'reference': '가',
              'prediction': prediction, 'frames': len(lp), 'status': 'ok', 'error': None, 'source_split': 'train',
              'raw_argmax_ids': raw, 'collapsed_ids': collapsed, 'target_token_ids': tokens,
              'replacement_characters': prediction.count('\ufffd'), 'strict_label_decoder_diagnostic': strict,
              'metrics': subject.metrics.counts('가', prediction), 'ctc_loss_mean': ctc,
              'token_metrics': {'token_edits': subject.diagnostics.edit_distance(tokens, collapsed),
                                'target_tokens': len(tokens), 'exact': tokens == collapsed}}
    source = {'key': 'seed42/A_ctc/normal/same/ref', 'seed': 42, 'arm': 'A_ctc', 'condition': 'normal',
              'cohort': 'same', 'id': 'ref', 'speaker_id': 's0', 'text': '가', 'tokens': tokens, 'frames': len(lp),
              'source_record': record, 'source_record_sha256': subject.canonical_sha(record),
              'source_evaluation': {'path': 'source.json', 'sha256': 'source'},
              'posterior_archive': {'path': 'posterior.npz', 'sha256': 'posterior'},
              'posterior_key': 'normal0', 'log_probs': lp}
    candidates = [{'kind': 'reference', 'id': 'ref', 'text': '가', 'tokens': tokens, 'speaker_id': 's0', 'same_speaker': True}]
    texts = ['나','다','라','마','바','사','아','자','차','카','타','파','하','거','너','더','러','머','버','서','어','저','처','커']
    for index, text in enumerate(texts):
        candidates.append({'kind': 'training', 'id': f'train{index:02d}', 'text': text,
                           'tokens': subject.tokenizer.encode_text(text),
                           'speaker_id': 's0' if index < 3 else f's{1 + (index-3)//3}', 'same_speaker': index < 3})
    return source, candidates


def candidate(cid, total, length, same=False, kind='training'):
    return {'id': cid, 'kind': kind, 'same_speaker': same, 'status': 'ok',
            'total_log_probability': total, 'mean_log_probability_per_target_token': total/length}


class ScoringTests(unittest.TestCase):
    def test_ctc_matches_exhaustive_path_sum_for_repeat_and_blank_cases(self):
        logits = torch.full((4, 69), -10., dtype=torch.float64)
        logits[:, :3] = torch.tensor([[1.,2.,0.], [2.,0.,1.], [0.,1.,2.], [1.,0.,2.]], dtype=torch.float64)
        lp = logits.log_softmax(-1)
        for target in ([], [1], [1, 1], [1, 2], [2, 1]):
            expected = 0.
            for path in itertools.product(range(3), repeat=4):
                if subject.diagnostics.collapse_ctc(list(path), blank_id=0) == target:
                    expected += float(torch.exp(sum(lp[t, value] for t, value in enumerate(path))))
            result = subject.ctc_score(lp, list(target))
            self.assertAlmostEqual(result['total_log_probability'], np.log(expected), places=12)

    def test_repeated_tokens_require_intervening_blank_frames(self):
        lp = torch.from_numpy(posterior([1, 1])).double()
        result = subject.ctc_score(lp, [1, 1])
        self.assertEqual(result['status'], 'infeasible')
        self.assertEqual(result['minimum_frames'], 3)
        self.assertIsNone(result['total_log_probability'])
        self.assertIsNone(result['mean_log_probability_per_target_token'])
        self.assertTrue(result['probability_is_zero'])

    def test_empty_greedy_is_blank_column_sum_without_normalized_score(self):
        source, candidates = fixture([0]*8)
        value = subject.score_source(source, candidates)
        self.assertEqual(value['status'], 'ok')
        greedy = value['greedy_score']
        self.assertEqual(greedy['tokens'], [])
        self.assertEqual(greedy['target_tokens'], 0)
        self.assertIsNone(greedy['mean_log_probability_per_target_token'])
        self.assertEqual(greedy['total_log_probability'], float(source['log_probs'][:, 0].astype(np.float64).sum()))

    def test_valid_source_produces25_plus_separate_greedy_and_preserves_rng(self):
        source, candidates = fixture()
        original = copy.deepcopy(source['source_record'])
        rng = torch.get_rng_state().clone()
        with mock.patch.object(torch, 'load', side_effect=AssertionError('Checkpoint loading is forbidden')):
            value = subject.score_source(source, candidates)
        self.assertEqual(value['status'], 'ok')
        self.assertEqual(len(value['candidates']), 25)
        self.assertEqual(value['candidate_pairs'], 25)
        self.assertEqual(value['auxiliary_greedy_pairs'], 1)
        self.assertEqual(value['reference_minus_greedy_total_log_probability'], 0.)
        self.assertEqual(value['source_validation']['raw_argmax_ids'], original['raw_argmax_ids'])
        self.assertEqual(source['source_record'], original)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_reference_rank_tie_epsilon_top_ties_and_stable_id_order(self):
        rows = [candidate('ref', -2., 1, kind='reference'), candidate('z', -2.+5e-9, 1, True),
                candidate('a', -2.+5e-9, 1), candidate('b', -1., 1)]
        value = subject.rank_candidates(rows, 'total_log_probability')
        self.assertEqual(value['reference_rank'], 2)
        self.assertEqual(value['reference_tie_ids'], ['a', 'ref', 'z'])
        self.assertEqual(value['ordered_candidate_ids'], ['b', 'a', 'z', 'ref'])
        self.assertEqual(value['top_candidate_ids'], ['b'])
        self.assertEqual(value['best_training_relation'], 'other_speaker')

    def test_training_top_ties_can_span_same_and_other_speaker(self):
        rows = [candidate('ref', -1., 1, kind='reference'), candidate('z', -2., 1, True), candidate('a', -2., 1)]
        value = subject.rank_candidates(rows, 'total_log_probability')
        self.assertEqual(value['best_training_relation'], 'mixed_tie')
        self.assertEqual(value['best_training_ids'], ['a', 'z'])
        self.assertEqual(value['reference_minus_best_training'], 1.)
        self.assertEqual(value['same_minus_other_training_score'], 0.)

    def test_length_heuristic_can_reverse_total_probability_rank(self):
        rows = [candidate('ref', -10., 10, kind='reference'), candidate('short', -3., 1)]
        self.assertEqual(subject.rank_candidates(rows, 'total_log_probability')['reference_rank'], 2)
        self.assertEqual(subject.rank_candidates(rows, 'mean_log_probability_per_target_token')['reference_rank'], 1)

    def test_infeasible_candidates_remain_in25_but_not_feasible_rank_order(self):
        source, candidates = fixture()
        candidates[-1].update(text='가'*6, tokens=subject.tokenizer.encode_text('가'*6))
        value = subject.score_source(source, candidates)
        self.assertEqual(value['status'], 'ok')
        self.assertEqual(len(value['candidates']), 25)
        for ranking in value['rankings'].values():
            self.assertEqual(ranking['infeasible_candidate_ids'], ['train23'])
            self.assertEqual(ranking['feasible_candidates'], 24)
            self.assertEqual(len(ranking['ordered_candidate_ids']), 24)
        self.assertEqual(value['candidates'][-1]['status'], 'infeasible')

    def test_no_feasible_training_candidates_keep_reference_rank_and_null_margins(self):
        rows = [candidate('ref', -1., 1, kind='reference'),
                {'id': 'bad', 'kind': 'training', 'status': 'infeasible', 'same_speaker': True,
                 'total_log_probability': None, 'mean_log_probability_per_target_token': None}]
        result = subject.rank_candidates(rows, 'total_log_probability')
        self.assertEqual(result['reference_rank'], 1)
        self.assertIsNone(result['best_training_relation'])
        self.assertIsNone(result['reference_minus_best_training'])

    def test_nonfinite_wrong_dtype_shape_and_unnormalized_sources_keep_denominators(self):
        for mode in ('nan', 'dtype', 'shape', 'normalization'):
            source, candidates = fixture()
            if mode == 'nan': source['log_probs'][0, 0] = np.nan
            elif mode == 'dtype': source['log_probs'] = source['log_probs'].astype(np.float64)
            elif mode == 'shape': source['log_probs'] = source['log_probs'][:, :68]
            else: source['log_probs'] += 1.
            value = subject.score_source(source, candidates)
            self.assertEqual(value['status'], 'failed')
            self.assertEqual(len(value['candidates']), 25)
            self.assertIsNone(value['rankings'])
            self.assertTrue(all(c['status'] == 'source_failed' and c['total_log_probability'] is None for c in value['candidates']))

    def test_failed_original_inference_is_explicit_without_row_deletion(self):
        source, candidates = fixture()
        source['source_record']['status'] = 'failed'
        source['source_record_sha256'] = subject.canonical_sha(source['source_record'])
        value = subject.score_source(source, candidates)
        self.assertEqual(value['status'], 'failed')
        self.assertIn('Original source inference failed', value['error'])
        self.assertEqual(value['candidate_pairs'], 25)

    def test_source_checksum_and_recomputed_greedy_tokens_metrics_ctc_are_enforced(self):
        changes = [('raw_argmax_ids', []), ('collapsed_ids', []), ('prediction', '나'),
                   ('replacement_characters', 10), ('token_metrics', {}), ('metrics', {}),
                   ('ctc_loss_mean', 99.), ('strict_label_decoder_diagnostic', {})]
        for field, replacement in changes:
            source, candidates = fixture()
            source['source_record'][field] = replacement
            # Rehash to exercise replay, not just the original row checksum.
            source['source_record_sha256'] = subject.canonical_sha(source['source_record'])
            self.assertEqual(subject.score_source(source, candidates)['status'], 'failed', field)
        source, candidates = fixture()
        source['source_record_sha256'] = 'bad'
        self.assertIn('checksum', subject.score_source(source, candidates)['error'])

    def test_uncomposable_greedy_tokens_scored_directly_without_text_reencoding(self):
        source, candidates = fixture([20, 20, 0, 20, 0, 0, 0, 0])
        value = subject.score_source(source, candidates)
        self.assertEqual(value['status'], 'ok')
        self.assertEqual(value['greedy_score']['tokens'], [20, 20])
        self.assertEqual(value['source_validation']['prediction'], '\ufffd\ufffd')
        self.assertEqual(value['source_validation']['replacement_characters'], 2)

    def test_invalid_target_ids_or_nonfloat64_scoring_tensor_rejected(self):
        lp = torch.from_numpy(posterior([1, 20])).double()
        for tokens in ([0], [69], [-1], [1.], [True]):
            with self.assertRaises(ValueError): subject.ctc_score(lp, tokens)
        with self.assertRaises(ValueError): subject.ctc_score(lp.float(), [1, 20])

    def test_duplicate_reference_wrong_speaker_reordering_and_candidate_target_abort(self):
        for mode in ('duplicate', 'speaker', 'reorder', 'target'):
            source, candidates = fixture()
            if mode == 'duplicate': candidates[-1]['id'] = 'ref'
            elif mode == 'speaker': candidates[-1]['same_speaker'] = True
            elif mode == 'reorder': candidates[1], candidates[2] = candidates[2], candidates[1]
            else: candidates[1]['tokens'] = [1, 20]
            with self.assertRaises(ValueError): subject.score_source(source, candidates)

    def test_tie_order_is_independent_of_original_training_candidate_order(self):
        rows = [candidate('ref', -1., 1, kind='reference'), candidate('z', -2., 1, True), candidate('a', -2., 1)]
        self.assertEqual(subject.rank_candidates(rows, 'total_log_probability'),
                         subject.rank_candidates(list(reversed(rows)), 'total_log_probability'))


if __name__ == '__main__':
    unittest.main()
