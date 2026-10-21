"""Deterministic CPU regressions for OLKAVS beam and EOS ownership defects.

Run: .venv-olkavs/Scripts/python.exe -m unittest discover -s
experiments/olkavs_official -p test_corrected_search.py -v
The fixtures supply scores directly, so no model, checkpoint or data is needed.
"""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from corrected_search import CorrectedSearchSequence, SearchSequence


class ScoreFixture:
    def get_scores(self, features, seq_length, g, g_att=None, ctc_output=None):
        self.seen_prefixes.append((g[0, :seq_length].tolist(), float(g_att[0])))
        scores = np.asarray([self.score_fn(g[0], seq_length)], dtype=float)
        return scores, scores.copy(), scores.copy()

    def _init_ctc(self, features):
        return torch.zeros((1, features.shape[1], self.vocab_size))


class BeamFixedOnly(SearchSequence):
    """Isolate the upstream output-copy defect after fixing beam ownership."""
    _beam_search = CorrectedSearchSequence._beam_search


def toy(search_class, score_fn, max_len=8):
    instance_class = type("Toy" + search_class.__name__, (ScoreFixture, search_class), {})
    instance = instance_class(SimpleNamespace(decoder=SimpleNamespace(), embedder=None),
                              vocab_size=7, method="hybrid", max_len=max_len,
                              ctc_rate=0.6, mp_num=1)
    instance.score_fn = score_fn
    instance.seen_prefixes = []
    return instance


def prefix(tokens, max_len=8, attention=0):
    ids = torch.zeros(max_len, dtype=torch.long)
    ids[:len(tokens)] = torch.tensor(tokens)
    return (ids, 0, len(tokens) - 1, attention, 0)


def one_step(search, prefixes=None, length=1, beam_size=2):
    return search._beam_search(
        torch.zeros(1, 3, 2), [prefixes or [prefix([1], search.max_len)]],
        length, 1, beam_size, torch.zeros(1, 3, 7), "cpu")


def run_search(search, **kwargs):
    ids = torch.full((1, search.max_len), 99, dtype=torch.long)
    lengths = torch.zeros(1, dtype=torch.long)
    search._search(torch.zeros(1, 3, 2), device="cpu", batch_idx=0,
                   batch_size=1, shared_outputs=[ids, lengths], **kwargs)
    return ids, lengths


def ordered_scores(_prefix, _length):
    return [-100, -100, 5, 9, 8, 7, 6]


class CorrectedBeamTests(unittest.TestCase):
    def test_upstream_reproduces_changed_eos_and_duplicate_top_two(self):
        endings, beams = one_step(toy(SearchSequence, ordered_scores))
        # These are observed failures of the same desired invariants tested below.
        self.assertNotEqual(int(endings[0][0][0][1]), 2)
        self.assertEqual([int(h[0][1]) for h in beams[0]], [3, 3])
        self.assertEqual(endings[0][0][0].data_ptr(), beams[0][0][0].data_ptr())

    def test_eos_and_top_two_have_distinct_owned_sequences(self):
        original = prefix([1])
        original_copy = original[0].clone()
        endings, beams = one_step(toy(CorrectedSearchSequence, ordered_scores), [original])
        self.assertEqual(int(endings[0][0][0][1]), 2)
        self.assertEqual([int(h[0][1]) for h in beams[0]], [3, 4])
        self.assertEqual([h[1] for h in beams[0]], [9, 8])
        sequences = [endings[0][0][0], beams[0][0][0], beams[0][1][0], original[0]]
        self.assertEqual(len({item.data_ptr() for item in sequences}), 4)
        beams[0][0][0][1] = 6
        self.assertEqual(int(endings[0][0][0][1]), 2)
        self.assertEqual(int(beams[0][1][0][1]), 4)
        torch.testing.assert_close(original[0], original_copy)

    def test_global_top_two_are_ranked_across_multiple_prefixes(self):
        def scores(ids, _length):
            return ([-100, -100, -20, -10, 7, 3, 9] if int(ids[1]) == 4
                    else [-100, -100, -21, 8, 4, 10, 1])
        search = toy(CorrectedSearchSequence, scores)
        originals = [prefix([1, 4], attention=13), prefix([1, 5], attention=23)]
        endings, beams = one_step(search, originals, length=2)
        self.assertEqual([h[0][:3].tolist() for h in beams[0]], [[1, 5, 5], [1, 4, 6]])
        self.assertEqual([h[1] for h in beams[0]], [10, 9])
        self.assertEqual([h[0][:3].tolist() for h in endings[0]], [[1, 4, 2], [1, 5, 2]])
        self.assertEqual(search.seen_prefixes, [([1, 4], 13.0), ([1, 5], 23.0)])
        self.assertEqual(len(originals), 2)

    def test_stable_ties_and_blank_policy_are_explicit(self):
        search = toy(CorrectedSearchSequence, lambda *_: [-100, -100, 5, 9, 9, 9, 9])
        _endings, beams = one_step(search)
        # Blank/unk ID 3 is intentionally still eligible, exactly as upstream.
        self.assertEqual([int(h[0][1]) for h in beams[0]], [3, 4])

    def test_model_scoring_and_ctc_prefix_initialization_are_inherited(self):
        self.assertIs(CorrectedSearchSequence.get_scores, SearchSequence.get_scores)
        self.assertIs(CorrectedSearchSequence._init_ctc, SearchSequence._init_ctc)

    def test_early_end_copies_only_selected_sequence_and_eos(self):
        def scores(_ids, length):
            return [-100, -100, 50, 0, -1, -2, -3] if length == 1 else [-100] * 7
        search = toy(CorrectedSearchSequence, scores)
        ids, lengths = run_search(search, M_end=1, D_end=-14)
        self.assertEqual(lengths.tolist(), [1])
        self.assertEqual(ids.tolist(), [[2, 0, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(len(search.seen_prefixes), 2)

    def test_upstream_final_copy_drops_last_iteration_eos(self):
        score_fn = lambda _ids, length: [-100, -100, length, 10, 9, 8, 7]
        ids, lengths = run_search(toy(BeamFixedOnly, score_fn, max_len=4), D_end=-14, M_end=16)
        self.assertEqual(lengths.tolist(), [3])
        self.assertNotEqual(int(ids[0, int(lengths[0]) - 1]), 2)

    def test_max_length_includes_eos_and_excludes_padding(self):
        score_fn = lambda _ids, length: [-100, -100, length, 10, 9, 8, 7]
        search = toy(CorrectedSearchSequence, score_fn, max_len=4)
        ids, lengths = search.decode_features(torch.zeros(1, 3, 2))
        self.assertEqual(lengths.tolist(), [3])
        self.assertEqual(ids.tolist(), [[3, 3, 2, 0]])
        self.assertEqual(ids.device, lengths.device)
        self.assertEqual(ids.device.type, "cpu")
        selected = ids[0, :int(lengths[0])]
        self.assertEqual(int((selected == 2).sum()), 1)
        self.assertFalse(bool((selected == 0).any()))

    def test_unsupported_batches_and_methods_fail_explicitly(self):
        search = toy(CorrectedSearchSequence, ordered_scores)
        with self.assertRaisesRegex(ValueError, "shape"):
            search.decode_features(torch.zeros(2, 3, 2))
        search.method = "att"
        with self.assertRaisesRegex(ValueError, "hybrid"):
            search.decode_features(torch.zeros(1, 3, 2))


if __name__ == "__main__":
    unittest.main()
