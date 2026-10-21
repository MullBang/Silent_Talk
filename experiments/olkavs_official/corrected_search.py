"""Minimal beam bookkeeping adapter for the pinned, unmodified OLKAVS code.

Candidate ownership, top-K selection and final output copying are corrected. The published
get_scores, CTC prefix scores, weighting and end-detection heuristic remain in
use. In particular, token 3 (also the CTC blank) remains a permitted continuation;
changing that policy is a separate experiment. The supported integration is
hybrid search with batch size one, because upstream multi-item active indexing
and attention-only dispatch have independent issues outside this experiment.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

UPSTREAM = Path(__file__).resolve().parent / "upstream"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

from avsr.utils.search import SearchSequence, end_detect  # noqa: E402


class CorrectedSearchSequence(SearchSequence):
    """Use distinct immutable-at-creation hypothesis tensors during beam search."""

    def _beam_search(
        self, features, current_hypothesis, length, batch_size, beam_size,
        ctc_output=None, device="cpu",
    ):
        if batch_size != 1 or len(current_hypothesis) != 1:
            raise ValueError("The audited corrected adapter supports batch size one")
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        if not 1 <= length < self.max_len:
            raise ValueError("length must leave room for the next token")
        if not current_hypothesis[0]:
            raise ValueError("At least one active prefix is required")

        completed = []
        continuations = []
        unused_tokens = {self.pad_id, self.eos_id, self.sos_id}
        allowed_tokens = [t for t in range(self.vocab_size) if t not in unused_tokens]
        if not allowed_tokens:
            raise ValueError("No continuation tokens are permitted")

        # Do not pop caller-owned prefixes or modify their tensor storage.
        for prefix, _score, _length, prefix_attention, _ctc in current_hypothesis[0]:
            scores, ctc_scores, att_scores = self.get_scores(
                features=features,
                seq_length=length,
                g=prefix.unsqueeze(0),
                g_att=np.array([prefix_attention], dtype=float),
                ctc_output=ctc_output,
            )

            def candidate(token):
                sequence = prefix.detach().clone()
                sequence[length] = token
                return (sequence, float(scores[0, token]), length,
                        float(att_scores[0, token]), float(ctc_scores[0, token]))

            completed.append(candidate(self.eos_id))
            continuations.extend(candidate(token) for token in allowed_tokens)

        # Stable ties follow prefix order, then ascending token ID. Each distinct
        # candidate is visited once; an early high scorer cannot fill K slots.
        continuations.sort(key=lambda item: item[1], reverse=True)
        return [completed], [continuations[:beam_size]]

    def _search(self, features=None, beam_size=1, D_end=-4, M_end=12,
                device="cpu", batch_idx=None, batch_size=1,
                shared_outputs=None, *args, **kwargs):
        if batch_size != 1 or features is None or features.shape[0] != 1:
            raise ValueError("The audited corrected adapter supports batch size one")
        if self.method != "hybrid":
            raise ValueError("This adapter is audited for hybrid search only")
        if self.max_len < 2:
            raise ValueError("max_len must allow SOS and EOS")
        if shared_outputs is None or batch_idx is None:
            raise ValueError("Output buffers and batch index are required")

        initial = torch.full((self.max_len,), self.pad_id, dtype=torch.long,
                             device=device)
        initial[0] = self.sos_id
        active = [[(initial, 0, 0, 0, 0)]]
        completed = []
        ctc_output = self._init_ctc(features)
        for length in range(1, self.max_len):
            endings, active = self._beam_search(
                features, active, length, 1, beam_size, ctc_output, device)
            # Preserve the original per-length EOS selection and stopping rule.
            completed.append(max(endings[0], key=lambda item: item[1]))
            if end_detect(completed, length, M_end, D_end, progressive=active[0]):
                break

        sequence, _score, best_length, _attention, _ctc = max(
            completed, key=lambda item: item[1])
        # best_length is the EOS index in a sequence whose SOS is at index 0.
        # Thus this slice has exactly best_length tokens, including the EOS.
        selected = sequence[1:best_length + 1].unsqueeze(0)
        lengths = torch.tensor([best_length], dtype=torch.long, device=device)
        cur_idx = batch_idx  # The public adapter intentionally only accepts B=1.
        shared_outputs[0][cur_idx].fill_(self.pad_id)
        shared_outputs[0][cur_idx, :best_length] = selected[0]
        shared_outputs[1][cur_idx] = best_length
        return selected, lengths

    def decode_features(self, features, beam_size=1, D_end=-14, M_end=16):
        """Return (1 x max_len IDs, 1 length), both on ``features.device``.

        Output excludes SOS, includes exactly one terminating EOS and has PAD
        only beyond the reported length. No encoder or audio path is invoked.
        """
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("Expected visual features with shape 1 x T x D")
        ids = torch.full((1, self.max_len), self.pad_id, dtype=torch.long,
                         device=features.device)
        lengths = torch.zeros(1, dtype=torch.long, device=features.device)
        self._search(features, beam_size, D_end, M_end, features.device,
                     0, 1, [ids, lengths])
        return ids, lengths
