"""Verify saved annotation posteriors, independent Viterbi paths and exact clocks."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any
import wave

import numpy as np

from audit_preflight import HERE, bind, check_binding, read_json
from independent_viterbi import align, collapse


def rational(value: dict[str, int]) -> Fraction:
    return Fraction(value["numerator"], value["denominator"])


def inspect() -> dict[str, Any]:
    base = HERE.parent
    plan_path = base / "alignment/execution_plan_after_parent_review.json"
    plan = read_json(plan_path)
    preflight = read_json(HERE / "alignment_plan_preflight.json")
    assert bind(plan_path) == preflight["plan"]
    check_binding(plan["execution_policy"])
    for source in plan["execution_code"]:
        check_binding(source)
    result_path = base / "alignment/execution/alignment_results.json"
    result = read_json(result_path)
    assert result["status"] == "automatic_phrase_proposals_pending_review"
    assert result["neural_calls"] == len(result["records"]) == 8
    assert result["training_updates"] == result["visual_model_calls"] == result["human_approvals_added"] == result["new_training_clips"] == 0
    assert result["training_ready"] is False
    setup = read_json(Path(plan["model_setup"]["path"]))
    model_dir = Path(setup["model_directory"])
    vocab = read_json(model_dir / "vocab.json")
    reverse = {value: key for key, value in vocab.items()}
    blank = vocab["[PAD]"]
    reports = []
    max_norm = max_score = 0.
    total_emissions = 0
    for saved, expected in zip(result["records"], plan["records"], strict=True):
        for key in ["review_id", "parent_id", "candidate_id", "canonical_parent_text"]:
            assert saved[key] == expected[key]
        assert saved["plan"] == bind(plan_path)
        assert saved["audio"] == bind(Path(expected["parent_audio"]["path"]))
        assert saved["parent_review_evidence"] == expected["parent_review_evidence"]
        assert saved["human_approvals_added_by_this_execution"] == 0
        assert saved["actual_split"] is None and saved["training_ready"] is False
        assert saved["greedy_is_independent_human_confirmation"] is False
        per_record_path = base / "alignment/execution" / (saved["review_id"] + "_alignment.json")
        assert read_json(per_record_path) == saved
        with wave.open(expected["parent_audio"]["path"], "rb") as stream:
            assert stream.getnframes() == expected["parent_audio"]["sample_count"]
        check_binding(saved["posterior"])
        with np.load(saved["posterior"]["path"], allow_pickle=False) as stored:
            assert stored.files == ["log_probs"]
            probs = stored["log_probs"].astype(np.float64)
        samples = expected["parent_audio"]["sample_count"]
        frames = (samples - 400) // 320 + 1
        assert probs.shape == (frames, 1205)
        assert list(probs.shape) == saved["posterior_shape"]
        assert np.isfinite(probs).all()
        error = float(np.abs(np.exp(probs).sum(axis=1) - 1).max())
        assert error <= 2e-5
        max_norm = max(max_norm, error)
        total_emissions += frames
        text = expected["canonical_parent_text"]
        targets = [vocab[ch] for ch in text.replace(" ", "|")]
        independently = align(probs, targets, blank)
        actual = saved["alignment"]
        for key in ["path_states", "expanded_tokens", "token_spans"]:
            assert independently[key] == actual[key]
        score_error = abs(independently["best_log_score"] - actual["best_log_score"])
        assert score_error < 1e-10
        max_score = max(max_score, score_error)
        greedy = collapse(probs.argmax(axis=1).tolist(), blank)
        assert saved["acoustic_greedy_ids"] == greedy
        assert saved["acoustic_greedy_text"] == "".join(reverse[i] for i in greedy).replace("|", " ").strip()
        matches = list(re.finditer(r"\S+", text))
        first_char = matches[expected["word_start"]].start()
        final_char = matches[expected["word_end"] - 1].end()
        assert text[first_char:final_char] == expected["canonical_phrase_text"]
        start = independently["token_spans"][first_char]["start_frame"] * 320 + 40
        end = independently["token_spans"][final_char - 1]["end_frame"] * 320 + 40
        assert 0 <= start < end <= samples
        phrase = saved["selected_phrase"]
        assert phrase["char_start"] == first_char and phrase["char_end"] == final_char
        assert phrase["word_start"] == expected["word_start"] and phrase["word_end"] == expected["word_end"]
        assert phrase["text"] == expected["canonical_phrase_text"]
        assert phrase["local_audio_start_sample_16k"] == start
        assert phrase["local_audio_end_sample_16k"] == end
        assert phrase["local_audio_sample_count"] == end - start
        origin = rational(expected["parent_audio"]["source_origin_seconds_exact"])
        offset = rational(expected["video_offset_seconds_exact"])
        assert rational(phrase["wav_start_seconds_exact"]) == origin + Fraction(start, 16000)
        assert rational(phrase["wav_end_seconds_exact"]) == origin + Fraction(end, 16000)
        assert rational(phrase["video_start_seconds_exact"]) == origin + Fraction(start, 16000) + offset
        assert rational(phrase["video_end_seconds_exact"]) == origin + Fraction(end, 16000) + offset
        assert phrase["video_offset_applied_count"] == 1
        assert phrase["frame_count"] is None and phrase["ctc_feasible"] is None
        assert phrase["actual_training_clip_created"] is False and phrase["training_ready"] is False
        assert phrase["human_phrase_audio_boundary_approved"] is False and phrase["human_phrase_visual_boundary_approved"] is False
        assert phrase["phrase_equals_whole_parent_text"] is (phrase["text"] == text)
        assert phrase["parent_review_is_separate_from_phrase_boundary_review"] is True
        reports.append({"review_id": saved["review_id"], "parent_id": saved["parent_id"],
                        "source_result": bind(per_record_path), "posterior": saved["posterior"],
                        "emission_frames": frames, "target_characters_including_spaces": len(targets),
                        "independent_path_exact": True, "independent_token_spans_exact": True,
                        "score_absolute_difference": score_error, "normalization_max_abs_error": error,
                        "phrase": phrase["text"], "phrase_sample_start_16k": start, "phrase_sample_end_16k": end,
                        "proposed_phrase_seconds": (end - start) / 16000,
                        "phrase_equals_whole_parent_text": phrase["phrase_equals_whole_parent_text"],
                        "leading_parent_audio_not_in_proposal_seconds": start / 16000,
                        "trailing_parent_audio_not_in_proposal_seconds": (samples - end) / 16000,
                        "exact_clock_offset_once_verified": True, "phrase_approval_added": False})
    assert not (base / "alignment/execution/failure.json").exists()
    return {"schema": "offline-stage5f-independent-acoustic-results-audit-v1", "status": "passed",
            "plan": bind(plan_path), "results": bind(result_path), "records": reports,
            "saved_posteriors_verified": 8, "emission_frames_verified": total_emissions,
            "independent_scalar_FP64_paths_verified": 8,
            "maximum_score_absolute_difference": max_score,
            "maximum_probability_normalization_error": max_norm,
            "phrase_clock_formula_checks": 8, "producer_DP_imports": 0,
            "new_model_loads": 0, "new_forward_calls": 0, "new_training_updates": 0,
            "phrase_audio_approvals": 0, "phrase_visual_approvals": 0,
            "scope": "The eight saved posteriors were checked without new model calls. The acoustic model and forced transcript produce tentative CTC occupancy spans, not verified speech or mouth boundaries.",
            "training_ready": False}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "alignment_execution_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "saved_posteriors_verified", "emission_frames_verified", "maximum_score_absolute_difference", "maximum_probability_normalization_error"]}))
