"""Apply narrowly scoped user transcript evidence and package unapproved spans."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

from ctc_alignment import forced_align
from segment_geometry import words_and_phrases, tokenizer
from transcript_audit import comparison_text, save, sha

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


def read(path: Path) -> Any:
    """Read bound local metadata."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def main() -> None:
    """Re-align two supported edits from saved probabilities, without neural calls."""
    try:
        media_path = BASE / "results/media_manifest.json"
        align_path = BASE / "results/alignment_r1/alignment_results.json"
        asr_path = BASE / "results/asr_results.json"
        evidence_path = BASE / "user_evidence/001_transcript_answers.json"
        media, align, asr, evidence = map(read, (media_path, align_path, asr_path, evidence_path))
        responses = {row["id"]: row for row in evidence["responses"]}
        old_align = {row["id"]: row for row in align["records"]}
        asr_by_id = {row["id"]: row for row in asr["records"]}
        vocab = read(ROOT / "models/alignment_audit/wav2vec2-large-xlsr-korean/vocab.json")
        visual = tokenizer()
        policy = {"schema": "offline-alignment-canonical-policy-v1",
                  "sources": {str(path): sha(path) for path in (media_path, align_path, asr_path, evidence_path)},
                  "execution_code": {name: sha(BASE / name) for name in
                                     ("compile_review_r2.py", "segment_geometry.py", "ctc_alignment.py", "transcript_audit.py")},
                  "raw_source_text_preserved": True, "frame_rounding": "round(Fraction(integer_duration_samples*25,16000)); half-even", "changed_transcripts": [],
                  "new_neural_calls": 0, "full_human_approvals_added": 0, "word_boundary_approvals_added": 0}
        records, reviews = [], []
        for original in media["records"]:
            row = copy.deepcopy(original)
            item = copy.deepcopy(old_align[row["id"]])
            response = responses.get(row["id"])
            old_text = row["text"]
            if response and response["change_required"]:
                if old_text.count(response["original_phrase"]) != 1:
                    raise ValueError("Ambiguous phrase replacement")
                row["text"] = old_text.replace(response["original_phrase"], response["canonical_phrase"])
                row["eojeols"] = row["text"].split()
                posterior = item["posterior"]
                if sha(Path(posterior["path"])) != posterior["sha256"]:
                    raise ValueError("Saved acoustic probabilities changed")
                with np.load(posterior["path"], allow_pickle=False) as archive:
                    probs = archive["log_probs"].astype(np.float64)
                target_chars = row["text"].replace(" ", "|")
                if any(ch not in vocab for ch in target_chars):
                    raise ValueError("User-supported edit has an unsupported acoustic token")
                item["text"] = row["text"]
                item["alignment"] = forced_align(probs, [vocab[ch] for ch in target_chars], vocab["[PAD]"])
                item["words"], item["phrases"] = words_and_phrases(row, item["alignment"], probs, 320, 400, visual)
                item["greedy_no_space_match"] = comparison_text(item["acoustic_greedy_text"]).replace(" ", "") == row["text"].replace(" ", "")
                policy["changed_transcripts"].append({"id": row["id"], "original_text": old_text,
                                                       "canonical_text": row["text"], "evidence_sha256": sha(evidence_path)})
            if item["alignment"] is not None:
                with np.load(item["posterior"]["path"], allow_pickle=False) as archive:
                    probs = archive["log_probs"].astype(np.float64)
                if sha(Path(item["posterior"]["path"])) != item["posterior"]["sha256"]:
                    raise ValueError("Changed posterior in geometry revision")
                item["words"], item["phrases"] = words_and_phrases(row, item["alignment"], probs, 320, 400, visual)
            item["original_text"] = old_text
            item["source_text"] = row["source_text"]
            item["canonical_text"] = row["text"]
            item["partial_user_expression_confirmation"] = response
            item["original_neural_elapsed_seconds"] = item.pop("elapsed_seconds")
            item["canonical_revision_neural_calls"] = 0
            item["source_alignment_results_sha256"] = sha(align_path)
            item["human_transcript_approved"] = False
            item["training_ready"] = False
            records.append(item)
            transcriptions = asr_by_id[row["id"]]["transcriptions"]
            comparisons = {key: comparison_text(value["text"]).replace(" ", "") == row["text"].replace(" ", "")
                           for key, value in transcriptions.items()}
            item["canonical_asr_no_space_matches"] = comparisons
            flags = ["어절 음성 경계와 입술 문맥은 아직 검수 전입니다."]
            if response:
                flags.append("사용자가 해당 표현만 확인했습니다. 문장 전체·경계 승인은 아닙니다.")
            if not all(comparisons.values()):
                flags.append("자동 전사와 차이가 있습니다. 사용자 표현 확인 근거와 함께 보세요.")
            if row["context_contains_neighbor_label_interval"]:
                flags.append("문맥에 이웃 문장의 라벨 구간이 포함됩니다. 정렬에는 본문 음성만 사용했습니다.")
            if item["oov_characters"]:
                flags.append("정렬 모델 사전 미지원 음절: " + ", ".join(item["oov_characters"]) + ". 문장 전체 자동 정렬 보류.")
            words = []
            for index, surface in enumerate(row["text"].split()):
                if item["words"]:
                    word = item["words"][index]
                    words.append({"index": index, "surface": surface,
                                  "start_in_context": word["start_in_context"], "end_in_context": word["end_in_context"],
                                  "flags": word["flags"], "proposed_frames": word["proposed_frames_25fps"],
                                  "minimum_ctc_frames": word["visual_ctc_min_frames"]})
                else:
                    words.append({"index": index, "surface": surface, "start_in_context": None,
                                  "end_in_context": None, "flags": ["unsupported_sentence_acoustic_vocabulary"]})
            context_origin = row["context_audio"]["actual_source_start_seconds"]
            reviews.append({"id": row["id"], "text": row["text"], "source_text": row["source_text"],
                            "context_audio_url": f"/media/{row['id']}/context.wav",
                            "context_video_url": f"/media/{row['id']}/context_video.mp4",
                            "roi_video_url": f"/media/{row['id']}/context_lips.mp4",
                            "sentence_start_in_context": row["sentence_audio"]["actual_source_start_seconds"] - context_origin,
                            "sentence_end_in_context": row["sentence_audio"]["actual_source_end_seconds"] - context_origin,
                            "asr_text": transcriptions["sentence_audio"]["text"],
                            "asr_match": comparisons["sentence_audio"], "automatic_flags": flags, "words": words})
        policy_path = BASE / "canonical_policy_r2.json"
        save(policy_path, policy)
        canonical_path = BASE / "results/revision2/canonical_alignment_results.json"
        save(canonical_path, {"schema": "offline-alignment-canonical-results-v1", "policy_sha256": sha(policy_path),
             "status": "automatic_candidates_pending_boundary_review", "records": records,
             "new_neural_calls": 0, "training_ready": False})
        save(BASE / "results/revision2/review_manifest.json", {"schema": "offline-alignment-review-manifest-v1",
             "canonical_results_sha256": sha(canonical_path), "records": reviews,
             "default_decision": "unanswered", "automatic_approval": False})
        words = [word for row in records for word in row["words"]]
        phrases = [phrase for row in records for phrase in row["phrases"]]
        stats = {"records": len(records), "speakers": len({row["speaker_id"] for row in media["records"]}),
                 "full_transcript_words": sum(len(row["canonical_text"].split()) for row in records),
                 "aligned_sentences": sum(bool(row["words"]) for row in records),
                 "oov_held_sentences": sum(bool(row["oov_characters"]) for row in records),
                 "automatic_word_proposals": len(words),
                 "ctc_feasible_word_proposals": sum(word["ctc_feasible_proposal"] for word in words),
                 "zero_video_frame_word_proposals": sum(word["proposed_frames_25fps"] == 0 for word in words),
                 "automatic_phrase_proposals": len(phrases),
                 "ctc_feasible_phrase_proposals": sum(phrase["ctc_feasible_proposal"] for phrase in phrases),
                 "word_flag_counts": dict(Counter(flag for word in words for flag in word["flags"])),
                 "asr_both_conditions_no_space_match_after_user_edits": sum(all(row["canonical_asr_no_space_matches"].values()) for row in records),
                 "partial_user_expression_confirmations": len(responses), "canonical_transcripts_changed": len(policy["changed_transcripts"]),
                 "full_human_transcript_approvals": 0, "human_word_boundary_approvals": 0,
                 "measured_boundary_error_ms": None, "approved_word_clips": 0,
                 "new_visual_model_training_updates": 0, "new_visual_model_inference": 0,
                 "whisper_calls": 32, "wav2vec2_calls_including_failed_first_attempt": 17,
                 "canonical_revision_neural_calls": 0, "service_weights_changed": False}
        save(BASE / "results/stage2_summary_r2.json", {"schema": "offline-alignment-stage2-summary-v1",
             "status": "automated_pilot_complete_word_boundary_review_pending", "counts": stats,
             "sources": {str(path): sha(path) for path in (media_path, align_path, asr_path, evidence_path, policy_path, canonical_path)},
             "limits": ["ASR agreement and valid CTC paths are not human transcript/boundary approval",
                        "CTC-feasible counts are proposals, not generated or validated training clips",
                        "Acoustic spikes are not ground-truth articulatory boundaries",
                        "Word boundary error cannot be measured without manually established reference intervals"],
             "training_ready": False})
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception as exc:
        raise RuntimeError("Review compilation failed; no labels approved") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
