"""Bind the completed eight-phrase review to a metadata-only extraction plan."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
MEDIA = BASE.parent / "offline_alignment_step02_20261012" / "media"


def read(path: Path) -> dict[str, Any]:
    """Read a JSON object without changing its source."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def sha(path: Path) -> str:
    """Hash a small local evidence artifact."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def rational(value: Fraction) -> dict[str, int | float]:
    """Retain exact rational arithmetic and a readable seconds value."""
    try:
        return {"numerator": value.numerator, "denominator": value.denominator, "seconds": float(value)}
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invalid time fraction") from exc


def main() -> None:
    """Prepare expected target times; never read TARs or create training pixels."""
    try:
        for item in read(BASE / "results/completion_manifest.json")["artifacts"]:
            if sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen stage2d artifact changed: {item['path']}")
        manifest_path = BASE / "review_manifest.json"
        manifest = read(manifest_path)
        evidence_path = BASE / "user_evidence/004_phrase_visual_answer.json"
        evidence = read(evidence_path)
        parent_evidence = read(BASE / "results/current_parent_evidence.json")
        parent_index = {row["id"]: row for row in parent_evidence["records"]}
        if evidence["manifest_sha256"] != sha(manifest_path):
            raise ValueError("Visual answer manifest mismatch")
        expected_targets = [dict(id=row["id"], rank=row["rank"], phrase=row["phrase"],
                                 visual_review_decision="no_obvious_cut_reported") for row in manifest["records"]]
        if evidence["targets"] != expected_targets or evidence["answer_verbatim"] != "8개 모두 뚜렷한 잘림은 보이지 않음":
            raise ValueError("Visual answer does not match these eight exact windows")
        tokenizer_path = ROOT / "backend/models/tokenizer.py"
        spec = importlib.util.spec_from_file_location("extraction_tokenizer", tokenizer_path)
        if spec is None or spec.loader is None:
            raise ValueError("Tokenizer unavailable")
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        reference_path = BASE.parent / "offline_boundary_review_step02c_20261013/results/current_audio_confirmed_phrases.json"
        reference = {row["parent_id"]: row for row in read(reference_path)["records"]}
        source_hashes = {str(path): sha(path) for path in (manifest_path, evidence_path, tokenizer_path, reference_path,
                                                          BASE / "results/current_parent_evidence.json")}
        records = []
        for row in manifest["records"]:
            parent = parent_index[row["id"]]
            if parent["parent_audio_evidence"]["transcript"] != "correct" or parent["parent_audio_evidence"]["boundary"] != "complete":
                raise ValueError("Parent audio not confirmed")
            record_path = MEDIA / row["id"] / "record.json"
            source = read(record_path)
            source_hashes[str(record_path)] = sha(record_path)
            context = source["context_audio"]
            if not context["source_sample_origin_verified"]:
                raise ValueError("Unknown source audio origin")
            source_rate = context["source_sample_rate"]
            origin = Fraction(context["source_first_sample"], source_rate)
            offset = Fraction(str(source["offset_seconds"])).limit_denominator(source_rate)
            if abs(float(offset) - source["offset_seconds"]) > 1e-12:
                raise ValueError("Cannot reconstruct frozen audio/video offset")
            native_fps = Fraction(str(source["video"]["native_fps"])).limit_denominator(100000)
            if abs(float(native_fps) - source["video"]["native_fps"]) > 1e-12:
                raise ValueError("Cannot represent recorded native FPS")
            segment = row["phrase"]
            left, right = segment["start_sample_16k"], segment["end_sample_16k"]
            if not row["sentence_start_sample_16k"] <= left < right <= row["sentence_end_sample_16k"]:
                raise ValueError("Selected phrase lies outside its reviewed parent")
            duration = Fraction(right - left, 16000)
            frames = round(duration * 25)
            tokens = codec.encode_text(segment["text"])
            minimum = codec.ctc_min_input_len(tokens)
            if frames < minimum or frames != reference[row["id"]]["frames_25fps"]:
                raise ValueError("CTC length failure or historical geometry mismatch")
            wav_start, wav_end = origin + Fraction(left, 16000), origin + Fraction(right, 16000)
            video_start, video_end = wav_start + offset, wav_end + offset
            target_times = [video_start + Fraction(index, 25) for index in range(frames)]
            expected_indices = [round(time * native_fps) for time in target_times]
            if len(set(expected_indices)) != frames or not all(0 <= index < source["video"]["native_frame_count"] for index in expected_indices):
                raise ValueError("Repeated or out-of-bounds expected source frame")
            if not all(video_start <= time < video_end for time in target_times):
                raise ValueError("Frame target outside selected range")
            records.append({
                "rank": row["rank"], "parent_id": row["id"], "speaker_id": row["speaker_id"],
                "recording_group": source["source_stem"], "parent_text": row["text"], "phrase": segment,
                "source_members": source["source_members"], "context_source_first_sample": context["source_first_sample"],
                "context_source_sample_rate": source_rate, "frozen_offset": rational(offset),
                "source_wav_start": rational(wav_start), "source_wav_end": rational(wav_end),
                "source_video_start": rational(video_start), "source_video_end": rational(video_end),
                "duration": rational(duration), "target_frame_times": [rational(time) for time in target_times],
                "historical_native_fps": {"numerator": native_fps.numerator, "denominator": native_fps.denominator},
                "expected_native_indices_if_prior_cfr_contract_holds": expected_indices,
                "grid_phase_differs_from_context_preview": True,
                "expected_last_native_pts_to_selected_end_seconds": float(video_end - Fraction(expected_indices[-1], 1) / native_fps),
                "native_indices_verified_this_run": False, "frames_25fps": frames, "label": tokens,
                "ctc_min_frames": minimum, "ctc_feasible": True,
                "parent_transcript_and_audio_scope": "actual parent audio replies 001-003",
                "phrase_audio_scope": "prior actual current phrase confirmation; unchanged interval",
                "phrase_visual_scope": "actual qualitative no-obvious-cut report for this current interval",
                "split": None, "training_ready": False, "original_pixels_extracted": False,
            })
        plan = {
            "schema": "offline-reviewed-eight-extraction-plan-v1", "status": "metadata_prepared_original_extraction_and_split_pending",
            "reviewed_subset_count": 8, "original_pilot_parent_count": 16, "original_pilot_complete": False,
            "records": records, "total_expected_frames": sum(row["frames_25fps"] for row in records),
            "expected_rgb_uint8_bytes_before_compression": sum(row["frames_25fps"] for row in records) * 96 * 96 * 3,
            "source_hashes": source_hashes,
            "policy": {
                "output_fps": 25, "frame_count": "round-half-even(exact integer-sample duration * 25)",
                "sampling": "At source decode, verify native CFR/PTS and select the nearest actual original frame to each target timestamp; stored indices are expectations, not reverified source data.",
                "time_mapping": "Verified native WAV first sample + selected context sample / 16000 + frozen WAV-to-video offset exactly once.",
                "roi_size": [96, 96], "roi_margin": 0.20, "color": "RGB",
                "bbox_coordinate_order": "yxyx; convert to xyxy, do not rotate image",
                "bbox_lookup": "selected actual original native frame index",
                "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
                "variable_temporal_length": True, "last_frame_padding_only_for_batches": True,
                "raw_clip_padding_allowed": False, "model_audio_input": False,
                "tokenizer_version": codec.TOKENIZER_VERSION, "normalization_version": codec.NORMALIZATION_VERSION,
                "vocab_sha256": codec.VOCAB_SHA256, "num_classes": codec.NUM_CLASSES,
                "extra_context_added_to_selected_windows": False,
                "duplicate_parent_texts_must_not_cross_an_unseen_sentence_split": ["C293", "C312"],
            },
            "gates_before_training": [
                "Assign parent-recording/view/duplicate-aware train/development roles before training cache generation.",
                "At original extraction verify archive/member identity and hashes, native timing, chosen frames, bbox range, ROI pixels, count and CTC feasibility.",
                "Compare first/last selected original frames against reviewed context frames: a phrase-origin 25fps grid differs in phase from the preview grid. The current qualitative reply does not approve newly sampled pixels. Do not silently add padding or context if terminal coverage needs revision.",
                "Do not use recompressed review MP4 pixels as training inputs.",
                "Keep exact phoneme boundaries unmeasured; this is qualitative review of selected phrases, not forced-aligner error calibration.",
                "Remaining eight original pilot parents are outside this reviewed subset.",
            ],
            "new_tar_reads": 0, "new_training_clips": 0, "training_ready": False,
        }
        target = BASE / "results/extraction_plan.json"
        if target.exists():
            raise ValueError("Refusing to overwrite an existing extraction plan")
        target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"reviewed_subset": len(records), "expected_frames": plan["total_expected_frames"],
                          "raw_rgb_bytes": plan["expected_rgb_uint8_bytes_before_compression"], "training_ready": False}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Reviewed extraction plan preparation failed") from exc


if __name__ == "__main__":
    main()
