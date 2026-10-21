"""Independently verify r2 phrase PCM slices and selected review-video frames."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np

from audit_preflight import HERE, bind, check_binding, read_json
from audit_media import FFMPEG, pcm_file, ratio


def decode(path: Path) -> list[np.ndarray]:
    """Decode all review frames, requiring explicit ordinal25fps timestamps."""
    capture = cv2.VideoCapture(str(path))
    values = []
    try:
        assert capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS) - 25) < 1e-9
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            good, frame = capture.read()
            if not good:
                break
            assert abs(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - len(values) / 25) < 1e-6
            values.append(frame)
        assert len(values) == expected
    finally:
        capture.release()
    return values


def inspect() -> dict[str, Any]:
    base = HERE.parent
    path = base / "phrase_media_r2/manifest.json"
    result = read_json(path)
    check_binding(result["source_plan"])
    plan = read_json(Path(result["source_plan"]["path"]))
    for source in plan["bindings"]:
        check_binding(source)
    alignment = read_json(base / "alignment/execution/alignment_results.json")["records"]
    assert result["records_count"] == len(result["records"]) == len(plan["records"]) == 8
    assert result["new_tar_payload_bytes"] == result["new_neural_calls"] == result["new_human_approvals"] == result["training_updates"] == 0
    reports = []
    total_samples = total_video_frames = 0
    for actual, source, aligned in zip(result["records"], plan["records"], alignment, strict=True):
        qid = actual["review_id"]
        assert qid == source["review_id"] == aligned["review_id"]
        assert actual["parent_id"] == source["parent_id"] == aligned["parent_id"]
        assert actual["candidate_id"] == source["candidate_id"] == aligned["candidate_id"]
        assert actual["text"] == aligned["selected_phrase"]["text"]
        assert actual["aligned_proposal"] == source["aligned_proposal"] == aligned["selected_phrase"]
        assert actual["source_plan"] == result["source_plan"]
        assert read_json(base / "phrase_media_r2" / qid / "record.json") == actual
        parent = source["parent"]
        original = source["original_media"]
        assert read_json(Path(parent["media_record"]["path"])) == original
        _, context_pcm = pcm_file(Path(original["context_audio"]["path"]))
        start = ratio(aligned["selected_phrase"]["wav_start_seconds_exact"])
        end = ratio(aligned["selected_phrase"]["wav_end_seconds_exact"])
        origin = Fraction(original["context_audio"]["source_first_sample"], 48000)
        left, right = (start - origin) * 16000, (end - origin) * 16000
        assert left.denominator == right.denominator == 1
        left, right = int(left), int(right)
        assert source["context_start_sample_16k"] == left and source["context_end_sample_16k"] == right
        checked_pcm = {}
        for role, first, stop in [("phrase_audio", left, right), ("phrase_context_audio", left - 3200, right + 3200)]:
            spec = actual[role]
            check_binding(spec)
            assert spec["source_context_binding"] == bind(Path(original["context_audio"]["path"]))
            assert spec["source_context_first_sample_16k"] == first and spec["source_context_end_sample_16k"] == stop
            info, pcm = pcm_file(Path(spec["path"]))
            assert info == (16000, 1, 2, stop - first)
            assert pcm == context_pcm[first * 2:stop * 2]
            assert hashlib.sha256(pcm).hexdigest() == spec["pcm_sha256"]
            assert spec["source_original_wav_first_sample_48k"] == original["context_audio"]["source_first_sample"] + 3 * first
            assert spec["source_original_wav_end_sample_48k"] == original["context_audio"]["source_first_sample"] + 3 * stop
            assert spec["new_resampling"] is False and spec["padding"] is False and spec["fade"] is False
            checked_pcm[role] = pcm
        assert checked_pcm["phrase_context_audio"][3200 * 2:(3200 + right - left) * 2] == checked_pcm["phrase_audio"]
        _, parent_pcm = pcm_file(Path(parent["audio"]["path"]))
        first = aligned["selected_phrase"]["local_audio_start_sample_16k"]
        stop = aligned["selected_phrase"]["local_audio_end_sample_16k"]
        difference = np.frombuffer(checked_pcm["phrase_audio"], dtype="<i2").astype(np.int32) - np.frombuffer(parent_pcm[first * 2:stop * 2], dtype="<i2").astype(np.int32)
        comparison = actual["parent_vs_context_pcm_comparison"]
        assert comparison["unequal_samples"] == int(np.count_nonzero(difference)) == 0
        assert comparison["maximum_absolute_int16_difference"] == int(np.max(np.abs(difference))) == 0
        display_origin = ratio(original["video"]["display_video_start"])
        offset = Fraction(original["offset_native_samples"], 48000)
        source_frames = decode(Path(original["video"]["roi_preview"]["path"]))
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in source_frames]
        video_results = []
        for role, semantic_start, semantic_end in [("phrase", start + offset, end + offset), ("phrase_context", start + offset - Fraction(1, 5), end + offset + Fraction(1, 5))]:
            spec = actual["muted_video"][role]
            check_binding(spec)
            first, stop = math.floor((semantic_start - display_origin) * 25), math.ceil((semantic_end - display_origin) * 25)
            assert spec["source_first_display_frame"] == first and spec["source_stop_display_frame"] == stop
            assert spec["source_decoded_selected_bgr_sha256"] == hashes[first:stop]
            frames = decode(Path(spec["path"]))
            assert len(frames) == spec["frames"] == stop - first
            assert all(frame.shape == (96, 96, 3) for frame in frames)
            total_video_frames += len(frames)
            probe = subprocess.run([str(FFMPEG), "-nostdin", "-v", "error", "-i", spec["path"], "-map", "0:a:0", "-f", "null", "-"], capture_output=True, timeout=30, check=False)
            assert probe.returncode != 0 and b"matches no streams" in probe.stderr
            window = spec["clock_window"]
            assert ratio(window["semantic_video_start"]) == semantic_start
            assert ratio(window["semantic_video_end"]) == semantic_end
            assert ratio(window["preview_display_video_start"]) == display_origin + Fraction(first, 25)
            assert ratio(window["preview_display_video_end"]) == display_origin + Fraction(stop, 25)
            assert 0 <= ratio(window["preview_before_semantic_seconds"]) < Fraction(1, 25)
            assert 0 <= ratio(window["preview_after_semantic_seconds"]) < Fraction(1, 25)
            assert spec["training_frame_count"] is None
            video_results.append({"role": role, "decoded_frames": len(frames), "selected_source_frame_hashes_exact": True, "no_audio_track": True})
        assert actual["actual_training_frame_count"] is None and actual["ctc_feasible"] is None
        assert actual["phrase_audio_boundary_approved"] is False and actual["phrase_visual_boundary_approved"] is False
        assert actual["training_ready"] is False and actual["actual_split"] is None
        total_samples += actual["phrase_audio"]["samples"]
        reports.append({"review_id": qid, "parent_id": actual["parent_id"], "phrase": actual["text"],
                        "phrase_samples_16k": actual["phrase_audio"]["samples"],
                        "two_context_pcm_slices_exact": True, "aligned_parent_phrase_pcm_equal": True,
                        "videos": video_results, "phrase_boundary_approval_added": False})
    failed = read_json(base / "media_prepare/phrase_preview_failure.json")
    for item in failed["first_attempt_outputs"] + [failed["first_plan"], failed["first_code"]]:
        check_binding(item)
    for name in ["phrase.wav", "phrase_context.wav"]:
        assert (base / "phrase_media/Q01" / name).read_bytes() == (base / "phrase_media_r2/Q01" / name).read_bytes()
    return {"schema": "offline-stage5f-independent-r2-phrase-media-audit-v1", "status": "passed",
            "manifest": bind(path), "records": reports,
            "phrase_pcm_samples_total": total_samples, "phrase_audio_seconds_total": total_samples / 16000,
            "exact_pcm_slices_verified": 16, "muted_videos_verified": 16,
            "decoded_video_frames_verified": total_video_frames,
            "selected_source_review_frame_hashes_verified": total_video_frames,
            "first_failed_attempt_preserved": True, "first_attempt_Q01_audio_unchanged_in_r2": True,
            "additional_tar_bytes_read": 0, "new_model_calls": 0, "training_ready": False,
            "limit": "Selected pixels are decoded review-preview pixels, not new original-TAR training inputs. Geometric and byte correctness do not approve phrase speech or articulatory boundaries."}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "phrase_media_r2_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ["status", "exact_pcm_slices_verified", "muted_videos_verified", "decoded_video_frames_verified", "phrase_audio_seconds_total"]}))
