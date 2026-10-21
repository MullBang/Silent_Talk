"""Verify Q02's separately preserved +120 ms review candidate from saved PCM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from audit_preflight import HERE, bind, check_binding, read_json
from audit_media import pcm_file, resample_source


def inspect() -> dict[str, Any]:
    base = HERE.parent
    path = base / "media_revisions/Q02_end_plus120ms/record.json"
    revision = read_json(path)
    for binding in revision["source_bindings"]:
        check_binding(binding)
    old_path = base / "media/lip_J_1_F_02_C004_A_009_049/record.json"
    old = read_json(old_path)
    assert revision["review_id"] == old["review_id"] == "Q02"
    assert revision["parent_id"] == old["parent_id"]
    assert revision["parent_text"] == old["parent_text"]
    assert revision["proposed_phrase_unchanged"] == old["proposed_phrase"]
    assert revision["context_native_source"] == old["context_audio"]["native_crop"]
    context_info, context_pcm = pcm_file(Path(revision["context_native_source"]["path"]))
    first, end = revision["context_slice_first_sample_48k"], revision["context_slice_end_sample_48k"]
    assert context_info[:3] == (48000, 2, 2)
    assert 0 <= first < end <= context_info[3]
    expected_pcm = context_pcm[first * 4:end * 4]
    context_origin = old["context_audio"]["source_first_sample"]
    assert revision["context_original_recording_first_sample_48k"] == context_origin
    assert context_origin + first == revision["new_parent_original_start_sample_48k"] == old["parent_native_start_sample"]
    assert context_origin + end == revision["new_parent_original_end_sample_48k"]
    assert revision["new_parent_original_end_sample_48k"] - old["parent_native_end_sample"] == revision["extension_native_samples"] == 5760
    assert revision["extension_seconds"] == .12 and revision["extension_is_candidate_not_measured_boundary_error"]
    audio = revision["sentence_audio"]
    check_binding(audio)
    check_binding(audio["native_crop"])
    native_info, actual_native = pcm_file(Path(audio["native_crop"]["path"]))
    review_info, actual_review = pcm_file(Path(audio["path"]))
    assert native_info == (48000, 2, 2, end - first)
    assert actual_native == expected_pcm
    assert hashlib.sha256(expected_pcm).hexdigest() == audio["native_pcm_sha256"]
    assert actual_review == resample_source(expected_pcm, 2)
    assert review_info == (16000, 1, 2, 49024)
    assert old["sentence_audio"]["samples"] == 47104
    assert 49024 - 47104 == 1920
    _, original_native = pcm_file(Path(old["sentence_audio"]["native_crop"]["path"]))
    assert actual_native[:len(original_native)] == original_native
    assert audio["clock"] == "saved_native_context_relative_samples"
    assert audio["source_first_sample"] == first and audio["source_end_sample_exclusive"] == end
    assert audio["original_recording_first_sample_48k"] == context_origin + first
    assert audio["original_recording_end_sample_48k"] == context_origin + end
    assert audio["offset_applied_to_audio"] is False
    assert not old["context_neighbor_intersections"]
    assert revision["new_user_answer_pending"]
    assert all(revision[k] is False for k in ["parent_transcript_confirmed", "parent_audio_boundary_confirmed", "phrase_boundary_approved", "training_ready"])
    assert revision["new_tar_payload_bytes"] == revision["neural_calls"] == revision["training_updates"] == 0
    return {"schema": "offline-stage5f-independent-Q02-revision-audit-v1", "status": "passed",
            "revision": bind(path), "source_original_record": bind(old_path),
            "native_context_slice_exact": True, "old_native_prefix_preserved": True,
            "separate_16k_resampling_exact": True, "native_extension_samples": 5760,
            "output_extension_samples_16k": 1920, "new_output_sample_count_16k": 49024,
            "source_clock_layers_distinguished": True,
            "alignment_origin_must_use_original_recording_sample": context_origin + first,
            "additional_tar_bytes_read": 0, "new_human_approvals_added": 0,
            "snapshot_pending_does_not_override_later_actual_answer": True,
            "limit": "This verifies the new candidate's technical construction, not whether its final sound is complete. The 120 ms extension is not a measured boundary error."}


if __name__ == "__main__":
    result = inspect()
    with (HERE / "Q02_revision_audit.json").open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "new_samples": result["new_output_sample_count_16k"]}))
