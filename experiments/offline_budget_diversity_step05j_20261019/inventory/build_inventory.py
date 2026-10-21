"""Read-only source selection. No media, model or existing source mutations."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def bind(path):
    blob = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}

def key(text):
    return re.sub(r"\s+", "", text)

def main():
    files = {
        "eligible": ROOT / "experiments/offline_phrase_design_step05e_20261018/data/eligible_sentences.json",
        "protected": ROOT / "experiments/offline_phrase_design_step05e_20261018/data/protected_scope.json",
        "proposal": ROOT / "experiments/offline_phrase_design_step05e_20261018/design/role_proposal.json",
        "previous": ROOT / "experiments/offline_diversity_motion_step05i_20261019/data_plan.json",
        "d_sources": ROOT / "experiments/local_source_step01_20261007/d_source_plan.json",
    }
    data = {name: read(path) for name, path in files.items()}
    previous = data["previous"]
    protected = data["protected"]
    train_stems = {r["parent_id"].rsplit("_", 1)[0] for r in previous["records"] if r["split"] == "train"}
    held_stems = {x.rsplit("_", 1)[0] for x in previous["held_parent_ids"]}
    reserved = data["proposal"]["reserved_recordings"]
    reserved_keys = {r["recording_key"] for r in reserved}
    reserved_videos = {h for r in reserved for h in r["blocked_video_sha256"]}
    reserved_labels = {h for r in reserved for h in r["blocked_label_sha256"]}
    records = {r["id"]: r for r in data["eligible"]["records"]}
    sources = {r["stem"]: r for r in data["d_sources"]["sources"]}
    selected_ids = [
        "lip_J_1_F_02_C017_A_003_003",  # one short candidate from each eligible speaker first
        "lip_J_1_F_02_C032_A_004_006",
        "lip_J_1_F_02_C004_A_007_024",
        "lip_J_1_F_03_C297_A_002_010",
        "lip_J_1_F_02_C017_A_005_021",
        "lip_J_1_F_02_C017_A_009_019",
    ]
    results = []
    for index, ident in enumerate(selected_ids, 1):
        r = records[ident]
        stem = r["source_stem"]
        assert stem not in train_stems | held_stems
        assert r["recording_key"] not in set(protected["protected_recording_keys"]) | reserved_keys
        assert ident not in protected["protected_parent_ids"]
        assert stem not in protected["blocked_source_stem"]
        assert r["video_sha256"] not in set(protected["blocked_video_sha256"]) | reserved_videos
        assert r["label_sha256"] not in set(protected["blocked_label_sha256"]) | reserved_labels
        assert key(r["text"]) not in protected["protected_full_text_keys"]
        assert not any(t in key(r["text"]) for t in protected["protected_phrase_text_keys"])
        assert not r["source_approval_status"]["parent_transcript_confirmed"]
        tokens = r["token_ids"]
        min_ctc = len(tokens) + sum(a == b for a, b in zip(tokens, tokens[1:]))
        assert r["frames"] >= min_ctc
        alignment_path = ROOT / f"experiments/training_alignment_step06/results/refined_sources/{stem}.json"
        alignment = read(alignment_path)
        assert alignment["status"] == "supported"
        assert alignment["summary"]["all_anchors_strict"]
        assert abs(r["offset_seconds"] - alignment["summary"]["median_offset_seconds"]) < 1e-12
        meta_path = ROOT / f"experiments/local_dataset_step04_20261007/inputs/metadata/{ident}.json"
        meta = read(meta_path)
        assert meta["source_verification"]["member_hashes_checked_on_materialization"]
        assert meta["source_verification"]["archive_stats_checked_before_and_after"]
        assert meta["source_verification"]["members"]["video"]["sha256"] == r["video_sha256"]
        assert meta["source_verification"]["members"]["label"]["sha256"] == r["label_sha256"]
        cache = Path(meta["npz_path"])
        cb = bind(cache)
        assert cb["sha256"] == meta["npz_sha256"]
        results.append({
            "candidate_rank": index, "candidate_id": f"N{index:02}", **r,
            "source_members": sources[stem]["members"],
            "source_alignment": bind(alignment_path), "source_alignment_summary": alignment["summary"],
            "existing_legacy_cache": cb, "existing_cache_metadata": bind(meta_path),
            "legacy_grid": "clip-start relative25fps withround-nearest nominalFPS; do not claimD-grididentity",
            "native_full_sentence_frame_indices_historical": {k: meta["teacher_context"][k] for k in ("official_first", "official_stop_exclusive", "student_union_first", "student_union_stop_exclusive")},
            "duration_seconds": r["end_time"] - r["start_time"],
            "ctc_min_frames": min_ctc,
            "human_transcript_approved": False,
            "human_boundary_approved": False,
            "automatic_asr_screening": "pending",
            "inclusion_status": "candidate_only_pending_fresh_technical_and_ASR_screening",
            "new_internal_word_or_phrase_cut": False,
            "source_audio_usage": "annotation_screening_only_never_student_model_input",
        })
    payload = {
        "schema": "offline-stage5j-new-recording-inventory-v1",
        "status": "candidate_metadata_protection_and_historical_integrity_passed",
        "source_bindings": [bind(p) for p in files.values()],
        "selection_policy": "Official whole utterances1..3s, no newinternalcut,6distinct nontrain/nonheld/nonreservedrecordings,4speakers; one candidate per speaker first; remaining two shortC017recordings are backups.",
        "records": results,
        "counts": {"selected_candidates": len(results), "distinct_recordings": len({r["source_stem"] for r in results}), "speakers": len({r["speaker_id"] for r in results}), "human_approved_new": 0, "ASR_screened_new": 0, "training_added": 0},
        "held_parent_ids_unchanged": previous["held_parent_ids"],
        "held_recordings_also_excluded_here": sorted(held_stems),
        "protected_counts": {"parent_ids": len(protected["protected_parent_ids"]), "recordings": len(protected["protected_recording_keys"]), "reserved_probe_recordings": len(reserved)},
        "provisional": True, "training_ready": False,
        "scope": {"raw_media_read": 0, "neural_calls": 0, "new_human_approvals": 0, "training_updates": 0},
    }
    dest = OUT / "new_recording_candidates.json"
    with dest.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"path": str(dest), "counts": payload["counts"], "ids": selected_ids}, ensure_ascii=False))

if __name__ == "__main__":
    main()
