"""Independently verify saved pixels and clocks, without re-reading any TAR."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from independent_clock import display_grid, parse_rational, nearest_actual_pts, token_ids, ctc_minimum

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
EXT = BASE / "extraction"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def fq(x):
    f = parse_rational(x)
    assert abs(float(f) - x.get("seconds", x.get("decimal_seconds", float(f)))) < 1e-12
    return f


def main():
    output = BASE / "audit/saved_cache_audit.json"
    assert not output.exists(), "Preserve prior audits"
    summary_path = EXT / "results/extraction_summary.json"
    summary = load(summary_path)
    plan_path = EXT / "extraction_plan.json"
    plan = load(plan_path)
    assert summary["plan_sha256"] == sha(plan_path)
    preflight_path = BASE / "audit/pre_extraction_plan_audit.json"
    preflight = load(preflight_path)
    assert preflight["source_sha256"][str(plan_path)] == sha(plan_path)
    source_hashes = {str(summary_path): sha(summary_path), str(plan_path): sha(plan_path),
                     str(preflight_path): sha(preflight_path)}
    for path, h in plan["source_hashes"].items():
        assert sha(path) == h
        source_hashes[path] = h
    refs = load(BASE / "audit/source_time_audit.json")
    source_hashes[str(BASE / "audit/source_time_audit.json")] = sha(BASE / "audit/source_time_audit.json")
    byref = {r["parent_id"]: r for r in refs["records"]}
    byplan = {r["parent_id"]: r for r in plan["records"]}
    assert {r["parent_id"] for r in summary["records"]} == set(byplan) and len(summary["records"]) == 15
    counted = Counter()
    clips_result, pairs, all_neighbors, replay_details = [], [], [], []
    npz_total = 0
    for row in summary["records"]:
        pid = row["parent_id"]
        entry, reference = byplan[pid], byref[pid]
        old = entry["historical_record"]
        old_trace = {t["native_frame_index"]: t for t in old["video"]["trace"]}
        period, origin = fq(row["native_period"]), fq(row["origin"])
        assert period > 0 and row["native_count"] == old["video"]["native_frame_count"]
        assert abs(float(1 / period) - old["video"]["native_fps"]) < 1e-10
        assert row["training_ready"] is False and row["source_members_sha_verified"] is True
        # Independently regenerate the complete CFR PTS values from the recorded exact cadence.
        # This checks saved-table/decoder consistency; it does not independently parse MP4 here.
        pts = [origin + k * period for k in range(row["native_count"])]
        expected_clips = {c["clip_id"]: c for c in entry["clips"]}
        assert {c["clip_id"] for c in row["clips"]} == set(expected_clips)
        arrays = {}
        for cr in row["clips"]:
            npz_path, meta_path = Path(cr["npz_path"]), Path(cr["metadata_path"])
            assert sha(npz_path) == cr["npz_sha256"] and npz_path.stat().st_size == cr["npz_bytes"]
            assert sha(meta_path) == cr["metadata_sha256"]
            source_hashes[str(npz_path)] = sha(npz_path)
            source_hashes[str(meta_path)] = sha(meta_path)
            m = load(meta_path)
            spec = expected_clips[cr["clip_id"]]
            ref = next(x for x in reference["intervals"] if x["kind"] == spec["kind"])
            assert all(m[k] == v for k, v in spec.items())
            assert m["parent_id"] == pid and m["recording_group"] == entry["recording_group"]
            assert m["source_members"] == entry["source_members"]
            assert m["plan_sha256"] == sha(plan_path)
            assert m["normalization"] == plan["normalization"] and m["tokenizer"] == plan["tokenizer"]
            assert m["training_ready"] is False and m["provisional"] is True
            assert m["new_original_pixels_human_approved"] is False
            assert m["audio_included"] is False and m["raw_padding_frames"] == 0
            labels = token_ids(m["text"])
            assert labels == spec["label"] == ref["token_ids"]
            assert ctc_minimum(labels) == m["ctc_min_frames"]
            cm = m["clock_mapping"]
            assert fq(cm["recording_origin"]) == origin and fq(cm["native_period"]) == period
            assert abs(cm["decoded_first_pts_seconds"] - float(origin)) <= 1e-6
            assert cm["exact_CFR_table_and_decoded_brackets_agree"] is True
            assert fq(cm["source_coverage_end"]) == origin + row["native_count"] * period
            assert fq(cm["context_origin"]) == Fraction(reference["context_source_first_sample_48k"], 48000)
            assert fq(cm["offset_applied_once"]) == fq(reference["offset"])
            s, e = fq(m["semantic_video_start"]), fq(m["semantic_video_end"])
            assert s == fq(ref["video_start"]) and e == fq(ref["video_end"])
            grid = display_grid(s, e, origin, origin_measured=True)
            g = m["grid"]
            assert g["policy"] == "D_recording_anchored_25fps_display_overlap_v1"
            assert (g["first"], g["stop"]) == (grid["first"], grid["stop"])
            assert fq(g["display_start"]) == grid["display_start"] and fq(g["display_end"]) == grid["display_end"]
            assert fq(g["before_semantic"]) == grid["left_extension"] and fq(g["after_semantic"]) == grid["right_extension"]
            assert g["outside_semantic_bins_require_QC"] is True
            chosen = nearest_actual_pts(grid["grid"], pts)
            assert len(set(chosen)) == len(chosen)
            with np.load(npz_path, allow_pickle=False) as bundle:
                assert set(bundle.files) == {"frames_rgb", "label", "length", "metadata_json", "schema", "training_ready", "provisional"}
                rgb = bundle["frames_rgb"].copy()
                label_array = bundle["label"]
                assert label_array.dtype == np.int64 and label_array.tolist() == labels
                assert bundle["length"].shape == () and bundle["length"].dtype == np.int64
                assert int(bundle["length"]) == len(rgb)
                assert json.loads(str(bundle["metadata_json"])) == m
                assert str(bundle["schema"]) == m["schema"] == plan["cache_schema"]
                assert bool(bundle["training_ready"]) is False and bool(bundle["provisional"]) is True
            assert rgb.dtype == np.uint8 and rgb.shape == (grid["count"], 96, 96, 3)
            assert list(rgb.shape) == m["shape"] and m["color"] == "RGB" and m["dtype"] == "uint8"
            assert len(rgb) == m["T"] == cr["T"] == len(m["trace"])
            assert len(rgb) >= ctc_minimum(labels) and m["ctc_feasible"] is True
            assert array_sha(rgb) == m["rgb_uint8_sha256"]
            overlaps = []
            common_old = 0
            for j, (frame, trace, target, index) in enumerate(zip(rgb, m["trace"], grid["grid"], chosen)):
                assert trace["grid_index"] == grid["first"] + j and fq(trace["grid_time"]) == target
                assert trace["native_frame_index"] == trace["annotation_index"] == index
                assert fq(trace["actual_pts"]) == pts[index]
                assert abs(trace["decoded_pts_seconds"] - float(pts[index])) <= 1e-6
                overlap = min(target + Fraction(1, 25), e) - max(target, s)
                assert 0 < overlap <= Fraction(1, 25) and fq(trace["semantic_overlap"]) == overlap
                overlaps.append(overlap)
                bracket = trace["selection_bracket"]
                assert 1 <= len(bracket) <= 3
                bs = [b["native_frame_index"] for b in bracket]
                bp = [fq(b["actual_pts"]) for b in bracket]
                assert all(t == pts[k] for k, t in zip(bs, bp))
                assert all(abs(b["decoded_pts_seconds"] - float(t)) <= 1e-6 for b, t in zip(bracket, bp))
                if len(bracket) == 1:
                    assert bp[0] == target
                elif len(bracket) == 2:
                    assert bs[1] == bs[0] + 1 and bp[0] <= target <= bp[1]
                else:
                    assert bs[1] == bs[0] + 1 and bs[2] == bs[1] + 1
                    assert bp[1] == target
                nearest_local = nearest_actual_pts([target], bp)[0]
                assert bs[nearest_local] == index
                assert array_sha(frame) == trace["roi_rgb_sha256"]
                assert array_sha(frame[:, :, ::-1].copy()) == trace["roi_bgr_sha256"]
                width, height = old["video"]["source_size"]
                x1, y1, x2, y2 = np.asarray(trace["annotation_box_xyxy"], dtype=np.float32).astype(float)
                assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
                a = max(0, min(math.floor(x1 - (x2 - x1) * .20), width - 1))
                b = max(0, min(math.floor(y1 - (y2 - y1) * .20), height - 1))
                c = max(a + 1, min(math.ceil(x2 + (x2 - x1) * .20), width))
                d = max(b + 1, min(math.ceil(y2 + (y2 - y1) * .20), height))
                assert trace["crop_rectangle_xyxy"] == [a, b, c, d]
                if index in old_trace:
                    before = old_trace[index]
                    for key in ("source_bgr_sha256", "roi_bgr_sha256", "annotation_index", "annotation_box_xyxy", "crop_rectangle_xyxy"):
                        assert before[key] == trace[key]
                    assert abs(before["actual_pts_seconds"] - trace["decoded_pts_seconds"]) < 1e-8
                    common_old += 1
                counted["RGB_frames"] += 1
                counted["decoded_bracket_entries"] += len(bracket)
            assert sum(overlaps) == e - s
            # Cached wider contexts contain every label intersecting these <=40ms extensions.
            offset = fq(reference["offset"])
            expected_neighbors = []
            for nr in old["context_neighbor_intersections"]:
                start = Fraction(str(nr["start_time"])) + offset
                end = Fraction(str(nr["end_time"])) + offset
                amount = min(grid["display_end"], end) - max(grid["display_start"], start)
                if amount > 0:
                    expected_neighbors.append((int(nr["sentence_id"]), nr["source_text"], amount))
            got_neighbors = [(int(n["sentence_id"]), n["source_text"], fq(n["overlap"])) for n in m["neighbor_label_intersections"]]
            assert got_neighbors == expected_neighbors
            if got_neighbors:
                all_neighbors.append({"clip_id": m["clip_id"], "neighbors": m["neighbor_label_intersections"]})
            counted[m["kind"] + "_frames"] += len(rgb)
            counted["historical_trace_replays_on_saved_frames"] += common_old
            npz_total += cr["npz_bytes"]
            clips_result.append({"clip_id": m["clip_id"], "kind": m["kind"], "split": m["split"],
                "T": len(rgb), "CTC_min": ctc_minimum(labels), "RGB_and_BGR_frame_hashes_pass": len(rgb),
                "historical_saved_frame_replays": common_old, "actual_PTS_nearest_pass": len(rgb),
                "parent_id": pid, "grid_first": grid["first"], "grid_stop": grid["stop"],
                "neighbor_label_intersection_count": len(got_neighbors), "training_ready": False})
            arrays[m["kind"]] = (rgb, m)
        phrase_rgb, phrase_meta = arrays["phrase"]
        full_rgb, full_meta = arrays["full_sentence"]
        start = phrase_meta["grid"]["first"] - full_meta["grid"]["first"]
        assert 0 <= start and start + len(phrase_rgb) <= len(full_rgb)
        assert np.array_equal(phrase_rgb, full_rgb[start:start + len(phrase_rgb)])
        assert phrase_meta["trace"] == full_meta["trace"][start:start + len(phrase_rgb)] or all(
            all(p[k] == f[k] for k in p if k != "semantic_overlap")
            for p, f in zip(phrase_meta["trace"], full_meta["trace"][start:start + len(phrase_rgb)]))
        pairs.append({"parent_id": pid, "phrase_T": len(phrase_rgb), "full_sentence_T": len(full_rgb),
            "phrase_offset_in_full_grid": start, "shared_RGB_exact": True,
            "shared_grid_native_indices_PTS_bbox_hashes_exact": True})
        for saved, expected in zip(row["historical_raw_comparisons"], entry["review_references"]):
            assert all(saved[k] == v for k, v in expected.items())
            assert saved["raw_roi_pts_bbox_match"] is True
        assert len(row["historical_raw_comparisons"]) == len(entry["review_references"])
        replay_details.append({"parent_id": pid, "producer_boundary_replays": len(row["historical_raw_comparisons"]),
            "independent_scope": "Directly check saved D frames against old traces where the native index is shared; boundary-only raw frames not saved in a D cache rely on producer execution evidence."})
    assert counted["RGB_frames"] == summary["total_rgb_frames"] == 2023
    assert counted["phrase_frames"] == 577 and counted["full_sentence_frames"] == 1446
    assert len(clips_result) == summary["cache_count"] == 30
    assert npz_total == 34834703
    assert summary["training_ready"] is False and summary["provisional"] is True
    for k in ("audio_members_read", "neural_calls", "training_updates", "temporary_originals_remaining"):
        assert summary[k] == 0
    assert summary["io"] == {"member_read_attempts": 30, "payload_bytes_read": 4666847421, "members_sha_verified": 30}
    assert not list((EXT / "_tmp").iterdir())
    # Final immutable-scope check, including the initially absent configured checkpoint.
    scope_path = BASE / "results/scope_snapshot.json"
    scope = load(scope_path)
    source_hashes[str(scope_path)] = sha(scope_path)
    assert sha(scope_path) == preflight["source_sha256"][str(scope_path)]
    for item in scope["artifacts"]:
        path = Path(item["path"])
        assert path.exists() == item["exists"]
        if item["exists"]:
            assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    frozen = []
    for snap in refs["snapshots"]:
        assert sha(snap["manifest"]) == snap["sha256"]
        artifact_list = load(snap["manifest"])["artifacts"]
        for item in artifact_list:
            path = Path(item["path"])
            assert sha(path) == item["sha256"] and path.stat().st_size == item["bytes"]
        frozen.append({"stage": snap["stage"], "preserved_files": len(artifact_list)})
    result = {"schema": "stage3-independent-saved-cache-audit-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed_saved_RGB_CTC_rational_PTS_ROI_paired_subset_and_preservation",
        "issues": [], "source_sha256": source_hashes, "counts": dict(counted),
        "auditor_correction_before_final_result": {
            "first_attempt": "Stopped at C004_A011_014 full_sentence grid2002: auditor incorrectly allowed only 1-2 PTS bracket entries.",
            "finding": "Producer saved previous/exact/next three consecutive PTS values for the exact 80.08s hit.",
            "correction": "Allow three entries only when indices are consecutive and the middle PTS exactly equals the target; all nearest-selection checks remain.",
            "source_or_cache_modified": False},
        "cache_NPZ_count": 30, "cache_NPZ_bytes": npz_total, "clips": clips_result,
        "paired_parent_checks": pairs, "historical_boundary_replays": replay_details,
        "neighbor_label_intersections_recomputed": all_neighbors,
        "frozen_preservation": frozen, "scope_snapshot_artifacts_preserved": len(scope["artifacts"]),
        "initially_absent_default_checkpoint_still_absent": True,
        "extraction_reported_IO": summary["io"], "temporary_originals_observed_remaining": 0,
        "limits": ["This audit reopened lossless saved caches, not source TAR payloads; source file hashes and first/last decode checks are producer evidence.",
            "Actual PTS and brackets were checked against exact arithmetic and recorded decoder values. This audit does not independently reparse MP4 timing tables.",
            "Coordinates were independently reconstructed from saved annotations; only frames shared with historical traces allow direct prior raw-hash comparison.",
            "Technical cache correctness is separate from precise articulation-boundary approval or authorization to train.",
            "Full-parent audio confirmation does not become full-parent visual approval; all source-side review scope remains unchanged.",
            "Frame sums count overlapping phrase and full-sentence roles; 2023 is not 2023 distinct decoded source images."],
        "training_ready": False, "new_exact_visual_approvals": 0,
        "execution": {"independent_TAR_reads": 0, "production_imports": 0, "neural_calls": 0, "training_updates": 0,
            "existing_artifacts_modified": 0}}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "npz_bytes": npz_total,
        "neighbor_clips": len(all_neighbors), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
