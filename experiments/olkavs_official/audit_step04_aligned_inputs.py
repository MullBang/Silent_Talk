"""Local-only pre-model QA of aligned20 by five geometry arms."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_step03_inputs import check_anchor, decoded_stats, sha256
from prepare_step04_aligned_inputs import ARMS, BASE, aligned_bounds


def main():
    path = BASE / "inputs_step04_aligned" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sidecar = json.loads(Path(manifest["alignment_sidecar"]["path"]).read_text(encoding="utf-8"))
    plan_path = BASE / "step03_source_plan.json"
    check_anchor(plan_path, manifest["source_plan_sha256"])
    original = {r["id"]: r for r in json.loads(plan_path.read_text(encoding="utf-8"))["records"]}
    offsets = {r["source_name"]: r["label_to_video_offset_seconds"] for r in sidecar["sources"]}
    for key in ("evaluation_plan", "alignment_sidecar", "previous_manifest", "contact_sheet"):
        check_anchor(Path(manifest[key]["path"]), manifest[key]["sha256"])
    check_anchor(BASE / "prepare_step04_aligned_inputs.py", manifest["preparation_script_sha256"])
    check_anchor(BASE / "roi_geometry_step04.py", manifest["geometry_script_sha256"])
    check_anchor(BASE / "automatic_roi.py", manifest["automatic_roi_script_sha256"])
    records = manifest["records"]
    expected_keys = {(sample, arm) for sample in original for arm in ARMS}
    if len(records) != len(expected_keys) or {(r["id"], r["arm"]) for r in records} != expected_keys:
        raise ValueError("Not the exact20-by5 input cohort")
    grouped, arm_rows = defaultdict(list), defaultdict(list)
    for row in records:
        source = original[row["id"]]
        if row["text"] != source["text"] or row["source_text"] != source["source_text"] or row["split"] != "val":
            raise ValueError("Original identity/reference modified")
        offset = offsets[row["source_name"]]
        first, stop = aligned_bounds(source["start_time"], source["end_time"], offset, row["native_fps"])
        if (first, stop, offset) != (row["start_frame"], row["stop_frame_exclusive"], row["label_to_video_offset_seconds"]):
            raise ValueError("Aligned native time contract differs")
        if stop-first != row["expected_frames"]:
            raise ValueError("Frame count differs from aligned native interval")
        check_anchor(Path(row["path"]), row["sha256"])
        check_anchor(Path(row["trace"]["path"]), row["trace"]["sha256"])
        trace = json.loads(Path(row["trace"]["path"]).read_text(encoding="utf-8"))
        reads = trace["source_read"]
        if [f["source_frame_index"] for f in reads] != list(range(first, stop)):
            raise ValueError("Trace omitted/reordered selected native frame")
        for frame in reads:
            index = frame["source_frame_index"]
            if frame["annotation_box_index"] != index or abs(frame["decoder_timestamp_seconds"]-index/row["native_fps"]) > 1e-5:
                raise ValueError("Actual video-indexed bbox/PTS mismatch")
        if row["status"] == "ok":
            stats, _ = decoded_stats(Path(row["path"]), row["expected_frames"])
            if stats["decoded_bgr_stream_sha256"] != row["decoded_bgr_stream_sha256"] or row["frames"] != row["expected_frames"]:
                raise ValueError("Saved input decode differs")
            if len(trace["geometry"]) != row["frames"]:
                raise ValueError("Successful geometry trace frame count differs")
            if row["arm"] == "rectangle_control" and not all(f["current_frame_rectangle_operation_exact"] for f in trace["geometry"]):
                raise ValueError("Rectangle operation not independently reproduced")
        elif row["valid_for_inference"]:
            raise ValueError("Failed input marked usable")
        if row["source_label_bbox_used"] != (row["arm"] == "annotation_bbox") or row["audio_used"]:
            raise ValueError("Unexpected annotation or audio use")
        grouped[row["id"]].append(row)
        arm_rows[row["arm"]].append(row)
    for sample, rows in grouped.items():
        if len({r["source_selected_frames_sha256"] for r in rows}) != 1:
            raise ValueError("Arms received different original frames")
        detector_traces = []
        for row in rows:
            trace = json.loads(Path(row["trace"]["path"]).read_text(encoding="utf-8"))
            detector_traces.append(hashlib.sha256(json.dumps(trace["source_read"], sort_keys=True).encode()).hexdigest())
        if len(set(detector_traces)) != 1:
            raise ValueError("Arms did not share identical detector and source observations")
    controls = arm_rows["rectangle_control"]
    result = {"schema": "olkavs-step04-aligned-input-qa-v1", "status": "pass",
        "manifest": {"path": str(path), "sha256": sha256(path)},
        "evaluation_plan": manifest["evaluation_plan"], "alignment_sidecar": manifest["alignment_sidecar"],
        "script_sha256": sha256(Path(__file__)), "source_plan_sha256": manifest["source_plan_sha256"],
        "counts": {"ids": len(grouped), "arms": len(ARMS), "records": len(records),
                   "successful": sum(r["status"] == "ok" for r in records),
                   "failed": sum(r["status"] != "ok" for r in records),
                   "selected_original_frames_once": sum(r["expected_frames"] for r in controls),
                   "reference_characters_per_arm": sum(len(r["text"].strip()) for r in controls),
                   "reference_words_per_arm": sum(len(r["text"].split()) for r in controls)},
        "all_input_bytes_and_trace_hashes_verified": True,
        "all_successful_input_frames_fully_decoded": True,
        "all5arms_share_each_original_frame_stream_and_detector_box_trace": True,
        "original_references_and_ids_unchanged": True,
        "current_rectangle_crop_operation_exact_every_frame": True,
        "source_frame_indices_and_pts_match_alignment": True,
        "zero_offset_controls_checked": manifest["zero_offset_controls_checked"],
        "detector": {"received": sum(r["automatic_stats"]["frames_received"] for r in controls),
                     "detected": sum(r["automatic_stats"]["frames_detected"] for r in controls),
                     "fallback_previous_box": sum(r["automatic_stats"]["frames_previous_bbox"] for r in controls)},
        "geometry": {}, "visual_qa": "pending human/agent contact-sheet inspection",
        "model_run": False, "audio_read": False, "test_payloads_read": False}
    for arm, rows in arm_rows.items():
        stats = [r["geometry_stats"] for r in rows if r["geometry_stats"]["frames"]]
        result["geometry"][arm] = {
            "median_of_clip_crop_area_ratios": float(np.median([s["median_crop_area_to_original_detector_box"] for s in stats])),
            "median_of_clip_center_step_pixels_mean": float(np.median([s["center_step_pixels_mean"] for s in stats])),
            "boundary_shifted_frames": sum(s["boundary_shifted_frames"] for s in stats),
            "median_crop_seconds_per_clip": float(np.median([r["crop_seconds"] for r in rows]))}
    destination = path.parent / "input_qa.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite prior QA")
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"qa": str(destination), "sha256": sha256(destination), "counts": result["counts"], "detector": result["detector"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
