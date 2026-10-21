"""Descriptive preparation statistics only; does not read model predictions."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_step03_inputs import sha256


def main():
    base = Path(__file__).resolve().parent
    path = base / "inputs_step04_aligned" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in manifest["records"]:
        grouped[row["id"]][row["arm"]] = row
    by_arm = defaultdict(lambda: {"areas": [], "area_ratios": [], "aspect_ratios": [],
                                 "center_steps": [], "width_steps": [], "crop_seconds": [],
                                 "boundary_shifted_frames": 0, "bytes": 0, "frames": 0})
    raw_automatic_stats = []
    detector_seconds = []
    shifts = defaultdict(list)
    for sample, rows in grouped.items():
        rectangle = rows["rectangle_control"]
        raw_automatic_stats.append(rectangle["automatic_stats"])
        detector_seconds.append(rectangle["shared_detector_seconds"])
        shifts[rectangle["speaker_id"]].append(rectangle["start_frame"] - rectangle["previous_start_frame"])
        automatic_trace = json.loads(Path(rectangle["trace"]["path"]).read_text(encoding="utf-8"))
        automatic_boxes = np.array([f["automatic_xyxy"] for f in automatic_trace["source_read"]], dtype=float)
        automatic_wh = automatic_boxes[:, 2:] - automatic_boxes[:, :2]
        automatic_areas = np.prod(automatic_wh, axis=1)
        for arm, row in rows.items():
            trace = json.loads(Path(row["trace"]["path"]).read_text(encoding="utf-8"))
            boxes = np.array([f.get("final_square_xyxy", f.get("final_rectangle_xyxy")) for f in trace["geometry"]], dtype=float)
            wh = boxes[:, 2:] - boxes[:, :2]
            areas = np.prod(wh, axis=1)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            target = by_arm[arm]
            target["areas"].extend(areas.tolist())
            target["area_ratios"].extend((areas / automatic_areas).tolist())
            target["aspect_ratios"].extend((wh[:, 0] / wh[:, 1]).tolist())
            target["center_steps"].extend(np.linalg.norm(np.diff(centers, axis=0), axis=1).tolist())
            target["width_steps"].extend(np.abs(np.diff(wh[:, 0])).tolist())
            target["crop_seconds"].append(row["crop_seconds"])
            target["boundary_shifted_frames"] += row["geometry_stats"]["boundary_shifted_frames"]
            target["bytes"] += row["size_bytes"]
            target["frames"] += row["frames"]
    sources = list(manifest["sources"].values())
    rectangles = [rows["rectangle_control"] for rows in grouped.values()]
    summary = {
        "schema": "olkavs-step04-aligned-input-summary-v1",
        "manifest": {"path": str(path), "sha256": sha256(path)},
        "qa": {"path": str(path.parent / "input_qa.json"), "sha256": sha256(path.parent / "input_qa.json")},
        "script_sha256": sha256(Path(__file__)),
        "clips": len(manifest["records"]), "ids": len(grouped), "arms": len(manifest["arms"]),
        "successful": manifest["prepared_count"], "failed": manifest["failed_count"],
        "unique_original_frames": sum(r["expected_frames"] for r in rectangles),
        "frames_per_arm": {arm: values["frames"] for arm, values in by_arm.items()},
        "input_video_bytes": sum(r["size_bytes"] for r in manifest["records"]),
        "input_directory_bytes_including_traces_at_summary_time": sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file()),
        "unique_source_bytes": sum(s["size_bytes"] for s in sources),
        "source_offsets_seconds": {Path(s["path"]).name: s["label_to_video_offset_seconds"] for s in sources},
        "start_frame_shift_counts_by_speaker": {speaker: dict(Counter(values)) for speaker, values in shifts.items()},
        "reference_characters_per_arm": sum(len(r["text"].strip()) for r in rectangles),
        "reference_words_per_arm": sum(len(r["text"].split()) for r in rectangles),
        "detector": {
            "frames_received": sum(s["frames_received"] for s in raw_automatic_stats),
            "frames_detected": sum(s["frames_detected"] for s in raw_automatic_stats),
            "frames_previous_bbox": sum(s["frames_previous_bbox"] for s in raw_automatic_stats),
            "clips_failed": sum(s["failed"] for s in raw_automatic_stats),
            "total_processing_seconds": sum(detector_seconds),
            "median_processing_seconds_per_clip": float(np.median(detector_seconds))},
        "preparation_wall_seconds": manifest["preparation_wall_seconds"],
        "source_decode_seconds": sum(segment["decode_seconds"] for source in sources for segment in source["segments"]),
        "zero_offset_control_equivalence": "not applicable: all four independently measured offsets are nonzero; selected frames changed",
        "all_five_arms_share_same_original_source_frames": True,
        "current_rectangle_operation_verified_every_frame": True,
        "visual_qa": {"reviewer": "step04_inputs agent",
            "method": "Viewed contact_sheet.png with all20 midpoint frames by five arms, after complete input-video decode verification",
            "contact_sheet": manifest["contact_sheet"],
            "observations": "Lips visible in every midpoint. Rectangle appears vertically stretched; square preserves aspect with tighter field of view; context1.5 and EMA-context show nose/chin context visually similar to supplied annotation. No blank/failed midpoint or clipped mouth seen.",
            "limit": "Midpoint visual inspection cannot certify every video frame, transcript alignment, or tracking jitter"},
        "geometry": {}, "model_predictions_read": False, "audio_read": False,
        "timing_limit": "Offline preparation timing including verification; not live-webcam latency. Rectangle crop time includes independent pixel-equivalence check.",
        "motion_limit": "Pixel motion reflects real facial motion plus tracker variation; lower motion alone does not imply improved recognition"}
    for arm, values in by_arm.items():
        summary["geometry"][arm] = {
            "median_source_crop_area_pixels": float(np.median(values["areas"])),
            "median_area_ratio_to_same_frame_automatic_rectangle": float(np.median(values["area_ratios"])),
            "median_source_crop_aspect_ratio": float(np.median(values["aspect_ratios"])),
            "center_step_pixels_mean": float(np.mean(values["center_steps"])),
            "center_step_pixels_p95": float(np.percentile(values["center_steps"], 95)),
            "width_step_pixels_mean": float(np.mean(values["width_steps"])),
            "boundary_shifted_frames": values["boundary_shifted_frames"],
            "crop_seconds_total": sum(values["crop_seconds"]),
            "crop_seconds_median_clip": float(np.median(values["crop_seconds"])),
            "input_video_bytes": values["bytes"]}
    destination = path.parent / "input_summary.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing input summary")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
