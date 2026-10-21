"""Prepare20 fixed targets at video times established by independent waveforms.

This module reads numeric offsets, video pixels and video-indexed lip boxes.
It reads no audio and never supplies text to a detector or geometry adapter.
All20 IDs and references stay unchanged. Zero-offset sources must reproduce
the stage03 source stream and both annotation/automatic control pixel streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from automatic_roi import AutomaticLipRoi, profile as detector_profile
from prepare_step03_inputs import check_anchor, decoded_stats, native_bounds, sha256
from prepare_step04_inputs import (PREVIOUS_MANIFEST_SHA, REFERENCE_SHA, SOURCE_PLAN_SHA,
                                   geometry_stats, rectangle_control, require_same_frames, save_json, verify_control)
from roi_geometry_step04 import RoiGeometry, profile as geometry_profile

BASE = Path(__file__).resolve().parent
DETECTOR_SHA = "80a94b59bdf025758bfb061832c34bc3d499f35f51887176b476c69f7819922b"
ARMS = ("annotation_bbox", "rectangle_control", "square", "context_square", "stable_context_square")
GEOMETRY_ARMS = ARMS[2:]


def aligned_bounds(start: float, end: float, offset: float, fps: float) -> tuple[int, int]:
    if not math.isfinite(offset):
        raise ValueError("Alignment offset must be finite")
    return native_bounds(start + offset, end + offset, fps)


def offsets_by_source(alignment: dict, expected_source_names: set[str]) -> dict[str, float]:
    """Require an explicit independently frozen offset for every source."""
    if alignment["status"] != "validated" or not isinstance(alignment["sources"], list):
        raise ValueError("Validated waveform sidecar with sources list required")
    result = {}
    for row in alignment["sources"]:
        name = row["source_name"]
        if name in result or name not in expected_source_names:
            raise ValueError("Alignment source duplicate or outside fixed cohort")
        value = float(row["label_to_video_offset_seconds"])
        if not math.isfinite(value):
            raise ValueError("Invalid waveform-derived offset")
        result[name] = value
    if set(result) != expected_source_names:
        raise ValueError("Every fixed source needs an independently fixed offset")
    return result


def annotation_crop(frame: np.ndarray, tlbr: list[int]) -> tuple[np.ndarray, dict]:
    top, left, bottom, right = map(int, tlbr)
    if not (0 <= top < bottom <= frame.shape[0] and 0 <= left < right <= frame.shape[1]):
        raise ValueError("Invalid video-indexed annotation bbox")
    crop = cv2.resize(frame[top:bottom, left:right], (96, 96), interpolation=cv2.INTER_LINEAR)
    return crop, {"arm": "annotation_bbox", "raw_bbox_xyxy": [left, top, right, bottom],
                  "final_rectangle_xyxy": [left, top, right, bottom], "boundary_shifted": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--alignment-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "inputs_step04_aligned")
    args = parser.parse_args()
    source_plan_path = BASE / "step03_source_plan.json"
    previous_path = BASE / "inputs_step03_recovery02" / "manifest.json"
    check_anchor(source_plan_path, SOURCE_PLAN_SHA)
    check_anchor(previous_path, PREVIOUS_MANIFEST_SHA)
    check_anchor(BASE / "reference_profile_96_v1.json", REFERENCE_SHA)
    check_anchor(BASE / "automatic_roi.py", DETECTOR_SHA)
    check_anchor(args.evaluation_plan, args.evaluation_plan_sha256)
    check_anchor(args.alignment, args.alignment_sha256)
    plan = json.loads(args.evaluation_plan.read_text(encoding="utf-8"))
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
    records = source_plan["records"]
    if len(records) != 20 or any(r["split"] != "val" for r in records):
        raise ValueError("Fixed existing validation20 required")
    if sorted(plan["source_ids"]) != sorted(row["id"] for row in records):
        raise ValueError("Evaluation plan changed fixed cohort")
    plan_alignment = plan.get("alignment_sidecar", plan.get("alignment", {}))
    if (plan["source_plan_sha256"] != SOURCE_PLAN_SHA
            or plan["reference_profile_sha256"] != REFERENCE_SHA
            or plan["parent_manifest"]["sha256"] != PREVIOUS_MANIFEST_SHA
            or tuple(plan["arms"]) != ARMS
            or plan_alignment.get("sha256") != args.alignment_sha256):
        raise ValueError("Plan does not bind every input/alignment anchor")
    check_anchor(BASE / "roi_geometry_step04.py", plan["geometry_script_sha256"])
    if plan["parent_detector_script_sha256"] != DETECTOR_SHA:
        raise ValueError("Plan detector differs from frozen stage03 detector")
    for evidence in alignment["evidence_files"]:
        check_anchor(Path(evidence["path"]), evidence["sha256"])
    if cv2.__version__ != "4.11.0" or np.__version__ != "1.26.4":
        raise ValueError("Requires unchanged stage03 ROI environment")
    previous = {(r["id"], r["arm"]): r for r in previous_manifest["records"]}
    if len(previous) != 40:
        raise ValueError("Parent input cohort malformed")
    source_names = {Path(row["source_video"]["path"]).name for row in records}
    offsets = offsets_by_source(alignment, source_names)
    alignment_anchor = {"path": str(args.alignment.resolve()), "sha256": args.alignment_sha256}
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output.exists():
        raise ValueError("Output must be a new directory inside isolated experiment")
    output.mkdir()
    for arm in ARMS:
        (output / arm).mkdir()
    (output / "traces").mkdir()
    groups = defaultdict(list)
    for row in records:
        groups[row["source_video"]["path"]].append(row)
    prepared, sources, tiles = [], {}, []
    run_started = time.perf_counter()
    for source_name, rows in groups.items():
        source = Path(source_name)
        offset = offsets[source.name]
        before = source.stat()
        expected = rows[0]["source_video"]
        if (before.st_size, before.st_mtime_ns) != (expected["size_bytes"], expected["mtime_ns"]):
            raise ValueError("Original source metadata changed")
        label_path = Path(rows[0]["label_path"])
        label_bytes = label_path.read_bytes()
        if hashlib.sha256(label_bytes).hexdigest() != rows[0]["label_sha256"]:
            raise ValueError("Original label changed")
        labels = json.loads(label_bytes.decode("utf-8-sig"))[0]
        boxes = labels["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        source_started = time.perf_counter()
        source_summary = {"path": str(source), "size_bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "read_mode": "selected-seek",
            "label_to_video_offset_seconds": offset,
            "annotation_box_frame_index": "Actual selected native video index; no label-time offset applied to bbox indices",
            "segments": [], "source_modified": False, "audio_read": False,
            "full_recording_decode_verified": False}
        for row in rows:
            prior = previous[(row["id"], "automatic_mediapipe")]
            prior_annotation = previous[(row["id"], "annotation_bbox")]
            if row["text"] != prior["text"] or row["text"] != prior_annotation["text"]:
                raise ValueError("Reference transcript changed")
            fps = prior["native_fps"]
            first, stop = aligned_bounds(row["start_time"], row["end_time"], offset, fps)
            count = stop - first
            if first < 0 or stop > len(boxes):
                raise ValueError("Aligned interval exceeds video-indexed boxes")
            paths = {arm: output / arm / f"{row['id']}.mp4" for arm in ARMS}
            writers = {arm: cv2.VideoWriter(str(paths[arm]), cv2.VideoWriter_fourcc(*"mp4v"),
                                            30.0, (96, 96)) for arm in ARMS}
            if not all(writer.isOpened() for writer in writers.values()):
                raise ValueError("Cannot initialize all mp4v writers")
            geometries = {arm: RoiGeometry(arm) for arm in GEOMETRY_ARMS}
            failures, written = dict.fromkeys(ARMS, None), dict.fromkeys(ARMS, 0)
            crop_seconds, write_seconds = dict.fromkeys(ARMS, 0.0), dict.fromkeys(ARMS, 0.0)
            geometry_records = {arm: [] for arm in ARMS}
            source_digest = hashlib.sha256()
            capture = cv2.VideoCapture(str(source))
            decode_seconds, detector_seconds = 0.0, 0.0
            read_records, tracker = [], None
            try:
                if not capture.isOpened() or not math.isclose(capture.get(cv2.CAP_PROP_FPS), fps, rel_tol=1e-9):
                    raise ValueError("Source cannot open or native FPS changed")
                if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(boxes):
                    raise ValueError("Source frame count and video-indexed bbox count differ")
                t0 = time.perf_counter()
                seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                start_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                seek_seconds = time.perf_counter() - t0
                if not seek_ok or abs(start_position - first) > 1e-6:
                    raise ValueError("Source exact selected-frame seek failed")
                tracker = AutomaticLipRoi()
                tracker.__enter__()
                for frame_index in range(first, stop):
                    t0 = time.perf_counter()
                    ok, frame = capture.read()
                    decode_seconds += time.perf_counter() - t0
                    if not ok:
                        raise ValueError(f"Source decode failed: {source.name} frame {frame_index}")
                    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if abs(position - (frame_index + 1)) > 1e-6 or abs(timestamp - frame_index / fps) > 1e-5:
                        raise ValueError("Decoded position or native timestamp differs")
                    source_digest.update(np.asarray([frame_index], dtype="<i8").tobytes())
                    source_digest.update(frame.tobytes(order="C"))
                    t0 = time.perf_counter()
                    auto_crop = tracker.process(frame)
                    detector_seconds += time.perf_counter() - t0
                    auto_box = None if tracker.last_box_xyxy is None else list(map(int, tracker.last_box_xyxy))
                    read_records.append({"source_frame_index": frame_index,
                        "decoder_position_after_read": position, "decoder_timestamp_seconds": timestamp,
                        "annotation_box_index": frame_index, "automatic_xyxy": auto_box,
                        "automatic_failed": tracker.failed,
                        "automatic_frame_record": dict(tracker.frame_records[-1]) if tracker.frame_records else None})
                    if auto_crop is None:
                        for arm in ARMS[1:]:
                            failures[arm] = failures[arm] or tracker.failure_reason or "Automatic ROI unavailable"
                    for arm in ARMS:
                        if failures[arm] is not None:
                            continue
                        t0 = time.perf_counter()
                        try:
                            if arm == "annotation_bbox":
                                crop, telemetry = annotation_crop(frame, boxes[frame_index])
                            elif arm == "rectangle_control":
                                crop = auto_crop
                                independent_crop, telemetry = rectangle_control(frame, auto_box)
                                if not np.array_equal(crop, independent_crop):
                                    raise ValueError("Current frozen detector crop differs from original rectangle operation")
                                telemetry["current_frame_rectangle_operation_exact"] = True
                            else:
                                crop, telemetry = geometries[arm].process(frame, auto_box)
                            crop_seconds[arm] += time.perf_counter() - t0
                            telemetry["source_frame_index"] = frame_index
                            geometry_records[arm].append(telemetry)
                            if crop is None or crop.shape != (96, 96, 3) or crop.dtype != np.uint8:
                                raise ValueError("Invalid geometry output shape or dtype")
                            t0 = time.perf_counter()
                            writers[arm].write(crop)
                            write_seconds[arm] += time.perf_counter() - t0
                            written[arm] += 1
                        except (ValueError, RuntimeError, cv2.error) as error:
                            failures[arm] = str(error)
                if offset == 0.0:
                    require_same_frames(source_digest.hexdigest(), prior["source_selected_frames_sha256"])
                auto_stats = tracker.stats()
            finally:
                capture.release()
                if tracker is not None:
                    tracker.__exit__(None, None, None)
                for writer in writers.values():
                    writer.release()
            middle_frames = {}
            for arm in ARMS:
                trace_path = output / "traces" / f"{row['id']}__{arm}.json"
                stats_geometry = geometry_stats(geometry_records[arm])
                save_json(trace_path, {"id": row["id"], "arm": arm,
                    "alignment_sidecar": alignment_anchor, "source_read": read_records,
                    "geometry": geometry_records[arm], "automatic_stats": auto_stats,
                    "annotation_boxes_used": arm == "annotation_bbox", "geometry_stats": stats_geometry})
                path, failed = paths[arm], failures[arm] is not None
                if not failed and written[arm] != count:
                    raise ValueError("Prepared frame count mismatch")
                if offset == 0.0 and arm in ("annotation_bbox", "rectangle_control") and failed:
                    raise ValueError("Zero-offset control failed; cannot establish reproduction")
                item = {"id": row["id"], "speaker_id": row["speaker_id"], "split": "val",
                    "arm": arm, "status": "failed" if failed else "ok", "valid_for_inference": not failed,
                    "failure_reason": failures[arm], "path": str(path), "sha256": sha256(path),
                    "size_bytes": path.stat().st_size, "frames": written[arm], "expected_frames": count,
                    "text": row["text"], "source_text": row["source_text"], "resolution": 96,
                    "source_name": source.name, "start_time": row["start_time"], "end_time": row["end_time"],
                    "aligned_video_start_time": row["start_time"] + offset,
                    "aligned_video_end_time": row["end_time"] + offset,
                    "label_to_video_offset_seconds": offset,
                    "start_frame": first, "stop_frame_exclusive": stop,
                    "previous_start_frame": prior["start_frame"],
                    "previous_stop_frame_exclusive": prior["stop_frame_exclusive"],
                    "native_fps": fps, "fps": 30.0, "temporal_method": "native-cfr-audio-aligned-v1",
                    "source_selected_frames_sha256": source_digest.hexdigest(),
                    "previous_source_selected_frames_sha256": prior["source_selected_frames_sha256"],
                    "original_frame_stream_identical_to_step03": source_digest.hexdigest() == prior["source_selected_frames_sha256"],
                    "zero_offset_reproduction_required": offset == 0.0,
                    "trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
                    "source_label_bbox_used": arm == "annotation_bbox", "detector_rerun": True,
                    "color_order": "BGR", "normalization_applied": False, "audio_used": False,
                    "automatic_stats": auto_stats, "geometry_stats": stats_geometry,
                    "crop_seconds": crop_seconds[arm], "write_seconds": write_seconds[arm],
                    "shared_detector_seconds": detector_seconds}
                if not failed:
                    stats, middle = decoded_stats(path, count)
                    item.update(stats)
                    middle_frames[arm] = middle
                    if offset == 0.0 and arm in ("annotation_bbox", "rectangle_control"):
                        target = prior_annotation if arm == "annotation_bbox" else prior
                        item["step03_control_reproduction"] = verify_control(path, stats, target)
                else:
                    item["partial_artifact_not_used_for_inference"] = True
                prepared.append(item)
            tile = np.full((140, 650, 3), 255, dtype=np.uint8)
            cv2.putText(tile, f"{row['speaker_id']} sentence{row['sentence_id']} offset{offset:+.4f}s", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            for index, arm in enumerate(ARMS):
                left = 130 * index
                cv2.putText(tile, ("annotation", "rectangle", "square", "context1.5", "context+EMA")[index],
                            (left + 4, 32), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
                if arm in middle_frames:
                    tile[40:136, left+17:left+113] = middle_frames[arm]
                else:
                    cv2.putText(tile, "FAILED", (left+15, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            tiles.append(tile)
            source_summary["segments"].append({"id": row["id"], "first_frame": first,
                "stop_frame_exclusive": stop, "seek_reported_position": start_position,
                "seek_seconds": seek_seconds, "decode_seconds": decode_seconds,
                "decoded_frames": count, "native_fps": fps, "detector_seconds": detector_seconds,
                "selected_source_sha256": source_digest.hexdigest(),
                "step03_frame_stream_identical": source_digest.hexdigest() == prior["source_selected_frames_sha256"]})
            print(f"EXTRACTED {row['id']} {count} frames offset={offset:+.6f}; failures={sum(v is not None for v in failures.values())}", flush=True)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Original source metadata changed during read")
        source_summary["source_wall_seconds"] = time.perf_counter() - source_started
        sources[source.name] = source_summary
        save_json(output / "preparation_checkpoint.json", {"records": prepared, "sources": sources,
            "evaluation_plan_sha256": args.evaluation_plan_sha256,
            "alignment_sha256": args.alignment_sha256, "complete": len(prepared) == len(records) * len(ARMS)})
    if len(prepared) != len(records) * len(ARMS):
        raise ValueError("Missing fixed-cohort arm records")
    sheet = np.full((140 * 10, 650 * 2, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        top, left = (index // 2) * 140, (index % 2) * 650
        sheet[top:top+140, left:left+650] = tile
    sheet_path = output / "contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise ValueError("Cannot write contact sheet")
    controls = [r for r in prepared if r["zero_offset_reproduction_required"] and r["arm"] in ("annotation_bbox", "rectangle_control")]
    manifest = {"schema": "olkavs-step04-aligned-geometry-input-manifest-v1", "status": "prepared",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_plan_sha256": SOURCE_PLAN_SHA, "reference_profile_sha256": REFERENCE_SHA,
        "previous_manifest": {"path": str(previous_path), "sha256": PREVIOUS_MANIFEST_SHA},
        "evaluation_plan": {"path": str(args.evaluation_plan.resolve()), "sha256": args.evaluation_plan_sha256},
        "alignment_sidecar": alignment_anchor, "preparation_script_sha256": sha256(Path(__file__)),
        "geometry_script_sha256": sha256(BASE / "roi_geometry_step04.py"), "geometry_profile": geometry_profile(),
        "automatic_roi_script_sha256": DETECTOR_SHA, "automatic_roi_profile": detector_profile(),
        "roi_environment_requirements_sha256": sha256(BASE / "requirements-lock-roi.txt"),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "source_read_mode": "selected-seek", "source_count": len(sources), "record_count": len(prepared),
        "arms": list(ARMS), "prepared_count": sum(r["status"] == "ok" for r in prepared),
        "failed_count": sum(r["status"] == "failed" for r in prepared),
        "zero_offset_controls_checked": len(controls),
        "all_zero_offset_frame_streams_and_control_pixels_identical_to_step03": all(r["original_frame_stream_identical_to_step03"] and r["step03_control_reproduction"]["decoded_bgr_stream_identical"] for r in controls) if controls else None,
        "prior_stream_equivalence_note": "Required only for sources with exact zero offset; nonzero offsets change selected source frames, so old pixel equivalence is not expected",
        "all_arms_use_identical_selected_source_frames": True,
        "rectangle_definition": "Current detector ROI verified frame by frame against original bbox slice and INTER_LINEAR resize; no square conversion",
        "detector_boxes": "Frozen MediaPipe detector rerun on aligned pixels; shared by automatic arms; no annotation fallback",
        "temporal_contract": "ceil((label_start+offset)*native_fps-1e-9):ceil((label_end+offset)*native_fps-1e-9), same indices for every arm, no resampling",
        "normalization": previous_manifest["normalization"],
        "training_run": False, "model_run": False, "audio_used": False, "audio_read": False,
        "alignment_audio_note": "Waveform evidence established numeric offsets independently; this preparation reads no waveform and model receives no audio",
        "test_payloads_read": False, "public_pretraining_overlap_unknown": True,
        "preparation_wall_seconds": time.perf_counter() - run_started,
        "sources": sources, "records": prepared,
        "contact_sheet": {"path": str(sheet_path), "sha256": sha256(sheet_path)},
        "visual_qa": "pending separate inspection",
        "limitations": ["Fixed development cohort; settings motivated by stage03 observations",
                        "Known aligned utterance intervals; not live webcam evaluation",
                        "Only selected original frames verified; whole-source health not asserted",
                        "Crop-motion statistics include genuine facial motion"]}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "sha256": sha256(output / "manifest.json"),
                      "prepared": manifest["prepared_count"], "failed": manifest["failed_count"]}), flush=True)


if __name__ == "__main__":
    main()
