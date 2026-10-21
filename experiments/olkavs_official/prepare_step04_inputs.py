"""Replay fixed automatic boxes on identical source frames for geometry ablation.

No detector or model is run here. Original video frames and stage03 automatic
boxes are reused; annotation boxes and text never enter the geometry adapter.
Every selected source-frame hash and rectangular control must reproduce stage03.
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

from prepare_step03_inputs import check_anchor, decoded_stats, native_bounds, sha256
from roi_geometry_step04 import RoiGeometry, profile

BASE = Path(__file__).resolve().parent
SOURCE_PLAN_SHA = "652692aaaf9778bfa51789de0e177797f9c769e7775df0d112676e0cafaab6f7"
PREVIOUS_MANIFEST_SHA = "a03a55eb67b78e92d2dab76a491bab2b4f2acfbf81e05c218ce396c8e7018613"
REFERENCE_SHA = "0f9b03f9d92189814c848faab0532c85210121866d6f8e72da3db0b6b436caa3"
ARMS = ("rectangle_control", "square", "context_square", "stable_context_square")


def rectangle_control(frame: np.ndarray, box: list[int]) -> tuple[np.ndarray, dict]:
    """Exact original crop operation, without square conversion or smoothing."""
    left, top, right, bottom = map(int, box)
    if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
        raise ValueError("Invalid frozen automatic rectangle")
    crop = cv2.resize(frame[top:bottom, left:right], (96, 96), interpolation=cv2.INTER_LINEAR)
    return crop, {"arm": "rectangle_control", "raw_bbox_xyxy": box,
                  "final_rectangle_xyxy": box, "boundary_shifted": False}


def geometry_stats(frames: list[dict]) -> dict:
    """Describe crop movement; these metrics are not detector accuracy."""
    if not frames:
        return {"frames": 0}
    boxes = np.array([f.get("final_square_xyxy", f.get("final_rectangle_xyxy")) for f in frames], dtype=float)
    raw = np.array([f["raw_bbox_xyxy"] for f in frames], dtype=float)
    wh = boxes[:, 2:] - boxes[:, :2]
    centers = (boxes[:, 2:] + boxes[:, :2]) / 2
    raw_wh = raw[:, 2:] - raw[:, :2]
    movement = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    return {"frames": len(frames), "boundary_shifted_frames": sum(f["boundary_shifted"] for f in frames),
            "median_crop_width": float(np.median(wh[:, 0])), "median_crop_height": float(np.median(wh[:, 1])),
            "median_crop_area_to_original_detector_box": float(np.median(np.prod(wh, axis=1) / np.prod(raw_wh, axis=1))),
            "median_output_crop_aspect_ratio": float(np.median(wh[:, 0] / wh[:, 1])),
            "center_step_pixels_mean": float(np.mean(movement)) if len(movement) else 0.0,
            "center_step_pixels_p95": float(np.percentile(movement, 95)) if len(movement) else 0.0,
            "width_step_pixels_mean": float(np.mean(np.abs(np.diff(wh[:, 0])))) if len(frames) > 1 else 0.0,
            "limitation": "Movement includes true subject motion and tracking variation; not detector jitter accuracy"}


def validate_trace(row: dict, previous: dict, trace: dict) -> None:
    """Reject a different cohort, shifted frame sequence or missing detector box."""
    if (row["id"], row["text"], row["split"]) != (previous["id"], previous["text"], "val"):
        raise ValueError("Previous input cohort or transcript changed")
    first, stop = native_bounds(row["start_time"], row["end_time"], previous["native_fps"])
    if (first, stop) != (previous["start_frame"], previous["stop_frame_exclusive"]):
        raise ValueError("Previous native frame selection changed")
    frames = trace["frames"]
    if trace["id"] != row["id"] or [f["source_frame_index"] for f in frames] != list(range(first, stop)):
        raise ValueError("Automatic trace is not the exact fixed native frame interval")
    if previous["status"] != "ok" or any(f["automatic_failed"] or f["automatic_xyxy"] is None for f in frames):
        raise ValueError("Existing automatic input failed; requires explicit failure-preserving policy")


def require_same_frames(actual: str, previous: str) -> None:
    if actual != previous:
        raise ValueError("Selected original source frames differ from stage03; no inference permitted")


def verify_control(path: Path, stats: dict, previous: dict) -> dict:
    """Decoded equality is mandatory; encoded equality is separately reported."""
    check_anchor(Path(previous["path"]), previous["sha256"])
    same = stats["decoded_bgr_stream_sha256"] == previous["decoded_bgr_stream_sha256"]
    if not same or stats["shape"] != previous["shape"]:
        raise ValueError("Rectangular control pixels differ from stage03; no inference permitted")
    return {"previous_path": previous["path"], "previous_sha256": previous["sha256"],
            "encoded_bytes_identical": sha256(path) == previous["sha256"],
            "decoded_bgr_stream_identical": same, "decoded_shape_identical": True,
            "previous_artifact_preserved": True}


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "inputs_step04")
    args = parser.parse_args()
    source_plan_path = BASE / "step03_source_plan.json"
    previous_path = BASE / "inputs_step03_recovery02" / "manifest.json"
    check_anchor(source_plan_path, SOURCE_PLAN_SHA)
    check_anchor(previous_path, PREVIOUS_MANIFEST_SHA)
    check_anchor(BASE / "reference_profile_96_v1.json", REFERENCE_SHA)
    check_anchor(args.evaluation_plan, args.evaluation_plan_sha256)
    plan = json.loads(args.evaluation_plan.read_text(encoding="utf-8"))
    source_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    previous_manifest = json.loads(previous_path.read_text(encoding="utf-8"))
    records = source_plan["records"]
    if len(records) != 20 or sorted(plan["source_ids"]) != sorted(row["id"] for row in records):
        raise ValueError("Evaluation plan must freeze exactly existing val20")
    if (plan["source_plan_sha256"] != SOURCE_PLAN_SHA
            or plan["reference_profile_sha256"] != REFERENCE_SHA
            or plan["parent_manifest"]["sha256"] != PREVIOUS_MANIFEST_SHA
            or tuple(plan["arms"]) != ARMS):
        raise ValueError("Frozen plan source anchors or geometry arms differ")
    if cv2.__version__ != "4.11.0" or np.__version__ != "1.26.4":
        raise ValueError("Reproduction requires the unchanged stage03 ROI environment")
    previous = {r["id"]: r for r in previous_manifest["records"] if r["arm"] == "automatic_mediapipe"}
    if set(previous) != {r["id"] for r in records}:
        raise ValueError("Previous automatic cohort differs")
    traces = {}
    for row in records:
        prior = previous[row["id"]]
        trace_path = Path(prior["trace"]["path"])
        check_anchor(trace_path, prior["trace"]["sha256"])
        traces[row["id"]] = json.loads(trace_path.read_text(encoding="utf-8"))
        validate_trace(row, prior, traces[row["id"]])
    output = args.output.resolve()
    if not output.is_relative_to(BASE) or output.exists():
        raise ValueError("Output must be a new directory inside the isolated experiment")
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
        before = source.stat()
        expected = rows[0]["source_video"]
        if (before.st_size, before.st_mtime_ns) != (expected["size_bytes"], expected["mtime_ns"]):
            raise ValueError("Original source metadata changed")
        source_started = time.perf_counter()
        source_summary = {"path": str(source), "size_bytes": before.st_size,
                          "mtime_ns": before.st_mtime_ns, "read_mode": "selected-seek",
                          "segments": [], "source_modified": False, "audio_read": False,
                          "full_recording_decode_verified": False}
        for row in rows:
            prior = previous[row["id"]]
            trace_frames = traces[row["id"]]["frames"]
            first, stop, fps = prior["start_frame"], prior["stop_frame_exclusive"], prior["native_fps"]
            count = stop - first
            paths = {arm: output / arm / f"{row['id']}.mp4" for arm in ARMS}
            writers = {arm: cv2.VideoWriter(str(paths[arm]), cv2.VideoWriter_fourcc(*"mp4v"),
                                            30.0, (96, 96)) for arm in ARMS}
            if not all(writer.isOpened() for writer in writers.values()):
                raise ValueError("Cannot initialize all mp4v writers")
            geometries = {arm: RoiGeometry(arm) for arm in ARMS if arm != "rectangle_control"}
            failures = dict.fromkeys(ARMS, None)
            written = dict.fromkeys(ARMS, 0)
            crop_seconds = dict.fromkeys(ARMS, 0.0)
            write_seconds = dict.fromkeys(ARMS, 0.0)
            geometry_records = {arm: [] for arm in ARMS}
            source_digest = hashlib.sha256()
            capture = cv2.VideoCapture(str(source))
            decode_seconds = 0.0
            read_records = []
            try:
                if not capture.isOpened() or not math.isclose(capture.get(cv2.CAP_PROP_FPS), fps, rel_tol=1e-9):
                    raise ValueError("Source cannot open or native FPS changed")
                t0 = time.perf_counter()
                seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                start_position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                seek_seconds = time.perf_counter() - t0
                if not seek_ok or abs(start_position - first) > 1e-6:
                    raise ValueError("Source exact selected-frame seek failed")
                for frame_index, prior_frame in zip(range(first, stop), trace_frames):
                    t0 = time.perf_counter()
                    ok, frame = capture.read()
                    decode_seconds += time.perf_counter() - t0
                    if not ok:
                        raise ValueError(f"Source decode failed: {source.name} frame {frame_index}")
                    position = capture.get(cv2.CAP_PROP_POS_FRAMES)
                    timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if abs(position - (frame_index + 1)) > 1e-6 or abs(timestamp - frame_index / fps) > 1e-5:
                        raise ValueError("Decoded source position or native timestamp differs")
                    source_digest.update(np.asarray([frame_index], dtype="<i8").tobytes())
                    source_digest.update(frame.tobytes(order="C"))
                    box = prior_frame["automatic_xyxy"]
                    read_records.append({"source_frame_index": frame_index,
                                         "decoder_position_after_read": position,
                                         "decoder_timestamp_seconds": timestamp,
                                         "automatic_xyxy": box})
                    for arm in ARMS:
                        if failures[arm] is not None:
                            continue
                        t0 = time.perf_counter()
                        try:
                            crop, telemetry = (rectangle_control(frame, box) if arm == "rectangle_control"
                                               else geometries[arm].process(frame, box))
                            crop_seconds[arm] += time.perf_counter() - t0
                            geometry_records[arm].append(telemetry)
                            if crop is None or crop.shape != (96, 96, 3) or crop.dtype != np.uint8:
                                raise ValueError("Invalid geometry output shape or dtype")
                            t0 = time.perf_counter()
                            writers[arm].write(crop)
                            write_seconds[arm] += time.perf_counter() - t0
                            written[arm] += 1
                        except (ValueError, RuntimeError, cv2.error) as error:
                            failures[arm] = str(error)
                require_same_frames(source_digest.hexdigest(), prior["source_selected_frames_sha256"])
            finally:
                capture.release()
                for writer in writers.values():
                    writer.release()
            if failures["rectangle_control"] is not None:
                raise ValueError("Rectangular control failed; cannot evaluate geometry ablation")
            middle_frames = {}
            for arm in ARMS:
                trace_path = output / "traces" / f"{row['id']}__{arm}.json"
                save_json(trace_path, {"id": row["id"], "arm": arm,
                                      "previous_automatic_trace": prior["trace"],
                                      "source_read": read_records,
                                      "geometry": geometry_records[arm],
                                      "annotation_boxes_used": False,
                                      "geometry_stats": geometry_stats(geometry_records[arm])})
                path = paths[arm]
                failed = failures[arm] is not None
                if not failed and written[arm] != count:
                    raise ValueError("Prepared frame count mismatch")
                item = {"id": row["id"], "speaker_id": row["speaker_id"], "split": "val",
                        "arm": arm, "status": "failed" if failed else "ok",
                        "valid_for_inference": not failed, "failure_reason": failures[arm],
                        "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
                        "frames": written[arm], "expected_frames": count,
                        "text": row["text"], "source_text": row["source_text"], "resolution": 96,
                        "source_name": source.name, "start_time": row["start_time"], "end_time": row["end_time"],
                        "start_frame": first, "stop_frame_exclusive": stop,
                        "native_fps": fps, "fps": 30.0, "temporal_method": "native-cfr-timestamps-ceil-v1",
                        "source_selected_frames_sha256": source_digest.hexdigest(),
                        "original_frame_stream_identical_to_step03": True,
                        "previous_automatic_trace": prior["trace"],
                        "trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
                        "source_label_bbox_used": False, "detector_rerun": False,
                        "color_order": "BGR", "normalization_applied": False, "audio_used": False,
                        "automatic_stats": prior["automatic_stats"],
                        "geometry_stats": geometry_stats(geometry_records[arm]),
                        "crop_seconds": crop_seconds[arm], "write_seconds": write_seconds[arm]}
                if not failed:
                    stats, middle = decoded_stats(path, count)
                    item.update(stats)
                    middle_frames[arm] = middle
                    if arm == "rectangle_control":
                        item["step03_control_reproduction"] = verify_control(path, stats, prior)
                else:
                    item["partial_artifact_not_used_for_inference"] = True
                prepared.append(item)
            tile = np.full((140, 520, 3), 255, dtype=np.uint8)
            cv2.putText(tile, f"{row['speaker_id']} sentence{row['sentence_id']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            for index, arm in enumerate(ARMS):
                left = 130 * index
                cv2.putText(tile, ("rectangle", "square", "context1.5", "context+EMA")[index],
                            (left + 4, 32), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
                if arm in middle_frames:
                    tile[40:136, left+17:left+113] = middle_frames[arm]
                else:
                    cv2.putText(tile, "FAILED", (left+15, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            tiles.append(tile)
            source_summary["segments"].append({"id": row["id"], "first_frame": first,
                "stop_frame_exclusive": stop, "seek_reported_position": start_position,
                "seek_seconds": seek_seconds, "decode_seconds": decode_seconds,
                "decoded_frames": count, "native_fps": fps,
                "selected_source_sha256": source_digest.hexdigest(), "step03_frame_stream_identical": True})
            print(f"EXTRACTED {row['id']} {count} frames; control identical; failures={sum(v is not None for v in failures.values())}", flush=True)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Original source metadata changed during read")
        source_summary["source_wall_seconds"] = time.perf_counter() - source_started
        sources[source.name] = source_summary
        save_json(output / "preparation_checkpoint.json", {"records": prepared, "sources": sources,
            "evaluation_plan_sha256": args.evaluation_plan_sha256, "complete": len(prepared) == 80})
    if len(prepared) != 80:
        raise ValueError("Expected all fixed val20 by four arms including failures")
    sheet = np.full((140 * 10, 520 * 2, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        top, left = (index // 2) * 140, (index % 2) * 520
        sheet[top:top+140, left:left+520] = tile
    sheet_path = output / "contact_sheet.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise ValueError("Cannot write contact sheet")
    manifest = {"schema": "olkavs-step04-geometry-input-manifest-v1", "status": "prepared",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_plan_sha256": SOURCE_PLAN_SHA, "reference_profile_sha256": REFERENCE_SHA,
                "previous_manifest": {"path": str(previous_path), "sha256": PREVIOUS_MANIFEST_SHA},
                "evaluation_plan": {"path": str(args.evaluation_plan.resolve()), "sha256": args.evaluation_plan_sha256},
                "preparation_script_sha256": sha256(Path(__file__)),
                "geometry_script_sha256": sha256(BASE / "roi_geometry_step04.py"), "geometry_profile": profile(),
                "roi_environment_requirements_sha256": sha256(BASE / "requirements-lock-roi.txt"),
                "opencv_version": cv2.__version__, "numpy_version": np.__version__,
                "source_read_mode": "selected-seek", "source_count": len(sources), "record_count": len(prepared),
                "prepared_count": sum(r["status"] == "ok" for r in prepared),
                "failed_count": sum(r["status"] == "failed" for r in prepared),
                "all_original_frame_streams_identical_to_step03": True,
                "all_rectangular_control_decoded_pixels_identical_to_step03": True,
                "detector_boxes": "Frozen stage03 automatic trace; no detector rerun or annotation fallback",
                "temporal_contract": previous_manifest["temporal_contract"],
                "normalization": previous_manifest["normalization"],
                "training_run": False, "model_run": False, "audio_used": False, "test_payloads_read": False,
                "public_pretraining_overlap_unknown": True,
                "preparation_wall_seconds": time.perf_counter() - run_started,
                "sources": sources, "records": prepared,
                "contact_sheet": {"path": str(sheet_path), "sha256": sha256(sheet_path)},
                "visual_qa": "pending separate inspection",
                "limitations": ["Fixed development cohort; settings chosen after stage03 observations",
                                "Known utterance boundaries; not live webcam evaluation",
                                "Only selected original frames verified; whole-source health not asserted",
                                "Jitter is descriptive crop motion and includes real facial motion"]}
    save_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "manifest.json"), "sha256": sha256(output / "manifest.json"),
                      "prepared": manifest["prepared_count"], "failed": manifest["failed_count"]}), flush=True)


if __name__ == "__main__":
    main()
