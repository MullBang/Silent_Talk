"""Prepare fixed val20 pairs: supplied lip boxes versus automatic mouth tracking.

Both arms see exactly the native-CFR frames inside each fixed labelled interval.
The automatic tracker receives only pixels; it never receives labels or boxes.
This script writes new local artifacts only and refuses to replace an existing run.
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

from automatic_roi import AutomaticLipRoi, profile
from prepare_inputs import inspect_mp4

BASE = Path(__file__).resolve().parent
SOURCE_PLAN_SHA = "652692aaaf9778bfa51789de0e177797f9c769e7775df0d112676e0cafaab6f7"
REFERENCE_SHA = "0f9b03f9d92189814c848faab0532c85210121866d6f8e72da3db0b6b436caa3"
ARMS = ("annotation_bbox", "automatic_mediapipe")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_anchor(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Frozen input changed: {path}")


def native_bounds(start: float, end: float, fps: float) -> tuple[int, int]:
    if not (math.isfinite(fps) and fps > 0 and 0 <= start < end):
        raise ValueError("Invalid temporal bounds")
    first, stop = math.ceil(start * fps - 1e-9), math.ceil(end * fps - 1e-9)
    if first >= stop:
        raise ValueError("Empty temporal interval")
    for index, seconds in ((first, start), (stop, end)):
        offset = index / fps - seconds
        if not -1e-8 <= offset < 1 / fps + 1e-8:
            raise ValueError("Native boundary offset exceeds one frame")
    return first, stop


def bbox_diagnostics(annotation: list[int], automatic: list[int] | None) -> dict:
    if automatic is None:
        return {"iou": None, "area_ratio_auto_to_annotation": None,
                "annotation_aspect_ratio": (annotation[2] - annotation[0]) / (annotation[3] - annotation[1]),
                "automatic_aspect_ratio": None}
    ax1, ay1, ax2, ay2 = annotation
    bx1, by1, bx2, by2 = automatic
    a_area, b_area = (ax2-ax1)*(ay2-ay1), (bx2-bx1)*(by2-by1)
    intersection = max(0, min(ax2,bx2)-max(ax1,bx1)) * max(0,min(ay2,by2)-max(ay1,by1))
    return {"iou": intersection / (a_area+b_area-intersection),
            "area_ratio_auto_to_annotation": b_area / a_area,
            "annotation_aspect_ratio": (ax2-ax1)/(ay2-ay1),
            "automatic_aspect_ratio": (bx2-bx1)/(by2-by1)}


def decoded_stats(path: Path, expected: int) -> tuple[dict, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != expected or expected == 0:
        raise ValueError(f"Saved clip did not fully decode: {path.name}: {len(frames)} != {expected}")
    array = np.stack(frames)
    if array.shape != (expected,96,96,3):
        raise ValueError(f"Invalid prepared shape: {array.shape}")
    std = float(array.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Prepared input has zero or invalid standard deviation")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "decoded_bgr_stream_sha256":hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "min": int(array.min()), "max": int(array.max()),
            "mean_before_normalization": float(array.mean()),
            "std_before_normalization_ddof1": std,
            "all_saved_frames_decoded": True}, array[len(array)//2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "inputs_step03")
    parser.add_argument("--read-mode", choices=("prefix", "selected-seek"), default="prefix")
    parser.add_argument("--compare-existing", type=Path)
    args = parser.parse_args()
    source_plan_path = BASE / "step03_source_plan.json"
    reference_path = BASE / "reference_profile_96_v1.json"
    check_anchor(source_plan_path, SOURCE_PLAN_SHA)
    check_anchor(reference_path, REFERENCE_SHA)
    check_anchor(args.evaluation_plan, args.evaluation_plan_sha256)
    evaluation_plan = json.loads(args.evaluation_plan.read_text(encoding="utf-8"))
    plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    for key in ("manifest", "inventory", "diagnostic_plan", "planning_script"):
        check_anchor(Path(plan[key]["path"]), plan[key]["sha256"])
    records = plan["records"]
    if len(records) != 20 or any(row["split"] != "val" for row in records):
        raise ValueError("Expected all and only fixed val20")
    if sorted(evaluation_plan["source_ids"]) != sorted(row["id"] for row in records):
        raise ValueError("Evaluation plan source cohort differs")
    if cv2.__version__ != "4.11.0":
        raise ValueError(f"Frozen preparation requires OpenCV4.11.0; got {cv2.__version__}")
    output = args.output.resolve()
    if not output.is_relative_to(BASE):
        raise ValueError("Output must remain within isolated experiment directory")
    if output.exists():
        raise FileExistsError("Refusing to overwrite step03 artifacts")
    output.mkdir()
    for arm in ARMS:
        (output / arm).mkdir()
    (output / "traces").mkdir()
    groups = defaultdict(list)
    for row in records:
        groups[row["source_video"]["path"]].append(row)
    prepared, sources, tiles = [], {}, []
    started = time.perf_counter()
    for source_name, rows in groups.items():
        source_started = time.perf_counter()
        source = Path(source_name)
        before = source.stat()
        expected = rows[0]["source_video"]
        if (before.st_size,before.st_mtime_ns) != (expected["size_bytes"],expected["mtime_ns"]):
            raise ValueError(f"Source metadata changed: {source.name}")
        container = inspect_mp4(source)
        label_path = Path(rows[0]["label_path"])
        label_bytes = label_path.read_bytes()
        if hashlib.sha256(label_bytes).hexdigest() != rows[0]["label_sha256"]:
            raise ValueError("Label bytes changed")
        labels = json.loads(label_bytes.decode("utf-8-sig"))[0]
        boxes = labels["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Cannot open source: {source.name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        segments = []
        for row in rows:
            first, stop = native_bounds(row["start_time"], row["end_time"], fps)
            if stop > min(frame_count,len(boxes)):
                raise ValueError("Selected interval exceeds source or label box count")
            paths = {arm: output / arm / f"{row['id']}.mp4" for arm in ARMS}
            writers = {arm: cv2.VideoWriter(str(paths[arm]),cv2.VideoWriter_fourcc(*"mp4v"),
                                           30.0,(96,96)) for arm in ARMS}
            if not all(writer.isOpened() for writer in writers.values()):
                raise ValueError("mp4v writer initialization failed")
            segments.append({"row": row, "first": first, "stop": stop,
                             "paths": paths, "writers": writers,
                             "written": dict.fromkeys(ARMS,0), "tracker": None,
                             "auto_stats": None, "trace": [],
                             "source_selected_frames_sha256": hashlib.sha256(),
                             "annotation_crop_seconds": 0.0, "automatic_process_seconds": 0.0,
                             "write_seconds": dict.fromkeys(ARMS,0.0)})
        max_stop = max(segment["stop"] for segment in segments)
        decode_seconds, tracker_setup_seconds = 0.0, 0.0
        selected_unique_frames = 0
        frame_decode_calls, seek_events = 0, []
        if args.read_mode == "prefix":
            frame_indices = range(max_stop)
        else:
            frame_indices = sorted({index for segment in segments for index in range(segment["first"],segment["stop"])})
        previous_index = None
        try:
            for frame_index in frame_indices:
                if args.read_mode == "selected-seek" and (previous_index is None or frame_index != previous_index+1):
                    capture.release()
                    capture=cv2.VideoCapture(str(source))
                    seek_started=time.perf_counter()
                    seek_ok=capture.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
                    actual_position=capture.get(cv2.CAP_PROP_POS_FRAMES)
                    if not seek_ok or abs(actual_position-frame_index)>1e-6:
                        raise ValueError(f"Cannot seek to exact selected start: {frame_index}")
                    seek_events.append({"target_frame":frame_index,"reported_position":actual_position,
                                        "seconds":time.perf_counter()-seek_started})
                t0 = time.perf_counter()
                ok, frame = capture.read()
                decode_seconds += time.perf_counter() - t0
                frame_decode_calls += 1
                if not ok:
                    raise ValueError(f"Source decode failed: {source.name} frame {frame_index}")
                if abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-(frame_index+1))>1e-6:
                    raise ValueError("Decoder frame position changed unexpectedly")
                decoded_timestamp_seconds=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
                timestamp_available=decoded_timestamp_seconds>0 or frame_index==0
                previous_index=frame_index
                active = [segment for segment in segments if segment["first"] <= frame_index < segment["stop"]]
                if active:
                    selected_unique_frames += 1
                for segment in active:
                    if segment["tracker"] is None:
                        t0 = time.perf_counter()
                        segment["tracker"] = AutomaticLipRoi()
                        segment["tracker"].__enter__()
                        tracker_setup_seconds += time.perf_counter() - t0
                    tracker = segment["tracker"]
                    segment["source_selected_frames_sha256"].update(np.asarray([frame_index],dtype="<i8").tobytes())
                    segment["source_selected_frames_sha256"].update(frame.tobytes(order="C"))
                    t0 = time.perf_counter()
                    top,left,bottom,right = map(int,boxes[frame_index])
                    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
                        raise ValueError("Invalid annotation bbox")
                    annotation = cv2.resize(frame[top:bottom,left:right],(96,96),interpolation=cv2.INTER_LINEAR)
                    segment["annotation_crop_seconds"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    automatic = tracker.process(frame)
                    segment["automatic_process_seconds"] += time.perf_counter() - t0
                    for arm, crop in zip(ARMS,(annotation,automatic)):
                        if crop is not None:
                            t0 = time.perf_counter()
                            segment["writers"][arm].write(crop)
                            segment["write_seconds"][arm] += time.perf_counter() - t0
                            segment["written"][arm] += 1
                    auto_box = None if tracker.last_box_xyxy is None else list(map(int,tracker.last_box_xyxy))
                    annotation_xyxy = [left,top,right,bottom]
                    segment["trace"].append({
                        "source_frame_index": frame_index, "source_timestamp_seconds": frame_index/fps,
                        "decoder_position_after_read":capture.get(cv2.CAP_PROP_POS_FRAMES),
                        "decoder_timestamp_seconds":decoded_timestamp_seconds,
                        "decoder_timestamp_available":timestamp_available,
                        "decoder_timestamp_minus_cfr_seconds":decoded_timestamp_seconds-frame_index/fps if timestamp_available else None,
                        "annotation_xyxy": annotation_xyxy, "automatic_xyxy": auto_box,
                        "automatic_failed": tracker.failed,
                        "automatic_frame_record": dict(tracker.frame_records[-1]) if tracker.frame_records else None,
                        **bbox_diagnostics(annotation_xyxy,auto_box if automatic is not None else None)})
                    if frame_index + 1 == segment["stop"]:
                        segment["auto_stats"] = tracker.stats()
                        segment["auto_failure_reason"] = tracker.failure_reason
                        segment["auto_failed"] = tracker.failed
                        tracker.__exit__(None,None,None)
                        segment["tracker"] = None
                        for writer in segment["writers"].values():
                            writer.release()
                        print(f"EXTRACTED {segment['row']['id']} {segment['stop']-segment['first']} frames; auto_failed={segment['auto_failed']}",flush=True)
        finally:
            capture.release()
            for segment in segments:
                if segment["tracker"] is not None:
                    segment["tracker"].__exit__(None,None,None)
                for writer in segment["writers"].values():
                    writer.release()
        after = source.stat()
        if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
            raise ValueError("Source metadata changed while reading")
        sources[source.name] = {
            "path": str(source), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "native_fps": fps, "reported_frames": frame_count,"width":width,"height":height,
            "decoded_prefix_frames": max_stop if args.read_mode=="prefix" else None,
            "frame_decode_calls":frame_decode_calls,"read_mode":args.read_mode,"seek_events":seek_events,
            "selected_unique_frames":selected_unique_frames,
            "decode_seconds": decode_seconds, "tracker_setup_seconds":tracker_setup_seconds,
            "extraction_source_wall_seconds":time.perf_counter()-source_started,
            "full_source_sha256":None,"source_modified":False,"audio_read":False,
            "top_level_container_screen":container,"full_recording_decode_verified":False}
        for segment in segments:
            row, count = segment["row"], segment["stop"]-segment["first"]
            trace_path = output / "traces" / f"{row['id']}.json"
            trace_path.write_text(json.dumps({"id":row["id"],"automatic_stats":segment["auto_stats"],
                "bbox_comparison_limitation":"Automatic boxes include fixed20% margin; annotation boxes do not. IoU is diagnostic, not detector accuracy.",
                "frames":segment["trace"]},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            common = {
                "id":row["id"],"speaker_id":row["speaker_id"],"split":"val","resolution":96,
                "source_name":source.name,"source_text":row["source_text"],"text":row["text"],
                "start_time":row["start_time"],"end_time":row["end_time"],
                "start_frame":segment["first"],"stop_frame_exclusive":segment["stop"],
                "expected_frames":count,"fps":30.0,"native_fps":fps,
                "temporal_method":"native-cfr-timestamps-ceil-v1",
                "first_frame_time_minus_label_seconds":segment["first"]/fps-row["start_time"],
                "stop_frame_time_minus_label_seconds":segment["stop"]/fps-row["end_time"],
                "previous_nominal30_start_frame":row["nominal_start_frame"],
                "previous_nominal30_stop_frame_exclusive":row["nominal_stop_frame_exclusive"],
                "previous_nominal30_first_time_minus_label_seconds":row["nominal_start_frame"]/fps-row["start_time"],
                "previous_nominal30_stop_time_minus_label_seconds":row["nominal_stop_frame_exclusive"]/fps-row["end_time"],
                "start_frame_delta_from_nominal30":segment["first"]-row["nominal_start_frame"],
                "stop_frame_delta_from_nominal30":segment["stop"]-row["nominal_stop_frame_exclusive"],
                "color_order":"BGR","normalization_applied":False,"audio_used":False,
                "source_selected_frames_sha256":segment["source_selected_frames_sha256"].hexdigest(),
                "source_selected_frames_hash_format":"Per frame: little-endian int64 source index, then full uint8 BGR source frame C-order",
                "trace":{"path":str(trace_path),"sha256":sha256(trace_path)},
            }
            middle_frames = {}
            for arm in ARMS:
                path = segment["paths"][arm]
                failed = arm == "automatic_mediapipe" and segment["auto_failed"]
                if not failed and segment["written"][arm] != count:
                    raise ValueError("Successful clip write count mismatch")
                item = {**common,"arm":arm,"status":"failed" if failed else "ok",
                        "valid_for_inference":not failed,"failure_reason":segment.get("auto_failure_reason") if failed else None,
                        "path":str(path),"sha256":sha256(path),"size_bytes":path.stat().st_size,
                        "frames":segment["written"][arm],"write_seconds":segment["write_seconds"][arm],
                        "crop_seconds":segment["annotation_crop_seconds"] if arm=="annotation_bbox" else None,
                        "automatic_process_seconds":segment["automatic_process_seconds"] if arm=="automatic_mediapipe" else None,
                        "automatic_stats":segment["auto_stats"] if arm=="automatic_mediapipe" else None,
                        "margin":0 if arm=="annotation_bbox" else 0.2,
                        "source_label_bbox_used":arm=="annotation_bbox"}
                if not failed:
                    t0=time.perf_counter()
                    stats,middle=decoded_stats(path,count)
                    item.update(stats)
                    item["verification_decode_seconds"]=time.perf_counter()-t0
                    middle_frames[arm]=middle
                    if args.compare_existing is not None:
                        previous=args.compare_existing/arm/path.name
                        if previous.exists():
                            previous_hash=sha256(previous)
                            prior_capture=cv2.VideoCapture(str(previous))
                            prior_frame_count=int(prior_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                            prior_capture.release()
                            if prior_frame_count != count:
                                item["previous_partial_run_comparison"]={
                                    "path":str(previous.resolve()),"sha256":previous_hash,
                                    "original_preserved":True,"comparison_performed":False,
                                    "reason":"Prior run stopped before completing this clip; empty/partial writer placeholder preserved",
                                    "prior_reported_frames":prior_frame_count,"required_frames":count}
                            else:
                                previous_stats,previous_middle=decoded_stats(previous,count)
                                item["previous_partial_run_comparison"]={
                                    "path":str(previous.resolve()),"sha256":previous_hash,
                                    "comparison_performed":True,
                                    "encoded_bytes_identical":previous_hash==item["sha256"],
                                    "decoded_shape_identical":previous_stats["shape"]==stats["shape"],
                                    "decoded_bgr_stream_identical":previous_stats["decoded_bgr_stream_sha256"]==stats["decoded_bgr_stream_sha256"],
                                    "middle_frame_pixels_identical":bool(np.array_equal(middle,previous_middle)),
                                    "original_preserved":True}
                                if previous_hash != item["sha256"]:
                                    raise ValueError(f"Reproduced selected input differs from completed prior artifact: {arm}/{row['id']}")
                else:
                    item["partial_artifact_not_used_for_inference"]=True
                    item["all_saved_frames_decoded"]=False
                prepared.append(item)
            tile=np.full((164,392,3),255,dtype=np.uint8)
            cv2.putText(tile,f"{row['speaker_id']} sentence{row['sentence_id']}",(4,15),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            for arm_index,arm in enumerate(ARMS):
                left=arm_index*196
                cv2.putText(tile,"annotation" if arm_index==0 else "automatic",(left+4,32),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
                if arm in middle_frames:
                    tile[40:160,left+36:left+156]=cv2.resize(middle_frames[arm],(120,120),interpolation=cv2.INTER_NEAREST)
                else:
                    cv2.putText(tile,"FAILED",(left+45,95),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
            tiles.append(tile)
        (output/"preparation_checkpoint.json").write_text(json.dumps({
            "schema":"olkavs-step03-preparation-checkpoint-v1","read_mode":args.read_mode,
            "created_utc":datetime.now(timezone.utc).isoformat(),"sources":sources,"records":prepared,
            "evaluation_plan_sha256":args.evaluation_plan_sha256,
            "source_plan_sha256":SOURCE_PLAN_SHA,"preparation_script_sha256":sha256(Path(__file__)),
            "complete":len(prepared)==40},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if len(prepared)!=40:
        raise ValueError("Expected40 paired records, including any failures")
    sheet=np.full((164*5,392*4,3),255,dtype=np.uint8)
    for index,tile in enumerate(tiles):
        top,left=(index//4)*164,(index%4)*392
        sheet[top:top+164,left:left+392]=tile
    sheet_path=output/"contact_sheet.png"
    if not cv2.imwrite(str(sheet_path),sheet):
        raise ValueError("Cannot write contact sheet")
    manifest={
        "schema":"olkavs-step03-paired-input-manifest-v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "status":"prepared","source_plan_sha256":SOURCE_PLAN_SHA,
        "reference_profile_sha256":REFERENCE_SHA,
        "evaluation_plan":{"path":str(args.evaluation_plan.resolve()),"sha256":args.evaluation_plan_sha256},
        "preparation_script_sha256":sha256(Path(__file__)),
        "automatic_roi_script_sha256":sha256(BASE/"automatic_roi.py"),
        "automatic_roi_profile":profile(),
        "roi_environment_requirements_sha256":sha256(BASE/"requirements-lock-roi.txt"),
        "opencv_version":cv2.__version__,"numpy_version":np.__version__,
        "source_read_mode":args.read_mode,
        "previous_partial_run_preserved":str(args.compare_existing.resolve()) if args.compare_existing else None,
        "training_run":False,"model_run":False,"audio_used":False,"test_payloads_read":False,
        "public_pretraining_overlap_unknown":True,"source_count":len(sources),"record_count":len(prepared),
        "prepared_count":sum(row["status"]=="ok" for row in prepared),
        "failed_count":sum(row["status"]=="failed" for row in prepared),
        "temporal_contract":"Same nativeCFR frame indices in both arms: ceil(start*fps-1e-9):ceil(end*fps-1e-9). No resampling. mp4v nominal30 only stores the selected frame sequence.",
        "normalization":"Runner: whole-utterance float32 mean/std correction=1; no normalization performed here",
        "automatic_policy":"Per-utterance reset; pixels only; fixed project landmarks/margin; no source-label fallback; failed clips remain in full denominator",
        "preparation_wall_seconds":time.perf_counter()-started,"sources":sources,"records":prepared,
        "contact_sheet":{"path":str(sheet_path),"sha256":sha256(sheet_path)},
        "visual_qa":"pending separate inspection",
        "limitations":["Provider may cache full cloud source despite bounded application reads",
                       "Whole-recording integrity not asserted beyond selected decoded prefix",
                       "Known utterance intervals supplied; automatic speech boundary detection not evaluated",
                       "Source CFR index/fps timestamps used; live variable-frame-rate timestamps not tested"]}
    destination=output/"manifest.json"
    destination.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"manifest":str(destination),"sha256":sha256(destination),
                      "prepared":manifest["prepared_count"],"failed":manifest["failed_count"]},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
