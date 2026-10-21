"""Read verified original TAR video/label members for review-only ROI diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import math
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
EXTRACTION = ROOT / "experiments/offline_boundary_finalize_step02d_20261013/results/extraction_plan.json"
MEDIA = ROOT / "experiments/offline_alignment_step02_20261012/results/media_manifest.json"
TAR_HELPER = ROOT / "experiments/local_source_step01_20261007/tar_source.py"
TAR_SHA = "05f72840576046683d6b5a3c1121a94b7fe2099fd0d85596c30da511801effb2"
RESERVE = 10 * 1024 ** 3


def sha(path: Path) -> str:
    """Hash without loading an entire video into RAM."""
    try:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Cannot hash {path}") from exc


def read(path: Path) -> dict[str, Any]:
    """Parse one UTF-8 metadata record."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Cannot read {path}") from exc


def save(path: Path, value: Any) -> None:
    """Write within this new diagnostic directory, refusing prior artifacts."""
    try:
        if not path.resolve().is_relative_to(BASE) or path.exists():
            raise ValueError(f"Unsafe or existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Cannot write diagnostic {path}") from exc


def exact(value: dict[str, Any]) -> Fraction:
    """Use the exact rational time already fixed by stage2d."""
    try:
        result = Fraction(value["numerator"], value["denominator"])
        if abs(float(result) - value["seconds"]) > 1e-10:
            raise ValueError("Rational/display seconds disagree")
        return result
    except Exception as exc:
        raise RuntimeError("Invalid planned time") from exc


def roi_crop(frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Use the preserved float32-corner 20% margin and 96x96 linear crop contract."""
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = box.tolist()
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("Lip box outside native image")
        points = np.asarray([[x1, y1], [x2, y2]], dtype=np.float32)
        left, top = float(points[:, 0].min()), float(points[:, 1].min())
        right, bottom = float(points[:, 0].max()), float(points[:, 1].max())
        bw, bh = right - left, bottom - top
        a = max(0, min(int(np.floor(left - bw * .20)), width - 1))
        b = max(0, min(int(np.floor(top - bh * .20)), height - 1))
        c = max(a + 1, min(int(np.ceil(right + bw * .20)), width))
        d = max(b + 1, min(int(np.ceil(bottom + bh * .20)), height))
        return cv2.resize(frame[b:d, a:c], (96, 96), interpolation=cv2.INTER_LINEAR), [a, b, c, d]
    except Exception as exc:
        raise RuntimeError("Native ROI crop failed") from exc


def reference_indices(row: dict[str, Any], old: dict[str, Any]) -> list[dict[str, Any]]:
    """Include context endpoints and the reviewed grid nearest both phrase boundaries."""
    try:
        n = len(old["video"]["trace"])
        start = row["phrase"]["start_sample_16k"] / 16000
        end = row["phrase"]["end_sample_16k"] / 16000
        refs = [("context_first", 0), ("reviewed_phrase_start", round(start * 25)),
            ("reviewed_phrase_end", round(end * 25)), ("context_last", n - 1)]
        if any(not 0 <= i < n for _, i in refs):
            raise ValueError("Reference preview frame outside context")
        return [{"role": role, "preview_frame_index": index, "trace": old["video"]["trace"][index]} for role, index in refs]
    except Exception as exc:
        raise RuntimeError("Cannot bind reviewed original frame references") from exc


def prepare() -> dict[str, Any]:
    """Freeze bounded member reads, exact target times and source-code hashes before I/O."""
    try:
        p, media = read(EXTRACTION), read(MEDIA)
        if p["total_expected_frames"] != 233 or len(p["records"]) != 8:
            raise ValueError("Unexpected frozen subset")
        for name, expected in p["source_hashes"].items():
            if sha(Path(name)) != expected:
                raise ValueError(f"Frozen source changed: {name}")
        if sha(TAR_HELPER) != TAR_SHA:
            raise ValueError("Frozen TAR reader changed")
        old = {r["id"]: r for r in media["records"]}
        entries = []
        for row in p["records"]:
            previous = old[row["parent_id"]]
            for role in ("video", "label"):
                if row["source_members"][role] != previous["source_members"][role]:
                    raise ValueError("Source binding differs from prior media")
            native_rate = Fraction(row["historical_native_fps"]["numerator"], row["historical_native_fps"]["denominator"])
            if float(native_rate) != previous["video"]["native_fps"]:
                raise ValueError("Rational native FPS disagrees")
            times = [exact(t) for t in row["target_frame_times"]]
            start, end = exact(row["source_video_start"]), exact(row["source_video_end"])
            if len(times) != row["frames_25fps"] or len(times) != round((end - start) * 25):
                raise ValueError("Fixed frame count differs")
            if times != [start + Fraction(k, 25) for k in range(len(times))]:
                raise ValueError("Fixed 25fps grid differs")
            refs = reference_indices(row, previous)
            left = Fraction(row["phrase"]["start_sample_16k"] * 25, 16000)
            right = Fraction(row["phrase"]["end_sample_16k"] * 25, 16000)
            first, stop = math.floor(left), math.ceil(right)
            brefs = [{"role":"grid_B_preview", "preview_frame_index":i,
                "trace":previous["video"]["trace"][i]} for i in range(first, stop)]
            entries.append({"record": row, "historical_record": previous, "review_references": refs,
                "grid_B_display_overlap_references":brefs,
                "grid_B_display_span_start_minus_selected_start_seconds":float(Fraction(first,25) - Fraction(row["phrase"]["start_sample_16k"],16000)),
                "grid_B_display_span_end_minus_selected_end_seconds":float(Fraction(stop,25) - Fraction(row["phrase"]["end_sample_16k"],16000))})
        unique_archives = sorted({row["source_members"][role]["archive_path"] for row in p["records"] for role in ("video", "label")})
        total = sum(row["source_members"][role]["size"] for row in p["records"] for role in ("video", "label"))
        maximum = max(sum(row["source_members"][role]["size"] for role in ("video", "label")) for row in p["records"])
        free = shutil.disk_usage(BASE).free
        if free < maximum + RESERVE:
            raise ValueError("Insufficient sequential temporary space")
        plan = {"schema": "offline-native-diagnostic-plan-v1", "project_date": "2026-10-13",
            "created_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared_review_only",
            "source_hashes": {str(EXTRACTION): sha(EXTRACTION), str(MEDIA): sha(MEDIA),
                str(TAR_HELPER): TAR_SHA, str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
            "planned_video_members": 8, "planned_label_members": 8, "planned_member_bytes": total,
            "archive_paths": unique_archives, "peak_temporary_member_bytes": maximum,
            "free_bytes_before": free, "reserve_bytes": RESERVE, "planned_target_roi_frames": 233,
            "planned_grid_B_roi_frames":sum(len(e["grid_B_display_overlap_references"]) for e in entries),
            "grid_B_policy":"Existing preview display bins overlapping selected [start,end); not an automatic boundary extension or approved training sampling policy.",
            "split": "unassigned", "review_only": True, "training_ready": False,
            "audio_members_to_read": 0, "neural_calls": 0, "training_updates": 0,
            "records": entries, "policy": "Do not alter targets or add/pad frames. Lossless NPZ is diagnostic only; not a training cache."}
        save(BASE / "native_execution_plan_r4.json", plan)
        return plan
    except Exception as exc:
        raise RuntimeError("Native preflight failed") from exc


def decode_record(path: Path, label_path: Path, entry: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray, dict[int, np.ndarray]]:
    """Decode the union of planned and historical reference indices with PTS validation."""
    capture = None
    try:
        row, historical = entry["record"], entry["historical_record"]
        raw_label = json.loads(label_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw_label,list) or len(raw_label) != 1 or not isinstance(raw_label[0],dict):
            raise ValueError("Expected single-record list label schema")
        label = raw_label[0]
        if Path(label["Video_info"]["video_Name"]).stem != row["recording_group"] or label["speaker_info"]["speaker_ID"] != row["speaker_id"]:
            raise ValueError("Native label identity differs")
        sentence = next(x for x in label["Sentence_info"] if int(x["ID"]) == int(row["parent_id"].rsplit("_",1)[1]))
        if sentence["sentence_text"] != historical["source_text"] or abs(sentence["start_time"]-historical["original_start_time"]) > 1e-6 or abs(sentence["end_time"]-historical["original_end_time"]) > 1e-6:
            raise ValueError("Original label sentence differs from frozen source")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError("Native video cannot open")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if abs(fps - float(Fraction(row["historical_native_fps"]["numerator"], row["historical_native_fps"]["denominator"]))) > 1e-10 or count != historical["video"]["native_frame_count"] or [width, height] != historical["video"]["source_size"]:
            raise ValueError("Native stream metadata differs")
        boxes = np.asarray(label["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.float64)
        if boxes.shape != (count, 4) or not np.isfinite(boxes).all():
            raise ValueError("Label bbox count/values invalid")
        boxes = boxes[:, [1, 0, 3, 2]]
        times = [float(exact(t)) for t in row["target_frame_times"]]
        indices = [round(t * fps) for t in times]
        if indices != row["expected_native_indices_if_prior_cfr_contract_holds"] or len(set(indices)) != len(indices):
            raise ValueError("Native sampling differs from fixed plan")
        all_references = entry["review_references"] + entry["grid_B_display_overlap_references"]
        ref_indices = [ref["trace"]["native_frame_index"] for ref in all_references]
        union = sorted(set(indices + ref_indices))
        if union[0] < 0 or union[-1] >= count:
            raise ValueError("Requested native frame outside stream")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, union[0]):
            raise ValueError("Native seek failed")
        current, full_trace, images = union[0], {}, {}
        for selected in union:
            frame = None
            while current <= selected:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Native video ended early")
                current += 1
            actual_index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if actual_index != selected or abs(pts - selected / fps) > .001:
                raise ValueError("Native decoded PTS/index differs")
            roi, rectangle = roi_crop(frame, boxes[selected])
            full_trace[selected] = {"native_frame_index": selected, "annotation_index": selected,
                "actual_pts_seconds": pts, "source_bgr_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "roi_bgr_sha256": hashlib.sha256(roi.tobytes()).hexdigest(),
                "roi_rgb_sha256": hashlib.sha256(roi[:, :, ::-1].copy().tobytes()).hexdigest(),
                "annotation_box_xyxy": boxes[selected].tolist(), "crop_rectangle_xyxy": rectangle}
            images[selected] = roi[:, :, ::-1].copy()
        comparisons = []
        for ref in all_references:
            prior = ref["trace"]
            actual = full_trace[prior["native_frame_index"]]
            for key in ("source_bgr_sha256", "roi_bgr_sha256", "annotation_box_xyxy", "crop_rectangle_xyxy", "annotation_index"):
                if actual[key] != prior[key]:
                    raise ValueError(f"Prior raw frame/ROI recheck failed: {row['parent_id']} {key}")
            if abs(actual["actual_pts_seconds"] - prior["actual_pts_seconds"]) > 1e-8:
                raise ValueError("Historical PTS differs")
            comparisons.append({**ref, "all_checked_fields_match": True})
        target_trace = []
        for target, index in zip(times, indices, strict=True):
            trace = {**full_trace[index], "target_video_time": target}
            if abs(trace["actual_pts_seconds"] - target) > .5 / fps + 1e-6:
                raise ValueError("Selected frame not nearest time")
            target_trace.append(trace)
        boundary = {ref["role"]:ref for ref in entry["review_references"]}
        start_preview = boundary["reviewed_phrase_start"]
        end_preview = boundary["reviewed_phrase_end"]
        last_pts = target_trace[-1]["actual_pts_seconds"]
        last_to_end = float(exact(row["source_video_end"])) - last_pts
        if abs(last_to_end - row["expected_last_native_pts_to_selected_end_seconds"]) > 1e-8:
            raise ValueError("Expected endpoint residual differs")
        result = {"parent_id": row["parent_id"], "rank": row["rank"], "phrase": row["phrase"],
            "native_fps": fps, "native_frame_count": count, "source_size": [width, height],
            "target_frames": len(indices), "decoded_native_frames_in_span": union[-1] - union[0] + 1,
            "retained_union_frames": len(union), "trace": target_trace, "historical_comparisons": comparisons,
            "start_to_reviewed_nearest_grid_seconds": float(exact(row["source_video_start"])) - start_preview["trace"]["target_video_time"],
            "first_native_index_minus_nearest_review_start": indices[0] - start_preview["trace"]["native_frame_index"],
            "last_native_index_minus_nearest_review_end": indices[-1] - end_preview["trace"]["native_frame_index"],
            "last_selected_native_pts_to_target_end_seconds": last_to_end,
            "last_target_grid_time_to_target_end_seconds": float(exact(row["source_video_end"])) - times[-1],
            "grid_B": {"frames":len(entry["grid_B_display_overlap_references"]),
                "trace":[{**full_trace[r["trace"]["native_frame_index"]],
                    "preview_frame_index":r["preview_frame_index"],
                    "target_video_time":r["trace"]["target_video_time"]} for r in entry["grid_B_display_overlap_references"]],
                "display_span_start_minus_selected_start_seconds":entry["grid_B_display_span_start_minus_selected_start_seconds"],
                "display_span_end_minus_selected_end_seconds":entry["grid_B_display_span_end_minus_selected_end_seconds"],
                "first_native_pts_minus_selected_start_seconds":entry["grid_B_display_overlap_references"][0]["trace"]["actual_pts_seconds"]-float(exact(row["source_video_start"])),
                "last_native_pts_to_selected_end_seconds":float(exact(row["source_video_end"]))-entry["grid_B_display_overlap_references"][-1]["trace"]["actual_pts_seconds"],
                "review_only":True, "sampling_policy_approved":False},
            "native_indices_verified_this_run": True, "audio_members_read": 0,
            "split": "unassigned", "review_only": True, "training_ready": False,
            "qualitative_prior_review_does_not_approve_new_grid": True, "raw_padding_frames": 0}
        return result, np.stack([images[i] for i in indices]), images
    except Exception as exc:
        raise RuntimeError(f"Native extraction failed for {entry['record']['parent_id']}") from exc
    finally:
        if capture is not None:
            capture.release()


def contact(result: dict[str, Any], images: dict[int, np.ndarray]) -> dict[str, Any]:
    """Show raw native endpoint ROIs beside raw sources of earlier reviewed frames."""
    try:
        refs = {r["role"]:r["trace"]["native_frame_index"] for r in result["historical_comparisons"]}
        items = [("context first", refs["context_first"]), ("review start", refs["reviewed_phrase_start"]),
            ("new first", result["trace"][0]["native_frame_index"]),
            ("new last", result["trace"][-1]["native_frame_index"]),
            ("review end", refs["reviewed_phrase_end"]), ("context last", refs["context_last"])]
        canvas = Image.new("RGB", (1000, 245), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
        draw.text((8, 3), result["parent_id"] + " | " + result["phrase"]["text"], font=font, fill="black")
        draw.text((8, 25), "원본 ROI 진단: 영상 입력만 / 학습자료 아님 / grid phase·끝 잔여 확인", font=font, fill="black")
        for column, (name, index) in enumerate(items):
            x, y = 10 + column * 164, 65
            canvas.paste(Image.fromarray(images[index]), (x, y))
            draw.text((x, y + 100), name, font=font, fill="black")
            draw.text((x, y + 120), f"native #{index}", font=font, fill="black")
        path = BASE / "contacts_r4" / (result["parent_id"] + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Contact exists")
        canvas.save(path)
        return {"path": str(path), "sha256": sha(path), "panels": [{"role":role, "native_index":idx,
            "roi_rgb_sha256": hashlib.sha256(images[idx].tobytes()).hexdigest()} for role,idx in items]}
    except Exception as exc:
        raise RuntimeError("Native contact creation failed") from exc


def execute() -> None:
    """Execute the frozen review-only plan one bounded recording at a time."""
    try:
        plan_path = BASE / "native_execution_plan_r4.json"
        plan = read(plan_path)
        for name, expected in plan["source_hashes"].items():
            if sha(Path(name)) != expected:
                raise ValueError("Preflight-bound source changed")
        spec = importlib.util.spec_from_file_location("nativecheck_frozen_tar", TAR_HELPER)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        temp_root = (BASE / "_tmp").resolve()
        if not temp_root.is_relative_to(BASE.resolve()) or temp_root == BASE.resolve():
            raise ValueError("Unsafe temporary root")
        temp_root.mkdir(exist_ok=True)
        if list(temp_root.iterdir()):
            raise ValueError("Temporary root not empty")
        results = []
        for entry in plan["records"]:
            row = entry["record"]
            print(f"START {row['rank']}/8 {row['parent_id']}", flush=True)
            started = time.perf_counter()
            source = {"stem": row["recording_group"], "members": {k:row["source_members"][k] for k in ("video", "label")}}
            with helper.materialize_source(source, temp_root, minimum_free_bytes=RESERVE) as paths:
                # The immutable helper owns its unique directory. Check its exact
                # resolved child paths before work and again before context cleanup.
                checked_dirs = {p.parent.resolve() for p in paths.values()}
                if len(checked_dirs) != 1 or any(not d.is_relative_to(temp_root) or d == temp_root for d in checked_dirs):
                    raise ValueError("Unsafe temporary child")
                try:
                    result, rgb, union_images = decode_record(paths["video"], paths["label"], entry)
                    result["source_members_verified"] = source["members"]
                    npz_path = BASE / "diagnostics_r4" / (row["parent_id"] + ".npz")
                    npz_path.parent.mkdir(parents=True, exist_ok=True)
                    if npz_path.exists():
                        raise ValueError("Diagnostic NPZ already exists")
                    np.savez_compressed(npz_path, review_roi_rgb=rgb,
                        native_indices=np.asarray([t["native_frame_index"] for t in result["trace"]], dtype=np.int64),
                        target_video_times=np.asarray([t["target_video_time"] for t in result["trace"]], dtype=np.float64),
                        actual_video_pts=np.asarray([t["actual_pts_seconds"] for t in result["trace"]], dtype=np.float64),
                        review_only=np.asarray(True), split=np.asarray("unassigned"), training_ready=np.asarray(False),
                        schema=np.asarray("diagnostic-native-rgb-v1-not-training-cache"))
                    with np.load(npz_path, allow_pickle=False) as packed:
                        if not np.array_equal(packed["review_roi_rgb"], rgb) or packed["review_roi_rgb"].shape != (row["frames_25fps"],96,96,3):
                            raise ValueError("Lossless NPZ roundtrip differs")
                    result["npz"] = {"path":str(npz_path), "sha256":sha(npz_path), "bytes":npz_path.stat().st_size,
                        "rgb_uint8_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(), "is_training_cache":False}
                    b_rgb = np.stack([union_images[t["native_frame_index"]] for t in result["grid_B"]["trace"]])
                    bpath = npz_path.with_name(npz_path.stem + "_grid_B.npz")
                    if bpath.exists():
                        raise ValueError("Grid B diagnostic already exists")
                    np.savez_compressed(bpath, review_roi_rgb=b_rgb,
                        native_indices=np.asarray([t["native_frame_index"] for t in result["grid_B"]["trace"]],dtype=np.int64),
                        target_video_times=np.asarray([t["target_video_time"] for t in result["grid_B"]["trace"]],dtype=np.float64),
                        actual_video_pts=np.asarray([t["actual_pts_seconds"] for t in result["grid_B"]["trace"]],dtype=np.float64),
                        review_only=np.asarray(True), split=np.asarray("unassigned"),training_ready=np.asarray(False),
                        schema=np.asarray("diagnostic-native-rgb-grid-B-v1-not-training-cache"))
                    with np.load(bpath,allow_pickle=False) as packed:
                        if not np.array_equal(packed["review_roi_rgb"],b_rgb):
                            raise ValueError("Grid B NPZ roundtrip differs")
                    result["grid_B"]["npz"]={"path":str(bpath),"sha256":sha(bpath),"bytes":bpath.stat().st_size,
                        "rgb_uint8_sha256":hashlib.sha256(b_rgb.tobytes()).hexdigest(),"is_training_cache":False}
                    result["contact"] = contact(result, union_images)
                finally:
                    for p in paths.values():
                        if p.parent.resolve() not in checked_dirs or not p.resolve().is_relative_to(temp_root):
                            raise ValueError("Temporary cleanup path changed")
            if any(d.exists() for d in checked_dirs):
                raise ValueError("Temporary original directory remains")
            result["temporary_originals_removed"] = True
            result["elapsed_seconds"] = time.perf_counter() - started
            save(BASE / "records_r4" / (row["parent_id"] + ".json"), result)
            results.append(result)
            print(f"DONE {row['rank']}/8 {row['frames_25fps']} ROI frames; prior raw/reference comparisons passed; {result['elapsed_seconds']:.2f}s", flush=True)
        output = {"schema":"offline-native-diagnostic-result-v1", "status":"completed_review_only_original_roi_check",
            "plan_sha256":sha(plan_path), "records":results, "source_video_members":8, "source_label_members":8,
            "unique_archive_paths":plan["archive_paths"], "source_member_bytes_read":plan["planned_member_bytes"],
            "target_roi_frames":sum(r["target_frames"] for r in results),
            "grid_B_roi_frames":sum(r["grid_B"]["frames"] for r in results),
            "native_frames_decoded_in_spans":sum(r["decoded_native_frames_in_span"] for r in results),
            "retained_union_frames":sum(r["retained_union_frames"] for r in results),
            "historical_raw_frame_comparisons":sum(len(r["historical_comparisons"]) for r in results),
            "temporary_originals_remaining":len(list(temp_root.iterdir())), "free_bytes_after":shutil.disk_usage(BASE).free,
            "audio_members_read":0, "new_downloads":0, "neural_calls":0, "training_updates":0,
            "training_cache_count":0, "review_only":True, "split":"unassigned", "training_ready":False,
            "new_grid_pixels_human_approved":False, "raw_padding_frames":0}
        save(BASE / "native_check_summary_r4.json", output)
        print(json.dumps({k:output[k] for k in ("status","source_member_bytes_read","target_roi_frames","historical_raw_frame_comparisons","temporary_originals_remaining")}))
    except Exception as exc:
        raise RuntimeError("Original ROI diagnostic execution failed") from exc


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        frozen = prepare()
        print(json.dumps({k:frozen[k] for k in ("planned_member_bytes","peak_temporary_member_bytes","free_bytes_before","planned_target_roi_frames","planned_grid_B_roi_frames")}))
    else:
        execute()
