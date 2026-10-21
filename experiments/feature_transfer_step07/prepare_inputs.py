"""Prepare isolated, time-aligned student/teacher inputs from a frozen plan.

The student calls the existing production extraction helper unchanged. The
teacher receives dense native frames with two real context frames per side,
where available. Integer source-frame identity connects the two sequences.
No audio, test payload, model inference, or original-file mutation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
_legacy_path = ROOT / "experiments/legacy_recheck_step05/prepare_inputs.py"
_spec = importlib.util.spec_from_file_location("step05_original_preparation", _legacy_path)
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
from models.tokenizer import encode_text, normalize_text  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def anchor(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"Frozen file changed: {path}")


def json_new(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def json_progress(path: Path, data: dict) -> None:
    temporary = path.with_suffix(f".pending-{os.getpid()}.json")
    json_new(temporary, data)
    os.replace(temporary, path)


def interval_contract(start: float, end: float, fps: float, frame_count: int,
                      context: int = 2) -> dict:
    """Preserve legacy rounding; include official interval and real context."""
    if context != 2 or not all(math.isfinite(x) for x in (start, end, fps)):
        raise ValueError("Finite times and exactly two teacher context frames required")
    if start < 0 or end <= start or fps <= 0 or frame_count < 1 or end > frame_count / fps + 1e-6:
        raise ValueError("Interval outside source")
    times, selected = legacy.target_selection(start, end, fps)
    if not selected or selected[0] < 0 or selected[-1] >= frame_count:
        raise ValueError("Student target outside source")
    official_first = math.ceil(start * fps - 1e-9)
    official_stop = math.ceil(end * fps - 1e-9)
    if official_first < 0 or official_stop > frame_count or official_stop <= official_first:
        raise ValueError("Official half-open interval outside source")
    union_first = min(official_first, selected[0])
    union_stop = max(official_stop, selected[-1] + 1)
    first, stop = max(0, union_first - context), min(frame_count, union_stop + context)
    student = np.asarray(selected, dtype=np.int64)
    teacher = np.arange(first, stop, dtype=np.int64)
    mapping = student - first
    if not np.array_equal(teacher[mapping], student):
        raise ValueError("Integer frame mapping failed")
    return {"student_indices": student, "teacher_indices": teacher, "student_teacher_rows": mapping,
            "student_target_times": np.asarray(times, dtype=np.float64),
            "official_first": official_first, "official_stop": official_stop,
            "union_first": union_first, "union_stop": union_stop,
            "teacher_first": first, "teacher_stop": stop,
            "left_real_context": np.minimum(context, mapping).astype(np.int64),
            "right_real_context": np.minimum(context, len(teacher) - 1 - mapping).astype(np.int64),
            "context_clipped_left": first != union_first - context,
            "context_clipped_right": stop != union_stop + context}


def teacher_crop(frame: np.ndarray, box_xyxy: list) -> tuple[np.ndarray, list[int]]:
    """Match the official annotation path: integer truncation, BGR, margin0."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected uint8 BGR frame")
    if len(box_xyxy) != 4 or not np.isfinite(box_xyxy).all():
        raise ValueError("Invalid annotation box")
    left, top, right, bottom = map(int, box_xyxy)
    if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
        raise ValueError("Annotation crop outside source")
    return cv2.resize(frame[top:bottom, left:right], (96, 96), interpolation=cv2.INTER_LINEAR), [left, top, right, bottom]


def extract_teacher(capture, indices: np.ndarray, boxes: list, fps: float) -> tuple[np.ndarray, dict]:
    if indices.dtype != np.int64 or len(indices) == 0 or not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
        raise ValueError("Teacher indices must be a nonempty dense native interval")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    pixels, selected = [], []
    source_digest = hashlib.sha256()
    for index64 in indices:
        index = int(index64)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Teacher source decode failed at frame {index}")
        actual_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if (not math.isfinite(actual_after) or abs(actual_after - index - 1) > 1e-6
                or not math.isfinite(pts) or abs(pts - index / fps) > 1e-3):
            raise ValueError("Teacher decoder does not follow verified native CFR time")
        crop, rectangle = teacher_crop(frame, boxes[index])
        raw = frame.tobytes(order="C")
        source_digest.update(np.asarray([index], dtype="<i8").tobytes())
        source_digest.update(raw)
        selected.append({"source_frame_index": index, "actual_pts_seconds": pts,
                         "source_bgr_pixel_sha256": hashlib.sha256(raw).hexdigest(),
                         "annotation_box_index": index, "annotation_box_xyxy": boxes[index],
                         "final_rectangle_xyxy": rectangle})
        pixels.append(crop)
    return np.stack(pixels), {"selected": selected, "selected_source_pixels_sha256": source_digest.hexdigest()}


def validate_plan(plan_path: Path, expected: str) -> dict:
    anchor(plan_path, expected)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    settings = plan["preparation"]
    output = Path(settings["output_directory"]).resolve()
    if output != (BASE / "inputs").resolve():
        raise ValueError("Output must remain inside the isolated step07 inputs directory")
    for bound in [*plan["bindings"], *settings["code"]]:
        anchor(Path(bound["path"]), bound["sha256"])
    env = settings["environment"]
    if (env["opencv"], env["numpy"]) != (cv2.__version__, np.__version__):
        raise ValueError("Preparation runtime differs from frozen plan")
    if settings["teacher_context_frames"] != 2 or (legacy.FPS, legacy.ROI_SIZE, legacy.ROI_MARGIN) != (25, (96, 96), .20):
        raise ValueError("Frozen spatial or temporal contract differs")
    rows = plan["records"]
    if len(rows) != 512 or len({r["id"] for r in rows}) != 512 or {r["split"] for r in rows} != {"train", "dev"}:
        raise ValueError("Exactly fixed512 train/dev records required; test is prohibited")
    ids = {r["id"] for r in rows}
    for name, count in (("reproduction_ids", 4), ("pilot_ids", 8)):
        chosen = settings[name]
        if len(chosen) != count or len(set(chosen)) != count or not set(chosen) <= ids:
            raise ValueError(f"Invalid fixed {name}")
    for row in rows:
        if Path(row["id"]).name != row["id"] or any(ch in row["id"] for ch in ('/', '\\', ':')):
            raise ValueError("Unsafe record ID")
        if not math.isclose(row["start_time"], row["original_start_time"] + row["offset_seconds"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Corrected start disagrees with frozen source mapping")
        if not math.isclose(row["end_time"], row["original_end_time"] + row["offset_seconds"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("Corrected end disagrees with frozen source mapping")
        if normalize_text(row["source_text"]) != row["text"] or encode_text(row["text"]) != row["token_ids"]:
            raise ValueError("Frozen original text/token mismatch")
    return plan


def verify_payload(path: Path, row: dict, plan_sha: str) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        meta = json.loads(str(payload["metadata_json"].item()))
        if meta["plan_sha256"] != plan_sha or meta["id"] != row["id"] or meta["record"] != row:
            raise ValueError("Existing record was made under a different plan")
        if set(payload.files) != {"student_rgb", "teacher_bgr", "student_source_indices", "teacher_source_indices",
                                  "student_teacher_rows", "label_tokens", "metadata_json"}:
            raise ValueError("Unexpected payload fields")
        contract = interval_contract(row["start_time"], row["end_time"], row["video_fps"], meta["source_frame_count"])
        arrays = {key: payload[key] for key in payload.files if key != "metadata_json"}
        for key in ("student_rgb", "teacher_bgr"):
            frames = arrays[key]
            expected_len = len(contract["student_indices"] if key == "student_rgb" else contract["teacher_indices"])
            if frames.dtype != np.uint8 or frames.shape != (expected_len, 96, 96, 3):
                raise ValueError(f"Invalid {key} payload")
            if hashlib.sha256(frames.tobytes()).hexdigest() != meta[f"{key}_sha256"]:
                raise ValueError("Payload pixels changed")
        for key, expected_key in (("student_source_indices", "student_indices"), ("teacher_source_indices", "teacher_indices"),
                                  ("student_teacher_rows", "student_teacher_rows")):
            if arrays[key].dtype != np.int64 or not np.array_equal(arrays[key], contract[expected_key]):
                raise ValueError("Saved frame linkage differs from time contract")
        if arrays["label_tokens"].dtype != np.int64 or arrays["label_tokens"].tolist() != row["token_ids"]:
            raise ValueError("Saved text tokens differ")
        if len(arrays["student_rgb"]) != row["frames"]:
            raise ValueError("Student frame count changed")
        student_trace, teacher_trace = meta["student_trace"]["selected"], meta["teacher_trace"]["selected"]
        if len(student_trace) != row["frames"] or len(teacher_trace) != len(arrays["teacher_bgr"]):
            raise ValueError("Saved trace length mismatch")
        for i, s in enumerate(student_trace):
            t = teacher_trace[int(arrays["student_teacher_rows"][i])]
            if (s["source_frame_index"] != int(arrays["student_source_indices"][i])
                    or t["source_frame_index"] != s["source_frame_index"]
                    or t["source_bgr_pixel_sha256"] != s["source_bgr_pixel_sha256"]):
                raise ValueError("Teacher/student source pixels differ at the mapped frame")
    return meta


def metadata_record(path: Path, meta: dict) -> dict:
    return {"id": meta["id"], "split": meta["record"]["split"], "source_stem": meta["record"]["source_stem"],
            "plan_sha256": meta["plan_sha256"], "npz_path": str(path.resolve()), "npz_sha256": sha256(path),
            "student_frames": meta["record"]["frames"], "teacher_frames": len(meta["teacher_trace"]["selected"]),
            "student_rgb_sha256": meta["student_rgb_sha256"], "teacher_bgr_sha256": meta["teacher_bgr_sha256"],
            "source_frame_count": meta["source_frame_count"], "video_fps": meta["record"]["video_fps"],
            "teacher_context": meta["teacher_context"], "source_verification": meta["source_verification"],
            "preparation_seconds": meta["preparation_seconds"]}


def existing_record(output: Path, row: dict, plan_sha: str) -> dict | None:
    path = output / "records" / f"{row['id']}.npz"
    description = output / "metadata" / f"{row['id']}.json"
    if not path.exists():
        if description.exists():
            raise ValueError("Existing metadata has no corresponding payload")
        return None
    meta = verify_payload(path, row, plan_sha)
    result = metadata_record(path, meta)
    if description.exists():
        if json.loads(description.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing record metadata/hash differs")
    else:
        # The NPZ is self-describing, so interruption between its publication and
        # metadata publication can recover without rereading the source.
        json_new(description, result)
    return result


def source_stat(path: Path, row: dict) -> tuple[int, int]:
    state = path.stat()
    pair = (state.st_size, state.st_mtime_ns)
    if pair != (row["video_size_bytes"], row["video_mtime_ns"]):
        raise ValueError("Source video size/mtime differs from audited inventory")
    return pair


def prepare_source(rows: list[dict], plan_sha: str, reproduction: bool) -> list[tuple[dict, dict]]:
    first = rows[0]
    video, label_path = Path(first["video_path"]), Path(first["label_path"])
    if any((r["video_path"], r["label_path"], r["label_sha256"], r["video_size_bytes"], r["video_mtime_ns"])
           != (first["video_path"], first["label_path"], first["label_sha256"], first["video_size_bytes"], first["video_mtime_ns"]) for r in rows):
        raise ValueError("Inconsistent metadata within a recording")
    before = source_stat(video, first)
    anchor(label_path, first["label_sha256"])
    raw_label = label_path.read_bytes()
    if hashlib.sha256(raw_label).hexdigest() != first["label_sha256"]:
        raise ValueError("Label changed while loading")
    label = json.loads(raw_label.decode("utf-8-sig"))
    if isinstance(label, list):
        if len(label) != 1:
            raise ValueError("Expected exactly one recording label")
        label = label[0]
    boxes, label_fps = legacy._extract_lip_boxes(label, "yxyx")
    if boxes is None:
        raise ValueError("Annotation boxes required; no automatic fallback")
    sentences = {s["ID"]: s for s in label["Sentence_info"]}
    if len(sentences) != len(label["Sentence_info"]):
        raise ValueError("Duplicate sentence IDs")
    capture = cv2.VideoCapture(str(video))
    outputs = []
    try:
        if not capture.isOpened():
            raise ValueError("Cannot open selected source video")
        fps, frame_count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(boxes) != frame_count:
            raise ValueError("Native video and annotation frame counts differ")
        for row in rows:
            started = time.perf_counter()
            s = sentences[row["sentence_id"]]
            if (s["sentence_text"], s["start_time"], s["end_time"]) != (row["source_text"], row["original_start_time"], row["original_end_time"]):
                raise ValueError("Original transcript or times changed")
            if not math.isclose(fps, row["video_fps"], rel_tol=1e-12):
                raise ValueError("Native source FPS changed")
            if reproduction:
                cached_path = Path(row["legacy_cache_path"])
                anchor(cached_path, row["legacy_cache_sha256"])
                pixels, trace = legacy.extract(capture, row["original_start_time"], row["original_end_time"], boxes, label_fps)
                cached = legacy.load_clip(cached_path)
                legacy.require_exact_reproduction(pixels, cached, row["text"], row["source_text"])
                if len(pixels) != row["frames"]:
                    raise ValueError("Original student frame count differs")
                outputs.append((row, {"id": row["id"], "plan_sha256": plan_sha, "pixel_exact": True,
                    "text_and_tokens_exact": True, "legacy_cache_path": str(cached_path),
                    "legacy_cache_sha256": row["legacy_cache_sha256"], "frames": len(pixels),
                    "student_rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(), "trace": trace}))
                continue
            contract = interval_contract(row["start_time"], row["end_time"], fps, frame_count)
            student, student_trace = legacy.extract(capture, row["start_time"], row["end_time"], boxes, label_fps)
            if len(student) != row["frames"]:
                raise ValueError("Corrected student frame count differs from legacy duration")
            teacher, teacher_trace = extract_teacher(capture, contract["teacher_indices"], boxes, fps)
            for index, trace in enumerate(student_trace["selected"]):
                mapped = teacher_trace["selected"][int(contract["student_teacher_rows"][index])]
                if (trace["source_frame_index"], trace["source_bgr_pixel_sha256"]) != (mapped["source_frame_index"], mapped["source_bgr_pixel_sha256"]):
                    raise ValueError("Independently decoded teacher/student source pixels differ")
            meta = {"schema": "feature-transfer-step07-input-v1", "id": row["id"], "record": row,
                    "plan_sha256": plan_sha, "source_frame_count": frame_count,
                    "student_rgb_sha256": hashlib.sha256(student.tobytes()).hexdigest(),
                    "teacher_bgr_sha256": hashlib.sha256(teacher.tobytes()).hexdigest(),
                    "student_trace": student_trace, "teacher_trace": teacher_trace,
                    "teacher_context": {"requested_real_frames_each_side": 2,
                        "official_first": contract["official_first"], "official_stop_exclusive": contract["official_stop"],
                        "student_union_first": contract["union_first"], "student_union_stop_exclusive": contract["union_stop"],
                        "actual_first": contract["teacher_first"], "actual_stop_exclusive": contract["teacher_stop"],
                        "clipped_left": contract["context_clipped_left"], "clipped_right": contract["context_clipped_right"],
                        "student_left_real_context": contract["left_real_context"].tolist(),
                        "student_right_real_context": contract["right_real_context"].tolist(),
                        "boundary_policy": "clip to available native frames; teacher front retains its own configured zero padding"},
                    "preparation_seconds": time.perf_counter() - started}
            arrays = {"student_rgb": student, "teacher_bgr": teacher,
                      "student_source_indices": contract["student_indices"], "teacher_source_indices": contract["teacher_indices"],
                      "student_teacher_rows": contract["student_teacher_rows"], "label_tokens": np.asarray(row["token_ids"], dtype=np.int64)}
            outputs.append((row, {"arrays": arrays, "meta": meta}))
    finally:
        capture.release()
    after = source_stat(video, first)
    anchor(label_path, first["label_sha256"])
    if before != after:
        raise ValueError("Video changed during source extraction")
    verification = {"video_size_bytes": after[0], "video_mtime_ns": after[1],
                    "label_sha256_before_and_after": first["label_sha256"], "source_video_unchanged": True}
    for _, value in outputs:
        (value if reproduction else value["meta"])["source_verification"] = verification
    return outputs


def write_record(output: Path, row: dict, value: dict) -> dict:
    path = output / "records" / f"{row['id']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(f".pending-{os.getpid()}.npz")
    with pending.open("xb") as stream:
        np.savez_compressed(stream, **value["arrays"], metadata_json=json.dumps(value["meta"], ensure_ascii=False, allow_nan=False))
    verify_payload(pending, row, value["meta"]["plan_sha256"])
    # Path.rename refuses replacement on Windows; the explicit existence check
    # additionally documents that previously accepted payloads are immutable.
    if path.exists():
        raise FileExistsError(path)
    pending.rename(path)
    result = metadata_record(path, value["meta"])
    json_new(output / "metadata" / f"{row['id']}.json", result)
    return result


def contact_sheet(output: Path, rows: list[dict]) -> Path:
    path = output / "pilot_contact_sheet.png"
    if path.exists():
        return path
    sheet = np.full((len(rows) * 132 + 32, 640, 3), 245, np.uint8)
    cv2.putText(sheet, "student first/mid/last | mapped teacher mid", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1)
    for i, row in enumerate(rows):
        with np.load(output / "records" / f"{row['id']}.npz", allow_pickle=False) as data:
            student, teacher, mapping = data["student_rgb"], data["teacher_bgr"], data["student_teacher_rows"]
            mid = len(student) // 2
            crops = [cv2.cvtColor(student[j], cv2.COLOR_RGB2BGR) for j in (0, mid, len(student) - 1)] + [teacher[int(mapping[mid])]]
            y = 32 + i * 132
            cv2.putText(sheet, row["id"], (8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (20, 20, 20), 1)
            for col, crop in enumerate(crops):
                sheet[y + 24:y + 120, 8 + col * 150:104 + col * 150] = crop
    ok, encoded = cv2.imencode(".png", sheet)
    if not ok:
        raise ValueError("Contact sheet encoding failed")
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())
    return path


def prepare(plan_path: Path, plan_sha: str, phase: str) -> dict:
    plan = validate_plan(plan_path, plan_sha)
    output = Path(plan["preparation"]["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    rows, settings = plan["records"], plan["preparation"]
    rows_by_id = {r["id"]: r for r in rows}
    reproduction_path, pilot_path = output / "reproduction_gate.json", output / "pilot_gate.json"
    started = time.perf_counter()
    if phase == "reproduction":
        if reproduction_path.exists():
            prior = json.loads(reproduction_path.read_text(encoding="utf-8"))
            if prior["plan_sha256"] != plan_sha or prior["ids"] != settings["reproduction_ids"] or prior["status"] != "passed":
                raise ValueError("Existing reproduction gate differs")
            return prior
        selected = [rows_by_id[sid] for sid in settings["reproduction_ids"]]
        groups = defaultdict(list)
        for row in selected:
            groups[row["video_path"]].append(row)
        controls = []
        for group in groups.values():
            for row, result in prepare_source(group, plan_sha, reproduction=True):
                controls.append(result)
                print(f"[reproduced] {row['id']} pixel-exact", flush=True)
        controls.sort(key=lambda item: settings["reproduction_ids"].index(item["id"]))
        result = {"schema": "feature-transfer-step07-reproduction-v1", "status": "passed", "plan_sha256": plan_sha,
                  "ids": settings["reproduction_ids"], "records": controls, "elapsed_seconds": time.perf_counter() - started}
        json_new(reproduction_path, result)
        return result
    if not reproduction_path.exists():
        raise ValueError("Run and pass the four exact original-pixel controls first")
    gate = json.loads(reproduction_path.read_text(encoding="utf-8"))
    if gate["status"] != "passed" or gate["plan_sha256"] != plan_sha or gate["ids"] != settings["reproduction_ids"]:
        raise ValueError("Reproduction gate is not for this plan")
    selected = [rows_by_id[sid] for sid in settings["pilot_ids"]] if phase == "pilot" else rows
    if phase == "all":
        if not pilot_path.exists():
            raise ValueError("Complete and review the fixed eight-input pilot before full extraction")
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot["status"] != "passed" or pilot["plan_sha256"] != plan_sha or pilot["ids"] != settings["pilot_ids"]:
            raise ValueError("Pilot gate is not for this plan")
    completed, pending = [], []
    for row in selected:
        existing = existing_record(output, row, plan_sha)
        (completed if existing is not None else pending).append(existing if existing is not None else row)
    if phase == "verify" and pending:
        raise ValueError("Cannot verify an incomplete512 cohort")
    groups = defaultdict(list)
    for row in pending:
        groups[row["video_path"]].append(row)
    for group in groups.values():
        if shutil.disk_usage(output).free < 10 * 1024**3:
            raise ValueError("Less than10 GiB free; stop before reading another source")
        values = prepare_source(group, plan_sha, reproduction=False)
        for row, value in values:
            completed.append(write_record(output, row, value))
            print(f"[input] {len(completed)}/{len(selected)} {row['id']} student={row['frames']} teacher={len(value['arrays']['teacher_bgr'])}", flush=True)
        json_progress(output / "progress.json", {"phase": phase, "plan_sha256": plan_sha, "completed": len(completed),
                      "target": len(selected), "last_source": group[0]["source_stem"], "elapsed_seconds": time.perf_counter() - started})
    completed.sort(key=lambda item: [r["id"] for r in selected].index(item["id"]))
    result = {"schema": "feature-transfer-step07-input-manifest-v1", "status": "passed", "phase": phase,
              "created_utc": datetime.now(timezone.utc).isoformat(), "plan_sha256": plan_sha,
              "record_count": len(completed), "ids": [r["id"] for r in selected], "records": completed,
              "student_frames": sum(r["student_frames"] for r in completed), "teacher_frames": sum(r["teacher_frames"] for r in completed),
              "source_recordings": len({r["source_stem"] for r in completed}), "elapsed_seconds": time.perf_counter() - started,
              "reproduction_gate_sha256": sha256(reproduction_path), "audio_read": False, "test_payloads_read": False,
              "original_files_modified": False, "teacher_model_run": False, "student_model_run": False}
    if phase == "pilot":
        sheet = contact_sheet(output, selected)
        result["contact_sheet"] = {"path": str(sheet), "sha256": sha256(sheet)}
        if not pilot_path.exists():
            json_new(pilot_path, result)
        pilot_manifest = output / "pilot_manifest.json"
        if not pilot_manifest.exists():
            json_new(pilot_manifest, result)
        else:
            prior = json.loads(pilot_manifest.read_text(encoding="utf-8"))
            if prior["plan_sha256"] != plan_sha or prior["records"] != completed:
                raise ValueError("Existing pilot manifest differs")
    elif phase == "all":
        destination = output / "manifest.json"
        if not destination.exists():
            json_new(destination, result)
        else:
            prior = json.loads(destination.read_text(encoding="utf-8"))
            if prior["plan_sha256"] != plan_sha or prior["records"] != completed:
                raise ValueError("Existing full manifest differs")
    else:
        destination = output / "verification.json"
        if not destination.exists():
            json_new(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=BASE / "data_plan.json")
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--phase", choices=["reproduction", "pilot", "all", "verify"], required=True)
    args = parser.parse_args()
    result = prepare(args.plan, args.plan_sha256, args.phase)
    print(json.dumps({key: result[key] for key in ("status", "plan_sha256", "ids")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
