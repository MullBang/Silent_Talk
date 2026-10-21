"""Independent Step9 arrays/geometry/source audit. No production imports.

All cached rows are checked against source label boxes and 360 older PNGs.
One predefined mixed candidate/hold source is separately materialized from D:
and decoded/cropped here to close the main array-only provenance gap.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

import audit_selection as selection
from audit_selection import BASE, ROOT, DATA_SHA, same, near, local, sha, binding, read, bound, tokens, ctc_min

REPRESENTATIVE_SOURCE = "lip_J_1_F_02_C045_A_007"
STUDENT_KEYS = {"student_rgb", "student_source_indices", "label_tokens", "metadata_json"}
TEACHER_KEYS = {"teacher_bgr", "teacher_source_indices", "student_teacher_rows"}


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def interval(start, end, fps, frame_count):
    if not all(math.isfinite(x) for x in (start, end, fps)) or not 0 <= start < end <= frame_count / fps + 1e-6:
        raise ValueError("Invalid physical interval")
    count = round((end - start) * 25)
    if count < 1:
        raise ValueError("Empty student sequence")
    student = np.asarray([round((start + k / 25) * fps) for k in range(count)], dtype=np.int64)
    if student[0] < 0 or student[-1] >= frame_count:
        raise ValueError("Student index outside source")
    official_first, official_stop = math.ceil(start * fps - 1e-9), math.ceil(end * fps - 1e-9)
    union_first, union_stop = min(official_first, int(student[0])), max(official_stop, int(student[-1]) + 1)
    first, stop = max(0, union_first - 2), min(frame_count, union_stop + 2)
    teacher = np.arange(first, stop, dtype=np.int64)
    mapping = student - first
    context = {"requested_real_frames_each_side": 2, "official_first": official_first,
               "official_stop_exclusive": official_stop, "student_union_first": union_first,
               "student_union_stop_exclusive": union_stop, "actual_first": first, "actual_stop_exclusive": stop,
               "clipped_left": first != union_first - 2, "clipped_right": stop != union_stop + 2,
               "student_left_real_context": np.minimum(2, mapping).tolist(),
               "student_right_real_context": np.minimum(2, len(teacher) - 1 - mapping).tolist(),
               "boundary_policy": "clip to available native frames; teacher front retains its own configured zero padding"}
    return student, teacher, mapping, context


def rectangle(box, width, height, margin):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError("Invalid source box")
    x1, y1, x2, y2 = map(float, np.asarray(box, dtype=np.float32))
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("Box outside image")
    if margin == 0:
        return [int(x1), int(y1), int(x2), int(y2)]
    left = max(0, min(math.floor(x1 - (x2 - x1) * margin), width - 1))
    top = max(0, min(math.floor(y1 - (y2 - y1) * margin), height - 1))
    right = max(left + 1, min(math.ceil(x2 + (x2 - x1) * margin), width))
    bottom = max(top + 1, min(math.ceil(y2 + (y2 - y1) * margin), height))
    return [left, top, right, bottom]


def recrop(frame_bgr, box, student):
    left, top, right, bottom = rectangle(box, frame_bgr.shape[1], frame_bgr.shape[0], .2 if student else 0)
    # Crop then swap channels independently from the producer's full-frame swap.
    crop = frame_bgr[top:bottom, left:right]
    if student:
        crop = np.ascontiguousarray(crop[:, :, ::-1])
    return cv2.resize(crop, (96, 96), interpolation=cv2.INTER_LINEAR)


def source_boxes(source):
    label = bound(source["label_copy"])
    if not isinstance(label, list) or len(label) != 1:
        raise ValueError("Unexpected source label wrapper")
    yxyx = np.asarray(label[0]["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"], dtype=np.float64)
    same(yxyx.shape, (source["video_frame_count"], 4), "Full source annotation shape")
    if not np.isfinite(yxyx).all():
        raise ValueError("Nonfinite source annotations")
    return yxyx[:, [1, 0, 3, 2]]


def verify_trace(trace, indices, row, boxes, source, student):
    selected = trace["selected"]
    same(len(selected), len(indices), "Trace length")
    if not digest_valid(trace["selected_source_pixels_sha256"]):
        raise ValueError("Invalid aggregate source digest")
    if student:
        same(trace["native_fps"], row["video_fps"], "Native trace fps")
        same(trace["target_fps"], 25, "Student target fps")
        same(trace["selected_count"], len(indices), "Selected trace count")
        if not len(indices) <= trace["source_frames_decoded"] <= source["video_frame_count"]:
            raise ValueError("Invalid decoded trace count")
    for k, (item, native) in enumerate(zip(selected, indices, strict=True)):
        native = int(native)
        same(item["source_frame_index"], native, "Native trace index")
        same(item["annotation_box_index"], native, "Annotation frame index")
        same(item["annotation_box_xyxy"], boxes[native].tolist(), "Raw JSON box identity")
        near(item["actual_pts_seconds"], native / row["video_fps"], .001, "Native PTS")
        if not digest_valid(item["source_bgr_pixel_sha256"]):
            raise ValueError("Invalid raw pixel digest")
        crop = rectangle(boxes[native], source["video_metadata"]["width"], source["video_metadata"]["height"], .2 if student else 0)
        if student:
            target = row["start_time"] + k / 25
            near(item["target_video_seconds"], target, 1e-12, "25fps target time")
            near(item["actual_pts_seconds"], target, .5 / row["video_fps"] + 1e-6, "Nearest-frame PTS")
        else:
            same(item["final_rectangle_xyxy"], crop, "Teacher native bbox integer crop")


def verify_arrays(arrays, meta, row, entry, boxes, source):
    candidate = row["teacher_candidate"]
    same(type(candidate), bool, "Boolean candidate")
    same(candidate, row["split"] == "train" and row["comparison_group"] == "both_supported", "Teacher train-only rule")
    same(set(arrays), STUDENT_KEYS - {"metadata_json"} | (TEACHER_KEYS if candidate else set()), "Exact cache fields")
    same(meta["record"], row, "Embedded frozen record")
    same(meta["id"], row["id"], "Embedded ID")
    same(meta["teacher_pixels_persisted"], candidate, "Teacher pixels role")
    same(meta["schema"], "lipnet-step09-paired-input-v1" if candidate else "lipnet-step09-student-input-v1", "Cache schema")
    for k in ("training_ready", "full_transcript_human_approved", "human_boundary_approved", "audio_used_as_model_input", "teacher_model_run"):
        same(meta[k], False, "No scope/approval expansion")
    same(meta["source_frame_count"], source["video_frame_count"], "Physical source count")
    same(row["source_frame_count"], source["video_frame_count"], "Plan source count")
    same(row["video_fps"], source["video_fps"], "Physical source fps")
    student, teacher, mapping, context = interval(row["start_time"], row["end_time"], row["video_fps"], meta["source_frame_count"])
    same(len(student), row["frames"], "Student frame count")
    _, label_tokens = tokens(row["text"])
    same(label_tokens, row["token_ids"], "Independent CTC labels")
    same(ctc_min(label_tokens), row["ctc_min_frames"], "CTC minimum")
    if ctc_min(label_tokens) > len(student):
        raise ValueError("CTC infeasible")
    integer_arrays = {"student_source_indices": student, "label_tokens": np.asarray(label_tokens, dtype=np.int64)}
    if candidate:
        integer_arrays.update(teacher_source_indices=teacher, student_teacher_rows=mapping)
    for key, expected in integer_arrays.items():
        same(arrays[key].dtype, np.dtype("int64"), key + " dtype")
        np.testing.assert_array_equal(arrays[key], expected, err_msg=key)
    for key, count in [("student_rgb", len(student))] + ([("teacher_bgr", len(teacher))] if candidate else []):
        same(arrays[key].dtype, np.dtype("uint8"), key + " dtype")
        same(arrays[key].shape, (count, 96, 96, 3), key + " shape")
        digest = pixel_sha(arrays[key])
        same(digest, meta[key + "_sha256"], key + " embedded digest")
        same(digest, entry[key + "_sha256"], key + " manifest digest")
    verify_trace(meta["student_trace"], student, row, boxes, source, True)
    if candidate:
        same(meta["teacher_context"], context, "Real frame context coverage")
        verify_trace(meta["teacher_trace"], teacher, row, boxes, source, False)
        np.testing.assert_array_equal(teacher[mapping], student, err_msg="Student/teacher native correspondence")
        for student_item, pos in zip(meta["student_trace"]["selected"], mapping, strict=True):
            teacher_item = meta["teacher_trace"]["selected"][int(pos)]
            same(student_item["source_bgr_pixel_sha256"], teacher_item["source_bgr_pixel_sha256"], "Paired raw pixel identity")
    else:
        for k in ("teacher_trace", "teacher_context", "teacher_bgr_sha256"):
            if k in meta or k in entry:
                raise ValueError("Teacher artifacts attached to held/development input")
    same(entry["teacher_frames"], len(teacher) if candidate else 0, "Stored teacher frame count")
    return student, teacher if candidate else np.empty(0, np.int64)


def verify_samples(rgb, meta, clip, row, source, boxes):
    samples = clip["roi_samples"]
    same([s["student_index"] for s in samples], [0, row["frames"] // 2, row["frames"] - 1], "Step6 three samples")
    for sample in samples:
        k = sample["student_index"]
        trace = meta["student_trace"]["selected"][k]
        same(trace["source_frame_index"], sample["native_frame_index"], "Step6 native sample index")
        same(trace["source_bgr_pixel_sha256"], sample["raw_frame_array"]["sha256"], "Step6 raw pixel source identity")
        same(trace["annotation_box_xyxy"], sample["bbox_xyxy"], "Step6 annotation identity")
        near(trace["actual_pts_seconds"], sample["decoded_pts_seconds"], 1e-9, "Step6 sample PTS")
        near(trace["target_video_seconds"], sample["target_time_seconds"], 1e-9, "Step6 target time")
        same(rectangle(boxes[sample["native_frame_index"]], source["video_metadata"]["width"], source["video_metadata"]["height"], .2), sample["crop_xyxy"], "Step6 independently derived margin rectangle")
        image_path = local(sample["image"]["path"])
        same(sha(image_path), sample["image"]["sha256"], "Step6 PNG bytes")
        png_bgr = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if png_bgr is None:
            raise ValueError("PNG decode failed")
        np.testing.assert_array_equal(rgb[k], png_bgr[:, :, ::-1], err_msg="Student RGB versus previous BGR PNG")
        same(pixel_sha(png_bgr), sample["roi_array"]["sha256"], "Step6 PNG decoded pixel hash")
        same(pixel_sha(png_bgr), sample["image"]["pixels"]["sha256"], "Step6 image pixel hash")
    return len(samples)


def verify_source_claim(claim, source, data_ref, execution_ref):
    expected = {"transport": "audited-local-tar-v1", "source_stem": source["stem"], "data_plan": data_ref,
                "execution_plan": execution_ref, "members": {k: source["members"][k] for k in ("video", "label")},
                "archive_stats_checked_before_and_after": True, "member_hashes_checked_on_materialization": True,
                "temporary_payload_cleaned_before_publish": True, "temporary_paths_persisted": False,
                "audio_used_as_model_input": False, "G_drive_accessed": False}
    same(claim, expected, "Source materialization declaration")


def representative_redecode(plan, entries, boxes_by_source):
    """Direct bounded D read and independent decode/crop of all selected frames of one source."""
    source = next(s for s in plan["sources"] if s["stem"] == REPRESENTATIVE_SOURCE)
    rows = [r for r in plan["records"] if r["source_stem"] == REPRESENTATIVE_SOURCE]
    same(len(rows), 4, "Representative source frozen four clips")
    same(sum(r["teacher_candidate"] for r in rows), 2, "Representative paired/held balance")
    member = source["members"]["video"]
    archive = Path(member["archive_path"])
    if archive.drive.upper() != "D:" or archive.suffix != ".tar":
        raise ValueError("Representative read must use planned D TAR")
    before = archive.stat()
    same((before.st_size, before.st_mtime_ns), (member["archive_size"], member["archive_mtime_ns"]), "Representative archive identity")
    temporary_root = local(BASE / "independent_temporary")
    temporary_root.mkdir(exist_ok=True)
    temp = (temporary_root / f"{REPRESENTATIVE_SOURCE}-{uuid.uuid4().hex}.mp4").resolve()
    if not temp.is_relative_to(temporary_root):
        raise ValueError("Temporary path leaves audited folder")
    streamed = hashlib.sha256()
    copied = 0
    arrays_by_id = {}
    metas = {}
    needed = {}
    for row in rows:
        e = entries[row["id"]]
        with np.load(local(e["npz_path"]), allow_pickle=False) as p:
            arrays_by_id[row["id"]] = {k: p[k] for k in p.files if k != "metadata_json"}
            metas[row["id"]] = json.loads(str(p["metadata_json"].item()))
        for k, native in enumerate(arrays_by_id[row["id"]]["student_source_indices"]):
            needed.setdefault(int(native), []).append((row["id"], "student", k))
        if row["teacher_candidate"]:
            for k, native in enumerate(arrays_by_id[row["id"]]["teacher_source_indices"]):
                needed.setdefault(int(native), []).append((row["id"], "teacher", k))
    cap = None
    try:
        with archive.open("rb") as original, temp.open("xb") as copied_file:
            original.seek(member["offset"])
            remaining = member["size"]
            while remaining:
                block = original.read(min(2**20, remaining))
                if not block:
                    raise ValueError("Short source member read")
                copied_file.write(block); streamed.update(block)
                copied += len(block); remaining -= len(block)
        same(streamed.hexdigest(), member["sha256"], "Independent D member SHA")
        cap = cv2.VideoCapture(str(temp))
        if not cap.isOpened():
            raise ValueError("Independent decoder could not open source")
        same(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), source["video_frame_count"], "Independent frame count")
        near(float(cap.get(cv2.CAP_PROP_FPS)), source["video_fps"], 1e-12, "Independent source fps")
        current = None
        raw_digests = {}
        pixel_checks = Counter()
        aggregate = {(r["id"], role): hashlib.sha256() for r in rows for role in (["student", "teacher"] if r["teacher_candidate"] else ["student"])}
        for native in sorted(needed):
            if current != native:
                cap.set(cv2.CAP_PROP_POS_FRAMES, native)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Independent decode failed: {native}")
            current = native + 1
            same(int(cap.get(cv2.CAP_PROP_POS_FRAMES)), current, "Independent decoder frame index")
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            near(pts, native / source["video_fps"], .001, "Independent decoder PTS")
            raw = frame.tobytes()
            digest = hashlib.sha256(raw).hexdigest()
            raw_digests[native] = digest
            for identity, role, k in needed[native]:
                meta = metas[identity]
                trace = meta[role + "_trace"]["selected"][k]
                same(digest, trace["source_bgr_pixel_sha256"], "Independent raw pixel hash")
                near(pts, trace["actual_pts_seconds"], 1e-9, "Independent exact trace PTS")
                crop = recrop(frame, boxes_by_source[REPRESENTATIVE_SOURCE][native], role == "student")
                np.testing.assert_array_equal(crop, arrays_by_id[identity]["student_rgb" if role == "student" else "teacher_bgr"][k], err_msg="Independent original-source recrop")
                aggregate[(identity, role)].update(np.asarray([native], dtype="<i8").tobytes())
                aggregate[(identity, role)].update(raw)
                pixel_checks[role] += 1
        for (identity, role), digest in aggregate.items():
            same(digest.hexdigest(), metas[identity][role + "_trace"]["selected_source_pixels_sha256"], "Independent aggregate raw source pixel hash")
        after = archive.stat()
        same((after.st_size, after.st_mtime_ns), (before.st_size, before.st_mtime_ns), "Archive stable across independent read")
        result = {"status": "passed", "source_stem": REPRESENTATIVE_SOURCE, "record_ids": [r["id"] for r in rows],
                  "candidate_records": 2, "held_records": 2, "source_member": member, "read_bytes": copied,
                  "unique_native_frames_decoded": len(raw_digests), "student_frames_recropped": pixel_checks["student"],
                  "teacher_frames_recropped": pixel_checks["teacher"], "raw_pixel_and_roi_arrays_exact": True,
                  "aggregate_raw_pixel_digests_exact": True, "temporary_copy_removed": True, "audio_read": False,
                  "limitations": ["One predefined mixed-role source was redecoded; 29 other sources use trace, label and previous-sample checks."]}
    finally:
        if cap is not None:
            cap.release()
        if temp.exists():
            # Single known file deletion, never a recursively computed target.
            temp.unlink()
        if temporary_root.exists() and not any(temporary_root.iterdir()):
            temporary_root.rmdir()
    return result


def audit(manifest_path, expected_sha, redecode=True):
    started = time.perf_counter()
    plan_path = BASE / "data_plan.json"
    plan = read(plan_path, DATA_SHA)
    selection.audit(plan_path)
    selection_path = BASE / "selection_audit.json"
    selected = read(selection_path)
    same(selected["status"], "passed", "Selection audit")
    same(selected["data_plan"], binding(plan_path), "Selection plan binding")
    same(selected["code"], binding(BASE / "audit_selection.py"), "Selection auditor unchanged")
    manifest_path = local(manifest_path)
    same(manifest_path, BASE / "inputs/manifest.json", "Fixed manifest path")
    manifest = read(manifest_path, expected_sha)
    same(manifest["schema"], "lipnet-step09-input-manifest-v1", "Input schema")
    same(manifest["status"], "passed", "Producer completion")
    data_ref = binding(plan_path)
    same(manifest["data_plan"], data_ref, "Manifest plan")
    execution_ref = manifest["execution_plan"]
    same(local(execution_ref["path"]), BASE / "input_execution_plan.json", "Execution plan path")
    execution = bound(execution_ref)
    same(execution["data_plan"], data_ref, "Execution data plan")
    for ref in execution["code"].values() if isinstance(execution["code"], dict) else execution["code"]:
        same(sha(ref["path"]), ref["sha256"], "Frozen execution code")
    same(manifest["ids"], plan["ids"], "Manifest exact ID order")
    same([e["id"] for e in manifest["records"]], plan["ids"], "Manifest row order")
    same(execution["ids"], plan["ids"], "Execution row order")
    same(execution["teacher_candidate_ids"], plan["teacher_candidate_ids"], "Execution teacher role IDs")
    source_by_id = {s["stem"]: s for s in plan["sources"]}
    media = bound(plan["inputs"]["media_manifest"])
    clips = {r["id"]: r for r in media["clips"]}
    boxes_by_source = {s["stem"]: source_boxes(s) for s in plan["sources"]}
    entries = {e["id"]: e for e in manifest["records"]}
    verified = []
    for row, entry in zip(plan["records"], manifest["records"], strict=True):
        source = source_by_id[row["source_stem"]]
        boxes = boxes_by_source[row["source_stem"]]
        path = local(entry["npz_path"])
        same(path, BASE / "inputs/records" / (row["id"] + ".npz"), "Isolated NPZ path")
        same(sha(path), entry["npz_sha256"], "Input archive hash")
        sidecar = BASE / "inputs/metadata" / (row["id"] + ".json")
        same(read(sidecar), entry, "Per-row manifest sidecar")
        with np.load(path, allow_pickle=False) as payload:
            same(set(payload.files), STUDENT_KEYS | (TEACHER_KEYS if row["teacher_candidate"] else set()), "NPZ precise key roles")
            arrays = {k: payload[k] for k in payload.files if k != "metadata_json"}
            meta = json.loads(str(payload["metadata_json"].item()))
        same(meta["data_plan"], data_ref, "Embedded data reference")
        same(meta["execution_plan"], execution_ref, "Embedded execution reference")
        same(meta["plan_sha256"], DATA_SHA, "Embedded logical plan SHA")
        student, teacher = verify_arrays(arrays, meta, row, entry, boxes, source)
        same(student.tolist(), clips[row["id"]]["timing"]["native_frame_indices"], "All Step6 frame identities")
        n_samples = verify_samples(arrays["student_rgb"], meta, clips[row["id"]], row, source, boxes)
        verify_source_claim(meta["source_verification"], source, data_ref, execution_ref)
        same(entry["source_verification"], meta["source_verification"], "Manifest source provenance")
        for key, value in (("split", row["split"]), ("speaker_id", row["speaker_id"]), ("source_stem", row["source_stem"]), ("student_frames", len(student)), ("teacher_candidate", row["teacher_candidate"]), ("source_frame_count", source["video_frame_count"]), ("video_fps", source["video_fps"]), ("plan_sha256", DATA_SHA), ("training_ready", False)):
            same(entry[key], value, "Manifest " + key)
        verified.append({"id": row["id"], "split": row["split"], "teacher_candidate": row["teacher_candidate"],
                         "student_frames": len(student), "teacher_frames": len(teacher), "npz": binding(path),
                         "metadata": binding(sidecar), "step06_png_samples_verified": n_samples,
                         "all_frame_indices_pts_and_source_label_boxes_checked": True,
                         "training_ready": False})
        if len(verified) % 20 == 0:
            print(json.dumps({"independent_input_audit": len(verified), "of": 120}), flush=True)
    expected_names = {r["id"] + ".npz" for r in plan["records"]}
    same({p.name for p in (BASE / "inputs/records").glob("*.npz")}, expected_names, "No extra cached rows")
    same(manifest["record_count"], 120, "120 inputs")
    same(manifest["paired_input_count"], 69, "69 paired inputs")
    same(manifest["student_only_count"], 51, "51 student-only inputs")
    same(manifest["student_frames"], sum(r["student_frames"] for r in verified), "Total student frames")
    same(manifest["teacher_frames"], sum(r["teacher_frames"] for r in verified), "Total teacher frames")
    for key, value in (("training_ready", False), ("student_optimization_steps", 0), ("teacher_model_runs", 0), ("audio_used_as_model_input", False), ("G_drive_accessed", False), ("temporary_payload_files_remaining", 0)):
        same(manifest[key], value, "Input scope " + key)
    temporary = BASE / "temporary"
    if temporary.exists() and any(p.is_file() for p in temporary.rglob("*")):
        raise ValueError("Producer source copy not cleaned")
    direct = representative_redecode(plan, entries, boxes_by_source) if redecode else {"status": "not_run", "limitation": "No independent original D-source decode/crop performed in this run."}
    return {"schema": "lipnet-step09-independent-input-audit-v1", "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "code": binding(Path(__file__)), "selection_code": binding(BASE / "audit_selection.py"),
            "data_plan": data_ref, "execution_plan": execution_ref, "input_manifest": binding(manifest_path),
            "selection_audit": binding(selection_path), "ids": plan["ids"], "records": verified,
            "record_count": len(verified), "student_frame_traces_checked": sum(r["student_frames"] for r in verified),
            "teacher_frame_traces_checked": sum(r["teacher_frames"] for r in verified),
            "step06_png_samples_checked": sum(r["step06_png_samples_verified"] for r in verified),
            "candidate_train_records": 69, "held_train_student_only_records": 27, "development_student_only_records": 24,
            "representative_source_redecode": direct, "training_ready": False, "teacher_model_run": False, "student_model_run": False,
            "source_audio_read": False, "G_drive_read": False, "elapsed_seconds": time.perf_counter() - started,
            "limitations": ["All frame indices, PTS declarations and source label boxes were checked; hashes alone do not reconstruct unsampled original pixels.",
                            "360 earlier Step6 sample images independently constrain RGB/channel and crop correspondence at first/middle/last frames.",
                            "The direct source re-decode covers one predefined mixed candidate/hold recording; the remaining29 are not fully redecoded by this auditor.",
                            "Input preparation validity does not confirm transcript truth, all phoneme boundaries, or model accuracy."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=BASE / "inputs/manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "results/input_audit.json")
    parser.add_argument("--skip-source-redecode", action="store_true")
    args = parser.parse_args()
    result = audit(args.manifest, args.manifest_sha256, not args.skip_source_redecode)
    output = local(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "record_count", "student_frame_traces_checked", "teacher_frame_traces_checked", "step06_png_samples_checked")}))
