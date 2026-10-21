"""Independent full-input audit; imports only the frozen independent Step2 auditor.

No producer, tokenizer, model, D: source or G: source is imported or reread.
The train numerical check below is forked from the pinned independent validator.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PILOT_BASE = ROOT / "experiments/local_pilot_step02_20261007"
ANCESTOR_PATH = PILOT_BASE / "verify_inputs.py"
ANCESTOR_SHA = "b531f7125df0386ac94777e06d3f79715fcf880743751944b948b4e498c86b81"
if hashlib.sha256(ANCESTOR_PATH.read_bytes()).hexdigest() != ANCESTOR_SHA:
    raise ValueError("Frozen independent Step2 validator changed")
_spec = importlib.util.spec_from_file_location("step04_independent_numerical_ancestor", ANCESTOR_PATH)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)
local, sha, read_json, same, near = old.local, old.sha, old.read_json, old.same, old.near
tokens_from_syllables, verify_source_verification = old.tokens_from_syllables, old.verify_source_verification
PARENT, PARENT_SHA, SOURCE_PLAN, SOURCE_SHA = old.PARENT, old.PARENT_SHA, old.SOURCE_PLAN, old.SOURCE_SHA
PILOT_SHA = "d807b5a67e109019b1cba82181dbba8c6204e736ecc78db09b5c73fbcf381fac"

def verify_train_record(row: dict, entry: dict, plan_sha: str, inputs: Path,
                  source: dict, source_path: Path, source_sha: str,
                  transport_code: list[dict], preparation_runner: dict) -> dict:
    path = local(entry["npz_path"])
    same(path, (inputs / "records" / (row["id"] + ".npz")).resolve(), "Isolated NPZ location")
    same(sha(path), entry["npz_sha256"], "NPZ hash")
    metadata_path = inputs / "metadata" / (row["id"] + ".json")
    same(read_json(metadata_path), entry, "Per-record sidecar equals manifest")
    text, label = tokens_from_syllables(row["source_text"])
    same(text, row["text"], "Independent normalized transcript")
    same(label, row["token_ids"], "Arithmetic Hangul encoding")
    if len(label) + sum(a == b for a, b in zip(label, label[1:])) > row["frames"]:
        raise ValueError("Insufficient student CTC length")
    start, end, fps = row["start_time"], row["end_time"], row["video_fps"]
    near(start, row["original_start_time"] + row["offset_seconds"], 1e-12, "Offset start")
    near(end, row["original_end_time"] + row["offset_seconds"], 1e-12, "Offset end")
    expected_count = round((end - start) * 25)
    same(expected_count, row["frames"], "Student count rounding")
    target_times = [start + k / 25 for k in range(expected_count)]
    expected_indices = np.array([round(t * fps) for t in target_times], dtype=np.int64)
    with np.load(path, allow_pickle=False) as payload:
        same(set(payload.files), {"student_rgb", "teacher_bgr", "student_source_indices", "teacher_source_indices",
                                  "student_teacher_rows", "label_tokens", "metadata_json"}, "Payload keys")
        meta = json.loads(str(payload["metadata_json"].item()))
        same(meta["record"], row, "Payload fixed record")
        same(meta["plan_sha256"], plan_sha, "Payload plan binding")
        same(meta["local_full_preparation"], preparation_runner, "Payload preparation runner")
        same(meta["id"], row["id"], "Payload ID")
        frames = meta["source_frame_count"]
        if not 0 <= start < end <= frames / fps + 1e-6:
            raise ValueError("Corrected interval exceeds physical bounds")
        official_first, official_stop = math.ceil(start * fps - 1e-9), math.ceil(end * fps - 1e-9)
        union_first, union_stop = min(official_first, int(expected_indices[0])), max(official_stop, int(expected_indices[-1]) + 1)
        first, stop = max(0, union_first - 2), min(frames, union_stop + 2)
        dense_indices = np.arange(first, stop, dtype=np.int64)
        mapping = expected_indices - first
        for field, expected in (("student_source_indices", expected_indices), ("teacher_source_indices", dense_indices),
                                ("student_teacher_rows", mapping), ("label_tokens", np.array(label, np.int64))):
            values = payload[field]
            same(values.dtype, np.dtype("int64"), f"{field} dtype")
            np.testing.assert_array_equal(values, expected, err_msg=field)
        np.testing.assert_array_equal(dense_indices[mapping], expected_indices, err_msg="Frame identity")
        for field, count in (("student_rgb", expected_count), ("teacher_bgr", len(dense_indices))):
            pixels = payload[field]
            same(pixels.dtype, np.dtype("uint8"), f"{field} dtype")
            same(pixels.shape, (count, 96, 96, 3), f"{field} shape")
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            same(digest, meta[field + "_sha256"], f"{field} embedded hash")
            same(digest, entry[field + "_sha256"], f"{field} manifest hash")
        context = {"requested_real_frames_each_side": 2, "official_first": official_first,
                   "official_stop_exclusive": official_stop, "student_union_first": union_first,
                   "student_union_stop_exclusive": union_stop, "actual_first": first, "actual_stop_exclusive": stop,
                   "clipped_left": first != union_first - 2, "clipped_right": stop != union_stop + 2,
                   "student_left_real_context": np.minimum(2, mapping).tolist(),
                   "student_right_real_context": np.minimum(2, len(dense_indices) - 1 - mapping).tolist(),
                   "boundary_policy": "clip to available native frames; teacher front retains its own configured zero padding"}
        same(meta["teacher_context"], context, "Every student context coverage")
        same(entry["teacher_context"], context, "Manifest context")
        student_trace, teacher_trace = meta["student_trace"]["selected"], meta["teacher_trace"]["selected"]
        same(len(student_trace), expected_count, "Student trace length")
        same(len(teacher_trace), len(dense_indices), "Teacher trace length")
        for index, trace in zip(dense_indices, teacher_trace):
            index = int(index)
            same(trace["source_frame_index"], index, "Dense native frame trace")
            same(trace["annotation_box_index"], index, "Native annotation index")
            near(trace["actual_pts_seconds"], index / fps, .001, "Native frame PTS")
            same(trace["final_rectangle_xyxy"], list(map(int, trace["annotation_box_xyxy"])), "Teacher margin-zero integer rectangle")
            rectangle = trace["final_rectangle_xyxy"]
            if not (0 <= rectangle[0] < rectangle[2] and 0 <= rectangle[1] < rectangle[3]):
                raise ValueError("Invalid teacher rectangle")
            if len(trace["source_bgr_pixel_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in trace["source_bgr_pixel_sha256"]):
                raise ValueError("Invalid raw-source pixel SHA")
        for i, trace in enumerate(student_trace):
            index = int(expected_indices[i])
            same(trace["source_frame_index"], index, "Student chosen frame")
            same(trace["annotation_box_index"], index, "Student annotation index")
            near(trace["target_video_seconds"], target_times[i], 1e-12, "25fps target time")
            near(trace["actual_pts_seconds"], index / fps, .001, "Student frame PTS")
            near(trace["actual_pts_seconds"], target_times[i], .5 / fps + 1e-6, "Nearest source frame tolerance")
            linked = teacher_trace[int(mapping[i])]
            same(trace["source_bgr_pixel_sha256"], linked["source_bgr_pixel_sha256"], "Raw source pixel identity across independent decodes")
            same(trace["annotation_box_xyxy"], linked["annotation_box_xyxy"], "Same-source annotation identity")
        verify_source_verification(meta["source_verification"], row, source, source_path,
                                   source_sha, transport_code, audio=False)
        same(entry["source_verification"], meta["source_verification"], "Manifest source gate")
        for key, expected in (("id", row["id"]), ("split", row["split"]), ("source_stem", row["source_stem"]),
                              ("student_frames", expected_count), ("teacher_frames", len(dense_indices)), ("video_fps", fps),
                              ("source_frame_count", frames), ("plan_sha256", plan_sha)):
            same(entry[key], expected, "Manifest " + key)
    return {"id": row["id"], "split": row["split"], "student_frames": expected_count,
            "teacher_frames": len(dense_indices), "context_clipped": context["clipped_left"] or context["clipped_right"],
            "metadata_sha256": sha(metadata_path), "npz_sha256": entry["npz_sha256"],
            "all_student_and_teacher_frame_traces_checked": True}


def verify_source_geometry(meta: dict, row: dict, source_metadata: dict) -> dict:
    """Independently check every source trace against the bound media audit."""
    frames = source_metadata["frame_count"]
    same(meta["source_frame_count"], frames, "Audited physical source frame count")
    same(row["video_fps"], source_metadata["fps"], "Audited physical source fps")
    width, height = source_metadata["width"], source_metadata["height"]
    trace = meta["student_trace"]
    same(trace["native_fps"], row["video_fps"], "Trace native fps")
    same(trace["target_fps"], 25, "Trace target fps")
    same(trace["selected_count"], row["frames"], "Trace selected count")
    same(len(trace["selected"]), row["frames"], "Trace complete count")
    same(trace["selection"], "round((video_start + k / 25) * actual_fps), k < round(duration * 25)", "Trace selection rule")
    if type(trace["source_frames_decoded"]) is not int or not row["frames"] <= trace["source_frames_decoded"] <= frames:
        raise ValueError("Invalid trace decoded-frame count")
    digest = trace["selected_source_pixels_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid trace source-pixel digest")
    for index, item in enumerate(trace["selected"]):
        expected = round((row["start_time"] + index / 25) * row["video_fps"])
        if not 0 <= expected < frames:
            raise ValueError("Selected student frame outside audited source bounds")
        same(item["source_frame_index"], expected, "Trace exact native index")
        same(item["annotation_box_index"], expected, "Trace annotation index")
        near(item["actual_pts_seconds"], expected / row["video_fps"], .001, "Trace source PTS")
        near(item["target_video_seconds"], row["start_time"] + index / 25, 1e-12, "Trace target time")
        box = item["annotation_box_xyxy"]
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError("Nonfinite source annotation box")
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError("Source annotation outside physical image bounds")
        digest = item["source_bgr_pixel_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid source-pixel digest")
    return {"audited_source_frame_count": frames, "all_student_trace_source_bounds_checked": True}


def verify_dev_record(row: dict, entry: dict, inputs: Path, source: dict,
                      transport_code: list[dict], preparation_runner: dict,
                      source_metadata: dict) -> dict:
    same(row["split"], "dev", "Development-only input")
    path = local(entry["npz_path"])
    same(path, (inputs / "records" / (row["id"] + ".npz")).resolve(), "Development NPZ location")
    same(sha(path), entry["npz_sha256"], "Development NPZ hash")
    sidecar = inputs / "metadata" / (row["id"] + ".json")
    same(read_json(sidecar), entry, "Development sidecar")
    if any(key in entry for key in ("teacher_context", "teacher_bgr_sha256", "features", "feature_path")):
        raise ValueError("Development entry contains teacher targets")
    text, labels = tokens_from_syllables(row["source_text"])
    same(text, row["text"], "Development transcript normalization")
    same(labels, row["token_ids"], "Development independent Hangul tokens")
    if len(labels) + sum(a == b for a, b in zip(labels, labels[1:])) > row["frames"]:
        raise ValueError("Development CTC length insufficient")
    start, end, fps = row["start_time"], row["end_time"], row["video_fps"]
    near(start, row["original_start_time"] + row["offset_seconds"], 1e-12, "Development offset start")
    near(end, row["original_end_time"] + row["offset_seconds"], 1e-12, "Development offset end")
    same(round((end - start) * 25), row["frames"], "Development student count")
    if not 0 <= start < end <= source_metadata["frame_count"] / fps + 1e-6:
        raise ValueError("Development interval outside audited source bounds")
    indices = np.asarray([round((start + k / 25) * fps) for k in range(row["frames"])], dtype=np.int64)
    with np.load(path, allow_pickle=False) as payload:
        same(set(payload.files), {"student_rgb", "student_source_indices", "label_tokens", "metadata_json"}, "Development student-only payload keys")
        meta = json.loads(str(payload["metadata_json"].item()))
        same(set(meta), {"schema", "id", "record", "plan_sha256", "source_frame_count", "student_rgb_sha256", "student_trace", "source_verification", "preparation_seconds", "local_full_preparation"}, "Development metadata fields")
        same(meta["schema"], "local-tar-step04-dev-input-v1", "Development payload schema")
        same(meta["record"], row, "Development frozen logical record")
        same(meta["plan_sha256"], PARENT_SHA, "Development plan hash")
        same(meta["id"], row["id"], "Development ID")
        same(meta["local_full_preparation"], preparation_runner, "Development producer binding")
        for field, expected in (("student_source_indices", indices), ("label_tokens", np.asarray(labels, dtype=np.int64))):
            same(payload[field].dtype, np.dtype("int64"), "Development " + field + " dtype")
            np.testing.assert_array_equal(payload[field], expected, err_msg="Development " + field)
        rgb = payload["student_rgb"]
        same(rgb.dtype, np.dtype("uint8"), "Development RGB uint8 storage")
        same(rgb.shape, (row["frames"], 96, 96, 3), "Development RGB shape")
        rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        same(rgb_hash, meta["student_rgb_sha256"], "Development embedded RGB hash")
        same(rgb_hash, entry["student_rgb_sha256"], "Development manifest RGB hash")
        # Frozen RGB storage is not pre-normalized. The existing student applies
        # channel-wise ImageNet normalization once, after uint8 -> FP32 / 255.
        normalized = (rgb.astype(np.float32) / np.float32(255) - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Development student normalization is nonfinite")
        verify_source_verification(meta["source_verification"], row, source, SOURCE_PLAN, SOURCE_SHA, transport_code, audio=False)
        same(entry["source_verification"], meta["source_verification"], "Development source verification")
        geometry = verify_source_geometry(meta, row, source_metadata)
        for field, expected in (("id", row["id"]), ("split", "dev"), ("source_stem", row["source_stem"]), ("student_frames", row["frames"]), ("teacher_frames", 0), ("source_frame_count", source_metadata["frame_count"]), ("video_fps", fps), ("plan_sha256", PARENT_SHA)):
            same(entry[field], expected, "Development entry " + field)
    return {"id": row["id"], "split": "dev", "student_frames": row["frames"], "teacher_frames": 0,
            "context_clipped": False, "metadata_sha256": sha(sidecar), "npz_sha256": entry["npz_sha256"],
            "all_student_and_teacher_frame_traces_checked": True, "teacher_target_allowed": False,
            "raw_rgb_uint8_and_student_normalization_finite": True, **geometry}


def verify_cohort(plan: dict, manifest: dict, pilot: dict) -> None:
    ids = [row["id"] for row in plan["records"]]
    same(manifest["ids"], ids, "Full512 frozen ID order")
    same([row["id"] for row in manifest["records"]], ids, "Full512 exact record coverage")
    same(manifest["record_count"], 512, "Full512 count")
    same(manifest["source_recordings"], 59, "Full59 recordings")
    if any("C025" in row["source_stem"] for row in plan["records"]):
        raise ValueError("Quarantined C025 entered frozen cohort")
    pilot_by_id = {entry["id"]: entry for entry in pilot["records"]}
    same(list(pilot_by_id), plan["preparation"]["pilot_ids"], "Original8 pilot IDs")
    same(len(pilot_by_id), 8, "Original8 count")
    for row, entry in zip(plan["records"], manifest["records"], strict=True):
        same(entry["split"], row["split"], "Frozen split")
        same(entry["source_stem"], row["source_stem"], "Frozen source association")
        if row["id"] in pilot_by_id:
            same(entry, pilot_by_id[row["id"]], "Original8 entry preserved exactly")


def binding(path: Path) -> dict:
    return {"path": str(local(path)), "sha256": sha(path)}


def verify_anchors() -> dict:
    step1_path = old.SOURCE_BASE / "results/independent_validation.json"
    step1 = read_json(step1_path, "3967eb30c31b066c151fe81d57ef81aef3c86ef0dd90ac61d6a4e705025a7a3e")
    same(step1["status"], "passed", "Step1 independent validation")
    for ref in step1["bindings"].values():
        old.verify_binding(ref)
    step2 = read_json(PILOT_BASE / "results/step02_summary.json", "c26a4b9b4a90ad553cacddecfd68fe22aad6835af507b9ce921ccfb6d5ec856e")
    same(step2["status"], "completed", "Step2 completion")
    for name, ref in step2["artifacts"].items():
        old.verify_binding(ref)
    same(step2["artifacts"]["inputs"]["sha256"], PILOT_SHA, "Frozen Step2 input manifest")
    step3_path = ROOT / "experiments/local_training_step03_20261007/results/step03_summary.json"
    step3 = read_json(step3_path, "aa62fc9f6ad16f6d2b05843096baf2cd3b3f4332108f2883417f39fa76b224fd")
    same(step3["status"], "completed", "Step3 completion")
    for key in ("preflight_plan", "replay_audit", "independent_review", "future_comparison_recipe"):
        old.verify_binding(step3[key])
    recipe = read_json(step3["future_comparison_recipe"]["path"], "d2612b2e96e6439505aca8b34af18517bd2bb32daa226c1c7adcb6a405127bb6")
    plan = read_json(PARENT, PARENT_SHA)
    same(recipe["train_ids"], [r["id"] for r in plan["records"] if r["split"] == "train"], "Frozen Step3 train IDs")
    return {"step01_independent_validation": binding(step1_path), "step02_summary": binding(PILOT_BASE / "results/step02_summary.json"), "step03_summary": binding(step3_path), "frozen_comparison_recipe": step3["future_comparison_recipe"]}


def audit(manifest_path: Path, manifest_sha: str, output: Path) -> dict:
    started = time.perf_counter()
    same(local(manifest_path), BASE / "inputs/manifest.json", "Full manifest fixed path")
    plan = read_json(PARENT, PARENT_SHA)
    source = read_json(SOURCE_PLAN, SOURCE_SHA)
    manifest = read_json(manifest_path, manifest_sha)
    pilot = read_json(PILOT_BASE / "inputs/pilot_manifest.json", PILOT_SHA)
    for item in [*plan["bindings"], *plan["preparation"]["code"]]:
        old.verify_binding(item)
    sources = old.verify_source_plan(plan, PARENT_SHA, source, SOURCE_PLAN, SOURCE_SHA)
    splits = old.verify_splits(plan)
    anchors = verify_anchors()
    same(manifest["schema"], "feature-transfer-step07-input-manifest-v1", "Full manifest schema")
    same(manifest["phase"], "all", "Full preparation phase")
    same(manifest["status"], "passed", "Full preparation status")
    same(manifest["plan_sha256"], PARENT_SHA, "Frozen logical plan SHA")
    same(manifest["source_plan"], {"path": str(SOURCE_PLAN), "sha256": SOURCE_SHA}, "Frozen D source plan")
    old.verify_transport_code(manifest["transport_code"])
    old.verify_binding(manifest["preparation_runner"], BASE / "prepare_full.py")
    old.verify_binding(manifest["preparation_plan"], BASE / "input_plan.json")
    preparation = read_json(manifest["preparation_plan"]["path"])
    # Check every explicit local binding, including nested lists and dictionaries.
    def check_bindings(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                old.verify_binding(value)
            else:
                for child in value.values():
                    check_bindings(child)
        elif isinstance(value, list):
            for child in value:
                check_bindings(child)
    check_bindings(preparation)
    verify_cohort(plan, manifest, pilot)
    controls = old.verify_controls(plan, PARENT_SHA, manifest["reproduction_gate"], sources, SOURCE_PLAN, SOURCE_SHA, manifest["transport_code"])
    mapping_ref = next(b for b in plan["bindings"] if Path(b["path"]).name == "audited_time_mapping_candidates.json")
    mappings = {r["source_stem"]: r for r in read_json(mapping_ref["path"], mapping_ref["sha256"])["sources"]}
    metadata = {}
    for stem in sources:
        ref = next(b for b in source["bindings"] if Path(b["path"]).name == stem + ".mp4.json")
        decoded = read_json(ref["path"], ref["sha256"])
        same(decoded["member_sha256"], sources[stem]["members"]["video"]["sha256"], "Audited source member hash")
        same(decoded["status"], "full_decode_pass", "Audited source decode status")
        metadata[stem] = decoded["video_metadata"]
    pilot_ids = set(pilot["ids"])
    verified = []
    for index, (row, entry) in enumerate(zip(plan["records"], manifest["records"], strict=True), 1):
        mapping = mappings[row["source_stem"]]
        same(row["offset_seconds"], mapping["offset_seconds"], "Audited time offset")
        same(row["label_sha256"], mapping["original_label_sha256"], "Audited time label hash")
        if row["id"] not in mapping["training_utterance_ids"]:
            raise ValueError("Sentence outside audited source-time cohort")
        if row["id"] in pilot_ids:
            record = old.verify_record(row, entry, PARENT_SHA, PILOT_BASE / "inputs", sources[row["source_stem"]], SOURCE_PLAN, SOURCE_SHA, pilot["transport_code"], pilot["preparation_runner"])
        elif row["split"] == "train":
            record = verify_train_record(row, entry, PARENT_SHA, BASE / "inputs", sources[row["source_stem"]], SOURCE_PLAN, SOURCE_SHA, manifest["transport_code"], manifest["preparation_runner"])
        else:
            record = verify_dev_record(row, entry, BASE / "inputs", sources[row["source_stem"]], manifest["transport_code"], manifest["preparation_runner"], metadata[row["source_stem"]])
        if row["split"] == "train":
            with np.load(local(entry["npz_path"]), allow_pickle=False) as payload:
                meta = json.loads(str(payload["metadata_json"].item()))
                record.update(verify_source_geometry(meta, row, metadata[row["source_stem"]]))
        record.update(reused_pilot=row["id"] in pilot_ids, teacher_target_allowed=row["split"] == "train")
        verified.append(record)
        if index % 32 == 0 or index == 512:
            print(f"[independent full input audit] {index}/512", flush=True)
    student_frames = sum(r["student_frames"] for r in verified)
    teacher_frames = sum(r["teacher_frames"] for r in verified)
    same(student_frames, manifest["student_frames"], "Total student frames")
    same(teacher_frames, manifest["teacher_frames"], "Train-only teacher frames")
    for key in ("audio_read", "test_payloads_read", "original_files_modified", "teacher_model_run", "student_model_run"):
        same(manifest[key], False, "Full input scope " + key)
    result = {"schema": "local-tar-step04-independent-input-audit-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code": binding(Path(__file__)),
              "numerical_ancestor": {"path": str(ANCESTOR_PATH), "sha256": ANCESTOR_SHA},
              "plan": binding(PARENT), "source_plan": binding(SOURCE_PLAN), "manifest": binding(manifest_path),
              "preparation_runner": manifest["preparation_runner"], "transport_code": manifest["transport_code"],
              "anchors": anchors, "scope": "all512", "record_count": 512, "split_checks": splits,
              "original_reproduction_checks": controls, "records": verified,
              "student_frame_traces_checked": student_frames, "teacher_frame_traces_checked": teacher_frames,
              "train_student_frames": sum(r["student_frames"] for r in verified if r["split"] == "train"),
              "dev_student_frames": sum(r["student_frames"] for r in verified if r["split"] == "dev"),
              "reused_pilot_records_unchanged": 8, "new_train_records": 433, "student_only_dev_records": 71,
              "context_boundary_clipped_records": sum(r["context_clipped"] for r in verified),
              "source_video_audio_labels_reread": False, "source_archive_stats_rechecked": False,
              "test_payloads_read": False, "teacher_model_run": False, "student_model_run": False,
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Saved frame hashes and crop/index traces were checked; no independent original-source re-decode or re-crop was performed.",
                              "RGB order/margin/resize semantics rely on the frozen producer and four golden reproductions; arrays alone do not establish source color or crop truth.",
                              "No D: archive or G: file was read. Source identity follows the audited bound evidence and preparation materialization checks.",
                              "This verifies data preparation, not transcript truth, new-speaker generalization or recognition accuracy."]}
    output = local(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=BASE / "inputs/manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "results/input_audit.json")
    args = parser.parse_args()
    result = audit(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "record_count", "student_frame_traces_checked", "teacher_frame_traces_checked")}), flush=True)


if __name__ == "__main__":
    main()

