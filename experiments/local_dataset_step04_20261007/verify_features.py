"""Independent NumPy-only all441 feature audit; never imports model code."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PILOT = ROOT / "experiments/local_pilot_step02_20261007"
_spec = importlib.util.spec_from_file_location("step04_independent_input_verifier_for_feature_audit", BASE / "verify_full.py")
inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inputs)
sha, local, read_json, same, near, binding = inputs.sha, inputs.local, inputs.read_json, inputs.same, inputs.near, inputs.binding
PILOT_FEATURE_SHA = "f36bc7bebbc30ab6dc3650a298488b9dc09abe52bd0ae36ceea182fa4eac2ab8"
PILOT_PLAN_SHA = "2e7f879539b80b9818afcc44645e4916a115fa31d96273191c52a4da3c3718e5"


def canonical_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def verify_numeric_features(features: np.ndarray, meta: dict, frames: int) -> dict:
    same(features.dtype, np.dtype("float32"), "Feature FP32 dtype")
    same(features.shape, (frames, 512), "Feature aligned shape")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite feature values")
    same(hashlib.sha256(features.tobytes()).hexdigest(), meta["feature_values_sha256"], "Feature value digest")
    values = features.astype(np.float64)
    norms = np.sqrt(np.sum(values * values, axis=1))
    if np.any(norms <= 1e-8):
        raise ValueError("Zero or near-zero feature target row")
    if np.all(features == features[0]):
        raise ValueError("Exactly constant temporal feature targets")
    temporal_std = values.std(axis=0)
    unit = values / norms[:, None]
    adjacent = np.sum(unit[:-1] * unit[1:], axis=1)
    expected = {"value_min": float(values.min()), "value_max": float(values.max()),
                "value_mean": float(values.mean()), "value_std": float(values.std()),
                "row_l2_norm_min": float(norms.min()), "row_l2_norm_median": float(np.median(norms)),
                "row_l2_norm_max": float(norms.max()), "per_dimension_temporal_std_mean": float(temporal_std.mean()),
                "per_dimension_temporal_std_max": float(temporal_std.max()),
                "adjacent_cosine_min": float(adjacent.min()), "adjacent_cosine_median": float(np.median(adjacent)),
                "adjacent_cosine_max": float(adjacent.max())}
    diag = meta["feature_diagnostics"]
    for key, value in expected.items():
        near(diag[key], value, 1e-12, "Independent feature diagnostic " + key)
    for key, value in (("finite", True), ("zero_or_near_zero_rows", 0), ("exactly_constant_across_time", False),
                       ("dimensions_with_temporal_variation", int(np.count_nonzero(temporal_std > 0)))):
        same(diag[key], value, "Independent feature diagnostic " + key)
    return {"minimum_row_l2_norm": float(norms.min()), "maximum_row_l2_norm": float(norms.max()),
            "dimensions_with_temporal_variation": int(np.count_nonzero(temporal_std > 0)),
            "all_finite": True, "zero_rows": 0, "exact_temporal_collapse": False}


def verify_normalization(bgr: np.ndarray, meta: dict) -> dict:
    same(bgr.dtype, np.dtype("uint8"), "Teacher input uint8")
    same(bgr.shape[1:], (96, 96, 3), "Teacher input native BGR shape")
    norm = meta["normalization"]
    for key, value in (("channel_order", "BGR"), ("scope", "whole native temporal window, all channels and pixels"), ("std_correction", 1), ("dtype", "float32")):
        same(norm[key], value, "Whole-window normalization " + key)
    mean = float(bgr.mean(dtype=np.float64))
    std = float(bgr.std(dtype=np.float64, ddof=1))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Teacher window has no valid variance")
    # Independent FP64 sums need not reproduce the FP32 parallel reduction bitwise.
    near(norm["raw_mean"], mean, 1e-4, "Independent whole-window raw mean")
    near(norm["raw_std"], std, 1e-4, "Independent whole-window sample standard deviation")
    normalized = (bgr.astype(np.float32) - np.float32(norm["raw_mean"])) / np.float32(norm["raw_std"])
    near(norm["normalized_mean"], float(normalized.mean(dtype=np.float64)), 2e-6, "Normalized window mean")
    near(norm["normalized_std"], float(normalized.std(dtype=np.float64, ddof=1)), 2e-6, "Normalized sample standard deviation")
    near(norm["normalized_mean"], 0., 2e-6, "Zero normalized mean")
    near(norm["normalized_std"], 1., 2e-6, "Unit normalized sample standard deviation")
    return {"independent_raw_mean_abs_difference": abs(norm["raw_mean"] - mean),
            "independent_raw_std_abs_difference": abs(norm["raw_std"] - std),
            "whole_native_window_sample_std_checked": True}


def verify_record(row: dict, entry: dict, input_entry: dict, plan: dict, plan_sha: str,
                  expected_state: str, reused: bool) -> dict:
    same(row["split"], "train", "Teacher source train-only")
    same(entry["split"], "train", "Teacher target train-only")
    feature_path = local(entry["feature_file"]["path"])
    expected_base = PILOT if reused else BASE
    same(feature_path, expected_base / "features" / (row["id"] + ".npz"), "Teacher target isolated path")
    inputs.old.verify_binding(entry["feature_file"])
    sidecar = expected_base / "teacher_results/records" / (row["id"] + ".json")
    same(read_json(sidecar), entry, "Feature sidecar matches manifest")
    same(sha(input_entry["npz_path"]), input_entry["npz_sha256"], "Prepared train input SHA")
    with np.load(feature_path, allow_pickle=False) as target, np.load(local(input_entry["npz_path"]), allow_pickle=False) as data:
        same(set(target.files), {"features", "student_source_indices", "teacher_source_indices", "gather_indices", "metadata_json"}, "Feature payload keys")
        meta = json.loads(str(target["metadata_json"].item()))
        same({**meta, "feature_file": entry["feature_file"], "status": "completed"}, entry, "Embedded feature metadata exact")
        for key, value in (("id", row["id"]), ("speaker_id", row["speaker_id"]), ("source_stem", row["source_stem"]),
                           ("teacher_plan_sha256", plan_sha), ("input_record", input_entry),
                           ("data_record_sha256", canonical_digest(row)), ("state_sha256_before", expected_state),
                           ("state_sha256_after", expected_state), ("aligned_feature_shape", [row["frames"], 512]),
                           ("feature_dtype", "float32"), ("optimization_steps", 0), ("trainable_parameters", 0),
                           ("audio_used", False), ("decoder_instantiated", False), ("transcript_used_for_teacher_forward", False)):
            same(meta[key], value, "Feature fixed contract " + key)
        for key in ("data_plan", "source_plan", "input_audit", "checkpoint", "runtime"):
            same(meta[key], plan[key], "Teacher provenance " + key)
        same(meta["input_manifest"], plan["pilot_manifest"] if reused else plan["input_manifest"], "Teacher input manifest provenance")
        for name, data_name in (("student_source_indices", "student_source_indices"), ("teacher_source_indices", "teacher_source_indices"), ("gather_indices", "student_teacher_rows")):
            same(target[name].dtype, np.dtype("int64"), "Feature indices int64")
            np.testing.assert_array_equal(target[name], data[data_name], err_msg="Feature/input original-frame identity")
        student_ids, teacher_ids, gather = (target[k] for k in ("student_source_indices", "teacher_source_indices", "gather_indices"))
        expected = np.asarray([round((row["start_time"] + k / 25) * row["video_fps"]) for k in range(row["frames"])], np.int64)
        np.testing.assert_array_equal(student_ids, expected)
        np.testing.assert_array_equal(teacher_ids, np.arange(teacher_ids[0], teacher_ids[-1] + 1, dtype=np.int64))
        if np.any(gather < 0) or np.any(gather >= len(teacher_ids)):
            raise ValueError("Feature gather outside native window")
        np.testing.assert_array_equal(teacher_ids[gather], expected)
        same(meta["native_feature_shape"], [1, len(teacher_ids), 512], "Native visual-front shape")
        numeric = verify_numeric_features(target["features"], meta, row["frames"])
        normalization = verify_normalization(data["teacher_bgr"], meta)
    contract = entry["checkpoint_contract"]
    for key, value in (("state_prefix", "encoder.front."), ("strict", True), ("parameters", 11218176),
                       ("other_state_entries_loaded", 0), ("missing", []), ("unexpected", []),
                       ("shape_mismatch", {}), ("dtype_mismatch", {})):
        same(contract[key], value, "Visual frontend checkpoint contract " + key)
    timing = entry["feature_timing"]
    if not math.isfinite(timing["seconds"]) or timing["seconds"] <= 0:
        raise ValueError("Invalid measured forward seconds")
    if not 0 < timing["baseline_allocated_bytes"] <= timing["peak_allocated_bytes"] <= timing["peak_reserved_bytes"]:
        raise ValueError("Invalid allocated/reserved GPU timing evidence")
    return {"id": row["id"], "split": "train", "reused_pilot": reused, "feature_file": entry["feature_file"],
            "input_file": {"path": input_entry["npz_path"], "sha256": input_entry["npz_sha256"]},
            "student_frames": row["frames"], "native_frames": entry["native_feature_shape"][1],
            "sidecar_sha256": sha(sidecar), "same_integer_source_frames": True, **numeric, **normalization}


def audit(manifest_sha: str, output: Path) -> dict:
    started = time.perf_counter()
    manifest_path = BASE / "teacher_results/train_manifest.json"
    manifest = read_json(manifest_path, manifest_sha)
    inputs.old.verify_binding(manifest["teacher_plan"], BASE / "teacher_plan.json")
    plan = read_json(BASE / "teacher_plan.json", manifest["teacher_plan"]["sha256"])
    same(plan["schema"], "lipnet-local-step04-teacher-plan-v1", "Full teacher plan schema")
    for key in ("data_plan", "source_plan", "input_manifest", "input_audit", "pilot_inputs", "pilot_features", "pilot_teacher_plan", "comparison_recipe", "runner", "pilot_runner", "primitives", "checkpoint"):
        inputs.old.verify_binding(plan[key])
    for ref in plan["contract_tests"]:
        inputs.old.verify_binding(ref)
    same(plan["data_plan"], binding(inputs.PARENT), "Exact frozen logical parent")
    same(plan["source_plan"], binding(inputs.SOURCE_PLAN), "Exact frozen D source plan")
    same(local(plan["runner"]["path"]), BASE / "teacher_full.py", "Full teacher runner location")
    same(local(plan["pilot_runner"]["path"]), PILOT / "teacher_pilot.py", "Frozen pilot runner location")
    for key in ("input_manifest", "input_audit", "pilot_features", "pilot_teacher_plan", "comparison_recipe", "runtime"):
        same(manifest[key], plan[key], "Manifest/teacher plan " + key)
    data_plan = read_json(inputs.PARENT, inputs.PARENT_SHA)
    full_inputs = read_json(plan["input_manifest"]["path"], plan["input_manifest"]["sha256"])
    input_audit = read_json(plan["input_audit"]["path"], plan["input_audit"]["sha256"])
    inputs.old.verify_binding(input_audit["code"], BASE / "verify_full.py")
    same(input_audit["status"], "passed", "Independent full-input audit passed")
    same(input_audit["record_count"], 512, "Independent full-input audit512")
    same(input_audit["manifest"], plan["input_manifest"], "Independent input audit binding")
    original_manifest = read_json(PILOT / "teacher_results/pilot_manifest.json", PILOT_FEATURE_SHA)
    original_plan = read_json(PILOT / "teacher_plan.json", PILOT_PLAN_SHA)
    original_rows = {r["id"]: r for r in original_manifest["records"]}
    train_rows = [r for r in data_plan["records"] if r["split"] == "train"]
    train_ids = [r["id"] for r in train_rows]
    dev_ids = [r["id"] for r in data_plan["records"] if r["split"] == "dev"]
    expected_state = original_manifest["state_sha256_before"]
    for key, value in (("schema", "lipnet-local-step04-teacher-manifest-v1"), ("status", "completed"), ("phase", "train"),
                       ("record_count", 441), ("pilot_references_reused", 8), ("new_feature_record_count", 433),
                       ("ids", train_ids), ("state_sha256_before", expected_state), ("state_sha256_after", expected_state),
                       ("audio_used", False), ("dev_input_npz_opened", False), ("dev_teacher_targets_created", False),
                       ("decoder_instantiated", False), ("student_optimization_steps", 0), ("teacher_optimization_steps", 0)):
        same(manifest[key], value, "Full teacher manifest " + key)
    same([r["id"] for r in manifest["records"]], train_ids, "Full teacher441 record order")
    same(plan["train_ids"], train_ids, "Teacher plan train cohort")
    same(plan["dev_ids"], dev_ids, "Teacher plan excluded dev cohort")
    same(plan["pilot_ids"], list(original_rows), "Teacher plan original eight pilot IDs")
    same(plan["new_feature_ids"], [rid for rid in train_ids if rid not in original_rows], "Teacher plan new433 IDs")
    same(plan["runtime"], original_plan["runtime"], "Pilot-identical runtime")
    same(plan["checkpoint"], original_plan["checkpoint"], "Pilot-identical checkpoint")
    same(plan["expected_state_sha256"], expected_state, "Pilot-identical visual frontend state")
    by_id = {r["id"]: r for r in full_inputs["records"]}
    verified = []
    for number, (row, entry) in enumerate(zip(train_rows, manifest["records"], strict=True), 1):
        reused = row["id"] in original_rows
        if reused:
            same(entry, original_rows[row["id"]], "Original8 feature entry preserved exactly")
        verified.append(verify_record(row, entry, by_id[row["id"]], original_plan if reused else plan,
                                      PILOT_PLAN_SHA if reused else manifest["teacher_plan"]["sha256"], expected_state, reused))
        if number % 32 == 0 or number == 441:
            print(f"[independent feature audit] {number}/441", flush=True)
    new_ids = {r["id"] for r in verified if not r["reused_pilot"]}
    same({p.stem for p in (BASE / "features").glob("*.npz")}, new_ids, "Exactly new433 feature files, no dev targets")
    same({p.stem for p in (BASE / "teacher_results/records").glob("*.json")}, new_ids, "Exactly new433 teacher sidecars")
    same(sum(r["student_frames"] for r in verified), manifest["student_frames"], "Teacher aligned frame total")
    same(sum(r["native_frames"] for r in verified), manifest["native_frames"], "Teacher native frame total")
    all_bytes = sum(local(r["feature_file"]["path"]).stat().st_size for r in verified)
    new_bytes = sum(local(r["feature_file"]["path"]).stat().st_size for r in verified if not r["reused_pilot"])
    same(all_bytes, manifest["all441_feature_file_bytes"], "All target NPZ bytes")
    same(new_bytes, manifest["new433_feature_file_bytes"], "New target NPZ bytes")
    new_records = [r for r in manifest["records"] if r["id"] not in original_rows]
    near(manifest["new433_gpu_forward_seconds"], sum(r["feature_timing"]["seconds"] for r in new_records), 1e-9, "New forward-time sum")
    same(manifest["new433_peak_allocated_bytes"], max(r["feature_timing"]["peak_allocated_bytes"] for r in new_records), "New peak allocated memory")
    same(manifest["new433_peak_reserved_bytes"], max(r["feature_timing"]["peak_reserved_bytes"] for r in new_records), "New peak reserved memory")
    result = {"schema": "local-tar-step04-independent-feature-audit-v1", "status": "passed",
              "created_utc": datetime.now(timezone.utc).isoformat(), "code": binding(Path(__file__)),
              "independent_input_verifier": binding(BASE / "verify_full.py"),
              "manifest": binding(manifest_path), "teacher_plan": binding(BASE / "teacher_plan.json"),
              "input_audit": plan["input_audit"], "input_manifest": plan["input_manifest"],
              "record_count": 441, "reused_pilot_records_unchanged": 8, "new_records": 433,
              "development_target_count": 0, "student_frames": manifest["student_frames"], "native_frames": manifest["native_frames"],
              "all_feature_file_bytes": all_bytes, "new_feature_file_bytes": new_bytes,
              "records": verified, "models_or_producers_imported": False, "source_drives_read": False,
              "teacher_inference_repeated": False, "student_optimization_steps": 0, "teacher_optimization_steps": 0,
              "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Independent checks cover saved values, shape, frame mapping, normalization and provenance; teacher forward was not independently rerun.",
                              "Nonconstant finite targets do not establish learning benefit or lipreading accuracy.",
                              "The teacher runner's no-dev-input behavior follows reviewed code and saved execution evidence; this is not an operating-system I/O trace."]}
    output = local(output)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=BASE / "results/feature_audit.json")
    args = parser.parse_args()
    result = audit(args.manifest_sha256, args.output)
    print(json.dumps({k: result[k] for k in ("status", "record_count", "student_frames", "native_frames")}), flush=True)


if __name__ == "__main__":
    main()
