"""Extract visual teacher targets for exactly the fixed eight training pilots.

This local-source wrapper reuses the unchanged step07 teacher primitives. It
never calls the old runner, redirects its globals, creates a decoder, or trains.
An independent input audit must pass before a teacher plan can be frozen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PARENT = ROOT / "experiments/feature_transfer_step07"
SOURCE = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"
EXPECTED_SOURCE_SHA = "d3a26f26e0f87f42928dcfff3f644fff2eaf0ab63e79d3213bd3d7bb5f0cadc9"
EXPECTED_PARENT_SHA = "c33dd5c2b77ed7298c497e58218d70a7552b476b947659f3accd33748af13b30"
PLAN_SCHEMA = "lipnet-local-pilot-step02-teacher-plan-v1"


def load_teacher_primitives():
    """Import the original teacher implementation without modifying it."""
    try:
        path = PARENT / "teacher_features.py"
        spec = importlib.util.spec_from_file_location("step02_frozen_teacher_primitives", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        raise RuntimeError("Cannot load original teacher primitives") from exc


teacher = load_teacher_primitives()


def validate_pilot(data_plan: dict, manifest: dict, source_plan: dict) -> dict:
    """Refuse a changed, duplicated, dev, or expanded pilot cohort."""
    try:
        rows = teacher.validate_data_plan(data_plan)
        inputs = teacher.validate_manifest(manifest, data_plan, EXPECTED_PARENT_SHA, "pilot")
        if manifest.get("phase") != "pilot":
            raise ValueError("Only the explicitly named pilot phase is permitted")
        ids = data_plan["preparation"]["pilot_ids"]
        if len(ids) != 8 or any(rows[sid]["split"] != "train" for sid in ids):
            raise ValueError("Exactly eight predetermined training records are permitted")
        if source_plan["records"] != data_plan["records"]:
            raise ValueError("D source plan changed original logical records")
        if source_plan["parent_plan"]["sha256"] != EXPECTED_PARENT_SHA:
            raise ValueError("D source plan belongs to another parent")
        return {"rows": rows, "inputs": inputs, "ids": ids}
    except Exception as exc:
        raise ValueError("Invalid fixed pilot cohort") from exc


def validate_audit(audit: dict, data_binding: dict, manifest_binding: dict,
                   source_binding: dict, ids: list[str]) -> None:
    """Require the independent saved-evidence audit for this exact input set."""
    try:
        if (audit.get("schema") != "local-tar-step02-independent-input-audit-v1"
                or audit.get("status") != "passed" or audit.get("scope") != "pilot8"
                or audit.get("record_count") != 8
                or audit.get("plan") != data_binding
                or audit.get("manifest") != manifest_binding
                or audit.get("source_plan") != source_binding
                or [record["id"] for record in audit["records"]] != ids):
            raise ValueError("Input audit does not prove this exact eight-record pilot")
        teacher.verify_binding(audit["code"])
    except Exception as exc:
        raise ValueError("Independent pilot input audit is missing or inconsistent") from exc


def validate_transport(arrays: dict, row: dict, source_plan: dict, source_binding: dict) -> None:
    """Require an explicit D TAR source identity, never reinterpret G paths."""
    try:
        provenance = arrays["metadata"]["source_verification"]
        source = next(s for s in source_plan["sources"] if s["stem"] == row["source_stem"])
        expected_members = {key: value for key, value in source["members"].items() if key != "audio"}
        if (provenance.get("transport") != "audited-local-tar-v1"
                or provenance.get("source_plan_path") != source_binding["path"]
                or provenance.get("source_plan_sha256") != source_binding["sha256"]
                or provenance.get("parent_plan_sha256") != EXPECTED_PARENT_SHA
                or provenance.get("source_stem") != row["source_stem"]
                or provenance.get("members") != expected_members
                or provenance.get("archive_stats_checked_before_and_after") is not True
                or provenance.get("member_hashes_checked_on_materialization") is not True
                or provenance.get("audio_used_as_model_input") is not False):
            raise ValueError("Prepared payload lacks the exact video/label TAR provenance")
        for item in provenance["transport_code"]:
            teacher.verify_binding(item)
    except Exception as exc:
        raise ValueError("D source input provenance failed") from exc


def freeze_plan(destination: Path, manifest_path: Path, audit_path: Path) -> dict:
    """Freeze the eight-input run, implementation, checkpoint and CUDA runtime."""
    try:
        data_binding = teacher.binding(PARENT / "data_plan.json")
        source_binding = teacher.binding(SOURCE)
        if data_binding["sha256"] != EXPECTED_PARENT_SHA or source_binding["sha256"] != EXPECTED_SOURCE_SHA:
            raise ValueError("Parent or D source plan changed")
        manifest_binding = teacher.binding(manifest_path)
        data = json.loads(Path(data_binding["path"]).read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = json.loads(SOURCE.read_text(encoding="utf-8"))
        selected = validate_pilot(data, manifest, source)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        validate_audit(audit, data_binding, manifest_binding, source_binding, selected["ids"])
        teacher.configure_runtime()
        checkpoint = teacher.binding(teacher.OFFICIAL / "weights/multiview_lip.pt")
        if checkpoint["sha256"] != teacher.EXPECTED_CHECKPOINT_SHA256:
            raise ValueError("The verified public checkpoint changed")
        plan = {
            "schema": PLAN_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_feature_extraction": True,
            "data_plan": data_binding, "source_plan": source_binding,
            "pilot_manifest": manifest_binding, "input_audit": teacher.binding(audit_path),
            "checkpoint": checkpoint, "upstream": teacher.upstream_contract(),
            "runner": teacher.binding(Path(__file__)),
            "primitives": teacher.binding(PARENT / "teacher_features.py"),
            "contract_tests": [teacher.binding(BASE / "test_teacher_pilot.py"),
                               teacher.binding(PARENT / "teacher_tests.py")],
            "runtime": teacher.runtime_info(), "pilot_ids": selected["ids"],
            "output_directory": str(BASE / "features"), "result_directory": str(BASE / "teacher_results"),
            "contract": {
                "scope": "fixed eight train-only pilots; no all-train option",
                "layer": "encoder.front", "module": "Resnet2D_front(3,512)",
                "checkpoint_prefix": "encoder.front.", "strict": True,
                "dtype": "float32", "autocast": False, "batch_size": 1,
                "eval": True, "frozen": True, "no_grad": True,
                "input": "whole native-rate BGR96 window, bbox margin0, two real context frames each side",
                "normalization": "one scalar mean and sample std(correction=1) across the whole window",
                "mapping": "exact source-frame gather onto unchanged student RGB96/25fps input",
                "output": "float32 student T x512, finite and each row norm >1e-8",
                "state_integrity": "all parameters and buffers hashed before/after each clip and repeat",
                "timing": "first clip first forward plus same-clip warm repeat; other clips one forward",
                "warm_repeat_acceptance": "bitwise identical gathered features under frozen deterministic runtime",
                "audio_used": False, "decoder_instantiated": False, "optimization_steps": 0,
                "dev_teacher_targets_created": False,
            },
        }
        teacher.save_new_json(destination, plan)
        return plan
    except Exception as exc:
        raise RuntimeError("Teacher pilot plan freeze failed") from exc


def measured_forward(front: torch.nn.Module, video: torch.Tensor,
                     native_frames: int, gather: np.ndarray) -> tuple[np.ndarray, dict, list[int]]:
    """Time a synchronized FP32 visual forward with resident model and input."""
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before_allocated = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
        started = time.perf_counter()
        with torch.inference_mode():
            output = front(video)
        torch.cuda.synchronize()
        timing = {"seconds": time.perf_counter() - started,
                  "baseline_allocated_bytes": before_allocated,
                  "baseline_reserved_bytes": before_reserved,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "scope": "resident model and input, whole native visual forward; excludes normalization, host/device transfer, gather, hashes, source preparation and file writes"}
        features = teacher.validate_front_output(output, native_frames, gather)
        shape = list(output.shape)
        del output
        return features, timing, shape
    except Exception as exc:
        raise RuntimeError("Visual teacher forward failed") from exc


def feature_diagnostics(features: np.ndarray) -> dict:
    """Detect exactly constant temporal targets and report descriptive variation."""
    try:
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 512 or len(features) < 2:
            raise ValueError("Expected at least two FP32 feature rows of width 512")
        if not np.isfinite(features).all():
            raise ValueError("Feature diagnostics require finite values")
        values = features.astype(np.float64)
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms <= 1e-8):
            raise ValueError("Feature diagnostics found a zero row")
        exact_constant = bool(np.all(features == features[0]))
        if exact_constant:
            raise ValueError("Teacher targets collapsed to identical temporal rows")
        normalized = values / norms[:, None]
        adjacent_cosines = np.sum(normalized[:-1] * normalized[1:], axis=1)
        temporal_std = values.std(axis=0)
        return {"finite": True, "zero_or_near_zero_rows": 0,
                "exactly_constant_across_time": False, "value_min": float(values.min()),
                "value_max": float(values.max()), "value_mean": float(values.mean()),
                "value_std": float(values.std()), "row_l2_norm_min": float(norms.min()),
                "row_l2_norm_median": float(np.median(norms)), "row_l2_norm_max": float(norms.max()),
                "per_dimension_temporal_std_mean": float(temporal_std.mean()),
                "per_dimension_temporal_std_max": float(temporal_std.max()),
                "dimensions_with_temporal_variation": int(np.count_nonzero(temporal_std > 0)),
                "adjacent_cosine_min": float(adjacent_cosines.min()),
                "adjacent_cosine_median": float(np.median(adjacent_cosines)),
                "adjacent_cosine_max": float(adjacent_cosines.max()),
                "interpretation": "Descriptive noncollapse check only; temporal variation is not evidence of recognition accuracy."}
    except Exception as exc:
        raise ValueError("Teacher target diagnostics failed") from exc


def run(plan_path: Path) -> dict:
    """Execute only the audited eight-input frozen pilot; preserve all outputs."""
    try:
        plan_binding = teacher.binding(plan_path)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("schema") != PLAN_SCHEMA or teacher.verify_binding(plan["runner"]) != Path(__file__).resolve():
            raise ValueError("This is not the frozen local pilot runner")
        if teacher.verify_binding(plan["primitives"]) != (PARENT / "teacher_features.py").resolve():
            raise ValueError("Teacher primitives differ")
        for contract in plan["contract_tests"]:
            teacher.verify_binding(contract)
        data_path = teacher.verify_binding(plan["data_plan"])
        source_path = teacher.verify_binding(plan["source_plan"])
        manifest_path = teacher.verify_binding(plan["pilot_manifest"])
        audit_path = teacher.verify_binding(plan["input_audit"])
        if plan["data_plan"]["sha256"] != EXPECTED_PARENT_SHA or plan["source_plan"]["sha256"] != EXPECTED_SOURCE_SHA:
            raise ValueError("Frozen parent/source identity differs")
        data = json.loads(data_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = validate_pilot(data, manifest, source)
        validate_audit(json.loads(audit_path.read_text(encoding="utf-8")), plan["data_plan"],
                       plan["pilot_manifest"], plan["source_plan"], selected["ids"])
        if plan["pilot_ids"] != selected["ids"]:
            raise ValueError("Frozen pilot selection differs")
        output, results = Path(plan["output_directory"]), Path(plan["result_directory"])
        if output.resolve() != BASE / "features" or results.resolve() != BASE / "teacher_results":
            raise ValueError("Teacher outputs must remain in this isolated Step2 folder")
        if teacher.upstream_contract() != plan["upstream"]:
            raise ValueError("Pinned upstream code changed")
        teacher.configure_runtime()
        runtime = teacher.runtime_info()
        if runtime != plan["runtime"]:
            raise ValueError("Frozen teacher runtime changed")
        started = time.perf_counter()
        front, checkpoint_contract, initial = teacher.build_frozen_front(plan)
        load_seconds = time.perf_counter() - started
        completed, reused, publication_seconds = [], 0, {}
        try:
            for number, sid in enumerate(selected["ids"], 1):
                row, input_record = selected["rows"][sid], selected["inputs"][sid]
                feature_path = output / f"{sid}.npz"
                result_path = results / "records" / f"{sid}.json"
                read_started = time.perf_counter()
                arrays = teacher.read_prepared_npz(Path(input_record["npz_path"]), input_record["npz_sha256"],
                                                   row, plan["data_plan"]["sha256"])
                validate_transport(arrays, row, source, plan["source_plan"])
                read_seconds = time.perf_counter() - read_started
                if feature_path.exists():
                    result = teacher.verify_saved_feature(feature_path, result_path, plan_binding["sha256"],
                                                          input_record, row, initial)
                    completed.append(result)
                    reused += 1
                    del arrays
                    continue
                if result_path.exists():
                    raise ValueError("Feature record exists without immutable NPZ")
                item_started = time.perf_counter()
                normalize_started = time.perf_counter()
                video, normalization = teacher.normalize_teacher(arrays["teacher_bgr"])
                normalization_seconds = time.perf_counter() - normalize_started
                torch.cuda.synchronize()
                transfer_started = time.perf_counter()
                video = video.to("cuda:0")
                torch.cuda.synchronize()
                transfer_seconds = time.perf_counter() - transfer_started
                before = teacher.state_sha256(front)
                if (before != initial or any(module.training for module in front.modules())
                        or any(p.requires_grad for p in front.parameters())):
                    raise ValueError("Teacher state, eval mode or parameter freezing changed")
                features, timing, native_shape = measured_forward(
                    front, video, len(arrays["teacher_bgr"]), arrays["gather_indices"])
                if teacher.state_sha256(front) != before:
                    raise ValueError("Teacher state changed during the first forward")
                repeat = None
                if number == 1:
                    warmed, warm_timing, warm_shape = measured_forward(
                        front, video, len(arrays["teacher_bgr"]), arrays["gather_indices"])
                    if warm_shape != native_shape or not np.array_equal(warmed, features):
                        raise ValueError("Warm repeat differs from first gathered feature values")
                    repeat = {"same_input_sha256": input_record["npz_sha256"],
                              "timing": warm_timing, "bitwise_equal": True,
                              "max_absolute_difference": float(np.max(np.abs(warmed - features)))}
                    del warmed
                after = teacher.state_sha256(front)
                if after != before:
                    raise ValueError("Teacher parameters or buffers changed")
                if teacher.sha256(input_record["npz_path"]) != input_record["npz_sha256"]:
                    raise ValueError("Prepared input changed during feature extraction")
                meta = {
                    "schema": "lipnet-local-pilot-step02-teacher-record-v1", "id": sid,
                    "split": "train", "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "teacher_plan_sha256": plan_binding["sha256"], "data_plan": plan["data_plan"],
                    "source_plan": plan["source_plan"], "input_audit": plan["input_audit"],
                    "input_manifest": plan["pilot_manifest"], "input_record": input_record,
                    "data_record_sha256": teacher.canonical_sha(row), "checkpoint": plan["checkpoint"],
                    "checkpoint_contract": checkpoint_contract, "runtime": runtime,
                    "state_sha256_before": before, "state_sha256_after": after,
                    "normalization": normalization, "native_feature_shape": native_shape,
                    "aligned_feature_shape": list(features.shape), "feature_dtype": "float32",
                    "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                    "feature_diagnostics": feature_diagnostics(features),
                    "feature_timing": timing,
                    "input_npz_read_and_verify_seconds": read_seconds,
                    "normalization_cpu_seconds": normalization_seconds,
                    "host_to_device_seconds": transfer_seconds,
                    "timing_kind": "first CUDA forward of process" if number == 1 else "first forward for this clip after initial warmup",
                    "warm_repeat": repeat, "processing_seconds_before_save": time.perf_counter() - item_started,
                    "transcript_used_for_teacher_forward": False, "audio_used": False,
                    "decoder_instantiated": False, "trainable_parameters": 0, "optimization_steps": 0,
                }
                publication_started = time.perf_counter()
                teacher.save_new_npz(feature_path, features=features,
                                     student_source_indices=arrays["student_source_indices"],
                                     teacher_source_indices=arrays["teacher_source_indices"],
                                     gather_indices=arrays["gather_indices"],
                                     metadata_json=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)))
                result = teacher.verify_saved_feature(feature_path, result_path, plan_binding["sha256"],
                                                      input_record, row, initial)
                publication_seconds[sid] = time.perf_counter() - publication_started
                completed.append(result)
                print(f"[step02 teacher] {number}/8 {sid} {native_shape[1]}->{len(features)} frames "
                      f"{timing['seconds']:.3f}s peak={timing['peak_allocated_bytes']/1024**2:.1f}MiB", flush=True)
                del video, features, arrays
            if teacher.state_sha256(front) != initial:
                raise ValueError("Teacher state changed over the pilot")
        finally:
            del front
            torch.cuda.empty_cache()
        summary = {
            "schema": "lipnet-local-pilot-step02-teacher-manifest-v1", "status": "completed", "phase": "pilot",
            "created_utc": datetime.now(timezone.utc).isoformat(), "teacher_plan": plan_binding,
            "input_manifest": plan["pilot_manifest"], "input_audit": plan["input_audit"],
            "runtime": runtime, "ids": selected["ids"], "records": completed,
            "record_count": len(completed), "reused_records": reused,
            "state_sha256_before": initial, "state_sha256_after": initial,
            "checkpoint_contract": checkpoint_contract, "teacher_load_seconds": load_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "native_frames": sum(r["native_feature_shape"][1] for r in completed),
            "student_frames": sum(r["aligned_feature_shape"][0] for r in completed),
            "feature_gpu_seconds_excluding_repeat": sum(r["feature_timing"]["seconds"] for r in completed),
            "peak_allocated_bytes": max(t["peak_allocated_bytes"] for r in completed
                                         for t in [r["feature_timing"], *([r["warm_repeat"]["timing"]] if r["warm_repeat"] else [])]),
            "peak_reserved_bytes": max(t["peak_reserved_bytes"] for r in completed
                                        for t in [r["feature_timing"], *([r["warm_repeat"]["timing"]] if r["warm_repeat"] else [])]),
            "feature_file_bytes": sum(Path(r["feature_file"]["path"]).stat().st_size for r in completed),
            "output_publication_and_reverification_seconds_by_new_id": publication_seconds,
            "audio_used": False, "dev_teacher_targets_created": False,
            "decoder_instantiated": False, "optimization_steps": 0,
            "limitations": ["Visual frontend extraction feasibility only; no accuracy or training-memory result.",
                            "This does not prove the full teacher encoder or student training fits memory.",
                            "Input auditing checks saved frame traces, without independent full source re-decoding."]}
        destination = results / "pilot_manifest.json"
        if destination.exists():
            prior = json.loads(destination.read_text(encoding="utf-8"))
            if (prior["teacher_plan"] != plan_binding or prior["records"] != summary["records"]):
                raise ValueError("Completed teacher manifest differs from immutable outputs")
            summary = prior
        else:
            teacher.save_new_json(destination, summary)
        print(json.dumps({"teacher_manifest": teacher.binding(destination), "record_count": 8,
                          "status": "completed"}, ensure_ascii=False), flush=True)
        return summary
    except Exception as exc:
        raise RuntimeError("Step2 teacher pilot did not complete; no success gate published") from exc


def main() -> None:
    """Freeze or run the isolated pilot without an expansion/training option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--plan", type=Path, default=BASE / "teacher_plan.json")
    parser.add_argument("--manifest", type=Path, default=BASE / "inputs/pilot_manifest.json")
    parser.add_argument("--audit", type=Path, default=BASE / "results/input_audit.json")
    args = parser.parse_args()
    if args.freeze:
        freeze_plan(args.plan.resolve(), args.manifest.resolve(), args.audit.resolve())
        print(json.dumps({"frozen_teacher_plan": teacher.binding(args.plan)}, ensure_ascii=False), flush=True)
    else:
        run(args.plan.resolve())


if __name__ == "__main__":
    main()
