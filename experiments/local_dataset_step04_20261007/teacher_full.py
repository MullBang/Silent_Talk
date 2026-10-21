"""Prepare exactly 441 frozen visual targets, reusing the eight pilot artifacts.

Development NPZ payloads are never opened here.  A passed independent all-input
audit must exist before the extraction plan is frozen.  No student, decoder,
optimizer, audio input or speech recognition model is constructed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
PILOT = ROOT / "experiments/local_pilot_step02_20261007"
PARENT = ROOT / "experiments/feature_transfer_step07"
SOURCE = ROOT / "experiments/local_source_step01_20261007/d_source_plan.json"
RECIPE = ROOT / "experiments/local_training_step03_20261007/comparison_protocol.json"
PLAN_SCHEMA = "lipnet-local-step04-teacher-plan-v1"
PILOT_PLAN_SHA = "2e7f879539b80b9818afcc44645e4916a115fa31d96273191c52a4da3c3718e5"
PILOT_INPUT_SHA = "d807b5a67e109019b1cba82181dbba8c6204e736ecc78db09b5c73fbcf381fac"
PILOT_FEATURE_SHA = "f36bc7bebbc30ab6dc3650a298488b9dc09abe52bd0ae36ceea182fa4eac2ab8"
RECIPE_SHA = "d2612b2e96e6439505aca8b34af18517bd2bb32daa226c1c7adcb6a405127bb6"


def load_pilot_helpers():
    spec = importlib.util.spec_from_file_location("step04_frozen_pilot_teacher", PILOT / "teacher_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load_pilot_helpers()
teacher = pilot.teacher


def read_bound(item: dict) -> dict:
    path = teacher.verify_binding(item)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("Bound JSON changed while reading")
    return json.loads(raw)


def validate_cohort(data: dict, source: dict, manifest: dict,
                    pilot_inputs: dict, pilot_features: dict, recipe: dict) -> dict:
    """Select only train441 and preserve every pilot reference exactly."""
    rows = teacher.validate_data_plan(data)
    inputs = teacher.validate_manifest(manifest, data, pilot.EXPECTED_PARENT_SHA, "all")
    prior_inputs = teacher.validate_manifest(pilot_inputs, data, pilot.EXPECTED_PARENT_SHA, "pilot")
    train_ids = [row["id"] for row in data["records"] if row["split"] == "train"]
    dev_ids = [row["id"] for row in data["records"] if row["split"] == "dev"]
    pilot_ids = data["preparation"]["pilot_ids"]
    if (manifest.get("phase") != "all" or source["records"] != data["records"]
            or source["parent_plan"]["sha256"] != pilot.EXPECTED_PARENT_SHA
            or pilot_features.get("status") != "completed" or pilot_features.get("phase") != "pilot"
            or pilot_features.get("record_count") != 8
            or pilot_features["ids"] != pilot_ids
            or [r["id"] for r in pilot_features["records"]] != pilot_ids
            or recipe.get("schema") != "lipnet-step03-frozen-comparison-recipe-v1"
            or recipe["train_ids"] != train_ids or recipe["development_ids"] != dev_ids):
        raise ValueError("Full cohort, pilot reuse or frozen comparison recipe differs")
    prior_features = {row["id"]: row for row in pilot_features["records"]}
    for sid in pilot_ids:
        if (rows[sid]["split"] != "train" or inputs[sid] != prior_inputs[sid]
                or prior_features[sid]["input_record"] != prior_inputs[sid]
                or prior_features[sid]["split"] != "train"
                or prior_features[sid]["teacher_plan_sha256"] != PILOT_PLAN_SHA):
            raise ValueError("Pilot entries must remain byte-identical artifact references")
    for sid in dev_ids:
        if inputs[sid].get("teacher_frames") != 0:
            raise ValueError("Development entries must contain student input only")
    return {"rows": rows, "inputs": inputs, "train_ids": train_ids, "dev_ids": dev_ids,
            "pilot_ids": pilot_ids, "pilot_features": prior_features,
            "new_ids": [sid for sid in train_ids if sid not in prior_features]}


def validate_audit(audit: dict, bindings: dict, all_ids: list[str], rows: dict) -> None:
    if (audit.get("schema") != "local-tar-step04-independent-input-audit-v1"
            or audit.get("status") != "passed" or audit.get("scope") != "all512"
            or audit.get("record_count") != 512
            or audit.get("plan") != bindings["data_plan"]
            or audit.get("manifest") != bindings["input_manifest"]
            or audit.get("source_plan") != bindings["source_plan"]
            or [r["id"] for r in audit["records"]] != all_ids):
        raise ValueError("Independent full input audit does not bind these exact 512 inputs")
    teacher.verify_binding(audit["code"])
    for record in audit["records"]:
        is_train = rows[record["id"]]["split"] == "train"
        if (record.get("teacher_target_allowed") is not is_train
                or (not is_train and record.get("teacher_frames") != 0)):
            raise ValueError("Independent audit does not prove train/dev teacher separation")


def validate_pilot_provenance(prior_plan: dict, prior_features: dict, bindings: dict) -> None:
    if (bindings["pilot_teacher_plan"]["sha256"] != PILOT_PLAN_SHA
            or bindings["pilot_inputs"]["sha256"] != PILOT_INPUT_SHA
            or bindings["pilot_features"]["sha256"] != PILOT_FEATURE_SHA
            or prior_plan["data_plan"] != bindings["data_plan"]
            or prior_plan["source_plan"] != bindings["source_plan"]
            or prior_plan["pilot_manifest"] != bindings["pilot_inputs"]
            or prior_features["teacher_plan"] != bindings["pilot_teacher_plan"]
            or prior_features["input_manifest"] != bindings["pilot_inputs"]
            or prior_features["runtime"] != prior_plan["runtime"]
            or prior_features["state_sha256_before"] != prior_features["state_sha256_after"]):
        raise ValueError("Immutable Step2 teacher provenance differs")
    for item in [prior_plan["runner"], prior_plan["primitives"], *prior_plan["contract_tests"],
                 prior_plan["input_audit"], prior_plan["checkpoint"]]:
        teacher.verify_binding(item)


def verify_feature(path: Path, sidecar: Path, plan_sha: str, input_record: dict,
                   row: dict, state_sha: str, expected: dict | None = None,
                   arrays: dict | None = None, provenance: dict | None = None) -> dict:
    """Validate immutable targets, with recovery only for new orphaned NPZs."""
    if row["split"] != "train" or input_record["split"] != "train":
        raise ValueError("Cannot read development targets")
    if expected is not None:
        if not sidecar.is_file() or read_bound(teacher.binding(sidecar)) != expected:
            raise ValueError("Reused feature sidecar is absent or changed")
        if teacher.verify_binding(expected["feature_file"]) != path.resolve():
            raise ValueError("Reused feature path differs")
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"].item()))
        diagnostics = pilot.feature_diagnostics(payload["features"])
        if arrays is not None:
            for key in ("student_source_indices", "teacher_source_indices", "gather_indices"):
                if not np.array_equal(payload[key], arrays[key]):
                    raise ValueError("Saved feature frame identities differ from exact prepared input")
    if metadata.get("feature_diagnostics") != diagnostics:
        raise ValueError("Saved feature diagnostic evidence differs")
    if (metadata.get("audio_used") is not False or metadata.get("decoder_instantiated") is not False
            or metadata.get("transcript_used_for_teacher_forward") is not False
            or metadata.get("optimization_steps") != 0 or metadata.get("trainable_parameters") != 0):
        raise ValueError("Teacher record is not frozen visual-only extraction")
    if provenance is not None:
        for key in ("data_plan", "source_plan", "input_audit", "input_manifest", "checkpoint", "runtime"):
            if metadata.get(key) != provenance[key]:
                raise ValueError(f"Saved feature provenance differs: {key}")
    # The unchanged primitive can recover a missing sidecar.  Run every extra
    # Step4 check first, so invalid orphaned NPZs never acquire a success record.
    result = teacher.verify_saved_feature(path, sidecar, plan_sha, input_record, row, state_sha)
    if expected is not None and result != expected:
        raise ValueError("Reused feature entry differs from the original manifest")
    return result


def validate_output_inventory(directory: Path, new_ids: list[str]) -> None:
    allowed = {f"{sid}.npz" for sid in new_ids}
    if directory.exists() and any(path.name not in allowed for path in directory.glob("*.npz")):
        raise ValueError("Unexpected pilot/development target or other NPZ in new feature directory")


def read_train_input(row: dict, item: dict, source: dict, source_binding: dict) -> dict:
    """Check membership before resolving, hashing or opening any NPZ path."""
    if row["split"] != "train" or item["split"] != "train" or item["id"] != row["id"]:
        raise ValueError("Teacher may open training NPZs only")
    arrays = teacher.read_prepared_npz(Path(item["npz_path"]), item["npz_sha256"],
                                      row, pilot.EXPECTED_PARENT_SHA)
    pilot.validate_transport(arrays, row, source, source_binding)
    return arrays


def freeze_plan(destination: Path, manifest_path: Path, audit_path: Path) -> dict:
    paths = {"data_plan": PARENT / "data_plan.json", "source_plan": SOURCE,
             "input_manifest": manifest_path, "input_audit": audit_path,
             "pilot_inputs": PILOT / "inputs/pilot_manifest.json",
             "pilot_features": PILOT / "teacher_results/pilot_manifest.json",
             "pilot_teacher_plan": PILOT / "teacher_plan.json", "comparison_recipe": RECIPE}
    bindings = {name: teacher.binding(path) for name, path in paths.items()}
    if (bindings["data_plan"]["sha256"] != pilot.EXPECTED_PARENT_SHA
            or bindings["source_plan"]["sha256"] != pilot.EXPECTED_SOURCE_SHA
            or bindings["comparison_recipe"]["sha256"] != RECIPE_SHA):
        raise ValueError("Logical parent, D source or frozen Step3 recipe changed")
    docs = {name: read_bound(value) for name, value in bindings.items()}
    selected = validate_cohort(docs["data_plan"], docs["source_plan"], docs["input_manifest"],
                               docs["pilot_inputs"], docs["pilot_features"], docs["comparison_recipe"])
    validate_audit(docs["input_audit"], bindings, [r["id"] for r in docs["data_plan"]["records"]], selected["rows"])
    validate_pilot_provenance(docs["pilot_teacher_plan"], docs["pilot_features"], bindings)
    recipe = docs["comparison_recipe"]
    if recipe["parent_plan"] != bindings["data_plan"] or recipe["source_plan"] != bindings["source_plan"]:
        raise ValueError("Comparison recipe uses another source")
    for item in (recipe["preflight_plan"], recipe["replay_audit"]):
        teacher.verify_binding(item)
    teacher.configure_runtime()
    prior = docs["pilot_teacher_plan"]
    if teacher.runtime_info() != prior["runtime"] or teacher.upstream_contract() != prior["upstream"]:
        raise ValueError("Full extraction must use the identical pilot runtime and upstream")
    state = docs["pilot_features"]["state_sha256_before"]
    for sid in selected["pilot_ids"]:
        item = selected["pilot_features"][sid]
        verify_feature(Path(item["feature_file"]["path"]), PILOT / "teacher_results/records" / f"{sid}.json",
                       PILOT_PLAN_SHA, selected["inputs"][sid], selected["rows"][sid], state, item)
    plan = {"schema": PLAN_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_before_feature_extraction": True, **bindings,
            "runner": teacher.binding(Path(__file__)),
            "pilot_runner": teacher.binding(PILOT / "teacher_pilot.py"),
            "primitives": teacher.binding(PARENT / "teacher_features.py"),
            "contract_tests": [teacher.binding(BASE / "test_teacher_full.py"), *prior["contract_tests"]],
            "runtime": prior["runtime"], "checkpoint": prior["checkpoint"], "upstream": prior["upstream"],
            "expected_state_sha256": state,
            "train_ids": selected["train_ids"], "dev_ids": selected["dev_ids"],
            "pilot_ids": selected["pilot_ids"], "new_feature_ids": selected["new_ids"],
            "output_directory": str(BASE / "features"), "result_directory": str(BASE / "teacher_results"),
            "contract": {"pilot_reference_contract": prior["contract"], "scope": "train441 only; no dev NPZ reads",
                         "reuse": "exact eight pilot dictionaries and existing NPZ paths/hashes; no copies",
                         "new_features": 433, "new_feature_provenance": "this new full teacher plan",
                         "pilot_feature_provenance": "original Step2 teacher plan unchanged",
                         "diagnostics": "reject nonfinite, zero/near-zero rows and temporally constant targets",
                         "resume": "verify immutable NPZ and sidecar against plan, input, state and diagnostics",
                         "student_optimization_steps": 0, "teacher_optimization_steps": 0}}
    teacher.save_new_json(destination, plan)
    return plan


def run(plan_path: Path) -> dict:
    binding = teacher.binding(plan_path)
    plan = read_bound(binding)
    if (plan.get("schema") != PLAN_SCHEMA
            or teacher.verify_binding(plan["runner"]) != Path(__file__).resolve()
            or teacher.verify_binding(plan["pilot_runner"]) != PILOT / "teacher_pilot.py"
            or teacher.verify_binding(plan["primitives"]) != PARENT / "teacher_features.py"):
        raise ValueError("This is not the frozen full teacher extraction runner")
    for item in plan["contract_tests"]:
        teacher.verify_binding(item)
    keys = ["data_plan", "source_plan", "input_manifest", "input_audit", "pilot_inputs",
            "pilot_features", "pilot_teacher_plan", "comparison_recipe"]
    docs = {key: read_bound(plan[key]) for key in keys}
    if (plan["data_plan"]["sha256"] != pilot.EXPECTED_PARENT_SHA
            or plan["source_plan"]["sha256"] != pilot.EXPECTED_SOURCE_SHA
            or plan["comparison_recipe"]["sha256"] != RECIPE_SHA):
        raise ValueError("Frozen parent/source/recipe identity differs")
    selected = validate_cohort(docs["data_plan"], docs["source_plan"], docs["input_manifest"],
                               docs["pilot_inputs"], docs["pilot_features"], docs["comparison_recipe"])
    validate_audit(docs["input_audit"], plan, [r["id"] for r in docs["data_plan"]["records"]], selected["rows"])
    validate_pilot_provenance(docs["pilot_teacher_plan"], docs["pilot_features"], plan)
    for name, selected_name in [("train_ids", "train_ids"), ("dev_ids", "dev_ids"),
                                ("pilot_ids", "pilot_ids"), ("new_feature_ids", "new_ids")]:
        if plan[name] != selected[selected_name]:
            raise ValueError("Frozen train/pilot/dev order differs")
    output, results = Path(plan["output_directory"]), Path(plan["result_directory"])
    if output.resolve() != BASE / "features" or results.resolve() != BASE / "teacher_results":
        raise ValueError("New teacher outputs must remain in the isolated Step4 folder")
    validate_output_inventory(output, selected["new_ids"])
    teacher.configure_runtime()
    prior = docs["pilot_teacher_plan"]
    if (teacher.runtime_info() != plan["runtime"] or plan["runtime"] != prior["runtime"]
            or teacher.upstream_contract() != plan["upstream"] or plan["upstream"] != prior["upstream"]
            or plan["checkpoint"] != prior["checkpoint"]
            or plan["expected_state_sha256"] != docs["pilot_features"]["state_sha256_before"]):
        raise ValueError("Teacher checkpoint, runtime or expected state differs from pilot")
    if shutil.disk_usage(BASE).free < 1024**3 + sum(selected["rows"][sid]["frames"] * 512 * 4 for sid in selected["new_ids"]):
        raise ValueError("Insufficient free disk space for immutable teacher outputs")
    started = time.perf_counter()
    front, checkpoint_contract, initial = teacher.build_frozen_front(plan)
    load_seconds = time.perf_counter() - started
    if initial != plan["expected_state_sha256"]:
        raise ValueError("Loaded teacher state differs from reused pilot targets")
    completed, resumed, created = [], [], []
    try:
        for number, sid in enumerate(selected["train_ids"], 1):
            row, item = selected["rows"][sid], selected["inputs"][sid]
            if sid in selected["pilot_features"]:
                prior_item = selected["pilot_features"][sid]
                completed.append(verify_feature(Path(prior_item["feature_file"]["path"]),
                                                PILOT / "teacher_results/records" / f"{sid}.json",
                                                PILOT_PLAN_SHA, item, row, initial, prior_item))
                continue
            feature_path = output / f"{sid}.npz"
            sidecar = results / "records" / f"{sid}.json"
            item_started = time.perf_counter()
            arrays = read_train_input(row, item, docs["source_plan"], plan["source_plan"])
            read_seconds = time.perf_counter() - item_started
            if feature_path.exists():
                completed.append(verify_feature(feature_path, sidecar, binding["sha256"], item, row, initial,
                                                arrays=arrays, provenance=plan))
                resumed.append(sid)
                del arrays
                continue
            if sidecar.exists():
                raise ValueError("Feature sidecar exists without immutable NPZ")
            normalize_started = time.perf_counter()
            video, normalization = teacher.normalize_teacher(arrays["teacher_bgr"])
            normalize_seconds = time.perf_counter() - normalize_started
            torch.cuda.synchronize()
            transfer_started = time.perf_counter()
            video = video.to("cuda:0")
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_started
            before = teacher.state_sha256(front)
            if (before != initial or any(module.training for module in front.modules())
                    or any(p.requires_grad for p in front.parameters())):
                raise ValueError("Teacher state, eval mode or freezing changed")
            features, timing, native_shape = pilot.measured_forward(front, video, len(arrays["teacher_bgr"]), arrays["gather_indices"])
            after = teacher.state_sha256(front)
            if after != before or teacher.sha256(item["npz_path"]) != item["npz_sha256"]:
                raise ValueError("Teacher state or prepared input changed during extraction")
            meta = {"schema": "lipnet-local-step04-teacher-record-v1", "id": sid, "split": "train",
                    "speaker_id": row["speaker_id"], "source_stem": row["source_stem"],
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "teacher_plan_sha256": binding["sha256"], "data_plan": plan["data_plan"],
                    "source_plan": plan["source_plan"], "input_audit": plan["input_audit"],
                    "input_manifest": plan["input_manifest"], "input_record": item,
                    "data_record_sha256": teacher.canonical_sha(row), "checkpoint": plan["checkpoint"],
                    "checkpoint_contract": checkpoint_contract, "runtime": plan["runtime"],
                    "state_sha256_before": before, "state_sha256_after": after,
                    "normalization": normalization, "native_feature_shape": native_shape,
                    "aligned_feature_shape": list(features.shape), "feature_dtype": "float32",
                    "feature_values_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
                    "feature_diagnostics": pilot.feature_diagnostics(features), "feature_timing": timing,
                    "input_npz_read_and_verify_seconds": read_seconds,
                    "normalization_cpu_seconds": normalize_seconds, "host_to_device_seconds": transfer_seconds,
                    "timing_kind": "first CUDA forward of process" if not created else "first forward for this clip after initial warmup",
                    "warm_repeat": None, "processing_seconds_before_save": time.perf_counter() - item_started,
                    "transcript_used_for_teacher_forward": False, "audio_used": False,
                    "decoder_instantiated": False, "trainable_parameters": 0, "optimization_steps": 0}
            teacher.save_new_npz(feature_path, features=features,
                                 student_source_indices=arrays["student_source_indices"],
                                 teacher_source_indices=arrays["teacher_source_indices"],
                                 gather_indices=arrays["gather_indices"],
                                 metadata_json=np.array(json.dumps(meta, ensure_ascii=False, allow_nan=False)))
            completed.append(verify_feature(feature_path, sidecar, binding["sha256"], item, row, initial,
                                            arrays=arrays, provenance=plan))
            created.append(sid)
            print(f"[step04 teacher] {number}/441 {sid} {native_shape[1]}->{len(features)} frames "
                  f"{timing['seconds']:.3f}s peak={timing['peak_allocated_bytes']/1024**2:.1f}MiB", flush=True)
            del video, features, arrays
        if teacher.state_sha256(front) != initial:
            raise ValueError("Teacher state changed over full extraction")
    finally:
        del front
        torch.cuda.empty_cache()
    new_records = [r for r in completed if r["id"] in selected["new_ids"]]
    validate_output_inventory(output, selected["new_ids"])
    summary = {"schema": "lipnet-local-step04-teacher-manifest-v1", "status": "completed", "phase": "train",
               "created_utc": datetime.now(timezone.utc).isoformat(), "teacher_plan": binding,
               "input_manifest": plan["input_manifest"], "input_audit": plan["input_audit"],
               "pilot_features": plan["pilot_features"], "pilot_teacher_plan": plan["pilot_teacher_plan"],
               "comparison_recipe": plan["comparison_recipe"], "runtime": plan["runtime"],
               "ids": selected["train_ids"], "records": completed, "record_count": 441,
               "pilot_references_reused": 8, "new_feature_record_count": 433,
               "newly_created_ids_this_run": created, "resumed_ids_this_run": resumed,
               "state_sha256_before": initial, "state_sha256_after": initial,
               "checkpoint_contract": checkpoint_contract, "teacher_load_seconds": load_seconds,
               "elapsed_seconds_this_run": time.perf_counter() - started,
               "native_frames": sum(r["native_feature_shape"][1] for r in completed),
               "student_frames": sum(r["aligned_feature_shape"][0] for r in completed),
               "new433_gpu_forward_seconds": sum(r["feature_timing"]["seconds"] for r in new_records),
               "new433_peak_allocated_bytes": max(r["feature_timing"]["peak_allocated_bytes"] for r in new_records),
               "new433_peak_reserved_bytes": max(r["feature_timing"]["peak_reserved_bytes"] for r in new_records),
               "all441_feature_file_bytes": sum(Path(r["feature_file"]["path"]).stat().st_size for r in completed),
               "new433_feature_file_bytes": sum(Path(r["feature_file"]["path"]).stat().st_size for r in new_records),
               "audio_used": False, "dev_input_npz_opened": False, "dev_teacher_targets_created": False,
               "decoder_instantiated": False, "student_optimization_steps": 0, "teacher_optimization_steps": 0,
               "limitations": ["Feature preparation only; no new student learning or recognition score.",
                               "Eight entries retain original Step2 provenance and timing.",
                               "GPU forward timings exclude input verification, normalization, transfers and publication."]}
    destination = results / "train_manifest.json"
    if destination.exists():
        prior_summary = read_bound(teacher.binding(destination))
        if prior_summary["teacher_plan"] != binding or prior_summary["records"] != completed:
            raise ValueError("Completed immutable manifest differs from current outputs")
        summary = prior_summary
    else:
        teacher.save_new_json(destination, summary)
    print(json.dumps({"teacher_manifest": teacher.binding(destination), "record_count": 441,
                      "status": "completed"}, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--plan", type=Path, default=BASE / "teacher_plan.json")
    parser.add_argument("--manifest", type=Path, default=BASE / "inputs/manifest.json")
    parser.add_argument("--audit", type=Path, default=BASE / "results/input_audit.json")
    args = parser.parse_args()
    if args.freeze:
        freeze_plan(args.plan.resolve(), args.manifest.resolve(), args.audit.resolve())
        print(json.dumps({"frozen_teacher_plan": teacher.binding(args.plan)}, ensure_ascii=False), flush=True)
    else:
        run(args.plan.resolve())


if __name__ == "__main__":
    main()
