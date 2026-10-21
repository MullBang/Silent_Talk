"""Independently verify stage14's paired learning-rate continuation artifacts.

Read saved tensors, telemetry and posteriors on CPU. No model forward pass,
training, beam search or validation/test cache load is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models import GRAPHEME_VOCAB
from models.dataset_manifest import sha256_file
from models.tokenizer import encode_text, validate_tokenizer_metadata
from diagnose_temporal_use import prediction_row
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import GROUP_PREFIXES, METRIC_DEFINITIONS, make_training_schedule
from summarize_learning_transition import (
    require, finite_tree, read_json, read_rows, compare, describe, checked_hash,
    inside, summarize_group, cohort_metadata, validate_telemetry,
)
import summarize_learning_budget as budget
from learning_budget_state import compare_nested, compare_state_dicts

BASE_STEP = 6400
END_STEP = 9600
BRANCHES = {"A_keep": .0003, "B_reduce": .0001}
STEPS = [6400, 6401, 6500, 6800, 7200, 8000, 8800, 9600]
CONTROL_STEPS = [6400, 7200, 8000, 9600]
BEAM_STEPS = [6400, 8000, 9600]
WINDOWS = ((6401, 6500), (6501, 6800), (6801, 7200), (7201, 8000), (8001, 8800), (8801, 9600))


def exact(actual, expected, where: str) -> None:
    require(compare_nested(expected, actual)["exact"], f"Exact state mismatch: {where}")


def parameter_norms(state: dict) -> dict:
    """Reproduce grouped float64 norms from saved CPU weights, without a model."""
    grouped = {name: [] for name in GROUP_PREFIXES}
    for name, tensor in state.items():
        matches = [group for group, prefix in GROUP_PREFIXES.items() if name.startswith(prefix)]
        require(len(matches) == 1, "Unclassified saved model parameter")
        require(isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                and bool(torch.isfinite(tensor).all()), "Invalid model parameter")
        grouped[matches[0]].append(float(tensor.to(torch.float64).square().sum()))
    require(all(grouped.values()), "Missing model parameter group")
    result = {name: math.sqrt(sum(parts)) for name, parts in grouped.items()}
    result["total"] = math.sqrt(sum(value * value for value in result.values()))
    return result


def scaler_after(rows: list[dict], initial: dict) -> dict:
    """Track successful AMP updates from the actual resumed growth tracker."""
    state = copy.deepcopy(initial)
    require(set(state) == {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"},
            "Invalid initial scaler state")
    require(0 <= state["_growth_tracker"] < state["growth_interval"], "Invalid scaler growth tracker")
    for row in rows:
        compare(row["scale_before"], state["scale"], "Resumed AMP scale before")
        if row["skipped"]:
            state["scale"] *= state["backoff_factor"]
            state["_growth_tracker"] = 0
        else:
            state["_growth_tracker"] += 1
            if state["_growth_tracker"] == state["growth_interval"]:
                state["scale"] *= state["growth_factor"]
                state["_growth_tracker"] = 0
        compare(row["scale_after"], state["scale"], "Resumed AMP scale after")
    return state


def validate_extension_telemetry(rows: list[dict], schedule: dict, records: dict,
                                 settings: dict, initial: dict) -> dict:
    """Validate original absolute attempts plus resumed counters, scale and norms."""
    start = initial["attempt"]
    require(len(rows) == schedule["attempt_updates"] - start and bool(rows), "Continuation telemetry length mismatch")
    batches = schedule["batches"][start:]
    exposures = Counter({sample_id: 0 for sample_id in schedule["sample_ids"]})
    for batch in batches:
        exposures.update(batch["ids"])
    counters = initial["counters"]
    adjusted = [{**row,
        "optimizer_steps": row["optimizer_steps"] - counters["optimizer_steps"],
        "skipped_optimizer_steps": row["skipped_optimizer_steps"] - counters["skipped_optimizer_steps"],
    } for row in rows]
    segment = {**schedule, "attempt_updates": len(rows), "batches": batches, "exposures": dict(exposures)}
    segment_settings = {**settings, "grad_scaler_init_scale": initial["scaler_state_dict"]["scale"]}
    result = validate_telemetry(adjusted, segment, records, segment_settings)
    initial_norms = parameter_norms(initial["model_state_dict"])
    for group, norm in initial_norms.items():
        value = rows[0]["parameter_update"]["total"] if group == "total" else rows[0]["parameter_update"]["groups"][group]
        compare(value["before_l2"], norm, "First update connects to resumed model", rtol=2e-6)
    final_scaler = scaler_after(rows, initial["scaler_state_dict"])
    cumulative = {
        "optimizer_steps": counters["optimizer_steps"] + result["optimizer_steps"],
        "skipped_optimizer_steps": counters["skipped_optimizer_steps"] + result["skipped_optimizer_steps"],
        "attempted_exposures": {s: counters["attempted_exposures"][s] + result["attempted_exposures"][s]
                                for s in schedule["sample_ids"]},
        "successful_update_exposures": {s: counters["successful_update_exposures"][s] + result["successful_update_exposures"][s]
                                        for s in schedule["sample_ids"]},
    }
    return {"extension": result, "cumulative": cumulative, "final_scaler_state": final_scaler}


def compare_initial_states(source: dict, candidate: dict, learning_rate: float) -> dict:
    """Require identical resumed training state, except the declared Adam LR."""
    fields = ("schema", "inference_only", "deployment", "boundary", "attempt", "model_class", "optimizer_class",
              "scaler_class", "scaler_enabled", "parameter_signature", "optimizer_parameter_names",
              "model_state_dict", "scaler_state_dict", "rng_state", "model_training", "module_training",
              "backend_state", "counters", "environment")
    checks = {}
    for field in fields:
        report = compare_nested(source[field], candidate[field])
        require(report["exact"], f"Initial continuation state changed: {field}")
        checks[field] = {key: report[key] for key in ("exact", "compared_leaves", "difference_count")}
    optimizer = copy.deepcopy(source["optimizer_state_dict"])
    require(len(optimizer["param_groups"]) == 1, "Single Adam group required")
    optimizer["param_groups"][0]["lr"] = learning_rate
    exact(candidate["optimizer_state_dict"], optimizer, "Initial Adam state except declared LR")
    return {"learning_rate": learning_rate, "source_training_state_exact_except_learning_rate": True,
            "verified_fields": checks, "adam_moments_and_step_counters_exact": True}


def validate_full_state(value: dict, source: dict, checkpoint_state: dict, *,
                        step: int, learning_rate: float, counters: dict, scaler: dict) -> dict:
    """Check full resume structure and independently derived counters on CPU."""
    compare(value["attempt"], step, "Full-state attempt")
    for field in ("schema", "inference_only", "deployment", "boundary", "model_class", "optimizer_class",
                  "scaler_class", "scaler_enabled", "parameter_signature", "optimizer_parameter_names",
                  "model_training", "module_training", "backend_state", "environment"):
        exact(value[field], source[field], f"Full-state topology/backend.{field}")
    exact(value["counters"], counters, "Full-state counters")
    exact(value["scaler_state_dict"], scaler, "Full-state scaler")
    require(compare_state_dicts(checkpoint_state, value["model_state_dict"])["exact"],
            "Full-state weights disagree with inference checkpoint")
    state, signature = value["model_state_dict"], value["parameter_signature"]
    names = [entry["name"] for entry in signature]
    require(len(names) == len(set(names)) and set(names) == set(state), "Full-state parameter inventory")
    for entry in signature:
        tensor = state[entry["name"]]
        compare(entry, {"name": entry["name"], "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "requires_grad": True}, "Full-state signature")
    exact(value["optimizer_parameter_names"], [names], "Adam parameter names")
    optimizer = value["optimizer_state_dict"]
    require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
            "Adam optimizer inventory")
    group = optimizer["param_groups"][0]
    expected_group = copy.deepcopy(source["optimizer_state_dict"]["param_groups"][0])
    expected_group["lr"] = learning_rate
    exact(group, expected_group, "Adam group settings")
    require(len(group["params"]) == len(names) and len(set(group["params"])) == len(names)
            and set(group["params"]) == set(optimizer["state"]), "Adam parameter-state inventory")
    for parameter_id, name in zip(group["params"], names):
        moment = optimizer["state"][parameter_id]
        require(set(moment) == {"step", "exp_avg", "exp_avg_sq"}, "Adam moment fields")
        require(isinstance(moment["step"], torch.Tensor) and moment["step"].numel() == 1
                and moment["step"].item() == counters["optimizer_steps"], "Adam update counter")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = moment[key]
            require(isinstance(tensor, torch.Tensor) and tensor.dtype == state[name].dtype
                    and tensor.shape == state[name].shape and bool(torch.isfinite(tensor).all()),
                    "Invalid Adam moment tensor")
        require(bool((moment["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")
    rng = value["rng_state"]
    require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG inventory")
    require(isinstance(rng["python"], tuple) and len(rng["python"]) == 3 and rng["python"][0] == 3,
            "Python RNG structure")
    for tensor in [rng["torch_cpu"], *rng["torch_cuda"]]:
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint8 and tensor.ndim == 1
                and tensor.numel() > 0, "Torch RNG structure")
    require(len(rng["torch_cuda"]) == len(source["rng_state"]["torch_cuda"]) == 1, "CUDA RNG inventory")
    numpy_rng = rng["numpy"]
    require(numpy_rng["bit_generator"] == "MT19937" and numpy_rng["keys"].dtype == torch.int64
            and numpy_rng["keys"].shape == (624,) and bool(((numpy_rng["keys"] >= 0)
            & (numpy_rng["keys"] < 2**32)).all()) and 0 <= numpy_rng["position"] <= 624
            and numpy_rng["has_gauss"] in (0, 1) and math.isfinite(numpy_rng["cached_gaussian"]),
            "NumPy RNG structure")
    return {"step": step, "checkpoint_tensors_exact": True, "model_tensor_count": len(state),
            "adam_parameter_states": len(optimizer["state"]), "optimizer_steps": counters["optimizer_steps"],
            "scaler_scale": scaler["scale"], "scaler_growth_tracker": scaler["_growth_tracker"],
            "note": "CPU inspection only. Model tensors and counters are cross-checked; later Adam moments and RNG are schema/hash-checked, not replayed."}


def compare_saved_evaluation(source_folder: Path, candidate_folder: Path) -> dict:
    """The baseline must match the original step6400 rows and raw arrays exactly."""
    a_rows, b_rows = read_rows(source_folder / "rows.jsonl"), read_rows(candidate_folder / "rows.jsonl")
    exact(b_rows, a_rows, "Reproduced initial evaluation rows")
    with np.load(source_folder / "posteriors.npz", allow_pickle=False) as a, \
            np.load(candidate_folder / "posteriors.npz", allow_pickle=False) as b:
        require(a.files == b.files, "Initial posterior array inventory/order changed")
        for key in a.files:
            require(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape and np.isfinite(a[key]).all()
                    and np.array_equal(a[key], b[key]), "Initial posterior array changed")
        count = len(a.files)
    require(count == len(a_rows), "Initial posterior/row inventory mismatch")
    return {"prediction_rows_exact": True, "posterior_arrays_exact": True,
            "posterior_arrays_compared": count, "posterior_max_absolute_difference": 0.}


def schedule_metadata(schedule: dict) -> dict:
    return {key: schedule[key] for key in ("sample_ids", "attempt_updates", "batch_size", "seed", "batches_sha256")}


def provenance(plan: dict, branch: str) -> dict:
    return {"mode": "smoke_only", "purpose": "train_subset_lr_diagnostic", "branch": branch,
            "manifest_sha256": plan["manifest_sha256"], "sample_ids": plan["cohorts"]["expanded64"],
            "generalization_evaluation": False, "deployment": False}


def validate_plan(plan: dict) -> tuple[dict, dict]:
    """Bind the paired protocol to the complete immutable stage13 artifact graph."""
    compare(plan["schema"], "lipnet-learning-rate-v1", "Plan schema")
    prior = Path(plan["prior_run"]).resolve()
    old_plan, old_completed = read_json(prior / "plan.json"), read_json(prior / "completed.json")
    require(old_completed["status"] == "completed", "Source experiment is incomplete")
    checked_hash(prior / "plan.json", plan["prior_plan_sha256"])
    checked_hash(prior / "plan.json", old_completed["plan_sha256"])
    records, _ = budget.validate_plan(old_plan)
    required = {"plan.json", "completed.json", "prefix_gate.json", "telemetry.jsonl"}
    for spec in old_completed["snapshots"]:
        folder = inside(prior, spec["folder"])
        checked_hash(folder / "summary.json", spec["summary_sha256"])
        stored = read_json(folder / "summary.json")
        for filename, field in (("rows.jsonl", "rows_sha256"), ("posteriors.npz", "posteriors_sha256")):
            checked_hash(folder / filename, stored[field])
        required.update((Path(spec["folder"]) / name).as_posix()
                        for name in ("summary.json", "rows.jsonl", "posteriors.npz"))
        if spec["step"]:
            checked_hash(inside(prior, spec["checkpoint"]), spec["checkpoint_sha256"])
            required.add(Path(spec["checkpoint"]).as_posix())
        else:
            checked_hash(Path(spec["checkpoint"]), spec["checkpoint_sha256"])
    required.update(spec["file"] for spec in old_completed["resume_states"])
    require(set(plan["prior_files"]) == required, "Prior artifact inventory mismatch")
    for relative, digest in plan["prior_files"].items():
        checked_hash(inside(prior, relative), digest)
    checked_hash(prior / "prefix_gate.json", old_completed["prefix_gate_sha256"])
    checked_hash(prior / "telemetry.jsonl", old_completed["telemetry_sha256"])
    old_telemetry = read_rows(prior / "telemetry.jsonl")
    old_counts = validate_telemetry(old_telemetry, old_plan["schedule"], records, old_plan["settings"])
    for field, expected in old_counts.items():
        compare(old_completed[field], expected, f"Source completion.{field}")
    budget.validate_resume_states(prior, old_plan, old_completed, old_telemetry)
    checked_hash(Path(plan["prior_summary"]), plan["prior_summary_sha256"])
    verified = read_json(Path(plan["prior_summary"]))
    compare(verified["status"], "completed_and_verified", "Source verified summary")
    compare(verified["provenance"]["completed_sha256"], sha256_file(prior / "completed.json"), "Source verified completion")
    compare(verified["provenance"]["plan_sha256"], old_completed["plan_sha256"], "Source verified plan")
    require(Path(plan["prior_state"]).resolve() == prior / "training_state_6400.pt", "Unexpected source full state")
    checked_hash(Path(plan["prior_state"]), plan["prior_state_sha256"])
    source = torch.load(plan["prior_state"], map_location="cpu", weights_only=True)
    contract = {"attempt": BASE_STEP, "source_hashes": old_plan["source_hashes"], "settings": old_plan["settings"],
        "schedule": schedule_metadata(old_plan["schedule"]),
        "provenance": {"mode": "smoke_only", "purpose": "train_subset_budget_diagnostic",
            "manifest_sha256": old_plan["manifest_sha256"], "sample_ids": old_plan["cohorts"]["expanded64"],
            "generalization_evaluation": False, "deployment": False, "plan_sha256": old_completed["plan_sha256"]},
        "counters": {key: old_completed[key] for key in ("optimizer_steps", "skipped_optimizer_steps",
                                                        "attempted_exposures", "successful_update_exposures")}}
    exact(plan["source_contract"], contract, "Source resume contract")
    exact({key: source[key] for key in contract}, contract, "Saved source resume contract")
    for field in ("manifest", "manifest_sha256", "cohorts", "records", "scope", "nonfinite_policy"):
        compare(plan[field], old_plan[field], f"Same source inputs.{field}")
    exact(plan["environment"], old_completed["environment"], "Same execution environment")
    required_sources = [*budget.SOURCE_PATHS, "scripts/diagnose_learning_rate.py"]
    require(set(plan["source_hashes"]) == set(required_sources), "Stage14 source inventory mismatch")
    for relative, digest in plan["source_hashes"].items():
        checked_hash(inside(ROOT, relative), digest)
    expected_settings = {**old_plan["settings"], "attempt_updates": END_STEP,
        "fresh_optimizer_per_branch": False, "grad_scaler_init_scale": source["scaler_state_dict"]["scale"],
        "normal_steps": STEPS, "control_steps": CONTROL_STEPS, "beam_steps": BEAM_STEPS}
    compare(plan["settings"], expected_settings, "Fixed settings")
    compare(plan["branches"], {key: {"learning_rate": lr, "settings": {**expected_settings, "learning_rate": lr}}
                               for key, lr in BRANCHES.items()}, "Branch learning rates/settings")
    schedule = make_training_schedule(plan["cohorts"]["expanded64"], attempt_updates=END_STEP)
    compare(plan["schedule"], schedule, "Full common schedule")
    compare(schedule["batches"][:BASE_STEP], old_plan["schedule"]["batches"], "Source schedule prefix")
    for field, expected in {"start_attempt": BASE_STEP, "continuation_start": BASE_STEP + 1,
                            "additional_attempt_updates": END_STEP - BASE_STEP,
                            "resume_state_steps": [6400, 8000, 9600],
                            "initial_counters": source["counters"], "initial_scaler": source["scaler_state_dict"]}.items():
        exact(plan[field], expected, f"Continuation plan.{field}")
    compare(plan["preflight_policy"],
        "Both branches restore source exactly, change only LR, save full initial state and reproduce all step6400 outputs before either branch trains.",
        "Preflight policy")
    return records, source


def validate_checkpoint(path: Path, plan: dict, *, branch: str, step: int,
                        counters: dict, plan_sha: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(path))
    require(checkpoint["architecture_version"] == "lipnet-compact-v1" and checkpoint["num_classes"] == 69
            and checkpoint["lstm_hidden"] == 256 and checkpoint["vocab"] == GRAPHEME_VOCAB,
            "Inference checkpoint architecture/tokenizer")
    require(not any(key in checkpoint for key in ("optimizer", "optimizer_state_dict", "scaler", "rng_state")),
            "Expected inference-only checkpoint")
    require(checkpoint["state_dict"] and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
            for v in checkpoint["state_dict"].values()), "Invalid inference model tensors")
    compare(checkpoint["training_data"], provenance(plan, branch), "Inference checkpoint provenance")
    compare(checkpoint["training_run"], {
        "lr_diagnostic": True, "branch": branch, "attempt_updates": step,
        "additional_attempt_updates": step - BASE_STEP, "optimizer_steps": counters["optimizer_steps"],
        "skipped_optimizer_steps": counters["skipped_optimizer_steps"], "plan_sha256": plan_sha,
        "source_training_state_sha256": plan["prior_state_sha256"], "inference_only": True,
        "selection": "fixed_attempt_schedule", "settings": plan["branches"][branch]["settings"],
    }, "Inference checkpoint run")
    return checkpoint["state_dict"]


def counters_at(source: dict, rows: list[dict], step: int) -> dict:
    counters = copy.deepcopy(source["counters"])
    attempted, successful = Counter(counters["attempted_exposures"]), Counter(counters["successful_update_exposures"])
    for row in rows:
        if row["attempt"] > step:
            break
        attempted.update(row["ids"])
        if not row["skipped"]:
            successful.update(row["ids"])
        counters["optimizer_steps"] = row["optimizer_steps"]
        counters["skipped_optimizer_steps"] = row["skipped_optimizer_steps"]
    counters.update(attempted_exposures=dict(attempted), successful_update_exposures=dict(successful))
    return counters


RUNTIME_FIELDS = ("model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state",
                  "model_training", "module_training", "backend_state")


def validate_gates(run: Path, plan: dict, completed: dict, source: dict,
                   initial_states: dict, branch_completed: dict) -> dict:
    """Recheck saved baseline states/outputs and the hash-bound preflight evidence."""
    checked_hash(run / "preflight_gate.json", completed["preflight_gate_sha256"])
    preflight = read_json(run / "preflight_gate.json")
    plan_sha = completed["plan_sha256"]
    source_runtime = {key: source[key] for key in RUNTIME_FIELDS}
    initial_snapshots, initial_specs, gate_digests, results = {}, {}, {}, {}
    for branch, learning_rate in BRANCHES.items():
        folder = run / branch
        state = initial_states[branch]
        initial_specs[branch] = branch_completed[branch]["resume_states"][0]
        initial_snapshots[branch] = branch_completed[branch]["snapshots"][0]
        report = compare_initial_states(source, state, learning_rate)
        outputs = compare_saved_evaluation(Path(plan["prior_run"]) / "step_6400", folder / "step_6400")
        candidate_runtime = {key: state[key] for key in RUNTIME_FIELDS}
        runtime_report = compare_nested(candidate_runtime, candidate_runtime)
        expected_restore = {"status": "passed", "source_runtime_exact": compare_nested(source_runtime, source_runtime),
            "lr_only_runtime_exact": runtime_report, "learning_rate": learning_rate,
            "before_lrs": [group["lr"] for group in source["optimizer_state_dict"]["param_groups"]],
            "evaluation_preserved_runtime": runtime_report, "initial_outputs": outputs,
            "source_training_state_sha256": plan["prior_state_sha256"],
            "initial_training_state_sha256": initial_specs[branch]["sha256"]}
        gate_digests[branch] = checked_hash(folder / "restore_gate.json", branch_completed[branch]["restore_gate_sha256"])
        exact(read_json(folder / "restore_gate.json"), expected_restore, "Stored source restoration gate")
        checked_hash(folder / "start_gate.json", branch_completed[branch]["start_gate_sha256"])
        expected_start = {"status": "passed", "source_runtime_exact": runtime_report,
            "lr_only_runtime_exact": runtime_report, "learning_rate": learning_rate, "before_lrs": [learning_rate],
            "initial_training_state_sha256": initial_specs[branch]["sha256"],
            "preflight_gate_sha256": completed["preflight_gate_sha256"], "next_attempt": BASE_STEP + 1}
        exact(read_json(folder / "start_gate.json"), expected_start, "Stored branch-start gate")
        results[branch] = {**report, "initial_outputs": outputs,
                          "restore_gate_sha256": gate_digests[branch],
                          "start_gate_sha256": branch_completed[branch]["start_gate_sha256"]}
    expected_runtime = copy.deepcopy({key: initial_states["A_keep"][key] for key in RUNTIME_FIELDS})
    for group in expected_runtime["optimizer_state_dict"]["param_groups"]:
        group["lr"] = BRANCHES["B_reduce"]
    cross = compare_nested(expected_runtime, {key: initial_states["B_reduce"][key] for key in RUNTIME_FIELDS})
    require(cross["exact"], "Branch initial runtimes differ beyond learning rate")
    exact(preflight, {"status": "passed", "plan_sha256": plan_sha,
        "checked_before_either_branch_training": True, "cross_branch_runtime_lr_only_exact": cross,
        "restore_gate_sha256": gate_digests, "initial_states": initial_specs,
        "initial_snapshots": initial_snapshots}, "Stored paired preflight gate")
    return {"passed": True, "cross_branch_runtime_exact_except_learning_rate": True,
            "branches": results, "preflight_gate_sha256": completed["preflight_gate_sha256"],
            "note": "Saved initial full states and all baseline outputs independently compared on CPU; live preflight/start checks are hash-bound runtime evidence."}


def validate_snapshot(folder: Path, plan: dict, records: dict, step: int, checkpoint_sha: str) -> dict:
    stored = read_json(folder / "summary.json")
    conditions = ["normal", "static_middle", "shuffle_42"] if step in CONTROL_STEPS else ["normal"]
    beam_performed = step in BEAM_STEPS
    for name, value in {"step": step, "beam_performed": beam_performed, "conditions": conditions,
                        "rows": len(records)*len(conditions), "checkpoint_sha256": checkpoint_sha}.items():
        compare(stored[name], value, f"snapshot.{name}")
    require(stored["seconds"] >= 0, "Negative snapshot duration")
    checked_hash(folder / "rows.jsonl", stored["rows_sha256"])
    checked_hash(folder / "posteriors.npz", stored["posteriors_sha256"])
    rows = read_rows(folder / "rows.jsonl")
    require(len(rows) == stored["rows"], "Snapshot row count mismatch")
    expected_pairs = [(sample_id, condition) for sample_id in plan["cohorts"]["expanded64"] for condition in conditions]
    compare([[r["id"], r["condition"]] for r in rows], [list(pair) for pair in expected_pairs], "snapshot row scope/order")
    normal = {}
    with np.load(folder / "posteriors.npz", allow_pickle=False) as posteriors:
        require(len(set(r["posterior_key"] for r in rows)) == len(rows)
                and set(posteriors.files) == {r["posterior_key"] for r in rows}, "Posterior inventory mismatch")
        for row in rows:
            source, condition = records[row["id"]], row["condition"]
            cohort = "short16" if row["id"] in plan["cohorts"]["short16"] else "additional48"
            for name, value in {"step": step, "split": "train", "cohort": cohort, "speaker_id": source["speaker_id"],
                                "reference": source["text"], "frames": source["frames"]}.items():
                compare(row[name], value, f"row.{name}")
            logp = posteriors[row["posterior_key"]]
            require(logp.dtype == np.float32 and logp.shape == (row["frames"], 69) and np.isfinite(logp).all(),
                    "Nonfinite or invalid posterior")
            require((row["beam"] is not None) == beam_performed, "Beam availability mismatch")
            labels = np.asarray(encode_text(row["reference"]), dtype=np.int64)
            diagnostics = analyze_ctc(logp, labels, beam_text=row["beam"]["prediction"] if beam_performed else "")
            if not beam_performed:
                diagnostics = {k: v for k, v in diagnostics.items() if not k.startswith("beam")}
            compare(row["ctc"], diagnostics, "posterior CTC statistics")
            compare(row["greedy"], prediction_row(source, diagnostics["greedy_text"]), "greedy prediction")
            if beam_performed:
                compare(row["beam"], prediction_row(source, row["beam"]["prediction"]), "beam scoring")
            loss = torch.nn.functional.ctc_loss(torch.from_numpy(logp[:, None, :]), torch.from_numpy(labels),
                torch.tensor([len(logp)]), torch.tensor([len(labels)]), blank=0, reduction="mean", zero_infinity=False)
            compare(row["ctc_loss_per_target_token"], float(loss), "posterior CTC loss", atol=2e-5, rtol=2e-5)
            if condition == "normal":
                require(row["comparison_to_normal"] is None, "Normal has control comparison")
                normal[row["id"]] = (logp, diagnostics["greedy_text"])
            else:
                normal_logp, normal_text = normal[row["id"]]
                compare(row["comparison_to_normal"], compare_posteriors(normal_logp, logp,
                        normal_text=normal_text, control_text=diagnostics["greedy_text"]), "normal/control posterior")
            # Source index metadata can be reconstructed without loading pixels.
            _, transform = transform_frames(np.zeros((row["frames"], 1, 1, 3), np.uint8),
                "shuffle" if condition == "shuffle_42" else condition, sample_id=row["id"], seed=42)
            transform["input_shape"] = [row["frames"], 96, 96, 3]
            changed = row["transform"]["pixel_changed_frame_count"]
            require(isinstance(changed, int) and not isinstance(changed, bool) and 0 <= changed <= row["frames"],
                    "Invalid changed-frame count")
            if condition == "normal":
                require(changed == 0, "Normal transform changed pixels")
            transform["pixel_changed_frame_count"] = changed
            compare(row["transform"], transform, "transform metadata")
    cohorts = {cohort: {condition: summarize_group([r for r in rows if r["cohort"] == cohort and r["condition"] == condition])
                        for condition in conditions} for cohort in ("short16", "additional48")}
    expected_quick = {cohort: {"count": values["normal"]["count"],
                              "greedy_exact": values["normal"]["greedy"]["exact_count"],
                              "greedy_mean_cer": values["normal"]["greedy"]["mean_sentence_jamo_cer"]}
                      for cohort, values in cohorts.items()}
    compare(stored["cohorts"], expected_quick, "snapshot quick summary")
    return {"step": step, "beam_performed": beam_performed, "conditions": conditions, "rows": len(rows),
            "checkpoint_sha256": checkpoint_sha, "seconds": stored["seconds"], "cohorts": cohorts}


def fixed_window_summary(rows: list[dict]) -> dict:
    result = {}
    for start, end in WINDOWS:
        values = [row for row in rows if start <= row["attempt"] <= end]
        if not values:
            continue
        result[f"{start:04d}-{end:04d}"] = {
            "first_attempt": start, "last_attempt": end, "attempts_observed": len(values),
            "optimizer_steps": sum(not r["skipped"] for r in values), "skipped_count": sum(r["skipped"] for r in values),
            "clip_invoked_count": sum(r["clip_applied"] for r in values),
            "effective_clipping_count": sum(r["clip_coefficient"] is not None and r["clip_coefficient"] < 1 for r in values),
            "zero_parameter_update_count": sum(r["parameter_update"]["total"]["absolute_l2"] == 0 for r in values),
            "loss": describe([r["loss"] for r in values]), "seconds_per_attempt": describe([r["seconds"] for r in values]),
            "clip_coefficient": describe([r["clip_coefficient"] for r in values]),
            "scale_before": describe([r["scale_before"] for r in values]),
            "scale_after": describe([r["scale_after"] for r in values]),
            "groups": {}}
        for group in [*GROUP_PREFIXES, "total"]:
            def metrics(row, key):
                return row[key]["total"] if group == "total" else row[key]["groups"][group]
            result[f"{start:04d}-{end:04d}"]["groups"][group] = {
                key: {field: describe([metrics(r, key)[field] for r in values]) for field in fields}
                for key, fields in {
                    "preclip": ("l2_norm", "finite_l2_norm", "nonfinite_grad_elements", "missing_grad_tensors", "zero_grad_tensors"),
                    "postclip": ("l2_norm", "finite_l2_norm", "nonfinite_grad_elements", "missing_grad_tensors", "zero_grad_tensors"),
                    "parameter_update": ("absolute_l2", "relative_l2", "changed_finite_elements"),
                }.items()}
    return result


def endpoint_comparison(branches: dict) -> dict:
    result = {}
    for cohort in ("short16", "additional48"):
        result[cohort] = {}
        for condition in ("normal", "static_middle", "shuffle_42"):
            baseline = branches["A_keep"]["snapshots"]["6400"]["cohorts"][cohort][condition]
            endpoints = {key: branches[key]["snapshots"]["9600"]["cohorts"][cohort][condition] for key in BRANCHES}
            entry = {"baseline6400": {key: baseline[key] for key in ("count", "greedy", "beam", "ctc_loss_per_target_token")},
                     "endpoints9600": {branch: {key: value[key] for key in ("count", "greedy", "beam", "ctc_loss_per_target_token")}
                                       for branch, value in endpoints.items()}}
            comparisons = {f"{branch}_minus_baseline": (value, baseline) for branch, value in endpoints.items()}
            comparisons["B_reduce_minus_A_keep"] = (endpoints["B_reduce"], endpoints["A_keep"])
            for label, (a, b) in comparisons.items():
                entry[label] = {decoder: {metric: a[decoder][metric] - b[decoder][metric]
                    for metric in ("mean_sentence_jamo_cer", "corpus_jamo_cer", "mean_sentence_wer", "corpus_wer", "exact_count")}
                    for decoder in ("greedy", "beam")}
            result[cohort][condition] = entry
    return result


def build_summary(runpath: str | Path) -> dict:
    run = Path(runpath).resolve()
    plan, completed = read_json(run / "plan.json"), read_json(run / "completed.json")
    require(completed["status"] == "completed", "Experiment is not completed")
    plan_sha = checked_hash(run / "plan.json", completed["plan_sha256"])
    records, source = validate_plan(plan)
    for field, expected in plan["scope"].items():
        compare(completed[field], expected, f"Completed scope.{field}")
    compare(completed["environment"], plan["environment"], "Completed environment")
    compare(completed["train_cache_ids_opened"], plan["cohorts"]["expanded64"], "Opened train caches")
    require(set(completed["branch_completed_sha256"]) == set(BRANCHES), "Completed branch inventory")
    branches, initial_states, branch_completed, telemetry_by_branch = {}, {}, {}, {}
    for branch, learning_rate in BRANCHES.items():
        folder = run / branch
        checked_hash(folder / "completed.json", completed["branch_completed_sha256"][branch])
        final = read_json(folder / "completed.json")
        branch_completed[branch] = final
        for field, expected in {"status": "completed", "branch": branch, "learning_rate": learning_rate,
                                "plan_sha256": plan_sha, "attempt_updates": END_STEP,
                                "additional_attempt_updates": END_STEP - BASE_STEP, **plan["scope"]}.items():
            compare(final[field], expected, f"Branch completion.{field}")
        checked_hash(folder / "telemetry.jsonl", final["telemetry_sha256"])
        rows = read_rows(folder / "telemetry.jsonl")
        telemetry_by_branch[branch] = rows
        validated = validate_extension_telemetry(rows, plan["schedule"], records,
                                                plan["branches"][branch]["settings"], source)
        for field, expected in validated["cumulative"].items():
            compare(final[field], expected, f"Branch cumulative.{field}")
        for target, origin in (("additional_optimizer_steps", "optimizer_steps"),
                               ("additional_skipped_optimizer_steps", "skipped_optimizer_steps")):
            compare(final[target], validated["extension"][origin], f"Branch additional.{target}")
        compare(final["attempted_exposures"], plan["schedule"]["exposures"], "Final common attempted exposures")
        compare([spec["step"] for spec in final["snapshots"]], STEPS, "Fixed snapshot inventory")
        snapshots, rows_by_step, checkpoint_states = {}, {}, {}
        for spec in final["snapshots"]:
            step = spec["step"]
            compare(spec["folder"], f"step_{step:04d}", "Snapshot folder")
            compare(spec["checkpoint"], f"step_{step:04d}.pt", "Inference checkpoint path")
            snapshot_folder = inside(folder, spec["folder"])
            checked_hash(snapshot_folder / "summary.json", spec["summary_sha256"])
            checkpoint_path = inside(folder, spec["checkpoint"])
            checked_hash(checkpoint_path, spec["checkpoint_sha256"])
            state = validate_checkpoint(checkpoint_path, plan, branch=branch, step=step,
                                        counters=counters_at(source, rows, step), plan_sha=plan_sha)
            if step in plan["resume_state_steps"]:
                checkpoint_states[step] = state
            snapshots[f"{step:04d}"] = validate_snapshot(snapshot_folder, plan, records, step, spec["checkpoint_sha256"])
            rows_by_step[step] = read_rows(snapshot_folder / "rows.jsonl")
        compare([spec["step"] for spec in final["resume_states"]], plan["resume_state_steps"], "Full-state inventory")
        resume_checks = []
        for spec in final["resume_states"]:
            step = spec["step"]
            compare(spec["file"], f"training_state_{step:04d}.pt", "Full-state filename")
            path = inside(folder, spec["file"])
            checked_hash(path, spec["sha256"])
            state = torch.load(path, map_location="cpu", weights_only=True)
            for field, expected in {"source_hashes": plan["source_hashes"],
                "settings": plan["branches"][branch]["settings"], "schedule": schedule_metadata(plan["schedule"]),
                "provenance": {**provenance(plan, branch), "plan_sha256": plan_sha}}.items():
                exact(state[field], expected, f"Full-state metadata.{field}")
            portion = [row for row in rows if row["attempt"] <= step]
            report = validate_full_state(state, source, checkpoint_states[step], step=step, learning_rate=learning_rate,
                counters=counters_at(source, rows, step), scaler=scaler_after(portion, source["scaler_state_dict"]))
            if step == BASE_STEP:
                initial_states[branch] = state
                compare_initial_states(source, state, learning_rate)
            resume_checks.append({**report, "file": spec["file"], "sha256": spec["sha256"]})
        for field in ("seconds", "peak_allocated_mib", "peak_reserved_mib"):
            require(final[field] >= 0, "Negative branch runtime/memory")
        branches[branch] = {
            "learning_rate": learning_rate, "attempt_updates": END_STEP,
            "additional_attempt_updates": END_STEP - BASE_STEP, **validated["cumulative"],
            "additional_optimizer_steps": validated["extension"]["optimizer_steps"],
            "additional_skipped_optimizer_steps": validated["extension"]["skipped_optimizer_steps"],
            "additional_attempted_exposures": validated["extension"]["attempted_exposures"],
            "additional_successful_update_exposures": validated["extension"]["successful_update_exposures"],
            "snapshots": snapshots, "resume_states": resume_checks, "fixed_windows": fixed_window_summary(rows),
            "skipped_attempts": [{key: row[key] for key in ("attempt", "ids", "loss", "scale_before", "scale_after")}
                                 for row in rows if row["skipped"]],
            "length_bins": {f"{step:04d}": budget.summarize_length_bins(values, "additional48")
                            for step, values in rows_by_step.items()},
            "sentence_trajectories": budget.sentence_trajectories(rows_by_step, plan["cohorts"]["expanded64"]),
            "exposure_summary": {kind: {cohort: describe([validated["extension"][kind][s] for s in plan["cohorts"][cohort]])
                for cohort in ("short16", "additional48")} for kind in ("attempted_exposures", "successful_update_exposures")},
            "runtime": {key: final[key] for key in ("seconds", "peak_allocated_mib", "peak_reserved_mib")},
            "telemetry_sha256": final["telemetry_sha256"], "completed_sha256": completed["branch_completed_sha256"][branch],
        }
    gates = validate_gates(run, plan, completed, source, initial_states, branch_completed)
    first_a, first_b = telemetry_by_branch["A_keep"][0], telemetry_by_branch["B_reduce"][0]
    first_fields = ("attempt", "epoch", "batch_in_epoch", "indices", "ids", "loss", "preclip", "postclip",
                    "clip_coefficient", "clip_applied", "scale_before", "scale_after", "skipped", "frames", "target_lengths")
    exact({key: first_a[key] for key in first_fields}, {key: first_b[key] for key in first_fields},
          "First paired loss/gradient telemetry")
    require(completed["seconds"] >= 0, "Negative experiment duration")
    snapshots_all = [value for branch in branches.values() for value in branch["snapshots"].values()]
    result = {"schema": "lipnet-learning-rate-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_and_verified", "run": str(run),
        "provenance": {"plan_sha256": plan_sha, "completed_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": plan["manifest_sha256"], "prior_state_sha256": plan["prior_state_sha256"],
            "prior_run": plan["prior_run"], "prior_files": plan["prior_files"],
            "prior_summary_sha256": plan["prior_summary_sha256"],
            "summarizer_sha256": sha256_file(Path(__file__)), "source_hashes": plan["source_hashes"],
            "train_cache_count_verified": len(records), "validation_cache_access": False, "test_cache_access": False,
            "model_forward_calls": 0, "beam_decodes": 0,
            "saved_posterior_rows_verified": sum(snapshot["rows"] for snapshot in snapshots_all),
            "saved_beam_rows_rescored": sum(snapshot["rows"] for snapshot in snapshots_all if snapshot["beam_performed"]),
            "raw_posterior_recalculation": ["greedy text and CER/WER", "CTC loss", "blank/entropy/token statistics", "normal/control comparison"],
            "limits": ["Saved beam strings are rescored; beam search is not rerun.",
                "Raw per-update gradients were not saved; telemetry is hash-checked and internally cross-checked.",
                "Cache hashes and transform indices are checked; changed-pixel counts are range-checked, not recomputed.",
                "Saved initial states are exactly cross-checked; later Adam moments and RNG are schema/hash-checked, not replayed."]},
        "protocol": {"settings": plan["settings"], "branches": plan["branches"], "scope": plan["scope"],
            "cohorts": plan["cohorts"], "cohort_metadata": cohort_metadata(plan, records),
            "schedule": {key: value for key, value in plan["schedule"].items() if key != "batches"},
            "start_attempt": BASE_STEP, "additional_attempt_updates": END_STEP - BASE_STEP,
            "initial_counters": plan["initial_counters"], "initial_scaler": plan["initial_scaler"],
            "metric_definitions": METRIC_DEFINITIONS},
        "preflight_verification": gates, "common": branches["A_keep"]["snapshots"]["6400"],
        "branches": branches, "comparison": endpoint_comparison(branches),
        "paired_first_attempt": {"attempt": BASE_STEP + 1, "same_forward_and_gradient_telemetry_exact": True,
            "verified_fields": list(first_fields), "ids": first_a["ids"], "loss": first_a["loss"],
            "note": "Saved loss and gradient telemetry values compare exactly. Raw gradient tensors were not saved and are not compared.",
            "actual_parameter_updates": {branch: rows[0]["parameter_update"]["total"]
                                         for branch, rows in telemetry_by_branch.items()}},
        "runtime": {"seconds": completed["seconds"], "environment": completed["environment"]},
        "interpretation": [
            "Both branches restore the same stage13 step6400 model, Adam moments/counters, GradScaler and RNG state.",
            "Only Adam learning rate changes: A=.0003 versus B=.0001. The common schedule adds3200 attempted updates to9600.",
            "This paired comparison has one shared initialization and no independent seed replicates.",
            "Fixed endpoint differences measure train64 reconstruction. No validation/test inference, best-checkpoint selection or deployment.",
            "Same attempted exposure does not imply same successful exposure if learning rates cause different later AMP skips.",
            "Static/shuffle effects measure sensitivity on training clips and do not establish recognition of unseen speakers or webcam speech.",
            "Sentence-mean CER and pooled edit-count CER have different denominators. All errors are jamo-token metrics.",
            "Residual errors may reflect alignment, representation, optimization or data constraints; this one comparison cannot establish their cause.",
        ]}
    finite_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/step14-learning-rate/run01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {args.out}")
    torch.set_num_threads(2)
    result = build_summary(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
