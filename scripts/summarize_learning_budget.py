"""Verify stage13's fixed 64-clip learning-budget experiment without model inference.

The first1600 attempts must reproduce stage12 B before extension to6400.
Posterior metrics are recomputed; saved telemetry is internally cross-checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from models import GRAPHEME_VOCAB
from models.dataset_manifest import load_manifest, sha256_file
from models.tokenizer import encode_text, validate_tokenizer_metadata
from diagnose_temporal_use import prediction_row
from temporal_diagnostics import analyze_ctc, compare_posteriors, transform_frames
from transition_diagnostics import (
    GROUP_PREFIXES, METRIC_DEFINITIONS, make_training_schedule, select_transition_cohorts,
)
from summarize_learning_transition import (
    require, finite_tree, read_json, read_rows, compare, describe, checked_hash,
    inside, summarize_group, cohort_metadata, validate_telemetry, first_observed_drop,
    validate_plan as validate_prior_plan,
)
from diagnose_learning_transition import SOURCE_PATHS as PRIOR_SOURCE_PATHS
from learning_budget_state import compare_state_dicts, compare_telemetry, compare_schedule_prefix

STEPS = [0, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 2400, 3200, 4800, 6400]
CONTROL_STEPS = [0, 100, 400, 1600, 3200, 6400]
BEAM_STEPS = [0, 1600, 3200, 6400]
WINDOWS = ((1, 100), (101, 400), (401, 800), (801, 1600),
           (1601, 2400), (2401, 3200), (3201, 4800), (4801, 6400))
SOURCE_PATHS = [*PRIOR_SOURCE_PATHS, "scripts/diagnose_learning_budget.py", "scripts/learning_budget_state.py"]


def validate_plan(plan: dict) -> tuple[dict, dict]:
    """Tie the longer schedule to immutable stage12 B artifacts and train-only data."""
    require(plan["schema"] == "lipnet-learning-budget-v1", "Unsupported budget plan schema")
    prior = Path(plan["prior_run"]).resolve()
    require(set(plan["source_hashes"]) == set(SOURCE_PATHS), "Frozen source inventory mismatch")
    for relative, digest in plan["source_hashes"].items():
        checked_hash(inside(ROOT, relative), digest)
    for relative, digest in plan["prior_files"].items():
        checked_hash(inside(prior, relative), digest)
    old_plan, old_completed = read_json(prior / "plan.json"), read_json(prior / "completed.json")
    old_branch = read_json(prior / "B_expand64/completed.json")
    checked_hash(prior / "plan.json", old_completed["plan_sha256"])
    checked_hash(prior / "B_expand64/completed.json", old_completed["branch_completed_sha256"]["B_expand64"])
    require(old_completed["status"] == old_branch["status"] == "completed", "Stage12 B is incomplete")
    records = validate_prior_plan(old_plan)
    for field in ("cohorts", "records", "manifest", "manifest_sha256", "initial_checkpoint", "initial_checkpoint_sha256"):
        compare(plan[field], old_plan[field], f"same stage12 input.{field}")
    expected_settings = {**old_plan["settings"], "attempt_updates": 6400,
                         "normal_steps": STEPS, "control_steps": CONTROL_STEPS, "beam_steps": BEAM_STEPS}
    compare(plan["settings"], expected_settings, "budget settings")
    compare(plan["nonfinite_policy"], old_plan["nonfinite_policy"], "same nonfinite policy")
    compare(plan["resume_state_steps"], [1600, 3200, 6400], "Resume state schedule")
    compare(plan["prefix_gate_policy"], {
        "comparison": "exact", "before_attempt": 1601,
        "model": "every state_dict tensor at original evaluation steps",
        "outputs": "every raw posterior and prediction row at original evaluation steps",
        "telemetry": "all first1600 numeric/ID/counter values, excluding wall time seconds only",
        "on_mismatch": "write failure and stop before any further training; do not relax silently",
    }, "Prefix gate policy")
    compare(plan["scope"], {key: old_plan["scope"][key] for key in
            ("train_only", "validation_cache_access", "test_cache_access", "generalization_evaluation", "deployment")},
            "same train-only scope")
    compare(plan["schedule"], make_training_schedule(plan["cohorts"]["expanded64"], attempt_updates=6400),
            "budget schedule")
    compare(plan["schedule"]["batches"][:1600], old_plan["schedules"]["B_expand64"]["batches"],
            "stage12 schedule prefix")
    required_prior = {"plan.json", "completed.json", "B_expand64/completed.json", "B_expand64/telemetry.jsonl",
                      "common/step_0000/summary.json", "common/step_0000/rows.jsonl", "common/step_0000/posteriors.npz"}
    for spec in old_branch["snapshots"]:
        required_prior.add(Path(spec["checkpoint"]).as_posix())
        for filename in ("summary.json", "rows.jsonl", "posteriors.npz"):
            required_prior.add((Path(spec["folder"]) / filename).as_posix())
    require(set(plan["prior_files"]) == required_prior, "Stage12 frozen artifact inventory mismatch")
    checked_hash(prior / "B_expand64/telemetry.jsonl", old_branch["telemetry_sha256"])
    checked_hash(prior / "common/step_0000/summary.json", old_completed["common_summary_sha256"])
    for spec in old_branch["snapshots"]:
        checked_hash(inside(prior, spec["checkpoint"]), spec["checkpoint_sha256"])
        checked_hash(inside(prior, spec["folder"]) / "summary.json", spec["summary_sha256"])
    return records, old_branch


def validate_checkpoint(path: Path, plan: dict, *, step: int, updates: int,
                        skips: int, plan_sha: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_tokenizer_metadata(checkpoint, str(path))
    require(checkpoint["architecture_version"] == "lipnet-compact-v1"
            and checkpoint["num_classes"] == 69 and checkpoint["lstm_hidden"] == 256
            and checkpoint["vocab"] == GRAPHEME_VOCAB, "Checkpoint architecture/tokenizer mismatch")
    require(not any(key in checkpoint for key in ("optimizer", "optimizer_state_dict", "scaler", "rng_state")),
            "Expected inference-only checkpoint")
    state = checkpoint["state_dict"]
    require(isinstance(state, dict) and bool(state) and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
            for v in state.values()), "Missing/nonfinite checkpoint tensors")
    compare(checkpoint["training_data"], {
        "mode": "smoke_only", "purpose": "train_subset_budget_diagnostic",
        "manifest_sha256": plan["manifest_sha256"], "sample_ids": plan["schedule"]["sample_ids"],
        "generalization_evaluation": False, "deployment": False,
    }, "checkpoint.training_data")
    compare(checkpoint["training_run"], {
        "budget_diagnostic": True, "attempt_updates": step, "optimizer_steps": updates,
        "skipped_optimizer_steps": skips, "plan_sha256": plan_sha,
        "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"], "inference_only": True,
        "selection": "fixed_attempt_schedule", "settings": plan["settings"],
    }, "checkpoint.training_run")


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


def summarize_length_bins(rows: list[dict], cohort: str = "additional48") -> dict:
    """Use fixed50-frame bins; report train memorization, never a validation score."""
    require(cohort in ("short16", "additional48"), "Unsupported cohort")
    selected = [row for row in rows if row["cohort"] == cohort and row["condition"] == "normal"]
    require(bool(selected), "Missing normal cohort rows")
    require(all(1 <= row["frames"] <= 200 for row in selected), "Frame length outside fixed bins")
    result = {}
    for first, last in ((1, 50), (51, 100), (101, 150), (151, 200)):
        values = [row for row in selected if first <= row["frames"] <= last]
        if values:
            result[f"{first:03d}-{last:03d}"] = {
                "minimum_frames": first, "maximum_frames": last,
                "sample_ids": [row["id"] for row in values],
                "summary": summarize_group(values),
            }
    require(sum(v["summary"]["count"] for v in result.values()) == len(selected), "Incomplete length bins")
    return result


def sentence_trajectories(rows_by_step: dict[int, list[dict]], sample_ids: list[str]) -> list[dict]:
    """Include every selected clip so examples cannot be selected for favorable scores."""
    result = []
    indexes = {}
    for step, rows in rows_by_step.items():
        normal = [row for row in rows if row["condition"] == "normal"]
        require(len(normal) == len(sample_ids) and {row["id"] for row in normal} == set(sample_ids),
                "Sentence trajectory scope mismatch")
        indexes[step] = {row["id"]: row for row in normal}
    for sample_id in sample_ids:
        baseline = indexes[min(indexes)][sample_id]
        item = {key: baseline[key] for key in ("id", "cohort", "speaker_id", "frames", "reference")}
        item["snapshots"] = {}
        for step, index in sorted(indexes.items()):
            row = index[sample_id]
            for key in ("cohort", "speaker_id", "frames", "reference"):
                compare(row[key], baseline[key], f"trajectory.{sample_id}.{key}")
            item["snapshots"][f"{step:04d}"] = {
                "greedy": row["greedy"], "beam": row["beam"],
                "ctc_loss_per_target_token": row["ctc_loss_per_target_token"],
                "blank_posterior_mean": row["ctc"]["blank_posterior_mean"],
                "argmax_blank_fraction": row["ctc"]["argmax_blank_fraction"],
                "greedy_collapsed_token_count": row["ctc"]["greedy_collapsed_token_count"],
            }
        result.append(item)
    return result


def validate_prefix(run: Path, plan: dict, completed: dict, telemetry: list[dict], old_branch: dict) -> dict:
    """Independently require bit-identical tensors/posteriors and exact stored rows."""
    prior = Path(plan["prior_run"]).resolve()
    old_plan = read_json(prior / "plan.json")
    compare(plan["schedule"]["batches"][:1600], old_plan["schedules"]["B_expand64"]["batches"],
            "reproduction schedule prefix", atol=0., rtol=0.)
    previous = read_rows(prior / "B_expand64/telemetry.jsonl")
    require(len(previous) == 1600 and len(telemetry) >= 1600, "Reproduction telemetry length mismatch")
    for step, (old, new) in enumerate(zip(previous, telemetry[:1600]), start=1):
        compare({k: v for k, v in new.items() if k != "seconds"},
                {k: v for k, v in old.items() if k != "seconds"},
                f"reproduction telemetry.{step}", atol=0., rtol=0.)
    specs = {spec["step"]: spec for spec in completed["snapshots"]}
    old_specs = {0: {"folder": "common/step_0000", "checkpoint": plan["initial_checkpoint"]},
                 **{spec["step"]: spec for spec in old_branch["snapshots"]}}
    checks, gate_snapshots = [], []
    for step, old_spec in old_specs.items():
        old_folder, new_folder = inside(prior, old_spec["folder"]), inside(run, specs[step]["folder"])
        compare(read_rows(new_folder / "rows.jsonl"), read_rows(old_folder / "rows.jsonl"),
                f"reproduction rows.{step}", atol=0., rtol=0.)
        with np.load(old_folder / "posteriors.npz", allow_pickle=False) as old_npz, \
                np.load(new_folder / "posteriors.npz", allow_pickle=False) as new_npz:
            require(set(old_npz.files) == set(new_npz.files), "Reproduction posterior inventory mismatch")
            for key in old_npz.files:
                a, b = old_npz[key], new_npz[key]
                require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b),
                        f"Reproduction posterior tensor mismatch: {step}/{key}")
            posterior_count = len(old_npz.files)
        old_checkpoint = Path(old_spec["checkpoint"]) if step == 0 else inside(prior, old_spec["checkpoint"])
        new_checkpoint = Path(specs[step]["checkpoint"]) if step == 0 else inside(run, specs[step]["checkpoint"])
        a = torch.load(old_checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        b = torch.load(new_checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        require(a.keys() == b.keys(), "Reproduction model tensor inventory mismatch")
        require(all(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
                    and torch.equal(a[key], b[key]) for key in a), f"Reproduction model tensor mismatch: {step}")
        checks.append({"step": step, "model_tensor_count": len(a), "model_tensors_exact": True,
                       "posterior_rows": posterior_count, "posteriors_exact": True, "rows_exact": True})
        gate_snapshots.append({"step": step, "model_state": compare_state_dicts(a, b),
            "prediction_rows_exact": True, "posterior_arrays_exact": True,
            "posterior_arrays_compared": posterior_count, "posterior_max_absolute_difference": 0.})
    checked_hash(run / "prefix_gate.json", completed["prefix_gate_sha256"])
    gate = read_json(run / "prefix_gate.json")
    require(gate["status"] == "passed", "Reproduction gate did not pass")
    compare(gate, {"status": "passed", "checked_before_attempt": 1601,
        "plan_sha256": sha256_file(run / "plan.json"), "steps": [item["step"] for item in checks],
        "snapshots": gate_snapshots,
        "schedule": compare_schedule_prefix(old_plan["schedules"]["B_expand64"]["batches"], plan["schedule"]["batches"], attempts=1600),
        "telemetry": compare_telemetry(previous, telemetry[:1600], excluded_keys=("seconds",)),
        "inference_only_source": True, "prior_optimizer_state_available": False,
        "skipped_attempts": [row["attempt"] for row in telemetry[:1600] if row["skipped"]],
    }, "stored reproduction gate", atol=0., rtol=0.)
    return {"passed": True, "schedule_prefix_exact": True, "telemetry_attempts_exact_excluding_seconds": 1600,
            "snapshots": checks, "stored_gate_sha256": completed["prefix_gate_sha256"],
            "note": "Independent CPU checks of saved tensors, posteriors, rows and telemetry; no forward or training rerun."}


def validate_resume_states(run: Path, plan: dict, completed: dict, telemetry: list[dict]) -> list[dict]:
    """Check full states against checkpoint tensors, Adam counters and the fixed plan.

    States are loaded on CPU and inspected, never restored or used for inference.
    Optimizer moments and RNG states have no separate historic reference; their
    schema, inventory, finiteness and hash binding are checked explicitly.
    """
    compare([entry["step"] for entry in completed["resume_states"]], [1600, 3200, 6400], "Resume snapshot schedule")
    compare(plan["resume_state_steps"], [1600, 3200, 6400], "Planned resume schedule")
    specs = {entry["step"]: entry for entry in completed["snapshots"]}
    results = []
    for spec in completed["resume_states"]:
        step = spec["step"]
        compare(Path(spec["file"]).as_posix(), f"training_state_{step:04d}.pt", "Resume artifact path")
        path = inside(run, spec["file"])
        checked_hash(path, spec["sha256"])
        value = torch.load(path, map_location="cpu", weights_only=True)
        for name, expected in {
            "schema": "lipnet-learning-budget-resume-v1", "inference_only": False, "deployment": False,
            "boundary": "after_scaler_update_before_next_zero_grad", "attempt": step,
            "model_class": "models.baseline.LipNetBaseline", "optimizer_class": "torch.optim.adam.Adam",
            "scaler_class": "torch.amp.grad_scaler.GradScaler", "scaler_enabled": True,
            "model_training": True, "source_hashes": plan["source_hashes"], "settings": plan["settings"],
            "schedule": {k: plan["schedule"][k] for k in ("sample_ids", "attempt_updates", "batch_size", "seed", "batches_sha256")},
            "provenance": {"mode": "smoke_only", "purpose": "train_subset_budget_diagnostic",
                "manifest_sha256": plan["manifest_sha256"], "sample_ids": plan["cohorts"]["expanded64"],
                "generalization_evaluation": False, "deployment": False, "plan_sha256": completed["plan_sha256"]},
        }.items():
            compare(value[name], expected, f"resume.{name}")
        subset = telemetry[:step]
        exposures = Counter({s: 0 for s in plan["schedule"]["sample_ids"]})
        successful = exposures.copy()
        for row in subset:
            exposures.update(row["ids"])
            if not row["skipped"]:
                successful.update(row["ids"])
        compare(value["counters"], {
            "optimizer_steps": subset[-1]["optimizer_steps"], "skipped_optimizer_steps": subset[-1]["skipped_optimizer_steps"],
            "attempted_exposures": dict(exposures), "successful_update_exposures": dict(successful),
        }, "resume.counters")
        checkpoint = torch.load(inside(run, specs[step]["checkpoint"]), map_location="cpu", weights_only=True)
        state, checkpoint_state = value["model_state_dict"], checkpoint["state_dict"]
        state_comparison = compare_state_dicts(checkpoint_state, state)
        require(state_comparison["exact"], "Resume tensors disagree with inference checkpoint")
        signature = value["parameter_signature"]
        require(len(signature) == len(state) and {entry["name"] for entry in signature} == set(state),
                "Resume parameter signature inventory mismatch")
        for entry in signature:
            tensor = state[entry["name"]]
            compare(entry, {"name": entry["name"], "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "requires_grad": True}, "resume.parameter_signature")
        names = [entry["name"] for entry in signature]
        compare(value["optimizer_parameter_names"], [names], "Resume optimizer parameter order")
        optimizer = value["optimizer_state_dict"]
        require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
                "Invalid Adam state inventory")
        group = optimizer["param_groups"][0]
        require(len(group["params"]) == len(names) and len(set(group["params"])) == len(names)
                and set(group["params"]) == set(optimizer["state"]), "Adam state parameter inventory mismatch")
        for name, expected in {"lr": .0003, "eps": 1e-8, "weight_decay": 0., "amsgrad": False,
                               "maximize": False, "capturable": False, "differentiable": False}.items():
            compare(group[name], expected, f"Adam.{name}")
        require(group["betas"] == (.9, .999), "Adam beta mismatch")
        for parameter_id, name in zip(group["params"], names):
            item = optimizer["state"][parameter_id]
            require(set(item) == {"step", "exp_avg", "exp_avg_sq"}, "Adam moment fields mismatch")
            require(isinstance(item["step"], torch.Tensor) and item["step"].numel() == 1
                    and item["step"].item() == subset[-1]["optimizer_steps"], "Adam successful-step counter mismatch")
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = item[key]
                require(isinstance(tensor, torch.Tensor) and tensor.dtype == state[name].dtype
                        and tensor.shape == state[name].shape and bool(torch.isfinite(tensor).all()),
                        "Invalid Adam moment tensor")
            require(bool((item["exp_avg_sq"] >= 0).all()), "Negative Adam second moment")
        scale, tracker = 1024., 0
        for row in subset:
            if row["skipped"]:
                scale, tracker = scale * .5, 0
            else:
                tracker += 1
                if tracker == 2000:
                    scale, tracker = scale * 2, 0
            compare(row["scale_after"], scale, "Resume scaler trajectory")
        compare(value["scaler_state_dict"], {"scale": scale, "growth_factor": 2., "backoff_factor": .5,
                "growth_interval": 2000, "_growth_tracker": tracker}, "Resume GradScaler state")
        rng = value["rng_state"]
        require(set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "RNG inventory mismatch")
        require(isinstance(rng["python"], tuple) and len(rng["python"]) == 3 and rng["python"][0] == 3,
                "Python RNG state mismatch")
        require(isinstance(rng["torch_cpu"], torch.Tensor) and rng["torch_cpu"].dtype == torch.uint8
                and rng["torch_cpu"].ndim == 1 and rng["torch_cpu"].numel() > 0, "CPU RNG state mismatch")
        require(len(rng["torch_cuda"]) == 1 and all(isinstance(v, torch.Tensor) and v.dtype == torch.uint8
                and v.ndim == 1 and v.numel() > 0 for v in rng["torch_cuda"]), "CUDA RNG state mismatch")
        numpy_rng = rng["numpy"]
        require(numpy_rng["bit_generator"] == "MT19937" and numpy_rng["keys"].dtype == torch.int64
                and numpy_rng["keys"].shape == (624,) and bool(((numpy_rng["keys"] >= 0)
                & (numpy_rng["keys"] < 2**32)).all()) and 0 <= numpy_rng["position"] <= 624
                and numpy_rng["has_gauss"] in (0, 1), "NumPy RNG state mismatch")
        require(value["module_training"] and all(flag is True for flag in value["module_training"].values()),
                "Resume module training mode mismatch")
        for name, expected in {"torch_threads": 2, "cudnn_benchmark": False, "cudnn_deterministic": True,
                               "cudnn_allow_tf32": False, "matmul_allow_tf32": False}.items():
            compare(value["backend_state"][name], expected, f"Resume backend.{name}")
        compare(value["environment"], {key: completed["environment"][key] for key in ("torch", "numpy", "cuda")},
                "Resume execution environment")
        results.append({"step": step, "file": spec["file"], "sha256": spec["sha256"],
            "checkpoint_tensors_exact": True, "model_tensor_count": len(state),
            "adam_parameter_states": len(optimizer["state"]), "optimizer_steps": subset[-1]["optimizer_steps"],
            "scaler_scale": scale, "scaler_growth_tracker": tracker,
            "note": "CPU inspection only; checkpoint tensors/counters are cross-checked, moments and RNG are schema/hash-checked, not replayed."})
    return results


def build_summary(runpath: str | Path) -> dict:
    run = Path(runpath).resolve()
    plan, completed = read_json(run / "plan.json"), read_json(run / "completed.json")
    require(completed["status"] == "completed", "Experiment is not completed")
    plan_sha = checked_hash(run / "plan.json", completed["plan_sha256"])
    for name in ("validation_cache_access", "test_cache_access", "deployment"):
        require(completed[name] is False, "Forbidden completed scope")
    records, old_branch = validate_plan(plan)
    compare(completed["train_cache_ids_opened"], plan["cohorts"]["expanded64"], "Opened train cache IDs")
    checked_hash(run / "telemetry.jsonl", completed["telemetry_sha256"])
    telemetry = read_rows(run / "telemetry.jsonl")
    validated = validate_telemetry(telemetry, plan["schedule"], records, plan["settings"])
    for name, value in validated.items():
        compare(completed[name], value, f"completed.{name}")
    compare([spec["step"] for spec in completed["snapshots"]], STEPS, "Fixed snapshot schedule")
    snapshots, rows_by_step = {}, {}
    for spec in completed["snapshots"]:
        step = spec["step"]
        compare(Path(spec["folder"]).as_posix(), f"step_{step:04d}", "Snapshot path")
        folder = inside(run, spec["folder"])
        checked_hash(folder / "summary.json", spec["summary_sha256"])
        if step == 0:
            compare(spec["checkpoint"], plan["initial_checkpoint"], "Initial source checkpoint")
            compare(spec["checkpoint_sha256"], plan["initial_checkpoint_sha256"], "Initial checkpoint SHA")
        else:
            compare(Path(spec["checkpoint"]).as_posix(), f"step_{step:04d}.pt", "Checkpoint path")
            checkpoint = inside(run, spec["checkpoint"])
            checked_hash(checkpoint, spec["checkpoint_sha256"])
            event = telemetry[step-1]
            validate_checkpoint(checkpoint, plan, step=step, updates=event["optimizer_steps"],
                                skips=event["skipped_optimizer_steps"], plan_sha=plan_sha)
        snapshots[f"{step:04d}"] = validate_snapshot(folder, plan, records, step, spec["checkpoint_sha256"])
        if step in (0, 1600, 3200, 6400):
            rows_by_step[step] = read_rows(folder / "rows.jsonl")
    require(snapshots["0000"]["cohorts"]["short16"]["normal"]["greedy"]["exact_count"] == 16,
            "Initial short16 reproduction failed")
    prefix = validate_prefix(run, plan, completed, telemetry, old_branch)
    resume_states = validate_resume_states(run, plan, completed, telemetry)
    for key in ("seconds", "peak_allocated_mib", "peak_reserved_mib"):
        require(completed[key] >= 0, "Negative runtime/memory")
    comparison = {}
    for cohort in ("short16", "additional48"):
        comparison[cohort] = {}
        for condition in snapshots["6400"]["conditions"]:
            entries = {key: snapshots[key]["cohorts"][cohort][condition] for key in ("1600", "3200", "6400")}
            comparison[cohort][condition] = {
                "budgets": {key: {field: row[field] for field in ("count", "greedy", "beam", "ctc_loss_per_target_token")}
                            for key, row in entries.items()},
                "6400_minus_1600": {decoder: {
                    metric: entries["6400"][decoder][metric] - entries["1600"][decoder][metric]
                    for metric in ("mean_sentence_jamo_cer", "corpus_jamo_cer", "mean_sentence_wer", "corpus_wer", "exact_count")}
                    for decoder in ("greedy", "beam")},
            }
    result = {"schema": "lipnet-learning-budget-summary-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_and_verified", "run": str(run), **validated,
        "provenance": {"plan_sha256": plan_sha, "completed_sha256": sha256_file(run / "completed.json"),
            "manifest_sha256": plan["manifest_sha256"], "initial_checkpoint_sha256": plan["initial_checkpoint_sha256"],
            "summarizer_sha256": sha256_file(Path(__file__)), "source_hashes": plan["source_hashes"],
            "prior_files": plan["prior_files"], "prior_run": plan["prior_run"],
            "train_cache_count_verified": len(records), "validation_cache_access": False, "test_cache_access": False,
            "model_forward_calls": 0, "beam_decodes": 0,
            "saved_posterior_rows_verified": sum(s["rows"] for s in snapshots.values()),
            "saved_beam_rows_rescored": sum(s["rows"] for s in snapshots.values() if s["beam_performed"]),
            "raw_posterior_recalculation": ["greedy text and CER/WER", "CTC loss", "blank/entropy/token statistics", "normal/control comparison"],
            "limits": ["Saved beam strings are rescored but beam search is not rerun.",
                "Raw per-update gradients were not saved; telemetry is hash-checked and internally cross-checked.",
                "Cache hashes and transform indices are checked; pixel_changed_frame_count is range-checked, not recomputed."]},
        "protocol": {"settings": plan["settings"], "scope": plan["scope"], "cohorts": plan["cohorts"],
            "cohort_metadata": cohort_metadata(plan, records),
            "schedule": {k: v for k, v in plan["schedule"].items() if k != "batches"},
            "metric_definitions": METRIC_DEFINITIONS},
        "prefix_reproduction": prefix, "resume_states": resume_states,
        "snapshots": snapshots, "budget_comparison": comparison,
        "fixed_windows": fixed_window_summary(telemetry),
        "first_observed_short16_greedy_exact_below16": first_observed_drop(snapshots),
        "exposure_summary": {kind: {cohort: describe([validated[kind].get(s, 0) for s in plan["cohorts"][cohort]])
            for cohort in ("short16", "additional48")} for kind in ("attempted_exposures", "successful_update_exposures")},
        "length_bins": {f"{step:04d}": summarize_length_bins(rows, "additional48") for step, rows in rows_by_step.items()},
        "sentence_trajectories": sentence_trajectories(rows_by_step, plan["cohorts"]["expanded64"]),
        "runtime": {k: completed[k] for k in ("seconds", "peak_allocated_mib", "peak_reserved_mib")},
        "telemetry_sha256": completed["telemetry_sha256"],
        "interpretation": [
            "All64 clips belong to the existing train split. Results measure reconstruction of training clips, not generalization.",
            "The run restarts from the same short16 inference weights with fresh Adam/GradScaler and reproduces stage12 B through1600 attempts.",
            "1600/3200/6400 are nested budgets on one training trajectory, not independent seeds or separate model-selection trials.",
            "Per-ID attempted presentations increase50/100/200; successful exposures exclude AMP-skipped batches.",
            "Cohorts16/48 and conditions remain separate. Longer training does not address unseen speakers or webcam domain differences.",
            "CER/WER are ratios. Sentence means and pooled edit-count rates have different denominators.",
            "Token top4 is pooled over raw collapsed greedy token IDs; blank posterior and argmax blank fraction are different metrics.",
            "Static/shuffle changes measure sensitivity and train memorization; they do not prove general visual speech recognition.",
            "No validation/test inference, best-checkpoint selection or deployment was performed.",
        ]}
    finite_tree(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/step13-budget/run01")
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

